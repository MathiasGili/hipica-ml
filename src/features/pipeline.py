"""Feature-engineering pipeline (training-serving skew–safe).

Contract
--------
A single class — :class:`FeatureEngineeringPipeline` — produces the model
input matrix in **both** the training notebook and the FastAPI serving
process. It is sklearn-compatible (``fit`` / ``transform``), so it can also
be embedded inside an MLflow ``sklearn.Pipeline`` for one-shot deployment.

Inputs
------
``fit(history_df)``
    A long-form DataFrame produced by
    :func:`src.ingestion.loader.build_long_form_dataset`. One row per
    (horse, past race) with columns ``horse_name``, ``race_date``,
    ``finish_pos``, ``kg``, ``distance_m``, ``racetrack_id``, etc.
    The pipeline indexes this by ``horse_name`` for fast lookup.

``transform(targets_df)``
    A DataFrame with one row per *target* race participation. Required
    columns: ``horse_name``, ``race_date``, ``distance_m``,
    ``racetrack_id``, ``kg``. Optional but recommended: ``post_position``,
    ``n_field``, ``horse_age``, ``sex_code``.

For every target row the pipeline looks up *all rows in the fitted history
that belong to the same horse with date strictly less than the target's
``race_date``* and computes summary features. Because the date filter is
strict, **the target row's own outcome is never seen by the feature
pipeline** — that's the leakage guard.

For training, ``targets_df`` is typically the history DataFrame itself, in
which case ``in_trifecta`` is the label and the prior-history lookup gives
each row a feature vector. For serving, ``targets_df`` is built from the
incoming API request.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RACE_DATE_COL,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight DTOs for the API. Kept here (not in ``api/``) so that the
# training notebook can also use them when synthesising prediction payloads
# from EDA — same contract everywhere.
# ---------------------------------------------------------------------------
@dataclass
class HorseEntry:
    """One horse entered in a target race."""

    horse_name: str
    kg: float
    post_position: int | None = None
    horse_age: int | None = None
    sex_code: str | None = None      # "M" / "H" / None
    jockey_name: str | None = None


@dataclass
class RaceContext:
    """Metadata of the target race shared by all entrants."""

    race_date: datetime
    racetrack_id: int
    distance_m: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """Stateful feature builder.

    The fitted state is just an indexed copy of the history. There is no
    statistical state (no fitted means/scalers) — those belong to the
    downstream sklearn Pipeline so they get persisted by MLflow alongside
    the model. Keeping this transformer stateless-ish means we can also
    rebuild the history at serving time from the database without retraining.
    """

    REQUIRED_HISTORY_COLS = [
        "horse_name", "race_date", "racetrack_id",
        "finish_pos", "kg", "distance_m",
    ]

    REQUIRED_TARGET_COLS = ["horse_name", "race_date", "distance_m", "racetrack_id", "kg"]

    def __init__(
        self,
        rolling_windows: tuple[int, ...] = (3,),
        max_history_days: int | None = None,
    ) -> None:
        self.rolling_windows = rolling_windows
        self.max_history_days = max_history_days
        self._history: pd.DataFrame | None = None
        self._history_by_horse: dict[str, pd.DataFrame] = {}
        self._history_by_jockey: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, history_df: pd.DataFrame, y=None) -> "FeatureEngineeringPipeline":  # noqa: ARG002
        self._validate_columns(history_df, self.REQUIRED_HISTORY_COLS, "history_df")
        hist = history_df.copy()
        hist["race_date"] = pd.to_datetime(hist["race_date"])
        hist["horse_name"] = hist["horse_name"].astype(str).str.upper().str.strip()
        if "jockey" in hist.columns:
            hist["jockey"] = hist["jockey"].astype(str).str.upper().str.strip()
        hist = hist.dropna(subset=["finish_pos"])
        hist = hist.sort_values(["horse_name", "race_date"])
        self._history = hist.reset_index(drop=True)
        self._history_by_horse = {
            name: g.reset_index(drop=True)
            for name, g in self._history.groupby("horse_name", sort=False)
        }
        if "jockey" in self._history.columns:
            self._history_by_jockey = {
                name: g.reset_index(drop=True)
                for name, g in self._history.groupby("jockey", sort=False)
                if name and name.lower() not in ("nan", "none", "")
            }
        else:
            self._history_by_jockey = {}
        logger.info(
            "FeatureEngineeringPipeline fitted on %d rows / %d horses / %d jockeys",
            len(self._history), len(self._history_by_horse), len(self._history_by_jockey),
        )
        return self

    # ------------------------------------------------------------- transform
    def transform(self, targets_df: pd.DataFrame) -> pd.DataFrame:
        if self._history is None:
            raise RuntimeError(
                "FeatureEngineeringPipeline.transform() called before fit(). "
                "Call .fit(history_df) with the long-form history first."
            )
        self._validate_columns(targets_df, self.REQUIRED_TARGET_COLS, "targets_df")

        targets = targets_df.copy()
        targets["race_date"] = pd.to_datetime(targets["race_date"])
        targets["horse_name"] = targets["horse_name"].astype(str).str.upper().str.strip()

        # Canonical pass-through columns. Derive once here so every downstream
        # path (training notebook + API) sees the same column set.
        targets["weight_kg"] = targets["kg"].astype(float)
        for opt in ("horse_age", "post_position", "sex_code"):
            if opt not in targets.columns:
                targets[opt] = np.nan
        if "jockey_name" not in targets.columns:
            # At training time we read the jockey from the history row itself.
            if "jockey" in targets.columns:
                targets["jockey_name"] = targets["jockey"]
            else:
                targets["jockey_name"] = None
        targets["jockey_name"] = (
            targets["jockey_name"].astype("object").map(
                lambda x: x.upper().strip() if isinstance(x, str) and x.strip() else None
            )
        )

        # Per-race competitive features (z-score of kg within the race, n_field).
        targets = self._add_within_race_features(targets)

        # Per-horse historical features (the leakage-safe ones).
        feature_rows: list[dict] = []
        for _, row in targets.iterrows():
            hist = self._history_for(row["horse_name"], row["race_date"])
            feature_rows.append(self._features_from_history(hist, row))
        feats = pd.DataFrame(feature_rows, index=targets.index)

        out = pd.concat([targets.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

        # Hard guard: the contract is "no duplicate columns ever".
        if out.columns.duplicated().any():
            dups = out.columns[out.columns.duplicated()].tolist()
            raise RuntimeError(
                f"FeatureEngineeringPipeline produced duplicate columns: {dups}. "
                "This is the symptom of a training-serving skew bug."
            )

        # Ensure every expected column exists (NaN if we have no history).
        for col in ALL_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
        return out

    # ------------------------------------------------------------- y helper
    @staticmethod
    def extract_target(df: pd.DataFrame) -> pd.Series:
        """Return the binary in_trifecta target from a labelled frame."""
        if TARGET_COL in df.columns:
            return df[TARGET_COL].astype(int)
        if "finish_pos" not in df.columns:
            raise KeyError("Target frame needs either 'in_trifecta' or 'finish_pos'.")
        return df["finish_pos"].between(1, 3, inclusive="both").astype(int)

    @staticmethod
    def feature_columns() -> list[str]:
        return list(ALL_FEATURES)

    # ----------------------------------------------------------- internals
    def _history_for(self, horse_name: str, race_date: pd.Timestamp) -> pd.DataFrame:
        g = self._history_by_horse.get(horse_name)
        if g is None or g.empty:
            return g if g is not None else self._history.iloc[0:0]
        mask = g["race_date"] < race_date
        if self.max_history_days is not None:
            cutoff = race_date - pd.Timedelta(days=self.max_history_days)
            mask &= g["race_date"] >= cutoff
        return g.loc[mask]

    def _features_from_history(self, hist: pd.DataFrame, row: pd.Series) -> dict:
        """Emit ONLY the leakage-safe historical features.

        Pass-through columns (``horse_age``, ``weight_kg``,
        ``post_position``, ``sex_code``, ``racetrack_id``,
        ``weight_kg_zscore_in_race``, ``n_field``) are populated upstream
        in :meth:`transform`. Re-emitting them here would create duplicate
        columns at concat time — which is the canonical symptom of a
        training-serving skew bug.
        """
        race_date = row["race_date"]
        target_weight = row.get("weight_kg")
        if pd.isna(target_weight):
            target_weight = row.get("kg")

        if hist.empty:
            # Rookie: counts are zero, but every *rate* / "last_X" is NaN
            # so the downstream imputer can fill it deterministically.
            jockey_runs, jockey_show_rate = self._jockey_features(
                row.get("jockey_name"), race_date
            )
            return {
                "career_runs": 0, "career_wins": 0, "career_places": 0, "career_shows": 0,
                "year_runs": 0, "year_wins": 0, "year_places": 0, "year_shows": 0,
                "career_win_rate": np.nan, "career_show_rate": np.nan,
                "year_win_rate": np.nan, "year_show_rate": np.nan,
                "last_finish_pos": np.nan, "avg_finish_last3": np.nan,
                "best_finish_last3": np.nan, "rest_days": np.nan,
                "track_runs": 0, "track_show_rate": np.nan,
                "dist_bucket_runs": 0, "dist_bucket_show_rate": np.nan,
                "days_since_last_win": np.nan,
                "dividend_career_mean": np.nan,
                "dividend_last3_mean": np.nan,
                "dividend_career_min": np.nan,
                "dist_diff_from_avg": np.nan,
                "weight_change_from_last": np.nan,
                "jockey_career_runs": jockey_runs,
                "jockey_career_show_rate": jockey_show_rate,
            }

        last_date = hist["race_date"].max()
        last_row = hist.loc[hist["race_date"].idxmax()]

        # Career counters
        finish_pos = hist["finish_pos"].astype(float)
        career_runs = int(len(hist))
        career_wins = int((finish_pos == 1).sum())
        career_places = int((finish_pos == 2).sum())
        career_shows = int((finish_pos == 3).sum())

        # Same year counters
        same_year = hist[hist["race_date"].dt.year == race_date.year]
        same_year_pos = same_year["finish_pos"].astype(float)
        year_runs = int(len(same_year))

        # Last-N rolling
        last_n = hist.sort_values("race_date").tail(max(self.rolling_windows))
        last_n_pos = last_n["finish_pos"].astype(float)

        # Track-specific career: same racetrack as the target race.
        target_track = row.get("racetrack_id")
        if pd.notna(target_track) and "racetrack_id" in hist.columns:
            track_hist = hist[hist["racetrack_id"] == target_track]
            track_runs = int(len(track_hist))
            track_show_rate = (
                track_hist["finish_pos"].astype(float).between(1, 3).mean()
                if track_runs else np.nan
            )
        else:
            track_runs, track_show_rate = 0, np.nan

        # Distance-bucket career: target distance ± 100 m.
        target_dist = row.get("distance_m")
        if pd.notna(target_dist) and "distance_m" in hist.columns:
            dist_hist = hist[
                hist["distance_m"].between(
                    float(target_dist) - 100, float(target_dist) + 100
                )
            ]
            dist_bucket_runs = int(len(dist_hist))
            dist_bucket_show_rate = (
                dist_hist["finish_pos"].astype(float).between(1, 3).mean()
                if dist_bucket_runs else np.nan
            )
        else:
            dist_bucket_runs, dist_bucket_show_rate = 0, np.nan

        # Days since the horse's last win (any track / distance).
        wins = hist[finish_pos == 1]
        days_since_last_win = (
            (race_date - wins["race_date"].max()).days if not wins.empty else np.nan
        )

        # Market-signal features: the horse's past dividends. Lower dividend
        # at the bookmaker = more favoured. NaN dividends are dropped.
        if "dividend" in hist.columns:
            div = hist["dividend"].astype(float).dropna()
            dividend_career_mean = float(div.mean()) if not div.empty else np.nan
            dividend_career_min = float(div.min()) if not div.empty else np.nan
            div_last3 = hist.sort_values("race_date")["dividend"].astype(float).dropna().tail(3)
            dividend_last3_mean = float(div_last3.mean()) if not div_last3.empty else np.nan
        else:
            dividend_career_mean = dividend_career_min = dividend_last3_mean = np.nan

        # Distance-fit: how far the target distance is from the horse's
        # average historical distance. Positive = horse running longer than usual.
        if pd.notna(target_dist) and "distance_m" in hist.columns:
            past_dist = hist["distance_m"].astype(float).dropna()
            dist_diff_from_avg = (
                float(target_dist) - float(past_dist.mean()) if not past_dist.empty else np.nan
            )
        else:
            dist_diff_from_avg = np.nan

        # Weight delta from the most recent race.
        last_weight = last_row.get("kg")
        if pd.notna(target_weight) and pd.notna(last_weight):
            weight_change_from_last = float(target_weight) - float(last_weight)
        else:
            weight_change_from_last = np.nan

        # Jockey features (independent of horse history — uses jockey index).
        jockey_runs, jockey_show_rate = self._jockey_features(
            row.get("jockey_name"), race_date
        )

        return {
            "rest_days": (race_date - last_date).days,
            "career_runs": career_runs,
            "career_wins": career_wins,
            "career_places": career_places,
            "career_shows": career_shows,
            "career_win_rate": career_wins / career_runs if career_runs else np.nan,
            "career_show_rate": (career_wins + career_places + career_shows) / career_runs
                if career_runs else np.nan,
            "year_runs": year_runs,
            "year_wins": int((same_year_pos == 1).sum()),
            "year_places": int((same_year_pos == 2).sum()),
            "year_shows": int((same_year_pos == 3).sum()),
            "year_win_rate": (same_year_pos == 1).mean() if year_runs else np.nan,
            "year_show_rate": same_year_pos.between(1, 3).mean() if year_runs else np.nan,
            "last_finish_pos": float(last_row["finish_pos"]) if pd.notna(last_row["finish_pos"]) else np.nan,
            "avg_finish_last3": float(last_n_pos.mean()) if not last_n_pos.empty else np.nan,
            "best_finish_last3": float(last_n_pos.min()) if not last_n_pos.empty else np.nan,
            "track_runs": track_runs,
            "track_show_rate": track_show_rate,
            "dist_bucket_runs": dist_bucket_runs,
            "dist_bucket_show_rate": dist_bucket_show_rate,
            "days_since_last_win": float(days_since_last_win) if pd.notna(days_since_last_win) else np.nan,
            "dividend_career_mean": dividend_career_mean,
            "dividend_last3_mean": dividend_last3_mean,
            "dividend_career_min": dividend_career_min,
            "dist_diff_from_avg": dist_diff_from_avg,
            "weight_change_from_last": weight_change_from_last,
            "jockey_career_runs": jockey_runs,
            "jockey_career_show_rate": jockey_show_rate,
        }

    def _jockey_features(
        self, jockey_name, race_date: pd.Timestamp
    ) -> tuple[int, float]:
        """Jockey career runs + show rate from rides strictly before race_date.

        Uses every horse the jockey has ridden — that's the point of this
        feature (cross-horse jockey skill). NaN/empty jockey names give
        ``(0, NaN)`` which the imputer handles.
        """
        if not isinstance(jockey_name, str) or not jockey_name:
            return 0, np.nan
        g = self._history_by_jockey.get(jockey_name)
        if g is None or g.empty:
            return 0, np.nan
        past = g[g["race_date"] < race_date]
        n = int(len(past))
        if n == 0:
            return 0, np.nan
        show_rate = float(past["finish_pos"].astype(float).between(1, 3).mean())
        return n, show_rate

    @staticmethod
    def _add_within_race_features(targets: pd.DataFrame) -> pd.DataFrame:
        """Compute kg z-score and field size *within* each target race."""
        race_keys = ["race_date", "racetrack_id", "distance_m"]
        if not all(k in targets.columns for k in race_keys):
            targets["weight_kg_zscore_in_race"] = np.nan
            targets["n_field"] = np.nan
            return targets

        grp = targets.groupby(race_keys, dropna=False)["kg"]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        targets["weight_kg_zscore_in_race"] = (targets["kg"] - mean) / std
        targets["n_field"] = grp.transform("count").astype(int)
        return targets

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def build_training_frame(history_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for training from a long-form labelled history.

    Each row in ``history_df`` is treated as a labelled target whose features
    are computed from strictly prior runs of the same horse.
    """
    pipe = FeatureEngineeringPipeline().fit(history_df)
    feats = pipe.transform(history_df)
    y = FeatureEngineeringPipeline.extract_target(history_df.assign(
        in_trifecta=history_df["finish_pos"].between(1, 3).astype(int)
    ))
    X = feats[ALL_FEATURES]
    return X, y


def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _safe_int(v) -> int | None:
    f = _safe_float(v)
    if f is None or (isinstance(f, float) and np.isnan(f)):
        return None
    return int(f)
