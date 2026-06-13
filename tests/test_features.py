"""Anti-skew / anti-leakage tests for the feature pipeline.

The whole point of :class:`FeatureEngineeringPipeline` is that the **same**
code runs at training and at serving time. These tests pin that contract.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeatureEngineeringPipeline


# ---------------------------------------------------------------------------
# Tiny synthetic history: 2 horses, 5 runs each, deterministic positions.
# ---------------------------------------------------------------------------
@pytest.fixture()
def history() -> pd.DataFrame:
    base = datetime(2024, 1, 1)
    rows = []
    for horse, positions in [
        ("ALPHA", [1, 2, 1, 4, 1]),     # 3 wins out of 5
        ("BRAVO", [9, 8, 7, 10, 6]),    # never trifecta
    ]:
        for i, pos in enumerate(positions):
            rows.append(
                {
                    "horse_name": horse,
                    "race_date": base + timedelta(days=30 * i),
                    "racetrack_id": 1,
                    "finish_pos": pos,
                    "kg": 55.0,
                    "distance_m": 1600,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def serving_targets() -> pd.DataFrame:
    """Mirrors what the API constructs from a request — extra columns
    that are *not* in the training history (horse_age, post_position,
    sex_code). This is the shape that exposed the duplicate-column bug.
    """
    rows = [
        {"horse_name": "ALPHA", "race_date": datetime(2024, 6, 1), "racetrack_id": 1,
         "distance_m": 1600, "kg": 55.0, "post_position": 1, "horse_age": 4, "sex_code": "M"},
        {"horse_name": "BRAVO", "race_date": datetime(2024, 6, 1), "racetrack_id": 1,
         "distance_m": 1600, "kg": 56.0, "post_position": 2, "horse_age": 5, "sex_code": "H"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def test_pipeline_emits_canonical_feature_columns(history: pd.DataFrame) -> None:
    pipe = FeatureEngineeringPipeline().fit(history)
    out = pipe.transform(history)
    expected = set(FeatureEngineeringPipeline.feature_columns())
    missing = expected - set(out.columns)
    assert not missing, f"feature columns missing: {missing}"


def test_no_self_leakage(history: pd.DataFrame) -> None:
    """At the i-th appearance of a horse, career_runs must equal i (not i+1)."""
    pipe = FeatureEngineeringPipeline().fit(history)
    out = pipe.transform(history)
    out = out.sort_values(["horse_name", "race_date"]).reset_index(drop=True)
    for horse, group in out.groupby("horse_name"):
        for i, (_, row) in enumerate(group.iterrows()):
            assert row["career_runs"] == i, (
                f"{horse} at appearance {i}: career_runs={row['career_runs']} "
                "(should match number of strictly prior runs)"
            )


def test_rookie_features_are_nan(history: pd.DataFrame) -> None:
    """First-ever run for each horse: counts are 0, rates/last-X are NaN."""
    pipe = FeatureEngineeringPipeline().fit(history)
    out = pipe.transform(history)
    firsts = out.sort_values(["horse_name", "race_date"]).groupby("horse_name").head(1)
    assert (firsts["career_runs"] == 0).all()
    assert firsts["last_finish_pos"].isna().all()
    assert firsts["career_win_rate"].isna().all()


def test_training_and_serving_produce_identical_features(history: pd.DataFrame) -> None:
    """Anti-skew check: the API path and the training path must agree.

    The training path computes features for the labelled history itself.
    The serving path computes features for a single 'incoming' entrant.
    Given identical inputs, both must yield byte-identical numbers.
    """
    pipe = FeatureEngineeringPipeline().fit(history)

    # --- training path ---
    train_feats = pipe.transform(history).sort_values(["horse_name", "race_date"])

    # --- serving path: one row at a time, as the API would call it ---
    serving_rows = []
    for _, row in history.sort_values(["horse_name", "race_date"]).iterrows():
        single = pd.DataFrame([row])
        serving_rows.append(pipe.transform(single))
    serving_feats = pd.concat(serving_rows, ignore_index=True)

    # within-race z-score depends on the whole race, so compare the leakage-safe,
    # purely-historical columns only (the API computes z-score per request).
    historical_cols = [
        "career_runs", "career_wins", "career_places", "career_shows",
        "career_win_rate", "career_show_rate",
        "year_runs", "year_wins", "year_places", "year_shows",
        "year_win_rate", "year_show_rate",
        "last_finish_pos", "avg_finish_last3", "best_finish_last3", "rest_days",
    ]
    a = train_feats[historical_cols].reset_index(drop=True)
    b = serving_feats[historical_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False)


def test_transform_without_fit_raises() -> None:
    pipe = FeatureEngineeringPipeline()
    with pytest.raises(RuntimeError, match="before fit"):
        pipe.transform(pd.DataFrame({"horse_name": ["X"], "race_date": ["2024-01-01"],
                                     "distance_m": [1600], "racetrack_id": [1], "kg": [55]}))


def test_serving_input_has_no_duplicate_columns(
    history: pd.DataFrame, serving_targets: pd.DataFrame
) -> None:
    """The shape that exposed the original duplicate-column bug.

    Serving payloads carry ``horse_age``, ``post_position`` and
    ``sex_code`` that the training history does not have. The pipeline
    must NOT emit duplicates of those columns when concatenating.
    """
    pipe = FeatureEngineeringPipeline().fit(history)
    out = pipe.transform(serving_targets)
    dups = out.columns[out.columns.duplicated()].tolist()
    assert not dups, f"duplicate columns: {dups}"
    # And every expected feature must be present.
    for col in FeatureEngineeringPipeline.feature_columns():
        assert col in out.columns, f"missing feature column: {col}"


def test_serving_pass_through_columns_are_preserved(
    history: pd.DataFrame, serving_targets: pd.DataFrame
) -> None:
    """horse_age / post_position / sex_code must reach the model from the request."""
    pipe = FeatureEngineeringPipeline().fit(history)
    out = pipe.transform(serving_targets).sort_values("horse_name").reset_index(drop=True)
    assert out.loc[0, "horse_age"] == 4
    assert out.loc[1, "horse_age"] == 5
    assert out.loc[0, "post_position"] == 1
    assert out.loc[1, "post_position"] == 2
    assert out.loc[0, "sex_code"] == "M"
    assert out.loc[1, "sex_code"] == "H"
