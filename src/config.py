"""Centralised configuration and constants.

Single source of truth for paths, schema names and feature definitions so
that the same values are used by the scraper, training notebook and the
serving API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Maroñas REST endpoints (discovered from the public AngularJS bundle)
# ---------------------------------------------------------------------------
MARONAS_REST_URL: Final[str] = os.getenv(
    "MARONAS_REST_URL",
    "https://mobile-rest-services-v3.azurewebsites.net/XTurfRestService.svc",
)
MARONAS_RESOURCES_URL: Final[str] = os.getenv(
    "MARONAS_RESOURCES_URL",
    "https://mobile-rest-services-v3.azurewebsites.net/XTurfResourcesService.svc",
)

# DocumentTypeEnum values copied verbatim from /bundles/app
DOC_TYPE_TABULADA: Final[int] = 2

# Race-track ids documented in the bundle
RACETRACKS: Final[dict[int, str]] = {
    1: "Maroñas",
    13: "Las Piedras",
    4: "Colonia",
    8: "Flores",
    9: "Florida",
    16: "Melo",
    21: "Paysandú",
    22: "Rocha",
}


# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
TARGET_COL: Final[str] = "in_trifecta"
RACE_DATE_COL: Final[str] = "race_date"

# Numeric features computed by FeatureEngineeringPipeline.
NUMERIC_FEATURES: Final[list[str]] = [
    "horse_age",
    "weight_kg",
    "weight_kg_zscore_in_race",
    "rest_days",
    "career_runs",
    "career_wins",
    "career_places",
    "career_shows",
    "career_win_rate",
    "career_show_rate",
    "year_runs",
    "year_wins",
    "year_places",
    "year_shows",
    "year_win_rate",
    "year_show_rate",
    "last_finish_pos",
    "avg_finish_last3",
    "best_finish_last3",
    "n_field",
    "post_position",
    # leakage-safe, target-conditioned features added in v2
    "track_runs",
    "track_show_rate",
    "dist_bucket_runs",
    "dist_bucket_show_rate",
    "days_since_last_win",
    # v4: market signal (dividend = the horse's payout, lower = more favoured)
    "dividend_career_mean",
    "dividend_last3_mean",
    "dividend_career_min",
    # v4: distance / weight fit relative to horse's own history
    "dist_diff_from_avg",
    "weight_change_from_last",
    # v4: jockey signal (computed from history of every horse the jockey rode)
    "jockey_career_runs",
    "jockey_career_show_rate",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "sex_code",
    "racetrack_id",
]

ALL_FEATURES: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ScraperSettings:
    """Runtime configuration for the scraper, populated from env vars."""

    racetrack_id: int = int(os.getenv("SCRAPER_RACETRACK_ID", "1"))
    date_from: str = os.getenv("SCRAPER_DATE_FROM", "2018-01-01")
    date_to: str = os.getenv("SCRAPER_DATE_TO", "2026-06-01")
    raw_dir: Path = Path(os.getenv("SCRAPER_RAW_DIR", str(RAW_DIR)))
    http_timeout: int = int(os.getenv("SCRAPER_HTTP_TIMEOUT", "60"))
    max_workers: int = int(os.getenv("SCRAPER_MAX_WORKERS", "4"))
