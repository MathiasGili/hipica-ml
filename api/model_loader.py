"""Loads the trained pipeline + feature engineer at API startup.

Resolution order (first hit wins):

1. MLflow Model Registry — ``models:/<MODEL_NAME>/<MODEL_STAGE>``.
2. Local file fallback — ``models/trifecta_pipeline/{estimator,feature_pipeline}.joblib``.

Keeping a local fallback means developers can run the API offline against a
checkpoint produced by the training notebook without standing up MLflow.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features.pipeline import FeatureEngineeringPipeline
from src.ingestion.loader import build_long_form_dataset

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "trifecta-classifier")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
LOCAL_MODEL_DIR = Path(os.getenv("LOCAL_MODEL_DIR", "models/trifecta_pipeline"))
HISTORY_CACHE = Path(os.getenv("HISTORY_CACHE", "data/processed/history.parquet"))


@dataclass
class LoadedModel:
    estimator: Any
    feature_pipeline: FeatureEngineeringPipeline
    source: str
    version: str | None


def _try_load_from_mlflow() -> LoadedModel | None:
    if not TRACKING_URI:
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            logger.warning("No %s/%s found in MLflow Registry", MODEL_NAME, MODEL_STAGE)
            return None
        version = versions[0]
        uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        estimator = mlflow.sklearn.load_model(uri)
        # The feature pipeline is logged as a separate artifact next to the model.
        run_id = version.run_id
        local_dir = client.download_artifacts(run_id, "local_artifacts")
        fe = joblib.load(Path(local_dir) / "feature_pipeline.joblib")
        logger.info("Loaded model %s v%s from MLflow", MODEL_NAME, version.version)
        return LoadedModel(estimator, fe, "mlflow", version.version)
    except Exception as exc:  # noqa: BLE001
        logger.exception("MLflow load failed: %s", exc)
        return None


def _try_load_local() -> LoadedModel | None:
    est_path = LOCAL_MODEL_DIR / "estimator.joblib"
    fe_path = LOCAL_MODEL_DIR / "feature_pipeline.joblib"
    if not (est_path.exists() and fe_path.exists()):
        return None
    estimator = joblib.load(est_path)
    fe = joblib.load(fe_path)
    logger.info("Loaded model from local fallback at %s", LOCAL_MODEL_DIR)
    return LoadedModel(estimator, fe, "local", None)


def _maybe_warmup_history(loaded: LoadedModel) -> None:
    """If the FE pipeline has no fitted history, try to load it from cache.

    This handles the case where the training process saved the estimator
    but not the FE state (or the API container starts fresh and we want to
    pull the latest history from the database/cache).
    """
    if loaded.feature_pipeline._history is not None:  # noqa: SLF001
        return
    if HISTORY_CACHE.exists():
        df = pd.read_parquet(HISTORY_CACHE)
        loaded.feature_pipeline.fit(df)
        logger.info("FE pipeline warmed up from %s (%d rows)", HISTORY_CACHE, len(df))
        return
    logger.warning(
        "FE pipeline has no fitted history and %s does not exist — "
        "predictions for unseen horses will return rookie defaults.",
        HISTORY_CACHE,
    )


def load_model() -> LoadedModel:
    loaded = _try_load_from_mlflow() or _try_load_local()
    if loaded is None:
        raise RuntimeError(
            f"Cannot load model: no MLflow registry entry for {MODEL_NAME}/{MODEL_STAGE} "
            f"and no local fallback at {LOCAL_MODEL_DIR}. "
            "Run `python -m src.training.train --register` first."
        )
    _maybe_warmup_history(loaded)
    return loaded
