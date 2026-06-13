"""End-to-end training script.

Pipeline (in order):

1. Load the long-form labelled history from ``data/raw`` (cached as parquet
   under ``data/processed/history.parquet``).
2. Temporal split — strict date cut, no random shuffling.
3. Build features with :class:`FeatureEngineeringPipeline`. Crucially we
   fit the pipeline on the **train** slice only, so test rows can't peek
   at their future neighbours via the in-race z-score either.
4. Wrap a ``ColumnTransformer`` (impute + one-hot) and an XGBoost
   classifier (``tree_method='hist'`` + ``device='cuda'``) in an
   ``sklearn.Pipeline`` so MLflow can serialise the entire artifact.
5. Log metrics, params and the registered model to MLflow.

Run it from the repo root:

    python -m src.training.train --cache --register

CLI flags:
    --cache        Use parquet cache when available (and write one when not).
    --register     Register the model in MLflow Model Registry.
    --device cpu   Force CPU even if CUDA is available.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ..config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RACE_DATE_COL,
    TARGET_COL,
)
from ..features.pipeline import FeatureEngineeringPipeline
from ..ingestion.loader import build_long_form_dataset
from .split import temporal_train_test_split

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trifecta-classifier")
MODEL_NAME = os.getenv("MODEL_NAME", "trifecta-classifier")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/trifecta_pipeline"))


# ---------------------------------------------------------------------------
def build_estimator(device: str = "cuda") -> Pipeline:
    """Return the full sklearn pipeline (preprocessor + XGBoost)."""
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device=device,           # 'cuda' if GPU available, else 'cpu'
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", pre), ("clf", xgb)])


# ---------------------------------------------------------------------------
def _binary_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1_at_05": float(f1_score(y_true, pred, zero_division=0)),
        "precision_at_05": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_05": float(recall_score(y_true, pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
def train(
    cache: bool = True,
    register: bool = False,
    device: str = "cuda",
    test_size: float = 0.2,
) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    cache_path = PROCESSED_DIR / "history.parquet" if cache else None
    history = build_long_form_dataset(cache_path=cache_path, use_cache=bool(cache_path))
    if history.empty:
        raise RuntimeError(
            "No labelled history available. Run the scraper first: "
            "python -m src.ingestion.scraper"
        )
    history = history.dropna(subset=["finish_pos"]).reset_index(drop=True)
    history[TARGET_COL] = history["finish_pos"].between(1, 3, inclusive="both").astype(int)
    logger.info("Loaded %d labelled rows. Positive rate: %.3f",
                len(history), history[TARGET_COL].mean())

    train_df, test_df, cutoff = temporal_train_test_split(history, test_size=test_size)

    fe = FeatureEngineeringPipeline().fit(train_df)
    X_train = fe.transform(train_df)[ALL_FEATURES]
    X_test = fe.transform(test_df)[ALL_FEATURES]
    y_train = train_df[TARGET_COL].astype(int)
    y_test = test_df[TARGET_COL].astype(int)

    estimator = build_estimator(device=device)
    estimator.fit(X_train, y_train)

    proba_test = estimator.predict_proba(X_test)[:, 1]
    proba_train = estimator.predict_proba(X_train)[:, 1]
    metrics = {f"test_{k}": v for k, v in _binary_metrics(y_test, proba_test).items()}
    metrics.update({f"train_{k}": v for k, v in _binary_metrics(y_train, proba_train).items()})
    metrics["temporal_cutoff"] = str(cutoff.date())

    # ---- MLflow ----
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"xgb_hist_{cutoff.date()}") as run:
        mlflow.log_params(
            {
                "test_size": test_size,
                "temporal_cutoff": str(cutoff.date()),
                "device": device,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "feature_count": len(ALL_FEATURES),
                **{f"xgb__{k}": v for k, v in estimator.named_steps["clf"].get_params().items()
                   if isinstance(v, (int, float, str, bool, type(None)))},
            }
        )
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            registered_model_name=MODEL_NAME if register else None,
            input_example=X_train.head(2),
        )

        # Persist the FE pipeline as a separate artifact (the API rebuilds
        # it on startup from the database, but we keep this for reproducibility).
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(fe, MODEL_PATH / "feature_pipeline.joblib")
        joblib.dump(estimator, MODEL_PATH / "estimator.joblib")
        mlflow.log_artifacts(str(MODEL_PATH), artifact_path="local_artifacts")

        run_id = run.info.run_id
        logger.info("MLflow run_id=%s | metrics=%s", run_id, metrics)

    return {"run_id": run_id, "metrics": metrics, "cutoff": cutoff}


# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train Trifecta classifier")
    parser.add_argument("--cache", action="store_true", help="Use cached parquet")
    parser.add_argument("--register", action="store_true", help="Register the model")
    parser.add_argument("--device", default=os.getenv("XGB_DEVICE", "cuda"),
                        choices=["cuda", "cpu"])
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    train(
        cache=args.cache,
        register=args.register,
        device=args.device,
        test_size=args.test_size,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
