"""Optuna hyperparameter tuning for the Trifecta classifier.

Search strategy
---------------
1. Load and label the long-form history (same as ``train.py``).
2. Carve a temporal **train / test** split.  ``test`` is kept aside and
   never seen during tuning — only the final refit is evaluated on it.
3. Carve an inner **train_inner / val** split *inside* the training
   slice, again temporally.  Optuna optimises against ``val``.
4. Each trial fits the same sklearn pipeline as ``train.build_estimator``
   but with sampled hyperparameters; the FE pipeline is fitted once on
   ``train_inner`` and reused across trials (it does not depend on the
   hyperparameters and is the slow part).
5. Optimise **PR-AUC** — better than ROC-AUC for our ~36% positive rate.
6. Each trial logs to MLflow as a child run under one parent.
7. After the search, refit on the full train slice with the best params
   and evaluate on test; persist artifacts mirroring ``train.py``.

Run from the repo root:

    python -m src.training.tune --cache --device cpu --n-trials 50

CLI flags:
    --cache         Use parquet cache when available.
    --device        ``cpu`` or ``cuda`` (default cuda).
    --n-trials      Number of Optuna trials (default 50).
    --val-size      Fraction of the train slice reserved for validation
                    (default 0.2).
    --register      Register the refitted final model in MLflow Registry.
    --persist       Persist the refitted final model under
                    ``models/trifecta_pipeline_tuned/`` (default True).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from ..config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    TARGET_COL,
)
from ..features.pipeline import FeatureEngineeringPipeline
from ..ingestion.loader import build_long_form_dataset
from .split import temporal_train_test_split
from .train import _binary_metrics

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "trifecta-classifier")
TUNED_MODEL_NAME = os.getenv("TUNED_MODEL_NAME", "trifecta-classifier-tuned")
TUNED_MODEL_PATH = Path(os.getenv("TUNED_MODEL_PATH", "models/trifecta_pipeline_tuned"))


# ---------------------------------------------------------------------------
def _build_pipeline(params: dict[str, Any], device: str) -> Pipeline:
    """Same shape as ``train.build_estimator`` but with sampled params."""
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
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
        device=device,
        random_state=42,
        n_jobs=-1,
        **params,
    )
    return Pipeline(steps=[("pre", pre), ("clf", xgb)])


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


# ---------------------------------------------------------------------------
def tune(
    cache: bool = True,
    device: str = "cuda",
    n_trials: int = 50,
    test_size: float = 0.2,
    val_size: float = 0.2,
    register: bool = False,
    persist: bool = True,
) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # ---- Data ----
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

    # Outer split: keep test untouched.
    train_df, test_df, outer_cutoff = temporal_train_test_split(history, test_size=test_size)
    # Inner split: tune against val, fit on train_inner.
    train_inner_df, val_df, inner_cutoff = temporal_train_test_split(train_df, test_size=val_size)

    # FE pipeline fits on train_inner (so val can't leak via in-race z-score).
    fe_inner = FeatureEngineeringPipeline().fit(train_inner_df)
    X_train_inner = fe_inner.transform(train_inner_df)[ALL_FEATURES]
    X_val = fe_inner.transform(val_df)[ALL_FEATURES]
    y_train_inner = train_inner_df[TARGET_COL].astype(int)
    y_val = val_df[TARGET_COL].astype(int)
    logger.info(
        "Tuning splits | train_inner=%d  val=%d  test_held_out=%d",
        len(X_train_inner), len(X_val), len(test_df),
    )

    # ---- Optuna search ----
    mlflow.set_experiment(EXPERIMENT_NAME)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"trifecta_optuna_{outer_cutoff.date()}")

    parent_run = mlflow.start_run(run_name=f"optuna_search_{outer_cutoff.date()}")
    parent_run_id = parent_run.info.run_id
    mlflow.log_params(
        {
            "n_trials": n_trials,
            "test_size": test_size,
            "val_size": val_size,
            "outer_cutoff": str(outer_cutoff.date()),
            "inner_cutoff": str(inner_cutoff.date()),
            "device": device,
            "n_train_inner": len(X_train_inner),
            "n_val": len(X_val),
            "feature_count": len(ALL_FEATURES),
            "objective": "val_pr_auc",
        }
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.log_params({f"xgb__{k}": v for k, v in params.items()})
            estimator = _build_pipeline(params, device=device)
            estimator.fit(X_train_inner, y_train_inner)
            proba_val = estimator.predict_proba(X_val)[:, 1]
            metrics = {f"val_{k}": v for k, v in _binary_metrics(y_val, proba_val).items()}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            score = metrics["val_pr_auc"]
            trial.set_user_attr("val_roc_auc", metrics["val_roc_auc"])
            trial.set_user_attr("val_log_loss", metrics["val_log_loss"])
            return score

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_value = study.best_value
        logger.info("Best PR-AUC=%.4f | best params=%s", best_value, best_params)

        mlflow.log_metric("best_val_pr_auc", best_value)
        mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

        # ---- Refit on the full train slice with the best params ----
        fe_full = FeatureEngineeringPipeline().fit(train_df)
        X_train = fe_full.transform(train_df)[ALL_FEATURES]
        X_test = fe_full.transform(test_df)[ALL_FEATURES]
        y_train = train_df[TARGET_COL].astype(int)
        y_test = test_df[TARGET_COL].astype(int)

        final_estimator = _build_pipeline(best_params, device=device)
        final_estimator.fit(X_train, y_train)

        proba_test = final_estimator.predict_proba(X_test)[:, 1]
        proba_train = final_estimator.predict_proba(X_train)[:, 1]
        final_metrics = {f"test_{k}": v for k, v in _binary_metrics(y_test, proba_test).items()}
        final_metrics.update({f"train_{k}": v for k, v in _binary_metrics(y_train, proba_train).items()})
        for k, v in final_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        logger.info("Refit on full train | test metrics: %s",
                    {k: v for k, v in final_metrics.items() if k.startswith("test_")})

        mlflow.sklearn.log_model(
            sk_model=final_estimator,
            artifact_path="model",
            registered_model_name=TUNED_MODEL_NAME if register else None,
            input_example=X_train.head(2),
        )

        if persist:
            TUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            joblib.dump(fe_full, TUNED_MODEL_PATH / "feature_pipeline.joblib")
            joblib.dump(final_estimator, TUNED_MODEL_PATH / "estimator.joblib")
            mlflow.log_artifacts(str(TUNED_MODEL_PATH), artifact_path="local_artifacts")
    finally:
        mlflow.end_run()

    return {
        "run_id": parent_run_id,
        "best_params": best_params,
        "best_val_pr_auc": best_value,
        "test_metrics": {k: v for k, v in final_metrics.items() if k.startswith("test_")},
        "outer_cutoff": outer_cutoff,
        "inner_cutoff": inner_cutoff,
    }


# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Optuna tuning for Trifecta classifier")
    parser.add_argument("--cache", action="store_true", help="Use cached parquet")
    parser.add_argument("--device", default=os.getenv("XGB_DEVICE", "cuda"),
                        choices=["cuda", "cpu"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--register", action="store_true",
                        help="Register the refitted model in MLflow Registry")
    parser.add_argument("--no-persist", action="store_true",
                        help="Do not write joblib artifacts to disk")
    args = parser.parse_args()
    tune(
        cache=args.cache,
        device=args.device,
        n_trials=args.n_trials,
        test_size=args.test_size,
        val_size=args.val_size,
        register=args.register,
        persist=not args.no_persist,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
