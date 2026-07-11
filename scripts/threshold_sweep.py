"""Threshold sweep on val fold for the v5-datafix-tuned model.

Fits a fresh model on train_inner only (winning Optuna params from the
post-datafix run) and evaluates F1/precision/recall at cutoffs 0.20..0.60
on the val fold. Selects the operating threshold *without* peeking at the
outer test fold.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, PROCESSED_DIR, TARGET_COL
from src.features.pipeline import FeatureEngineeringPipeline
from src.ingestion.loader import build_long_form_dataset
from src.training.split import temporal_train_test_split

BEST_PARAMS = {
    "n_estimators": 850,
    "max_depth": 8,
    "learning_rate": 0.012289395781451795,
    "min_child_weight": 4,
    "reg_lambda": 4.437932581831696,
    "reg_alpha": 1.8725404521393565,
    "subsample": 0.8274210541217935,
    "colsample_bytree": 0.8485896935001601,
    "gamma": 2.2519774865432596,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    log = logging.getLogger(__name__)

    history = build_long_form_dataset(cache_path=PROCESSED_DIR / "history.parquet", use_cache=True)
    history = history.dropna(subset=["finish_pos"]).reset_index(drop=True)
    history[TARGET_COL] = history["finish_pos"].between(1, 3, inclusive="both").astype(int)

    train_df, _test_df, outer_cutoff = temporal_train_test_split(history, test_size=0.2)
    train_inner_df, val_df, inner_cutoff = temporal_train_test_split(train_df, test_size=0.2)
    log.info("outer_cutoff=%s inner_cutoff=%s | train_inner=%d val=%d",
             outer_cutoff.date(), inner_cutoff.date(), len(train_inner_df), len(val_df))

    fe = FeatureEngineeringPipeline().fit(train_inner_df)
    X_train_inner = fe.transform(train_inner_df)[ALL_FEATURES]
    X_val = fe.transform(val_df)[ALL_FEATURES]
    y_train_inner = train_inner_df[TARGET_COL].astype(int)
    y_val = val_df[TARGET_COL].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
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
        device="cpu",
        random_state=42,
        n_jobs=-1,
        **BEST_PARAMS,
    )
    estimator = Pipeline([("pre", pre), ("clf", xgb)])
    log.info("Fitting on train_inner with best Optuna params...")
    estimator.fit(X_train_inner, y_train_inner)
    proba_val = estimator.predict_proba(X_val)[:, 1]

    rows = []
    for cutoff in np.arange(0.20, 0.601, 0.025):
        pred = (proba_val >= cutoff).astype(int)
        rows.append({
            "threshold": round(float(cutoff), 3),
            "f1": f1_score(y_val, pred),
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred),
            "positive_rate": float(pred.mean()),
        })
    sweep = pd.DataFrame(rows)
    print("\nThreshold sweep on val fold (train_inner-only fit):")
    print(sweep.to_string(index=False))

    best = sweep.loc[sweep["f1"].idxmax()]
    print("\nBest F1 threshold on val:")
    print(best.to_string())

    sweep.to_csv("reports/threshold_sweep_v5_datafix.csv", index=False)
    print("\nSaved -> reports/threshold_sweep_v5_datafix.csv")


if __name__ == "__main__":
    main()
