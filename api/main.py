"""FastAPI inference service for the Trifecta classifier.

Endpoints:

* ``GET  /health`` — liveness + which model version is loaded.
* ``POST /predict_online`` — single horse prediction. Note that without
  the rest of the field, the in-race z-score for kg cannot be computed,
  so it falls back to NaN (the imputer fills with the training median).
* ``POST /predict_batch`` — full field. Recommended path: produces the
  most accurate predictions because all in-race competitive features
  (kg z-score, field size) are computed correctly.

The same :class:`FeatureEngineeringPipeline` that ran in the training
notebook is reused here verbatim — that is the contract that prevents
training-serving skew.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.config import ALL_FEATURES

from .model_loader import LoadedModel, load_model
from .schemas import (
    FeatureContribution,
    HorsePrediction,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictExplainRequest,
    PredictExplainResponse,
    PredictOnlineRequest,
    PredictOnlineResponse,
)

logger = logging.getLogger("hipica.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info("Loading model …")
    app_state["model"] = load_model()
    logger.info("Model ready: source=%s version=%s",
                app_state["model"].source, app_state["model"].version)
    yield
    logger.info("Shutting down")


app_state: dict[str, LoadedModel] = {}
app = FastAPI(
    title="Trifecta Classifier API",
    version="0.1.0",
    description=(
        "Predicts whether a horse will finish in the Trifecta (1st-3rd) "
        "based on tabular Maroñas/Las Piedras racing data."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
def _model() -> LoadedModel:
    m = app_state.get("model")
    if m is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return m


def _entries_to_targets(req_race, entries) -> pd.DataFrame:
    """Convert request DTOs to the long-form DataFrame the FE pipeline expects."""
    rows = []
    for e in entries:
        rows.append(
            {
                "horse_name": e.horse_name,
                "race_date": pd.Timestamp(req_race.race_date),
                "racetrack_id": req_race.racetrack_id,
                "distance_m": req_race.distance_m,
                "kg": e.kg,
                "post_position": e.post_position,
                "horse_age": e.horse_age,
                "sex_code": e.sex_code,
                "jockey_name": e.jockey_name,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    m = app_state.get("model")
    return {
        "status": "ok" if m is not None else "loading",
        "model_name": m and m.source,
        "model_version": m and m.version,
    }


@app.post("/predict_online", response_model=PredictOnlineResponse)
def predict_online(req: PredictOnlineRequest) -> PredictOnlineResponse:
    m = _model()
    targets = _entries_to_targets(req.race, [req.entry])
    feats = m.feature_pipeline.transform(targets)
    proba = float(m.estimator.predict_proba(feats[ALL_FEATURES])[0, 1])
    return PredictOnlineResponse(
        horse_name=req.entry.horse_name,
        p_trifecta=proba,
        model_name=m.source,
        model_version=m.version,
        served_at=datetime.now(timezone.utc),
    )


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
    m = _model()
    targets = _entries_to_targets(req.race, req.entries)
    feats = m.feature_pipeline.transform(targets)
    probas = m.estimator.predict_proba(feats[ALL_FEATURES])[:, 1]

    preds = [
        HorsePrediction(horse_name=e.horse_name, p_trifecta=float(p))
        for e, p in zip(req.entries, probas)
    ]
    # Rank by probability descending — useful for the UI / API consumers.
    ordered = sorted(preds, key=lambda p: p.p_trifecta, reverse=True)
    for rank, item in enumerate(ordered, start=1):
        item.rank = rank
    # Preserve original entry order in the response.
    by_name = {p.horse_name: p for p in ordered}
    final = [by_name[e.horse_name] for e in req.entries]

    return PredictBatchResponse(
        race_date=req.race.race_date,
        racetrack_id=req.race.racetrack_id,
        distance_m=req.race.distance_m,
        predictions=final,
        model_name=m.source,
        model_version=m.version,
        served_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
@app.post("/predict_explain", response_model=PredictExplainResponse)
def predict_explain(req: PredictExplainRequest) -> PredictExplainResponse:
    """Return the prediction plus the top-k SHAP-style contributions.

    Uses the booster's ``pred_contribs=True`` path, which is
    mathematically identical to TreeSHAP and avoids the
    ``TreeExplainer(clf)`` failure mode that SHAP 0.49 exhibits with
    XGBoost 2.x classifiers loaded from joblib.
    """
    m = _model()
    targets = _entries_to_targets(req.race, [req.entry])
    feats = m.feature_pipeline.transform(targets)[ALL_FEATURES]

    pre = m.estimator.named_steps["pre"]
    clf = m.estimator.named_steps["clf"]
    X_pre = pre.transform(feats)
    # The ColumnTransformer may return a pandas DataFrame (with `object` dtype
    # on the OHE-expanded categorical columns) depending on the global
    # ``set_config`` state. Coerce to a dense float matrix that DMatrix
    # accepts, and trust the booster as the source of truth for feature
    # names — that's the same view it was trained against.
    booster = clf.get_booster()
    X_pre_arr = np.asarray(getattr(X_pre, "to_numpy", lambda: X_pre)(), dtype=float)
    feat_names = booster.feature_names or list(pre.get_feature_names_out())

    dmat = xgb.DMatrix(X_pre_arr, feature_names=feat_names)
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    base_value = float(contribs[-1])
    feat_contribs = contribs[:-1]

    proba = float(m.estimator.predict_proba(feats)[0, 1])

    order = np.argsort(np.abs(feat_contribs))[::-1][: req.top_k]
    row_values = X_pre_arr[0]
    top = [
        FeatureContribution(
            feature=feat_names[i],
            value=(None if not np.isfinite(row_values[i]) else float(row_values[i])),
            contribution=float(feat_contribs[i]),
        )
        for i in order
    ]
    return PredictExplainResponse(
        horse_name=req.entry.horse_name,
        p_trifecta=proba,
        base_value=base_value,
        top_contributions=top,
        model_name=m.source,
        model_version=m.version,
        served_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
