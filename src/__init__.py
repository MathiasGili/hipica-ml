"""Hipica-ML: end-to-end ML pipeline for predicting Trifecta finishes.

The package is intentionally split into three layers:

* `ingestion` — talks to the public XTurf REST endpoints behind
  https://hipica.maronas.com.uy/ and parses the per-race-day Excel
  ("tabulada") files into a tidy long-form DataFrame.
* `features` — pure pandas/sklearn transformations that turn the long-form
  history into a model-ready matrix. **This module is imported by both the
  training notebook and the FastAPI app**, which is the contract that
  prevents training-serving skew.
* `training` — temporal train/test split, XGBoost training and MLflow
  logging. Importing the trained model from MLflow Model Registry is
  handled by `api.model_loader`.
"""

__version__ = "0.1.0"
