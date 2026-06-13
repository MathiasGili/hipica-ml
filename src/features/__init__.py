"""Feature-engineering sub-package.

Importing :class:`FeatureEngineeringPipeline` from this module gives the
**same** transformer to the training notebook and the FastAPI serving app,
which is the contract that prevents training-serving skew.
"""
from .pipeline import (
    FeatureEngineeringPipeline,
    HorseEntry,
    RaceContext,
    build_training_frame,
)

__all__ = [
    "FeatureEngineeringPipeline",
    "HorseEntry",
    "RaceContext",
    "build_training_frame",
]
