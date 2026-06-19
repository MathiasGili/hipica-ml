"""Pydantic schemas for the FastAPI service."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
class HorseEntryIn(BaseModel):
    """One horse entered in the target race."""

    horse_name: str = Field(..., min_length=1, max_length=120)
    kg: float = Field(..., gt=30, lt=80, description="Kilos to be carried")
    post_position: int | None = Field(default=None, ge=1, le=25)
    horse_age: int | None = Field(default=None, ge=2, le=20)
    sex_code: Literal["M", "H"] | None = None
    jockey_name: str | None = Field(default=None, max_length=120)


class RaceContextIn(BaseModel):
    """Metadata of the race that is shared across all entrants."""

    race_date: date
    racetrack_id: int = Field(..., description="1=Maroñas, 13=Las Piedras, ...")
    distance_m: int = Field(..., ge=600, le=4000)


class PredictOnlineRequest(BaseModel):
    """One horse + race context."""

    race: RaceContextIn
    entry: HorseEntryIn


class PredictBatchRequest(BaseModel):
    """All horses entered in the same race — needed for in-race z-score."""

    race: RaceContextIn
    entries: list[HorseEntryIn] = Field(..., min_length=1, max_length=25)

    @field_validator("entries")
    @classmethod
    def unique_horses(cls, v: list[HorseEntryIn]) -> list[HorseEntryIn]:
        names = [e.horse_name.upper().strip() for e in v]
        if len(set(names)) != len(names):
            raise ValueError("Horse names must be unique within a race")
        return v


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
class HorsePrediction(BaseModel):
    horse_name: str
    p_trifecta: float = Field(..., ge=0, le=1)
    rank: int | None = Field(default=None, description="1=highest probability in race")


class PredictOnlineResponse(BaseModel):
    horse_name: str
    p_trifecta: float
    model_name: str
    model_version: str | None
    served_at: datetime


class PredictBatchResponse(BaseModel):
    race_date: date
    racetrack_id: int
    distance_m: int
    predictions: list[HorsePrediction]
    model_name: str
    model_version: str | None
    served_at: datetime


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
class PredictExplainRequest(BaseModel):
    """Same shape as :class:`PredictOnlineRequest` plus a top_k knob."""

    race: RaceContextIn
    entry: HorseEntryIn
    top_k: int = Field(default=10, ge=1, le=40,
                       description="Number of top contributions to return")


class FeatureContribution(BaseModel):
    feature: str
    value: float | None = Field(default=None,
                                description="Feature value after preprocessing (NaN -> None)")
    contribution: float = Field(..., description="SHAP value in log-odds space")


class PredictExplainResponse(BaseModel):
    horse_name: str
    p_trifecta: float
    base_value: float = Field(..., description="Model bias in log-odds space")
    top_contributions: list[FeatureContribution]
    model_name: str
    model_version: str | None
    served_at: datetime


# ---------------------------------------------------------------------------
# Race-day program (live scrape + predict)
# ---------------------------------------------------------------------------
class PredictProgramRequest(BaseModel):
    """Ask the API to scrape and predict a whole race day."""

    race_date: date
    racetrack_id: int = Field(default=1, description="1=Mara\u00f1as, 13=Las Piedras, ...")
    force_refresh: bool = Field(default=False,
                                description="Re-download the Programa even if it exists on disk")


class ProgramHorsePrediction(BaseModel):
    horse_name: str
    post_position: int
    kg: float
    horse_age: int | None = None
    sex_code: str | None = None
    jockey_name: str | None = None
    p_trifecta: float = Field(..., ge=0, le=1)
    rank: int


class ProgramRacePrediction(BaseModel):
    race_index: int = Field(..., ge=1, description="1-based within the day")
    distance_m: int
    post_time: str | None = None
    predictions: list[ProgramHorsePrediction]


class PredictProgramResponse(BaseModel):
    race_date: date
    racetrack_id: int
    n_races: int
    races: list[ProgramRacePrediction]
    model_name: str
    model_version: str | None
    served_at: datetime
