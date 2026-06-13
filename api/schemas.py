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
