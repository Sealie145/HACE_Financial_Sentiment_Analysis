"""
schemas.py — Pydantic request/response models for the HACE API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Financial text to analyze.")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank.")
        return value


PredictRequest = PredictionRequest


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of financial texts.")


class ExpertPrediction(BaseModel):
    sentiment: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class HedgingResponse(BaseModel):
    hedge_flag: bool
    hedge_probability: float = Field(ge=0.0, le=1.0)
    detected_terms: list[str]
    hedge_count: int


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float = Field(ge=0.0, le=1.0)
    hedging: HedgingResponse
    hedge_probability: float
    hedge_count: int
    hedge_density: float
    hedge_words: list[str]
    experts: dict[str, ExpertPrediction]
    expert_agreement: float
    base_ensemble: dict
    explanation: list[str]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
