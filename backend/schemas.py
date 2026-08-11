"""
schemas.py — Pydantic request/response models for the HACE API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Financial text to analyze.")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, description="List of financial texts.")


class ExpertPrediction(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
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
