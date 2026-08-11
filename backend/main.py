"""
main.py — FastAPI application for HACE.

Endpoints:
  POST /predict
  POST /batch_predict
  GET  /health
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.schemas import (
    BatchPredictRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictRequest,
    PredictionResponse,
)
from backend.service import HACEService

# ── Service singleton ─────────────────────────────────────────────────────────

service = HACEService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup."""
    service.load()
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HACE — Hedging-Aware Cross-Domain Ensemble",
    description="Financial Sentiment Analysis with Uncertainty Awareness",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health check."""
    return HealthResponse(status="ok", models_loaded=service.is_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    """Run the full HACE inference pipeline on a single text."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty.")

    try:
        result = service.predict(request.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")

    return PredictionResponse(**result)


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictRequest) -> BatchPredictionResponse:
    """Run inference on a list of texts."""
    predictions = []
    for text in request.texts:
        if not text or not text.strip():
            raise HTTPException(status_code=422, detail="All texts must be non-empty.")
        try:
            result = service.predict(text)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return BatchPredictionResponse(predictions=predictions)
