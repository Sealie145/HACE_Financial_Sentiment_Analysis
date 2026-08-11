"""Inference service shell with real hedge detection and mock sentiment outputs.

Trained FinBERT experts and the HACE meta-learner are deliberately not loaded
here.  They can replace the mock prediction section without changing the API.
"""

from __future__ import annotations

from typing import Protocol

from src.hedging.detector import HedgeDetector


class ExpertPredictor(Protocol):
    """Interface that trained expert adapters will implement later."""

    def predict_all(self, text: str) -> dict[str, dict]: ...


class MockExpertPredictor:
    """MOCK ONLY — replace with trained FinBERT expert adapters later."""

    _PREDICTIONS = {
        "fiqa": ("positive", 0.82),
        "twitter": ("positive", 0.79),
        "phrasebank": ("positive", 0.87),
        "finance_news": ("neutral", 0.61),
    }

    def predict_all(self, text: str) -> dict[str, dict]:
        """Return stable demo values; text is unused by design for now."""
        return {
            name: {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": self._probabilities(sentiment, confidence),
            }
            for name, (sentiment, confidence) in self._PREDICTIONS.items()
        }

    @staticmethod
    def _probabilities(sentiment: str, confidence: float) -> dict[str, float]:
        remaining = round((1.0 - confidence) / 2, 2)
        probabilities = {"negative": remaining, "neutral": remaining, "positive": remaining}
        probabilities[sentiment] = confidence
        return probabilities


class HACEService:
    """Provide an API-facing prediction interface for the HACE application."""

    def __init__(self, expert_predictor: ExpertPredictor | None = None) -> None:
        self._hedge_detector = HedgeDetector()
        self._expert_predictor = expert_predictor or MockExpertPredictor()
        self._loaded = False

    def load(self) -> None:
        """Mark the lightweight service ready; no ML models are loaded."""
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> dict:
        """Return real hedging features and deterministic mock predictions."""
        if not self._loaded:
            raise RuntimeError("HACEService is not loaded. Call load() first.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must not be empty.")

        hedge = self._hedge_detector.detect(text)
        experts = self._expert_predictor.predict_all(text)
        hedging = hedge.to_dict()
        return {
            "text": text,
            "sentiment": "positive",
            "confidence": 0.84,
            "hedging": hedging,
            # Legacy flat fields remain available for the existing Gradio formatter.
            "hedge_probability": hedge.hedge_probability,
            "hedge_count": hedge.hedge_count,
            "hedge_density": hedge.hedge_density,
            "hedge_words": hedge.detected_terms,
            "experts": experts,
            "expert_agreement": 1.0,
            "base_ensemble": {"sentiment": "positive", "confidence": 0.84},
            "explanation": [
                "Sentiment and expert outputs are deterministic placeholders.",
                "Hedging features are produced by the rule-based lexicon detector.",
            ],
        }
