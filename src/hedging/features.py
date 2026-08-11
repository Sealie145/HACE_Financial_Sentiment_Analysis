"""Structured and ML-ready representations of hedge-detection output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HedgeFeatures:
    """Deterministic hedge features returned by :class:`HedgeDetector`."""

    hedge_flag: bool = False
    hedge_probability: float = 0.0
    detected_terms: list[str] = field(default_factory=list)
    hedge_count: int = 0
    hedge_density: float = 0.0
    hedge_categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def detected_hedge_words(self) -> list[str]:
        """Backward-compatible alias for older UI/service consumers."""
        return self.detected_terms

    def to_dict(self) -> dict[str, Any]:
        return {
            "hedge_flag": self.hedge_flag,
            "hedge_probability": self.hedge_probability,
            "detected_terms": list(self.detected_terms),
            "hedge_count": self.hedge_count,
        }

    def __getitem__(self, key: str) -> Any:
        """Permit the result to be consumed as a dictionary by API callers."""
        return self.to_dict()[key]

    def to_feature_vector(self) -> list[float]:
        """Compatibility helper for a future feature-fusion implementation."""
        return [self.hedge_probability, self.hedge_density, float(self.hedge_count)]


class HedgeFeatureExtractor:
    """Convert detector results to the numeric features for a future learner."""

    def __init__(self, detector: Any | None = None) -> None:
        if detector is None:
            from .detector import HedgeDetector
            detector = HedgeDetector()
        self._detector = detector

    def extract(self, text: str) -> dict[str, float | int]:
        result = self._detector.detect(text)
        return {
            "hedge_flag": int(result.hedge_flag),
            "hedge_probability": result.hedge_probability,
            "hedge_count": result.hedge_count,
        }
