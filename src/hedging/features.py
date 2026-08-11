"""
features.py — HedgeFeatures dataclass.

Holds the structured output of the hedge detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HedgeFeatures:
    """Structured output from the hedge detector.

    Attributes:
        hedge_probability: Float in [0, 1] representing hedging signal strength.
        hedge_count: Number of hedge cues detected.
        hedge_density: hedge_count / total_token_count (0.0 if no tokens).
        detected_hedge_words: List of matched hedge tokens/phrases.
        hedge_categories: Optional dict mapping category → matched terms.
    """

    hedge_probability: float = 0.0
    hedge_count: int = 0
    hedge_density: float = 0.0
    detected_hedge_words: list[str] = field(default_factory=list)
    hedge_categories: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialization."""
        return {
            "hedge_probability": self.hedge_probability,
            "hedge_count": self.hedge_count,
            "hedge_density": self.hedge_density,
            "detected_hedge_words": self.detected_hedge_words,
            "hedge_categories": self.hedge_categories,
        }

    def to_feature_vector(self) -> list[float]:
        """Return the three numeric features used by the meta-learner.

        Returns:
            [hedge_probability, hedge_density, hedge_count_normalized]
        """
        return [self.hedge_probability, self.hedge_density, float(self.hedge_count)]
