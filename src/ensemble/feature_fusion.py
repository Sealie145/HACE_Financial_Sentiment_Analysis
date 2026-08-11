"""
feature_fusion.py — Assemble the meta-learner feature vector.

Base Ensemble (18 features):
  15 expert probabilities + 3 ensemble features

HACE (21 features):
  15 expert probabilities + 3 ensemble features + 3 hedge features

Feature vector order is fixed and must be identical across training and inference.
"""

from __future__ import annotations

import math

import numpy as np

from src.models.predictor import EXPERT_KEYS
from src.hedging.features import HedgeFeatures


# Feature vector indices (for documentation and debugging)
FEATURE_NAMES_BASE = (
    [f"{k}_{c}" for k in EXPERT_KEYS for c in ("neg", "neu", "pos")]  # 15
    + ["token_length", "prediction_entropy", "expert_agreement"]        # 3
)
FEATURE_NAMES_HACE = FEATURE_NAMES_BASE + [
    "hedge_probability", "hedge_density", "hedge_count"                 # 3
]


class FeatureFusion:
    """Assemble the meta-learner input vector from expert predictions and hedge features.

    Args:
        use_hedge_features: If True, include hedge features (HACE mode).
            If False, produce the Base Ensemble feature vector.
    """

    def __init__(self, use_hedge_features: bool = True) -> None:
        self.use_hedge_features = use_hedge_features

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES_HACE if self.use_hedge_features else FEATURE_NAMES_BASE

    def assemble(
        self,
        expert_probas: dict[str, list[float]],
        hedge_features: HedgeFeatures | None = None,
        text: str = "",
    ) -> np.ndarray:
        """Build the feature vector for one sample.

        Args:
            expert_probas: Dict mapping expert key → [P(neg), P(neu), P(pos)].
            hedge_features: HedgeFeatures instance (required for HACE mode).
            text: Original text (used for token_length computation).

        Returns:
            1D numpy array of floats.
        """
        # ── Expert probabilities (15) ──────────────────────────────────────
        expert_vec = []
        all_preds = []
        for key in EXPERT_KEYS:
            p = expert_probas[key]          # [neg, neu, pos]
            expert_vec.extend(p)
            all_preds.append(int(np.argmax(p)))

        # ── Ensemble features (3) ──────────────────────────────────────────
        token_length = float(len(text.split())) if text else 0.0
        entropy = self._mean_entropy(expert_probas)
        agreement = self._expert_agreement(all_preds)

        ensemble_vec = [token_length, entropy, agreement]

        # ── Hedge features (3, HACE only) ──────────────────────────────────
        if self.use_hedge_features:
            if hedge_features is None:
                raise ValueError("hedge_features is required when use_hedge_features=True.")
            hedge_vec = hedge_features.to_feature_vector()  # [prob, density, count]
        else:
            hedge_vec = []

        return np.array(expert_vec + ensemble_vec + hedge_vec, dtype=np.float32)

    def assemble_batch(
        self,
        expert_probas_list: list[dict[str, list[float]]],
        hedge_features_list: list[HedgeFeatures] | None = None,
        texts: list[str] | None = None,
    ) -> np.ndarray:
        """Assemble feature matrix for a batch of samples.

        Returns:
            2D numpy array of shape (n_samples, n_features).
        """
        n = len(expert_probas_list)
        hedge_features_list = hedge_features_list or [None] * n
        texts = texts or [""] * n

        rows = [
            self.assemble(ep, hf, t)
            for ep, hf, t in zip(expert_probas_list, hedge_features_list, texts)
        ]
        return np.vstack(rows)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _mean_entropy(expert_probas: dict[str, list[float]]) -> float:
        """Mean Shannon entropy across all expert probability distributions."""
        entropies = []
        for p_list in expert_probas.values():
            h = -sum(p * math.log(p + 1e-9) for p in p_list)
            entropies.append(h)
        return float(np.mean(entropies))

    @staticmethod
    def _expert_agreement(predictions: list[int]) -> float:
        """Fraction of experts agreeing with the plurality prediction."""
        if not predictions:
            return 0.0
        plurality = max(set(predictions), key=predictions.count)
        return predictions.count(plurality) / len(predictions)
