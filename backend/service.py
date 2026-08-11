"""
service.py — HACE inference service.

Loads all models once at startup and exposes a single predict() method
used by both the FastAPI backend and the Gradio frontend.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import config
from src.utils.seed import set_seed
from src.preprocessing.cleaner import clean_text
from src.hedging.detector import HedgeDetector
from src.models.predictor import ExpertPredictor
from src.ensemble.feature_fusion import FeatureFusion
from src.ensemble.meta_learner import MetaLearner


class HACEService:
    """Singleton inference service for HACE.

    Args:
        models_dir: Root models directory.
        device: 'cuda' or 'cpu'.
    """

    def __init__(self, models_dir: Path = None, device: str | None = None) -> None:
        set_seed(config.seed)
        self._models_dir = Path(models_dir or config.models_dir)
        self._device = device
        self._loaded = False

        self._hedge_detector: HedgeDetector | None = None
        self._predictor: ExpertPredictor | None = None
        self._base_fusion: FeatureFusion | None = None
        self._hace_fusion: FeatureFusion | None = None
        self._base_learner: MetaLearner | None = None
        self._hace_learner: MetaLearner | None = None

    def load(self) -> None:
        """Load all models into memory. Call once at application startup."""
        print("[HACEService] Loading hedge detector...")
        self._hedge_detector = HedgeDetector()

        print("[HACEService] Loading expert models...")
        self._predictor = ExpertPredictor(self._models_dir, self._device)
        self._predictor.load_all()

        self._base_fusion = FeatureFusion(use_hedge_features=False)
        self._hace_fusion = FeatureFusion(use_hedge_features=True)

        meta_dir = self._models_dir / "meta_learner"
        base_path = meta_dir / "base_ensemble.pkl"
        hace_path = meta_dir / "hace_meta_learner.pkl"

        if base_path.exists():
            self._base_learner = MetaLearner.load(base_path, use_hedge_features=False)
        else:
            print(f"[HACEService] Warning: {base_path} not found. Base Ensemble unavailable.")

        if hace_path.exists():
            self._hace_learner = MetaLearner.load(hace_path, use_hedge_features=True)
        else:
            print(f"[HACEService] Warning: {hace_path} not found. HACE meta-learner unavailable.")

        self._loaded = True
        print("[HACEService] Ready.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> dict:
        """Run the full HACE inference pipeline.

        Args:
            text: Raw financial text.

        Returns:
            Full prediction response dict (matches PredictionResponse schema).
        """
        self._check_loaded()

        # 1. Preprocess
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Input text is empty after preprocessing.")

        # 2. Hedge detection
        hedge_features = self._hedge_detector.detect(cleaned)

        # 3. All-expert inference
        expert_probas = self._predictor.predict_proba_all(cleaned)
        expert_preds = self._predictor.predict_all(cleaned)

        # 4. Feature fusion
        base_vec = self._base_fusion.assemble(expert_probas, text=cleaned)
        hace_vec = self._hace_fusion.assemble(expert_probas, hedge_features, cleaned)

        # 5. Meta-learner predictions
        base_result = (
            self._base_learner.predict_single(base_vec)
            if self._base_learner else {"sentiment": "unavailable", "confidence": 0.0}
        )
        hace_result = (
            self._hace_learner.predict_single(hace_vec)
            if self._hace_learner else {"sentiment": "unavailable", "confidence": 0.0}
        )

        # 6. Explanation
        explanation = self._build_explanation(expert_preds, hedge_features, hace_result)

        # 7. Expert agreement
        sentiments = [v["sentiment"] for v in expert_preds.values()]
        plurality = max(set(sentiments), key=sentiments.count)
        agreement = sentiments.count(plurality) / len(sentiments)

        return {
            "sentiment":         hace_result["sentiment"],
            "confidence":        hace_result["confidence"],
            "hedge_probability": hedge_features.hedge_probability,
            "hedge_count":       hedge_features.hedge_count,
            "hedge_density":     hedge_features.hedge_density,
            "hedge_words":       hedge_features.detected_hedge_words,
            "experts":           expert_preds,
            "expert_agreement":  round(agreement, 2),
            "base_ensemble":     base_result,
            "explanation":       explanation,
        }

    @staticmethod
    def _build_explanation(
        expert_preds: dict,
        hedge_features,
        hace_result: dict,
    ) -> list[str]:
        explanation = []
        sentiments = [v["sentiment"] for v in expert_preds.values()]
        plurality = max(set(sentiments), key=sentiments.count)
        count = sentiments.count(plurality)
        explanation.append(f"{count} of {len(sentiments)} experts predict {plurality.capitalize()}.")

        if hedge_features.hedge_count > 0:
            words = ", ".join(f"'{w}'" for w in hedge_features.detected_hedge_words[:3])
            explanation.append(f"Hedge cue(s) detected: {words}.")
            explanation.append(
                f"Hedge probability: {hedge_features.hedge_probability:.2f}. "
                "HACE incorporates this uncertainty signal."
            )
        else:
            explanation.append("No hedging language detected.")

        explanation.append(
            f"Final HACE prediction: {hace_result['sentiment'].capitalize()} "
            f"(confidence: {hace_result['confidence']:.2f})."
        )
        return explanation

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("HACEService is not loaded. Call load() first.")
