"""
service.py — Real HACE inference service.

Loads:
- Five trained FinBERT experts
- HACE meta-learner
- Hedge detector
- Feature fusion
"""

from __future__ import annotations

import numpy as np

from src.models.predictor import ExpertPredictor
from src.ensemble.feature_fusion import FeatureFusion
from src.ensemble.meta_learner import MetaLearner
from src.hedging.detector import HedgeDetector
from src.utils.config import config


class HACEService:

    def __init__(self) -> None:
        self._expert_predictor = None
        self._hace_model = None

        self._hedge_detector = HedgeDetector()

        self._hace_fusion = FeatureFusion(
            use_hedge_features=True
        )

        self._loaded = False

    # ============================================================
    # LOAD MODELS
    # ============================================================

    def load(self) -> None:

        print("=" * 60)
        print("LOADING HACE MODELS")
        print("=" * 60)

        # --------------------------------------------------------
        # Load five experts
        # --------------------------------------------------------

        print("\nLoading expert models...")

        self._expert_predictor = ExpertPredictor(
            models_dir=config.models_dir,
            device=None
        )

        self._expert_predictor.load_all()

        print("✓ Five expert models loaded")

        # --------------------------------------------------------
        # Load HACE meta-learner
        # --------------------------------------------------------

        meta_path = (
            config.models_dir
            / "meta_learner"
            / "hace_meta_learner.pkl"
        )

        if not meta_path.exists():
            raise FileNotFoundError(
                f"HACE meta-learner not found: {meta_path}"
            )

        print("\nLoading HACE meta-learner...")
        print("Path:", meta_path)

        self._hace_model = MetaLearner.load(
            meta_path,
            use_hedge_features=True
        )

        print("✓ HACE meta-learner loaded")

        self._loaded = True

        print("\n" + "=" * 60)
        print("✓ HACE SERVICE READY")
        print("=" * 60)

    # ============================================================
    # STATUS
    # ============================================================

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(self, text: str) -> dict:

        if not self._loaded:
            raise RuntimeError(
                "HACEService is not loaded. Call load() first."
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must not be empty.")

        # --------------------------------------------------------
        # 1. Expert predictions
        # --------------------------------------------------------

        expert_predictions = (
            self._expert_predictor.predict_all(text)
        )

        expert_probas = (
            self._expert_predictor.predict_proba_all(text)
        )

        # --------------------------------------------------------
        # 2. Hedge detection
        # --------------------------------------------------------

        hedge = self._hedge_detector.detect(text)

        # --------------------------------------------------------
        # 3. Build 21 HACE features
        # --------------------------------------------------------

        X = self._hace_fusion.assemble(
            expert_probas=expert_probas,
            hedge_features=hedge,
            text=text
        )

        assert X.shape == (21,)

        # --------------------------------------------------------
        # 4. HACE prediction
        # --------------------------------------------------------

        prediction = self._hace_model.predict_single(X)

        # --------------------------------------------------------
        # 5. Expert agreement
        # --------------------------------------------------------

        expert_labels = [
            int(np.argmax(expert_probas[key]))
            for key in expert_probas
        ]

        if expert_labels:

            plurality = max(
                set(expert_labels),
                key=expert_labels.count
            )

            agreement = (
                expert_labels.count(plurality)
                / len(expert_labels)
            )

        else:
            agreement = 0.0

        # --------------------------------------------------------
        # 6. Explanation
        # --------------------------------------------------------

        explanation = []

        explanation.append(
            f"HACE predicted {prediction['sentiment']} "
            f"sentiment with "
            f"{prediction['confidence']:.4f} confidence."
        )

        if hedge.hedge_flag:

            explanation.append(
                "Hedging language detected: "
                + ", ".join(hedge.detected_terms)
            )

        else:

            explanation.append(
                "No predefined financial hedging cues detected."
            )

        # --------------------------------------------------------
        # 7. Response
        # --------------------------------------------------------

        return {
            "text": text,

            "sentiment": prediction["sentiment"],

            "confidence": prediction["confidence"],

            "hedging": hedge.to_dict(),

            "hedge_probability": hedge.hedge_probability,

            "hedge_count": hedge.hedge_count,

            "hedge_density": hedge.hedge_density,

            "hedge_words": hedge.detected_terms,

            "experts": expert_predictions,

            "expert_agreement": round(
                float(agreement),
                4
            ),

            "base_ensemble": {
                "sentiment": prediction["sentiment"],
                "confidence": prediction["confidence"],
            },

            "explanation": explanation,
        }