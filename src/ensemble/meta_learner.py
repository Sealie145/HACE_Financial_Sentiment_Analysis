"""
meta_learner.py — Stacking meta-learner (Logistic Regression).

Trains and serializes two models:
  base_ensemble.pkl  — no hedge features
  hace_meta_learner.pkl — with hedge features
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.config import config


class MetaLearner:
    """Logistic Regression meta-learner for the HACE stacking ensemble.

    Args:
        use_hedge_features: Determines the artifact name and documents intent.
        C: Inverse regularization strength (LogReg hyperparameter).
        max_iter: Maximum number of solver iterations.
        seed: Random seed.
    """

    def __init__(
        self,
        use_hedge_features: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        seed: int = None,
    ) -> None:
        self.use_hedge_features = use_hedge_features
        self.seed = seed or config.seed
        self._artifact_name = (
            "hace_meta_learner.pkl" if use_hedge_features else "base_ensemble.pkl"
        )

        # Pipeline: StandardScaler → LogisticRegression
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=self.seed,
            )),
        ])
        self._is_fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MetaLearner":
        """Fit the meta-learner pipeline.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Integer labels (0/1/2).

        Returns:
            self
        """
        self._pipeline.fit(X, y)
        self._is_fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix of shape (n_samples, 3)."""
        self._check_fitted()
        return self._pipeline.predict_proba(X)

    def predict_single(self, x: np.ndarray) -> dict:
        """Predict a single feature vector.

        Returns:
            Dict with keys: sentiment, confidence, probabilities.
        """
        self._check_fitted()
        x = x.reshape(1, -1)
        proba = self._pipeline.predict_proba(x)[0]
        pred_id = int(np.argmax(proba))
        id2label = config.id2label
        return {
            "sentiment": id2label[pred_id],
            "confidence": round(float(proba[pred_id]), 4),
            "probabilities": {
                "negative": round(float(proba[0]), 4),
                "neutral":  round(float(proba[1]), 4),
                "positive": round(float(proba[2]), 4),
            },
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path = None) -> Path:
        """Serialize the fitted pipeline to disk.

        Args:
            directory: Save directory. Defaults to models/meta_learner/.

        Returns:
            Path to the saved artifact.
        """
        self._check_fitted()
        directory = Path(directory or (config.models_dir / "meta_learner"))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self._artifact_name
        joblib.dump(self._pipeline, path)
        print(f"[MetaLearner] Saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path, use_hedge_features: bool = True) -> "MetaLearner":
        """Load a saved meta-learner from disk.

        Args:
            path: Path to the .pkl artifact.
            use_hedge_features: Passed to __init__ for context only.

        Returns:
            MetaLearner instance with a fitted pipeline.
        """
        instance = cls(use_hedge_features=use_hedge_features)
        instance._pipeline = joblib.load(path)
        instance._is_fitted = True
        return instance

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("MetaLearner is not fitted. Call fit() first.")
