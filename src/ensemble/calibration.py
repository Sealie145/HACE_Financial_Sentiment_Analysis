"""
calibration.py — Temperature Scaling for probability calibration.

Purpose: Improve calibration of confidence estimates (ECE reduction).
Does NOT improve classification accuracy.
This component is optional — see PRD Section 21 for priority guidance.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


class TemperatureScaler:
    """Post-hoc temperature scaling calibration.

    Fits a single scalar temperature T by minimizing NLL on a held-out
    calibration set (typically the validation set).

    T > 1 softens probabilities (less confident).
    T < 1 sharpens probabilities (more confident).
    T = 1 leaves probabilities unchanged.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self._is_fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Find the optimal temperature T on calibration data.

        Args:
            logits: Raw logits of shape (n_samples, n_classes).
            labels: Integer ground-truth labels of shape (n_samples,).

        Returns:
            self
        """
        def nll(T: float) -> float:
            scaled = softmax(logits / max(T, 1e-6), axis=1)
            # Negative log-likelihood
            n = len(labels)
            return -np.sum(np.log(scaled[np.arange(n), labels] + 1e-9)) / n

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        self._is_fitted = True
        print(f"[TemperatureScaler] Optimal T = {self.temperature:.4f}")
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw logits of shape (n_samples, n_classes).

        Returns:
            Calibrated probability matrix of same shape.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before calibrate().")
        return softmax(logits / self.temperature, axis=1)

    def calibrate_proba(self, proba: np.ndarray) -> np.ndarray:
        """Apply temperature scaling using probabilities (re-derives logits via log).

        Args:
            proba: Probability matrix of shape (n_samples, n_classes).

        Returns:
            Calibrated probability matrix.
        """
        logits = np.log(proba + 1e-9)
        return self.calibrate(logits)

    def save(self, path: Path) -> None:
        joblib.dump({"temperature": self.temperature}, path)
        print(f"[TemperatureScaler] Saved T={self.temperature:.4f} to {path}")

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        data = joblib.load(path)
        instance = cls()
        instance.temperature = data["temperature"]
        instance._is_fitted = True
        return instance
