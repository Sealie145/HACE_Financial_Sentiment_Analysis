"""
ablation.py — Ablation study runner.

Evaluates all required configurations:
  1. General FinBERT
  2. Best individual expert
  3. Base Ensemble (no hedge features)
  4. HACE without hedge features (same as base ensemble — verify during Phase 6)
  5. HACE with hedge features
  6. HACE + calibration (optional)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import config
from .metrics import compute_metrics, print_metrics_table


class AblationStudy:
    """Collect and report ablation results.

    Usage:
        ablation = AblationStudy()
        ablation.add("General FinBERT", y_true, y_pred_general)
        ablation.add("Base Ensemble", y_true, y_pred_base)
        ablation.add("HACE", y_true, y_pred_hace)
        ablation.report()
        ablation.save()
    """

    def __init__(self) -> None:
        self._results: dict[str, dict] = {}

    def add(
        self,
        name: str,
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
    ) -> None:
        """Register predictions for one ablation configuration.

        Args:
            name: Human-readable configuration name.
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
        """
        self._results[name] = compute_metrics(y_true, y_pred)

    def report(self) -> None:
        """Print the ablation table to stdout."""
        print("\n=== Ablation Study ===")
        print_metrics_table(self._results)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, m in self._results.items():
            rows.append({
                "Configuration": name,
                "Accuracy":      m["accuracy"],
                "Precision":     m["precision_macro"],
                "Recall":        m["recall_macro"],
                "F1 Macro":      m["f1_macro"],
                "F1 Weighted":   m["f1_weighted"],
            })
        return pd.DataFrame(rows)

    def save(self, path: Path = None) -> Path:
        path = path or (config.outputs_dir / "metrics" / "ablation_results.csv")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[AblationStudy] Saved to {path}")
        return path
