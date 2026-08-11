"""
hedged_analysis.py — Hedged vs. non-hedged evaluation.

Partitions the test set by hedge detection output and computes
separate metrics for each partition.

This is the key experiment for testing the primary research hypothesis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import config
from .metrics import compute_metrics


class HedgedAnalysis:
    """Evaluate Base Ensemble and HACE separately on hedged and non-hedged samples.

    Args:
        hedge_threshold: hedge_probability threshold above which a sample is
            classified as hedged. Defaults to 0.5 (presence of any hedge cue).
    """

    def __init__(self, hedge_threshold: float = 0.5) -> None:
        self.hedge_threshold = hedge_threshold
        self._results: dict = {}

    def evaluate(
        self,
        y_true: list[int] | np.ndarray,
        hedge_probs: list[float] | np.ndarray,
        y_pred_base: list[int] | np.ndarray,
        y_pred_hace: list[int] | np.ndarray,
    ) -> dict:
        """Run the hedged vs. non-hedged analysis.

        Args:
            y_true: Ground-truth labels for the full test set.
            hedge_probs: hedge_probability values for each test sample.
            y_pred_base: Base Ensemble predictions.
            y_pred_hace: HACE predictions.

        Returns:
            Nested dict with keys: hedged, non_hedged, overall.
            Each contains metrics for both Base Ensemble and HACE.
        """
        y_true = np.array(y_true)
        hedge_probs = np.array(hedge_probs)
        y_pred_base = np.array(y_pred_base)
        y_pred_hace = np.array(y_pred_hace)

        hedged_mask = hedge_probs >= self.hedge_threshold
        non_hedged_mask = ~hedged_mask

        n_hedged = int(hedged_mask.sum())
        n_non_hedged = int(non_hedged_mask.sum())
        print(f"[HedgedAnalysis] Hedged: {n_hedged} | Non-hedged: {n_non_hedged} | Total: {len(y_true)}")

        self._results = {
            "hedged": {
                "n_samples":    n_hedged,
                "base_ensemble": compute_metrics(y_true[hedged_mask], y_pred_base[hedged_mask]) if n_hedged > 0 else {},
                "hace":          compute_metrics(y_true[hedged_mask], y_pred_hace[hedged_mask]) if n_hedged > 0 else {},
            },
            "non_hedged": {
                "n_samples":    n_non_hedged,
                "base_ensemble": compute_metrics(y_true[non_hedged_mask], y_pred_base[non_hedged_mask]) if n_non_hedged > 0 else {},
                "hace":          compute_metrics(y_true[non_hedged_mask], y_pred_hace[non_hedged_mask]) if n_non_hedged > 0 else {},
            },
            "overall": {
                "n_samples":    len(y_true),
                "base_ensemble": compute_metrics(y_true, y_pred_base),
                "hace":          compute_metrics(y_true, y_pred_hace),
            },
        }
        return self._results

    def report(self) -> None:
        """Print the hedged vs. non-hedged comparison table."""
        if not self._results:
            print("[HedgedAnalysis] No results. Call evaluate() first.")
            return

        print("\n=== Hedged vs. Non-Hedged Evaluation ===")
        print(f"{'Metric':<25} {'Base Ensemble':>15} {'HACE':>10} {'Δ':>8}")
        print("-" * 62)

        partitions = [
            ("Hedged F1 Macro",        "hedged",     "f1_macro"),
            ("Hedged Accuracy",         "hedged",     "accuracy"),
            ("Non-Hedged F1 Macro",     "non_hedged", "f1_macro"),
            ("Non-Hedged Accuracy",     "non_hedged", "accuracy"),
            ("Overall F1 Macro",        "overall",    "f1_macro"),
            ("Overall Accuracy",        "overall",    "accuracy"),
        ]

        for label, partition, metric in partitions:
            base_val = self._results[partition]["base_ensemble"].get(metric, float("nan"))
            hace_val = self._results[partition]["hace"].get(metric, float("nan"))
            delta = hace_val - base_val if isinstance(hace_val, float) else float("nan")
            print(f"{label:<25} {base_val:>15.4f} {hace_val:>10.4f} {delta:>+8.4f}")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for partition in ["hedged", "non_hedged", "overall"]:
            for model in ["base_ensemble", "hace"]:
                m = self._results.get(partition, {}).get(model, {})
                rows.append({
                    "partition": partition,
                    "model":     model,
                    "n_samples": self._results.get(partition, {}).get("n_samples", 0),
                    "accuracy":  m.get("accuracy"),
                    "f1_macro":  m.get("f1_macro"),
                })
        return pd.DataFrame(rows)

    def save(self, path: Path = None) -> Path:
        path = path or (config.outputs_dir / "metrics" / "hedged_analysis.csv")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[HedgedAnalysis] Saved to {path}")
        return path
