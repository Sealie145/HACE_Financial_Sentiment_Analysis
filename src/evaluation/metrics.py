"""
metrics.py — Standard evaluation metrics for HACE experiments.

Primary metric: Macro F1 (treats all classes equally regardless of imbalance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    label_names: list[str] = None,
) -> dict:
    """Compute the full HACE evaluation metric set.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        label_names: Optional class names for reporting.

    Returns:
        Dict with keys: accuracy, precision_macro, recall_macro,
        f1_macro, f1_weighted, confusion_matrix, classification_report.
    """
    label_names = label_names or ["negative", "neutral", "positive"]
    return {
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "precision_macro":  round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":     round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":         round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_weighted":      round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0
        ),
    }


def print_metrics_table(results: dict[str, dict]) -> None:
    """Print a formatted comparison table.

    Args:
        results: Dict mapping model_name → metrics dict from compute_metrics().
    """
    rows = []
    for model, m in results.items():
        rows.append({
            "Model":       model,
            "Accuracy":    m.get("accuracy", ""),
            "Precision":   m.get("precision_macro", ""),
            "Recall":      m.get("recall_macro", ""),
            "F1 Macro":    m.get("f1_macro", ""),
            "F1 Weighted": m.get("f1_weighted", ""),
        })
    df = pd.DataFrame(rows).set_index("Model")
    print(df.to_string())
