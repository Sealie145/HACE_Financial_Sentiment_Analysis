"""
cross_domain.py — Cross-domain evaluation matrix.

Evaluates each domain-trained expert on every other domain's test set.
Produces a 4×4 Macro F1 matrix and saves it to outputs/metrics/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import config
from .metrics import compute_metrics


class CrossDomainEvaluator:
    """Build and save the cross-domain evaluation matrix.

    Args:
        predictor: Loaded ExpertPredictor instance.
        test_sets: Dict mapping domain → test DataFrame (text, label columns).
    """

    DOMAINS = ["fiqa", "phrasebank", "twitter", "finance_news"]

    def __init__(self, predictor, test_sets: dict[str, pd.DataFrame]) -> None:
        self._predictor = predictor
        self._test_sets = test_sets

    def evaluate(self) -> pd.DataFrame:
        """Run all cross-domain combinations and return a results DataFrame.

        Returns:
            DataFrame with columns: train_domain, test_domain, accuracy,
            precision_macro, recall_macro, f1_macro.
        """
        rows = []
        for train_domain in self.DOMAINS:
            for test_domain in self.DOMAINS:
                if train_domain == test_domain:
                    continue  # Skip in-domain; add separately if needed

                test_df = self._test_sets.get(test_domain)
                if test_df is None:
                    print(f"[CrossDomain] Skipping {train_domain}→{test_domain}: no test set.")
                    continue

                # Use the single expert trained on train_domain
                expert = self._predictor._experts.get(train_domain)
                if expert is None:
                    print(f"[CrossDomain] Expert '{train_domain}' not loaded.")
                    continue

                preds = [expert.predict(t)["sentiment"] for t in test_df["text"]]
                label2id = config.label2id
                y_pred = [label2id[p] for p in preds]
                y_true = list(test_df["label"])

                m = compute_metrics(y_true, y_pred)
                rows.append({
                    "train_domain":    train_domain,
                    "test_domain":     test_domain,
                    "accuracy":        m["accuracy"],
                    "precision_macro": m["precision_macro"],
                    "recall_macro":    m["recall_macro"],
                    "f1_macro":        m["f1_macro"],
                })

        return pd.DataFrame(rows)

    def save(self, df: pd.DataFrame, path: Path = None) -> Path:
        path = path or (config.outputs_dir / "metrics" / "cross_domain_results.csv")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[CrossDomain] Saved to {path}")
        return path
