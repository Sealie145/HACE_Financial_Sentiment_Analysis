"""
predictor.py — Load all five experts and run all-expert inference.

All five experts run on every input. No domain gating.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .finbert_expert import FinBERTExpert
from src.utils.config import config


# Canonical expert keys (order matters for feature vector assembly)
EXPERT_KEYS = ["fiqa", "phrasebank", "twitter", "finance_news", "general"]


class ExpertPredictor:
    """Loads all five FinBERT experts and runs them on every input.

    Args:
        models_dir: Root models directory. Defaults to config.models_dir.
        device: Compute device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        models_dir: Path = None,
        device: str | None = None,
    ) -> None:
        self._models_dir = Path(models_dir or config.models_dir)
        self._device = device
        self._experts: dict[str, FinBERTExpert] = {}

    def load_all(self) -> None:
        """Load all five expert models into memory."""
        domain_paths = {
            "fiqa":         self._models_dir / "fiqa",
            "phrasebank":   self._models_dir / "phrasebank",
            "twitter":      self._models_dir / "twitter",
            "finance_news": self._models_dir / "finance_news",
            "general":      self._models_dir / "general_finbert",
        }
        for key, path in domain_paths.items():
            if not path.exists():
                # Fall back to base FinBERT for un-trained experts
                print(f"[ExpertPredictor] {key}: model not found at {path}, loading base FinBERT.")
                path = config.finbert_model_name
            self._experts[key] = FinBERTExpert(
                model_path=path,
                domain=key,
                device=self._device,
            )
        print(f"[ExpertPredictor] Loaded {len(self._experts)} experts.")

    def predict_all(self, text: str) -> dict[str, dict]:
        """Run all five experts on a single text.

        Args:
            text: Preprocessed financial text.

        Returns:
            Dict mapping expert key → prediction dict
            (sentiment, confidence, probabilities).
        """
        self._check_loaded()
        return {key: expert.predict(text) for key, expert in self._experts.items()}

    def predict_proba_all(self, text: str) -> dict[str, list[float]]:
        """Return raw probability lists [P(neg), P(neu), P(pos)] per expert."""
        self._check_loaded()
        return {
            key: expert.predict_proba(text)
            for key, expert in self._experts.items()
        }

    def predict_df(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Run all experts on every row of a DataFrame.

        Returns a new DataFrame with one column per expert per class
        (e.g., fiqa_neg, fiqa_neu, fiqa_pos, ...).
        """
        self._check_loaded()
        rows = []
        for text in df[text_col]:
            proba = self.predict_proba_all(text)
            row = {}
            for key in EXPERT_KEYS:
                p = proba[key]
                row[f"{key}_neg"] = p[0]
                row[f"{key}_neu"] = p[1]
                row[f"{key}_pos"] = p[2]
            rows.append(row)
        return pd.DataFrame(rows, index=df.index)

    def _check_loaded(self) -> None:
        if not self._experts:
            raise RuntimeError("Call load_all() before running predictions.")
