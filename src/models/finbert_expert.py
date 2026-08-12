"""
finbert_expert.py — FinBERT expert model wrapper.

Wraps a fine-tuned (or pre-trained) FinBERT model and exposes a consistent
predict() interface returning probability distributions over three classes.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.config import config


class FinBERTExpert:
    """Wraps a FinBERT classification model for inference.

    Args:
        model_path: Path to a saved model directory or a HF model name.
        domain: Human-readable domain label (e.g., 'fiqa', 'general').
        device: 'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        model_path: str | Path,
        domain: str = "unknown",
        device: str | None = None,
    ) -> None:
        self.domain = domain
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path),use_fast=False)        
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.eval()
        self.model.to(self.device)

        # Map model's id2label to HACE standard — override if needed
        self._id2label = config.id2label

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """Run inference on a single text.

        Args:
            text: Preprocessed financial text.

        Returns:
            Dict with keys: sentiment, confidence, probabilities.
        """
        probs = self._get_probabilities(text)
        pred_id = int(probs.argmax())
        return {
            "sentiment": self._id2label[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "probabilities": {
                "negative": round(float(probs[0]), 4),
                "neutral":  round(float(probs[1]), 4),
                "positive": round(float(probs[2]), 4),
            },
        }

    def predict_proba(self, text: str) -> list[float]:
        """Return [P(neg), P(neu), P(pos)] as a plain list."""
        return self._get_probabilities(text).tolist()

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Run predict() over a list of texts.

        Args:
            texts: List of preprocessed strings.
            batch_size: Number of texts per inference batch.

        Returns:
            List of prediction dicts.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            probs_batch = self._get_probabilities_batch(batch)
            for probs in probs_batch:
                pred_id = int(probs.argmax())
                results.append({
                    "sentiment": self._id2label[pred_id],
                    "confidence": round(float(probs[pred_id]), 4),
                    "probabilities": {
                        "negative": round(float(probs[0]), 4),
                        "neutral":  round(float(probs[1]), 4),
                        "positive": round(float(probs[2]), 4),
                    },
                })
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_probabilities(self, text: str) -> torch.Tensor:
        encoding = self.tokenizer(
            text,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits
        return F.softmax(logits[0], dim=-1).cpu()

    def _get_probabilities_batch(self, texts: list[str]) -> torch.Tensor:
        encoding = self.tokenizer(
            texts,
            max_length=config.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits
        return F.softmax(logits, dim=-1).cpu()
