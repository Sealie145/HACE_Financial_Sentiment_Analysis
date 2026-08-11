"""
config.py — Central configuration for HACE.

All paths and hyperparameters are defined here.
Override via environment variables or by subclassing HACEConfig.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root is two levels up from this file (src/utils/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class HACEConfig:
    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = int(os.getenv("HACE_SEED", 42))

    # ── Model ─────────────────────────────────────────────────────────────
    finbert_model_name: str = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
    max_seq_length: int = 128

    # ── Training hyperparameters (defaults) ───────────────────────────────
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # ── Label mapping ─────────────────────────────────────────────────────
    label2id: dict = field(default_factory=lambda: {
        "negative": 0, "neutral": 1, "positive": 2
    })
    id2label: dict = field(default_factory=lambda: {
        0: "negative", 1: "neutral", 2: "positive"
    })
    num_labels: int = 3

    # ── Paths ──────────────────────────────────────────────────────────────
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_interim_dir: Path = PROJECT_ROOT / "data" / "interim"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    # ── Dataset source keys ───────────────────────────────────────────────
    domains: tuple = ("fiqa", "phrasebank", "twitter", "finance_news")

    # ── Split ratios ──────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def model_dir(self, domain: str) -> Path:
        return self.models_dir / domain

    def processed_dir(self, domain: str) -> Path:
        return self.data_processed_dir / domain


# Singleton instance — import this in notebooks and modules
config = HACEConfig()
