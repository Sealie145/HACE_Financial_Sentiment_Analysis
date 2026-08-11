"""
splitter.py — Stratified train/validation/test splitting.

Produces three DataFrames with the schema: text | label | source.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import config as default_config


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = None,
    val_ratio: float = None,
    seed: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, validation, and test sets.

    Splitting is stratified on the 'label' column to preserve class balance.

    Args:
        df: Standardized DataFrame with columns: text, label, source.
        train_ratio: Fraction for training (default from config).
        val_ratio: Fraction for validation (default from config).
        seed: Random seed (default from config).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    cfg = default_config
    train_ratio = train_ratio or cfg.train_ratio
    val_ratio = val_ratio or cfg.val_ratio
    seed = seed or cfg.seed

    test_ratio = round(1.0 - train_ratio - val_ratio, 6)
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label"],
        random_state=seed,
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        stratify=temp_df["label"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
