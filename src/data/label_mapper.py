"""
label_mapper.py — Normalize heterogeneous dataset labels to HACE standard.

Standard mapping:
    negative / bearish / -1  → 0
    neutral  / none    /  0  → 1
    positive / bullish / +1  → 2
"""

import pandas as pd

# Exhaustive alias map — extend as new datasets are added
_ALIAS: dict[str, int] = {
    # Negative
    "negative": 0,
    "neg": 0,
    "bearish": 0,
    "-1": 0,
    -1: 0,
    # Neutral
    "neutral": 1,
    "neu": 1,
    "none": 1,
    "0": 1,
    0: 1,
    # Positive
    "positive": 2,
    "pos": 2,
    "bullish": 2,
    "1": 2,
    1: 2,
}

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}


def normalize_label(raw_label) -> int:
    """Map a raw label value to 0 / 1 / 2.

    Args:
        raw_label: Original label (string or int).

    Returns:
        Normalized integer label.

    Raises:
        ValueError: If the label cannot be mapped.
    """
    key = raw_label.strip().lower() if isinstance(raw_label, str) else raw_label
    if key not in _ALIAS:
        raise ValueError(
            f"Unknown label '{raw_label}'. "
            f"Add it to label_mapper._ALIAS or pre-process the dataset."
        )
    return _ALIAS[key]


def map_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Apply normalize_label to an entire DataFrame column in-place.

    Args:
        df: DataFrame containing the label column.
        label_col: Name of the column to normalize.

    Returns:
        DataFrame with the label column replaced by integer labels 0/1/2.
    """
    df = df.copy()
    df[label_col] = df[label_col].apply(normalize_label)
    return df
