"""
cleaner.py — Financial text preprocessing.

Design principles:
  - Do NOT over-clean financial text.
  - Preserve numbers, percentages, currency symbols, and domain terminology.
  - Remove only genuinely noisy content (HTML tags, excess whitespace, bare URLs).
"""

from __future__ import annotations

import re

import pandas as pd


# ── Regex patterns ────────────────────────────────────────────────────────────

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTI_SPACE = re.compile(r"\s{2,}")
_NEWLINE = re.compile(r"[\r\n]+")


def clean_text(text: str, replace_url: str = "[URL]") -> str:
    """Clean a single financial text string.

    Steps applied (in order):
    1. Return empty string for non-string or whitespace-only input.
    2. Strip leading/trailing whitespace.
    3. Remove HTML tags.
    4. Replace URLs with a placeholder token (preserves sentence structure).
    5. Normalize internal whitespace (collapse multiple spaces/newlines).

    Financial content preserved:
    - Numbers, percentages, currency symbols ($, €, £, ¥)
    - Abbreviations, tickers, financial terminology
    - Meaningful punctuation (., %, -, ())

    Args:
        text: Raw input string.
        replace_url: Token to substitute for URLs. Use "" to remove entirely.

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.strip()
    text = _HTML_TAG.sub("", text)
    text = _URL.sub(replace_url, text)
    text = _NEWLINE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def clean_series(series: pd.Series, replace_url: str = "[URL]") -> pd.Series:
    """Apply clean_text to a pandas Series.

    Args:
        series: Series of raw text strings.
        replace_url: URL placeholder token.

    Returns:
        Cleaned Series (same index).
    """
    return series.apply(lambda x: clean_text(x, replace_url=replace_url))
