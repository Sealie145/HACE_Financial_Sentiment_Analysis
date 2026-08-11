"""
dataset_loader.py — Load raw datasets and return standardized DataFrames.

Each loader returns a DataFrame with columns: text | label | source
Labels are NOT yet normalized here — call map_labels() from label_mapper.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import config


# ── FiQA ──────────────────────────────────────────────────────────────────────

def load_fiqa(raw_dir: Path = None) -> pd.DataFrame:
    """Load FiQA 2018 from the locally saved JSON files.

    Confirmed schema (TheFinAI/fiqa-sentiment-classification):
        _id, sentence, target, aspect, score (float), type

    The 'score' is a continuous sentiment value in roughly [-1, 1].
    We bucket it into three classes:
        score < -0.1  → Negative (0)
        score > +0.1  → Positive (2)
        otherwise     → Neutral  (1)

    All available splits (train / valid / test) are combined and
    re-split by the splitter to ensure consistent stratified splits.

    Args:
        raw_dir: Path to data/raw/fiqa/. Defaults to config value.

    Returns:
        DataFrame with columns: text, label, source.
    """
    raw_dir = Path(raw_dir or (config.data_raw_dir / "fiqa"))
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {raw_dir}. "
            "Run: python scripts/download_datasets.py"
        )

    dfs = []
    for f in files:
        dfs.append(pd.read_json(f, lines=True))
    df = pd.concat(dfs, ignore_index=True)

    # Rename confirmed columns
    df = df.rename(columns={"sentence": "text", "score": "label"})

    # Bucket continuous score → 0 / 1 / 2
    def _bucket(score: float) -> int:
        if score < -0.1:
            return 0   # Negative
        elif score > 0.1:
            return 2   # Positive
        else:
            return 1   # Neutral

    df["label"] = df["label"].apply(_bucket)
    df["source"] = "fiqa"

    # Drop rows with missing text
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    return df[["text", "label", "source"]].reset_index(drop=True)


# ── FinancialPhraseBank ────────────────────────────────────────────────────────

def load_phrasebank(
    path: Path = None,
    encoding: str = "latin-1",
) -> pd.DataFrame:
    """Load FinancialPhraseBank from the local Sentences_AllAgree.txt file.

    File format (each line):
        <sentence>@<label>

    The HuggingFace loader (takala/financial_phrasebank) uses a deprecated
    dataset script and cannot be used. The local file is the authoritative source.

    Place the file at: data/raw/phrasebank/Sentences_AllAgree.txt
    Download from the original FinancialPhraseBank distribution.

    Labels in file: positive / negative / neutral (lowercase string)
    Mapped to: 2 / 0 / 1

    Args:
        path: Path to Sentences_AllAgree.txt. Defaults to data/raw/phrasebank/.
        encoding: File encoding (latin-1 is typical for this dataset).

    Returns:
        DataFrame with columns: text, label, source.
    """
    path = path or (config.data_raw_dir / "phrasebank" / "Sentences_AllAgree.txt")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"FinancialPhraseBank file not found at: {path}\n"
            "Please place Sentences_AllAgree.txt at data/raw/phrasebank/\n"
            "Download from the original FinancialPhraseBank distribution."
        )

    from src.data.label_mapper import normalize_label

    records = []
    with open(path, encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            text, raw_label = line.rsplit("@", 1)
            text = text.strip()
            raw_label = raw_label.strip().lower()
            if not text:
                continue
            try:
                label_int = normalize_label(raw_label)
            except ValueError:
                continue  # Skip unrecognised labels
            records.append({"text": text, "label": label_int, "source": "phrasebank"})

    return pd.DataFrame(records, columns=["text", "label", "source"])


# ── Twitter Financial News Sentiment ──────────────────────────────────────────

def load_twitter(raw_dir: Path = None) -> pd.DataFrame:
    """Load Twitter Financial News Sentiment from locally saved JSON files.

    Confirmed schema (zeroshot/twitter-financial-news-sentiment):
        text (string), label (int)

    Label mapping confirmed from dataset card:
        0 → Bearish  → Negative (0)
        1 → Bullish  → Positive (2)
        2 → Neutral  → Neutral  (1)

    All available splits (train / validation) are combined and
    re-split by the splitter.

    Args:
        raw_dir: Path to data/raw/twitter/. Defaults to config value.

    Returns:
        DataFrame with columns: text, label, source.
    """
    raw_dir = Path(raw_dir or (config.data_raw_dir / "twitter"))
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {raw_dir}. "
            "Run: python scripts/download_datasets.py"
        )

    dfs = []
    for f in files:
        dfs.append(pd.read_json(f, lines=True))
    df = pd.concat(dfs, ignore_index=True)

    # Remap: 0=Bearish→Neg(0), 1=Bullish→Pos(2), 2=Neutral→Neu(1)
    _twitter_map = {0: 0, 1: 2, 2: 1}
    df["label"] = df["label"].map(_twitter_map)
    df["source"] = "twitter"

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)

    return df[["text", "label", "source"]].reset_index(drop=True)


# ── Finance News Sentiments (Kaggle) ──────────────────────────────────────────

def load_finance_news(
    path: Path = None,
) -> pd.DataFrame:
    """Load the Kaggle Financial News Sentiments CSV.

    Download from: https://www.kaggle.com/datasets/antobenedetti/finance-news-sentiments
    Place the file at: data/raw/finance_news/dataset.csv

    Confirmed columns: text, sentiment (positive / neutral / negative)

    Args:
        path: Path to dataset.csv. Defaults to data/raw/finance_news/dataset.csv.

    Returns:
        DataFrame with columns: text, label, source.
    """
    path = path or (config.data_raw_dir / "finance_news" / "dataset.csv")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Finance News dataset not found at: {path}\n"
            "Download from: https://www.kaggle.com/datasets/antobenedetti/finance-news-sentiments\n"
            "Place dataset.csv at data/raw/finance_news/"
        )

    from src.data.label_mapper import normalize_label

    df = pd.read_csv(path)

    # Confirmed columns: 'text', 'sentiment'
    if "sentiment" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"sentiment": "label"})

    df["label"] = df["label"].apply(normalize_label)
    df["source"] = "finance_news"

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    return df[["text", "label", "source"]].reset_index(drop=True)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def load_raw_dataset(domain: str, **kwargs) -> pd.DataFrame:
    """Load a raw dataset by domain name.

    Args:
        domain: One of 'fiqa', 'phrasebank', 'twitter', 'finance_news'.
        **kwargs: Passed to the domain-specific loader.

    Returns:
        Raw DataFrame with columns: text, label, source.
    """
    loaders = {
        "fiqa": load_fiqa,
        "phrasebank": load_phrasebank,
        "twitter": load_twitter,
        "finance_news": load_finance_news,
    }
    if domain not in loaders:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(loaders)}")
    return loaders[domain](**kwargs)
