"""
download_datasets.py — Download all four HACE datasets.

Usage:
    python scripts/download_datasets.py

For Kaggle dataset, either:
  1. Place kaggle.json at ~/.kaggle/kaggle.json  (standard)
  2. Set KAGGLE_USERNAME and KAGGLE_KEY env vars
  3. Manually download from:
     https://www.kaggle.com/datasets/antobenedetti/finance-news-sentiments
     and place the CSV at: data/raw/finance_news/dataset.csv
"""

import os
import sys
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RAW = PROJECT_ROOT / "data" / "raw"


def download_fiqa():
    print("\n[1/4] Downloading FiQA 2018 (TheFinAI/fiqa-sentiment-classification)...")
    from datasets import load_dataset
    ds = load_dataset("TheFinAI/fiqa-sentiment-classification")
    # FiQA splits: train / valid / test  (NOT 'validation')
    print(f"  Available splits: {list(ds.keys())}")
    out = RAW / "fiqa"
    out.mkdir(parents=True, exist_ok=True)
    for split in ds.keys():
        path = out / f"{split}.json"
        ds[split].to_json(str(path))
        print(f"  Saved {split} → {path}  ({len(ds[split])} rows)")
    print("  FiQA done.")


def download_phrasebank():
    print("\n[2/4] Downloading FinancialPhraseBank (takala/financial_phrasebank)...")
    from datasets import load_dataset
    # Use the sentences_allagree config which is 100% annotator agreement
    ds = load_dataset("takala/financial_phrasebank", "sentences_allagree")
    out = RAW / "phrasebank"
    out.mkdir(parents=True, exist_ok=True)
    for split in ds.keys():
        path = out / f"{split}.json"
        ds[split].to_json(str(path))
        print(f"  Saved {split} → {path}  ({len(ds[split])} rows)")
    print("  FinancialPhraseBank done.")


def download_twitter():
    print("\n[3/4] Downloading Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment)...")
    from datasets import load_dataset
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    out = RAW / "twitter"
    out.mkdir(parents=True, exist_ok=True)
    for split in ds.keys():
        path = out / f"{split}.json"
        ds[split].to_json(str(path))
        print(f"  Saved {split} → {path}  ({len(ds[split])} rows)")
    print("  Twitter done.")


def download_finance_news_kaggle():
    print("\n[4/4] Downloading Finance News Sentiments (Kaggle)...")
    out = RAW / "finance_news"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "dataset.csv"

    if dest.exists():
        print(f"  Already exists: {dest}. Skipping.")
        return

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "antobenedetti/finance-news-sentiments",
            path=str(out),
            unzip=True,
        )
        # The Kaggle dataset may extract with a different filename — find it
        csvs = list(out.glob("*.csv"))
        if not csvs:
            print("  ERROR: No CSV found after Kaggle download. Check Kaggle credentials.")
            return
        # Rename the first CSV to dataset.csv if needed
        if dest not in csvs:
            csvs[0].rename(dest)
            print(f"  Renamed {csvs[0].name} → dataset.csv")
        print(f"  Saved → {dest}")
        print("  Finance News done.")

    except Exception as e:
        print(f"\n  Kaggle download failed: {e}")
        print("  Manual download instructions:")
        print("    1. Go to: https://www.kaggle.com/datasets/antobenedetti/finance-news-sentiments")
        print("    2. Download the CSV file")
        print(f"    3. Place it at: {dest}")


if __name__ == "__main__":
    download_fiqa()
    download_phrasebank()
    download_twitter()
    download_finance_news_kaggle()
    print("\nAll downloads complete.")
    print("Next step: run notebooks/01_data_preparation.ipynb")
