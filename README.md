# HACE — Hedging-Aware Cross-Domain Ensemble
**Financial Sentiment Analysis with Uncertainty Awareness**

## Overview

HACE is a research system that investigates whether adding hedging/uncertainty features to a stacking ensemble of domain-specialized FinBERT models improves financial sentiment classification — particularly on hedged financial text — and cross-domain generalization.

**Research Question:**  
Can hedging information added to a stacking ensemble of domain-specialized FinBERT models improve financial sentiment classification and cross-domain generalization?

## Sentiment Classes

| Label | Integer |
|---|---|
| Negative | 0 |
| Neutral | 1 |
| Positive | 2 |

## Project Structure

```
HACE/
├── data/           Raw, interim, and processed datasets
├── notebooks/      Experiment notebooks (run in order)
├── src/            Core Python modules
├── models/         Trained model checkpoints
├── outputs/        Metrics, figures, predictions
├── backend/        FastAPI service
└── frontend/       Gradio demo
```

## Notebooks (run in order)

| Notebook | Purpose |
|---|---|
| 01_data_preparation.ipynb | Load, clean, standardize, split all datasets |
| 02_train_experts.ipynb | Fine-tune four domain FinBERT experts |
| 03_ensemble_evaluation.ipynb | Train Base Ensemble & HACE; full evaluation |
| 04_demo.ipynb | Interactive demonstration |

## Quickstart

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run notebooks in order. See individual notebooks for dataset download instructions.

## Running the API

```bash
uvicorn backend.main:app --reload --port 8000
```

## Running the Demo

```bash
python frontend/app.py
```

## Datasets

| Dataset | Domain | Expert |
|---|---|---|
| FiQA 2018 | Financial QA / microblogs | FiQA Expert |
| FinancialPhraseBank (AllAgree) | Financial news | PhraseBank Expert |
| Twitter Financial News Sentiment | Social media | Twitter Expert |
| Financial News Sentiments (Kaggle) | Financial news | Finance News Expert |
| ProsusAI/finbert | General | General FinBERT |

See `data/raw/*/README.md` for dataset acquisition instructions.

## Reproducibility

All random seeds are fixed via `src/utils/seed.py`. Pinned dependency versions are in `requirements.txt`.

## License

Research use only. Not for investment or trading decisions.
