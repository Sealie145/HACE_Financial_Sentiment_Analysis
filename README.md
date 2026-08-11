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

## Hedging detection

`src.hedging.HedgeDetector` is a lightweight, rule-based auxiliary feature
layer. It detects token-aware financial hedging cues (modals, epistemic and
uncertainty terms, approximators, and conditional phrases) without changing
the input text or sentiment labels.

```python
from src.hedging.detector import HedgeDetector

HedgeDetector().detect("Results may improve depending on market conditions.").to_dict()
```

Version 1's `hedge_probability` is a deterministic lexicon-derived confidence
score, **not** a calibrated machine-learning probability. Run the FiQA sanity
check with `notebooks/02_hedging_detection.ipynb`; it writes only the enriched
output to `data/interim/fiqa_hedging.csv`, preserving raw data unchanged.

Run the dependency-free tests with:

```bash
python -m unittest tests/test_hedging.py
```

## API demo

The FastAPI application currently combines real hedge detection with clearly
marked deterministic mock predictions for the four future FinBERT experts and
the final sentiment. It does not load model checkpoints or a meta-learner.

```bash
uvicorn backend.main:app --reload --port 8000
```

Open Swagger documentation at `http://127.0.0.1:8000/docs`. The `POST /predict`
response includes the nested `hedging` object and mock `fiqa`, `twitter`,
`phrasebank`, and `finance_news` experts.

## License

Research use only. Not for investment or trading decisions.

