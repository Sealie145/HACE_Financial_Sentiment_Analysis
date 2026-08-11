"""
trainer.py — Fine-tune a FinBERT model on a domain dataset.

Uses Hugging Face Trainer for training loop management.
Saves the best checkpoint (by validation Macro F1) to models/<domain>/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score

from src.utils.config import config, HACEConfig
from src.utils.seed import set_seed


def _tokenize(batch: dict, tokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


class ExpertTrainer:
    """Fine-tune FinBERT on a domain-specific training set.

    Args:
        domain: Domain name (e.g., 'fiqa').
        cfg: HACEConfig instance. Defaults to global config.
    """

    def __init__(self, domain: str, cfg: HACEConfig = None) -> None:
        self.domain = domain
        self.cfg = cfg or config
        set_seed(self.cfg.seed)

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: Path = None,
    ) -> Path:
        """Fine-tune FinBERT and save the best checkpoint.

        Args:
            train_df: Training DataFrame with columns: text, label.
            val_df: Validation DataFrame with columns: text, label.
            output_dir: Where to save the final model. Defaults to models/<domain>/.

        Returns:
            Path to the saved model directory.
        """
        output_dir = output_dir or self.cfg.model_dir(self.domain)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.finbert_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.finbert_model_name,
            num_labels=self.cfg.num_labels,
            id2label=self.cfg.id2label,
            label2id=self.cfg.label2id,
        )

        train_ds = Dataset.from_pandas(train_df[["text", "label"]])
        val_ds = Dataset.from_pandas(val_df[["text", "label"]])

        train_ds = train_ds.map(
            lambda b: _tokenize(b, tokenizer, self.cfg.max_seq_length),
            batched=True,
        )
        val_ds = val_ds.map(
            lambda b: _tokenize(b, tokenizer, self.cfg.max_seq_length),
            batched=True,
        )

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=self.cfg.num_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            per_device_eval_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            seed=self.cfg.seed,
            fp16=False,  # Set to True on GPU for faster training
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        trainer.train()

        # Save final best model
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"[ExpertTrainer] Saved {self.domain} expert to {output_dir}")
        return output_dir
