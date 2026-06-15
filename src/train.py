"""Fine-tune DistilBERT for content moderation classification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DISTILBERT_MODEL_NAME,
    ID2LABEL,
    LABEL2ID,
    MAX_SEQ_LENGTH,
    METRICS_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PYTORCH_MODEL_DIR,
)
from src.dataset import get_class_weights, prepare_data
from src.metrics import compute_metrics, plot_confusion_matrix, print_metrics_summary, save_metrics


@dataclass
class ModerationDataset(Dataset):
    texts: list[str]
    labels: list[int]
    tokenizer: AutoTokenizer
    max_length: int = MAX_SEQ_LENGTH

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced moderation datasets."""

    def __init__(self, class_weights: Optional[dict[int, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight_tensor = torch.tensor(
                [self.class_weights[i] for i in range(len(self.class_weights))],
                dtype=torch.float32,
                device=logits.device,
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_trainer_metrics(eval_pred) -> dict:
    """HuggingFace Trainer metrics callback."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    m = compute_metrics(labels.tolist(), predictions.tolist())
    return {
        "accuracy": m["accuracy"],
        "precision": m["macro_precision"],
        "recall": m["macro_recall"],
        "f1": m["macro_f1"],
    }


def train_distilbert(
    prepare: bool = True,
    source: str = "huggingface",
    sample_size: int = 20000,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Fine-tune DistilBERT for 4-class content moderation.

    Why DistilBERT:
    - 40% smaller and 60% faster than BERT while retaining ~97% of its performance
    - Ideal for real-time inference with <100ms latency target on CPU
    - Strong contextual understanding vs. bag-of-words baselines
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir or str(PYTORCH_MODEL_DIR)

    if prepare:
        prepare_data(source=source, sample_size=sample_size)

    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = ModerationDataset(train_df["text"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_dataset = ModerationDataset(val_df["text"].tolist(), val_df["label_id"].tolist(), tokenizer)
    test_dataset = ModerationDataset(test_df["text"].tolist(), test_df["label_id"].tolist(), tokenizer)

    class_weights = get_class_weights(train_df["label_id"].tolist())

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Training DistilBERT on {len(train_df)} samples ({epochs} epochs)...")
    trainer.train()

    # Evaluate on test set
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = test_df["label_id"].tolist()

    metrics = compute_metrics(y_true, y_pred.tolist())
    metrics["model"] = "distilbert-base-uncased"
    metrics["train_samples"] = len(train_df)
    metrics["test_samples"] = len(test_df)
    metrics["epochs"] = epochs

    # Save final model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_metrics(metrics, MODELS_DIR / "distilbert_metrics.json")
    save_metrics(metrics, METRICS_PATH)
    plot_confusion_matrix(y_true, y_pred.tolist(), output_path=MODELS_DIR / "distilbert_confusion_matrix.png")

    print_metrics_summary(metrics, model_name="DistilBERT Fine-tuned")
    print(f"\nModel saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for content moderation")
    parser.add_argument("--source", default="huggingface")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--no-prepare", action="store_true")
    args = parser.parse_args()

    train_distilbert(
        prepare=not args.no_prepare,
        source=args.source,
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
