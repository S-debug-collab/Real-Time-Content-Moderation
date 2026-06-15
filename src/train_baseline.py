"""Baseline model: TF-IDF + Logistic Regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import BASELINE_MODEL_PATH, MODELS_DIR, PROCESSED_DATA_DIR
from src.dataset import prepare_data
from src.metrics import compute_metrics, plot_confusion_matrix, print_metrics_summary, save_metrics


def build_baseline_pipeline(classifier: str = "logistic_regression") -> Pipeline:
    """
    Build TF-IDF + classifier pipeline.

    TF-IDF captures n-gram patterns effective for toxic keyword detection.
    Logistic Regression provides calibrated probabilities and fast inference.
    """
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    if classifier == "svm":
        clf = LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)
    else:
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        )

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def train_baseline(
    train_path: Path | None = None,
    test_path: Path | None = None,
    classifier: str = "logistic_regression",
    prepare: bool = True,
    source: str = "huggingface",
    sample_size: int = 20000,
) -> dict:
    """Train baseline model and evaluate on test set."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if prepare:
        _, _, test_df = prepare_data(source=source, sample_size=sample_size)
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    else:
        train_df = pd.read_csv(train_path or PROCESSED_DATA_DIR / "train.csv")
        test_df = pd.read_csv(test_path or PROCESSED_DATA_DIR / "test.csv")

    pipeline = build_baseline_pipeline(classifier)
    print(f"Training baseline ({classifier}) on {len(train_df)} samples...")
    pipeline.fit(train_df["text"], train_df["label_id"])

    y_pred = pipeline.predict(test_df["text"])
    metrics = compute_metrics(test_df["label_id"].tolist(), y_pred.tolist())
    metrics["model"] = f"tfidf_{classifier}"
    metrics["train_samples"] = len(train_df)
    metrics["test_samples"] = len(test_df)

    joblib.dump(pipeline, BASELINE_MODEL_PATH)
    save_metrics(metrics, MODELS_DIR / "baseline_metrics.json")
    plot_confusion_matrix(
        test_df["label_id"].tolist(),
        y_pred.tolist(),
        output_path=MODELS_DIR / "baseline_confusion_matrix.png",
    )

    print_metrics_summary(metrics, model_name=f"Baseline (TF-IDF + {classifier})")
    print(f"\nModel saved to {BASELINE_MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF model")
    parser.add_argument("--classifier", default="logistic_regression", choices=["logistic_regression", "svm"])
    parser.add_argument("--source", default="huggingface")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--no-prepare", action="store_true")
    args = parser.parse_args()

    train_baseline(
        classifier=args.classifier,
        prepare=not args.no_prepare,
        source=args.source,
        sample_size=args.sample_size,
    )
