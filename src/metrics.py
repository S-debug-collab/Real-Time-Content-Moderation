"""Evaluation metrics and reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.config import ID2LABEL, LABEL_NAMES, METRICS_PATH, MODELS_DIR


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Compute full classification metrics including per-class breakdown."""
    label_names = label_names or LABEL_NAMES

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(label_names))), zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {},
    }

    for i, name in enumerate(label_names):
        metrics["per_class"][name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(len(label_names)))
    ).tolist()

    return metrics


def save_metrics(metrics: dict[str, Any], path: Optional[Path] = None) -> Path:
    """Save metrics JSON (excluding large text fields)."""
    path = path or METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report_text"] = metrics.get("classification_report", "")

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    return path


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: Optional[Path] = None,
    label_names: Optional[list[str]] = None,
) -> Path:
    """Generate and save confusion matrix heatmap."""
    label_names = label_names or LABEL_NAMES
    output_path = output_path or MODELS_DIR / "confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Content Moderation — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def print_metrics_summary(metrics: dict[str, Any], model_name: str = "Model") -> None:
    """Pretty-print metrics to console."""
    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"\n  Per-class metrics:")
    for label, scores in metrics["per_class"].items():
        print(
            f"    {label:14s}  P={scores['precision']:.3f}  "
            f"R={scores['recall']:.3f}  F1={scores['f1']:.3f}  (n={scores['support']})"
        )
    print(f"\n{metrics.get('classification_report', '')}")
