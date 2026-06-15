"""Dataset loading, label mapping, and class imbalance handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    JIGSAW_PRIORITY,
    LABEL2ID,
    LABEL_NAMES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.preprocessing import clean_text, filter_empty_texts


def map_jigsaw_multilabel(row: pd.Series, label_columns: list[str]) -> str:
    """
    Map Jigsaw's 6 multi-label columns to a single moderation class.

    Priority: hate_speech > toxic > offensive > neutral
    This ensures the most severe violation is captured when multiple labels apply.
    """
    for target_label, source_cols in JIGSAW_PRIORITY:
        for col in source_cols:
            if col in row.index and row[col] == 1:
                return target_label
    return "neutral"


def map_twitter_hate_label(label: int | str) -> str:
    """Map binary Twitter hate speech labels to our taxonomy."""
    label_str = str(label).lower()
    if label_str in {"1", "hate", "hatespeech", "hate_speech"}:
        return "hate_speech"
    if label_str in {"2", "offensive"}:
        return "offensive"
    return "neutral"


def load_jigsaw_dataset(
    csv_path: Optional[Path] = None,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Jigsaw Toxic Comment Classification dataset from CSV.

    Expected columns: comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
    Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    """
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "train.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Jigsaw dataset not found at {csv_path}. "
            "Download train.csv from Kaggle and place it in data/raw/. "
            "Alternatively, run: python -m src.dataset --source huggingface"
        )

    df = pd.read_csv(csv_path)
    text_col = "comment_text" if "comment_text" in df.columns else "text"
    jigsaw_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    available_cols = [c for c in jigsaw_cols if c in df.columns]

    df["text"] = df[text_col].apply(clean_text)
    df["label"] = df.apply(lambda row: map_jigsaw_multilabel(row, available_cols), axis=1)
    df["label_id"] = df["label"].map(LABEL2ID)

    df = filter_empty_texts(df, text_col="text", label_col="label")

    if sample_size and len(df) > sample_size:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample_size // len(LABEL_NAMES)), random_state=42)
        ).reset_index(drop=True)

    return df[["text", "label", "label_id"]]


def load_huggingface_jigsaw(sample_size: Optional[int] = 20000) -> pd.DataFrame:
    """
    Load Jigsaw data via HuggingFace datasets (no Kaggle account required).
    Falls back to a small synthetic demo set if download fails.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("google/jigsaw_toxicity_pred", split="train")
        df = ds.to_pandas()

        text_col = "comment_text" if "comment_text" in df.columns else "text"
        jigsaw_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        available_cols = [c for c in jigsaw_cols if c in df.columns]

        df["text"] = df[text_col].apply(clean_text)
        df["label"] = df.apply(lambda row: map_jigsaw_multilabel(row, available_cols), axis=1)
        df["label_id"] = df["label"].map(LABEL2ID)
        df = filter_empty_texts(df, text_col="text", label_col="label")

        if sample_size and len(df) > sample_size:
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), sample_size // len(LABEL_NAMES)), random_state=42))
                .reset_index(drop=True)
            )

        return df[["text", "label", "label_id"]]

    except Exception as exc:
        print(f"HuggingFace download failed ({exc}). Using synthetic demo dataset.")
        return generate_demo_dataset(n_samples=2000)


def load_twitter_hate_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load Twitter hate speech dataset (CSV with 'text' and 'label' columns)."""
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "twitter_hate.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Twitter hate dataset not found at {csv_path}. "
            "Place twitter_hate.csv in data/raw/ or use --source huggingface"
        )

    df = pd.read_csv(csv_path)
    text_col = "tweet" if "tweet" in df.columns else "text"
    label_col = "class" if "class" in df.columns else "label"

    df["text"] = df[text_col].apply(clean_text)
    df["label"] = df[label_col].apply(map_twitter_hate_label)
    df["label_id"] = df["label"].map(LABEL2ID)
    return filter_empty_texts(df, text_col="text", label_col="label")[["text", "label", "label_id"]]


def generate_demo_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset for development/demo when real data is unavailable.
    Not intended for production training — use Jigsaw/Twitter datasets for real models.
    """
    rng = np.random.default_rng(seed)
    templates = {
        "neutral": [
            "Thanks for sharing this helpful article.",
            "Looking forward to the meeting tomorrow.",
            "Great work on the project presentation.",
            "Can you send me the report by Friday?",
            "The weather is nice today.",
        ],
        "toxic": [
            "You are completely useless and worthless.",
            "Nobody wants you here, just leave.",
            "This is the worst garbage I have ever seen.",
            "You should be ashamed of yourself.",
        ],
        "hate_speech": [
            "People like you don't belong in this country.",
            "Your entire group is inferior and dangerous.",
            "Violence against them is justified.",
        ],
        "offensive": [
            "What an idiot, can't believe this nonsense.",
            "Shut up you moron.",
            "This is so stupid it hurts.",
        ],
    }

    rows = []
    per_class = n_samples // len(LABEL_NAMES)
    for label in LABEL_NAMES:
        for _ in range(per_class):
            base = rng.choice(templates[label])
            noise = rng.choice(["", " lol", " seriously", " ugh", " wow"])
            rows.append({"text": clean_text(base + noise), "label": label, "label_id": LABEL2ID[label]})

    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def get_class_weights(labels: list[int]) -> dict[int, float]:
    """
    Compute inverse-frequency class weights for imbalanced datasets.
    Used during DistilBERT training via CrossEntropyLoss weight parameter.
    """
    unique_labels = np.unique(labels)
    weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
    return {int(label): float(weight) for label, weight in zip(unique_labels, weights)}


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split preserving label distribution."""
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, stratify=df["label"], random_state=random_state
    )
    relative_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val, stratify=temp_df["label"], random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> None:
    """Persist processed splits to disk."""
    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    label_dist = {
        "train": train_df["label"].value_counts().to_dict(),
        "val": val_df["label"].value_counts().to_dict(),
        "test": test_df["label"].value_counts().to_dict(),
    }
    with open(output_dir / "label_distribution.json", "w") as f:
        json.dump(label_dist, f, indent=2)


def prepare_data(
    source: str = "huggingface",
    sample_size: Optional[int] = 20000,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end data preparation pipeline.

    Args:
        source: 'huggingface', 'jigsaw', 'twitter', or 'demo'
        sample_size: Limit dataset size for faster iteration
        save: Whether to persist processed splits
    """
    loaders = {
        "huggingface": lambda: load_huggingface_jigsaw(sample_size),
        "jigsaw": lambda: load_jigsaw_dataset(sample_size=sample_size),
        "twitter": load_twitter_hate_dataset,
        "demo": lambda: generate_demo_dataset(n_samples=sample_size or 2000),
    }

    if source not in loaders:
        raise ValueError(f"Unknown source '{source}'. Choose from: {list(loaders.keys())}")

    df = loaders[source]()
    print(f"Loaded {len(df)} samples. Label distribution:\n{df['label'].value_counts()}")

    train_df, val_df, test_df = split_dataset(df)
    if save:
        save_processed_splits(train_df, val_df, test_df)

    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare moderation dataset")
    parser.add_argument("--source", default="huggingface", choices=["huggingface", "jigsaw", "twitter", "demo"])
    parser.add_argument("--sample-size", type=int, default=20000)
    args = parser.parse_args()

    prepare_data(source=args.source, sample_size=args.sample_size)
