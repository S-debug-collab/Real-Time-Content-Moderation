import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)

from feature_extraction import build_tfidf_features
from preprocessing import clean_text, handle_missing_values
from labels import LABEL_COLUMNS

DATA_PATH = "dataset/train.csv"
MODEL_DIR = "model"
TFIDF_MAX_FEATURES = 8000

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    return pd.read_csv(DATA_PATH)


def build_label_matrix(df):
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df


def evaluate_model(model, X_test, Y_test, labels):
    Y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(Y_test, Y_pred, target_names=labels, zero_division=0))

    hamming = hamming_loss(Y_test, Y_pred)
    exact_acc = accuracy_score(Y_test, Y_pred)
    f1_micro = f1_score(Y_test, Y_pred, average="micro")
    f1_macro = f1_score(Y_test, Y_pred, average="macro")

    print("\nMetrics:")
    print("Hamming Loss      :", hamming)
    print("Exact Match Acc   :", exact_acc)
    print("F1 Micro          :", f1_micro)
    print("F1 Macro          :", f1_macro)

    cms = multilabel_confusion_matrix(Y_test, Y_pred)

    cols = 3
    rows = (len(labels) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = np.array(axes).flatten()

    for i, label in enumerate(labels):
        sns.heatmap(
            cms[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
            ax=axes[i]
        )
        axes[i].set_title(label)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(path)

    print("\nSaved confusion matrix:", path)

    return {
        "hamming_loss": hamming,
        "exact_match_accuracy": exact_acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


def main():
    print("Loading dataset...")
    df = load_dataset()

    print("Cleaning missing values...")
    df = handle_missing_values(df, "comment_text")

    print("Building labels...")
    df = build_label_matrix(df)

    print("Cleaning text...")
    df["clean_text"] = df["comment_text"].apply(clean_text)

    df = df[df["clean_text"].str.strip() != ""]

    Y = df[LABEL_COLUMNS].values

    print("Vectorizing text (TF-IDF)...")
    vectorizer, X = build_tfidf_features(
        df["clean_text"],
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2)
    )

    stratify = Y.any(axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    print("Training model...")

    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced"
        )
    )

    model.fit(X_train, Y_train)

    print("Evaluating...")
    evaluate_model(model, X_test, Y_test, LABEL_COLUMNS)

    print("Saving model...")

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(LABEL_COLUMNS, os.path.join(MODEL_DIR, "label_columns.pkl"))

    print("Done ✔")


if __name__ == "__main__":
    main()
