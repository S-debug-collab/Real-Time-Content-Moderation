"""
train.py
--------
End-to-end training pipeline for the Real-Time Content Moderation System.

Pipeline stages (each mirrors a step from the project spec):
    1.  Load dataset
    2.  Data exploration
    3.  Handle missing values
    4.  Clean text (lowercase, URL/HTML/punctuation/number removal)
    5.  Tokenization, stopword removal, lemmatization (see preprocessing.py)
    6.  Derive the 4-class moderation label from Jigsaw's binary columns
    7.  Compare Bag-of-Words vs TF-IDF feature extraction
    8.  Train the final Logistic Regression model on the winning features
    9.  Evaluate: accuracy, precision, recall, F1, confusion matrix
    10. Persist model + vectorizer + label encoder to model/ for the API

Dataset note
------------
The real dataset is the Jigsaw Toxic Comment Classification Challenge
(Kaggle: train.csv with columns id, comment_text, toxic, severe_toxic,
obscene, threat, insult, identity_hate). Kaggle requires an account, so
this script:
    - Looks for dataset/train.csv (the real Jigsaw file) first.
    - Falls back to a small bundled synthetic sample with the SAME
      schema, so the whole pipeline is runnable end-to-end without
      external downloads. Swap in the real Kaggle CSV for production
      results (see README.md for the download link).

Label mapping (Jigsaw's 6 binary tags -> our 4 classes)
--------------------------------------------------------
Jigsaw doesn't ship single-label "Neutral/Toxic/Offensive/Hate Speech"
tags, so we derive one deterministic label per comment using a priority
order (most severe wins when multiple tags fire on the same comment):

    identity_hate == 1                       -> "Hate Speech"
    toxic == 1 or severe_toxic == 1          -> "Toxic"
    obscene == 1 or insult == 1 or threat==1 -> "Offensive"
    none of the above                        -> "Neutral"

This is a reasonable, explainable heuristic for turning a multi-label
dataset into the single-label 4-class problem this project asks for.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_extraction import (
    build_tfidf_features,
    compare_bow_vs_tfidf,
    compute_dampened_class_weights,
    print_comparison,
)
from preprocessing import clean_text, handle_missing_values

DATASET_DIR = "dataset"
MODEL_DIR = "model"
REAL_DATA_PATH = os.path.join(DATASET_DIR, "train.csv")
SAMPLE_DATA_PATH = os.path.join(DATASET_DIR, "sample_data.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load dataset (real Jigsaw CSV if present, else generate a small
#    synthetic sample with an identical schema so the project still runs).
# ---------------------------------------------------------------------------
def generate_synthetic_sample(path, n_per_class=120, seed=42):
    """Creates a small Jigsaw-schema CSV so the pipeline is runnable
    without a Kaggle account. NOT a substitute for the real dataset —
    swap in dataset/train.csv from Kaggle for real performance numbers.
    """
    rng = np.random.default_rng(seed)

    neutral_templates = [
        "I really enjoyed this movie, great acting!",
        "Thanks for sharing this article, very helpful.",
        "Can someone explain how this feature works?",
        "Congratulations on the new job, well deserved!",
        "The weather has been lovely this week.",
        "This recipe turned out amazing, thank you!",
        "Looking forward to the meeting tomorrow.",
        "Great presentation, learned a lot today.",
        "I disagree with the conclusion but respect the effort.",
        "The documentation could use a few more examples.",
    ]
    toxic_templates = [
        "This is the dumbest thing I have ever read.",
        "Nobody asked for your opinion, shut up.",
        "You clearly have no idea what you're talking about.",
        "What a waste of time this whole thread is.",
        "Get lost, no one wants you here.",
        "This is garbage, just like everything else you post.",
        "You are an idiot if you believe that.",
        "Stop spreading nonsense, it's embarrassing.",
        "This comment section is full of morons.",
        "Your logic is trash and so is your attitude.",
    ]
    offensive_templates = [
        "You are such a pathetic loser, get a life.",
        "Shut your stupid mouth before I make you.",
        "You're a worthless piece of garbage.",
        "I hope you fail at everything you do, jerk.",
        "Only a complete moron would say something like that.",
        "You disgust me, go away forever.",
        "What an ugly, stupid thing to post.",
        "You're a joke and everyone laughs at you.",
        "Quit whining, you spineless coward.",
        "You are trash and everyone knows it.",
    ]
    hate_templates = [
        "People like you don't belong in this country.",
        "Your entire race is the problem with society.",
        "I hate everyone from that religion, they're all the same.",
        "Women like you shouldn't be allowed to work here.",
        "That ethnic group is ruining everything, send them back.",
        "People of that faith are all dangerous criminals.",
        "Your kind should not be allowed to speak in public.",
        "I despise that community and everyone in it.",
        "That gender doesn't deserve any rights at all.",
        "Immigrants like you are destroying this nation.",
    ]

    rows = []
    idx = 0
    for templates, tags in [
        (neutral_templates, dict(toxic=0, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0)),
        (toxic_templates, dict(toxic=1, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0)),
        (offensive_templates, dict(toxic=0, severe_toxic=0, obscene=1, threat=0, insult=1, identity_hate=0)),
        (hate_templates, dict(toxic=1, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=1)),
    ]:
        for _ in range(n_per_class):
            base = templates[rng.integers(0, len(templates))]
            rows.append({"id": idx, "comment_text": base, **tags})
            idx += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    # Inject a few missing values to demonstrate step 3 (handle missing values)
    missing_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[missing_idx, "comment_text"] = np.nan
    df.to_csv(path, index=False)
    return df


def load_dataset():
    if os.path.exists(REAL_DATA_PATH):
        print(f"Loading real Jigsaw dataset from {REAL_DATA_PATH}")
        df = pd.read_csv(REAL_DATA_PATH)
    elif os.path.exists(SAMPLE_DATA_PATH):
        print(f"Loading bundled synthetic sample from {SAMPLE_DATA_PATH}")
        df = pd.read_csv(SAMPLE_DATA_PATH)
    else:
        print("No dataset found. Generating a synthetic sample dataset...")
        df = generate_synthetic_sample(SAMPLE_DATA_PATH)
    return df


# ---------------------------------------------------------------------------
# 2. Data exploration
# ---------------------------------------------------------------------------
def explore_data(df):
    print("\n--- Data Exploration ---")
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nSample rows:\n", df.head(3))


# ---------------------------------------------------------------------------
# 6. Derive the 4-class label from Jigsaw's binary tag columns
# ---------------------------------------------------------------------------
def derive_label(row):
    if row.get("identity_hate", 0) == 1:
        return "Hate Speech"
    if row.get("toxic", 0) == 1 or row.get("severe_toxic", 0) == 1:
        return "Toxic"
    if row.get("obscene", 0) == 1 or row.get("insult", 0) == 1 or row.get("threat", 0) == 1:
        return "Offensive"
    return "Neutral"


def build_labels(df):
    tag_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for col in tag_cols:
        if col not in df.columns:
            df[col] = 0
    df["label"] = df.apply(derive_label, axis=1)
    print("\n--- Class Distribution ---")
    print(df["label"].value_counts())
    return df


# ---------------------------------------------------------------------------
# 9. Evaluation: accuracy, precision, recall, F1, confusion matrix
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, label_encoder):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted", zero_division=0)
    recall = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    # 1. Load dataset
    df = load_dataset()

    # 2. Data exploration
    explore_data(df)

    # 3. Handle missing values
    df = handle_missing_values(df, "comment_text")

    # 6. Derive 4-class labels
    df = build_labels(df)

    # 4-5. Clean text (lowercase, URL/HTML/punct/number removal, tokenize,
    # remove stopwords, lemmatize) — implemented in preprocessing.py
    print("\nCleaning text (this applies the full NLP normalization pipeline)...")
    df["clean_text"] = df["comment_text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

    # Encode string labels -> integers for scikit-learn
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    # 7. Compare Bag of Words vs TF-IDF on the SAME split/model settings
    print("\nComparing Bag-of-Words vs TF-IDF feature extraction...")
    comparison = compare_bow_vs_tfidf(df["clean_text"], y, max_features=5000)
    best_method = print_comparison(comparison)

    # 8. Train the final model using the winning feature extraction method
    #    (re-fit cleanly on the full corpus before the final train/test split
    #    so the saved vectorizer/model pair is self-consistent).
    if best_method == "tfidf":
        vectorizer, X = build_tfidf_features(df["clean_text"], max_features=5000)
    else:
        from feature_extraction import build_bow_features
        vectorizer, X = build_bow_features(df["clean_text"], max_features=5000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    from sklearn.linear_model import LogisticRegression

    class_weights = compute_dampened_class_weights(y_train)
    model = LogisticRegression(max_iter=2000, class_weight=class_weights)
    model.fit(X_train, y_train)

    # 9. Evaluate
    evaluate_model(model, X_test, y_test, label_encoder)

    # 10. Persist artifacts for the FastAPI service
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(best_method, os.path.join(MODEL_DIR, "feature_method.pkl"))

    print(f"\nSaved model, vectorizer ({best_method}), and label encoder to '{MODEL_DIR}/'")


if __name__ == "__main__":
    main()
