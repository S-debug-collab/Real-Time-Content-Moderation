"""
feature_extraction.py
----------------------
Converts cleaned text into numerical features for Logistic Regression.

Implements and compares two classic NLP feature-extraction techniques:
    1. Bag of Words (CountVectorizer)   - raw term frequency counts
    2. TF-IDF (TfidfVectorizer)         - term frequency weighted by how
                                          rare/informative a word is across
                                          the whole corpus

Why compare them?
Bag of Words treats every word as equally important, so very common words
(that survive stopword removal) can dominate the feature space. TF-IDF
down-weights words that appear in many documents and up-weights words that
are distinctive to a document, which usually helps a linear classifier
like Logistic Regression separate classes better on text data.
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


def compute_dampened_class_weights(labels):
    """
    Computes class weights like sklearn's class_weight="balanced"
    (n_samples / (n_classes * n_c)) but takes the SQUARE ROOT of that
    ratio before using it.

    Why: plain "balanced" weighting is n_samples / (n_classes * n_c), which
    for a very rare class (e.g. 828 out of 159,571 comments) produces a
    huge multiplier (~48x here). On a linear model like Logistic
    Regression, a single coincidental word co-occurring with that rare
    class then gets a massively inflated coefficient — one accidental
    training example can make an unrelated, benign word (e.g. "loved")
    look like strong evidence for the rare class. Taking the square root
    keeps rare classes meaningfully up-weighted (so recall doesn't collapse
    to zero) while preventing single coincidental examples from dominating
    the decision boundary. This is a common practical compromise between
    class_weight=None (ignores imbalance, rare classes get ~0 recall) and
    class_weight="balanced" (can overreact to noise in very small classes).
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(classes)
    weights = {}
    for cls, count in zip(classes, counts):
        linear_weight = n_samples / (n_classes * count)
        weights[cls] = float(np.sqrt(linear_weight))
    return weights


def build_bow_features(corpus, max_features=5000, ngram_range=(1, 1), min_df=3):
    """Fit a CountVectorizer (Bag of Words) on the corpus.

    min_df=3 drops tokens that appear in fewer than 3 documents, which
    reduces noise from ultra-rare words/typos that a linear model could
    otherwise latch onto due to random co-occurrence with a class.
    """
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def build_tfidf_features(corpus, max_features=5000, ngram_range=(1, 1), min_df=3):
    """Fit a TfidfVectorizer on the corpus. See build_bow_features for min_df."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def compare_bow_vs_tfidf(corpus, labels, max_features=5000, random_state=42):
    """
    Trains a quick Logistic Regression model with BOTH feature extraction
    methods on the same train/test split and reports accuracy + weighted
    F1 for each, so we can justify picking one for the final model.

    Returns a dict summary plus the fitted vectorizers/models so the
    caller (train.py) can reuse whichever performed better without
    re-vectorizing.
    """
    results = {}

    # BAG OF WORDS
    bow_vectorizer, X_bow = build_bow_features(corpus, max_features=max_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bow, labels, test_size=0.2, random_state=random_state, stratify=labels
    )
    class_weights = compute_dampened_class_weights(y_train)
    bow_model = LogisticRegression(max_iter=2000, class_weight=class_weights)
    bow_model.fit(X_train, y_train)
    bow_preds = bow_model.predict(X_test)
    results["bow"] = {
        "vectorizer": bow_vectorizer,
        "model": bow_model,
        "accuracy": accuracy_score(y_test, bow_preds),
        "f1_weighted": f1_score(y_test, bow_preds, average="weighted"),
    }

    # TF-IDF
    tfidf_vectorizer, X_tfidf = build_tfidf_features(corpus, max_features=max_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, labels, test_size=0.2, random_state=random_state, stratify=labels
    )
    class_weights = compute_dampened_class_weights(y_train)
    tfidf_model = LogisticRegression(max_iter=2000, class_weight=class_weights)
    tfidf_model.fit(X_train, y_train)
    tfidf_preds = tfidf_model.predict(X_test)
    results["tfidf"] = {
        "vectorizer": tfidf_vectorizer,
        "model": tfidf_model,
        "accuracy": accuracy_score(y_test, tfidf_preds),
        "f1_weighted": f1_score(y_test, tfidf_preds, average="weighted"),
    }

    return results


def print_comparison(results):
    print("\n--- Feature Extraction Comparison ---")
    print(f"{'Method':<10}{'Accuracy':<12}{'Weighted F1':<12}")
    for name in ("bow", "tfidf"):
        r = results[name]
        print(f"{name.upper():<10}{r['accuracy']:<12.4f}{r['f1_weighted']:<12.4f}")
    winner = max(results, key=lambda k: results[k]["f1_weighted"])
    print(f"\nBest method by weighted F1: {winner.upper()}")
    return winner
