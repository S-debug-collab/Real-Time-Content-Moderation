"""
preprocessing.py
----------------
Text cleaning and NLP normalization pipeline for the Real-Time Content
Moderation System.

This module implements every text-preprocessing step required before
feature extraction:
    1. Handle missing values
    2. Lowercasing
    3. URL removal
    4. HTML tag removal
    5. Punctuation removal
    6. Number removal
    7. Tokenization
    8. Stopword removal
    9. Stemming
    10. Lemmatization

Each step is a small, testable function so it can be reused independently
(e.g. in train.py for batch cleaning and in predict.py for single-comment
cleaning at inference time).
"""

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# One-time NLTK resource download (safe to call repeatedly; NLTK skips
# re-downloading if the resource already exists locally).
# ---------------------------------------------------------------------------
def download_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
    ]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


download_nltk_resources()

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NUMBER_PATTERN = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# 1. Handle missing values
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Drop rows with missing/empty comment text and reset the index."""
    df = df.copy()
    df[text_column] = df[text_column].fillna("")
    df = df[df[text_column].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2-6. Individual cleaning steps (kept separate so each is independently
# demonstrable in an interview walkthrough).
# ---------------------------------------------------------------------------
def to_lowercase(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_html_tags(text: str) -> str:
    return HTML_PATTERN.sub(" ", text)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return NUMBER_PATTERN.sub(" ", text)


# ---------------------------------------------------------------------------
# 7. Tokenization
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list:
    return word_tokenize(text)


# ---------------------------------------------------------------------------
# 8. Stopword removal
# ---------------------------------------------------------------------------
def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# 9. Stemming
# ---------------------------------------------------------------------------
def stem_tokens(tokens: list) -> list:
    return [STEMMER.stem(t) for t in tokens]


# ---------------------------------------------------------------------------
# 10. Lemmatization
# ---------------------------------------------------------------------------
def lemmatize_tokens(tokens: list) -> list:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


# ---------------------------------------------------------------------------
# Full pipeline used by both training and inference.
#
# NOTE: Stemming and lemmatization both reduce words to a root/base form,
# so applying both in sequence is unusual in a real production pipeline.
# We expose both as separate functions to demonstrate/compare them, but
# the `clean_text` pipeline used for the actual model uses LEMMATIZATION
# only, since lemmatization produces real dictionary words and tends to
# preserve more signal for a linear model like Logistic Regression.
# `clean_text_with_stemming` is provided for side-by-side comparison.
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Full cleaning pipeline -> returns a cleaned, lemmatized string."""
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


def clean_text_with_stemming(text: str) -> str:
    """Alternate pipeline using stemming instead of lemmatization, kept
    for comparison/demo purposes (see train.py exploration section)."""
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Check THIS out!! <b>Visit</b> http://spam.com NOW, you're stupid 123!!!"
    print("Original :", sample)
    print("Lemmatized:", clean_text(sample))
    print("Stemmed   :", clean_text_with_stemming(sample))
