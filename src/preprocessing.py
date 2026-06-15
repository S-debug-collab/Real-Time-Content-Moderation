"""Text preprocessing utilities for content moderation."""

import re
import unicodedata
from typing import Optional

import pandas as pd

# Common contraction map for social text normalization
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII-compatible form."""
    return unicodedata.normalize("NFKC", text)


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    lowered = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        lowered = lowered.replace(contraction, expansion)
    return lowered


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Production preprocessing pipeline:
    1. Handle missing values
    2. Unicode normalization
    3. Contraction expansion
    4. URL/email removal
    5. Whitespace normalization
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    text = normalize_unicode(text)
    text = expand_contractions(text)

    # Remove URLs and email addresses (common noise in social datasets)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Preserve apostrophes within words; strip other special chars
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower() if lowercase else text


def preprocess_batch(texts: list[str], lowercase: bool = True) -> list[str]:
    """Apply preprocessing to a batch of texts."""
    return [clean_text(t, lowercase=lowercase) for t in texts]


def filter_empty_texts(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Remove rows with empty text or missing labels."""
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 0]
    df = df.dropna(subset=[label_col])
    return df.reset_index(drop=True)


def truncate_text(text: str, max_chars: Optional[int] = None) -> str:
    """Character-level truncation for logging/display (tokenizer handles model truncation)."""
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "..."
    return text
