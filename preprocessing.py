import re
import string

import emoji
import nltk
import contractions
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from contractions import contractions_dict


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
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

CONTRACTIONS_MAP = dict(contractions_dict)

_CONTRACTIONS_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS_MAP, key=len, reverse=True)) + r")\b"
)


def handle_missing_values(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].fillna("")
    df = df[df[text_column].str.strip() != ""]
    df = df.reset_index(drop=True)
    return df


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_html_tags(text: str) -> str:
    return HTML_PATTERN.sub(" ", text)


def convert_emojis_to_text(text: str) -> str:
    text = emoji.demojize(text, delimiters=(" emoji ", " "))
    return text.replace("_", " ")


def expand_contractions(text: str) -> str:
    return _CONTRACTIONS_PATTERN.sub(lambda match: CONTRACTIONS_MAP[match.group(0)], text)


def normalize_repeated_chars(text: str) -> str:
    return REPEATED_CHAR_PATTERN.sub(r"\1\1", text)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return NUMBER_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def stem_tokens(tokens: list) -> list:
    return [STEMMER.stem(t) for t in tokens]


def lemmatize_tokens(tokens: list) -> list:
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def clean_text(text: str) -> str:
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = convert_emojis_to_text(text)
    text = expand_contractions(text)
    text = normalize_repeated_chars(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize_whitespace(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Check THIS out!! <b>Visit</b> http://spam.com NOW, you're stupidddd 😡🤬 123!!!"
    print("Original  :", sample)
    print("Lemmatized:", clean_text(sample))
