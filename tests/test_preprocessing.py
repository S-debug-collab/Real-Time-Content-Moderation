"""Tests for preprocessing module."""

import pytest

from src.preprocessing import clean_text, preprocess_batch


def test_clean_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_url_removal():
    assert "http" not in clean_text("Check this out https://spam.com now")


def test_clean_lowercase():
    assert clean_text("HELLO World") == "hello world"


def test_preprocess_batch():
    texts = ["Hello!", "  ", "Test URL https://x.com"]
    result = preprocess_batch(texts)
    assert len(result) == 3
    assert result[1] == ""
