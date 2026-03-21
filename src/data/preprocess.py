"""
Tokenization and light normalization.

Limitations (good for Methods / Discussion): bag-of-words style token lists
discard order, irony, multimodal context, and cross-lingual nuance — standard
limits of frequency and TF-IDF baselines before probabilistic or neural models.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z][a-z0-9']*", re.IGNORECASE)

# sklearn stopwords are lowercase set-like
_STOP = frozenset(w.lower() for w in ENGLISH_STOP_WORDS)


def tokenize_text(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, drop English stopwords and short tokens."""
    if not text or not str(text).strip():
        return []
    out: list[str] = []
    for m in _TOKEN_RE.finditer(str(text).lower()):
        t = m.group(0).strip("'")
        if len(t) < 2 or t in _STOP:
            continue
        out.append(t)
    return out


def tokenize_documents(texts: Iterable[str]) -> list[list[str]]:
    return [tokenize_text(t) for t in texts]


def add_tokens_column(
    df: pd.DataFrame,
    text_col: str = "text",
    out_col: str = "tokens",
) -> pd.DataFrame:
    df = df.copy()
    df[out_col] = tokenize_documents(df[text_col].astype(str))
    return df
