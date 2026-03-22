"""
Tokenization, lemmatization, and corpus-level statistics.

Offers two modes: lightweight regex tokenization (default) and WordNet
lemmatization via NLTK. Lemmatization reduces inflectional forms (e.g. "floods"
→ "flood", "running" → "run") which improves vocabulary consistency for both
frequency baselines and LDA. The trade-off is slower processing and a
dependency on NLTK data.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z][a-z0-9']*", re.IGNORECASE)

_STOP = frozenset(w.lower() for w in ENGLISH_STOP_WORDS)

_lemmatizer = None


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        import nltk
        for resource in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
        from nltk.stem import WordNetLemmatizer
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def tokenize_text(
    text: str,
    *,
    lemmatize: bool = False,
    min_len: int = 2,
) -> list[str]:
    """Lowercase, extract alphanumeric tokens, drop English stopwords and short tokens."""
    if not text or not str(text).strip():
        return []
    lem = _get_lemmatizer() if lemmatize else None
    out: list[str] = []
    for m in _TOKEN_RE.finditer(str(text).lower()):
        t = m.group(0).strip("'")
        if len(t) < min_len or t in _STOP:
            continue
        if lem is not None:
            t = lem.lemmatize(t)
            if len(t) < min_len or t in _STOP:
                continue
        out.append(t)
    return out


def tokenize_documents(
    texts: Iterable[str],
    *,
    lemmatize: bool = False,
) -> list[list[str]]:
    return [tokenize_text(t, lemmatize=lemmatize) for t in texts]


def build_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Build n-gram strings from a token list (e.g. bigrams: ['carbon_emissions', ...])."""
    if n < 2 or len(tokens) < n:
        return []
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def add_tokens_column(
    df: pd.DataFrame,
    text_col: str = "text",
    out_col: str = "tokens",
    *,
    lemmatize: bool = False,
    ngram: int | None = None,
) -> pd.DataFrame:
    """
    Tokenize texts; optionally lemmatize and/or append n-gram tokens.

    When ngram is set (e.g. 2 for bigrams), the n-gram strings are appended to
    the unigram list so downstream counters / LDA see both.
    """
    df = df.copy()
    tok_lists = tokenize_documents(df[text_col].astype(str), lemmatize=lemmatize)
    if ngram is not None and ngram >= 2:
        tok_lists = [t + build_ngrams(t, ngram) for t in tok_lists]
    df[out_col] = tok_lists
    return df


def corpus_statistics(df: pd.DataFrame, tokens_col: str = "tokens") -> dict[str, float]:
    """Summary stats for the tokenized corpus (useful for report tables)."""
    lengths = df[tokens_col].apply(len)
    from collections import Counter
    vocab: Counter[str] = Counter()
    for toks in df[tokens_col]:
        vocab.update(toks)
    return {
        "n_documents": int(len(df)),
        "vocab_size": int(len(vocab)),
        "total_tokens": int(lengths.sum()),
        "mean_doc_length": float(np.mean(lengths)) if len(lengths) > 0 else 0.0,
        "median_doc_length": float(np.median(lengths)) if len(lengths) > 0 else 0.0,
        "min_doc_length": int(lengths.min()) if len(lengths) > 0 else 0,
        "max_doc_length": int(lengths.max()) if len(lengths) > 0 else 0,
    }