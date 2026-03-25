"""Frequency and TF-IDF baselines per time window."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _filter_tokens(tokens, extra_stop: frozenset | None) -> list[str]:
    if not extra_stop:
        return list(tokens)
    return [t for t in tokens if t not in extra_stop]


def frequency_top_terms(df, k=15, extra_stop: frozenset | None = None):
    """Top-k terms by count in each time bin."""
    results = {}
    for bin_val in sorted(df["time_bin"].unique()):
        mask = df["time_bin"] == bin_val
        counter = Counter()
        for toks in df.loc[mask, "tokens"]:
            counter.update(_filter_tokens(toks, extra_stop))
        results[bin_val] = counter.most_common(k)
    return results


def tfidf_top_terms(df, k=15, extra_stop: frozenset | None = None):
    """Top-k TF-IDF terms per time bin (each bin = one pseudo-document)."""
    bins = sorted(df["time_bin"].unique())
    docs = []
    for b in bins:
        mask = df["time_bin"] == b
        all_tokens = []
        for toks in df.loc[mask, "tokens"]:
            all_tokens.extend(_filter_tokens(toks, extra_stop))
        docs.append(" ".join(all_tokens))

    if not docs:
        return {}

    vec = TfidfVectorizer(token_pattern=r"(?u)\b[a-z][a-z0-9']+\b", lowercase=True)
    X = vec.fit_transform(docs)
    names = vec.get_feature_names_out()

    results = {}
    for i, b in enumerate(bins):
        row = np.asarray(X[i].todense()).ravel()
        if row.sum() == 0:
            results[b] = []
            continue
        top_idx = np.argsort(-row)[:k]
        results[b] = [(str(names[j]), float(row[j])) for j in top_idx if row[j] > 0]
    return results
