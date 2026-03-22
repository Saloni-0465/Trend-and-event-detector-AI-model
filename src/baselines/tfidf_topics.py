"""
TF-IDF discriminative terms per time window.

Each window is treated as one document (concatenated posts). High weight terms
are frequent in that window relative to others — a standard strong baseline
before LDA or embeddings.

Supports unigrams and bigrams via `ngram_range` to capture multi-word concepts
like "carbon_emissions" or "gpu benchmark."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_top_terms_per_window(
    window_to_text: dict[pd.Timestamp, str],
    *,
    top_k: int = 15,
    max_features: int = 4096,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 1),
) -> dict[pd.Timestamp, list[tuple[str, float]]]:
    """
    Return top_k terms per window by TF-IDF weight for that window's row.

    Set ngram_range=(1,2) to include bigrams alongside unigrams.
    """
    if not window_to_text:
        return {}

    keys = sorted(window_to_text.keys())
    docs = [window_to_text[k] for k in keys]
    vec = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-z][a-z0-9']+\b",
        lowercase=True,
        ngram_range=ngram_range,
    )
    X = vec.fit_transform(docs)
    names = vec.get_feature_names_out()

    out: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    for row_idx, key in enumerate(keys):
        row = np.asarray(X[row_idx].todense()).ravel()
        if row.sum() == 0:
            out[key] = []
            continue
        idx = np.argsort(-row)[:top_k]
        out[key] = [(str(names[j]), float(row[j])) for j in idx if row[j] > 0]
    return out