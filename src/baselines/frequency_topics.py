"""Simple frequency-based pseudo-topics (baseline)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable

import pandas as pd


def top_terms_by_frequency(
    token_lists: list[list[str]], k: int = 20
) -> list[tuple[str, int]]:
    """Return top-k tokens by count across documents."""
    counts: Counter[str] = Counter()
    for doc in token_lists:
        counts.update(doc)
    return counts.most_common(k)


def top_terms_per_time_bin(
    df: pd.DataFrame,
    *,
    bin_col: str = "time_bin",
    tokens_col: str = "tokens",
    k: int = 15,
) -> dict[Hashable, list[tuple[str, int]]]:
    """Aggregate tokens within each time bin and return top-k counts per bin."""
    if bin_col not in df.columns or tokens_col not in df.columns:
        raise ValueError("DataFrame must contain bin and tokens columns")
    buckets: dict[Hashable, Counter[str]] = {}
    for _, row in df.iterrows():
        b = row[bin_col]
        toks = row[tokens_col]
        if not isinstance(toks, list):
            toks = list(toks) if hasattr(toks, "__iter__") and not isinstance(toks, str) else []
        if b not in buckets:
            buckets[b] = Counter()
        buckets[b].update(toks)
    return {b: cnt.most_common(k) for b, cnt in sorted(buckets.items(), key=lambda kv: kv[0])}
