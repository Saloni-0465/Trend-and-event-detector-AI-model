"""
Simple burst / spike detection on term frequency time series.

Marks a bin as a "burst" if a term's count exceeds its rolling mean by more
than `z_thresh` standard deviations. This is a lightweight z-score method
(not Kleinberg's automaton) but is sufficient for the baseline track and
easy to explain in terms of stationarity and threshold sensitivity.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable

import numpy as np
import pandas as pd


def term_frequency_series(
    df: pd.DataFrame,
    term: str,
    *,
    bin_col: str = "time_bin",
    tokens_col: str = "tokens",
) -> pd.Series:
    """Count occurrences of `term` in each time bin → Series indexed by bin."""
    bins = sorted(df[bin_col].unique())
    counts: dict[Hashable, int] = {}
    for b in bins:
        mask = df[bin_col] == b
        c = 0
        for toks in df.loc[mask, tokens_col]:
            if isinstance(toks, list):
                c += toks.count(term)
        counts[b] = c
    return pd.Series(counts, name=term)


def detect_bursts(
    series: pd.Series,
    *,
    window: int = 3,
    z_thresh: float = 1.5,
    min_count: int = 2,
) -> pd.DataFrame:
    """
    Return DataFrame of bins flagged as burst for this term.

    `window`: number of preceding bins for rolling stats.
    `z_thresh`: how many std above rolling mean to call a burst.
    `min_count`: ignore bins with fewer than this many occurrences.
    """
    s = series.astype(float)
    roll_mean = s.rolling(window=window, min_periods=1).mean().shift(1)
    roll_std = s.rolling(window=window, min_periods=1).std(ddof=0).shift(1)

    roll_mean = roll_mean.fillna(s.expanding().mean())
    roll_std = roll_std.fillna(0.0)

    # When std is 0 (constant history), use deviation from mean directly:
    # any value above mean by at least z_thresh counts as a burst.
    deviation = s - roll_mean
    z = np.where(
        roll_std > 0,
        deviation / roll_std,
        np.where(deviation > 0, z_thresh + 1, 0.0),
    )
    is_burst = (z > z_thresh) & (s >= min_count)

    out = pd.DataFrame(
        {
            "time_bin": series.index,
            "count": s.values,
            "rolling_mean": roll_mean.values,
            "rolling_std": roll_std.values,
            "z_score": z,
            "is_burst": is_burst,
        }
    )
    return out


def detect_all_bursts(
    df: pd.DataFrame,
    terms: list[str],
    *,
    bin_col: str = "time_bin",
    tokens_col: str = "tokens",
    window: int = 3,
    z_thresh: float = 1.5,
    min_count: int = 2,
) -> pd.DataFrame:
    """Run burst detection for each term; return long-format DataFrame of burst bins."""
    rows: list[pd.DataFrame] = []
    for term in terms:
        ts = term_frequency_series(df, term, bin_col=bin_col, tokens_col=tokens_col)
        bd = detect_bursts(ts, window=window, z_thresh=z_thresh, min_count=min_count)
        bd["term"] = term
        bursts_only = bd[bd["is_burst"]]
        if not bursts_only.empty:
            rows.append(bursts_only)
    if not rows:
        return pd.DataFrame(columns=["term", "time_bin", "count", "z_score"])
    return pd.concat(rows, ignore_index=True)[
        ["term", "time_bin", "count", "rolling_mean", "z_score"]
    ]