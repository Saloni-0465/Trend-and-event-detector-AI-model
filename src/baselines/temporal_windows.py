"""Assign documents to calendar time bins for temporal topic baselines."""

from __future__ import annotations

import pandas as pd


def add_time_bin(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    freq: str = "7D",
    out_col: str = "time_bin",
) -> pd.DataFrame:
    """
    Floor timestamps to a pandas offset alias (e.g. '1D', '12H', '7D').

    Empty bins can appear downstream if the stream is sparse; callers should handle.
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing {time_col!r}")
    out = df.copy()
    ts = pd.to_datetime(out[time_col], utc=True)
    out[out_col] = ts.dt.floor(freq)
    return out


def aggregate_window_texts(
    df: pd.DataFrame,
    *,
    bin_col: str = "time_bin",
    tokens_col: str = "tokens",
) -> dict[pd.Timestamp, list[str]]:
    """Concatenate all tokens per time bin (order not meaningful for BOW baselines)."""
    if bin_col not in df.columns or tokens_col not in df.columns:
        raise ValueError("DataFrame must contain bin and tokens columns")
    grouped: dict[pd.Timestamp, list[str]] = {}
    for _, row in df.iterrows():
        b = row[bin_col]
        toks = row[tokens_col]
        if not isinstance(toks, list):
            toks = list(toks) if hasattr(toks, "__iter__") and not isinstance(toks, str) else []
        grouped.setdefault(b, []).extend(toks)
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def window_strings_from_tokens(
    window_tokens: dict[pd.Timestamp, list[str]],
) -> dict[pd.Timestamp, str]:
    """Join tokens for sklearn vectorizers (single string per bin)."""
    return {k: " ".join(v) for k, v in window_tokens.items()}
