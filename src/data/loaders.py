"""Load timestamped text streams from CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_stream_csv(
    path: str | Path,
    *,
    text_col: str = "text",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Read a CSV with at least text and timestamp columns.

    Timestamps are parsed with pandas (ISO-8601 recommended). Rows with null
    time or empty text are dropped.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    for col in (text_col, time_col):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r}; have {list(df.columns)}")

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df.dropna(subset=[time_col])
    df = df[df[text_col].str.len() > 0]
    return df.reset_index(drop=True)
