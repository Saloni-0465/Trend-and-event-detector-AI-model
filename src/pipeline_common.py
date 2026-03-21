"""Shared data loading for pipeline parts (Part 1, Part 2, …)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data.loaders import load_stream_csv
from src.data.synthetic import generate_synthetic_social_stream

Source = Literal["synthetic", "csv"]


def load_raw_stream(
    *,
    source: Source = "synthetic",
    csv_path: str | Path | None = None,
    text_col: str = "text",
    time_col: str = "timestamp",
    seed: int = 42,
    n_docs: int = 800,
    start: str = "2024-01-01",
    end: str = "2024-03-01",
) -> pd.DataFrame:
    if source == "synthetic":
        return generate_synthetic_social_stream(
            n_docs=n_docs, seed=seed, start=start, end=end
        )
    if not csv_path:
        raise ValueError("csv_path is required when source='csv'")
    return load_stream_csv(csv_path, text_col=text_col, time_col=time_col)


def time_ordered_split(n: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices for [train | test] along a pre-sorted timeline (no shuffling)."""
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    cut = max(1, int(np.floor(n * train_frac)))
    cut = min(cut, n - 1) if n > 1 else 1
    idx = np.arange(n)
    return idx[:cut], idx[cut:]
