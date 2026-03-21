"""Aggregate LDA document-topic mixtures over calendar bins."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
import pandas as pd


def mean_topic_mixture_by_bin(
    df: pd.DataFrame,
    theta: np.ndarray,
    *,
    bin_col: str = "time_bin",
) -> tuple[list[Hashable], np.ndarray]:
    """
    Row-aligned `theta` (n_docs, K) with `df`; return sorted bin keys and mean vectors.
    """
    if len(df) != len(theta):
        raise ValueError("df rows must match theta rows")
    if bin_col not in df.columns:
        raise ValueError(f"Missing {bin_col!r}")

    tmp = pd.DataFrame({bin_col: df[bin_col].values})
    for k in range(theta.shape[1]):
        tmp[f"t{k}"] = theta[:, k]
    g = tmp.groupby(bin_col, sort=True).mean(numeric_only=True)
    keys_sorted: list[Hashable] = list(g.index)
    cols = [f"t{k}" for k in range(theta.shape[1])]
    mat = g[cols].to_numpy(dtype=np.float64)
    return keys_sorted, mat
