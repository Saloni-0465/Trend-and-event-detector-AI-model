"""Supervised clustering scores when a reference label exists (e.g. synthetic theme)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def encode_categorical_labels(series: pd.Series) -> np.ndarray:
    """Stable integer codes for string labels."""
    cat = pd.Categorical(series.astype(str))
    return np.asarray(cat.codes, dtype=int)


def clustering_supervised_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    subset: np.ndarray | None = None,
) -> dict[str, float]:
    """NMI and ARI; if subset indices provided, score only those rows."""
    if subset is not None:
        y_true = y_true[subset]
        y_pred = y_pred[subset]
    if len(y_true) == 0:
        return {"nmi": float("nan"), "ari": float("nan")}
    return {
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }


def silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    if len(X) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))
