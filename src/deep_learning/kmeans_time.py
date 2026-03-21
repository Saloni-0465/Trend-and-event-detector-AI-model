"""
K-Means on embeddings with **k chosen on the training time slice** via silhouette.

This is explicit **model selection / complexity control** (pick K without peeking at
the held-out tail of the timeline). `max_iter` caps coordinate-descent steps — a simple
analogy to iterative optimization limits discussed in the course.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class KMeansSelectionResult:
    model: KMeans
    best_k: int
    silhouette_by_k: dict[int, float]


def select_kmeans_by_train_silhouette(
    X_train: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 12,
    random_state: int = 42,
    max_iter: int = 300,
) -> KMeansSelectionResult:
    """
    Grid-search cluster count k on **training** embeddings only; maximize silhouette.

    Refits a final `KMeans` with `best_k` on all of `X_train`.
    """
    n = len(X_train)
    if n < k_min + 1:
        raise ValueError(f"Need more than k_min training points; got n={n}")

    upper = min(k_max, n - 1)
    if upper < k_min:
        raise ValueError(f"Cannot try k in [{k_min}, {k_max}] with only n={n} training points")

    scores: dict[int, float] = {}
    best_k = k_min
    best_s = -np.inf

    for k in range(k_min, upper + 1):
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto",
            max_iter=max_iter,
        )
        labels = km.fit_predict(X_train)
        if len(np.unique(labels)) < 2:
            scores[k] = float("nan")
            continue
        s = float(silhouette_score(X_train, labels))
        scores[k] = s
        if not np.isnan(s) and s > best_s:
            best_s = s
            best_k = k

    final = KMeans(
        n_clusters=best_k,
        random_state=random_state,
        n_init="auto",
        max_iter=max_iter,
    )
    final.fit(X_train)
    return KMeansSelectionResult(
        model=final, best_k=best_k, silhouette_by_k=scores
    )
