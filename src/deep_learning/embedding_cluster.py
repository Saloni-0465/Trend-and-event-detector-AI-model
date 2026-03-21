"""High-level embedding + clustering entry points (used by `part3_runner`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.deep_learning.embedder import encode_texts
from src.deep_learning.kmeans_time import select_kmeans_by_train_silhouette


def embed_stream_texts(
    df: pd.DataFrame,
    text_col: str,
    **encode_kw,
) -> np.ndarray:
    texts = df[text_col].astype(str).tolist()
    return encode_texts(texts, **encode_kw)


def cluster_embeddings_time_split(
    X: np.ndarray,
    train_idx: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 12,
    random_state: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, KMeans, int, dict[int, float]]:
    """
    Returns (labels_all, fitted_kmeans, best_k, silhouette_by_k).

    `labels_all` assigns every row of X using the model fit on X[train_idx] only.
    """
    X_train = X[train_idx]
    sel = select_kmeans_by_train_silhouette(
        X_train,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
        max_iter=max_iter,
    )
    labels_all = sel.model.predict(X)
    return labels_all, sel.model, sel.best_k, sel.silhouette_by_k
