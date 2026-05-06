"""Sentence embeddings + K-Means clustering.

DL path: a *pretrained* sentence encoder (MiniLM) maps each document to a
fixed-size vector. We do not fine-tune; K-Means groups documents in that space.
This matches a common, defensible "transfer learning" setup for text.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_encoder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
    """Load a SentenceTransformer once; reuse for multiple encode() calls."""
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    texts,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=None,
    model=None,
    show_progress_bar: bool = False,
):
    """
    Encode texts. Pass ``model`` from a previous call to avoid reloading weights.

    Returns
    -------
    embeddings : np.ndarray, shape (n, d), float32
    model : SentenceTransformer
    """
    if not texts:
        if model is not None:
            return np.zeros((0, 0), dtype=np.float32), model
        m = load_encoder(model_name=model_name, device=device)
        return np.zeros((0, 0), dtype=np.float32), m

    m = model if model is not None else load_encoder(model_name=model_name, device=device)
    emb = m.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return np.asarray(emb, dtype=np.float32), m


def mean_l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize each row; safe for zero rows."""
    v = np.asarray(vecs, dtype=np.float32)
    if v.size == 0:
        return v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return (v / n).astype(np.float32)


def mean_embedding_per_bin(df, X, time_col="time_bin") -> tuple[list, list[np.ndarray]]:
    """
    For each time bin (ordered), mean-pool document embeddings, then L2-normalize
    the centroid (for comparable cosine-based drift).
    """
    if len(df) != len(X):
        raise ValueError("df and X must have the same number of rows")
    bins = sorted(df[time_col].unique())
    centroids: list[np.ndarray] = []
    for b in bins:
        mask = (df[time_col] == b).values
        c = X[mask].mean(axis=0)
        c2 = mean_l2_normalize(c.reshape(1, -1))[0]
        centroids.append(c2)
    return bins, centroids


def find_best_k(X, k_min=2, k_max=10, seed=42):
    """Try different k values and pick the one with best silhouette score."""
    best_k = k_min
    best_score = -1
    scores = {}

    upper = min(k_max, len(X) - 1)
    for k in range(k_min, upper + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        scores[k] = s
        if s > best_score:
            best_score = s
            best_k = k

    return best_k, scores


def cluster(X, k, seed=42):
    """Run K-Means with given k on all rows. Returns labels for X."""
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(X), km


def cluster_time_split(X_train, X_test, k, seed=42):
    """Fit K-Means on train embeddings only; assign train + test labels."""
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    y_train = km.fit_predict(X_train)
    y_test = km.predict(X_test)
    labels = np.concatenate([y_train, y_test])
    return labels, km
