"""Sentence embeddings + K-Means clustering."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def encode_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
    """Encode texts using a pretrained sentence transformer (frozen weights)."""
    from sentence_transformers import SentenceTransformer
    import torch

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


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
