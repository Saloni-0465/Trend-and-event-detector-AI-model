"""
Simple evaluation metrics (easy to explain in viva).
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def jaccard(a, b):
    """Overlap between two word sets (0 = different, 1 = same)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mean_adjacent_jaccard(bins, terms):
    """How similar words are between consecutive time windows."""
    if len(bins) < 2:
        return float("nan")
    scores = []
    for b1, b2 in zip(bins[:-1], bins[1:]):
        scores.append(jaccard(terms[b1], terms[b2]))
    return float(np.mean(scores))


def mean_adjacent_jsd(bins, vectors):
    """How much topic distribution changes between adjacent time windows."""
    if len(bins) < 2:
        return float("nan")
    dists = []
    for i in range(len(bins) - 1):
        p = np.asarray(vectors[i], dtype=float)
        q = np.asarray(vectors[i + 1], dtype=float)
        dists.append(float(jensenshannon(p, q)))
    return float(np.mean(dists))


def safe_silhouette(X, labels):
    """Silhouette score; nan if fewer than 2 clusters or too few points."""
    if len(X) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def cluster_category_alignment(y_category, y_cluster):
    """
    Compare unsupervised cluster ids to editorial category labels (weak signals).

    NMI / ARI are standard for clustering vs external labels. Neither requires
    the cluster id to match the category id (permutation-invariant).
    """
    y_category = np.asarray(y_category)
    y_cluster = np.asarray(y_cluster)
    if len(y_category) != len(y_cluster):
        raise ValueError("y_category and y_cluster must have the same length")
    if len(y_category) < 2:
        return float("nan"), float("nan")
    if len(np.unique(y_cluster)) < 2:
        return float("nan"), float("nan")

    nmi = float(normalized_mutual_info_score(y_category, y_cluster))
    ari = float(adjusted_rand_score(y_category, y_cluster))
    return nmi, ari


def mean_adjacent_cosine_distance(vectors):
    """Mean pairwise 1 - cos_sim between consecutive L2-normalized row vectors."""
    if len(vectors) < 2:
        return float("nan")
    dists = []
    for i in range(len(vectors) - 1):
        a = np.asarray(vectors[i], dtype=float).ravel()
        b = np.asarray(vectors[i + 1], dtype=float).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        a = a / na
        b = b / nb
        sim = float(np.clip(np.dot(a, b), -1.0, 1.0))
        dists.append(1.0 - sim)
    if not dists:
        return float("nan")
    return float(np.mean(dists))
