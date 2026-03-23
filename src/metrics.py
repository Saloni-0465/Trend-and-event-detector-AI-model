"""
Simple evaluation metrics (easy to explain in viva).
"""

import numpy as np
from scipy.spatial.distance import jensenshannon


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
