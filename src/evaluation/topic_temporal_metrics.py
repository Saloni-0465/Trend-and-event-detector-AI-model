"""Metrics for topic mixtures over time (complement perplexity / coherence)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon


def mean_adjacent_jsd(
    ordered_bins: Sequence[Hashable],
    topic_vectors: np.ndarray,
) -> float:
    """
    Mean Jensen–Shannon distance between consecutive windows' topic distributions.

    Each row of `topic_vectors` must correspond to `ordered_bins` in the same order.
    """
    if topic_vectors.shape[0] != len(ordered_bins):
        raise ValueError("topic_vectors rows must match ordered_bins length")
    if len(ordered_bins) < 2:
        return float("nan")
    dists: list[float] = []
    for i in range(len(ordered_bins) - 1):
        p = np.asarray(topic_vectors[i], dtype=np.float64)
        q = np.asarray(topic_vectors[i + 1], dtype=np.float64)
        dists.append(float(jensenshannon(p, q)))
    return float(np.mean(dists))
