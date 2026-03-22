"""
HDBSCAN clustering as an alternative to K-Means.

Advantages over K-Means for event detection:
- **No need to prespecify K** — the algorithm finds the natural cluster count.
- **Noise label (-1)** — outlier documents that don't belong to any cluster,
  which maps well to "miscellaneous" social chatter or low-signal posts.
- Hierarchical density approach handles clusters of varying density.

Trade-offs:
- Slower on very large datasets than K-Means.
- Sensitive to `min_cluster_size` and `min_samples` hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class HDBSCANResult:
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    probabilities: np.ndarray


def run_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> HDBSCANResult:
    """Run HDBSCAN and return labels, cluster count, and noise count."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(X)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int((labels == -1).sum())
    probs = clusterer.probabilities_ if hasattr(clusterer, "probabilities_") else np.ones(len(X))
    return HDBSCANResult(
        labels=np.asarray(labels, dtype=int),
        n_clusters=n_clusters,
        n_noise=n_noise,
        probabilities=np.asarray(probs, dtype=np.float64),
    )


def hdbscan_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette excluding noise points (label == -1)."""
    mask = labels >= 0
    unique = np.unique(labels[mask])
    if mask.sum() < 2 or len(unique) < 2:
        return float("nan")
    return float(silhouette_score(X[mask], labels[mask]))