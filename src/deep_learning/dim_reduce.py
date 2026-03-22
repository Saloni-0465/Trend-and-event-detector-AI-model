"""
UMAP dimensionality reduction on sentence embeddings.

UMAP preserves both local and global structure better than t-SNE for
downstream clustering and produces deterministic results with a fixed
`random_state`. The 2D projection is mainly for visualization and report
figures; clustering is performed in the original high-dimensional space.
"""

from __future__ import annotations

import numpy as np


def umap_reduce(
    X: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Project high-dimensional embeddings to `n_components` dims via UMAP."""
    from umap import UMAP

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(X), dtype=np.float32)