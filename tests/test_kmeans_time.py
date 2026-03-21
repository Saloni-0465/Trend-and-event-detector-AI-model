import numpy as np

from src.deep_learning.kmeans_time import select_kmeans_by_train_silhouette


def test_select_k_returns_fitted_model():
    rng = np.random.default_rng(0)
    X = np.vstack(
        [
            rng.standard_normal((40, 8)) + np.array([3.0, 0, 0, 0, 0, 0, 0, 0]),
            rng.standard_normal((40, 8)) + np.array([-3.0, 0, 0, 0, 0, 0, 0, 0]),
        ]
    ).astype(np.float32)
    res = select_kmeans_by_train_silhouette(X, k_min=2, k_max=6, random_state=0)
    assert res.best_k >= 2
    assert res.model.n_clusters == res.best_k
    lab = res.model.predict(X)
    assert len(lab) == len(X)
