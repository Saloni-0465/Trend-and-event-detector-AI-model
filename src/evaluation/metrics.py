"""Central place for MAE/RMSE, clustering metrics, perplexity/coherence, etc."""

from __future__ import annotations


def rmse(y_true, y_pred) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))
