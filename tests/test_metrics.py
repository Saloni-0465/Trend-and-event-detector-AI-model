import numpy as np

from src.evaluation.metrics import mae, rmse


def test_rmse_mae():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert rmse(y, y + 1) > 0
