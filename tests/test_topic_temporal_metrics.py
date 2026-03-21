import numpy as np

from src.evaluation.topic_temporal_metrics import mean_adjacent_jsd


def test_mean_adjacent_jsd():
    v_same = np.array([[1.0, 0, 0], [1.0, 0, 0]], dtype=float)
    assert mean_adjacent_jsd(["a", "b"], v_same) == 0.0
    v_diff = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=float)
    assert mean_adjacent_jsd(["x", "y"], v_diff) > 0.0
