import numpy as np
import pandas as pd
import pytest

from src.hybrid import (
    ablation_table,
    category_distribution_drift,
    fuse_transition_scores,
    minmax_scale,
    precision_recall_f1_at_percentile,
)


def test_minmax_scale_constant_returns_zeros():
    assert minmax_scale([3, 3, 3]) == [0.0, 0.0, 0.0]


def test_category_distribution_drift_detects_shift():
    bins = pd.to_datetime(["2022-01-01", "2022-01-08", "2022-01-15"], utc=True)
    df = pd.DataFrame(
        {
            "time_bin": [bins[0]] * 10 + [bins[1]] * 10 + [bins[2]] * 10,
            "category": ["A"] * 9 + ["B"] + ["A"] * 8 + ["B"] * 2 + ["A"] + ["B"] * 9,
        }
    )
    drift = category_distribution_drift(df, list(bins))
    assert len(drift) == 2
    assert drift[1] > drift[0]


def test_fuse_transition_scores_aligns_and_weights_components():
    transitions = [(0, 1), (1, 2), (2, 3)]
    rows = fuse_transition_scores(
        transitions=transitions,
        lexical_similarity=[0.9, 0.5, 0.1],
        lda_jsd=[0.0, 0.5, 1.0],
        semantic_cosine_distance=[0.0, 0.2, 0.4],
    )
    assert len(rows) == 3
    assert rows[0]["start_bin"] == 0
    assert rows[-1]["hybrid_score"] == pytest.approx(1.0)
    assert rows[0]["hybrid_score"] == pytest.approx(0.0)


def test_precision_recall_f1_at_percentile_matches_top_events():
    metrics = precision_recall_f1_at_percentile(
        scores=[0.1, 0.9, 0.2, 0.8],
        targets=[0.0, 1.0, 0.1, 0.7],
        percentile=75,
    )
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_ablation_table_reports_three_models():
    fused_rows = [
        {"ml_only_score": 0.1, "dl_only_score": 0.2, "hybrid_score": 0.15},
        {"ml_only_score": 0.8, "dl_only_score": 0.7, "hybrid_score": 0.9},
        {"ml_only_score": 0.2, "dl_only_score": 0.1, "hybrid_score": 0.25},
    ]
    table = ablation_table(fused_rows, [0.0, 1.0, 0.2], percentile=80)
    assert [r["model"] for r in table] == ["ML-only", "DL-only", "Hybrid"]
    assert np.isfinite([r["f1"] for r in table]).all()
