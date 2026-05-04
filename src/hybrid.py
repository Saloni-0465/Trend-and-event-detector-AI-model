"""Hybrid trend/event scoring and ablation utilities.

The hybrid layer fuses three transition-level signals:

* lexical drift from top terms (symbolic ML baseline)
* LDA topic-mixture drift (probabilistic ML)
* embedding centroid drift (neural semantic signal)

All signals are computed on the same adjacent time-bin transitions. The fusion
score is intentionally transparent: normalize each signal within a run, then
take a weighted average. This makes ablations meaningful because ML-only,
DL-only, and Hybrid are scored against the same weak category-drift proxy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def minmax_scale(values: Iterable[float]) -> list[float]:
    """Scale a numeric sequence to [0, 1], returning zeros for constants."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return [0.0 for _ in arr]
    return [float((v - lo) / (hi - lo)) for v in arr]


def invert_similarity(similarities: Iterable[float]) -> list[float]:
    """Convert similarity values into drift values."""
    return [float(1.0 - s) for s in similarities]


def category_distribution_drift(
    df: pd.DataFrame,
    bins: list[Any],
    *,
    category_col: str = "category",
    time_col: str = "time_bin",
) -> list[float]:
    """Weak event target: adjacent Jensen-Shannon drift over editor categories."""
    if category_col not in df.columns or len(bins) < 2:
        return []

    categories = sorted(df[category_col].dropna().astype(str).unique())
    if not categories:
        return []

    rows: list[np.ndarray] = []
    for b in bins:
        part = df.loc[df[time_col] == b, category_col].dropna().astype(str)
        counts = part.value_counts()
        vec = np.asarray([float(counts.get(c, 0.0)) for c in categories], dtype=float)
        total = vec.sum()
        rows.append(vec / total if total > 0 else np.ones(len(categories)) / len(categories))

    dists: list[float] = []
    for i in range(len(rows) - 1):
        p = rows[i]
        q = rows[i + 1]
        m = 0.5 * (p + q)
        eps = 1e-12
        kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0))
        kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0))
        dists.append(float(np.sqrt(0.5 * (kl_pm + kl_qm))))
    return dists


def precision_recall_f1_at_percentile(
    scores: Iterable[float],
    targets: Iterable[float],
    *,
    percentile: float = 90.0,
) -> dict[str, float]:
    """Evaluate top-score transitions against top-target transitions."""
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(targets), dtype=float)
    if s.size == 0 or y.size == 0 or s.size != y.size:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    pred_threshold = float(np.percentile(s, percentile))
    target_threshold = float(np.percentile(y, percentile))
    pred = s >= pred_threshold
    truth = y >= target_threshold

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def fuse_transition_scores(
    *,
    transitions: list[tuple[Any, Any]],
    lexical_similarity: list[float],
    lda_jsd: list[float],
    semantic_cosine_distance: list[float],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Create aligned transition rows with ML-only, DL-only, and Hybrid scores."""
    n = min(
        len(transitions),
        len(lexical_similarity),
        len(lda_jsd),
        len(semantic_cosine_distance),
    )
    if n == 0:
        return []

    # Short news text benefits from heavier probabilistic topic weight for
    # interpretability, with neural drift acting as a semantic correction.
    w = {"lexical": 0.30, "lda": 0.56, "semantic": 0.14}
    if weights:
        w.update(weights)
    total = sum(max(0.0, float(v)) for v in w.values())
    if total <= 0:
        raise ValueError("At least one fusion weight must be positive")
    w = {k: max(0.0, float(v)) / total for k, v in w.items()}

    lexical_drift = invert_similarity(lexical_similarity[:n])
    lex_norm = minmax_scale(lexical_drift)
    lda_norm = minmax_scale(lda_jsd[:n])
    sem_norm = minmax_scale(semantic_cosine_distance[:n])

    rows: list[dict[str, Any]] = []
    for i in range(n):
        ml_only = 0.5 * lex_norm[i] + 0.5 * lda_norm[i]
        dl_only = sem_norm[i]
        hybrid = w["lexical"] * lex_norm[i] + w["lda"] * lda_norm[i] + w["semantic"] * sem_norm[i]
        b0, b1 = transitions[i]
        rows.append(
            {
                "start_bin": b0,
                "end_bin": b1,
                "lexical_drift": lexical_drift[i],
                "lda_jsd": float(lda_jsd[i]),
                "semantic_cosine_distance": float(semantic_cosine_distance[i]),
                "ml_only_score": float(ml_only),
                "dl_only_score": float(dl_only),
                "hybrid_score": float(hybrid),
            }
        )
    return rows


def ablation_table(
    fused_rows: list[dict[str, Any]],
    target_drift: list[float],
    *,
    percentile: float = 90.0,
) -> list[dict[str, float | str]]:
    """Return comparable ML-only, DL-only, and Hybrid metrics."""
    specs = [
        ("ML-only", "ml_only_score"),
        ("DL-only", "dl_only_score"),
        ("Hybrid", "hybrid_score"),
    ]
    out: list[dict[str, float | str]] = []
    for name, key in specs:
        scores = [float(r[key]) for r in fused_rows]
        metrics = precision_recall_f1_at_percentile(
            scores,
            target_drift[: len(fused_rows)],
            percentile=percentile,
        )
        try:
            from scipy.stats import spearmanr

            rho = float(spearmanr(scores, target_drift[: len(fused_rows)]).statistic)
        except Exception:
            rho = float("nan")
        out.append({"model": name, **metrics, "spearman": rho})
    return out
