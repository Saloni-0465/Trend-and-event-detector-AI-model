"""
Diagnostics for *temporal* baselines (not full topic-model metrics yet).

Adjacent-window Jaccard on top-k term sets measures how much the lexical
signature changes from one bin to the next — useful to discuss stationarity
and drift before ARIMA/HMM/LDA in later phases.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def mean_adjacent_jaccard(
    ordered_bins: Sequence[Hashable],
    bin_to_terms: dict[Hashable, Sequence[str]],
) -> float:
    """
    Mean Jaccard between top-term sets of consecutive time bins.

    Returns float('nan') if fewer than two bins.
    """
    if len(ordered_bins) < 2:
        return float("nan")
    scores: list[float] = []
    for b0, b1 in zip(ordered_bins[:-1], ordered_bins[1:]):
        t0 = list(bin_to_terms.get(b0, []))
        t1 = list(bin_to_terms.get(b1, []))
        scores.append(jaccard_similarity(t0, t1))
    return sum(scores) / len(scores)
