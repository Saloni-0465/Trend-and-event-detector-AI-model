"""Event detection from drift series.

This module implements a simple, explainable "spike" detector:
1) compute drift scores between adjacent time bins
2) pick bins where drift is unusually high (percentile threshold)
3) merge consecutive spikes into event windows

We intentionally keep this baseline simple so the project can move
from "trend metrics" to "event windows" quickly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.metrics import jaccard


@dataclass(frozen=True)
class Event:
    event_id: int
    start_bin: Any
    end_bin: Any
    max_score: float
    # For debug/explanation: which adjacent transition caused max_score.
    spike_transition_index: int

    # Optional drivers (model-dependent)
    drivers: dict[str, Any]


def detect_spike_events(
    drift_scores: list[float],
    transitions: list[tuple[Any, Any]],
    *,
    percentile: float = 90.0,
    threshold: float | None = None,
    merge_consecutive: bool = True,
) -> list[Event]:
    """Detect event windows from adjacent-bin drift scores.

    drift_scores[i] corresponds to transitions[i] = (bin_i, bin_{i+1}).
    """
    if len(drift_scores) == 0:
        return []
    if len(drift_scores) != len(transitions):
        raise ValueError("drift_scores and transitions must have same length")
    if len(drift_scores) == 1:
        threshold_val = drift_scores[0] if threshold is None else threshold
        if drift_scores[0] >= threshold_val:
            b0, b1 = transitions[0]
            return [
                Event(
                    event_id=1,
                    start_bin=b0,
                    end_bin=b1,
                    max_score=float(drift_scores[0]),
                    spike_transition_index=0,
                    drivers={},
                )
            ]
        return []

    threshold_val = float(np.percentile(drift_scores, percentile)) if threshold is None else float(threshold)
    spike_indices = [i for i, s in enumerate(drift_scores) if s >= threshold_val]
    if not spike_indices:
        return []

    groups: list[list[int]] = []
    current = [spike_indices[0]]
    for idx in spike_indices[1:]:
        if merge_consecutive and idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    events: list[Event] = []
    for event_id, g in enumerate(groups, start=1):
        first_i, last_i = g[0], g[-1]
        start_bin = transitions[first_i][0]
        end_bin = transitions[last_i][1]
        # Find the transition with the max drift score in this event window.
        local_max_i = max(g, key=lambda i: drift_scores[i])
        max_score = float(drift_scores[local_max_i])
        events.append(
            Event(
                event_id=event_id,
                start_bin=start_bin,
                end_bin=end_bin,
                max_score=max_score,
                spike_transition_index=local_max_i,
                drivers={},
            )
        )
    return events


def jsd_adjacent_from_topic_means(
    bin_means: list[np.ndarray],
) -> list[float]:
    """Compute Jensen-Shannon distance between adjacent topic mixtures."""
    if len(bin_means) < 2:
        return []
    dists: list[float] = []
    for i in range(len(bin_means) - 1):
        p = np.asarray(bin_means[i], dtype=float)
        q = np.asarray(bin_means[i + 1], dtype=float)
        # If numerical issues produce non-probability vectors, renormalize.
        if p.sum() > 0:
            p = p / p.sum()
        if q.sum() > 0:
            q = q / q.sum()
        dists.append(float(jensenshannon(p, q)))
    return dists


def lexical_jaccard_drift_from_top_terms(
    freq_terms_by_bin: dict[Any, list[tuple[str, int]]],
    *,
    top_k: int,
) -> tuple[list[float], list[tuple[Any, Any]]]:
    """Compute adjacent-bin lexical drift as Jaccard over top terms."""
    bins = sorted(freq_terms_by_bin.keys())
    if len(bins) < 2:
        return [], []

    scores: list[float] = []
    transitions: list[tuple[Any, Any]] = []
    for i in range(len(bins) - 1):
        b0, b1 = bins[i], bins[i + 1]
        t0 = [w for w, _ in freq_terms_by_bin.get(b0, [])[:top_k]]
        t1 = [w for w, _ in freq_terms_by_bin.get(b1, [])[:top_k]]
        scores.append(float(jaccard(t0, t1)))
        transitions.append((b0, b1))
    return scores, transitions


def _dominant_category_in_range(
    test_df: pd.DataFrame,
    *,
    start_bin: Any,
    end_bin: Any,
) -> str | None:
    if "category" not in test_df.columns or test_df.empty:
        return None

    ts = pd.to_datetime(test_df["time_bin"], utc=True, errors="coerce")
    tb0 = pd.to_datetime(start_bin, utc=True)
    tb1 = pd.to_datetime(end_bin, utc=True)
    mask = (ts >= tb0) & (ts <= tb1)
    cats = test_df.loc[mask, "category"].dropna()
    if cats.empty:
        return None

    counts = cats.value_counts()
    top_count = counts.max()
    tied = sorted([c for c, n in counts.items() if n == top_count])
    return tied[0] if tied else None


def label_events(
    events_out: dict[str, list[dict[str, Any]]],
    test_df: pd.DataFrame,
    *,
    include_category: bool = True,
    lexical_label_top_terms: int = 6,
    lda_label_top_words: int = 6,
) -> dict[str, list[dict[str, Any]]]:
    """Attach a human-readable label to each detected event.

    For lexical events: uses `drivers.new_terms` / `drivers.dropped_terms`.
    For LDA events: uses `drivers.top_topics[*].words`.
    Optionally adds the dominant dataset `category` within the event window.
    """
    labeled = {"lexical": [], "lda": []}

    for ev in events_out.get("lexical", []):
        new_terms = ev.get("drivers", {}).get("new_terms") or []
        dropped_terms = ev.get("drivers", {}).get("dropped_terms") or []
        chosen = new_terms[:lexical_label_top_terms] or dropped_terms[:lexical_label_top_terms]
        label = "Lexical change: " + (" ".join(chosen) if chosen else "topic drift")

        cat = None
        if include_category:
            cat = _dominant_category_in_range(
                test_df, start_bin=ev["start_bin"], end_bin=ev["end_bin"]
            )

        labeled["lexical"].append(
            {
                **ev,
                "label": label,
                "dominant_category": cat,
            }
        )

    for ev in events_out.get("lda", []):
        top_topics = ev.get("drivers", {}).get("top_topics") or []
        words: list[str] = []
        for t in top_topics:
            for w in (t.get("words") or []):
                if w not in words:
                    words.append(w)
                if len(words) >= lda_label_top_words:
                    break
            if len(words) >= lda_label_top_words:
                break

        label = "Topic-mixture change: " + (" ".join(words) if words else "topic drift")

        cat = None
        if include_category:
            cat = _dominant_category_in_range(
                test_df, start_bin=ev["start_bin"], end_bin=ev["end_bin"]
            )

        labeled["lda"].append(
            {
                **ev,
                "label": label,
                "dominant_category": cat,
            }
        )

    return labeled

