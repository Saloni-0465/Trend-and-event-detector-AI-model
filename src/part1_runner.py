"""
Part 1 — end-to-end baseline track: data → time bins → frequency & TF-IDF → metrics.

This is intentionally *not* probabilistic topic modeling yet (Phase 2); it establishes
the temporal evaluation harness and strong frequency baselines required by the brief.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines.frequency_topics import top_terms_per_time_bin
from src.baselines.temporal_windows import (
    add_time_bin,
    aggregate_window_texts,
    window_strings_from_tokens,
)
from src.baselines.tfidf_topics import tfidf_top_terms_per_window
from src.data.preprocess import add_tokens_column
from src.evaluation.temporal_baseline_metrics import mean_adjacent_jaccard
from src.pipeline_common import Source, load_raw_stream


def _terms_only(ranked: list[tuple[str, float | int]]) -> list[str]:
    return [w for w, _ in ranked]


def run_part1(
    *,
    source: Source = "synthetic",
    csv_path: str | Path | None = None,
    text_col: str = "text",
    time_col: str = "timestamp",
    window_freq: str = "7D",
    top_k: int = 15,
    seed: int = 42,
    n_docs: int = 800,
    start: str = "2024-01-01",
    end: str = "2024-03-01",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load or synthesize data, run baselines, compute adjacent-window Jaccard.

    If output_dir is set, writes part1_window_terms.csv (long format).
    """
    df = load_raw_stream(
        source=source,
        csv_path=csv_path,
        text_col=text_col,
        time_col=time_col,
        seed=seed,
        n_docs=n_docs,
        start=start,
        end=end,
    )

    df = add_tokens_column(df, text_col=text_col, out_col="tokens")
    df = add_time_bin(df, time_col=time_col, freq=window_freq, out_col="time_bin")

    freq_ranked = top_terms_per_time_bin(df, bin_col="time_bin", tokens_col="tokens", k=top_k)
    ordered_bins = list(freq_ranked.keys())
    freq_terms_only = {b: _terms_only(freq_ranked[b]) for b in ordered_bins}

    wtoks = aggregate_window_texts(df, bin_col="time_bin", tokens_col="tokens")
    wtext = window_strings_from_tokens(wtoks)
    tfidf_ranked = tfidf_top_terms_per_window(wtext, top_k=top_k)
    tfidf_terms_only = {b: _terms_only(tfidf_ranked.get(b, [])) for b in ordered_bins}

    j_freq = mean_adjacent_jaccard(ordered_bins, freq_terms_only)
    j_tfidf = mean_adjacent_jaccard(ordered_bins, tfidf_terms_only)

    long_rows: list[dict[str, Any]] = []
    for b in ordered_bins:
        for w, c in freq_ranked[b]:
            long_rows.append(
                {
                    "time_bin": b.isoformat() if hasattr(b, "isoformat") else str(b),
                    "method": "frequency",
                    "term": w,
                    "score": int(c),
                }
            )
        for w, s in tfidf_ranked.get(b, []):
            long_rows.append(
                {
                    "time_bin": b.isoformat() if hasattr(b, "isoformat") else str(b),
                    "method": "tfidf",
                    "term": w,
                    "score": float(s),
                }
            )

    summary = pd.DataFrame(
        [
            {
                "metric": "mean_adjacent_jaccard_top_terms",
                "method": "frequency",
                "value": j_freq,
                "n_windows": len(ordered_bins),
                "window_freq": window_freq,
                "top_k": top_k,
            },
            {
                "metric": "mean_adjacent_jaccard_top_terms",
                "method": "tfidf",
                "value": j_tfidf,
                "n_windows": len(ordered_bins),
                "window_freq": window_freq,
                "top_k": top_k,
            },
        ]
    )

    out: dict[str, Any] = {
        "df": df,
        "freq_ranked": freq_ranked,
        "tfidf_ranked": tfidf_ranked,
        "summary_metrics": summary,
        "long_table": pd.DataFrame(long_rows),
    }

    if output_dir is not None:
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        out["long_table"].to_csv(od / "part1_window_terms.csv", index=False)
        summary.to_csv(od / "part1_summary_metrics.csv", index=False)

    return out
