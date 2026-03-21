"""
Part 2 — Advanced ML: LDA (probabilistic topics) + temporal aggregation.

- **Train / test split by time** (no random shuffling): earlier posts train, later hold out.
- **Metrics:** held-out log perplexity, topic coherence (default `u_mass` for speed).
- **Temporal view:** mean document-topic vector per `time_bin` + adjacent-window JSD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.advanced_ml.lda_gensim import (
    bows_with_fixed_dictionary,
    coherence_for_model,
    document_topic_matrix,
    log_perplexity_on_corpus,
    top_words_per_topic,
    train_lda,
)
from src.advanced_ml.temporal_lda import mean_topic_mixture_by_bin
from src.baselines.temporal_windows import add_time_bin
from src.data.preprocess import add_tokens_column
from src.evaluation.topic_temporal_metrics import mean_adjacent_jsd
from src.pipeline_common import Source, load_raw_stream


def _time_ordered_split(n: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    cut = max(1, int(np.floor(n * train_frac)))
    cut = min(cut, n - 1) if n > 1 else 1
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


def run_part2(
    *,
    source: Source = "synthetic",
    csv_path: str | Path | None = None,
    text_col: str = "text",
    time_col: str = "timestamp",
    window_freq: str = "7D",
    num_topics: int = 8,
    passes: int = 15,
    train_frac: float = 0.75,
    seed: int = 42,
    n_docs: int = 800,
    start: str = "2024-01-01",
    end: str = "2024-03-01",
    coherence: Literal["u_mass", "c_v"] = "u_mass",
    no_below: int = 1,
    no_above: float = 0.85,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
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
    df = df.sort_values(time_col).reset_index(drop=True)
    df = add_tokens_column(df, text_col=text_col, out_col="tokens")
    df = add_time_bin(df, time_col=time_col, freq=window_freq, out_col="time_bin")

    texts = df["tokens"].tolist()
    n = len(texts)
    train_idx, test_idx = _time_ordered_split(n, train_frac)
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]

    artifacts = train_lda(
        train_texts,
        num_topics=num_topics,
        passes=passes,
        random_state=seed,
        no_below=no_below,
        no_above=no_above,
    )
    model, dictionary = artifacts.model, artifacts.dictionary

    test_corpus = bows_with_fixed_dictionary(test_texts, dictionary)
    train_corpus = artifacts.corpus_train

    perp_test = log_perplexity_on_corpus(model, test_corpus)
    perp_train = log_perplexity_on_corpus(model, train_corpus)

    try:
        coh = coherence_for_model(
            model,
            train_texts,
            dictionary,
            coherence=coherence,
            topn=10,
        )
    except Exception:
        coh = float("nan")

    all_bows = bows_with_fixed_dictionary(texts, dictionary)
    theta = document_topic_matrix(model, all_bows, n_topics=model.num_topics)
    bin_keys, bin_theta = mean_topic_mixture_by_bin(df, theta, bin_col="time_bin")
    jsd = mean_adjacent_jsd(bin_keys, bin_theta)

    topics = top_words_per_topic(model, num_words=12)
    topic_rows: list[dict[str, Any]] = []
    for tid, terms in enumerate(topics):
        topic_rows.append(
            {
                "topic_id": tid,
                "top_words": " ".join(w for w, _ in terms[:12]),
            }
        )
    topics_df = pd.DataFrame(topic_rows)

    mix_rows: list[dict[str, Any]] = []
    for j, b in enumerate(bin_keys):
        row: dict[str, Any] = {
            "time_bin": b.isoformat() if hasattr(b, "isoformat") else str(b),
        }
        for k in range(bin_theta.shape[1]):
            row[f"topic_{k}_mean"] = float(bin_theta[j, k])
        mix_rows.append(row)
    mix_df = pd.DataFrame(mix_rows)

    summary = pd.DataFrame(
        [
            {
                "metric": "log_perplexity_train_bow",
                "value": perp_train,
                "num_topics": num_topics,
                "passes": passes,
                "train_frac": train_frac,
                "n_train_docs": int(len(train_idx)),
                "n_test_docs": int(len(test_idx)),
                "coherence": coherence,
            },
            {
                "metric": "log_perplexity_test_bow",
                "value": perp_test,
                "num_topics": num_topics,
                "passes": passes,
                "train_frac": train_frac,
                "n_train_docs": int(len(train_idx)),
                "n_test_docs": int(len(test_idx)),
                "coherence": coherence,
            },
            {
                "metric": f"coherence_{coherence}",
                "value": coh,
                "num_topics": num_topics,
                "passes": passes,
                "train_frac": train_frac,
                "n_train_docs": int(len(train_idx)),
                "n_test_docs": int(len(test_idx)),
                "coherence": coherence,
            },
            {
                "metric": "mean_adjacent_jsd_topic_mix",
                "value": jsd,
                "num_topics": num_topics,
                "passes": passes,
                "train_frac": train_frac,
                "n_train_docs": int(len(train_idx)),
                "n_test_docs": int(len(test_idx)),
                "coherence": coherence,
            },
        ]
    )

    out: dict[str, Any] = {
        "df": df,
        "lda_model": model,
        "dictionary": dictionary,
        "theta_docs": theta,
        "topic_mix_by_window": mix_df,
        "topics": topics_df,
        "summary_metrics": summary,
    }

    if output_dir is not None:
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        mix_df.to_csv(od / "part2_topic_mix_by_window.csv", index=False)
        topics_df.to_csv(od / "part2_lda_top_words.csv", index=False)
        summary.to_csv(od / "part2_summary_metrics.csv", index=False)

    return out
