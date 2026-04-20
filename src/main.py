"""
Main script to run the pipeline (Phase 1).

Uses one fixed chronological train/test split for all modes (fair comparison).
Default: train 2018-01-01 .. 2018-05-01 (exclusive), test 2018-05-01 .. 2018-07-01 (exclusive).
"""

import os
import argparse

import numpy as np
import pandas as pd

from src.data_utils import (
    load_csv,
    tokenize_df,
    add_time_bins,
    corpus_stats,
    split_train_test,
    rebin_time_bins,
    parse_extra_stopwords,
)
from src.baselines import frequency_top_terms, tfidf_top_terms
from src.lda_model import train_lda, get_topic_words, get_perplexity, get_coherence, doc_topic_matrix
from src.metrics import (
    mean_adjacent_jaccard,
    mean_adjacent_jsd,
    safe_silhouette,
    cluster_category_alignment,
    mean_adjacent_cosine_distance,
)
from src.events import (
    detect_spike_events,
    jsd_adjacent_from_topic_means,
    lexical_jaccard_drift_from_top_terms,
    label_events,
)

DEFAULT_CSV = os.path.join("data", "sample", "news_2018_h1.csv")

# Default split matches typical Jan–Jun 2018 sample: train Jan–Apr, test May–Jun
DEFAULT_TRAIN_START = "2018-01-01"
DEFAULT_TRAIN_END = "2018-05-01"
DEFAULT_TEST_START = "2018-05-01"
DEFAULT_TEST_END = "2018-07-01"


def prepare_data(args):
    csv_path = args.csv_path or DEFAULT_CSV
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        print("Run: python scripts/download_data.py")
        raise SystemExit(1)

    df = load_csv(csv_path)
    df = tokenize_df(df)
    df = add_time_bins(df, freq=args.window)

    print(f"\nLoaded {len(df)} documents from {csv_path}")
    stats = corpus_stats(df)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return df


def print_split(train_df, test_df, args):
    print("\n--- Evaluation split (same for baselines, LDA, embeddings) ---")
    print(f"  Train: [{args.train_start}, {args.train_end})  →  {len(train_df)} docs")
    print(f"  Test:  [{args.test_start}, {args.test_end})   →  {len(test_df)} docs")
    if len(train_df) == 0 or len(test_df) == 0:
        print("  WARNING: empty train or test — check dates vs your CSV date range.")
    print()


def _run_baselines_single_window(
    train_df,
    test_df,
    *,
    top_k,
    extra_stop,
    window_label: str,
):
    """One baseline pass for a given time-binning (train/test already split)."""
    print(f"\n=== BASELINES (frequency + TF-IDF) — window={window_label} ===")

    for name, part in (("train", train_df), ("test", test_df)):
        if len(part) == 0:
            print(f"\n{name}: (no documents in this period)")
            continue
        freq = frequency_top_terms(part, k=top_k, extra_stop=extra_stop)
        tfidf = tfidf_top_terms(part, k=top_k, extra_stop=extra_stop)
        bins = sorted(freq.keys())
        if len(bins) < 2:
            print(f"\n{name}: only one time bin — Jaccard N/A")
        else:
            freq_terms = {b: [w for w, _ in freq[b]] for b in bins}
            j = mean_adjacent_jaccard(bins, freq_terms)
            print(f"\n{name}: adjacent-window Jaccard (freq top terms): {j:.3f}")

    if len(test_df) > 0:
        freq = frequency_top_terms(test_df, k=top_k, extra_stop=extra_stop)
        tfidf = tfidf_top_terms(test_df, k=top_k, extra_stop=extra_stop)
        bins = sorted(freq.keys())
        for b in bins[:2]:
            print(f"\n  Test window {b}:")
            print("    Top freq:", [w for w, _ in freq[b][:5]])
            print("    Top tf-idf:", [w for w, _ in tfidf.get(b, [])[:5]])


def run_baselines(
    train_df,
    test_df,
    top_k=15,
    extra_stop=None,
    compare_windows=False,
    primary_window="7D",
):
    """Optional extra stopwords for counts; optional 7D vs 14D rebin for baselines only."""
    if compare_windows:
        for freq, title in (("7D", "7D"), ("14D", "14D")):
            tr = rebin_time_bins(train_df, freq)
            te = rebin_time_bins(test_df, freq)
            _run_baselines_single_window(
                tr,
                te,
                top_k=top_k,
                extra_stop=extra_stop,
                window_label=title,
            )
    else:
        _run_baselines_single_window(
            train_df,
            test_df,
            top_k=top_k,
            extra_stop=extra_stop,
            window_label=primary_window,
        )


def run_lda(train_df, test_df, num_topics=6, passes=10, seed=42):
    print("\n=== LDA TOPIC MODEL ===")

    if len(train_df) == 0:
        print("No training documents — skip LDA.")
        return

    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)
    train_texts = train_df["tokens"].tolist()
    test_texts = test_df["tokens"].tolist()

    model, dictionary, train_corpus = train_lda(
        train_texts, num_topics=num_topics, passes=passes, seed=seed
    )

    perp_train = get_perplexity(model, train_corpus)
    test_corpus = [dictionary.doc2bow(t, allow_update=False) for t in test_texts]
    perp_test = get_perplexity(model, test_corpus) if test_corpus else float("nan")

    try:
        coh = get_coherence(model, train_texts, dictionary)
    except Exception:
        coh = float("nan")

    print(f"Perplexity train (log, gensim): {perp_train:.3f}")
    print(f"Perplexity test  (log, gensim): {perp_test:.3f}")
    print(f"Coherence (u_mass, on train): {coh:.3f}")

    print("\nSample topics (from model trained on train set only):")
    for t in get_topic_words(model)[:3]:
        print(" ", t["words"])

    if len(test_df) > 0 and test_corpus:
        theta_test = doc_topic_matrix(model, test_corpus)
        bins = sorted(test_df["time_bin"].unique())
        if len(bins) >= 2:
            bin_means = []
            for b in bins:
                mask = test_df["time_bin"] == b
                bin_means.append(theta_test[mask.values].mean(axis=0))
            jsd = mean_adjacent_jsd(bins, bin_means)
            print(f"\nTopic mix drift on TEST (mean adjacent JSD): {jsd:.3f}")
        else:
            print("\nTest period has <2 time bins — JSD on test N/A.")
    else:
        print("\nNo test documents — skip test drift.")


def run_embeddings(
    train_df,
    test_df,
    k_min=2,
    k_max=6,
    seed=42,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    print("\n=== EMBEDDINGS (DL: pretrained encoder + K-Means) ===")

    try:
        from sklearn.preprocessing import LabelEncoder

        from src.embeddings import (
            encode_texts,
            find_best_k,
            cluster_time_split,
            mean_embedding_per_bin,
        )
    except ImportError:
        print("sentence-transformers not installed — skip embeddings.")
        return None

    if len(train_df) == 0:
        print("No training documents — skip embeddings.")
        return None

    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)
    texts_train = train_df["text"].tolist()
    texts_test = test_df["text"].tolist()

    print(f"Encoder: {model_name} (frozen weights; no fine-tuning)")
    print("Encoding train...")
    X_train, enc_model = encode_texts(texts_train, model_name=model_name)
    print("Encoding test...")
    X_test, _ = (
        encode_texts(texts_test, model_name=model_name, model=enc_model)
        if texts_test
        else (None, enc_model)
    )

    upper = min(k_max, len(X_train) - 1)
    if upper < k_min:
        print("Not enough train rows for k search.")
        return None

    best_k, scores = find_best_k(X_train, k_min, upper, seed=seed)
    print(f"Best k (silhouette on TRAIN only): {best_k}")
    print(f"  Silhouette by k: {scores}")

    if X_test is None or len(X_test) == 0:
        print("No test embeddings — fit clusters on train only for demo.")
        from src.embeddings import cluster

        labels_tr, _ = cluster(X_train, best_k, seed=seed)
        sil_tr = safe_silhouette(X_train, labels_tr)
        print(f"Silhouette train: {sil_tr:.4f}")
        if "category" in train_df.columns:
            le = LabelEncoder()
            y_tr = le.fit_transform(train_df["category"].astype(str))
            nmi_tr, ari_tr = cluster_category_alignment(y_tr, labels_tr)
            print(f"Cluster vs category (train): NMI={nmi_tr:.4f}, ARI={ari_tr:.4f}")
        return best_k

    labels, km = cluster_time_split(X_train, X_test, best_k, seed=seed)
    n_tr = len(X_train)
    sil_train = safe_silhouette(X_train, labels[:n_tr])
    sil_test = safe_silhouette(X_test, labels[n_tr:])
    print(f"Silhouette train: {sil_train:.4f}")
    print(f"Silhouette test:  {sil_test:.4f}")
    print(f"Total labeled docs: {len(labels)}")

    # Weak supervision: compare clusters to HuffPost editor categories (not ground truth topics).
    if "category" in train_df.columns and "category" in test_df.columns:
        le = LabelEncoder()
        le.fit(
            pd.concat([train_df["category"], test_df["category"]], ignore_index=True).astype(str)
        )
        y_tr = le.transform(train_df["category"].astype(str))
        y_te = le.transform(test_df["category"].astype(str))
        nmi_tr, ari_tr = cluster_category_alignment(y_tr, labels[:n_tr])
        nmi_te, ari_te = cluster_category_alignment(y_te, labels[n_tr:])
        print(f"Cluster vs category (train): NMI={nmi_tr:.4f}, ARI={ari_tr:.4f}")
        print(f"Cluster vs category (test):  NMI={nmi_te:.4f}, ARI={ari_te:.4f}")

    # Temporal semantic drift on TEST: mean adjacent cosine distance between weekly bin centroids.
    if len(test_df) >= 2 and test_df["time_bin"].nunique() >= 2:
        _, cents = mean_embedding_per_bin(test_df, X_test)
        sem_drift = mean_adjacent_cosine_distance(cents)
        print(f"Semantic drift on TEST (mean adjacent bin-centroid cosine distance): {sem_drift:.4f}")

    return best_k


def run_events(
    train_df,
    test_df,
    *,
    num_topics=6,
    passes=10,
    seed=42,
    extra_stop=None,
    event_percentile=90.0,
    event_top_k_terms=15,
    lda_driver_topics=3,
):
    """
    Run a simple spike-based event detector on the TEST period.

    Returns a structured event list so it can be reused downstream for
    labeling/evaluation/plotting.

    Output format:
      {
        "lexical": [{"start_bin": ..., "end_bin": ..., "score": ..., "drivers": {...}}, ...],
        "lda": [{"start_bin": ..., "end_bin": ..., "score": ..., "drivers": {...}}, ...],
      }
    """
    events_out = {"lexical": [], "lda": []}

    print("\n=== EVENT DETECTION (simple spike baseline) ===")
    if len(test_df) == 0:
        print("No test documents — skip events.")
        return events_out

    # ---- 1) Lexical events from top-term Jaccard drift ----
    print(f"\n[Lexical] Using top-{event_top_k_terms} freq terms + adjacent Jaccard drift")
    freq_terms_by_bin = frequency_top_terms(test_df, k=event_top_k_terms, extra_stop=extra_stop)
    drift_scores, transitions = lexical_jaccard_drift_from_top_terms(
        freq_terms_by_bin, top_k=event_top_k_terms
    )
    if not drift_scores:
        print("Not enough test bins for lexical events.")
    else:
        events = detect_spike_events(drift_scores, transitions, percentile=event_percentile)
        print(f"  Drift threshold: {event_percentile}th percentile; events: {len(events)}")
        for e in events:
            b0, b1 = transitions[e.spike_transition_index]
            t0 = freq_terms_by_bin.get(b0, [])[:event_top_k_terms]
            t1 = freq_terms_by_bin.get(b1, [])[:event_top_k_terms]
            prev_set = {w for w, _ in t0}
            next_set = {w for w, _ in t1}
            new_terms = [w for w, _ in t1 if w not in prev_set][:8]
            dropped_terms = [w for w, _ in t0 if w not in next_set][:8]

            events_out["lexical"].append(
                {
                    "event_id": e.event_id,
                    "start_bin": b0,
                    "end_bin": b1,
                    "score": float(e.max_score),
                    "drivers": {
                        "new_terms": new_terms,
                        "dropped_terms": dropped_terms,
                    },
                }
            )

            print(
                f"  Event {e.event_id}: {b0} -> {b1} | max_drift={e.max_score:.3f} | "
                f"new={new_terms} | dropped={dropped_terms}"
            )

    # ---- 2) LDA events from topic-mixture JSD drift ----
    if len(train_df) == 0:
        print("\n[LDA] No training documents — skip LDA events.")
        return events_out

    print(f"\n[LDA] Training LDA on train, detecting spikes in adjacent topic-mixture JSD")
    train_texts = train_df.sort_values("timestamp")["tokens"].tolist()
    test_texts = test_df.sort_values("timestamp")["tokens"].tolist()

    model, dictionary, train_corpus = train_lda(
        train_texts, num_topics=num_topics, passes=passes, seed=seed
    )

    test_corpus = [dictionary.doc2bow(t, allow_update=False) for t in test_texts]
    if not test_corpus:
        print("No test corpus for LDA — skip LDA events.")
        return events_out

    theta_test = doc_topic_matrix(model, test_corpus)
    bins = sorted(test_df.sort_values("timestamp")["time_bin"].unique())
    if len(bins) < 2:
        print("Not enough test bins for LDA events.")
        return events_out

    # Mean topic mixture per time bin.
    bin_means: list[np.ndarray] = []
    test_df_sorted = test_df.sort_values("timestamp").reset_index(drop=True)
    for b in bins:
        mask = test_df_sorted["time_bin"] == b
        bin_means.append(theta_test[mask.values].mean(axis=0))

    dists = jsd_adjacent_from_topic_means(bin_means)
    transitions_lda = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    events_lda = detect_spike_events(dists, transitions_lda, percentile=event_percentile)
    print(f"  Drift threshold: {event_percentile}th percentile; events: {len(events_lda)}")
    for e in events_lda:
        prev_vec = bin_means[e.spike_transition_index]
        next_vec = bin_means[e.spike_transition_index + 1]
        deltas = np.abs(next_vec - prev_vec)
        top_topic_ids = list(np.argsort(-deltas)[:lda_driver_topics])
        top_topics = []
        for tid in top_topic_ids:
            words = [w for w, _ in model.show_topic(tid, topn=5)]
            top_topics.append({"topic_id": int(tid), "words": words})
        b0, b1 = transitions_lda[e.spike_transition_index]

        events_out["lda"].append(
            {
                "event_id": e.event_id,
                "start_bin": b0,
                "end_bin": b1,
                "score": float(e.max_score),
                "drivers": {
                    "top_topics": top_topics,
                },
            }
        )

        print(
            f"  Event {e.event_id}: {b0} -> {b1} | max_drift={e.max_score:.3f} | "
            f"top_topics={top_topics}"
        )

    return events_out


def main():
    parser = argparse.ArgumentParser(description="Trend detector — Phase 1")
    parser.add_argument(
        "--mode",
        choices=["baselines", "lda", "embeddings", "events", "all"],
        default="all",
    )
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--window", type=str, default="7D")
    parser.add_argument("--num-topics", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-start",
        type=str,
        default=DEFAULT_TRAIN_START,
        help="Train interval start (inclusive), ISO date",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=DEFAULT_TRAIN_END,
        help="Train interval end (exclusive)",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=DEFAULT_TEST_START,
        help="Test interval start (inclusive)",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=DEFAULT_TEST_END,
        help="Test interval end (exclusive)",
    )
    parser.add_argument(
        "--extra-stopwords",
        type=str,
        default="none",
        help="Baselines only: 'none' | 'default' (headline boilerplate) | comma-separated words",
    )
    parser.add_argument(
        "--compare-baseline-windows",
        action="store_true",
        help="Run baselines for 7D and 14D (weekly vs biweekly); LDA/embeddings still use --window",
    )
    parser.add_argument(
        "--event-percentile",
        type=float,
        default=90.0,
        help="Spike detector threshold as percentile of adjacent drift scores",
    )
    parser.add_argument(
        "--event-top-k-terms",
        type=int,
        default=15,
        help="Top terms per bin for lexical event detection (frequency-based)",
    )
    parser.add_argument(
        "--lda-passes",
        type=int,
        default=10,
        help="LDA passes for event detection LDA model",
    )
    parser.add_argument(
        "--lda-driver-topics",
        type=int,
        default=3,
        help="How many topics to list as drivers per LDA event",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer checkpoint for embeddings + clustering (frozen encoder)",
    )
    args = parser.parse_args()

    extra_stop = parse_extra_stopwords(args.extra_stopwords)

    df = prepare_data(args)
    train_df, test_df = split_train_test(
        df,
        args.train_start,
        args.train_end,
        args.test_start,
        args.test_end,
    )
    print_split(train_df, test_df, args)

    if args.mode in ("baselines", "all"):
        run_baselines(
            train_df,
            test_df,
            extra_stop=extra_stop,
            compare_windows=args.compare_baseline_windows,
            primary_window=args.window,
        )
    if args.mode in ("lda", "all"):
        run_lda(train_df, test_df, num_topics=args.num_topics, seed=args.seed)
    if args.mode in ("embeddings", "all"):
        run_embeddings(train_df, test_df, seed=args.seed, model_name=args.embedding_model)
    if args.mode in ("events",):
        events_out = run_events(
            train_df,
            test_df,
            num_topics=args.num_topics,
            passes=args.lda_passes,
            seed=args.seed,
            extra_stop=extra_stop,
            event_percentile=args.event_percentile,
            event_top_k_terms=args.event_top_k_terms,
            lda_driver_topics=args.lda_driver_topics,
        )

        labeled = label_events(events_out, test_df)
        print("\n=== Event labels (simple, explainable) ===")
        for ev in labeled.get("lexical", []):
            print(
                f"  [Lexical] {ev['start_bin']} -> {ev['end_bin']} | "
                f"score={ev['score']:.3f} | label={ev['label']} | category={ev['dominant_category']}"
            )
        for ev in labeled.get("lda", []):
            print(
                f"  [LDA] {ev['start_bin']} -> {ev['end_bin']} | "
                f"score={ev['score']:.3f} | label={ev['label']} | category={ev['dominant_category']}"
            )


if __name__ == "__main__":
    main()
