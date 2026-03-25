"""
Main script to run the pipeline (Phase 1).

Uses one fixed chronological train/test split for all modes (fair comparison).
Default: train 2018-01-01 .. 2018-05-01 (exclusive), test 2018-05-01 .. 2018-07-01 (exclusive).
"""

import os
import argparse

import pandas as pd

from src.data_utils import (
    load_csv,
    tokenize_df,
    add_time_bins,
    corpus_stats,
    split_train_test,
)
from src.baselines import frequency_top_terms, tfidf_top_terms
from src.lda_model import train_lda, get_topic_words, get_perplexity, get_coherence, doc_topic_matrix
from src.metrics import mean_adjacent_jaccard, mean_adjacent_jsd, safe_silhouette

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


def run_baselines(train_df, test_df, top_k=15):
    print("\n=== BASELINES (frequency + TF-IDF) ===")

    for name, part in (("train", train_df), ("test", test_df)):
        if len(part) == 0:
            print(f"\n{name}: (no documents in this period)")
            continue
        freq = frequency_top_terms(part, k=top_k)
        tfidf = tfidf_top_terms(part, k=top_k)
        bins = sorted(freq.keys())
        if len(bins) < 2:
            print(f"\n{name}: only one time bin — Jaccard N/A")
            continue
        freq_terms = {b: [w for w, _ in freq[b]] for b in bins}
        j = mean_adjacent_jaccard(bins, freq_terms)
        print(f"\n{name}: adjacent-window Jaccard (freq top terms): {j:.3f}")

    if len(test_df) > 0:
        freq = frequency_top_terms(test_df, k=top_k)
        tfidf = tfidf_top_terms(test_df, k=top_k)
        bins = sorted(freq.keys())
        for b in bins[:2]:
            print(f"\n  Test window {b}:")
            print("    Top freq:", [w for w, _ in freq[b][:5]])
            print("    Top tf-idf:", [w for w, _ in tfidf.get(b, [])[:5]])


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


def run_embeddings(train_df, test_df, k_min=2, k_max=6, seed=42):
    print("\n=== EMBEDDINGS ===")

    try:
        from src.embeddings import encode_texts, find_best_k, cluster_time_split
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

    print("Encoding train...")
    X_train = encode_texts(texts_train)
    print("Encoding test...")
    X_test = encode_texts(texts_test) if texts_test else None

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
        return best_k

    labels, km = cluster_time_split(X_train, X_test, best_k, seed=seed)
    n_tr = len(X_train)
    sil_train = safe_silhouette(X_train, labels[:n_tr])
    sil_test = safe_silhouette(X_test, labels[n_tr:])
    print(f"Silhouette train: {sil_train:.4f}")
    print(f"Silhouette test:  {sil_test:.4f}")
    print(f"Total labeled docs: {len(labels)}")
    return best_k


def main():
    parser = argparse.ArgumentParser(description="Trend detector — Phase 1")
    parser.add_argument(
        "--mode",
        choices=["baselines", "lda", "embeddings", "all"],
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
    args = parser.parse_args()

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
        run_baselines(train_df, test_df)
    if args.mode in ("lda", "all"):
        run_lda(train_df, test_df, num_topics=args.num_topics, seed=args.seed)
    if args.mode in ("embeddings", "all"):
        run_embeddings(train_df, test_df, seed=args.seed)


if __name__ == "__main__":
    main()
