"""
Main script to run the pipeline (simplified for Phase 1).
"""

import os
import argparse

from src.data_utils import load_csv, tokenize_df, add_time_bins, corpus_stats
from src.baselines import frequency_top_terms, tfidf_top_terms
from src.lda_model import train_lda, get_topic_words, get_perplexity, get_coherence, doc_topic_matrix
from src.metrics import mean_adjacent_jaccard, mean_adjacent_jsd

DEFAULT_CSV = os.path.join("data", "sample", "news_2018_h1.csv")


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


def run_baselines(df, top_k=15):
    print("\n=== BASELINES (frequency + TF-IDF) ===")

    freq = frequency_top_terms(df, k=top_k)
    tfidf = tfidf_top_terms(df, k=top_k)
    bins = sorted(freq.keys())

    for b in bins[:2]:
        print(f"\nWindow {b}:")
        print("  Top freq:", [w for w, _ in freq[b][:5]])
        print("  Top tf-idf:", [w for w, _ in tfidf.get(b, [])[:5]])

    freq_terms = {b: [w for w, _ in freq[b]] for b in bins}
    j = mean_adjacent_jaccard(bins, freq_terms)
    print(f"\nWord overlap between adjacent windows (Jaccard): {j:.3f}")
    return j


def run_lda(df, num_topics=6, passes=10, seed=42):
    print("\n=== LDA TOPIC MODEL ===")

    df = df.sort_values("timestamp").reset_index(drop=True)
    texts = df["tokens"].tolist()

    model, dictionary, corpus = train_lda(
        texts, num_topics=num_topics, passes=passes, seed=seed
    )

    perp = get_perplexity(model, corpus)
    try:
        coh = get_coherence(model, texts, dictionary)
    except Exception:
        coh = float("nan")

    print(f"Perplexity (log, gensim): {perp:.3f}")
    print(f"Coherence (u_mass): {coh:.3f}")

    print("\nSample topics:")
    for t in get_topic_words(model)[:3]:
        print(" ", t["words"])

    theta = doc_topic_matrix(model, corpus)
    bins = sorted(df["time_bin"].unique())
    bin_means = []
    for b in bins:
        mask = df["time_bin"] == b
        bin_means.append(theta[mask.values].mean(axis=0))

    jsd = mean_adjacent_jsd(bins, bin_means)
    print(f"\nTopic mix change over time (mean JSD): {jsd:.3f}")
    return jsd


def run_embeddings(df, k_min=2, k_max=6, seed=42):
    print("\n=== EMBEDDINGS ===")

    try:
        from src.embeddings import encode_texts, find_best_k, cluster
    except ImportError:
        print("sentence-transformers not installed — skip embeddings.")
        return None

    texts = df["text"].tolist()
    print("Encoding (first run may download the model)...")
    X = encode_texts(texts)

    upper = min(k_max, len(X) - 1)
    if upper < k_min:
        print("Not enough rows for clustering.")
        return None

    best_k, _ = find_best_k(X, k_min, upper, seed=seed)
    labels, _ = cluster(X, best_k, seed=seed)
    print(f"Chosen k (silhouette on full set): {best_k}")
    print(f"Cluster labels assigned: {len(labels)} documents")
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
    args = parser.parse_args()

    df = prepare_data(args)

    if args.mode in ("baselines", "all"):
        run_baselines(df)
    if args.mode in ("lda", "all"):
        run_lda(df, num_topics=args.num_topics, seed=args.seed)
    if args.mode in ("embeddings", "all"):
        run_embeddings(df, seed=args.seed)


if __name__ == "__main__":
    main()
