"""
Exploratory data analysis (plots + printouts for the report).

This is a normal Python script.
Run from the repository root:

    python scripts/eda.py

Requires: matplotlib, pandas, and the sample CSV from download_data.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import load_csv, tokenize_df, add_time_bins, corpus_stats
import matplotlib.pyplot as plt

DATA_PATH = "data/sample/news_2018_h1.csv"


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}")
        print("Run: python scripts/download_data.py")
        return

    df = load_csv(DATA_PATH)
    df = tokenize_df(df)
    df = add_time_bins(df, freq="7D")

    print(corpus_stats(df))
    if "category" in df.columns:
        print(f"\nTop categories:\n{df['category'].value_counts().head(10)}")

    if "category" in df.columns:
        top_cats = df["category"].value_counts().head(6).index
        subset = df[df["category"].isin(top_cats)]
        ct = subset.groupby(["time_bin", "category"]).size().unstack(fill_value=0)
        ct.plot.bar(stacked=True, figsize=(12, 5))
        plt.title("Top 6 categories over time")
        plt.ylabel("# articles")
        plt.tight_layout()
        os.makedirs("docs", exist_ok=True)
        plt.savefig("docs/categories_over_time.png", dpi=120)
        plt.show()

    df["doc_len"] = df["tokens"].apply(len)
    df["doc_len"].hist(bins=30, figsize=(7, 3))
    plt.title("Document length (tokens after preprocessing)")
    plt.xlabel("# tokens")
    plt.tight_layout()
    plt.savefig("docs/doc_lengths.png", dpi=120)
    plt.show()

    print(f"Mean doc length: {df['doc_len'].mean():.1f}")
    print(f"Median doc length: {df['doc_len'].median():.1f}")

    # Observations (for report / viva):
    # - Many news categories; a few dominate (e.g. politics, wellness).
    # - Category mix shifts across weeks — useful for trend-style analysis.
    # - Short texts after preprocessing — fits LDA and embeddings without heavy stemming.


if __name__ == "__main__":
    main()
