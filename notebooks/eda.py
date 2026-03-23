
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import load_csv, tokenize_df, add_time_bins, corpus_stats
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = "data/sample/news_2018_h1.csv"

if not os.path.exists(DATA_PATH):
    print(f"Dataset not found at {DATA_PATH}")
    print("Run: python scripts/download_data.py")
else:
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

    # %% [markdown]
    # ### Observations
    # - Dataset has 40+ news categories with POLITICS and WELLNESS dominating
    # - Category distribution shifts across weeks — good for trend detection
    # - Headlines + descriptions average ~15-25 tokens after preprocessing
    # - No heavy class imbalance that would break time-window analysis
