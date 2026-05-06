"""
Download and prepare the News Category Dataset from Kaggle.

Prerequisites:
    pip install kaggle
    Place your kaggle.json in ~/.kaggle/

Usage:
    python scripts/download_data.py
"""

import os
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")


def download():
    """Download dataset using Kaggle API."""
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Downloading News Category Dataset from Kaggle...")
    os.system(f"kaggle datasets download -d rmisra/news-category-dataset -p {RAW_DIR} --unzip")
    print("Download complete.")


def prepare():
    """Convert the raw JSON to a clean CSV for the pipeline."""
    raw_path = os.path.join(RAW_DIR, "News_Category_Dataset_v3.json")
    if not os.path.exists(raw_path):
        print(f"Raw file not found at {raw_path}")
        print("Run: kaggle datasets download -d rmisra/news-category-dataset -p data/raw --unzip")
        return

    print("Reading raw JSON...")
    records = []
    with open(raw_path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)
    df = df.rename(columns={"headline": "text", "date": "timestamp", "category": "category"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # combine headline + short description for richer text
    df["text"] = df["text"].str.strip() + ". " + df["short_description"].str.strip()
    df = df[["timestamp", "text", "category"]].dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Categories: {df['category'].nunique()}")

    # save full processed CSV
    out_path = os.path.join(DATA_DIR, "news_full.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved full dataset to {out_path}")

    # save a 6-month subset for quicker experiments
    subset = df[(df["timestamp"] >= "2018-01-01") & (df["timestamp"] < "2018-07-01")]
    subset_path = os.path.join(SAMPLE_DIR, "news_2018_h1.csv")
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    subset.to_csv(subset_path, index=False)
    print(f"Saved 6-month subset ({len(subset)} docs) to {subset_path}")


if __name__ == "__main__":
    download()
    prepare()
