"""Data loading, preprocessing, and time binning."""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = frozenset(w.lower() for w in ENGLISH_STOP_WORDS)


def load_csv(path, text_col="text", time_col="timestamp"):
    """Load a CSV with text and timestamp columns."""
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df.dropna(subset=[time_col])
    df = df[df[text_col].str.len() > 0]
    return df.reset_index(drop=True)


# --- Preprocessing ---

TOKEN_RE = re.compile(r"[a-z][a-z0-9']*", re.IGNORECASE)


def tokenize(text):
    """Lowercase, extract tokens, remove stopwords and short words."""
    tokens = []
    for m in TOKEN_RE.finditer(str(text).lower()):
        t = m.group(0).strip("'")
        if len(t) >= 2 and t not in STOP:
            tokens.append(t)
    return tokens


def tokenize_df(df, text_col="text"):
    """Add a 'tokens' column to the dataframe."""
    df = df.copy()
    df["tokens"] = df[text_col].apply(tokenize)
    return df


def add_time_bins(df, time_col="timestamp", freq="7D"):
    """Bin timestamps into fixed intervals."""
    df = df.copy()
    df["time_bin"] = pd.to_datetime(df[time_col], utc=True).dt.floor(freq)
    return df


def corpus_stats(df):
    """Basic stats about the tokenized corpus."""
    lengths = df["tokens"].apply(len)
    all_tokens = [t for toks in df["tokens"] for t in toks]
    return {
        "n_docs": len(df),
        "vocab_size": len(set(all_tokens)),
        "total_tokens": len(all_tokens),
        "mean_length": round(float(np.mean(lengths)), 1),
    }
