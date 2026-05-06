"""Data loading, preprocessing, and time binning."""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = frozenset(w.lower() for w in ENGLISH_STOP_WORDS)

# Headline/news boilerplate (not in sklearn list) — optional extra filter for baselines
NEWS_DOMAIN_STOPWORDS = frozenset(
    {
        "new",
        "said",
        "says",
        "say",
        "according",
        "year",
        "years",
        "day",
        "days",
        "week",
        "weeks",
        "time",
        "times",
        "just",
        "also",
        "news",
        "report",
        "reports",
        "source",
        "sources",
        "people",
        "way",
        "man",
        "men",
        "woman",
        "women",
        "many",
        "much",
        "still",
        "even",
        "may",
        "well",
        "including",
        "today",
        "yesterday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }
)


def parse_extra_stopwords(spec: str) -> frozenset:
    """
    Parse --extra-stopwords: 'none' | 'default' | comma-separated words.

    'default' uses NEWS_DOMAIN_STOPWORDS (headline boilerplate).
    """
    s = (spec or "none").strip().lower()
    if s in ("", "none"):
        return frozenset()
    if s == "default":
        return NEWS_DOMAIN_STOPWORDS
    return frozenset(w.strip() for w in s.split(",") if w.strip())


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


def rebin_time_bins(df, freq: str, time_col="timestamp"):
    """Recompute ``time_bin`` (e.g. compare weekly vs biweekly without reloading CSV)."""
    out = df.copy()
    if "time_bin" in out.columns:
        out = out.drop(columns=["time_bin"])
    return add_time_bins(out, time_col=time_col, freq=freq)


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


def split_train_test(
    df,
    train_start,
    train_end_excl,
    test_start,
    test_end_excl,
    time_col="timestamp",
):
    """
    Chronological train/test split. All end boundaries are exclusive.

    Example (default for 2018 H1 sample): train Jan–Apr, test May–Jun.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    ts = pd.to_datetime(df[time_col], utc=True)
    t_tr0 = pd.Timestamp(train_start, tz="UTC")
    t_tr1 = pd.Timestamp(train_end_excl, tz="UTC")
    t_te0 = pd.Timestamp(test_start, tz="UTC")
    t_te1 = pd.Timestamp(test_end_excl, tz="UTC")

    train_df = df[(ts >= t_tr0) & (ts < t_tr1)].copy()
    test_df = df[(ts >= t_te0) & (ts < t_te1)].copy()
    return train_df, test_df
