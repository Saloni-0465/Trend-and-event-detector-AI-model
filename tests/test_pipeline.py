"""Basic tests for the trend detector pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data_utils import (
    tokenize,
    tokenize_df,
    add_time_bins,
    corpus_stats,
    load_csv,
    split_train_test,
    rebin_time_bins,
    parse_extra_stopwords,
)
from src.baselines import frequency_top_terms, tfidf_top_terms
from src.lda_model import train_lda, get_topic_words, get_perplexity, doc_topic_matrix
from src.metrics import jaccard, mean_adjacent_jaccard, mean_adjacent_jsd, safe_silhouette


def _make_df(n=60):
    """Create a small test dataframe that mimics real news data."""
    rng = np.random.default_rng(0)
    categories = ["POLITICS", "TECH", "SPORTS", "HEALTH"]
    word_pools = {
        "POLITICS": ["election", "vote", "senate", "campaign", "poll", "government"],
        "TECH": ["software", "chip", "data", "cloud", "platform", "startup"],
        "SPORTS": ["game", "player", "team", "score", "season", "league"],
        "HEALTH": ["vaccine", "hospital", "cases", "treatment", "study", "patients"],
    }
    rows = []
    base = pd.Timestamp("2018-01-01", tz="UTC")
    for i in range(n):
        cat = categories[i % len(categories)]
        words = list(rng.choice(word_pools[cat], size=10))
        rows.append({
            "timestamp": base + pd.Timedelta(days=i),
            "text": " ".join(words),
            "category": cat,
        })
    return pd.DataFrame(rows)


# --- data_utils ---

def test_tokenize():
    tokens = tokenize("The quick brown FOX jumps over a lazy dog")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "the" not in tokens


def test_tokenize_df():
    df = _make_df(10)
    df = tokenize_df(df)
    assert "tokens" in df.columns
    assert all(len(t) > 0 for t in df["tokens"])


def test_add_time_bins():
    df = _make_df(30)
    df = add_time_bins(df, freq="14D")
    assert "time_bin" in df.columns
    assert df["time_bin"].nunique() >= 2


def test_corpus_stats():
    df = _make_df(20)
    df = tokenize_df(df)
    st = corpus_stats(df)
    assert st["n_docs"] == 20
    assert st["vocab_size"] > 0


def test_split_train_test():
    rows = []
    base = pd.Timestamp("2018-01-01", tz="UTC")
    for i in range(180):
        rows.append(
            {
                "timestamp": base + pd.Timedelta(days=i),
                "text": f"word{i % 5} election vote news",
                "category": "POLITICS",
            }
        )
    df = pd.DataFrame(rows)
    df = tokenize_df(df)
    df = add_time_bins(df, freq="14D")
    tr, te = split_train_test(
        df, "2018-01-01", "2018-05-01", "2018-05-01", "2018-07-01"
    )
    assert len(tr) > 0 and len(te) > 0
    assert tr["timestamp"].max() < pd.Timestamp("2018-05-01", tz="UTC")
    assert te["timestamp"].min() >= pd.Timestamp("2018-05-01", tz="UTC")


def test_load_csv(tmp_path):
    csv_path = tmp_path / "test.csv"
    df = _make_df(15)
    df.to_csv(csv_path, index=False)
    loaded = load_csv(str(csv_path))
    assert len(loaded) == 15
    assert "text" in loaded.columns


# --- baselines ---

def test_frequency_top_terms():
    df = _make_df(60)
    df = tokenize_df(df)
    df = add_time_bins(df, freq="14D")
    result = frequency_top_terms(df, k=5)
    assert len(result) > 0
    for _, terms in result.items():
        assert len(terms) <= 5


def test_tfidf_top_terms():
    df = _make_df(60)
    df = tokenize_df(df)
    df = add_time_bins(df, freq="14D")
    result = tfidf_top_terms(df, k=5)
    assert len(result) > 0


def test_frequency_top_terms_extra_stop():
    rows = []
    base = pd.Timestamp("2018-01-01", tz="UTC")
    for i in range(40):
        rows.append(
            {
                "timestamp": base + pd.Timedelta(days=i),
                "text": "said said election vote senate" if i % 2 == 0 else "election vote campaign poll",
                "category": "POLITICS",
            }
        )
    df = pd.DataFrame(rows)
    df = tokenize_df(df)
    df = add_time_bins(df, freq="7D")
    raw = frequency_top_terms(df, k=5)
    first_bin = sorted(raw.keys())[0]
    assert "said" in [w for w, _ in raw[first_bin]]

    extra = frozenset({"said"})
    filtered = frequency_top_terms(df, k=5, extra_stop=extra)
    assert "said" not in [w for w, _ in filtered[first_bin]]


def test_rebin_time_bins_has_fewer_or_equal_bins():
    df = _make_df(120)
    df = tokenize_df(df)
    df = add_time_bins(df, freq="7D")
    n7 = df["time_bin"].nunique()
    df14 = rebin_time_bins(df, "14D")
    assert df14["time_bin"].nunique() <= n7


def test_parse_extra_stopwords():
    assert parse_extra_stopwords("none") == frozenset()
    assert "said" in parse_extra_stopwords("default")
    assert parse_extra_stopwords("foo,bar") == frozenset({"foo", "bar"})


# --- lda ---

def test_lda_basic():
    df = _make_df(80)
    df = tokenize_df(df)
    texts = df["tokens"].tolist()
    model, dictionary, corpus = train_lda(texts, num_topics=3, passes=3)
    topics = get_topic_words(model, num_words=5)
    assert len(topics) == 3

    perp = get_perplexity(model, corpus)
    assert np.isfinite(perp)

    theta = doc_topic_matrix(model, corpus)
    assert theta.shape == (80, 3)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-3)


# --- metrics ---

def test_jaccard():
    assert jaccard(["a", "b", "c"], ["b", "c", "d"]) == pytest.approx(0.5)
    assert jaccard([], []) == 1.0


def test_mean_adjacent_jsd():
    bins = [0, 1]
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert mean_adjacent_jsd(bins, [p, q]) == pytest.approx(0.0, abs=1e-6)


def test_safe_silhouette():
    X = np.random.randn(30, 4)
    y = np.array([0] * 15 + [1] * 15)
    s = safe_silhouette(X, y)
    assert -1 <= s <= 1
    assert np.isnan(safe_silhouette(X, np.zeros(30)))


# --- embeddings (no model download, just clustering) ---

def test_find_best_k_and_cluster():
    from src.embeddings import find_best_k, cluster, cluster_time_split
    X = np.random.randn(50, 16).astype(np.float32)
    best_k, scores = find_best_k(X, k_min=2, k_max=5)
    assert 2 <= best_k <= 5

    labels, km = cluster(X, best_k)
    assert len(labels) == 50
    assert len(np.unique(labels)) == best_k

    X_tr, X_te = X[:30], X[30:]
    lab, km2 = cluster_time_split(X_tr, X_te, best_k)
    assert len(lab) == 50
    assert len(np.unique(lab[:30])) <= best_k
