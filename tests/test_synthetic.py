import pandas as pd

from src.data.synthetic import generate_synthetic_social_stream


def test_synthetic_shape_and_monotonic_time():
    df = generate_synthetic_social_stream(n_docs=100, seed=0)
    assert list(df.columns) == ["timestamp", "text", "theme"]
    assert len(df) == 100
    assert df["timestamp"].is_monotonic_increasing
    assert df["theme"].isin(["climate", "election", "tech", "health"]).all()


def test_synthetic_reproducible():
    a = generate_synthetic_social_stream(n_docs=20, seed=123)
    b = generate_synthetic_social_stream(n_docs=20, seed=123)
    assert a["text"].tolist() == b["text"].tolist()
