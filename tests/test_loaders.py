from pathlib import Path

from src.data.loaders import load_stream_csv


def test_load_sample_csv():
    root = Path(__file__).resolve().parents[1]
    p = root / "data" / "sample" / "stream_sample.csv"
    df = load_stream_csv(p)
    assert "text" in df.columns and "timestamp" in df.columns
    assert len(df) > 0
    assert df["timestamp"].notna().all()
