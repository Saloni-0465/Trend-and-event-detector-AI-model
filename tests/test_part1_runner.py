from pathlib import Path

import pandas as pd

from src.part1_runner import run_part1


def test_run_part1_synthetic_produces_metrics():
    r = run_part1(source="synthetic", n_docs=200, seed=1, window_freq="7D", top_k=10)
    sm = r["summary_metrics"]
    assert set(sm["method"]) == {"frequency", "tfidf"}
    assert (sm["n_windows"] > 0).all()
    assert not r["long_table"].empty


def test_run_part1_csv_sample(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "sample" / "stream_sample.csv"
    r = run_part1(
        source="csv",
        csv_path=csv_path,
        window_freq="7D",
        top_k=8,
        output_dir=tmp_path,
    )
    assert (tmp_path / "part1_window_terms.csv").is_file()
    assert (tmp_path / "part1_summary_metrics.csv").is_file()
    assert isinstance(r["df"], pd.DataFrame)
