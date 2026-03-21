from pathlib import Path

import pandas as pd

from src.part2_runner import run_part2


def test_run_part2_synthetic():
    r = run_part2(
        source="synthetic",
        n_docs=250,
        seed=2,
        passes=5,
        num_topics=6,
        window_freq="7D",
    )
    assert "lda_model" in r
    assert len(r["topics"]) == 6
    assert not r["topic_mix_by_window"].empty
    sm = r["summary_metrics"]
    assert set(sm["metric"].astype(str)) >= {
        "log_perplexity_train_bow",
        "log_perplexity_test_bow",
    }


def test_run_part2_sample_csv(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "data" / "sample" / "stream_sample.csv"
    r = run_part2(
        source="csv",
        csv_path=csv_path,
        passes=4,
        num_topics=4,
        window_freq="7D",
        no_below=1,
        no_above=1.0,
        output_dir=tmp_path,
    )
    assert (tmp_path / "part2_summary_metrics.csv").is_file()
    assert isinstance(r["summary_metrics"], pd.DataFrame)
