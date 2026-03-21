import pandas as pd

from src.baselines.frequency_topics import top_terms_per_time_bin
from src.baselines.temporal_windows import add_time_bin, aggregate_window_texts
from src.baselines.tfidf_topics import tfidf_top_terms_per_window
from src.data.preprocess import add_tokens_column
from src.evaluation.temporal_baseline_metrics import jaccard_similarity, mean_adjacent_jaccard


def test_jaccard():
    assert jaccard_similarity(["a", "b"], ["b", "c"]) == 1 / 3
    assert jaccard_similarity([], []) == 1.0


def test_mean_adjacent_requires_two_bins():
    import math

    assert math.isnan(mean_adjacent_jaccard(["only"], {"only": ["a"]}))
    x = mean_adjacent_jaccard(["a", "b"], {"a": ["x"], "b": ["x"]})
    assert x == 1.0


def test_frequency_and_tfidf_pipeline_smoke():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"], utc=True
            ),
            "text": [
                "vote ballot campaign poll senate",
                "debate turnout district primary fraud",
                "climate carbon emissions renewable warming",
                "flood drought methane energy cop",
            ],
        }
    )
    df = add_tokens_column(df)
    df = add_time_bin(df, time_col="timestamp", freq="7D")
    freq = top_terms_per_time_bin(df, k=5)
    assert len(freq) >= 1
    wtoks = aggregate_window_texts(df)
    wtext = {k: " ".join(v) for k, v in wtoks.items()}
    tfidf = tfidf_top_terms_per_window(wtext, top_k=5)
    assert len(tfidf) == len(wtoks)
