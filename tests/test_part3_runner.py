from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.part3_runner import run_part3


def _fake_embed_stream(df: pd.DataFrame, text_col: str, **kw) -> np.ndarray:
    rng = np.random.default_rng(42)
    n = len(df)
    # Two separable blobs so k=2 is natural
    base = rng.standard_normal((n, 16)).astype(np.float32)
    if "theme" in df.columns:
        codes = pd.Categorical(df["theme"].astype(str)).codes
        base[codes % 2 == 0, 0] += 4.0
        base[codes % 2 == 1, 0] -= 4.0
    return base


@pytest.fixture
def patch_embed(monkeypatch):
    monkeypatch.setattr(
        "src.part3_runner.embed_stream_texts",
        _fake_embed_stream,
    )


def test_run_part3_synthetic_with_fake_embed(patch_embed):
    r = run_part3(
        source="synthetic",
        n_docs=120,
        seed=1,
        k_min=2,
        k_max=6,
        train_frac=0.7,
    )
    assert r["embeddings"].shape[0] == 120
    assert len(r["labels"]) == 120
    assert r["best_k"] >= 2
    sm = r["summary_metrics"]
    assert set(sm["metric"].astype(str)) >= {"silhouette_train", "silhouette_test"}
