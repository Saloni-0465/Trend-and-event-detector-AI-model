"""
Reproducible synthetic social-style text stream.

Themes mix over time so simple frequency/TF-IDF baselines show *temporal dynamics*
without API keys. For the report: this is a controlled sandbox; real streams add
noise, bots, and sampling bias (document limitations there).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# Compact vocabularies — “pseudo–entities” and domain words per narrative cluster.
_THEME_WORDS: dict[str, list[str]] = {
    "climate": [
        "climate",
        "carbon",
        "emissions",
        "renewable",
        "cop",
        "warming",
        "flood",
        "drought",
        "methane",
        "energy",
    ],
    "election": [
        "vote",
        "ballot",
        "campaign",
        "poll",
        "senate",
        "primary",
        "debate",
        "turnout",
        "district",
        "fraud",
    ],
    "tech": [
        "chip",
        "gpu",
        "model",
        "inference",
        "latency",
        "open",
        "source",
        "benchmark",
        "dataset",
        "scaling",
    ],
    "health": [
        "vaccine",
        "variant",
        "hospital",
        "cdc",
        "outbreak",
        "cases",
        "immunity",
        "trial",
        "dose",
        "surge",
    ],
}

_FILLER = [
    "today",
    "people",
    "report",
    "breaking",
    "thread",
    "update",
    "source",
    "claims",
    "video",
    "live",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def generate_synthetic_social_stream(
    n_docs: int = 800,
    *,
    seed: int = 42,
    start: str = "2024-01-01",
    end: str = "2024-03-01",
    min_words: int = 8,
    max_words: int = 22,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns: timestamp, text, theme (latent label for analysis).

    Theme probabilities vary with normalized time so different windows emphasize
    different clusters — useful for visualizing drift under baselines.
    """
    if n_docs < 1:
        raise ValueError("n_docs must be >= 1")

    rng = np.random.default_rng(seed)
    themes = list(_THEME_WORDS.keys())
    t_start = pd.Timestamp(start, tz="UTC")
    t_end = pd.Timestamp(end, tz="UTC")
    span_s = (t_end - t_start).total_seconds()
    if span_s <= 0:
        raise ValueError("end must be after start")

    rows: list[dict[str, Any]] = []
    for i in range(n_docs):
        # Time advances with document index so theme statistics are non-stationary
        # over the window (small jitter within each slot).
        frac = (i + float(rng.random())) / n_docs
        u = i / max(n_docs - 1, 1)
        ts = t_start + pd.Timedelta(seconds=frac * span_s)

        # Two slow drifts + one faster oscillation → non-stationary theme mix.
        logits = np.array(
            [
                1.6 * math.sin(2 * math.pi * u),
                1.4 * math.cos(2 * math.pi * u),
                0.9 * math.sin(4 * math.pi * u),
                0.35 * rng.standard_normal(),
            ],
            dtype=float,
        )
        p = _softmax(logits)
        z = rng.choice(len(themes), p=p)
        theme = themes[int(z)]

        n_w = int(rng.integers(min_words, max_words + 1))
        words: list[str] = []
        pool_theme = _THEME_WORDS[theme]
        for _ in range(n_w):
            if rng.random() < 0.72:
                words.append(rng.choice(pool_theme))
            else:
                words.append(rng.choice(_FILLER))

        rng.shuffle(words)
        text = " ".join(words)

        rows.append({"timestamp": ts, "text": text, "theme": theme})

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df
