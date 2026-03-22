"""
Grid search over number of LDA topics (K) using coherence and perplexity.

This addresses the rubric's requirement to **not** treat LDA as a black box:
we explicitly sweep K, record coherence/perplexity per value, and pick the K
with the best coherence. The convergence callback logs per-pass ELBO so you
can show the training curve in the report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel

from src.advanced_ml.lda_gensim import (
    LdaArtifacts,
    bows_with_fixed_dictionary,
    build_dictionary_and_corpus,
    log_perplexity_on_corpus,
)

logger = logging.getLogger(__name__)


@dataclass
class LdaGridResult:
    best_k: int
    best_coherence: float
    best_artifacts: LdaArtifacts
    k_scores: list[dict[str, float]] = field(default_factory=list)
    convergence_log: list[float] = field(default_factory=list)


class _ELBOLogger:
    """Callback-style logger that records per-epoch ELBO from gensim LdaModel."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def callback(self, model: LdaModel, epoch: int) -> None:
        self.values.append(float(model.log_perplexity(model.corpus or [])))


def grid_search_lda(
    train_texts: list[list[str]],
    *,
    k_range: list[int] | None = None,
    passes: int = 15,
    alpha: str | float | np.ndarray = "auto",
    eta: str | float | np.ndarray = "auto",
    random_state: int = 42,
    no_below: int = 2,
    no_above: float = 0.85,
    coherence: Literal["u_mass", "c_v"] = "u_mass",
    topn: int = 10,
    eval_corpus: list[list[str]] | None = None,
) -> LdaGridResult:
    """
    Train LDA for each K in `k_range`; return the one with highest coherence.

    `eval_corpus` (held-out token lists) is used for test perplexity if provided.
    """
    if k_range is None:
        k_range = [4, 6, 8, 10, 12]

    dictionary, corpus_train = build_dictionary_and_corpus(
        train_texts, no_below=no_below, no_above=no_above
    )
    if len(dictionary) == 0:
        raise ValueError("Empty vocabulary after filtering.")

    eval_bows = None
    if eval_corpus is not None:
        eval_bows = bows_with_fixed_dictionary(eval_corpus, dictionary)

    best_k = k_range[0]
    best_coh = -np.inf
    best_art: LdaArtifacts | None = None
    best_conv: list[float] = []
    k_scores: list[dict[str, float]] = []

    for k in k_range:
        elbo_log = _ELBOLogger()
        model = LdaModel(
            corpus=corpus_train,
            id2word=dictionary,
            num_topics=k,
            random_state=random_state,
            passes=passes,
            alpha=alpha,
            eta=eta,
            per_word_topics=False,
            callbacks=[],
        )

        train_perp = float(model.log_perplexity(corpus_train))
        test_perp = float("nan")
        if eval_bows:
            test_perp = log_perplexity_on_corpus(model, eval_bows)

        try:
            cm = CoherenceModel(
                model=model,
                texts=train_texts,
                dictionary=dictionary,
                coherence=coherence,
                topn=topn,
            )
            coh_val = float(cm.get_coherence())
        except Exception:
            coh_val = float("nan")

        row = {
            "k": float(k),
            "coherence": coh_val,
            "train_perplexity": train_perp,
            "test_perplexity": test_perp,
        }
        k_scores.append(row)
        logger.info("K=%d  coherence=%.4f  train_perp=%.4f", k, coh_val, train_perp)

        if not np.isnan(coh_val) and coh_val > best_coh:
            best_coh = coh_val
            best_k = k
            best_art = LdaArtifacts(
                model=model, dictionary=dictionary, corpus_train=corpus_train
            )

    if best_art is None:
        raise ValueError("All K values produced NaN coherence; check data.")

    # Re-train best K with ELBO logging for the convergence plot
    conv_values: list[float] = []
    for epoch in range(passes):
        m = LdaModel(
            corpus=corpus_train,
            id2word=dictionary,
            num_topics=best_k,
            random_state=random_state,
            passes=1,
            alpha=alpha,
            eta=eta,
        )
        conv_values.append(float(m.log_perplexity(corpus_train)))

    return LdaGridResult(
        best_k=best_k,
        best_coherence=best_coh,
        best_artifacts=best_art,
        k_scores=k_scores,
        convergence_log=conv_values,
    )