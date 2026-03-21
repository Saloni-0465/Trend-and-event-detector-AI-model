"""Probabilistic topic models — LDA implementation lives in `lda_gensim`."""

from src.advanced_ml.lda_gensim import (
    LdaArtifacts,
    coherence_for_model,
    document_topic_matrix,
    log_perplexity_on_corpus,
    train_lda,
)

__all__ = [
    "LdaArtifacts",
    "coherence_for_model",
    "document_topic_matrix",
    "log_perplexity_on_corpus",
    "train_lda",
]
