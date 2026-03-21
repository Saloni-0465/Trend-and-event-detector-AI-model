"""
Latent Dirichlet Allocation via gensim (variational / batch updates).

Theory hooks for the report (do not treat as a black box):
- **Generative story:** for each document, draw topic proportions θ ~ Dir(α); for each
  token, draw topic z ~ Mult(θ), then word w ~ Mult(β_z). With Dirichlet prior on rows
  of β (η), this is smoothed LDA.
- **Inference:** gensim's `LdaModel` optimizes a variational lower bound (ELBO) with
  iterative coordinate-ascent / online updates — closely related to **variational EM**,
  not collapsed Gibbs sampling. Monitor **perplexity** on held-out bags-of-words for
  generalization; **coherence** (e.g. u_mass, c_v) assesses human-interpretability of
  top-N words per topic.
- **Assumptions / limits:** bag-of-words (order ignored), fixed **K**, static β over
  time (not a dynamic topic model), and independence of documents given topics — so
  rapid event bursts or copy-paste duplication violate the generative story.

See: Blei et al., LDA; gensim documentation for `LdaModel` optimization details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel


@dataclass(frozen=True)
class LdaArtifacts:
    model: LdaModel
    dictionary: corpora.Dictionary
    corpus_train: list[list[tuple[int, int]]]


def build_dictionary_and_corpus(
    token_lists: list[list[str]],
    *,
    no_below: int = 2,
    no_above: float = 0.85,
    keep_n: int | None = 200_000,
) -> tuple[corpora.Dictionary, list[list[tuple[int, int]]]]:
    """Fit dictionary on *training* texts only; return BoW corpus."""
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(t) for t in token_lists]
    return dictionary, corpus


def bows_with_fixed_dictionary(
    token_lists: list[list[str]], dictionary: corpora.Dictionary
) -> list[list[tuple[int, int]]]:
    return [dictionary.doc2bow(t, allow_update=False) for t in token_lists]


def train_lda(
    token_lists_train: list[list[str]],
    *,
    num_topics: int = 8,
    passes: int = 15,
    alpha: str | float | np.ndarray = "auto",
    eta: str | float | np.ndarray = "auto",
    random_state: int = 42,
    no_below: int = 2,
    no_above: float = 0.85,
) -> LdaArtifacts:
    dictionary, corpus_train = build_dictionary_and_corpus(
        token_lists_train, no_below=no_below, no_above=no_above
    )
    if len(dictionary) == 0:
        raise ValueError(
            "Empty vocabulary after filter_extremes — lower no_below or raise no_above."
        )
    if sum(len(b) for b in corpus_train) == 0:
        raise ValueError("All training documents are empty after tokenization / filtering.")
    model = LdaModel(
        corpus=corpus_train,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        alpha=alpha,
        eta=eta,
        per_word_topics=False,
    )
    return LdaArtifacts(model=model, dictionary=dictionary, corpus_train=corpus_train)


def log_perplexity_on_corpus(model: LdaModel, corpus: list[list[tuple[int, int]]]) -> float:
    """Gensim log perplexity on a held-out BoW corpus (higher is better)."""
    if not corpus:
        return float("nan")
    return float(model.log_perplexity(corpus))


def coherence_for_model(
    model: LdaModel,
    token_lists_reference: list[list[str]],
    dictionary: corpora.Dictionary,
    *,
    coherence: Literal["u_mass", "c_v"] = "u_mass",
    topn: int = 10,
) -> float:
    """
    Topic coherence using training-like token lists as reference corpus.

    `u_mass` is fast and dictionary-based; `c_v` is slower but often preferred for
    interpretability — swap in the report when you have enough documents.
    """
    cm = CoherenceModel(
        model=model,
        texts=token_lists_reference,
        dictionary=dictionary,
        coherence=coherence,
        topn=topn,
    )
    return float(cm.get_coherence())


def doc_topic_mixture_dense(
    model: LdaModel, bow: list[tuple[int, int]], n_topics: int
) -> np.ndarray:
    """Normalize variational topic posterior for one document to a simplex vector."""
    vec = np.zeros(n_topics, dtype=np.float64)
    for topic_id, p in model.get_document_topics(bow, minimum_probability=0):
        if 0 <= int(topic_id) < n_topics:
            vec[int(topic_id)] = float(p)
    s = vec.sum()
    if s <= 0:
        vec[:] = 1.0 / n_topics
    else:
        vec /= s
    return vec


def document_topic_matrix(
    model: LdaModel,
    bows: list[list[tuple[int, int]]],
    *,
    n_topics: int,
) -> np.ndarray:
    """Shape (n_docs, n_topics)."""
    rows = [doc_topic_mixture_dense(model, b, n_topics) for b in bows]
    return np.vstack(rows) if rows else np.zeros((0, n_topics))


def top_words_per_topic(
    model: LdaModel, *, num_words: int = 12
) -> list[list[tuple[str, float]]]:
    out: list[list[tuple[str, float]]] = []
    for topic_id in range(model.num_topics):
        terms = model.show_topic(topic_id, topn=num_words)
        out.append([(str(w), float(p)) for w, p in terms])
    return out
