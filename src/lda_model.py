"""Basic LDA topic model using gensim."""

import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel


def train_lda(token_lists, num_topics=8, passes=15, seed=42):
    """Train LDA on a list of token lists. Returns model, dictionary, corpus."""
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=1, no_above=0.85)
    corpus = [dictionary.doc2bow(doc) for doc in token_lists]

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=passes,
        alpha="auto",
        eta="auto",
    )
    return model, dictionary, corpus


def get_topic_words(model, num_words=10):
    """Get top words for each topic."""
    topics = []
    for tid in range(model.num_topics):
        words = model.show_topic(tid, topn=num_words)
        topics.append({"topic_id": tid, "words": " ".join(w for w, _ in words)})
    return topics


def get_perplexity(model, corpus):
    """Log perplexity on a corpus (higher = better in gensim)."""
    if not corpus:
        return float("nan")
    return float(model.log_perplexity(corpus))


def get_coherence(model, texts, dictionary, measure="u_mass"):
    """Topic coherence score."""
    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                        coherence=measure, topn=10)
    return float(cm.get_coherence())


def doc_topic_matrix(model, corpus):
    """Get topic distribution for each document. Returns (n_docs, n_topics) array."""
    n_topics = model.num_topics
    rows = []
    for bow in corpus:
        vec = np.zeros(n_topics)
        for tid, prob in model.get_document_topics(bow, minimum_probability=0):
            vec[tid] = prob
        s = vec.sum()
        if s > 0:
            vec /= s
        else:
            vec[:] = 1.0 / n_topics
        rows.append(vec)
    return np.array(rows)
