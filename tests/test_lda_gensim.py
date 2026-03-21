import numpy as np

from src.advanced_ml.lda_gensim import (
    bows_with_fixed_dictionary,
    document_topic_matrix,
    log_perplexity_on_corpus,
    train_lda,
)
from src.data.synthetic import generate_synthetic_social_stream
from src.data.preprocess import add_tokens_column


def test_train_lda_and_perplexity_finite():
    df = generate_synthetic_social_stream(n_docs=200, seed=7)
    df = add_tokens_column(df)
    texts = df["tokens"].tolist()
    train = texts[:150]
    test = texts[150:]
    art = train_lda(train, num_topics=6, passes=5, random_state=0, no_below=1, no_above=0.9)
    test_bow = bows_with_fixed_dictionary(test, art.dictionary)
    lp = log_perplexity_on_corpus(art.model, test_bow)
    assert np.isfinite(lp)

    theta = document_topic_matrix(art.model, test_bow, n_topics=art.model.num_topics)
    assert theta.shape == (len(test), art.model.num_topics)
    assert np.allclose(theta.sum(axis=1), 1.0, atol=1e-5)
