from src.data.preprocess import build_ngrams, corpus_statistics, tokenize_text

import pandas as pd


def test_tokenize_drops_stopwords():
    toks = tokenize_text("The climate and emissions report")
    assert "the" not in toks
    assert "climate" in toks
    assert "emissions" in toks


def test_tokenize_with_lemmatize():
    import pytest

    try:
        toks = tokenize_text("floods flooding flooded", lemmatize=True)
    except LookupError:
        pytest.skip("NLTK wordnet data not available in this environment")
    assert all(t == "flood" for t in toks)


def test_build_bigrams():
    grams = build_ngrams(["climate", "carbon", "emissions"], n=2)
    assert grams == ["climate_carbon", "carbon_emissions"]
    assert build_ngrams(["single"], n=2) == []


def test_corpus_statistics():
    df = pd.DataFrame({"tokens": [["a", "b", "c"], ["d", "e"]]})
    stats = corpus_statistics(df)
    assert stats["n_documents"] == 2
    assert stats["vocab_size"] == 5
    assert stats["total_tokens"] == 5
    assert stats["mean_doc_length"] == 2.5