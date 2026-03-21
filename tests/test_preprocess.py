from src.data.preprocess import tokenize_text


def test_tokenize_drops_stopwords():
    toks = tokenize_text("The climate and emissions report")
    assert "the" not in toks
    assert "climate" in toks
    assert "emissions" in toks
