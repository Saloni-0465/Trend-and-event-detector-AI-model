# Dynamic Trend & Event Detector

Detects and tracks evolving topics in news article streams using three approaches:
1. **Baselines** — frequency counts and TF-IDF per time window
2. **LDA** — probabilistic topic model (gensim)
3. **Embeddings + KMeans** — sentence transformer embeddings clustered with KMeans

## Dataset

We use the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle (~210k HuffPost headlines, 2012–2022).

### Download

1. Install Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` in `~/.kaggle/`
3. Run: `python scripts/download_data.py`

This downloads the dataset and creates a 6-month subset at `data/sample/news_2018_h1.csv`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Usage

```bash
# run all parts (baselines + LDA + embeddings)
python -m src.main --mode all

# just baselines
python -m src.main --mode baselines

# just LDA with custom topic count
python -m src.main --mode lda --num-topics 6

# just embeddings (pretrained SentenceTransformer + K-Means)
python -m src.main --mode embeddings

# optional: another Hugging Face sentence-transformers checkpoint
python -m src.main --mode embeddings --embedding-model sentence-transformers/all-MiniLM-L6-v2

# lexical + LDA event spikes + simple labels (test period)
python -m src.main --mode events

# use full dataset instead of 6-month subset
python -m src.main --mode all --csv-path data/news_full.csv

# custom chronological split (ISO dates; end dates exclusive)
python -m src.main --mode all \
  --train-start 2018-01-01 --train-end 2018-05-01 \
  --test-start 2018-05-01 --test-end 2018-07-01
```

### Quick test / evaluation commands

```bash
# unit tests (no model download if embeddings tests use synthetic matrices only)
pytest tests/ -q

# end-to-end on the 6-month sample (downloads MiniLM on first run)
python -m src.main --mode all

# DL path only: NMI/ARI vs category + semantic drift on test bins
python -m src.main --mode embeddings
```

## Tests

```bash
pytest tests/ -v
```

## Project structure

```
src/
  data_utils.py    — data loading, tokenization, time bins
  baselines.py     — frequency + TF-IDF per time window
  lda_model.py     — LDA topic model (gensim)
  embeddings.py    — sentence embeddings + KMeans
  events.py        — spike-style event detection helpers + labeling
  metrics.py       — Jaccard, JSD, silhouette, NMI, ARI
  main.py          — CLI entry point
scripts/
  download_data.py — download + preprocess Kaggle dataset
  eda.py           — exploratory plots (run: python scripts/eda.py from repo root)
tests/
  test_pipeline.py — unit tests
docs/
  literature_review.md — background research
  report.md        — Phase 1 report
```
