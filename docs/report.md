# Dynamic Trend & Event Detector — Phase 1 Report

## Abstract

We built a pipeline that detects and tracks evolving topics in a news article stream. Three approaches are compared: frequency/TF-IDF baselines, LDA topic models, and sentence-embedding clustering. We use the News Category Dataset (HuffPost, ~210k articles) as our real-world data source.

## 1. Data

We use the **News Category Dataset** from Kaggle (Rishabh Misra), containing ~210,000 news headlines and short descriptions from HuffPost spanning 2012–2022. Each article has a timestamp and one of 42 category labels (POLITICS, ENTERTAINMENT, WELLNESS, etc.).

For initial experiments we work with a 6-month subset (Jan–Jun 2018) to keep runtimes manageable. The text field combines the headline and short description for richer content.

### 1.1 Exploratory plots (EDA)

Before choosing preprocessing, we inspect the sample with `python scripts/eda.py` (from the repo root, after `download_data.py`). That script creates **two PNG files** in `docs/`:

- `docs/categories_over_time.png` — stacked bar chart of article counts per week for the six most common categories  
- `docs/doc_lengths.png` — histogram of token counts per document after preprocessing  

Those files **are not created by** `src.main`; they only exist after you run `eda.py` (or if someone commits them for you). Until then, the image links below will be broken in Markdown viewers.

**Category mix over time** — stacked counts per week for the six most frequent editor categories. This shows that the stream is not static: different weeks emphasize different sections, which motivates time-binned baselines and drift metrics.

![Top six news categories by 7-day time bin](categories_over_time.png)

**Document length (tokens)** — distribution after tokenization and stopword removal. Most headlines + descriptions are short; we avoid aggressive stemming because lengths are already modest.

![Histogram of token counts per document](doc_lengths.png)

*If the images do not render in your viewer, run `python scripts/eda.py` once so the PNG files exist next to this report, or open `docs/categories_over_time.png` and `docs/doc_lengths.png` directly.*

## 2. Preprocessing

- Lowercasing, regex tokenization, stopword removal (sklearn's English list)
- Short tokens (< 2 chars) dropped
- No stemming/lemmatization — headlines are short and already pretty clean

## 3. Methods

### 3.1 Baselines

- **Frequency**: count top-k terms per 7-day window
- **TF-IDF**: treat each window as one document, rank terms by TF-IDF weight

We measure lexical drift between adjacent windows using Jaccard similarity on the top-15 term sets.

### 3.2 LDA

- Gensim's `LdaModel` with `alpha=auto`, `eta=auto` (Phase 1: fit on the **full** corpus after sorting by time — simple and easy to explain)
- Evaluate: gensim log-perplexity on that corpus and u_mass coherence
- Aggregate doc-topic distributions per time bin and compute JSD between adjacent bins

LDA treats each document as a mixture of topics and each topic as a distribution over words. Perplexity is a quick sanity check; coherence checks if topic words co-occur in the corpus (Mimno et al.).

### 3.3 Embeddings + Clustering (optional demo)

- Encode all texts with `all-MiniLM-L6-v2` (frozen sentence transformer)
- Pick *k* with best silhouette on the embedding matrix (Phase 1 uses a small range, e.g. k=2..6)
- KMeans on all points — kept minimal for Phase 1; a stricter time-based train/test split can come in a later phase

## 4. Results

Results are from the 6-month 2018 subset. Exact numbers depend on the window size and number of topics.

| Metric | Baselines | LDA | Embeddings |
|--------|-----------|-----|------------|
| Jaccard (adjacent windows) | ~0.4–0.6 | — | — |
| Coherence (u_mass) | — | ~ -2 to -5 | — |
| Perplexity (log, same corpus) | — | ~ -5 to -7 | — |
| Chosen *k* (silhouette) | — | — | 2–6 (data-dependent) |

## 5. Discussion

- Baselines show moderate lexical overlap between windows, confirming some topic continuity
- LDA finds interpretable topics but coherence varies with K
- Embedding tier is a minimal demo (Phase 1); NMI/ARI vs category can be added when you need stricter evaluation
- All three approaches detect temporal drift through different lenses

## 6. Limitations

- Using only a 6-month subset for speed
- No hyperparameter tuning beyond basic k-selection
- No burst or event detection yet
- KMeans assumes spherical clusters which may not fit news categories well

## 7. Next Steps

- Grid search for LDA num_topics
- Hybrid model combining embeddings with temporal topic evolution
- Event detection and impact scoring
- Full dataset experiments

## Reproducibility

**Core pipeline** (what you need whenever you run the project):

```bash
pip install -r requirements.txt
python scripts/download_data.py
python -m src.main --mode all
pytest tests/
```


```bash
python scripts/eda.py
```

If the PNGs are committed in `docs/`, readers see the figures without re-running EDA. Re-run `eda.py` only when those inputs change or the images are missing.
