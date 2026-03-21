# Dynamic Trend & Event Detector

Social-media–style text streams: emerging topics, temporal dynamics, and event-like shifts. This repository supports course **Project 8** baselines (frequency topics), advanced probabilistic topic models, deep embedding–based clustering, and a planned hybrid pipeline with ablation-friendly evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you add API keys or paths later.

## Layout

| Path | Role |
|------|------|
| `src/baselines/` | Frequency / simple topic extraction |
| `src/advanced_ml/` | Probabilistic topic models (e.g., LDA) |
| `src/deep_learning/` | Transformer embeddings + clustering |
| `src/hybrid/` | Temporal + semantic integration |
| `src/evaluation/` | Metrics and ablation helpers |
| `data/sample/` | Tiny example stream (committed) |
| `data/raw`, `data/processed` | Your large data (gitignored except `.gitkeep`) |
| `src/data/` | Synthetic generator, CSV loader, preprocessing |
| `src/pipeline_common.py` | Shared CSV / synthetic loading for all parts |
| `src/part1_runner.py` | Part 1 orchestration (baselines + metrics) |
| `src/part2_runner.py` | Part 2: LDA training, perplexity, coherence, topic mix over time |
| `src/part3_runner.py` | Part 3: sentence embeddings + K-Means (k via train silhouette) |
| `src/advanced_ml/lda_gensim.py` | LDA training + theory hooks (ELBO / assumptions) in docstring |
| `src/deep_learning/embedder.py` | Frozen MiniLM-class encoder (`sentence-transformers`) |
| `src/deep_learning/kmeans_time.py` | K-Means with k selected on the train time slice |
| `notebooks/` | Exploratory work |
| `docs/` | Report sources, architecture diagram |

## Part 1 (complete): data + temporal baselines

**Scope:** reproducible stream → time bins → **frequency** and **TF-IDF** “topics” per window → **adjacent-window Jaccard** on top-*k* terms (temporal drift diagnostic). This satisfies the course **baseline ML** slice and sets up **time-aware evaluation** for Phases 2–4 (LDA, embeddings, hybrid).

```bash
# Synthetic stream (default): no external data required
python -m src.pipeline part1 --source synthetic --n-docs 800 --window 7D

# Small committed sample CSV
python -m src.pipeline part1 --source csv --csv-path data/sample/stream_sample.csv --window 7D

# Write results under data/processed/ (gitignored except .gitkeep)
python -m src.pipeline part1 --source synthetic --save --output-dir data/processed
```

**Your own data:** CSV with `timestamp` (ISO-8601) and `text` columns (or pass `--time-col` / `--text-col`). Place large files under `data/raw/` (ignored by git).

**Tests:** `pytest -q` from repo root (uses `pyproject.toml` `pythonpath`).

## Part 2 (complete): Advanced ML — LDA

**Scope:** **Latent Dirichlet Allocation** via [gensim](https://radimrehurek.com/gensim/) `LdaModel` (variational / batch optimization of the ELBO — relate to **variational EM** in your report). **Train/test split follows time** (earlier documents train, later held out — no random shuffle). Metrics: **held-out log perplexity**, topic **coherence** (`u_mass` default for speed; try `c_v` on larger corpora), and **mean adjacent Jensen–Shannon distance** between window-averaged topic mixtures (temporal dynamics).

```bash
python -m src.pipeline part2 --source synthetic --n-docs 800 --num-topics 8 --window 7D
python -m src.pipeline part2 --source csv --csv-path data/sample/stream_sample.csv \
  --num-topics 4 --passes 8 --no-below 1 --no-above 1.0
python -m src.pipeline part2 --source synthetic --save --output-dir data/processed
```

**Report angles (rubric):** bag-of-words assumption, fixed **K**, document independence, what **perplexity** measures vs what **coherence** measures, and why static LDA is not a full **dynamic topic model** (optional stretch: cite DTM / neural topics for future work).

## Part 3 (complete): Deep learning — embeddings + clustering

**Scope (Model B):** **Frozen** [sentence-transformers](https://www.sbert.net/) encoder (default `all-MiniLM-L6-v2`) → **L2-normalized** vectors → **K-Means**. **K** is chosen on the **training time slice** by maximizing **silhouette** (no peeking at the held-out tail). **Metrics:** silhouette on train/test assignments, **NMI/ARI vs `theme`** when using the synthetic generator (CSV sample has no labels → those scores stay NaN), and **mean adjacent JSD** on per-window cluster distributions. **Rubric hooks:** frozen backbone as **regularization**; dropout/LayerNorm live inside the pretrained checkpoint; **K-Means `max_iter`** caps the clustering optimization; optional note that **FlashAttention** applies to *training* large models — not required for this small inference-only path.

```bash
# First run downloads the MiniLM weights (~90MB) into the HF cache
python -m src.pipeline part3 --source synthetic --n-docs 400 --window 7D
python -m src.pipeline part3 --source csv --csv-path data/sample/stream_sample.csv --k-max 8
python -m src.pipeline part3 --source synthetic --save --output-dir data/processed
```

**Tests** mock embeddings so CI does not download models; a full local run exercises `sentence-transformers` + `torch`.

Next: **Part 4 / Model C (hybrid)** and the **ablation table** (A vs B vs C).

## Deliverables checklist (course)

- Ablation table: Advanced ML only | DL only | Hybrid
- Architecture diagram under `docs/`
- Short conference-style report (6–8 pages)
- Reproducible `requirements.txt` (or conda lockfile)

## License

MIT (adjust if your course requires otherwise).
