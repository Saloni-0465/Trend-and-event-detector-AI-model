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
| `src/part1_runner.py` | Part 1 orchestration (baselines + metrics) |
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

Later phases will add **Model A (probabilistic topics)**, **Model B (DL embeddings)**, **Model C (hybrid)**, and the required **ablation table**.

## Deliverables checklist (course)

- Ablation table: Advanced ML only | DL only | Hybrid
- Architecture diagram under `docs/`
- Short conference-style report (6–8 pages)
- Reproducible `requirements.txt` (or conda lockfile)

## License

MIT (adjust if your course requires otherwise).
