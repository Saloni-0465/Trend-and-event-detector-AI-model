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
| `data/raw`, `data/processed` | Data (gitignored by default except `.gitkeep`) |
| `notebooks/` | Exploratory work |
| `docs/` | Report sources, architecture diagram |

## Running (placeholder)

```bash
python -m src.pipeline --help
```

Implementation and experiments will be filled in as the project progresses.

## Deliverables checklist (course)

- Ablation table: Advanced ML only | DL only | Hybrid
- Architecture diagram under `docs/`
- Short conference-style report (6–8 pages)
- Reproducible `requirements.txt` (or conda lockfile)

## License

MIT (adjust if your course requires otherwise).
