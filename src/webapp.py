"""Flask UI for exploring drift + events interactively."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_file

from src.baselines import frequency_top_terms
from src.data_utils import add_time_bins, load_csv, parse_extra_stopwords, split_train_test, tokenize_df
from src.embeddings import encode_texts, mean_embedding_per_bin
from src.events import detect_spike_events, label_events, lexical_jaccard_drift_from_top_terms
from src.main import (
    DEFAULT_CSV,
    DEFAULT_TEST_END,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
)
from src.metrics import mean_adjacent_cosine_distance

app = Flask(__name__, static_folder=None)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _dashboard_file() -> str:
    return os.path.join(_repo_root(), "dashboard.html")


def _fmt_bin(v: Any) -> str:
    s = str(v)
    return s[:10] if len(s) >= 10 else s


def _normalize_counts(term_counts: list[tuple[str, int]]) -> list[tuple[str, int]]:
    if not term_counts:
        return []
    total = sum(c for _, c in term_counts)
    if total <= 0:
        return [(t, 0) for t, _ in term_counts]
    return [(term, int(round((count / total) * 100))) for term, count in term_counts]


def _status_for_score(score: float) -> str:
    if score >= 0.85:
        return "Anomalous"
    if score >= 0.65:
        return "Active Drift"
    return "Stable"


@lru_cache(maxsize=32)
def _compute_payload(
    *,
    csv_path: str,
    window: str,
    event_percentile: float,
    event_top_k_terms: int,
    extra_stopwords_spec: str,
    embedding_model: str,
) -> dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    extra_stop = parse_extra_stopwords(extra_stopwords_spec)
    df = load_csv(csv_path)
    df = tokenize_df(df)
    df = add_time_bins(df, freq=window)

    train_df, test_df = split_train_test(
        df,
        DEFAULT_TRAIN_START,
        DEFAULT_TRAIN_END,
        DEFAULT_TEST_START,
        DEFAULT_TEST_END,
    )

    # -------- Lexical drift + lexical events (explainable baseline) --------
    freq_terms_by_bin = frequency_top_terms(test_df, k=event_top_k_terms, extra_stop=extra_stop)
    lex_drift, lex_transitions = lexical_jaccard_drift_from_top_terms(
        freq_terms_by_bin, top_k=event_top_k_terms
    )
    lex_events = detect_spike_events(
        drift_scores=lex_drift,
        transitions=lex_transitions,
        percentile=event_percentile,
    )

    events_out = {"lexical": [], "lda": []}
    for e in lex_events:
        b0, b1 = lex_transitions[e.spike_transition_index]
        t0 = freq_terms_by_bin.get(b0, [])[:event_top_k_terms]
        t1 = freq_terms_by_bin.get(b1, [])[:event_top_k_terms]
        prev_set = {w for w, _ in t0}
        next_set = {w for w, _ in t1}
        new_terms = [w for w, _ in t1 if w not in prev_set][:8]
        dropped_terms = [w for w, _ in t0 if w not in next_set][:8]
        events_out["lexical"].append(
            {
                "event_id": e.event_id,
                "start_bin": b0,
                "end_bin": b1,
                "score": float(e.max_score),
                "drivers": {"new_terms": new_terms, "dropped_terms": dropped_terms},
            }
        )

    labeled = label_events(events_out, test_df=test_df, include_category=True)
    labeled_lex = labeled.get("lexical", [])

    # -------- Semantic drift (embeddings) --------
    semantic = {"available": False}
    try:
        if len(test_df) >= 2 and test_df["time_bin"].nunique() >= 2:
            X_test, _ = encode_texts(
                test_df.sort_values("timestamp")["text"].tolist(),
                model_name=embedding_model,
                show_progress_bar=False,
            )
            bins, cents = mean_embedding_per_bin(test_df.sort_values("timestamp"), X_test)
            sem_drift_mean = mean_adjacent_cosine_distance(cents)
            semantic = {
                "available": True,
                "mean_adjacent_cosine_distance": float(sem_drift_mean),
                "n_bins": int(len(bins)),
            }
    except Exception as exc:  # pragma: no cover
        semantic = {"available": False, "error": str(exc)}

    # -------- Summary & UI-friendly formatting --------
    lex_arr = np.asarray(lex_drift if lex_drift else [0.0], dtype=float)
    avg_lex = float(np.mean(lex_arr))
    peak_lex = float(np.max(lex_arr))

    bins_test = sorted(test_df["time_bin"].unique())
    latest_bin = bins_test[-1] if bins_test else None
    latest_terms = freq_terms_by_bin.get(latest_bin, []) if latest_bin is not None else []
    topic_distribution = [
        {"label": term.replace("_", " ").title(), "percent": pct}
        for term, pct in _normalize_counts(latest_terms[:6])
    ]

    # bars: normalize first 14 points for plotting
    raw = [float(x) for x in (lex_drift[:14] if lex_drift else [0.0])]
    max_v = max(raw) if raw else 1.0
    bars = [round((v / max_v) * 100, 1) if max_v else 0.0 for v in raw]
    peak_index = int(np.argmax(raw)) if raw else 0

    event_rows = []
    for idx, ev in enumerate(
        sorted(labeled_lex, key=lambda e: float(e.get("score", 0.0)), reverse=True)[:12],
        start=1,
    ):
        score = float(ev.get("score", 0.0))
        event_rows.append(
            {
                "id": f"EV-{88000 + idx}",
                "label": ev.get("label") or "Lexical drift",
                "dominant_category": ev.get("dominant_category") or "—",
                "score": round(score, 3),
                "status": _status_for_score(score),
                "start_bin": _fmt_bin(ev.get("start_bin")),
                "end_bin": _fmt_bin(ev.get("end_bin")),
                "drivers": ev.get("drivers", {}),
            }
        )

    return {
        "params": {
            "csv_path": csv_path,
            "window": window,
            "event_percentile": float(event_percentile),
            "event_top_k_terms": int(event_top_k_terms),
            "extra_stopwords": extra_stopwords_spec,
            "embedding_model": embedding_model,
        },
        "summary": {
            "total_documents": int(len(df)),
            "train_documents": int(len(train_df)),
            "test_documents": int(len(test_df)),
            "detected_events": int(len(labeled_lex)),
            "avg_lexical_drift": round(avg_lex, 4),
            "peak_lexical_drift": round(peak_lex, 4),
            "test_range": (
                f"{_fmt_bin(bins_test[0])} - {_fmt_bin(bins_test[-1])}" if bins_test else "N/A"
            ),
        },
        "drift_chart": {
            "bars": bars,
            "raw_values": [round(v, 4) for v in raw],
            "peak_index": peak_index,
            "peak_value": round(max(raw), 4) if raw else 0.0,
        },
        "topic_distribution": topic_distribution,
        "events": event_rows,
        "semantic": semantic,
    }


@app.get("/")
def serve_dashboard():
    return send_file(_dashboard_file())


@app.get("/api/dashboard")
def api_dashboard():
    csv_path = os.path.abspath(request.args.get("csv_path", DEFAULT_CSV))
    window = request.args.get("window", "7D")
    event_percentile = float(request.args.get("event_percentile", 90.0))
    event_top_k_terms = int(request.args.get("event_top_k_terms", 15))
    extra_stopwords = request.args.get("extra_stopwords", "none")
    embedding_model = request.args.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        payload = _compute_payload(
            csv_path=csv_path,
            window=window,
            event_percentile=event_percentile,
            event_top_k_terms=event_top_k_terms,
            extra_stopwords_spec=extra_stopwords,
            embedding_model=embedding_model,
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Failed to compute dashboard: {exc}"}), 500

    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

