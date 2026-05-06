"""Generate frontend dashboard data from the HuffPost news dataset.

Pipeline:
1. load raw JSONL articles
2. build weekly activity/KPI metrics
3. run TF-IDF over recent article text
4. train LDA topic model
5. encode recent articles with SentenceTransformer and cluster with KMeans
6. rank current clusters as trends and forecast near-future topics

The output is a static JSON artifact consumed directly by the Next.js frontend.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "News_Category_Dataset_v3.json"
DEFAULT_OUTPUT = ROOT.parent / "frontend" / "src" / "data" / "dashboard.json"

STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "new",
    "said",
    "says",
    "say",
    "people",
    "year",
    "years",
    "week",
    "day",
    "time",
    "huffpost",
    "video",
    "photos",
}


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["headline"] = df["headline"].fillna("").astype(str)
    df["short_description"] = df["short_description"].fillna("").astype(str)
    df["text"] = (df["headline"] + ". " + df["short_description"]).str.strip()
    df["category"] = df["category"].fillna("UNKNOWN").astype(str)
    df = df.dropna(subset=["timestamp"])
    df = df[df["text"].str.len() > 10]
    return df.sort_values("timestamp").reset_index(drop=True)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.lower().replace("'", " ").split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) >= 3 and token not in STOPWORDS and not token.isnumeric():
            out.append(token)
    return out


def compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def pct_change(current: int, previous: int) -> float:
    if previous <= 0:
        return 0.0
    return round(((current - previous) / previous) * 100.0, 1)


def choose_k(embeddings: np.ndarray, seed: int) -> int:
    upper = min(10, len(embeddings) - 1)
    if upper <= 3:
        return max(2, upper)

    best_k = 6
    best_score = -1.0
    for k in range(4, upper + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k = k
            best_score = float(score)
    return best_k


def cluster_keywords(texts: list[str], labels: np.ndarray, cluster_id: int, top_n: int = 6) -> list[str]:
    docs = [text for text, label in zip(texts, labels) if int(label) == cluster_id]
    if not docs:
        return []
    vec = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        max_features=3000,
        min_df=2,
    )
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:
        return []
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    names = vec.get_feature_names_out()
    order = np.argsort(-scores)[:top_n]
    return [str(names[i]).title() for i in order if scores[i] > 0]


def lda_topics(token_lists: list[list[str]], *, num_topics: int, seed: int) -> list[dict[str, Any]]:
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=5000)
    corpus = [dictionary.doc2bow(tokens) for tokens in token_lists]
    corpus = [bow for bow in corpus if bow]
    if not corpus or len(dictionary) == 0:
        return []

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=seed,
        passes=5,
        alpha="auto",
        eta="auto",
    )
    topics: list[dict[str, Any]] = []
    for topic_id in range(num_topics):
        words = [word for word, _ in model.show_topic(topic_id, topn=6)]
        topics.append({"id": topic_id, "words": [w.title() for w in words]})
    return topics


def direction_from_velocity(velocity: float, score: float) -> str:
    if velocity >= 35 or score >= 85:
        return "breakout"
    if velocity < 0:
        return "cooling"
    return "rising"


def event_type_from_velocity(velocity: float) -> str:
    if velocity >= 40:
        return "spike"
    if velocity < 0:
        return "declining"
    return "emerging"


def alert_severity(velocity: float, score: float) -> str:
    if velocity >= 80 or score >= 80:
        return "critical"
    if velocity < 0 or velocity >= 25:
        return "warning"
    return "info"


def build_dashboard(df: pd.DataFrame, *, sample_size: int, seed: int) -> dict[str, Any]:
    latest = df["timestamp"].max()
    latest_week_start = latest.floor("7D")
    previous_week_start = latest_week_start - pd.Timedelta(days=7)

    last_week_count = int((df["timestamp"] >= latest_week_start).sum())
    previous_week_count = int(((df["timestamp"] >= previous_week_start) & (df["timestamp"] < latest_week_start)).sum())
    change = pct_change(last_week_count, previous_week_count)

    weekly = (
        df.assign(time_bin=df["timestamp"].dt.floor("7D"))
        .groupby("time_bin")
        .size()
        .tail(8)
        .reset_index(name="value")
    )
    activity = [
        {"name": row.time_bin.strftime("%b %d"), "value": int(row.value)}
        for row in weekly.itertuples(index=False)
    ]

    recent_start = latest - pd.Timedelta(days=180)
    recent = df[df["timestamp"] >= recent_start].copy()
    if len(recent) > sample_size:
        recent = recent.sample(sample_size, random_state=seed).sort_values("timestamp").reset_index(drop=True)
    else:
        recent = recent.reset_index(drop=True)

    recent["tokens"] = recent["text"].apply(tokenize)
    texts = recent["text"].tolist()

    print(f"Encoding {len(texts)} recent articles with SentenceTransformer...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    k = choose_k(embeddings, seed)
    print(f"Clustering embeddings with k={k}...")
    labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(embeddings)
    recent["cluster"] = labels

    current_start = latest - pd.Timedelta(days=14)
    previous_start = latest - pd.Timedelta(days=28)
    cluster_rows: list[dict[str, Any]] = []
    total_current = max(1, int((recent["timestamp"] >= current_start).sum()))

    for cluster_id in sorted(set(int(x) for x in labels)):
        part = recent[recent["cluster"] == cluster_id]
        current = int((part["timestamp"] >= current_start).sum())
        previous = int(((part["timestamp"] >= previous_start) & (part["timestamp"] < current_start)).sum())
        if current == 0 and previous == 0:
            continue

        velocity = pct_change(current, previous)
        volume_share = current / total_current
        velocity_component = max(0.0, min(1.0, (velocity + 20.0) / 120.0))
        score = round(min(99.0, 100.0 * (0.58 * volume_share + 0.42 * velocity_component)), 1)
        keywords = cluster_keywords(texts, labels, cluster_id)
        if not keywords:
            keywords = [str(x).title() for x, _ in Counter(sum(part["tokens"].tolist(), [])).most_common(6)]
        category = str(part["category"].mode().iloc[0]) if not part.empty else "UNKNOWN"
        label = " ".join(keywords[:3]) if keywords else category.title()
        sentiment = "Positive" if velocity >= 10 else "Negative" if velocity <= -10 else "Neutral"

        cluster_rows.append(
            {
                "cluster": cluster_id,
                "name": label,
                "score": score,
                "velocity": velocity,
                "sentiment": sentiment,
                "keywords": keywords[:4],
                "articles_count": current,
                "category": category.title(),
                "previous_mentions": previous,
            }
        )

    trends = sorted(cluster_rows, key=lambda row: (row["score"], row["articles_count"]), reverse=True)[:6]

    topics = lda_topics(recent["tokens"].tolist(), num_topics=8, seed=seed)
    topic_words = [topic["words"] for topic in topics]
    predictions: list[dict[str, Any]] = []
    for idx, trend in enumerate(sorted(cluster_rows, key=lambda row: row["velocity"], reverse=True)[:6], start=1):
        current = int(trend["articles_count"])
        previous = int(trend["previous_mentions"])
        velocity = float(trend["velocity"])
        forecast_multiplier = 1.0 + max(-0.6, min(1.4, velocity / 100.0))
        predicted_mentions = int(max(1, round(current * forecast_multiplier)))
        confidence = round(min(0.94, max(0.55, 0.58 + (trend["score"] / 250.0) + min(0.12, abs(velocity) / 300.0))), 2)
        predicted_score = round(min(99.0, trend["score"] * 0.72 + max(0, velocity) * 0.35 + confidence * 10), 1)
        drivers = trend["keywords"][:3]
        if idx - 1 < len(topic_words):
            drivers = list(dict.fromkeys(drivers + topic_words[idx - 1][:2]))[:4]
        predictions.append(
            {
                "id": idx,
                "topic": trend["name"],
                "predicted_score": predicted_score,
                "velocity_forecast": round(velocity, 1),
                "current_mentions": current,
                "predicted_mentions": predicted_mentions,
                "confidence": confidence,
                "horizon": "14d",
                "drivers": drivers,
                "category": trend["category"],
                "direction": direction_from_velocity(velocity, predicted_score),
            }
        )

    event_candidates = sorted(
        cluster_rows,
        key=lambda row: (abs(float(row["velocity"])), float(row["score"]), int(row["articles_count"])),
        reverse=True,
    )[:8]
    events: list[dict[str, Any]] = []
    for idx, trend in enumerate(event_candidates, start=1):
        velocity = float(trend["velocity"])
        event_type = event_type_from_velocity(velocity)
        confidence = round(min(0.96, max(0.58, 0.62 + float(trend["score"]) / 220.0 + min(0.16, abs(velocity) / 400.0))), 2)
        direction_word = "increased" if velocity >= 0 else "decreased"
        events.append(
            {
                "id": idx,
                "title": f"{trend['name']} {event_type.title()} Detected",
                "summary": (
                    f"{trend['category']} cluster volume {direction_word} by {abs(velocity):.1f}% "
                    f"from {trend['previous_mentions']} to {trend['articles_count']} articles in the latest 14-day window. "
                    f"Drivers: {', '.join(trend['keywords'][:3])}."
                ),
                "type": event_type,
                "confidence": confidence,
                "trend": trend["name"],
                "time": f"Window ending {latest.date().isoformat()}",
                "articles_count": int(trend["articles_count"]),
                "velocity": round(velocity, 1),
            }
        )

    alerts: list[dict[str, Any]] = []
    for idx, trend in enumerate(event_candidates[:6], start=1):
        velocity = float(trend["velocity"])
        severity = alert_severity(velocity, float(trend["score"]))
        if velocity >= 0:
            title = f"Spike Detected: {trend['name']}"
            message = (
                f"Article volume increased by {velocity:.1f}% versus the previous 14-day window. "
                f"The model score is {float(trend['score']):.1f} with {trend['articles_count']} current mentions."
            )
        else:
            title = f"Cooling Trend: {trend['name']}"
            message = (
                f"Article volume decreased by {abs(velocity):.1f}% versus the previous 14-day window. "
                f"Monitor whether this cluster is leaving the active trend set."
            )
        alerts.append(
            {
                "id": idx,
                "title": title,
                "message": message,
                "severity": severity,
                "trend": trend["name"],
                "time": f"{latest.date().isoformat()} window",
                "read": idx > 3,
            }
        )

    source_counts = df["category"].value_counts().head(12)
    source_total = max(1, int(len(df)))
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "meta": {
            "generated_at": generated_at,
            "dataset": "News_Category_Dataset_v3.json",
            "latest_article_date": latest.date().isoformat(),
            "sampled_recent_articles": len(recent),
            "pipeline": ["TF-IDF", "LDA", "SentenceTransformer(all-MiniLM-L6-v2)", "KMeans"],
        },
        "stats": {
            "total_articles": int(len(df)),
            "total_articles_label": compact_number(int(len(df))),
            "weekly_change_pct": change,
            "last_week_articles": last_week_count,
            "previous_week_articles": previous_week_count,
            "active_sources": int(df["category"].nunique()),
        },
        "activity": activity,
        "trends": trends,
        "predictions": predictions,
        "events": events,
        "alerts": alerts,
        "sources": [
            {
                "name": str(category).title(),
                "articlesTotal": int(count),
                "status": "active",
                "type": "dataset",
                "lastFetch": latest.date().isoformat(),
                "coveragePct": round((int(count) / source_total) * 100.0, 1),
            }
            for category, count in source_counts.items()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = read_jsonl(args.input)
    dashboard = build_dashboard(df, sample_size=args.sample_size, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
