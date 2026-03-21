"""
Part 3 — Deep learning track: frozen Transformer sentence embeddings + K-Means.

- **Embeddings:** `sentence-transformers` (MiniLM-class encoder); weights frozen → strong
  regularization vs fitting noise on small streams.
- **Clustering:** K chosen on the **training time slice** by maximizing **silhouette**
  (complexity control without looking at the held-out tail).
- **Metrics:** silhouette (train/test), optional **NMI/ARI** vs `theme` on synthetic data;
  **temporal stability:** JSD between adjacent time-bin cluster distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.temporal_windows import add_time_bin
from src.deep_learning.embedding_cluster import cluster_embeddings_time_split, embed_stream_texts
from src.evaluation.clustering_metrics import (
    clustering_supervised_scores,
    encode_categorical_labels,
    silhouette_safe,
)
from src.evaluation.topic_temporal_metrics import mean_adjacent_jsd
from src.pipeline_common import Source, load_raw_stream, time_ordered_split


def _cluster_mixture_matrix(
    df: pd.DataFrame,
    labels: np.ndarray,
    *,
    bin_col: str = "time_bin",
    n_clusters: int,
) -> tuple[list, np.ndarray]:
    sub = df[[bin_col]].copy()
    sub["cluster"] = labels.astype(int)
    raw_bins = sub[bin_col].unique()
    keys = sorted(raw_bins, key=lambda t: pd.Timestamp(t))
    mat = np.zeros((len(keys), n_clusters), dtype=np.float64)
    for i, b in enumerate(keys):
        m = sub[bin_col] == b
        if not m.any():
            continue
        hist = np.bincount(sub.loc[m, "cluster"].to_numpy(), minlength=n_clusters)
        mat[i] = hist / max(int(m.sum()), 1)
    return keys, mat


def run_part3(
    *,
    source: Source = "synthetic",
    csv_path: str | Path | None = None,
    text_col: str = "text",
    time_col: str = "timestamp",
    window_freq: str = "7D",
    train_frac: float = 0.75,
    seed: int = 42,
    n_docs: int = 800,
    start: str = "2024-01-01",
    end: str = "2024-03-01",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str | None = None,
    k_min: int = 2,
    k_max: int = 12,
    max_iter: int = 300,
    output_dir: str | Path | None = None,
    show_embedding_progress: bool = False,
) -> dict[str, Any]:
    df = load_raw_stream(
        source=source,
        csv_path=csv_path,
        text_col=text_col,
        time_col=time_col,
        seed=seed,
        n_docs=n_docs,
        start=start,
        end=end,
    )
    df = df.sort_values(time_col).reset_index(drop=True)
    df = add_time_bin(df, time_col=time_col, freq=window_freq, out_col="time_bin")

    n = len(df)
    train_idx, test_idx = time_ordered_split(n, train_frac)

    X = embed_stream_texts(
        df,
        text_col,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=True,
        show_progress=show_embedding_progress,
    )

    labels_all, km, best_k, sil_by_k = cluster_embeddings_time_split(
        X,
        train_idx,
        k_min=k_min,
        k_max=k_max,
        random_state=seed,
        max_iter=max_iter,
    )

    lab_train = labels_all[train_idx]
    lab_test = labels_all[test_idx]

    sil_train = silhouette_safe(X[train_idx], lab_train)
    sil_test = silhouette_safe(X[test_idx], lab_test)

    sup_train = {"nmi": float("nan"), "ari": float("nan")}
    sup_test = {"nmi": float("nan"), "ari": float("nan")}
    if "theme" in df.columns:
        y = encode_categorical_labels(df["theme"])
        sup_train = clustering_supervised_scores(y, labels_all, subset=train_idx)
        sup_test = clustering_supervised_scores(y, labels_all, subset=test_idx)

    bin_keys, mix = _cluster_mixture_matrix(
        df, labels_all, n_clusters=int(km.n_clusters)
    )
    jsd_bins = mean_adjacent_jsd(bin_keys, mix)

    k_curve_rows = [{"k": k, "silhouette_train": v} for k, v in sorted(sil_by_k.items())]
    k_curve_df = pd.DataFrame(k_curve_rows)

    split_mask = np.zeros(n, dtype=bool)
    split_mask[train_idx] = True
    assign_df = pd.DataFrame(
        {
            "timestamp": df[time_col],
            "time_bin": df["time_bin"],
            "cluster": labels_all.astype(int),
            "split": np.where(split_mask, "train", "test"),
        }
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "best_k_silhouette_train",
                "value": float(best_k),
                "train_frac": train_frac,
                "model_name": model_name,
            },
            {
                "metric": "silhouette_train",
                "value": sil_train,
                "train_frac": train_frac,
                "model_name": model_name,
            },
            {
                "metric": "silhouette_test",
                "value": sil_test,
                "train_frac": train_frac,
                "model_name": model_name,
            },
            {
                "metric": "nmi_train_vs_theme",
                "value": sup_train["nmi"],
                "train_frac": train_frac,
                "model_name": model_name,
            },
            {
                "metric": "nmi_test_vs_theme",
                "value": sup_test["nmi"],
                "train_frac": train_frac,
                "model_name": model_name,
            },
            {
                "metric": "ari_train_vs_theme",
                "value": sup_train["ari"],
                "train_frac": train_frac,
                "model_name": model_name,
            },
            {
                "metric": "ari_test_vs_theme",
                "value": sup_test["ari"],
                "train_frac": train_frac,
                "model_name": model_name,
            },
            {
                "metric": "mean_adjacent_jsd_cluster_mix",
                "value": jsd_bins,
                "train_frac": train_frac,
                "model_name": model_name,
            },
        ]
    )

    out: dict[str, Any] = {
        "df": df,
        "embeddings": X,
        "kmeans": km,
        "labels": labels_all,
        "best_k": best_k,
        "silhouette_by_k": sil_by_k,
        "k_curve": k_curve_df,
        "assignments": assign_df,
        "summary_metrics": summary,
    }

    if output_dir is not None:
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)
        assign_df.to_csv(od / "part3_cluster_assignments.csv", index=False)
        k_curve_df.to_csv(od / "part3_k_silhouette_curve.csv", index=False)
        summary.to_csv(od / "part3_summary_metrics.csv", index=False)

    return out
