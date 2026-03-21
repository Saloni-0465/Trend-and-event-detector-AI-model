"""
Entry point for experiments and ablation runs.

Usage (from repository root):
    python -m src.pipeline part1 --source synthetic
    python -m src.pipeline part2 --source synthetic --num-topics 8
"""

from __future__ import annotations

import argparse
import sys

from src.part1_runner import run_part1
from src.part2_runner import run_part2


def _cmd_part1(ns: argparse.Namespace) -> int:
    out_dir = ns.output_dir if ns.save else None
    result = run_part1(
        source=ns.source,
        csv_path=ns.csv_path,
        text_col=ns.text_col,
        time_col=ns.time_col,
        window_freq=ns.window,
        top_k=ns.top_k,
        seed=ns.seed,
        n_docs=ns.n_docs,
        start=ns.start,
        end=ns.end,
        output_dir=out_dir,
    )
    sm = result["summary_metrics"]
    print(sm.to_string(index=False))
    if ns.save:
        print(f"\nWrote CSVs under {ns.output_dir}")
    return 0


def _cmd_part2(ns: argparse.Namespace) -> int:
    out_dir = ns.output_dir if ns.save else None
    result = run_part2(
        source=ns.source,
        csv_path=ns.csv_path,
        text_col=ns.text_col,
        time_col=ns.time_col,
        window_freq=ns.window,
        num_topics=ns.num_topics,
        passes=ns.passes,
        train_frac=ns.train_frac,
        seed=ns.seed,
        n_docs=ns.n_docs,
        start=ns.start,
        end=ns.end,
        coherence=ns.coherence,
        no_below=ns.no_below,
        no_above=ns.no_above,
        output_dir=out_dir,
    )
    sm = result["summary_metrics"]
    print(sm[["metric", "value"]].to_string(index=False))
    print("\nTop-level LDA topics (preview):")
    print(result["topics"].head(min(5, len(result["topics"]))).to_string(index=False))
    if ns.save:
        print(f"\nWrote CSVs under {ns.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Dynamic trend & event detector — project pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("part1", help="Part 1: data + temporal baselines (frequency, TF-IDF)")
    p1.add_argument(
        "--source",
        choices=["synthetic", "csv"],
        default="synthetic",
        help="synthetic: reproducible generator; csv: your own stream",
    )
    p1.add_argument("--csv-path", type=str, default=None, help="Required when --source csv")
    p1.add_argument("--text-col", type=str, default="text")
    p1.add_argument("--time-col", type=str, default="timestamp")
    p1.add_argument(
        "--window",
        type=str,
        default="7D",
        help="Pandas offset alias for binning, e.g. 1D, 12H, 7D",
    )
    p1.add_argument("--top-k", type=int, default=15)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--n-docs", type=int, default=800)
    p1.add_argument("--start", type=str, default="2024-01-01")
    p1.add_argument("--end", type=str, default="2024-03-01")
    p1.add_argument(
        "--save",
        action="store_true",
        help="Write part1_window_terms.csv and part1_summary_metrics.csv",
    )
    p1.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory for Part 1 CSV outputs when --save is set",
    )
    p1.set_defaults(func=_cmd_part1)

    p2 = sub.add_parser("part2", help="Part 2: LDA (gensim) + perplexity, coherence, temporal topic mix")
    p2.add_argument("--source", choices=["synthetic", "csv"], default="synthetic")
    p2.add_argument("--csv-path", type=str, default=None)
    p2.add_argument("--text-col", type=str, default="text")
    p2.add_argument("--time-col", type=str, default="timestamp")
    p2.add_argument("--window", type=str, default="7D")
    p2.add_argument("--num-topics", type=int, default=8)
    p2.add_argument("--passes", type=int, default=15)
    p2.add_argument("--train-frac", type=float, default=0.75)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--n-docs", type=int, default=800)
    p2.add_argument("--start", type=str, default="2024-01-01")
    p2.add_argument("--end", type=str, default="2024-03-01")
    p2.add_argument("--coherence", choices=["u_mass", "c_v"], default="u_mass")
    p2.add_argument(
        "--no-below",
        type=int,
        default=1,
        help="gensim Dictionary: min document frequency (use 1 for tiny corpora)",
    )
    p2.add_argument("--no-above", type=float, default=0.85)
    p2.add_argument("--save", action="store_true")
    p2.add_argument("--output-dir", type=str, default="data/processed")
    p2.set_defaults(func=_cmd_part2)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
