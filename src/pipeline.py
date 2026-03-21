"""
Entry point for experiments and ablation runs.

Usage (from repository root):
    python -m src.pipeline part1 --source synthetic
    python -m src.pipeline part1 --source csv --csv-path data/sample/stream_sample.csv
"""

from __future__ import annotations

import argparse
import sys

from src.part1_runner import run_part1


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

    ns = parser.parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())
