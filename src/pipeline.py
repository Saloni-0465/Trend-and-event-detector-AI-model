"""
Entry point for experiments and ablation runs.

Usage (from repository root):
    python -m src.pipeline --help
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend & event detector — pipeline placeholder"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "advanced", "dl", "hybrid", "ablation"],
        default="baseline",
        help="Which stack to run (stubs until implemented).",
    )
    args = parser.parse_args()
    print(f"Mode: {args.mode} — implement data load, train, and evaluate here.")


if __name__ == "__main__":
    main()
