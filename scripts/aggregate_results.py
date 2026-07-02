#!/usr/bin/env python3
"""Aggregate Sprint 2 eval JSONs into the MicroNet-vs-ImageNet benchmark table.

Scans ``results/`` for ``eval_<split>.json`` sidecars, pivots them per
(dataset, encoder), writes ``results/benchmark_matrix.csv``, and prints a
markdown table plus the majority-win summary (the Sprint 2 exit criterion).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from microscopy_analysis.orchestration.aggregate import (
    build_comparison,
    load_eval_scores,
    load_paper_targets,
    majority_summary,
    render_markdown,
    write_comparison_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Sprint 2 benchmark eval results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-csv", type=Path, default=Path("results/benchmark_matrix.csv"))
    parser.add_argument(
        "--target-metrics",
        type=Path,
        default=Path("paper/target_metrics.csv"),
        help="Paper IoU targets to compare against (blank rows are ignored)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = load_eval_scores(args.results_dir, split=args.split)
    if not scores:
        print(f"No eval_{args.split}.json files found under {args.results_dir}.")
        return 1
    targets = load_paper_targets(args.target_metrics) if args.target_metrics.exists() else {}
    rows = build_comparison(scores, targets)
    summary = majority_summary(rows)
    csv_path = write_comparison_csv(rows, args.out_csv)
    print(render_markdown(rows, summary))
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
