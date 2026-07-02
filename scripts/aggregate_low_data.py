#!/usr/bin/env python3
"""Aggregate Sprint 3 low-data eval JSONs into curves + a relative-error table.

Scans ``results/`` for ``eval_<split>.json`` sidecars tagged with a training-set
size, pivots them per (dataset, #train-images), writes
``results/low_data_curves.csv``, prints the MicroNet-vs-ImageNet relative
IoU-error-reduction table, and (``--plot``) renders the IoU-vs-#images figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from microscopy_analysis.orchestration.low_data_analysis import (
    build_curves,
    build_low_data_rows,
    load_low_data_scores,
    plot_curves,
    render_markdown,
    summarize,
    write_low_data_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Sprint 3 low-data ablation results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-csv", type=Path, default=Path("results/low_data_curves.csv"))
    parser.add_argument("--plot", type=Path, nargs="?", const=Path("results/low_data_curves.png"),
                        default=None, help="Also render the IoU-vs-#images figure (optional path)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    points = load_low_data_scores(args.results_dir, split=args.split)
    if not points:
        print(f"No low-data eval_{args.split}.json files found under {args.results_dir}.")
        return 1

    curves = build_curves(points)
    rows = build_low_data_rows(points)
    summary = summarize(rows, curves)
    csv_path = write_low_data_csv(rows, args.out_csv)
    print(render_markdown(rows, summary))
    print(f"\nWrote {csv_path}")
    if args.plot is not None:
        fig_path = plot_curves(curves, args.plot)
        print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
