#!/usr/bin/env python3
"""Save qualitative prediction panels for a trained segmentation run.

Each output PNG is a 4-column figure: input | GT overlay | prediction overlay | errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from microscopy_analysis.eval.predictions import run_prediction_panels
from microscopy_analysis.train.config import load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions for a trained run")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML used for training")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model weights (default: results/<run_name>/model_best.pth)",
    )
    parser.add_argument("--split", default="val", help="Dataset split to visualize (train|val|test)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG panels (default: results/<run_name>/predictions/<split>)",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Cap number of panels written")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config)

    checkpoint = args.checkpoint or (cfg.output_dir / "model_best.pth")
    out_dir = args.output_dir or (cfg.output_dir / "predictions" / args.split)

    paths = run_prediction_panels(
        cfg,
        checkpoint=checkpoint,
        split=args.split,
        output_dir=out_dir,
        device=args.device,
        max_images=args.max_images,
    )
    for path in paths:
        print(f"Wrote {path}")
    print(f"Done. {len(paths)} panel(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
