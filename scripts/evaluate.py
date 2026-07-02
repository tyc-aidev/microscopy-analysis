#!/usr/bin/env python3
"""Evaluate a trained segmentation run on a held-out split (Sprint 2).

Loads ``results/<run_name>/model_best.pth`` and computes per-class + mean IoU on
the test split, writing ``results/<run_name>/eval_<split>.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from microscopy_analysis.eval.evaluate import evaluate_run
from microscopy_analysis.train.config import load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained run on a held-out split")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML used for training")
    parser.add_argument("--split", default="test", help="Dataset split to score (default: test)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model weights (default: results/<run_name>/model_best.pth)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config)
    result = evaluate_run(
        cfg, split=args.split, checkpoint=args.checkpoint, device_preference=args.device
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
