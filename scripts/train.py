#!/usr/bin/env python3
"""Config-driven training entrypoint for Sprint 1.

Optional overrides (epoch caps / batch size / device) let the same baseline
config drive a quick local MPS smoke run without editing the YAML.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import asdict
from pathlib import Path

from microscopy_analysis.train.config import load_train_config
from microscopy_analysis.train.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run microscopy-analysis segmentation training")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--batch-size", type=int, default=None, help="Override trainer batch size")
    parser.add_argument("--max-epochs-phase1", type=int, default=None, help="Override phase-1 epoch cap")
    parser.add_argument("--max-epochs-phase2", type=int, default=None, help="Override phase-2 epoch cap")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from results/<run>/checkpoint.pth if present")
    parser.add_argument(
        "--log-backend", default=None, choices=("none", "wandb", "mlflow"), help="Override experiment logging backend"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config)

    overrides = {
        "batch_size": args.batch_size,
        "max_epochs_phase1": args.max_epochs_phase1,
        "max_epochs_phase2": args.max_epochs_phase2,
        "patience": args.patience,
        "log_backend": args.log_backend,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.resume:
        overrides["resume"] = True
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    result = run_training(cfg, device_preference=args.device)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
