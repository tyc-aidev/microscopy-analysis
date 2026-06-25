#!/usr/bin/env python3
"""Config-driven training entrypoint for Sprint 1."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from amat.train.config import load_train_config
from amat.train.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMAT segmentation training")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config)
    result = run_training(cfg)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

