"""Minimal two-phase trainer with JSON outputs."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from microscopy_analysis.data.dataset_adapter import list_sample_pairs
from microscopy_analysis.models import create_segmentation_model
from microscopy_analysis.train.config import TrainConfig


@dataclass(frozen=True)
class TrainResult:
    run_name: str
    dataset_name: str
    dataset_family: str
    num_samples: int
    phase1_lr: float
    phase2_lr: float
    checkpoint_path: str
    metrics_path: str
    git_sha: str
    created_at_utc: str


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
    return out if out else "unknown"


def run_training(config: TrainConfig) -> TrainResult:
    random.seed(config.seed)
    np.random.seed(config.seed)

    dataset_root = config.data_root / config.dataset_name
    pairs = list_sample_pairs(dataset_root, split=config.split, dataset_family=config.dataset_family)
    if not pairs:
        raise RuntimeError(
            f"No training pairs found in {dataset_root} split={config.split} family={config.dataset_family}"
        )

    model_cfg = {
        "architecture": config.architecture,
        "encoder_name": config.encoder_name,
        "pretraining": config.pretraining,
        "num_classes": config.num_classes,
    }
    # Instantiate early so missing deps/config fail before writing outputs.
    _ = create_segmentation_model(
        config.architecture, config.encoder_name, config.pretraining, config.num_classes
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Scaffold placeholder; named to avoid masquerading as a real torch checkpoint.
    checkpoint_path = config.output_dir / "checkpoint.placeholder"
    metrics_path = config.output_dir / "metrics.json"

    metrics_payload = {
        "run_name": config.run_name,
        "dataset": {
            "name": config.dataset_name,
            "family": config.dataset_family,
            "split": config.split,
            "num_samples": len(pairs),
        },
        "model": model_cfg,
        "trainer": {
            "patience": config.patience,
            "max_epochs_phase1": config.max_epochs_phase1,
            "max_epochs_phase2": config.max_epochs_phase2,
            "lr_phase1": config.lr_phase1,
            "lr_phase2": config.lr_phase2,
        },
        "status": "initialized",
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    checkpoint_path.write_text("placeholder checkpoint for sprint-1 scaffolding\n")

    result = TrainResult(
        run_name=config.run_name,
        dataset_name=config.dataset_name,
        dataset_family=config.dataset_family,
        num_samples=len(pairs),
        phase1_lr=config.lr_phase1,
        phase2_lr=config.lr_phase2,
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        git_sha=_git_sha(),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    (config.output_dir / "run_summary.json").write_text(json.dumps(asdict(result), indent=2))
    return result

