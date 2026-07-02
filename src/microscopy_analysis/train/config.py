"""Training config loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    data_root: Path
    dataset_name: str
    dataset_family: str
    split: str
    architecture: str
    encoder_name: str
    pretraining: str
    # Super (multiclass) = 3; EBC binary-oxide = 1 (background implicit, NASA asserts != 2).
    num_classes: int
    output_dir: Path
    seed: int = 42
    lr_phase1: float = 2e-4
    lr_phase2: float = 1e-5
    patience: int = 30
    max_epochs_phase1: int = 120
    max_epochs_phase2: int = 60
    # Sprint 1 real-trainer knobs (defaults keep older configs working unchanged).
    val_split: str = "val"
    batch_size: int = 6
    num_workers: int = 0
    crop_size: int = 512
    loss_weight: float = 0.7
    metric_threshold: float = 0.5
    resume: bool = False
    # Low-data ablation (Sprint 3): cap the training split to this many images
    # (deterministic, seeded). None uses the full split. Val/test are never capped.
    train_subsample: int | None = None
    # Structured logging (#13): none keeps runs offline. Raw augmentation overrides
    # (#12) are kept as a plain dict here so this module stays torch/albumentations-free.
    log_backend: str = "none"
    log_project: str | None = None
    augmentation: dict | None = None


def load_train_config(path: Path, base_dir: Path | None = None) -> TrainConfig:
    """Load a training config.

    Relative ``data_root`` / ``output_dir`` are resolved against ``base_dir``
    (defaults to the current working directory), matching how the repo treats
    ``data/`` and ``results/`` at the project root when running from there.
    """
    raw = yaml.safe_load(path.read_text())
    base = (base_dir or Path.cwd()).resolve()
    trainer = raw.get("trainer", {})
    optimizer = raw.get("optimizer", {})
    dataset = raw["dataset"]
    logging_cfg = raw.get("logging", {})

    data_root = Path(raw["data_root"])
    if not data_root.is_absolute():
        data_root = (base / data_root).resolve()
    output_root = Path(raw.get("output_dir", "results"))
    if not output_root.is_absolute():
        output_root = (base / output_root).resolve()

    return TrainConfig(
        run_name=raw["run_name"],
        data_root=data_root,
        dataset_name=dataset["name"],
        dataset_family=dataset["family"],
        split=dataset.get("split", "train"),
        architecture=raw["model"]["architecture"],
        encoder_name=raw["model"]["encoder_name"],
        pretraining=raw["model"]["pretraining"],
        num_classes=int(raw["model"]["num_classes"]),
        output_dir=output_root / raw["run_name"],
        seed=int(raw.get("seed", 42)),
        lr_phase1=float(optimizer.get("lr_phase1", 2e-4)),
        lr_phase2=float(optimizer.get("lr_phase2", 1e-5)),
        patience=int(trainer.get("patience", 30)),
        max_epochs_phase1=int(trainer.get("max_epochs_phase1", 120)),
        max_epochs_phase2=int(trainer.get("max_epochs_phase2", 60)),
        val_split=dataset.get("val_split", "val"),
        batch_size=int(trainer.get("batch_size", 6)),
        num_workers=int(trainer.get("num_workers", 0)),
        crop_size=int(trainer.get("crop_size", 512)),
        loss_weight=float(trainer.get("loss_weight", 0.7)),
        metric_threshold=float(trainer.get("metric_threshold", 0.5)),
        resume=bool(trainer.get("resume", False)),
        train_subsample=(
            int(trainer["train_subsample"]) if trainer.get("train_subsample") is not None else None
        ),
        log_backend=str(logging_cfg.get("backend", "none")),
        log_project=logging_cfg.get("project"),
        augmentation=raw.get("augmentation"),
    )

