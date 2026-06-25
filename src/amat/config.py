"""Experiment configuration: typed dataclasses loaded from YAML.

Sprint 0 only needs enough config to drive the smoke test (which dataset, which
encoder/architecture/pretraining, the reference hyperparameters). Sprint 1
extends this into the full training pipeline, but the schema is kept stable here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models.weights import PAPER_VERSION, normalize_version

_FAMILIES = ("super", "ebc")
_ARCHITECTURES = ("Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3Plus", "FPN", "PSPNet", "Linknet", "PAN")
_PRETRAINING = ("random", "imagenet", "micronet", "image-micronet")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    family: str
    classes: int
    class_values: dict[str, list[int]]

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError(f"dataset.family must be one of {_FAMILIES}, got {self.family!r}")
        if self.classes == 2:
            raise ValueError("Binary tasks use classes=1 (background implicit); classes=2 is invalid.")
        if self.classes < 1:
            raise ValueError("dataset.classes must be >= 1")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    encoder: str
    pretraining: str
    micronet_version: str = PAPER_VERSION

    def __post_init__(self) -> None:
        if self.architecture not in _ARCHITECTURES:
            raise ValueError(f"model.architecture must be one of {_ARCHITECTURES}, got {self.architecture!r}")
        if self.pretraining not in _PRETRAINING:
            raise ValueError(f"model.pretraining must be one of {_PRETRAINING}, got {self.pretraining!r}")
        object.__setattr__(self, "micronet_version", normalize_version(self.micronet_version))


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 2e-4
    phase2_lr: float = 1e-5
    patience: int = 30
    batch_size: int = 6
    loss_weight: float = 0.7
    metric_threshold: float = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate an experiment YAML into an :class:`ExperimentConfig`."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    try:
        return ExperimentConfig(
            name=raw["name"],
            dataset=DatasetConfig(**raw["dataset"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw.get("training", {})),
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from None
