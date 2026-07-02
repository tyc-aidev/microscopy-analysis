"""Tests for Sprint 1 structured experiment logging (#13)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from microscopy_analysis.train import trainer as trainer_mod
from microscopy_analysis.train.config import TrainConfig
from microscopy_analysis.train.logging import NoOpLogger, TensorBoardLogger, build_logger
from microscopy_analysis.train.trainer import run_training


class _TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return torch.softmax(y, dim=1) if self.num_classes > 1 else torch.sigmoid(y)


class _RecordingLogger:
    def __init__(self) -> None:
        self.params: dict = {}
        self.epochs: list[dict] = []
        self.summary: dict = {}

    def log_params(self, params: dict) -> None:
        self.params = params

    def log_epoch(self, record: dict) -> None:
        self.epochs.append(record)

    def finish(self, summary: dict) -> None:
        self.summary = summary


def _make_super_split(root: Path, split: str, n: int, size: int = 24) -> None:
    image_dir, annot_dir = root / "Super1" / split, root / "Super1" / f"{split}_annot"
    image_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size, 3), dtype=np.uint8)
        image[: size // 2], mask[: size // 2] = (255, 0, 0), (255, 0, 0)
        image[size // 2 :], mask[size // 2 :] = (0, 0, 255), (0, 0, 255)
        Image.fromarray(image, mode="RGB").save(image_dir / f"s{i}.tif")
        Image.fromarray(mask, mode="RGB").save(annot_dir / f"s{i}_mask.tif")


def _tiny_config(tmp_path: Path, **overrides) -> TrainConfig:
    base = dict(
        run_name="log_run",
        data_root=tmp_path,
        dataset_name="Super1",
        dataset_family="super",
        split="train",
        architecture="UnetPlusPlus",
        encoder_name="resnet50",
        pretraining="micronet",
        num_classes=3,
        output_dir=tmp_path / "results" / "log_run",
        lr_phase1=1e-2,
        lr_phase2=1e-3,
        patience=50,
        max_epochs_phase1=2,
        max_epochs_phase2=1,
        batch_size=2,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_default_backend_is_offline_noop() -> None:
    assert isinstance(build_logger("none", "r"), NoOpLogger)
    # NoOp methods accept the full protocol surface without side effects.
    logger = build_logger("none", "r")
    logger.log_params({"a": 1})
    logger.log_epoch({"epoch": 1})
    logger.finish({"best_score": 0.5})


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown logging backend"):
        build_logger("not-a-backend", "r")


def test_tensorboard_logger_writes_scalars(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    logger = build_logger("tensorboard", "tb_run", log_dir=tmp_path / "results" / "tb_run")
    assert isinstance(logger, TensorBoardLogger)
    logger.log_params({"run_name": "tb_run", "lr_phase1": 2e-4})
    logger.log_epoch(
        {
            "epoch": 1,
            "phase": 1,
            "lr": 2e-4,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "val_mean_iou": 0.3,
            "val_score": 0.3,
            "val_iou_per_class": [0.2, 0.3, 0.4],
        }
    )
    logger.finish({"best_score": 0.3, "epochs_trained": 1})
    assert (tmp_path / "results" / "tb_run" / "tensorboard").is_dir()


def test_run_training_drives_logger(tmp_path: Path, monkeypatch) -> None:
    _make_super_split(tmp_path, "train", n=4)
    _make_super_split(tmp_path, "val", n=2)
    cfg = _tiny_config(tmp_path)

    recorder = _RecordingLogger()
    monkeypatch.setattr(trainer_mod, "build_logger", lambda *a, **k: recorder)
    monkeypatch.setattr(
        trainer_mod, "create_segmentation_model", lambda *a, **k: _TinyModel(num_classes=cfg.num_classes)
    )
    result = run_training(cfg, device_preference="cpu")

    # Params carry the resolved config plus run metadata (git SHA + device).
    assert "git_sha" in recorder.params and recorder.params["run_name"] == "log_run"
    # Per-epoch records match what was written to metrics.json.
    written = json.loads(Path(result.metrics_path).read_text())
    assert len(recorder.epochs) == len(written) == result.epochs_trained
    # Final summary is the run summary (best score reported).
    assert recorder.summary["best_score"] == result.best_score
