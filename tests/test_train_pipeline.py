"""Tests for Sprint 1 config loading and the real two-phase trainer.

The trainer is exercised on a tiny synthetic dataset with a tiny real
``nn.Module`` (1x1 conv) swapped in for the heavy encoder model, so it actually
trains a few steps fast on CPU. A real-encoder MPS run is covered separately as
a slow/integration test.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from microscopy_analysis.train.config import TrainConfig, load_train_config
from microscopy_analysis.train.trainer import run_training


class _TinyModel(torch.nn.Module):
    """Pixelwise 1x1 conv with the activation the factory models carry."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return torch.softmax(y, dim=1) if self.num_classes > 1 else torch.sigmoid(y)


def _make_super_split(root: Path, split: str, n: int, size: int = 24) -> None:
    image_dir, annot_dir = root / "Super1" / split, root / "Super1" / f"{split}_annot"
    image_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # Color-separable image so a 1x1 conv can learn the mapping and loss drops.
        image = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size, 3), dtype=np.uint8)
        image[: size // 2], mask[: size // 2] = (255, 0, 0), (255, 0, 0)  # class 1
        image[size // 2 :], mask[size // 2 :] = (0, 0, 255), (0, 0, 255)  # class 2
        Image.fromarray(image, mode="RGB").save(image_dir / f"s{i}.tif")
        Image.fromarray(mask, mode="RGB").save(annot_dir / f"s{i}_mask.tif")


def _tiny_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        run_name="tiny_run",
        data_root=tmp_path,
        dataset_name="Super1",
        dataset_family="super",
        split="train",
        architecture="UnetPlusPlus",
        encoder_name="resnet50",
        pretraining="micronet",
        num_classes=3,
        output_dir=tmp_path / "results" / "tiny_run",
        lr_phase1=1e-2,
        lr_phase2=1e-3,
        patience=50,
        max_epochs_phase1=8,
        max_epochs_phase2=2,
        batch_size=2,
    )


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiments"
BASELINE_CONFIGS = sorted(CONFIG_DIR.glob("*_baseline.yaml"))
SE_RESNET50_CONFIGS = sorted(CONFIG_DIR.glob("*_se_resnet50.yaml"))


@pytest.mark.parametrize("config_path", BASELINE_CONFIGS, ids=lambda p: p.stem)
def test_baseline_experiment_configs_load(config_path: Path) -> None:
    expected_name = yaml.safe_load(config_path.read_text())["run_name"]
    cfg = load_train_config(config_path)
    assert cfg.run_name == expected_name
    assert cfg.dataset_family in ("super", "ebc")
    assert cfg.encoder_name == "senet154"
    if cfg.dataset_family == "super":
        assert cfg.num_classes == 3
    else:
        assert cfg.num_classes == 1
    assert cfg.pretraining == "micronet"


@pytest.mark.parametrize("config_path", SE_RESNET50_CONFIGS, ids=lambda p: p.stem)
def test_se_resnet50_experiment_configs_load(config_path: Path) -> None:
    expected_name = yaml.safe_load(config_path.read_text())["run_name"]
    cfg = load_train_config(config_path)
    assert cfg.run_name == expected_name
    assert cfg.dataset_family in ("super", "ebc")
    assert cfg.encoder_name == "se_resnet50"
    if cfg.dataset_family == "super":
        assert cfg.num_classes == 3
    else:
        assert cfg.num_classes == 1
    assert cfg.pretraining == "micronet"


def test_load_train_config(tmp_path: Path) -> None:
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: test_run",
                f"data_root: {tmp_path.as_posix()}",
                "output_dir: results",
                "dataset:",
                "  name: Super1",
                "  family: super",
                "model:",
                "  architecture: UnetPlusPlus",
                "  encoder_name: resnet50",
                "  pretraining: micronet",
                "  num_classes: 3",
                "trainer:",
                "  batch_size: 4",
            ]
        )
    )
    cfg = load_train_config(config_path, base_dir=tmp_path)
    assert cfg.run_name == "test_run"
    assert cfg.dataset_name == "Super1"
    assert cfg.pretraining == "micronet"
    assert cfg.lr_phase2 == 1e-5
    assert cfg.batch_size == 4  # overridden
    assert cfg.val_split == "val"  # default
    assert cfg.data_root.is_absolute()
    assert cfg.output_dir == (tmp_path / "results" / "test_run").resolve()


def test_run_training_trains_and_writes_real_outputs(tmp_path: Path, monkeypatch) -> None:
    _make_super_split(tmp_path, "train", n=4)
    _make_super_split(tmp_path, "val", n=2)
    cfg = _tiny_config(tmp_path)

    monkeypatch.setattr(
        "microscopy_analysis.train.trainer.create_segmentation_model",
        lambda *a, **k: _TinyModel(num_classes=cfg.num_classes),
    )
    result = run_training(cfg, device_preference="cpu")

    # Real torch checkpoint that round-trips.
    assert Path(result.checkpoint_path).suffix == ".pth"
    ckpt = torch.load(result.checkpoint_path, map_location="cpu")
    assert "model_state" in ckpt and "optimizer_state" in ckpt

    # Per-epoch metrics with loss + IoU; loss decreases and IoU is computed.
    records = json.loads(Path(result.metrics_path).read_text())
    assert len(records) == result.epochs_trained >= cfg.max_epochs_phase1
    # Val loss is the smoothed learning signal (train loss is noisy under per-batch aug).
    phase1 = [r for r in records if r["phase"] == 1]
    assert phase1[-1]["val_loss"] < phase1[0]["val_loss"]
    assert all(0.0 <= record["val_mean_iou"] <= 1.0 for record in records)
    assert len(records[-1]["val_iou_per_class"]) == cfg.num_classes

    # Run summary reflects the best epoch.
    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["num_samples"] == 4
    assert summary["num_val_samples"] == 2
    assert 0.0 <= result.best_mean_iou <= 1.0
    assert result.best_score > 0.0


def test_train_subsample_limits_training_images(tmp_path: Path, monkeypatch) -> None:
    _make_super_split(tmp_path, "train", n=6)
    _make_super_split(tmp_path, "val", n=2)
    monkeypatch.setattr(
        "microscopy_analysis.train.trainer.create_segmentation_model",
        lambda *a, **k: _TinyModel(num_classes=3),
    )
    cfg = replace(_tiny_config(tmp_path), train_subsample=2)
    result = run_training(cfg, device_preference="cpu")
    # Low-data cap: only 2 of the 6 training images are used; val is untouched.
    assert result.num_samples == 2
    assert result.num_val_samples == 2
    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["num_samples"] == 2


def test_run_training_binary_ebc_path(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "EBC1" / "train"
    annot_dir = tmp_path / "EBC1" / "train_annot"
    image_dir.mkdir(parents=True)
    annot_dir.mkdir(parents=True)
    for i in range(3):
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        image[:12], mask[:12] = 255, 1
        Image.fromarray(image, mode="RGB").save(image_dir / f"t{i}.tif")
        Image.fromarray(mask, mode="L").save(annot_dir / f"t{i}.tif")

    cfg = replace(
        _tiny_config(tmp_path),
        dataset_name="EBC1",
        dataset_family="ebc",
        num_classes=1,
        crop_size=16,
        run_name="tiny_ebc",
        output_dir=tmp_path / "results" / "tiny_ebc",
    )
    monkeypatch.setattr(
        "microscopy_analysis.train.trainer.create_segmentation_model",
        lambda *a, **k: _TinyModel(num_classes=1),
    )
    result = run_training(cfg, device_preference="cpu")
    records = json.loads(Path(result.metrics_path).read_text())
    # Binary path exercises sigmoid + BCE/Dice; trivial data converges fast.
    assert min(r["train_loss"] for r in records) <= records[0]["train_loss"]
    assert result.best_score >= 0.99
    assert len(records[-1]["val_iou_per_class"]) == 2  # [background, foreground]
    assert Path(result.checkpoint_path).exists()


def test_resume_continues_from_latest_checkpoint(tmp_path: Path, monkeypatch) -> None:
    _make_super_split(tmp_path, "train", n=4)
    _make_super_split(tmp_path, "val", n=2)
    monkeypatch.setattr(
        "microscopy_analysis.train.trainer.create_segmentation_model",
        lambda *a, **k: _TinyModel(num_classes=3),
    )

    # First run: 3 phase-1 epochs, no phase 2 — leaves a resumable checkpoint at epoch 3.
    first = replace(_tiny_config(tmp_path), max_epochs_phase1=3, max_epochs_phase2=0)
    r1 = run_training(first, device_preference="cpu")
    assert r1.epochs_trained == 3
    assert Path(r1.checkpoint_path).exists() and Path(r1.best_checkpoint_path).exists()

    # Resume into the same run dir with a higher phase-1 cap: continues from epoch 3.
    second = replace(first, max_epochs_phase1=5, resume=True)
    r2 = run_training(second, device_preference="cpu")
    assert r2.resumed_from_epoch == 3
    assert r2.epochs_trained == 5
    # metrics.json was appended to, not restarted.
    records = json.loads(Path(r2.metrics_path).read_text())
    assert [r["epoch"] for r in records] == [1, 2, 3, 4, 5]


def test_resume_into_phase2_keeps_latest_weights(tmp_path: Path, monkeypatch) -> None:
    _make_super_split(tmp_path, "train", n=4)
    _make_super_split(tmp_path, "val", n=2)
    monkeypatch.setattr(
        "microscopy_analysis.train.trainer.create_segmentation_model",
        lambda *a, **k: _TinyModel(num_classes=3),
    )

    # First run completes phase 1 and one phase-2 epoch, leaving the checkpoint in phase 2.
    first = replace(_tiny_config(tmp_path), max_epochs_phase1=2, max_epochs_phase2=1)
    r1 = run_training(first, device_preference="cpu")
    ckpt1 = torch.load(r1.checkpoint_path, map_location="cpu")
    assert ckpt1["phase"] == 2
    weights_before = {k: v.clone() for k, v in ckpt1["model_state"].items()}

    # Resume into phase 2 with a higher phase-2 cap. The resumed (latest) weights must be
    # the starting point — NOT silently reset to model_best — and training must continue.
    second = replace(first, max_epochs_phase2=3, resume=True)
    r2 = run_training(second, device_preference="cpu")
    assert r2.resumed_from_epoch == 3
    assert r2.epochs_trained == 5  # 2 (phase1) + 1 (prior phase2) + 2 (resumed phase2)
    records = json.loads(Path(r2.metrics_path).read_text())
    assert [r["epoch"] for r in records] == [1, 2, 3, 4, 5]
    assert [r["phase"] for r in records][-1] == 2
    # The first resumed epoch trained from the latest weights, so params moved from there.
    resumed = torch.load(r2.checkpoint_path, map_location="cpu")["model_state"]
    assert any(not torch.equal(resumed[k], weights_before[k]) for k in weights_before)


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 for the real-model run")
def test_run_training_with_real_smp_model(tmp_path: Path) -> None:
    """End-to-end through the real smp UnetPlusPlus (random init, no download)."""
    _make_super_split(tmp_path, "train", n=2, size=64)
    _make_super_split(tmp_path, "val", n=2, size=64)
    cfg = replace(
        _tiny_config(tmp_path),
        pretraining="random",
        max_epochs_phase1=1,
        max_epochs_phase2=1,
        lr_phase1=1e-3,
    )
    result = run_training(cfg, device_preference="auto")
    assert Path(result.checkpoint_path).exists()
    assert result.epochs_trained == 2
    assert len(result.best_per_class_iou) == cfg.num_classes
