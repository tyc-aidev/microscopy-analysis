"""Tests for Sprint 1 config and train pipeline scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from amat.train.config import load_train_config
from amat.train.trainer import run_training


def _make_super_train_data(root: Path) -> None:
    dataset_dir = root / "Super1"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "train_annot").mkdir(parents=True)
    Image.new("RGB", (4, 4), (128, 128, 128)).save(dataset_dir / "train" / "sample.tif")
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(
        dataset_dir / "train_annot" / "sample_mask.tif"
    )


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
            ]
        )
    )
    cfg = load_train_config(config_path, base_dir=tmp_path)
    assert cfg.run_name == "test_run"
    assert cfg.dataset_name == "Super1"
    assert cfg.pretraining == "micronet"
    assert cfg.lr_phase2 == 1e-5
    assert cfg.data_root.is_absolute()
    assert cfg.output_dir.is_absolute()
    # Relative output_dir resolves against base_dir, not the config's directory.
    assert cfg.output_dir == (tmp_path / "results" / "test_run").resolve()


def test_run_training_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    _make_super_train_data(tmp_path)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: test_run",
                f"data_root: {tmp_path.as_posix()}",
                f"output_dir: {(tmp_path / 'results').as_posix()}",
                "dataset:",
                "  name: Super1",
                "  family: super",
                "  split: train",
                "model:",
                "  architecture: UnetPlusPlus",
                "  encoder_name: resnet50",
                "  pretraining: micronet",
                "  num_classes: 3",
            ]
        )
    )
    cfg = load_train_config(config_path)

    monkeypatch.setattr(
        "amat.train.trainer.create_segmentation_model", lambda *args, **kwargs: object()
    )
    result = run_training(cfg)

    metrics = json.loads(Path(result.metrics_path).read_text())
    assert metrics["dataset"]["num_samples"] == 1
    assert metrics["model"]["pretraining"] == "micronet"
    assert Path(result.checkpoint_path).exists()

