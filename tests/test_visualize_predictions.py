"""Tests for scripts/visualize_predictions.py."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


class _TinyModel(torch.nn.Module):
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
        image = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size, 3), dtype=np.uint8)
        image[: size // 2], mask[: size // 2] = (255, 0, 0), (255, 0, 0)
        image[size // 2 :], mask[size // 2 :] = (0, 0, 255), (0, 0, 255)
        Image.fromarray(image, mode="RGB").save(image_dir / f"s{i}.tif")
        Image.fromarray(mask, mode="RGB").save(annot_dir / f"s{i}_mask.tif")


def _make_ebc_split(root: Path, split: str, n: int, size: int = 768) -> None:
    image_dir, annot_dir = root / "EBC1" / split, root / "EBC1" / f"{split}_annot"
    image_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        image[: size // 2] = 255
        mask[: size // 2] = 1
        Image.fromarray(image, mode="RGB").save(image_dir / f"e{i}.tif")
        Image.fromarray(mask, mode="L").save(annot_dir / f"e{i}.tif")


def _load_viz_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "visualize_predictions.py"
    spec = importlib.util.spec_from_file_location("visualize_predictions", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_visualize_predictions_script(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("segmentation_models_pytorch")
    data_root = tmp_path / "data"
    _make_super_split(data_root, "val", n=1)

    run_dir = tmp_path / "results" / "viz_run"
    run_dir.mkdir(parents=True)
    model = _TinyModel(num_classes=3)
    torch.save({"model_state": model.state_dict(), "score": 0.0}, run_dir / "model_best.pth")

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: viz_run",
                f"data_root: {data_root}",
                f"output_dir: {tmp_path / 'results'}",
                "seed: 0",
                "dataset:",
                "  name: Super1",
                "  family: super",
                "  split: train",
                "model:",
                "  architecture: UnetPlusPlus",
                "  encoder_name: resnet50",
                "  pretraining: random",
                "  num_classes: 3",
            ]
        )
    )

    viz_mod = _load_viz_module()
    from microscopy_analysis.eval import predictions as pred_mod

    monkeypatch.setattr(pred_mod, "create_segmentation_model", lambda *a, **k: _TinyModel(num_classes=3))
    monkeypatch.setattr(
        viz_mod,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "config": config_path,
                "checkpoint": run_dir / "model_best.pth",
                "split": "val",
                "device": "cpu",
                "output_dir": run_dir / "predictions" / "val",
                "max_images": 1,
            },
        )(),
    )

    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert viz_mod.main() == 0
    finally:
        os.chdir(old_cwd)

    panels = list((run_dir / "predictions" / "val").glob("*_panel.png"))
    assert len(panels) == 1


def test_run_prediction_panels_ebc_crop_path(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("segmentation_models_pytorch")
    data_root = tmp_path / "data"
    _make_ebc_split(data_root, "val", n=1)

    run_dir = tmp_path / "results" / "viz_ebc"
    run_dir.mkdir(parents=True)
    model = _TinyModel(num_classes=1)
    torch.save({"model_state": model.state_dict(), "score": 0.0}, run_dir / "model_best.pth")

    config_path = tmp_path / "cfg_ebc.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: viz_ebc",
                f"data_root: {data_root}",
                f"output_dir: {tmp_path / 'results'}",
                "seed: 0",
                "dataset:",
                "  name: EBC1",
                "  family: ebc",
                "  split: train",
                "model:",
                "  architecture: UnetPlusPlus",
                "  encoder_name: senet154",
                "  pretraining: random",
                "  num_classes: 1",
            ]
        )
    )

    from microscopy_analysis.eval import predictions as pred_mod
    from microscopy_analysis.train.config import load_train_config

    monkeypatch.setattr(pred_mod, "create_segmentation_model", lambda *a, **k: _TinyModel(num_classes=1))
    cfg = load_train_config(config_path)
    written = pred_mod.run_prediction_panels(
        cfg,
        checkpoint=run_dir / "model_best.pth",
        split="val",
        output_dir=run_dir / "predictions" / "val",
        device="cpu",
        max_images=1,
    )

    assert len(written) == 1
    assert written[0].exists()
