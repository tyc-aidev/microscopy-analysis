"""Tests for held-out evaluation (Sprint 2).

Uses a tiny real ``nn.Module`` (1x1 conv) on synthetic color-separable data, so
whole-image inference + IoU scoring run fast on CPU without downloading weights.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from microscopy_analysis.eval.evaluate import _pad_to_multiple, evaluate_run
from microscopy_analysis.train.config import TrainConfig


class _TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return torch.softmax(y, dim=1) if self.num_classes > 1 else torch.sigmoid(y)


def _make_super_split(root: Path, split: str, n: int, size: int = 20) -> None:
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


def _cfg(tmp_path: Path, **over) -> TrainConfig:
    base = TrainConfig(
        run_name="eval_run",
        data_root=tmp_path,
        dataset_name="Super1",
        dataset_family="super",
        split="train",
        architecture="UnetPlusPlus",
        encoder_name="resnet50",
        pretraining="micronet",
        num_classes=3,
        output_dir=tmp_path / "results" / "eval_run",
    )
    return replace(base, **over)


def test_pad_to_multiple_rounds_up_and_reports_original() -> None:
    padded, h, w = _pad_to_multiple(torch.zeros(3, 20, 30))
    assert (h, w) == (20, 30)
    assert padded.shape[1] % 32 == 0 and padded.shape[2] % 32 == 0
    assert padded.shape[1] >= 20 and padded.shape[2] >= 30


def test_evaluate_run_writes_json_and_scores_separable_super(tmp_path: Path) -> None:
    _make_super_split(tmp_path, "test", n=3)
    cfg = _cfg(tmp_path)

    # Hand-set the 1x1 conv so red->class1, blue->class2, else class0 — a perfect
    # classifier on this synthetic split, so IoU on present classes should be ~1.
    model = _TinyModel(num_classes=3)
    with torch.no_grad():
        model.conv.weight.zero_()
        model.conv.bias.zero_()
        model.conv.weight[1, 0] = 10.0  # class 1 responds to R channel
        model.conv.weight[2, 2] = 10.0  # class 2 responds to B channel

    result = evaluate_run(cfg, split="test", model=model, device_preference="cpu")

    assert result.num_samples == 3
    assert result.split == "test"
    assert len(result.per_class_iou) == 3
    assert 0.0 <= result.mean_iou <= 1.0
    assert result.per_class_iou[1] > 0.99 and result.per_class_iou[2] > 0.99

    payload = json.loads((cfg.output_dir / "eval_test.json").read_text())
    assert payload["dataset_name"] == "Super1"
    assert payload["pretraining"] == "micronet"
    assert payload["metrics_path"].endswith("eval_test.json")
    assert payload["train_subsample"] is None  # full split by default


def test_evaluate_run_records_train_subsample(tmp_path: Path) -> None:
    _make_super_split(tmp_path, "test", n=2)
    cfg = _cfg(tmp_path, train_subsample=1)
    result = evaluate_run(cfg, split="test", model=_TinyModel(num_classes=3), device_preference="cpu")
    assert result.train_subsample == 1
    payload = json.loads((cfg.output_dir / "eval_test.json").read_text())
    assert payload["train_subsample"] == 1


def test_evaluate_run_binary_ebc_scores_two_rows(tmp_path: Path) -> None:
    image_dir = tmp_path / "EBC1" / "test"
    annot_dir = tmp_path / "EBC1" / "test_annot"
    image_dir.mkdir(parents=True)
    annot_dir.mkdir(parents=True)
    for i in range(2):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        image[:10], mask[:10] = 255, 1
        Image.fromarray(image, mode="RGB").save(image_dir / f"t{i}.tif")
        Image.fromarray(mask, mode="L").save(annot_dir / f"t{i}.tif")

    cfg = _cfg(tmp_path, dataset_name="EBC1", dataset_family="ebc", num_classes=1,
               run_name="eval_ebc", output_dir=tmp_path / "results" / "eval_ebc")
    model = _TinyModel(num_classes=1)
    with torch.no_grad():
        model.conv.weight.zero_()
        model.conv.bias.fill_(-10.0)
        model.conv.weight[0, 0] = 20.0  # bright R (foreground) -> positive

    result = evaluate_run(cfg, split="test", model=model, device_preference="cpu")
    assert len(result.per_class_iou) == 2  # [background, foreground]
    assert result.score == result.per_class_iou[1]  # binary headline = foreground IoU
    assert result.score > 0.99


def test_evaluate_run_raises_without_samples(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model = _TinyModel(num_classes=3)
    try:
        evaluate_run(cfg, split="test", model=model, device_preference="cpu")
    except RuntimeError as exc:
        assert "No samples" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for empty split")
