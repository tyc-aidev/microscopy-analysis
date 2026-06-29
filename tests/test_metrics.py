"""Unit tests for the IoU metric on synthetic predictions."""

from __future__ import annotations

import torch

from microscopy_analysis.eval import IoU, per_class_iou


def test_per_class_iou_perfect_is_one() -> None:
    labels = torch.tensor([[0, 1], [2, 0]])
    iou = per_class_iou(labels, labels, num_rows=3)
    assert torch.allclose(iou, torch.ones(3), atol=1e-4)


def test_multiclass_iou_perfect_prediction() -> None:
    target = torch.tensor([[[0, 1], [2, 0]]])
    probs = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float()
    metric = IoU(num_classes=3)
    metric.update(probs, target)
    assert abs(metric.mean() - 1.0) < 1e-4
    assert abs(metric.score() - 1.0) < 1e-4


def test_multiclass_half_wrong_iou() -> None:
    target = torch.tensor([[[0, 0], [1, 1]]])
    pred = torch.tensor([[[0, 0], [0, 1]]])  # one of two class-1 pixels correct
    probs = torch.nn.functional.one_hot(pred, 2).permute(0, 3, 1, 2).float()
    metric = IoU(num_classes=2)
    metric.update(probs, target)
    per_class = metric.per_class()
    assert abs(float(per_class[0]) - 2 / 3) < 1e-4  # bg: 2 correct, 1 false positive
    assert abs(float(per_class[1]) - 1 / 2) < 1e-4  # fg: 1 correct, 1 missed


def test_binary_iou_score_is_foreground() -> None:
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    probs = torch.tensor([[[[0.9, 0.1], [0.1, 0.1]]]])  # (1, 1, 2, 2) -> threshold 0.5
    metric = IoU(num_classes=1, threshold=0.5)
    metric.update(probs, target)
    assert abs(metric.score() - 1.0) < 1e-4


def test_accumulation_across_batches() -> None:
    metric = IoU(num_classes=2)
    target = torch.tensor([[[0, 1]]])
    good = torch.nn.functional.one_hot(target, 2).permute(0, 3, 1, 2).float()
    bad = good.flip(1)
    metric.update(good, target)
    metric.update(bad, target)
    assert 0.0 < metric.mean() < 1.0
