"""Unit tests for DiceBCELoss on activated probabilities."""

from __future__ import annotations

import torch

from microscopy_analysis.eval import DiceBCELoss, soft_dice_loss


def test_perfect_multiclass_prediction_has_low_loss() -> None:
    target = torch.tensor([[[0, 1], [2, 0]]])  # (N=1, H=2, W=2)
    probs = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float()
    loss = DiceBCELoss(weight=0.7)(probs, target)
    assert loss.item() < 1e-2


def test_wrong_prediction_has_higher_loss_than_correct() -> None:
    target = torch.tensor([[[0, 1], [1, 0]]])
    correct = torch.nn.functional.one_hot(target, 2).permute(0, 3, 1, 2).float().clamp(0.01, 0.99)
    wrong = correct.flip(1)
    loss_fn = DiceBCELoss(weight=0.5)
    assert loss_fn(wrong, target) > loss_fn(correct, target)


def test_binary_loss_runs_and_is_lower_when_correct() -> None:
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (N=1, H=2, W=2)
    probs_good = target.unsqueeze(1).clamp(0.02, 0.98)
    probs_bad = (1.0 - target).unsqueeze(1).clamp(0.02, 0.98)
    loss_fn = DiceBCELoss(weight=0.7)
    assert loss_fn(probs_good, target) < loss_fn(probs_bad, target)


def test_soft_dice_perfect_is_zero() -> None:
    probs = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # (1, 2, 1, 2)
    assert soft_dice_loss(probs, probs).item() < 1e-3
