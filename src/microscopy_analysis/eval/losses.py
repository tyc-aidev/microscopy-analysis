"""Dice + BCE/CE combined loss on activated probabilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_EPS = 1e-7


def _onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss over class channels for probabilities and one-hot targets."""
    dims = (0, 2, 3)
    intersection = (probs * target).sum(dims)
    cardinality = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + cross-entropy loss for activated (post-softmax/sigmoid) outputs.

    ``weight`` is the Dice contribution; the cross-entropy term gets ``1 - weight``
    (BCE for binary ``classes=1``, categorical CE for multiclass). Inputs are
    probabilities in ``[0, 1]``; targets are class indices ``(N, H, W)`` or, for
    binary, a float mask ``(N, H, W)`` / ``(N, 1, H, W)``.
    """

    def __init__(self, weight: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {weight}")
        self.weight = weight
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(_EPS, 1.0 - _EPS)
        if probs.shape[1] == 1:
            target = target.float()
            if target.dim() == probs.dim() - 1:
                target = target.unsqueeze(1)
            ce = F.binary_cross_entropy(probs, target)
            dice = soft_dice_loss(probs, target, self.smooth)
        else:
            num_classes = probs.shape[1]
            target_oh = target.float() if target.dim() == probs.dim() else _onehot(target, num_classes)
            ce = -(target_oh * probs.log()).sum(dim=1).mean()
            dice = soft_dice_loss(probs, target_oh, self.smooth)
        return self.weight * dice + (1.0 - self.weight) * ce
