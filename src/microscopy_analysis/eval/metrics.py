"""IoU metric for activated segmentation predictions."""

from __future__ import annotations

import torch

_EPS = 1e-7


def per_class_iou(pred_labels: torch.Tensor, target_labels: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Per-class IoU from integer label maps; absent-in-both classes score 1.0."""
    ious = torch.empty(num_rows)
    for c in range(num_rows):
        p = pred_labels == c
        t = target_labels == c
        intersection = (p & t).sum().float()
        union = (p | t).sum().float()
        ious[c] = (intersection + _EPS) / (union + _EPS)
    return ious


class IoU:
    """Accumulates IoU across batches of activated predictions.

    ``num_classes`` follows the smp convention (foreground classes; background
    implicit). Binary (``1``) is scored as two rows ``[background, foreground]``
    and :meth:`score` reports foreground IoU; multiclass reports mean IoU.
    """

    def __init__(self, num_classes: int, threshold: float = 0.5) -> None:
        self.binary = num_classes == 1
        self.rows = 2 if self.binary else num_classes
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.rows)
        self.union = torch.zeros(self.rows)

    @torch.no_grad()
    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        if self.binary:
            pred = (probs.squeeze(1) >= self.threshold).long()
            tgt = target.squeeze(1) if target.dim() == probs.dim() else target
        else:
            pred = probs.argmax(dim=1)
            tgt = target
        pred = pred.cpu()
        tgt = tgt.long().cpu()
        for c in range(self.rows):
            p = pred == c
            t = tgt == c
            self.intersection[c] += (p & t).sum()
            self.union[c] += (p | t).sum()

    def per_class(self) -> torch.Tensor:
        return (self.intersection + _EPS) / (self.union + _EPS)

    def mean(self) -> float:
        return float(self.per_class().mean())

    def score(self) -> float:
        """Early-stopping scalar: foreground IoU (binary) or mean IoU (multiclass)."""
        per_class = self.per_class()
        return float(per_class[1]) if self.binary else float(per_class.mean())
