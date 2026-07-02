"""Loss and metric primitives for segmentation training.

Both operate on **activated** model outputs (probabilities in ``[0, 1]``),
matching the :mod:`microscopy_analysis.models` factory convention where models
carry their own ``softmax2d`` / ``sigmoid`` head — not raw logits.
"""

from .evaluate import EvalResult, evaluate_run
from .losses import DiceBCELoss, soft_dice_loss
from .metrics import IoU, per_class_iou

__all__ = ["DiceBCELoss", "soft_dice_loss", "IoU", "per_class_iou", "EvalResult", "evaluate_run"]
