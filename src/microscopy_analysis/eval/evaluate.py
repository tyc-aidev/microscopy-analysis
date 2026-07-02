"""Held-out test-set evaluation for a trained segmentation run (Sprint 2).

Loads the best checkpoint (``model_best.pth``) produced by the trainer and
computes IoU on a held-out split (default ``test``). Reports per-class IoU, mean
IoU, and the paper's headline scalar (foreground IoU for binary EBC tasks, mean
IoU for multiclass Super tasks).

Whole-image inference is used: each image is ImageNet-normalized and padded up to
a multiple of 32 (smp encoders require that), run through the model, then cropped
back to the original size before scoring. Sliding-window patch inference
(512/stride 256) is a Sprint 5 refinement and intentionally not done here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from microscopy_analysis.data.dataset_adapter import (
    decode_ebc_mask,
    decode_super_mask,
    list_sample_pairs,
)
from microscopy_analysis.data.segmentation_dataset import IMAGENET_MEAN, IMAGENET_STD
from microscopy_analysis.device import enable_mps_fallback, resolve_device
from microscopy_analysis.eval.metrics import IoU
from microscopy_analysis.models import create_segmentation_model
from microscopy_analysis.train.config import TrainConfig

_STRIDE = 32  # smp encoder downsampling factor; inputs must be a multiple of this.


@dataclass(frozen=True)
class EvalResult:
    run_name: str
    dataset_name: str
    dataset_family: str
    architecture: str
    encoder_name: str
    pretraining: str
    split: str
    num_samples: int
    num_classes: int
    checkpoint_path: str
    device: str
    per_class_iou: tuple[float, ...]
    mean_iou: float
    score: float  # foreground IoU (binary) or mean IoU (multiclass) — paper headline
    created_at_utc: str
    metrics_path: str = ""
    extra: dict = field(default_factory=dict)


def _decode_mask(mask_path: Path, dataset_family: str) -> np.ndarray:
    if dataset_family == "super":
        return decode_super_mask(mask_path)
    return decode_ebc_mask(mask_path)


def _normalize_image(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> normalized CHW float tensor (ImageNet stats, per notebooks)."""
    arr = image.astype(np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def _pad_to_multiple(image_t: torch.Tensor, stride: int = _STRIDE) -> tuple[torch.Tensor, int, int]:
    """Pad a CHW tensor's H/W up to a multiple of ``stride``; return (padded, H, W)."""
    _, h, w = image_t.shape
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    if pad_h or pad_w:
        image_t = torch.nn.functional.pad(image_t, (0, pad_w, 0, pad_h), mode="reflect" if min(h, w) > 1 else "constant")
    return image_t, h, w


@torch.no_grad()
def _predict_probs(model: torch.nn.Module, image_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    padded, h, w = _pad_to_multiple(image_t)
    probs = model(padded.unsqueeze(0).to(device))
    return probs[..., :h, :w].cpu()


def evaluate_run(
    config: TrainConfig,
    *,
    split: str = "test",
    checkpoint: Path | None = None,
    device_preference: str = "auto",
    model: torch.nn.Module | None = None,
    write: bool = True,
) -> EvalResult:
    """Evaluate a trained run's best checkpoint on a held-out ``split``.

    ``model`` may be injected (tests); otherwise the smp model is built and the
    checkpoint weights are loaded. Set ``write=False`` to skip the JSON sidecar.
    """
    enable_mps_fallback()
    device = resolve_device(device_preference)

    checkpoint_path = checkpoint or (config.output_dir / "model_best.pth")
    if model is None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = create_segmentation_model(
            config.architecture, config.encoder_name, config.pretraining, config.num_classes
        )
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"])
    model = model.to(device)
    model.eval()

    dataset_root = config.data_root / config.dataset_name
    pairs = list_sample_pairs(dataset_root, split=split, dataset_family=config.dataset_family)
    if not pairs:
        raise RuntimeError(f"No samples found for {config.dataset_name} split={split} in {dataset_root}")

    from PIL import Image  # local import keeps module import torch-light for callers

    metric = IoU(config.num_classes, threshold=config.metric_threshold)
    for pair in pairs:
        image = np.asarray(Image.open(pair.image_path).convert("RGB"), dtype=np.uint8)
        target = _decode_mask(pair.mask_path, config.dataset_family)
        probs = _predict_probs(model, _normalize_image(image), device)
        target_t = torch.from_numpy(target.astype(np.int64)).unsqueeze(0)
        metric.update(probs, target_t)

    per_class = tuple(round(v, 6) for v in metric.per_class().tolist())
    result = EvalResult(
        run_name=config.run_name,
        dataset_name=config.dataset_name,
        dataset_family=config.dataset_family,
        architecture=config.architecture,
        encoder_name=config.encoder_name,
        pretraining=config.pretraining,
        split=split,
        num_samples=len(pairs),
        num_classes=config.num_classes,
        checkpoint_path=str(checkpoint_path),
        device=str(device),
        per_class_iou=per_class,
        mean_iou=round(metric.mean(), 6),
        score=round(metric.score(), 6),
        created_at_utc=datetime.now(UTC).isoformat(),
    )

    if write:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = config.output_dir / f"eval_{split}.json"
        payload = asdict(result) | {"metrics_path": str(metrics_path)}
        metrics_path.write_text(json.dumps(payload, indent=2))
        result = EvalResult(**payload)
    return result
