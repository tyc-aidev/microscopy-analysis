"""Generate qualitative prediction panels for a trained segmentation run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from microscopy_analysis.data.dataset_adapter import decode_ebc_mask, decode_super_mask, list_sample_pairs
from microscopy_analysis.data.segmentation_dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from microscopy_analysis.device import enable_mps_fallback, resolve_device
from microscopy_analysis.eval.visualize import class_index_to_rgb, error_map_rgba, make_prediction_panel
from microscopy_analysis.models import create_segmentation_model
from microscopy_analysis.train.config import TrainConfig


def _decode_mask(mask_path: Path, dataset_family: str) -> np.ndarray:
    if dataset_family == "super":
        return decode_super_mask(mask_path)
    return decode_ebc_mask(mask_path)


def _predict_labels(model: torch.nn.Module, image_t: torch.Tensor, *, num_classes: int) -> np.ndarray:
    with torch.no_grad():
        probs = model(image_t.unsqueeze(0))
    if num_classes == 1:
        return (probs.squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
    return probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _tensor_to_display_image(image_t: torch.Tensor) -> Image.Image:
    """Convert a normalized CHW tensor back to a displayable RGB image."""
    image_np = image_t.detach().cpu().permute(1, 2, 0).numpy()
    image_np = image_np * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    image_np = np.clip(image_np, 0.0, 1.0)
    return Image.fromarray((image_np * 255).astype(np.uint8), mode="RGB")


def _mask_to_labels(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(np.uint8)
    return np.asarray(mask, dtype=np.uint8)


def run_prediction_panels(
    cfg: TrainConfig,
    *,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    device: str = "auto",
    max_images: int | None = None,
) -> list[Path]:
    """Run inference and write 4-column prediction PNGs. Returns written paths."""
    enable_mps_fallback()
    dev = resolve_device(device)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = create_segmentation_model(
        cfg.architecture, cfg.encoder_name, cfg.pretraining, cfg.num_classes
    ).to(dev)
    state = torch.load(checkpoint, map_location=dev, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()

    dataset_root = cfg.data_root / cfg.dataset_name
    pairs = list_sample_pairs(dataset_root, split=split, dataset_family=cfg.dataset_family)
    if not pairs:
        raise RuntimeError(f"No samples found for {cfg.dataset_name} split={split}")

    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transforms(cfg.dataset_family, train=False, crop_size=cfg.crop_size)
    limit = len(pairs) if max_images is None else min(max_images, len(pairs))

    written: list[Path] = []
    for pair in pairs[:limit]:
        image_np = np.asarray(Image.open(pair.image_path).convert("RGB"), dtype=np.uint8)
        target = _decode_mask(pair.mask_path, cfg.dataset_family)

        transformed = transform(image=image_np, mask=target)
        image_t = transformed["image"].to(dev)
        target_labels = _mask_to_labels(transformed["mask"])
        display_image = _tensor_to_display_image(image_t)
        pred = _predict_labels(model, image_t, num_classes=cfg.num_classes)

        target_rgb = class_index_to_rgb(target_labels, cfg.dataset_family, num_classes=cfg.num_classes)
        pred_rgb = class_index_to_rgb(pred, cfg.dataset_family, num_classes=cfg.num_classes)
        panel = make_prediction_panel(display_image, target_rgb, pred_rgb, error_map_rgba(pred, target_labels))

        out_path = output_dir / f"{pair.image_path.stem}_panel.png"
        panel.save(out_path)
        written.append(out_path)
    return written
