#!/usr/bin/env python3
"""Save qualitative prediction panels for a trained segmentation run.

Each output PNG is a 4-column figure: input | GT overlay | prediction overlay | errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from microscopy_analysis.data.dataset_adapter import decode_ebc_mask, decode_super_mask, list_sample_pairs
from microscopy_analysis.data.segmentation_dataset import build_transforms
from microscopy_analysis.device import enable_mps_fallback, resolve_device
from microscopy_analysis.eval.visualize import class_index_to_rgb, error_map_rgba, make_prediction_panel
from microscopy_analysis.models import create_segmentation_model
from microscopy_analysis.train.config import load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize segmentation predictions for a trained run")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML used for training")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model weights (default: results/<run_name>/model_best.pth)",
    )
    parser.add_argument("--split", default="val", help="Dataset split to visualize (train|val|test)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG panels (default: results/<run_name>/predictions/<split>)",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Cap number of panels written")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    cfg = load_train_config(args.config)
    enable_mps_fallback()
    device = resolve_device(args.device)

    checkpoint = args.checkpoint or (cfg.output_dir / "model_best.pth")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = create_segmentation_model(
        cfg.architecture, cfg.encoder_name, cfg.pretraining, cfg.num_classes
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()

    dataset_root = cfg.data_root / cfg.dataset_name
    pairs = list_sample_pairs(dataset_root, split=args.split, dataset_family=cfg.dataset_family)
    if not pairs:
        raise RuntimeError(f"No samples found for {cfg.dataset_name} split={args.split}")

    out_dir = args.output_dir or (cfg.output_dir / "predictions" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transforms(cfg.dataset_family, train=False, crop_size=cfg.crop_size)
    limit = len(pairs) if args.max_images is None else min(args.max_images, len(pairs))

    for pair in pairs[:limit]:
        image = Image.open(pair.image_path).convert("RGB")
        image_np = np.asarray(image, dtype=np.uint8)
        target = _decode_mask(pair.mask_path, cfg.dataset_family)

        transformed = transform(image=image_np, mask=target)
        image_t = transformed["image"].to(device)
        pred = _predict_labels(model, image_t, num_classes=cfg.num_classes)

        target_rgb = class_index_to_rgb(target, cfg.dataset_family, num_classes=cfg.num_classes)
        pred_rgb = class_index_to_rgb(pred, cfg.dataset_family, num_classes=cfg.num_classes)
        panel = make_prediction_panel(image, target_rgb, pred_rgb, error_map_rgba(pred, target))

        out_path = out_dir / f"{pair.image_path.stem}_panel.png"
        panel.save(out_path)
        print(f"Wrote {out_path}")

    print(f"Done. {limit} panel(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
