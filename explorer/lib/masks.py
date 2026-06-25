"""Mask parsing and overlay rendering for Super (RGB) and EBC (grayscale) benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from explorer.lib.catalog import get_family


def _rgb_array(mask: Image.Image) -> np.ndarray:
    if mask.mode != "RGB":
        mask = mask.convert("RGB")
    return np.asarray(mask)


def super_class_masks(rgb: np.ndarray, family: dict[str, Any]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for cls in family["classes"]:
        color = np.array(cls["color"], dtype=np.uint8)
        masks[cls["id"]] = np.all(rgb == color, axis=-1)
    return masks


def ebc_class_masks(gray: np.ndarray, family: dict[str, Any]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for cls in family["classes"]:
        value = cls["value"]
        if value is None:
            continue
        masks[cls["id"]] = gray == value
    return masks


def load_class_masks(mask_path: str | Any, family_id: str) -> dict[str, np.ndarray]:
    family = get_family(family_id)
    mask = Image.open(mask_path)
    if family_id == "super":
        return super_class_masks(_rgb_array(mask), family)
    gray = np.asarray(mask.convert("L"))
    return ebc_class_masks(gray, family)


def class_pixel_counts(class_masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    total = next(iter(class_masks.values())).size if class_masks else 0
    if total == 0:
        return {}
    counts: dict[str, dict[str, float]] = {}
    for cls_id, mask in class_masks.items():
        pixels = int(mask.sum())
        counts[cls_id] = {
            "pixels": pixels,
            "percent": round(100.0 * pixels / total, 2),
        }
    return counts


def colored_mask_rgba(
    class_masks: dict[str, np.ndarray],
    family_id: str,
    *,
    visible_classes: set[str] | None = None,
) -> np.ndarray:
    family = get_family(family_id)
    if not class_masks:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    shape = next(iter(class_masks.values())).shape
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    color_by_id = {cls["id"]: cls["color"] for cls in family["classes"]}

    for cls_id, mask in class_masks.items():
        if visible_classes is not None and cls_id not in visible_classes:
            continue
        color = color_by_id.get(cls_id, [128, 128, 128])
        rgba[mask, :3] = color
        rgba[mask, 3] = 255
    return rgba


def overlay_image(
    image: Image.Image,
    mask_rgba: np.ndarray,
    *,
    opacity: float = 0.5,
) -> Image.Image:
    base = image.convert("RGBA")
    if base.size[::-1] != mask_rgba.shape[:2]:
        mask_img = Image.fromarray(mask_rgba, mode="RGBA").resize(base.size, Image.Resampling.NEAREST)
        mask_rgba = np.asarray(mask_img)

    overlay = Image.fromarray(mask_rgba, mode="RGBA")
    overlay.putalpha(int(max(0, min(1, opacity)) * 255))
    return Image.alpha_composite(base, overlay)
