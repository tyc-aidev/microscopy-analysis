"""Qualitative segmentation figures: colored masks, overlays, and prediction panels."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# NASA Super RGB convention (matrix / secondary / tertiary).
SUPER_CLASS_COLORS: tuple[tuple[int, int, int], ...] = ((0, 0, 0), (255, 0, 0), (0, 0, 255))
EBC_BINARY_COLORS: tuple[tuple[int, int, int], ...] = ((0, 0, 0), (255, 128, 0))


def class_index_to_rgb(mask: np.ndarray, dataset_family: str, *, num_classes: int) -> np.ndarray:
    """Render an integer (or binary float) label map as an RGB image."""
    labels = mask.astype(np.uint8)
    if dataset_family == "ebc" or num_classes == 1:
        palette = EBC_BINARY_COLORS
    else:
        palette = SUPER_CLASS_COLORS
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        rgb[labels == idx] = color
    return rgb


def error_map_rgba(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Highlight misclassified pixels in red; correct pixels are transparent."""
    pred_labels = pred.astype(np.uint8)
    target_labels = target.astype(np.uint8)
    wrong = pred_labels != target_labels
    rgba = np.zeros((*pred_labels.shape, 4), dtype=np.uint8)
    rgba[wrong, :3] = (255, 0, 0)
    rgba[wrong, 3] = 255
    return rgba


def overlay_mask(
    image: Image.Image,
    mask_rgb: np.ndarray,
    *,
    opacity: float = 0.45,
) -> Image.Image:
    """Alpha-composite a solid RGB mask over an image."""
    base = image.convert("RGBA")
    overlay = Image.fromarray(mask_rgb, mode="RGB").convert("RGBA")
    overlay.putalpha(int(max(0.0, min(1.0, opacity)) * 255))
    if overlay.size != base.size:
        overlay = overlay.resize(base.size, Image.Resampling.NEAREST)
    return Image.alpha_composite(base, overlay)


def _label_panel(image: Image.Image, text: str) -> Image.Image:
    bar_h = 22
    panel = Image.new("RGB", (image.width, image.height + bar_h), (32, 32, 32))
    panel.paste(image.convert("RGB"), (0, bar_h))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), text, fill=(240, 240, 240))
    return panel


def make_prediction_panel(
    image: Image.Image,
    target_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    error_rgba: np.ndarray,
    *,
    titles: tuple[str, str, str, str] = ("Input", "Ground truth", "Prediction", "Errors"),
) -> Image.Image:
    """Build a 4-column figure: image | GT overlay | pred overlay | error map."""
    gt_overlay = overlay_mask(image, target_rgb)
    pred_overlay = overlay_mask(image, pred_rgb)
    error_base = image.convert("RGBA")
    error_img = Image.alpha_composite(error_base, Image.fromarray(error_rgba, mode="RGBA")).convert("RGB")

    panels = [
        _label_panel(image.convert("RGB"), titles[0]),
        _label_panel(gt_overlay.convert("RGB"), titles[1]),
        _label_panel(pred_overlay.convert("RGB"), titles[2]),
        _label_panel(error_img, titles[3]),
    ]
    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas
