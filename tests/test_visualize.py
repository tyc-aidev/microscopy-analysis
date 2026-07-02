"""Tests for qualitative segmentation visualization helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from microscopy_analysis.eval.visualize import (
    class_index_to_rgb,
    error_map_rgba,
    make_prediction_panel,
    overlay_mask,
)


def test_class_index_to_rgb_super() -> None:
    mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    rgb = class_index_to_rgb(mask, "super", num_classes=3)
    assert tuple(rgb[0, 0]) == (0, 0, 0)
    assert tuple(rgb[0, 1]) == (255, 0, 0)
    assert tuple(rgb[1, 0]) == (0, 0, 255)


def test_error_map_marks_mismatches() -> None:
    pred = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    target = np.array([[0, 0], [1, 0]], dtype=np.uint8)
    rgba = error_map_rgba(pred, target)
    assert rgba[0, 1, 3] == 255  # pred=1, target=0
    assert rgba[0, 0, 3] == 0
    assert rgba[1, 0, 3] == 0
    assert rgba[1, 1, 3] == 0


def test_make_prediction_panel_dimensions() -> None:
    image = Image.new("RGB", (32, 24), color=(128, 128, 128))
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[:12, :] = 1
    target_rgb = class_index_to_rgb(mask, "super", num_classes=3)
    pred_rgb = class_index_to_rgb(mask, "super", num_classes=3)
    panel = make_prediction_panel(image, target_rgb, pred_rgb, error_map_rgba(mask, mask))
    assert panel.width == 32 * 4
    assert panel.height >= 24


def test_overlay_mask_resizes_to_image() -> None:
    image = Image.new("RGB", (40, 30), color=(0, 0, 0))
    mask_rgb = np.full((20, 10, 3), 255, dtype=np.uint8)
    out = overlay_mask(image, mask_rgb)
    assert out.size == image.size
