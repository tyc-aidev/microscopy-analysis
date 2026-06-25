"""Tests for explorer/lib index and mask utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from explorer.lib.index import (
    _ebc_mask_path,
    _super_mask_path,
    scan_benchmarks,
    split_counts,
)
from explorer.lib.masks import class_pixel_counts, colored_mask_rgba, load_class_masks, overlay_image


@pytest.fixture
def super_fixture(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "benchmark_segmentation_data" / "Super1"
    train = dataset_dir / "train"
    train_annot = dataset_dir / "train_annot"
    train.mkdir(parents=True)
    train_annot.mkdir(parents=True)

    Image.new("RGB", (4, 4), (128, 128, 128)).save(train / "sample.tif")
    Image.new("RGB", (4, 4), (0, 0, 0)).save(train_annot / "sample_mask.tif")
    return tmp_path


@pytest.fixture
def ebc_fixture(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "benchmark_segmentation_data" / "EBC1"
    train = dataset_dir / "train"
    train_annot = dataset_dir / "train_annot"
    train.mkdir(parents=True)
    train_annot.mkdir(parents=True)

    Image.new("L", (4, 4), 0).save(train / "tile.tif")
    Image.fromarray(np.array([[0, 1], [2, 0]], dtype=np.uint8)).resize((4, 4)).save(
        train_annot / "tile.tif"
    )
    return tmp_path


def test_super_mask_pairing(super_fixture: Path) -> None:
    image = super_fixture / "benchmark_segmentation_data/Super1/train/sample.tif"
    annot = super_fixture / "benchmark_segmentation_data/Super1/train_annot"
    mask = _super_mask_path(image, annot)
    assert mask is not None
    assert mask.name == "sample_mask.tif"


def test_ebc_mask_pairing(ebc_fixture: Path) -> None:
    image = ebc_fixture / "benchmark_segmentation_data/EBC1/train/tile.tif"
    annot = ebc_fixture / "benchmark_segmentation_data/EBC1/train_annot"
    mask = _ebc_mask_path(image, annot)
    assert mask is not None
    assert mask.name == "tile.tif"


def test_scan_benchmarks(benchmark_fixture: Path) -> None:
    records = scan_benchmarks(benchmark_fixture)
    assert len(records) == 2
    counts = split_counts(records)
    assert counts["Super1"]["train"] == 1
    assert counts["EBC1"]["train"] == 1


def test_super_class_masks(super_fixture: Path) -> None:
    mask_path = super_fixture / "benchmark_segmentation_data/Super1/train_annot/sample_mask.tif"
    masks = load_class_masks(mask_path, "super")
    assert set(masks) == {"matrix", "secondary", "tertiary"}
    counts = class_pixel_counts(masks)
    assert counts["matrix"]["pixels"] == 16


def test_ebc_class_masks(ebc_fixture: Path) -> None:
    mask_path = ebc_fixture / "benchmark_segmentation_data/EBC1/train_annot/tile.tif"
    masks = load_class_masks(mask_path, "ebc")
    assert "oxide" in masks
    rgba = colored_mask_rgba(masks, "ebc", visible_classes={"oxide"})
    assert rgba.shape == (4, 4, 4)


def test_splits_for_dataset(benchmark_fixture: Path) -> None:
    from explorer.lib.index import splits_for_dataset

    records = scan_benchmarks(benchmark_fixture)
    assert splits_for_dataset(records, "Super1") == ["train"]
    assert splits_for_dataset(records, "EBC1") == ["train"]


def test_overlay_image(super_fixture: Path) -> None:
    image_path = super_fixture / "benchmark_segmentation_data/Super1/train/sample.tif"
    mask_path = super_fixture / "benchmark_segmentation_data/Super1/train_annot/sample_mask.tif"
    image = Image.open(image_path)
    masks = load_class_masks(mask_path, "super")
    rgba = colored_mask_rgba(masks, "super")
    result = overlay_image(image, rgba, opacity=0.5)
    assert result.mode == "RGBA"
    assert result.size == image.size
