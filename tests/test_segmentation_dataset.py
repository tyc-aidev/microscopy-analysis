"""Tests for the torch SegmentationDataset + albumentations transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from microscopy_analysis.data.segmentation_dataset import (
    AugmentationConfig,
    SegmentationDataset,
    build_transforms,
)


def _make_super(root: Path, n: int = 2, size: int = 32) -> Path:
    dataset_dir = root / "Super1"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "train_annot").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
            dataset_dir / "train" / f"img{i}.tif"
        )
        mask = np.zeros((size, size, 3), dtype=np.uint8)
        mask[: size // 2, :] = (255, 0, 0)  # class 1
        mask[size // 2 :, : size // 2] = (0, 0, 255)  # class 2
        Image.fromarray(mask, mode="RGB").save(dataset_dir / "train_annot" / f"img{i}_mask.tif")
    return dataset_dir


def _make_ebc(root: Path, n: int = 2, size: int = 600) -> Path:
    dataset_dir = root / "EBC1"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "train_annot").mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
            dataset_dir / "train" / f"t{i}.tif"
        )
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[: size // 2] = 1  # oxide
        Image.fromarray(mask, mode="L").save(dataset_dir / "train_annot" / f"t{i}.tif")
    return dataset_dir


def test_super_dataset_yields_long_class_masks(tmp_path: Path) -> None:
    dataset_dir = _make_super(tmp_path)
    transform = build_transforms("super", train=True)
    ds = SegmentationDataset(dataset_dir, "train", "super", transform)
    image, mask = ds[0]
    assert len(ds) == 2
    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert mask.shape == (32, 32) and mask.dtype == torch.long
    assert set(mask.unique().tolist()) <= {0, 1, 2}


def test_ebc_dataset_yields_float_binary_mask_and_crops(tmp_path: Path) -> None:
    dataset_dir = _make_ebc(tmp_path)
    transform = build_transforms("ebc", train=True, crop_size=512)
    ds = SegmentationDataset(dataset_dir, "train", "ebc", transform)
    image, mask = ds[0]
    assert image.shape == (3, 512, 512)
    assert mask.shape == (512, 512) and mask.dtype == torch.float32
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_eval_transform_keeps_full_image(tmp_path: Path) -> None:
    dataset_dir = _make_super(tmp_path, size=48)
    ds = SegmentationDataset(dataset_dir, "train", "super", build_transforms("super", train=False))
    image, mask = ds[0]
    assert image.shape == (3, 48, 48)
    assert mask.shape == (48, 48)


def test_augmentation_config_disables_ops() -> None:
    # Every probability at 0 should leave only normalize + ToTensor in the pipeline.
    off = AugmentationConfig(flip_p=0, vflip_p=0, rotate90_p=0, clahe_p=0, gauss_noise_p=0)
    assert len(build_transforms("super", train=True, aug=off).transforms) == 2
    # Defaults keep the full Super profile (5 aug ops + normalize + ToTensor).
    assert len(build_transforms("super", train=True).transforms) == 7


def test_augmentation_from_dict_overrides_crop_and_ignores_unknown() -> None:
    aug = AugmentationConfig.from_dict({"crop_size": 256, "clahe_p": 0.9, "bogus": 1}, crop_size=512)
    assert aug.crop_size == 256 and aug.clahe_p == 0.9


def test_ebc_crop_size_follows_augmentation_config(tmp_path: Path) -> None:
    dataset_dir = _make_ebc(tmp_path, size=600)
    aug = AugmentationConfig(crop_size=256)
    ds = SegmentationDataset(dataset_dir, "train", "ebc", build_transforms("ebc", train=True, aug=aug))
    image, mask = ds[0]
    assert image.shape == (3, 256, 256) and mask.shape == (256, 256)
