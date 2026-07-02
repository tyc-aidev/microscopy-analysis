"""Torch ``Dataset`` + albumentations transforms for the NASA benchmarks.

Wraps the torch-free :mod:`dataset_adapter` helpers (pairing + mask decoding)
and yields ``(image, mask)`` tensors ready for training:

- **Super** (multiclass): mask is a ``long`` class-index map ``{0, 1, 2}``.
- **EBC** (binary): mask is a ``float`` ``{0, 1}`` map.

ImageNet normalization is applied for every encoder (per the paper notebooks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from .dataset_adapter import decode_ebc_mask, decode_super_mask, list_sample_pairs, subsample_pairs

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentationConfig:
    """YAML-overridable augmentation knobs; defaults reproduce the PLAN.md profiles.

    Super = flip/rotate/CLAHE/noise; EBC = horizontal flip + random ``crop_size``
    crop (center crop on val). Probabilities of ``0`` disable an op entirely.
    """

    crop_size: int = 512
    flip_p: float = 0.5
    vflip_p: float = 0.5
    rotate90_p: float = 0.5
    clahe_p: float = 0.3
    gauss_noise_p: float = 0.3
    extra: tuple[str, ...] = field(default_factory=tuple)  # reserved; kept for forward-compat

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, *, crop_size: int = 512) -> AugmentationConfig:
        raw = raw or {}
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs.setdefault("crop_size", crop_size)
        return cls(**kwargs)


def _normalize() -> list[A.BasicTransform]:
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


def build_transforms(
    dataset_family: str,
    train: bool,
    crop_size: int = 512,
    aug: AugmentationConfig | None = None,
) -> A.Compose:
    """Augmentation pipeline per PLAN.md, overridable via :class:`AugmentationConfig`."""
    aug = aug or AugmentationConfig(crop_size=crop_size)
    crop = aug.crop_size

    if not train:
        if dataset_family == "ebc":
            return A.Compose([A.PadIfNeeded(crop, crop), A.CenterCrop(crop, crop), *_normalize()])
        return A.Compose(_normalize())

    if dataset_family == "super":
        ops = [
            (A.HorizontalFlip, aug.flip_p),
            (A.VerticalFlip, aug.vflip_p),
            (A.RandomRotate90, aug.rotate90_p),
            (A.CLAHE, aug.clahe_p),
            (A.GaussNoise, aug.gauss_noise_p),
        ]
        steps = [op(p=p) for op, p in ops if p > 0]
    elif dataset_family == "ebc":
        steps = [A.PadIfNeeded(crop, crop), A.RandomCrop(crop, crop)]
        if aug.flip_p > 0:
            steps.insert(0, A.HorizontalFlip(p=aug.flip_p))
    else:
        raise ValueError(f"Unsupported dataset family: {dataset_family}")
    return A.Compose([*steps, *_normalize()])


class SegmentationDataset(Dataset):
    """Image/mask pairs for one dataset split, decoded and transformed to tensors."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        dataset_family: str,
        transform: A.Compose | None = None,
        subsample: int | None = None,
        subsample_seed: int = 42,
    ) -> None:
        self.dataset_family = dataset_family
        self.transform = transform
        self.pairs = list_sample_pairs(Path(dataset_root), split=split, dataset_family=dataset_family)
        # Low-data ablation (Sprint 3): keep a deterministic subset of the split.
        if subsample is not None:
            self.pairs = subsample_pairs(self.pairs, subsample, seed=subsample_seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def _decode_mask(self, path: Path) -> np.ndarray:
        if self.dataset_family == "super":
            return decode_super_mask(path)
        return decode_ebc_mask(path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        image = np.asarray(Image.open(pair.image_path).convert("RGB"), dtype=np.uint8)
        mask = self._decode_mask(pair.mask_path)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_t, mask_t = out["image"], out["mask"]
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask)

        mask_t = mask_t.long() if self.dataset_family == "super" else mask_t.float()
        return image_t, mask_t
