"""Torch ``Dataset`` + albumentations transforms for the NASA benchmarks.

Wraps the torch-free :mod:`dataset_adapter` helpers (pairing + mask decoding)
and yields ``(image, mask)`` tensors ready for training:

- **Super** (multiclass): mask is a ``long`` class-index map ``{0, 1, 2}``.
- **EBC** (binary): mask is a ``float`` ``{0, 1}`` map.

ImageNet normalization is applied for every encoder (per the paper notebooks).
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from .dataset_adapter import decode_ebc_mask, decode_super_mask, list_sample_pairs

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize() -> list[A.BasicTransform]:
    return [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]


def build_transforms(dataset_family: str, train: bool, crop_size: int = 512) -> A.Compose:
    """Augmentation pipeline per PLAN.md: Super = flip/rotate/CLAHE/noise; EBC = flip + random crop."""
    if not train:
        if dataset_family == "ebc":
            return A.Compose([A.PadIfNeeded(crop_size, crop_size), A.CenterCrop(crop_size, crop_size), *_normalize()])
        return A.Compose(_normalize())

    if dataset_family == "super":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.CLAHE(p=0.3),
            A.GaussNoise(p=0.3),
        ]
    elif dataset_family == "ebc":
        aug = [A.HorizontalFlip(p=0.5), A.PadIfNeeded(crop_size, crop_size), A.RandomCrop(crop_size, crop_size)]
    else:
        raise ValueError(f"Unsupported dataset family: {dataset_family}")
    return A.Compose([*aug, *_normalize()])


class SegmentationDataset(Dataset):
    """Image/mask pairs for one dataset split, decoded and transformed to tensors."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        dataset_family: str,
        transform: A.Compose | None = None,
    ) -> None:
        self.dataset_family = dataset_family
        self.transform = transform
        self.pairs = list_sample_pairs(Path(dataset_root), split=split, dataset_family=dataset_family)

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
