"""Dataset adapters for Super and EBC segmentation layouts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path


def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _super_mask_path(image_path: Path, annot_dir: Path) -> Path | None:
    stem = image_path.stem
    candidate = annot_dir / f"{stem}_mask{image_path.suffix}"
    if candidate.exists():
        return candidate
    for ext in (".tif", ".tiff"):
        alt = annot_dir / f"{stem}_mask{ext}"
        if alt.exists():
            return alt
    return None


def _ebc_mask_path(image_path: Path, annot_dir: Path) -> Path | None:
    candidate = annot_dir / image_path.name
    return candidate if candidate.exists() else None


def list_sample_pairs(dataset_root: Path, split: str, dataset_family: str) -> list[SamplePair]:
    image_dir = dataset_root / split
    annot_dir = dataset_root / f"{split}_annot"
    if not image_dir.exists() or not annot_dir.exists():
        return []

    pairs: list[SamplePair] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or not _is_tiff(image_path):
            continue
        if dataset_family == "super":
            mask_path = _super_mask_path(image_path, annot_dir)
        elif dataset_family == "ebc":
            mask_path = _ebc_mask_path(image_path, annot_dir)
        else:
            raise ValueError(f"Unsupported dataset family: {dataset_family}")
        if mask_path is not None:
            pairs.append(SamplePair(image_path=image_path, mask_path=mask_path))
    return pairs


def subsample_pairs(pairs: list[SamplePair], n: int | None, *, seed: int = 42) -> list[SamplePair]:
    """Deterministically pick ``n`` sample pairs for the low-data ablation (Sprint 3).

    A seeded RNG selects a reproducible subset (same ``seed`` -> same images), so
    the ``{1, 2, 4, 8, all}`` training-set-size sweep is comparable across
    pretraining regimes. ``n`` of ``None`` / ``<= 0`` / ``>= len(pairs)`` returns
    every pair unchanged. The chosen pairs are returned in stable path order.
    """
    if n is None or n <= 0 or n >= len(pairs):
        return list(pairs)
    chosen = random.Random(seed).sample(pairs, n)
    return sorted(chosen, key=lambda p: p.image_path.name)


def decode_super_mask(mask_path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    out[np.all(arr == (255, 0, 0), axis=-1)] = 1
    out[np.all(arr == (0, 0, 255), axis=-1)] = 2
    return out


def decode_ebc_mask(mask_path: Path, positive_values: tuple[int, ...] = (1,)) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for value in positive_values:
        out[arr == value] = 1
    return out

