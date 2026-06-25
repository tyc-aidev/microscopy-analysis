"""Image/mask pairing for the NASA semantic benchmarks.

Pairing follows NASA ``io.py`` conventions:

- **Super** (RGB masks): ``image.tif`` pairs with ``image_mask.tif`` in the
  ``<split>_annot`` folder.
- **EBC** (grayscale masks): the mask shares the **same filename** in the
  ``<split>_annot`` folder.

This is the torch-free seed of the Sprint 1 ``DatasetAdapter`` — enough for the
Sprint 0 smoke test to locate real train/val/test pairs and sanity-check counts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SUPER_MASK_SUFFIX = "_mask"
ANNOT_SUFFIX = "_annot"
SPLITS = ("train", "val", "test", "different_test")


@dataclass(frozen=True)
class ImagePair:
    split: str
    image_path: Path
    mask_path: Path | None


def get_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "./data")).expanduser().resolve()


def benchmark_root(data_root: Path | None = None) -> Path:
    return (data_root or get_data_root()) / "benchmark_segmentation_data"


def _image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _super_mask(image_path: Path, annot_dir: Path) -> Path | None:
    if image_path.stem.endswith(SUPER_MASK_SUFFIX):
        return None
    candidate = annot_dir / f"{image_path.stem}{SUPER_MASK_SUFFIX}{image_path.suffix}"
    return candidate if candidate.is_file() else None


def _ebc_mask(image_path: Path, annot_dir: Path) -> Path | None:
    candidate = annot_dir / image_path.name
    return candidate if candidate.is_file() else None


def discover_pairs(
    dataset: str,
    family: str,
    *,
    data_root: Path | None = None,
    splits: tuple[str, ...] = SPLITS,
) -> list[ImagePair]:
    """Return image/mask pairs for ``dataset`` across the requested ``splits``."""
    if family not in ("super", "ebc"):
        raise ValueError(f"family must be 'super' or 'ebc', got {family!r}")
    dataset_dir = benchmark_root(data_root) / dataset
    pair_fn = _super_mask if family == "super" else _ebc_mask

    pairs: list[ImagePair] = []
    for split in splits:
        annot_dir = dataset_dir / f"{split}{ANNOT_SUFFIX}"
        for image_path in _image_files(dataset_dir / split):
            pairs.append(ImagePair(split, image_path, pair_fn(image_path, annot_dir)))
    return pairs


def split_counts(pairs: list[ImagePair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.split] = counts.get(pair.split, 0) + 1
    return counts
