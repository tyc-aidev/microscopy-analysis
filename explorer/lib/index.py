"""Scan DATA_ROOT and index benchmark image/mask pairs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from explorer.lib.catalog import get_family_id_for_dataset, list_benchmark_datasets

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SUPER_SUFFIX = "_mask"
SPLITS = ("train", "val", "test", "different_test")
ANNOT_SUFFIX = "_annot"


@dataclass(frozen=True)
class ImageRecord:
    dataset: str
    family_id: str
    split: str
    image_path: Path
    mask_path: Path | None


def get_data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", "./data")).expanduser().resolve()


def benchmark_root(data_root: Path | None = None) -> Path:
    root = data_root or get_data_root()
    return root / "benchmark_segmentation_data"


def is_data_populated(data_root: Path | None = None) -> bool:
    root = benchmark_root(data_root)
    if not root.is_dir():
        return False
    for dataset in list_benchmark_datasets():
        if (root / dataset / "train").is_dir():
            return True
    return False


def _image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _super_mask_path(image_path: Path, annot_dir: Path) -> Path | None:
    stem = image_path.stem
    if stem.endswith(SUPER_SUFFIX):
        return None
    candidate = annot_dir / f"{stem}{SUPER_SUFFIX}{image_path.suffix}"
    return candidate if candidate.is_file() else None


def _ebc_mask_path(image_path: Path, annot_dir: Path) -> Path | None:
    candidate = annot_dir / image_path.name
    return candidate if candidate.is_file() else None


def pair_split(dataset: str, family_id: str, split: str, dataset_dir: Path) -> list[ImageRecord]:
    image_dir = dataset_dir / split
    annot_dir = dataset_dir / f"{split}{ANNOT_SUFFIX}"
    pair_fn = _super_mask_path if family_id == "super" else _ebc_mask_path

    records: list[ImageRecord] = []
    for image_path in _image_files(image_dir):
        mask_path = pair_fn(image_path, annot_dir)
        records.append(
            ImageRecord(
                dataset=dataset,
                family_id=family_id,
                split=split,
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return records


def scan_benchmarks(data_root: Path | None = None) -> list[ImageRecord]:
    root = benchmark_root(data_root)
    records: list[ImageRecord] = []

    for dataset in list_benchmark_datasets():
        dataset_dir = root / dataset
        if not dataset_dir.is_dir():
            continue
        family_id = get_family_id_for_dataset(dataset)
        for split in SPLITS:
            records.extend(pair_split(dataset, family_id, split, dataset_dir))
    return records


def split_counts(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        counts.setdefault(record.dataset, {})
        counts[record.dataset][record.split] = counts[record.dataset].get(record.split, 0) + 1
    return counts


def records_for(
    records: list[ImageRecord],
    *,
    dataset: str | None = None,
    split: str | None = None,
) -> list[ImageRecord]:
    result = records
    if dataset is not None:
        result = [r for r in result if r.dataset == dataset]
    if split is not None:
        result = [r for r in result if r.split == split]
    return result


def splits_for_dataset(records: list[ImageRecord], dataset: str) -> list[str]:
    present = {r.split for r in records if r.dataset == dataset}
    return [split for split in SPLITS if split in present]
