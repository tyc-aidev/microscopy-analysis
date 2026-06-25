"""Aggregate dataset statistics for the home page."""

from __future__ import annotations

from typing import Any

import pandas as pd

from explorer.lib.catalog import get_dataset_info, list_benchmark_datasets
from explorer.lib.index import ImageRecord, SPLITS
from explorer.lib.masks import load_class_masks


def image_counts_dataframe(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in list_benchmark_datasets():
        for split, total in sorted(counts.get(dataset, {}).items()):
            rows.append({"dataset": dataset, "split": split, "images": total})
    return pd.DataFrame(rows)


def image_counts_pivot(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    df = image_counts_dataframe(counts)
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot(index="dataset", columns="split", values="images").fillna(0).astype(int)
    ordered_splits = [split for split in SPLITS if split in pivot.columns]
    return pivot[ordered_splits]


def split_summary_table(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in list_benchmark_datasets():
        if dataset not in counts:
            continue
        info = get_dataset_info(dataset)
        ds_counts = counts[dataset]
        for split in SPLITS:
            if split not in ds_counts:
                continue
            notes: list[str] = []
            if info.get("highlight") == "low_data" and split == "train":
                notes.append("low-data train")
            if split == "different_test":
                notes.append("different_test")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "images": ds_counts[split],
                    "notes": ", ".join(notes),
                }
            )
    return pd.DataFrame(rows)


def aggregate_class_pixels(records: list[ImageRecord]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for record in records:
        if record.mask_path is None:
            continue
        class_masks = load_class_masks(record.mask_path, record.family_id)
        dataset_totals = totals.setdefault(record.dataset, {})
        for cls_id, mask in class_masks.items():
            dataset_totals[cls_id] = dataset_totals.get(cls_id, 0) + int(mask.sum())
    return totals


def class_distribution_dataframe(
    class_totals: dict[str, dict[str, int]],
    dataset: str,
) -> pd.DataFrame:
    info = get_dataset_info(dataset)
    labels = {cls["id"]: cls["label"] for cls in info["family"]["classes"]}
    dataset_totals = class_totals.get(dataset, {})
    rows = [
        {"class": labels.get(cls_id, cls_id), "pixels": pixels}
        for cls_id, pixels in sorted(dataset_totals.items())
    ]
    return pd.DataFrame(rows)
