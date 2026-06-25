"""Cached data loaders for Streamlit pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from explorer.lib.index import ImageRecord, get_data_root, scan_benchmarks

RecordTuple = tuple[str, str, str, str, str | None]


def _serialize(records: list[ImageRecord]) -> tuple[RecordTuple, ...]:
    return tuple(
        (
            r.dataset,
            r.family_id,
            r.split,
            str(r.image_path),
            str(r.mask_path) if r.mask_path else None,
        )
        for r in records
    )


def deserialize_records(rows: tuple[RecordTuple, ...]) -> list[ImageRecord]:
    return [
        ImageRecord(
            dataset=dataset,
            family_id=family_id,
            split=split,
            image_path=Path(image_path),
            mask_path=Path(mask_path) if mask_path else None,
        )
        for dataset, family_id, split, image_path, mask_path in rows
    ]


@st.cache_data(show_spinner="Scanning benchmark images...")
def cached_benchmark_records(data_root: str) -> tuple[RecordTuple, ...]:
    return _serialize(scan_benchmarks(Path(data_root)))


def load_benchmark_records() -> list[ImageRecord]:
    return deserialize_records(cached_benchmark_records(str(get_data_root())))


@st.cache_data
def cached_pil_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.copy()


@st.cache_data(show_spinner="Loading COCO annotations...")
def cached_coco_split(instance_root: str, split: str) -> tuple[str, tuple, dict[int, tuple], tuple]:
    from explorer.lib.coco import load_coco_split

    index = load_coco_split(Path(instance_root), split)
    image_rows = tuple(
        (
            img.image_id,
            img.split,
            img.file_name,
            str(img.image_path),
            img.width,
            img.height,
        )
        for img in index.images
    )
    ann_rows: dict[int, tuple] = {}
    for image_id, anns in index.annotations_by_image.items():
        ann_rows[image_id] = tuple(
            (a.id, a.category_id, a.category_name, a.bbox, a.segmentation, a.area) for a in anns
        )
    return (index.split, image_rows, ann_rows, index.categories)


def load_coco_split_index(instance_root: Path, split: str):
    from explorer.lib.coco import CocoAnnotation, CocoImageRecord, CocoSplitIndex

    split_name, image_rows, ann_rows, categories = cached_coco_split(str(instance_root), split)
    images = tuple(
        CocoImageRecord(
            image_id=row[0],
            split=row[1],
            file_name=row[2],
            image_path=Path(row[3]),
            width=row[4],
            height=row[5],
        )
        for row in image_rows
    )
    annotations_by_image = {
        image_id: tuple(
            CocoAnnotation(
                id=row[0],
                category_id=row[1],
                category_name=row[2],
                bbox=row[3],
                segmentation=row[4],
                area=row[5],
            )
            for row in rows
        )
        for image_id, rows in ann_rows.items()
    }
    return CocoSplitIndex(
        split=split_name,
        images=images,
        annotations_by_image=annotations_by_image,
        categories=categories,
    )
