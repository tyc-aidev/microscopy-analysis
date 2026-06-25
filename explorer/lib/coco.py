"""Parse COCO JSON and render instance-segmentation overlays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from explorer.lib.catalog import load_catalog


@dataclass(frozen=True)
class CocoAnnotation:
    id: int
    category_id: int
    category_name: str
    bbox: tuple[float, float, float, float]
    segmentation: tuple[tuple[float, ...], ...]
    area: float


@dataclass(frozen=True)
class CocoImageRecord:
    image_id: int
    split: str
    file_name: str
    image_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class CocoSplitIndex:
    split: str
    images: tuple[CocoImageRecord, ...]
    annotations_by_image: dict[int, tuple[CocoAnnotation, ...]]
    categories: tuple[dict[str, Any], ...]

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def annotation_count(self) -> int:
        return sum(len(v) for v in self.annotations_by_image.values())

    def bbox_areas(self) -> list[float]:
        areas: list[float] = []
        for anns in self.annotations_by_image.values():
            for ann in anns:
                _, _, w, h = ann.bbox
                areas.append(w * h)
        return areas


def instance_seg_root(data_root: Path) -> Path:
    return data_root / "instance_segmentation"


def is_instance_data_populated(data_root: Path) -> bool:
    root = instance_seg_root(data_root)
    return (root / "annotations" / "train.json").is_file()


def annotation_json_path(instance_root: Path, split: str) -> Path:
    catalog = load_catalog()["instance_segmentation"]["annotation_files"]
    return instance_root / catalog[split]


def resolve_image_path(instance_root: Path, file_name: str, split: str) -> Path:
    basename = Path(file_name).name
    return instance_root / split / basename


def _parse_segmentation(raw: list[Any]) -> tuple[tuple[float, ...], ...]:
    polygons: list[tuple[float, ...]] = []
    for item in raw:
        if isinstance(item, list):
            polygons.append(tuple(float(v) for v in item))
    return tuple(polygons)


def load_coco_split(instance_root: Path, split: str) -> CocoSplitIndex:
    json_path = annotation_json_path(instance_root, split)
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    categories = {cat["id"]: cat for cat in payload.get("categories", [])}
    annotations_by_image: dict[int, list[CocoAnnotation]] = {}

    for raw_ann in payload.get("annotations", []):
        image_id = int(raw_ann["image_id"])
        category = categories.get(int(raw_ann["category_id"]), {})
        ann = CocoAnnotation(
            id=int(raw_ann["id"]),
            category_id=int(raw_ann["category_id"]),
            category_name=str(category.get("name", "unknown")),
            bbox=tuple(float(v) for v in raw_ann["bbox"]),
            segmentation=_parse_segmentation(raw_ann.get("segmentation", [])),
            area=float(raw_ann.get("area", 0.0)),
        )
        annotations_by_image.setdefault(image_id, []).append(ann)

    images: list[CocoImageRecord] = []
    for raw_image in payload.get("images", []):
        image_id = int(raw_image["id"])
        file_name = str(raw_image["file_name"])
        image_path = resolve_image_path(instance_root, file_name, split)
        images.append(
            CocoImageRecord(
                image_id=image_id,
                split=split,
                file_name=file_name,
                image_path=image_path,
                width=int(raw_image["width"]),
                height=int(raw_image["height"]),
            )
        )

    images.sort(key=lambda record: record.image_path.name)
    frozen_annotations = {
        image_id: tuple(anns) for image_id, anns in annotations_by_image.items()
    }
    return CocoSplitIndex(
        split=split,
        images=tuple(images),
        annotations_by_image=frozen_annotations,
        categories=tuple(payload.get("categories", [])),
    )


def split_summary(index: CocoSplitIndex) -> dict[str, float | int]:
    areas = index.bbox_areas()
    if not areas:
        return {
            "images": index.image_count,
            "annotations": index.annotation_count,
            "bbox_area_min": 0,
            "bbox_area_mean": 0,
            "bbox_area_max": 0,
        }
    return {
        "images": index.image_count,
        "annotations": index.annotation_count,
        "bbox_area_min": min(areas),
        "bbox_area_mean": mean(areas),
        "bbox_area_max": max(areas),
    }


def bbox_area_histogram(areas: list[float], bins: int = 10) -> pd.DataFrame:
    """Histogram of bbox areas with string bin labels (Streamlit-compatible)."""
    if not areas:
        return pd.DataFrame({"count": []})
    counts, edges = np.histogram(areas, bins=bins)
    labels = [f"{edges[i]:.0f}–{edges[i + 1]:.0f}" for i in range(len(counts))]
    return pd.DataFrame({"count": counts}, index=labels)


def _polygon_points(flat: tuple[float, ...]) -> list[tuple[float, float]]:
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]


def render_annotations(
    image: Image.Image,
    annotations: tuple[CocoAnnotation, ...],
    *,
    show_bbox: bool = True,
    show_mask: bool = True,
    mask_alpha: int = 100,
) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in annotations:
        if show_mask:
            for polygon in ann.segmentation:
                points = _polygon_points(polygon)
                if len(points) >= 3:
                    draw.polygon(
                        points,
                        fill=(255, 128, 0, mask_alpha),
                        outline=(255, 128, 0, 255),
                    )
        if show_bbox:
            x, y, w, h = ann.bbox
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=(0, 255, 255, 255),
                width=2,
            )

    return Image.alpha_composite(base, overlay)
