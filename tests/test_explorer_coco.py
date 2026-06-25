"""Tests for COCO parsing and examples scanning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from explorer.lib.coco import (
    load_coco_split,
    render_annotations,
    resolve_image_path,
    split_summary,
)
from explorer.lib.examples import list_example_items


@pytest.fixture
def coco_fixture(tmp_path: Path) -> Path:
    instance_root = tmp_path / "instance_segmentation"
    train_dir = instance_root / "train"
    annot_dir = instance_root / "annotations"
    train_dir.mkdir(parents=True)
    annot_dir.mkdir(parents=True)

    Image.new("RGB", (64, 64), (100, 100, 100)).save(train_dir / "tile_a.png")

    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "./data/train/tile_a.png",
                "width": 64,
                "height": 64,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10.0, 10.0, 20.0, 15.0],
                "area": 300.0,
                "segmentation": [[10, 10, 30, 10, 30, 25, 10, 25]],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 0, "name": "melt pool", "supercategory": "melt pool"}],
    }
    (annot_dir / "train.json").write_text(json.dumps(payload), encoding="utf-8")
    return instance_root


@pytest.fixture
def examples_fixture(tmp_path: Path) -> Path:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(examples_dir / "npg.png")
    Image.new("RGB", (8, 8), (0, 255, 0)).save(examples_dir / "dog.jpeg")

    heatmaps = examples_dir / "regression-and-xai" / "sample_heatmaps"
    heatmaps.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(heatmaps / "sample1.png")
    return examples_dir


def test_resolve_image_path(coco_fixture: Path) -> None:
    path = resolve_image_path(coco_fixture, "./data/train/tile_a.png", "train")
    assert path == coco_fixture / "train" / "tile_a.png"
    assert path.is_file()


def test_load_coco_split(coco_fixture: Path) -> None:
    index = load_coco_split(coco_fixture, "train")
    assert index.image_count == 1
    assert index.annotation_count == 1
    summary = split_summary(index)
    assert summary["images"] == 1
    assert summary["annotations"] == 1
    assert summary["bbox_area_mean"] == 300.0


def test_render_annotations(coco_fixture: Path) -> None:
    index = load_coco_split(coco_fixture, "train")
    image = Image.open(index.images[0].image_path)
    anns = index.annotations_by_image[index.images[0].image_id]
    rendered = render_annotations(image, anns, show_bbox=True, show_mask=True)
    assert rendered.mode == "RGBA"
    assert rendered.size == image.size


def test_list_example_items(examples_fixture: Path) -> None:
    items = list_example_items(examples_fixture)
    names = {item.path.name for item in items}
    assert "npg.png" in names
    assert "dog.jpeg" in names
    assert "sample1.png" in names
