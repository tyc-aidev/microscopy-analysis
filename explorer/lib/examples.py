"""Scan example/reference assets listed in catalog.json."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from explorer.lib.catalog import load_catalog
from explorer.lib.index import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class GalleryItem:
    path: Path
    caption: str
    section: str


def examples_root(data_root: Path) -> Path:
    return data_root / "examples"


def is_examples_data_populated(data_root: Path) -> bool:
    root = examples_root(data_root)
    if not root.is_dir():
        return False
    return (root / "npg.png").is_file() or (root / "dog.jpeg").is_file()


def list_example_items(examples_dir: Path) -> list[GalleryItem]:
    catalog = load_catalog()["examples"]
    items: list[GalleryItem] = []

    for entry in catalog["paths"]:
        section = entry["caption"]
        if "file" in entry:
            path = examples_dir / entry["file"]
            if path.is_file():
                items.append(GalleryItem(path=path, caption=entry["caption"], section=section))
            continue

        if "dir" not in entry:
            continue

        dir_path = examples_dir / entry["dir"]
        if not dir_path.is_dir():
            continue

        for path in sorted(dir_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append(
                    GalleryItem(
                        path=path,
                        caption=f"{entry['caption']} — {path.name}",
                        section=section,
                    )
                )

    return items
