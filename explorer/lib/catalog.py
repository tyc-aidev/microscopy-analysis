"""Load curated dataset metadata from catalog.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "datasets" / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Any]:
    with _CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_family_id_for_dataset(dataset: str) -> str:
    catalog = load_catalog()
    return catalog["datasets"][dataset]["family"]


def get_family(family_id: str) -> dict[str, Any]:
    catalog = load_catalog()
    for family in catalog["families"]:
        if family["id"] == family_id:
            return family
    raise KeyError(f"Unknown family: {family_id}")


def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    catalog = load_catalog()
    if dataset_id not in catalog["datasets"]:
        raise KeyError(f"Unknown dataset: {dataset_id}")
    info = dict(catalog["datasets"][dataset_id])
    info["id"] = dataset_id
    info["family"] = get_family(info["family"])
    return info


def list_benchmark_datasets() -> list[str]:
    catalog = load_catalog()
    return list(catalog["datasets"].keys())
