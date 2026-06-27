"""Tests for Super/EBC image-mask pairing (reuses conftest benchmark_fixture)."""

from __future__ import annotations

from pathlib import Path

import pytest

from amat.data import discover_pairs, split_counts


def test_super_pairs_via_mask_suffix(benchmark_fixture: Path):
    pairs = discover_pairs("Super1", "super", data_root=benchmark_fixture)
    assert split_counts(pairs) == {"train": 1}
    pair = pairs[0]
    assert pair.mask_path is not None
    assert pair.mask_path.name == "sample_mask.tif"


def test_ebc_pairs_via_same_name(benchmark_fixture: Path):
    pairs = discover_pairs("EBC1", "ebc", data_root=benchmark_fixture)
    assert split_counts(pairs) == {"train": 1}
    pair = pairs[0]
    assert pair.mask_path is not None
    assert pair.mask_path.name == pair.image_path.name == "tile.tif"


def test_missing_dataset_returns_empty(benchmark_fixture: Path):
    assert discover_pairs("Super4", "super", data_root=benchmark_fixture) == []


def test_invalid_family_rejected(benchmark_fixture: Path):
    with pytest.raises(ValueError, match="family"):
        discover_pairs("Super1", "bogus", data_root=benchmark_fixture)
