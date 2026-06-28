"""Tests for Sprint 1 dataset adapter behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from microscopy_analysis.data.dataset_adapter import decode_ebc_mask, decode_super_mask, list_sample_pairs


def _make_super_fixture(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "Super1"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "train_annot").mkdir(parents=True)
    Image.new("RGB", (3, 3), (128, 128, 128)).save(dataset_dir / "train" / "a.tif")
    Image.fromarray(
        np.array(
            [
                [[0, 0, 0], [255, 0, 0], [0, 0, 255]],
                [[0, 0, 0], [255, 0, 0], [0, 0, 255]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    ).save(dataset_dir / "train_annot" / "a_mask.tif")
    return dataset_dir


def _make_ebc_fixture(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "EBC1"
    (dataset_dir / "train").mkdir(parents=True)
    (dataset_dir / "train_annot").mkdir(parents=True)
    Image.new("L", (3, 3), 0).save(dataset_dir / "train" / "tile.tif")
    Image.fromarray(
        np.array(
            [
                [0, 1, 2],
                [0, 1, 0],
                [2, 0, 0],
            ],
            dtype=np.uint8,
        ),
        mode="L",
    ).save(dataset_dir / "train_annot" / "tile.tif")
    return dataset_dir


def test_super_pairs_and_decode(tmp_path: Path) -> None:
    dataset_dir = _make_super_fixture(tmp_path)
    pairs = list_sample_pairs(dataset_dir, split="train", dataset_family="super")
    assert len(pairs) == 1
    decoded = decode_super_mask(pairs[0].mask_path)
    assert decoded.shape == (3, 3)
    assert int((decoded == 1).sum()) == 2
    assert int((decoded == 2).sum()) == 2


def test_ebc_pairs_and_decode(tmp_path: Path) -> None:
    dataset_dir = _make_ebc_fixture(tmp_path)
    pairs = list_sample_pairs(dataset_dir, split="train", dataset_family="ebc")
    assert len(pairs) == 1
    decoded = decode_ebc_mask(pairs[0].mask_path, positive_values=(1,))
    assert decoded.shape == (3, 3)
    assert int(decoded.sum()) == 2

