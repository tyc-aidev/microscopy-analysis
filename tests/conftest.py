"""Shared pytest fixtures for explorer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def benchmark_fixture(tmp_path: Path) -> Path:
    super_dir = tmp_path / "benchmark_segmentation_data" / "Super1"
    (super_dir / "train").mkdir(parents=True)
    (super_dir / "train_annot").mkdir(parents=True)
    Image.new("RGB", (4, 4), (128, 128, 128)).save(super_dir / "train" / "sample.tif")
    Image.new("RGB", (4, 4), (0, 0, 0)).save(super_dir / "train_annot" / "sample_mask.tif")

    ebc_dir = tmp_path / "benchmark_segmentation_data" / "EBC1"
    (ebc_dir / "train").mkdir(parents=True)
    (ebc_dir / "train_annot").mkdir(parents=True)
    Image.new("L", (4, 4), 0).save(ebc_dir / "train" / "tile.tif")
    Image.fromarray(np.array([[0, 1], [2, 0]], dtype=np.uint8)).resize((4, 4)).save(
        ebc_dir / "train_annot" / "tile.tif"
    )
    return tmp_path
