"""Tests for the torch-free validation guards in create_segmentation_model.

These run without the PyTorch stack because the factory validates its arguments
before the lazy torch/smp import.
"""

from __future__ import annotations

import pytest

from microscopy_analysis.models import create_segmentation_model


def test_rejects_unknown_pretraining():
    with pytest.raises(ValueError, match="pretraining"):
        create_segmentation_model("UnetPlusPlus", "resnet50", "bogus", 3)


def test_rejects_two_classes():
    with pytest.raises(ValueError, match="classes=1"):
        create_segmentation_model("UnetPlusPlus", "resnet50", "micronet", 2)
