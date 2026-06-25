"""Tests for MicroNet v1.0 weight-URL pinning (the key reproduction pitfall)."""

from __future__ import annotations

import pytest

from amat.models.weights import S3_BASE, micronet_weight_url, normalize_version


def test_resnet50_micronet_pins_v1_0_by_default():
    url = micronet_weight_url("resnet50", "micronet")
    assert url == f"{S3_BASE}resnet50_pretrained_microscopynet_v1.0.pth.tar"


def test_image_micronet_alias():
    url = micronet_weight_url("resnet50", "image-micronet")
    assert url == f"{S3_BASE}resnet50_pretrained_imagenet-microscopynet_v1.0.pth.tar"


def test_se_resnext_default_v1_0():
    url = micronet_weight_url("se_resnext50_32x4d", "micronet")
    assert url == f"{S3_BASE}se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar"


def test_resnext101_special_case():
    url = micronet_weight_url("resnext101_32x8d", "micronet")
    assert url == f"{S3_BASE}resnext101_pretrained_microscopynet_v1.0.pth.tar"


def test_non_paper_version_rejected():
    with pytest.raises(ValueError, match="non-paper"):
        micronet_weight_url("resnet50", "micronet", version=1.1)


def test_non_paper_version_allowed_when_opted_in():
    url = micronet_weight_url("resnet50", "micronet", version=1.1, allow_non_paper_version=True)
    assert url.endswith("_v1.1.pth.tar")


def test_invalid_encoder_weights():
    with pytest.raises(ValueError, match="micronet"):
        micronet_weight_url("resnet50", "imagenet")


def test_self_supervision_forces_v1_0_token():
    url = micronet_weight_url("resnet50", "micronet", self_supervision="moco")
    assert url == f"{S3_BASE}resnet50_moco_pretrained_microscopynet_v1.0.pth.tar"


@pytest.mark.parametrize("value,expected", [(1.0, "1.0"), ("1.0", "1.0"), (1, "1.0"), (1.1, "1.1")])
def test_normalize_version(value, expected):
    assert normalize_version(value) == expected
