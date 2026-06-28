"""Tests for experiment config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from microscopy_analysis.config import load_experiment_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiments"


def test_load_super1_smoke():
    cfg = load_experiment_config(CONFIG_DIR / "super1_smoke.yaml")
    assert cfg.name == "super1_smoke"
    assert cfg.dataset.family == "super"
    assert cfg.dataset.classes == 3
    assert cfg.model.architecture == "UnetPlusPlus"
    assert cfg.model.encoder == "resnet50"
    assert cfg.model.micronet_version == "1.0"
    assert cfg.training.lr == pytest.approx(2e-4)


def test_load_ebc1_smoke():
    cfg = load_experiment_config(CONFIG_DIR / "ebc1_smoke.yaml")
    assert cfg.dataset.family == "ebc"
    assert cfg.dataset.classes == 1
    assert cfg.model.encoder == "se_resnext50_32x4d"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_rejects_binary_as_two_classes(tmp_path: Path):
    body = """
name: bad
dataset: {name: EBC1, family: ebc, classes: 2, class_values: {oxide: [1]}}
model: {architecture: Unet, encoder: resnet50, pretraining: micronet}
"""
    with pytest.raises(ValueError, match="classes=1"):
        load_experiment_config(_write(tmp_path, body))


def test_rejects_unknown_pretraining(tmp_path: Path):
    body = """
name: bad
dataset: {name: Super1, family: super, classes: 3, class_values: {matrix: [0,0,0]}}
model: {architecture: Unet, encoder: resnet50, pretraining: bogus}
"""
    with pytest.raises(ValueError, match="pretraining"):
        load_experiment_config(_write(tmp_path, body))


def test_micronet_version_normalized(tmp_path: Path):
    body = """
name: norm
dataset: {name: Super1, family: super, classes: 3, class_values: {matrix: [0,0,0]}}
model: {architecture: Unet, encoder: resnet50, pretraining: micronet, micronet_version: 1.0}
"""
    cfg = load_experiment_config(_write(tmp_path, body))
    assert cfg.model.micronet_version == "1.0"


def test_missing_key_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="Missing required config key"):
        load_experiment_config(_write(tmp_path, "name: x\n"))
