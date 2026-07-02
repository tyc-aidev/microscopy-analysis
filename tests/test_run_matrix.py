"""Tests for the Sprint 2 benchmark-matrix generator."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from microscopy_analysis.orchestration import (
    DATASETS,
    ENCODERS,
    PRETRAININGS,
    generate_jobs,
    render_config,
    write_configs,
    write_manifest,
)
from microscopy_analysis.train.config import load_train_config


def test_generate_jobs_produces_full_cross_product() -> None:
    jobs = generate_jobs()
    assert len(jobs) == len(DATASETS) * len(ENCODERS) * len(PRETRAININGS) == 56
    assert len({j.run_name for j in jobs}) == len(jobs)  # unique run names
    assert {j.pretraining for j in jobs} == set(PRETRAININGS)
    assert {j.encoder_name for j in jobs} == set(ENCODERS)
    assert all(j.architecture == "UnetPlusPlus" for j in jobs)


def test_jobs_carry_correct_class_counts_per_family() -> None:
    jobs = generate_jobs()
    for j in jobs:
        if j.dataset_family == "super":
            assert j.num_classes == 3
        else:
            assert j.dataset_family == "ebc" and j.num_classes == 1


def test_render_config_is_loadable_train_config(tmp_path: Path) -> None:
    job = next(j for j in generate_jobs() if j.dataset_family == "ebc")
    cfg_dict = render_config(job)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))

    cfg = load_train_config(path, base_dir=tmp_path)
    assert cfg.run_name == job.run_name
    assert cfg.dataset_family == "ebc"
    assert cfg.num_classes == 1
    assert cfg.lr_phase1 == 2e-4 and cfg.lr_phase2 == 1e-5


def test_write_manifest_and_configs_roundtrip(tmp_path: Path) -> None:
    jobs = generate_jobs()
    manifest = write_manifest(jobs, tmp_path / "manifest.json")
    loaded = json.loads(manifest.read_text())
    assert len(loaded) == 56
    assert loaded[0]["run_name"] == jobs[0].run_name

    paths = write_configs(jobs, tmp_path / "configs")
    assert len(paths) == 56
    assert all(p.exists() and p.suffix == ".yaml" for p in paths)
