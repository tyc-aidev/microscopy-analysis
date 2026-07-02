"""Tests for the Sprint 3 low-data ablation sweep generator."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from microscopy_analysis.orchestration.low_data import (
    DEFAULT_SIZES,
    LOW_DATA_PRETRAININGS,
    SUPER_DATASETS,
    generate_low_data_jobs,
    render_config,
    resolve_sizes,
    write_configs,
    write_manifest,
)
from microscopy_analysis.train.config import load_train_config


def test_super_datasets_are_the_four_multiclass_benchmarks() -> None:
    assert {d.name for d in SUPER_DATASETS} == {"Super1", "Super2", "Super3", "Super4"}
    assert all(d.family == "super" and d.num_classes == 3 for d in SUPER_DATASETS)


def test_resolve_sizes_clamps_and_dedupes_with_known_count() -> None:
    # Super3 has a single training image -> the whole sweep collapses to {1}.
    assert resolve_sizes(DEFAULT_SIZES, 1) == [1]
    # 4 images: 8 clamps to 4 and "all" (None) also becomes 4 -> {1,2,4}.
    assert resolve_sizes(DEFAULT_SIZES, 4) == [1, 2, 4]
    # 10 images: "all" becomes the concrete 10 -> {1,2,4,8,10}.
    assert resolve_sizes(DEFAULT_SIZES, 10) == [1, 2, 4, 8, 10]


def test_resolve_sizes_without_count_keeps_none_sentinel() -> None:
    assert resolve_sizes((1, 2, None), None) == [1, 2, None]
    assert resolve_sizes((4, 2), None) == [2, 4]


def test_generate_jobs_covers_sizes_and_regimes() -> None:
    counts = {"Super1": 10, "Super2": 4, "Super3": 1, "Super4": 4}
    jobs = generate_low_data_jobs(train_counts=counts)
    # sizes per dataset: Super1 {1,2,4,8,10}=5, Super2/4 {1,2,4}=3, Super3 {1}=1.
    expected_sizes = 5 + 3 + 1 + 3
    assert len(jobs) == expected_sizes * len(LOW_DATA_PRETRAININGS)
    assert len({j.run_name for j in jobs}) == len(jobs)  # unique run names
    assert {j.pretraining for j in jobs} == set(LOW_DATA_PRETRAININGS)
    # Super3's single job pair uses the whole (1-image) split.
    super3 = [j for j in jobs if j.dataset_name == "Super3"]
    assert {j.train_subsample for j in super3} == {1}


def test_run_name_encodes_size_and_all_label() -> None:
    jobs = generate_low_data_jobs(train_counts=None)  # None sizes stay as "all"
    all_jobs = [j for j in jobs if j.train_subsample is None]
    assert all_jobs and all(j.run_name.endswith(f"_nall_seed{j.seed}") for j in all_jobs)
    numbered = [j for j in jobs if j.train_subsample == 4]
    assert numbered and all("_n4_" in j.run_name for j in numbered)


def test_render_config_roundtrips_train_subsample(tmp_path: Path) -> None:
    job = next(j for j in generate_low_data_jobs(train_counts={"Super1": 10}) if j.train_subsample == 4)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(render_config(job), sort_keys=False))
    cfg = load_train_config(path, base_dir=tmp_path)
    assert cfg.train_subsample == 4
    assert cfg.dataset_name == job.dataset_name
    assert cfg.num_classes == 3


def test_render_config_omits_subsample_for_full_split(tmp_path: Path) -> None:
    job = next(j for j in generate_low_data_jobs(train_counts=None) if j.train_subsample is None)
    cfg_dict = render_config(job)
    assert "train_subsample" not in cfg_dict["trainer"]
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    assert load_train_config(path, base_dir=tmp_path).train_subsample is None


def test_write_manifest_and_configs(tmp_path: Path) -> None:
    jobs = generate_low_data_jobs(train_counts={"Super1": 10, "Super2": 4, "Super3": 1, "Super4": 4})
    manifest = write_manifest(jobs, tmp_path / "manifest.json")
    loaded = json.loads(manifest.read_text())
    assert len(loaded) == len(jobs)
    assert "train_subsample" in loaded[0]

    paths = write_configs(jobs, tmp_path / "configs")
    assert len(paths) == len(jobs)
    assert all(p.exists() and p.suffix == ".yaml" for p in paths)
