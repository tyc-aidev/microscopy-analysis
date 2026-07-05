"""Tests for explorer/lib training run scanning and charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

from explorer.lib.prediction_panels import list_panels
from explorer.lib.runs import (
    filter_runs,
    find_config_for_run,
    load_metrics,
    scan_runs,
)
from explorer.lib.training_charts import plot_iou_curves, plot_loss_curves, plot_per_class_iou


def _sample_metrics() -> list[dict]:
    return [
        {
            "epoch": 1,
            "phase": 1,
            "lr": 0.0002,
            "train_loss": 0.7,
            "val_loss": 0.8,
            "val_iou_per_class": [0.33, 0.03, 0.30],
            "val_mean_iou": 0.22,
            "val_score": 0.22,
        },
        {
            "epoch": 2,
            "phase": 2,
            "lr": 1e-05,
            "train_loss": 0.65,
            "val_loss": 0.75,
            "val_iou_per_class": [0.38, 0.01, 0.47],
            "val_mean_iou": 0.29,
            "val_score": 0.29,
        },
    ]


def _write_run(root: Path, run_name: str, *, device: str) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    summary = {
        "run_name": run_name,
        "dataset_name": "Super1",
        "dataset_family": "super",
        "device": device,
        "best_mean_iou": 0.29,
        "best_epoch": 2,
        "epochs_trained": 2,
        "git_sha": "abc123def456",
        "num_samples": 10,
        "num_val_samples": 4,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary))
    (run_dir / "metrics.json").write_text(json.dumps(_sample_metrics()))
    (run_dir / "model_best.pth").write_bytes(b"stub")
    pred_dir = run_dir / "predictions" / "val"
    pred_dir.mkdir(parents=True)
    (pred_dir / "img_panel.png").write_bytes(b"not-a-png")
    return run_dir


def test_scan_runs_discovers_summaries(tmp_path: Path) -> None:
    _write_run(tmp_path, "local_run", device="mps")
    _write_run(tmp_path, "cuda_run", device="cuda")

    runs = scan_runs(tmp_path)
    assert len(runs) == 2
    names = {r.run_name for r in runs}
    assert names == {"local_run", "cuda_run"}
    local = next(r for r in runs if r.run_name == "local_run")
    assert local.has_metrics
    assert local.has_best
    assert local.prediction_splits == ("val",)


def test_filter_runs_by_track(tmp_path: Path) -> None:
    _write_run(tmp_path, "local_run", device="mps")
    _write_run(tmp_path, "cpu_run", device="cpu")
    _write_run(tmp_path, "cuda_run", device="cuda")

    runs = scan_runs(tmp_path)
    assert len(filter_runs(runs, "local")) == 2
    assert len(filter_runs(runs, "cuda")) == 1
    assert filter_runs(runs, "cuda")[0].run_name == "cuda_run"


def test_find_config_for_run(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "super1_baseline.yaml"
    cfg_path.write_text(yaml.dump({"run_name": "my_run", "data_root": "data"}))

    found = find_config_for_run("my_run", cfg_dir)
    assert found == cfg_path
    assert find_config_for_run("missing", cfg_dir) is None


def test_load_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_sample_metrics()))
    metrics = load_metrics(path)
    assert len(metrics) == 2
    assert metrics[0]["epoch"] == 1


def test_training_charts_smoke() -> None:
    metrics = _sample_metrics()
    loss_fig = plot_loss_curves(metrics)
    iou_fig = plot_iou_curves(metrics, dataset_family="super")
    per_class_fig = plot_per_class_iou(metrics, dataset_family="super")
    assert loss_fig is not None
    assert iou_fig is not None
    assert per_class_fig is not None


def test_list_panels(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "panel_run", device="mps")
    panels = list_panels(run_dir, "val")
    assert len(panels) == 1
    assert panels[0].name == "img_panel.png"
    assert list_panels(run_dir, "train") == []
