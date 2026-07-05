"""Scan and parse training run artifacts under RESULTS_ROOT (torch-free)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Track = Literal["local", "cuda"]
LOCAL_DEVICES = frozenset({"mps", "cpu"})


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    run_dir: Path
    summary: dict[str, Any]
    has_metrics: bool
    has_checkpoint: bool
    has_best: bool
    prediction_splits: tuple[str, ...]

    @property
    def device(self) -> str:
        return str(self.summary.get("device", "unknown"))

    @property
    def dataset_name(self) -> str:
        return str(self.summary.get("dataset_name", ""))

    @property
    def best_mean_iou(self) -> float:
        return float(self.summary.get("best_mean_iou", 0.0))

    @property
    def best_epoch(self) -> int:
        return int(self.summary.get("best_epoch", 0))

    @property
    def epochs_trained(self) -> int:
        return int(self.summary.get("epochs_trained", 0))

    @property
    def git_sha(self) -> str:
        return str(self.summary.get("git_sha", "unknown"))


def get_results_root() -> Path:
    return Path(os.environ.get("RESULTS_ROOT", "./results")).expanduser().resolve()


def get_configs_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    return root / "configs" / "experiments"


def load_metrics(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected metrics array in {path}")
    return raw


def load_run_summary(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected summary object in {path}")
    return raw


def _prediction_splits(run_dir: Path) -> tuple[str, ...]:
    pred_root = run_dir / "predictions"
    if not pred_root.is_dir():
        return ()
    return tuple(sorted(d.name for d in pred_root.iterdir() if d.is_dir()))


def scan_runs(root: Path | None = None) -> list[RunInfo]:
    """Discover runs that have a ``run_summary.json`` under *root*."""
    results_root = root or get_results_root()
    if not results_root.is_dir():
        return []

    runs: list[RunInfo] = []
    for summary_path in sorted(results_root.glob("*/run_summary.json")):
        run_dir = summary_path.parent
        try:
            summary = load_run_summary(summary_path)
        except (json.JSONDecodeError, ValueError):
            continue
        run_name = str(summary.get("run_name", run_dir.name))
        runs.append(
            RunInfo(
                run_name=run_name,
                run_dir=run_dir,
                summary=summary,
                has_metrics=(run_dir / "metrics.json").is_file(),
                has_checkpoint=(run_dir / "checkpoint.pth").is_file(),
                has_best=(run_dir / "model_best.pth").is_file(),
                prediction_splits=_prediction_splits(run_dir),
            )
        )
    return sorted(runs, key=lambda r: r.run_name)


def filter_runs(runs: list[RunInfo], track: Track) -> list[RunInfo]:
    if track == "local":
        return [r for r in runs if r.device in LOCAL_DEVICES]
    return [r for r in runs if r.device == "cuda"]


def find_config_for_run(run_name: str, configs_dir: Path | None = None) -> Path | None:
    """Return the experiment YAML whose ``run_name`` matches *run_name*."""
    cfg_dir = configs_dir or get_configs_dir()
    if not cfg_dir.is_dir():
        return None
    for path in sorted(cfg_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(raw, dict) and raw.get("run_name") == run_name:
            return path
    return None


def list_config_files(configs_dir: Path | None = None) -> list[Path]:
    cfg_dir = configs_dir or get_configs_dir()
    if not cfg_dir.is_dir():
        return []
    return sorted(cfg_dir.glob("*.yaml"))


def count_runs(root: Path | None = None) -> tuple[int, int]:
    """Return (local_count, cuda_count) for status chips."""
    runs = scan_runs(root)
    return len(filter_runs(runs, "local")), len(filter_runs(runs, "cuda"))
