"""Tests for explorer/lib/remote_results slim archive fetching."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from explorer.lib import remote_results
from explorer.lib.remote_data import READY_MARKER, RemoteConfig
from explorer.lib.remote_results import (
    is_results_populated,
    resolve_results_remote_config,
)


_ENV_KEYS = (
    "RESULTS_ARCHIVE_URL",
    "REMOTE_RESULTS_ROOT",
    "RESULTS_ROOT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _make_slim_run(root: Path, run_name: str = "super1_baseline") -> None:
    run_dir = root / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "device": "mps",
                "best_mean_iou": 0.9,
                "best_epoch": 1,
                "epochs_trained": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("[]", encoding="utf-8")
    panels = run_dir / "predictions" / "val"
    panels.mkdir(parents=True)
    (panels / "sample_panel.png").write_bytes(b"fake-png")


def _tar_gz(src: Path, out: Path) -> Path:
    with tarfile.open(out, "w:gz") as tar:
        for entry in src.iterdir():
            tar.add(entry, arcname=entry.name)
    return out


def test_resolve_results_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESULTS_ARCHIVE_URL", "https://cdn.example.com/amat-results-slim.tar.zst")
    cfg = resolve_results_remote_config()
    assert cfg == RemoteConfig(url="https://cdn.example.com/amat-results-slim.tar.zst")


def test_resolve_results_unconfigured_is_none() -> None:
    assert resolve_results_remote_config() is None


def test_is_results_populated(tmp_path: Path) -> None:
    assert not is_results_populated(tmp_path)
    _make_slim_run(tmp_path)
    assert is_results_populated(tmp_path)


def test_ensure_results_skips_when_local_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_slim_run(tmp_path)
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path))
    assert remote_results._ensure_results() == tmp_path.resolve()


def test_ensure_results_none_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "empty"))
    assert remote_results._ensure_results() is None


def test_ensure_results_downloads_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    _make_slim_run(payload)
    archive = _tar_gz(payload, tmp_path / "results.tar.gz")

    root = tmp_path / "remote-results"
    monkeypatch.setenv("RESULTS_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("REMOTE_RESULTS_ROOT", str(root))
    monkeypatch.setenv("RESULTS_ARCHIVE_URL", archive.as_uri())

    result = remote_results._ensure_results()
    assert result == root
    assert (root / READY_MARKER).is_file()
    assert (root / "super1_baseline" / "run_summary.json").is_file()
    assert (root / "super1_baseline" / "predictions" / "val" / "sample_panel.png").is_file()
    import os

    assert os.environ["RESULTS_ROOT"] == str(root)
