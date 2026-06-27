"""Tests for explorer/lib/remote_data remote dataset fetching."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from explorer.lib import remote_data
from explorer.lib.remote_data import (
    RemoteConfig,
    extract_archive,
    populate_from_remote,
    resolve_remote_config,
)

_R2_KEYS = (
    "DATA_ARCHIVE_URL",
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_OBJECT_KEY",
    "REMOTE_DATA_ROOT",
    "DATA_ROOT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _R2_KEYS:
        monkeypatch.delenv(key, raising=False)


def _make_payload(root: Path) -> None:
    bench = root / "benchmark_segmentation_data" / "Super1" / "train"
    bench.mkdir(parents=True)
    (bench / "sample.tif").write_bytes(b"fake-tiff")


def _tar_gz(src: Path, out: Path) -> Path:
    with tarfile.open(out, "w:gz") as tar:
        for entry in src.iterdir():
            tar.add(entry, arcname=entry.name)
    return out


def _zip(src: Path, out: Path) -> Path:
    with zipfile.ZipFile(out, "w") as zf:
        for path in src.rglob("*"):
            zf.write(path, path.relative_to(src))
    return out


def test_archive_name_from_url() -> None:
    cfg = RemoteConfig(url="https://cdn.example.com/path/amat-data-sample.tar.zst?v=1")
    assert cfg.archive_name == "amat-data-sample.tar.zst"


def test_archive_name_from_object_key() -> None:
    cfg = RemoteConfig(object_key="dir/data.tar.gz")
    assert cfg.archive_name == "data.tar.gz"


def test_resolve_config_prefers_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ARCHIVE_URL", "https://x/data.zip")
    monkeypatch.setenv("R2_ENDPOINT", "https://r2")
    cfg = resolve_remote_config()
    assert cfg == RemoteConfig(url="https://x/data.zip")


def test_resolve_config_r2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ENDPOINT", "https://r2")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "microscopy-analysis-datasets")
    monkeypatch.setenv("R2_OBJECT_KEY", "data.tar.zst")
    cfg = resolve_remote_config()
    assert cfg is not None and cfg.bucket == "microscopy-analysis-datasets" and cfg.url is None


def test_resolve_config_incomplete_r2_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("R2_ENDPOINT", "https://r2")
    assert resolve_remote_config() is None


def test_extract_tar_gz(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    _make_payload(payload)
    archive = _tar_gz(payload, tmp_path / "data.tar.gz")

    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "benchmark_segmentation_data" / "Super1" / "train" / "sample.tif").is_file()


def test_extract_zip(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    _make_payload(payload)
    archive = _zip(payload, tmp_path / "data.zip")

    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "benchmark_segmentation_data" / "Super1" / "train" / "sample.tif").is_file()


def test_extract_rejects_zip_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "nope")

    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out")


def test_extract_tar_zst(tmp_path: Path) -> None:
    zstandard = pytest.importorskip("zstandard")
    payload = tmp_path / "payload"
    payload.mkdir()
    _make_payload(payload)

    tar_bytes = tmp_path / "data.tar"
    with tarfile.open(tar_bytes, "w") as tar:
        for entry in payload.iterdir():
            tar.add(entry, arcname=entry.name)
    archive = tmp_path / "data.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(tar_bytes.read_bytes()))

    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "benchmark_segmentation_data" / "Super1" / "train" / "sample.tif").is_file()


def test_extract_unsupported_format(tmp_path: Path) -> None:
    archive = tmp_path / "data.rar"
    archive.write_bytes(b"x")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out")


def test_populate_from_remote_via_file_url(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    _make_payload(payload)
    archive = _tar_gz(payload, tmp_path / "data.tar.gz")

    root = tmp_path / "remote-root"
    config = RemoteConfig(url=archive.as_uri())
    populate_from_remote(config, root)

    assert (root / remote_data.READY_MARKER).is_file()
    assert (root / "benchmark_segmentation_data" / "Super1" / "train" / "sample.tif").is_file()


def test_ensure_data_skips_when_local_populated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_payload(tmp_path)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert remote_data._ensure_data() == tmp_path.resolve()


def test_ensure_data_none_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "empty"))
    assert remote_data._ensure_data() is None


def test_ensure_data_downloads_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "payload"
    payload.mkdir()
    _make_payload(payload)
    archive = _tar_gz(payload, tmp_path / "data.tar.gz")

    root = tmp_path / "remote-root"
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("REMOTE_DATA_ROOT", str(root))
    monkeypatch.setenv("DATA_ARCHIVE_URL", archive.as_uri())

    result = remote_data._ensure_data()
    assert result == root
    assert (root / "benchmark_segmentation_data" / "Super1" / "train" / "sample.tif").is_file()
    import os

    assert os.environ["DATA_ROOT"] == str(root)


def test_ensure_data_raises_on_bad_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("REMOTE_DATA_ROOT", str(tmp_path / "remote-root"))
    monkeypatch.setenv("DATA_ARCHIVE_URL", (tmp_path / "missing.tar.gz").as_uri())
    with pytest.raises(Exception):
        remote_data._ensure_data()
