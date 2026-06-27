"""Fetch and extract explorer datasets from a remote archive (Cloudflare R2).

On Streamlit Community Cloud the local ``data/`` flow (``scripts/download_data.sh``)
never runs and the filesystem is ephemeral, so the app downloads a prepackaged
archive at runtime and extracts it into a writable ``DATA_ROOT``.

Configuration (env var or ``st.secrets``):

* ``DATA_ARCHIVE_URL`` — public URL to the archive (preferred; data is MIT).
* or ``R2_ENDPOINT`` + ``R2_ACCESS_KEY_ID`` + ``R2_SECRET_ACCESS_KEY`` +
  ``R2_BUCKET`` + ``R2_OBJECT_KEY`` — private bucket via ``boto3``.
* ``REMOTE_DATA_ROOT`` — extraction target (default ``/tmp/amat-data``).

Remote fetching is **explicit opt-in**: if none of the above are configured,
``ensure_data()`` returns ``None`` and the app shows its download prompt instead
of pulling anything. On Streamlit Cloud, set ``DATA_ARCHIVE_URL`` to the public
archive you want served (see ``README.md`` for the canonical URLs).

Local development is unaffected: if ``DATA_ROOT`` already points at populated
data, ``ensure_data()`` returns immediately without any download.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from explorer.lib.index import get_data_root, is_data_populated

DEFAULT_REMOTE_ROOT = "/tmp/amat-data"
READY_MARKER = ".ready"
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class RemoteConfig:
    """Resolved source for the dataset archive."""

    url: str | None = None
    endpoint: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    bucket: str | None = None
    object_key: str | None = None

    @property
    def archive_name(self) -> str:
        source = self.url or self.object_key or ""
        name = Path(urlparse(source).path if "://" in source else source).name
        return name or "data.tar.zst"


def _secret(key: str) -> str | None:
    """Read config from the environment first, then ``st.secrets`` if available."""
    value = os.environ.get(key)
    if value:
        return value
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return None


def resolve_remote_config() -> RemoteConfig | None:
    """Build a :class:`RemoteConfig` from env/secrets, or ``None`` if unconfigured."""
    url = _secret("DATA_ARCHIVE_URL")
    if url:
        return RemoteConfig(url=url)

    endpoint = _secret("R2_ENDPOINT")
    access_key_id = _secret("R2_ACCESS_KEY_ID")
    secret_access_key = _secret("R2_SECRET_ACCESS_KEY")
    bucket = _secret("R2_BUCKET")
    object_key = _secret("R2_OBJECT_KEY")
    if all((endpoint, access_key_id, secret_access_key, bucket, object_key)):
        return RemoteConfig(
            endpoint=endpoint,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket=bucket,
            object_key=object_key,
        )
    return None


def remote_data_root() -> Path:
    return Path(os.environ.get("REMOTE_DATA_ROOT", DEFAULT_REMOTE_ROOT)).expanduser()


def _is_within(base: Path, target: Path) -> bool:
    """Guard against archive path traversal (``../`` or absolute members)."""
    base = base.resolve()
    try:
        target.resolve().relative_to(base)
    except ValueError:
        return False
    return True


def extract_archive(archive_path: Path, dest: Path) -> None:
    """Extract ``archive_path`` into ``dest``, dispatching on file extension.

    Supports ``.zip``, ``.tar``, ``.tar.gz``/``.tgz`` and ``.tar.zst``. Extraction
    is constrained to ``dest`` to prevent path-traversal from untrusted members.
    """
    dest.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if not _is_within(dest, dest / member):
                    raise ValueError(f"Unsafe archive member: {member}")
            zf.extractall(dest)
        return

    if name.endswith((".tar.zst", ".tzst")):
        import zstandard  # optional dependency, only needed for zstd archives

        dctx = zstandard.ZstdDecompressor()
        with archive_path.open("rb") as raw, dctx.stream_reader(raw) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                tar.extractall(dest, filter="data")
        return

    if name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive_path) as tar:
            tar.extractall(dest, filter="data")
        return

    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def _download(config: RemoteConfig, dest_file: Path) -> None:
    if config.url:
        # Cloudflare r2.dev rejects the default Python-urllib User-Agent with 403,
        # so send an explicit one.
        request = Request(config.url, headers={"User-Agent": "amat-explorer/1.0"})
        with urlopen(request) as response, dest_file.open("wb") as out:
            shutil.copyfileobj(response, out, _DOWNLOAD_CHUNK)
        return

    import boto3  # optional dependency, only needed for private R2 buckets

    client = boto3.client(
        "s3",
        endpoint_url=config.endpoint,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
    )
    client.download_file(config.bucket, config.object_key, str(dest_file))


def populate_from_remote(config: RemoteConfig, root: Path) -> Path:
    """Download the archive for ``config`` and extract it into ``root``.

    Extraction goes to a temporary sibling directory first, then is swapped into
    place, so a crashed download never leaves a half-populated ``root``.
    """
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as staging:
        staging_path = Path(staging)
        archive_file = staging_path / config.archive_name
        _download(config, archive_file)
        extract_path = staging_path / "extract"
        extract_archive(archive_file, extract_path)

        for entry in extract_path.iterdir():
            target = root / entry.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(entry), str(target))

    (root / READY_MARKER).write_text("ok", encoding="utf-8")
    return root


def _ensure_data() -> Path | None:
    """Resolve a populated data root, downloading from remote only if needed."""
    if is_data_populated():
        return get_data_root()

    config = resolve_remote_config()
    if config is None:
        return None

    root = remote_data_root()
    if not (root / READY_MARKER).is_file():
        populate_from_remote(config, root)

    os.environ["DATA_ROOT"] = str(root)
    return root


try:
    import streamlit as _st

    @_st.cache_resource(show_spinner="Downloading datasets...")
    def _ensure_data_cached() -> str | None:
        try:
            root = _ensure_data()
        except Exception as exc:  # degrade to the download-prompt UI instead of crashing
            _st.error(f"Could not fetch datasets from the configured source: {exc}")
            return None
        return str(root) if root else None

except ImportError:  # pragma: no cover - streamlit always present in the app env
    _ensure_data_cached = None


def ensure_data() -> str | None:
    """Cached entry point: ensure datasets are present and return the root path.

    Cached with ``st.cache_resource`` so the download/extract happens once per
    container. Returns the data root as a string, or ``None`` when no data is
    available locally and no remote source is configured.
    """
    if _ensure_data_cached is not None:
        return _ensure_data_cached()
    root = _ensure_data()
    return str(root) if root else None
