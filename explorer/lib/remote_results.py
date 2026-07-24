"""Fetch and extract slim training-result archives from a remote URL (Cloudflare R2).

On Streamlit Community Cloud ``results/`` is gitignored and the filesystem is
ephemeral, so the app can download a prepackaged slim archive (metrics,
summaries, pre-rendered prediction panels — no checkpoints) at runtime.

Configuration (env var or ``st.secrets``):

* ``RESULTS_ARCHIVE_URL`` — public URL to the archive (preferred).
* ``REMOTE_RESULTS_ROOT`` — extraction target (default ``/tmp/amat-results``).

Remote fetching is **explicit opt-in**: if ``RESULTS_ARCHIVE_URL`` is unset,
``ensure_results()`` leaves ``RESULTS_ROOT`` alone. Local development is
unaffected: if ``./results`` (or ``RESULTS_ROOT``) already has run summaries,
nothing is downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

from explorer.lib.remote_data import (
    READY_MARKER,
    RemoteConfig,
    populate_from_remote,
    _secret,
)
from explorer.lib.runs import get_results_root

DEFAULT_REMOTE_RESULTS_ROOT = "/tmp/amat-results"


def resolve_results_remote_config() -> RemoteConfig | None:
    """Build a :class:`RemoteConfig` from ``RESULTS_ARCHIVE_URL``, or ``None``."""
    url = _secret("RESULTS_ARCHIVE_URL")
    if url:
        return RemoteConfig(url=url)
    return None


def remote_results_root() -> Path:
    return Path(
        os.environ.get("REMOTE_RESULTS_ROOT", DEFAULT_REMOTE_RESULTS_ROOT)
    ).expanduser()


def is_results_populated(root: Path | None = None) -> bool:
    """True when *root* contains at least one ``run_summary.json``."""
    results_root = root if root is not None else get_results_root()
    if not results_root.is_dir():
        return False
    return any(results_root.glob("*/run_summary.json"))


def _ensure_results() -> Path | None:
    """Resolve a populated results root, downloading from remote only if needed."""
    if is_results_populated():
        return get_results_root()

    config = resolve_results_remote_config()
    if config is None:
        return None

    root = remote_results_root()
    if not (root / READY_MARKER).is_file():
        populate_from_remote(config, root)

    os.environ["RESULTS_ROOT"] = str(root)
    return root


try:
    import streamlit as _st

    @_st.cache_resource(show_spinner="Downloading training results...")
    def _ensure_results_cached() -> str | None:
        try:
            root = _ensure_results()
        except Exception as exc:  # degrade to empty Local Training UI
            _st.error(f"Could not fetch training results from the configured source: {exc}")
            return None
        return str(root) if root else None

except ImportError:  # pragma: no cover
    _ensure_results_cached = None


def ensure_results() -> str | None:
    """Cached entry point: ensure slim results are present and return the root.

    Cached with ``st.cache_resource`` so the download/extract happens once per
    container. Returns the results root as a string, or ``None`` when no local
    runs exist and no remote source is configured.
    """
    if _ensure_results_cached is not None:
        return _ensure_results_cached()
    root = _ensure_results()
    return str(root) if root else None
