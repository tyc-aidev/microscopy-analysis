"""Tests for explorer bootstrap path setup (Streamlit Cloud import resolution)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_bootstrap_puts_src_on_path_for_microscopy_analysis(monkeypatch) -> None:
    # Simulate a clean path without the editable-install site-packages entry.
    repo = Path(__file__).resolve().parents[1]
    src = repo / "src"
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p).resolve() != src])

    # Drop a cached package so we re-resolve from the new path.
    for name in list(sys.modules):
        if name == "microscopy_analysis" or name.startswith("microscopy_analysis."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    from explorer._bootstrap import ensure_repo_root_on_path

    root = ensure_repo_root_on_path()
    assert root == repo
    assert str(src) in sys.path

    mod = importlib.import_module("microscopy_analysis.train.config")
    assert hasattr(mod, "load_train_config")
