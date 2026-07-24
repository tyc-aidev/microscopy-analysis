"""Ensure repo root is on sys.path when Streamlit runs app/pages as scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    for path in (str(root), str(root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root
