"""Bootstrap import path before explorer package imports (Streamlit runs scripts directly)."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPLORER_DIR = Path(__file__).resolve().parent


def ensure_repo_root_on_path() -> Path:
    """Put the repo root and ``src/`` on ``sys.path`` for Cloud / script runs.

    Streamlit Community Cloud never runs ``pip install -e .``, so
    ``microscopy_analysis`` (under ``src/``) is not installed as a package.
    Adding ``src/`` lets Local Training load ``load_train_config`` without an
    editable install. Repo root stays on the path so ``explorer.*`` imports work.
    """
    root = _EXPLORER_DIR.parent
    # Insert src first so it ends up ahead of root after both inserts.
    for path in (str(root), str(root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root
