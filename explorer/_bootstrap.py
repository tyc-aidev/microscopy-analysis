"""Bootstrap import path before explorer package imports (Streamlit runs scripts directly)."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPLORER_DIR = Path(__file__).resolve().parent


def ensure_repo_root_on_path() -> Path:
    root = _EXPLORER_DIR.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
