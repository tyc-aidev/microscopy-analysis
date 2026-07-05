"""Hybrid prediction panel loading and Streamlit browser (optional torch)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import streamlit as st

from explorer.lib.streamlit_data import cached_pil_image


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec(
        "segmentation_models_pytorch"
    ) is not None


def list_panels(run_dir: Path, split: str) -> list[Path]:
    pred_dir = run_dir / "predictions" / split
    if not pred_dir.is_dir():
        return []
    return sorted(pred_dir.glob("*_panel.png"))


def ensure_panels(
    config_path: Path,
    run_dir: Path,
    split: str,
    *,
    device: str = "auto",
    force: bool = False,
) -> list[Path]:
    """Return saved panels, generating them first when missing (or when *force*)."""
    existing = list_panels(run_dir, split)
    if existing and not force:
        return existing
    return generate_panels(config_path, run_dir, split, device=device)


def generate_panels(
    config_path: Path,
    run_dir: Path,
    split: str,
    *,
    device: str = "auto",
    max_images: int | None = None,
) -> list[Path]:
    if not torch_available():
        raise RuntimeError("PyTorch stack not installed; run: uv pip install -r requirements-apple.txt")

    from microscopy_analysis.eval.predictions import run_prediction_panels
    from microscopy_analysis.train.config import load_train_config

    cfg = load_train_config(config_path)
    checkpoint = run_dir / "model_best.pth"
    out_dir = run_dir / "predictions" / split
    return run_prediction_panels(
        cfg,
        checkpoint=checkpoint,
        split=split,
        output_dir=out_dir,
        device=device,
        max_images=max_images,
    )


def render_panel_browser(panels: list[Path], *, key_prefix: str) -> None:
    if not panels:
        st.caption("No prediction panels available.")
        return

    idx_key = f"{key_prefix}_panel_idx"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    idx = st.session_state[idx_key]
    idx = max(0, min(idx, len(panels) - 1))
    st.session_state[idx_key] = idx

    nav_l, nav_m, nav_r = st.columns([1, 3, 1])
    with nav_l:
        if st.button("← Prev", key=f"{key_prefix}_prev", disabled=idx == 0):
            st.session_state[idx_key] = idx - 1
            st.rerun()
    with nav_r:
        if st.button("Next →", key=f"{key_prefix}_next", disabled=idx >= len(panels) - 1):
            st.session_state[idx_key] = idx + 1
            st.rerun()
    with nav_m:
        names = [p.name for p in panels]
        picked = st.selectbox("Panel", names, index=idx, label_visibility="collapsed", key=f"{key_prefix}_pick")
        st.session_state[idx_key] = names.index(picked)

    panel_path = panels[st.session_state[idx_key]]
    st.caption(f"`{panel_path.name}` — input | ground truth | prediction | errors")
    st.image(cached_pil_image(str(panel_path)), width="stretch")
