"""Local training results — Apple Silicon / MPS iteration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_bootstrap", Path(__file__).resolve().parent.parent / "_bootstrap.py"
)
assert _spec and _spec.loader
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
_bootstrap.ensure_repo_root_on_path()

import streamlit as st

from explorer.lib.prediction_panels import torch_available
from explorer.lib.remote_results import ensure_results
from explorer.lib.results_view import render_run_dashboard
from explorer.lib.runs import (
    filter_runs,
    find_config_for_run,
    get_configs_dir,
    get_results_root,
    list_config_files,
    scan_runs,
)

st.set_page_config(page_title="Local Training", layout="wide")

st.title("Local Training (Apple Silicon)")
st.caption("Browse metrics and validation panels from local MPS/CPU training runs.")

ensure_results()

if torch_available():
    try:
        import torch

        mps = "available" if torch.backends.mps.is_available() else "unavailable"
        st.success(f"PyTorch {torch.__version__} · MPS {mps}")
    except Exception:
        st.success("PyTorch stack installed")
else:
    st.warning(
        "PyTorch not installed — metrics and saved panels still work. "
        "For live panel generation: `uv pip install -r requirements-apple.txt`"
    )

results_root = get_results_root()
configs_dir = get_configs_dir()
all_runs = filter_runs(scan_runs(results_root), "local")

with st.sidebar:
    st.header("Filters")
    st.caption(f"Results: `{results_root}`")

    if st.button("Refresh runs"):
        st.cache_data.clear()
        st.rerun()

    if not all_runs:
        st.info(
            "No local runs found. Train with:\n\n"
            "```bash\npython scripts/train.py "
            "--config configs/experiments/super1_baseline.yaml "
            "--device mps --batch-size 4 "
            "--max-epochs-phase1 6 --max-epochs-phase2 3\n```"
        )
        st.stop()

    run_names = [r.run_name for r in all_runs]
    selected_run = st.selectbox("Run", run_names)
    run = next(r for r in all_runs if r.run_name == selected_run)

    matched = find_config_for_run(run.run_name, configs_dir)
    config_options = list_config_files(configs_dir)
    config_labels = [p.name for p in config_options]
    default_idx = config_labels.index(matched.name) if matched and matched in config_options else 0
    picked_config = st.selectbox("Experiment config", config_labels, index=default_idx)
    config_path = configs_dir / picked_config

    split = st.selectbox("Prediction split", ["val", "train", "test"])

if not all_runs:
    st.stop()

render_run_dashboard(
    run,
    config_path,
    split=split,
    allow_live_inference=True,
    key_prefix="local",
    inference_device="mps",
)
