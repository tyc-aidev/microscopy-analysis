"""Paper reproduction results — CUDA host runs imported locally."""

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

from explorer.lib.results_view import render_run_dashboard
from explorer.lib.runs import (
    filter_runs,
    find_config_for_run,
    get_configs_dir,
    get_results_root,
    list_config_files,
    scan_runs,
)

st.set_page_config(page_title="CUDA Reproduction", layout="wide")

st.title("Paper Reproduction (CUDA)")
st.caption("Browse bit-exact reproduction runs synced from a CUDA Linux host.")

with st.expander("Reproduction environment requirements", expanded=False):
    st.markdown(
        "The paper-pinned stack targets **CUDA Linux + Python 3.8–3.10** "
        "([`requirements.txt`](https://github.com/tyc-aidev/microscopy-analysis/blob/main/requirements.txt)):\n\n"
        "| Package | Version |\n"
        "|---------|--------|\n"
        "| torch | 1.10.1 |\n"
        "| torchvision | 0.11.2 |\n"
        "| segmentation-models-pytorch | 0.2.1 |\n"
        "| timm | 0.4.12 |\n\n"
        "MicroNet encoder weights are pinned to **v1.0** (never v1.1). "
        "See README *Reproduction environment (CUDA host)* for setup."
    )

with st.expander("Import results from CUDA host", expanded=True):
    st.markdown(
        "Train on the CUDA machine, then copy the run directory into local `RESULTS_ROOT` "
        "(default `./results`):\n\n"
        "```bash\n"
        "# From your Mac — sync one run\n"
        "rsync -avz user@cuda-host:~/microscopy-analysis/results/<run_name>/ \\\n"
        "  ./results/<run_name>/\n\n"
        "# Or copy a single run with scp\n"
        "scp -r user@cuda-host:~/microscopy-analysis/results/<run_name> ./results/\n"
        "```\n\n"
        "Generate prediction panels **on the CUDA host** before sync:\n\n"
        "```bash\n"
        "python scripts/visualize_predictions.py \\\n"
        "  --config configs/experiments/super1_baseline.yaml \\\n"
        "  --split val --device cuda\n"
        "```\n\n"
        "Imported runs are tagged by `device: cuda` in `run_summary.json`."
    )

results_root = get_results_root()
configs_dir = get_configs_dir()
cuda_runs = filter_runs(scan_runs(results_root), "cuda")

with st.sidebar:
    st.header("Filters")
    st.caption(f"Results: `{results_root}`")

    if st.button("Refresh runs"):
        st.cache_data.clear()
        st.rerun()

    if not cuda_runs:
        st.info(
            "No CUDA runs imported yet.\n\n"
            "1. Provision CUDA venv with `requirements.txt`\n"
            "2. Train on remote host\n"
            "3. `rsync` `results/<run_name>/` here\n"
            "4. Refresh this page"
        )
        st.stop()

    run_names = [r.run_name for r in cuda_runs]
    selected_run = st.selectbox("Run", run_names)
    run = next(r for r in cuda_runs if r.run_name == selected_run)

    matched = find_config_for_run(run.run_name, configs_dir)
    config_options = list_config_files(configs_dir)
    config_labels = [p.name for p in config_options]
    default_idx = config_labels.index(matched.name) if matched and matched in config_options else 0
    picked_config = st.selectbox("Experiment config", config_labels, index=default_idx)
    config_path = configs_dir / picked_config

    split = st.selectbox("Prediction split", ["val", "train", "test"])

render_run_dashboard(
    run,
    config_path,
    split=split,
    allow_live_inference=False,
    key_prefix="cuda",
)
