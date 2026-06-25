"""Dataset Explorer — home page."""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Streamlit executes this file as a script, not as a package module.
_spec = importlib.util.spec_from_file_location(
    "_bootstrap", Path(__file__).resolve().parent / "_bootstrap.py"
)
assert _spec and _spec.loader
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)
_bootstrap.ensure_repo_root_on_path()

import streamlit as st

from explorer.lib.catalog import load_catalog, list_benchmark_datasets
from explorer.lib.index import get_data_root, is_data_populated, scan_benchmarks, split_counts
from explorer.lib.streamlit_data import cached_benchmark_records, deserialize_records

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="🔬",
    layout="wide",
)

catalog = load_catalog()

st.title("Microscopy Dataset Explorer")
st.markdown(
    "Browse NASA public benchmark datasets before training. "
    f"License: **{catalog['license']}**. "
    f"[Paper]({catalog['paper_url']}) · "
    f"[NASA repo]({catalog['source_repo']}) · "
    "[Reproduction plan](../PLAN.md)"
)
st.caption("Use **Benchmarks** in the sidebar to browse images with mask overlays.")

data_root = get_data_root()
populated = is_data_populated()

col_status, col_path = st.columns([1, 2])
with col_status:
    if populated:
        st.success("Data available")
    else:
        st.warning("Data not downloaded")
with col_path:
    st.code(str(data_root), language=None)

if not populated:
    st.info(
        "Download benchmark data with:\n\n"
        "```bash\n./scripts/download_data.sh\n```\n\n"
        "Or set `DATA_ROOT` to an existing dataset directory."
    )

st.subheader("Benchmark datasets")

if populated:
    records = deserialize_records(cached_benchmark_records(str(data_root)))
    counts = split_counts(records)
else:
    counts = {ds: info["split_counts"] for ds, info in catalog["datasets"].items()}

cols = st.columns(2)
for idx, dataset_id in enumerate(list_benchmark_datasets()):
    info = catalog["datasets"][dataset_id]
    family = next(f for f in catalog["families"] if f["id"] == info["family"])
    ds_counts = counts.get(dataset_id, info.get("split_counts", {}))
    total = sum(ds_counts.values())
    highlight = info.get("highlight")
    label = f"**{dataset_id}** — {info['material']}"
    if highlight == "low_data":
        label += " ⚠️ low-data"

    with cols[idx % 2]:
        with st.container(border=True):
            st.markdown(label)
            st.caption(family["name"])
            st.write(f"**{total}** images indexed")
            split_parts = [f"{split}: {n}" for split, n in sorted(ds_counts.items())]
            st.caption(" · ".join(split_parts))

st.subheader("Other assets")
st.markdown(
    f"- **Instance segmentation** — {catalog['instance_segmentation']['description']}\n"
    f"- **Examples** — {catalog['examples']['description']}"
)
st.caption("Open Instance Segmentation or Examples from the sidebar.")
