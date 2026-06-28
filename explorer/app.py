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

import matplotlib.pyplot as plt
import streamlit as st

from explorer.lib.catalog import load_catalog, list_benchmark_datasets
from explorer.lib.coco import instance_seg_root, is_instance_data_populated
from explorer.lib.examples import examples_root, is_examples_data_populated
from explorer.lib.index import get_data_root, is_data_populated, split_counts
from explorer.lib.stats import (
    aggregate_class_pixels,
    class_distribution_dataframe,
    image_counts_pivot,
    split_summary_table,
)
from explorer.lib.streamlit_data import cached_benchmark_records, deserialize_records
from explorer.lib.benchmark_panel import render_benchmark_panel

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="🔬",
    layout="wide",
)


@st.cache_data(show_spinner="Computing class pixel totals...")
def cached_class_pixel_totals(record_rows: tuple) -> dict[str, dict[str, int]]:
    return aggregate_class_pixels(deserialize_records(record_rows))


catalog = load_catalog()

st.title("Microscopy Dataset Explorer")
st.markdown(
    "Browse NASA public benchmark datasets before training. "
    f"License: **{catalog['license']}**. "
    f"[Paper]({catalog['paper_url']}) · "
    f"[NASA repo]({catalog['source_repo']}) · "
    "[Reproduction plan](https://github.com/tyc-aidev/microscopy-analysis/blob/main/PLAN.md)"
)

data_root = get_data_root()
benchmark_ready = is_data_populated()
instance_ready = is_instance_data_populated(data_root)
examples_ready = is_examples_data_populated(data_root)

status_bits = [
    ("Benchmarks", benchmark_ready),
    ("Instance seg", instance_ready),
    ("Examples", examples_ready),
]
status_text = " · ".join(
    f"{'✅' if ready else '⚠️'} {label}" for label, ready in status_bits
)
st.caption(f"{status_text} — use the sidebar pages to browse each asset family.")

col_status, col_path = st.columns([1, 2])
with col_status:
    if benchmark_ready:
        st.success("Benchmark data available")
    else:
        st.warning("Benchmark data not downloaded")
with col_path:
    st.code(str(data_root), language=None)

if not benchmark_ready:
    st.info(
        "Download datasets with:\n\n"
        "```bash\n./scripts/download_data.sh\n```\n\n"
        "Or set `DATA_ROOT` to an existing dataset directory."
    )

st.subheader("Benchmark datasets")

record_rows = cached_benchmark_records(str(data_root)) if benchmark_ready else ()
records = deserialize_records(record_rows) if benchmark_ready else []
counts = split_counts(records) if benchmark_ready else {
    ds: info["split_counts"] for ds, info in catalog["datasets"].items()
}

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

with st.expander("Benchmark images", expanded=benchmark_ready):
    if not benchmark_ready:
        st.caption("Download benchmark data to load and preview TIFF images here.")
    else:
        st.caption("Select a dataset, split, and image to load from disk.")
        render_benchmark_panel(records, key_prefix="home_bench")

with st.expander("Dataset statistics", expanded=False):
    if not benchmark_ready:
        st.caption("Statistics use catalog defaults until benchmark data is downloaded.")
    st.markdown("**Images per dataset and split**")
    pivot = image_counts_pivot(counts)
    if not pivot.empty:
        st.bar_chart(pivot)
    else:
        st.caption("No benchmark images indexed.")

    st.markdown("**Split summary**")
    summary = split_summary_table(counts)
    if not summary.empty:
        st.dataframe(summary, width="stretch", hide_index=True)
    else:
        st.caption("No split rows to display.")

    if benchmark_ready and records:
        st.markdown("**Class pixel distribution**")
        class_totals = cached_class_pixel_totals(record_rows)
        present = [ds for ds in list_benchmark_datasets() if ds in class_totals]
        if present:
            selected = st.selectbox("Dataset", present, key="stats_dataset")
            distribution = class_distribution_dataframe(class_totals, selected)
            if not distribution.empty:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(
                    distribution["pixels"],
                    labels=distribution["class"],
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.set_title(f"{selected} — aggregated over all splits")
                st.pyplot(fig, clear_figure=True)
            else:
                st.caption("No class pixels found for this dataset.")
        else:
            st.caption("No mask annotations available for class totals.")

st.subheader("Other assets")
st.markdown(
    f"- **Instance segmentation** — {catalog['instance_segmentation']['description']}\n"
    f"- **Examples** — {catalog['examples']['description']}"
)
if instance_ready:
    st.caption(f"Instance data: `{instance_seg_root(data_root)}`")
if examples_ready:
    st.caption(f"Examples: `{examples_root(data_root)}`")
