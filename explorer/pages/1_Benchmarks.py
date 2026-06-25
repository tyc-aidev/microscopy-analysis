"""Semantic benchmark browser with mask overlay viewer."""

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
from PIL import Image

from explorer.lib.catalog import get_dataset_info, list_benchmark_datasets
from explorer.lib.index import is_data_populated, records_for, split_counts, splits_for_dataset
from explorer.lib.masks import (
    class_pixel_counts,
    colored_mask_rgba,
    load_class_masks,
    overlay_image,
)
from explorer.lib.streamlit_data import cached_pil_image, load_benchmark_records

st.set_page_config(page_title="Benchmarks", layout="wide")

st.title("Semantic Benchmarks")
st.caption("Browse Super1–4 and EBC1–3 with mask overlays.")

if not is_data_populated():
    st.warning("Benchmark data not found. Run `./scripts/download_data.sh` from the repo root.")
    st.stop()

records = load_benchmark_records()
counts = split_counts(records)

with st.sidebar:
    st.header("Filters")
    present = {r.dataset for r in records}
    dataset = st.selectbox("Dataset", [d for d in list_benchmark_datasets() if d in present])
    available_splits = splits_for_dataset(records, dataset)
    split = st.selectbox("Split", available_splits)

    filtered = records_for(records, dataset=dataset, split=split)
    st.divider()
    st.subheader("Split counts")
    ds_counts = counts.get(dataset, {})
    for split_name in available_splits:
        count = ds_counts.get(split_name, 0)
        marker = " →" if split_name == split else ""
        st.write(f"**{split_name}**{marker}: {count}")

    if not filtered:
        st.stop()

    family = get_dataset_info(dataset)["family"]
    st.divider()
    st.subheader("Classes")
    visible_classes: set[str] = set()
    for cls in family["classes"]:
        if st.checkbox(cls["label"], value=True, key=f"class_{cls['id']}"):
            visible_classes.add(cls["id"])

    opacity = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)
    view_mode = st.radio(
        "View",
        ["overlay", "original", "mask only", "side-by-side"],
        horizontal=True,
    )

filter_key = f"{dataset}:{split}"
if st.session_state.get("benchmark_filter") != filter_key:
    st.session_state["benchmark_filter"] = filter_key
    st.session_state["benchmark_idx"] = 0

idx = st.session_state.get("benchmark_idx", 0)
idx = max(0, min(idx, len(filtered) - 1))
st.session_state["benchmark_idx"] = idx

nav_l, nav_m, nav_r = st.columns([1, 3, 1])
with nav_l:
    if st.button("← Prev", disabled=idx == 0):
        st.session_state["benchmark_idx"] = idx - 1
        st.rerun()
with nav_r:
    if st.button("Next →", disabled=idx >= len(filtered) - 1):
        st.session_state["benchmark_idx"] = idx + 1
        st.rerun()
with nav_m:
    image_names = [r.image_path.name for r in filtered]
    picked = st.selectbox("Image", image_names, index=idx, label_visibility="collapsed")
    st.session_state["benchmark_idx"] = image_names.index(picked)

record = filtered[st.session_state["benchmark_idx"]]
image = cached_pil_image(str(record.image_path))

info_col, view_col = st.columns([1, 2])

with info_col:
    st.subheader("Image metadata")
    st.write(f"**File:** `{record.image_path.name}`")
    st.write(f"**Size:** {image.width} × {image.height} px")
    st.write(f"**Mask:** `{record.mask_path.name if record.mask_path else 'missing'}`")

    if record.mask_path:
        class_masks = load_class_masks(record.mask_path, record.family_id)
        pixel_counts = class_pixel_counts(class_masks)
        st.write("**Class pixels:**")
        for cls in family["classes"]:
            stats = pixel_counts.get(cls["id"], {"pixels": 0, "percent": 0.0})
            st.write(f"- {cls['label']}: {stats['percent']}% ({stats['pixels']:,} px)")

    with st.expander("Dataset info", expanded=False):
        ds_info = get_dataset_info(dataset)
        st.markdown(ds_info["family"]["description"])
        if ds_info["family"].get("notes"):
            st.info(ds_info["family"]["notes"])
        if ds_info.get("highlight") == "low_data":
            st.warning("Super3 has only 2 training images — a low-data benchmark regime.")

with view_col:
    if record.mask_path and visible_classes:
        class_masks = load_class_masks(record.mask_path, record.family_id)
        mask_rgba = colored_mask_rgba(
            class_masks, record.family_id, visible_classes=visible_classes
        )
        mask_img = Image.fromarray(mask_rgba, mode="RGBA")
        composite = overlay_image(image, mask_rgba, opacity=opacity)
    elif record.mask_path:
        class_masks = load_class_masks(record.mask_path, record.family_id)
        mask_rgba = colored_mask_rgba(class_masks, record.family_id)
        mask_img = Image.fromarray(mask_rgba, mode="RGBA")
        composite = image
    else:
        mask_img = None
        composite = image

    if view_mode == "original":
        st.image(image, use_container_width=True)
    elif view_mode == "mask only" and mask_img is not None:
        st.image(mask_img, use_container_width=True)
    elif view_mode == "overlay" and mask_img is not None:
        st.image(composite, use_container_width=True)
    elif view_mode == "side-by-side" and mask_img is not None:
        left, right = st.columns(2)
        with left:
            st.caption("Original")
            st.image(image, use_container_width=True)
        with right:
            st.caption("Mask")
            st.image(mask_img, use_container_width=True)
    else:
        st.image(image, use_container_width=True)
        if mask_img is None:
            st.warning("No mask paired for this image.")
