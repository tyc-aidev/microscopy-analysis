"""Instance segmentation tile browser with COCO overlays."""

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

from explorer.lib.catalog import load_catalog
from explorer.lib.coco import (
    bbox_area_histogram,
    instance_seg_root,
    is_instance_data_populated,
    render_annotations,
    split_summary,
)
from explorer.lib.index import get_data_root
from explorer.lib.remote_data import ensure_data
from explorer.lib.streamlit_data import cached_pil_image, load_coco_split_index

st.set_page_config(page_title="Instance Segmentation", layout="wide")

catalog = load_catalog()["instance_segmentation"]
st.title("Instance Segmentation")
st.caption(catalog["description"])

ensure_data()

data_root = get_data_root()
instance_root = instance_seg_root(data_root)

if not is_instance_data_populated(data_root):
    st.warning("Instance segmentation data not found. Run `./scripts/download_data.sh` from the repo root.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    split = st.selectbox("Split", catalog["splits"])
    index = load_coco_split_index(instance_root, split)
    summary = split_summary(index)

    st.divider()
    st.subheader("Summary")
    st.write(f"**Images:** {summary['images']}")
    st.write(f"**Annotations:** {summary['annotations']}")
    st.write(f"**BBox area (px²):** {summary['bbox_area_min']:.0f} – {summary['bbox_area_max']:.0f}")
    st.caption(f"Mean bbox area: {summary['bbox_area_mean']:.0f} px²")

    areas = index.bbox_areas()
    if areas:
        with st.expander("BBox size distribution"):
            st.bar_chart(bbox_area_histogram(areas))

    st.divider()
    show_bbox = st.checkbox("Show bounding boxes", value=True)
    show_mask = st.checkbox("Show segmentation masks", value=True)
    mask_alpha = st.slider("Mask opacity", 0, 255, 100, 5)

if not index.images:
    st.info("No images in this split.")
    st.stop()

filter_key = f"instance:{split}"
if st.session_state.get("instance_filter") != filter_key:
    st.session_state["instance_filter"] = filter_key
    st.session_state["instance_idx"] = 0

idx = st.session_state.get("instance_idx", 0)
idx = max(0, min(idx, len(index.images) - 1))
st.session_state["instance_idx"] = idx

nav_l, nav_m, nav_r = st.columns([1, 3, 1])
with nav_l:
    if st.button("← Prev", disabled=idx == 0, key="inst_prev"):
        st.session_state["instance_idx"] = idx - 1
        st.rerun()
with nav_r:
    if st.button("Next →", disabled=idx >= len(index.images) - 1, key="inst_next"):
        st.session_state["instance_idx"] = idx + 1
        st.rerun()
with nav_m:
    names = [img.image_path.name for img in index.images]
    picked = st.selectbox("Tile", names, index=idx, label_visibility="collapsed", key="inst_pick")
    st.session_state["instance_idx"] = names.index(picked)

record = index.images[st.session_state["instance_idx"]]
annotations = index.annotations_by_image.get(record.image_id, ())

info_col, view_col = st.columns([1, 2])
with info_col:
    st.subheader("Tile metadata")
    st.write(f"**File:** `{record.image_path.name}`")
    st.write(f"**Size:** {record.width} × {record.height} px")
    st.write(f"**Annotations:** {len(annotations)}")
    for ann in annotations:
        x, y, w, h = ann.bbox
        st.write(
            f"- **{ann.category_name}** — bbox ({x:.0f}, {y:.0f}, {w:.0f}×{h:.0f}), "
            f"area {ann.area:.0f} px²"
        )

with view_col:
    if not record.image_path.is_file():
        st.error(f"Image file missing: `{record.image_path}`")
        st.stop()

    image = cached_pil_image(str(record.image_path))
    rendered = render_annotations(
        image,
        annotations,
        show_bbox=show_bbox,
        show_mask=show_mask,
        mask_alpha=mask_alpha,
    )
    st.image(rendered, use_container_width=True)
