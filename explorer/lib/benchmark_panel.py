"""Reusable Streamlit panel for browsing benchmark images."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from explorer.lib.catalog import get_dataset_info, list_benchmark_datasets
from explorer.lib.index import ImageRecord, records_for, splits_for_dataset
from explorer.lib.masks import class_pixel_counts, colored_mask_rgba, load_class_masks, overlay_image
from explorer.lib.streamlit_data import cached_pil_image


def _filter_key(prefix: str, dataset: str, split: str) -> str:
    return f"{prefix}:{dataset}:{split}"


def render_benchmark_panel(
    records: list[ImageRecord],
    *,
    key_prefix: str = "bench_panel",
    default_dataset: str | None = None,
    show_overlay_controls: bool = True,
) -> None:
    """Render dataset/split/image selectors and load the selected benchmark image."""
    if not records:
        st.caption("No benchmark images indexed.")
        return

    present = {r.dataset for r in records}
    datasets = [d for d in list_benchmark_datasets() if d in present]
    if not datasets:
        st.caption("No benchmark datasets found on disk.")
        return

    default_idx = datasets.index(default_dataset) if default_dataset in datasets else 0

    filter_col, view_col = st.columns([1, 2])

    with filter_col:
        dataset = st.selectbox(
            "Dataset",
            datasets,
            index=default_idx,
            key=f"{key_prefix}_dataset",
        )
        available_splits = splits_for_dataset(records, dataset)
        split = st.selectbox(
            "Split",
            available_splits,
            key=f"{key_prefix}_split",
        )
        filtered = records_for(records, dataset=dataset, split=split)
        if not filtered:
            st.warning("No images in this split.")
            return

        filter_key = _filter_key(key_prefix, dataset, split)
        if st.session_state.get(f"{key_prefix}_filter") != filter_key:
            st.session_state[f"{key_prefix}_filter"] = filter_key
            st.session_state[f"{key_prefix}_idx"] = 0

        idx = st.session_state.get(f"{key_prefix}_idx", 0)
        idx = max(0, min(idx, len(filtered) - 1))
        image_names = [r.image_path.name for r in filtered]
        picked = st.selectbox(
            "Image",
            image_names,
            index=idx,
            key=f"{key_prefix}_image",
        )
        st.session_state[f"{key_prefix}_idx"] = image_names.index(picked)
        record = filtered[st.session_state[f"{key_prefix}_idx"]]

        family = get_dataset_info(dataset)["family"]
        st.markdown("**Metadata**")
        st.write(f"`{record.image_path.name}`")
        if record.mask_path:
            st.write(f"Mask: `{record.mask_path.name}`")
        else:
            st.write("Mask: missing")

        if show_overlay_controls and record.mask_path:
            show_overlay = st.checkbox("Show mask overlay", value=True, key=f"{key_prefix}_overlay")
            opacity = st.slider(
                "Overlay opacity",
                0.0,
                1.0,
                0.5,
                0.05,
                key=f"{key_prefix}_opacity",
            )
        else:
            show_overlay = False
            opacity = 0.5

    with view_col:
        image = cached_pil_image(str(record.image_path))
        st.caption(f"{record.dataset} / {record.split} — {image.width}×{image.height} px")

        if show_overlay_controls and show_overlay and record.mask_path:
            class_masks = load_class_masks(record.mask_path, record.family_id)
            mask_rgba = colored_mask_rgba(class_masks, record.family_id)
            display = overlay_image(image, mask_rgba, opacity=opacity)
            st.image(display, width="stretch")

            pixel_counts = class_pixel_counts(class_masks)
            parts = []
            for cls in family["classes"]:
                stats = pixel_counts.get(cls["id"], {"percent": 0.0})
                parts.append(f"{cls['label']} {stats['percent']}%")
            st.caption(" · ".join(parts))
        else:
            st.image(image, width="stretch")
            if record.mask_path is None:
                st.warning("No mask paired for this image.")

        with st.expander("File paths"):
            st.code(str(record.image_path), language=None)
            if record.mask_path:
                st.code(str(record.mask_path), language=None)
