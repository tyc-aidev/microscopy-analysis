"""Gallery of NASA reference and demo assets."""

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
from explorer.lib.examples import examples_root, is_examples_data_populated, list_example_items
from explorer.lib.index import get_data_root
from explorer.lib.remote_data import ensure_data
from explorer.lib.streamlit_data import cached_pil_image

st.set_page_config(page_title="Examples", layout="wide")

examples_meta = load_catalog()["examples"]
st.title("Examples and Reference Assets")
st.caption(examples_meta["description"])

ensure_data()

data_root = get_data_root()
examples_dir = examples_root(data_root)

if not is_examples_data_populated(data_root):
    st.warning("Example assets not found. Run `./scripts/download_data.sh` from the repo root.")
    st.stop()

items = list_example_items(examples_dir)
if not items:
    st.info("No example images found under the configured paths.")
    st.stop()

sections: dict[str, list] = {}
for item in items:
    sections.setdefault(item.section, []).append(item)

for section, section_items in sections.items():
    st.subheader(section)
    cols = st.columns(min(3, len(section_items)))
    for idx, item in enumerate(section_items):
        with cols[idx % len(cols)]:
            image = cached_pil_image(str(item.path))
            st.image(image, width="stretch")
            st.caption(item.caption)
            st.caption(f"`{item.path.relative_to(examples_dir)}`")
