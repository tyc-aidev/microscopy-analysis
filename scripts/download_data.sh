#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_ROOT:-$ROOT/data}"
UPSTREAM="$DATA_DIR/_upstream"
REPO_URL="https://github.com/nasa/pretrained-microscopy-models.git"

# --sample: trimmed checkout (Ni-superalloy benchmarks only) for a smaller footprint,
# matching the Streamlit Cloud archive built by scripts/build_data_archive.sh.
SAMPLE=0
[[ "${1:-}" == "--sample" ]] && SAMPLE=1

if [[ "$SAMPLE" -eq 1 ]]; then
  SPARSE_PATHS=(
    benchmark_segmentation_data/Super1
    benchmark_segmentation_data/Super2
    benchmark_segmentation_data/Super3
    benchmark_segmentation_data/Super4
    instance_segmentation/data
    examples
  )
else
  SPARSE_PATHS=(
    benchmark_segmentation_data
    instance_segmentation/data
    examples
  )
fi

mkdir -p "$DATA_DIR"

if [[ -d "$UPSTREAM/.git" ]]; then
  echo "Updating existing sparse checkout at $UPSTREAM ..."
  git -C "$UPSTREAM" sparse-checkout set "${SPARSE_PATHS[@]}"
  git -C "$UPSTREAM" pull --ff-only
else
  echo "Cloning NASA pretrained-microscopy-models (sparse) into $UPSTREAM ..."
  git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$UPSTREAM"
  git -C "$UPSTREAM" sparse-checkout set "${SPARSE_PATHS[@]}"
fi

link() {
  local target="$1"
  local link_path="$2"
  ln -sfn "$target" "$link_path"
  echo "  $link_path -> $target"
}

echo "Linking dataset paths under $DATA_DIR ..."
link "_upstream/benchmark_segmentation_data" "$DATA_DIR/benchmark_segmentation_data"
link "_upstream/instance_segmentation/data" "$DATA_DIR/instance_segmentation"
link "_upstream/examples" "$DATA_DIR/examples"

echo
echo "Done. Dataset root: $DATA_DIR"
echo "Start the explorer: streamlit run explorer/app.py"
