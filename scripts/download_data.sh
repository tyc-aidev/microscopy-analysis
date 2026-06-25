#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_ROOT:-$ROOT/data}"
UPSTREAM="$DATA_DIR/_upstream"
REPO_URL="https://github.com/nasa/pretrained-microscopy-models.git"

mkdir -p "$DATA_DIR"

if [[ -d "$UPSTREAM/.git" ]]; then
  echo "Updating existing sparse checkout at $UPSTREAM ..."
  git -C "$UPSTREAM" pull --ff-only
else
  echo "Cloning NASA pretrained-microscopy-models (sparse) into $UPSTREAM ..."
  git clone --depth 1 --filter=blob:none --sparse "$REPO_URL" "$UPSTREAM"
  git -C "$UPSTREAM" sparse-checkout set \
    benchmark_segmentation_data \
    instance_segmentation/data \
    examples
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
