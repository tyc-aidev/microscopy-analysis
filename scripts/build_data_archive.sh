#!/usr/bin/env bash
set -euo pipefail

# Build a dataset archive for upload to Cloudflare R2 (see issue #17).
#
# The archive is rooted at the asset directories (benchmark_segmentation_data/,
# instance_segmentation/, examples/) so explorer.lib.remote_data.ensure_data()
# can extract it straight into DATA_ROOT.
#
# Usage:
#   scripts/build_data_archive.sh [--sample|--full] [--format tar.zst|tar.gz|zip] [--out PATH]
#
# Defaults: --sample, --format tar.zst, --out data/dist/amat-data-<mode>.<ext>
# Source data is read from $DATA_ROOT (default ./data); symlinks are dereferenced.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_ROOT:-$ROOT/data}"

MODE="sample"
FORMAT="tar.zst"
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) MODE="sample"; shift ;;
    --full) MODE="full"; shift ;;
    --format) FORMAT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$DATA_DIR/benchmark_segmentation_data" ]] || {
  echo "No data found under $DATA_DIR. Run ./scripts/download_data.sh first." >&2
  exit 1
}

case "$FORMAT" in
  tar.zst)
    command -v zstd >/dev/null 2>&1 || {
      echo "zstd CLI not found (needed for --format tar.zst). Install it or use --format tar.gz." >&2
      exit 1
    } ;;
  tar.gz|zip) ;;
  *) echo "Unsupported --format: $FORMAT (use tar.zst, tar.gz, or zip)" >&2; exit 2 ;;
esac

OUT="${OUT:-$DATA_DIR/dist/amat-data-$MODE.$FORMAT}"
mkdir -p "$(dirname "$OUT")"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging $MODE dataset (dereferencing symlinks) ..."
copy() {  # copy SRC (relative to DATA_DIR) into the staging tree if it exists
  local src="$DATA_DIR/$1"
  [[ -e "$src" ]] || { echo "  skip (missing): $1"; return; }
  mkdir -p "$STAGE/$(dirname "$1")"
  cp -RL "$src" "$STAGE/$1"
  echo "  + $1"
}

if [[ "$MODE" == "full" ]]; then
  copy "benchmark_segmentation_data"
  copy "instance_segmentation"
  copy "examples"
else
  # Trimmed subset: Ni-superalloy benchmarks + instance demo (annotations + val)
  # + example images only (notebooks/heatmap dirs dropped to fit ephemeral disk).
  for ds in Super1 Super2 Super3 Super4; do
    copy "benchmark_segmentation_data/$ds"
  done
  copy "instance_segmentation/annotations"
  copy "instance_segmentation/validation"
  copy "examples/npg.png"
  copy "examples/dog.jpeg"
fi

echo "Writing $OUT ..."
case "$FORMAT" in
  tar.zst) tar -C "$STAGE" -cf - . | zstd -19 -T0 -o "$OUT" -f ;;
  tar.gz)  tar -C "$STAGE" -czf "$OUT" . ;;
  zip)     ( cd "$STAGE" && zip -qr -X "$OUT" . ) ;;
esac

SIZE="$(du -h "$OUT" | cut -f1)"
echo
echo "Built $OUT ($SIZE)."
echo
echo "Upload to Cloudflare R2 (public bucket, data is MIT):"
echo "  wrangler r2 object put amat-datasets/$(basename "$OUT") --file \"$OUT\" --remote"
echo "Then set DATA_ARCHIVE_URL in Streamlit secrets to the bucket's public URL + object key."
