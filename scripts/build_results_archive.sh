#!/usr/bin/env bash
set -euo pipefail

# Build a slim training-results archive for Streamlit Community Cloud.
#
# Packs per-run metrics, summaries, and pre-rendered prediction panels only —
# checkpoints are excluded (too large for ephemeral /tmp alongside datasets).
#
# Usage:
#   scripts/build_results_archive.sh [--format tar.zst|tar.gz|zip] [--out PATH]
#                                    [--results-dir PATH]
#
# Defaults: --format tar.zst, --out data/dist/amat-results-slim.<ext>,
#           --results-dir ./results
#
# Archive layout: <run_name>/{run_summary.json,metrics.json,predictions/...}
# so explorer.lib.remote_results.ensure_results() can extract into RESULTS_ROOT.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_ROOT:-$ROOT/results}"
FORMAT="tar.zst"
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --format) FORMAT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$RESULTS_DIR" ]] || {
  echo "No results found under $RESULTS_DIR." >&2
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

OUT="${OUT:-$ROOT/data/dist/amat-results-slim.$FORMAT}"
mkdir -p "$(dirname "$OUT")"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging slim results from $RESULTS_DIR ..."
copied=0
for run_dir in "$RESULTS_DIR"/*/ ; do
  [[ -d "$run_dir" ]] || continue
  name="$(basename "$run_dir")"
  summary="$run_dir/run_summary.json"
  [[ -f "$summary" ]] || { echo "  skip (no summary): $name"; continue; }

  dest="$STAGE/$name"
  mkdir -p "$dest"
  cp "$summary" "$dest/run_summary.json"
  if [[ -f "$run_dir/metrics.json" ]]; then
    cp "$run_dir/metrics.json" "$dest/metrics.json"
  else
    echo "  warn: missing metrics.json for $name"
  fi
  if [[ -d "$run_dir/predictions" ]]; then
    mkdir -p "$dest/predictions"
    # Copy panel trees only (PNGs); skip any accidental weight files.
    find "$run_dir/predictions" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) \
      -print0 | while IFS= read -r -d '' f; do
      rel="${f#"$run_dir"}"
      mkdir -p "$dest/$(dirname "$rel")"
      cp "$f" "$dest/$rel"
    done
  else
    echo "  warn: no predictions/ for $name"
  fi
  echo "  + $name"
  copied=$((copied + 1))
done

[[ "$copied" -gt 0 ]] || {
  echo "No runs with run_summary.json under $RESULTS_DIR." >&2
  exit 1
}

echo "Writing $OUT ..."
case "$FORMAT" in
  tar.zst) tar -C "$STAGE" -cf - . | zstd -19 -T0 -o "$OUT" -f ;;
  tar.gz)  tar -C "$STAGE" -czf "$OUT" . ;;
  zip)     ( cd "$STAGE" && zip -qr -X "$OUT" . ) ;;
esac

SIZE="$(du -h "$OUT" | cut -f1)"
echo
echo "Built $OUT ($SIZE) from $copied run(s)."
echo
echo "Upload to Cloudflare R2:"
echo "  wrangler r2 object put microscopy-analysis-datasets/$(basename "$OUT") \\"
echo "    --file \"$OUT\" --content-type application/zstd --remote"
echo "Then set RESULTS_ARCHIVE_URL in Streamlit secrets to the public URL + object key."
