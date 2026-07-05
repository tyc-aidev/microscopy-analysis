#!/usr/bin/env bash
# Train all baseline experiments except Super2 (local MPS/CPU defaults).
#
# Override caps for a full run:
#   DEVICE=mps BATCH_SIZE=2 MAX_EPOCHS_PHASE1=120 MAX_EPOCHS_PHASE2=60 \
#     ./scripts/train_baselines_except_super2.sh
#
# Quick smoke (default): phase1=6, phase2=3, batch-size=2.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_EPOCHS_PHASE1="${MAX_EPOCHS_PHASE1:-6}"
MAX_EPOCHS_PHASE2="${MAX_EPOCHS_PHASE2:-3}"
PATIENCE="${PATIENCE:-5}"

CONFIGS=(
  super1_baseline
  super3_baseline
  super4_baseline
  ebc1_baseline
  ebc2_baseline
  ebc3_baseline
)

for name in "${CONFIGS[@]}"; do
  config="configs/experiments/${name}.yaml"
  echo "=== Training ${name} (${config}) ==="
  python scripts/train.py \
    --config "$config" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --max-epochs-phase1 "$MAX_EPOCHS_PHASE1" \
    --max-epochs-phase2 "$MAX_EPOCHS_PHASE2" \
    --patience "$PATIENCE"
  echo
done

echo "Done. ${#CONFIGS[@]} run(s) except Super2."
