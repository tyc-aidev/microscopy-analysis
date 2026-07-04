#!/usr/bin/env bash
# Train all seven datasets with UnetPlusPlus + se_resnet50.
#
# By default this uses each YAML's full training schedule and batch size.
# Optional overrides:
#   DEVICE=mps BATCH_SIZE=4 MAX_EPOCHS_PHASE1=10 MAX_EPOCHS_PHASE2=5 PATIENCE=8 \
#     ./scripts/train_all_se_resnet50.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_EPOCHS_PHASE1="${MAX_EPOCHS_PHASE1:-}"
MAX_EPOCHS_PHASE2="${MAX_EPOCHS_PHASE2:-}"
PATIENCE="${PATIENCE:-}"

CONFIGS=(
  super1_se_resnet50
  super2_se_resnet50
  super3_se_resnet50
  super4_se_resnet50
  ebc1_se_resnet50
  ebc2_se_resnet50
  ebc3_se_resnet50
)

for name in "${CONFIGS[@]}"; do
  config="configs/experiments/${name}.yaml"
  args=(
    --config "$config"
    --device "$DEVICE"
  )

  if [[ -n "$BATCH_SIZE" ]]; then
    args+=(--batch-size "$BATCH_SIZE")
  fi
  if [[ -n "$MAX_EPOCHS_PHASE1" ]]; then
    args+=(--max-epochs-phase1 "$MAX_EPOCHS_PHASE1")
  fi
  if [[ -n "$MAX_EPOCHS_PHASE2" ]]; then
    args+=(--max-epochs-phase2 "$MAX_EPOCHS_PHASE2")
  fi
  if [[ -n "$PATIENCE" ]]; then
    args+=(--patience "$PATIENCE")
  fi

  echo "=== Training ${name} (${config}) ==="
  python scripts/train.py "${args[@]}"
  echo
done

echo "Done. ${#CONFIGS[@]} se_resnet50 run(s)."
