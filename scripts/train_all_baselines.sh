#!/usr/bin/env bash
# Train all seven baseline experiments on the current machine.
#
# Defaults are safe for local Apple Silicon with SENet-154:
#   DEVICE=mps BATCH_SIZE=2 ./scripts/train_all_baselines.sh
#
# Epoch and patience overrides are optional. If omitted, each YAML's full
# baseline schedule is used (typically best for overnight runs).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

DEVICE="${DEVICE:-mps}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_EPOCHS_PHASE1="${MAX_EPOCHS_PHASE1:-}"
MAX_EPOCHS_PHASE2="${MAX_EPOCHS_PHASE2:-}"
PATIENCE="${PATIENCE:-}"

CONFIGS=(
  super1_baseline
  super2_baseline
  super3_baseline
  super4_baseline
  ebc1_baseline
  ebc2_baseline
  ebc3_baseline
)

for name in "${CONFIGS[@]}"; do
  config="configs/experiments/${name}.yaml"
  args=(
    --config "$config"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
  )

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

echo "Done. ${#CONFIGS[@]} baseline run(s)."
