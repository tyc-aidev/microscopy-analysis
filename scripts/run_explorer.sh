#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec streamlit run explorer/app.py "$@"
