#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/getting-started/installation/)" >&2
  exit 1
fi

echo "Creating virtualenv with Python 3.12 at .venv ..."
uv venv --python 3.12 .venv

# shellcheck source=/dev/null
source .venv/bin/activate

echo "Installing explorer package (editable) ..."
uv pip install -e .

if [[ -f requirements-explorer.txt ]]; then
  echo "Installing explorer dependencies ..."
  uv pip install -r requirements-explorer.txt
elif [[ -f requirements.txt ]]; then
  echo "Installing dependencies ..."
  uv pip install -r requirements.txt
else
  echo "No requirements-explorer.txt or requirements.txt found; venv is ready for manual installs."
fi

echo
echo "Environment ready. Activate with:"
echo "  source .venv/bin/activate"
