# AGENTS.md

## Cursor Cloud specific instructions

This repo is a Python 3.12 monorepo with two products (see `README.md`): the
**Dataset Explorer** (Streamlit, no PyTorch) and the **MicroNet reproduction**
(PyTorch training pipeline). The cloud VM is **CPU-only Linux** (not Apple
Silicon / CUDA). The update script already creates `.venv` and installs all deps,
so the notes below are only the non-obvious caveats.

### Environment / dependencies
- `uv` is the package manager, installed at `~/.local/bin/uv`. The `.venv` is
  created by `uv venv` and therefore has **no `pip`** — install with
  `uv pip install --python .venv/bin/python ...`, or just run tools directly via
  `.venv/bin/python -m <tool>`.
- The reproduction stack is installed from `requirements-apple.txt` (modern
  `torch>=2.2` + `segmentation-models-pytorch`) using the CPU wheel index
  (`--extra-index-url https://download.pytorch.org/whl/cpu`). The paper-pinned
  `requirements.txt` (torch 1.10.1, CUDA, Python 3.8–3.10) is **not installable**
  on this Python 3.12 / CPU VM — do not try to install it here. It only matters
  for bit-exact paper numbers on a dedicated CUDA host.
- There is **no linter/formatter** configured (no ruff/black/flake8/mypy, no CI
  workflows). "Lint" is not a runnable step in this repo.

### Dataset Explorer (Streamlit)
- Requires data on disk first: `./scripts/download_data.sh --sample` sparse-clones
  NASA's repo into `data/_upstream` and symlinks it under `data/` (needs network).
  Without data the app still loads but only shows download prompts.
- Run headless: `streamlit run explorer/app.py --server.headless true --server.port 8501`
  (or `./scripts/run_explorer.sh`). Imports resolve via `explorer/_bootstrap.py`,
  so an editable install is not required.

### MicroNet reproduction (PyTorch)
- Smoke test: `python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml`
  builds the real model and **downloads the NASA v1.0 encoder weights from S3**
  (`nasa-public-data.s3.amazonaws.com`, ~98 MB, cached under `~/.cache/torch`).
  Needs network the first time.
- Train: `python scripts/train.py --config configs/experiments/super1_baseline.yaml`.
  For a fast CPU check, cap epochs/batch:
  `--device cpu --batch-size 2 --max-epochs-phase1 1 --max-epochs-phase2 1`.
- Outputs go to `results/<run_name>/` (gitignored). Checkpoints are large
  (`checkpoint.pth` ~580 MB); leave them out of commits.

### Tests
- Run with `.venv/bin/python -m pytest tests/ -q` (config in `pytest.ini`).
- The single `slow` test (real smp model end-to-end, CPU, random init — no
  download) is deselected by default; run it with `RUN_SLOW=1 .venv/bin/python -m pytest tests/test_train_pipeline.py -q`.
