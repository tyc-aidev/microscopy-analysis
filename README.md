# microscopy-analysis

Reproduction of [Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset](https://www.nature.com/articles/s41524-022-00878-5) (Stuckner et al., 2022), using the [NASA pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models) codebase and benchmark datasets.

## Current status

- **Dataset Explorer** — Streamlit app to browse NASA public datasets before training ([#8](https://github.com/tyc-aidev/microscopy-analysis/issues/8), merged in [#9](https://github.com/tyc-aidev/microscopy-analysis/pull/9)).
- **Sprint 0** — reproduction foundation and smoke test ([#1](https://github.com/tyc-aidev/microscopy-analysis/issues/1)): MicroNet **v1.0** weight pinning, model factory, smoke-test harness.

See [PLAN.md](PLAN.md) for the full reproduction plan and sprint breakdown. Explorer design: [PLAN_DATASET_EXPLORER.md](PLAN_DATASET_EXPLORER.md).

## Python environment

This project uses [uv](https://docs.astral.sh/uv/) for virtualenv and package management. Virtualenvs are created with **Python 3.12** explicitly.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. From the repo root, run:

   ```bash
   ./scripts/setup_env.sh
   ```

3. Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

Dependencies for the explorer live in `requirements-explorer.txt` (Streamlit, Pillow, pandas, matplotlib — no PyTorch). `./scripts/setup_env.sh` creates the venv, installs the explorer package in editable mode, and installs those requirements.

## Dataset Explorer

Local Streamlit app to browse semantic benchmarks (Super1–4, EBC1–3), instance-segmentation melt-pool tiles, and NASA reference examples.

### 1. Download data

Benchmark TIFFs, COCO instance-seg tiles, and example assets are downloaded into `data/` (gitignored):

```bash
./scripts/download_data.sh
```

Optional custom location:

```bash
export DATA_ROOT=/path/to/datasets
./scripts/download_data.sh
```

See [data/README.md](data/README.md) for layout details.

### 2. Run the app

From the repo root with the venv activated:

```bash
./scripts/run_explorer.sh
```

Opens at [http://localhost:8501](http://localhost:8501). Alternative:

```bash
streamlit run explorer/app.py
```

(`uv pip install -e .` is required if you skip `./scripts/setup_env.sh`.)

### 3. Pages

| Page | Content |
|------|---------|
| **Home** | Dataset overview cards, download status, aggregate stats (images per split, class pixel distribution) |
| **Benchmarks** | Filterable Super/EBC browser with mask overlays |
| **Instance Segmentation** | Melt-pool COCO tiles with bbox and polygon overlays |
| **Examples** | Reference images from NASA notebooks |

### 4. Tests

```bash
pytest tests/ -q
```

## MicroNet reproduction (Sprint 0+)

The PyTorch reproduction (`src/amat/`) reproduces Stuckner et al. 2022 using NASA's
released **MicroNet v1.0** encoder weights. NASA's `create_segmentation_model`
loads weights without a version and defaults to **v1.1** for `resnet50/micronet`,
so `src/amat/models/` pins **v1.0** explicitly — never v1.1.

### Reproduction environment (CUDA host)

The reproduction stack pins torch 1.10.1 / smp 0.2.1 to match NASA's frozen
environment, so v1.0 weights load without state-dict drift. These wheels target
CUDA Linux + CPython 3.8–3.10 and are **not** expected to install on Python 3.12 /
Apple Silicon:

```bash
python -m venv .venv-repro && source .venv-repro/bin/activate
pip install -r requirements.txt
```

### Smoke test (Sprint 0)

The smoke test runs every check the environment allows (config, data pairing,
pinned v1.0 weight URL, and — on a torch host — model build + forward pass):

```bash
# torch-free checks anywhere (config, data, weight URL pinning):
python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml --check-url

# full end-to-end on the CUDA reproduction host:
python scripts/smoke_test.py --config configs/experiments/ebc1_smoke.yaml --build --check-url
```

Record notebook-vs-reproduction metrics in [paper/notebook_baseline.md](paper/notebook_baseline.md).

### Reproduction tests

```bash
pytest tests/test_repro_weights.py tests/test_repro_config.py tests/test_repro_datasets.py -q
```

## Sprints

| Sprint | Focus | Issue |
|--------|-------|-------|
| — | Dataset Explorer (pre-Sprint-0) | [#8](https://github.com/tyc-aidev/microscopy-analysis/issues/8) |
| 0 | Foundation and smoke test | [#1](https://github.com/tyc-aidev/microscopy-analysis/issues/1) |
| 1 | Production training pipeline | [#2](https://github.com/tyc-aidev/microscopy-analysis/issues/2) |
| 2 | Core benchmark matrix | [#3](https://github.com/tyc-aidev/microscopy-analysis/issues/3) |
| 3 | Low-data ablations | [#4](https://github.com/tyc-aidev/microscopy-analysis/issues/4) |
| 4 | Architecture and encoder sweep | [#5](https://github.com/tyc-aidev/microscopy-analysis/issues/5) |
| 5 | Validation and reproduction report | [#6](https://github.com/tyc-aidev/microscopy-analysis/issues/6) |
| 6 | Instance segmentation (optional) | [#7](https://github.com/tyc-aidev/microscopy-analysis/issues/7) |

## License

Apache 2.0 — see [LICENSE](LICENSE).
