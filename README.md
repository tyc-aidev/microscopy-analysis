# microscopy-analysis

Reproduction of [Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset](https://www.nature.com/articles/s41524-022-00878-5) (Stuckner et al., 2022), using the [NASA pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models) codebase and benchmark datasets.

## Current status

- **Dataset Explorer** — Streamlit app to browse NASA public datasets before training ([#8](https://github.com/tyc-aidev/microscopy-analysis/issues/8), merged in [#9](https://github.com/tyc-aidev/microscopy-analysis/pull/9)).
- **Sprint 0** — reproduction foundation and smoke test ([#1](https://github.com/tyc-aidev/microscopy-analysis/issues/1)): MicroNet **v1.0** weight pinning, model factory, smoke-test harness.
- **Sprint 1 scaffold** — config-driven training entrypoint with dataset adapter and two-phase trainer skeleton ([#2](https://github.com/tyc-aidev/microscopy-analysis/issues/2)).

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

Dependencies for the explorer live in `requirements-explorer.txt` (Streamlit, Pillow, pandas, matplotlib — no PyTorch). `./scripts/setup_env.sh` creates the venv, installs the explorer package in editable mode, and installs those requirements. The editable install is a local convenience only — at runtime imports resolve via `explorer/_bootstrap.py`, which puts the repo root on `sys.path`.

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

You only need the explorer dependencies installed (`pip install -r requirements-explorer.txt`); the editable package install is not required because `explorer/_bootstrap.py` handles imports.

### Deploy to Streamlit Community Cloud

- **Entrypoint:** `explorer/app.py`
- **Dependencies:** Cloud auto-installs from `explorer/requirements.txt` (found before any root `requirements.txt`, which is reserved for the PyTorch stack). That file re-uses the canonical `requirements-explorer.txt`.
- **Imports:** resolved by `explorer/_bootstrap.py` — no `pip install -e .` needed (Cloud never runs `setup_env.sh`). `pyproject.toml` is ignored by Cloud and is for local installs only.
- **Data:** `data/` is gitignored, so a fresh Cloud deploy starts empty and the app shows download prompts. To ship data, point `DATA_ROOT` at a mounted volume or bake assets into the image/repo.

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

## Deploy to Streamlit Community Cloud

Streamlit Cloud never runs `setup_env.sh`/`download_data.sh` and has an ephemeral
filesystem, so the app fetches a prepackaged dataset archive from
[Cloudflare R2](https://developers.cloudflare.com/r2/) at runtime
([#17](https://github.com/tyc-aidev/microscopy-analysis/issues/17)). R2 has **$0
egress**, which suits Cloud re-downloading on every cold start. NASA data is MIT,
so a public bucket is fine.

### 1. Build the archive

Download data locally, then package the trimmed (Cloud) subset:

```bash
./scripts/download_data.sh            # or --sample for a lighter local checkout
./scripts/build_data_archive.sh       # --sample (default) | --full
```

This writes `data/dist/amat-data-<mode>.tar.zst`, rooted at
`benchmark_segmentation_data/`, `instance_segmentation/`, and `examples/` so the
app can extract it straight into `DATA_ROOT`. Use `--format tar.gz` or `--format
zip` if you prefer not to ship `zstandard`.

| Mode | Contents | Compressed | Extracted |
|------|----------|-----------|-----------|
| `--sample` (default) | Super1–4 (all splits) + instance annotations/validation + 2 example images | ~33 MB | ~135 MB |
| `--full` | All 7 benchmarks (Super1–4 + EBC1–3), full instance-seg (train/val/annotations), complete examples gallery | ~169 MB | ~483 MB |

### 2. Upload to R2

```bash
wrangler r2 bucket create microscopy-analysis-datasets
wrangler r2 object put microscopy-analysis-datasets/amat-data-sample.tar.zst \
  --file data/dist/amat-data-sample.tar.zst --content-type application/zstd --remote
wrangler r2 bucket dev-url enable microscopy-analysis-datasets   # or attach a custom domain
```

Both archives are provisioned in the bucket
([#19](https://github.com/tyc-aidev/microscopy-analysis/issues/19),
[#22](https://github.com/tyc-aidev/microscopy-analysis/issues/22)):

```
https://pub-9aef84b8fae545b9a233bfb899a636ae.r2.dev/amat-data-full.tar.zst     # ~169 MB (all 7 benchmarks)
https://pub-9aef84b8fae545b9a233bfb899a636ae.r2.dev/amat-data-sample.tar.zst   # ~33 MB  (trimmed, disk-constrained)
```

The **full** archive extracts to ~483 MB and peaks near ~660 MB transiently
during download+extract; this fits Streamlit Community Cloud's ephemeral `/tmp`
([#25](https://github.com/tyc-aidev/microscopy-analysis/issues/25)). Prefer the
trimmed **sample** only on disk-constrained hosts.

### 3. Configure Streamlit secrets (required for remote fetch)

Remote fetching is **explicit opt-in**: with no source configured the app shows
its download prompt and fetches nothing. To serve data on Streamlit Cloud, set
the public archive URL in the app's **Settings → Secrets** (use the full URL to
browse all 7 benchmarks), then **reboot the app**:

```toml
DATA_ARCHIVE_URL = "https://pub-9aef84b8fae545b9a233bfb899a636ae.r2.dev/amat-data-full.tar.zst"
```

For a private bucket, use S3-compatible credentials instead (downloaded via `boto3`):

```toml
R2_ENDPOINT = "https://<account-id>.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = "..."
R2_SECRET_ACCESS_KEY = "..."
R2_BUCKET = "microscopy-analysis-datasets"
R2_OBJECT_KEY = "amat-data-sample.tar.zst"
```

On first load, `explorer.lib.remote_data.ensure_data()` downloads and extracts the
archive once per container into `/tmp/amat-data` (override with `REMOTE_DATA_ROOT`)
and drops a `.ready` marker. Local runs with populated `DATA_ROOT` skip the
download entirely. Set the app entry point to `explorer/app.py`.

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

### Apple Silicon (MPS) — local training and iteration

To train/iterate locally on an Apple Silicon Mac (Metal / MPS), use the modern
stack in `requirements-apple.txt` (torch 2.x + modern `segmentation_models_pytorch`).
`amat.models.create_segmentation_model` builds via smp directly and loads the v1.0
encoder weights itself, so NASA's `pmm` (and its torch-1.10 pins) is **not** needed
for build / forward / smoke training:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-apple.txt
python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml --build --device auto
```

`scripts/smoke_test.py` and `amat.device.resolve_device()` auto-select
`cuda → mps → cpu` and enable `PYTORCH_ENABLE_MPS_FALLBACK` for the handful of ops
not yet implemented on MPS. Verified on Apple Silicon (torch 2.x / smp 0.5.x): the
v1.0 weights for `resnet50` and `se_resnext50_32x4d` load cleanly and a forward pass
runs on `mps`.

> Caveat: MPS uses a different torch/smp than the paper-pinned env, so it is for
> **development and iteration**. Bit-exact paper numbers still come from the CUDA
> host with `requirements.txt` (issue #16).

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

## Sprint 1 CLI (scaffold)

Run the baseline config:

```bash
python scripts/train.py --config configs/experiments/super1_baseline.yaml
```

Current scaffold writes `metrics.json`, `checkpoint.pth.tar`, and `run_summary.json` under `results/<run_name>/`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
