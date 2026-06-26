# microscopy-analysis

Reproduction of [Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset](https://www.nature.com/articles/s41524-022-00878-5) (Stuckner et al., 2022), using the [NASA pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models) codebase and benchmark datasets.

## Current status

**Dataset Explorer** — Streamlit app to browse NASA public datasets before training ([#8](https://github.com/tyc-aidev/microscopy-analysis/issues/8)).

Implementation is tracked in PR [#9](https://github.com/tyc-aidev/microscopy-analysis/pull/9). Next: **Sprint 0** (environment setup and smoke tests).

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

This writes `data/dist/amat-data-sample.tar.zst`, rooted at
`benchmark_segmentation_data/`, `instance_segmentation/`, and `examples/` so the
app can extract it straight into `DATA_ROOT`. Use `--format tar.gz` or `--format
zip` if you prefer not to ship `zstandard`.

### 2. Upload to R2

```bash
wrangler r2 bucket create amat-datasets
wrangler r2 object put amat-datasets/amat-data-sample.tar.zst \
  --file data/dist/amat-data-sample.tar.zst --remote
```

Enable a public bucket URL (or attach a custom domain) in the Cloudflare dashboard.

### 3. Configure Streamlit secrets

In the app's **Settings → Secrets**, set the public archive URL:

```toml
DATA_ARCHIVE_URL = "https://<public-r2-domain>/amat-data-sample.tar.zst"
```

For a private bucket, use S3-compatible credentials instead (downloaded via `boto3`):

```toml
R2_ENDPOINT = "https://<account-id>.r2.cloudflarestorage.com"
R2_ACCESS_KEY_ID = "..."
R2_SECRET_ACCESS_KEY = "..."
R2_BUCKET = "amat-datasets"
R2_OBJECT_KEY = "amat-data-sample.tar.zst"
```

On first load, `explorer.lib.remote_data.ensure_data()` downloads and extracts the
archive once per container into `/tmp/amat-data` (override with `REMOTE_DATA_ROOT`)
and drops a `.ready` marker. Local runs with populated `DATA_ROOT` skip the
download entirely. Set the app entry point to `explorer/app.py`.

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
