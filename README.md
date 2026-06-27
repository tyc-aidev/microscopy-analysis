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
https://pub-9aef84b8fae545b9a233bfb899a636ae.r2.dev/amat-data-full.tar.zst     # ~169 MB (all 7 benchmarks — default)
https://pub-9aef84b8fae545b9a233bfb899a636ae.r2.dev/amat-data-sample.tar.zst   # ~33 MB  (trimmed, disk-constrained)
```

The app **defaults to the full archive** ([`DEFAULT_ARCHIVE_URL`](explorer/lib/remote_data.py)),
so a fresh deploy serves all 7 benchmarks with no secret configured. The full
archive extracts to ~483 MB and peaks near ~660 MB transiently during
download+extract; this fits Streamlit Community Cloud's ephemeral `/tmp`
([#25](https://github.com/tyc-aidev/microscopy-analysis/issues/25)). On a
disk-constrained host, pin the trimmed **sample** instead (see below).

### 3. Configure Streamlit secrets (optional)

No secret is required — the app fetches the full archive by default. To pin a
different source (e.g. the trimmed sample on a constrained host), set the public
archive URL in the app's **Settings → Secrets**:

```toml
DATA_ARCHIVE_URL = "https://pub-9aef84b8fae545b9a233bfb899a636ae.r2.dev/amat-data-sample.tar.zst"
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
