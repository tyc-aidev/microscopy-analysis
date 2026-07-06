# microscopy-analysis

Reproduction of [Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset](https://www.nature.com/articles/s41524-022-00878-5) (Stuckner et al., 2022), using the [NASA pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models) codebase and benchmark datasets.

## Current status

- **Dataset Explorer** — Streamlit app to browse NASA public datasets before training ([#8](https://github.com/tyc-aidev/microscopy-analysis/issues/8), merged in [#9](https://github.com/tyc-aidev/microscopy-analysis/pull/9)).
- **Sprint 0** — reproduction foundation and smoke test ([#1](https://github.com/tyc-aidev/microscopy-analysis/issues/1)): MicroNet **v1.0** weight pinning, model factory, smoke-test harness.
- **Sprint 1 scaffold** — config-driven training entrypoint with dataset adapter and two-phase trainer skeleton ([#2](https://github.com/tyc-aidev/microscopy-analysis/issues/2)).
- **Local training explorer** — Streamlit pages for local MPS runs and imported CUDA reproduction runs (metrics, qualitative panels, config lookup).
- **Baseline configs** — ready-to-run local YAMLs for all 7 semantic benchmarks (`Super1`–`Super4`, `EBC1`–`EBC3`) using `senet154` + MicroNet v1.0.

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
| **Local Training** | Browse local MPS/CPU runs under `results/`, plot loss/IoU curves, and generate validation prediction panels |
| **Paper Reproduction CUDA** | Browse imported CUDA runs, review paper-pinned env requirements, and inspect synced prediction panels |

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

The PyTorch reproduction (`src/microscopy_analysis/`) reproduces Stuckner et al. 2022 using NASA's
released **MicroNet v1.0** encoder weights. NASA's `create_segmentation_model`
loads weights without a version and defaults to **v1.1** for `resnet50/micronet`,
so `src/microscopy_analysis/models/` pins **v1.0** explicitly — never v1.1.

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
`microscopy_analysis.models.create_segmentation_model` builds via smp directly and loads the v1.0
encoder weights itself, so NASA's `pmm` (and its torch-1.10 pins) is **not** needed
for build / forward / smoke training:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e . -r requirements-apple.txt
python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml --build --device auto
```

`scripts/smoke_test.py` and `microscopy_analysis.device.resolve_device()` auto-select
`cuda → mps → cpu` and enable `PYTORCH_ENABLE_MPS_FALLBACK` for the handful of ops
not yet implemented on MPS. Verified on Apple Silicon (torch 2.x / smp 0.5.x):
the baseline `senet154` MicroNet v1.0 weights load and build correctly on `mps`.
Because `senet154` is much larger than `resnet50`, local runs typically need a
smaller batch size (`1`–`2` on Apple Silicon).

### Baseline configs for local training

All baseline configs live under `configs/experiments/` and use:

- `architecture: UnetPlusPlus`
- `encoder_name: senet154`
- `pretraining: micronet` (MicroNet **v1.0**, never v1.1)
- `optimizer: Adam` with the paper-style two-phase schedule (`2e-4` → `1e-5`)
- `trainer: patience=30, max_epochs_phase1=120, max_epochs_phase2=60`

| Dataset | Config | Classes | Notes |
|---------|--------|---------|-------|
| `Super1` | `configs/experiments/super1_baseline.yaml` | 3 | Ni-superalloy multiclass benchmark |
| `Super2` | `configs/experiments/super2_baseline.yaml` | 3 | Ni-superalloy multiclass benchmark |
| `Super3` | `configs/experiments/super3_baseline.yaml` | 3 | Very low-data benchmark |
| `Super4` | `configs/experiments/super4_baseline.yaml` | 3 | Ni-superalloy multiclass benchmark |
| `EBC1` | `configs/experiments/ebc1_baseline.yaml` | 1 | Binary oxide task |
| `EBC2` | `configs/experiments/ebc2_baseline.yaml` | 1 | Binary oxide task |
| `EBC3` | `configs/experiments/ebc3_baseline.yaml` | 1 | Binary oxide task |

### Local training workflow

1. **Install the Apple Silicon training stack**

   ```bash
   uv venv --python 3.12 .venv && source .venv/bin/activate
   uv pip install -e . -r requirements-apple.txt
   ```

2. **Download the datasets**

   ```bash
   ./scripts/download_data.sh
   ```

   Use the full download for any `EBC*` run. `--sample` is only enough for the
   `Super1`–`Super4` benchmarks.

3. **Run one dataset locally**

   Quick smoke run:

   ```bash
   python scripts/train.py --config configs/experiments/super1_baseline.yaml \
     --device mps --batch-size 2 --max-epochs-phase1 6 --max-epochs-phase2 3 --patience 5
   ```

   Overnight / full baseline run:

   ```bash
   python scripts/train.py --config configs/experiments/super1_baseline.yaml \
     --device mps --batch-size 2
   ```

   Omitting the epoch overrides uses the config defaults (`120 / 60`) with early
   stopping on validation IoU, which is the recommended overnight setting.

4. **Loop all baselines except `Super2`**

   ```bash
   ./scripts/train_baselines_except_super2.sh
   ```

   The helper script defaults to a quick local smoke profile. For an overnight
   run, override the environment variables:

   ```bash
   DEVICE=mps BATCH_SIZE=2 PATIENCE=30 \
     MAX_EPOCHS_PHASE1=120 MAX_EPOCHS_PHASE2=60 \
     ./scripts/train_baselines_except_super2.sh
   ```

5. **Visualize metrics and predictions**

   ```bash
   ./scripts/run_explorer.sh
   ```

   Open the **Local Training** page in Streamlit to:

   - browse runs discovered under `results/`
   - inspect `metrics.json` loss / IoU curves automatically
   - select the matching config YAML
   - generate `val` prediction panels on demand (saved under `results/<run_name>/predictions/val/`)

The Sprint 1 trainer runs on MPS too. For a quick local smoke run, cap the epochs
so it finishes in well under a minute (full baseline uses 120 / 60 epochs):

```bash
python scripts/train.py --config configs/experiments/super1_baseline.yaml \
  --device mps --batch-size 4 --max-epochs-phase1 6 --max-epochs-phase2 3 --patience 5
```

This trains for real (loss decreases, val IoU computed per epoch) and writes a real
`checkpoint.pth`, `metrics.json`, and `run_summary.json` under `results/<run_name>/`.

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

## Sprint 1 CLI

Run the baseline two-phase training:

```bash
python scripts/train.py --config configs/experiments/super1_baseline.yaml
```

This runs the real pipeline — config-driven dataset + augmentation, `DiceBCELoss`,
two-phase Adam (`2e-4` → resume `1e-5`) with early stopping on validation IoU — and
writes a resumable `checkpoint.pth`, per-epoch `metrics.json` (train/val loss + per-class
and mean IoU), and `run_summary.json` under `results/<run_name>/`. Optional
`--device`, `--batch-size`, `--max-epochs-phase{1,2}`, and `--patience` flags override
the config (e.g. for a fast local MPS smoke run; see the Apple Silicon section above).

## Sprint 2 CLI

Sprint 2 reproduces the paper's central claim — MicroNet vs ImageNet pretraining
across all 7 benchmarks — via a generate → train → evaluate → aggregate pipeline.

**Evaluate** a trained run on the held-out test split (per-class + mean IoU):

```bash
python scripts/evaluate.py --config configs/experiments/super1_baseline.yaml --split test
```

Loads `results/<run_name>/model_best.pth` and writes `results/<run_name>/eval_test.json`.
Inference is whole-image (inputs padded up to a multiple of 32, then cropped back);
sliding-window patch inference is a Sprint 5 refinement.

**Generate** the 56-job benchmark matrix (7 datasets × 4 pretraining regimes ×
`UnetPlusPlus` × `{senet154, se_resnext50_32x4d}`) as a manifest + per-job configs,
and optionally train+evaluate locally (heavy — meant for a GPU/cloud fleet, so cap
with `--max-jobs` for a smoke run; cloud fan-out is [#39](https://github.com/tyc-aidev/microscopy-analysis/issues/39)):

```bash
python scripts/run_matrix.py                          # write manifest + configs only
python scripts/run_matrix.py --dispatch local --max-jobs 1 --device mps  # smoke: 1 job train+eval
```

**Aggregate** all `eval_test.json` files into the MicroNet-vs-ImageNet table (and,
where transcribed, a reproduced-vs-paper comparison from `paper/target_metrics.csv`):

```bash
python scripts/aggregate_results.py
```

Writes `results/benchmark_matrix.csv` and prints a markdown table plus the majority-win
summary (the Sprint 2 exit criterion). Populating paper baselines + running the full
matrix on CUDA is [#40](https://github.com/tyc-aidev/microscopy-analysis/issues/40).

## Sprint 3 CLI (low-data ablations)

Sprint 3 reproduces the paper's headline low-data finding — the large relative
IoU-**error** reduction from MicroNet pretraining when very few training images
are available — by sweeping the training-set size on the Super benchmarks and
comparing MicroNet vs ImageNet.

Training runs honour a deterministic, seeded `train_subsample` cap (val/test are
never capped), so `{1, 2, 4, 8, all}`-image runs use reproducible, comparable
subsets. A single-image config is provided:

```bash
python scripts/train.py --config configs/experiments/super3_low_data.yaml
```

**Generate** the low-data sweep (Super1–4 × `{1,2,4,8,all}` × `{imagenet, micronet}`
× `{senet154, se_resnext50_32x4d}` × `UnetPlusPlus`) as a manifest + per-job configs,
and optionally train+evaluate locally. Each encoder gets its own curve. When the
benchmark data is present, the size sweep is clamped to each dataset's real image
count (Super3 → just 1):

```bash
python scripts/run_low_data.py                                  # manifest + configs only
python scripts/run_low_data.py --dispatch local --max-jobs 2 --device cpu  # smoke
```

**Aggregate** the low-data `eval_test.json` files into per-`(dataset, encoder, #images)`
curves and the relative-error-reduction table, optionally rendering the figure:

```bash
python scripts/aggregate_low_data.py --plot
```

Writes `results/low_data_curves.csv` (+ `results/low_data_curves.png`) and prints
the MicroNet-vs-ImageNet relative IoU-error-reduction table plus the Sprint 3 exit
criteria (min-image MicroNet advantage and curve monotonicity). The relative error
reduction is `(IoU_micronet − IoU_imagenet) / (1 − IoU_imagenet)`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
