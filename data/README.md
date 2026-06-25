# Local datasets

Downloaded NASA benchmark and demo assets live here. This directory is **gitignored** except for this README.

## Download

From the repo root:

```bash
./scripts/download_data.sh
```

Optional: set a custom location via `DATA_ROOT` (used by the explorer and download script):

```bash
export DATA_ROOT=/path/to/datasets
./scripts/download_data.sh
```

## Layout (after download)

```
data/
├── benchmark_segmentation_data/   # Super1–4, EBC1–3 semantic benchmarks
├── instance_segmentation/       # COCO melt-pool demo tiles
├── examples/                    # Reference images from NASA notebooks
└── _upstream/                   # sparse git clone (source of symlinks)
```

## Explorer

```bash
source .venv/bin/activate
streamlit run explorer/app.py
```

See [PLAN_DATASET_EXPLORER.md](../PLAN_DATASET_EXPLORER.md) for details.
