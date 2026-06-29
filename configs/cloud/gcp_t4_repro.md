# GCP T4 runbook — Sprint 0 reproduction (issue #16)

Copy-pasteable runbook to execute the Sprint 0 smoke test and notebook baseline
([#16](https://github.com/tyc-aidev/microscopy-analysis/issues/16)) on a NVIDIA
T4 GPU host on Google Cloud. The pinned reproduction stack
(`torch==1.10.1` / `torchvision==0.11.2` / `segmentation-models-pytorch==0.2.1`,
cu113 wheels) cannot install on Apple Silicon / Python 3.12, so the model-build,
forward-pass, and smoke-train acceptance criteria must run here.

> Cost note: a preemptible/spot `n1-standard-4` + 1× T4 in `us-central1` is roughly
> **$0.15–0.25/hr**. The whole runbook is well under an hour of GPU time. **Delete
> the instance when done** (see [§7](#7-cleanup-stop-the-billing)).

---

## 0. Why T4 / this machine shape

| Choice | Value | Rationale |
|---|---|---|
| GPU | `nvidia-tesla-t4` (Turing, **sm_75**) | cu113 wheels ship SASS for sm_75. **Avoid L4/Ada (sm_89) and Hopper (sm_90)** — those need CUDA ≥ 11.8, so cu113 PyTorch 1.10.1 will fail with "no kernel image is available for execution on the device". P100/V100 also work; T4 is the cheapest safe match. |
| Machine | `n1-standard-4` (4 vCPU / 15 GB) | T4 only attaches to N1. 15 GB RAM is plenty for batch=6 at 512². |
| Python | **3.9** | torch 1.10.1 wheels are `cp36`–`cp39` only. There is no cp310+ build, so 3.10/3.11/3.12 cannot install the pinned stack. |
| CUDA driver | ≥ 465 (CUDA 11.3 runtime) | The cu113 wheels bundle their own CUDA runtime; only a compatible **driver** is needed. The Deep Learning VM ships one. |
| Provisioning | spot/preemptible | Smoke run is short and restartable; cheapest option. |

---

## 1. Prerequisites (local)

```bash
# This repo's gcloud is broken under Python 3.12 (No module named 'imp').
# Point it at an older interpreter for every gcloud invocation:
export CLOUDSDK_PYTHON="$(command -v python3.11 || command -v python3.10 || command -v python3.9)"

gcloud auth login                      # interactive: cached creds may be expired
gcloud config set project YOUR_PROJECT_ID
gcloud config set account YOUR_ACCOUNT

# Sanity: confirm billing is enabled and a T4 quota exists in your region.
gcloud billing projects describe YOUR_PROJECT_ID
gcloud compute regions describe us-central1 --format='value(quotas)' | tr ';' '\n' | grep -i nvidia
```

Pick a zone with T4 stock, e.g. `us-central1-a`, `us-central1-b`, `us-east1-c`,
`europe-west4-a`. If you hit `ZONE_RESOURCE_POOL_EXHAUSTED`, try another zone.

---

## 2. Create the instance

### Option A — Deep Learning VM (recommended, driver pre-installed)

```bash
export ZONE=us-central1-b
export INSTANCE=micronet-repro-t4

gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu113-debian-11 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True"
```

On first SSH the DLVM prompts to install the NVIDIA driver (the metadata flag
auto-accepts). The `common-cu113-*` image ships a CUDA 11.3-compatible driver,
matching our wheels.

### Option B — Ubuntu 22.04 + manual driver

Use only if a DLVM image is unavailable. cu113 wheels bundle their CUDA runtime,
so just a recent driver is required.

```bash
gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced

gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command='
  sudo apt-get update -y &&
  sudo apt-get install -y ubuntu-drivers-common python3.9 python3.9-venv git &&
  sudo ubuntu-drivers autoinstall &&
  sudo reboot'   # wait ~60s for reboot, then re-SSH
```

---

## 3. SSH in and verify the GPU

```bash
gcloud compute ssh "$INSTANCE" --zone="$ZONE"
# on the VM:
nvidia-smi          # expect a Tesla T4, driver >= 465
python3.9 --version # expect 3.9.x
```

---

## 4. Clone repo + build the reproduction env (on the VM)

```bash
git clone https://github.com/tyc-aidev/microscopy-analysis.git
cd microscopy-analysis
git checkout repro/sprint0-cuda-baseline      # this PR's branch

# Python 3.9 venv (torch 1.10.1 has no cp310+ wheels).
python3.9 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install --upgrade pip

# Install the pinned GPU stack. requirements-cuda.txt bakes in the cu113 index;
# equivalently: pip install -r requirements.txt with the find-links flag.
pip install -r requirements-cuda.txt

# Record the exact resolved pmm commit for paper/notebook_baseline.md:
pip show pretrained-microscopy-models | grep -i version
python - <<'PY'
import importlib.metadata as m; print("pmm dist version:", m.version("pretrained-microscopy-models"))
PY
pip freeze | grep -i pretrained-microscopy-models   # shows the @ <sha> it built from
```

Sanity-check the GPU torch build:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
# expect: torch 1.10.1+cu113 cuda 11.3 avail True / Tesla T4
```

---

## 5. Download data (Super1 + EBC1 both needed)

```bash
./scripts/download_data.sh        # sparse checkout of NASA benchmark_segmentation_data + examples
ls data/benchmark_segmentation_data/Super1 data/benchmark_segmentation_data/EBC1
ls data/examples                  # NASA example notebooks land here
```

`download_data.sh` (no flag) fetches all 7 benchmarks; Super1 and EBC1 are both
included. The NASA example notebooks used for the baseline land in `data/examples/`.

---

## 6. Run the acceptance criteria

### Criterion 2 + 3 — smoke test (build + v1.0 weight load + forward)

```bash
python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml --build --check-url
python scripts/smoke_test.py --config configs/experiments/ebc1_smoke.yaml  --build --check-url
```

Expect every line `[PASS]` and a final `Result: OK`. Key lines to confirm:

- `[PASS] weights: HTTP 200 ... resnet50_pretrained_microscopynet_v1.0.pth.tar`
  (and `se_resnext50_..._v1.0...` for EBC1) — the pin is **v1.0**, never v1.1.
- `[PASS] model: built UnetPlusPlus/<encoder> (micronet)` — v1.0 encoder weights
  loaded into `model.encoder`.
- `[PASS] forward: output shape (1, C, 256, 256) on cuda`.

The first `--build` downloads the ~100 MB v1.0 encoder weights into the torch hub
cache (`~/.cache/torch/hub/checkpoints/`); later runs reuse it.

### Criterion 4 — short smoke train via NASA's loop

Use NASA's own training loop (`pmm.segmentation_training.train_segmentation_model`),
**not** this repo's `scripts/train.py`. Match the NASA example-notebook
hyperparameters from PLAN.md, but cap epochs low so it's a smoke run:

| Hyperparameter | Value (PLAN.md) |
|---|---|
| Optimizer | Adam, `lr=2e-4` (phase 2 resumes at `1e-5`) |
| Loss | `DiceBCELoss(weight=0.7)` |
| Metric | `IoU(threshold=0.5)` (per-class + mean) |
| Early stopping | `patience=30` on val IoU (smoke: set tiny, see below) |
| Batch size | 6 |
| Normalization | ImageNet (even for MicroNet encoders) |

A smoke run should cap epochs (e.g. `epochs=5–10`, `patience=3`) just to confirm
the loss curve falls and val IoU climbs in the right shape — not to reach the full
baseline. Reference adapter (run on the VM; adjust dataset wiring to NASA's
`pmm.io` loaders / the example notebook for the dataset's augmentation profile):

```python
# smoke_train.py — minimal NASA-loop smoke train. Run on the GPU host only.
import segmentation_models_pytorch as smp
from pretrained_microscopy_models import segmentation_training as st
from microscopy_analysis.models import create_segmentation_model

# Build with paper-correct v1.0 weights (mirrors the smoke_test build).
model = create_segmentation_model("UnetPlusPlus", "resnet50", "micronet", classes=3)  # Super1
# model = create_segmentation_model("UnetPlusPlus", "se_resnext50_32x4d", "micronet", classes=1)  # EBC1

loss = smp.utils.losses.DiceLoss() + smp.utils.losses.BCELoss()   # ~DiceBCELoss(0.7); see notebook
metrics = [smp.utils.metrics.IoU(threshold=0.5)]

# train_dataset / valid_dataset: build per the NASA example notebook in data/examples/
# (Super1 multiclass notebook for resnet50; EBC1 binary notebook for se_resnext50).
# Cap epochs for a smoke run:
st.train_segmentation_model(
    model, "UnetPlusPlus", "resnet50",
    train_dataset, valid_dataset,
    batch_size=6, lr=2e-4, patience=3, epochs=10,   # SMOKE caps; full run uses patience=30 two-phase
    save_name="super1_smoke",
)
```

> Defer to the NASA example notebooks (`data/examples/`) for the exact dataset
> construction, augmentation, and the precise `train_segmentation_model` signature
> of the installed pmm commit — the call above is the shape, not a guaranteed API.
> Run the matching notebook **once unmodified** to capture the *notebook baseline*
> test IoU, then run the smoke train above and compare (target: within ~2–5%, same
> loss-curve shape).

### Criterion 5 — fill the baseline record

Record measured numbers in `paper/notebook_baseline.md`:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch, segmentation_models_pytorch as smp; print(torch.__version__, smp.__version__)"
pip freeze | grep -i pretrained-microscopy-models   # the @ <sha> commit
```

Fill: notebook test IoU, our smoke-train test IoU, Δ, GPU/CUDA, torch/smp
versions, pmm commit SHA, and the seed used. Commit and push from the VM (or copy
the file back) on the same branch.

---

## 7. Cleanup (stop the billing)

```bash
# from your LOCAL machine (remember CLOUDSDK_PYTHON):
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
gcloud compute instances list   # confirm it's gone
```

`--instance-termination-action=DELETE` on a SPOT VM also auto-deletes on
preemption, but always run the explicit delete to be safe — a forgotten T4 is the
main cost risk here.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'imp'` from gcloud | `export CLOUDSDK_PYTHON=$(command -v python3.11)` before any gcloud call. |
| `invalid_grant: Bad Request` from gcloud | Cached creds expired; run `gcloud auth login` (interactive). |
| `no kernel image is available for execution` | Wrong GPU (L4/Ada). Recreate with `nvidia-tesla-t4` (sm_75). |
| `torch.cuda.is_available()` is False | Driver not installed/loaded — re-run driver install / reboot; check `nvidia-smi`. |
| pip can't find `torch==1.10.1+cu113` | Python is not 3.9 (no cp310+ wheels) or the `-f .../cu113/torch_stable.html` find-links is missing. |
| `ZONE_RESOURCE_POOL_EXHAUSTED` | Try another T4 zone (`us-east1-c`, `europe-west4-a`, ...). |
| weight URL not v1.0 | Bug — `micronet_weight_url` pins v1.0; never pass `allow_non_paper_version`. |
