# Sprint 0 — notebook baseline record

Manually recorded results from running the NASA example notebooks, used as the
reference for the Sprint 0 exit criterion: **smoke-train test IoU within ~2–5% of
the notebook baseline, with the same loss-curve shape.**

> Status: measured values are `_TBD (run on GPU host)`. The pinned reproduction
> stack (torch 1.10.1 / cu113) cannot run on Apple Silicon / Python 3.12, so the
> build, forward, and smoke-train numbers must be filled in on a CUDA host.
> Full runbook: [`configs/cloud/gcp_t4_repro.md`](../configs/cloud/gcp_t4_repro.md).

Source notebooks (NASA `pretrained-microscopy-models/examples/`, land in
`data/examples/` after `./scripts/download_data.sh`):

- Multiclass: `Super1` — `UnetPlusPlus` + `resnet50` + `micronet` (v1.0), 3 classes
- Binary: `EBC1` — `UnetPlusPlus` + `se_resnext50_32x4d` + `micronet` (v1.0), 1 class

Reproduce with:

```bash
python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml --build --check-url
python scripts/smoke_test.py --config configs/experiments/ebc1_smoke.yaml  --build --check-url
```

## Model / config (static — from configs/experiments/*.yaml)

| Config | Dataset | Task | Architecture | Encoder | Pretraining | MicroNet ver | Classes |
|---|---|---|---|---|---|---|---|
| `super1_smoke` | Super1 | multiclass | `UnetPlusPlus` | `resnet50` | `micronet` | **1.0** | 3 |
| `ebc1_smoke` | EBC1 | binary oxide | `UnetPlusPlus` | `se_resnext50_32x4d` | `micronet` | **1.0** | 1 |

v1.0 encoder weight URLs (asserted by the smoke test; HTTP 200 verified from S3):

- `https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/resnet50_pretrained_microscopynet_v1.0.pth.tar`
- `https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/se_resnext50_32x4d_pretrained_microscopynet_v1.0.pth.tar`

## Hyperparameters (static — PLAN.md / NASA notebooks)

- Optimizer: Adam, `lr=2e-4`; two-phase — phase 2 resumes at `1e-5`
- Loss: `DiceBCELoss(weight=0.7)`
- Metric: `IoU(threshold=0.5)` (per-class + mean)
- Early stopping: `patience=30` on val IoU (smoke run caps epochs low)
- Batch size: 6
- Normalization: ImageNet (even for MicroNet encoders)

## Baseline metrics (fill in on the CUDA reproduction host)

| Config | Notebook test IoU | Our smoke test IoU | Δ | Within 2–5%? | Loss-curve shape matches? |
|---|---|---|---|---|---|
| `super1_smoke` (Super1, resnet50, micronet v1.0) | _TBD (run on GPU host)_ | _TBD (run on GPU host)_ | _TBD_ | _TBD_ | _TBD_ |
| `ebc1_smoke` (EBC1, se_resnext50_32x4d, micronet v1.0) | _TBD (run on GPU host)_ | _TBD (run on GPU host)_ | _TBD_ | _TBD_ | _TBD_ |

## Environment

- GPU / CUDA: _TBD (run on GPU host)_ — target NVIDIA T4 (sm_75), CUDA 11.3 (cu113)
- `torch`: 1.10.1, `torchvision`: 0.11.2, `segmentation-models-pytorch`: 0.2.1 (pinned in `requirements.txt` / `requirements-cuda.txt`)
- Python: 3.9 (torch 1.10.1 wheels are cp36–cp39 only)
- `pretrained-microscopy-models` commit: _TBD (run on GPU host)_ — record `pip freeze | grep pretrained-microscopy-models`; remote HEAD at authoring time was `9b7c4abc1321e81eca7a68d548e5371676fa74fa`
- Seed: _TBD (run on GPU host)_

## Notes

- MicroNet weights **must** be v1.0 (paper version); the smoke test asserts the
  pinned URL and confirms HTTP 200 from NASA S3.
- Torch-free checks verified locally on Apple Silicon: config loads, and both v1.0
  weight URLs return HTTP 200. The `data` check FAILs only until
  `./scripts/download_data.sh` has run; the `model`/`forward` checks SKIP without
  the torch stack.
- Record any deviations from NASA notebook hyperparameters here with rationale
  (e.g. epoch cap for the smoke run vs. full two-phase `patience=30` schedule).
