# Sprint 0 — notebook baseline record

Manually recorded results from running the NASA example notebooks, used as the
reference for the Sprint 0 exit criterion: **smoke-train test IoU within ~2–5% of
the notebook baseline, with the same loss-curve shape.**

Source notebooks (NASA `pretrained-microscopy-models/examples/`):

- Multiclass: `Super1` — `UnetPlusPlus` + `resnet50` + `micronet` (v1.0)
- Binary: `EBC1` — `UnetPlusPlus` + `se_resnext50_32x4d` + `micronet` (v1.0)

Reproduce with:

```bash
python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml --build --check-url
python scripts/smoke_test.py --config configs/experiments/ebc1_smoke.yaml  --build --check-url
```

## Baseline metrics (fill in on the CUDA reproduction host)

| Config | Notebook test IoU | Our smoke test IoU | Δ | Within 2–5%? | Loss-curve shape matches? |
|---|---|---|---|---|---|
| `super1_smoke` (Super1, resnet50, micronet v1.0) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| `ebc1_smoke` (EBC1, se_resnext50_32x4d, micronet v1.0) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## Environment

- GPU / CUDA: _TBD_
- `torch`: 1.10.1, `segmentation-models-pytorch`: 0.2.1 (per `requirements.txt`)
- `pretrained-microscopy-models` commit: _TBD_
- Seed: _TBD_

## Notes

- MicroNet weights **must** be v1.0 (paper version); the smoke test asserts the
  pinned URL and confirms HTTP 200 from NASA S3.
- Record any deviations from NASA notebook hyperparameters here with rationale.
