# microscopy-analysis

Reproduction of [Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset](https://www.nature.com/articles/s41524-022-00878-5) (Stuckner et al., 2022), using the [NASA pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models) codebase and benchmark datasets.

## Current status

**Planning phase** — building a [Dataset Explorer](#dataset-explorer) Streamlit app to visualize the public datasets, then **Sprint 0** (environment setup and smoke tests).

See [PLAN.md](PLAN.md) for the full reproduction plan, dataset availability audit, and sprint breakdown.

## Dataset Explorer

Local Streamlit app to browse and understand the NASA benchmark datasets before training. See [PLAN_DATASET_EXPLORER.md](PLAN_DATASET_EXPLORER.md) — tracked in [#8](https://github.com/tyc-aidev/microscopy-analysis/issues/8).

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