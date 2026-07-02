"""Sprint 3 low-data ablation sweep: Super datasets x {1,2,4,8,all} train images.

Reproduces the paper's headline low-data finding (large relative IoU-error
reduction from MicroNet pretraining when very few training images are available)
by sweeping the training-set size for each Super benchmark and comparing MicroNet
vs ImageNet pretraining. Like :mod:`microscopy_analysis.orchestration.matrix`,
this module only generates the job manifest + per-job training configs; dispatch
lives in ``scripts/run_low_data.py`` and analysis in :mod:`low_data_analysis`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .matrix import DATASETS, DatasetSpec

# Only the Super (multiclass) benchmarks carry a meaningful low-data curve; Super3
# is the paper's extreme case with a single training image.
SUPER_DATASETS: tuple[DatasetSpec, ...] = tuple(d for d in DATASETS if d.family == "super")

# The headline comparison is MicroNet vs ImageNet pretraining. Pass all four
# regimes explicitly to widen the sweep (random / image-micronet baselines).
LOW_DATA_PRETRAININGS: tuple[str, ...] = ("imagenet", "micronet")

# ``None`` marks the full training split ("all"); numeric caps are clamped to the
# available image count per dataset at generation time when counts are known.
DEFAULT_SIZES: tuple[int | None, ...] = (1, 2, 4, 8, None)

ENCODER = "resnet50"  # paper's low-data curves use resnet50
ARCHITECTURE = "UnetPlusPlus"


@dataclass(frozen=True)
class LowDataJob:
    run_name: str
    dataset_name: str
    dataset_family: str
    num_classes: int
    architecture: str
    encoder_name: str
    pretraining: str
    train_subsample: int | None  # None = full split ("all")
    seed: int


def _size_label(n: int | None) -> str:
    return "all" if n is None else str(n)


def resolve_sizes(sizes: tuple[int | None, ...], train_count: int | None) -> list[int | None]:
    """Reduce a requested size sweep to the distinct sizes runnable for a dataset.

    With a known ``train_count``, ``None`` and any cap ``>= train_count`` collapse
    to the concrete full count (so the "all" point is a real integer and dedupes
    against an equal numeric cap). Without a count, numeric caps pass through and
    ``None`` stays as the sentinel full-split marker.
    """
    if train_count is not None:
        effective = {min(s, train_count) if s is not None else train_count for s in sizes}
        return sorted(effective)
    numeric = sorted({s for s in sizes if s is not None})
    return [*numeric, None] if None in sizes else numeric


def _run_name(dataset: str, encoder: str, pretraining: str, n: int | None, seed: int) -> str:
    return f"{dataset.lower()}_{ARCHITECTURE.lower()}_{encoder}_{pretraining}_n{_size_label(n)}_seed{seed}"


def generate_low_data_jobs(
    *,
    datasets: tuple[DatasetSpec, ...] = SUPER_DATASETS,
    sizes: tuple[int | None, ...] = DEFAULT_SIZES,
    pretrainings: tuple[str, ...] = LOW_DATA_PRETRAININGS,
    encoder: str = ENCODER,
    seed: int = 42,
    train_counts: dict[str, int] | None = None,
) -> list[LowDataJob]:
    """Cross product of datasets x resolved sizes x pretrainings (one encoder)."""
    train_counts = train_counts or {}
    jobs: list[LowDataJob] = []
    for ds in datasets:
        for n in resolve_sizes(sizes, train_counts.get(ds.name)):
            for pre in pretrainings:
                jobs.append(
                    LowDataJob(
                        run_name=_run_name(ds.name, encoder, pre, n, seed),
                        dataset_name=ds.name,
                        dataset_family=ds.family,
                        num_classes=ds.num_classes,
                        architecture=ARCHITECTURE,
                        encoder_name=encoder,
                        pretraining=pre,
                        train_subsample=n,
                        seed=seed,
                    )
                )
    return jobs


def render_config(
    job: LowDataJob, *, data_root: str = "data/benchmark_segmentation_data", output_dir: str = "results"
) -> dict:
    """Render a low-data job into a ``load_train_config``-compatible dict."""
    trainer: dict = {"patience": 30, "max_epochs_phase1": 120, "max_epochs_phase2": 60}
    if job.train_subsample is not None:
        trainer["train_subsample"] = job.train_subsample
    return {
        "run_name": job.run_name,
        "data_root": data_root,
        "output_dir": output_dir,
        "seed": job.seed,
        "dataset": {"name": job.dataset_name, "family": job.dataset_family, "split": "train"},
        "model": {
            "architecture": job.architecture,
            "encoder_name": job.encoder_name,
            "pretraining": job.pretraining,
            "num_classes": job.num_classes,
        },
        "optimizer": {"lr_phase1": 2e-4, "lr_phase2": 1e-5},
        "trainer": trainer,
        "logging": {"backend": "none"},
    }


def write_manifest(jobs: list[LowDataJob], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(j) for j in jobs], indent=2))
    return path


def write_configs(jobs: list[LowDataJob], configs_dir: Path, **render_kwargs) -> list[Path]:
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for job in jobs:
        out = configs_dir / f"{job.run_name}.yaml"
        out.write_text(yaml.safe_dump(render_config(job, **render_kwargs), sort_keys=False))
        paths.append(out)
    return paths
