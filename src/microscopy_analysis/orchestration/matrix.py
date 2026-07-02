"""Sprint 2 core benchmark matrix: 7 datasets x 4 regimes x UNet++ x 2 encoders.

Generates the 56-job manifest for the paper's central comparison (MicroNet vs
ImageNet pretraining across all 7 semantic benchmarks) and renders a ready-to-run
training config per job. Dispatch is intentionally decoupled: this module only
produces the manifest + configs; ``scripts/run_matrix.py`` runs them locally
(sequentially) and cloud fan-out is a documented follow-up.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str  # "super" | "ebc"
    num_classes: int  # foreground classes (background implicit); EBC binary = 1


# The 7 semantic benchmarks from Stuckner et al. 2022 (Super = 3-class RGB masks,
# EBC = binary oxide task, classes=1 per the NASA convention).
DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("Super1", "super", 3),
    DatasetSpec("Super2", "super", 3),
    DatasetSpec("Super3", "super", 3),
    DatasetSpec("Super4", "super", 3),
    DatasetSpec("EBC1", "ebc", 1),
    DatasetSpec("EBC2", "ebc", 1),
    DatasetSpec("EBC3", "ebc", 1),
)

# All four regimes so MicroNet can be compared against every baseline in one sweep.
PRETRAININGS: tuple[str, ...] = ("random", "imagenet", "micronet", "image-micronet")

# Two v1.0 top-performing encoders from the NASA README leaderboard.
ENCODERS: tuple[str, ...] = ("resnet50", "se_resnext50_32x4d")

ARCHITECTURE = "UnetPlusPlus"


@dataclass(frozen=True)
class Job:
    run_name: str
    dataset_name: str
    dataset_family: str
    num_classes: int
    architecture: str
    encoder_name: str
    pretraining: str
    seed: int


def _run_name(dataset: str, encoder: str, pretraining: str, seed: int) -> str:
    return f"{dataset.lower()}_{ARCHITECTURE.lower()}_{encoder}_{pretraining}_seed{seed}"


def generate_jobs(
    *,
    datasets: tuple[DatasetSpec, ...] = DATASETS,
    encoders: tuple[str, ...] = ENCODERS,
    pretrainings: tuple[str, ...] = PRETRAININGS,
    seed: int = 42,
) -> list[Job]:
    """Cartesian product of datasets x pretrainings x encoders (56 jobs by default)."""
    jobs: list[Job] = []
    for ds in datasets:
        for enc in encoders:
            for pre in pretrainings:
                jobs.append(
                    Job(
                        run_name=_run_name(ds.name, enc, pre, seed),
                        dataset_name=ds.name,
                        dataset_family=ds.family,
                        num_classes=ds.num_classes,
                        architecture=ARCHITECTURE,
                        encoder_name=enc,
                        pretraining=pre,
                        seed=seed,
                    )
                )
    return jobs


def render_config(job: Job, *, data_root: str = "data/benchmark_segmentation_data", output_dir: str = "results") -> dict:
    """Render a job into a training-config dict consumable by ``load_train_config``."""
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
        "trainer": {"patience": 30, "max_epochs_phase1": 120, "max_epochs_phase2": 60},
        "logging": {"backend": "none"},
    }


def write_manifest(jobs: list[Job], path: Path) -> Path:
    """Write the job list as a JSON manifest (one object per job)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(j) for j in jobs], indent=2))
    return path


def write_configs(jobs: list[Job], configs_dir: Path, **render_kwargs) -> list[Path]:
    """Write one ``<run_name>.yaml`` training config per job; return the paths."""
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for job in jobs:
        out = configs_dir / f"{job.run_name}.yaml"
        out.write_text(yaml.safe_dump(render_config(job, **render_kwargs), sort_keys=False))
        paths.append(out)
    return paths
