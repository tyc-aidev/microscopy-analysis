#!/usr/bin/env python3
"""Generate and (optionally) run the Sprint 3 low-data ablation sweep.

Default: emit a job manifest + one training config per job under
``configs/experiments/low_data/`` without training. When the benchmark data is on
disk, per-dataset training-image counts are used to clamp the ``{1,2,4,8,all}``
size sweep to what each dataset actually has (Super3 has a single image). Pass
``--dispatch local`` to train + evaluate each job sequentially (slow — meant for a
GPU/cloud fleet).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from microscopy_analysis.data.dataset_adapter import list_sample_pairs
from microscopy_analysis.orchestration.low_data import (
    DEFAULT_SIZES,
    ENCODERS,
    LOW_DATA_PRETRAININGS,
    SUPER_DATASETS,
    generate_low_data_jobs,
    render_config,
    write_configs,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/dispatch the Sprint 3 low-data sweep")
    parser.add_argument("--manifest", type=Path, default=Path("results/low_data/manifest.json"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs/experiments/low_data"))
    parser.add_argument("--data-root", default="data/benchmark_segmentation_data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated training sizes; use 'all' for the full split (default: 1,2,4,8,all)",
    )
    parser.add_argument(
        "--pretrainings",
        nargs="+",
        default=list(LOW_DATA_PRETRAININGS),
        help="Pretraining regimes to compare (default: imagenet micronet)",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=list(ENCODERS),
        help="Encoders to sweep, each gets its own curve (default: senet154 se_resnext50_32x4d)",
    )
    parser.add_argument("--dispatch", choices=("none", "local"), default="none")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--max-jobs", type=int, default=None, help="Cap dispatched jobs (smoke run)")
    parser.add_argument("--eval-split", default="test", help="Split to evaluate after each job")
    parser.add_argument("--no-eval", action="store_true", help="Train only; skip post-training evaluation")
    return parser.parse_args()


def _parse_sizes(raw: str | None) -> tuple[int | None, ...]:
    if raw is None:
        return DEFAULT_SIZES
    out: list[int | None] = []
    for token in raw.split(","):
        token = token.strip().lower()
        out.append(None if token in ("all", "max", "") else int(token))
    return tuple(out)


def _count_train_images(data_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ds in SUPER_DATASETS:
        pairs = list_sample_pairs(data_root / ds.name, split="train", dataset_family=ds.family)
        if pairs:
            counts[ds.name] = len(pairs)
    return counts


def main() -> int:
    args = parse_args()
    base = Path.cwd()
    data_root = (base / args.data_root) if not Path(args.data_root).is_absolute() else Path(args.data_root)
    train_counts = _count_train_images(data_root) or None

    jobs = generate_low_data_jobs(
        sizes=_parse_sizes(args.sizes),
        pretrainings=tuple(args.pretrainings),
        encoders=tuple(args.encoders),
        seed=args.seed,
        train_counts=train_counts,
    )
    manifest_path = write_manifest(jobs, args.manifest)
    config_paths = write_configs(
        jobs, args.configs_dir, data_root=args.data_root, output_dir=args.output_dir
    )
    counts_note = f" (clamped to {train_counts})" if train_counts else " (data not found; sizes unclamped)"
    print(f"Generated {len(jobs)} low-data jobs -> {manifest_path}, {len(config_paths)} configs{counts_note}")

    if args.dispatch == "none":
        return 0

    from microscopy_analysis.eval.evaluate import evaluate_run  # noqa: PLC0415
    from microscopy_analysis.train.config import TrainConfig  # noqa: PLC0415
    from microscopy_analysis.train.trainer import run_training  # noqa: PLC0415

    limit = len(jobs) if args.max_jobs is None else min(args.max_jobs, len(jobs))
    output_root = (base / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    for job in jobs[:limit]:
        cfg = TrainConfig(
            run_name=job.run_name,
            data_root=data_root,
            dataset_name=job.dataset_name,
            dataset_family=job.dataset_family,
            split="train",
            architecture=job.architecture,
            encoder_name=job.encoder_name,
            pretraining=job.pretraining,
            num_classes=job.num_classes,
            output_dir=output_root / job.run_name,
            seed=job.seed,
            train_subsample=job.train_subsample,
        )
        print(f"[dispatch] training {job.run_name} (n_train={job.train_subsample})")
        result = run_training(cfg, device_preference=args.device)
        print(f"[dispatch] trained {job.run_name} on {result.num_samples} images, best IoU={result.best_score:.4f}")
        if args.no_eval:
            continue
        try:
            eval_result = evaluate_run(cfg, split=args.eval_split, device_preference=args.device)
            print(f"[dispatch] eval {job.run_name} {args.eval_split} score={eval_result.score:.4f}")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[dispatch] eval skipped for {job.run_name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
