#!/usr/bin/env python3
"""Generate and (optionally) run the Sprint 2 benchmark matrix.

Default: emit a job manifest + one training config per job under
``configs/experiments/matrix/`` without running anything. Pass ``--dispatch
local`` to train each job sequentially on the current machine (slow — 56 real
runs are meant for a GPU/cloud fleet; cloud fan-out is a documented follow-up).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from microscopy_analysis.orchestration import generate_jobs, write_configs, write_manifest
from microscopy_analysis.orchestration.matrix import render_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/dispatch the Sprint 2 benchmark matrix")
    parser.add_argument("--manifest", type=Path, default=Path("results/matrix/manifest.json"))
    parser.add_argument("--configs-dir", type=Path, default=Path("configs/experiments/matrix"))
    parser.add_argument("--data-root", default="data/benchmark_segmentation_data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dispatch", choices=("none", "local"), default="none")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--max-jobs", type=int, default=None, help="Cap dispatched jobs (smoke run)")
    parser.add_argument("--eval-split", default="test", help="Split to evaluate after each job (default: test)")
    parser.add_argument("--no-eval", action="store_true", help="Train only; skip post-training evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = generate_jobs(seed=args.seed)

    manifest_path = write_manifest(jobs, args.manifest)
    config_paths = write_configs(
        jobs, args.configs_dir, data_root=args.data_root, output_dir=args.output_dir
    )
    print(f"Generated {len(jobs)} jobs -> manifest {manifest_path}, {len(config_paths)} configs in {args.configs_dir}")

    if args.dispatch == "none":
        return 0

    # Local sequential dispatch: build a TrainConfig per job from the rendered
    # dict, train it, then evaluate on the held-out split so aggregate_results.py
    # has an eval_<split>.json to read. Heavy — cap with --max-jobs for a smoke run.
    from microscopy_analysis.eval.evaluate import evaluate_run  # noqa: PLC0415
    from microscopy_analysis.train.config import TrainConfig  # noqa: PLC0415
    from microscopy_analysis.train.trainer import run_training  # noqa: PLC0415

    limit = len(jobs) if args.max_jobs is None else min(args.max_jobs, len(jobs))
    base = Path.cwd()
    for job in jobs[:limit]:
        rendered = render_config(job, data_root=args.data_root, output_dir=args.output_dir)
        data_root = base / rendered["data_root"]
        output_root = base / rendered["output_dir"]
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
        )
        print(f"[dispatch] training {job.run_name}")
        result = run_training(cfg, device_preference=args.device)
        print(json.dumps(dataclasses.asdict(result), indent=2))
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
