"""Config-driven two-phase segmentation trainer with checkpoint + JSON metrics.

Ports the NASA two-phase schedule (Adam @ ``lr_phase1`` with early stopping on
validation IoU, then resume @ ``lr_phase2``) to a reproducible CLI. Models carry
their own ``softmax2d`` / ``sigmoid`` activation, so the loss and metric operate
on probabilities. Runs on CUDA, Apple Silicon (MPS), or CPU via
:func:`resolve_device`.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from microscopy_analysis.data.dataset_adapter import list_sample_pairs
from microscopy_analysis.data.segmentation_dataset import (
    AugmentationConfig,
    SegmentationDataset,
    build_transforms,
)
from microscopy_analysis.device import enable_mps_fallback, resolve_device
from microscopy_analysis.eval import DiceBCELoss, IoU
from microscopy_analysis.models import create_segmentation_model
from microscopy_analysis.train.config import TrainConfig
from microscopy_analysis.train.logging import build_logger


@dataclass(frozen=True)
class TrainResult:
    run_name: str
    dataset_name: str
    dataset_family: str
    num_samples: int
    phase1_lr: float
    phase2_lr: float
    checkpoint_path: str
    metrics_path: str
    git_sha: str
    created_at_utc: str
    # Sprint 1 real-training outcome (defaults preserve the old constructor shape).
    num_val_samples: int = 0
    device: str = "cpu"
    epochs_trained: int = 0
    best_epoch: int = 0
    best_score: float = 0.0
    best_mean_iou: float = 0.0
    best_per_class_iou: tuple[float, ...] = field(default_factory=tuple)
    summary_path: str = ""
    best_checkpoint_path: str = ""
    resumed_from_epoch: int = 0


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
    return out if out else "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loader(config: TrainConfig, split: str, *, train: bool) -> DataLoader:
    aug = AugmentationConfig.from_dict(config.augmentation, crop_size=config.crop_size)
    transform = build_transforms(config.dataset_family, train=train, crop_size=config.crop_size, aug=aug)
    dataset = SegmentationDataset(
        config.data_root / config.dataset_name, split, config.dataset_family, transform
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.num_workers,
        drop_last=False,
    )


def _run_epoch(model, loader, loss_fn, device, optimizer=None, metric=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    if metric is not None:
        metric.reset()
    total_loss, seen = 0.0, 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.set_grad_enabled(is_train):
            probs = model(images)
            loss = loss_fn(probs, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * images.size(0)
        seen += images.size(0)
        if metric is not None:
            metric.update(probs.detach(), masks)
    return total_loss / max(seen, 1)


def run_training(config: TrainConfig, *, device_preference: str = "auto") -> TrainResult:
    _seed_everything(config.seed)
    enable_mps_fallback()
    device = resolve_device(device_preference)

    dataset_root = config.data_root / config.dataset_name
    train_pairs = list_sample_pairs(dataset_root, split=config.split, dataset_family=config.dataset_family)
    if not train_pairs:
        raise RuntimeError(
            f"No training pairs found in {dataset_root} split={config.split} family={config.dataset_family}"
        )
    val_pairs = list_sample_pairs(dataset_root, split=config.val_split, dataset_family=config.dataset_family)
    val_split = config.val_split if val_pairs else config.split  # fall back to train for tiny smoke sets

    train_loader = _make_loader(config, config.split, train=True)
    val_loader = _make_loader(config, val_split, train=False)

    model = create_segmentation_model(
        config.architecture, config.encoder_name, config.pretraining, config.num_classes
    ).to(device)
    loss_fn = DiceBCELoss(weight=config.loss_weight)
    metric = IoU(config.num_classes, threshold=config.metric_threshold)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "checkpoint.pth"  # latest, for resume
    best_checkpoint_path = config.output_dir / "model_best.pth"  # best val IoU, final model
    metrics_path = config.output_dir / "metrics.json"
    summary_path = config.output_dir / "run_summary.json"

    logger = build_logger(config.log_backend, config.run_name, config.log_project)
    # Stringify non-scalars (e.g. Path data_root/output_dir) so every backend accepts them.
    params = {k: (v if isinstance(v, (int, float, bool, str, type(None))) else str(v)) for k, v in asdict(config).items()}
    logger.log_params({**params, "git_sha": _git_sha(), "device": str(device)})

    epoch_records: list[dict] = []
    best = {"score": -1.0, "mean_iou": 0.0, "per_class": [], "epoch": 0}
    global_epoch = 0

    def save_latest(epoch: int, optimizer, phase: int, phase_epoch: int) -> None:
        """Latest-state checkpoint (every epoch) so a preempted run can resume."""
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "best_score": best["score"],
                "best_mean_iou": best["mean_iou"],
                "best_per_class": best["per_class"],
                "best_epoch": best["epoch"],
                "config": {"run_name": config.run_name, "num_classes": config.num_classes},
            },
            checkpoint_path,
        )

    def save_best() -> None:
        torch.save({"model_state": model.state_dict(), "score": best["score"]}, best_checkpoint_path)

    # Resume from a prior latest checkpoint (#11): restore weights, optimizer, and bests.
    resume_phase, resume_phase_epoch, resume_opt_state, resumed_from = 1, 0, None, 0
    if config.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        global_epoch = int(ckpt.get("epoch", 0))
        resume_phase = int(ckpt.get("phase", 1))
        resume_phase_epoch = int(ckpt.get("phase_epoch", 0))
        resume_opt_state = ckpt.get("optimizer_state")
        best.update(
            score=float(ckpt.get("best_score", -1.0)),
            mean_iou=float(ckpt.get("best_mean_iou", 0.0)),
            per_class=list(ckpt.get("best_per_class", [])),
            epoch=int(ckpt.get("best_epoch", 0)),
        )
        resumed_from = global_epoch
        if metrics_path.exists():
            epoch_records = json.loads(metrics_path.read_text())

    def run_phase(phase: int, max_epochs: int, lr: float, *, start_epoch: int = 0, opt_state=None) -> None:
        nonlocal global_epoch
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
        no_improve = 0  # patience counter restarts on resume (a preemption gets a fresh window)
        for phase_epoch in range(start_epoch, max_epochs):
            global_epoch += 1
            train_loss = _run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer)
            val_loss = _run_epoch(model, val_loader, loss_fn, device, metric=metric)
            per_class = [round(v, 6) for v in metric.per_class().tolist()]
            mean_iou, score = metric.mean(), metric.score()
            record = {
                "epoch": global_epoch,
                "phase": phase,
                "lr": lr,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_iou_per_class": per_class,
                "val_mean_iou": round(mean_iou, 6),
                "val_score": round(score, 6),
            }
            epoch_records.append(record)
            metrics_path.write_text(json.dumps(epoch_records, indent=2))
            logger.log_epoch(record)
            print(
                f"[phase {phase}] epoch {global_epoch} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mean_iou={mean_iou:.4f} val_score={score:.4f}"
            )
            if score > best["score"] + 1e-6:
                best.update(score=score, mean_iou=mean_iou, per_class=per_class, epoch=global_epoch)
                save_best()
                no_improve = 0
            else:
                no_improve += 1
            save_latest(global_epoch, optimizer, phase, phase_epoch + 1)
            if no_improve >= config.patience:
                break

    if resume_phase <= 1:
        run_phase(1, config.max_epochs_phase1, config.lr_phase1,
                  start_epoch=resume_phase_epoch if resume_phase == 1 else 0, opt_state=resume_opt_state)
        # Normal phase-1 -> phase-2 transition: fine-tune from the best phase-1 weights.
        # Skipped when resuming directly into phase 2, where the latest checkpoint
        # already restored the in-progress phase-2 weights (don't clobber them).
        if best_checkpoint_path.exists():
            model.load_state_dict(torch.load(best_checkpoint_path, map_location=device)["model_state"])
    run_phase(2, config.max_epochs_phase2, config.lr_phase2,
              start_epoch=resume_phase_epoch if resume_phase == 2 else 0,
              opt_state=resume_opt_state if resume_phase == 2 else None)

    if not checkpoint_path.exists():  # degenerate (e.g. zero epochs): still emit a real checkpoint
        save_latest(global_epoch, torch.optim.Adam(model.parameters(), lr=config.lr_phase2), 2, 0)
    if not best_checkpoint_path.exists():
        save_best()

    result = TrainResult(
        run_name=config.run_name,
        dataset_name=config.dataset_name,
        dataset_family=config.dataset_family,
        num_samples=len(train_pairs),
        phase1_lr=config.lr_phase1,
        phase2_lr=config.lr_phase2,
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        git_sha=_git_sha(),
        created_at_utc=datetime.now(UTC).isoformat(),
        num_val_samples=len(val_pairs),
        device=str(device),
        epochs_trained=global_epoch,
        best_epoch=int(best["epoch"]),
        best_score=round(float(best["score"]), 6),
        best_mean_iou=round(float(best["mean_iou"]), 6),
        best_per_class_iou=tuple(best["per_class"]),
        summary_path=str(summary_path),
        best_checkpoint_path=str(best_checkpoint_path),
        resumed_from_epoch=resumed_from,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2))
    logger.finish(asdict(result))
    return result
