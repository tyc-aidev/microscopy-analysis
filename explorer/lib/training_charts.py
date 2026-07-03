"""Matplotlib training curves from metrics.json (torch-free)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

SUPER_CLASS_LABELS = ("matrix", "secondary γ′", "tertiary γ′")
EBC_CLASS_LABELS = ("background", "oxide")


def _class_labels(metrics: list[dict[str, Any]], dataset_family: str) -> tuple[str, ...]:
    if not metrics:
        return ()
    n = len(metrics[-1].get("val_iou_per_class", []))
    if dataset_family == "ebc" or n == 2:
        return EBC_CLASS_LABELS[:n]
    return SUPER_CLASS_LABELS[:n]


def plot_loss_curves(metrics: list[dict[str, Any]]) -> plt.Figure:
    epochs = [m["epoch"] for m in metrics]
    train = [m["train_loss"] for m in metrics]
    val = [m["val_loss"] for m in metrics]
    phases = [m.get("phase", 1) for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train, label="train", marker=".", markersize=4)
    ax.plot(epochs, val, label="val", marker=".", markersize=4)
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            ax.axvline(epochs[i], color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_iou_curves(metrics: list[dict[str, Any]], *, dataset_family: str = "super") -> plt.Figure:
    epochs = [m["epoch"] for m in metrics]
    mean_iou = [m["val_mean_iou"] for m in metrics]
    score = [m.get("val_score", m["val_mean_iou"]) for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, mean_iou, marker="o", markersize=4, color="C2")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Val mean IoU")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, score, marker="o", markersize=4, color="C3")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Val score (early-stop)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_per_class_iou(metrics: list[dict[str, Any]], *, dataset_family: str = "super") -> plt.Figure | None:
    if not metrics:
        return None
    per_class = metrics[0].get("val_iou_per_class", [])
    if not per_class:
        return None

    labels = _class_labels(metrics, dataset_family)
    epochs = [m["epoch"] for m in metrics]
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, label in enumerate(labels):
        series = [m["val_iou_per_class"][idx] for m in metrics if len(m.get("val_iou_per_class", [])) > idx]
        ax.plot(epochs[: len(series)], series, marker=".", markersize=4, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Val IoU per class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
