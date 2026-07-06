"""Analyse the Sprint 3 low-data sweep: IoU-vs-#images curves + error reduction.

Reads the ``eval_<split>.json`` sidecars produced for low-data runs (each tagged
with ``train_subsample`` = the number of training images the checkpoint saw) and
builds the paper's headline artefacts:

- **Learning curves**: test IoU as a function of training-set size, per
  (dataset, encoder, pretraining regime).
- **Relative IoU-error reduction**: how much of ImageNet's residual error MicroNet
  eliminates at each training-set size — the paper's headline metric (~72% at a
  single training image).

Torch-free (stdlib only) for the tables; plotting pulls in matplotlib lazily.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPoint:
    dataset_name: str
    pretraining: str
    n_train: int
    encoder_name: str
    score: float


@dataclass(frozen=True)
class LowDataRow:
    dataset_name: str
    encoder_name: str
    n_train: int
    imagenet: float | None
    micronet: float | None
    delta: float | None  # micronet - imagenet
    rel_error_reduction: float | None  # fraction of ImageNet's IoU-error removed


def relative_error_reduction(imagenet_iou: float, micronet_iou: float) -> float | None:
    """Fraction of ImageNet's residual IoU error that MicroNet removes.

    ``(err_imagenet - err_micronet) / err_imagenet`` with ``err = 1 - IoU``,
    i.e. ``(IoU_micronet - IoU_imagenet) / (1 - IoU_imagenet)``. Positive when
    MicroNet beats ImageNet. ``None`` when ImageNet is already perfect (no error
    left to reduce).
    """
    residual = 1.0 - imagenet_iou
    if residual <= 0:
        return None
    return round((micronet_iou - imagenet_iou) / residual, 6)


def load_low_data_scores(results_dir: Path, *, split: str = "test") -> list[RunPoint]:
    """Load low-data ``eval_<split>.json`` sidecars beneath ``results_dir``.

    A run's training-set size comes from ``train_subsample``; when that is null
    (a full-split "all" run) it falls back to the sibling ``run_summary.json``'s
    ``num_samples``. Sidecars with neither are skipped (not low-data runs).
    """
    results_dir = Path(results_dir)
    points: list[RunPoint] = []
    for path in sorted(results_dir.rglob(f"eval_{split}.json")):
        data = json.loads(path.read_text())
        n_train = data.get("train_subsample")
        if n_train is None:
            summary = path.with_name("run_summary.json")
            if summary.exists():
                n_train = json.loads(summary.read_text()).get("num_samples")
        if n_train is None:
            continue
        points.append(
            RunPoint(
                dataset_name=data["dataset_name"],
                pretraining=data["pretraining"],
                n_train=int(n_train),
                encoder_name=data.get("encoder_name", ""),
                score=float(data["score"]),
            )
        )
    return points


def build_curves(points: list[RunPoint]) -> dict[tuple[str, str, str], list[tuple[int, float]]]:
    """Group points into ``(dataset, encoder, pretraining) -> [(n_train, iou), ...]``.

    Keyed by encoder so each encoder gets its own curve. Repeated points at the
    same size (e.g. multiple seeds) are averaged into a single curve point.
    """
    grouped: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in points:
        grouped[(p.dataset_name, p.encoder_name, p.pretraining)][p.n_train].append(p.score)
    curves: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    for key, by_n in grouped.items():
        curves[key] = [(n, round(sum(v) / len(v), 6)) for n, v in sorted(by_n.items())]
    return curves


def build_low_data_rows(points: list[RunPoint]) -> list[LowDataRow]:
    """Pivot to per-(dataset, encoder, n_train) ImageNet-vs-MicroNet error-reduction rows."""
    by_key: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in points:
        by_key[(p.dataset_name, p.encoder_name, p.n_train)][p.pretraining].append(p.score)

    rows: list[LowDataRow] = []
    for (dataset, encoder, n_train), regimes in sorted(by_key.items()):
        mean = {r: sum(v) / len(v) for r, v in regimes.items()}
        imagenet = mean.get("imagenet")
        micronet = mean.get("micronet")
        delta = round(micronet - imagenet, 6) if imagenet is not None and micronet is not None else None
        rel = (
            relative_error_reduction(imagenet, micronet)
            if imagenet is not None and micronet is not None
            else None
        )
        rows.append(
            LowDataRow(
                dataset_name=dataset,
                encoder_name=encoder,
                n_train=n_train,
                imagenet=round(imagenet, 6) if imagenet is not None else None,
                micronet=round(micronet, 6) if micronet is not None else None,
                delta=delta,
                rel_error_reduction=rel,
            )
        )
    return rows


def summarize(rows: list[LowDataRow], curves: dict[tuple[str, str, str], list[tuple[int, float]]]) -> dict:
    """Sprint 3 exit-criteria tally: min-n MicroNet advantage + curve monotonicity."""
    # Smallest training-set size per (dataset, encoder) — the low-data extreme.
    min_n: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r.dataset_name, r.encoder_name)
        if key not in min_n or r.n_train < min_n[key]:
            min_n[key] = r.n_train
    reduction_at_min_n = {
        f"{r.dataset_name}/{r.encoder_name}": r.rel_error_reduction
        for r in rows
        if r.n_train == min_n.get((r.dataset_name, r.encoder_name)) and r.rel_error_reduction is not None
    }
    reductions = list(reduction_at_min_n.values())

    monotonic = 0
    multi = 0
    for series in curves.values():
        if len(series) < 2:
            continue
        multi += 1
        ious = [iou for _, iou in series]
        if all(b >= a - 1e-9 for a, b in zip(ious, ious[1:], strict=False)):
            monotonic += 1

    return {
        "reduction_at_min_n": {k: round(v, 6) for k, v in reduction_at_min_n.items()},
        "max_reduction_at_min_n": round(max(reductions), 6) if reductions else None,
        "curves_with_multiple_points": multi,
        "monotonic_curves": monotonic,
        "monotonic_fraction": round(monotonic / multi, 6) if multi else None,
    }


def write_low_data_csv(rows: list[LowDataRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["dataset", "encoder", "n_train", "imagenet_iou", "micronet_iou", "delta", "rel_error_reduction"]
        )
        for r in rows:
            writer.writerow(
                [r.dataset_name, r.encoder_name, r.n_train, r.imagenet, r.micronet, r.delta,
                 r.rel_error_reduction]
            )
    return path


def render_markdown(rows: list[LowDataRow], summary: dict) -> str:
    header = ["Dataset", "Encoder", "# Train", "ImageNet IoU", "MicroNet IoU", "Δ", "Rel. err. reduction"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    pct = lambda v: "—" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [r.dataset_name, r.encoder_name, str(r.n_train), fmt(r.imagenet), fmt(r.micronet),
                 fmt(r.delta), pct(r.rel_error_reduction)]
            )
            + " |"
        )
    lines.append("")
    peak = summary.get("max_reduction_at_min_n")
    lines.append(
        "Max relative IoU-error reduction at the smallest training set: "
        + (f"{peak * 100:.1f}%" if peak is not None else "—")
        + "."
    )
    frac = summary.get("monotonic_fraction")
    if frac is not None:
        lines.append(
            f"Monotonic (IoU non-decreasing with more data) on "
            f"{summary['monotonic_curves']}/{summary['curves_with_multiple_points']} curves."
        )
    return "\n".join(lines)


def plot_curves(
    curves: dict[tuple[str, str, str], list[tuple[int, float]]],
    path: Path,
    *,
    title: str = "Low-data IoU curves",
) -> Path:
    """Render test IoU vs #training-images (one line per dataset/encoder/regime)."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dataset, encoder, pretraining), series in sorted(curves.items()):
        xs = [n for n, _ in series]
        ys = [iou for _, iou in series]
        style = "-o" if pretraining == "micronet" else "--s"
        ax.plot(xs, ys, style, label=f"{dataset} · {encoder} · {pretraining}")
    ax.set_xlabel("# training images")
    ax.set_ylabel("Test IoU")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
