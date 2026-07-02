"""Aggregate per-run ``eval_<split>.json`` files into the Sprint 2 results table.

Collects the held-out scores produced by :mod:`microscopy_analysis.eval.evaluate`
across a ``results/`` tree and pivots them into the paper's central comparison:
per (dataset, encoder), MicroNet vs ImageNet test IoU and their delta. The
``micronet >= imagenet on the majority of datasets`` exit criterion is computed
directly from these rows. Torch-free (stdlib only) so it runs anywhere.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunScore:
    dataset_name: str
    encoder_name: str
    pretraining: str
    score: float
    mean_iou: float
    split: str


@dataclass(frozen=True)
class ComparisonRow:
    dataset_name: str
    encoder_name: str
    imagenet: float | None
    micronet: float | None
    delta: float | None  # micronet - imagenet
    micronet_ge_imagenet: bool | None
    paper_micronet: float | None = None  # transcribed target (paper/target_metrics.csv)
    micronet_vs_paper: float | None = None  # reproduced micronet - paper micronet


def load_paper_targets(csv_path: Path) -> dict[tuple[str, str], float]:
    """Map ``(dataset, pretraining) -> paper_test_iou`` from ``target_metrics.csv``.

    Rows with a blank ``paper_test_iou`` (not yet transcribed) are skipped.
    """
    targets: dict[tuple[str, str], float] = {}
    with Path(csv_path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            value = (row.get("paper_test_iou") or "").strip()
            if not value:
                continue
            targets[(row["dataset"], row["pretraining"])] = float(value)
    return targets


def load_eval_scores(results_dir: Path, *, split: str = "test") -> list[RunScore]:
    """Load all ``eval_<split>.json`` sidecars beneath ``results_dir``."""
    results_dir = Path(results_dir)
    scores: list[RunScore] = []
    for path in sorted(results_dir.rglob(f"eval_{split}.json")):
        data = json.loads(path.read_text())
        scores.append(
            RunScore(
                dataset_name=data["dataset_name"],
                encoder_name=data["encoder_name"],
                pretraining=data["pretraining"],
                score=float(data["score"]),
                mean_iou=float(data["mean_iou"]),
                split=data.get("split", split),
            )
        )
    return scores


def build_comparison(
    scores: list[RunScore], targets: dict[tuple[str, str], float] | None = None
) -> list[ComparisonRow]:
    """Pivot scores to per-(dataset, encoder) ImageNet-vs-MicroNet comparison rows.

    Optional ``targets`` (from :func:`load_paper_targets`) adds the paper MicroNet
    IoU and the reproduced-vs-paper delta for each dataset.
    """
    targets = targets or {}
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for s in scores:
        by_key.setdefault((s.dataset_name, s.encoder_name), {})[s.pretraining] = s.score

    rows: list[ComparisonRow] = []
    for (dataset, encoder), regimes in sorted(by_key.items()):
        imagenet = regimes.get("imagenet")
        micronet = regimes.get("micronet")
        delta = round(micronet - imagenet, 6) if imagenet is not None and micronet is not None else None
        paper_micronet = targets.get((dataset, "micronet"))
        vs_paper = (
            round(micronet - paper_micronet, 6)
            if micronet is not None and paper_micronet is not None
            else None
        )
        rows.append(
            ComparisonRow(
                dataset_name=dataset,
                encoder_name=encoder,
                imagenet=imagenet,
                micronet=micronet,
                delta=delta,
                micronet_ge_imagenet=(delta >= 0) if delta is not None else None,
                paper_micronet=paper_micronet,
                micronet_vs_paper=vs_paper,
            )
        )
    return rows


def majority_summary(rows: list[ComparisonRow]) -> dict:
    """Sprint 2 exit-criterion tally: MicroNet >= ImageNet on the majority of pairs."""
    compared = [r for r in rows if r.micronet_ge_imagenet is not None]
    wins = sum(1 for r in compared if r.micronet_ge_imagenet)
    total = len(compared)
    return {
        "compared_pairs": total,
        "micronet_ge_imagenet": wins,
        "majority_met": total > 0 and wins * 2 >= total,
    }


def write_comparison_csv(rows: list[ComparisonRow], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["dataset", "encoder", "imagenet_iou", "micronet_iou", "delta",
             "micronet_ge_imagenet", "paper_micronet_iou", "micronet_vs_paper"]
        )
        for r in rows:
            writer.writerow(
                [r.dataset_name, r.encoder_name, r.imagenet, r.micronet, r.delta,
                 r.micronet_ge_imagenet, r.paper_micronet, r.micronet_vs_paper]
            )
    return path


def render_markdown(rows: list[ComparisonRow], summary: dict) -> str:
    has_paper = any(r.paper_micronet is not None for r in rows)
    header = ["Dataset", "Encoder", "ImageNet IoU", "MicroNet IoU", "Δ", "MicroNet ≥ ImageNet"]
    if has_paper:
        header += ["Paper MicroNet", "Repro − Paper"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
        flag = "—" if r.micronet_ge_imagenet is None else ("✅" if r.micronet_ge_imagenet else "❌")
        cells = [r.dataset_name, r.encoder_name, fmt(r.imagenet), fmt(r.micronet), fmt(r.delta), flag]
        if has_paper:
            cells += [fmt(r.paper_micronet), fmt(r.micronet_vs_paper)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"MicroNet ≥ ImageNet on {summary['micronet_ge_imagenet']}/{summary['compared_pairs']} pairs "
        f"— majority criterion {'MET' if summary['majority_met'] else 'NOT met'}."
    )
    return "\n".join(lines)
