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


def build_comparison(scores: list[RunScore]) -> list[ComparisonRow]:
    """Pivot scores to per-(dataset, encoder) ImageNet-vs-MicroNet comparison rows."""
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for s in scores:
        by_key.setdefault((s.dataset_name, s.encoder_name), {})[s.pretraining] = s.score

    rows: list[ComparisonRow] = []
    for (dataset, encoder), regimes in sorted(by_key.items()):
        imagenet = regimes.get("imagenet")
        micronet = regimes.get("micronet")
        delta = round(micronet - imagenet, 6) if imagenet is not None and micronet is not None else None
        rows.append(
            ComparisonRow(
                dataset_name=dataset,
                encoder_name=encoder,
                imagenet=imagenet,
                micronet=micronet,
                delta=delta,
                micronet_ge_imagenet=(delta >= 0) if delta is not None else None,
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
        writer.writerow(["dataset", "encoder", "imagenet_iou", "micronet_iou", "delta", "micronet_ge_imagenet"])
        for r in rows:
            writer.writerow(
                [r.dataset_name, r.encoder_name, r.imagenet, r.micronet, r.delta, r.micronet_ge_imagenet]
            )
    return path


def render_markdown(rows: list[ComparisonRow], summary: dict) -> str:
    lines = [
        "| Dataset | Encoder | ImageNet IoU | MicroNet IoU | Δ | MicroNet ≥ ImageNet |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
        flag = "—" if r.micronet_ge_imagenet is None else ("✅" if r.micronet_ge_imagenet else "❌")
        lines.append(
            f"| {r.dataset_name} | {r.encoder_name} | {fmt(r.imagenet)} | {fmt(r.micronet)} | {fmt(r.delta)} | {flag} |"
        )
    lines.append("")
    lines.append(
        f"MicroNet ≥ ImageNet on {summary['micronet_ge_imagenet']}/{summary['compared_pairs']} pairs "
        f"— majority criterion {'MET' if summary['majority_met'] else 'NOT met'}."
    )
    return "\n".join(lines)
