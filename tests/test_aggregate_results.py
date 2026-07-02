"""Tests for Sprint 2 results aggregation (MicroNet vs ImageNet table)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from microscopy_analysis.orchestration.aggregate import (
    build_comparison,
    load_eval_scores,
    majority_summary,
    render_markdown,
    write_comparison_csv,
)


def _write_eval(root: Path, dataset: str, encoder: str, pretraining: str, score: float) -> None:
    run_dir = root / f"{dataset}_{encoder}_{pretraining}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_test.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset,
                "encoder_name": encoder,
                "pretraining": pretraining,
                "score": score,
                "mean_iou": score,
                "split": "test",
            }
        )
    )


def test_load_and_build_comparison_computes_delta(tmp_path: Path) -> None:
    _write_eval(tmp_path, "Super1", "resnet50", "imagenet", 0.80)
    _write_eval(tmp_path, "Super1", "resnet50", "micronet", 0.85)
    _write_eval(tmp_path, "EBC1", "resnet50", "imagenet", 0.70)
    _write_eval(tmp_path, "EBC1", "resnet50", "micronet", 0.65)

    scores = load_eval_scores(tmp_path)
    assert len(scores) == 4

    rows = build_comparison(scores)
    by_ds = {r.dataset_name: r for r in rows}
    assert by_ds["Super1"].delta == 0.05
    assert by_ds["Super1"].micronet_ge_imagenet is True
    assert by_ds["EBC1"].delta == -0.05
    assert by_ds["EBC1"].micronet_ge_imagenet is False


def test_majority_summary_and_markdown(tmp_path: Path) -> None:
    _write_eval(tmp_path, "Super1", "resnet50", "imagenet", 0.80)
    _write_eval(tmp_path, "Super1", "resnet50", "micronet", 0.85)
    _write_eval(tmp_path, "Super2", "resnet50", "imagenet", 0.60)
    _write_eval(tmp_path, "Super2", "resnet50", "micronet", 0.62)
    _write_eval(tmp_path, "EBC1", "resnet50", "imagenet", 0.70)
    _write_eval(tmp_path, "EBC1", "resnet50", "micronet", 0.65)

    rows = build_comparison(load_eval_scores(tmp_path))
    summary = majority_summary(rows)
    assert summary["compared_pairs"] == 3
    assert summary["micronet_ge_imagenet"] == 2
    assert summary["majority_met"] is True

    md = render_markdown(rows, summary)
    assert "MicroNet" in md and "2/3" in md


def test_missing_regime_yields_none_delta(tmp_path: Path) -> None:
    _write_eval(tmp_path, "Super3", "resnet50", "micronet", 0.90)  # no imagenet counterpart
    rows = build_comparison(load_eval_scores(tmp_path))
    assert rows[0].imagenet is None
    assert rows[0].delta is None
    assert rows[0].micronet_ge_imagenet is None
    assert majority_summary(rows)["compared_pairs"] == 0


def test_write_comparison_csv(tmp_path: Path) -> None:
    _write_eval(tmp_path, "Super1", "resnet50", "imagenet", 0.80)
    _write_eval(tmp_path, "Super1", "resnet50", "micronet", 0.85)
    rows = build_comparison(load_eval_scores(tmp_path))
    out = write_comparison_csv(rows, tmp_path / "matrix.csv")

    with out.open() as fh:
        reader = list(csv.DictReader(fh))
    assert reader[0]["dataset"] == "Super1"
    assert float(reader[0]["delta"]) == 0.05
