"""Tests for the Sprint 3 low-data analysis (curves + relative error reduction)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from microscopy_analysis.orchestration.low_data_analysis import (
    RunPoint,
    build_curves,
    build_low_data_rows,
    load_low_data_scores,
    plot_curves,
    relative_error_reduction,
    render_markdown,
    summarize,
    write_low_data_csv,
)


def _write_eval(
    root: Path, dataset: str, pretraining: str, n_train, score: float, encoder: str = "senet154"
) -> None:
    run_dir = root / f"{dataset}_{encoder}_{pretraining}_n{n_train}"
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
                "train_subsample": n_train,
            }
        )
    )


def test_relative_error_reduction_formula() -> None:
    # ImageNet leaves 0.40 error; MicroNet halves it -> 0.5 reduction.
    assert relative_error_reduction(0.60, 0.80) == 0.5
    # No advantage -> 0; worse -> negative.
    assert relative_error_reduction(0.60, 0.60) == 0.0
    assert relative_error_reduction(0.80, 0.60) == -1.0
    # ImageNet already perfect -> undefined.
    assert relative_error_reduction(1.0, 1.0) is None


def test_load_scores_uses_train_subsample(tmp_path: Path) -> None:
    _write_eval(tmp_path, "Super3", "imagenet", 1, 0.30)
    _write_eval(tmp_path, "Super3", "micronet", 1, 0.75)
    points = load_low_data_scores(tmp_path)
    assert len(points) == 2
    assert {p.n_train for p in points} == {1}


def test_load_scores_falls_back_to_run_summary_for_all_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "super1_all"
    run_dir.mkdir()
    (run_dir / "eval_test.json").write_text(
        json.dumps(
            {
                "dataset_name": "Super1",
                "encoder_name": "resnet50",
                "pretraining": "micronet",
                "score": 0.9,
                "mean_iou": 0.9,
                "split": "test",
                "train_subsample": None,  # full split
            }
        )
    )
    (run_dir / "run_summary.json").write_text(json.dumps({"num_samples": 10}))
    points = load_low_data_scores(tmp_path)
    assert len(points) == 1 and points[0].n_train == 10


def test_load_scores_skips_untagged_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "plain_run"
    run_dir.mkdir()
    (run_dir / "eval_test.json").write_text(
        json.dumps(
            {"dataset_name": "Super1", "encoder_name": "resnet50", "pretraining": "micronet",
             "score": 0.9, "mean_iou": 0.9, "split": "test"}
        )
    )
    assert load_low_data_scores(tmp_path) == []


def test_build_curves_sorts_and_averages_seeds() -> None:
    points = [
        RunPoint("Super1", "micronet", 4, "senet154", 0.8),
        RunPoint("Super1", "micronet", 4, "senet154", 0.6),  # second seed -> averaged
        RunPoint("Super1", "micronet", 1, "senet154", 0.5),
    ]
    curves = build_curves(points)
    assert curves[("Super1", "senet154", "micronet")] == [(1, 0.5), (4, 0.7)]


def test_build_curves_separates_encoders() -> None:
    points = [
        RunPoint("Super1", "micronet", 1, "senet154", 0.7),
        RunPoint("Super1", "micronet", 1, "se_resnext50_32x4d", 0.6),
    ]
    curves = build_curves(points)
    assert curves[("Super1", "senet154", "micronet")] == [(1, 0.7)]
    assert curves[("Super1", "se_resnext50_32x4d", "micronet")] == [(1, 0.6)]


def test_build_rows_and_summary(tmp_path: Path) -> None:
    _write_eval(tmp_path, "Super1", "imagenet", 1, 0.30)
    _write_eval(tmp_path, "Super1", "micronet", 1, 0.75)
    _write_eval(tmp_path, "Super1", "imagenet", 4, 0.70)
    _write_eval(tmp_path, "Super1", "micronet", 4, 0.80)

    points = load_low_data_scores(tmp_path)
    rows = build_low_data_rows(points)
    by_n = {r.n_train: r for r in rows}
    assert by_n[1].delta == 0.45
    # (0.75 - 0.30) / (1 - 0.30) = 0.642857
    assert by_n[1].rel_error_reduction == round(0.45 / 0.70, 6)

    curves = build_curves(points)
    summary = summarize(rows, curves)
    assert summary["reduction_at_min_n"]["Super1/senet154"] == round(0.45 / 0.70, 6)
    assert summary["max_reduction_at_min_n"] == round(0.45 / 0.70, 6)
    # Both regimes' IoU rises with more data -> fully monotonic.
    assert summary["monotonic_fraction"] == 1.0
    assert summary["monotonic_curves"] == 2


def test_write_csv_and_markdown(tmp_path: Path) -> None:
    _write_eval(tmp_path, "Super1", "imagenet", 1, 0.30)
    _write_eval(tmp_path, "Super1", "micronet", 1, 0.75)
    points = load_low_data_scores(tmp_path)
    rows = build_low_data_rows(points)
    out = write_low_data_csv(rows, tmp_path / "curves.csv")
    with out.open() as fh:
        reader = list(csv.DictReader(fh))
    assert reader[0]["dataset"] == "Super1"
    assert reader[0]["encoder"] == "senet154"
    assert reader[0]["n_train"] == "1"

    md = render_markdown(rows, summarize(rows, build_curves(points)))
    assert "Encoder" in md and "Rel. err. reduction" in md and "%" in md


def test_plot_curves_writes_png(tmp_path: Path) -> None:
    # Two encoders x two regimes -> four labelled curves in one figure.
    points = [
        RunPoint("Super1", "imagenet", 1, "senet154", 0.3),
        RunPoint("Super1", "imagenet", 4, "senet154", 0.7),
        RunPoint("Super1", "micronet", 1, "senet154", 0.75),
        RunPoint("Super1", "micronet", 4, "senet154", 0.8),
        RunPoint("Super1", "imagenet", 1, "se_resnext50_32x4d", 0.25),
        RunPoint("Super1", "micronet", 4, "se_resnext50_32x4d", 0.78),
    ]
    out = plot_curves(build_curves(points), tmp_path / "curves.png")
    assert out.exists() and out.stat().st_size > 0
