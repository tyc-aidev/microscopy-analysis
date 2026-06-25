"""Tests for aggregate dataset statistics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from explorer.lib.index import scan_benchmarks, split_counts
from explorer.lib.stats import (
    aggregate_class_pixels,
    class_distribution_dataframe,
    image_counts_pivot,
    split_summary_table,
)


def test_image_counts_pivot(benchmark_fixture: Path) -> None:
    records = scan_benchmarks(benchmark_fixture)
    counts = split_counts(records)
    pivot = image_counts_pivot(counts)
    assert isinstance(pivot, pd.DataFrame)
    assert pivot.loc["Super1", "train"] == 1
    assert pivot.loc["EBC1", "train"] == 1


def test_split_summary_table(benchmark_fixture: Path) -> None:
    records = scan_benchmarks(benchmark_fixture)
    counts = split_counts(records)
    table = split_summary_table(counts)
    assert set(table["dataset"]) == {"Super1", "EBC1"}
    super1_train = table[(table["dataset"] == "Super1") & (table["split"] == "train")]
    assert super1_train.iloc[0]["images"] == 1


def test_aggregate_class_pixels(benchmark_fixture: Path) -> None:
    records = scan_benchmarks(benchmark_fixture)
    totals = aggregate_class_pixels(records)
    assert totals["Super1"]["matrix"] == 16
    distribution = class_distribution_dataframe(totals, "Super1")
    assert "Matrix" in distribution["class"].values
