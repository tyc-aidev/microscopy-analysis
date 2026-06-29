"""Benchmark dataset discovery and image/mask pairing.

The torch ``SegmentationDataset`` lives in :mod:`.segmentation_dataset` and is
imported from there directly so this package stays torch/albumentations-free for
lightweight (explorer / repro) consumers.
"""

from .datasets import ImagePair, benchmark_root, discover_pairs, split_counts

__all__ = ["ImagePair", "benchmark_root", "discover_pairs", "split_counts"]
