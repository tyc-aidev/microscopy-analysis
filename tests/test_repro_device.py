"""Tests for compute-device resolution (cuda/mps/cpu). Requires torch."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from microscopy_analysis.device import enable_mps_fallback, resolve_device  # noqa: E402


def test_cpu_is_always_resolvable():
    assert resolve_device("cpu").type == "cpu"


def test_auto_returns_a_known_device():
    assert resolve_device("auto").type in ("cuda", "mps", "cpu")


def test_explicit_unavailable_falls_back_to_cpu():
    # At most one accelerator is present; the absent one must fall back to CPU.
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type == "cpu"
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        assert resolve_device("mps").type == "cpu"


def test_invalid_preference_rejected():
    with pytest.raises(ValueError, match="prefer must be"):
        resolve_device("tpu")


def test_enable_mps_fallback_sets_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    enable_mps_fallback()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
