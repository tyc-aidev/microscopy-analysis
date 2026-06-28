"""Compute-device selection across CUDA, Apple Silicon (MPS), and CPU.

Lets the same code run on a CUDA reproduction host and on a local Apple Silicon
Mac (Metal / MPS) for development and iteration. Torch is imported lazily so
torch-free callers can still import this module.
"""

from __future__ import annotations

import os

PREFERENCES = ("auto", "cuda", "mps", "cpu")


def _mps_available() -> bool:
    import torch  # noqa: PLC0415

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def enable_mps_fallback() -> None:
    """Allow ops unimplemented on MPS to fall back to CPU instead of erroring.

    Must be set before the ops run; torch reads it at dispatch time.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(prefer: str = "auto"):
    """Return a ``torch.device`` for ``prefer`` in {auto, cuda, mps, cpu}.

    ``auto`` picks CUDA, then MPS, then CPU. An explicit ``cuda``/``mps`` that is
    unavailable warns and falls back to CPU rather than raising, so scripts keep
    running on whatever hardware is present.
    """
    import torch  # noqa: PLC0415

    prefer = prefer.lower()
    if prefer not in PREFERENCES:
        raise ValueError(f"prefer must be one of {PREFERENCES}, got {prefer!r}")

    if prefer != "cpu" and prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer != "cpu" and prefer in ("auto", "mps") and _mps_available():
        enable_mps_fallback()
        return torch.device("mps")

    if prefer == "cuda":
        print("[device] CUDA requested but unavailable; falling back to CPU.")
    elif prefer == "mps":
        print("[device] MPS requested but unavailable; falling back to CPU.")
    return torch.device("cpu")
