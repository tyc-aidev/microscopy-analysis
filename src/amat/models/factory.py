"""Thin wrapper around NASA pretrained microscopy model creation."""

from __future__ import annotations

import inspect


def create_segmentation_model(config: dict) -> object:
    """Create a segmentation model with paper-correct defaults."""
    try:
        from pretrained_microscopy_models import model as pmm_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pretrained_microscopy_models is required for training. "
            "Install it with `uv pip install git+https://github.com/nasa/pretrained-microscopy-models`."
        ) from exc

    fn = pmm_model.create_segmentation_model
    sig = inspect.signature(fn)
    kwargs = {}

    if "architecture" in sig.parameters:
        kwargs["architecture"] = config["architecture"]
    if "encoder_name" in sig.parameters:
        kwargs["encoder_name"] = config["encoder_name"]
    if "n_classes" in sig.parameters:
        kwargs["n_classes"] = config["num_classes"]
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = config["num_classes"]
    if "pretraining" in sig.parameters:
        kwargs["pretraining"] = config["pretraining"]
    if "encoder_weights" in sig.parameters:
        kwargs["encoder_weights"] = config["pretraining"]

    # Paper reproduction requires v1.0 encoder weights whenever MicroNet is used.
    if config["pretraining"] in {"micronet", "image-micronet"} and "version" in sig.parameters:
        kwargs["version"] = 1.0

    return fn(**kwargs)

