"""Thin wrapper around NASA pretrained microscopy model creation."""

from __future__ import annotations


def create_segmentation_model(config: dict) -> object:
    """Create a segmentation model with paper-correct defaults."""
    try:
        from pretrained_microscopy_models import model as pmm_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pretrained_microscopy_models is required for training. "
            "Install it with `uv pip install git+https://github.com/nasa/pretrained-microscopy-models`."
        ) from exc

    kwargs = {
        "architecture": config["architecture"],
        "encoder_name": config["encoder_name"],
        "encoder_weights": config["pretraining"],
        "n_classes": config["num_classes"],
    }
    # Paper reproduction requires v1.0 encoder weights whenever MicroNet is used.
    if config["pretraining"] in {"micronet", "image-micronet"}:
        kwargs["version"] = 1.0

    return pmm_model.create_segmentation_model(**kwargs)

