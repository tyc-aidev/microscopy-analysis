"""Segmentation model factory pinned to MicroNet v1.0 for paper reproduction.

This mirrors NASA's ``create_segmentation_model`` (build an ``smp`` model, then
load encoder weights) but forces ``version=1.0`` for MicroNet weights. NASA's own
``util.get_pretrained_microscopynet_url`` defaults to v1.1 for ``resnet50/micronet``,
which is the exact reproduction pitfall called out in PLAN.md.
"""

from __future__ import annotations

MICRONET_PRETRAINING = frozenset({"micronet", "image-micronet"})
IMAGENET_5K_ENCODERS = frozenset({"dpn68b", "dpn92", "dpn137", "dpn107"})


def _micronet_url(encoder: str, encoder_weights: str, url_fn=None) -> str:
    """Resolve a MicroNet weight URL, always pinned to v1.0."""
    if url_fn is None:
        from pretrained_microscopy_models import util

        url_fn = util.get_pretrained_microscopynet_url
    return url_fn(encoder, encoder_weights, version=1.0)


def create_segmentation_model(config: dict) -> object:
    """Create a segmentation model with paper-correct (v1.0) pretrained weights."""
    try:
        import segmentation_models_pytorch as smp
        import torch
        import torch.utils.model_zoo as model_zoo
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch and segmentation_models_pytorch are required for training. "
            "Install the reproduction stack (see requirements) plus "
            "`uv pip install git+https://github.com/nasa/pretrained-microscopy-models`."
        ) from exc

    architecture = config["architecture"]
    encoder = config["encoder_name"]
    encoder_weights = config["pretraining"]
    classes = int(config["num_classes"])

    if classes == 2:
        raise ValueError(
            "Binary segmentation must use num_classes=1 (background is implicit); "
            "got num_classes=2."
        )

    activation = "softmax2d" if classes > 1 else "sigmoid"
    initial_weights = "imagenet" if encoder_weights == "imagenet" else None
    if initial_weights == "imagenet" and encoder in IMAGENET_5K_ENCODERS:
        initial_weights = "imagenet+5k"

    try:
        model = getattr(smp, architecture)(
            encoder_name=encoder,
            encoder_weights=initial_weights,
            classes=classes,
            activation=activation,
        )
    except ValueError as exc:
        raise ValueError(
            f"{encoder} does not support the dilated mode needed for {architecture}."
        ) from exc

    if encoder_weights in MICRONET_PRETRAINING:
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        url = _micronet_url(encoder, encoder_weights)
        model.encoder.load_state_dict(model_zoo.load_url(url, map_location=map_location))

    return model
