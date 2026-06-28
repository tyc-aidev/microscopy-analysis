"""Segmentation model factory pinned to MicroNet v1.0 for paper reproduction.

Mirrors NASA's ``create_segmentation_model`` (build an ``smp`` model, then load
the pretrained encoder weights) but resolves MicroNet URLs through
:mod:`microscopy_analysis.models.weights`, which forces ``version=1.0`` — NASA's own helper
defaults to v1.1 for ``resnet50/micronet`` (the PLAN.md reproduction pitfall).

Argument validation runs before the lazy torch/smp import so config mistakes are
caught on any machine, torch-free.
"""

from __future__ import annotations

from .weights import PAPER_VERSION, micronet_weight_url

_PRETRAINING = ("random", "imagenet", "micronet", "image-micronet")
_MICRONET_PRETRAINING = frozenset({"micronet", "image-micronet"})
_IMAGENET_5K_ENCODERS = frozenset({"dpn68b", "dpn92", "dpn137", "dpn107"})


def create_segmentation_model(
    architecture: str,
    encoder: str,
    pretraining: str,
    classes: int,
    micronet_version: str | float = PAPER_VERSION,
    allow_non_paper_version: bool = False,
):
    """Build a segmentation model with paper-correct (v1.0) pretrained weights.

    ``classes`` counts foreground classes (background is implicit); binary tasks
    use ``classes=1`` and NASA explicitly rejects ``classes=2``.
    """
    if pretraining not in _PRETRAINING:
        raise ValueError(f"pretraining must be one of {_PRETRAINING}, got {pretraining!r}")
    classes = int(classes)
    if classes == 2:
        raise ValueError("binary segmentation must use classes=1 (background implicit); got classes=2")
    if classes < 1:
        raise ValueError(f"classes must be >= 1, got {classes}")

    # Resolve (and validate the pin on) the weight URL before any heavy import.
    weight_url = None
    if pretraining in _MICRONET_PRETRAINING:
        weight_url = micronet_weight_url(
            encoder, pretraining, version=micronet_version, allow_non_paper_version=allow_non_paper_version
        )

    try:
        import segmentation_models_pytorch as smp
        import torch
        import torch.utils.model_zoo as model_zoo
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch and segmentation_models_pytorch are required to build models. "
            "Install the reproduction stack (see requirements.txt) plus "
            "`uv pip install git+https://github.com/nasa/pretrained-microscopy-models`."
        ) from exc

    activation = "softmax2d" if classes > 1 else "sigmoid"
    initial_weights = "imagenet" if pretraining == "imagenet" else None
    if initial_weights == "imagenet" and encoder in _IMAGENET_5K_ENCODERS:
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

    if weight_url is not None:
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        model.encoder.load_state_dict(model_zoo.load_url(weight_url, map_location=map_location))

    return model
