"""MicroNet pretrained-encoder weight URLs, pinned to the paper's v1.0.

Mirrors NASA's ``pmm.util.get_pretrained_microscopynet_url`` but inverts the
default: that helper falls back to **v1.1** for ``resnet50/micronet``, which is
the exact reproduction pitfall called out in PLAN.md. Here the paper version is
the default and any other version must be opted into explicitly.
"""

from __future__ import annotations

S3_BASE = "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
PAPER_VERSION = "1.0"

_WEIGHTS_NAMES = {"micronet": "microscopynet", "image-micronet": "imagenet-microscopynet"}
# resnext101_32x8d weights are published under a shortened name (v1.0 only).
_ENCODER_ALIASES = {"resnext101_32x8d": "resnext101"}


def normalize_version(version: str | float | int) -> str:
    """Normalize a version (``1``, ``1.0``, ``"1.0"``) to a canonical string."""
    return str(float(version))


def micronet_weight_url(
    encoder: str,
    encoder_weights: str,
    version: str | float = PAPER_VERSION,
    self_supervision: str = "",
    allow_non_paper_version: bool = False,
) -> str:
    """Resolve the S3 URL for a MicroNet-pretrained encoder.

    Defaults to the paper's v1.0; requesting any other version raises unless
    ``allow_non_paper_version`` is set (the Sprint 4 ablation escape hatch).
    """
    version_str = normalize_version(version)
    if version_str != PAPER_VERSION and not allow_non_paper_version:
        raise ValueError(
            f"non-paper MicroNet version {version_str!r} requested; reproduction "
            f"pins v{PAPER_VERSION}. Pass allow_non_paper_version=True to override."
        )

    try:
        weights_name = _WEIGHTS_NAMES[encoder_weights]
    except KeyError:
        raise ValueError("encoder_weights must be 'micronet' or 'image-micronet'") from None

    encoder = _ENCODER_ALIASES.get(encoder, encoder)
    suffix = f"_{self_supervision}" if self_supervision else ""
    return f"{S3_BASE}{encoder}{suffix}_pretrained_{weights_name}_v{version_str}.pth.tar"
