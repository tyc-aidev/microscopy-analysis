"""Tests for MicroNet v1.0 weight pinning in the model factory."""

from __future__ import annotations

from amat.models.factory import _micronet_url


def test_micronet_url_pins_version_1_0() -> None:
    captured: dict[str, object] = {}

    def fake_url_fn(encoder: str, encoder_weights: str, version: float) -> str:
        captured["encoder"] = encoder
        captured["encoder_weights"] = encoder_weights
        captured["version"] = version
        return f"https://example/{encoder}_{encoder_weights}_v{version}.pth.tar"

    url = _micronet_url("resnet50", "micronet", url_fn=fake_url_fn)

    assert captured["version"] == 1.0
    assert captured["encoder"] == "resnet50"
    assert "v1.0" in url
