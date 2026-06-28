#!/usr/bin/env python3
"""Sprint 0 smoke test: verify the reproduction foundation end to end.

Runs as many checks as the current environment allows and prints a PASS/SKIP/FAIL
line per step:

1. Load + validate the experiment config.
2. Discover benchmark image/mask pairs (needs ``./scripts/download_data.sh``).
3. Resolve the pinned MicroNet **v1.0** weight URL (optionally HTTP-HEAD it).
4. Build the segmentation model and load v1.0 encoder weights (needs torch + pmm).
5. Run a forward pass and report the output shape (needs torch + pmm).

Steps 4-5 are skipped (not failed) when the PyTorch reproduction stack is not
installed, so the torch-free checks can run on any machine. Use ``--build`` to
require the model steps. The first ``--build`` downloads the ~100MB v1.0 encoder
weights into the torch hub cache; subsequent runs reuse the cached file.

Examples:
    python scripts/smoke_test.py --config configs/experiments/super1_smoke.yaml
    python scripts/smoke_test.py --config configs/experiments/ebc1_smoke.yaml --check-url --build
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from microscopy_analysis.config import ExperimentConfig, load_experiment_config  # noqa: E402
from microscopy_analysis.data import discover_pairs, split_counts  # noqa: E402
from microscopy_analysis.device import resolve_device  # noqa: E402
from microscopy_analysis.models import create_segmentation_model, micronet_weight_url  # noqa: E402

PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"


class Report:
    def __init__(self) -> None:
        self.failed = False

    def log(self, status: str, step: str, detail: str = "") -> None:
        if status == FAIL:
            self.failed = True
        line = f"[{status:4}] {step}"
        print(f"{line}: {detail}" if detail else line)


def _torch_stack_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "segmentation_models_pytorch"))


def check_data(report: Report, cfg: ExperimentConfig, data_root: Path | None) -> None:
    pairs = discover_pairs(cfg.dataset.name, cfg.dataset.family, data_root=data_root)
    if not pairs:
        report.log(
            FAIL,
            "data",
            f"no images found for {cfg.dataset.name}; run ./scripts/download_data.sh",
        )
        return
    counts = split_counts(pairs)
    missing = [p.image_path.name for p in pairs if p.mask_path is None]
    if missing:
        report.log(FAIL, "data", f"{len(missing)} image(s) without a mask, e.g. {missing[0]}")
        return
    report.log(PASS, "data", f"{cfg.dataset.name} pairs by split: {counts}")


def check_weight_url(report: Report, cfg: ExperimentConfig, check_url: bool) -> None:
    if cfg.model.pretraining not in ("micronet", "image-micronet"):
        report.log(SKIP, "weights", f"pretraining={cfg.model.pretraining} needs no MicroNet URL")
        return
    url = micronet_weight_url(cfg.model.encoder, cfg.model.pretraining, version=cfg.model.micronet_version)
    if f"_v{cfg.model.micronet_version}." not in url:
        report.log(FAIL, "weights", f"URL not pinned to v{cfg.model.micronet_version}: {url}")
        return
    if not check_url:
        report.log(PASS, "weights", f"v1.0 URL: {url}")
        return
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            ok = resp.status == 200
        report.log(PASS if ok else FAIL, "weights", f"HTTP {resp.status} for {url}")
    except Exception as exc:  # noqa: BLE001 — network failure is a soft signal here
        report.log(FAIL, "weights", f"HEAD {url} failed: {exc}")


def check_model(report: Report, cfg: ExperimentConfig, require_build: bool, prefer_device: str) -> None:
    if not _torch_stack_available():
        status = FAIL if require_build else SKIP
        report.log(status, "model", "torch + segmentation_models_pytorch not installed")
        return
    try:
        import torch  # noqa: PLC0415

        device = resolve_device(prefer_device)
        report.log(PASS, "device", f"using {device.type}")

        model = create_segmentation_model(
            cfg.model.architecture,
            cfg.model.encoder,
            cfg.model.pretraining,
            cfg.dataset.classes,
            micronet_version=cfg.model.micronet_version,
        )
        model = model.to(device).eval()
        report.log(PASS, "model", f"built {cfg.model.architecture}/{cfg.model.encoder} ({cfg.model.pretraining})")

        with torch.no_grad():
            out = model(torch.randn(1, 3, 256, 256, device=device))
        report.log(PASS, "forward", f"output shape {tuple(out.shape)} on {device.type}")
    except Exception as exc:  # noqa: BLE001
        report.log(FAIL, "model", f"{type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 0 reproduction smoke test")
    parser.add_argument("--config", required=True, type=Path, help="experiment YAML path")
    parser.add_argument("--data-root", type=Path, default=None, help="override DATA_ROOT")
    parser.add_argument("--check-url", action="store_true", help="HTTP-HEAD the weight URL")
    parser.add_argument("--build", action="store_true", help="require model build/forward (fail if torch missing)")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"), help="preferred device")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report()

    try:
        cfg = load_experiment_config(args.config)
        report.log(PASS, "config", f"loaded {cfg.name}")
    except Exception as exc:  # noqa: BLE001
        report.log(FAIL, "config", f"{type(exc).__name__}: {exc}")
        return 1

    check_data(report, cfg, args.data_root)
    check_weight_url(report, cfg, args.check_url)
    check_model(report, cfg, args.build, args.device)

    print("\nResult:", "FAIL" if report.failed else "OK")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
