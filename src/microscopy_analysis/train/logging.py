"""Pluggable structured experiment logging for Sprint 1.

Backends are selected by name (``none`` | ``wandb`` | ``mlflow``); the default
``none`` keeps training fully offline (no imports, no network) so unit tests and
local smoke runs never touch a tracking server. Heavy backends are imported
lazily inside their constructors so the dependency is only required when used.

Every backend logs the same three things: the resolved run params (full config
+ git SHA), per-epoch metric records, and a final run summary.
"""

from __future__ import annotations

from typing import Any, Protocol


class ExperimentLogger(Protocol):
    """Minimal sink for run params, per-epoch metrics, and a final summary."""

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_epoch(self, record: dict[str, Any]) -> None: ...
    def finish(self, summary: dict[str, Any]) -> None: ...


class NoOpLogger:
    """Default backend: records nothing, makes no network calls."""

    def log_params(self, params: dict[str, Any]) -> None:  # noqa: D102
        pass

    def log_epoch(self, record: dict[str, Any]) -> None:  # noqa: D102
        pass

    def finish(self, summary: dict[str, Any]) -> None:  # noqa: D102
        pass


class WandbLogger:
    """Weights & Biases backend (lazy ``wandb`` import)."""

    def __init__(self, run_name: str, project: str | None) -> None:
        import wandb  # noqa: PLC0415 — optional dependency, only imported when selected

        self._wandb = wandb
        self._run = wandb.init(project=project or "microscopy-analysis", name=run_name, reinit=True)

    def log_params(self, params: dict[str, Any]) -> None:
        self._run.config.update(params, allow_val_change=True)

    def log_epoch(self, record: dict[str, Any]) -> None:
        self._run.log(record, step=record.get("epoch"))

    def finish(self, summary: dict[str, Any]) -> None:
        self._run.summary.update(summary)
        self._run.finish()


class MLflowLogger:
    """MLflow backend (lazy ``mlflow`` import)."""

    def __init__(self, run_name: str, project: str | None) -> None:
        import mlflow  # noqa: PLC0415 — optional dependency, only imported when selected

        self._mlflow = mlflow
        if project:
            mlflow.set_experiment(project)
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_epoch(self, record: dict[str, Any]) -> None:
        step = record.get("epoch")
        metrics = {k: v for k, v in record.items() if isinstance(v, (int, float))}
        self._mlflow.log_metrics(metrics, step=step)

    def finish(self, summary: dict[str, Any]) -> None:
        self._mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        self._mlflow.end_run()


def build_logger(backend: str, run_name: str, project: str | None = None) -> ExperimentLogger:
    """Construct the logger for ``backend`` (``none`` | ``wandb`` | ``mlflow``)."""
    key = (backend or "none").lower()
    if key == "none":
        return NoOpLogger()
    if key == "wandb":
        return WandbLogger(run_name, project)
    if key == "mlflow":
        return MLflowLogger(run_name, project)
    raise ValueError(f"Unknown logging backend: {backend!r} (expected none|wandb|mlflow)")
