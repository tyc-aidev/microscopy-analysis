"""Shared Streamlit dashboard for a single training run."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from explorer.lib.prediction_panels import (
    ensure_panels,
    list_panels,
    render_panel_browser,
    torch_available,
)
from explorer.lib.runs import RunInfo, load_metrics
from explorer.lib.training_charts import plot_iou_curves, plot_loss_curves, plot_per_class_iou


def _render_config_expander(config_path: Path | None) -> None:
    if config_path is None or not config_path.is_file():
        st.caption("No matching experiment YAML found in `configs/experiments/`.")
        return
    try:
        from microscopy_analysis.train.config import load_train_config

        cfg = load_train_config(config_path)
    except Exception as exc:
        st.warning(f"Could not load config: {exc}")
        return

    with st.expander("Experiment config", expanded=False):
        st.write(f"**Architecture:** {cfg.architecture}")
        st.write(f"**Encoder:** {cfg.encoder_name} ({cfg.pretraining})")
        st.write(f"**Dataset:** {cfg.dataset_name} ({cfg.dataset_family})")
        st.write(f"**Classes:** {cfg.num_classes}")
        st.write(f"**Phase 1 LR:** {cfg.lr_phase1} · **Phase 2 LR:** {cfg.lr_phase2}")
        st.write(f"**Patience:** {cfg.patience} · **Batch size:** {cfg.batch_size}")
        st.write(f"**Max epochs:** {cfg.max_epochs_phase1} + {cfg.max_epochs_phase2}")
        st.caption(f"Config: `{config_path.name}`")


def render_run_dashboard(
    run: RunInfo,
    config_path: Path | None,
    *,
    split: str,
    allow_live_inference: bool,
    key_prefix: str,
    inference_device: str = "auto",
) -> None:
    summary = run.summary
    st.subheader(run.run_name)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Best val IoU", f"{run.best_mean_iou:.4f}")
    m2.metric("Best epoch", run.best_epoch)
    m3.metric("Epochs trained", run.epochs_trained)
    m4.metric("Device", run.device)
    sha = run.git_sha
    m5.metric("Git SHA", sha[:8] if len(sha) >= 8 else sha)

    st.caption(
        f"**{summary.get('dataset_name', '?')}** · "
        f"{summary.get('num_samples', '?')} train / {summary.get('num_val_samples', '?')} val samples"
    )

    _render_config_expander(config_path)

    if not run.has_metrics:
        st.warning("No `metrics.json` found for this run.")
        return

    metrics = load_metrics(run.run_dir / "metrics.json")
    family = str(summary.get("dataset_family", "super"))

    st.markdown("**Training curves**")
    col_loss, col_iou = st.columns(2)
    with col_loss:
        st.pyplot(plot_loss_curves(metrics), clear_figure=True)
    with col_iou:
        st.pyplot(plot_iou_curves(metrics, dataset_family=family), clear_figure=True)

    per_class_fig = plot_per_class_iou(metrics, dataset_family=family)
    if per_class_fig is not None:
        st.pyplot(per_class_fig, clear_figure=True)

    st.markdown("**Validation predictions**")
    panels = list_panels(run.run_dir, split)
    can_generate = (
        allow_live_inference and run.has_best and config_path is not None and torch_available()
    )
    auto_key = f"{key_prefix}_panels_autogen_{run.run_name}_{split}"

    if not panels and not can_generate:
        if allow_live_inference:
            if not run.has_best:
                st.warning("No `model_best.pth` — train or sync checkpoints before generating panels.")
            elif config_path is None:
                st.info("Select a matching experiment YAML in the sidebar to generate panels.")
            elif not torch_available():
                st.info(
                    "Install the PyTorch stack for live inference:\n\n"
                    "```bash\nuv pip install -r requirements-apple.txt\n```\n\n"
                    "Or pre-generate panels:\n\n"
                    f"```bash\npython scripts/visualize_predictions.py "
                    f"--config {config_path} --split {split}\n```"
                )
        else:
            st.info(
                "Prediction panels must be generated on the CUDA host before sync:\n\n"
                f"```bash\npython scripts/visualize_predictions.py "
                f"--config configs/experiments/<experiment>.yaml --split {split} --device cuda\n```\n\n"
                "Then copy `results/<run_name>/predictions/` to this machine."
            )
    elif not panels and can_generate and not st.session_state.get(auto_key):
        with st.spinner("Generating prediction panels…"):
            try:
                panels = ensure_panels(
                    config_path, run.run_dir, split, device=inference_device
                )
                st.session_state[auto_key] = True
                st.session_state[f"{key_prefix}_panel_idx"] = 0
            except Exception as exc:
                st.error(str(exc))

    if can_generate:
        if st.button("Regenerate panels", key=f"{key_prefix}_regen"):
            with st.spinner("Running inference…"):
                try:
                    panels = ensure_panels(
                        config_path,
                        run.run_dir,
                        split,
                        device=inference_device,
                        force=True,
                    )
                    st.session_state[auto_key] = True
                    st.session_state[f"{key_prefix}_panel_idx"] = 0
                    st.success(f"Generated {len(panels)} panel(s).")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    render_panel_browser(panels, key_prefix=key_prefix)
