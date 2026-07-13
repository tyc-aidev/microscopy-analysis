"""Job-matrix generation, low-data sweeps, and dispatch for the reproduction."""

from .low_data import (
    DEFAULT_SIZES,
    LOW_DATA_PRETRAININGS,
    SUPER_DATASETS,
    LowDataJob,
    generate_low_data_jobs,
    resolve_sizes,
)
from .low_data import ENCODERS as LOW_DATA_ENCODERS
from .low_data_analysis import (
    LowDataRow,
    RunPoint,
    build_curves,
    build_low_data_rows,
    load_low_data_scores,
    relative_error_reduction,
    summarize,
)
from .matrix import (
    DATASETS,
    ENCODERS,
    PRETRAININGS,
    Job,
    generate_jobs,
    render_config,
    write_configs,
    write_manifest,
)

__all__ = [
    "DATASETS",
    "ENCODERS",
    "PRETRAININGS",
    "Job",
    "generate_jobs",
    "render_config",
    "write_configs",
    "write_manifest",
    # Sprint 3 low-data ablation
    "DEFAULT_SIZES",
    "LOW_DATA_PRETRAININGS",
    "LOW_DATA_ENCODERS",
    "SUPER_DATASETS",
    "LowDataJob",
    "generate_low_data_jobs",
    "resolve_sizes",
    "LowDataRow",
    "RunPoint",
    "build_curves",
    "build_low_data_rows",
    "load_low_data_scores",
    "relative_error_reduction",
    "summarize",
]
