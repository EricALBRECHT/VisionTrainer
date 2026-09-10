from __future__ import annotations

import os
from pathlib import Path

# Environment variable used in Docker and optional local overrides.
DATA_DIR_ENV = "VISIONTRAINER_DATA_DIR"

# Read-only bind-mount root for large/external datasets (container path).
# Host path is configured in docker-compose only — never hard-code D: here.
EXTERNAL_DATASETS_ENV = "VISION_TRAINER_EXTERNAL_DATASETS"
DEFAULT_EXTERNAL_DATASETS_ROOT = "/datasets"


def get_data_root() -> Path:
    """
    Return the persistent data root for datasets, runs and related artifacts.

    - If ``VISIONTRAINER_DATA_DIR`` is set: use that path.
    - Otherwise: ``./artifacts`` relative to the process working directory
      (backward-compatible local default).
    """
    raw = os.environ.get(DATA_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "artifacts").resolve()


def get_datasets_dir() -> Path:
    return get_data_root() / "datasets"


def get_external_datasets_root() -> Path:
    """
    Root directory for external (mounted) datasets.

    Default ``/datasets``. Override with ``VISION_TRAINER_EXTERNAL_DATASETS``.
    The path is not created automatically (mount may be absent).
    """
    raw = os.environ.get(EXTERNAL_DATASETS_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(DEFAULT_EXTERNAL_DATASETS_ROOT)


def get_runs_dir() -> Path:
    return get_data_root() / "runs"


def get_tmp_dir() -> Path:
    """Writable temp directory under the data root (created on demand)."""
    path = get_data_root() / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_pipelines_dir() -> Path:
    """Persistent directory for detection→classification pipeline configs."""
    path = get_data_root() / "pipelines"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_dir(run_id: str) -> Path:
    """Return the directory for a single training run under the data root."""
    return get_runs_dir() / run_id
