from __future__ import annotations

import os
from pathlib import Path

# Environment variable used in Docker and optional local overrides.
DATA_DIR_ENV = "VISIONTRAINER_DATA_DIR"


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


def get_runs_dir() -> Path:
    return get_data_root() / "runs"


def get_tmp_dir() -> Path:
    """Writable temp directory under the data root (created on demand)."""
    path = get_data_root() / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path
