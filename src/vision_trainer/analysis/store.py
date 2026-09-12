"""Persist and load run analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from vision_trainer.analysis.models import (
    ANALYSIS_FILENAME,
    DATASET_STATS_FILENAME,
    RunAnalysis,
)
from vision_trainer.io_utils import atomic_write_json


def analysis_path(run_dir: Path) -> Path:
    return Path(run_dir) / ANALYSIS_FILENAME


def dataset_stats_path(run_dir: Path) -> Path:
    return Path(run_dir) / DATASET_STATS_FILENAME


def load_analysis(run_dir: Path) -> RunAnalysis | None:
    path = analysis_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return RunAnalysis.from_dict(data)
    except (TypeError, ValueError, KeyError):
        return None


def save_analysis(run_dir: Path, analysis: RunAnalysis) -> Path:
    path = analysis_path(run_dir)
    atomic_write_json(path, analysis.to_dict())
    return path


def load_dataset_stats_cache(run_dir: Path) -> dict | None:
    path = dataset_stats_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_dataset_stats_cache(run_dir: Path, stats: dict) -> Path:
    path = dataset_stats_path(run_dir)
    atomic_write_json(path, stats)
    return path
