"""Helpers to wire TrackingSession onto cached YOLO models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_trainer.inference.predictor import extract_class_names
from vision_trainer.pipeline.engine import get_cached_model
from vision_trainer.tracking.models import DEFAULT_TRAJECTORY_MAX_POINTS, TrackerName
from vision_trainer.tracking.session import TrackingSession
from vision_trainer.video.processors import ModelCache, ModelFactory


def create_tracking_session(
    weights_path: str | Path,
    *,
    tracker: TrackerName = "bytetrack",
    model_cache: ModelCache | None = None,
    model_factory: ModelFactory | None = None,
    trajectory_max_points: int = DEFAULT_TRAJECTORY_MAX_POINTS,
    fps: float | None = None,
) -> tuple[TrackingSession, Any]:
    """Load/cache detector once and return a TrackingSession bound to it."""
    cache = model_cache if model_cache is not None else {}
    model = get_cached_model(
        weights_path,
        cache=cache,
        model_factory=model_factory,
    )
    names = extract_class_names(model)
    session = TrackingSession.from_ultralytics_model(
        model,
        tracker=tracker,
        class_names=names,
        trajectory_max_points=trajectory_max_points,
        fps=fps,
    )
    return session, model
