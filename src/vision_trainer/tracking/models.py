"""Tracking data models (structured results, no rendering)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TrackerName = Literal["bytetrack", "botsort"]

DEFAULT_TRAJECTORY_MAX_POINTS = 50


@dataclass(frozen=True)
class TrackingConfig:
    """User-facing tracking options for video / camera."""

    enabled: bool = False
    tracker: TrackerName = "bytetrack"
    show_trajectories: bool = False
    reuse_classification: bool = True
    reclassify_every_n_frames: int = 30
    uncertain_reclassify_every_n_frames: int = 5
    trajectory_max_points: int = DEFAULT_TRAJECTORY_MAX_POINTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tracker": self.tracker,
            "show_trajectories": self.show_trajectories,
            "reuse_classification": self.reuse_classification,
            "reclassify_every_n_frames": self.reclassify_every_n_frames,
            "uncertain_reclassify_every_n_frames": self.uncertain_reclassify_every_n_frames,
            "trajectory_max_points": self.trajectory_max_points,
        }


@dataclass
class TrackedObject:
    """One active object identity at the current frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    center: tuple[float, float]
    first_seen_frame: int
    last_seen_frame: int
    age_frames: int
    frames_seen: int
    trajectory: list[tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": [float(v) for v in self.bbox],
            "center": [float(self.center[0]), float(self.center[1])],
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "age_frames": self.age_frames,
            "frames_seen": self.frames_seen,
            "trajectory": [[float(x), float(y)] for x, y in self.trajectory],
        }


@dataclass
class TrackSummary:
    """Lightweight finished (or still-active) track summary — no images."""

    track_id: int
    class_id: int
    class_name: str
    first_seen_frame: int
    last_seen_frame: int
    frames_seen: int
    first_center: tuple[float, float] | None = None
    last_center: tuple[float, float] | None = None
    finished: bool = False

    def duration_seconds(self, fps: float | None) -> float | None:
        if fps is None or fps <= 0:
            return None
        span = max(0, self.last_seen_frame - self.first_seen_frame)
        # Inclusive frame span ≈ (last - first) / fps for consecutive observations.
        return float(span) / float(fps)

    def to_dict(self, *, fps: float | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "frames_seen": self.frames_seen,
            "finished": self.finished,
            "first_center": (
                None
                if self.first_center is None
                else [float(self.first_center[0]), float(self.first_center[1])]
            ),
            "last_center": (
                None
                if self.last_center is None
                else [float(self.last_center[0]), float(self.last_center[1])]
            ),
        }
        duration = self.duration_seconds(fps)
        if duration is not None:
            payload["duration_seconds"] = duration
        return payload


@dataclass
class TrackingStats:
    active_tracks: int = 0
    tracks_created: int = 0
    tracks_finished: int = 0
    by_class: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_tracks": self.active_tracks,
            "tracks_created": self.tracks_created,
            "tracks_finished": self.tracks_finished,
            "by_class": dict(self.by_class),
            "total": self.tracks_created,
        }


@dataclass
class TrackingFrameResult:
    """Structured tracking output for one processed frame."""

    frame_index: int
    objects: list[TrackedObject] = field(default_factory=list)
    detection_tracking_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "objects": [obj.to_dict() for obj in self.objects],
            "detection_tracking_ms": self.detection_tracking_ms,
        }


def tracking_config_asdict(config: TrackingConfig) -> dict[str, Any]:
    return asdict(config)
