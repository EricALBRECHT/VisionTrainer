from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class AvailableModel:
    run_id: str
    weights_path: str
    label: str


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)
    class_names: dict[int, str] = field(default_factory=dict)
    image_width: int = 0
    image_height: int = 0

    @property
    def count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class VideoMetadata:
    """Lightweight probe of an input video (missing fields stay None)."""

    path: str
    display_name: str
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class VideoProgress:
    frames_done: int
    frames_total: int | None
    percent: float | None
    elapsed_seconds: float
    processing_fps: float | None
    eta_seconds: float | None


@dataclass
class VideoInferenceResult:
    """
    Outcome of a video inference pass.

    ``total_detections`` and ``detections_by_class`` are cumulative counts across
    frames (no tracking) — not unique object counts.
    """

    output_path: str
    frames_analyzed: int
    elapsed_seconds: float
    mean_processing_fps: float
    total_detections: int
    detections_by_class: dict[str, int] = field(default_factory=dict)
    source_width: int = 0
    source_height: int = 0
    source_fps: float | None = None
    output_fps: float = 0.0
    # Explicit reminder for UI / API consumers.
    counts_are_cumulative: bool = True
