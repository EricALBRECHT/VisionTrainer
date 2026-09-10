"""Result models for generic video / camera processing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

VideoMode = Literal["detect", "segment", "pipeline"]


@dataclass
class FrameTimingsAgg:
    """Incremental timing aggregates (avoid storing per-frame lists)."""

    count: int = 0
    detection_ms_sum: float = 0.0
    classification_ms_sum: float = 0.0
    segmentation_ms_sum: float = 0.0
    total_ms_sum: float = 0.0

    def add(
        self,
        *,
        detection_ms: float = 0.0,
        classification_ms: float = 0.0,
        segmentation_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> None:
        self.count += 1
        self.detection_ms_sum += float(detection_ms)
        self.classification_ms_sum += float(classification_ms)
        self.segmentation_ms_sum += float(segmentation_ms)
        self.total_ms_sum += float(total_ms)

    def averages(self) -> dict[str, float | None]:
        if self.count <= 0:
            return {
                "detection_ms": None,
                "classification_ms": None,
                "segmentation_ms": None,
                "total_ms": None,
            }
        n = float(self.count)
        return {
            "detection_ms": self.detection_ms_sum / n,
            "classification_ms": self.classification_ms_sum / n,
            "segmentation_ms": self.segmentation_ms_sum / n,
            "total_ms": self.total_ms_sum / n,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_with_timings": self.count,
            "averages_ms": self.averages(),
        }


@dataclass
class VideoJobSummary:
    mode: VideoMode
    output_path: str | None
    source_path: str | None
    source_width: int
    source_height: int
    source_fps: float | None
    output_fps: float
    frames_total: int | None
    frames_read: int
    frames_processed: int
    frames_skipped: int
    elapsed_seconds: float
    mean_processing_fps: float
    mean_ms_per_frame: float | None
    frame_stride: int
    total_detections: int = 0
    total_classifications: int = 0
    total_segmentations: int = 0
    secondary_errors: int = 0
    warnings: list[str] = field(default_factory=list)
    timings: FrameTimingsAgg = field(default_factory=FrameTimingsAgg)
    # Optional per-frame lightweight records (disabled by default).
    frame_details: list[dict[str, Any]] = field(default_factory=list)
    audio_preserved: bool = False

    def to_dict(self, *, include_frames: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video": {
                "path": self.source_path,
                "width": self.source_width,
                "height": self.source_height,
                "fps": self.source_fps,
                "frames": self.frames_total,
            },
            "processing": {
                "mode": self.mode,
                "frame_stride": self.frame_stride,
                "frames_read": self.frames_read,
                "frames_processed": self.frames_processed,
                "frames_skipped": self.frames_skipped,
                "elapsed_seconds": self.elapsed_seconds,
                "mean_processing_fps": self.mean_processing_fps,
                "mean_ms_per_frame": self.mean_ms_per_frame,
                "output_fps": self.output_fps,
                "audio_preserved": self.audio_preserved,
            },
            "counts": {
                "detections": self.total_detections,
                "classifications": self.total_classifications,
                "segmentations": self.total_segmentations,
                "secondary_errors": self.secondary_errors,
            },
            "timings": self.timings.to_dict(),
            "warnings": list(self.warnings),
            "output_path": self.output_path,
        }
        if include_frames:
            payload["frames"] = list(self.frame_details)
        return payload
