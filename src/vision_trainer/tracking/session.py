"""Stateful tracking session across video / camera frames."""

from __future__ import annotations

import time
from typing import Any

from PIL import Image

from vision_trainer.tracking.backend import (
    FakeTrackerBackend,
    RawTrackDetection,
    TrackerBackend,
    TrackingBackendError,
    UltralyticsTrackerBackend,
)
from vision_trainer.tracking.geometry import bbox_center
from vision_trainer.tracking.models import (
    DEFAULT_TRAJECTORY_MAX_POINTS,
    TrackedObject,
    TrackerName,
    TrackingFrameResult,
    TrackingStats,
    TrackSummary,
)


class TrackingSession:
    """
    Keeps tracker state between frames.

    Reset on: new video, camera source change, model change, or explicit user reset.
    """

    def __init__(
        self,
        backend: TrackerBackend,
        *,
        tracker_name: TrackerName = "bytetrack",
        trajectory_max_points: int = DEFAULT_TRAJECTORY_MAX_POINTS,
        fps: float | None = None,
    ) -> None:
        self.backend = backend
        self.tracker_name = tracker_name
        self.trajectory_max_points = max(1, int(trajectory_max_points))
        self.fps = fps
        self._active: dict[int, TrackedObject] = {}
        self._finished: dict[int, TrackSummary] = {}
        self._created_ids: set[int] = set()
        self._class_of_track: dict[int, str] = {}

    @classmethod
    def from_ultralytics_model(
        cls,
        model: Any,
        *,
        tracker: TrackerName = "bytetrack",
        class_names: dict[int, str] | None = None,
        trajectory_max_points: int = DEFAULT_TRAJECTORY_MAX_POINTS,
        fps: float | None = None,
    ) -> TrackingSession:
        backend = UltralyticsTrackerBackend(
            model,
            tracker=tracker,
            class_names=class_names,
        )
        return cls(
            backend,
            tracker_name=tracker,
            trajectory_max_points=trajectory_max_points,
            fps=fps,
        )

    @classmethod
    def from_fake(
        cls,
        provider,
        **kwargs: Any,
    ) -> TrackingSession:
        return cls(FakeTrackerBackend(provider), tracker_name="bytetrack", **kwargs)

    def reset(self) -> None:
        self.backend.reset()
        self._active.clear()
        self._finished.clear()
        self._created_ids.clear()
        self._class_of_track.clear()

    def process(
        self,
        image: Image.Image,
        *,
        frame_index: int,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | int | None = None,
    ) -> TrackingFrameResult:
        t0 = time.perf_counter()
        try:
            raw = self.backend.track_frame(
                image, conf=conf, iou=iou, device=device
            )
        except TrackingBackendError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TrackingBackendError(str(exc)) from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        objects = self._ingest(raw, frame_index=int(frame_index))
        return TrackingFrameResult(
            frame_index=int(frame_index),
            objects=objects,
            detection_tracking_ms=elapsed_ms,
        )

    def ingest_raw(
        self,
        detections: list[RawTrackDetection],
        *,
        frame_index: int,
        detection_tracking_ms: float = 0.0,
    ) -> TrackingFrameResult:
        """Update session state from already-associated detections (tests / adapters)."""
        objects = self._ingest(detections, frame_index=int(frame_index))
        return TrackingFrameResult(
            frame_index=int(frame_index),
            objects=objects,
            detection_tracking_ms=float(detection_tracking_ms),
        )

    def _ingest(
        self,
        detections: list[RawTrackDetection],
        *,
        frame_index: int,
    ) -> list[TrackedObject]:
        seen_ids: set[int] = set()
        current: list[TrackedObject] = []

        for det in detections:
            track_id = int(det.track_id)
            seen_ids.add(track_id)
            center = det.center
            existing = self._active.get(track_id)
            if existing is None:
                # Revive finished track if the backend reuses the id (rare).
                self._finished.pop(track_id, None)
                is_new = track_id not in self._created_ids
                if is_new:
                    self._created_ids.add(track_id)
                trajectory = [center]
                obj = TrackedObject(
                    track_id=track_id,
                    class_id=int(det.class_id),
                    class_name=str(det.class_name),
                    confidence=float(det.confidence),
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                    center=center,
                    first_seen_frame=frame_index,
                    last_seen_frame=frame_index,
                    age_frames=0,
                    frames_seen=1,
                    trajectory=trajectory,
                )
            else:
                trajectory = list(existing.trajectory)
                trajectory.append(center)
                if len(trajectory) > self.trajectory_max_points:
                    trajectory = trajectory[-self.trajectory_max_points :]
                # Keep current detection class (Ultralytics may flicker; document as current).
                obj = TrackedObject(
                    track_id=track_id,
                    class_id=int(det.class_id),
                    class_name=str(det.class_name),
                    confidence=float(det.confidence),
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                    center=center,
                    first_seen_frame=existing.first_seen_frame,
                    last_seen_frame=frame_index,
                    age_frames=frame_index - existing.first_seen_frame,
                    frames_seen=existing.frames_seen + 1,
                    trajectory=trajectory,
                )
            self._active[track_id] = obj
            self._class_of_track[track_id] = obj.class_name
            current.append(obj)

        # Finish tracks not seen on this processed frame.
        for track_id in list(self._active.keys()):
            if track_id in seen_ids:
                continue
            finished_obj = self._active.pop(track_id)
            self._finished[track_id] = TrackSummary(
                track_id=track_id,
                class_id=finished_obj.class_id,
                class_name=finished_obj.class_name,
                first_seen_frame=finished_obj.first_seen_frame,
                last_seen_frame=finished_obj.last_seen_frame,
                frames_seen=finished_obj.frames_seen,
                first_center=finished_obj.trajectory[0] if finished_obj.trajectory else finished_obj.center,
                last_center=finished_obj.center,
                finished=True,
            )

        return current

    def active_objects(self) -> list[TrackedObject]:
        return list(self._active.values())

    def finished_summaries(self) -> list[TrackSummary]:
        return list(self._finished.values())

    def all_track_summaries(self) -> list[TrackSummary]:
        """Active + finished summaries (active marked finished=False)."""
        items = list(self._finished.values())
        for obj in self._active.values():
            items.append(
                TrackSummary(
                    track_id=obj.track_id,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    first_seen_frame=obj.first_seen_frame,
                    last_seen_frame=obj.last_seen_frame,
                    frames_seen=obj.frames_seen,
                    first_center=obj.trajectory[0] if obj.trajectory else obj.center,
                    last_center=obj.center,
                    finished=False,
                )
            )
        return items

    def stats(self) -> TrackingStats:
        by_class: dict[str, int] = {}
        for track_id in self._created_ids:
            name = self._class_of_track.get(track_id, "unknown")
            by_class[name] = by_class.get(name, 0) + 1
        return TrackingStats(
            active_tracks=len(self._active),
            tracks_created=len(self._created_ids),
            tracks_finished=len(self._finished),
            by_class=by_class,
        )

    def summary_dict(self) -> dict[str, Any]:
        stats = self.stats()
        return {
            "enabled": True,
            "tracker": self.tracker_name,
            "tracks": stats.to_dict(),
            "track_summaries": [
                item.to_dict(fps=self.fps) for item in self.all_track_summaries()
            ],
        }
