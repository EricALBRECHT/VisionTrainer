"""Tracker backends (Ultralytics ByteTrack / BoT-SORT + injectable fake)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from vision_trainer.tracking.geometry import bbox_center
from vision_trainer.tracking.models import TrackerName

TRACKER_YAML: dict[TrackerName, str] = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


@dataclass(frozen=True)
class RawTrackDetection:
    """One detection with a tracker-assigned id for a single frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.x1, self.y1, self.x2, self.y2)


class TrackerBackend(Protocol):
    def track_frame(
        self,
        image: Image.Image,
        *,
        conf: float,
        iou: float,
        device: str | int | None = None,
    ) -> list[RawTrackDetection]:
        ...

    def reset(self) -> None:
        ...


class TrackingBackendError(RuntimeError):
    """Raised when the tracker backend fails."""


class UltralyticsTrackerBackend:
    """
    Frame-by-frame Ultralytics ``model.track(..., persist=True)``.

    Detection + association run together → timings reported as detection_tracking_ms.
    """

    def __init__(
        self,
        model: Any,
        *,
        tracker: TrackerName = "bytetrack",
        class_names: dict[int, str] | None = None,
    ) -> None:
        self.model = model
        self.tracker = tracker
        self.class_names = dict(class_names or {})
        self._tracker_arg = TRACKER_YAML.get(tracker, "bytetrack.yaml")

    def reset(self) -> None:
        """Clear Ultralytics persistent trackers on this model instance."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None:
            if hasattr(predictor, "trackers"):
                predictor.trackers = []
            if hasattr(predictor, "vid_path"):
                try:
                    predictor.vid_path = [None] * len(getattr(predictor, "vid_path", []) or [None])
                except Exception:  # noqa: BLE001
                    pass
        # Also clear common alternate attributes across Ultralytics versions.
        if hasattr(self.model, "trackers"):
            try:
                self.model.trackers = []
            except Exception:  # noqa: BLE001
                pass

    def track_frame(
        self,
        image: Image.Image,
        *,
        conf: float,
        iou: float,
        device: str | int | None = None,
    ) -> list[RawTrackDetection]:
        rgb = np.asarray(image.convert("RGB"))
        kwargs: dict[str, Any] = {
            "source": rgb,
            "persist": True,
            "tracker": self._tracker_arg,
            "conf": float(conf),
            "iou": float(iou),
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device
        try:
            results = self.model.track(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise TrackingBackendError(f"Échec du tracking Ultralytics : {exc}") from exc

        if not results:
            return []
        result = results[0] if isinstance(results, (list, tuple)) else results
        names = dict(self.class_names)
        result_names = getattr(result, "names", None)
        if isinstance(result_names, dict):
            for key, value in result_names.items():
                try:
                    names[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _as_list2d(getattr(boxes, "xyxy", None))
        confs = _as_list1d(getattr(boxes, "conf", None))
        clss = _as_list1d(getattr(boxes, "cls", None))
        ids_raw = getattr(boxes, "id", None)
        ids = _as_list1d(ids_raw) if ids_raw is not None else []

        out: list[RawTrackDetection] = []
        count = len(xyxy)
        for index in range(count):
            if index >= len(ids) or ids[index] is None:
                # Ultralytics may omit ids on the first frame in edge cases.
                continue
            try:
                track_id = int(ids[index])
            except (TypeError, ValueError):
                continue
            class_id = int(clss[index]) if index < len(clss) else 0
            confidence = float(confs[index]) if index < len(confs) else 0.0
            box = xyxy[index]
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            out.append(
                RawTrackDetection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=names.get(class_id, str(class_id)),
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return out


class FakeTrackerBackend:
    """Deterministic backend for unit tests (no YOLO)."""

    def __init__(
        self,
        provider: Callable[[Image.Image], list[RawTrackDetection]] | None = None,
    ) -> None:
        self._provider = provider or (lambda _img: [])
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def track_frame(
        self,
        image: Image.Image,
        *,
        conf: float,
        iou: float,
        device: str | int | None = None,
    ) -> list[RawTrackDetection]:
        _ = (conf, iou, device)
        return list(self._provider(image))


def resolve_project_tracker_yaml(tracker: TrackerName) -> str:
    """Prefer bundled Ultralytics name; optional project override if present."""
    project_cfg = (
        Path(__file__).resolve().parents[3] / "config" / "trackers" / f"{tracker}.yaml"
    )
    if project_cfg.is_file():
        return str(project_cfg)
    return TRACKER_YAML.get(tracker, "bytetrack.yaml")


def _as_list2d(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    rows: list[list[float]] = []
    for row in value:
        if isinstance(row, (list, tuple)):
            rows.append([float(x) for x in row])
        else:
            rows.append([float(row)])
    return rows


def _as_list1d(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return [value]
    flat: list[Any] = []
    for item in value:
        if isinstance(item, list) and len(item) == 1:
            flat.append(item[0])
        else:
            flat.append(item)
    return flat
