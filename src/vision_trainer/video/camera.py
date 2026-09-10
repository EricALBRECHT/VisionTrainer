"""Camera / webcam helpers with explicit environment limitations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from vision_trainer.video.color import bgr_to_pil


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened or read."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    opened: bool
    width: int | None = None
    height: int | None = None
    message: str | None = None


WSL_DOCKER_CAMERA_NOTE = (
    "Sous Docker / WSL2, `cv2.VideoCapture(index)` accède à une caméra du "
    "**serveur Linux**, pas automatiquement à la webcam Windows du navigateur. "
    "Si aucune caméra n'est listée, utilisez plutôt un fichier vidéo, ou lancez "
    "VisionTrainer nativement sous Linux avec périphérique `/dev/video*` monté."
)


def probe_camera(index: int, *, warmup_frames: int = 1) -> CameraInfo:
    """Try to open a camera index briefly and report capabilities."""
    import cv2

    if index < 0:
        raise CameraError("L'index caméra doit être ≥ 0.")

    capture = cv2.VideoCapture(int(index))
    try:
        if not capture.isOpened():
            return CameraInfo(
                index=index,
                opened=False,
                message=f"Caméra {index} inaccessible.",
            )
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        for _ in range(max(0, warmup_frames)):
            capture.read()
        ok, frame = capture.read()
        if not ok or frame is None:
            return CameraInfo(
                index=index,
                opened=False,
                width=width,
                height=height,
                message=f"Caméra {index} ouverte mais aucune frame lue.",
            )
        if width is None:
            width = int(frame.shape[1])
        if height is None:
            height = int(frame.shape[0])
        return CameraInfo(index=index, opened=True, width=width, height=height)
    finally:
        capture.release()


def list_camera_indices(max_index: int = 5) -> list[CameraInfo]:
    """Probe camera indices ``0..max_index-1`` (does not require a real device in tests)."""
    results: list[CameraInfo] = []
    for index in range(max(0, int(max_index))):
        try:
            results.append(probe_camera(index))
        except Exception as exc:  # noqa: BLE001
            results.append(
                CameraInfo(index=index, opened=False, message=str(exc))
            )
    return results


def capture_camera_frame(index: int = 0) -> Image.Image:
    """Capture a single RGB frame from a camera index."""
    import cv2

    if index < 0:
        raise CameraError("L'index caméra doit être ≥ 0.")
    capture = cv2.VideoCapture(int(index))
    try:
        if not capture.isOpened():
            raise CameraError(
                f"Impossible d'ouvrir la caméra {index}. {WSL_DOCKER_CAMERA_NOTE}"
            )
        # Discard one warm-up frame when possible.
        capture.read()
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraError(f"Aucune frame lue depuis la caméra {index}.")
        return bgr_to_pil(frame)
    finally:
        capture.release()


def open_camera_capture(index: int = 0) -> Any:
    """
    Open a ``cv2.VideoCapture`` for continuous reading.

    Caller **must** release it. Prefer higher-level helpers when possible.
    """
    import cv2

    if index < 0:
        raise CameraError("L'index caméra doit être ≥ 0.")
    capture = cv2.VideoCapture(int(index))
    if not capture.isOpened():
        capture.release()
        raise CameraError(
            f"Impossible d'ouvrir la caméra {index}. {WSL_DOCKER_CAMERA_NOTE}"
        )
    return capture
