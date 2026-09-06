from __future__ import annotations

import subprocess
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from vision_trainer.inference.models import VideoInferenceResult, VideoMetadata, VideoProgress
from vision_trainer.inference.predictor import (
    DEFAULT_IOU,
    InferenceError,
    build_predict_kwargs,
    extract_class_names,
    normalize_ultralytics_results,
)
from vision_trainer.inference.render import AnnotationScale, compute_annotation_style, draw_detections
from vision_trainer.training.device import DeviceChoice, DeviceError, cuda_oom_user_message, resolve_device

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
VIDEO_DEFAULT_CONF = 0.50
FALLBACK_FPS = 25.0

ProgressCallback = Callable[[VideoProgress], None]


class VideoInferenceError(InferenceError):
    """Raised when video probing or inference cannot complete."""


def is_supported_video_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_SUFFIXES


def build_video_output_filename(original_name: str) -> str:
    stem = Path(original_name).stem or "video"
    return f"{stem}_inference.mp4"


def probe_video(path: Path | str) -> VideoMetadata:
    """Open a video briefly and read metadata. Does not keep the capture open."""
    import cv2

    video_path = Path(path)
    if not video_path.is_file():
        raise VideoInferenceError(f"Vidéo introuvable : {video_path}")
    if not is_supported_video_path(video_path):
        raise VideoInferenceError(
            f"Format vidéo non supporté : {video_path.suffix or '(sans extension)'}. "
            f"Formats acceptés : {', '.join(sorted(SUPPORTED_VIDEO_SUFFIXES))}."
        )

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoInferenceError(
                f"Impossible d'ouvrir la vidéo (fichier corrompu ou codec non supporté) : "
                f"{video_path.name}"
            )

        width = _positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = _positive_float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = _positive_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sanity check: try to read one frame.
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoInferenceError(
                f"Vidéo illisible ou vide : {video_path.name}"
            )
        if width is None:
            width = int(frame.shape[1])
        if height is None:
            height = int(frame.shape[0])

        duration = None
        if fps is not None and frame_count is not None and fps > 0:
            duration = frame_count / fps

        return VideoMetadata(
            path=str(video_path.resolve()),
            display_name=video_path.name,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )
    finally:
        capture.release()


def run_video_inference(
    *,
    weights_path: Path | str,
    video_path: Path | str,
    output_path: Path | str,
    conf: float = VIDEO_DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    device_choice: DeviceChoice | str = "auto",
    model_factory: Callable[[str], Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    annotation_scale: AnnotationScale | str = "auto",
) -> VideoInferenceResult:
    """
    Run detection on each frame and write an annotated MP4.

    Frames are processed sequentially (not held in RAM). Detection counts are
    cumulative across frames — no object tracking / unique IDs.
    """
    import cv2
    import numpy as np

    weights = Path(weights_path)
    source = Path(video_path)
    destination = Path(output_path)

    if not weights.is_file():
        raise VideoInferenceError(f"Poids introuvables : {weights}")
    if not source.is_file():
        raise VideoInferenceError(f"Vidéo introuvable : {source}")
    if not is_supported_video_path(source):
        raise VideoInferenceError(
            f"Format vidéo non supporté : {source.suffix or '(sans extension)'}."
        )

    try:
        device = resolve_device(device_choice)
    except DeviceError as exc:
        raise VideoInferenceError(str(exc)) from exc

    try:
        predict_kwargs = build_predict_kwargs(conf=conf, iou=iou, device=device)
    except InferenceError as exc:
        raise VideoInferenceError(str(exc)) from exc

    meta = probe_video(source)
    width = meta.width or 0
    height = meta.height or 0
    if width <= 0 or height <= 0:
        raise VideoInferenceError("Résolution vidéo invalide.")

    output_fps = meta.fps if meta.fps and meta.fps > 0 else FALLBACK_FPS
    frames_total = meta.frame_count

    factory = model_factory or _default_yolo_factory
    try:
        model = factory(str(weights))
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VideoInferenceError(f"Impossible de charger le modèle : {exc}") from exc

    class_names = extract_class_names(model)
    destination.parent.mkdir(parents=True, exist_ok=True)

    raw_output = destination.with_suffix(".raw.mp4")
    if raw_output.exists():
        raw_output.unlink()
    if destination.exists():
        destination.unlink()

    capture: Any = None
    writer: Any = None
    frames_done = 0
    total_detections = 0
    class_counts: Counter[str] = Counter()
    started = time.perf_counter()

    try:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoInferenceError(
                f"Impossible d'ouvrir la vidéo : {source.name}"
            )

        writer = _open_video_writer(raw_output, output_fps, width, height)
        annotation_style = compute_annotation_style(width, height, annotation_scale)

        while True:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break

            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height))

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)

            try:
                results = model.predict(source=np.asarray(pil_frame), **predict_kwargs)
            except Exception as exc:  # noqa: BLE001
                oom = cuda_oom_user_message(exc)
                raise VideoInferenceError(
                    oom
                    or f"Erreur pendant l'inférence (frame {frames_done + 1}) : {exc}"
                ) from exc

            normalized = normalize_ultralytics_results(results, class_names=class_names)
            total_detections += len(normalized.detections)
            for detection in normalized.detections:
                class_counts[detection.class_name] += 1

            annotated = draw_detections(
                pil_frame,
                normalized.detections,
                scale=annotation_scale,
                style=annotation_style,
            )
            out_bgr = cv2.cvtColor(np.asarray(annotated.convert("RGB")), cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)
            frames_done += 1

            if progress_callback is not None:
                progress_callback(
                    _build_progress(frames_done, frames_total, started)
                )

        if frames_done == 0:
            raise VideoInferenceError("Aucune frame n'a pu être lue dans la vidéo.")

        writer.release()
        writer = None
        capture.release()
        capture = None

        _transcode_to_browser_mp4(raw_output, destination)
    except VideoInferenceError:
        _cleanup_path(raw_output)
        _cleanup_path(destination)
        raise
    except Exception as exc:  # noqa: BLE001
        _cleanup_path(raw_output)
        _cleanup_path(destination)
        raise VideoInferenceError(f"Échec du traitement vidéo : {exc}") from exc
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        _cleanup_path(raw_output)

    elapsed = max(time.perf_counter() - started, 1e-6)
    mean_fps = frames_done / elapsed

    if progress_callback is not None:
        progress_callback(
            VideoProgress(
                frames_done=frames_done,
                frames_total=frames_done if frames_total is None else frames_total,
                percent=100.0,
                elapsed_seconds=elapsed,
                processing_fps=mean_fps,
                eta_seconds=0.0,
            )
        )

    return VideoInferenceResult(
        output_path=str(destination.resolve()),
        frames_analyzed=frames_done,
        elapsed_seconds=elapsed,
        mean_processing_fps=mean_fps,
        total_detections=total_detections,
        detections_by_class=dict(sorted(class_counts.items())),
        source_width=width,
        source_height=height,
        source_fps=meta.fps,
        output_fps=output_fps,
        counts_are_cumulative=True,
    )


def _build_progress(
    frames_done: int,
    frames_total: int | None,
    started: float,
) -> VideoProgress:
    elapsed = max(time.perf_counter() - started, 1e-6)
    processing_fps = frames_done / elapsed if frames_done else None
    percent = None
    eta = None
    if frames_total and frames_total > 0:
        percent = min(100.0, 100.0 * frames_done / frames_total)
        remaining = max(0, frames_total - frames_done)
        if processing_fps and processing_fps > 0:
            eta = remaining / processing_fps
    return VideoProgress(
        frames_done=frames_done,
        frames_total=frames_total,
        percent=percent,
        elapsed_seconds=elapsed,
        processing_fps=processing_fps,
        eta_seconds=eta,
    )


def _open_video_writer(path: Path, fps: float, width: int, height: int) -> Any:
    import cv2

    # mp4v is widely available with OpenCV wheels; remuxed later to H.264 for browsers.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        writer.release()
        raise VideoInferenceError(
            f"Impossible de créer la vidéo de sortie : {path.name}"
        )
    return writer


def _transcode_to_browser_mp4(source: Path, destination: Path) -> None:
    """Convert OpenCV mp4v output to H.264 / yuv420p for Streamlit browsers."""
    if not source.is_file() or source.stat().st_size <= 0:
        raise VideoInferenceError("La vidéo temporaire de sortie est vide ou absente.")

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise VideoInferenceError(
            "imageio-ffmpeg est requis pour produire une vidéo lisible dans le navigateur. "
            "Installez les dépendances du projet."
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Even dimensions required by yuv420p.
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise VideoInferenceError(
            f"Impossible de lancer ffmpeg pour finaliser la vidéo : {exc}"
        ) from exc

    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size <= 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise VideoInferenceError(
            "Échec de la conversion vers MP4 compatible navigateur."
            + (f" Détail : {detail[-500:]}" if detail else "")
        )


def _default_yolo_factory(weights_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise VideoInferenceError(
            "Ultralytics n'est pas disponible. Installez le projet avec ses dépendances."
        ) from exc
    try:
        return YOLO(weights_path)
    except Exception as exc:  # noqa: BLE001
        raise VideoInferenceError(f"Modèle illisible ou invalide : {exc}") from exc


def _positive_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:  # NaN or non-positive
        return None
    return number


def _cleanup_path(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
