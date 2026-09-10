"""Generic frame-by-frame video processing engine (Streamlit-independent)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from vision_trainer.inference.models import VideoProgress
from vision_trainer.inference.video import (
    FALLBACK_FPS,
    VideoInferenceError,
    _build_progress,
    _cleanup_path,
    _open_video_writer,
    _transcode_to_browser_mp4,
    is_supported_video_path,
    probe_video,
)
from vision_trainer.io_utils import atomic_write_text
from vision_trainer.training.device import cuda_oom_user_message
from vision_trainer.video.color import bgr_to_pil, pil_to_bgr
from vision_trainer.video.models import FrameTimingsAgg, VideoJobSummary, VideoMode
from vision_trainer.video.processors import FrameProcessor, FrameProcessResult

ProgressCallback = Callable[[VideoProgress], None]
PreviewCallback = Callable[[int, Image.Image], None]
CancelCheck = Callable[[], bool]


def frame_timestamp_seconds(frame_index: int, fps: float | None) -> float | None:
    if fps is None or fps <= 0:
        return None
    return float(frame_index) / float(fps)


def process_video(
    *,
    video_path: Path | str,
    output_path: Path | str,
    processor: FrameProcessor,
    mode: VideoMode,
    frame_stride: int = 1,
    progress_callback: ProgressCallback | None = None,
    preview_callback: PreviewCallback | None = None,
    preview_every: int = 10,
    collect_frame_details: bool = False,
    cancel_check: CancelCheck | None = None,
) -> VideoJobSummary:
    """
    Process a video file frame-by-frame.

    - Models must already be loaded / cached inside ``processor``.
    - Skipped frames (stride > 1) are written as the **original** frame (no stale labels).
    - Output FPS matches source FPS (fallback if invalid) — independent of processing speed.
    - Audio is not preserved (OpenCV writer + ffmpeg ``-an``).
    """
    import cv2

    source = Path(video_path)
    destination = Path(output_path)
    stride = max(1, int(frame_stride))
    preview_every = max(1, int(preview_every))

    if not source.is_file():
        raise VideoInferenceError(f"Vidéo introuvable : {source}")
    if not is_supported_video_path(source):
        raise VideoInferenceError(
            f"Format vidéo non supporté : {source.suffix or '(sans extension)'}."
        )

    meta = probe_video(source)
    width = meta.width or 0
    height = meta.height or 0
    if width <= 0 or height <= 0:
        raise VideoInferenceError("Résolution vidéo invalide.")

    output_fps = meta.fps if meta.fps and meta.fps > 0 else FALLBACK_FPS
    frames_total = meta.frame_count

    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_output = destination.with_suffix(".raw.mp4")
    _cleanup_path(raw_output)
    _cleanup_path(destination)

    capture: Any = None
    writer: Any = None
    frames_read = 0
    frames_processed = 0
    frames_skipped = 0
    total_detections = 0
    total_classifications = 0
    total_segmentations = 0
    secondary_errors = 0
    warnings: list[str] = []
    timings = FrameTimingsAgg()
    frame_details: list[dict[str, Any]] = []
    started = time.perf_counter()
    cancelled = False

    try:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoInferenceError(f"Impossible d'ouvrir la vidéo : {source.name}")

        writer = _open_video_writer(raw_output, output_fps, width, height)

        while True:
            if cancel_check is not None and cancel_check():
                cancelled = True
                break

            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break

            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height))

            frame_index = frames_read
            frames_read += 1
            pil_frame = bgr_to_pil(frame_bgr)

            should_process = (frame_index % stride) == 0
            out_image: Image.Image

            if should_process:
                try:
                    frame_result = processor.process(pil_frame, frame_index=frame_index)
                except Exception as exc:  # noqa: BLE001
                    oom = cuda_oom_user_message(exc)
                    raise VideoInferenceError(
                        oom
                        or f"Erreur pendant l'inférence (frame {frame_index}) : {exc}"
                    ) from exc

                frames_processed += 1
                total_detections += frame_result.detections
                total_classifications += frame_result.classifications
                total_segmentations += frame_result.segmentations
                secondary_errors += frame_result.secondary_errors
                warnings.extend(frame_result.warnings)
                timings.add(
                    detection_ms=frame_result.detection_ms,
                    classification_ms=frame_result.classification_ms,
                    segmentation_ms=frame_result.segmentation_ms,
                    total_ms=frame_result.total_ms,
                )
                out_image = frame_result.annotated or pil_frame

                if collect_frame_details:
                    frame_details.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ms": (
                                None
                                if frame_timestamp_seconds(frame_index, meta.fps) is None
                                else round(
                                    frame_timestamp_seconds(frame_index, meta.fps) * 1000.0,
                                    3,
                                )
                            ),
                            "processed": True,
                            "objects": frame_result.structured,
                        }
                    )

                if preview_callback is not None and (
                    frames_processed == 1 or frames_processed % preview_every == 0
                ):
                    preview_callback(frame_index, out_image)
            else:
                frames_skipped += 1
                out_image = pil_frame
                if collect_frame_details:
                    frame_details.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ms": (
                                None
                                if frame_timestamp_seconds(frame_index, meta.fps) is None
                                else round(
                                    frame_timestamp_seconds(frame_index, meta.fps) * 1000.0,
                                    3,
                                )
                            ),
                            "processed": False,
                            "objects": {},
                        }
                    )

            # Writer expects original resolution.
            if out_image.size != (width, height):
                out_image = out_image.resize((width, height), Image.Resampling.BILINEAR)
            writer.write(pil_to_bgr(out_image))

            if progress_callback is not None:
                progress_callback(_build_progress(frames_read, frames_total, started))

        if frames_read == 0:
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
        oom = cuda_oom_user_message(exc)
        raise VideoInferenceError(oom or f"Échec du traitement vidéo : {exc}") from exc
    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                pass
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001
                pass
        _cleanup_path(raw_output)

    elapsed = max(time.perf_counter() - started, 1e-6)
    mean_fps = frames_processed / elapsed if frames_processed else 0.0
    mean_ms = (elapsed * 1000.0 / frames_processed) if frames_processed else None

    if cancelled:
        warnings.append("Traitement interrompu par l'utilisateur.")

    if progress_callback is not None:
        progress_callback(
            VideoProgress(
                frames_done=frames_read,
                frames_total=frames_read if frames_total is None else frames_total,
                percent=100.0,
                elapsed_seconds=elapsed,
                processing_fps=mean_fps if frames_processed else None,
                eta_seconds=0.0,
            )
        )

    return VideoJobSummary(
        mode=mode,
        output_path=str(destination.resolve()) if destination.is_file() else None,
        source_path=str(source.resolve()),
        source_width=width,
        source_height=height,
        source_fps=meta.fps,
        output_fps=float(output_fps),
        frames_total=frames_total,
        frames_read=frames_read,
        frames_processed=frames_processed,
        frames_skipped=frames_skipped,
        elapsed_seconds=elapsed,
        mean_processing_fps=mean_fps,
        mean_ms_per_frame=mean_ms,
        frame_stride=stride,
        total_detections=total_detections,
        total_classifications=total_classifications,
        total_segmentations=total_segmentations,
        secondary_errors=secondary_errors,
        warnings=warnings,
        timings=timings,
        frame_details=frame_details,
        audio_preserved=False,
    )


def process_image_with_processor(
    image: Image.Image,
    processor: FrameProcessor,
    *,
    frame_index: int = 0,
) -> FrameProcessResult:
    """Single-frame helper for camera capture / robot / API."""
    return processor.process(image.convert("RGB"), frame_index=frame_index)


def export_video_summary_json(
    summary: VideoJobSummary,
    destination: Path | str,
    *,
    include_frames: bool = False,
) -> Path:
    path = Path(destination)
    atomic_write_text(
        path,
        json.dumps(summary.to_dict(include_frames=include_frames), indent=2, ensure_ascii=False)
        + "\n",
    )
    return path


def video_summary_json_bytes(
    summary: VideoJobSummary,
    *,
    include_frames: bool = False,
) -> bytes:
    return (
        json.dumps(summary.to_dict(include_frames=include_frames), indent=2, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
