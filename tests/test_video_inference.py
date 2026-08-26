from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from vision_trainer.inference.models import Detection
from vision_trainer.inference.video import (
    SUPPORTED_VIDEO_SUFFIXES,
    VIDEO_DEFAULT_CONF,
    VideoInferenceError,
    build_video_output_filename,
    is_supported_video_path,
    probe_video,
    run_video_inference,
)


def _write_synthetic_video(path: Path, *, frames: int = 4, size=(64, 48), fps: float = 10.0) -> Path:
    import cv2

    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (index * 40) % 255
        frame[:, :, 1] = 80
        frame[:, :, 2] = 160
        writer.write(frame)
    writer.release()
    assert path.is_file() and path.stat().st_size > 0
    return path


def test_is_supported_video_path() -> None:
    assert is_supported_video_path("demo.mp4")
    assert is_supported_video_path("demo.AVI")
    assert is_supported_video_path("clip.mov")
    assert is_supported_video_path("clip.mkv")
    assert not is_supported_video_path("photo.jpg")
    assert SUPPORTED_VIDEO_SUFFIXES == {".mp4", ".avi", ".mov", ".mkv"}


def test_build_video_output_filename() -> None:
    assert build_video_output_filename("cam_01.mov") == "cam_01_inference.mp4"
    assert build_video_output_filename("../../evil.mp4") == "evil_inference.mp4"


def test_probe_valid_video(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "sample.mp4", frames=5, size=(80, 60), fps=12.0)
    meta = probe_video(video)
    assert meta.width == 80
    assert meta.height == 60
    assert meta.fps is not None and meta.fps > 0
    assert meta.frame_count is None or meta.frame_count >= 1
    assert meta.display_name == "sample.mp4"


def test_probe_invalid_video(tmp_path: Path) -> None:
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not-a-video")
    with pytest.raises(VideoInferenceError):
        probe_video(bad)


def test_probe_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "clip.webm"
    path.write_bytes(b"x")
    with pytest.raises(VideoInferenceError, match="non supporté"):
        probe_video(path)


def test_run_video_inference_creates_output(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=3, size=(64, 48), fps=8.0)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    output = tmp_path / "out_inference.mp4"

    mock_model = MagicMock()
    mock_model.names = {0: "valve"}

    empty_boxes = MagicMock()
    empty_boxes.xyxy = []
    empty_boxes.conf = []
    empty_boxes.cls = []
    empty = MagicMock()
    empty.boxes = empty_boxes
    empty.names = mock_model.names
    empty.orig_shape = (48, 64)
    mock_model.predict.return_value = [empty]

    progress_events: list[int] = []

    result = run_video_inference(
        weights_path=weights,
        video_path=video,
        output_path=output,
        conf=VIDEO_DEFAULT_CONF,
        iou=0.45,
        device_choice="cpu",
        model_factory=lambda _: mock_model,
        progress_callback=lambda p: progress_events.append(p.frames_done),
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert result.frames_analyzed == 3
    assert result.total_detections == 0
    assert result.detections_by_class == {}
    assert result.source_width == 64
    assert result.source_height == 48
    assert result.output_fps == pytest.approx(8.0)
    assert result.counts_are_cumulative is True
    assert progress_events[-1] >= 3
    assert mock_model.predict.call_count == 3

    # Approximate resolution / fps preserved on the written file.
    meta = probe_video(output)
    assert meta.width == 64
    assert meta.height == 48
    assert meta.fps is None or abs(float(meta.fps) - 8.0) < 1.5


def test_run_video_inference_zero_detections_and_stats(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=2)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    output = tmp_path / "annotated.mp4"

    mock_model = MagicMock()
    mock_model.names = {0: "pump", 1: "valve"}

    boxes = MagicMock()
    boxes.xyxy = [[5.0, 5.0, 20.0, 20.0], [30.0, 10.0, 50.0, 40.0]]
    boxes.conf = [0.9, 0.8]
    boxes.cls = [0, 1]
    hit = MagicMock()
    hit.boxes = boxes
    hit.names = mock_model.names
    hit.orig_shape = (48, 64)

    empty_boxes = MagicMock()
    empty_boxes.xyxy = []
    empty_boxes.conf = []
    empty_boxes.cls = []
    miss = MagicMock()
    miss.boxes = empty_boxes
    miss.names = mock_model.names
    miss.orig_shape = (48, 64)

    mock_model.predict.side_effect = [[hit], [miss]]

    result = run_video_inference(
        weights_path=weights,
        video_path=video,
        output_path=output,
        conf=0.5,
        device_choice="cpu",
        model_factory=lambda _: mock_model,
    )

    assert result.frames_analyzed == 2
    assert result.total_detections == 2
    assert result.detections_by_class == {"pump": 1, "valve": 1}
    assert result.elapsed_seconds > 0
    assert result.mean_processing_fps > 0
    assert output.is_file()


def test_run_video_inference_missing_weights(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4")
    with pytest.raises(VideoInferenceError, match="Poids"):
        run_video_inference(
            weights_path=tmp_path / "missing.pt",
            video_path=video,
            output_path=tmp_path / "out.mp4",
            model_factory=lambda _: MagicMock(),
            device_choice="cpu",
        )


def test_run_video_inference_invalid_input(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(VideoInferenceError):
        run_video_inference(
            weights_path=weights,
            video_path=bad,
            output_path=tmp_path / "out.mp4",
            model_factory=lambda _: MagicMock(),
            device_choice="cpu",
        )


def test_draw_detections_still_usable_on_video_frame() -> None:
    from vision_trainer.inference.render import draw_detections

    image = Image.new("RGB", (40, 30), (10, 10, 10))
    annotated = draw_detections(
        image,
        [
            Detection(
                class_id=0,
                class_name="valve",
                confidence=0.91,
                x1=2,
                y1=2,
                x2=20,
                y2=18,
            )
        ],
    )
    assert annotated.size == (40, 30)
