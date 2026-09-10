"""Tests for generic video / camera processing layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from vision_trainer.inference.models import Detection
from vision_trainer.inference.video import VideoInferenceError, probe_video
from vision_trainer.pipeline.models import (
    ClassificationRefinement,
    EnrichedDetection,
    PipelineConfig,
    PipelineResult,
    PipelineTimings,
)
from vision_trainer.video.camera import CameraError, capture_camera_frame, list_camera_indices
from vision_trainer.video.color import (
    bgr_to_pil,
    bgr_to_rgb_array,
    pil_to_bgr,
    rgb_to_bgr_array,
)
from vision_trainer.video.engine import (
    export_video_summary_json,
    frame_timestamp_seconds,
    process_image_with_processor,
    process_video,
    video_summary_json_bytes,
)
from vision_trainer.video.models import FrameTimingsAgg, VideoJobSummary
from vision_trainer.video.processors import (
    DetectFrameProcessor,
    FrameProcessResult,
    PipelineFrameProcessor,
    SegmentFrameProcessor,
)


def _write_synthetic_video(
    path: Path, *, frames: int = 6, size=(64, 48), fps: float = 10.0
) -> Path:
    import cv2

    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (index * 30) % 255
        frame[:, :, 1] = 90
        frame[:, :, 2] = 180
        writer.write(frame)
    writer.release()
    return path


class _CountingProcessor:
    def __init__(self) -> None:
        self.calls = 0
        self.indices: list[int] = []

    def process(self, frame: Image.Image, *, frame_index: int) -> FrameProcessResult:
        self.calls += 1
        self.indices.append(frame_index)
        # Draw a tiny mark so annotated ≠ original easily detectable
        annotated = frame.copy()
        annotated.putpixel((0, 0), (255, 0, 0))
        return FrameProcessResult(
            processed=True,
            annotated=annotated,
            structured={"frame_index": frame_index},
            detections=1,
            total_ms=1.0,
        )


def test_rgb_bgr_roundtrip() -> None:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]  # red
    bgr = rgb_to_bgr_array(rgb)
    assert list(bgr[0, 0]) == [0, 0, 255]
    back = bgr_to_rgb_array(bgr)
    assert list(back[0, 0]) == [255, 0, 0]
    pil = bgr_to_pil(bgr)
    assert pil.getpixel((0, 0)) == (255, 0, 0)
    assert list(pil_to_bgr(pil)[0, 0]) == [0, 0, 255]


def test_frame_timestamp() -> None:
    assert frame_timestamp_seconds(30, 30.0) == pytest.approx(1.0)
    assert frame_timestamp_seconds(0, 25.0) == 0.0
    assert frame_timestamp_seconds(10, None) is None
    assert frame_timestamp_seconds(10, 0.0) is None


def test_probe_fps_and_invalid(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "ok.mp4", frames=4, fps=12.0)
    meta = probe_video(video)
    assert meta.fps is not None and meta.fps > 0
    assert meta.width == 64
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(VideoInferenceError):
        probe_video(bad)


def test_process_video_stride_and_skip_original(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=6, fps=8.0)
    out = tmp_path / "out.mp4"
    processor = _CountingProcessor()
    summary = process_video(
        video_path=video,
        output_path=out,
        processor=processor,
        mode="detect",
        frame_stride=2,
        collect_frame_details=True,
    )
    assert summary.frames_read == 6
    assert summary.frames_processed == 3
    assert summary.frames_skipped == 3
    assert processor.calls == 3
    assert processor.indices == [0, 2, 4]
    assert summary.output_fps == pytest.approx(8.0)
    assert summary.audio_preserved is False
    assert out.is_file()
    # Detailed JSON contains processed flags
    assert any(not item["processed"] for item in summary.frame_details)
    assert any(item["processed"] for item in summary.frame_details)


def test_process_video_stride_1_all_frames(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=3, fps=10.0)
    processor = _CountingProcessor()
    summary = process_video(
        video_path=video,
        output_path=tmp_path / "out.mp4",
        processor=processor,
        mode="segment",
        frame_stride=1,
    )
    assert summary.frames_processed == 3
    assert summary.frames_skipped == 0
    assert processor.calls == 3


def test_process_video_progress_and_summary(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=3, fps=5.0)
    events: list[int] = []
    processor = _CountingProcessor()
    summary = process_video(
        video_path=video,
        output_path=tmp_path / "out.mp4",
        processor=processor,
        mode="detect",
        progress_callback=lambda p: events.append(p.frames_done),
    )
    assert events[-1] >= 3
    assert summary.mean_processing_fps >= 0
    assert summary.total_detections == 3
    payload = summary.to_dict(include_frames=False)
    assert "frames" not in payload
    assert payload["processing"]["mode"] == "detect"


def test_export_json_summary_and_detailed(tmp_path: Path) -> None:
    summary = VideoJobSummary(
        mode="pipeline",
        output_path=None,
        source_path="x.mp4",
        source_width=10,
        source_height=10,
        source_fps=25.0,
        output_fps=25.0,
        frames_total=2,
        frames_read=2,
        frames_processed=1,
        frames_skipped=1,
        elapsed_seconds=1.0,
        mean_processing_fps=1.0,
        mean_ms_per_frame=1000.0,
        frame_stride=2,
        frame_details=[{"frame_index": 0, "processed": True, "objects": {}}],
    )
    path = export_video_summary_json(summary, tmp_path / "s.json", include_frames=False)
    text = path.read_text(encoding="utf-8")
    assert "pipeline" in text
    assert "frame_index" not in text
    raw = video_summary_json_bytes(summary, include_frames=True)
    assert b"frame_index" in raw


def test_timings_agg_incremental() -> None:
    agg = FrameTimingsAgg()
    agg.add(detection_ms=10, classification_ms=20, segmentation_ms=30, total_ms=60)
    agg.add(detection_ms=30, classification_ms=10, segmentation_ms=10, total_ms=50)
    avg = agg.averages()
    assert avg["detection_ms"] == pytest.approx(20.0)
    assert avg["total_ms"] == pytest.approx(55.0)


def test_writer_failure(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=2)
    processor = _CountingProcessor()
    with patch(
        "vision_trainer.video.engine._open_video_writer",
        side_effect=VideoInferenceError("writer boom"),
    ):
        with pytest.raises(VideoInferenceError, match="writer"):
            process_video(
                video_path=video,
                output_path=tmp_path / "out.mp4",
                processor=processor,
                mode="detect",
            )


def test_exception_during_process_cleans_output(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=2)
    out = tmp_path / "out.mp4"

    class Boom:
        def process(self, frame, *, frame_index):
            raise RuntimeError("cuda exploded")

    with pytest.raises(VideoInferenceError):
        process_video(
            video_path=video,
            output_path=out,
            processor=Boom(),
            mode="detect",
        )
    assert not out.is_file()


def test_detect_processor_uses_cache(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    cache: dict = {}
    created = []

    def factory(path: str):
        created.append(path)
        model = MagicMock()
        model.names = {0: "a"}
        boxes = MagicMock()
        boxes.xyxy = []
        boxes.conf = []
        boxes.cls = []
        result = MagicMock()
        result.boxes = boxes
        result.names = model.names
        result.orig_shape = (48, 64)
        model.predict.return_value = [result]
        return model

    proc = DetectFrameProcessor(
        weights_path=weights,
        model_cache=cache,
        model_factory=factory,
    )
    frame = Image.new("RGB", (64, 48), color=(10, 20, 30))
    proc.process(frame, frame_index=0)
    proc.process(frame, frame_index=1)
    assert len(created) == 1
    assert len(cache) == 1


def test_segment_processor_calls_run_segmentation(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")

    fake_result = MagicMock()
    fake_result.instances = []
    with patch(
        "vision_trainer.video.processors.run_segmentation",
        return_value=fake_result,
    ) as mocked:
        with patch(
            "vision_trainer.video.processors.draw_segmentation_result",
            return_value=Image.new("RGB", (32, 32)),
        ):
            with patch(
                "vision_trainer.video.processors.get_cached_model",
                return_value=MagicMock(),
            ):
                proc = SegmentFrameProcessor(
                    weights_path=weights,
                    model_cache={},
                    model_factory=lambda _: MagicMock(),
                )
                # bypass __post_init__ cache by constructing after patches
                frame = Image.new("RGB", (32, 32))
                result = proc.process(frame, frame_index=0)
                assert result.processed is True
                assert mocked.call_count == 1


def test_pipeline_processor_reuses_run_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    run = runs / "det-1"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    (run / "config.json").write_text('{"task":"detect"}', encoding="utf-8")

    config = PipelineConfig(
        pipeline_id="p",
        name="P",
        detector_run_id="det-1",
        mappings={},
    )
    pipe_result = PipelineResult(
        items=[
            EnrichedDetection(
                detection=Detection(0, "A", 0.9, 1, 2, 3, 4),
                refined=False,
            )
        ],
        timings=PipelineTimings(detection_ms=5, total_ms=5),
    )
    with patch(
        "vision_trainer.video.processors.get_cached_model",
        return_value=MagicMock(),
    ):
        with patch(
            "vision_trainer.video.processors.run_pipeline",
            return_value=pipe_result,
        ) as mocked:
            with patch(
                "vision_trainer.video.processors.draw_pipeline_result",
                return_value=Image.new("RGB", (20, 20)),
            ):
                proc = PipelineFrameProcessor(
                    config=config,
                    model_cache={},
                    model_factory=lambda _: MagicMock(),
                )
                out = proc.process(Image.new("RGB", (20, 20)), frame_index=0)
                assert mocked.call_count == 1
                assert out.detections == 1


def test_pipeline_secondary_error_counted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    run = runs / "det-1"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    (run / "config.json").write_text('{"task":"detect"}', encoding="utf-8")
    config = PipelineConfig(
        pipeline_id="p", name="P", detector_run_id="det-1", mappings={}
    )
    pipe_result = PipelineResult(
        items=[
            EnrichedDetection(
                detection=Detection(0, "A", 0.9, 1, 2, 3, 4),
                refined=True,
                classification=ClassificationRefinement(
                    status="error", warning="cls down"
                ),
            )
        ],
        warnings=["cls down"],
        timings=PipelineTimings(total_ms=3),
    )
    with patch("vision_trainer.video.processors.get_cached_model", return_value=MagicMock()):
        with patch(
            "vision_trainer.video.processors.run_pipeline", return_value=pipe_result
        ):
            with patch(
                "vision_trainer.video.processors.draw_pipeline_result",
                return_value=Image.new("RGB", (10, 10)),
            ):
                proc = PipelineFrameProcessor(config=config, model_cache={})
                out = proc.process(Image.new("RGB", (10, 10)), frame_index=0)
                assert out.secondary_errors == 1
                assert out.warnings


def test_camera_index_validation() -> None:
    with pytest.raises(CameraError):
        capture_camera_frame(-1)


def test_list_camera_indices_mocked() -> None:
    fake = MagicMock()
    fake.isOpened.return_value = False
    with patch("cv2.VideoCapture", return_value=fake):
        infos = list_camera_indices(2)
    assert len(infos) == 2
    assert all(not info.opened for info in infos)
    fake.release.assert_called()


def test_process_image_with_processor() -> None:
    proc = _CountingProcessor()
    result = process_image_with_processor(Image.new("RGB", (8, 8)), proc)
    assert result.processed is True
    assert proc.calls == 1


def test_cancel_check_stops_early(tmp_path: Path) -> None:
    video = _write_synthetic_video(tmp_path / "in.mp4", frames=10, fps=10.0)
    processor = _CountingProcessor()
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    summary = process_video(
        video_path=video,
        output_path=tmp_path / "out.mp4",
        processor=processor,
        mode="detect",
        cancel_check=cancel,
    )
    assert summary.frames_read < 10
    assert any("interrompu" in w.lower() for w in summary.warnings)
