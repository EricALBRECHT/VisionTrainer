"""Unit tests for Tracking V1 (mocked backends — no YOLO / webcam)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vision_trainer.inference.models import Detection
from vision_trainer.pipeline.models import ClassificationRefinement
from vision_trainer.tracking.backend import FakeTrackerBackend, RawTrackDetection
from vision_trainer.tracking.cache import ClassificationTrackCache
from vision_trainer.tracking.geometry import bbox_center, frame_stride_tracking_warning
from vision_trainer.tracking.models import TrackingConfig
from vision_trainer.tracking.render import (
    draw_track_id_overlays,
    draw_tracked_objects,
    format_track_label,
)
from vision_trainer.tracking.session import TrackingSession
from vision_trainer.video.models import VideoJobSummary


def _raw(
    track_id: int,
    class_id: int = 0,
    class_name: str = "person",
    conf: float = 0.9,
    box=(10.0, 10.0, 50.0, 50.0),
) -> RawTrackDetection:
    x1, y1, x2, y2 = box
    return RawTrackDetection(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _session_with_sequence(frames: list[list[RawTrackDetection]], **kwargs) -> TrackingSession:
    state = {"i": 0}

    def provider(_img: Image.Image) -> list[RawTrackDetection]:
        idx = state["i"]
        state["i"] += 1
        if idx >= len(frames):
            return []
        return frames[idx]

    return TrackingSession.from_fake(provider, **kwargs)


def test_bbox_center() -> None:
    assert bbox_center(0, 0, 10, 20) == (5.0, 10.0)


def test_frame_stride_warning() -> None:
    assert frame_stride_tracking_warning(1) is None
    assert "stabilité" in (frame_stride_tracking_warning(5) or "")


def test_tracking_session_create_reset_and_persist_id() -> None:
    session = _session_with_sequence(
        [
            [_raw(1)],
            [_raw(1, box=(12, 12, 52, 52))],
            [_raw(1, box=(14, 14, 54, 54))],
        ]
    )
    img = Image.new("RGB", (64, 64), "black")
    r0 = session.process(img, frame_index=0)
    r1 = session.process(img, frame_index=1)
    r2 = session.process(img, frame_index=2)
    assert [o.track_id for o in r0.objects] == [1]
    assert [o.track_id for o in r1.objects] == [1]
    assert [o.track_id for o in r2.objects] == [1]
    assert r2.objects[0].frames_seen == 3
    assert r2.objects[0].age_frames == 2
    session.reset()
    assert session.stats().tracks_created == 0
    assert session.active_objects() == []


def test_multiple_tracks_and_classes() -> None:
    session = _session_with_sequence(
        [
            [_raw(1, 0, "person"), _raw(2, 1, "car", box=(60, 10, 90, 40))],
            [_raw(1, 0, "person"), _raw(2, 1, "car", box=(62, 12, 92, 42))],
        ]
    )
    img = Image.new("RGB", (100, 100))
    session.process(img, frame_index=0)
    session.process(img, frame_index=1)
    stats = session.stats()
    assert stats.tracks_created == 2
    assert stats.by_class == {"person": 1, "car": 1}


def test_track_disappears_and_finishes() -> None:
    session = _session_with_sequence(
        [
            [_raw(7)],
            [],
        ]
    )
    img = Image.new("RGB", (64, 64))
    session.process(img, frame_index=0)
    assert session.stats().active_tracks == 1
    session.process(img, frame_index=1)
    assert session.stats().active_tracks == 0
    assert session.stats().tracks_finished == 1
    finished = session.finished_summaries()
    assert finished[0].track_id == 7
    assert finished[0].finished is True


def test_trajectory_limit() -> None:
    frames = [[_raw(1, box=(i, i, i + 10, i + 10))] for i in range(12)]
    session = _session_with_sequence(frames, trajectory_max_points=5)
    img = Image.new("RGB", (128, 128))
    last = None
    for index in range(12):
        last = session.process(img, frame_index=index)
    assert last is not None
    assert len(last.objects[0].trajectory) == 5


def test_duration_with_valid_and_invalid_fps() -> None:
    session = _session_with_sequence(
        [
            [_raw(1)],
            [_raw(1)],
            [],
        ],
        fps=10.0,
    )
    img = Image.new("RGB", (32, 32))
    session.process(img, frame_index=0)
    session.process(img, frame_index=10)
    session.process(img, frame_index=11)
    summary = session.finished_summaries()[0]
    assert summary.duration_seconds(10.0) == pytest.approx(1.0)
    assert summary.duration_seconds(None) is None
    assert summary.duration_seconds(0.0) is None


def test_summary_and_json_track_fields() -> None:
    session = _session_with_sequence([[_raw(3, 0, "dog")]], fps=25.0)
    img = Image.new("RGB", (40, 40))
    result = session.process(img, frame_index=0)
    payload = result.objects[0].to_dict()
    assert payload["track_id"] == 3
    assert payload["center"] == [30.0, 30.0]
    summary = session.summary_dict()
    assert summary["enabled"] is True
    assert summary["tracker"] == "bytetrack"
    assert summary["tracks"]["total"] == 1
    assert summary["tracks"]["by_class"]["dog"] == 1


def test_video_job_summary_includes_tracking() -> None:
    summary = VideoJobSummary(
        mode="detect",
        output_path=None,
        source_path=None,
        source_width=10,
        source_height=10,
        source_fps=30.0,
        output_fps=30.0,
        frames_total=1,
        frames_read=1,
        frames_processed=1,
        frames_skipped=0,
        elapsed_seconds=1.0,
        mean_processing_fps=1.0,
        mean_ms_per_frame=1000.0,
        frame_stride=1,
        tracking={"enabled": True, "tracker": "bytetrack", "tracks": {"total": 2}},
    )
    data = summary.to_dict()
    assert data["tracking"]["enabled"] is True
    assert data["tracking"]["tracks"]["total"] == 2


def test_renderer_id_and_trajectory_flags() -> None:
    session = _session_with_sequence(
        [
            [_raw(1)],
            [_raw(1, box=(20, 20, 60, 60))],
        ]
    )
    img = Image.new("RGB", (80, 80), "white")
    session.process(img, frame_index=0)
    result = session.process(img, frame_index=1)
    obj = result.objects[0]
    assert "ID 1" in format_track_label(obj)
    with_traj = draw_tracked_objects(img, [obj], show_trajectories=True)
    without = draw_tracked_objects(img, [obj], show_trajectories=False)
    assert with_traj.size == without.size == img.size
    overlay = draw_track_id_overlays(img, [obj], show_trajectories=True)
    assert overlay.size == img.size


def test_classification_cache_policy() -> None:
    cache = ClassificationTrackCache(
        reclassify_every_n_frames=30,
        uncertain_reclassify_every_n_frames=5,
    )
    ok = ClassificationRefinement(status="ok", class_name="Golden", confidence=0.9)
    unknown = ClassificationRefinement(status="unknown", reason="low conf")
    uncertain = ClassificationRefinement(status="uncertain", reason="margin")

    assert cache.should_classify(12, 0) is True
    cache.put(12, 0, ok, status="ok")
    assert cache.should_classify(12, 10) is False
    assert cache.should_classify(12, 30) is True
    assert cache.get(12) is ok

    cache.put(12, 30, unknown, status="unknown")
    assert cache.should_classify(12, 34) is False
    assert cache.should_classify(12, 35) is True

    cache.put(99, 0, uncertain, status="uncertain")
    assert cache.should_classify(99, 4) is False
    assert cache.should_classify(99, 5) is True

    # Separate per track_id
    cache.put(1, 0, ok, status="ok")
    cache.put(2, 0, unknown, status="unknown")
    assert cache.get(1) is ok
    assert cache.get(2) is unknown

    cache.reset()
    assert cache.get(1) is None
    assert cache.should_classify(1, 0) is True


def test_ingest_raw_without_backend_call() -> None:
    session = TrackingSession(FakeTrackerBackend())
    result = session.ingest_raw([_raw(5)], frame_index=3, detection_tracking_ms=12.5)
    assert result.detection_tracking_ms == 12.5
    assert result.objects[0].track_id == 5


def test_tracking_config_dict() -> None:
    cfg = TrackingConfig(enabled=True, show_trajectories=True)
    data = cfg.to_dict()
    assert data["enabled"] is True
    assert data["tracker"] == "bytetrack"


def test_tracking_disabled_summary_absent_by_default() -> None:
    summary = VideoJobSummary(
        mode="detect",
        output_path=None,
        source_path=None,
        source_width=1,
        source_height=1,
        source_fps=None,
        output_fps=1.0,
        frames_total=None,
        frames_read=0,
        frames_processed=0,
        frames_skipped=0,
        elapsed_seconds=0.0,
        mean_processing_fps=0.0,
        mean_ms_per_frame=None,
        frame_stride=1,
    )
    assert "tracking" not in summary.to_dict()


def test_new_session_is_independent() -> None:
    s1 = _session_with_sequence([[_raw(1)]])
    s2 = _session_with_sequence([[_raw(2)]])
    img = Image.new("RGB", (20, 20))
    s1.process(img, frame_index=0)
    s2.process(img, frame_index=0)
    assert s1.stats().tracks_created == 1
    assert s2.stats().tracks_created == 1
    assert s1.active_objects()[0].track_id == 1
    assert s2.active_objects()[0].track_id == 2


def test_detailed_frame_objects_include_track_id() -> None:
    session = _session_with_sequence([[_raw(9)]])
    img = Image.new("RGB", (30, 30))
    frame = session.process(img, frame_index=0)
    detail = {"frame_index": 0, "objects": {"objects": [o.to_dict() for o in frame.objects]}}
    assert detail["objects"]["objects"][0]["track_id"] == 9


def test_fake_backend_reset_and_model_cache_unchanged() -> None:
    calls = {"n": 0}
    cache: dict = {}

    def factory(path: str):
        calls["n"] += 1
        return object()

    # Simulate get_cached_model behaviour used by tracking factory.
    from vision_trainer.pipeline.engine import get_cached_model

    class _Dummy:
        names = {0: "a"}

        def track(self, **kwargs):
            return []

    def yolo_factory(path: str):
        calls["n"] += 1
        return _Dummy()

    m1 = get_cached_model("w.pt", cache=cache, model_factory=yolo_factory)
    m2 = get_cached_model("w.pt", cache=cache, model_factory=yolo_factory)
    assert m1 is m2
    assert calls["n"] == 1

    backend = FakeTrackerBackend(lambda _i: [_raw(1)])
    session = TrackingSession(backend)
    img = Image.new("RGB", (16, 16))
    session.process(img, frame_index=0)
    session.reset()
    assert backend.reset_count == 1


def test_run_pipeline_image_api_still_importable() -> None:
    from vision_trainer.pipeline.engine import run_pipeline

    assert callable(run_pipeline)


def test_class_change_keeps_track_id() -> None:
    """V1: keep current detection class; identity is track_id."""
    session = _session_with_sequence(
        [
            [_raw(1, 0, "person")],
            [_raw(1, 0, "person")],
            [_raw(1, 1, "dog")],  # rare flicker
        ]
    )
    img = Image.new("RGB", (40, 40))
    session.process(img, frame_index=0)
    session.process(img, frame_index=1)
    last = session.process(img, frame_index=2)
    assert last.objects[0].track_id == 1
    assert last.objects[0].class_name == "dog"
