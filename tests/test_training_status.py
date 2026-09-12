from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vision_trainer.training.status import (
    RunMetrics,
    RunStatus,
    compute_progress_percent,
    find_active_run,
    read_status,
    reconcile_run_status,
    write_status,
)
from vision_trainer.training.trainer import (
    EpochProgressTracker,
    TrainingRequest,
    _build_progress_callbacks,
    build_train_kwargs,
    execute_training_from_run_dir,
    prepare_training_run,
    start_training_subprocess,
)
from vision_trainer.yolo.models import DatasetInfo, SplitInfo


def _dataset(tmp_path: Path) -> DatasetInfo:
    root = tmp_path / "dataset"
    train_images = root / "train" / "images"
    train_images.mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    return DatasetInfo(
        root=root,
        yaml_path=root / "data.yaml",
        class_names={0: "pothole"},
        splits={
            "train": SplitInfo(
                name="train",
                images_dir=train_images,
                labels_dir=root / "train" / "labels",
                image_count=1,
            )
        },
    )


def test_compute_progress_percent() -> None:
    assert compute_progress_percent(0, 3) == 0.0
    assert compute_progress_percent(1, 3) == 33.3
    assert compute_progress_percent(2, 3) == 66.7
    assert compute_progress_percent(3, 3) == 100.0


def test_status_json_created_and_transitions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    status = RunStatus(
        run_id="run-1",
        state="created",
        model="YOLO11n",
        epochs_total=3,
        imgsz=640,
        batch=-1,
        device="cpu",
    )
    write_status(run_dir, status)
    loaded = read_status(run_dir)
    assert loaded is not None
    assert loaded.state == "created"
    assert (run_dir / "status.json").is_file()

    loaded.state = "running"
    loaded.epoch_current = 1
    loaded.progress_percent = compute_progress_percent(1, 3)
    loaded.pid = 123456
    write_status(run_dir, loaded)

    running = read_status(run_dir)
    assert running is not None
    assert running.state == "running"
    assert running.progress_percent == 33.3

    running.state = "completed"
    running.epoch_current = 3
    running.progress_percent = 100.0
    running.metrics = RunMetrics(precision=0.9, recall=0.8, map50=0.85, map50_95=0.6)
    running.best_model_path = str(run_dir / "weights" / "best.pt")
    write_status(run_dir, running)

    done = read_status(run_dir)
    assert done is not None
    assert done.state == "completed"
    assert done.metrics.map50 == 0.85


def test_status_failed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-fail"
    run_dir.mkdir()
    status = RunStatus(run_id="run-fail", state="running", model="YOLO11n", epochs_total=3)
    write_status(run_dir, status)
    status.state = "failed"
    status.error_message = "boom"
    write_status(run_dir, status)
    loaded = read_status(run_dir)
    assert loaded is not None
    assert loaded.state == "failed"
    assert loaded.error_message == "boom"


def test_interrupted_when_pid_dead(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run-dead"
    run_dir.mkdir()
    write_status(
        run_dir,
        RunStatus(
            run_id="run-dead",
            state="running",
            model="YOLO11n",
            epochs_total=3,
            pid=999999,
        ),
    )
    monkeypatch.setattr(
        "vision_trainer.training.status.is_pid_alive",
        lambda pid: False,
    )
    status = reconcile_run_status(run_dir)
    assert status is not None
    assert status.state == "interrupted"
    assert status.error_message is not None


def test_prepare_run_writes_artifacts_under_runs_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "artifacts" / "runs"
    prepared = prepare_training_run(
        TrainingRequest(dataset=_dataset(tmp_path), runs_root=runs_root, epochs=3)
    )
    assert prepared.run_dir.parent == runs_root.resolve()
    assert prepared.run_dir.name == prepared.run_id
    assert "runs/detect" not in str(prepared.run_dir)
    assert (prepared.run_dir / "status.json").is_file()
    assert (prepared.run_dir / "request.json").is_file()
    assert (prepared.run_dir / "data.resolved.yaml").is_file()
    status = read_status(prepared.run_dir)
    assert status is not None
    assert status.state == "created"


def test_build_train_kwargs_uses_absolute_project(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.resolved.yaml"
    data_yaml.write_text("names: [a]\n", encoding="utf-8")
    run_dir = tmp_path / "artifacts" / "runs" / "abc123"
    run_dir.mkdir(parents=True)
    kwargs = build_train_kwargs(
        data_yaml=data_yaml,
        run_dir=run_dir,
        epochs=3,
        imgsz=640,
        batch=-1,
        device="cpu",
    )
    assert Path(kwargs["project"]) == run_dir.parent.resolve()
    assert kwargs["name"] == "abc123"
    assert Path(kwargs["project"]).is_absolute()
    assert "runs/detect" not in kwargs["project"]


def test_start_subprocess_does_not_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "artifacts" / "runs"
    prepared = prepare_training_run(
        TrainingRequest(dataset=_dataset(tmp_path), runs_root=runs_root)
    )

    fake_proc = MagicMock()
    fake_proc.pid = 4242
    popen_calls: list[tuple] = []

    def _fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return fake_proc

    with patch("subprocess.Popen", side_effect=_fake_popen):
        pid = start_training_subprocess(prepared.run_dir, python_executable="python")

    assert pid == 4242
    assert len(popen_calls) == 1
    cmd = popen_calls[0][0][0]
    assert cmd[1:3] == ["-m", "vision_trainer.training.worker"]
    status = read_status(prepared.run_dir)
    assert status is not None
    assert status.state == "running"
    assert status.pid == 4242
    # Refresh/read path must not spawn another process
    before = len(popen_calls)
    _ = read_status(prepared.run_dir)
    assert len(popen_calls) == before


def test_epoch_progress_tracker_percentages_and_no_redundant_writes(tmp_path: Path) -> None:
    run_dir = tmp_path / "prog"
    run_dir.mkdir()
    write_status(
        run_dir,
        RunStatus(
            run_id="prog",
            state="running",
            model="YOLO11n",
            epochs_total=3,
            epoch_current=0,
            progress_percent=0.0,
            pid=1,
        ),
    )
    tracker = EpochProgressTracker(run_dir)

    assert tracker.apply_completed_epoch(1, 3) is True
    status = read_status(run_dir)
    assert status is not None
    assert status.epoch_current == 1
    assert status.progress_percent == 33.3

    assert tracker.apply_completed_epoch(1, 3) is False  # same epoch → no rewrite
    assert tracker.write_count == 1

    assert tracker.apply_completed_epoch(2, 3) is True
    status = read_status(run_dir)
    assert status is not None
    assert status.epoch_current == 2
    assert status.progress_percent == 66.7

    assert tracker.apply_completed_epoch(3, 3) is True
    status = read_status(run_dir)
    assert status is not None
    assert status.epoch_current == 3
    assert status.progress_percent == 100.0
    assert tracker.write_count == 3


@pytest.mark.parametrize("device_label", ["cpu", "0"])
def test_epoch_progress_callbacks_device_agnostic(tmp_path: Path, device_label: str) -> None:
    """Progress callbacks only read trainer.epoch — independent of CUDA/CPU device string."""
    run_dir = tmp_path / f"run-{device_label}"
    run_dir.mkdir()
    write_status(
        run_dir,
        RunStatus(
            run_id=run_dir.name,
            state="running",
            model="YOLO11n",
            epochs_total=3,
            epoch_current=0,
            device=device_label,
            progress_percent=0.0,
            pid=1,
        ),
    )
    tracker = EpochProgressTracker(run_dir)
    callbacks = _build_progress_callbacks(run_dir, tracker=tracker)
    trainer = MagicMock()
    trainer.epochs = 3
    trainer.device = device_label

    # Simulate Ultralytics loop: epoch_start → batches → epoch_end
    for epoch in range(3):
        trainer.epoch = epoch
        callbacks["on_train_epoch_start"](trainer)
        callbacks["on_train_batch_end"](trainer)  # same epoch → no extra write
        callbacks["on_train_batch_end"](trainer)
        callbacks["on_train_epoch_end"](trainer)
        status = read_status(run_dir)
        assert status is not None
        assert status.epoch_current == epoch + 1
        assert status.progress_percent == compute_progress_percent(epoch + 1, 3)

    # Extra batch/epoch callbacks must not rewrite the same completed epoch.
    writes_after_loop = tracker.write_count
    callbacks["on_train_epoch_end"](trainer)
    callbacks["on_train_batch_end"](trainer)
    assert tracker.write_count == writes_after_loop


def test_batch_end_fallback_detects_epoch_advance(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch-fallback"
    run_dir.mkdir()
    write_status(
        run_dir,
        RunStatus(
            run_id="batch-fallback",
            state="running",
            model="YOLO11n",
            epochs_total=3,
            epoch_current=0,
            progress_percent=0.0,
            pid=1,
        ),
    )
    tracker = EpochProgressTracker(run_dir)
    callbacks = _build_progress_callbacks(run_dir, tracker=tracker)
    trainer = MagicMock()
    trainer.epochs = 3

    trainer.epoch = 0
    callbacks["on_train_batch_end"](trainer)
    assert read_status(run_dir).epoch_current == 0

    # Epoch advanced without on_train_epoch_end (fallback).
    trainer.epoch = 1
    callbacks["on_train_batch_end"](trainer)
    status = read_status(run_dir)
    assert status is not None
    assert status.epoch_current == 1
    assert status.progress_percent == 33.3

    trainer.epoch = 1
    callbacks["on_train_batch_end"](trainer)
    assert tracker.write_count == 1


def test_execute_training_updates_progress_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "artifacts" / "runs"
    prepared = prepare_training_run(
        TrainingRequest(
            dataset=_dataset(tmp_path),
            runs_root=runs_root,
            epochs=3,
            imgsz=640,
            batch=8,
        )
    )

    callbacks: dict[str, list] = {}
    progress_snapshots: list[tuple[int, float]] = []

    mock_model = MagicMock()

    def add_callback(event: str, callback) -> None:
        callbacks.setdefault(event, []).append(callback)

    mock_model.add_callback.side_effect = add_callback

    def _train(**kwargs):
        assert Path(kwargs["project"]) == prepared.run_dir.parent
        assert kwargs["name"] == prepared.run_id
        trainer = MagicMock()
        trainer.epochs = 3
        trainer.callbacks = callbacks
        for callback in callbacks.get("on_train_start", []):
            callback(trainer)
        for epoch in range(3):
            trainer.epoch = epoch
            for callback in callbacks.get("on_train_epoch_start", []):
                callback(trainer)
            for callback in callbacks.get("on_train_batch_end", []):
                callback(trainer)
            for callback in callbacks.get("on_train_epoch_end", []):
                callback(trainer)
            status = read_status(prepared.run_dir)
            assert status is not None
            progress_snapshots.append((status.epoch_current, status.progress_percent))
        weights = prepared.run_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        trainer.metrics = {
            "metrics/precision(B)": 0.91,
            "metrics/recall(B)": 0.88,
            "metrics/mAP50(B)": 0.77,
            "metrics/mAP50-95(B)": 0.55,
        }
        trainer.best = weights / "best.pt"
        trainer.last = weights / "last.pt"
        for callback in callbacks.get("on_train_end", []):
            callback(trainer)

    mock_model.train.side_effect = _train

    final_status = execute_training_from_run_dir(
        prepared.run_dir,
        yolo_factory=lambda _: mock_model,
    )
    assert final_status.state == "completed"
    assert final_status.progress_percent == 100.0
    assert final_status.epoch_current == 3
    assert final_status.best_model_path is not None
    assert final_status.last_model_path is not None
    assert final_status.export_model_path is not None
    export_path = Path(final_status.export_model_path)
    assert export_path.is_file()
    assert export_path.name.endswith("_detect_best.pt")
    assert export_path.read_bytes() == (prepared.run_dir / "weights" / "best.pt").read_bytes()
    assert final_status.metrics.map50 == 0.77
    assert final_status.metrics.map50_95 == 0.55

    assert progress_snapshots == [
        (1, 33.3),
        (2, 66.7),
        (3, 100.0),
    ]

    # Intermediate progress was written during epochs (last epoch end = 100 already).
    raw = json.loads((prepared.run_dir / "status.json").read_text(encoding="utf-8"))
    assert raw["state"] == "completed"


def test_find_active_run_ignores_dead_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "dead"
    run_dir.mkdir()
    write_status(
        run_dir,
        RunStatus(run_id="dead", state="running", model="YOLO11n", epochs_total=1, pid=1),
    )
    monkeypatch.setattr("vision_trainer.training.status.is_pid_alive", lambda pid: False)
    assert find_active_run(tmp_path) is None


def test_training_log_display_height_is_compact() -> None:
    from vision_trainer.training.log_display import TRAINING_LOG_DISPLAY_HEIGHT_PX

    assert 250 <= TRAINING_LOG_DISPLAY_HEIGHT_PX <= 300
    assert TRAINING_LOG_DISPLAY_HEIGHT_PX == 280


def test_training_log_body_preserves_full_text() -> None:
    from vision_trainer.training.log_display import (
        DEFAULT_EMPTY_LOG_PLACEHOLDER,
        training_log_body,
    )

    long_log = "\n".join(f"line-{i}" for i in range(200))
    assert training_log_body(long_log) == long_log
    assert training_log_body(None) == DEFAULT_EMPTY_LOG_PLACEHOLDER
    assert training_log_body("") == DEFAULT_EMPTY_LOG_PLACEHOLDER


def test_escape_training_log_html_neutralizes_markup() -> None:
    from vision_trainer.training.log_display import escape_training_log_html

    raw = '<script>alert("x")</script> & <b>bold</b>'
    escaped = escape_training_log_html(raw)
    assert "<script>" not in escaped
    assert "<b>" not in escaped
    assert "&lt;script&gt;" in escaped
    assert "&amp;" in escaped


def test_build_training_log_html_auto_scrolls_and_escapes() -> None:
    from vision_trainer.training.log_display import (
        TRAINING_LOG_DISPLAY_HEIGHT_PX,
        build_training_log_html,
    )

    html_doc = build_training_log_html(
        'epoch 1\n<script>evil()</script>\nepoch 2',
        height_px=TRAINING_LOG_DISPLAY_HEIGHT_PX,
        auto_scroll=True,
    )
    assert 'id="training-log"' in html_doc
    assert f"height: {TRAINING_LOG_DISPLAY_HEIGHT_PX}px" in html_doc
    assert "overflow-y: auto" in html_doc
    assert "el.scrollTop = el.scrollHeight" in html_doc
    assert "<script>evil()</script>" not in html_doc
    assert "&lt;script&gt;evil()&lt;/script&gt;" in html_doc
