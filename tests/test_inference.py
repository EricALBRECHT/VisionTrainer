from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.inference.models import Detection
from vision_trainer.inference.predictor import (
    InferenceError,
    build_predict_kwargs,
    load_image_rgb,
    normalize_ultralytics_results,
    run_inference,
)
from vision_trainer.inference.render import (
    annotated_image_to_jpeg_bytes,
    build_download_filename,
    draw_detections,
    format_detection_label,
)


def _write_best(run_dir: Path) -> Path:
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    best.write_bytes(b"fake-weights")
    return best


def test_discover_trained_models_finds_best(tmp_path: Path) -> None:
    run_a = tmp_path / "20260822-232302-e5a882"
    run_b = tmp_path / "20260822-180000-aaaaaa"
    _write_best(run_a)
    _write_best(run_b)

    models = discover_trained_models(tmp_path)
    assert len(models) == 2
    assert models[0].run_id == "20260822-232302-e5a882"
    assert models[0].label == "20260822-232302-e5a882 — best.pt"
    assert Path(models[0].weights_path).name == "best.pt"


def test_discover_excludes_runs_without_best(tmp_path: Path) -> None:
    complete = tmp_path / "run-ok"
    incomplete = tmp_path / "run-incomplete"
    _write_best(complete)
    incomplete.mkdir()
    (incomplete / "weights").mkdir()
    (incomplete / "status.json").write_text('{"state":"running"}', encoding="utf-8")

    models = discover_trained_models(tmp_path)
    assert len(models) == 1
    assert models[0].run_id == "run-ok"


def test_normalize_ultralytics_results_coordinates_and_names() -> None:
    boxes = MagicMock()
    boxes.xyxy = [[10.0, 20.0, 110.0, 220.0]]
    boxes.conf = [0.87]
    boxes.cls = [0]

    result = MagicMock()
    result.boxes = boxes
    result.names = {0: "pothole"}
    result.orig_shape = (480, 640)

    normalized = normalize_ultralytics_results([result])
    assert normalized.count == 1
    det = normalized.detections[0]
    assert det.class_id == 0
    assert det.class_name == "pothole"
    assert det.confidence == pytest.approx(0.87)
    assert det.x1 == 10.0
    assert det.y1 == 20.0
    assert det.x2 == 110.0
    assert det.y2 == 220.0
    assert normalized.image_width == 640
    assert normalized.image_height == 480


def test_normalize_empty_detections() -> None:
    result = MagicMock()
    result.boxes = None
    result.names = {0: "pothole"}
    result.orig_shape = (100, 100)

    normalized = normalize_ultralytics_results([result], class_names={0: "pothole"})
    assert normalized.count == 0
    assert normalized.detections == []


def test_class_id_to_class_name_mapping() -> None:
    boxes = MagicMock()
    boxes.xyxy = [[1, 2, 3, 4], [5, 6, 7, 8]]
    boxes.conf = [0.5, 0.6]
    boxes.cls = [1, 0]
    result = MagicMock()
    result.boxes = boxes
    result.names = None
    result.orig_shape = (10, 10)

    normalized = normalize_ultralytics_results(
        [result],
        class_names={0: "cat", 1: "dog"},
    )
    assert normalized.detections[0].class_name == "dog"
    assert normalized.detections[1].class_name == "cat"


def test_build_predict_kwargs_passes_thresholds() -> None:
    kwargs = build_predict_kwargs(conf=0.3, iou=0.5, device="cpu")
    assert kwargs["conf"] == 0.3
    assert kwargs["iou"] == 0.5
    assert kwargs["device"] == "cpu"


def test_build_predict_kwargs_rejects_invalid_conf() -> None:
    with pytest.raises(InferenceError, match="confiance"):
        build_predict_kwargs(conf=0.01, iou=0.45, device="cpu")


def test_run_inference_passes_conf_iou_device(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    image = Image.new("RGB", (64, 64), (10, 20, 30))

    mock_model = MagicMock()
    mock_model.names = {0: "pothole"}
    captured: dict = {}

    def _predict(**kwargs):
        captured.update(kwargs)
        boxes = MagicMock()
        boxes.xyxy = []
        boxes.conf = []
        boxes.cls = []
        result = MagicMock()
        result.boxes = boxes
        result.names = {0: "pothole"}
        result.orig_shape = (64, 64)
        return [result]

    mock_model.predict.side_effect = _predict

    result = run_inference(
        weights_path=weights,
        image=image,
        conf=0.4,
        iou=0.6,
        device_choice="cpu",
        model_factory=lambda _: mock_model,
    )
    assert result.count == 0
    assert captured["conf"] == 0.4
    assert captured["iou"] == 0.6
    assert captured["device"] == "cpu"


def test_run_inference_no_detection_is_ok(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    mock_model = MagicMock()
    mock_model.names = {0: "pothole"}
    empty = MagicMock()
    empty.boxes = None
    empty.names = {0: "pothole"}
    empty.orig_shape = (32, 32)
    mock_model.predict.return_value = [empty]

    result = run_inference(
        weights_path=weights,
        image=Image.new("RGB", (32, 32), (1, 1, 1)),
        model_factory=lambda _: mock_model,
        device_choice="cpu",
    )
    assert result.count == 0


def test_run_inference_rejects_cuda_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: False,
    )
    with pytest.raises(InferenceError, match="CUDA"):
        run_inference(
            weights_path=weights,
            image=Image.new("RGB", (16, 16), (0, 0, 0)),
            device_choice="cuda",
            model_factory=lambda _: MagicMock(),
        )


def test_run_inference_missing_weights(tmp_path: Path) -> None:
    with pytest.raises(InferenceError, match="introuvables"):
        run_inference(
            weights_path=tmp_path / "missing.pt",
            image=Image.new("RGB", (8, 8), (0, 0, 0)),
            device_choice="cpu",
            model_factory=lambda _: MagicMock(),
        )


def test_load_image_rejects_unsupported_and_corrupt(tmp_path: Path) -> None:
    bad_ext = tmp_path / "file.gif"
    bad_ext.write_bytes(b"gif")
    with pytest.raises(InferenceError, match="non supporté"):
        load_image_rgb(bad_ext)

    corrupt = tmp_path / "broken.jpg"
    corrupt.write_bytes(b"not-an-image")
    with pytest.raises(InferenceError, match="illisible|corrompue"):
        load_image_rgb(corrupt)


def test_draw_detections_and_export() -> None:
    image = Image.new("RGB", (200, 150), (240, 240, 240))
    detections = [
        Detection(
            class_id=0,
            class_name="pothole",
            confidence=0.87,
            x1=20,
            y1=30,
            x2=120,
            y2=100,
        )
    ]
    assert format_detection_label(detections[0]) == "pothole 87%"
    annotated = draw_detections(image, detections)
    assert annotated.size == image.size
    assert annotated.getpixel((25, 35)) != image.getpixel((25, 35)) or annotated != image

    jpeg_bytes = annotated_image_to_jpeg_bytes(annotated)
    assert jpeg_bytes[:2] == b"\xff\xd8"
    assert build_download_filename("street.png") == "prediction_street.jpg"
