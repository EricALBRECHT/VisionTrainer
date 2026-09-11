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
    DISPLAY_MAX_WIDTH,
    annotated_image_to_jpeg_bytes,
    build_download_filename,
    compute_annotation_style,
    compute_display_transform,
    draw_detections,
    format_detection_label,
    scale_box_to_display,
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
    assert models[0].label == "20260822-232302-e5a882 [det] — best.pt"
    assert Path(models[0].weights_path).name == "best.pt"


def test_discover_trained_models_filters_by_task(tmp_path: Path) -> None:
    det = tmp_path / "run-det"
    cls = tmp_path / "run-cls"
    _write_best(det)
    _write_best(cls)
    (cls / "status.json").write_text(
        '{"run_id":"run-cls","state":"completed","task":"classify"}',
        encoding="utf-8",
    )
    assert len(discover_trained_models(tmp_path, task="detect")) == 1
    assert discover_trained_models(tmp_path, task="detect")[0].run_id == "run-det"
    assert len(discover_trained_models(tmp_path, task="classify")) == 1
    assert discover_trained_models(tmp_path, task="classify")[0].run_id == "run-cls"


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


def test_normalize_empty_tensor_boxes_like_ultralytics() -> None:
    """Ultralytics returns Boxes with empty tensors, not None, when conf filters all."""
    boxes = MagicMock()
    boxes.xyxy = []
    boxes.conf = []
    boxes.cls = []
    result = MagicMock()
    result.boxes = boxes
    result.names = {0: "Carrot", 1: "Onion", 2: "Potato", 3: "Tomato"}
    result.orig_shape = (720, 1280)

    normalized = normalize_ultralytics_results([result])
    assert normalized.count == 0
    assert normalized.detections == []
    assert normalized.class_names[3] == "Tomato"


def test_run_inference_parity_when_ultralytics_returns_zero_boxes(tmp_path: Path) -> None:
    """Reproduce légumes case: predict @ conf=0.05 yields 0 boxes → VT also returns 0."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    mock_model = MagicMock()
    mock_model.names = {0: "Carrot", 1: "Onion", 2: "Potato", 3: "Tomato"}
    empty_boxes = MagicMock()
    empty_boxes.xyxy = []
    empty_boxes.conf = []
    empty_boxes.cls = []
    empty = MagicMock()
    empty.boxes = empty_boxes
    empty.names = mock_model.names
    empty.orig_shape = (720, 1280)
    mock_model.predict.return_value = [empty]

    result = run_inference(
        weights_path=weights,
        image=Image.new("RGB", (1280, 720), (1, 1, 1)),
        conf=0.05,
        iou=0.45,
        device_choice="cpu",
        model_factory=lambda _: mock_model,
    )
    assert result.count == 0
    assert result.class_names == mock_model.names
    assert mock_model.predict.call_args.kwargs["conf"] == 0.05


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


def test_compute_display_transform_and_box_scaling() -> None:
    # Smartphone photo → capped display canvas.
    transform = compute_display_transform(4032, 3024, max_width=1200)
    assert transform.display_width == 1200
    assert transform.display_height == 900
    assert transform.was_resized is True
    assert abs(transform.scale_x - (1200 / 4032)) < 1e-9
    assert abs(transform.scale_y - (900 / 3024)) < 1e-9

    x1, y1, x2, y2 = scale_box_to_display(403.2, 302.4, 2016.0, 1512.0, transform)
    assert abs(x1 - 120.0) < 1e-6
    assert abs(y1 - 90.0) < 1e-6
    assert abs(x2 - 600.0) < 1e-6
    assert abs(y2 - 450.0) < 1e-6

    # Already small: no resize.
    native = compute_display_transform(640, 480, max_width=1200)
    assert native.display_width == 640
    assert native.display_height == 480
    assert native.was_resized is False
    assert native.scale_x == 1.0

    # Portrait smartphone.
    portrait = compute_display_transform(3024, 4032, max_width=1200)
    assert portrait.display_width == 1200
    assert portrait.display_height == 1600


def test_compute_annotation_style_targets_display_canvas() -> None:
    s1200 = compute_annotation_style(1200, 900, "auto")
    assert s1200.font_size == 22
    assert s1200.line_width == 3

    s640 = compute_annotation_style(640, 640, "auto")
    assert 16 <= s640.font_size <= 24
    assert 2 <= s640.line_width <= 4

    # Style follows the display canvas, not a 4K original.
    s_display_from_phone = compute_annotation_style(1200, 900, "auto")
    s_raw_4k = compute_annotation_style(4032, 3024, "auto")
    # If someone wrongly styled on the original, font would hit the 40px clamp;
    # display styling stays in the readable mid-20s.
    assert s_display_from_phone.font_size == 22
    assert s_raw_4k.font_size == 40  # clamped — why we must style after resize

    large = compute_annotation_style(1200, 900, "large")
    small = compute_annotation_style(1200, 900, "small")
    assert large.font_size > s1200.font_size
    assert small.font_size < s1200.font_size
    assert 2 <= small.line_width <= 4
    assert 2 <= large.line_width <= 4


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
    assert format_detection_label(detections[0]) == "pothole 0.87"
    annotated = draw_detections(image, detections)
    assert annotated.size == image.size
    assert annotated.getpixel((25, 35)) != image.getpixel((25, 35)) or annotated != image

    jpeg_bytes = annotated_image_to_jpeg_bytes(annotated)
    assert jpeg_bytes[:2] == b"\xff\xd8"
    assert build_download_filename("street.png") == "prediction_street.jpg"


def test_draw_detections_downscales_hires_before_annotating() -> None:
    image = Image.new("RGB", (4032, 3024), (240, 240, 240))
    detections = [
        Detection(
            class_id=0,
            class_name="Tomato",
            confidence=0.96,
            x1=403.2,
            y1=302.4,
            x2=2016.0,
            y2=1512.0,
        )
    ]
    annotated = draw_detections(image, detections, scale="auto", max_display_width=1200)
    assert annotated.size == (1200, 900)
    # Outline sits on the scaled top-left corner ≈ (120, 90).
    assert annotated.getpixel((120, 90)) != (240, 240, 240)
    # Label badge is drawn just above the box.
    assert annotated.getpixel((130, 70)) != (240, 240, 240)


def test_draw_detections_respects_scale_preset() -> None:
    image = Image.new("RGB", (1920, 1080), (240, 240, 240))
    detections = [
        Detection(
            class_id=0,
            class_name="valve",
            confidence=0.9,
            x1=100,
            y1=100,
            x2=400,
            y2=300,
        )
    ]
    small = draw_detections(image, detections, scale="small")
    large = draw_detections(image, detections, scale="large")
    # Both fit to DISPLAY_MAX_WIDTH.
    expected_w = min(1920, DISPLAY_MAX_WIDTH)
    expected_h = round(1080 * (expected_w / 1920))
    assert small.size == large.size == (expected_w, expected_h)
    assert small.tobytes() != large.tobytes()

def test_inference_display_max_width_fractions() -> None:
    from vision_trainer.inference.render import (
        DISPLAY_MAX_WIDTH,
        DEFAULT_INFERENCE_DISPLAY_FRACTION,
        inference_display_max_width,
        preview_image_for_ui,
    )

    assert DEFAULT_INFERENCE_DISPLAY_FRACTION == 0.50
    assert inference_display_max_width(1.0) == DISPLAY_MAX_WIDTH
    assert inference_display_max_width(0.5) == DISPLAY_MAX_WIDTH // 2
    assert inference_display_max_width(0.25) == DISPLAY_MAX_WIDTH // 4
    assert inference_display_max_width(0.75) == int(round(DISPLAY_MAX_WIDTH * 0.75))


def test_preview_image_for_ui_does_not_mutate_original() -> None:
    from vision_trainer.inference.render import preview_image_for_ui

    image = Image.new("RGB", (4000, 2000), color=(10, 20, 30))
    original_size = image.size
    preview_50 = preview_image_for_ui(image, display_fraction=0.50)
    preview_100 = preview_image_for_ui(image, display_fraction=1.00)
    assert image.size == original_size
    assert preview_50.width <= 640
    assert preview_100.width <= 1280
    assert preview_50.width < preview_100.width

