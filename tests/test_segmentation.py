"""Tests for YOLO segmentation support."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.inference.render import compute_display_transform
from vision_trainer.pipeline.store import list_classify_run_ids
from vision_trainer.segment.area import mask_area_pixels, mask_area_ratio, polygon_area_pixels
from vision_trainer.segment.export import export_segmentation_json, segmentation_result_json_bytes
from vision_trainer.segment.labels import (
    looks_like_detection_bbox_line,
    parse_segment_label_line,
    polygon_to_pixel_points,
)
from vision_trainer.segment.models import SegmentInstance, SegmentationResult
from vision_trainer.segment.predictor import normalize_segmentation_results
from vision_trainer.segment.render import draw_segmentation_result, scale_polygon_to_display
from vision_trainer.segment.validator import validate_segment_dataset
from vision_trainer.tasks import normalize_task, task_label_fr
from vision_trainer.training.status import RunMetrics, extract_metrics_from_trainer
from vision_trainer.yolo.parser import extract_zip_dataset


def _write_seg_layout(root: Path, *, with_yaml: bool = True, detect_labels: bool = False) -> Path:
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    val_img = root / "val" / "images"
    val_lbl = root / "val" / "labels"
    for path in (train_img, train_lbl, val_img, val_lbl):
        path.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(train_img / "a.jpg")
    Image.new("RGB", (64, 48), color=(11, 21, 31)).save(val_img / "b.jpg")

    if detect_labels:
        (train_lbl / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (val_lbl / "b.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    else:
        # triangle polygon
        (train_lbl / "a.txt").write_text(
            "0 0.2 0.2 0.8 0.2 0.5 0.8\n1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n",
            encoding="utf-8",
        )
        (val_lbl / "b.txt").write_text(
            "0 0.2 0.2 0.8 0.2 0.5 0.8\n",
            encoding="utf-8",
        )

    if with_yaml:
        (root / "data.yaml").write_text(
            "path: .\ntrain: train/images\nval: val/images\nnames: {0: Apple, 1: Leaf}\n",
            encoding="utf-8",
        )
    return root


def test_task_segment_normalize_and_label() -> None:
    assert normalize_task("segment") == "segment"
    assert task_label_fr("segment") == "Segmentation"
    assert normalize_task(None) == "detect"


def test_valid_segment_dataset(tmp_path: Path) -> None:
    root = _write_seg_layout(tmp_path / "ds")
    result = validate_segment_dataset(root)
    assert result.is_valid
    assert result.dataset is not None
    assert result.dataset.num_classes == 2
    info = getattr(result, "segment_info", None)
    assert info is not None
    assert info.total_instances == 3


def test_segment_dataset_without_yaml_generates(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    train_img.mkdir(parents=True)
    train_lbl.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=1).save(train_img / "x.jpg")
    (train_lbl / "x.txt").write_text("0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8")
    (root / "train" / "classes.txt").write_text("Fruit\n", encoding="utf-8")
    result = validate_segment_dataset(root)
    assert result.dataset is not None
    assert result.dataset.yaml_path.is_file()


def test_detect_labels_rejected_as_segment(tmp_path: Path) -> None:
    root = _write_seg_layout(tmp_path / "ds", detect_labels=True)
    result = validate_segment_dataset(root)
    assert not result.is_valid
    assert any("bounding boxes de détection" in issue.message for issue in result.errors)


def test_invalid_polygon_coords() -> None:
    with pytest.raises(ValueError, match="hors"):
        parse_segment_label_line("0 1.5 0.2 0.8 0.2 0.5 0.8")
    with pytest.raises(ValueError, match="impair"):
        parse_segment_label_line("0 0.1 0.1 0.2 0.2 0.3 0.3 0.4")
    with pytest.raises(ValueError, match="insuffisant"):
        parse_segment_label_line("0 0.1 0.1")
    with pytest.raises(ValueError, match="bounding boxes"):
        parse_segment_label_line("0 0.5 0.5 0.2 0.2")


def test_class_out_of_range_in_validator(tmp_path: Path) -> None:
    root = _write_seg_layout(tmp_path / "ds")
    (root / "train" / "labels" / "a.txt").write_text(
        "9 0.2 0.2 0.8 0.2 0.5 0.8\n",
        encoding="utf-8",
    )
    result = validate_segment_dataset(root)
    assert not result.is_valid
    assert any("classe 9" in issue.message.lower() or "9" in issue.message for issue in result.errors)


def test_zip_extra_root_folder(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    _write_seg_layout(payload / "nested")
    zip_path = tmp_path / "seg.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file_path in (payload / "nested").rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=str(file_path.relative_to(payload)))
    extract = tmp_path / "out"
    extract.mkdir()
    extract_zip_dataset(zip_path, extract)
    result = validate_segment_dataset(extract, containment_root=extract)
    assert result.dataset is not None


def test_looks_like_detection_bbox() -> None:
    assert looks_like_detection_bbox_line("0 0.5 0.5 0.2 0.2".split())
    assert not looks_like_detection_bbox_line("0 0.2 0.2 0.8 0.2 0.5 0.8".split())


def test_instance_counting(tmp_path: Path) -> None:
    root = _write_seg_layout(tmp_path / "ds")
    result = validate_segment_dataset(root)
    info = getattr(result, "segment_info")
    assert info.instances_by_class()["Apple"] == 2
    assert info.instances_by_class()["Leaf"] == 1


def test_polygon_display_transform() -> None:
    points = ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))
    pixels = polygon_to_pixel_points(points, image_width=200, image_height=100)
    transform = compute_display_transform(200, 100, max_width=100)
    scaled = scale_polygon_to_display(tuple(pixels), transform)
    assert scaled[0] == (0.0, 0.0)
    assert scaled[1][0] == pytest.approx(100.0)


def test_discover_segment_and_exclude_from_classify(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    for run_id, task in (("seg-1", "segment"), ("cls-1", "classify"), ("det-1", "detect")):
        run = runs / run_id
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "best.pt").write_bytes(b"x")
        (run / "config.json").write_text(json.dumps({"task": task}), encoding="utf-8")
        (run / "status.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "state": "completed",
                    "task": task,
                    "model": "m",
                    "epochs_total": 1,
                }
            ),
            encoding="utf-8",
        )

    models = discover_trained_models(runs, task="segment")
    assert [m.run_id for m in models] == ["seg-1"]
    assert "[seg]" in models[0].label

    classify_models = discover_trained_models(runs, task="classify")
    assert all(m.run_id != "seg-1" for m in classify_models)

    assert "seg-1" not in list_classify_run_ids()
    assert "cls-1" in list_classify_run_ids()


def test_legacy_run_without_task_is_detect(tmp_path: Path) -> None:
    run = tmp_path / "legacy"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    models = discover_trained_models(tmp_path, task="detect")
    assert models[0].run_id == "legacy"
    assert "[det]" in models[0].label


def test_segmentation_result_structure_and_json(tmp_path: Path) -> None:
    result = SegmentationResult(
        instances=[
            SegmentInstance(
                class_id=0,
                class_name="Apple",
                confidence=0.96,
                x1=1,
                y1=2,
                x2=10,
                y2=20,
                polygon=((1.0, 2.0), (10.0, 2.0), (5.0, 20.0)),
                mask_area_pixels=100,
                mask_area_ratio=0.01,
            )
        ],
        image_width=100,
        image_height=100,
    )
    payload = result.to_json_dict()
    assert "instances" in payload
    assert "mask" not in json.dumps(payload).lower() or "mask_area" in json.dumps(payload)
    raw = segmentation_result_json_bytes(result)
    assert b"Apple" in raw
    path = export_segmentation_json(result, tmp_path / "out.json")
    assert path.is_file()


def test_mask_area_pixels_and_ratio() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, 0:4] = 1
    area = mask_area_pixels(mask)
    assert area == 20
    assert mask_area_ratio(area, image_width=10, image_height=10) == pytest.approx(0.2)


def test_polygon_area_pixels() -> None:
    # Full image rectangle via polygon
    area = polygon_area_pixels(
        [(0, 0), (9, 0), (9, 9), (0, 9)],
        image_width=10,
        image_height=10,
    )
    assert area >= 80  # rasterization may miss edge pixels slightly


def test_normalize_without_masks() -> None:
    boxes = MagicMock()
    boxes.xyxy = [[10.0, 20.0, 40.0, 50.0]]
    boxes.conf = [0.9]
    boxes.cls = [0]
    result = MagicMock()
    result.boxes = boxes
    result.masks = None
    result.names = {0: "Apple"}
    result.orig_shape = (100, 200)

    normalized = normalize_segmentation_results([result], class_names={0: "Apple"})
    assert len(normalized.instances) == 1
    assert normalized.instances[0].polygon == ()
    assert normalized.instances[0].class_name == "Apple"


def test_normalize_multiple_instances_with_polygons() -> None:
    boxes = MagicMock()
    boxes.xyxy = [[1, 1, 10, 10], [20, 20, 40, 40]]
    boxes.conf = [0.8, 0.7]
    boxes.cls = [0, 0]
    masks = MagicMock()
    masks.xy = [
        np.array([[1.0, 1.0], [10.0, 1.0], [5.0, 10.0]]),
        np.array([[20.0, 20.0], [40.0, 20.0], [30.0, 40.0]]),
    ]
    masks.data = None
    result = MagicMock()
    result.boxes = boxes
    result.masks = masks
    result.names = {0: "Apple"}
    result.orig_shape = (80, 80)

    normalized = normalize_segmentation_results([result])
    assert len(normalized.instances) == 2
    assert all(inst.mask_area_pixels > 0 for inst in normalized.instances)


def test_render_without_mask_does_not_crash() -> None:
    image = Image.new("RGB", (200, 150), color=(30, 30, 30))
    result = SegmentationResult(
        instances=[
            SegmentInstance(0, "Apple", 0.9, 10, 10, 50, 50, polygon=(), mask_area_pixels=0)
        ],
        image_width=200,
        image_height=150,
    )
    out = draw_segmentation_result(image, result, show_masks=True, show_boxes=True)
    assert out.size[0] <= 200


def test_extract_mask_metrics_from_trainer() -> None:
    metrics = MagicMock()
    metrics.results_dict = {
        "metrics/precision(B)": 0.1,
        "metrics/recall(B)": 0.2,
        "metrics/mAP50(B)": 0.3,
        "metrics/mAP50-95(B)": 0.4,
        "metrics/precision(M)": 0.5,
        "metrics/recall(M)": 0.6,
        "metrics/mAP50(M)": 0.7,
        "metrics/mAP50-95(M)": 0.8,
    }
    box = MagicMock()
    box.mp = box.mr = box.map50 = box.map = None
    seg = MagicMock()
    seg.mp = seg.mr = seg.map50 = seg.map = None
    metrics.box = box
    metrics.seg = seg
    trainer = MagicMock()
    trainer.metrics = metrics
    extracted = extract_metrics_from_trainer(trainer)
    assert isinstance(extracted, RunMetrics)
    assert extracted.map50 == pytest.approx(0.3)
    assert extracted.mask_map50 == pytest.approx(0.7)
    assert extracted.mask_precision == pytest.approx(0.5)


def test_available_seg_models() -> None:
    from vision_trainer.training.segment_trainer import AVAILABLE_SEG_MODELS, DEFAULT_SEG_MODEL_KEY

    assert DEFAULT_SEG_MODEL_KEY in AVAILABLE_SEG_MODELS
    assert AVAILABLE_SEG_MODELS[DEFAULT_SEG_MODEL_KEY].endswith("-seg.pt")
