"""Pipeline V3: detect → classify? → segment? + v1 compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from vision_trainer.pipeline.crop import CropRegion, crop_region_from_detection
from vision_trainer.pipeline.engine import get_cached_model, run_pipeline
from vision_trainer.pipeline.export import pipeline_result_json_bytes
from vision_trainer.pipeline.models import (
    ClassificationStage,
    ClassMapping,
    PipelineConfig,
    SegmentationStage,
)
from vision_trainer.pipeline.store import (
    load_pipeline,
    save_pipeline,
    validate_pipeline_config,
)


def _write_run(runs_root: Path, run_id: str, *, task: str) -> Path:
    run_dir = runs_root / run_id
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"fake-weights")
    (run_dir / "config.json").write_text(
        json.dumps({"task": task, "run_id": run_id}),
        encoding="utf-8",
    )
    return run_dir


def _blank(width: int = 200, height: int = 150) -> Image.Image:
    return Image.new("RGB", (width, height), color=(40, 40, 40))


def _mock_detect(detections):
    name_to_id: dict[str, int] = {}
    for name, _, _ in detections:
        if name not in name_to_id:
            name_to_id[name] = len(name_to_id)
    id_to_name = {i: n for n, i in name_to_id.items()}
    boxes = MagicMock()
    boxes.xyxy = [[*box] for _, _, box in detections]
    boxes.conf = [c for _, c, _ in detections]
    boxes.cls = [name_to_id[name] for name, _, _ in detections]
    result = MagicMock()
    result.boxes = boxes
    result.names = id_to_name
    result.orig_shape = (150, 200)
    model = MagicMock()
    model.names = id_to_name
    model.predict.return_value = [result]
    return model


def _mock_classify(scores: dict[str, float]):
    names = {i: name for i, name in enumerate(scores)}
    values = [scores[names[i]] for i in range(len(names))]
    probs = MagicMock()
    probs.data = values
    result = MagicMock()
    result.probs = probs
    result.names = names
    model = MagicMock()
    model.names = names
    model.predict.return_value = [result]
    return model


def _mock_segment(instances: list[tuple[str, float, list[list[float]]]]):
    """instances: (class_name, conf, polygon_xy local)."""
    name_to_id: dict[str, int] = {}
    for name, _, _ in instances:
        if name not in name_to_id:
            name_to_id[name] = len(name_to_id)
    id_to_name = {i: n for n, i in name_to_id.items()}
    boxes = MagicMock()
    boxes.xyxy = []
    boxes.conf = []
    boxes.cls = []
    polys = []
    for name, conf, poly in instances:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.xyxy.append([min(xs), min(ys), max(xs), max(ys)])
        boxes.conf.append(conf)
        boxes.cls.append(name_to_id[name])
        import numpy as np

        polys.append(np.array(poly, dtype=float))
    masks = MagicMock()
    masks.xy = polys
    masks.data = None
    result = MagicMock()
    result.boxes = boxes
    result.masks = masks
    result.names = id_to_name
    result.orig_shape = (40, 40)
    model = MagicMock()
    model.names = id_to_name
    model.predict.return_value = [result]
    return model


def test_v1_pipeline_still_readable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    pipelines = tmp_path / "data" / "pipelines"
    pipelines.mkdir(parents=True)
    payload = {
        "format_version": 1,
        "pipeline_id": "old-v1",
        "name": "Legacy",
        "detector_run_id": "det-1",
        "crop_padding": 0.05,
        "detect_conf": 0.25,
        "detect_iou": 0.45,
        "mappings": {
            "Apple": {
                "enabled": True,
                "classifier_run_id": "cls-apple",
                "confidence_threshold": 0.8,
                "margin_threshold": 0.1,
                "top_n": 3,
            },
            "Carrot": {"enabled": False},
        },
    }
    (pipelines / "old-v1.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_pipeline("old-v1")
    assert loaded.format_version == 1
    assert loaded.mappings["Apple"].classification.enabled is True
    assert loaded.mappings["Apple"].classification.run_id == "cls-apple"
    assert loaded.mappings["Apple"].segmentation.enabled is False
    assert loaded.mappings["Carrot"].enabled is False
    # Disk file not rewritten
    on_disk = json.loads((pipelines / "old-v1.json").read_text(encoding="utf-8"))
    assert on_disk["format_version"] == 1
    assert "classifier_run_id" in on_disk["mappings"]["Apple"]


def test_v1_normalize_then_save_becomes_v2(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-apple", task="classify")
    pipelines = tmp_path / "data" / "pipelines"
    pipelines.mkdir(parents=True)
    (pipelines / "old-v1.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "pipeline_id": "old-v1",
                "name": "Legacy",
                "detector_run_id": "det-1",
                "mappings": {
                    "Apple": {
                        "enabled": True,
                        "classifier_run_id": "cls-apple",
                        "confidence_threshold": 0.8,
                        "margin_threshold": 0.1,
                        "top_n": 3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_pipeline("old-v1")
    save_pipeline(loaded)
    saved = json.loads((pipelines / "old-v1.json").read_text(encoding="utf-8"))
    assert saved["format_version"] == 2
    assert saved["mappings"]["Apple"]["classification"]["run_id"] == "cls-apple"


def test_v1_runtime_compat_classify_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-apple", task="classify")
    config = PipelineConfig.from_dict(
        {
            "format_version": 1,
            "pipeline_id": "p1",
            "name": "V1",
            "detector_run_id": "det-1",
            "crop_padding": 0.0,
            "mappings": {
                "Apple": {
                    "enabled": True,
                    "classifier_run_id": "cls-apple",
                    "confidence_threshold": 0.5,
                    "margin_threshold": 0.01,
                    "top_n": 3,
                }
            },
        }
    )
    det = _mock_detect([("Apple", 0.94, (10, 10, 50, 50))])
    cls = _mock_classify({"Golden": 0.96, "Gala": 0.03})
    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())
    cls_path = str((runs / "cls-apple" / "weights" / "best.pt").resolve())

    def factory(path: str):
        return det if str(Path(path).resolve()) == det_path else cls

    result = run_pipeline(_blank(), config, model_factory=factory, model_cache={})
    assert result.items[0].classification is not None
    assert result.items[0].classification.class_name == "Golden"
    assert result.items[0].segmentation is None
    assert "Golden" in result.items[0].display_label


def test_detect_only_v2(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    config = PipelineConfig(
        pipeline_id="p1",
        name="Detect only",
        detector_run_id="det-1",
        mappings={"Apple": ClassMapping()},
    )
    det = _mock_detect([("Apple", 0.9, (10, 10, 40, 40))])
    result = run_pipeline(
        _blank(), config, model_factory=lambda _: det, model_cache={}
    )
    assert result.items[0].refined is False


def test_detect_plus_segment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "seg-1", task="segment")
    config = PipelineConfig(
        pipeline_id="p1",
        name="Seg",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "Bearing": ClassMapping(
                segmentation=SegmentationStage(
                    enabled=True,
                    run_id="seg-1",
                    confidence_threshold=0.25,
                )
            )
        },
    )
    det = _mock_detect([("Bearing", 0.93, (10, 10, 50, 50))])
    seg = _mock_segment(
        [("Rust", 0.88, [[5.0, 5.0], [30.0, 5.0], [20.0, 25.0]])]
    )
    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())
    seg_path = str((runs / "seg-1" / "weights" / "best.pt").resolve())

    def factory(path: str):
        return det if str(Path(path).resolve()) == det_path else seg

    result = run_pipeline(_blank(), config, model_factory=factory, model_cache={})
    item = result.items[0]
    assert item.classification is None
    assert item.segmentation is not None
    assert item.segmentation.status == "ok"
    assert len(item.segmentations) == 1
    mask = item.segmentations[0]
    assert mask.class_name == "Rust"
    # local (5,5) + crop origin (10,10) => (15,15)
    assert mask.polygon_global[0] == pytest.approx((15.0, 15.0))
    assert mask.mask_area_ratio_crop > 0
    assert mask.mask_area_ratio_image > 0


def test_detect_classify_segment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-1", task="classify")
    _write_run(runs, "seg-1", task="segment")
    config = PipelineConfig(
        pipeline_id="p1",
        name="Full",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "Bearing": ClassMapping(
                classification=ClassificationStage(
                    enabled=True,
                    run_id="cls-1",
                    confidence_threshold=0.5,
                    margin_threshold=0.01,
                ),
                segmentation=SegmentationStage(
                    enabled=True, run_id="seg-1", confidence_threshold=0.2
                ),
            ),
            "Other": ClassMapping(),
        },
    )
    det = _mock_detect(
        [
            ("Bearing", 0.94, (10, 10, 50, 50)),
            ("Other", 0.91, (60, 10, 90, 40)),
        ]
    )
    cls = _mock_classify({"TypeA": 0.92, "TypeB": 0.05})
    seg = _mock_segment(
        [
            ("Rust", 0.88, [[2.0, 2.0], [20.0, 2.0], [10.0, 15.0]]),
            ("Crack", 0.70, [[25.0, 25.0], [35.0, 25.0], [30.0, 35.0]]),
        ]
    )
    paths = {
        "det": str((runs / "det-1" / "weights" / "best.pt").resolve()),
        "cls": str((runs / "cls-1" / "weights" / "best.pt").resolve()),
        "seg": str((runs / "seg-1" / "weights" / "best.pt").resolve()),
    }
    models = {paths["det"]: det, paths["cls"]: cls, paths["seg"]: seg}
    counts = {"cls": 0, "seg": 0}

    def factory(path: str):
        key = str(Path(path).resolve())
        if key == paths["cls"]:
            counts["cls"] += 1
        if key == paths["seg"]:
            counts["seg"] += 1
        return models[key]

    cache: dict = {}
    result = run_pipeline(_blank(), config, model_factory=factory, model_cache=cache)
    bearing, other = result.items
    assert bearing.classification and bearing.classification.class_name == "TypeA"
    assert len(bearing.segmentations) == 2
    assert other.refined is False
    assert counts["cls"] == 1
    assert counts["seg"] == 1
    assert len(cache) == 3
    assert result.timings.total_ms >= 0
    raw = pipeline_result_json_bytes(result)
    assert b"TypeA" in raw
    assert b"polygon_global" in raw
    assert b"mask_area_ratio_crop" in raw


def test_invalid_tasks_on_validate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "seg-as-cls", task="segment")
    _write_run(runs, "cls-as-seg", task="classify")
    bad_cls = PipelineConfig(
        pipeline_id="p1",
        name="Bad",
        detector_run_id="det-1",
        mappings={
            "A": ClassMapping(
                classification=ClassificationStage(enabled=True, run_id="seg-as-cls")
            )
        },
    )
    with pytest.raises(Exception, match="classificateur"):
        validate_pipeline_config(bad_cls, check_weights=True)

    bad_seg = PipelineConfig(
        pipeline_id="p2",
        name="Bad2",
        detector_run_id="det-1",
        mappings={
            "A": ClassMapping(
                segmentation=SegmentationStage(enabled=True, run_id="cls-as-seg")
            )
        },
    )
    with pytest.raises(Exception, match="segmentation"):
        validate_pipeline_config(bad_seg, check_weights=True)


def test_missing_segmenter_keeps_classification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-1", task="classify")
    config = PipelineConfig(
        pipeline_id="p1",
        name="Missing seg",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "Bearing": ClassMapping(
                classification=ClassificationStage(
                    enabled=True, run_id="cls-1", confidence_threshold=0.5, margin_threshold=0.01
                ),
                segmentation=SegmentationStage(enabled=True, run_id="missing-seg"),
            )
        },
    )
    det = _mock_detect([("Bearing", 0.9, (10, 10, 50, 50))])
    cls = _mock_classify({"TypeA": 0.9, "TypeB": 0.05})
    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())
    cls_path = str((runs / "cls-1" / "weights" / "best.pt").resolve())

    def factory(path: str):
        return det if str(Path(path).resolve()) == det_path else cls

    result = run_pipeline(_blank(), config, model_factory=factory, model_cache={})
    item = result.items[0]
    assert item.detection.class_name == "Bearing"
    assert item.classification and item.classification.class_name == "TypeA"
    assert item.segmentation and item.segmentation.status == "error"
    assert result.warnings


def test_crop_local_to_global_with_padding() -> None:
    image = _blank(100, 80)
    region = crop_region_from_detection(image, 20, 20, 40, 40, padding=0.05)
    # 5% of 20 = 1 → box approx (19,19,41,41)
    assert region.x1 <= 20 and region.y1 <= 20
    gx, gy = region.local_to_global_point(5.0, 7.0)
    assert gx == pytest.approx(region.x1 + 5.0)
    assert gy == pytest.approx(region.y1 + 7.0)
    poly = region.local_to_global_polygon(((0.0, 0.0), (10.0, 0.0), (5.0, 10.0)))
    assert poly[0] == (float(region.x1), float(region.y1))
    bbox = region.local_to_global_bbox(1, 2, 8, 9)
    assert bbox[0] == pytest.approx(region.x1 + 1)


def test_crop_region_dataclass() -> None:
    crop = Image.new("RGB", (30, 20))
    region = CropRegion(10, 15, 40, 35, crop)
    assert region.width == 30
    assert region.height == 20
    assert region.box == (10, 15, 40, 35)


def test_segment_error_non_fatal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "seg-1", task="segment")
    config = PipelineConfig(
        pipeline_id="p1",
        name="Boom seg",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "A": ClassMapping(
                segmentation=SegmentationStage(enabled=True, run_id="seg-1")
            )
        },
    )
    det = _mock_detect([("A", 0.9, (10, 10, 40, 40))])
    seg = MagicMock()
    seg.names = {0: "x"}
    seg.predict.side_effect = RuntimeError("seg fail")
    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())

    def factory(path: str):
        return det if str(Path(path).resolve()) == det_path else seg

    result = run_pipeline(_blank(), config, model_factory=factory, model_cache={})
    assert result.items[0].detection.confidence == pytest.approx(0.9)
    assert result.items[0].segmentation is not None
    assert result.items[0].segmentation.status == "error"


def test_multiple_objects_same_class_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "seg-1", task="segment")
    config = PipelineConfig(
        pipeline_id="p1",
        name="Multi",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "A": ClassMapping(
                segmentation=SegmentationStage(enabled=True, run_id="seg-1")
            )
        },
    )
    det = _mock_detect(
        [
            ("A", 0.9, (10, 10, 30, 30)),
            ("A", 0.88, (40, 10, 60, 30)),
            ("A", 0.85, (70, 10, 90, 30)),
        ]
    )
    seg = _mock_segment([("Rust", 0.8, [[1, 1], [10, 1], [5, 10]])])
    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())
    seg_path = str((runs / "seg-1" / "weights" / "best.pt").resolve())
    loads = {"seg": 0}

    def factory(path: str):
        key = str(Path(path).resolve())
        if key == seg_path:
            loads["seg"] += 1
            return seg
        return det

    cache: dict = {}
    result = run_pipeline(_blank(), config, model_factory=factory, model_cache=cache)
    assert len(result.items) == 3
    assert loads["seg"] == 1
    assert get_cached_model(seg_path, cache=cache, model_factory=factory) is seg
