"""Tests for detection → classification pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from vision_trainer.inference.models import Detection
from vision_trainer.pipeline.crop import CropError, crop_from_detection, padded_crop_box
from vision_trainer.pipeline.engine import (
    PipelineEngineError,
    get_cached_model,
    result_table_rows,
    run_pipeline,
)
from vision_trainer.pipeline.models import (
    ClassMapping,
    EnrichedDetection,
    PipelineConfig,
)
from vision_trainer.pipeline.store import (
    PipelineStoreError,
    list_pipelines,
    load_pipeline,
    make_pipeline_id,
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


def _blank_image(width: int = 200, height: int = 150) -> Image.Image:
    return Image.new("RGB", (width, height), color=(40, 40, 40))


def test_pipeline_create_and_load_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-apple", task="classify")

    config = PipelineConfig(
        pipeline_id="pipe-fruits",
        name="Fruits détaillés",
        detector_run_id="det-1",
        mappings={
            "Apple": ClassMapping(
                enabled=True,
                classifier_run_id="cls-apple",
                confidence_threshold=0.8,
                margin_threshold=0.1,
                top_n=3,
            ),
            "Carrot": ClassMapping(enabled=False),
        },
    )
    validate_pipeline_config(config, check_weights=True)
    path = save_pipeline(config)
    assert path.is_file()

    loaded = load_pipeline("pipe-fruits")
    assert loaded.name == "Fruits détaillés"
    assert loaded.detector_run_id == "det-1"
    assert loaded.mappings["Apple"].enabled is True
    assert loaded.mappings["Apple"].classifier_run_id == "cls-apple"
    assert loaded.mappings["Carrot"].enabled is False
    assert loaded.format_version == 1


def test_serialization_deserialization_dict() -> None:
    config = PipelineConfig(
        pipeline_id="p1",
        name="N",
        detector_run_id="d1",
        mappings={"A": ClassMapping(enabled=True, classifier_run_id="c1")},
    )
    restored = PipelineConfig.from_dict(config.to_dict())
    assert restored.to_dict()["mappings"]["A"]["classifier_run_id"] == "c1"


def test_invalid_pipeline_skipped_in_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    pipelines = tmp_path / "data" / "pipelines"
    pipelines.mkdir(parents=True)
    (pipelines / "broken.json").write_text("{not-json", encoding="utf-8")
    (pipelines / "empty.json").write_text("{}", encoding="utf-8")
    assert list_pipelines() == []


def test_load_pipeline_missing_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(PipelineStoreError, match="introuvable"):
        load_pipeline("missing-pipe")


def test_validate_detect_model_not_found(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    config = PipelineConfig(
        pipeline_id="p1",
        name="N",
        detector_run_id="no-such-run",
        mappings={},
    )
    with pytest.raises(PipelineStoreError, match="Détecteur"):
        validate_pipeline_config(config, check_weights=True)


def test_crop_basic() -> None:
    image = _blank_image(100, 80)
    crop, box = crop_from_detection(image, 10, 10, 50, 50, padding=0.0)
    assert box == (10, 10, 50, 50)
    assert crop.size == (40, 40)


def test_crop_near_edge() -> None:
    image = _blank_image(100, 80)
    crop, box = crop_from_detection(image, -5, -5, 20, 20, padding=0.0)
    assert box[0] == 0 and box[1] == 0
    assert crop.size[0] > 0 and crop.size[1] > 0


def test_crop_with_padding() -> None:
    image = _blank_image(200, 200)
    # box 40x40 centered-ish; 5% pad => 2 px each side
    box = padded_crop_box(40, 40, 80, 80, image_width=200, image_height=200, padding=0.05)
    assert box == (38, 38, 82, 82)
    crop, _ = crop_from_detection(image, 40, 40, 80, 80, padding=0.05)
    assert crop.size == (44, 44)


def test_crop_empty_box_raises() -> None:
    with pytest.raises(CropError):
        padded_crop_box(10, 10, 10, 10, image_width=100, image_height=100, padding=0.0)


def _mock_detect_model(detections: list[tuple[str, float, tuple[float, float, float, float]]]):
    names = {}
    for index, (name, _, _) in enumerate(detections):
        names[index] = name
    # unique names map
    unique = {name: i for i, (name, _, _) in enumerate(detections)}
    # rebuild properly by class
    name_to_id: dict[str, int] = {}
    for name, _, _ in detections:
        if name not in name_to_id:
            name_to_id[name] = len(name_to_id)
    id_to_name = {i: n for n, i in name_to_id.items()}

    boxes = MagicMock()
    boxes.xyxy = [[x1, y1, x2, y2] for _, _, (x1, y1, x2, y2) in detections]
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


def _mock_classify_model(scores: dict[str, float]):
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


def test_run_pipeline_mapping_and_skip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-apple", task="classify")

    config = PipelineConfig(
        pipeline_id="p1",
        name="Test",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "Apple": ClassMapping(
                enabled=True,
                classifier_run_id="cls-apple",
                confidence_threshold=0.5,
                margin_threshold=0.05,
                top_n=3,
            ),
            "Carrot": ClassMapping(enabled=False),
        },
    )

    det_model = _mock_detect_model(
        [
            ("Apple", 0.94, (10, 10, 60, 60)),
            ("Carrot", 0.93, (70, 10, 120, 60)),
        ]
    )
    cls_model = _mock_classify_model({"Golden": 0.96, "Gala": 0.03, "Granny": 0.01})

    models = {
        str((runs / "det-1" / "weights" / "best.pt").resolve()): det_model,
        str((runs / "cls-apple" / "weights" / "best.pt").resolve()): cls_model,
    }

    def factory(path: str):
        return models[str(Path(path).resolve())]

    result = run_pipeline(
        _blank_image(),
        config,
        model_factory=factory,
        model_cache={},
    )
    assert len(result.items) == 2
    apple, carrot = result.items
    assert apple.refined is True
    assert apple.classification is not None
    assert apple.classification.status == "ok"
    assert apple.classification.class_name == "Golden"
    assert carrot.refined is False
    assert "Golden" in apple.display_label
    assert "Carrot" in carrot.display_label


def test_run_pipeline_multiple_same_class_and_classifiers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-a", task="classify")
    _write_run(runs, "cls-b", task="classify")

    config = PipelineConfig(
        pipeline_id="p1",
        name="Multi",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "Apple": ClassMapping(
                enabled=True,
                classifier_run_id="cls-a",
                confidence_threshold=0.5,
                margin_threshold=0.01,
            ),
            "Tomato": ClassMapping(
                enabled=True,
                classifier_run_id="cls-b",
                confidence_threshold=0.5,
                margin_threshold=0.01,
            ),
            "Carrot": ClassMapping(enabled=False),
        },
    )

    det_model = _mock_detect_model(
        [
            ("Apple", 0.9, (10, 10, 40, 40)),
            ("Apple", 0.88, (50, 10, 80, 40)),
            ("Apple", 0.85, (90, 10, 120, 40)),
            ("Tomato", 0.91, (10, 50, 40, 80)),
            ("Tomato", 0.87, (50, 50, 80, 80)),
            ("Carrot", 0.93, (90, 50, 120, 80)),
        ]
    )
    cls_a = _mock_classify_model({"Golden": 0.9, "Gala": 0.1})
    cls_b = _mock_classify_model({"Cherry": 0.91, "Roma": 0.05})
    call_counts = {"cls-a": 0, "cls-b": 0, "det": 0}

    paths = {
        "det": str((runs / "det-1" / "weights" / "best.pt").resolve()),
        "a": str((runs / "cls-a" / "weights" / "best.pt").resolve()),
        "b": str((runs / "cls-b" / "weights" / "best.pt").resolve()),
    }
    real_models = {paths["det"]: det_model, paths["a"]: cls_a, paths["b"]: cls_b}

    def factory(path: str):
        key = str(Path(path).resolve())
        if key == paths["det"]:
            call_counts["det"] += 1
        elif key == paths["a"]:
            call_counts["cls-a"] += 1
        elif key == paths["b"]:
            call_counts["cls-b"] += 1
        return real_models[key]

    cache: dict = {}
    result = run_pipeline(
        _blank_image(),
        config,
        model_factory=factory,
        model_cache=cache,
    )
    assert len(result.items) == 6
    apples = [i for i in result.items if i.detection.class_name == "Apple"]
    tomatoes = [i for i in result.items if i.detection.class_name == "Tomato"]
    carrots = [i for i in result.items if i.detection.class_name == "Carrot"]
    assert len(apples) == 3 and all(a.refined for a in apples)
    assert all(a.classification and a.classification.class_name == "Golden" for a in apples)
    assert len(tomatoes) == 2 and all(
        t.classification and t.classification.class_name == "Cherry" for t in tomatoes
    )
    assert len(carrots) == 1 and carrots[0].refined is False
    # Factory called once per unique weights via cache (preload + use).
    assert call_counts["cls-a"] == 1
    assert call_counts["cls-b"] == 1
    assert call_counts["det"] == 1
    assert len(cache) == 3


def test_unknown_and_uncertain(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-1", task="classify")

    config = PipelineConfig(
        pipeline_id="p1",
        name="Reject",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "Apple": ClassMapping(
                enabled=True,
                classifier_run_id="cls-1",
                confidence_threshold=0.80,
                margin_threshold=0.10,
            )
        },
    )

    # First object: low confidence → unknown
    # Second: high but tight margin → uncertain
    # We run twice with different classify models by mutating predict side_effect.

    det_model = _mock_detect_model([("Apple", 0.9, (10, 10, 50, 50))])
    unknown_model = _mock_classify_model({"Golden": 0.55, "Gala": 0.40})
    uncertain_model = _mock_classify_model({"Golden": 0.82, "Gala": 0.78})

    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())
    cls_path = str((runs / "cls-1" / "weights" / "best.pt").resolve())

    result_u = run_pipeline(
        _blank_image(),
        config,
        model_factory=lambda p: det_model if Path(p).resolve() == Path(det_path) else unknown_model,
        model_cache={},
    )
    assert result_u.items[0].classification is not None
    assert result_u.items[0].classification.status == "unknown"
    assert "INCONNU" in result_u.items[0].display_label

    result_i = run_pipeline(
        _blank_image(),
        config,
        model_factory=lambda p: (
            det_model if Path(p).resolve() == Path(det_path) else uncertain_model
        ),
        model_cache={},
    )
    assert result_i.items[0].classification is not None
    assert result_i.items[0].classification.status == "uncertain"
    assert "INCERTAIN" in result_i.items[0].display_label


def test_missing_classifier_keeps_detection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")

    config = PipelineConfig(
        pipeline_id="p1",
        name="Missing cls",
        detector_run_id="det-1",
        mappings={
            "Apple": ClassMapping(
                enabled=True,
                classifier_run_id="missing-cls",
                confidence_threshold=0.8,
            )
        },
    )
    det_model = _mock_detect_model([("Apple", 0.94, (10, 10, 50, 50))])
    result = run_pipeline(
        _blank_image(),
        config,
        model_factory=lambda _: det_model,
        model_cache={},
    )
    assert len(result.items) == 1
    assert result.items[0].detection.class_name == "Apple"
    assert result.items[0].classification is not None
    assert result.items[0].classification.status == "error"
    assert result.warnings


def test_classifier_error_keeps_detection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-1", task="classify")

    config = PipelineConfig(
        pipeline_id="p1",
        name="Boom",
        detector_run_id="det-1",
        crop_padding=0.0,
        mappings={
            "Apple": ClassMapping(enabled=True, classifier_run_id="cls-1")
        },
    )
    det_model = _mock_detect_model([("Apple", 0.9, (10, 10, 50, 50))])
    cls_model = MagicMock()
    cls_model.names = {0: "x"}
    cls_model.predict.side_effect = RuntimeError("cuda exploded")

    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())
    cls_path = str((runs / "cls-1" / "weights" / "best.pt").resolve())

    def factory(path: str):
        return det_model if str(Path(path).resolve()) == det_path else cls_model

    result = run_pipeline(
        _blank_image(),
        config,
        model_factory=factory,
        model_cache={},
    )
    assert result.items[0].detection.confidence == pytest.approx(0.9)
    assert result.items[0].classification is not None
    assert result.items[0].classification.status == "error"


def test_detector_missing_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    config = PipelineConfig(
        pipeline_id="p1",
        name="X",
        detector_run_id="gone",
        mappings={},
    )
    with pytest.raises(PipelineEngineError):
        run_pipeline(_blank_image(), config, model_factory=lambda _: MagicMock())


def test_detector_classes_changed_warning(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "data"))
    runs = tmp_path / "data" / "runs"
    _write_run(runs, "det-1", task="detect")
    _write_run(runs, "cls-1", task="classify")

    config = PipelineConfig(
        pipeline_id="p1",
        name="Legacy class",
        detector_run_id="det-1",
        mappings={
            "OldFruit": ClassMapping(enabled=True, classifier_run_id="cls-1"),
        },
    )
    det_model = _mock_detect_model([("Apple", 0.9, (10, 10, 50, 50))])
    cls_model = _mock_classify_model({"A": 0.9, "B": 0.1})
    det_path = str((runs / "det-1" / "weights" / "best.pt").resolve())
    cls_path = str((runs / "cls-1" / "weights" / "best.pt").resolve())

    def factory(path: str):
        return det_model if str(Path(path).resolve()) == det_path else cls_model

    result = run_pipeline(
        _blank_image(),
        config,
        model_factory=factory,
        model_cache={},
    )
    assert any("OldFruit" in w for w in result.warnings)
    assert result.items[0].refined is False


def test_result_table_rows_and_labels() -> None:
    det = Detection(0, "Apple", 0.94, 1, 2, 3, 4)
    item = EnrichedDetection(
        detection=det,
        refined=True,
        classification=None,
    )
    # without classification still refined flag — treat carefully
    rows = result_table_rows(
        __import__("vision_trainer.pipeline.models", fromlist=["PipelineResult"]).PipelineResult(
            items=[
                EnrichedDetection(detection=Detection(0, "Carrot", 0.93, 0, 0, 1, 1), refined=False)
            ]
        )
    )
    assert rows[0]["Affinement"] == "Non"
    assert rows[0]["Statut"] == "Non affiné"


def test_make_pipeline_id_stable_chars() -> None:
    pid = make_pipeline_id("Fruits détaillés!")
    assert " " not in pid
    assert pid


def test_get_cached_model_reuses() -> None:
    cache: dict = {}
    created = []

    def factory(path: str):
        created.append(path)
        return MagicMock(name=path)

    # Use a real temp file path for resolve()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        path = handle.name
        a = get_cached_model(path, cache=cache, model_factory=factory)
        b = get_cached_model(path, cache=cache, model_factory=factory)
        assert a is b
        assert len(created) == 1


def test_paths_pipelines_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "d"))
    from vision_trainer.paths import get_pipelines_dir

    path = get_pipelines_dir()
    assert path.is_dir()
    assert path.name == "pipelines"
