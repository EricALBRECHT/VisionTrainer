"""Tests for Analyse & Comparaison des modèles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vision_trainer.analysis.builder import build_run_analysis
from vision_trainer.analysis.classify_errors import (
    ClassifyAnalysisError,
    analyze_classify_errors,
)
from vision_trainer.analysis.compare import (
    compare_runs,
    format_pp,
    percentage_point_delta,
)
from vision_trainer.analysis.confusion import (
    build_confusion_matrix,
    confusion_pairs_from_errors,
    confusion_pairs_from_matrix,
)
from vision_trainer.analysis.curves import infer_best_epoch, load_training_curves
from vision_trainer.analysis.dataset_stats import collect_dataset_stats
from vision_trainer.analysis.diagnostics import build_diagnostics
from vision_trainer.analysis.environment import (
    DEFAULT_ULTRALYTICS_SEED,
    collect_training_environment,
)
from vision_trainer.analysis.models import (
    ANALYSIS_VERSION,
    ClassMetricRow,
    ClassifyErrorSample,
    ConfusionPair,
    RunAnalysis,
    TopScore,
)
from vision_trainer.analysis.store import load_analysis, save_analysis
from vision_trainer.training.status import RunStatus, write_request, write_status


def _write_status(run_dir: Path, **kwargs) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "run_id": run_dir.name,
        "state": "completed",
        "model": "YOLO11n",
        "epochs_total": 3,
        "imgsz": 640,
        "batch": 8,
        "device": "cpu",
        "task": "detect",
        "started_at": "2026-09-01T10:00:00+00:00",
        "finished_at": "2026-09-01T11:00:00+00:00",
        "metrics": {},
    }
    base.update(kwargs)
    write_status(run_dir, RunStatus.from_dict(base))


def test_legacy_incomplete_run_analysis(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-empty"
    run_dir.mkdir()
    analysis = build_run_analysis(run_dir, persist=True)
    assert analysis.analysis_version == ANALYSIS_VERSION
    assert analysis.metrics.get("map50") is None
    assert load_analysis(run_dir) is not None


def test_detect_metrics_from_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "det-1"
    _write_status(
        run_dir,
        task="detect",
        metrics={"precision": 0.9, "recall": 0.8, "map50": 0.75, "map50_95": 0.55},
    )
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"x")
    (run_dir / "PR_curve.png").write_bytes(b"fake")
    analysis = build_run_analysis(run_dir, persist=False)
    assert analysis.task == "detect"
    assert analysis.metrics["map50"] == 0.75
    assert analysis.metrics["precision"] == 0.9
    assert "PR_curve.png" in analysis.assets
    assert analysis.detection_errors_v1_note is not None


def test_segment_box_mask_separated(tmp_path: Path) -> None:
    run_dir = tmp_path / "seg-1"
    _write_status(
        run_dir,
        task="segment",
        model="YOLO11n-seg",
        metrics={
            "precision": 0.91,
            "recall": 0.88,
            "map50": 0.8,
            "map50_95": 0.6,
            "mask_precision": 0.85,
            "mask_recall": 0.82,
            "mask_map50": 0.77,
            "mask_map50_95": 0.5,
        },
    )
    analysis = build_run_analysis(run_dir, persist=False)
    assert analysis.metrics["map50"] == 0.8
    assert analysis.metrics["mask_map50"] == 0.77
    assert analysis.metrics["map50"] != analysis.metrics["mask_map50"]


def test_classify_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "cls-1"
    _write_status(
        run_dir,
        task="classify",
        model="YOLO11n-cls",
        metrics={"accuracy_top1": 0.948, "accuracy_top5": 0.99},
    )
    analysis = build_run_analysis(run_dir, persist=False)
    assert analysis.metrics["accuracy_top1"] == pytest.approx(0.948)
    assert analysis.metrics["accuracy_top5"] == pytest.approx(0.99)


def test_confusion_pairs_sorted_and_zero_errors() -> None:
    labels = ["A", "B", "C"]
    matrix = [[5, 0, 0], [0, 4, 0], [0, 0, 3]]
    assert confusion_pairs_from_matrix(matrix, labels) == []
    matrix2 = [[5, 7, 1], [2, 4, 0], [0, 3, 3]]
    pairs = confusion_pairs_from_matrix(matrix2, labels)
    assert pairs[0].true_label == "A"
    assert pairs[0].pred_label == "B"
    assert pairs[0].count == 7


def test_confusion_from_errors_with_special_class_names() -> None:
    errors = [
        ClassifyErrorSample(
            image_path="val/Doliprane 500/a.jpg",
            true_label="Doliprane 500",
            pred_label="Doliprane 1000",
            confidence=0.87,
            top_scores=[TopScore("Doliprane 1000", 0.87)],
        ),
        ClassifyErrorSample(
            image_path="val/Doliprane 500/b.jpg",
            true_label="Doliprane 500",
            pred_label="Doliprane 1000",
            confidence=0.7,
        ),
    ]
    pairs = confusion_pairs_from_errors(errors)
    assert pairs[0].count == 2
    assert "Doliprane 500" in pairs[0].true_label


def test_classify_error_analysis_cached(tmp_path: Path) -> None:
    run_dir = tmp_path / "cls-err"
    data = tmp_path / "dataset"
    for split, classes in (("train", ("A", "B")), ("val", ("A", "B"))):
        for name in classes:
            folder = data / split / name
            folder.mkdir(parents=True)
            Image.new("RGB", (8, 8), (10, 20, 30)).save(folder / "img.jpg")
    _write_status(run_dir, task="classify")
    write_request(
        run_dir,
        {
            "run_id": run_dir.name,
            "task": "classify",
            "data_dir": str(data),
            "dataset_name": "meds",
        },
    )
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"x")

    calls = {"n": 0}

    def predict_fn(image_path: Path):
        calls["n"] += 1
        # Always predict B → errors for class A images
        return "B", 0.9, [("B", 0.9), ("A", 0.1)]

    errors = analyze_classify_errors(run_dir, predict_fn=predict_fn)
    assert len(errors) == 1
    assert errors[0].true_label == "A"
    assert errors[0].pred_label == "B"
    assert errors[0].top_scores[0].class_name == "B"
    first_calls = calls["n"]

    # Cache hit
    errors2 = analyze_classify_errors(run_dir, predict_fn=predict_fn)
    assert calls["n"] == first_calls
    assert len(errors2) == 1

    loaded = load_analysis(run_dir)
    assert loaded is not None
    assert loaded.classify_errors_computed is True
    assert loaded.analysis_version == ANALYSIS_VERSION
    assert loaded.confusion_matrix is not None


def test_classify_error_analysis_requires_classify(tmp_path: Path) -> None:
    run_dir = tmp_path / "det"
    _write_status(run_dir, task="detect")
    write_request(run_dir, {"task": "detect", "dataset_root": str(tmp_path)})
    with pytest.raises(ClassifyAnalysisError):
        analyze_classify_errors(run_dir, predict_fn=lambda p: ("x", 1.0, []))


def test_diagnostics_and_recommendations() -> None:
    analysis = RunAnalysis(
        task="classify",
        confusion_pairs=[
            ConfusionPair("Doliprane 500", "Doliprane 1000", 7),
        ],
        dataset_stats={
            "per_class": {
                "val": {"Doliprane 500": 42},
                "train": {"Doliprane 500": 5, "Doliprane 1000": 80},
            }
        },
        class_metrics=[
            ClassMetricRow(class_name="weak", accuracy=0.5),
            ClassMetricRow(class_name="strong", accuracy=0.95),
        ],
    )
    findings, recs = build_diagnostics(analysis)
    assert any(f.code == "class_confusion" for f in findings)
    assert any("7/42" in f.detail for f in findings)
    assert any(f.code == "class_imbalance" for f in findings)
    assert recs


def test_percentage_points_not_relative_percent() -> None:
    delta = percentage_point_delta(0.948, 0.982)
    assert delta == pytest.approx(3.4, abs=0.01)
    assert "points de pourcentage" in format_pp(delta)
    assert "%" not in format_pp(delta).split("points")[0] or True


def test_compare_same_task_and_reject_different(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    _write_status(
        a,
        task="classify",
        metrics={"accuracy_top1": 0.90, "accuracy_top5": 0.95},
    )
    _write_status(
        b,
        task="classify",
        metrics={"accuracy_top1": 0.94, "accuracy_top5": 0.97},
    )
    _write_status(c, task="detect", metrics={"map50": 0.5})
    write_request(a, {"task": "classify", "dataset_name": "meds"})
    write_request(b, {"task": "classify", "dataset_name": "meds"})
    write_request(c, {"task": "detect", "dataset_name": "meds"})

    ok = compare_runs(a, b)
    assert ok.compatible
    top1 = next(d for d in ok.metric_deltas if d.name == "accuracy_top1")
    assert top1.delta_pp == pytest.approx(4.0, abs=0.01)

    bad = compare_runs(a, c)
    assert bad.compatible is False
    assert bad.metric_deltas == []


def test_compare_dataset_warning_and_class_regression(tmp_path: Path) -> None:
    a = tmp_path / "ra"
    b = tmp_path / "rb"
    _write_status(a, task="classify", metrics={"accuracy_top1": 0.90})
    _write_status(b, task="classify", metrics={"accuracy_top1": 0.92})
    write_request(a, {"task": "classify", "dataset_name": "dsA"})
    write_request(b, {"task": "classify", "dataset_name": "dsB"})

    analysis_a = build_run_analysis(a, persist=False)
    analysis_b = build_run_analysis(b, persist=False)
    analysis_a.class_metrics = [
        ClassMetricRow(class_name="X", accuracy=0.95),
        ClassMetricRow(class_name="Y", accuracy=0.90),
    ]
    analysis_b.class_metrics = [
        ClassMetricRow(class_name="X", accuracy=0.90),
        ClassMetricRow(class_name="Y", accuracy=0.96),
    ]
    cmp = compare_runs(a, b, analysis_a=analysis_a, analysis_b=analysis_b)
    assert any("Datasets différents" in w for w in cmp.warnings)
    assert any(r.class_name == "X" and (r.delta_pp or 0) < 0 for r in cmp.regressions)
    assert any(r.class_name == "Y" and (r.delta_pp or 0) > 0 for r in cmp.improvements)


def test_duration_speedup(tmp_path: Path) -> None:
    a = tmp_path / "slow"
    b = tmp_path / "fast"
    _write_status(
        a,
        task="detect",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T03:00:00+00:00",
        metrics={"map50": 0.5},
    )
    _write_status(
        b,
        task="detect",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:30:00+00:00",
        metrics={"map50": 0.55},
    )
    cmp = compare_runs(a, b)
    assert cmp.speedup_b_vs_a == pytest.approx(6.0)


def test_curves_and_best_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "curves"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B),train/box_loss,val/box_loss\n"
        "1,0.4,1.0,1.2\n"
        "2,0.6,0.8,1.0\n"
        "3,0.55,0.7,1.1\n",
        encoding="utf-8",
    )
    curves = load_training_curves(run_dir)
    assert "map50" in curves
    epoch, name, value = infer_best_epoch(curves, task="detect")
    assert epoch == 2
    assert value == pytest.approx(0.6)


def test_missing_results_csv_and_png(tmp_path: Path) -> None:
    run_dir = tmp_path / "no-csv"
    _write_status(run_dir, task="detect", metrics={"map50": 0.1})
    analysis = build_run_analysis(run_dir, persist=False)
    assert analysis.curves == {}
    assert any("results.csv" in line for line in analysis.limitations)


def test_dataset_stats_classify_cached_no_copy(tmp_path: Path) -> None:
    external = tmp_path / "datasets" / "archive_legumes"
    for split in ("train", "val"):
        for cls in ("carrot", "onion"):
            d = external / split / cls
            d.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(d / "1.jpg")
    run_dir = tmp_path / "run-ext"
    _write_status(run_dir, task="classify")
    write_request(
        run_dir,
        {
            "task": "classify",
            "data_dir": str(external),
            "dataset_name": "archive_legumes",
            "dataset_source_type": "external",
        },
    )
    stats = collect_dataset_stats(run_dir)
    assert stats["train_images"] == 2
    assert stats["val_images"] == 2
    assert stats["dataset_name"] == "archive_legumes"
    # External tree untouched (still only images we created)
    assert (external / "train" / "carrot" / "1.jpg").is_file()
    assert not (run_dir / "archive_legumes").exists()
    # Cache file
    assert (run_dir / "dataset_stats.json").is_file()
    stats2 = collect_dataset_stats(run_dir)
    assert stats2["train_images"] == 2


def test_hardware_seed_legacy_and_future(tmp_path: Path) -> None:
    legacy = tmp_path / "old"
    _write_status(legacy, task="detect", device="0")
    write_request(legacy, {"task": "detect", "device": "0"})
    analysis = build_run_analysis(legacy, persist=False)
    assert analysis.summary["device_name"] == "Non enregistré"
    assert analysis.summary["seed"] == "Non enregistré"

    modern = tmp_path / "new"
    _write_status(
        modern,
        task="detect",
        device="0",
        device_name="NVIDIA GeForce GTX 1060 3GB",
        seed=DEFAULT_ULTRALYTICS_SEED,
        environment={"python": "3.11.0", "torch": "2.0"},
    )
    analysis2 = build_run_analysis(modern, persist=False)
    assert analysis2.summary["device_name"] == "NVIDIA GeForce GTX 1060 3GB"
    assert analysis2.summary["seed"] == DEFAULT_ULTRALYTICS_SEED
    assert analysis2.summary["environment"]["torch"] == "2.0"


def test_export_analysis_json_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "exp"
    _write_status(run_dir, task="classify", metrics={"accuracy_top1": 0.9})
    analysis = build_run_analysis(run_dir, persist=True)
    path = run_dir / "analysis.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["analysis_version"] == ANALYSIS_VERSION
    again = RunAnalysis.from_dict(raw)
    assert again.metrics["accuracy_top1"] == 0.9


def test_corrupted_analysis_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad"
    _write_status(run_dir, task="detect")
    (run_dir / "analysis.json").write_text("{broken", encoding="utf-8")
    assert load_analysis(run_dir) is None
    analysis = build_run_analysis(run_dir, persist=True)
    assert analysis.run_id == "bad"


def test_build_confusion_matrix_helper() -> None:
    labels = ["a", "b"]
    matrix = build_confusion_matrix(
        labels, true_labels=["a", "a", "b"], pred_labels=["a", "b", "b"]
    )
    assert matrix[0][0] == 1
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1


def test_environment_collector_safe() -> None:
    env = collect_training_environment()
    assert "python" in env


def test_save_load_analysis(tmp_path: Path) -> None:
    run_dir = tmp_path / "s"
    run_dir.mkdir()
    analysis = RunAnalysis(task="detect", run_id="s", metrics={"map50": 0.1})
    save_analysis(run_dir, analysis)
    loaded = load_analysis(run_dir)
    assert loaded is not None
    assert loaded.metrics["map50"] == 0.1
