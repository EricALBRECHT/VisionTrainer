"""Tests for named best.pt export copies (user-facing filenames)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.results.catalog import load_run
from vision_trainer.training.export_weights import (
    build_export_weights_filename,
    export_named_best_weights,
    find_export_weights,
    resolve_export_dataset_name,
    sanitize_dataset_slug,
)
from vision_trainer.training.status import RunStatus, read_status, write_status
from vision_trainer.training.trainer import (
    TrainingError,
    TrainingRequest,
    execute_training_from_run_dir,
    prepare_training_run,
)
from vision_trainer.yolo.models import DatasetInfo, SplitInfo


def _detect_dataset(tmp_path: Path, *, folder_name: str = "archive_legumes") -> DatasetInfo:
    root = tmp_path / folder_name
    train_images = root / "train" / "images"
    train_images.mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    return DatasetInfo(
        root=root,
        yaml_path=root / "data.yaml",
        class_names={0: "tomato"},
        splits={
            "train": SplitInfo(
                name="train",
                images_dir=train_images,
                labels_dir=root / "train" / "labels",
                image_count=1,
            )
        },
    )


def test_sanitize_spaces_accents_and_specials() -> None:
    assert sanitize_dataset_slug("Mes médicaments 2026") == "mes_medicaments_2026"
    assert sanitize_dataset_slug("archive_legumes") == "archive_legumes"
    assert sanitize_dataset_slug("A/B\\C") == "c"
    assert sanitize_dataset_slug("../etc/passwd") == "passwd"
    assert sanitize_dataset_slug(".../../../x") == "x"
    assert sanitize_dataset_slug("@@@") == "dataset"
    assert sanitize_dataset_slug("") == "dataset"
    assert sanitize_dataset_slug(None) == "dataset"
    long = "a" * 200
    assert len(sanitize_dataset_slug(long)) == 80
    assert "/" not in sanitize_dataset_slug("foo/bar")
    assert "\\" not in sanitize_dataset_slug("foo\\bar")


def test_build_export_filenames_for_tasks() -> None:
    assert (
        build_export_weights_filename("archive_legumes", "detect")
        == "archive_legumes_detect_best.pt"
    )
    assert (
        build_export_weights_filename("archive_legumes", "classify")
        == "archive_legumes_classify_best.pt"
    )
    assert (
        build_export_weights_filename("archive_legumes", "segment")
        == "archive_legumes_segment_best.pt"
    )
    assert (
        build_export_weights_filename("Mes médicaments 2026", "classify")
        == "mes_medicaments_2026_classify_best.pt"
    )


def test_resolve_dataset_name_zip_external_and_legacy() -> None:
    assert resolve_export_dataset_name({"dataset_name": "archive_legumes"}) == "archive_legumes"
    assert (
        resolve_export_dataset_name({"dataset_root": "/datasets/archive_legumes"})
        == "archive_legumes"
    )
    assert (
        resolve_export_dataset_name({"data_dir": "/data/uploads/zip_extract/my_cls"})
        == "my_cls"
    )
    assert resolve_export_dataset_name({}) == "dataset"
    assert resolve_export_dataset_name(None) == "dataset"


def test_export_copy_keeps_best_and_matches_bytes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    best = weights / "best.pt"
    best.write_bytes(b"canonical-weights")

    export = export_named_best_weights(
        run_dir,
        dataset_name="archive_legumes",
        task="detect",
        best_path=best,
    )
    assert export is not None
    assert export.name == "archive_legumes_detect_best.pt"
    assert best.is_file()
    assert best.read_bytes() == b"canonical-weights"
    assert export.read_bytes() == best.read_bytes()
    assert export.parent == weights.resolve()


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("detect", "archive_legumes_detect_best.pt"),
        ("classify", "archive_legumes_classify_best.pt"),
        ("segment", "archive_legumes_segment_best.pt"),
    ],
)
def test_export_named_for_each_task(tmp_path: Path, task: str, expected: str) -> None:
    run_dir = tmp_path / f"run-{task}"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"w")
    export = export_named_best_weights(
        run_dir,
        dataset_name="archive_legumes",
        task=task,
    )
    assert export is not None
    assert export.name == expected


def test_export_skipped_when_best_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    (run_dir / "weights").mkdir(parents=True)
    assert (
        export_named_best_weights(run_dir, dataset_name="x", task="detect") is None
    )


def test_path_traversal_cannot_escape_weights_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"safe")
    export = export_named_best_weights(
        run_dir,
        dataset_name="../../evil",
        task="detect",
    )
    assert export is not None
    assert export.parent == weights.resolve()
    assert export.name == "evil_detect_best.pt"
    assert not (tmp_path / "evil_detect_best.pt").exists()


def test_failed_training_does_not_create_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "runs"
    prepared = prepare_training_run(
        TrainingRequest(
            dataset=_detect_dataset(tmp_path),
            runs_root=runs_root,
            epochs=1,
            imgsz=640,
            batch=8,
            dataset_name="archive_legumes",
        )
    )
    mock_model = MagicMock()
    mock_model.train.side_effect = RuntimeError("boom")

    with pytest.raises(TrainingError):
        execute_training_from_run_dir(
            prepared.run_dir,
            yolo_factory=lambda _: mock_model,
        )

    status = read_status(prepared.run_dir)
    assert status is not None
    assert status.state == "failed"
    assert status.export_model_path is None
    weights_dir = prepared.run_dir / "weights"
    if weights_dir.is_dir():
        assert list(weights_dir.glob("*_*_best.pt")) == []
    else:
        assert not (prepared.run_dir / "archive_legumes_detect_best.pt").exists()


def test_successful_training_writes_export_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "runs"
    prepared = prepare_training_run(
        TrainingRequest(
            dataset=_detect_dataset(tmp_path, folder_name="archive_legumes"),
            runs_root=runs_root,
            epochs=1,
            imgsz=640,
            batch=8,
            dataset_name="archive_legumes",
        )
    )

    mock_model = MagicMock()

    def _train(**_kwargs):
        weights = prepared.run_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_bytes(b"best-bytes")
        (weights / "last.pt").write_bytes(b"last-bytes")

    mock_model.train.side_effect = _train
    final = execute_training_from_run_dir(
        prepared.run_dir,
        yolo_factory=lambda _: mock_model,
    )
    assert final.state == "completed"
    best = prepared.run_dir / "weights" / "best.pt"
    export = Path(final.export_model_path or "")
    assert best.is_file()
    assert export.name == "archive_legumes_detect_best.pt"
    assert export.read_bytes() == best.read_bytes() == b"best-bytes"
    assert final.best_model_path == str(best)


def test_external_dataset_name_from_root(tmp_path: Path) -> None:
    assert (
        resolve_export_dataset_name(
            {"dataset_root": str(tmp_path / "datasets" / "archive_legumes")}
        )
        == "archive_legumes"
    )
    assert (
        build_export_weights_filename("archive_legumes", "segment")
        == "archive_legumes_segment_best.pt"
    )


def test_legacy_run_without_export_still_loads(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"old")
    write_status(
        run_dir,
        RunStatus(run_id="legacy-run", state="completed", task="detect"),
    )
    detail = load_run(run_dir)
    assert detail.best_pt is not None
    assert detail.export_pt is None
    models = discover_trained_models(tmp_path)
    assert models[0].label.endswith("best.pt")
    assert Path(models[0].weights_path).name == "best.pt"


def test_discovery_prefers_export_label(tmp_path: Path) -> None:
    run_dir = tmp_path / "named-run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"b")
    export = weights / "mes_medicaments_2026_classify_best.pt"
    export.write_bytes(b"b")
    write_status(
        run_dir,
        RunStatus(
            run_id="named-run",
            state="completed",
            task="classify",
            best_model_path=str(weights / "best.pt"),
            export_model_path=str(export),
        ),
    )
    models = discover_trained_models(tmp_path, task="classify")
    assert len(models) == 1
    assert models[0].label.endswith("mes_medicaments_2026_classify_best.pt")
    assert Path(models[0].weights_path).name == "best.pt"


def test_find_export_weights_from_disk_scan(tmp_path: Path) -> None:
    run_dir = tmp_path / "scan"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"b")
    named = weights / "foo_segment_best.pt"
    named.write_bytes(b"b")
    found = find_export_weights(run_dir)
    assert found == named
