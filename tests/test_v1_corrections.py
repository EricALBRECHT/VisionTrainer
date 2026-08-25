from __future__ import annotations

import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from PIL import Image

from vision_trainer.datasets.store import import_zip_to_persistent_dataset
from vision_trainer.inference.uploads import display_upload_name, safe_internal_upload_path
from vision_trainer.io_utils import atomic_write_json
from vision_trainer.results.catalog import load_run
from vision_trainer.results.models import ULTRALYTICS_PLOT_FILES
from vision_trainer.training.status import (
    RunStatus,
    attach_worker_pid,
    extract_metrics_from_trainer,
    read_status,
    should_auto_refresh,
    write_status,
)
from vision_trainer.training.trainer import (
    TrainingError,
    TrainingRequest,
    execute_training_from_run_dir,
    prepare_training_run,
    start_training_subprocess,
)
from vision_trainer.yolo.models import DatasetInfo, SampleAnnotation, SplitInfo
from vision_trainer.yolo.parser import (
    MAX_ZIP_MEMBERS,
    ZipExtractionError,
    extract_zip_dataset,
    load_dataset_from_directory,
)
from vision_trainer.yolo.validator import validate_dataset
from vision_trainer.yolo.visualization import draw_annotations


def _dataset(tmp_path: Path) -> DatasetInfo:
    root = tmp_path / "dataset"
    train_images = root / "train" / "images"
    train_images.mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    Image.new("RGB", (16, 16), (1, 2, 3)).save(train_images / "a.jpg")
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
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


def test_safe_internal_upload_path_blocks_traversal(tmp_path: Path) -> None:
    dest = tmp_path / "uploads"
    dest.mkdir()
    for original in ("../../target.jpg", "/etc/passwd.jpg", "normal photo.JPG"):
        path = safe_internal_upload_path(dest, original)
        assert path.parent == dest.resolve()
        assert path.name.startswith("upload_")
        assert ".." not in path.name
    assert display_upload_name("../../target.jpg") == "target.jpg"
    assert display_upload_name("/tmp/abs/photo.png") == "photo.png"
    assert display_upload_name("normal.jpg") == "normal.jpg"


def test_atomic_status_write_and_invalid_read(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status = RunStatus(run_id="run", state="created", model="YOLO11n", epochs_total=1)
    write_status(run_dir, status)
    loaded = read_status(run_dir)
    assert loaded is not None
    assert loaded.state == "created"

    (run_dir / "status.json").write_text("{incomplete", encoding="utf-8")
    assert read_status(run_dir) is None

    atomic_write_json(run_dir / "request.json", {"ok": True})
    assert json.loads((run_dir / "request.json").read_text(encoding="utf-8"))["ok"] is True


def test_attach_worker_pid_does_not_clobber_failed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_status(
        run_dir,
        RunStatus(
            run_id="run",
            state="failed",
            model="YOLO11n",
            epochs_total=1,
            error_message="boom",
        ),
    )
    result = attach_worker_pid(run_dir, pid=1234)
    assert result is not None
    assert result.state == "failed"
    assert result.error_message == "boom"
    assert result.pid is None


def test_worker_immediate_failure_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "artifacts" / "runs"
    prepared = prepare_training_run(
        TrainingRequest(dataset=_dataset(tmp_path), runs_root=runs_root)
    )

    fake_proc = MagicMock()
    fake_proc.pid = 5555

    def _popen(*args, **kwargs):
        # Worker fails before parent attaches pid.
        write_status(
            prepared.run_dir,
            RunStatus(
                run_id=prepared.run_id,
                state="failed",
                model="YOLO11n",
                epochs_total=3,
                error_message="worker died",
            ),
        )
        return fake_proc

    with patch("subprocess.Popen", side_effect=_popen):
        start_training_subprocess(prepared.run_dir, python_executable="python")

    status = read_status(prepared.run_dir)
    assert status is not None
    assert status.state == "failed"
    assert status.error_message == "worker died"


def test_completed_requires_best_pt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "artifacts" / "runs"
    prepared = prepare_training_run(
        TrainingRequest(dataset=_dataset(tmp_path), runs_root=runs_root)
    )
    mock_model = MagicMock()
    mock_model.add_callback = MagicMock()
    mock_model.train = MagicMock()  # does not create best.pt

    with pytest.raises(TrainingError, match="best.pt"):
        execute_training_from_run_dir(prepared.run_dir, yolo_factory=lambda _: mock_model)

    status = read_status(prepared.run_dir)
    assert status is not None
    assert status.state == "failed"
    assert "best.pt" in (status.error_message or "")


def test_double_prepare_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )
    runs_root = tmp_path / "artifacts" / "runs"
    dataset = _dataset(tmp_path)
    prepare_training_run(TrainingRequest(dataset=dataset, runs_root=runs_root))
    with pytest.raises(TrainingError, match="déjà"):
        prepare_training_run(TrainingRequest(dataset=dataset, runs_root=runs_root))


def test_persistent_dataset_import(tmp_path: Path) -> None:
    zip_path = tmp_path / "ds.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "data.yaml",
            yaml.dump({"names": ["a"], "train": "train/images"}),
        )
        # minimal image+label
    # Build proper zip with image
    root = tmp_path / "src"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    Image.new("RGB", (8, 8), (1, 1, 1)).save(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        yaml.dump({"names": ["a"], "train": "train/images"}),
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())

    dataset_id, extract_dir = import_zip_to_persistent_dataset(
        zip_path.read_bytes(),
        datasets_root=tmp_path / "artifacts" / "datasets",
    )
    assert dataset_id
    assert extract_dir.is_dir()
    assert (extract_dir / "data.yaml").is_file()
    assert str(extract_dir).startswith(str((tmp_path / "artifacts" / "datasets").resolve()))


def test_zip_member_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vision_trainer.yolo.parser.MAX_ZIP_MEMBERS", 2)
    zip_path = tmp_path / "many.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("a.txt", "1")
        archive.writestr("b.txt", "2")
        archive.writestr("c.txt", "3")
    with pytest.raises(ZipExtractionError, match="entrées"):
        extract_zip_dataset(zip_path, tmp_path / "out")


def _minimal_internal_dataset(root: Path) -> Path:
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    Image.new("RGB", (8, 8), (1, 1, 1)).save(train_images / "a.jpg")
    (train_labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "train: train/images\nnames: ['a']\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("created", True),
        ("running", True),
        ("completed", False),
        ("failed", False),
        ("interrupted", False),
        (None, False),
    ],
)
def test_should_auto_refresh_only_for_active_states(state: str | None, expected: bool) -> None:
    assert should_auto_refresh(state) is expected


def test_precision_metric_reads_ultralytics_precision_b_column() -> None:
    """Precision displayed for run légumes comes from metrics/precision(B), not another field."""
    trainer = MagicMock()
    trainer.metrics = {
        "metrics/precision(B)": 0.00623,
        "metrics/recall(B)": 0.74444,
        "metrics/mAP50(B)": 0.67695,
        "metrics/mAP50-95(B)": 0.43156,
        # distractors that must not win over precision(B)
        "precision": 0.99,
        "train/box_loss": 1.25,
    }
    metrics = extract_metrics_from_trainer(trainer)
    assert metrics.precision == pytest.approx(0.00623)
    assert metrics.recall == pytest.approx(0.74444)
    assert metrics.map50 == pytest.approx(0.67695)
    assert metrics.map50_95 == pytest.approx(0.43156)


def test_validate_dataset_accepts_containment_root(tmp_path: Path) -> None:
    dataset_path = _minimal_internal_dataset(tmp_path / "extracted")
    result = validate_dataset(dataset_path, containment_root=dataset_path)
    assert result.dataset is not None
    assert result.is_valid


def test_validate_dataset_without_containment_root_still_works(tmp_path: Path) -> None:
    dataset_path = _minimal_internal_dataset(tmp_path / "extracted")
    result = validate_dataset(dataset_path)
    assert result.dataset is not None
    assert result.is_valid


def test_split_confinement_rejects_external_absolute(tmp_path: Path) -> None:
    extract = tmp_path / "extracted"
    extract.mkdir()
    outside = tmp_path / "outside" / "images"
    outside.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(outside / "x.jpg")
    (extract / "data.yaml").write_text(
        yaml.dump(
            {
                "names": ["a"],
                "train": str(outside),
            }
        ),
        encoding="utf-8",
    )
    dataset, errors = load_dataset_from_directory(extract, containment_root=extract)
    assert dataset is None
    assert any("sort" in error.lower() or "extérieur" in error.lower() or "racine" in error.lower() for error in errors)

    result = validate_dataset(extract, containment_root=extract)
    assert result.dataset is None
    assert any(
        "sort" in issue.message.lower() or "racine" in issue.message.lower()
        for issue in result.errors
    )


def test_split_confinement_rejects_parent_escape(tmp_path: Path) -> None:
    extract = tmp_path / "extracted"
    extract.mkdir()
    outside = tmp_path / "outside" / "images"
    outside.mkdir(parents=True)
    Image.new("RGB", (8, 8), (0, 0, 0)).save(outside / "x.jpg")
    (extract / "data.yaml").write_text(
        "train: ../../outside/images\nnames: ['a']\n",
        encoding="utf-8",
    )
    result = validate_dataset(extract, containment_root=extract)
    assert result.dataset is None
    assert any(
        "sort" in issue.message.lower() or "racine" in issue.message.lower()
        for issue in result.errors
    )


def test_roboflow_parent_fallback_still_works_under_containment(tmp_path: Path) -> None:
    extract = tmp_path / "extracted"
    for split in ("train", "valid"):
        (extract / split / "images").mkdir(parents=True)
        (extract / split / "labels").mkdir(parents=True)
        Image.new("RGB", (8, 8), (2, 2, 2)).save(extract / split / "images" / "a.jpg")
        (extract / split / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (extract / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\nnames: ['pothole']\n",
        encoding="utf-8",
    )
    dataset, errors = load_dataset_from_directory(extract, containment_root=extract)
    assert errors == []
    assert dataset is not None
    assert dataset.splits["train"].images_dir == (extract / "train" / "images").resolve()
    assert dataset.splits["val"].resolved_via_fallback is True

    result = validate_dataset(extract, containment_root=extract)
    assert result.is_valid, [issue.message for issue in result.errors]
    assert result.dataset is not None
    assert result.dataset.splits["train"].images_dir == (extract / "train" / "images").resolve()
    assert result.dataset.splits["val"].resolved_via_fallback is True


def test_box_plot_names_supported(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"x")
    for name in ("BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png"):
        Image.new("RGB", (4, 4), (9, 9, 9)).save(run_dir / name)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run", "state": "completed"}),
        encoding="utf-8",
    )
    detail = load_run(run_dir)
    for name in ("BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png"):
        assert name in ULTRALYTICS_PLOT_FILES
        assert name in detail.plots


def test_label_background_has_positive_height(tmp_path: Path) -> None:
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 80), (200, 200, 200)).save(image_path)
    annotated = draw_annotations(
        image_path,
        [
            SampleAnnotation(
                class_id=0,
                class_name="pothole",
                x_center=0.5,
                y_center=0.5,
                width=0.4,
                height=0.4,
            )
        ],
    )
    # Background fill should change some pixels near the top of the box.
    assert annotated.size == (100, 80)
    assert annotated.getpixel((30, 20)) != (200, 200, 200) or annotated.getpixel((50, 25)) != (
        200,
        200,
        200,
    )


def test_max_zip_members_constant_is_generous() -> None:
    assert MAX_ZIP_MEMBERS >= 100_000
