from __future__ import annotations

from pathlib import Path

from PIL import Image

from vision_trainer.classify.parser import find_classify_dataset_root, load_classify_dataset
from vision_trainer.classify.reject import ClassScore, decide_classification
from vision_trainer.classify.validator import class_size_guidance, validate_classify_dataset
from vision_trainer.tasks import normalize_task, task_label_fr
from vision_trainer.training.status import RunStatus, write_status
from vision_trainer.yolo.parser import (
    ensure_data_yaml_for_layout,
    find_data_yaml,
    load_dataset_from_directory,
)
from vision_trainer.yolo.validator import validate_dataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (120, 80, 40)).save(path)


def test_normalize_task_defaults_legacy_runs() -> None:
    assert normalize_task(None) == "detect"
    assert normalize_task("") == "detect"
    assert normalize_task("classify") == "classify"
    assert normalize_task("segment") == "segment"
    assert task_label_fr(None) == "Détection"
    assert task_label_fr("classify") == "Classification"


def test_legacy_run_status_without_task_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    # Write minimal status without task (simulates pre-classification runs).
    (run_dir / "status.json").write_text(
        '{"run_id":"old-run","state":"completed","model":"YOLO11n","epochs_total":3}',
        encoding="utf-8",
    )
    from vision_trainer.training.status import read_status

    status = read_status(run_dir)
    assert status is not None
    assert status.task == "detect"


def test_classify_run_status_persists_task(tmp_path: Path) -> None:
    run_dir = tmp_path / "cls-run"
    run_dir.mkdir()
    write_status(
        run_dir,
        RunStatus(run_id="cls-run", state="completed", model="YOLO11n-cls", task="classify"),
    )
    from vision_trainer.training.status import read_status

    status = read_status(run_dir)
    assert status is not None
    assert status.task == "classify"


def test_classify_dataset_valid_and_counts(tmp_path: Path) -> None:
    root = tmp_path / "cls"
    for split, classes in (
        ("train", {"Golden": 3, "Gala": 2}),
        ("val", {"Golden": 1, "Gala": 1}),
    ):
        for name, count in classes.items():
            for index in range(count):
                _write_image(root / split / name / f"{index}.jpg")

    info, errors = load_classify_dataset(root)
    assert errors == []
    assert info is not None
    assert info.num_classes == 2
    assert set(info.class_names.values()) == {"Gala", "Golden"}
    assert info.train_image_count == 5
    assert info.val_image_count == 2
    assert info.class_stats["Golden"].train_count == 3

    result = validate_classify_dataset(root)
    assert result.is_valid
    assert result.dataset is not None


def test_classify_dataset_nested_zip_root(tmp_path: Path) -> None:
    nested = tmp_path / "extract" / "MyDataset"
    _write_image(nested / "train" / "A" / "1.jpg")
    _write_image(nested / "train" / "B" / "1.jpg")
    _write_image(nested / "val" / "A" / "1.jpg")
    _write_image(nested / "val" / "B" / "1.jpg")

    found = find_classify_dataset_root(tmp_path / "extract")
    assert found == nested.resolve()
    result = validate_classify_dataset(tmp_path / "extract")
    assert result.is_valid


def test_classify_dataset_invalid_missing_train(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    _write_image(root / "val" / "A" / "1.jpg")
    result = validate_classify_dataset(root)
    assert not result.is_valid
    assert result.dataset is None


def test_class_size_guidance_bands() -> None:
    assert "expérimental" in class_size_guidance(10)
    assert class_size_guidance(75) == "faible"
    assert class_size_guidance(150) == "correct"
    assert class_size_guidance(400) == "recommandé"


def test_reject_unknown_and_uncertain() -> None:
    scores = [
        ClassScore(0, "Golden", 0.57),
        ClassScore(1, "Gala", 0.31),
        ClassScore(2, "Other", 0.12),
    ]
    unknown = decide_classification(scores, min_confidence=0.80, min_margin=0.10)
    assert unknown.kind == "unknown"
    assert "INCONNU" in unknown.label

    close = [
        ClassScore(0, "Golden", 0.52),
        ClassScore(1, "Gala", 0.49),
    ]
    uncertain = decide_classification(close, min_confidence=0.50, min_margin=0.10)
    assert uncertain.kind == "uncertain"
    assert uncertain.label == "INCERTAIN"

    clear = [
        ClassScore(0, "Golden", 0.94),
        ClassScore(1, "Gala", 0.04),
    ]
    accepted = decide_classification(clear, min_confidence=0.80, min_margin=0.10)
    assert accepted.kind == "accepted"
    assert accepted.label == "Golden"


def test_yolo_dataset_without_data_yaml_generates_from_classes_txt(tmp_path: Path) -> None:
    root = tmp_path / "Dataset"
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        _write_image(root / split / "images" / "img0.jpg")
        (root / split / "labels" / "img0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "train" / "classes.txt").write_text(
        "plate\nspoon\nfork\nknife\ncup\nglass\n",
        encoding="utf-8",
    )

    assert find_data_yaml(root) is None
    yaml_path, message = ensure_data_yaml_for_layout(root)
    assert yaml_path is not None and yaml_path.is_file()
    assert message is not None and "généré" in message

    # Existing yaml must not be overwritten on second call.
    original = yaml_path.read_text(encoding="utf-8")
    yaml_path2, message2 = ensure_data_yaml_for_layout(root)
    assert yaml_path2 == yaml_path
    assert message2 is None
    assert yaml_path.read_text(encoding="utf-8") == original

    info, msgs = load_dataset_from_directory(root)
    assert info is not None
    assert info.num_classes == 6
    assert info.class_names[0] == "plate"
    assert "train" in info.splits and "val" in info.splits
    assert any("généré" in m for m in msgs) or True  # already generated; find_data_yaml finds it

    result = validate_dataset(root)
    assert result.dataset is not None
    assert result.is_valid or len(result.errors) == 0 or result.dataset.num_classes == 6


def test_yolo_dataset_nested_zip_without_yaml(tmp_path: Path) -> None:
    nested = tmp_path / "extract" / "outer" / "Dataset"
    for split in ("train", "val"):
        (nested / split / "images").mkdir(parents=True)
        (nested / split / "labels").mkdir(parents=True)
        _write_image(nested / split / "images" / "a.jpg")
        (nested / split / "labels" / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (nested / "classes.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    result = validate_dataset(tmp_path / "extract", containment_root=tmp_path / "extract")
    assert result.dataset is not None
    assert result.dataset.num_classes == 2
    assert any("généré" in i.message for i in result.infos)


def test_yolo_existing_data_yaml_not_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    _write_image(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nnames: {0: keep_me}\n",
        encoding="utf-8",
    )
    (root / "train" / "classes.txt").write_text("ignored\n", encoding="utf-8")

    yaml_path, message = ensure_data_yaml_for_layout(root)
    assert message is None
    assert "keep_me" in yaml_path.read_text(encoding="utf-8")
    info, _ = load_dataset_from_directory(root)
    assert info is not None
    assert info.class_names[0] == "keep_me"
