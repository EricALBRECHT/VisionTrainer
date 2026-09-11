"""External / mounted datasets (no full copy)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from vision_trainer.classify.validator import validate_classify_dataset
from vision_trainer.datasets.external import (
    SOURCE_EXTERNAL,
    SOURCE_UPLOADED,
    ExternalDatasetError,
    assert_external_dataset_accessible,
    external_root_exists,
    is_under_root,
    list_external_dataset_dirs,
    normalize_source_type,
    register_external_dataset_meta,
    resolve_external_dataset_path,
)
from vision_trainer.segment.validator import validate_segment_dataset
from vision_trainer.training.data_yaml import write_resolved_data_yaml
from vision_trainer.training.session_dataset import dataset_to_session_payload
from vision_trainer.yolo.parser import ensure_data_yaml_for_layout, load_dataset_from_directory
from vision_trainer.yolo.validator import validate_dataset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (10, 20, 30)).save(path)


def _write_detect_dataset(root: Path, *, with_yaml: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_image(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _write_image(root / "val" / "images" / "b.jpg")
    (root / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "val" / "labels" / "b.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    (root / "classes.txt").write_text("objet\n", encoding="utf-8")
    if with_yaml:
        (root / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "train/images",
                    "val": "val/images",
                    "names": {0: "objet"},
                }
            ),
            encoding="utf-8",
        )
    return root


def _write_classify_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_image(root / "train" / "Pomme" / "1.jpg")
    _write_image(root / "train" / "Poire" / "1.jpg")
    _write_image(root / "val" / "Pomme" / "1.jpg")
    _write_image(root / "val" / "Poire" / "1.jpg")
    return root


def _write_segment_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _write_image(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    # polygon: class + at least 3 points
    (root / "train" / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.5 0.9\n",
        encoding="utf-8",
    )
    _write_image(root / "val" / "images" / "b.jpg")
    (root / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "val" / "labels" / "b.txt").write_text(
        "0 0.2 0.2 0.8 0.2 0.5 0.8\n",
        encoding="utf-8",
    )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train/images",
                "val": "val/images",
                "names": {0: "piece"},
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def external_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "datasets_mount"
    root.mkdir()
    monkeypatch.setenv("VISION_TRAINER_EXTERNAL_DATASETS", str(root))
    monkeypatch.setenv("VISIONTRAINER_DATA_DIR", str(tmp_path / "appdata"))
    return root


def test_external_root_exists_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setenv("VISION_TRAINER_EXTERNAL_DATASETS", str(missing))
    assert external_root_exists() is False

    missing.mkdir()
    assert external_root_exists() is True


def test_list_first_level_only(external_root: Path) -> None:
    (external_root / "Cuisine").mkdir()
    (external_root / "Cuisine" / "nested").mkdir()
    (external_root / "Fruits").mkdir()
    (external_root / "readme.txt").write_text("x", encoding="utf-8")
    (external_root / ".hidden").mkdir()
    names = [p.name for p in list_external_dataset_dirs()]
    assert names == ["Cuisine", "Fruits"]


def test_detect_external_valid(external_root: Path) -> None:
    ds = _write_detect_dataset(external_root / "TestDetect")
    path = resolve_external_dataset_path("TestDetect")
    assert path == ds.resolve()
    result = validate_dataset(path, containment_root=path)
    assert result.is_valid
    assert result.dataset is not None
    assert result.dataset.splits["train"].image_count == 1


def test_classify_external_valid(external_root: Path) -> None:
    ds = _write_classify_dataset(external_root / "TestClassify")
    result = validate_classify_dataset(ds, containment_root=ds)
    assert result.is_valid
    assert result.dataset is not None
    assert result.dataset.num_classes == 2


def test_segment_external_valid(external_root: Path) -> None:
    ds = _write_segment_dataset(external_root / "TestSegment")
    result = validate_segment_dataset(ds, containment_root=ds)
    assert result.is_valid
    assert result.dataset is not None


def test_invalid_external_dataset(external_root: Path) -> None:
    (external_root / "Empty").mkdir()
    result = validate_dataset(external_root / "Empty", containment_root=external_root / "Empty")
    assert not result.is_valid


def test_path_traversal_rejected(external_root: Path) -> None:
    with pytest.raises(ExternalDatasetError):
        resolve_external_dataset_path("../etc")
    with pytest.raises(ExternalDatasetError):
        resolve_external_dataset_path("/etc")
    with pytest.raises(ExternalDatasetError):
        resolve_external_dataset_path("Cuisine/../etc")


def test_path_outside_root_rejected(external_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    assert is_under_root(outside, external_root) is False
    with pytest.raises(ExternalDatasetError):
        assert_external_dataset_accessible(outside)


def test_symlink_outside_root_rejected(external_root: Path, tmp_path: Path) -> None:
    outside = tmp_path / "secret"
    outside.mkdir()
    link = external_root / "Escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not available")
    listed = [p.name for p in list_external_dataset_dirs()]
    assert "Escape" not in listed
    with pytest.raises(ExternalDatasetError):
        resolve_external_dataset_path("Escape")


def test_register_external_does_not_copy_and_is_readonly_meta(
    external_root: Path,
) -> None:
    ds = _write_detect_dataset(external_root / "Cuisine")
    before = {p.relative_to(ds) for p in ds.rglob("*") if p.is_file()}
    dataset_id, meta_dir = register_external_dataset_meta(ds, task="detect")
    after = {p.relative_to(ds) for p in ds.rglob("*") if p.is_file()}
    assert before == after
    assert (meta_dir / "meta.json").is_file()
    assert not (meta_dir / "extracted").exists()
    assert dataset_id.startswith("ext-detect-")
    # source tree unchanged even if we chmod files read-only
    for path in ds.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    result = validate_dataset(ds, containment_root=ds)
    assert result.is_valid


def test_source_type_uploaded_default_and_external() -> None:
    assert normalize_source_type(None) == SOURCE_UPLOADED
    assert normalize_source_type("") == SOURCE_UPLOADED
    assert normalize_source_type("external") == SOURCE_EXTERNAL
    assert normalize_source_type("uploaded") == SOURCE_UPLOADED


def test_session_payload_source_type(external_root: Path) -> None:
    ds = _write_detect_dataset(external_root / "Cuisine")
    result = validate_dataset(ds, containment_root=ds)
    assert result.dataset is not None
    payload = dataset_to_session_payload(
        result.dataset,
        ds,
        dataset_id="ext-detect-Cuisine",
        source_type=SOURCE_EXTERNAL,
        display_name="Cuisine",
    )
    assert payload["source_type"] == SOURCE_EXTERNAL
    legacy = dataset_to_session_payload(result.dataset, ds)
    assert legacy["source_type"] == SOURCE_UPLOADED


def test_external_missing_after_selection(external_root: Path) -> None:
    ds = _write_detect_dataset(external_root / "Gone")
    path = resolve_external_dataset_path("Gone")
    # remove dataset
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()
    with pytest.raises(ExternalDatasetError, match="n'est plus accessible"):
        assert_external_dataset_accessible(path)


def test_training_path_external_resolved(external_root: Path, tmp_path: Path) -> None:
    ds = _write_detect_dataset(external_root / "Cuisine")
    result = validate_dataset(ds, containment_root=ds)
    assert result.dataset is not None
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved = write_resolved_data_yaml(result.dataset, run_dir)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert Path(raw["path"]).resolve() == ds.resolve()
    assert Path(raw["train"]).resolve() == (ds / "train" / "images").resolve()


def test_auto_yaml_without_writing_source(external_root: Path, tmp_path: Path) -> None:
    ds = _write_detect_dataset(external_root / "NoYaml", with_yaml=False)
    assert not (ds / "data.yaml").exists()
    meta = tmp_path / "meta"
    meta.mkdir()
    yaml_path, message = ensure_data_yaml_for_layout(ds, generated_yaml_dir=meta)
    assert yaml_path is not None
    assert message is not None
    assert yaml_path.parent == meta
    assert not (ds / "data.yaml").exists()
    info, msgs = load_dataset_from_directory(
        ds,
        containment_root=ds,
        generated_yaml_dir=meta,
    )
    assert info is not None
    assert info.root.resolve() == ds.resolve()
    assert any("généré" in m for m in msgs)


def test_compose_cpu_and_gpu_preserve_external_mount() -> None:
    root = Path(__file__).resolve().parents[1]
    cpu = (root / "docker-compose.yml").read_text(encoding="utf-8")
    gpu = (root / "docker-compose.gpu.yml").read_text(encoding="utf-8")
    wsl = (root / "docker-compose.wsl.yml").read_text(encoding="utf-8")
    assert "/datasets:ro" in cpu
    assert "VISION_TRAINER_EXTERNAL_DATASETS: /datasets" in cpu
    assert ":/datasets:ro" in cpu
    # GPU overlay must not redefine volumes (inherits /datasets from base).
    assert "\n    volumes:" not in gpu
    assert "/datasets:ro" in wsl
