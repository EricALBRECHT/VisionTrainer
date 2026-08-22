from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from vision_trainer.yolo.parser import (
    ZipExtractionError,
    extract_zip_dataset,
    find_data_yaml,
    label_path_for_image,
    load_dataset_from_directory,
)
from vision_trainer.yolo.validator import validate_dataset

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _write_data_yaml(
    root: Path,
    *,
    include_train: bool = True,
    train_value: object = "train/images",
    include_names: bool = True,
    names_value: object | None = None,
    include_val: bool = True,
    val_value: object = "val/images",
) -> None:
    content: dict = {"path": str(root)}
    if include_names:
        content["names"] = names_value if names_value is not None else {0: "cat", 1: "dog"}
    if include_train:
        content["train"] = train_value
    if include_val:
        content["val"] = val_value
    (root / "data.yaml").write_text(yaml.dump(content), encoding="utf-8")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


def _write_corrupted_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-valid-image")


def _write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "valid_dataset"
    root.mkdir()
    _write_data_yaml(root)

    _write_image(root / "train" / "images" / "cat_1.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "cat_1.txt", ["0 0.5 0.5 0.4 0.4"])

    _write_image(root / "train" / "images" / "dog_1.jpg", (0, 255, 0))
    _write_label(root / "train" / "labels" / "dog_1.txt", ["1 0.3 0.3 0.2 0.2"])

    _write_image(root / "val" / "images" / "cat_2.jpg", (0, 0, 255))
    _write_label(root / "val" / "labels" / "cat_2.txt", ["0 0.6 0.6 0.3 0.3"])

    return root


@pytest.fixture
def dataset_missing_names(tmp_path: Path) -> Path:
    root = tmp_path / "missing_names"
    root.mkdir()
    _write_data_yaml(root, include_names=False)
    return root


@pytest.fixture
def dataset_missing_train(tmp_path: Path) -> Path:
    root = tmp_path / "missing_train"
    root.mkdir()
    _write_data_yaml(root, include_train=False)
    _write_image(root / "val" / "images" / "cat_2.jpg", (0, 0, 255))
    _write_label(root / "val" / "labels" / "cat_2.txt", ["0 0.6 0.6 0.3 0.3"])
    return root


@pytest.fixture
def dataset_invalid_label_format(tmp_path: Path) -> Path:
    root = tmp_path / "invalid_label_format"
    root.mkdir()
    _write_data_yaml(root, include_val=False)
    _write_image(root / "train" / "images" / "cat_1.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "cat_1.txt", ["invalid line"])
    return root


@pytest.fixture
def dataset_invalid_class_id(tmp_path: Path) -> Path:
    root = tmp_path / "invalid_class_id"
    root.mkdir()
    _write_data_yaml(root, include_val=False)
    _write_image(root / "train" / "images" / "cat_1.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "cat_1.txt", ["99 0.5 0.5 0.4 0.4"])
    return root


@pytest.fixture
def dataset_invalid_coordinates(tmp_path: Path) -> Path:
    root = tmp_path / "invalid_coordinates"
    root.mkdir()
    _write_data_yaml(root, include_val=False)
    _write_image(root / "train" / "images" / "cat_1.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "cat_1.txt", ["0 1.5 0.5 0.4 0.4"])
    return root


@pytest.fixture
def dataset_zero_width(tmp_path: Path) -> Path:
    root = tmp_path / "zero_width"
    root.mkdir()
    _write_data_yaml(root, include_val=False)
    _write_image(root / "train" / "images" / "cat_1.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "cat_1.txt", ["0 0.5 0.5 0.0 0.4"])
    return root


@pytest.fixture
def dataset_image_without_label(tmp_path: Path) -> Path:
    root = tmp_path / "image_without_label"
    root.mkdir()
    _write_data_yaml(root, include_val=False)
    _write_image(root / "train" / "images" / "cat_1.jpg", (255, 0, 0))
    (root / "train" / "labels").mkdir(parents=True)
    return root


@pytest.fixture
def dataset_label_without_image(tmp_path: Path) -> Path:
    root = tmp_path / "label_without_image"
    root.mkdir()
    _write_data_yaml(root, include_val=False)
    _write_image(root / "train" / "images" / "cat_1.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "cat_1.txt", ["0 0.5 0.5 0.4 0.4"])
    _write_label(root / "train" / "labels" / "orphan.txt", ["1 0.2 0.2 0.1 0.1"])
    return root


@pytest.fixture
def dataset_same_stem_subdirs(tmp_path: Path) -> Path:
    root = tmp_path / "same_stem_subdirs"
    root.mkdir()
    _write_data_yaml(root, include_val=False)

    _write_image(root / "train" / "images" / "site_a" / "photo.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "site_a" / "photo.txt", ["0 0.5 0.5 0.4 0.4"])

    _write_image(root / "train" / "images" / "site_b" / "photo.jpg", (0, 255, 0))
    _write_label(root / "train" / "labels" / "site_b" / "orphan_photo.txt", ["1 0.2 0.2 0.1 0.1"])

    return root


@pytest.fixture
def dataset_corrupted_image(tmp_path: Path) -> Path:
    root = tmp_path / "corrupted_image"
    root.mkdir()
    _write_data_yaml(root, include_val=False)
    _write_corrupted_image(root / "train" / "images" / "broken.jpg")
    _write_label(root / "train" / "labels" / "broken.txt", ["0 0.5 0.5 0.4 0.4"])
    return root


def test_find_data_yaml_at_root(valid_dataset: Path) -> None:
    assert find_data_yaml(valid_dataset) == valid_dataset / "data.yaml"


def test_load_valid_dataset(valid_dataset: Path) -> None:
    dataset, errors = load_dataset_from_directory(valid_dataset)
    assert errors == []
    assert dataset is not None
    assert dataset.num_classes == 2
    assert dataset.class_names[0] == "cat"
    assert set(dataset.splits) == {"train", "val"}


def test_validate_valid_dataset(valid_dataset: Path) -> None:
    result = validate_dataset(valid_dataset)
    assert result.is_valid
    assert result.dataset is not None
    assert result.dataset.splits["train"].image_count == 2
    assert result.dataset.splits["val"].image_count == 1
    assert any(issue.severity.value == "info" for issue in result.issues)


def test_missing_data_yaml(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    dataset, errors = load_dataset_from_directory(empty)
    assert dataset is None
    assert any("data.yaml" in error for error in errors)


def test_missing_names(dataset_missing_names: Path) -> None:
    dataset, errors = load_dataset_from_directory(dataset_missing_names)
    assert dataset is None
    assert any("names" in error.lower() for error in errors)


def test_missing_train_split(dataset_missing_train: Path) -> None:
    result = validate_dataset(dataset_missing_train)
    assert not result.is_valid
    assert any("absent" in issue.message.lower() for issue in result.errors)


def test_invalid_label_format(dataset_invalid_label_format: Path) -> None:
    result = validate_dataset(dataset_invalid_label_format)
    assert not result.is_valid
    assert any("format yolo" in issue.message.lower() for issue in result.errors)


def test_invalid_class_id(dataset_invalid_class_id: Path) -> None:
    result = validate_dataset(dataset_invalid_class_id)
    assert not result.is_valid
    assert any("inconnue" in issue.message.lower() for issue in result.errors)


def test_invalid_coordinates(dataset_invalid_coordinates: Path) -> None:
    result = validate_dataset(dataset_invalid_coordinates)
    assert not result.is_valid
    assert any("hors [0, 1]" in issue.message.lower() for issue in result.errors)


def test_zero_width(dataset_zero_width: Path) -> None:
    result = validate_dataset(dataset_zero_width)
    assert not result.is_valid
    assert any("width nulle" in issue.message.lower() for issue in result.errors)


def test_image_without_label_is_warning(dataset_image_without_label: Path) -> None:
    result = validate_dataset(dataset_image_without_label)
    assert result.is_valid
    assert any("sans label" in issue.message.lower() for issue in result.warnings)


def test_label_without_image_is_warning(dataset_label_without_image: Path) -> None:
    result = validate_dataset(dataset_label_without_image)
    assert result.is_valid
    assert any("sans image" in issue.message.lower() for issue in result.warnings)


def test_same_stem_subdirs_use_relative_paths(dataset_same_stem_subdirs: Path) -> None:
    root = dataset_same_stem_subdirs
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"

    site_a_image = root / "train" / "images" / "site_a" / "photo.jpg"
    site_a_label = label_path_for_image(site_a_image, images_dir, labels_dir)
    assert site_a_label == labels_dir / "site_a" / "photo.txt"

    result = validate_dataset(root)
    assert result.is_valid
    assert any("site_b/orphan_photo" in issue.context for issue in result.warnings)


def test_corrupted_image(dataset_corrupted_image: Path) -> None:
    result = validate_dataset(dataset_corrupted_image)
    assert not result.is_valid
    assert any("illisible ou corrompue" in issue.message.lower() for issue in result.errors)


def test_names_list_valid(tmp_path: Path) -> None:
    root = tmp_path / "names_list"
    root.mkdir()
    _write_data_yaml(root, names_value=["cat", "dog"], include_val=False)
    _write_image(root / "train" / "images" / "img.jpg", (255, 0, 0))
    _write_label(root / "train" / "labels" / "img.txt", ["0 0.5 0.5 0.2 0.2"])

    dataset, errors = load_dataset_from_directory(root)
    assert errors == []
    assert dataset is not None
    assert dataset.class_names == {0: "cat", 1: "dog"}


@pytest.mark.parametrize(
    "names_value,expected_fragment",
    [
        ([None, "dog"], "names[0]"),
        ([123, "dog"], "names[0]"),
        ([{"bad": "object"}, "dog"], "names[0]"),
    ],
)
def test_names_list_invalid_entries(
    tmp_path: Path,
    names_value: list[object],
    expected_fragment: str,
) -> None:
    root = tmp_path / "names_invalid"
    root.mkdir()
    _write_data_yaml(root, names_value=names_value, include_val=False)

    dataset, errors = load_dataset_from_directory(root)
    assert dataset is None
    assert any(expected_fragment in error for error in errors)


def test_names_empty_list(tmp_path: Path) -> None:
    root = tmp_path / "names_empty_list"
    root.mkdir()
    _write_data_yaml(root, names_value=[], include_val=False)

    dataset, errors = load_dataset_from_directory(root)
    assert dataset is None
    assert any("vide" in error.lower() for error in errors)


def test_names_empty_dict(tmp_path: Path) -> None:
    root = tmp_path / "names_empty_dict"
    root.mkdir()
    _write_data_yaml(root, names_value={}, include_val=False)

    dataset, errors = load_dataset_from_directory(root)
    assert dataset is None
    assert any("vide" in error.lower() for error in errors)


def test_names_dict_negative_key(tmp_path: Path) -> None:
    root = tmp_path / "names_negative"
    root.mkdir()
    _write_data_yaml(root, names_value={-1: "cat", 0: "dog"}, include_val=False)

    dataset, errors = load_dataset_from_directory(root)
    assert dataset is None
    assert any("négatif" in error.lower() for error in errors)


def test_names_dict_non_contiguous_indices(tmp_path: Path) -> None:
    root = tmp_path / "names_non_contiguous"
    root.mkdir()
    _write_data_yaml(root, names_value={0: "cat", 2: "dog"}, include_val=False)

    dataset, errors = load_dataset_from_directory(root)
    assert dataset is None
    assert any("contigus" in error.lower() for error in errors)


def test_train_split_invalid_type(tmp_path: Path) -> None:
    root = tmp_path / "train_invalid_type"
    root.mkdir()
    _write_data_yaml(root, train_value=["train/images"], include_val=False)

    dataset, errors = load_dataset_from_directory(root)
    assert dataset is None
    assert any("chaîne" in error.lower() for error in errors)


def test_train_split_empty_string(tmp_path: Path) -> None:
    root = tmp_path / "train_empty"
    root.mkdir()
    _write_data_yaml(root, train_value="   ", include_val=False)

    dataset, errors = load_dataset_from_directory(root)
    assert dataset is None
    assert any("vide" in error.lower() for error in errors)


def test_zip_slip_rejected(tmp_path: Path) -> None:
    zip_path = tmp_path / "malicious.zip"
    destination = tmp_path / "extract"
    outside_file = tmp_path / "outside.txt"

    _create_zip(zip_path, {"../../outside.txt": b"escaped"})

    with pytest.raises(ZipExtractionError):
        extract_zip_dataset(zip_path, destination)

    assert not outside_file.exists()
    assert not destination.exists() or not any(destination.iterdir())


def test_invalid_zip_rejected(tmp_path: Path) -> None:
    zip_path = tmp_path / "invalid.zip"
    zip_path.write_bytes(b"not-a-zip")
    destination = tmp_path / "extract"

    with pytest.raises(ZipExtractionError, match="invalide"):
        extract_zip_dataset(zip_path, destination)


def test_committed_valid_fixture() -> None:
    fixture_root = FIXTURES_DIR / "valid_dataset"
    if not fixture_root.exists():
        pytest.skip("Committed fixture not generated yet")
    result = validate_dataset(fixture_root)
    assert result.is_valid


def _build_kaggle_style_extracted_dataset(
    tmp_path: Path,
    *,
    path_value: object | None,
) -> Path:
    """Reproduce the import layout: temp/extracted/{data.yaml, train, valid, test}."""
    temp_root = tmp_path / "vision-trainer-dataset-XXXX"
    extracted = temp_root / "extracted"
    extracted.mkdir(parents=True)

    for split in ("train", "valid", "test"):
        _write_image(extracted / split / "images" / "img.jpg", (200, 100, 50))
        _write_label(extracted / split / "labels" / "img.txt", ["0 0.5 0.5 0.3 0.3"])

    content: dict = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "object"},
    }
    if path_value is not None:
        content["path"] = path_value
    (extracted / "data.yaml").write_text(yaml.dump(content), encoding="utf-8")
    return extracted


@pytest.mark.parametrize("path_value", [".", None, ".."])
def test_relative_splits_resolved_under_extracted(
    tmp_path: Path,
    path_value: object | None,
) -> None:
    extracted = _build_kaggle_style_extracted_dataset(tmp_path, path_value=path_value)
    temp_root = extracted.parent

    result = validate_dataset(extracted)
    assert result.is_valid, [issue.message for issue in result.errors]
    assert result.dataset is not None

    for split_name, folder_name in (("train", "train"), ("val", "valid"), ("test", "test")):
        split = result.dataset.splits[split_name]
        expected = extracted / folder_name / "images"
        assert split.images_dir == expected.resolve()
        assert split.image_count == 1
        assert str(temp_root / folder_name / "images") not in str(split.images_dir)


def test_relative_splits_not_resolved_against_temp_parent(tmp_path: Path) -> None:
    extracted = _build_kaggle_style_extracted_dataset(tmp_path, path_value="..")
    temp_root = extracted.parent

    dataset, errors = load_dataset_from_directory(extracted)
    assert errors == []
    assert dataset is not None
    assert dataset.root == extracted.resolve()
    assert dataset.root != temp_root.resolve()

    wrong_train = temp_root / "train" / "images"
    assert not wrong_train.exists()
    assert dataset.splits["train"].images_dir == (extracted / "train" / "images").resolve()
    assert dataset.splits["val"].images_dir == (extracted / "valid" / "images").resolve()
    assert dataset.splits["test"].images_dir == (extracted / "test" / "images").resolve()


def test_parent_relative_split_paths_fallback_with_warning(tmp_path: Path) -> None:
    """Reproduce Kaggle-style data.yaml with train: ../train/images under extracted/."""
    extracted = tmp_path / "extracted"
    extracted.mkdir()

    for split in ("train", "valid", "test"):
        _write_image(extracted / split / "images" / "img.jpg", (180, 90, 40))
        _write_label(extracted / split / "labels" / "img.txt", ["0 0.5 0.5 0.25 0.25"])

    yaml_content = (
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        "nc: 1\n"
        "names: ['pothole']\n"
    )
    yaml_path = extracted / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    original_yaml = yaml_path.read_text(encoding="utf-8")

    # Strict resolution would point outside extracted and miss the real folders.
    strict_train = (extracted / "../train/images").resolve()
    assert strict_train == (extracted.parent / "train" / "images").resolve()
    assert not strict_train.exists()
    assert (extracted / "train" / "images").is_dir()

    result = validate_dataset(extracted)
    assert result.is_valid, [issue.message for issue in result.errors]
    assert result.dataset is not None

    for split_name, folder_name in (("train", "train"), ("val", "valid"), ("test", "test")):
        split = result.dataset.splits[split_name]
        assert split.images_dir == (extracted / folder_name / "images").resolve()
        assert split.image_count == 1
        assert split.resolved_via_fallback is True

    path_warnings = [
        issue
        for issue in result.warnings
        if "introuvable" in issue.message and "chemin détecté" in issue.message
    ]
    assert len(path_warnings) == 3
    assert any("../train/images" in issue.message and "train/images" in issue.message for issue in path_warnings)
    assert any("../valid/images" in issue.message and "valid/images" in issue.message for issue in path_warnings)
    assert any("../test/images" in issue.message and "test/images" in issue.message for issue in path_warnings)

    assert not any("introuvable" in issue.message and issue.severity.value == "error" for issue in result.errors)
    assert yaml_path.read_text(encoding="utf-8") == original_yaml
