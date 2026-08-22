from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from PIL import Image

from vision_trainer.training.data_yaml import build_resolved_data_dict, write_resolved_data_yaml
from vision_trainer.training.device import DeviceError, describe_device, resolve_device
from vision_trainer.training.runs import create_run_directory, generate_run_id
from vision_trainer.training.trainer import (
    AVAILABLE_MODELS,
    BATCH_AUTO,
    TrainingError,
    TrainingRequest,
    build_train_kwargs,
    find_best_weights,
    run_training,
)
from vision_trainer.yolo.models import DatasetInfo, SplitInfo


def _make_dataset(tmp_path: Path) -> DatasetInfo:
    root = tmp_path / "dataset"
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    val_images = root / "val" / "images"
    val_labels = root / "val" / "labels"
    for directory in (train_images, train_labels, val_images, val_labels):
        directory.mkdir(parents=True)

    Image.new("RGB", (32, 32), (1, 2, 3)).save(train_images / "a.jpg")
    (train_labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    Image.new("RGB", (32, 32), (3, 2, 1)).save(val_images / "b.jpg")
    (val_labels / "b.txt").write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")

    yaml_path = root / "data.yaml"
    yaml_path.write_text("names: [pothole]\ntrain: train/images\nval: val/images\n", encoding="utf-8")

    return DatasetInfo(
        root=root,
        yaml_path=yaml_path,
        class_names={0: "pothole"},
        splits={
            "train": SplitInfo(
                name="train",
                images_dir=train_images,
                labels_dir=train_labels,
                image_count=1,
            ),
            "val": SplitInfo(
                name="val",
                images_dir=val_images,
                labels_dir=val_labels,
                image_count=1,
            ),
        },
    )


def test_generate_run_id_unique_format() -> None:
    first = generate_run_id(datetime(2026, 8, 22, 19, 0, 0))
    second = generate_run_id(datetime(2026, 8, 22, 19, 0, 0))
    assert first.startswith("20260822-190000-")
    assert second.startswith("20260822-190000-")
    assert first != second
    assert len(first.split("-")[-1]) == 6


def test_create_run_directory_unique(tmp_path: Path) -> None:
    run_id, run_dir = create_run_directory("demo-run", runs_root=tmp_path)
    assert run_id == "demo-run"
    assert run_dir == tmp_path / "demo-run"
    assert run_dir.is_dir()

    other_id, other_dir = create_run_directory("demo-run", runs_root=tmp_path)
    assert other_id != "demo-run"
    assert other_dir.is_dir()
    assert other_dir != run_dir


def test_write_resolved_data_yaml(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    out_dir = tmp_path / "run"
    yaml_path = write_resolved_data_yaml(dataset, out_dir)

    assert yaml_path.name == "data.resolved.yaml"
    assert yaml_path.is_file()
    # Original data.yaml must stay untouched.
    original = dataset.yaml_path.read_text(encoding="utf-8")
    assert "train: train/images" in original

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert payload["nc"] == 1
    assert payload["names"] == ["pothole"]
    assert payload["train"] == str(dataset.splits["train"].images_dir.resolve())
    assert payload["val"] == str(dataset.splits["val"].images_dir.resolve())
    assert Path(payload["train"]).is_dir()


def test_build_resolved_data_dict_requires_train(tmp_path: Path) -> None:
    dataset = DatasetInfo(
        root=tmp_path,
        yaml_path=tmp_path / "data.yaml",
        class_names={0: "a"},
        splits={},
    )
    with pytest.raises(ValueError, match="train"):
        build_resolved_data_dict(dataset)


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu") == "cpu"
    assert describe_device("cpu") == "CPU"


def test_resolve_device_auto_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: True,
    )
    assert resolve_device("auto") == "0"
    assert describe_device("0") == "CUDA (GPU 0)"


def test_resolve_device_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: False,
    )
    assert resolve_device("auto") == "cpu"


def test_resolve_device_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: False,
    )
    with pytest.raises(DeviceError, match="CUDA"):
        resolve_device("cuda")


def test_build_train_kwargs(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.resolved.yaml"
    data_yaml.write_text("names: [a]\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)

    kwargs = build_train_kwargs(
        data_yaml=data_yaml,
        run_dir=run_dir,
        epochs=3,
        imgsz=640,
        batch=BATCH_AUTO,
        device="cpu",
    )
    assert kwargs["data"] == str(data_yaml)
    assert kwargs["epochs"] == 3
    assert kwargs["imgsz"] == 640
    assert kwargs["batch"] == -1
    assert kwargs["device"] == "cpu"
    assert kwargs["project"] == str(run_dir.parent)
    assert kwargs["name"] == "abc"
    assert kwargs["exist_ok"] is True


def test_find_best_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    assert find_best_weights(run_dir) is None

    best = weights / "best.pt"
    best.write_bytes(b"fake")
    assert find_best_weights(run_dir) == best


def test_run_training_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset(tmp_path)
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )

    mock_model = MagicMock()

    def _factory(weights_name: str) -> MagicMock:
        assert weights_name == AVAILABLE_MODELS["YOLO11n"]

        def _train(**kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            (weights_dir / "best.pt").write_bytes(b"weights")

        mock_model.train.side_effect = _train
        return mock_model

    result = run_training(
        TrainingRequest(
            dataset=dataset,
            model_key="YOLO11n",
            epochs=2,
            imgsz=320,
            batch=8,
            device_choice="cpu",
            runs_root=tmp_path / "artifacts" / "runs",
        ),
        yolo_factory=_factory,
    )

    assert result.status == "completed"
    assert result.device == "cpu"
    assert result.epochs == 2
    assert result.imgsz == 320
    assert result.batch == 8
    assert result.resolved_data_yaml.is_file()
    assert result.best_weights is not None
    assert result.best_weights.name == "best.pt"
    mock_model.train.assert_called_once()
    call_kwargs = mock_model.train.call_args.kwargs
    assert call_kwargs["data"] == str(result.resolved_data_yaml)
    assert call_kwargs["device"] == "cpu"


def test_run_training_rejects_cuda_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _make_dataset(tmp_path)
    monkeypatch.setattr(
        "vision_trainer.training.device.is_cuda_available",
        lambda: False,
    )
    with pytest.raises(TrainingError, match="CUDA"):
        run_training(
            TrainingRequest(dataset=dataset, device_choice="cuda", runs_root=tmp_path / "runs"),
            yolo_factory=lambda _: MagicMock(),
        )


def test_run_training_model_load_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset(tmp_path)
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )

    def _factory(_weights: str):
        raise RuntimeError("weights missing")

    with pytest.raises(TrainingError, match="Impossible de charger"):
        run_training(
            TrainingRequest(dataset=dataset, runs_root=tmp_path / "runs"),
            yolo_factory=_factory,
        )


def test_run_training_ultralytics_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _make_dataset(tmp_path)
    monkeypatch.setattr(
        "vision_trainer.training.trainer.resolve_device",
        lambda choice: "cpu",
    )

    import builtins

    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            raise ImportError("No module named ultralytics")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_guarded_import):
        with pytest.raises(TrainingError, match="Ultralytics n'est pas disponible"):
            run_training(
                TrainingRequest(dataset=dataset, runs_root=tmp_path / "runs"),
            )
