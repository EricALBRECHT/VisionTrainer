from __future__ import annotations

from pathlib import Path

from vision_trainer.yolo.models import DatasetInfo, SplitInfo


SESSION_DATASET_KEY = "validated_dataset"


def dataset_to_session_payload(dataset: DatasetInfo, extract_dir: Path) -> dict:
    """Serialize a validated dataset for Streamlit session_state."""
    splits: dict[str, dict] = {}
    for name, split in dataset.splits.items():
        splits[name] = {
            "images_dir": str(split.images_dir) if split.images_dir else None,
            "labels_dir": str(split.labels_dir) if split.labels_dir else None,
            "image_count": split.image_count,
        }
    return {
        "extract_dir": str(extract_dir),
        "root": str(dataset.root),
        "yaml_path": str(dataset.yaml_path),
        "class_names": {int(key): value for key, value in dataset.class_names.items()},
        "splits": splits,
    }


def dataset_from_session_payload(payload: dict) -> DatasetInfo:
    """Rebuild DatasetInfo from a session_state payload."""
    splits: dict[str, SplitInfo] = {}
    for name, raw in payload.get("splits", {}).items():
        images_dir = Path(raw["images_dir"]) if raw.get("images_dir") else None
        labels_dir = Path(raw["labels_dir"]) if raw.get("labels_dir") else None
        splits[name] = SplitInfo(
            name=name,
            images_dir=images_dir,
            labels_dir=labels_dir,
            image_count=int(raw.get("image_count", 0)),
        )
    class_names = {int(key): str(value) for key, value in payload["class_names"].items()}
    return DatasetInfo(
        root=Path(payload["root"]),
        yaml_path=Path(payload["yaml_path"]),
        class_names=class_names,
        splits=splits,
    )
