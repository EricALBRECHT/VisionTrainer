from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vision_trainer.yolo.models import DatasetInfo

RESOLVED_DATA_YAML_NAME = "data.resolved.yaml"


def build_resolved_data_dict(dataset: DatasetInfo) -> dict[str, Any]:
    """Build a YOLO data.yaml mapping with absolute resolved split paths."""
    if "train" not in dataset.splits or dataset.splits["train"].images_dir is None:
        raise ValueError("Le split 'train' résolu est requis pour générer data.resolved.yaml.")

    names_list = [dataset.class_names[index] for index in range(dataset.num_classes)]
    payload: dict[str, Any] = {
        "path": str(dataset.root.resolve()),
        "names": names_list,
        "nc": dataset.num_classes,
    }

    for split_name in ("train", "val", "test"):
        split = dataset.splits.get(split_name)
        if split is None or split.images_dir is None:
            continue
        payload[split_name] = str(split.images_dir.resolve())

    return payload


def write_resolved_data_yaml(dataset: DatasetInfo, destination_dir: Path) -> Path:
    """
    Write ``data.resolved.yaml`` into ``destination_dir``.

    Does not modify the original imported data.yaml.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / RESOLVED_DATA_YAML_NAME
    payload = build_resolved_data_dict(dataset)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path
