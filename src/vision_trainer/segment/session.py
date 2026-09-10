"""Session helpers for validated segmentation datasets."""

from __future__ import annotations

from pathlib import Path

from vision_trainer.datasets.external import SOURCE_UPLOADED, normalize_source_type
from vision_trainer.segment.labels import SegmentDatasetInfo, SegmentSplitStats
from vision_trainer.yolo.models import DatasetInfo, SplitInfo

SESSION_SEG_DATASET_KEY = "validated_segment_dataset"


def segment_dataset_to_session_payload(
    dataset: DatasetInfo,
    extract_dir: Path,
    *,
    segment_info: SegmentDatasetInfo | None = None,
    dataset_id: str | None = None,
    source_type: str = SOURCE_UPLOADED,
    display_name: str | None = None,
) -> dict:
    payload: dict = {
        "task": "segment",
        "extract_dir": str(extract_dir),
        "root": str(dataset.root),
        "yaml_path": str(dataset.yaml_path),
        "class_names": {int(k): v for k, v in dataset.class_names.items()},
        "source_type": normalize_source_type(source_type),
        "splits": {},
    }
    for name, split in dataset.splits.items():
        payload["splits"][name] = {
            "name": split.name,
            "images_dir": str(split.images_dir) if split.images_dir else None,
            "labels_dir": str(split.labels_dir) if split.labels_dir else None,
            "image_count": split.image_count,
            "images_without_labels": list(split.images_without_labels),
        }
    if segment_info is not None:
        payload["split_stats"] = {
            name: {
                "image_count": stats.image_count,
                "instance_count": stats.instance_count,
                "instances_by_class": dict(stats.instances_by_class),
                "images_without_labels": list(stats.images_without_labels),
                "invalid_label_files": stats.invalid_label_files,
                "total_polygon_points": stats.total_polygon_points,
                "polygon_count": stats.polygon_count,
            }
            for name, stats in segment_info.split_stats.items()
        }
        payload["detection_label_hits"] = segment_info.detection_label_hits
    if dataset_id:
        payload["dataset_id"] = dataset_id
    if display_name:
        payload["display_name"] = display_name
    return payload


def segment_dataset_from_session_payload(payload: dict) -> DatasetInfo:
    class_names = {int(k): str(v) for k, v in payload["class_names"].items()}
    splits: dict[str, SplitInfo] = {}
    for name, raw in (payload.get("splits") or {}).items():
        splits[str(name)] = SplitInfo(
            name=str(raw.get("name") or name),
            images_dir=Path(raw["images_dir"]) if raw.get("images_dir") else None,
            labels_dir=Path(raw["labels_dir"]) if raw.get("labels_dir") else None,
            image_count=int(raw.get("image_count") or 0),
            images_without_labels=list(raw.get("images_without_labels") or []),
        )
    return DatasetInfo(
        root=Path(payload["root"]),
        yaml_path=Path(payload["yaml_path"]),
        class_names=class_names,
        splits=splits,
    )


def segment_info_from_session_payload(payload: dict) -> SegmentDatasetInfo | None:
    if "split_stats" not in payload:
        return None
    class_names = {int(k): str(v) for k, v in payload["class_names"].items()}
    split_stats = {
        name: SegmentSplitStats(
            name=name,
            image_count=int(raw.get("image_count") or 0),
            instance_count=int(raw.get("instance_count") or 0),
            instances_by_class={
                str(k): int(v) for k, v in (raw.get("instances_by_class") or {}).items()
            },
            images_without_labels=list(raw.get("images_without_labels") or []),
            invalid_label_files=int(raw.get("invalid_label_files") or 0),
            total_polygon_points=int(raw.get("total_polygon_points") or 0),
            polygon_count=int(raw.get("polygon_count") or 0),
        )
        for name, raw in (payload.get("split_stats") or {}).items()
    }
    return SegmentDatasetInfo(
        root=Path(payload["root"]),
        yaml_path=Path(payload["yaml_path"]),
        class_names=class_names,
        split_stats=split_stats,
        detection_label_hits=int(payload.get("detection_label_hits") or 0),
    )
