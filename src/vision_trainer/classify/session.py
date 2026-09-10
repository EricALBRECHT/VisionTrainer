from __future__ import annotations

from pathlib import Path

from vision_trainer.classify.models import ClassifyClassStats, ClassifyDatasetInfo

SESSION_CLS_DATASET_KEY = "validated_classify_dataset"


def classify_dataset_to_session_payload(
    dataset: ClassifyDatasetInfo,
    extract_dir: Path,
    *,
    dataset_id: str | None = None,
) -> dict:
    payload: dict = {
        "task": "classify",
        "extract_dir": str(extract_dir),
        "root": str(dataset.root),
        "class_names": {int(k): v for k, v in dataset.class_names.items()},
        "train_image_count": dataset.train_image_count,
        "val_image_count": dataset.val_image_count,
        "test_image_count": dataset.test_image_count,
        "class_stats": {
            name: {
                "train_count": stats.train_count,
                "val_count": stats.val_count,
                "test_count": stats.test_count,
            }
            for name, stats in dataset.class_stats.items()
        },
    }
    if dataset_id:
        payload["dataset_id"] = dataset_id
    return payload


def classify_dataset_from_session_payload(payload: dict) -> ClassifyDatasetInfo:
    class_names = {int(k): str(v) for k, v in payload["class_names"].items()}
    class_stats: dict[str, ClassifyClassStats] = {}
    for name, raw in (payload.get("class_stats") or {}).items():
        class_stats[str(name)] = ClassifyClassStats(
            name=str(name),
            train_count=int(raw.get("train_count", 0)),
            val_count=int(raw.get("val_count", 0)),
            test_count=int(raw.get("test_count", 0)),
        )
    root = Path(payload["root"])
    return ClassifyDatasetInfo(
        root=root,
        class_names=class_names,
        class_stats=class_stats,
        train_dir=root / "train" if (root / "train").is_dir() else None,
        val_dir=(root / "val") if (root / "val").is_dir() else None,
        test_dir=(root / "test") if (root / "test").is_dir() else None,
        train_image_count=int(payload.get("train_image_count", 0)),
        val_image_count=int(payload.get("val_image_count", 0)),
        test_image_count=int(payload.get("test_image_count", 0)),
    )
