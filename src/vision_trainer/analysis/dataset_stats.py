"""Dataset statistics for analysis (cached; never copies external datasets)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from vision_trainer.analysis.store import load_dataset_stats_cache, save_dataset_stats_cache
from vision_trainer.tasks import normalize_task
from vision_trainer.training.status import read_request_safe


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_dataset_stats(
    run_dir: Path,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """
    Collect train/val image counts (and per-class for classify).

    Results are persisted as ``dataset_stats.json`` beside the run.
    External datasets are scanned in place — never copied.
    """
    run_dir = Path(run_dir)
    if not force_refresh:
        cached = load_dataset_stats_cache(run_dir)
        if cached is not None:
            return cached

    request = read_request_safe(run_dir) or {}
    task = normalize_task(request.get("task"))
    dataset_name = request.get("dataset_name")
    source_type = request.get("dataset_source_type")
    root = _dataset_root(request)

    stats: dict[str, Any] = {
        "version": 1,
        "task": task,
        "dataset_name": dataset_name or (root.name if root else None),
        "dataset_root": str(root) if root else None,
        "source_type": source_type,
        "train_images": None,
        "val_images": None,
        "per_class": {},
        "scanned": False,
        "note": None,
    }

    if root is None or not root.is_dir():
        stats["note"] = "Racine dataset introuvable pour ce run."
        save_dataset_stats_cache(run_dir, stats)
        return stats

    if task == "classify":
        train_dir = root / "train"
        val_dir = _first_existing(root / "val", root / "valid", root / "test")
        train_pc = _classify_split_counts(train_dir) if train_dir.is_dir() else {}
        val_pc = _classify_split_counts(val_dir) if val_dir and val_dir.is_dir() else {}
        stats["per_class"] = {
            "train": train_pc,
            "val": val_pc,
        }
        stats["train_images"] = sum(train_pc.values()) if train_pc else 0
        stats["val_images"] = sum(val_pc.values()) if val_pc else 0
        stats["scanned"] = True
    else:
        # detect / segment YOLO layout
        train_images = _yolo_split_image_count(root, "train")
        val_images = _yolo_split_image_count(root, "val")
        if val_images is None:
            val_images = _yolo_split_image_count(root, "valid")
        stats["train_images"] = train_images
        stats["val_images"] = val_images
        label_counts = _yolo_class_label_counts(root, "train")
        if label_counts:
            stats["per_class"] = {"train_labels": label_counts}
        stats["scanned"] = train_images is not None or val_images is not None
        if not stats["scanned"]:
            stats["note"] = "Impossible de scanner le dataset (chemins absents)."

    save_dataset_stats_cache(run_dir, stats)
    return stats


def _dataset_root(request: dict[str, Any]) -> Path | None:
    for key in ("dataset_root", "data_dir"):
        raw = request.get(key)
        if raw:
            path = Path(str(raw))
            if path.is_dir():
                return path
    return None


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path
    return None


def _classify_split_counts(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        counts[child.name] = _count_images(child)
    return counts


def _count_images(folder: Path) -> int:
    total = 0
    try:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                total += 1
    except OSError:
        return total
    return total


def _yolo_split_image_count(root: Path, split: str) -> int | None:
    candidates = [
        root / split / "images",
        root / "images" / split,
        root / split,
    ]
    for folder in candidates:
        if folder.is_dir():
            return _count_images(folder)
    return None


def _yolo_class_label_counts(root: Path, split: str) -> dict[str, int]:
    """Count class ids in YOLO txt labels (keys are string class ids)."""
    label_dirs = [
        root / split / "labels",
        root / "labels" / split,
    ]
    counter: Counter[str] = Counter()
    for label_dir in label_dirs:
        if not label_dir.is_dir():
            continue
        try:
            for path in label_dir.glob("*.txt"):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for line in text.splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    counter[parts[0]] += 1
        except OSError:
            continue
        if counter:
            break
    return dict(counter)
