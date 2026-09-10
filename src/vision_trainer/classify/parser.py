"""Parse Ultralytics ImageFolder classification datasets."""

from __future__ import annotations

from pathlib import Path

from vision_trainer.classify.models import ClassifyClassStats, ClassifyDatasetInfo
from vision_trainer.yolo.parser import IMAGE_EXTENSIONS

SPLIT_DIR_ALIASES = {
    "train": ("train",),
    "val": ("val", "validation", "valid"),
    "test": ("test",),
}


def find_classify_dataset_root(extract_root: Path) -> Path | None:
    """
    Locate a classification dataset root (directory containing ``train/<class>/``).

    Handles an optional single top-level folder inside a ZIP extract.
    """
    extract_root = extract_root.resolve()
    if not extract_root.is_dir():
        return None

    if _looks_like_classify_root(extract_root):
        return extract_root

    children = [c for c in sorted(extract_root.iterdir()) if c.is_dir() and not c.name.startswith(".")]
    if len(children) == 1 and _looks_like_classify_root(children[0]):
        return children[0]

    for child in children:
        if _looks_like_classify_root(child):
            return child
    return None


def _looks_like_classify_root(path: Path) -> bool:
    train = path / "train"
    if not train.is_dir():
        return False
    # Reject YOLO detect layout: train/images + train/labels
    if (train / "images").is_dir() and (train / "labels").is_dir():
        return False
    class_dirs = [d for d in train.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return bool(class_dirs)


def load_classify_dataset(root: Path) -> tuple[ClassifyDatasetInfo | None, list[str]]:
    """
    Load a classification dataset from ``root/train/<class>/…``.

    Returns ``(info, hard_errors)``. Soft issues are left to the validator.
    """
    root = root.resolve()
    errors: list[str] = []
    if not root.is_dir():
        return None, [f"Dossier dataset introuvable : {root}"]

    train_dir = _resolve_split_dir(root, "train")
    if train_dir is None:
        return None, ["Dossier 'train/' introuvable (structure classification requise)."]

    val_dir = _resolve_split_dir(root, "val")
    test_dir = _resolve_split_dir(root, "test")

    class_names_list = sorted(
        d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not class_names_list:
        return None, ["Aucune classe trouvée dans train/ (dossiers attendus)."]

    if "images" in class_names_list and "labels" in class_names_list and len(class_names_list) <= 3:
        return None, [
            "Structure de type détection YOLO détectée (train/images + train/labels). "
            "Utilisez la page Dataset YOLO, pas Classification."
        ]

    class_names = {index: name for index, name in enumerate(class_names_list)}
    class_stats: dict[str, ClassifyClassStats] = {
        name: ClassifyClassStats(name=name) for name in class_names_list
    }
    unsupported: list[str] = []
    empty_dirs: list[str] = []

    train_count = _scan_split(train_dir, class_stats, "train", unsupported, empty_dirs)
    val_count = _scan_split(val_dir, class_stats, "val", unsupported, empty_dirs) if val_dir else 0
    test_count = _scan_split(test_dir, class_stats, "test", unsupported, empty_dirs) if test_dir else 0

    if train_count == 0:
        errors.append("Aucune image d'entraînement valide trouvée dans train/<classe>/.")

    if errors:
        return None, errors

    info = ClassifyDatasetInfo(
        root=root,
        class_names=class_names,
        class_stats=class_stats,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_image_count=train_count,
        val_image_count=val_count,
        test_image_count=test_count,
        unsupported_files=unsupported,
        empty_class_dirs=empty_dirs,
    )
    return info, []


def _resolve_split_dir(root: Path, split: str) -> Path | None:
    for name in SPLIT_DIR_ALIASES[split]:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _scan_split(
    split_dir: Path | None,
    class_stats: dict[str, ClassifyClassStats],
    split_name: str,
    unsupported: list[str],
    empty_dirs: list[str],
) -> int:
    if split_dir is None or not split_dir.is_dir():
        return 0
    total = 0
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        stats = class_stats.setdefault(class_dir.name, ClassifyClassStats(name=class_dir.name))
        images = 0
        for path in class_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images += 1
            else:
                if path.name.lower() in {".ds_store", "thumbs.db"}:
                    continue
                if path.suffix.lower() in {".txt", ".json", ".md", ".csv"}:
                    continue
                unsupported.append(str(path.relative_to(split_dir.parent)))
        if images == 0:
            empty_dirs.append(f"{split_name}/{class_dir.name}")
        if split_name == "train":
            stats.train_count = images
        elif split_name == "val":
            stats.val_count = images
        else:
            stats.test_count = images
        total += images
    return total
