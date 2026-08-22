from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from vision_trainer.yolo.models import DatasetInfo, Severity, SplitInfo, ValidationIssue, ValidationResult
from vision_trainer.yolo.parser import (
    image_path_for_label,
    label_path_for_image,
    list_images,
    load_dataset_from_directory,
    relative_path_without_extension,
)

YOLO_LINE_PATTERN = re.compile(
    r"^\s*(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def validate_dataset(root: Path) -> ValidationResult:
    """Validate a YOLO dataset directory and return structured findings."""
    dataset, parse_errors = load_dataset_from_directory(root)
    issues: list[ValidationIssue] = [
        ValidationIssue(Severity.ERROR, message) for message in parse_errors
    ]

    if dataset is None:
        return ValidationResult(dataset=None, issues=issues)

    issues.append(
        ValidationIssue(
            Severity.INFO,
            f"Dataset chargé depuis {dataset.yaml_path}.",
        )
    )
    issues.append(
        ValidationIssue(
            Severity.INFO,
            f"{dataset.num_classes} classe(s) définie(s).",
        )
    )

    if "train" not in dataset.splits:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                "Le split 'train' est absent de data.yaml.",
            )
        )

    available_splits = [name for name in ("train", "val", "test") if name in dataset.splits]
    if available_splits:
        issues.append(
            ValidationIssue(
                Severity.INFO,
                f"Splits disponibles : {', '.join(available_splits)}.",
            )
        )

    for split_info in dataset.splits.values():
        if split_info.resolved_via_fallback and split_info.declared_ref and split_info.fallback_ref:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    (
                        f"Le chemin déclaré '{split_info.declared_ref}' est introuvable. "
                        f"Utilisation du chemin détecté '{split_info.fallback_ref}'."
                    ),
                    context=split_info.name,
                )
            )
        issues.extend(_validate_split(dataset, split_info))

    return ValidationResult(dataset=dataset, issues=issues)


def _validate_split(dataset: DatasetInfo, split: SplitInfo) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if split.images_dir is None or not split.images_dir.exists():
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                f"Le chemin images du split '{split.name}' est introuvable.",
                context=str(split.images_dir),
            )
        )
        return issues

    images = list_images(split.images_dir)
    split.image_count = len(images)

    if split.image_count == 0:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                f"Aucune image trouvée pour le split '{split.name}'.",
                context=str(split.images_dir),
            )
        )
        return issues

    issues.append(
        ValidationIssue(
            Severity.INFO,
            f"Split '{split.name}' : {split.image_count} image(s).",
        )
    )

    for image in images:
        relative_image = relative_path_without_extension(image, split.images_dir)
        image_issue = _validate_image_readable(image, relative_image)
        if image_issue is not None:
            issues.append(image_issue)

    if split.labels_dir is None or not split.labels_dir.exists():
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                f"Le dossier labels du split '{split.name}' est introuvable.",
                context=str(split.labels_dir),
            )
        )
        return issues

    label_files = sorted(path for path in split.labels_dir.rglob("*.txt") if path.is_file())
    split.label_file_count = len(label_files)

    images_without_labels: list[str] = []
    labels_without_images: list[str] = []
    matched_pairs = 0
    matched_labels: set[Path] = set()

    for image in images:
        label_path = label_path_for_image(image, split.images_dir, split.labels_dir)
        relative_image = relative_path_without_extension(image, split.images_dir)
        if not label_path.is_file():
            images_without_labels.append(relative_image)
            continue

        matched_pairs += 1
        matched_labels.add(label_path.resolve())
        issues.extend(
            _validate_label_file(label_path, dataset.class_names, relative_image)
        )

    for label in label_files:
        if label.resolve() in matched_labels:
            continue
        image_path = image_path_for_label(label, split.labels_dir, split.images_dir)
        relative_label = relative_path_without_extension(label, split.labels_dir)
        if image_path is None or not image_path.is_file():
            labels_without_images.append(relative_label)
            continue

    split.matched_pairs = matched_pairs
    split.images_without_labels = images_without_labels
    split.labels_without_images = labels_without_images

    if images_without_labels:
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                f"{len(images_without_labels)} image(s) sans label dans '{split.name}'.",
                context=", ".join(images_without_labels[:5]),
            )
        )

    if labels_without_images:
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                f"{len(labels_without_images)} label(s) sans image dans '{split.name}'.",
                context=", ".join(labels_without_images[:5]),
            )
        )

    return issues


def _validate_image_readable(image_path: Path, relative_path: str) -> ValidationIssue | None:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        return ValidationIssue(
            Severity.ERROR,
            f"Image illisible ou corrompue : {relative_path}.",
            context=str(exc),
        )
    return None


def _validate_label_file(
    label_path: Path,
    class_names: dict[int, str],
    relative_label: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        content = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [
            ValidationIssue(
                Severity.ERROR,
                f"Impossible de lire le label {relative_label}.",
                context=str(exc),
            )
        ]

    if not content.strip():
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                f"Fichier label vide : {relative_label}.",
            )
        )
        return issues

    for line_number, line in enumerate(content.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        match = YOLO_LINE_PATTERN.match(stripped)
        if not match:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"Format YOLO invalide dans {relative_label} (ligne {line_number}).",
                    context=stripped,
                )
            )
            continue

        class_id = int(match.group(1))
        coords = [float(match.group(index)) for index in range(2, 6)]

        if class_id not in class_names:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"Classe {class_id} inconnue dans {relative_label} (ligne {line_number}).",
                )
            )

        for coord_name, value in zip(
            ("x_center", "y_center", "width", "height"), coords, strict=True
        ):
            if not 0.0 <= value <= 1.0:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        f"{coord_name} hors [0, 1] dans {relative_label} (ligne {line_number}).",
                        context=f"{coord_name}={value}",
                    )
                )
            elif coord_name in {"width", "height"} and value == 0.0:
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        f"{coord_name} nulle dans {relative_label} (ligne {line_number}).",
                        context=f"{coord_name}={value}",
                    )
                )

    return issues
