"""Validate YOLO segmentation datasets (polygon labels)."""

from __future__ import annotations

from pathlib import Path

from vision_trainer.segment.labels import (
    SegmentDatasetInfo,
    SegmentSplitStats,
    looks_like_detection_bbox_line,
    parse_segment_label_file,
)
from vision_trainer.yolo.models import Severity, ValidationIssue, ValidationResult
from vision_trainer.yolo.parser import (
    label_path_for_image,
    list_images,
    load_dataset_from_directory,
    relative_path_without_extension,
)


def validate_segment_dataset(
    root: Path,
    *,
    containment_root: Path | None = None,
    max_label_files_detailed: int = 5000,
) -> ValidationResult:
    """
    Validate a YOLO segmentation dataset.

    Reuses detect layout parsing (data.yaml / classes.txt) but rejects
    detection-style bbox labels and validates polygon geometry.
    """
    dataset, parse_messages = load_dataset_from_directory(
        root,
        containment_root=containment_root,
    )
    issues: list[ValidationIssue] = []
    for message in parse_messages:
        if dataset is None:
            issues.append(ValidationIssue(Severity.ERROR, message))
        elif message.startswith("data.yaml généré"):
            issues.append(ValidationIssue(Severity.INFO, message))
        else:
            issues.append(ValidationIssue(Severity.WARNING, message))

    if dataset is None:
        return ValidationResult(dataset=None, issues=issues)

    issues.append(
        ValidationIssue(
            Severity.INFO,
            f"Dataset segmentation chargé depuis {dataset.yaml_path}.",
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

    split_stats: dict[str, SegmentSplitStats] = {}
    detection_hits = 0
    labels_scanned = 0

    for split_info in dataset.splits.values():
        stats = SegmentSplitStats(name=split_info.name)
        if split_info.images_dir is None or not split_info.images_dir.exists():
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"Le chemin images du split '{split_info.name}' est introuvable.",
                    context=str(split_info.images_dir),
                )
            )
            split_stats[split_info.name] = stats
            continue

        images = list_images(split_info.images_dir)
        stats.image_count = len(images)
        split_info.image_count = len(images)

        if not images:
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"Aucune image trouvée pour le split '{split_info.name}'.",
                )
            )
            split_stats[split_info.name] = stats
            continue

        issues.append(
            ValidationIssue(
                Severity.INFO,
                f"Split '{split_info.name}' : {stats.image_count} image(s).",
            )
        )

        if split_info.labels_dir is None or not split_info.labels_dir.exists():
            issues.append(
                ValidationIssue(
                    Severity.ERROR,
                    f"Le dossier labels du split '{split_info.name}' est introuvable.",
                    context=str(split_info.labels_dir),
                )
            )
            split_stats[split_info.name] = stats
            continue

        images_without: list[str] = []
        for image in images:
            relative = relative_path_without_extension(image, split_info.images_dir)
            label_path = label_path_for_image(
                image, split_info.images_dir, split_info.labels_dir
            )
            if not label_path.is_file():
                images_without.append(relative)
                continue

            labels_scanned += 1
            # Cap detailed parsing on huge datasets (still count structure).
            if labels_scanned > max_label_files_detailed:
                continue

            # Quick detection-format scan before full parse
            try:
                raw = label_path.read_text(encoding="utf-8")
            except OSError as exc:
                stats.invalid_label_files += 1
                issues.append(
                    ValidationIssue(
                        Severity.ERROR,
                        f"Impossible de lire le label {relative}.",
                        context=str(exc),
                    )
                )
                continue

            for line_number, line in enumerate(raw.splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if looks_like_detection_bbox_line(parts):
                    detection_hits += 1
                    issues.append(
                        ValidationIssue(
                            Severity.ERROR,
                            "Ce dataset semble contenir des bounding boxes de détection "
                            "et non des polygones de segmentation.",
                            context=f"{relative}:{line_number}",
                        )
                    )

            instances, errors = parse_segment_label_file(
                label_path, class_names=dataset.class_names
            )
            if errors and not instances:
                stats.invalid_label_files += 1
            for err in errors:
                # Avoid duplicating the detection message flood
                if "bounding boxes de détection" in err:
                    continue
                issues.append(
                    ValidationIssue(Severity.ERROR, err, context=split_info.name)
                )

            for instance in instances:
                class_name = dataset.class_names[instance.class_id]
                stats.instance_count += 1
                stats.instances_by_class[class_name] = (
                    stats.instances_by_class.get(class_name, 0) + 1
                )
                stats.polygon_count += 1
                stats.total_polygon_points += instance.num_points

        stats.images_without_labels = images_without
        split_info.images_without_labels = images_without
        if images_without:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"{len(images_without)} image(s) sans label dans '{split_info.name}'.",
                    context=", ".join(images_without[:5]),
                )
            )
        split_stats[split_info.name] = stats

    # Classes missing from a split (informational)
    for split_name, stats in split_stats.items():
        if stats.image_count == 0:
            continue
        present = set(stats.instances_by_class)
        missing = [name for name in dataset.class_names.values() if name not in present]
        if missing and stats.instance_count > 0:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"Classes absentes du split '{split_name}' : {', '.join(missing)}.",
                )
            )

    if detection_hits > 0:
        # Already emitted per-line; add a summary once
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                f"{detection_hits} ligne(s) au format détection (bbox) détectée(s). "
                "Aucune conversion automatique bbox → polygone n'est effectuée.",
            )
        )

    mean_points = None
    total_poly = sum(s.polygon_count for s in split_stats.values())
    total_pts = sum(s.total_polygon_points for s in split_stats.values())
    if total_poly > 0:
        mean_points = total_pts / total_poly
        issues.append(
            ValidationIssue(
                Severity.INFO,
                f"{sum(s.instance_count for s in split_stats.values())} instance(s) ; "
                f"moyenne {mean_points:.1f} points/polygone.",
            )
        )

    segment_info = SegmentDatasetInfo(
        root=dataset.root,
        yaml_path=dataset.yaml_path,
        class_names=dataset.class_names,
        splits=dataset.splits,
        split_stats=split_stats,
        detection_label_hits=detection_hits,
    )
    # Attach on dataset via ValidationResult — keep DatasetInfo for training YAML
    # by storing SegmentDatasetInfo in a custom way: return ValidationResult with
    # the base DatasetInfo (needed by write_resolved_data_yaml) and put segment
    # stats on issues / session separately.
    result = ValidationResult(dataset=dataset, issues=issues)
    # Stash segment enrichment for callers (session layer reads this attribute).
    result.segment_info = segment_info  # type: ignore[attr-defined]
    return result
