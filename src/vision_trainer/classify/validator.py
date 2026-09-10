"""Validate classification datasets and produce UI guidance."""

from __future__ import annotations

from pathlib import Path

from vision_trainer.classify.models import ClassifyDatasetInfo, ClassifyValidationResult
from vision_trainer.classify.parser import find_classify_dataset_root, load_classify_dataset
from vision_trainer.yolo.models import Severity, ValidationIssue


def class_size_guidance(image_count: int) -> str:
    """Informative (non-blocking) guidance for per-class image counts."""
    if image_count < 50:
        return "insuffisant / expérimental"
    if image_count < 100:
        return "faible"
    if image_count < 300:
        return "correct"
    return "recommandé"


def validate_classify_dataset(
    extract_or_root: Path,
    *,
    containment_root: Path | None = None,
) -> ClassifyValidationResult:
    """
    Analyse a classification dataset under ``extract_or_root``.

    ``containment_root`` is accepted for API symmetry with YOLO validation
    (future local-path imports) but is not required for ImageFolder layouts.
    """
    _ = containment_root
    issues: list[ValidationIssue] = []
    root = find_classify_dataset_root(extract_or_root)
    if root is None:
        issues.append(
            ValidationIssue(
                Severity.ERROR,
                "Dataset de classification introuvable. "
                "Attendu : train/<classe>/*.jpg (et idéalement val/<classe>/).",
            )
        )
        return ClassifyValidationResult(dataset=None, issues=issues)

    info, errors = load_classify_dataset(root)
    for message in errors:
        issues.append(ValidationIssue(Severity.ERROR, message))
    if info is None:
        return ClassifyValidationResult(dataset=None, issues=issues)

    if info.val_dir is None:
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                "Aucun dossier val/validation/valid trouvé. "
                "Ultralytics pourra se rabattre sur test/ ou d'autres splits.",
            )
        )
    elif info.val_image_count == 0:
        issues.append(
            ValidationIssue(Severity.WARNING, "Le split validation ne contient aucune image.")
        )

    for empty in info.empty_class_dirs:
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                f"Dossier de classe vide : {empty}",
                context=empty,
            )
        )

    if info.unsupported_files:
        preview = ", ".join(info.unsupported_files[:5])
        more = "" if len(info.unsupported_files) <= 5 else f" (+{len(info.unsupported_files) - 5})"
        issues.append(
            ValidationIssue(
                Severity.WARNING,
                f"Fichiers non image ignorés : {preview}{more}",
            )
        )

    # Per-class guidance + imbalance
    totals = [stats.total for stats in info.class_stats.values() if stats.total > 0]
    max_total = max(totals) if totals else 0
    for name, stats in sorted(info.class_stats.items()):
        guidance = class_size_guidance(stats.total)
        if stats.total < 50:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"Classe « {name} » : {stats.total} image(s) — {guidance} "
                    f"(< 50). L'entraînement reste possible.",
                    context=name,
                )
            )
        elif stats.total < 100:
            issues.append(
                ValidationIssue(
                    Severity.INFO,
                    f"Classe « {name} » : {stats.total} image(s) — {guidance}.",
                    context=name,
                )
            )
        if max_total >= 50 and stats.total > 0 and stats.total * 5 < max_total:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"Déséquilibre : la classe « {name} » ({stats.total}) "
                    f"a beaucoup moins d'images que la plus fournie ({max_total}).",
                    context=name,
                )
            )

    # Classes present in val/test but missing in train
    train_names = set(info.class_names.values())
    for name, stats in info.class_stats.items():
        if name not in train_names and stats.total > 0:
            issues.append(
                ValidationIssue(
                    Severity.WARNING,
                    f"Classe « {name} » présente hors train mais absente de train/.",
                    context=name,
                )
            )

    issues.append(
        ValidationIssue(
            Severity.INFO,
            f"Dataset classification : {info.num_classes} classe(s), "
            f"{info.total_images} image(s) "
            f"(train={info.train_image_count}, val={info.val_image_count}, "
            f"test={info.test_image_count}).",
        )
    )
    return ClassifyValidationResult(dataset=info, issues=issues)
