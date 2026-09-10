from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vision_trainer.yolo.models import Severity, ValidationIssue


@dataclass
class ClassifyClassStats:
    name: str
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0

    @property
    def total(self) -> int:
        return self.train_count + self.val_count + self.test_count


@dataclass
class ClassifyDatasetInfo:
    """Folder-based ImageFolder classification dataset (Ultralytics / torchvision)."""

    root: Path
    class_names: dict[int, str]
    class_stats: dict[str, ClassifyClassStats] = field(default_factory=dict)
    train_dir: Path | None = None
    val_dir: Path | None = None
    test_dir: Path | None = None
    train_image_count: int = 0
    val_image_count: int = 0
    test_image_count: int = 0
    unsupported_files: list[str] = field(default_factory=list)
    empty_class_dirs: list[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def total_images(self) -> int:
        return self.train_image_count + self.val_image_count + self.test_image_count


@dataclass
class ClassifyValidationResult:
    dataset: ClassifyDatasetInfo | None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def is_valid(self) -> bool:
        return self.dataset is not None and len(self.errors) == 0
