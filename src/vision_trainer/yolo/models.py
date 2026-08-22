from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    message: str
    context: str | None = None


@dataclass
class SplitInfo:
    name: str
    images_dir: Path | None
    labels_dir: Path | None
    image_count: int = 0
    label_file_count: int = 0
    matched_pairs: int = 0
    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)
    declared_ref: str | None = None
    resolved_via_fallback: bool = False
    fallback_ref: str | None = None


@dataclass
class DatasetInfo:
    root: Path
    yaml_path: Path
    class_names: dict[int, str]
    splits: dict[str, SplitInfo] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass
class SampleAnnotation:
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class ValidationResult:
    dataset: DatasetInfo | None
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == Severity.WARNING]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == Severity.INFO]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
