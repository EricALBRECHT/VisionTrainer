from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class CategoryKind(str, Enum):
    DATASET_ARCHIVE = "dataset_archive"
    DATASET_EXTRACTED = "dataset_extracted"
    RUN_CHECKPOINTS = "run_checkpoints"
    RUN_ARTIFACTS = "run_artifacts"
    FINAL_MODEL = "final_model"
    TEMPS = "temps"


CATEGORY_LABELS: dict[CategoryKind, str] = {
    CategoryKind.DATASET_ARCHIVE: "Dataset uploadé (ZIP)",
    CategoryKind.DATASET_EXTRACTED: "Dataset extrait",
    CategoryKind.RUN_CHECKPOINTS: "Checkpoints intermédiaires",
    CategoryKind.RUN_ARTIFACTS: "Artefacts du run",
    CategoryKind.FINAL_MODEL: "Modèle final (best.*)",
    CategoryKind.TEMPS: "Fichiers temporaires / cache",
}


# Categories removed by the quick "Nettoyer" action (never final model / dataset).
CLEANABLE_CATEGORIES = frozenset(
    {
        CategoryKind.RUN_CHECKPOINTS,
        CategoryKind.TEMPS,
    }
)


@dataclass(frozen=True)
class StoragePathRef:
    """A concrete path belonging to a category, always under an allowed root."""

    path: Path
    is_dir: bool = False


@dataclass
class StorageCategory:
    kind: CategoryKind
    label: str
    paths: list[StoragePathRef] = field(default_factory=list)
    size_bytes: int = 0
    shared: bool = False  # e.g. dataset used by several runs
    important: bool = False  # final model


@dataclass
class StorageGroup:
    """
    A selectable unit in the storage UI.

    Typically one training run (+ optional linked dataset), or an orphan dataset.
    """

    group_id: str
    title: str
    subtitle: str
    kind: str  # "run" | "orphan_dataset"
    run_id: str | None = None
    dataset_id: str | None = None
    run_dir: Path | None = None
    dataset_dir: Path | None = None
    created_at: str | None = None
    state: str | None = None
    categories: list[StorageCategory] = field(default_factory=list)
    size_bytes: int = 0
    protected: bool = False
    protection_reason: str | None = None
    legacy: bool = False  # incomplete metadata / old layout


@dataclass
class StorageOverview:
    total_bytes: int
    dataset_count: int
    run_count: int
    model_count: int
    free_disk_bytes: int | None
    groups: list[StorageGroup] = field(default_factory=list)


@dataclass
class DeletionItem:
    path: Path
    is_dir: bool
    category: CategoryKind
    group_id: str
    size_bytes: int = 0


@dataclass
class DeletionPlan:
    items: list[DeletionItem] = field(default_factory=list)
    estimated_bytes: int = 0
    group_ids: list[str] = field(default_factory=list)
    includes_final_model: bool = False
    includes_dataset: bool = False
    blocked: list[str] = field(default_factory=list)


@dataclass
class DeletionResult:
    files_removed: int = 0
    dirs_removed: int = 0
    bytes_freed: int = 0
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
