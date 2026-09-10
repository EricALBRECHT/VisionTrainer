from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ULTRALYTICS_PLOT_FILES = (
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "MaskPR_curve.png",
    "MaskP_curve.png",
    "MaskR_curve.png",
    "MaskF1_curve.png",
    # Legacy Ultralytics names (kept for older runs).
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
)

SESSION_INFERENCE_WEIGHTS_KEY = "inference_preferred_weights_path"


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    run_dir: Path
    state: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    model: str | None = None
    epochs: int | None = None
    imgsz: int | None = None
    batch: int | None = None
    device: str | None = None
    num_classes: int | None = None
    has_best: bool = False
    has_last: bool = False
    has_status: bool = False
    load_warning: str | None = None
    task: str = "detect"


@dataclass
class RunDetail:
    summary: RunSummary
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    accuracy_top1: float | None = None
    accuracy_top5: float | None = None
    mask_precision: float | None = None
    mask_recall: float | None = None
    mask_map50: float | None = None
    mask_map50_95: float | None = None
    best_pt: Path | None = None
    last_pt: Path | None = None
    plots: dict[str, Path] = field(default_factory=dict)
    error_message: str | None = None
    data_yaml: Path | None = None


@dataclass
class MetricsHistory:
    epochs: list[int] = field(default_factory=list)
    map50: list[float | None] = field(default_factory=list)
    map50_95: list[float | None] = field(default_factory=list)
    mask_map50: list[float | None] = field(default_factory=list)
    mask_map50_95: list[float | None] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    @property
    def has_map50(self) -> bool:
        return any(value is not None for value in self.map50)

    @property
    def has_map50_95(self) -> bool:
        return any(value is not None for value in self.map50_95)

    @property
    def has_mask_map50(self) -> bool:
        return any(value is not None for value in self.mask_map50)

    @property
    def has_mask_map50_95(self) -> bool:
        return any(value is not None for value in self.mask_map50_95)
