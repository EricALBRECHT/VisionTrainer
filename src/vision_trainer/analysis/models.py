"""Run analysis & comparison models (versioned, UI-agnostic)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ANALYSIS_VERSION = 1
ANALYSIS_FILENAME = "analysis.json"
DATASET_STATS_FILENAME = "dataset_stats.json"
COMPARISON_VERSION = 1


@dataclass
class ConfusionPair:
    true_label: str
    pred_label: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfusionPair:
        return cls(
            true_label=str(data.get("true_label") or ""),
            pred_label=str(data.get("pred_label") or ""),
            count=int(data.get("count") or 0),
        )


@dataclass
class ClassMetricRow:
    class_name: str
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    accuracy: float | None = None
    support: int | None = None
    # Segmentation mask metrics (optional)
    mask_precision: float | None = None
    mask_recall: float | None = None
    mask_map50: float | None = None
    mask_map50_95: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassMetricRow:
        return cls(
            class_name=str(data.get("class_name") or ""),
            precision=_opt_float(data.get("precision")),
            recall=_opt_float(data.get("recall")),
            map50=_opt_float(data.get("map50")),
            map50_95=_opt_float(data.get("map50_95")),
            accuracy=_opt_float(data.get("accuracy")),
            support=_opt_int(data.get("support")),
            mask_precision=_opt_float(data.get("mask_precision")),
            mask_recall=_opt_float(data.get("mask_recall")),
            mask_map50=_opt_float(data.get("mask_map50")),
            mask_map50_95=_opt_float(data.get("mask_map50_95")),
        )


@dataclass
class TopScore:
    class_name: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopScore:
        return cls(
            class_name=str(data.get("class_name") or ""),
            confidence=float(data.get("confidence") or 0.0),
        )


@dataclass
class ClassifyErrorSample:
    """One misclassified validation image (paths relative to dataset root when possible)."""

    image_path: str
    true_label: str
    pred_label: str
    confidence: float
    top_scores: list[TopScore] = field(default_factory=list)
    # Active-learning hooks (populated when available)
    margin_top1_top2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "true_label": self.true_label,
            "pred_label": self.pred_label,
            "confidence": self.confidence,
            "top_scores": [s.to_dict() for s in self.top_scores],
            "margin_top1_top2": self.margin_top1_top2,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassifyErrorSample:
        tops = data.get("top_scores") or []
        return cls(
            image_path=str(data.get("image_path") or ""),
            true_label=str(data.get("true_label") or ""),
            pred_label=str(data.get("pred_label") or ""),
            confidence=float(data.get("confidence") or 0.0),
            top_scores=[TopScore.from_dict(t) for t in tops if isinstance(t, dict)],
            margin_top1_top2=_opt_float(data.get("margin_top1_top2")),
        )


@dataclass
class DiagnosticFinding:
    code: str
    severity: str  # info | warning | critical
    title: str
    detail: str
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticFinding:
        return cls(
            code=str(data.get("code") or ""),
            severity=str(data.get("severity") or "info"),
            title=str(data.get("title") or ""),
            detail=str(data.get("detail") or ""),
            recommendation=(
                str(data["recommendation"]) if data.get("recommendation") else None
            ),
        )


@dataclass
class CurveSeries:
    epochs: list[int] = field(default_factory=list)
    values: list[float | None] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunAnalysis:
    """Persisted / in-memory analysis for one training run."""

    analysis_version: int = ANALYSIS_VERSION
    task: str = "detect"
    run_id: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    class_metrics: list[ClassMetricRow] = field(default_factory=list)
    confusion_pairs: list[ConfusionPair] = field(default_factory=list)
    confusion_matrix: list[list[int]] | None = None
    confusion_labels: list[str] = field(default_factory=list)
    classify_errors: list[ClassifyErrorSample] = field(default_factory=list)
    classify_errors_computed: bool = False
    diagnostics: list[DiagnosticFinding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    curves: dict[str, CurveSeries] = field(default_factory=dict)
    best_epoch: int | None = None
    best_metric_name: str | None = None
    best_metric_value: float | None = None
    assets: dict[str, str] = field(default_factory=dict)
    detection_errors_v1_note: str | None = None
    limitations: list[str] = field(default_factory=list)
    dataset_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_version": self.analysis_version,
            "task": self.task,
            "run_id": self.run_id,
            "summary": self.summary,
            "metrics": self.metrics,
            "class_metrics": [c.to_dict() for c in self.class_metrics],
            "confusion_pairs": [c.to_dict() for c in self.confusion_pairs],
            "confusion_matrix": self.confusion_matrix,
            "confusion_labels": list(self.confusion_labels),
            "classify_errors": [e.to_dict() for e in self.classify_errors],
            "classify_errors_computed": self.classify_errors_computed,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "recommendations": list(self.recommendations),
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
            "best_epoch": self.best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "assets": dict(self.assets),
            "detection_errors_v1_note": self.detection_errors_v1_note,
            "limitations": list(self.limitations),
            "dataset_stats": dict(self.dataset_stats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunAnalysis:
        curves_raw = data.get("curves") or {}
        curves: dict[str, CurveSeries] = {}
        if isinstance(curves_raw, dict):
            for key, value in curves_raw.items():
                if isinstance(value, dict):
                    curves[str(key)] = CurveSeries(
                        epochs=[int(e) for e in (value.get("epochs") or [])],
                        values=[_opt_float(v) for v in (value.get("values") or [])],
                        label=str(value.get("label") or key),
                    )
        return cls(
            analysis_version=int(data.get("analysis_version") or ANALYSIS_VERSION),
            task=str(data.get("task") or "detect"),
            run_id=str(data.get("run_id") or ""),
            summary=dict(data.get("summary") or {}),
            metrics=dict(data.get("metrics") or {}),
            class_metrics=[
                ClassMetricRow.from_dict(c)
                for c in (data.get("class_metrics") or [])
                if isinstance(c, dict)
            ],
            confusion_pairs=[
                ConfusionPair.from_dict(c)
                for c in (data.get("confusion_pairs") or [])
                if isinstance(c, dict)
            ],
            confusion_matrix=data.get("confusion_matrix"),
            confusion_labels=[str(x) for x in (data.get("confusion_labels") or [])],
            classify_errors=[
                ClassifyErrorSample.from_dict(e)
                for e in (data.get("classify_errors") or [])
                if isinstance(e, dict)
            ],
            classify_errors_computed=bool(data.get("classify_errors_computed")),
            diagnostics=[
                DiagnosticFinding.from_dict(d)
                for d in (data.get("diagnostics") or [])
                if isinstance(d, dict)
            ],
            recommendations=[str(r) for r in (data.get("recommendations") or [])],
            curves=curves,
            best_epoch=_opt_int(data.get("best_epoch")),
            best_metric_name=(
                str(data["best_metric_name"]) if data.get("best_metric_name") else None
            ),
            best_metric_value=_opt_float(data.get("best_metric_value")),
            assets={str(k): str(v) for k, v in (data.get("assets") or {}).items()},
            detection_errors_v1_note=(
                str(data["detection_errors_v1_note"])
                if data.get("detection_errors_v1_note")
                else None
            ),
            limitations=[str(x) for x in (data.get("limitations") or [])],
            dataset_stats=dict(data.get("dataset_stats") or {}),
        )


@dataclass
class MetricDelta:
    name: str
    value_a: float | None
    value_b: float | None
    delta_pp: float | None  # percentage points when values are ratios 0–1 or %

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassDelta:
    class_name: str
    metric_name: str
    value_a: float | None
    value_b: float | None
    delta_pp: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunComparison:
    comparison_version: int = COMPARISON_VERSION
    task: str = "detect"
    run_a_id: str = ""
    run_b_id: str = ""
    compatible: bool = True
    warnings: list[str] = field(default_factory=list)
    summary_a: dict[str, Any] = field(default_factory=dict)
    summary_b: dict[str, Any] = field(default_factory=dict)
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    class_deltas: list[ClassDelta] = field(default_factory=list)
    regressions: list[ClassDelta] = field(default_factory=list)
    improvements: list[ClassDelta] = field(default_factory=list)
    duration_a_seconds: float | None = None
    duration_b_seconds: float | None = None
    speedup_b_vs_a: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_version": self.comparison_version,
            "task": self.task,
            "run_a_id": self.run_a_id,
            "run_b_id": self.run_b_id,
            "compatible": self.compatible,
            "warnings": list(self.warnings),
            "summary_a": dict(self.summary_a),
            "summary_b": dict(self.summary_b),
            "metric_deltas": [m.to_dict() for m in self.metric_deltas],
            "class_deltas": [c.to_dict() for c in self.class_deltas],
            "regressions": [c.to_dict() for c in self.regressions],
            "improvements": [c.to_dict() for c in self.improvements],
            "duration_a_seconds": self.duration_a_seconds,
            "duration_b_seconds": self.duration_b_seconds,
            "speedup_b_vs_a": self.speedup_b_vs_a,
        }


def _opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
