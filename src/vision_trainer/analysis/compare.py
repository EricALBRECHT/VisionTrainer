"""Compare two training runs (same task only for full comparison)."""

from __future__ import annotations

from pathlib import Path

from vision_trainer.analysis.builder import build_run_analysis
from vision_trainer.analysis.models import (
    ClassDelta,
    MetricDelta,
    RunAnalysis,
    RunComparison,
)
from vision_trainer.tasks import normalize_task


def compare_runs(
    run_a_dir: Path,
    run_b_dir: Path,
    *,
    analysis_a: RunAnalysis | None = None,
    analysis_b: RunAnalysis | None = None,
) -> RunComparison:
    a = analysis_a or build_run_analysis(run_a_dir, persist=False)
    b = analysis_b or build_run_analysis(run_b_dir, persist=False)

    task_a = normalize_task(a.task)
    task_b = normalize_task(b.task)
    warnings: list[str] = []
    compatible = True

    if task_a != task_b:
        compatible = False
        warnings.append(
            f"Tâches incompatibles : {task_a} vs {task_b}. Comparaison métrique refusée."
        )

    ds_a = (a.summary.get("dataset_name") or "").strip()
    ds_b = (b.summary.get("dataset_name") or "").strip()
    if ds_a and ds_b and ds_a != ds_b:
        warnings.append(
            f"Datasets différents : « {ds_a} » vs « {ds_b} ». "
            "Les deltas restent calculés mais l'interprétation doit rester prudente."
        )

    classes_a = {row.class_name for row in a.class_metrics}
    classes_b = {row.class_name for row in b.class_metrics}
    if classes_a and classes_b and classes_a != classes_b:
        warnings.append(
            "Jeux de classes différents entre les deux runs — "
            "comparaison par classe limitée à l'intersection."
        )

    comparison = RunComparison(
        task=task_a if task_a == task_b else task_a,
        run_a_id=a.run_id or Path(run_a_dir).name,
        run_b_id=b.run_id or Path(run_b_dir).name,
        compatible=compatible,
        warnings=warnings,
        summary_a=dict(a.summary),
        summary_b=dict(b.summary),
        duration_a_seconds=_as_float(a.summary.get("duration_seconds")),
        duration_b_seconds=_as_float(b.summary.get("duration_seconds")),
    )

    if not compatible:
        return comparison

    comparison.metric_deltas = _metric_deltas(a, b, task_a)
    comparison.class_deltas = _class_deltas(a, b, task_a)
    comparison.regressions = [
        d for d in comparison.class_deltas if d.delta_pp is not None and d.delta_pp < -1e-9
    ]
    comparison.improvements = [
        d for d in comparison.class_deltas if d.delta_pp is not None and d.delta_pp > 1e-9
    ]
    comparison.speedup_b_vs_a = _speedup(
        comparison.duration_a_seconds, comparison.duration_b_seconds
    )
    return comparison


def percentage_point_delta(value_a: float | None, value_b: float | None) -> float | None:
    """
    Delta in percentage points.

    Values may be ratios (0–1) or already percentages (0–100). If both look like
    ratios (<=1.5), convert to percent before subtracting.
    """
    if value_a is None or value_b is None:
        return None
    a = float(value_a)
    b = float(value_b)
    if abs(a) <= 1.5 and abs(b) <= 1.5:
        a *= 100.0
        b *= 100.0
    return round(b - a, 4)


def format_pp(delta: float | None) -> str:
    if delta is None:
        return "Non disponible"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.2f} points de pourcentage"


def _metric_deltas(a: RunAnalysis, b: RunAnalysis, task: str) -> list[MetricDelta]:
    keys: tuple[str, ...]
    if task == "classify":
        keys = ("accuracy_top1", "accuracy_top5")
    elif task == "segment":
        keys = (
            "precision",
            "recall",
            "map50",
            "map50_95",
            "mask_precision",
            "mask_recall",
            "mask_map50",
            "mask_map50_95",
        )
    else:
        keys = ("precision", "recall", "map50", "map50_95")

    out: list[MetricDelta] = []
    for key in keys:
        va = _as_float(a.metrics.get(key))
        vb = _as_float(b.metrics.get(key))
        out.append(
            MetricDelta(
                name=key,
                value_a=va,
                value_b=vb,
                delta_pp=percentage_point_delta(va, vb),
            )
        )
    return out


def _class_deltas(a: RunAnalysis, b: RunAnalysis, task: str) -> list[ClassDelta]:
    metric_name = (
        "accuracy"
        if task == "classify"
        else "mask_map50"
        if task == "segment"
        else "map50"
    )

    def value_of(row) -> float | None:
        if task == "classify":
            return row.accuracy if row.accuracy is not None else row.recall
        if task == "segment":
            return row.mask_map50 if row.mask_map50 is not None else row.map50
        return row.map50 if row.map50 is not None else row.recall

    map_a = {r.class_name: value_of(r) for r in a.class_metrics}
    map_b = {r.class_name: value_of(r) for r in b.class_metrics}
    names = sorted(set(map_a) & set(map_b))
    out: list[ClassDelta] = []
    for name in names:
        va = map_a.get(name)
        vb = map_b.get(name)
        out.append(
            ClassDelta(
                class_name=name,
                metric_name=metric_name,
                value_a=va,
                value_b=vb,
                delta_pp=percentage_point_delta(va, vb),
            )
        )
    out.sort(key=lambda d: (d.delta_pp is None, d.delta_pp or 0.0, d.class_name))
    return out


def _speedup(duration_a: float | None, duration_b: float | None) -> float | None:
    if duration_a is None or duration_b is None:
        return None
    if duration_a <= 0 or duration_b <= 0:
        return None
    return round(duration_a / duration_b, 2)


def _as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
