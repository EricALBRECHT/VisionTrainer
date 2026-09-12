"""Build a RunAnalysis from existing VisionTrainer / Ultralytics artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vision_trainer.analysis.curves import infer_best_epoch, load_training_curves
from vision_trainer.analysis.dataset_stats import collect_dataset_stats
from vision_trainer.analysis.diagnostics import build_diagnostics
from vision_trainer.analysis.models import (
    ANALYSIS_VERSION,
    ClassMetricRow,
    RunAnalysis,
)
from vision_trainer.analysis.store import load_analysis, save_analysis
from vision_trainer.results.catalog import format_duration, load_run
from vision_trainer.results.models import ULTRALYTICS_PLOT_FILES
from vision_trainer.tasks import normalize_task, task_label_fr
from vision_trainer.training.export_weights import find_export_weights
from vision_trainer.training.status import read_request_safe, read_status


DETECTION_ERRORS_V1_NOTE = (
    "V1 détection : les faux positifs / faux négatifs image par image ne sont pas "
    "encore reconstruits automatiquement. Les métriques globales, courbes, assets "
    "Ultralytics et diagnostics basés sur les métriques/classes disponibles sont fournis. "
    "Une évaluation détaillée FP/FN pourra être ajoutée plus tard."
)


def build_run_analysis(
    run_dir: Path,
    *,
    persist: bool = True,
    refresh_dataset_stats: bool = False,
) -> RunAnalysis:
    """
    Assemble analysis from status/request/results.csv/assets.

    Merges previously persisted classify_errors when present.
    Never crashes on incomplete legacy runs.
    """
    run_dir = Path(run_dir)
    previous = load_analysis(run_dir)

    try:
        detail = load_run(run_dir)
    except Exception:  # noqa: BLE001
        detail = None

    status = read_status(run_dir)
    request = read_request_safe(run_dir) or {}
    task = normalize_task(
        (status.task if status else None)
        or request.get("task")
        or (detail.summary.task if detail else None)
    )

    summary = _build_summary(run_dir, status, request, detail)
    metrics = _build_metrics(status, detail, task)
    curves = load_training_curves(run_dir)
    best_epoch, best_name, best_val = infer_best_epoch(curves, task=task)
    assets = _list_assets(run_dir)
    dataset_stats = collect_dataset_stats(
        run_dir, force_refresh=refresh_dataset_stats
    )
    class_metrics = _load_optional_class_metrics(run_dir, request)

    limitations: list[str] = []
    if not (run_dir / "results.csv").is_file():
        limitations.append("results.csv absent — courbes d'entraînement indisponibles.")
    if task != "classify" and not class_metrics:
        limitations.append(
            "Métriques par classe numériques non disponibles pour ce run "
            "(Ultralytics ne les persiste pas toujours hors PNG)."
        )

    analysis = RunAnalysis(
        analysis_version=ANALYSIS_VERSION,
        task=task,
        run_id=summary.get("run_id") or run_dir.name,
        summary=summary,
        metrics=metrics,
        class_metrics=class_metrics,
        curves=curves,
        best_epoch=best_epoch,
        best_metric_name=best_name,
        best_metric_value=best_val,
        assets=assets,
        dataset_stats=dataset_stats,
        limitations=limitations,
        detection_errors_v1_note=(
            DETECTION_ERRORS_V1_NOTE if task in {"detect", "segment"} else None
        ),
    )

    if previous is not None:
        if previous.classify_errors_computed:
            analysis.classify_errors = list(previous.classify_errors)
            analysis.classify_errors_computed = True
        if previous.confusion_matrix is not None:
            analysis.confusion_matrix = previous.confusion_matrix
            analysis.confusion_labels = list(previous.confusion_labels)
            analysis.confusion_pairs = list(previous.confusion_pairs)

    analysis.diagnostics, analysis.recommendations = build_diagnostics(analysis)

    if persist:
        save_analysis(run_dir, analysis)
    return analysis


def _build_summary(
    run_dir: Path,
    status: Any,
    request: dict[str, Any],
    detail: Any,
) -> dict[str, Any]:
    run_id = run_dir.name
    model = None
    epochs = imgsz = batch = None
    device = None
    device_name = None
    started = finished = None
    duration = None
    state = "inconnu"
    seed = None
    environment: dict[str, Any] = {}

    if status is not None:
        run_id = status.run_id or run_id
        model = status.model or None
        epochs = status.epochs_total or None
        imgsz = status.imgsz or None
        batch = status.batch
        device = status.device or None
        started = status.started_at
        finished = status.finished_at
        state = status.state
        # Optional future fields stored on status dict via to_dict extras — read raw
    raw_status = _raw_status(run_dir)
    if raw_status:
        device_name = raw_status.get("device_name") or request.get("device_name")
        seed = raw_status.get("seed") if raw_status.get("seed") is not None else request.get("seed")
        environment = dict(raw_status.get("environment") or request.get("environment") or {})

    if detail is not None:
        model = model or detail.summary.model
        epochs = epochs if epochs is not None else detail.summary.epochs
        imgsz = imgsz if imgsz is not None else detail.summary.imgsz
        batch = batch if batch is not None else detail.summary.batch
        device = device or detail.summary.device
        started = started or detail.summary.started_at
        finished = finished or detail.summary.finished_at
        duration = detail.summary.duration_seconds
        state = detail.summary.state or state

    if duration is None and started and finished:
        duration = _duration_seconds(started, finished)

    export = find_export_weights(
        run_dir,
        export_model_path=getattr(status, "export_model_path", None) if status else None,
    )
    task = normalize_task(
        (status.task if status else None) or request.get("task")
    )

    return {
        "run_id": run_id,
        "task": task,
        "task_label": task_label_fr(task),
        "state": state,
        "dataset_name": request.get("dataset_name")
        or Path(str(request.get("dataset_root") or request.get("data_dir") or "")).name
        or None,
        "dataset_root": request.get("dataset_root") or request.get("data_dir"),
        "dataset_source_type": request.get("dataset_source_type"),
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "device_name": device_name or ("Non enregistré" if device else "Non disponible"),
        "started_at": started,
        "finished_at": finished,
        "duration_seconds": duration,
        "duration_label": format_duration(duration),
        "seed": seed if seed is not None else "Non enregistré",
        "environment": environment,
        "best_model_path": getattr(status, "best_model_path", None) if status else None,
        "export_model_path": str(export) if export else getattr(status, "export_model_path", None),
        "export_model_name": export.name if export else None,
    }


def _build_metrics(status: Any, detail: Any, task: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    src = None
    if status is not None:
        src = status.metrics
    if task == "classify":
        metrics["accuracy_top1"] = _pick(
            getattr(src, "accuracy_top1", None) if src else None,
            getattr(detail, "accuracy_top1", None) if detail else None,
        )
        metrics["accuracy_top5"] = _pick(
            getattr(src, "accuracy_top5", None) if src else None,
            getattr(detail, "accuracy_top5", None) if detail else None,
        )
    elif task == "segment":
        metrics["precision"] = _pick(
            getattr(src, "precision", None) if src else None,
            getattr(detail, "precision", None) if detail else None,
        )
        metrics["recall"] = _pick(
            getattr(src, "recall", None) if src else None,
            getattr(detail, "recall", None) if detail else None,
        )
        metrics["map50"] = _pick(
            getattr(src, "map50", None) if src else None,
            getattr(detail, "map50", None) if detail else None,
        )
        metrics["map50_95"] = _pick(
            getattr(src, "map50_95", None) if src else None,
            getattr(detail, "map50_95", None) if detail else None,
        )
        metrics["mask_precision"] = _pick(
            getattr(src, "mask_precision", None) if src else None,
            getattr(detail, "mask_precision", None) if detail else None,
        )
        metrics["mask_recall"] = _pick(
            getattr(src, "mask_recall", None) if src else None,
            getattr(detail, "mask_recall", None) if detail else None,
        )
        metrics["mask_map50"] = _pick(
            getattr(src, "mask_map50", None) if src else None,
            getattr(detail, "mask_map50", None) if detail else None,
        )
        metrics["mask_map50_95"] = _pick(
            getattr(src, "mask_map50_95", None) if src else None,
            getattr(detail, "mask_map50_95", None) if detail else None,
        )
    else:
        metrics["precision"] = _pick(
            getattr(src, "precision", None) if src else None,
            getattr(detail, "precision", None) if detail else None,
        )
        metrics["recall"] = _pick(
            getattr(src, "recall", None) if src else None,
            getattr(detail, "recall", None) if detail else None,
        )
        metrics["map50"] = _pick(
            getattr(src, "map50", None) if src else None,
            getattr(detail, "map50", None) if detail else None,
        )
        metrics["map50_95"] = _pick(
            getattr(src, "map50_95", None) if src else None,
            getattr(detail, "map50_95", None) if detail else None,
        )
    return metrics


def _load_optional_class_metrics(
    run_dir: Path,
    request: dict[str, Any],
) -> list[ClassMetricRow]:
    """Load per-class metrics from request extras or class_metrics.json if present."""
    raw = request.get("class_metrics")
    if isinstance(raw, list):
        return [ClassMetricRow.from_dict(c) for c in raw if isinstance(c, dict)]

    path = run_dir / "class_metrics.json"
    if path.is_file():
        import json

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return []
        if isinstance(data, list):
            return [ClassMetricRow.from_dict(c) for c in data if isinstance(c, dict)]
        if isinstance(data, dict) and isinstance(data.get("classes"), list):
            return [
                ClassMetricRow.from_dict(c)
                for c in data["classes"]
                if isinstance(c, dict)
            ]
    return []


def _list_assets(run_dir: Path) -> dict[str, str]:
    assets: dict[str, str] = {}
    extra = (
        "labels.jpg",
        "labels_correlogram.jpg",
    )
    names = list(ULTRALYTICS_PLOT_FILES) + list(extra)
    for path in sorted(run_dir.glob("val_batch*_pred.jpg")):
        assets[path.name] = str(path)
    for path in sorted(run_dir.glob("val_batch*_labels.jpg")):
        assets[path.name] = str(path)
    for name in names:
        path = run_dir / name
        if path.is_file():
            assets[name] = str(path)
    return assets


def _raw_status(run_dir: Path) -> dict[str, Any] | None:
    import json

    from vision_trainer.training.status import status_path

    path = status_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _pick(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    from datetime import datetime

    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())
