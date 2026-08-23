from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from vision_trainer.results.models import (
    ULTRALYTICS_PLOT_FILES,
    MetricsHistory,
    RunDetail,
    RunSummary,
)
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR
from vision_trainer.training.status import request_path, status_path

STATE_LABELS = {
    "completed": "terminé",
    "failed": "failed",
    "interrupted": "interrupted",
    "running": "running",
    "created": "created",
}


def discover_runs(runs_root: Path | None = None) -> list[RunSummary]:
    """
    Discover training runs under ``artifacts/runs/``, newest first.

    A corrupted run never prevents discovery of the others.
    """
    root = runs_root if runs_root is not None else ARTIFACTS_RUNS_DIR
    if not root.is_dir():
        return []

    summaries: list[RunSummary] = []
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        try:
            summaries.append(_load_run_summary(child))
        except Exception as exc:  # noqa: BLE001 - isolate bad runs
            summaries.append(
                RunSummary(
                    run_id=child.name,
                    run_dir=child.resolve(),
                    state="inconnu",
                    load_warning=f"Run illisible : {exc}",
                )
            )
    return summaries


def load_run(run_dir: Path) -> RunDetail:
    """Load full detail for one run directory (tolerant to missing artifacts)."""
    summary = _load_run_summary(run_dir)
    precision = recall = map50 = map50_95 = None
    error_message = summary.load_warning
    status_data = _safe_read_status_dict(run_dir)

    if status_data is not None:
        metrics = status_data.get("metrics") or {}
        if isinstance(metrics, dict):
            precision = _as_float(metrics.get("precision"))
            recall = _as_float(metrics.get("recall"))
            map50 = _as_float(metrics.get("map50"))
            map50_95 = _as_float(metrics.get("map50_95"))
        error_message = status_data.get("error_message") or error_message

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    plots = {
        name: path
        for name in ULTRALYTICS_PLOT_FILES
        for path in [run_dir / name]
        if path.is_file()
    }
    data_yaml = run_dir / "data.resolved.yaml"
    return RunDetail(
        summary=summary,
        precision=precision,
        recall=recall,
        map50=map50,
        map50_95=map50_95,
        best_pt=best if best.is_file() else None,
        last_pt=last if last.is_file() else None,
        plots=plots,
        error_message=error_message,
        data_yaml=data_yaml if data_yaml.is_file() else None,
    )


def load_metrics_history(run_dir: Path) -> MetricsHistory | None:
    """
    Read Ultralytics ``results.csv`` when present and extract mAP curves.

    Column names are normalized (strip / collapse spaces) before matching.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return None

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return MetricsHistory()
            raw_fields = list(reader.fieldnames)
            normalized_map = {_normalize_column(name): name for name in raw_fields}
            epoch_key = _find_column(normalized_map, ("epoch",))
            map50_key = _find_column(
                normalized_map,
                (
                    "metrics/map50(b)",
                    "metrics/map50",
                    "map50(b)",
                    "map50",
                    "metrics/map50(box)",
                ),
            )
            map5095_key = _find_column(
                normalized_map,
                (
                    "metrics/map50-95(b)",
                    "metrics/map50-95",
                    "map50-95(b)",
                    "map50-95",
                    "metrics/map50-95(box)",
                ),
            )

            epochs: list[int] = []
            map50_values: list[float | None] = []
            map5095_values: list[float | None] = []

            for index, row in enumerate(reader, start=1):
                epoch_raw = row.get(epoch_key) if epoch_key else None
                epoch_value = _as_int(epoch_raw)
                epochs.append(epoch_value if epoch_value is not None else index)

                map50_values.append(_as_float(row.get(map50_key)) if map50_key else None)
                map5095_values.append(
                    _as_float(row.get(map5095_key)) if map5095_key else None
                )

            return MetricsHistory(
                epochs=epochs,
                map50=map50_values,
                map50_95=map5095_values,
                columns=[_normalize_column(name) for name in raw_fields],
            )
    except (OSError, csv.Error, UnicodeError):
        return MetricsHistory()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "Non disponible"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_optional(value: Any, *, percent: bool = False) -> str:
    if value is None or value == "":
        return "Non disponible"
    if isinstance(value, float):
        if percent:
            return f"{value:.4f}"
        if value == int(value):
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


def _load_run_summary(run_dir: Path) -> RunSummary:
    run_dir = run_dir.resolve()
    run_id = run_dir.name
    has_best = (run_dir / "weights" / "best.pt").is_file()
    has_last = (run_dir / "weights" / "last.pt").is_file()
    status_file = status_path(run_dir)
    has_status = status_file.is_file()

    state = "inconnu"
    started_at = finished_at = None
    model = None
    epochs = imgsz = batch = None
    device = None
    load_warning = None
    duration_seconds = None

    status_data = _safe_read_status_dict(run_dir)
    if has_status and status_data is None:
        load_warning = "status.json invalide ou illisible."
    elif status_data is not None:
        raw_state = str(status_data.get("state") or "inconnu")
        state = STATE_LABELS.get(raw_state, raw_state)
        started_at = status_data.get("started_at")
        finished_at = status_data.get("finished_at")
        model = status_data.get("model") or None
        epochs = _as_int(status_data.get("epochs_total"))
        imgsz = _as_int(status_data.get("imgsz"))
        batch = _as_int(status_data.get("batch"))
        device = status_data.get("device") or None
        duration_seconds = _duration_seconds(started_at, finished_at)
    elif not has_status:
        # Fallback to request.json for older / incomplete runs
        request_data = _safe_read_request_dict(run_dir)
        if request_data is not None:
            model = request_data.get("model_key") or request_data.get("model")
            epochs = _as_int(request_data.get("epochs"))
            imgsz = _as_int(request_data.get("imgsz"))
            batch = _as_int(request_data.get("batch"))
            device = request_data.get("device")
            state = "terminé" if has_best else "incomplet"

    if state == "inconnu" and has_best:
        state = "terminé"
    elif state == "inconnu" and not has_status:
        state = "incomplet"

    num_classes = _read_num_classes(run_dir)

    return RunSummary(
        run_id=run_id,
        run_dir=run_dir,
        state=state,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        model=str(model) if model else None,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=str(device) if device else None,
        num_classes=num_classes,
        has_best=has_best,
        has_last=has_last,
        has_status=has_status,
        load_warning=load_warning,
    )


def _safe_read_status_dict(run_dir: Path) -> dict[str, Any] | None:
    path = status_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _safe_read_request_dict(run_dir: Path) -> dict[str, Any] | None:
    path = request_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_num_classes(run_dir: Path) -> int | None:
    yaml_path = run_dir / "data.resolved.yaml"
    if not yaml_path.is_file():
        return None
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("nc") is not None:
        return _as_int(payload.get("nc"))
    names = payload.get("names")
    if isinstance(names, list):
        return len(names)
    if isinstance(names, dict):
        return len(names)
    return None


def _duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    start = _parse_iso(started_at)
    end = _parse_iso(finished_at)
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds())


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalize_column(name: str) -> str:
    return " ".join(str(name).strip().split()).lower()


def _find_column(normalized_map: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        key = _normalize_column(candidate)
        if key in normalized_map:
            return normalized_map[key]
    # fuzzy contains for mAP variants
    for normalized, original in normalized_map.items():
        compact = normalized.replace(" ", "")
        for candidate in candidates:
            if _normalize_column(candidate).replace(" ", "") in compact:
                return original
    return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
