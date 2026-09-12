"""Parse Ultralytics results.csv for curves and best-epoch hints."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from vision_trainer.analysis.models import CurveSeries


def load_training_curves(run_dir: Path) -> dict[str, CurveSeries]:
    """Extract useful epoch curves from results.csv when present."""
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.is_file():
        return {}

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return {}
            field_map = {_norm(name): name for name in reader.fieldnames}
            epoch_key = _match(field_map, ("epoch",))
            series_specs = (
                ("train_loss", ("train/box_loss", "train/loss", "train/cls_loss")),
                ("val_loss", ("val/box_loss", "val/loss", "val/cls_loss")),
                ("map50", ("metrics/map50(b)", "metrics/map50", "map50")),
                ("map50_95", ("metrics/map50-95(b)", "metrics/map50-95", "map50-95")),
                ("mask_map50", ("metrics/map50(m)", "map50(m)")),
                ("mask_map50_95", ("metrics/map50-95(m)", "map50-95(m)")),
                ("accuracy_top1", ("metrics/accuracy_top1", "accuracy_top1", "metrics/top1")),
                ("accuracy_top5", ("metrics/accuracy_top5", "accuracy_top5", "metrics/top5")),
            )
            keys: dict[str, str | None] = {
                name: _match(field_map, candidates) for name, candidates in series_specs
            }

            epochs: list[int] = []
            buffers: dict[str, list[float | None]] = {name: [] for name in keys}
            for index, row in enumerate(reader, start=1):
                epoch_raw = row.get(epoch_key) if epoch_key else None
                epoch_val = _as_int(epoch_raw)
                epochs.append(epoch_val if epoch_val is not None else index)
                for name, col in keys.items():
                    buffers[name].append(_as_float(row.get(col)) if col else None)

            curves: dict[str, CurveSeries] = {}
            labels = {
                "train_loss": "Train loss",
                "val_loss": "Val loss",
                "map50": "Box mAP50",
                "map50_95": "Box mAP50-95",
                "mask_map50": "Mask mAP50",
                "mask_map50_95": "Mask mAP50-95",
                "accuracy_top1": "Top-1 accuracy",
                "accuracy_top5": "Top-5 accuracy",
            }
            for name, values in buffers.items():
                if any(v is not None for v in values):
                    curves[name] = CurveSeries(
                        epochs=list(epochs),
                        values=values,
                        label=labels.get(name, name),
                    )
            return curves
    except (OSError, csv.Error, UnicodeError):
        return {}


def infer_best_epoch(
    curves: dict[str, CurveSeries],
    *,
    task: str,
) -> tuple[int | None, str | None, float | None]:
    """
    Best-effort best epoch from curves.

    Prefers accuracy_top1 (classify), map50 (detect), mask_map50 then map50 (segment).
    Returns (epoch, metric_name, value) or (None, None, None) if unknown.
    """
    preference: tuple[str, ...]
    if task == "classify":
        preference = ("accuracy_top1", "accuracy_top5")
    elif task == "segment":
        preference = ("mask_map50", "map50", "mask_map50_95", "map50_95")
    else:
        preference = ("map50", "map50_95")

    for key in preference:
        series = curves.get(key)
        if series is None or not series.values:
            continue
        best_idx = None
        best_val = None
        for idx, value in enumerate(series.values):
            if value is None:
                continue
            if best_val is None or value > best_val:
                best_val = value
                best_idx = idx
        if best_idx is not None and best_val is not None:
            epoch = series.epochs[best_idx] if best_idx < len(series.epochs) else best_idx + 1
            return int(epoch), series.label or key, float(best_val)
    return None, None, None


def _norm(name: str) -> str:
    return " ".join(str(name).strip().split()).lower()


def _match(field_map: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        key = _norm(candidate)
        if key in field_map:
            return field_map[key]
    for normalized, original in field_map.items():
        compact = normalized.replace(" ", "")
        for candidate in candidates:
            if _norm(candidate).replace(" ", "") in compact:
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
