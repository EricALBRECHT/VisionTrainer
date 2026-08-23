from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

STATUS_FILENAME = "status.json"
REQUEST_FILENAME = "request.json"
LOG_FILENAME = "train.log"
PID_FILENAME = "worker.pid"

RunState = Literal["created", "running", "completed", "failed", "interrupted"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunMetrics:
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None


@dataclass
class RunStatus:
    run_id: str
    state: RunState = "created"
    model: str = ""
    epochs_total: int = 0
    epoch_current: int = 0
    imgsz: int = 640
    batch: int = -1
    device: str = "cpu"
    started_at: str | None = None
    finished_at: str | None = None
    progress_percent: float = 0.0
    best_model_path: str | None = None
    last_model_path: str | None = None
    error_message: str | None = None
    pid: int | None = None
    metrics: RunMetrics = field(default_factory=RunMetrics)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunStatus:
        metrics_raw = data.get("metrics") or {}
        metrics = RunMetrics(
            precision=_as_optional_float(metrics_raw.get("precision")),
            recall=_as_optional_float(metrics_raw.get("recall")),
            map50=_as_optional_float(metrics_raw.get("map50")),
            map50_95=_as_optional_float(metrics_raw.get("map50_95")),
        )
        return cls(
            run_id=str(data["run_id"]),
            state=data.get("state", "created"),  # type: ignore[arg-type]
            model=str(data.get("model") or ""),
            epochs_total=int(data.get("epochs_total") or 0),
            epoch_current=int(data.get("epoch_current") or 0),
            imgsz=int(data.get("imgsz") or 640),
            batch=int(data.get("batch") if data.get("batch") is not None else -1),
            device=str(data.get("device") or "cpu"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            progress_percent=float(data.get("progress_percent") or 0.0),
            best_model_path=data.get("best_model_path"),
            last_model_path=data.get("last_model_path"),
            error_message=data.get("error_message"),
            pid=int(data["pid"]) if data.get("pid") is not None else None,
            metrics=metrics,
        )


def status_path(run_dir: Path) -> Path:
    return run_dir / STATUS_FILENAME


def request_path(run_dir: Path) -> Path:
    return run_dir / REQUEST_FILENAME


def log_path(run_dir: Path) -> Path:
    return run_dir / LOG_FILENAME


def write_status(run_dir: Path, status: RunStatus) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = status_path(run_dir)
    path.write_text(json.dumps(status.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_status(run_dir: Path) -> RunStatus | None:
    path = status_path(run_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunStatus.from_dict(data)


def write_request(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = request_path(run_dir)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_request(run_dir: Path) -> dict[str, Any]:
    return json.loads(request_path(run_dir).read_text(encoding="utf-8"))


def compute_progress_percent(epoch_current: int, epochs_total: int) -> float:
    """Epoch-based progress: 0/3→0%, 1/3→33.3%, 3/3→100%."""
    if epochs_total <= 0:
        return 0.0
    current = max(0, min(epoch_current, epochs_total))
    return round(100.0 * current / epochs_total, 1)


def is_pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def reconcile_run_status(run_dir: Path) -> RunStatus | None:
    """
    If a run is marked running but its worker process is gone, mark it interrupted.
    """
    status = read_status(run_dir)
    if status is None:
        return None
    if status.state == "running" and not is_pid_alive(status.pid):
        status.state = "interrupted"
        status.finished_at = status.finished_at or utc_now_iso()
        status.error_message = (
            status.error_message
            or "Le processus d'entraînement n'existe plus. Run marqué comme interrupted."
        )
        write_status(run_dir, status)
    return status


def find_active_run(runs_root: Path) -> Path | None:
    """Return the first run directory still actively running (alive pid)."""
    if not runs_root.is_dir():
        return None
    for child in sorted(runs_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        status = reconcile_run_status(child)
        if status is not None and status.state == "running" and is_pid_alive(status.pid):
            return child
    return None


def read_log_tail(run_dir: Path, max_lines: int = 40) -> str:
    path = log_path(run_dir)
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def extract_metrics_from_trainer(trainer: Any) -> RunMetrics:
    """Best-effort extraction of final detection metrics from an Ultralytics trainer."""
    raw: dict[str, Any] = {}
    metrics_obj = getattr(trainer, "metrics", None)
    if isinstance(metrics_obj, dict):
        raw.update(metrics_obj)
    elif metrics_obj is not None:
        results_dict = getattr(metrics_obj, "results_dict", None)
        if isinstance(results_dict, dict):
            raw.update(results_dict)
        box = getattr(metrics_obj, "box", None)
        if box is not None:
            raw.setdefault("metrics/precision(B)", getattr(box, "mp", None))
            raw.setdefault("metrics/recall(B)", getattr(box, "mr", None))
            raw.setdefault("metrics/mAP50(B)", getattr(box, "map50", None))
            raw.setdefault("metrics/mAP50-95(B)", getattr(box, "map", None))

    return RunMetrics(
        precision=_first_metric(raw, ("metrics/precision(B)", "precision", "precision(B)")),
        recall=_first_metric(raw, ("metrics/recall(B)", "recall", "recall(B)")),
        map50=_first_metric(raw, ("metrics/mAP50(B)", "mAP50", "map50")),
        map50_95=_first_metric(raw, ("metrics/mAP50-95(B)", "mAP50-95", "map", "map50_95")),
    )


def _first_metric(raw: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in raw and raw[key] is not None:
            return _as_optional_float(raw[key])
    # case-insensitive fallback
    lowered = {str(k).lower(): v for k, v in raw.items()}
    for key in keys:
        if key.lower() in lowered and lowered[key.lower()] is not None:
            return _as_optional_float(lowered[key.lower()])
    return None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
