from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from vision_trainer.io_utils import atomic_write_json
from vision_trainer.tasks import DEFAULT_TASK, TaskType, normalize_task


STATUS_FILENAME = "status.json"
REQUEST_FILENAME = "request.json"
LOG_FILENAME = "train.log"
PID_FILENAME = "worker.pid"
LAUNCH_LOCK_FILENAME = ".launch.lock"

TERMINAL_STATES = frozenset({"completed", "failed", "interrupted"})
ACTIVE_STATES = frozenset({"created", "running"})

RunState = Literal["created", "running", "completed", "failed", "interrupted"]

# UI polling interval for the Training page while a run is active.
TRAINING_UI_REFRESH_SECONDS = 2


def should_auto_refresh(state: str | None) -> bool:
    """Return True only while the Training UI should poll status/log updates."""
    return state in ACTIVE_STATES


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunMetrics:
    precision: float | None = None
    recall: float | None = None
    map50: float | None = None
    map50_95: float | None = None
    accuracy_top1: float | None = None
    accuracy_top5: float | None = None


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
    task: TaskType = DEFAULT_TASK

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
            accuracy_top1=_as_optional_float(metrics_raw.get("accuracy_top1")),
            accuracy_top5=_as_optional_float(metrics_raw.get("accuracy_top5")),
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
            task=normalize_task(data.get("task")),
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
    atomic_write_json(path, status.to_dict())
    return path


def read_status(run_dir: Path) -> RunStatus | None:
    """Read status.json; return None if absent or temporarily unreadable/invalid."""
    path = status_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "run_id" not in data:
        return None
    try:
        return RunStatus.from_dict(data)
    except (TypeError, ValueError, KeyError):
        return None


def write_request(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = request_path(run_dir)
    atomic_write_json(path, payload)
    return path


def read_request(run_dir: Path) -> dict[str, Any]:
    data = read_request_safe(run_dir)
    if data is None:
        raise FileNotFoundError(f"request.json introuvable ou illisible dans {run_dir}")
    return data


def read_request_safe(run_dir: Path) -> dict[str, Any] | None:
    path = request_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


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
    Reconcile disk status with process liveness.

    - running + dead pid → interrupted
    - created without pid and without launch lock → interrupted (orphaned prepare)
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

    if status.state == "created" and status.pid is None and not is_launch_lock_held(run_dir.parent):
        # Only reclaim clearly orphaned prepares (not mid-launch).
        try:
            age = time.time() - status_path(run_dir).stat().st_mtime
        except OSError:
            age = 0
        if age > 120:
            status.state = "interrupted"
            status.finished_at = status.finished_at or utc_now_iso()
            status.error_message = (
                status.error_message
                or "Préparation orpheline (created sans worker). Run marqué comme interrupted."
            )
            write_status(run_dir, status)

    return status


def find_active_run(runs_root: Path) -> Path | None:
    """Return a run that is actively training (running + alive pid)."""
    if not runs_root.is_dir():
        return None
    for child in sorted(runs_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        status = reconcile_run_status(child)
        if status is not None and status.state == "running" and is_pid_alive(status.pid):
            return child
    return None


def find_reserved_run(runs_root: Path) -> Path | None:
    """
    Return a run that blocks a new launch (created or running).

    ``created`` and ``running`` are both considered reserved/active.
    """
    if not runs_root.is_dir():
        return None
    for child in sorted(runs_root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        status = reconcile_run_status(child)
        if status is None:
            continue
        if status.state == "created":
            return child
        if status.state == "running" and is_pid_alive(status.pid):
            return child
    return None


def launch_lock_path(runs_root: Path) -> Path:
    return runs_root / LAUNCH_LOCK_FILENAME


def is_launch_lock_held(runs_root: Path) -> bool:
    path = launch_lock_path(runs_root)
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return True
    return is_pid_alive(pid)


def acquire_launch_lock(runs_root: Path) -> Path:
    """Atomically create a launch lock file. Raises OSError if already locked."""
    runs_root.mkdir(parents=True, exist_ok=True)
    path = launch_lock_path(runs_root)
    if is_launch_lock_held(runs_root):
        raise BlockingIOError(f"Verrou d'entraînement déjà actif : {path}")
    # Clear stale lock file then create exclusively
    if path.exists() and not is_launch_lock_held(runs_root):
        path.unlink(missing_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
    finally:
        os.close(fd)
    return path


def release_launch_lock(runs_root: Path) -> None:
    path = launch_lock_path(runs_root)
    try:
        if path.is_file():
            owner = int(path.read_text(encoding="utf-8").strip() or "0")
            if owner in {0, os.getpid()} or not is_pid_alive(owner):
                path.unlink(missing_ok=True)
    except (OSError, ValueError):
        path.unlink(missing_ok=True)


def attach_worker_pid(run_dir: Path, pid: int) -> RunStatus | None:
    """
    Parent-only transition: created → running with pid.

    Never overwrites a terminal state already written by the worker.
    Preserves any progress fields already advanced by a fast worker.
    """
    status = read_status(run_dir)
    if status is None:
        return None
    if status.state in TERMINAL_STATES:
        return status
    if status.state not in {"created", "running"}:
        return status
    status.pid = pid
    if status.state == "created":
        status.state = "running"
    status.started_at = status.started_at or utc_now_iso()
    # Re-read once so we do not clobber epoch progress written between our
    # first read and this write (GPU runs can finish an epoch very quickly).
    latest = read_status(run_dir)
    if latest is not None:
        if latest.state in TERMINAL_STATES:
            return latest
        status.epoch_current = max(int(status.epoch_current), int(latest.epoch_current))
        status.progress_percent = max(float(status.progress_percent), float(latest.progress_percent))
        status.epochs_total = max(int(status.epochs_total), int(latest.epochs_total))
        if latest.metrics is not None and (
            latest.metrics.precision is not None
            or latest.metrics.recall is not None
            or latest.metrics.map50 is not None
            or latest.metrics.map50_95 is not None
        ):
            status.metrics = latest.metrics
        if latest.best_model_path:
            status.best_model_path = latest.best_model_path
        if latest.last_model_path:
            status.last_model_path = latest.last_model_path
    write_status(run_dir, status)
    return status


def read_log_tail(run_dir: Path, max_lines: int = 40) -> str:
    path = log_path(run_dir)
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def extract_metrics_from_trainer(trainer: Any) -> RunMetrics:
    """Best-effort extraction of Ultralytics trainer metrics (detect or classify)."""
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
        for key in ("top1", "top5", "accuracy_top1", "accuracy_top5"):
            if hasattr(metrics_obj, key):
                raw.setdefault(f"metrics/{key}", getattr(metrics_obj, key))

    return RunMetrics(
        precision=_first_metric(raw, ("metrics/precision(B)", "precision", "precision(B)")),
        recall=_first_metric(raw, ("metrics/recall(B)", "recall", "recall(B)")),
        map50=_first_metric(raw, ("metrics/mAP50(B)", "mAP50", "map50")),
        map50_95=_first_metric(raw, ("metrics/mAP50-95(B)", "mAP50-95", "map", "map50_95")),
        accuracy_top1=_first_metric(
            raw,
            (
                "metrics/accuracy_top1",
                "metrics/top1",
                "accuracy_top1",
                "top1",
                "train/accuracy_top1",
                "val/accuracy_top1",
            ),
        ),
        accuracy_top5=_first_metric(
            raw,
            (
                "metrics/accuracy_top5",
                "metrics/top5",
                "accuracy_top5",
                "top5",
                "train/accuracy_top5",
                "val/accuracy_top5",
            ),
        ),
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
