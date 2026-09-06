from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vision_trainer.training.data_yaml import write_resolved_data_yaml
from vision_trainer.training.device import DeviceChoice, DeviceError, cuda_oom_user_message, describe_device, resolve_device
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR, create_run_directory
from vision_trainer.training.status import (
    TERMINAL_STATES,
    RunStatus,
    acquire_launch_lock,
    attach_worker_pid,
    compute_progress_percent,
    extract_metrics_from_trainer,
    find_reserved_run,
    read_request,
    read_status,
    reconcile_run_status,
    release_launch_lock,
    request_path,
    utc_now_iso,
    write_request,
    write_status,
)
from vision_trainer.yolo.models import DatasetInfo

AVAILABLE_MODELS: dict[str, str] = {
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLO11m": "yolo11m.pt",
}

DEFAULT_MODEL_KEY = "YOLO11n"
DEFAULT_EPOCHS = 3
DEFAULT_IMGSZ = 640
IMGSZ_CHOICES = (320, 480, 640)
BATCH_AUTO = -1

SESSION_ACTIVE_RUN_KEY = "active_training_run_dir"


class TrainingError(Exception):
    """Raised when training cannot be prepared or completed."""


@dataclass(frozen=True)
class TrainingRequest:
    dataset: DatasetInfo
    model_key: str = DEFAULT_MODEL_KEY
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: int = BATCH_AUTO
    device_choice: DeviceChoice | str = "auto"
    runs_root: Path | None = None
    dataset_id: str | None = None


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    run_dir: Path
    resolved_data_yaml: Path
    status: RunStatus
    device: str
    device_label: str
    weights_name: str


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    run_dir: Path
    resolved_data_yaml: Path
    model_key: str
    weights_name: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    device_label: str
    best_weights: Path | None
    status: str


def build_train_kwargs(
    *,
    data_yaml: Path,
    run_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    """Build the keyword arguments passed to ``model.train``."""
    if epochs < 1:
        raise TrainingError("Le nombre d'epochs doit être au moins 1.")
    if imgsz not in IMGSZ_CHOICES:
        raise TrainingError(f"imgsz invalide : {imgsz}. Choix autorisés : {IMGSZ_CHOICES}.")

    # Absolute project/name avoids Ultralytics nesting under runs/detect/...
    run_dir_abs = run_dir.resolve()
    return {
        "data": str(Path(data_yaml).resolve()),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "device": device,
        "project": str(run_dir_abs.parent),
        "name": run_dir_abs.name,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }


def find_best_weights(run_dir: Path) -> Path | None:
    candidate = run_dir / "weights" / "best.pt"
    return candidate if candidate.is_file() else None


def find_last_weights(run_dir: Path) -> Path | None:
    candidate = run_dir / "weights" / "last.pt"
    return candidate if candidate.is_file() else None


def prepare_training_run(request: TrainingRequest) -> PreparedRun:
    """Create run directory, resolved YAML, status.json and request.json (no training yet)."""
    if "train" not in request.dataset.splits:
        raise TrainingError("Le dataset ne contient pas de split 'train' valide.")

    weights_name = AVAILABLE_MODELS.get(request.model_key)
    if weights_name is None:
        raise TrainingError(
            f"Modèle non supporté : {request.model_key!r}. "
            f"Choix : {', '.join(AVAILABLE_MODELS)}."
        )

    try:
        device = resolve_device(request.device_choice)
    except DeviceError as exc:
        raise TrainingError(str(exc)) from exc

    runs_root = request.runs_root if request.runs_root is not None else ARTIFACTS_RUNS_DIR
    reserved = find_reserved_run(runs_root)
    if reserved is not None:
        raise TrainingError(
            f"Un entraînement est déjà réservé ou en cours ({reserved.name}). "
            "Attendez la fin avant d'en lancer un autre."
        )

    try:
        acquire_launch_lock(runs_root)
    except BlockingIOError as exc:
        raise TrainingError(
            "Un autre lancement d'entraînement est déjà en cours. Réessayez dans un instant."
        ) from exc

    try:
        # Re-check under lock
        reserved = find_reserved_run(runs_root)
        if reserved is not None:
            raise TrainingError(
                f"Un entraînement est déjà réservé ou en cours ({reserved.name})."
            )

        run_id, run_dir = create_run_directory(runs_root=runs_root)
        resolved_yaml = write_resolved_data_yaml(request.dataset, run_dir)

        status = RunStatus(
            run_id=run_id,
            state="created",
            model=request.model_key,
            epochs_total=int(request.epochs),
            epoch_current=0,
            imgsz=int(request.imgsz),
            batch=int(request.batch),
            device=device,
            progress_percent=0.0,
        )
        write_status(run_dir, status)

        request_payload = {
            "run_id": run_id,
            "model_key": request.model_key,
            "weights_name": weights_name,
            "epochs": int(request.epochs),
            "imgsz": int(request.imgsz),
            "batch": int(request.batch),
            "device": device,
            "data_yaml": str(resolved_yaml.resolve()),
            "run_dir": str(run_dir.resolve()),
            "dataset_root": str(request.dataset.root.resolve()),
        }
        if request.dataset_id:
            request_payload["dataset_id"] = request.dataset_id
        write_request(run_dir, request_payload)

        return PreparedRun(
            run_id=run_id,
            run_dir=run_dir.resolve(),
            resolved_data_yaml=resolved_yaml.resolve(),
            status=status,
            device=device,
            device_label=describe_device(device),
            weights_name=weights_name,
        )
    finally:
        release_launch_lock(runs_root)


def start_training_subprocess(run_dir: Path, *, python_executable: str | None = None) -> int:
    """
    Launch the training worker in a dedicated subprocess and return immediately.

    Parent ownership: may only attach pid / created→running.
    Never overwrites terminal states written by a fast-failing worker.
    Releases the launch lock after spawn attempt.
    """
    import subprocess

    run_dir = run_dir.resolve()
    runs_root = run_dir.parent

    try:
        acquire_launch_lock(runs_root)
    except BlockingIOError as exc:
        raise TrainingError(
            "Un autre lancement d'entraînement est déjà en cours. Réessayez dans un instant."
        ) from exc

    try:
        status = read_status(run_dir)
        if status is None:
            raise TrainingError(f"status.json introuvable dans {run_dir}")
        if status.state in TERMINAL_STATES:
            raise TrainingError(
                f"Le run est déjà terminé ({status.state}); relance refusée."
            )
        if status.state == "running" and status.pid is not None:
            raise TrainingError("Ce run est déjà en cours d'exécution.")
        if not request_path(run_dir).is_file():
            raise TrainingError(f"request.json introuvable dans {run_dir}")

        reserved = find_reserved_run(runs_root)
        if reserved is not None and reserved.resolve() != run_dir:
            raise TrainingError(f"Un autre entraînement est déjà en cours ({reserved.name}).")

        python = python_executable or sys.executable
        log_file = run_dir / "train.log"
        log_handle = log_file.open("a", encoding="utf-8")

        try:
            try:
                process = subprocess.Popen(
                    [python, "-m", "vision_trainer.training.worker", "--run-dir", str(run_dir)],
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(Path.cwd()),
                    start_new_session=True,
                )
            except Exception as exc:
                _mark_failed(run_dir, status, f"Impossible de démarrer le worker : {exc}")
                raise TrainingError(f"Impossible de démarrer le worker : {exc}") from exc
        finally:
            log_handle.close()

        attach_worker_pid(run_dir, process.pid)
        return process.pid
    finally:
        release_launch_lock(runs_root)


def execute_training_from_run_dir(
    run_dir: Path,
    *,
    yolo_factory: Callable[[str], Any] | None = None,
) -> RunStatus:
    """
    Execute training for an already prepared run directory (used by the worker).

    Worker ownership: running → completed|failed.
    """
    run_dir = run_dir.resolve()
    status = read_status(run_dir)
    if status is None:
        raise TrainingError(f"status.json introuvable dans {run_dir}")
    if status.state in TERMINAL_STATES:
        return status

    request = read_request(run_dir)

    status.state = "running"
    status.pid = status.pid or os_getpid()
    status.started_at = status.started_at or utc_now_iso()
    status.epoch_current = 0
    status.progress_percent = 0.0
    status.error_message = None
    write_status(run_dir, status)

    weights_name = str(request["weights_name"])
    train_kwargs = build_train_kwargs(
        data_yaml=Path(request["data_yaml"]),
        run_dir=run_dir,
        epochs=int(request["epochs"]),
        imgsz=int(request["imgsz"]),
        batch=int(request["batch"]),
        device=str(request["device"]),
    )

    factory = yolo_factory or _default_yolo_factory
    try:
        model = factory(weights_name)
    except TrainingError:
        _mark_failed(run_dir, status, "Impossible de charger le modèle.")
        raise
    except Exception as exc:  # noqa: BLE001
        message = f"Impossible de charger le modèle {weights_name} : {exc}"
        _mark_failed(run_dir, status, message)
        raise TrainingError(message) from exc

    _register_progress_callbacks(model, run_dir)

    try:
        model.train(**train_kwargs)
    except Exception as exc:  # noqa: BLE001
        oom = cuda_oom_user_message(exc)
        message = oom or f"Erreur pendant l'entraînement : {exc}"
        _mark_failed(run_dir, status, message)
        raise TrainingError(message) from exc

    status = read_status(run_dir) or status
    best = find_best_weights(run_dir)
    last = find_last_weights(run_dir)

    if best is None or not best.is_file():
        message = (
            "Entraînement terminé sans weights/best.pt exploitable. "
            "Le run est marqué failed."
        )
        _mark_failed(run_dir, status, message)
        raise TrainingError(message)

    try:
        with best.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        message = f"weights/best.pt illisible : {exc}"
        _mark_failed(run_dir, status, message)
        raise TrainingError(message) from exc

    status.state = "completed"
    status.finished_at = utc_now_iso()
    status.epoch_current = status.epochs_total
    status.progress_percent = 100.0
    status.best_model_path = str(best)
    status.last_model_path = str(last) if last else None
    status.error_message = None
    write_status(run_dir, status)
    return status



def run_training(
    request: TrainingRequest,
    *,
    yolo_factory: Callable[[str], Any] | None = None,
) -> TrainingResult:
    """
    Synchronous helper used by unit tests: prepare + execute in-process.
    Production UI uses prepare_training_run + start_training_subprocess instead.
    """
    prepared = prepare_training_run(request)
    final_status = execute_training_from_run_dir(prepared.run_dir, yolo_factory=yolo_factory)

    return TrainingResult(
        run_id=prepared.run_id,
        run_dir=prepared.run_dir,
        resolved_data_yaml=prepared.resolved_data_yaml,
        model_key=request.model_key,
        weights_name=prepared.weights_name,
        epochs=request.epochs,
        imgsz=request.imgsz,
        batch=request.batch,
        device=prepared.device,
        device_label=prepared.device_label,
        best_weights=find_best_weights(prepared.run_dir),
        status=final_status.state,
    )


def load_run_for_ui(run_dir: Path) -> RunStatus | None:
    """Reconcile and return status for Streamlit display."""
    return reconcile_run_status(run_dir)


def os_getpid() -> int:
    import os

    return os.getpid()


def _mark_failed(run_dir: Path, status: RunStatus, message: str) -> None:
    current = read_status(run_dir) or status
    if current.state in TERMINAL_STATES and current.state != "failed":
        # Do not downgrade completed/interrupted.
        if current.state == "completed":
            return
    status.state = "failed"
    status.finished_at = utc_now_iso()
    status.error_message = message
    status.progress_percent = current.progress_percent
    write_status(run_dir, status)


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    """Best-effort conversion of trainer epoch/epochs fields (int, float, 0-dim tensor)."""
    if value is None:
        return default
    try:
        if hasattr(value, "item") and callable(value.item):
            value = value.item()
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _read_status_retry(run_dir: Path, *, attempts: int = 3) -> RunStatus | None:
    """Read status.json with short retries (bind-mount / concurrent UI readers)."""
    status: RunStatus | None = None
    for attempt in range(max(1, attempts)):
        status = read_status(run_dir)
        if status is not None:
            return status
        if attempt + 1 < attempts:
            import time

            time.sleep(0.01 * (attempt + 1))
    return status


def _trainer_epochs_total(trainer: Any, fallback: int) -> int:
    total = _coerce_non_negative_int(getattr(trainer, "epochs", None), 0)
    if total <= 0:
        args = getattr(trainer, "args", None)
        total = _coerce_non_negative_int(getattr(args, "epochs", None), 0)
    return total if total > 0 else max(1, fallback)


class EpochProgressTracker:
    """
    Persist epoch progress to status.json via Ultralytics callbacks.

    Ultralytics ``trainer.epoch`` is 0-based. We store 1-based completed epochs
    (after epoch 0 ends → epoch_current=1 → ~33% for a 3-epoch run).

    Writes only when the completed-epoch counter advances (never every batch).
    ``on_train_batch_end`` / ``on_train_epoch_start`` are fallbacks when
    ``on_train_epoch_end`` is skipped (e.g. OOM epoch restart paths).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._last_written_epoch = 0
        self._last_seen_epoch_0based: int | None = None
        self.write_count = 0

    def apply_completed_epoch(self, epoch_current: int, epochs_total: int | None = None) -> bool:
        """Update status.json when ``epoch_current`` advances. Returns True if a write occurred."""
        if epoch_current <= self._last_written_epoch:
            return False

        status = _read_status_retry(self.run_dir)
        if status is None or status.state in TERMINAL_STATES:
            return False

        total = int(epochs_total) if epochs_total and int(epochs_total) > 0 else int(status.epochs_total)
        if total <= 0:
            total = 1
        epoch_current = min(max(0, int(epoch_current)), total)

        if epoch_current <= self._last_written_epoch:
            return False
        if status.epoch_current >= epoch_current and status.epoch_current > 0:
            self._last_written_epoch = max(self._last_written_epoch, status.epoch_current)
            return False

        status.epoch_current = epoch_current
        status.epochs_total = total
        status.progress_percent = compute_progress_percent(epoch_current, total)
        status.state = "running"
        write_status(self.run_dir, status)
        self._last_written_epoch = epoch_current
        self.write_count += 1
        print(
            f"[vision-trainer] progression {epoch_current}/{total} "
            f"({status.progress_percent}%)",
            flush=True,
        )
        return True

    def on_train_epoch_end(self, trainer: Any) -> None:
        epoch_0 = _coerce_non_negative_int(getattr(trainer, "epoch", None), -1)
        if epoch_0 < 0:
            return
        status = _read_status_retry(self.run_dir)
        fallback_total = status.epochs_total if status is not None else 1
        total = _trainer_epochs_total(trainer, fallback_total)
        # After 0-based epoch N finishes, N+1 epochs are completed.
        self.apply_completed_epoch(epoch_0 + 1, total)
        self._last_seen_epoch_0based = epoch_0

    def on_train_epoch_start(self, trainer: Any) -> None:
        epoch_0 = _coerce_non_negative_int(getattr(trainer, "epoch", None), -1)
        if epoch_0 < 0:
            return
        self._last_seen_epoch_0based = epoch_0
        if epoch_0 <= 0:
            return
        status = _read_status_retry(self.run_dir)
        fallback_total = status.epochs_total if status is not None else 1
        total = _trainer_epochs_total(trainer, fallback_total)
        # Entering epoch N (0-based) means epochs 0..N-1 are done.
        self.apply_completed_epoch(epoch_0, total)

    def on_train_batch_end(self, trainer: Any) -> None:
        epoch_0 = _coerce_non_negative_int(getattr(trainer, "epoch", None), -1)
        if epoch_0 < 0:
            return
        if self._last_seen_epoch_0based is None:
            self._last_seen_epoch_0based = epoch_0
            return
        if epoch_0 == self._last_seen_epoch_0based:
            return
        # Epoch index advanced without a prior epoch_end write (fallback path).
        self._last_seen_epoch_0based = epoch_0
        status = _read_status_retry(self.run_dir)
        fallback_total = status.epochs_total if status is not None else 1
        total = _trainer_epochs_total(trainer, fallback_total)
        self.apply_completed_epoch(epoch_0, total)

    def on_train_end(self, trainer: Any) -> None:
        status = _read_status_retry(self.run_dir)
        if status is None or status.state in TERMINAL_STATES:
            return
        status.metrics = extract_metrics_from_trainer(trainer)
        best = find_best_weights(self.run_dir)
        last = find_last_weights(self.run_dir)
        best_attr = getattr(trainer, "best", None)
        last_attr = getattr(trainer, "last", None)
        if best is None and best_attr is not None:
            best_path = Path(str(best_attr))
            if best_path.is_file():
                best = best_path
        if last is None and last_attr is not None:
            last_path = Path(str(last_attr))
            if last_path.is_file():
                last = last_path
        status.best_model_path = str(best) if best else status.best_model_path
        status.last_model_path = str(last) if last else status.last_model_path
        status.epoch_current = status.epochs_total
        status.progress_percent = compute_progress_percent(status.epoch_current, status.epochs_total)
        write_status(self.run_dir, status)
        self._last_written_epoch = max(self._last_written_epoch, status.epoch_current)
        self.write_count += 1


def _build_progress_callbacks(
    run_dir: Path,
    tracker: EpochProgressTracker | None = None,
) -> dict[str, Callable[[Any], None]]:
    """Build Ultralytics callback map for live status.json progress updates."""
    progress = tracker if tracker is not None else EpochProgressTracker(run_dir)

    def on_train_start(trainer: Any) -> None:
        # Re-bind on the live trainer in case integrations reshuffled callbacks.
        callback_map = {
            "on_train_epoch_start": progress.on_train_epoch_start,
            "on_train_epoch_end": progress.on_train_epoch_end,
            "on_fit_epoch_end": progress.on_train_epoch_end,
            "on_train_batch_end": progress.on_train_batch_end,
            "on_train_end": progress.on_train_end,
        }
        trainer_callbacks = getattr(trainer, "callbacks", None)
        if not isinstance(trainer_callbacks, dict):
            return
        for event, callback in callback_map.items():
            bucket = trainer_callbacks.setdefault(event, [])
            if callback not in bucket:
                bucket.append(callback)

    return {
        "on_train_start": on_train_start,
        "on_train_epoch_start": progress.on_train_epoch_start,
        "on_train_epoch_end": progress.on_train_epoch_end,
        "on_fit_epoch_end": progress.on_train_epoch_end,
        "on_train_batch_end": progress.on_train_batch_end,
        "on_train_end": progress.on_train_end,
    }


def _register_progress_callbacks(model: Any, run_dir: Path) -> EpochProgressTracker:
    """Attach progress callbacks on a YOLO model (add_callback + direct list append)."""
    tracker = EpochProgressTracker(run_dir)
    callbacks = _build_progress_callbacks(run_dir, tracker=tracker)
    model_callbacks = getattr(model, "callbacks", None)

    for event, callback in callbacks.items():
        registered = False
        add_cb = getattr(model, "add_callback", None)
        if callable(add_cb):
            try:
                add_cb(event, callback)
                registered = True
            except Exception:  # noqa: BLE001 - mocks / alternate YOLO wrappers
                registered = False
        if not registered and isinstance(model_callbacks, dict):
            bucket = model_callbacks.setdefault(event, [])
            if callback not in bucket:
                bucket.append(callback)
    return tracker


def _default_yolo_factory(weights_name: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainingError(
            "Ultralytics n'est pas disponible. Installez le projet avec ses dépendances."
        ) from exc

    try:
        return YOLO(weights_name)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Impossible de charger le modèle {weights_name} : {exc}") from exc
