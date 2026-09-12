"""Prepare Ultralytics YOLO segmentation training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_trainer.analysis.environment import (
    DEFAULT_ULTRALYTICS_SEED,
    collect_training_environment,
    resolve_device_display_name,
)
from vision_trainer.tasks import normalize_task
from vision_trainer.training.data_yaml import write_resolved_data_yaml
from vision_trainer.training.device import DeviceChoice, DeviceError, describe_device, resolve_device
from vision_trainer.training.runs import ARTIFACTS_RUNS_DIR, create_run_directory
from vision_trainer.training.status import (
    RunStatus,
    acquire_launch_lock,
    find_reserved_run,
    release_launch_lock,
    write_request,
    write_status,
)
from vision_trainer.training.trainer import (
    BATCH_AUTO,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    IMGSZ_CHOICES,
    TrainingError,
    build_train_kwargs,
)
from vision_trainer.yolo.models import DatasetInfo

AVAILABLE_SEG_MODELS: dict[str, str] = {
    "YOLO11n-seg": "yolo11n-seg.pt",
    "YOLO11s-seg": "yolo11s-seg.pt",
    "YOLO11m-seg": "yolo11m-seg.pt",
}

DEFAULT_SEG_MODEL_KEY = "YOLO11n-seg"
SESSION_ACTIVE_SEG_RUN_KEY = "active_segment_training_run_dir"


@dataclass(frozen=True)
class SegmentTrainingRequest:
    dataset: DatasetInfo
    model_key: str = DEFAULT_SEG_MODEL_KEY
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: int = BATCH_AUTO
    device_choice: DeviceChoice | str = "auto"
    runs_root: Path | None = None
    dataset_id: str | None = None
    dataset_source_type: str | None = None
    dataset_name: str | None = None


@dataclass(frozen=True)
class PreparedSegmentRun:
    run_id: str
    run_dir: Path
    resolved_data_yaml: Path
    status: RunStatus
    device: str
    device_label: str
    weights_name: str


def prepare_segment_training_run(request: SegmentTrainingRequest) -> PreparedSegmentRun:
    """Create run directory + status/request for a segmentation training."""
    if "train" not in request.dataset.splits:
        raise TrainingError("Le dataset de segmentation ne contient pas de split 'train'.")

    weights_name = AVAILABLE_SEG_MODELS.get(request.model_key)
    if weights_name is None:
        raise TrainingError(
            f"Modèle de segmentation non supporté : {request.model_key!r}. "
            f"Choix : {', '.join(AVAILABLE_SEG_MODELS)}."
        )

    if request.imgsz not in IMGSZ_CHOICES:
        raise TrainingError(
            f"imgsz invalide : {request.imgsz}. Choix autorisés : {IMGSZ_CHOICES}."
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
            task="segment",
            device_name=resolve_device_display_name(device) or (
                "CPU" if str(device).lower() == "cpu" else None
            ),
            seed=DEFAULT_ULTRALYTICS_SEED,
            environment=collect_training_environment(),
        )
        write_status(run_dir, status)

        # Validate train kwargs early (same shape as detect).
        build_train_kwargs(
            data_yaml=resolved_yaml,
            run_dir=run_dir,
            epochs=int(request.epochs),
            imgsz=int(request.imgsz),
            batch=int(request.batch),
            device=device,
        )

        request_payload = {
            "run_id": run_id,
            "task": "segment",
            "model_key": request.model_key,
            "weights_name": weights_name,
            "epochs": int(request.epochs),
            "imgsz": int(request.imgsz),
            "batch": int(request.batch),
            "device": device,
            "device_name": status.device_name,
            "seed": DEFAULT_ULTRALYTICS_SEED,
            "environment": status.environment,
            "data_yaml": str(resolved_yaml.resolve()),
            "run_dir": str(run_dir.resolve()),
            "dataset_root": str(request.dataset.root.resolve()),
            "num_classes": request.dataset.num_classes,
            "class_names": {
                str(k): v for k, v in request.dataset.class_names.items()
            },
            "dataset_name": request.dataset_name or request.dataset.root.name,
        }
        if request.dataset_id:
            request_payload["dataset_id"] = request.dataset_id
        if request.dataset_source_type:
            request_payload["dataset_source_type"] = request.dataset_source_type
        write_request(run_dir, request_payload)

        return PreparedSegmentRun(
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


def assert_segment_task(value: str | None) -> str:
    task = normalize_task(value)
    if task != "segment":
        raise TrainingError(f"Task attendue 'segment', reçu {task!r}.")
    return task
