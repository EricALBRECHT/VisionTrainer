"""Prepare / execute Ultralytics classification training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_trainer.analysis.environment import (
    DEFAULT_ULTRALYTICS_SEED,
    collect_training_environment,
    resolve_device_display_name,
)
from vision_trainer.classify.models import ClassifyDatasetInfo
from vision_trainer.tasks import normalize_task
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
    TrainingError,
)

AVAILABLE_CLS_MODELS: dict[str, str] = {
    "YOLO11n-cls": "yolo11n-cls.pt",
    "YOLO11s-cls": "yolo11s-cls.pt",
    "YOLO11m-cls": "yolo11m-cls.pt",
}

DEFAULT_CLS_MODEL_KEY = "YOLO11n-cls"
DEFAULT_CLS_IMGSZ = 224
CLS_IMGSZ_CHOICES = (224, 320, 480, 640)

SESSION_ACTIVE_CLS_RUN_KEY = "active_classify_training_run_dir"


@dataclass(frozen=True)
class ClassifyTrainingRequest:
    dataset: ClassifyDatasetInfo
    model_key: str = DEFAULT_CLS_MODEL_KEY
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_CLS_IMGSZ
    batch: int = BATCH_AUTO
    device_choice: DeviceChoice | str = "auto"
    runs_root: Path | None = None
    dataset_id: str | None = None
    dataset_source_type: str | None = None
    dataset_name: str | None = None


@dataclass(frozen=True)
class PreparedClassifyRun:
    run_id: str
    run_dir: Path
    data_dir: Path
    status: RunStatus
    device: str
    device_label: str
    weights_name: str


def build_classify_train_kwargs(
    *,
    data_dir: Path,
    run_dir: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    if epochs < 1:
        raise TrainingError("Le nombre d'epochs doit être au moins 1.")
    if imgsz not in CLS_IMGSZ_CHOICES:
        raise TrainingError(
            f"imgsz invalide : {imgsz}. Choix autorisés : {CLS_IMGSZ_CHOICES}."
        )
    run_dir_abs = run_dir.resolve()
    return {
        "data": str(Path(data_dir).resolve()),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "device": device,
        "project": str(run_dir_abs.parent),
        "name": run_dir_abs.name,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
        "seed": DEFAULT_ULTRALYTICS_SEED,
    }


def prepare_classify_training_run(request: ClassifyTrainingRequest) -> PreparedClassifyRun:
    """Create run directory + status/request for a classification training."""
    if request.dataset.train_image_count < 1:
        raise TrainingError("Le dataset de classification ne contient pas d'images train.")

    weights_name = AVAILABLE_CLS_MODELS.get(request.model_key)
    if weights_name is None:
        raise TrainingError(
            f"Modèle de classification non supporté : {request.model_key!r}. "
            f"Choix : {', '.join(AVAILABLE_CLS_MODELS)}."
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
        data_dir = request.dataset.root.resolve()

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
            task="classify",
            device_name=resolve_device_display_name(device) or (
                "CPU" if str(device).lower() == "cpu" else None
            ),
            seed=DEFAULT_ULTRALYTICS_SEED,
            environment=collect_training_environment(),
        )
        write_status(run_dir, status)

        request_payload = {
            "run_id": run_id,
            "task": "classify",
            "model_key": request.model_key,
            "weights_name": weights_name,
            "epochs": int(request.epochs),
            "imgsz": int(request.imgsz),
            "batch": int(request.batch),
            "device": device,
            "device_name": status.device_name,
            "seed": DEFAULT_ULTRALYTICS_SEED,
            "environment": status.environment,
            "data_dir": str(data_dir),
            "run_dir": str(run_dir.resolve()),
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

        return PreparedClassifyRun(
            run_id=run_id,
            run_dir=run_dir.resolve(),
            data_dir=data_dir,
            status=status,
            device=device,
            device_label=describe_device(device),
            weights_name=weights_name,
        )
    finally:
        release_launch_lock(runs_root)
