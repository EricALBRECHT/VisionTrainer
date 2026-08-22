from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from vision_trainer.training.data_yaml import write_resolved_data_yaml
from vision_trainer.training.device import DeviceChoice, DeviceError, describe_device, resolve_device
from vision_trainer.training.runs import create_run_directory
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

    return {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "device": device,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "plots": True,
        "verbose": True,
    }


def find_best_weights(run_dir: Path) -> Path | None:
    """Return ``weights/best.pt`` under the run directory when present."""
    candidate = run_dir / "weights" / "best.pt"
    if candidate.is_file():
        return candidate
    return None


def run_training(
    request: TrainingRequest,
    *,
    yolo_factory: Callable[[str], Any] | None = None,
) -> TrainingResult:
    """
    Prepare artifacts and run Ultralytics training synchronously.

    ``yolo_factory`` may be injected in tests to avoid loading real weights.
    """
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

    run_id, run_dir = create_run_directory(runs_root=request.runs_root)
    resolved_yaml = write_resolved_data_yaml(request.dataset, run_dir)

    train_kwargs = build_train_kwargs(
        data_yaml=resolved_yaml,
        run_dir=run_dir,
        epochs=request.epochs,
        imgsz=request.imgsz,
        batch=request.batch,
        device=device,
    )

    factory = yolo_factory or _default_yolo_factory
    try:
        model = factory(weights_name)
    except TrainingError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as user-facing training error
        raise TrainingError(f"Impossible de charger le modèle {weights_name} : {exc}") from exc

    try:
        model.train(**train_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Erreur pendant l'entraînement : {exc}") from exc

    return TrainingResult(
        run_id=run_id,
        run_dir=run_dir,
        resolved_data_yaml=resolved_yaml,
        model_key=request.model_key,
        weights_name=weights_name,
        epochs=request.epochs,
        imgsz=request.imgsz,
        batch=request.batch,
        device=device,
        device_label=describe_device(device),
        best_weights=find_best_weights(run_dir),
        status="completed",
    )


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
