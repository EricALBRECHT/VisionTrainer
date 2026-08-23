from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from vision_trainer.inference.models import Detection, InferenceResult
from vision_trainer.training.device import DeviceChoice, DeviceError, resolve_device

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


class InferenceError(Exception):
    """Raised when inference cannot be prepared or executed."""


def load_image_rgb(source: Path | str) -> Image.Image:
    """Load an image as RGB, validating format and readability."""
    path = Path(source)
    if not path.is_file():
        raise InferenceError(f"Image introuvable : {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise InferenceError(
            f"Format d'image non supporté : {suffix or '(sans extension)'}. "
            f"Formats acceptés : {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}."
        )

    try:
        with Image.open(path) as image:
            image.verify()
        image = Image.open(path).convert("RGB")
        image.load()
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        raise InferenceError(f"Image illisible ou corrompue : {path.name}") from exc

    return image


def extract_class_names(model: Any) -> dict[int, str]:
    """Extract class id → name mapping from an Ultralytics model when available."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        parsed: dict[int, str] = {}
        for key, value in names.items():
            try:
                parsed[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return parsed
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def normalize_ultralytics_results(
    results: Any,
    *,
    class_names: dict[int, str] | None = None,
) -> InferenceResult:
    """
    Normalize Ultralytics predict() output into pixel-space Detection objects.

    Accepts a list/tuple of Results or a single Results-like object.
    """
    names = dict(class_names or {})
    items = results
    if not isinstance(items, (list, tuple)):
        items = [items]

    detections: list[Detection] = []
    width = 0
    height = 0

    for result in items:
        result_names = getattr(result, "names", None)
        if not names and isinstance(result_names, dict):
            names = extract_class_names(type("M", (), {"names": result_names})())
        elif not names and isinstance(result_names, (list, tuple)):
            names = {index: str(name) for index, name in enumerate(result_names)}

        orig_shape = getattr(result, "orig_shape", None)
        if orig_shape is not None and len(orig_shape) >= 2:
            height = int(orig_shape[0])
            width = int(orig_shape[1])

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        xyxy = _to_list_of_lists(getattr(boxes, "xyxy", None))
        confs = _to_flat_list(getattr(boxes, "conf", None))
        clss = _to_flat_list(getattr(boxes, "cls", None))

        count = max(len(xyxy), len(confs), len(clss))
        for index in range(count):
            if index >= len(xyxy) or len(xyxy[index]) < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[index][:4])
            confidence = float(confs[index]) if index < len(confs) else 0.0
            class_id = int(clss[index]) if index < len(clss) else -1
            class_name = names.get(class_id, f"class_{class_id}")
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return InferenceResult(
        detections=detections,
        class_names=names,
        image_width=width,
        image_height=height,
    )


def build_predict_kwargs(
    *,
    conf: float,
    iou: float,
    device: str,
) -> dict[str, Any]:
    if not 0.05 <= conf <= 0.95:
        raise InferenceError("Le seuil de confiance doit être compris entre 0.05 et 0.95.")
    if not 0.05 <= iou <= 0.95:
        raise InferenceError("Le seuil IoU doit être compris entre 0.05 et 0.95.")
    return {
        "conf": float(conf),
        "iou": float(iou),
        "device": device,
        "verbose": False,
    }


def run_inference(
    *,
    weights_path: Path | str,
    image: Path | str | Image.Image,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    device_choice: DeviceChoice | str = "auto",
    model_factory: Callable[[str], Any] | None = None,
) -> InferenceResult:
    """
    Run object detection and return normalized detections.

    ``model_factory`` may be injected in tests to avoid loading real weights.
    """
    weights = Path(weights_path)
    if not weights.is_file():
        raise InferenceError(f"Poids introuvables : {weights}")

    try:
        device = resolve_device(device_choice)
    except DeviceError as exc:
        raise InferenceError(str(exc)) from exc

    if isinstance(image, Image.Image):
        source: Path | Image.Image = image.convert("RGB")
        image_width, image_height = source.size
    else:
        source_path = Path(image)
        loaded = load_image_rgb(source_path)
        image_width, image_height = loaded.size
        source = source_path

    predict_kwargs = build_predict_kwargs(conf=conf, iou=iou, device=device)
    factory = model_factory or _default_yolo_factory

    try:
        model = factory(str(weights))
    except InferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Impossible de charger le modèle : {exc}") from exc

    class_names = extract_class_names(model)

    try:
        results = model.predict(source=source, **predict_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Erreur pendant l'inférence : {exc}") from exc

    normalized = normalize_ultralytics_results(results, class_names=class_names)
    if normalized.image_width == 0 or normalized.image_height == 0:
        return InferenceResult(
            detections=normalized.detections,
            class_names=normalized.class_names or class_names,
            image_width=image_width,
            image_height=image_height,
        )
    if not normalized.class_names and class_names:
        return InferenceResult(
            detections=normalized.detections,
            class_names=class_names,
            image_width=normalized.image_width,
            image_height=normalized.image_height,
        )
    return normalized


def _default_yolo_factory(weights_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise InferenceError(
            "Ultralytics n'est pas disponible. Installez le projet avec ses dépendances."
        ) from exc
    try:
        return YOLO(weights_path)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Modèle illisible ou invalide : {exc}") from exc


def _to_list_of_lists(value: Any) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return []
    # single box [x1,y1,x2,y2]
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
        return [list(map(float, value))]
    return [list(map(float, row)) for row in value]


def _to_flat_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    flat: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flat.extend(float(v) for v in item)
        else:
            flat.append(float(item))
    return flat
