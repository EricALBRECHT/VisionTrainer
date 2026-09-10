"""Classification inference helpers (Ultralytics YOLO-cls)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from vision_trainer.classify.reject import ClassScore, ClassifyDecision, decide_classification
from vision_trainer.inference.predictor import InferenceError, load_image_rgb
from vision_trainer.training.device import DeviceChoice, DeviceError, resolve_device


@dataclass(frozen=True)
class ClassifyInferenceResult:
    decision: ClassifyDecision
    scores: tuple[ClassScore, ...]
    image_width: int
    image_height: int
    device: str


def run_classify_inference(
    *,
    weights_path: Path | str,
    image: Image.Image | Path | str,
    device_choice: DeviceChoice | str = "auto",
    min_confidence: float = 0.80,
    min_margin: float = 0.10,
    top_n: int = 5,
    model_factory: Callable[[str], Any] | None = None,
) -> ClassifyInferenceResult:
    """Run a classification model and apply rejection rules."""
    weights = Path(weights_path)
    if not weights.is_file():
        raise InferenceError(f"Poids introuvables : {weights}")

    if isinstance(image, (str, Path)):
        pil = load_image_rgb(image)
    else:
        pil = image.convert("RGB")

    try:
        device = resolve_device(device_choice)
    except DeviceError as exc:
        raise InferenceError(str(exc)) from exc

    factory = model_factory or _default_factory
    try:
        model = factory(str(weights))
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Impossible de charger le modèle : {exc}") from exc

    try:
        results = model.predict(source=pil, device=device, verbose=False)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"Erreur pendant l'inférence classification : {exc}") from exc

    scores = _scores_from_ultralytics(results, model)
    decision = decide_classification(
        list(scores),
        min_confidence=min_confidence,
        min_margin=min_margin,
        top_n=top_n,
    )
    return ClassifyInferenceResult(
        decision=decision,
        scores=scores,
        image_width=pil.width,
        image_height=pil.height,
        device=device,
    )


def _scores_from_ultralytics(results: Any, model: Any) -> tuple[ClassScore, ...]:
    if not results:
        return ()
    first = results[0]
    probs = getattr(first, "probs", None)
    if probs is None:
        return ()

    names = getattr(first, "names", None) or getattr(model, "names", None) or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    data = getattr(probs, "data", None)
    if data is None:
        return ()
    try:
        values = data.tolist() if hasattr(data, "tolist") else list(data)
    except Exception:  # noqa: BLE001
        return ()

    scores: list[ClassScore] = []
    for index, confidence in enumerate(values):
        name = str(names.get(index, index)) if isinstance(names, dict) else str(index)
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            continue
        scores.append(ClassScore(class_id=index, class_name=name, confidence=conf))
    return tuple(scores)


def _default_factory(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)
