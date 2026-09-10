"""Pipeline inference engine (Streamlit-independent)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PIL import Image

from vision_trainer.classify.predictor import run_classify_inference
from vision_trainer.inference.models import Detection
from vision_trainer.inference.predictor import (
    InferenceError,
    extract_class_names,
    load_image_rgb,
    run_inference,
)
from vision_trainer.pipeline.crop import CropError, crop_from_detection
from vision_trainer.pipeline.models import (
    ClassificationRefinement,
    ClassMapping,
    EnrichedDetection,
    PipelineConfig,
    PipelineResult,
)
from vision_trainer.pipeline.store import (
    PipelineStoreError,
    resolve_run_weights,
)
from vision_trainer.training.device import DeviceChoice

ModelFactory = Callable[[str], Any]
ModelCache = dict[str, Any]


class PipelineEngineError(RuntimeError):
    """Fatal pipeline error (detector failure, invalid image, etc.)."""


def _default_yolo_factory(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)


def get_cached_model(
    weights_path: str | Path,
    *,
    cache: ModelCache | None = None,
    model_factory: ModelFactory | None = None,
) -> Any:
    """Load a YOLO model once per weights path (shared detector / classifiers)."""
    key = str(Path(weights_path).resolve())
    if cache is not None and key in cache:
        return cache[key]
    factory = model_factory or _default_yolo_factory
    model = factory(key if Path(key).is_file() else str(weights_path))
    if cache is not None:
        cache[key] = model
    return model


def load_detector_class_names(
    detector_run_id: str,
    *,
    cache: ModelCache | None = None,
    model_factory: ModelFactory | None = None,
) -> dict[int, str]:
    """Load class names from a detector run's weights (for pipeline UI)."""
    weights = resolve_run_weights(detector_run_id)
    model = get_cached_model(weights, cache=cache, model_factory=model_factory)
    return extract_class_names(model)


def _decision_to_refinement(decision_kind: str, *, decision: Any) -> ClassificationRefinement:
    status_map = {
        "accepted": "ok",
        "unknown": "unknown",
        "uncertain": "uncertain",
    }
    status = status_map.get(decision_kind, "error")
    top1 = decision.top1
    return ClassificationRefinement(
        status=status,  # type: ignore[arg-type]
        class_name=top1.class_name if status == "ok" and top1 is not None else None,
        confidence=float(top1.confidence) if top1 is not None else None,
        top_n=tuple(decision.top_n),
        reason=decision.reason,
        warning=None,
    )


def _refine_detection(
    *,
    image: Image.Image,
    detection: Detection,
    mapping: ClassMapping,
    padding: float,
    device_choice: DeviceChoice | str,
    cache: ModelCache,
    model_factory: ModelFactory | None,
) -> EnrichedDetection:
    if not mapping.enabled or not mapping.classifier_run_id:
        return EnrichedDetection(detection=detection, refined=False)

    try:
        weights = resolve_run_weights(mapping.classifier_run_id)
    except PipelineStoreError as exc:
        return EnrichedDetection(
            detection=detection,
            refined=True,
            classification=ClassificationRefinement(
                status="error",
                warning=f"Affinement indisponible : {exc}",
            ),
        )

    try:
        crop, crop_box = crop_from_detection(
            image,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            padding=padding,
        )
    except CropError as exc:
        return EnrichedDetection(
            detection=detection,
            refined=True,
            classification=ClassificationRefinement(
                status="error",
                warning=f"Crop invalide : {exc}",
            ),
        )

    weights_key = str(weights.resolve())

    def _factory(_path: str) -> Any:
        return get_cached_model(
            weights_key,
            cache=cache,
            model_factory=model_factory,
        )

    try:
        result = run_classify_inference(
            weights_path=weights,
            image=crop,
            device_choice=device_choice,
            min_confidence=mapping.confidence_threshold,
            min_margin=mapping.margin_threshold,
            top_n=mapping.top_n,
            model_factory=_factory,
        )
    except InferenceError as exc:
        return EnrichedDetection(
            detection=detection,
            refined=True,
            crop_box=crop_box,
            classification=ClassificationRefinement(
                status="error",
                warning=f"Affinement indisponible : {exc}",
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return EnrichedDetection(
            detection=detection,
            refined=True,
            crop_box=crop_box,
            classification=ClassificationRefinement(
                status="error",
                warning=f"Affinement indisponible : {exc}",
            ),
        )

    refinement = _decision_to_refinement(result.decision.kind, decision=result.decision)
    return EnrichedDetection(
        detection=detection,
        refined=True,
        crop_box=crop_box,
        classification=refinement,
    )


def run_pipeline(
    image: Image.Image | Path | str,
    config: PipelineConfig,
    *,
    device_choice: DeviceChoice | str = "auto",
    model_cache: ModelCache | None = None,
    model_factory: ModelFactory | None = None,
    conf: float | None = None,
    iou: float | None = None,
) -> PipelineResult:
    """
    Run detection then optional per-class classification on crops.

    Independent of Streamlit: suitable for image, and later video frames / API.
    Classifier models are loaded once per weights path via ``model_cache``.
    """
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = load_image_rgb(image)

    cache: ModelCache = model_cache if model_cache is not None else {}
    warnings: list[str] = []

    try:
        detector_weights = resolve_run_weights(config.detector_run_id)
    except PipelineStoreError as exc:
        raise PipelineEngineError(str(exc)) from exc

    detector_key = str(detector_weights.resolve())

    def _detector_factory(_path: str) -> Any:
        return get_cached_model(
            detector_key,
            cache=cache,
            model_factory=model_factory,
        )

    detect_conf = float(conf if conf is not None else config.detect_conf)
    detect_iou = float(iou if iou is not None else config.detect_iou)

    try:
        detection_result = run_inference(
            weights_path=detector_weights,
            image=pil,
            conf=detect_conf,
            iou=detect_iou,
            device_choice=device_choice,
            model_factory=_detector_factory,
        )
    except InferenceError as exc:
        raise PipelineEngineError(str(exc)) from exc

    detector_names = dict(detection_result.class_names)
    known_class_names = set(detector_names.values())

    for class_name, mapping in config.mappings.items():
        if mapping.enabled and class_name not in known_class_names:
            warnings.append(
                f"La classe « {class_name} » du pipeline n'existe plus dans le détecteur "
                f"(classes actuelles : {', '.join(sorted(known_class_names)) or 'aucune'})."
            )

    # Preload classifiers used by this pipeline (once each).
    for mapping in config.mappings.values():
        if not mapping.enabled or not mapping.classifier_run_id:
            continue
        try:
            weights = resolve_run_weights(mapping.classifier_run_id)
            get_cached_model(weights, cache=cache, model_factory=model_factory)
        except PipelineStoreError as exc:
            warnings.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Impossible de précharger un classificateur : {exc}")

    items: list[EnrichedDetection] = []
    for detection in detection_result.detections:
        mapping = config.mappings.get(detection.class_name)
        if mapping is None or not mapping.enabled:
            items.append(EnrichedDetection(detection=detection, refined=False))
            continue

        enriched = _refine_detection(
            image=pil,
            detection=detection,
            mapping=mapping,
            padding=config.crop_padding,
            device_choice=device_choice,
            cache=cache,
            model_factory=model_factory,
        )
        if (
            enriched.classification is not None
            and enriched.classification.status == "error"
            and enriched.classification.warning
        ):
            warnings.append(
                f"{detection.class_name} : {enriched.classification.warning}"
            )
        items.append(enriched)

    return PipelineResult(
        items=items,
        detector_class_names=detector_names,
        image_width=detection_result.image_width or pil.width,
        image_height=detection_result.image_height or pil.height,
        warnings=warnings,
        pipeline_id=config.pipeline_id,
        pipeline_name=config.name,
    )


def status_label_fr(status: str | None, *, refined: bool) -> str:
    if not refined:
        return "Non affiné"
    if status == "ok":
        return "OK"
    if status == "unknown":
        return "INCONNU"
    if status == "uncertain":
        return "INCERTAIN"
    if status == "error":
        return "Erreur"
    return status or "—"


def result_table_rows(result: PipelineResult) -> list[dict[str, Any]]:
    """Rows for Streamlit dataframe / display."""
    rows: list[dict[str, Any]] = []
    for item in result.items:
        det = item.detection
        refinement = item.classification
        final_class = det.class_name
        cls_conf: float | None = None
        status = status_label_fr(
            refinement.status if refinement else None,
            refined=item.refined,
        )
        if item.refined and refinement is not None:
            if refinement.status == "ok" and refinement.class_name:
                final_class = refinement.class_name
                cls_conf = refinement.confidence
            elif refinement.status in {"unknown", "uncertain"}:
                final_class = "—"
                cls_conf = refinement.confidence
            elif refinement.status == "error":
                final_class = "—"
        rows.append(
            {
                "Objet": det.class_name,
                "Conf. détection": round(det.confidence, 4),
                "Affinement": "Oui" if item.refined else "Non",
                "Classe finale": final_class,
                "Conf. classification": (
                    round(cls_conf, 4) if cls_conf is not None else None
                ),
                "Statut": status,
            }
        )
    return rows
