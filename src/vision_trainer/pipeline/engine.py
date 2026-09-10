"""Pipeline inference engine (Streamlit-independent): detect → classify? → segment?."""

from __future__ import annotations

import time
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
from vision_trainer.pipeline.crop import CropError, CropRegion, crop_region_from_detection
from vision_trainer.pipeline.models import (
    ClassificationRefinement,
    ClassMapping,
    EnrichedDetection,
    PipelineConfig,
    PipelineResult,
    PipelineTimings,
    SegmentationInstanceResult,
    SegmentationRefinement,
)
from vision_trainer.pipeline.store import (
    PipelineStoreError,
    resolve_run_weights,
)
from vision_trainer.segment.area import mask_area_ratio
from vision_trainer.segment.predictor import run_segmentation
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
    """Load a YOLO model once per weights path (detector / classifiers / segmenters)."""
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


def _resolve_crop_padding(mapping: ClassMapping, pipeline_padding: float) -> float:
    if mapping.segmentation.enabled and mapping.segmentation.crop_padding is not None:
        return float(mapping.segmentation.crop_padding)
    return float(pipeline_padding)


def _run_classification(
    *,
    region: CropRegion,
    mapping: ClassMapping,
    device_choice: DeviceChoice | str,
    cache: ModelCache,
    model_factory: ModelFactory | None,
) -> ClassificationRefinement:
    stage = mapping.classification
    try:
        weights = resolve_run_weights(stage.run_id)  # type: ignore[arg-type]
    except PipelineStoreError as exc:
        return ClassificationRefinement(
            status="error",
            warning=f"Classification indisponible : {exc}",
        )

    weights_key = str(weights.resolve())

    def _factory(_path: str) -> Any:
        return get_cached_model(weights_key, cache=cache, model_factory=model_factory)

    try:
        result = run_classify_inference(
            weights_path=weights,
            image=region.image,
            device_choice=device_choice,
            min_confidence=stage.confidence_threshold,
            min_margin=stage.margin_threshold,
            top_n=stage.top_n,
            model_factory=_factory,
        )
    except Exception as exc:  # noqa: BLE001
        return ClassificationRefinement(
            status="error",
            warning=f"Classification indisponible : {exc}",
        )
    return _decision_to_refinement(result.decision.kind, decision=result.decision)


def _run_segmentation(
    *,
    region: CropRegion,
    mapping: ClassMapping,
    image_width: int,
    image_height: int,
    device_choice: DeviceChoice | str,
    cache: ModelCache,
    model_factory: ModelFactory | None,
) -> SegmentationRefinement:
    stage = mapping.segmentation
    try:
        weights = resolve_run_weights(stage.run_id)  # type: ignore[arg-type]
    except PipelineStoreError as exc:
        return SegmentationRefinement(
            status="error",
            warning=f"Segmentation indisponible : {exc}",
        )

    weights_key = str(weights.resolve())

    def _factory(_path: str) -> Any:
        return get_cached_model(weights_key, cache=cache, model_factory=model_factory)

    try:
        result = run_segmentation(
            weights_path=weights,
            image=region.image,
            conf=float(stage.confidence_threshold),
            device_choice=device_choice,
            model_factory=_factory,
        )
    except Exception as exc:  # noqa: BLE001
        return SegmentationRefinement(
            status="error",
            warning=f"Segmentation indisponible : {exc}",
        )

    crop_area = max(1, region.width * region.height)
    instances: list[SegmentationInstanceResult] = []
    for item in result.instances:
        polygon_global = region.local_to_global_polygon(item.polygon) if item.polygon else ()
        bbox_global = region.local_to_global_bbox(item.x1, item.y1, item.x2, item.y2)
        ratio_crop = (
            float(item.mask_area_pixels) / float(crop_area)
            if item.mask_area_pixels
            else 0.0
        )
        ratio_image = mask_area_ratio(
            item.mask_area_pixels,
            image_width=image_width,
            image_height=image_height,
        )
        instances.append(
            SegmentationInstanceResult(
                class_id=item.class_id,
                class_name=item.class_name,
                confidence=item.confidence,
                polygon_global=polygon_global,
                bbox_global=bbox_global,
                mask_area_pixels=item.mask_area_pixels,
                mask_area_ratio_crop=ratio_crop,
                mask_area_ratio_image=ratio_image,
            )
        )

    return SegmentationRefinement(status="ok", instances=tuple(instances))


def _refine_detection(
    *,
    image: Image.Image,
    detection: Detection,
    mapping: ClassMapping,
    pipeline_padding: float,
    device_choice: DeviceChoice | str,
    cache: ModelCache,
    model_factory: ModelFactory | None,
    timings: PipelineTimings,
) -> EnrichedDetection:
    need_cls = mapping.classification.enabled and bool(mapping.classification.run_id)
    need_seg = mapping.segmentation.enabled and bool(mapping.segmentation.run_id)
    if not need_cls and not need_seg:
        return EnrichedDetection(detection=detection, refined=False)

    padding = _resolve_crop_padding(mapping, pipeline_padding)
    try:
        region = crop_region_from_detection(
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
            classification=(
                ClassificationRefinement(status="error", warning=f"Crop invalide : {exc}")
                if need_cls
                else None
            ),
            segmentation=(
                SegmentationRefinement(status="error", warning=f"Crop invalide : {exc}")
                if need_seg
                else None
            ),
        )

    classification: ClassificationRefinement | None = None
    segmentation: SegmentationRefinement | None = None

    if need_cls:
        t0 = time.perf_counter()
        classification = _run_classification(
            region=region,
            mapping=mapping,
            device_choice=device_choice,
            cache=cache,
            model_factory=model_factory,
        )
        timings.classification_ms += (time.perf_counter() - t0) * 1000.0

    if need_seg:
        t0 = time.perf_counter()
        segmentation = _run_segmentation(
            region=region,
            mapping=mapping,
            image_width=image.width,
            image_height=image.height,
            device_choice=device_choice,
            cache=cache,
            model_factory=model_factory,
        )
        timings.segmentation_ms += (time.perf_counter() - t0) * 1000.0

    return EnrichedDetection(
        detection=detection,
        refined=True,
        crop_box=region.box,
        classification=classification,
        segmentation=segmentation,
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
    Run detection then optional per-class classification and/or segmentation.

    Independent of Streamlit. Secondary models are loaded once per weights path.
    """
    t_total = time.perf_counter()
    if isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = load_image_rgb(image)

    cache: ModelCache = model_cache if model_cache is not None else {}
    warnings: list[str] = []
    timings = PipelineTimings()

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

    t0 = time.perf_counter()
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
    timings.detection_ms = (time.perf_counter() - t0) * 1000.0

    detector_names = dict(detection_result.class_names)
    known_class_names = set(detector_names.values())

    for class_name, mapping in config.mappings.items():
        if mapping.enabled and class_name not in known_class_names:
            warnings.append(
                f"La classe « {class_name} » du pipeline n'existe plus dans le détecteur "
                f"(classes actuelles : {', '.join(sorted(known_class_names)) or 'aucune'})."
            )

    # Preload secondary models once each.
    for mapping in config.mappings.values():
        for stage, label in (
            (mapping.classification, "classificateur"),
            (mapping.segmentation, "segmenter"),
        ):
            if not stage.enabled or not stage.run_id:
                continue
            try:
                weights = resolve_run_weights(stage.run_id)
                get_cached_model(weights, cache=cache, model_factory=model_factory)
            except PipelineStoreError as exc:
                warnings.append(str(exc))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Impossible de précharger un {label} : {exc}")

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
            pipeline_padding=config.crop_padding,
            device_choice=device_choice,
            cache=cache,
            model_factory=model_factory,
            timings=timings,
        )
        if (
            enriched.classification is not None
            and enriched.classification.status == "error"
            and enriched.classification.warning
        ):
            warnings.append(f"{detection.class_name} : {enriched.classification.warning}")
        if (
            enriched.segmentation is not None
            and enriched.segmentation.status == "error"
            and enriched.segmentation.warning
        ):
            warnings.append(f"{detection.class_name} : {enriched.segmentation.warning}")
        items.append(enriched)

    timings.total_ms = (time.perf_counter() - t_total) * 1000.0
    return PipelineResult(
        items=items,
        detector_class_names=detector_names,
        image_width=detection_result.image_width or pil.width,
        image_height=detection_result.image_height or pil.height,
        warnings=warnings,
        pipeline_id=config.pipeline_id,
        pipeline_name=config.name,
        timings=timings,
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
        cls_label = "—"
        cls_conf: float | None = None
        if item.refined and refinement is not None:
            if refinement.status == "ok" and refinement.class_name:
                cls_label = refinement.class_name
                cls_conf = refinement.confidence
            elif refinement.status == "unknown":
                cls_label = "INCONNU"
                cls_conf = refinement.confidence
            elif refinement.status == "uncertain":
                cls_label = "INCERTAIN"
                cls_conf = refinement.confidence
            elif refinement.status == "error":
                cls_label = "Erreur"

        seg_summary = "—"
        surface_summary = "—"
        if item.segmentation is not None:
            if item.segmentation.status == "error":
                seg_summary = "Erreur"
            elif item.segmentation.instances:
                bits = [
                    f"{m.class_name} {m.confidence * 100:.0f}%"
                    for m in item.segmentation.instances[:3]
                ]
                if len(item.segmentation.instances) > 3:
                    bits.append("…")
                seg_summary = ", ".join(bits)
                surfaces = [
                    f"{m.class_name} ({m.mask_area_ratio_crop * 100:.1f} % crop)"
                    for m in item.segmentation.instances[:3]
                ]
                surface_summary = ", ".join(surfaces)

        rows.append(
            {
                "Objet": det.class_name,
                "Detect": round(det.confidence, 4),
                "Classification": cls_label,
                "Conf. classif": (
                    round(cls_conf, 4) if cls_conf is not None else None
                ),
                "Segmentation": seg_summary,
                "Surface crop": surface_summary,
            }
        )
    return rows
