"""Frame processors: detect / segment / pipeline (reuse existing engines)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image

from vision_trainer.inference.predictor import InferenceError, run_inference
from vision_trainer.inference.render import AnnotationScale, draw_detections
from vision_trainer.pipeline.engine import get_cached_model, run_pipeline
from vision_trainer.pipeline.models import PipelineConfig, PipelineResult
from vision_trainer.pipeline.render import draw_pipeline_result
from vision_trainer.segment.predictor import run_segmentation
from vision_trainer.segment.render import draw_segmentation_result
from vision_trainer.training.device import DeviceChoice

ModelFactory = Callable[[str], Any]
ModelCache = dict[str, Any]


@dataclass
class FrameProcessResult:
    """Inference result for one frame — structured data separate from rendering."""

    processed: bool
    annotated: Image.Image | None = None
    structured: dict[str, Any] = field(default_factory=dict)
    detections: int = 0
    classifications: int = 0
    segmentations: int = 0
    secondary_errors: int = 0
    warnings: list[str] = field(default_factory=list)
    detection_ms: float = 0.0
    classification_ms: float = 0.0
    segmentation_ms: float = 0.0
    total_ms: float = 0.0


class FrameProcessor(Protocol):
    def process(self, frame: Image.Image, *, frame_index: int) -> FrameProcessResult:
        """Run inference on an RGB PIL frame. Must not mutate ``frame`` unexpectedly."""


@dataclass
class DetectFrameProcessor:
    weights_path: str | Path
    conf: float = 0.25
    iou: float = 0.45
    device_choice: DeviceChoice | str = "auto"
    model_cache: ModelCache = field(default_factory=dict)
    model_factory: ModelFactory | None = None
    annotation_scale: AnnotationScale | str = "auto"

    def __post_init__(self) -> None:
        weights = Path(self.weights_path)
        if not weights.is_file():
            raise InferenceError(f"Poids introuvables : {weights}")
        get_cached_model(
            weights,
            cache=self.model_cache,
            model_factory=self.model_factory,
        )

    def process(self, frame: Image.Image, *, frame_index: int) -> FrameProcessResult:
        t0 = time.perf_counter()
        weights = Path(self.weights_path)
        key = str(weights.resolve())

        def _factory(_path: str) -> Any:
            return get_cached_model(
                key, cache=self.model_cache, model_factory=self.model_factory
            )

        result = run_inference(
            weights_path=weights,
            image=frame,
            conf=self.conf,
            iou=self.iou,
            device_choice=self.device_choice,
            model_factory=_factory,
        )
        annotated = draw_detections(
            frame,
            result.detections,
            scale=self.annotation_scale,
            fit_to_display=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        return FrameProcessResult(
            processed=True,
            annotated=annotated,
            structured={
                "detections": [
                    {
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox": [d.x1, d.y1, d.x2, d.y2],
                    }
                    for d in result.detections
                ]
            },
            detections=len(result.detections),
            detection_ms=elapsed,
            total_ms=elapsed,
        )


@dataclass
class SegmentFrameProcessor:
    weights_path: str | Path
    conf: float = 0.25
    iou: float = 0.45
    device_choice: DeviceChoice | str = "auto"
    model_cache: ModelCache = field(default_factory=dict)
    model_factory: ModelFactory | None = None
    annotation_scale: AnnotationScale | str = "auto"
    mask_opacity: float = 0.40
    show_masks: bool = True
    show_contours: bool = True
    show_labels: bool = True
    show_boxes: bool = False

    def __post_init__(self) -> None:
        weights = Path(self.weights_path)
        if not weights.is_file():
            raise InferenceError(f"Poids introuvables : {weights}")
        get_cached_model(
            weights,
            cache=self.model_cache,
            model_factory=self.model_factory,
        )

    def process(self, frame: Image.Image, *, frame_index: int) -> FrameProcessResult:
        t0 = time.perf_counter()
        weights = Path(self.weights_path)
        key = str(weights.resolve())

        def _factory(_path: str) -> Any:
            return get_cached_model(
                key, cache=self.model_cache, model_factory=self.model_factory
            )

        result = run_segmentation(
            weights_path=weights,
            image=frame,
            conf=self.conf,
            iou=self.iou,
            device_choice=self.device_choice,
            model_factory=_factory,
        )
        annotated = draw_segmentation_result(
            frame,
            result,
            show_masks=self.show_masks,
            show_contours=self.show_contours,
            show_labels=self.show_labels,
            show_boxes=self.show_boxes,
            mask_opacity=self.mask_opacity,
            scale=self.annotation_scale,
            fit_to_display=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        return FrameProcessResult(
            processed=True,
            annotated=annotated,
            structured={"instances": [item.to_dict() for item in result.instances]},
            detections=len(result.instances),
            segmentations=len(result.instances),
            segmentation_ms=elapsed,
            total_ms=elapsed,
        )


@dataclass
class PipelineFrameProcessor:
    config: PipelineConfig
    device_choice: DeviceChoice | str = "auto"
    model_cache: ModelCache = field(default_factory=dict)
    model_factory: ModelFactory | None = None
    annotation_scale: AnnotationScale | str = "auto"
    show_boxes: bool = True
    show_classification: bool = True
    show_masks: bool = True
    show_contours: bool = True
    mask_opacity: float = 0.40

    def __post_init__(self) -> None:
        # Warm detector (+ secondary models) once via a dry preload path in run_pipeline.
        # Preload by resolving weights through first pipeline call is expensive;
        # instead warm detector + mapped models explicitly.
        from vision_trainer.pipeline.store import resolve_run_weights

        det = resolve_run_weights(self.config.detector_run_id)
        get_cached_model(det, cache=self.model_cache, model_factory=self.model_factory)
        for mapping in self.config.mappings.values():
            for stage in (mapping.classification, mapping.segmentation):
                if stage.enabled and stage.run_id:
                    try:
                        weights = resolve_run_weights(stage.run_id)
                        get_cached_model(
                            weights,
                            cache=self.model_cache,
                            model_factory=self.model_factory,
                        )
                    except Exception:  # noqa: BLE001
                        continue

    def process(self, frame: Image.Image, *, frame_index: int) -> FrameProcessResult:
        t0 = time.perf_counter()
        result: PipelineResult = run_pipeline(
            frame,
            self.config,
            device_choice=self.device_choice,
            model_cache=self.model_cache,
            model_factory=self.model_factory,
        )
        annotated = draw_pipeline_result(
            frame,
            result,
            show_boxes=self.show_boxes,
            show_classification=self.show_classification,
            show_masks=self.show_masks,
            show_contours=self.show_contours,
            mask_opacity=self.mask_opacity,
            scale=self.annotation_scale,
            fit_to_display=False,
        )
        classifications = sum(
            1
            for item in result.items
            if item.classification is not None and item.classification.status != "error"
        )
        segmentations = sum(len(item.segmentations) for item in result.items)
        secondary_errors = 0
        for item in result.items:
            if item.classification is not None and item.classification.status == "error":
                secondary_errors += 1
            if item.segmentation is not None and item.segmentation.status == "error":
                secondary_errors += 1
        elapsed = (time.perf_counter() - t0) * 1000.0
        return FrameProcessResult(
            processed=True,
            annotated=annotated,
            structured={"items": [item.to_dict() for item in result.items]},
            detections=len(result.items),
            classifications=classifications,
            segmentations=segmentations,
            secondary_errors=secondary_errors,
            warnings=list(result.warnings),
            detection_ms=result.timings.detection_ms,
            classification_ms=result.timings.classification_ms,
            segmentation_ms=result.timings.segmentation_ms,
            total_ms=result.timings.total_ms or elapsed,
        )
