"""Video frame processors with optional TrackingSession (detect / pipeline)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from vision_trainer.inference.models import Detection
from vision_trainer.inference.predictor import InferenceError
from vision_trainer.inference.render import AnnotationScale
from vision_trainer.pipeline.crop import CropError, crop_region_from_detection
from vision_trainer.pipeline.engine import (
    _resolve_crop_padding,
    _run_classification,
    _run_segmentation,
    get_cached_model,
)
from vision_trainer.pipeline.models import (
    ClassificationRefinement,
    EnrichedDetection,
    PipelineConfig,
    PipelineResult,
    PipelineTimings,
)
from vision_trainer.pipeline.render import draw_pipeline_result
from vision_trainer.pipeline.store import resolve_run_weights
from vision_trainer.tracking.cache import ClassificationTrackCache
from vision_trainer.tracking.models import TrackedObject, TrackingConfig
from vision_trainer.tracking.render import draw_track_id_overlays, draw_tracked_objects
from vision_trainer.tracking.session import TrackingSession
from vision_trainer.training.device import DeviceChoice, resolve_device
from vision_trainer.video.processors import FrameProcessResult, ModelCache, ModelFactory


def _tracked_to_detection(obj: TrackedObject) -> Detection:
    x1, y1, x2, y2 = obj.bbox
    return Detection(
        class_id=obj.class_id,
        class_name=obj.class_name,
        confidence=obj.confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


@dataclass
class TrackedDetectFrameProcessor:
    """Detection + ByteTrack (or other Ultralytics tracker) with persistent IDs."""

    weights_path: str | Path
    tracking_session: TrackingSession
    conf: float = 0.25
    iou: float = 0.45
    device_choice: DeviceChoice | str = "auto"
    model_cache: ModelCache = field(default_factory=dict)
    model_factory: ModelFactory | None = None
    annotation_scale: AnnotationScale | str = "auto"
    show_trajectories: bool = False
    tracking_config: TrackingConfig | None = None

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
        try:
            device = resolve_device(self.device_choice)
        except Exception:  # noqa: BLE001
            device = None

        tracked = self.tracking_session.process(
            frame,
            frame_index=frame_index,
            conf=self.conf,
            iou=self.iou,
            device=device,
        )
        annotated = draw_tracked_objects(
            frame,
            tracked.objects,
            scale=self.annotation_scale,
            fit_to_display=False,
            show_trajectories=self.show_trajectories,
        )
        return FrameProcessResult(
            processed=True,
            annotated=annotated,
            structured={
                "tracking": True,
                "objects": [obj.to_dict() for obj in tracked.objects],
                "stats": self.tracking_session.stats().to_dict(),
            },
            detections=len(tracked.objects),
            detection_ms=tracked.detection_tracking_ms,
            total_ms=tracked.detection_tracking_ms,
        )


@dataclass
class TrackedPipelineFrameProcessor:
    """
    Video pipeline with tracking on the detector stage.

    Does **not** call ``run_pipeline`` (stateless). Classification may be cached
    per ``track_id``; segmentation is recomputed every processed frame.
    """

    config: PipelineConfig
    tracking_session: TrackingSession
    classify_cache: ClassificationTrackCache
    device_choice: DeviceChoice | str = "auto"
    model_cache: ModelCache = field(default_factory=dict)
    model_factory: ModelFactory | None = None
    annotation_scale: AnnotationScale | str = "auto"
    show_boxes: bool = True
    show_classification: bool = True
    show_masks: bool = True
    show_contours: bool = True
    mask_opacity: float = 0.40
    show_trajectories: bool = False
    reuse_classification: bool = True
    tracking_config: TrackingConfig | None = None

    def __post_init__(self) -> None:
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

    def reset_tracking(self) -> None:
        self.tracking_session.reset()
        self.classify_cache.reset()

    def process(self, frame: Image.Image, *, frame_index: int) -> FrameProcessResult:
        t0 = time.perf_counter()
        try:
            device = resolve_device(self.device_choice)
        except Exception:  # noqa: BLE001
            device = None

        conf = float(self.config.detect_conf)
        iou = float(self.config.detect_iou)

        tracked = self.tracking_session.process(
            frame,
            frame_index=frame_index,
            conf=conf,
            iou=iou,
            device=device,
        )
        detection_ms = tracked.detection_tracking_ms

        items: list[EnrichedDetection] = []
        classification_ms = 0.0
        segmentation_ms = 0.0
        warnings: list[str] = []
        classifications = 0
        segmentations = 0
        secondary_errors = 0

        for obj in tracked.objects:
            detection = _tracked_to_detection(obj)
            mapping = self.config.mappings.get(detection.class_name)
            classification: ClassificationRefinement | None = None
            segmentation_ref = None
            crop_box = None
            region = None
            refined = False

            needs_cls = bool(
                mapping is not None
                and mapping.classification.enabled
                and mapping.classification.run_id
            )
            needs_seg = bool(
                mapping is not None
                and mapping.segmentation.enabled
                and mapping.segmentation.run_id
            )

            if needs_cls or needs_seg:
                refined = True
                padding = (
                    _resolve_crop_padding(mapping, self.config.crop_padding)
                    if mapping is not None
                    else float(self.config.crop_padding)
                )
                try:
                    region = crop_region_from_detection(
                        frame,
                        detection.x1,
                        detection.y1,
                        detection.x2,
                        detection.y2,
                        padding=padding,
                    )
                    crop_box = (region.x1, region.y1, region.x2, region.y2)
                except CropError as exc:
                    warnings.append(f"Crop track {obj.track_id} : {exc}")
                    region = None

            if needs_cls and mapping is not None and region is not None:
                use_cache = (
                    self.reuse_classification
                    and not self.classify_cache.should_classify(obj.track_id, frame_index)
                )
                if use_cache:
                    cached = self.classify_cache.get(obj.track_id)
                    if cached is not None:
                        classification = cached
                    else:
                        use_cache = False
                if not use_cache:
                    t_cls = time.perf_counter()
                    classification = _run_classification(
                        region=region,
                        mapping=mapping,
                        device_choice=self.device_choice,
                        cache=self.model_cache,
                        model_factory=self.model_factory,
                    )
                    classification_ms += (time.perf_counter() - t_cls) * 1000.0
                    if self.reuse_classification:
                        self.classify_cache.put(
                            obj.track_id,
                            frame_index,
                            classification,
                            status=classification.status,
                        )
                if classification is not None and classification.status != "error":
                    classifications += 1
                if classification is not None and classification.status == "error":
                    secondary_errors += 1

            if needs_seg and mapping is not None and region is not None:
                # Segmentation is never cached across frames (shape/pose change).
                t_seg = time.perf_counter()
                segmentation_ref = _run_segmentation(
                    region=region,
                    mapping=mapping,
                    image_width=frame.width,
                    image_height=frame.height,
                    device_choice=self.device_choice,
                    cache=self.model_cache,
                    model_factory=self.model_factory,
                )
                segmentation_ms += (time.perf_counter() - t_seg) * 1000.0
                if segmentation_ref is not None:
                    if segmentation_ref.status == "error":
                        secondary_errors += 1
                    else:
                        segmentations += len(segmentation_ref.instances)

            items.append(
                EnrichedDetection(
                    detection=detection,
                    refined=refined,
                    classification=classification,
                    segmentation=segmentation_ref,
                    crop_box=crop_box,
                )
            )

        pipeline_result = PipelineResult(
            items=items,
            image_width=frame.width,
            image_height=frame.height,
            warnings=warnings,
            pipeline_id=self.config.pipeline_id,
            pipeline_name=self.config.name,
            timings=PipelineTimings(
                detection_ms=detection_ms,
                classification_ms=classification_ms,
                segmentation_ms=segmentation_ms,
                total_ms=(time.perf_counter() - t0) * 1000.0,
            ),
        )

        annotated = draw_pipeline_result(
            frame,
            pipeline_result,
            show_boxes=self.show_boxes,
            show_classification=self.show_classification,
            show_masks=self.show_masks,
            show_contours=self.show_contours,
            mask_opacity=self.mask_opacity,
            scale=self.annotation_scale,
            fit_to_display=False,
        )
        annotated = draw_track_id_overlays(
            annotated,
            tracked.objects,
            scale=self.annotation_scale,
            fit_to_display=False,
            show_trajectories=self.show_trajectories,
        )

        structured_items = []
        for obj, item in zip(tracked.objects, items):
            payload = item.to_dict()
            payload["track_id"] = obj.track_id
            payload["center"] = [obj.center[0], obj.center[1]]
            structured_items.append(payload)

        total_ms = (time.perf_counter() - t0) * 1000.0
        return FrameProcessResult(
            processed=True,
            annotated=annotated,
            structured={
                "tracking": True,
                "items": structured_items,
                "stats": self.tracking_session.stats().to_dict(),
            },
            detections=len(items),
            classifications=classifications,
            segmentations=segmentations,
            secondary_errors=secondary_errors,
            warnings=warnings,
            detection_ms=detection_ms,
            classification_ms=classification_ms,
            segmentation_ms=segmentation_ms,
            total_ms=total_ms,
        )
