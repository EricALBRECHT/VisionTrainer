"""YOLO instance segmentation: datasets, training helpers, inference."""

from vision_trainer.segment.area import mask_area_pixels, mask_area_ratio, polygon_area_pixels
from vision_trainer.segment.export import export_segmentation_json, segmentation_result_json_bytes
from vision_trainer.segment.labels import (
    PolygonInstance,
    SegmentDatasetInfo,
    parse_segment_label_line,
)
from vision_trainer.segment.models import SegmentInstance, SegmentationResult
from vision_trainer.segment.predictor import normalize_segmentation_results, run_segmentation
from vision_trainer.segment.render import draw_segmentation_result, scale_polygon_to_display
from vision_trainer.segment.validator import validate_segment_dataset

__all__ = [
    "PolygonInstance",
    "SegmentDatasetInfo",
    "SegmentInstance",
    "SegmentationResult",
    "draw_segmentation_result",
    "export_segmentation_json",
    "mask_area_pixels",
    "mask_area_ratio",
    "normalize_segmentation_results",
    "parse_segment_label_line",
    "polygon_area_pixels",
    "run_segmentation",
    "scale_polygon_to_display",
    "segmentation_result_json_bytes",
    "validate_segment_dataset",
]
