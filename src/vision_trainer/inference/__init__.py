"""Inference helpers for trained YOLO models."""

from vision_trainer.inference.discovery import discover_trained_models
from vision_trainer.inference.models import (
    AvailableModel,
    Detection,
    InferenceResult,
    VideoInferenceResult,
    VideoMetadata,
    VideoProgress,
)
from vision_trainer.inference.predictor import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    InferenceError,
    build_predict_kwargs,
    load_image_rgb,
    normalize_ultralytics_results,
    run_inference,
)
from vision_trainer.inference.render import (
    annotated_image_to_jpeg_bytes,
    build_download_filename,
    compute_annotation_style,
    compute_display_transform,
    draw_detections,
    format_detection_label,
    scale_box_to_display,
)
from vision_trainer.inference.video import (
    SUPPORTED_VIDEO_SUFFIXES,
    VIDEO_DEFAULT_CONF,
    VideoInferenceError,
    build_video_output_filename,
    is_supported_video_path,
    probe_video,
    run_video_inference,
)

__all__ = [
    "AvailableModel",
    "DEFAULT_CONF",
    "DEFAULT_IOU",
    "Detection",
    "InferenceError",
    "InferenceResult",
    "SUPPORTED_VIDEO_SUFFIXES",
    "VIDEO_DEFAULT_CONF",
    "VideoInferenceError",
    "VideoInferenceResult",
    "VideoMetadata",
    "VideoProgress",
    "annotated_image_to_jpeg_bytes",
    "build_download_filename",
    "build_predict_kwargs",
    "build_video_output_filename",
    "compute_annotation_style",
    "compute_display_transform",
    "discover_trained_models",
    "draw_detections",
    "format_detection_label",
    "is_supported_video_path",
    "load_image_rgb",
    "normalize_ultralytics_results",
    "probe_video",
    "run_inference",
    "run_video_inference",
    "scale_box_to_display",
]
