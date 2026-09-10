"""Generic video / camera processing built on existing inference engines."""

from vision_trainer.video.camera import (
    WSL_DOCKER_CAMERA_NOTE,
    CameraError,
    CameraInfo,
    capture_camera_frame,
    list_camera_indices,
    open_camera_capture,
    probe_camera,
)
from vision_trainer.video.color import (
    bgr_to_pil,
    bgr_to_rgb_array,
    pil_to_bgr,
    rgb_to_bgr_array,
)
from vision_trainer.video.engine import (
    export_video_summary_json,
    frame_timestamp_seconds,
    process_image_with_processor,
    process_video,
    video_summary_json_bytes,
)
from vision_trainer.video.models import FrameTimingsAgg, VideoJobSummary
from vision_trainer.video.processors import (
    DetectFrameProcessor,
    FrameProcessResult,
    PipelineFrameProcessor,
    SegmentFrameProcessor,
)

__all__ = [
    "WSL_DOCKER_CAMERA_NOTE",
    "CameraError",
    "CameraInfo",
    "DetectFrameProcessor",
    "FrameProcessResult",
    "FrameTimingsAgg",
    "PipelineFrameProcessor",
    "SegmentFrameProcessor",
    "VideoJobSummary",
    "bgr_to_pil",
    "bgr_to_rgb_array",
    "capture_camera_frame",
    "export_video_summary_json",
    "frame_timestamp_seconds",
    "list_camera_indices",
    "open_camera_capture",
    "pil_to_bgr",
    "probe_camera",
    "process_image_with_processor",
    "process_video",
    "rgb_to_bgr_array",
    "video_summary_json_bytes",
]
