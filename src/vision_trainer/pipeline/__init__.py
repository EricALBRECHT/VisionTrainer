"""Detection → optional classification → optional segmentation pipelines."""

from vision_trainer.pipeline.crop import (
    CropError,
    CropRegion,
    crop_from_detection,
    crop_region_from_detection,
    padded_crop_box,
)
from vision_trainer.pipeline.engine import (
    PipelineEngineError,
    load_detector_class_names,
    result_table_rows,
    run_pipeline,
    status_label_fr,
)
from vision_trainer.pipeline.export import export_pipeline_json, pipeline_result_json_bytes
from vision_trainer.pipeline.models import (
    PIPELINE_FORMAT_VERSION,
    ClassificationRefinement,
    ClassificationStage,
    ClassMapping,
    EnrichedDetection,
    PipelineConfig,
    PipelineResult,
    PipelineTimings,
    SegmentationInstanceResult,
    SegmentationRefinement,
    SegmentationStage,
)
from vision_trainer.pipeline.render import draw_pipeline_result
from vision_trainer.pipeline.store import (
    PipelineStoreError,
    delete_pipeline,
    duplicate_pipeline,
    list_pipelines,
    load_pipeline,
    make_pipeline_id,
    save_pipeline,
    validate_pipeline_config,
)

__all__ = [
    "PIPELINE_FORMAT_VERSION",
    "ClassificationRefinement",
    "ClassificationStage",
    "ClassMapping",
    "CropError",
    "CropRegion",
    "EnrichedDetection",
    "PipelineConfig",
    "PipelineEngineError",
    "PipelineResult",
    "PipelineStoreError",
    "PipelineTimings",
    "SegmentationInstanceResult",
    "SegmentationRefinement",
    "SegmentationStage",
    "crop_from_detection",
    "crop_region_from_detection",
    "delete_pipeline",
    "draw_pipeline_result",
    "duplicate_pipeline",
    "export_pipeline_json",
    "list_pipelines",
    "load_detector_class_names",
    "load_pipeline",
    "make_pipeline_id",
    "padded_crop_box",
    "pipeline_result_json_bytes",
    "result_table_rows",
    "run_pipeline",
    "save_pipeline",
    "status_label_fr",
    "validate_pipeline_config",
]
