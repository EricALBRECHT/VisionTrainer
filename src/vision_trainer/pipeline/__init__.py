"""Detection → optional classification pipelines."""

from vision_trainer.pipeline.crop import CropError, crop_from_detection, padded_crop_box
from vision_trainer.pipeline.engine import (
    PipelineEngineError,
    load_detector_class_names,
    result_table_rows,
    run_pipeline,
    status_label_fr,
)
from vision_trainer.pipeline.models import (
    PIPELINE_FORMAT_VERSION,
    ClassificationRefinement,
    ClassMapping,
    EnrichedDetection,
    PipelineConfig,
    PipelineResult,
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
    "ClassMapping",
    "CropError",
    "EnrichedDetection",
    "PipelineConfig",
    "PipelineEngineError",
    "PipelineResult",
    "PipelineStoreError",
    "crop_from_detection",
    "delete_pipeline",
    "draw_pipeline_result",
    "duplicate_pipeline",
    "list_pipelines",
    "load_detector_class_names",
    "load_pipeline",
    "make_pipeline_id",
    "padded_crop_box",
    "result_table_rows",
    "run_pipeline",
    "save_pipeline",
    "status_label_fr",
    "validate_pipeline_config",
]
