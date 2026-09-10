"""Classification package (ImageFolder / Ultralytics YOLO-cls)."""

from vision_trainer.classify.models import (
    ClassifyClassStats,
    ClassifyDatasetInfo,
    ClassifyValidationResult,
)
from vision_trainer.classify.parser import find_classify_dataset_root, load_classify_dataset
from vision_trainer.classify.predictor import ClassifyInferenceResult, run_classify_inference
from vision_trainer.classify.reject import ClassScore, ClassifyDecision, decide_classification
from vision_trainer.classify.session import (
    SESSION_CLS_DATASET_KEY,
    classify_dataset_from_session_payload,
    classify_dataset_to_session_payload,
)
from vision_trainer.classify.validator import class_size_guidance, validate_classify_dataset

__all__ = [
    "SESSION_CLS_DATASET_KEY",
    "ClassScore",
    "ClassifyClassStats",
    "ClassifyDatasetInfo",
    "ClassifyDecision",
    "ClassifyInferenceResult",
    "ClassifyValidationResult",
    "class_size_guidance",
    "classify_dataset_from_session_payload",
    "classify_dataset_to_session_payload",
    "decide_classification",
    "find_classify_dataset_root",
    "load_classify_dataset",
    "run_classify_inference",
    "validate_classify_dataset",
]
