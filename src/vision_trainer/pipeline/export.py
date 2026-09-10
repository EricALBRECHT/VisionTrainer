"""JSON export for pipeline inference results."""

from __future__ import annotations

import json
from pathlib import Path

from vision_trainer.io_utils import atomic_write_text
from vision_trainer.pipeline.models import PipelineResult


def export_pipeline_json(result: PipelineResult, destination: Path | str) -> Path:
    path = Path(destination)
    atomic_write_text(
        path,
        json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False) + "\n",
    )
    return path


def pipeline_result_json_bytes(result: PipelineResult) -> bytes:
    return (
        json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
