"""JSON export helpers for segmentation inference."""

from __future__ import annotations

import json
from pathlib import Path

from vision_trainer.io_utils import atomic_write_text
from vision_trainer.segment.models import SegmentationResult


def export_segmentation_json(
    result: SegmentationResult,
    destination: Path | str,
) -> Path:
    """Write polygon-based inference results (no bitmap masks)."""
    path = Path(destination)
    payload = result.to_json_dict()
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    return path


def segmentation_result_json_bytes(result: SegmentationResult) -> bytes:
    return (json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
