"""Extensible training / inference task identifiers."""

from __future__ import annotations

from typing import Literal

TaskType = Literal["detect", "classify", "segment"]

KNOWN_TASKS: frozenset[str] = frozenset({"detect", "classify", "segment"})

TASK_LABELS_FR: dict[str, str] = {
    "detect": "Détection",
    "classify": "Classification",
    "segment": "Segmentation",
}

DEFAULT_TASK: TaskType = "detect"


def normalize_task(value: str | None) -> TaskType:
    """
    Normalize a stored task value.

    Missing / unknown values default to ``detect`` so legacy runs without a
    ``task`` field remain readable. ``segment`` is reserved for a future
    feature and is accepted when present without enabling UI for it yet.
    """
    raw = (value or "").strip().lower()
    if raw in KNOWN_TASKS:
        return raw  # type: ignore[return-value]
    return DEFAULT_TASK


def task_label_fr(task: str | None) -> str:
    return TASK_LABELS_FR.get(normalize_task(task), TASK_LABELS_FR[DEFAULT_TASK])
