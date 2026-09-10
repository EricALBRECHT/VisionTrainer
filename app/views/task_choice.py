"""Task selectors shared by Datasets / Entraînement / Inférence hubs."""

from __future__ import annotations

from typing import Literal

import streamlit as st

TaskKey = Literal["detect", "classify", "segment"]

TASK_LABELS: tuple[str, ...] = ("Détection", "Classification", "Segmentation")
LABEL_TO_TASK: dict[str, TaskKey] = {
    "Détection": "detect",
    "Classification": "classify",
    "Segmentation": "segment",
}
TASK_TO_LABEL: dict[TaskKey, str] = {
    "detect": "Détection",
    "classify": "Classification",
    "segment": "Segmentation",
}


def task_key_from_label(label: str) -> TaskKey:
    try:
        return LABEL_TO_TASK[label]
    except KeyError as exc:
        raise ValueError(f"Tâche inconnue: {label!r}") from exc


def label_from_task_key(task_key: str) -> str:
    try:
        return TASK_TO_LABEL[task_key]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"Clé de tâche inconnue: {task_key!r}") from exc


def render_task_selector(
    *,
    key: str,
    label: str = "Tâche",
    default_label: str = "Détection",
) -> TaskKey:
    """Show a compact task chooser and return the internal task key."""
    default = default_label if default_label in TASK_LABELS else TASK_LABELS[0]
    selected = st.segmented_control(
        label,
        options=list(TASK_LABELS),
        default=default,
        key=key,
        required=True,
    )
    if selected is None:
        selected = default
    return task_key_from_label(str(selected))
