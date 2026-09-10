"""Tests for navigation task selectors (no Streamlit UI)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from views.task_choice import (  # noqa: E402
    LABEL_TO_TASK,
    TASK_LABELS,
    TASK_TO_LABEL,
    label_from_task_key,
    task_key_from_label,
)


def test_task_labels_cover_three_tasks() -> None:
    assert TASK_LABELS == ("Détection", "Classification", "Segmentation")
    assert set(LABEL_TO_TASK) == set(TASK_LABELS)
    assert set(TASK_TO_LABEL.values()) == set(TASK_LABELS)


def test_task_key_roundtrip() -> None:
    for label, key in LABEL_TO_TASK.items():
        assert task_key_from_label(label) == key
        assert label_from_task_key(key) == label


def test_task_key_from_label_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Tâche inconnue"):
        task_key_from_label("Tracking")


def test_label_from_task_key_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Clé de tâche inconnue"):
        label_from_task_key("track")


def test_navigation_entrypoint_defines_expected_pages() -> None:
    """Smoke: entrypoint source lists the seven sidebar rubriques."""
    text = (_APP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    for title in (
        "Accueil",
        "Datasets",
        "Entraînement",
        "Inférence",
        "Pipelines",
        "Vidéo & Caméra",
        "Résultats",
    ):
        assert title in text
    assert "st.navigation" in text
    # Old numbered pages must not remain as auto-discovered Streamlit pages.
    pages_dir = _APP_DIR / "pages"
    leftover = list(pages_dir.glob("*.py")) if pages_dir.is_dir() else []
    assert leftover == [], f"Anciennes pages encore dans pages/: {leftover}"
