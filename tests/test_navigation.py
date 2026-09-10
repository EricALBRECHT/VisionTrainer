"""Tests for navigation task selectors and internal page links."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from navigation import (  # noqa: E402
    EXPECTED_SIDEBAR_TITLES,
    EXPECTED_URL_PATHS,
    INTERNAL_SWITCH_TARGETS,
    NAV_PAGES,
    NAV_SPECS,
    PAGE_BY_KEY,
    PAGE_DATASETS,
    PAGE_HOME,
    PAGE_INFERENCE,
    PAGE_PIPELINES,
    PAGE_RESULTS,
    PAGE_TRAINING,
    PAGE_VIDEO,
)
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


def test_navigation_entrypoint_uses_central_nav_pages() -> None:
    text = (_APP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    assert "from navigation import NAV_PAGES" in text
    assert "st.navigation(NAV_PAGES)" in text
    assert "assert_nav_pages_ready" in text
    pages_dir = _APP_DIR / "pages"
    leftover = list(pages_dir.glob("*.py")) if pages_dir.is_dir() else []
    assert leftover == [], f"Anciennes pages encore dans pages/: {leftover}"


def test_nav_specs_match_expected_sidebar() -> None:
    assert EXPECTED_SIDEBAR_TITLES == (
        "Accueil",
        "Datasets",
        "Entraînement",
        "Inférence",
        "Pipelines",
        "Vidéo & Caméra",
        "Résultats",
    )
    assert EXPECTED_URL_PATHS == (
        "accueil",
        "datasets",
        "entrainement",
        "inference",
        "pipelines",
        "video-camera",
        "resultats",
    )
    assert len(NAV_SPECS) == len(NAV_PAGES) == len(PAGE_BY_KEY)
    assert list(PAGE_BY_KEY) == list(EXPECTED_URL_PATHS)


def test_central_page_objects_are_unique_and_registered() -> None:
    pages = [
        PAGE_HOME,
        PAGE_DATASETS,
        PAGE_TRAINING,
        PAGE_INFERENCE,
        PAGE_PIPELINES,
        PAGE_VIDEO,
        PAGE_RESULTS,
    ]
    assert pages == NAV_PAGES
    assert len({id(p) for p in pages}) == len(pages)
    for key, page in PAGE_BY_KEY.items():
        assert page is PAGE_BY_KEY[key]
        assert page in NAV_PAGES


def test_internal_switch_targets_are_registered_nav_pages() -> None:
    assert set(INTERNAL_SWITCH_TARGETS).issubset(set(NAV_PAGES))
    assert PAGE_HOME not in INTERNAL_SWITCH_TARGETS
    assert PAGE_INFERENCE in INTERNAL_SWITCH_TARGETS
    assert PAGE_DATASETS in INTERNAL_SWITCH_TARGETS


def test_switch_page_calls_use_page_objects_not_url_strings() -> None:
    """AST scan: active app code must not switch_page to bare url_path strings."""
    forbidden_string_targets = set(EXPECTED_URL_PATHS) | {
        "training",
        "video",
        "results",
    }
    app_roots = [
        _APP_DIR / "views",
        _APP_DIR / "navigation.py",
        _APP_DIR / "streamlit_app.py",
    ]
    offenders: list[str] = []

    files: list[Path] = []
    for root in app_roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_switch = (
                isinstance(func, ast.Attribute) and func.attr == "switch_page"
            ) or (isinstance(func, ast.Name) and func.id == "switch_page")
            if not is_switch or not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                if arg0.value in forbidden_string_targets or arg0.value.startswith(
                    "pages/"
                ):
                    offenders.append(
                        f"{path}:{node.lineno}: switch_page({arg0.value!r})"
                    )

    assert offenders == [], "Liens internes invalides:\n" + "\n".join(offenders)


def test_home_and_results_import_central_pages() -> None:
    home = (_APP_DIR / "views" / "home.py").read_text(encoding="utf-8")
    results = (_APP_DIR / "views" / "partials" / "results_history.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "PAGE_DATASETS",
        "PAGE_TRAINING",
        "PAGE_INFERENCE",
        "PAGE_PIPELINES",
        "PAGE_VIDEO",
        "PAGE_RESULTS",
    ):
        assert name in home
        assert f"st.switch_page({name})" in home
    assert "PAGE_INFERENCE" in results
    assert "st.switch_page(PAGE_INFERENCE)" in results
    assert 'switch_page("datasets")' not in home
    assert 'switch_page("inference")' not in results
