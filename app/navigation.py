"""Central Streamlit navigation for VisionTrainer.

All ``st.Page`` objects used by ``st.navigation`` live here so internal
links can call ``st.switch_page(PAGE_…)`` with the **same** instances.

Callables import view modules lazily to avoid circular imports.

Note: ``st.Page`` only stores title/url_path when a ScriptRunContext exists
(i.e. under ``streamlit run``). Metadata below is therefore the source of
truth for tests and documentation outside a live Streamlit session.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st


def _run_home() -> None:
    from views.home import render

    render()


def _run_datasets() -> None:
    from views.datasets import render

    render()


def _run_training() -> None:
    from views.training import render

    render()


def _run_inference() -> None:
    from views.inference import render

    render()


def _run_pipelines() -> None:
    from views.pipelines import render

    render()


def _run_results() -> None:
    from views.results import render

    render()


def _run_video() -> None:
    from views.video import render

    render()


# Ordered sidebar rubriques: (url_path key, title, icon, default?, runner)
NAV_SPECS: tuple[tuple[str, str, str, bool, Callable[[], None]], ...] = (
    ("accueil", "Accueil", "🏠", True, _run_home),
    ("datasets", "Datasets", "📦", False, _run_datasets),
    ("entrainement", "Entraînement", "🧠", False, _run_training),
    ("inference", "Inférence", "🔍", False, _run_inference),
    ("pipelines", "Pipelines", "🔗", False, _run_pipelines),
    ("video-camera", "Vidéo & Caméra", "🎬", False, _run_video),
    ("resultats", "Résultats", "📊", False, _run_results),
)

EXPECTED_SIDEBAR_TITLES: tuple[str, ...] = tuple(spec[1] for spec in NAV_SPECS)
EXPECTED_URL_PATHS: tuple[str, ...] = tuple(spec[0] for spec in NAV_SPECS)

PAGE_BY_KEY: dict[str, Any] = {}
NAV_PAGES: list[Any] = []

for _key, _title, _icon, _default, _runner in NAV_SPECS:
    _page = st.Page(
        _runner,
        title=_title,
        icon=_icon,
        default=_default,
        url_path=_key,
    )
    PAGE_BY_KEY[_key] = _page
    NAV_PAGES.append(_page)

PAGE_HOME = PAGE_BY_KEY["accueil"]
PAGE_DATASETS = PAGE_BY_KEY["datasets"]
PAGE_TRAINING = PAGE_BY_KEY["entrainement"]
PAGE_INFERENCE = PAGE_BY_KEY["inference"]
PAGE_PIPELINES = PAGE_BY_KEY["pipelines"]
PAGE_VIDEO = PAGE_BY_KEY["video-camera"]
PAGE_RESULTS = PAGE_BY_KEY["resultats"]

# Internal destinations used by Accueil / Résultats (must be NAV_PAGES members).
INTERNAL_SWITCH_TARGETS: tuple[Any, ...] = (
    PAGE_DATASETS,
    PAGE_TRAINING,
    PAGE_INFERENCE,
    PAGE_PIPELINES,
    PAGE_VIDEO,
    PAGE_RESULTS,
)


def assert_nav_pages_ready() -> None:
    """Fail fast if Pages were built without a Streamlit script context."""
    missing = [
        key
        for key, page in PAGE_BY_KEY.items()
        if not getattr(page, "_title", None)
    ]
    if missing:
        raise RuntimeError(
            "st.Page objects incomplets (pas de ScriptRunContext) pour: "
            + ", ".join(missing)
        )
