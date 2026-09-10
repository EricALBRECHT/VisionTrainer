from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `from views...` when Streamlit runs this entrypoint.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

st.set_page_config(
    page_title="Vision Trainer",
    page_icon="👁️",
    layout="wide",
)


def _page_home() -> None:
    from views.home import render

    render()


def _page_datasets() -> None:
    from views.datasets import render

    render()


def _page_training() -> None:
    from views.training import render

    render()


def _page_inference() -> None:
    from views.inference import render

    render()


def _page_pipelines() -> None:
    from views.pipelines import render

    render()


def _page_video() -> None:
    from views.video import render

    render()


def _page_results() -> None:
    from views.results import render

    render()


_pages = [
    st.Page(_page_home, title="Accueil", icon="🏠", default=True, url_path="accueil"),
    st.Page(_page_datasets, title="Datasets", icon="📦", url_path="datasets"),
    st.Page(_page_training, title="Entraînement", icon="🧠", url_path="entrainement"),
    st.Page(_page_inference, title="Inférence", icon="🔍", url_path="inference"),
    st.Page(_page_pipelines, title="Pipelines", icon="🔗", url_path="pipelines"),
    st.Page(_page_video, title="Vidéo & Caméra", icon="🎬", url_path="video-camera"),
    st.Page(_page_results, title="Résultats", icon="📊", url_path="resultats"),
]

st.navigation(_pages).run()
