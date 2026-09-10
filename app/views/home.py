"""Page d'accueil VisionTrainer."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Vision Trainer")
    st.markdown(
        """
        Application locale pour entraîner et évaluer des modèles **YOLO / Ultralytics**
        (détection, classification, segmentation), chaîner des **pipelines**,
        et traiter de la **vidéo** ou une **caméra**.
        """
    )

    st.subheader("Parcours recommandé")
    st.markdown(
        """
        **Datasets** → **Entraînement** → **Inférence** → **Pipelines**
        → **Vidéo & Caméra** → **Résultats**
        """
    )

    cols = st.columns(3)
    with cols[0]:
        if st.button("📦 Datasets", use_container_width=True):
            st.switch_page("datasets")
        if st.button("🧠 Entraînement", use_container_width=True):
            st.switch_page("entrainement")
        if st.button("🔍 Inférence", use_container_width=True):
            st.switch_page("inference")
    with cols[1]:
        if st.button("🔗 Pipelines", use_container_width=True):
            st.switch_page("pipelines")
        if st.button("🎬 Vidéo & Caméra", use_container_width=True):
            st.switch_page("video-camera")
        if st.button("📊 Résultats", use_container_width=True):
            st.switch_page("resultats")
    with cols[2]:
        st.info(
            "CPU ou GPU NVIDIA (Docker Compose GPU). "
            "Les datasets, runs et poids restent sous le dossier de données local."
        )

    st.divider()
    st.caption(
        "Choisissez une rubrique dans la barre latérale. "
        "Chaque rubrique (Datasets, Entraînement, Inférence) propose ensuite "
        "Détection / Classification / Segmentation."
    )
