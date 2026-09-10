from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Vision Trainer",
    page_icon="👁️",
    layout="wide",
)

st.title("Vision Trainer")
st.markdown(
    """
    Application locale pour entraîner des modèles **YOLO / Ultralytics**.

    ### Détection d'objets
    1. **Dataset** — ZIP YOLO (`data.yaml` ou `classes.txt` + `train/images`/`labels`)
    2. **Entraînement** — YOLO11 detect
    3. **Inférence** — image / vidéo + bounding boxes

    ### Classification d'images
    1. **Classification — Dataset** — ZIP ImageFolder (`train/<classe>/…`)
    2. **Classification — Entraînement** — YOLO11-cls
    3. **Inférence** — mode Classification (Top-N, seuil *Inconnu*)

    ### Commun
    - **Résultats** — historique détection **et** classification
    - **Stockage** — datasets, runs, modèles
    - CPU / GPU NVIDIA (Docker Compose GPU)
    """
)
