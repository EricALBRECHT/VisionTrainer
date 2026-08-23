import streamlit as st

st.set_page_config(
    page_title="Vision Trainer",
    page_icon="👁️",
    layout="wide",
)

st.title("Vision Trainer")
st.markdown(
    """
    Application locale pour préparer et entraîner des modèles de détection d'objets.

    1. Page **Dataset** — chargez et validez un dataset YOLO (ZIP).
    2. Page **Entraînement** — lancez un entraînement Ultralytics YOLO11.
    3. Page **Inférence** — testez un modèle (`best.pt`) sur une image.
    """
)
