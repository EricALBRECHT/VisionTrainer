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

    Utilisez la page **Dataset** pour charger et valider un dataset YOLO.
    """
)
