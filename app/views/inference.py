"""Hub Inférence — détection / classification / segmentation."""

from __future__ import annotations

import streamlit as st

from views.partials import inference_all
from views.task_choice import render_task_selector


def render() -> None:
    st.title("Inférence")
    st.caption("Testez un modèle entraîné sur une image (ou une vidéo en détection).")
    task = render_task_selector(key="hub_infer_task", label="Tâche")
    st.divider()
    inference_all.render(task_key=task)
