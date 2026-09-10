"""Hub Entraînement — détection / classification / segmentation."""

from __future__ import annotations

import streamlit as st

from views.partials import train_classify, train_detect, train_segment
from views.task_choice import render_task_selector


def render() -> None:
    st.title("Entraînement")
    st.caption("Configurez et lancez un entraînement Ultralytics.")
    task = render_task_selector(key="hub_train_task", label="Tâche")
    st.divider()
    if task == "detect":
        train_detect.render()
    elif task == "classify":
        train_classify.render()
    else:
        train_segment.render()
