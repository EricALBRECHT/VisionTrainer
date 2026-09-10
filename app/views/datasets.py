"""Hub Datasets — détection / classification / segmentation."""

from __future__ import annotations

import streamlit as st

from views.partials import dataset_classify, dataset_detect, dataset_segment
from views.task_choice import render_task_selector


def render() -> None:
    st.title("Datasets")
    st.caption("Importez et validez un dataset selon la tâche.")
    task = render_task_selector(key="hub_datasets_task", label="Tâche")
    st.divider()
    if task == "detect":
        dataset_detect.render()
    elif task == "classify":
        dataset_classify.render()
    else:
        dataset_segment.render()
