"""Hub Pipelines — configuration + inférence."""

from __future__ import annotations

import streamlit as st

from views.partials import pipelines_config, pipelines_infer


def render() -> None:
    st.title("Pipelines")
    st.caption(
        "Chaînez détection → classification et/ou segmentation optionnelles par classe."
    )
    tab_cfg, tab_infer = st.tabs(["Configuration", "Inférence"])
    with tab_cfg:
        pipelines_config.render()
    with tab_infer:
        pipelines_infer.render()
