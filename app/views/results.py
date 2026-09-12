"""Hub Résultats — historique, analyse, comparaison, stockage."""

from __future__ import annotations

import streamlit as st

from views.partials import (
    results_analysis,
    results_compare,
    results_history,
    storage,
)


def render() -> None:
    st.title("Résultats")
    st.caption(
        "Historique des entraînements, analyse, comparaison et stockage local."
    )
    tab_hist, tab_analysis, tab_compare, tab_storage = st.tabs(
        ["Historique", "Analyse", "Comparaison", "Stockage"]
    )
    with tab_hist:
        results_history.render(embedded=True)
    with tab_analysis:
        results_analysis.render()
    with tab_compare:
        results_compare.render()
    with tab_storage:
        storage.render(embedded=True)
