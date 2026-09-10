"""Hub Résultats — historique des runs + stockage."""

from __future__ import annotations

import streamlit as st

from views.partials import results_history, storage


def render() -> None:
    st.title("Résultats")
    st.caption("Historique des entraînements et gestion du stockage local.")
    tab_hist, tab_storage = st.tabs(["Historique", "Stockage"])
    with tab_hist:
        results_history.render(embedded=True)
    with tab_storage:
        storage.render(embedded=True)
