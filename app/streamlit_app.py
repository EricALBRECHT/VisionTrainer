from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `from views...` / `from navigation...` when Streamlit runs this entrypoint.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

st.set_page_config(
    page_title="Vision Trainer",
    page_icon="👁️",
    layout="wide",
)

from navigation import NAV_PAGES, assert_nav_pages_ready  # noqa: E402

assert_nav_pages_ready()
st.navigation(NAV_PAGES).run()
