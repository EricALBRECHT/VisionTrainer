"""Streamlit presentation for the training journal (display only)."""

from __future__ import annotations

import streamlit.components.v1 as components

from vision_trainer.training.log_display import (
    DEFAULT_EMPTY_LOG_PLACEHOLDER,
    DEFAULT_TRAINING_LOG_STORAGE_KEY,
    TRAINING_LOG_DISPLAY_HEIGHT_PX,
    build_training_log_html,
    training_log_body,
)


def show_training_log(
    log_text: str | None,
    *,
    empty_placeholder: str = DEFAULT_EMPTY_LOG_PLACEHOLDER,
    auto_scroll: bool = True,
    storage_key: str = DEFAULT_TRAINING_LOG_STORAGE_KEY,
) -> None:
    """Show the training log in a compact HTML panel (~280 px) with bottom auto-scroll.

    Uses ``components.html`` instead of ``st.code`` so each Streamlit rerun can
    re-apply ``scrollTop = scrollHeight`` inside the iframe. Log capture is untouched.
    """
    body = training_log_body(log_text, empty_placeholder=empty_placeholder)
    html_doc = build_training_log_html(
        body,
        height_px=TRAINING_LOG_DISPLAY_HEIGHT_PX,
        auto_scroll=auto_scroll,
        storage_key=storage_key,
    )
    # iframe height matches the panel; scrolling happens inside #training-log.
    components.html(html_doc, height=TRAINING_LOG_DISPLAY_HEIGHT_PX, scrolling=False)
