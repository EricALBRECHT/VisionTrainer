"""Streamlit presentation for the training journal (display only)."""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from vision_trainer.training.log_display import (
    DEFAULT_EMPTY_LOG_PLACEHOLDER,
    TRAINING_LOG_DISPLAY_HEIGHT_PX,
    training_log_body,
)


def show_training_log(
    log_text: str | None,
    *,
    empty_placeholder: str = DEFAULT_EMPTY_LOG_PLACEHOLDER,
    auto_scroll: bool = True,
) -> None:
    """Show the training log in a compact scrollable ``st.code`` panel (~280 px).

    Does not alter log capture or content — presentation only.
    When ``auto_scroll`` is True, attempts to scroll to the latest lines after render.
    """
    body = training_log_body(log_text, empty_placeholder=empty_placeholder)
    st.code(body, language="text", height=TRAINING_LOG_DISPLAY_HEIGHT_PX)
    if auto_scroll:
        _scroll_training_log_to_bottom()


def _scroll_training_log_to_bottom() -> None:
    """Best-effort scroll of the last ``st.code`` block to the newest lines."""
    components.html(
        """
        <script>
        (function () {
          const roots = [];
          try { roots.push(window.parent.document); } catch (e) {}
          roots.push(document);
          for (const doc of roots) {
            const blocks = doc.querySelectorAll('[data-testid="stCode"]');
            if (!blocks.length) continue;
            const el = blocks[blocks.length - 1];
            const targets = [
              el.querySelector('[data-testid="stCodeScrollableContainer"]'),
              el.querySelector("pre"),
              el,
            ].filter(Boolean);
            for (const node of targets) {
              node.scrollTop = node.scrollHeight;
            }
          }
        })();
        </script>
        """,
        height=0,
    )
