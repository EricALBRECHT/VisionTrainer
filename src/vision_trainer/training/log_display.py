"""Presentation helpers for the training log panel (UI only)."""

from __future__ import annotations

# Compact scrollable journal height on the Training page (px).
TRAINING_LOG_DISPLAY_HEIGHT_PX = 280

DEFAULT_EMPTY_LOG_PLACEHOLDER = "(journal encore vide)"


def training_log_body(
    log_text: str | None,
    *,
    empty_placeholder: str = DEFAULT_EMPTY_LOG_PLACEHOLDER,
) -> str:
    """Return the text shown in the journal panel (never truncates further here)."""
    if log_text:
        return log_text
    return empty_placeholder
