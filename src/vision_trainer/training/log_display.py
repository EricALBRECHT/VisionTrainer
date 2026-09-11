"""Presentation helpers for the training log panel (UI only)."""

from __future__ import annotations

import html
import json

# Compact scrollable journal height on the Training page (px).
TRAINING_LOG_DISPLAY_HEIGHT_PX = 280

DEFAULT_EMPTY_LOG_PLACEHOLDER = "(journal encore vide)"
DEFAULT_TRAINING_LOG_STORAGE_KEY = "vt-training-log"


def training_log_body(
    log_text: str | None,
    *,
    empty_placeholder: str = DEFAULT_EMPTY_LOG_PLACEHOLDER,
) -> str:
    """Return the text shown in the journal panel (never truncates further here)."""
    if log_text:
        return log_text
    return empty_placeholder


def escape_training_log_html(text: str) -> str:
    """Escape log text so it cannot be interpreted as HTML/JS in the viewer."""
    return html.escape(text, quote=True)


def build_training_log_html(
    log_text: str,
    *,
    height_px: int = TRAINING_LOG_DISPLAY_HEIGHT_PX,
    auto_scroll: bool = True,
    storage_key: str = DEFAULT_TRAINING_LOG_STORAGE_KEY,
) -> str:
    """Build a self-contained HTML log viewer with sticky bottom auto-scroll.

    Streamlit recreates the iframe on each rerun; ``st.code`` therefore resets
    ``scrollTop`` to 0. This document sets ``scrollTop = scrollHeight`` after
    load when the user was already near the bottom (default / sticky mode).
    """
    escaped = escape_training_log_html(log_text)
    auto_js = "true" if auto_scroll else "false"
    key_js = json.dumps(storage_key)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
  }}
  #training-log {{
    box-sizing: border-box;
    height: {int(height_px)}px;
    max-height: {int(height_px)}px;
    overflow-x: auto;
    overflow-y: auto;
    margin: 0;
    padding: 0.75rem 1rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
      "Liberation Mono", "Courier New", monospace;
    font-size: 0.85rem;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
    color: #fafafa;
    background: #0e1117;
    border: 1px solid rgba(250, 250, 250, 0.2);
    border-radius: 0.5rem;
  }}
</style>
</head>
<body>
<pre id="training-log">{escaped}</pre>
<script>
(function () {{
  const el = document.getElementById("training-log");
  if (!el) return;
  const AUTO = {auto_js};
  const KEY = {key_js};
  const NEAR_PX = 48;

  function nearBottom() {{
    return (el.scrollHeight - el.scrollTop - el.clientHeight) <= NEAR_PX;
  }}

  function loadState() {{
    try {{
      const raw = sessionStorage.getItem(KEY);
      return raw ? JSON.parse(raw) : {{ sticky: true, scrollTop: 0 }};
    }} catch (e) {{
      return {{ sticky: true, scrollTop: 0 }};
    }}
  }}

  function saveState(sticky, scrollTop) {{
    try {{
      sessionStorage.setItem(
        KEY,
        JSON.stringify({{ sticky: !!sticky, scrollTop: scrollTop | 0 }})
      );
    }} catch (e) {{}}
  }}

  el.addEventListener("scroll", function () {{
    saveState(nearBottom(), el.scrollTop);
  }});

  const state = loadState();
  if (AUTO && state.sticky !== false) {{
    el.scrollTop = el.scrollHeight;
    saveState(true, el.scrollTop);
  }} else if (typeof state.scrollTop === "number") {{
    el.scrollTop = state.scrollTop;
  }}
}})();
</script>
</body>
</html>
"""
