"""Hub Vidéo & Caméra."""

from __future__ import annotations

from views.partials import video_camera


def render() -> None:
    video_camera.render()
