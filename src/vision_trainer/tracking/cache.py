"""Per-track classification cache for video pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_RECLASSIFY_FRAMES = 30
DEFAULT_UNCERTAIN_RECLASSIFY_FRAMES = 5

_UNCERTAIN_STATUSES = frozenset({"unknown", "uncertain"})


@dataclass
class _CacheEntry:
    refinement: Any
    last_classified_frame: int
    status: str


class ClassificationTrackCache:
    """
    Reuse classification results for a stable ``track_id``.

    Reliable (ok) results are kept for ``reclassify_every_n_frames``.
    INCONNU / INCERTAIN expire sooner (``uncertain_reclassify_every_n_frames``).
    """

    def __init__(
        self,
        *,
        reclassify_every_n_frames: int = DEFAULT_RECLASSIFY_FRAMES,
        uncertain_reclassify_every_n_frames: int = DEFAULT_UNCERTAIN_RECLASSIFY_FRAMES,
    ) -> None:
        self.reclassify_every_n_frames = max(1, int(reclassify_every_n_frames))
        self.uncertain_reclassify_every_n_frames = max(
            1, int(uncertain_reclassify_every_n_frames)
        )
        self._entries: dict[int, _CacheEntry] = {}

    def reset(self) -> None:
        self._entries.clear()

    def invalidate_track(self, track_id: int) -> None:
        self._entries.pop(int(track_id), None)

    def should_classify(self, track_id: int, frame_index: int) -> bool:
        entry = self._entries.get(int(track_id))
        if entry is None:
            return True
        interval = self._interval_for_status(entry.status)
        return (int(frame_index) - entry.last_classified_frame) >= interval

    def get(self, track_id: int) -> Any | None:
        entry = self._entries.get(int(track_id))
        return None if entry is None else entry.refinement

    def put(
        self,
        track_id: int,
        frame_index: int,
        refinement: Any,
        *,
        status: str | None = None,
    ) -> None:
        resolved_status = status
        if resolved_status is None:
            resolved_status = str(getattr(refinement, "status", "") or "")
        self._entries[int(track_id)] = _CacheEntry(
            refinement=refinement,
            last_classified_frame=int(frame_index),
            status=resolved_status,
        )

    def _interval_for_status(self, status: str) -> int:
        if status in _UNCERTAIN_STATUSES:
            return self.uncertain_reclassify_every_n_frames
        return self.reclassify_every_n_frames
