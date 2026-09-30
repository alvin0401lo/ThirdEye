from __future__ import annotations

import time

from thirdeye.repo_object_finding import RepoGuidance


class RepoGuidanceAnnouncer:
    def __init__(
        self,
        repeat_interval_s: float = 1.5,
    ) -> None:
        self._repeat_interval = max(0.2, repeat_interval_s)
        self._last_guidance: RepoGuidance | None = None
        self._last_time = float("-inf")

    def update(
        self,
        guidance: RepoGuidance,
        now: float | None = None,
    ) -> str | None:
        timestamp = time.monotonic() if now is None else now
        elapsed = timestamp - self._last_time
        changed = guidance is not self._last_guidance
        if elapsed < self._repeat_interval:
            return None
        if not changed and elapsed < self._repeat_interval * 2:
            return None
        self._last_guidance = guidance
        self._last_time = timestamp
        return guidance.value
