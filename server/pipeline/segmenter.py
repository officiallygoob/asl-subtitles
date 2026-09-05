"""Server-side utterance segmentation from per-frame activity.

Mirrors iOS `UtteranceSegmenter` so `/v1/stream` can finalize phrases on
pause/rest even if the client forgets to send `utterance_end`.
"""

from __future__ import annotations


class UtteranceSegmenter:
    def __init__(
        self,
        rest_threshold: float = 0.055,
        rest_duration: float = 0.75,
        min_utterance: float = 0.50,
    ) -> None:
        self.rest_threshold = rest_threshold
        self.rest_duration = rest_duration
        self.min_utterance = min_utterance
        self.rest_started_at: float | None = None
        self.utterance_started_at: float | None = None
        self.is_in_utterance = False

    def reset(self) -> None:
        self.rest_started_at = None
        self.utterance_started_at = None
        self.is_in_utterance = False

    def push(self, activity: float, timestamp: float) -> str:
        """Return 'none' | 'began' | 'ended'."""
        if activity >= self.rest_threshold:
            self.rest_started_at = None
            if not self.is_in_utterance:
                self.is_in_utterance = True
                self.utterance_started_at = timestamp
                return "began"
            return "none"

        if self.rest_started_at is None:
            self.rest_started_at = timestamp
        if (
            self.is_in_utterance
            and self.rest_started_at is not None
            and self.utterance_started_at is not None
            and (timestamp - self.rest_started_at) >= self.rest_duration
            and (timestamp - self.utterance_started_at) >= self.min_utterance
        ):
            self.is_in_utterance = False
            self.utterance_started_at = None
            self.rest_started_at = None
            return "ended"
        return "none"
