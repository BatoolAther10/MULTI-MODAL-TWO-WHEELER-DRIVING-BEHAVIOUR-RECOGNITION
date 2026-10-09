"""Lightweight wall-clock stage timing helpers."""

from __future__ import annotations

from time import perf_counter_ns
from typing import MutableMapping


class StageTimer:
    """Record one named stage duration in milliseconds."""

    def __init__(self, stage: str, timings: MutableMapping[str, float]) -> None:
        """Create a timer that writes to the supplied timing mapping."""
        self.stage = stage
        self.timings = timings
        self._started_ns = 0

    def __enter__(self) -> "StageTimer":
        """Start the monotonic wall-clock timer."""
        self._started_ns = perf_counter_ns()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Store the elapsed duration in milliseconds."""
        self.timings[self.stage] = (perf_counter_ns() - self._started_ns) / 1_000_000.0
