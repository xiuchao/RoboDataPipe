from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    name: str
    local_index: int
    score: float
    method: str
    signal_before: float | None = None
    signal_after: float | None = None


@dataclass(frozen=True)
class EventTimeline:
    events: dict[str, Event]