"""Device-side FL state machine used by ``ClientMission``.

Design ref: HERMES_FL_Scheduler_Design.md §5.3 (ClientMission lifecycle)
and Implementation Plan §3 Phase 2 tasks.

The mission-scope state is intentionally tiny — three values, explicit
transitions. Anything richer (training epoch counters, gradient stats)
lives in the program, not in the state tag.
"""

from __future__ import annotations

from enum import Enum


class FLState(str, Enum):
    """Per-device FL readiness state observed by the mule."""

    BUSY = "busy"             # in-mission work already running (local training, etc.)
    UNAVAILABLE = "unavailable"  # not willing/able to run FL this round
    FL_OPEN = "fl_open"       # ready for a mule to open an FL session

    def can_open_session(self) -> bool:
        return self is FLState.FL_OPEN
