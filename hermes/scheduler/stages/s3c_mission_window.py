"""Stage 3c — mission-level deadline-window adaptation (toggle-able).

**The signal the per-device rule cannot see.** S3's adaptation is *per device*:
a clean contact shrinks that device's window, a missed one widens it. That
responds to "this device is unreliable" but is blind to "the mule keeps failing
to complete its circuit at all" — a systemic condition where every window is too
tight for the geometry, budget and speed the mule actually has. No amount of
per-device feedback diagnoses that, because from any single device's point of
view nothing looks wrong.

This stage adds that second, global signal: track how much of each planned
mission actually got served, and when the mule is systematically falling short,
widen **everyone's** effective window until it can keep up.

**Design choices worth stating:**

* **The scale is a pure function of recent history, not an accumulator.** It is
  recomputed from the rolling record each time rather than nudged up and down,
  so it cannot wind up, cannot drift, and is trivially testable: the same
  history always yields the same scale.
* **It only ever widens.** Shrinking is left to the per-device rule, which has
  better information about whom to reward. Below the success target this
  widens; at or above it, it returns exactly 1.0 and is invisible.
* **It is bounded.** `max_scale` caps how far the system will stretch itself, so
  a persistently impossible configuration degrades to "windows are wide" rather
  than "windows are infinite".
* **It is off by default.** Every recorded sweep predates it; with
  ``enabled=False`` the scale is exactly 1.0 and the stage is inert.

Toggle it on with ``--mission-window-adaptation`` so its effect can be measured
against an otherwise identical configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple


@dataclass
class MissionWindowAdapter:
    """Rolling mission-success tracker that yields a deadline-window multiplier.

    ``record(served, planned)`` after each mission; read :attr:`scale` when
    computing deadlines.
    """

    enabled: bool = False
    #: how many recent missions inform the scale
    window: int = 5
    #: served/planned at or above which no widening is applied
    target_success: float = 0.8
    #: how hard to widen per unit of shortfall (1.0 shortfall -> +gain)
    gain: float = 2.0
    min_scale: float = 1.0
    max_scale: float = 4.0

    _history: Deque[Tuple[int, int]] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if not (0.0 <= self.target_success <= 1.0):
            raise ValueError(
                f"target_success must be in [0,1], got {self.target_success}"
            )
        if self.min_scale < 1.0 or self.max_scale < self.min_scale:
            raise ValueError(
                f"need 1.0 <= min_scale <= max_scale, got "
                f"{self.min_scale}, {self.max_scale}"
            )
        self._history = deque(maxlen=self.window)

    # -- recording ---------------------------------------------------------- #

    def record(self, served: int, planned: int) -> None:
        """Log one mission's outcome. ``planned`` of 0 is ignored (nothing to judge)."""
        if planned <= 0:
            return
        self._history.append((max(0, int(served)), int(planned)))

    def reset(self) -> None:
        self._history.clear()

    # -- reading ------------------------------------------------------------ #

    @property
    def n_missions(self) -> int:
        return len(self._history)

    @property
    def success_rate(self) -> Optional[float]:
        """Served / planned over the rolling window; None until any data."""
        if not self._history:
            return None
        served = sum(s for s, _ in self._history)
        planned = sum(p for _, p in self._history)
        return served / planned if planned else None

    @property
    def scale(self) -> float:
        """Multiplier applied to every device's fulfilment window.

        1.0 when disabled, when no history has been recorded, or when the mule
        is meeting ``target_success`` — so the stage is invisible unless the
        mission is actually falling short.
        """
        if not self.enabled:
            return 1.0
        rate = self.success_rate
        if rate is None:
            return 1.0
        shortfall = max(0.0, self.target_success - rate)
        return min(self.max_scale, max(self.min_scale, 1.0 + self.gain * shortfall))

    def describe(self) -> str:
        rate = self.success_rate
        return (
            f"MissionWindowAdapter(enabled={self.enabled}, "
            f"missions={self.n_missions}, "
            f"success={'n/a' if rate is None else f'{rate:.2f}'}, "
            f"scale={self.scale:.2f})"
        )
