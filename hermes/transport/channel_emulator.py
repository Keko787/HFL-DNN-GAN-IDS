"""Sprint 2 — wireless channel emulator stub.

Mirrors what AERPAW's wireless channel emulator does between AVNs:
applies probabilistic packet drop + mean / jitter delay to each message
crossing the link. The TCP RFLink uses one emulator instance per
endpoint, applied symmetrically to inbound and outbound frames.

This is a *stub* — not modelling fading, multipath, or per-device SINR.
Sprint 6 (real AERPAW) replaces it with the testbed's emulator. The
shape matches what the Sprint 2 paper experiments need: a single
``rf_range_m`` knob in ``ContactSim`` plus configurable drop / delay
in the live transport, so the paper can sweep both axes.

Documented limitations:

* **S2-M2** — drop probability and delay are *per-message*, not
  per-byte. A 200 MB DiscPush gets the same drop chance and the same
  single delay as a 64-byte FLOpenSolicit. Real radio loses long
  messages more often (per-bit error rate × message length). For the
  paper's ``rf_range_m`` sweep this doesn't matter — that sweep happens
  in the offline ``ContactSim``, not over the live transport. The live
  transport's drop/delay are independent fault-injection knobs.

Design refs:
* HERMES_FL_Scheduler_Implementation_Plan.md §3.6 Sprint 2 task 1
* AERPAW user manual §3.2 (development environment includes wireless
  channel emulator)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ChannelEmulator:
    """Probabilistic drop + mean-with-jitter delay model.

    Parameters:
        drop_prob: Per-message probability of being dropped. 0.0 disables
            drops; 1.0 drops every message.
        mean_delay_s: Mean per-message latency in seconds. 0.0 disables.
        jitter_s: Half-range of uniform jitter around ``mean_delay_s``.
            Effective delay sits in ``[mean - jitter, mean + jitter]``,
            floored at 0.
        seed: Optional RNG seed for reproducible drops + jitter.
    """

    drop_prob: float = 0.0
    mean_delay_s: float = 0.0
    jitter_s: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.drop_prob <= 1.0):
            raise ValueError(
                f"drop_prob must be in [0, 1], got {self.drop_prob}"
            )
        if self.mean_delay_s < 0.0:
            raise ValueError(f"mean_delay_s must be >= 0, got {self.mean_delay_s}")
        if self.jitter_s < 0.0:
            raise ValueError(f"jitter_s must be >= 0, got {self.jitter_s}")
        self._rng = random.Random(self.seed)

    def apply(self) -> Tuple[bool, float]:
        """Roll one drop + delay decision.

        Returns:
            ``(drop, delay_seconds)``. If ``drop`` is True, the caller
            should silently swallow the message; otherwise it sleeps
            ``delay_seconds`` (which may be 0) before proceeding.
        """
        if self.drop_prob > 0.0 and self._rng.random() < self.drop_prob:
            return True, 0.0
        if self.mean_delay_s == 0.0 and self.jitter_s == 0.0:
            return False, 0.0
        delay = self.mean_delay_s + self._rng.uniform(-self.jitter_s, self.jitter_s)
        return False, max(0.0, delay)


def no_op_emulator() -> ChannelEmulator:
    """Drop-free, zero-delay emulator — used as the default in tests."""
    return ChannelEmulator(drop_prob=0.0, mean_delay_s=0.0, jitter_s=0.0)
