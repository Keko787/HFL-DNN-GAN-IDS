"""EX-4.3 — RF channel environment for the L1 experiment (arm H3).

Models the mule->base-station backhaul as a set of RF channels whose
effective SNR varies over the mission sequence, and turns a channel choice
into a per-mission backhaul-loss probability. This is what lets adaptive
channel selection (L1, arm H3) *matter*: a good channel -> high SNR -> low
loss -> more rounds close.

Validity discipline (learned from the jittery remediation):

* **Bands cross over.** Each band peaks at a different time (distinct
  phases), so **no single fixed band is best throughout** — otherwise the
  static-best baseline (H2) would tie the adaptive controller (H3) and L1
  would have no honest value. Adaptation only helps when the best band
  changes, which is the realistic time-varying condition.
* **Fair baseline.** H2 uses ``best_average_band`` — the band a deployer
  picks from historical averages *without* real-time tracking (Exp 2's
  "Expected fixed"), NOT the retrospective per-instant oracle.
* **Same conditions.** H2 and H3 face the identical seeded SNR trace.
* **Clean ~ no L1 benefit.** Under clean links all bands sit high and
  stable, so fixed ~ adaptive and L1's effect is (correctly) negligible;
  the benefit, if any, appears under jittery.

The whole model is deterministic in the paired seed and runs in the driver
(in-process), producing a per-mission loss schedule the cluster applies —
so no cross-process channel coordination is needed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from hermes.l1.channel_utility import AdaptiveChannelController, best_average_band


@dataclass
class ChannelModel:
    """Per-band effective SNR (dB) over the mission sequence.

    ``snr(m, c) = base + g[c] + amplitude * sin(2*pi*(m/period + phase[c])) + noise``.
    Jittery lowers the base and raises the amplitude/noise, so bands dip into
    lossy troughs at different times.
    """

    n_bands: int = 3
    n_missions: int = 4
    seed: int = 0
    jittery: bool = False

    def __post_init__(self) -> None:
        rng = random.Random((self.seed ^ 0x0C0FFEE) & 0x7FFFFFFF)
        # Per-band static gain g(c) — modest spread so no band dominates.
        self._g = [rng.uniform(0.0, 3.0) for _ in range(self.n_bands)]
        # Distinct phases -> bands peak at different times (crossover).
        self._phase = [i / self.n_bands for i in range(self.n_bands)]
        rng.shuffle(self._phase)
        self._noise = [
            [rng.gauss(0.0, 1.5 if self.jittery else 0.4) for _ in range(self.n_bands)]
            for _ in range(self.n_missions)
        ]
        self._base = 6.0 if self.jittery else 12.0
        self._amp = 5.0 if self.jittery else 1.0
        self._period = max(2, self.n_missions)

    def snr(self, mission: int, band: int) -> float:
        wave = self._amp * math.sin(
            2.0 * math.pi * (mission / self._period + self._phase[band])
        )
        return self._base + self._g[band] + wave + self._noise[mission][band]

    def snr_by_mission(self) -> List[List[float]]:
        return [
            [self.snr(m, b) for b in range(self.n_bands)]
            for m in range(self.n_missions)
        ]


def loss_from_snr(snr_db: float, *, mid: float = 3.0, scale: float = 2.0) -> float:
    """Backhaul upload-loss probability from effective SNR (logistic).

    High SNR -> ~0 loss; SNR below ~``mid`` -> loss climbs toward 1. Tuned so
    a healthy channel (~12 dB) loses ~1% and a deep trough (~1 dB) loses
    ~70%+.
    """
    x = (snr_db - mid) / scale
    return 1.0 / (1.0 + math.exp(x))


@dataclass(frozen=True)
class BackhaulPlan:
    """Per-mission backhaul-loss schedule + the L1 trace behind it."""

    loss_schedule: List[float]
    chosen_bands: List[int]
    mean_chosen_snr_db: float
    adaptive: bool


def backhaul_plan(
    model: ChannelModel,
    *,
    adaptive: bool,
    switch_cost: float = 0.5,
    channel_use_cost: Optional[Tuple[float, ...]] = None,
) -> BackhaulPlan:
    """Turn the channel trace into a per-mission loss schedule.

    ``adaptive=True`` (arm H3) runs the ``U(c,t)`` controller, tracking the
    best band each mission; ``adaptive=False`` (arms H1/H2) holds the single
    best-average band. Both read the same SNR trace.
    """
    snr_by_m = model.snr_by_mission()
    chosen: List[int] = []
    losses: List[float] = []
    snrs: List[float] = []

    if adaptive:
        ctrl = AdaptiveChannelController(
            channel_use_cost=channel_use_cost or tuple(0.0 for _ in range(model.n_bands)),
            switch_cost=switch_cost,
        )
        current = -1
        for snr_per_band in snr_by_m:
            band = ctrl.select(snr_per_band, current)
            current = band
            chosen.append(band)
            snrs.append(snr_per_band[band])
            losses.append(loss_from_snr(snr_per_band[band]))
    else:
        fixed = best_average_band(snr_by_m)
        for snr_per_band in snr_by_m:
            chosen.append(fixed)
            snrs.append(snr_per_band[fixed])
            losses.append(loss_from_snr(snr_per_band[fixed]))

    mean_snr = sum(snrs) / len(snrs) if snrs else 0.0
    return BackhaulPlan(
        loss_schedule=losses,
        chosen_bands=chosen,
        mean_chosen_snr_db=mean_snr,
        adaptive=adaptive,
    )
