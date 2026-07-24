"""L1 RF-adaptive channel controller — the deterministic U(c,t) utility.

The SEC26 code audit found that ``hermes/l1`` shipped an inference-only
``ChannelDDQN`` and a read-only ``RFPriorStore`` but **not** the paper's
deterministic HERMES-Heuristic controller (§III-A Eq. 1). This module
implements it:

    U(c, t) = R(gamma1(t) + g(c)) - kappa(c) - lambda(c, t)

where ``R(.)`` is the rate-tier mapping, ``gamma1(t)`` the baseline SNR
observation at time ``t``, ``g(c)`` the channel effective-SNR gain,
``kappa(c)`` the channel-use cost, and ``lambda(c, t)`` the switching cost
incurred when the selected channel differs from the current one. The
controller changes channels only when the expected communication benefit
exceeds the switching overhead — exactly the paper's design intent.

It is deterministic, causal (uses only the current/observed SNR, never the
future), and lightweight — the "straightforward to certify for embedded
execution" property §III-A claims. It is the L1 policy that arm H3 of
Experiment 4 exercises; the DDQN variant remains future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


def rate_tier(effective_snr_db: float) -> float:
    """Map an effective SNR (dB) to an achievable-rate tier (monotone).

    A coarse Shannon-like staircase: no throughput below ~0 dB, then
    increasing tiers. Monotone in SNR so a better channel never scores
    lower; the exact breakpoints are not load-bearing (the controller
    only compares tiers across candidate channels).
    """
    # ~log2(1 + SNR_linear) capped into tiers; cheap and monotone.
    if effective_snr_db <= 0.0:
        return 0.0
    # dB -> a smooth increasing tier value.
    import math
    return math.log2(1.0 + 10.0 ** (effective_snr_db / 10.0))


@dataclass
class AdaptiveChannelController:
    """HERMES-Heuristic — pick the channel maximising U(c, t).

    ``channel_use_cost[c]`` (kappa) discourages unconditionally selecting the
    highest-gain channel; ``switch_cost`` (lambda) is charged when the chosen
    channel differs from the current one, preventing oscillation on
    short-term SNR fluctuation. ``current_band = -1`` means "no band held
    yet" (first decision, no switch charged).
    """

    channel_use_cost: Tuple[float, ...] = (0.0, 0.0, 0.0)
    switch_cost: float = 0.5

    def utility(self, effective_snr_db: float, band: int, current_band: int) -> float:
        switch = self.switch_cost if (current_band >= 0 and band != current_band) else 0.0
        kappa = self.channel_use_cost[band] if band < len(self.channel_use_cost) else 0.0
        return rate_tier(effective_snr_db) - kappa - switch

    def select(self, snr_per_band: Sequence[float], current_band: int) -> int:
        """Return the band maximising U(c, t) given per-band effective SNR.

        Ties keep the current band (no gratuitous switch). Deterministic.
        """
        best_band = current_band if current_band >= 0 else 0
        best_u = float("-inf")
        # Evaluate the incumbent first so ties favour staying put.
        order: List[int] = []
        if current_band >= 0:
            order.append(current_band)
        order.extend(b for b in range(len(snr_per_band)) if b != current_band)
        for band in order:
            u = self.utility(float(snr_per_band[band]), band, current_band)
            if u > best_u:
                best_u = u
                best_band = band
        return best_band


def best_average_band(snr_by_mission: Sequence[Sequence[float]]) -> int:
    """The single fixed band a deployer would pick *without* real-time
    adaptation — the one with the best mean SNR over the trace.

    This is the fair static baseline (Exp 2's "Expected fixed", not the
    retrospective per-instant oracle): it uses historical averages but
    cannot track which band is best *right now*. Arm H2 uses this; arm H3's
    advantage is exactly the value of real-time adaptation over it.
    """
    if not snr_by_mission:
        return 0
    n_bands = len(snr_by_mission[0])
    means = [
        sum(m[b] for m in snr_by_mission) / len(snr_by_mission)
        for b in range(n_bands)
    ]
    return int(max(range(n_bands), key=lambda b: means[b]))
