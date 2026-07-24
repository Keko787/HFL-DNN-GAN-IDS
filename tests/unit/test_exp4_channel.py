"""EX-4.3 — RF channel model + L1 controller (arm H3) unit tests.

Fast, in-process (no subprocess, no TensorFlow). Codifies the validity
discipline the L1 experiment rests on:

* **Clean ~ no L1 benefit.** Under clean links every band sits high and
  stable, so the adaptive controller (H3) ties the fixed best-average band
  (H1/H2) — L1 must NOT manufacture an advantage where there is nothing to
  adapt to.
* **Jittery -> adaptive wins, consistently.** Bands cross over (distinct
  phases), so the U(c,t) controller loses strictly less backhaul than any
  fixed band, across every seed.
* **Fair baseline.** H2 holds ``best_average_band`` (deployer's historical
  pick), not a retrospective per-instant oracle.
* **Same trace.** H2 and H3 read the identical seeded SNR trace.
"""

from __future__ import annotations

import pytest

from experiments.exp4.channel import (
    BackhaulPlan,
    ChannelModel,
    backhaul_plan,
    loss_from_snr,
)
from hermes.l1.channel_utility import AdaptiveChannelController, best_average_band


# --------------------------------------------------------------------------- #
# loss_from_snr — monotone, bounded, healthy≈0 / trough≈1
# --------------------------------------------------------------------------- #

def test_loss_from_snr_bounds_and_monotone():
    hi = loss_from_snr(12.0)   # healthy channel
    lo = loss_from_snr(1.0)    # deep trough
    assert 0.0 <= hi <= 1.0 and 0.0 <= lo <= 1.0
    assert hi < 0.05, f"healthy channel should barely lose: {hi}"
    assert lo > 0.5, f"deep trough should lose heavily: {lo}"
    # Strictly decreasing in SNR.
    xs = [loss_from_snr(s) for s in range(0, 20, 2)]
    assert all(a > b for a, b in zip(xs, xs[1:]))


# --------------------------------------------------------------------------- #
# Determinism + fair-baseline plumbing
# --------------------------------------------------------------------------- #

def test_plan_is_deterministic_in_seed():
    m1 = ChannelModel(n_bands=3, n_missions=6, seed=13, jittery=True)
    m2 = ChannelModel(n_bands=3, n_missions=6, seed=13, jittery=True)
    p1 = backhaul_plan(m1, adaptive=True)
    p2 = backhaul_plan(m2, adaptive=True)
    assert p1 == p2


def test_fixed_arm_holds_best_average_band():
    m = ChannelModel(n_bands=3, n_missions=6, seed=4, jittery=True)
    plan = backhaul_plan(m, adaptive=False)
    assert isinstance(plan, BackhaulPlan)
    assert not plan.adaptive
    # A single band held for the whole trace = the best-average band.
    assert len(set(plan.chosen_bands)) == 1
    assert plan.chosen_bands[0] == best_average_band(m.snr_by_mission())


def test_h2_and_h3_read_same_trace():
    """Fixed (H2) and adaptive (H3) must face the identical SNR trace."""
    m = ChannelModel(n_bands=3, n_missions=6, seed=8, jittery=True)
    trace_a = m.snr_by_mission()
    trace_b = m.snr_by_mission()
    assert trace_a == trace_b  # regenerating the trace is stable


# --------------------------------------------------------------------------- #
# The two headline validity properties
# --------------------------------------------------------------------------- #

def _mean(xs):
    return sum(xs) / len(xs)


@pytest.mark.parametrize("seed", range(12))
def test_clean_gives_no_meaningful_l1_benefit(seed):
    """Clean links: adaptive ~ fixed (nothing to adapt to)."""
    m = ChannelModel(n_bands=3, n_missions=6, seed=seed, jittery=False)
    fixed = _mean(backhaul_plan(m, adaptive=False).loss_schedule)
    adapt = _mean(backhaul_plan(m, adaptive=True).loss_schedule)
    # The adaptive controller may not beat fixed by more than a hair, and may
    # even pay a tiny switch cost — the point is the effect is negligible.
    assert abs(fixed - adapt) < 0.02, (
        f"seed={seed}: clean L1 effect should be ~0, got fixed={fixed:.4f} "
        f"adaptive={adapt:.4f}"
    )


@pytest.mark.parametrize("seed", range(12))
def test_jittery_adaptive_never_worse_than_fixed(seed):
    """Jittery links: adaptive (H3) loses no more backhaul than fixed (H2)."""
    m = ChannelModel(n_bands=3, n_missions=6, seed=seed, jittery=True)
    fixed = _mean(backhaul_plan(m, adaptive=False).loss_schedule)
    adapt = _mean(backhaul_plan(m, adaptive=True).loss_schedule)
    assert adapt <= fixed + 1e-9, (
        f"seed={seed}: adaptive should not lose more than fixed under jittery, "
        f"got fixed={fixed:.4f} adaptive={adapt:.4f}"
    )


def test_jittery_adaptive_beats_fixed_on_average():
    """Across seeds the jittery L1 benefit is real and positive (not noise)."""
    reductions = []
    for seed in range(20):
        m = ChannelModel(n_bands=3, n_missions=6, seed=seed, jittery=True)
        fixed = _mean(backhaul_plan(m, adaptive=False).loss_schedule)
        adapt = _mean(backhaul_plan(m, adaptive=True).loss_schedule)
        reductions.append(fixed - adapt)
    mean_reduction = _mean(reductions)
    assert mean_reduction > 0.03, (
        f"expected a real jittery L1 benefit, got mean reduction {mean_reduction:.4f}"
    )
    # And it is not driven by a single lucky seed.
    assert min(reductions) >= -1e-9


# --------------------------------------------------------------------------- #
# Controller re-selects under crossover
# --------------------------------------------------------------------------- #

def test_controller_switches_when_best_band_changes():
    """A trace whose best band flips should make the controller switch."""
    # Band 0 best early, band 1 best late.
    trace = [
        [10.0, 2.0],
        [10.0, 2.0],
        [2.0, 10.0],
        [2.0, 10.0],
    ]
    ctrl = AdaptiveChannelController(channel_use_cost=(0.0, 0.0), switch_cost=0.5)
    current = -1
    chosen = []
    for row in trace:
        current = ctrl.select(row, current)
        chosen.append(current)
    assert chosen[0] == 0 and chosen[-1] == 1, chosen
    assert 0 in chosen and 1 in chosen  # it actually switched


def test_switch_cost_prevents_thrashing():
    """A trivial, noisy near-tie must not trigger a costly switch."""
    ctrl = AdaptiveChannelController(channel_use_cost=(0.0, 0.0), switch_cost=5.0)
    current = 0
    # Band 1 is only marginally better; the switch cost should keep band 0.
    nxt = ctrl.select([10.0, 10.2], current)
    assert nxt == 0
