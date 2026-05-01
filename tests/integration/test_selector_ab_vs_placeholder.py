"""Phase 5 DoD — multi-metric A/B between trained selector and distance placeholder.

Implementation Plan §3 Phase 5 originally framed the DoD as a single
completion-rate margin, which doesn't reflect what the selector actually
buys at the system level (energy + retry savings on a battery-powered
NUC). This test scores both policies on the four-axis scoreboard used
in paper §V.C Experiment 3:

    completion rate · energy · retry rate · per-decision compute

Pass criterion (see :meth:`ABResult.passes_dod`):

* selector matches the placeholder on completion (within tolerance), AND
* selector wins by ≥ 5 % on at least one of {energy, retry rate}, AND
* selector's per-decision compute cost stays within Z× of the placeholder
  (no "win by burning a GPU").
"""

from __future__ import annotations

import pytest

from hermes.scheduler.selector import TargetSelectorRL
from hermes.scheduler.selector.selector_train import (
    ABResult,
    PolicyScore,
    TrainConfig,
    run_ab_evaluation,
    train_selector,
)


@pytest.mark.slow
def test_selector_wins_on_at_least_one_axis():
    cfg = TrainConfig(
        episodes=500,
        bucket_size=6,
        batch_size=32,
        warmup=64,
        buffer_capacity=4_000,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay_episodes=350,
        seed=0,
    )
    selector, metrics = train_selector(cfg=cfg)
    assert len(metrics.mean_reward_by_episode) == cfg.episodes
    assert len(metrics.loss_by_step) > 0

    ab = run_ab_evaluation(
        selector, episodes=200, bucket_size=cfg.bucket_size, seed=1
    )

    assert ab.passes_dod(
        completion_tolerance=0.02,
        win_margin=0.05,
        compute_ceiling_x=50.0,
    ), (
        "Selector did not pass the multi-metric DoD.\n"
        f"  completion_lift  = {ab.completion_rate_lift:+.2%} "
        f"(placeholder={ab.placeholder.completion_rate:.3f}, "
        f"selector={ab.selector.completion_rate:.3f})\n"
        f"  energy_savings   = {ab.energy_savings:+.2%} "
        f"(placeholder={ab.placeholder.energy_per_episode:.4f}, "
        f"selector={ab.selector.energy_per_episode:.4f})\n"
        f"  retry_savings    = {ab.retry_rate_savings:+.2%} "
        f"(placeholder={ab.placeholder.retry_rate:.3f}, "
        f"selector={ab.selector.retry_rate:.3f})\n"
        f"  path_savings     = {ab.path_savings:+.2%} "
        f"(placeholder={ab.placeholder.path_length_per_episode:.2f}, "
        f"selector={ab.selector.path_length_per_episode:.2f})\n"
        f"  compute_overhead = {ab.compute_overhead_x:.2f}× "
        f"(placeholder={ab.placeholder.mean_compute_us:.2f}µs, "
        f"selector={ab.selector.mean_compute_us:.2f}µs)"
    )


@pytest.mark.slow
def test_ab_scoreboard_emits_all_axes():
    """Cheap end-to-end check that the rollout actually populates every axis.

    Untrained selector — we don't care about *winning* here, only that the
    scoreboard isn't silently zero on any field.
    """
    selector = TargetSelectorRL(rng_seed=0)
    ab = run_ab_evaluation(selector, episodes=20, bucket_size=4, seed=2)

    for arm_name, arm in (("placeholder", ab.placeholder), ("selector", ab.selector)):
        assert arm.energy_per_episode > 0.0, f"{arm_name} energy not recorded"
        assert arm.path_length_per_episode > 0.0, f"{arm_name} path not recorded"
        assert arm.decisions_per_episode > 0.0, f"{arm_name} decisions not recorded"
        # mean_compute_us can be very small but must be measured (>0).
        assert arm.mean_compute_us > 0.0, f"{arm_name} compute not recorded"


def test_ab_result_completion_rate_lift_arithmetic():
    """Unit-level sanity on ABResult comparison properties."""
    r = ABResult(
        placeholder=PolicyScore(mean_reward=-5.0, completion_rate=0.5),
        selector=PolicyScore(mean_reward=-3.0, completion_rate=0.6),
    )
    assert r.completion_rate_lift == pytest.approx(0.2)

    # Zero-baseline edge case.
    r0 = ABResult(
        placeholder=PolicyScore(),
        selector=PolicyScore(completion_rate=0.1),
    )
    assert r0.completion_rate_lift == float("inf")

    r0b = ABResult(placeholder=PolicyScore(), selector=PolicyScore())
    assert r0b.completion_rate_lift == 0.0


def test_ab_result_savings_arithmetic():
    """Energy and retry savings flip sign correctly."""
    r = ABResult(
        placeholder=PolicyScore(
            energy_per_episode=1.0, retry_rate=0.4,
        ),
        selector=PolicyScore(
            energy_per_episode=0.8, retry_rate=0.2,
        ),
    )
    # Selector uses 20% less energy.
    assert r.energy_savings == pytest.approx(0.2)
    # Selector retries half as often.
    assert r.retry_rate_savings == pytest.approx(0.5)


def test_ab_result_compute_overhead_arithmetic():
    """Compute-overhead is selector µs / placeholder µs."""
    r = ABResult(
        placeholder=PolicyScore(mean_compute_us=10.0),
        selector=PolicyScore(mean_compute_us=25.0),
    )
    assert r.compute_overhead_x == pytest.approx(2.5)

    # Zero-baseline edge case (placeholder unrealistically free).
    r0 = ABResult(
        placeholder=PolicyScore(mean_compute_us=0.0),
        selector=PolicyScore(mean_compute_us=5.0),
    )
    assert r0.compute_overhead_x == float("inf")


def test_ab_passes_dod_requires_completion_within_tolerance():
    """Selector that tanks completion fails even with a big energy win."""
    r = ABResult(
        placeholder=PolicyScore(
            completion_rate=0.5, energy_per_episode=1.0,
            retry_rate=0.5, mean_compute_us=1.0,
        ),
        selector=PolicyScore(
            completion_rate=0.30,  # ~40 % below baseline — clearly tanked
            energy_per_episode=0.1,
            retry_rate=0.7,
            mean_compute_us=10.0,
        ),
    )
    assert not r.passes_dod()


def test_ab_passes_dod_with_energy_win():
    """Tied completion + ≥ 5 % energy win → pass."""
    r = ABResult(
        placeholder=PolicyScore(
            completion_rate=0.50, energy_per_episode=1.0,
            retry_rate=0.5, mean_compute_us=1.0,
        ),
        selector=PolicyScore(
            completion_rate=0.495, energy_per_episode=0.90,
            retry_rate=0.51, mean_compute_us=10.0,
        ),
    )
    assert r.passes_dod()


def test_ab_passes_dod_compute_ceiling_blocks_pyrrhic_win():
    """A win bought with absurd compute is rejected."""
    r = ABResult(
        placeholder=PolicyScore(
            completion_rate=0.50, energy_per_episode=1.0,
            retry_rate=0.5, mean_compute_us=1.0,
        ),
        selector=PolicyScore(
            completion_rate=0.50, energy_per_episode=0.5,
            retry_rate=0.25, mean_compute_us=10_000.0,  # 10 000×
        ),
    )
    assert not r.passes_dod(compute_ceiling_x=50.0)
