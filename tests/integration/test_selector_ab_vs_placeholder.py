"""Phase 5 DoD — trained selector beats the distance placeholder.

Implementation Plan §3 Phase 5 Definition of Done:

    A/B test: selector beats distance-sorted placeholder by ≥5 %
    completion rate on the sim.

We train for a modest number of episodes (enough to converge on the tiny
BucketSim; full AERPAW training is Phase 6) and evaluate both policies
on identically-seeded rollouts.
"""

from __future__ import annotations

import pytest

from hermes.scheduler.selector import TargetSelectorRL
from hermes.scheduler.selector.selector_train import (
    TrainConfig,
    run_ab_evaluation,
    train_selector,
)


@pytest.mark.slow
def test_selector_beats_distance_placeholder_by_at_least_5pct():
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

    # DoD: ≥5 % lift over placeholder completion rate.
    # (If the placeholder already hits ~0 we let the lift property report +inf.)
    assert ab.completion_rate_lift >= 0.05, (
        f"Selector only achieved {ab.completion_rate_lift:.2%} lift "
        f"(placeholder={ab.placeholder_completion_rate:.3f}, "
        f"selector={ab.selector_completion_rate:.3f})"
    )


def test_ab_result_completion_rate_lift_arithmetic():
    """Unit-level sanity on the ABResult.completion_rate_lift property."""
    from hermes.scheduler.selector.selector_train import ABResult

    r = ABResult(
        placeholder_mean_reward=-5.0,
        placeholder_completion_rate=0.5,
        selector_mean_reward=-3.0,
        selector_completion_rate=0.6,
    )
    assert r.completion_rate_lift == pytest.approx(0.2)

    # Zero-baseline edge case.
    r0 = ABResult(0.0, 0.0, 0.0, 0.1)
    assert r0.completion_rate_lift == float("inf")

    r0b = ABResult(0.0, 0.0, 0.0, 0.0)
    assert r0b.completion_rate_lift == 0.0
