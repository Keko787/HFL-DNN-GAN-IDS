"""Phase 5 demo — TargetSelectorRL vs distance-placeholder.

Run with:
    python -m hermes.scheduler.selector

Implementation Plan §3 Phase 5 DoD item:

    A/B test: selector beats distance-sorted placeholder by ≥5 %
    completion rate.

This demo trains a :class:`TargetSelectorRL` on the in-process
:class:`BucketSim`, then rolls out both policies over identically-seeded
episodes and prints the head-to-head numbers. It also demonstrates the
scope-guard (principle #12) by asking the selector to score a device it
wasn't admitted for and catching the :class:`SelectorScopeViolation`.
"""

from __future__ import annotations

import sys

from hermes.types import Bucket, DeviceID

from .features import SelectorEnv
from .scope_guard import SelectorScopeViolation
from .selector_train import TrainConfig, run_ab_evaluation, train_selector


def _hr(title: str) -> None:
    print("\n=== " + title + " ===")


def run_demo() -> int:
    # ---------------------------------------------------------------- #
    # 1. Train.
    # ---------------------------------------------------------------- #
    _hr("Phase 5 demo: training TargetSelectorRL on BucketSim")
    cfg = TrainConfig(
        episodes=250,
        bucket_size=6,
        batch_size=32,
        warmup=64,
        buffer_capacity=4_000,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay_episodes=200,
        seed=0,
    )
    print(
        f"  episodes={cfg.episodes} bucket_size={cfg.bucket_size} "
        f"batch={cfg.batch_size} warmup={cfg.warmup}"
    )
    selector, metrics = train_selector(cfg=cfg)
    first = (
        sum(metrics.mean_reward_by_episode[:20]) / 20
        if metrics.mean_reward_by_episode
        else 0.0
    )
    last = (
        sum(metrics.mean_reward_by_episode[-20:]) / 20
        if metrics.mean_reward_by_episode
        else 0.0
    )
    print(f"  mean reward first-20 ep : {first:+.3f}")
    print(f"  mean reward last-20 ep  : {last:+.3f}")
    print(f"  SGD steps logged        : {len(metrics.loss_by_step)}")

    # ---------------------------------------------------------------- #
    # 2. A/B.
    # ---------------------------------------------------------------- #
    _hr("A/B: selector vs distance-placeholder (200 episodes, same seeds)")
    ab = run_ab_evaluation(selector, episodes=200, bucket_size=6, seed=1)
    print(
        "  placeholder  : "
        f"mean_reward={ab.placeholder_mean_reward:+.3f} "
        f"completion_rate={ab.placeholder_completion_rate:.3f}"
    )
    print(
        "  selector     : "
        f"mean_reward={ab.selector_mean_reward:+.3f} "
        f"completion_rate={ab.selector_completion_rate:.3f}"
    )
    lift = ab.completion_rate_lift
    print(f"  completion-rate lift    : {lift:+.2%}")
    dod_ok = lift >= 0.05
    print(f"  DoD (≥ +5 % lift)       : {'OK' if dod_ok else 'MISS'}")

    # ---------------------------------------------------------------- #
    # 3. Scope guard demo.
    # ---------------------------------------------------------------- #
    _hr("Scope guard (design §7 principle 12)")
    env = SelectorEnv(
        mule_pose=(0.0, 0.0, 0.0),
        mule_energy=1.0,
        rf_prior_snr_db=20.0,
        now=0.0,
    )
    try:
        selector.rank(
            candidates=[DeviceID("ghost")],
            device_states={},
            bucket=Bucket.NEW,
            env=env,
            admitted=[DeviceID("real-admitted")],
        )
    except SelectorScopeViolation as exc:
        print(f"  refused foreign device: {exc}")
        guard_ok = True
    else:
        print("  scope guard did NOT fire — bug")
        guard_ok = False

    _hr("Phase 5 summary")
    print(f"  trained selector                     : OK")
    print(f"  selector beats placeholder by ≥5 %   : {'OK' if dod_ok else 'MISS'}")
    print(f"  scope guard rejects foreign device   : {'OK' if guard_ok else 'MISS'}")
    return 0 if (dod_ok and guard_ok) else 1


if __name__ == "__main__":
    sys.exit(run_demo())
