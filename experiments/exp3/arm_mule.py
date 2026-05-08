"""Shared mule-arm driver for A2 / A3 / A4.

Wraps :class:`~experiments.exp3.sim_env.Exp3Sim` and a
ranking-policy callable with the ``rank_contacts(...) -> List[ContactWaypoint]``
shape — the API both A2/A3 (in :mod:`hermes.scheduler.policies`) and
A4 (:meth:`hermes.scheduler.selector.TargetSelectorRL.rank_contacts`)
already expose.

One driver, three arms — what changes between calls is only the
``policy`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from hermes.types import ContactWaypoint, DeviceID, MissionPass

from experiments.calibration import Exp3Calibration

from .metrics import Exp3MetricSummary, Exp3RoundLog, summarise_trial
from .sim_env import Exp3Sim, Exp3SimConfig


# --------------------------------------------------------------------------- #
# Policy protocol — what every mule arm exposes
# --------------------------------------------------------------------------- #

class ContactRankingPolicy(Protocol):
    """Subset of the selector / policy surface this driver invokes."""

    def rank_contacts(  # pragma: no cover - protocol
        self,
        candidates: Sequence[ContactWaypoint],
        device_states,
        env,
        *,
        pass_kind: MissionPass = MissionPass.COLLECT,
        admitted=None,
    ) -> List[ContactWaypoint]: ...


# --------------------------------------------------------------------------- #
# Per-trial driver
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MuleArmConfig:
    """Trial-level config the driver passes to :class:`Exp3Sim`.

    ``arm_name`` is informational only — it's the column the CSV
    writes; the policy itself is what actually drives behaviour.
    """

    arm_name: str
    sim: Exp3SimConfig
    n_rounds: int = 1  # one mission = one round in the contact-event view


def run_mule_trial(
    *,
    cfg: MuleArmConfig,
    policy: ContactRankingPolicy,
    cal: Optional[Exp3Calibration] = None,
) -> Exp3MetricSummary:
    """Run one mule trial as a faithful two-pass mission.

    Mirrors the §V mission lifecycle in
    HERMES_FL_Scheduler_Design.md:

    * **Pass 1 (COLLECT)** — selector-driven walk over admitted-device
      contacts. Each step exchanges θ↔Δθ via :meth:`Exp3Sim.step`
      (transit + collect + per-contact mule→BS upload).
    * **Inter-pass dock** — :meth:`Exp3Sim.dock_at_bs` flies the mule
      back to the nearest BS, charges ``dock_time_s`` for the UP/DOWN
      bundle exchange, and updates the mule pose to the BS.
    * **Pass 2 (DELIVER)** — greedy nearest-first walk over every
      original contact (selector bypassed per design.md:148), pushing
      ``θ_disc'`` via :meth:`Exp3Sim.step_deliver`. Stops when the
      contact list exhausts or residual budget can't cover one more
      delivery.

    **Round semantics.** One mule mission produces one FL aggregation
    cycle (the cluster's cross-mule FedAvg at the dock folds the
    mission_aggregate into the global θ exactly once per mission).
    This driver therefore emits a single ``Exp3RoundLog`` per mission
    with ``n_updates`` summed across every visited contact and
    ``n_target = n_devices`` (the admitted slice, not the visited
    cluster sizes). That puts ``update_yield`` and ``round_close_rate``
    on the same denominator as A1's per-FedAvg-round metrics, so
    cross-arm comparisons on those panels are now apples-to-apples.
    """
    sim = Exp3Sim(cfg.sim)
    sim.reset()

    rounds: List[Exp3RoundLog] = []
    n_devices = cfg.sim.n_devices
    # Mission-level accumulators. The mule's per-contact stops are
    # *not* FL aggregation cycles — they are sub-events within Pass 1
    # that fold into a single ``mission_aggregate``. The cluster's
    # cross-mule FedAvg at the dock then folds that mission_aggregate
    # into the global θ once per mission, which IS the FL round.
    # See HERMES_FL_Scheduler_Design.md §V — "one mission ≡ one FL
    # aggregation cycle." This driver therefore emits exactly one
    # ``Exp3RoundLog`` per mission, summing completions across every
    # visited contact, with ``n_target = n_devices`` (the slice's
    # admitted count, not just the cluster sizes the mule managed to
    # visit). That keeps ``update_yield`` and ``round_close_rate`` on
    # the same denominator as A1's per-FedAvg-round metrics.
    mission_completions = 0

    # ------------------------------------------------------------------ #
    # Pass 1 — collect Δθ via selector-driven contact walk
    # ------------------------------------------------------------------ #
    while not sim.done:
        candidates = sim.candidates()
        if not candidates:
            break
        env = sim.selector_env()
        device_states = sim.device_states()
        admitted = list(device_states.keys())
        ranked = policy.rank_contacts(
            candidates,
            device_states,
            env,
            pass_kind=MissionPass.COLLECT,
            admitted=admitted,
        )
        if not ranked:
            # Policy declared nothing feasible. End Pass 1.
            break
        chosen = ranked[0]
        result = sim.step(chosen)
        mission_completions += result.completed_count

    # Emit the mission as ONE round.  ``deadline_met = True`` because
    # the mission itself ran to its scheduled close — per-device
    # readiness deadlines are already accounted for in ``mission_completions``
    # via the deadline gate inside ``Exp3Sim.step``. With round = mission,
    # ghost padding for unvisited contacts is no longer needed: the
    # denominator is now ``n_devices`` (the admitted slice), not the
    # number of visited clusters, so a strategy that truncates Pass 1
    # naturally reports a lower per-round count without survivorship
    # bias.
    rounds.append(Exp3RoundLog(
        round_index=0,
        n_updates=mission_completions,
        n_target=n_devices,
        deadline_met=True,
    ))

    # ------------------------------------------------------------------ #
    # Inter-pass dock — UP/DOWN bundle exchange at nearest BS
    # ------------------------------------------------------------------ #
    dock_result = sim.dock_at_bs()

    # ------------------------------------------------------------------ #
    # Pass 2 — greedy nearest-first delivery walk
    # ------------------------------------------------------------------ #
    if dock_result.success:
        from .sim_env import _euclid

        remaining: List[ContactWaypoint] = list(sim.pass2_candidates())
        # Residual-budget guard: a delivery's lower-bound cost is just
        # the session (transit ≥ 0). If even that doesn't fit, stop.
        while remaining and sim.budget_remaining > cfg.sim.delivery_session_s:
            current = sim.mule_pose
            # Greedy nearest-first from the current pose.
            remaining.sort(key=lambda c: _euclid(current, c.position))
            chosen = remaining[0]
            transit = (
                _euclid(current, chosen.position) / cfg.sim.cruise_speed_m_s
            )
            cost = transit + cfg.sim.delivery_session_s
            if cost > sim.budget_remaining:
                break
            sim.step_deliver(chosen)
            remaining.pop(0)

    summary = summarise_trial(
        rounds=rounds,
        metrics=sim.episode_metrics,
        cal=cal,
        n_devices=n_devices,
        is_mule_arm=True,
    )
    return summary
