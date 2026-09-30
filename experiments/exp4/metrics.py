"""Experiment-4 federation-side metrics (chunk EX-4.0).

These are the metrics computable *today* from the real orchestrator's
event stream — the "E5 ✔ set" of the design doc. Every definition is
reused **verbatim** from :mod:`experiments.exp3.metrics` (the metric
functions are pure roll-ups over a per-round / per-device log, so they
apply to any source that emits that accounting, sim or real). Only the
*source signal* changes: here it comes from a real
:class:`~experiments.exp4.events_consumer.Exp4Observation`, not the
abstracted sim.

Deferred to later chunks (documented, not silently dropped):

* **Communication metrics** (Bpw, Ttx, η) — need transport-level byte
  instrumentation; land with the shaped-radio work in EX-4.2.
* **Convergence / accuracy / T@τ** — need the real DNN-IDS in the loop
  (EX-4.1). The stub trainer produces no meaningful accuracy.
* **Propulsion energy** — needs a path-length integrator + a reconciled
  ``[exp4]`` calibration table (EX-4.3).

Metric semantics note for EX-4.0: the loopback orchestrator does not yet
enforce a mission deadline, so ``round_close_rate`` here measures the
*quorum* hit-rate among rounds that completed (``deadline_met`` is True
for every completed mission). Real deadline semantics arrive with the
shaped link in EX-4.2. One mule mission == one FL round (with the
cluster's ``min_participation=1``, each Pass-1 dock closes exactly one
cluster round).

With several mules (FeRRy Phase 2) each mission is still one round log, but
missions are told apart by ``(mule_id, mission_round)``, since every mule
numbers its own from 1, and a mission's slice-relative figures (Pass-2
coverage, the quorum target when Pass 1 scheduled nobody) are taken against
its own mule's slice rather than the whole population. With one mule the
slice is every device, and nothing changes. ``t_at_tau_round`` still counts
cluster rounds, about K per mission period with K mules; the trace scorer's
``missions_to_τ`` is the unit that compares across fleet sizes.

FeRRy Phase 3 — the simulated mission clock (``Exp4Observation.mission_clock
== "sim"``, critic B13). ``mission_duration_s_mean`` stays what it always
was, the wall time of a mission, which on the mission clock measures host
compute and TTL waits, not flight. Its simulated equivalent is
``sim_mission_duration_s_mean``, takeoff to the Pass-2 landing on the clock;
beside it, the clock's ledger per mission (transit, dwell, listen, return,
upload, turnaround, dock_wait; they sum to the duration), the SIMULATED
energy (``energy_status``; the Zeng-Xu-Zhang 2019 model, a labelled
simulation, not a measurement), the budget overrun (its mean, and the share
of budgeted missions that overran: the overrun rate the build plan's
deviation 11 asks to report) and the trial's re-plans, aborts and beacon
inserts. All of these are blank on the wall clock, and every column above
keeps its meaning there. The simulated time to τ is the trace scorer's
(``sim_s_to_τ``). On the mission clock solicits are targeted at a contact's
members, so ``coverage``, ``jains_fairness`` and ``participation_entropy``
(from the devices' serve counts) count member contacts only, where on the
wall clock they also count every non-member timing out on a broadcast (see
:mod:`experiments.exp4.events_consumer`): ferry and wall-clock rows do not
compare on those three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from experiments.exp3.metrics import (
    Exp3RoundLog,
    aggregate_round_logs,
    completion_fairness,
    coverage,
    jains_fairness,
    mission_completion_rate,
    participation_entropy,
)
from hermes.processes.config import CLOCK_SIM

from .events_consumer import Exp4Observation, MissionKey, MissionRecord, ModelEvalPoint

#: The mission clock's ledger kinds (``hermes.l1.mission_clock.LEDGER_KINDS``)
#: and the column each one's per-mission mean goes to.
SIM_LEDGER_COLUMNS = (
    ("transit", "sim_transit_s_mean"),
    ("dwell", "sim_dwell_s_mean"),
    ("listen", "sim_listen_s_mean"),
    ("return", "sim_return_s_mean"),
    ("upload", "sim_upload_s_mean"),
    ("turnaround", "sim_turnaround_s_mean"),
    ("dock_wait", "sim_dock_wait_s_mean"),
)

#: What ``energy_status`` says on the mission clock: the energy is modelled.
ENERGY_SIMULATED = "simulated"


@dataclass(frozen=True)
class Exp4MetricSummary:
    """Federation-side reportables for one (arm, cell, trial) row."""

    # Yield + round-close (quorum) rates — reused from Exp 3.
    update_yield: float
    round_close_rate_kmin1: float
    round_close_rate_kmin2: float
    round_close_rate_kminhalf: float
    round_close_rate_kminN: float
    # Coverage + fairness over the device population.
    coverage: float
    jains_fairness: float
    participation_entropy: float
    mission_completion_rate: float
    completion_fairness: float
    # Two-pass / contact-event structure. None (blank) for mule-less arms
    # such as H0 traditional FL, per the paper's A1 "N/A" convention.
    pass2_coverage: Optional[float] = None
    rho_contact: Optional[float] = None
    # Run-shape counters (integration health + denominators).
    rounds_closed: int = 0
    missions_completed: int = 0
    mission_failures: int = 0
    pass1_contacts_mean: float = 0.0
    pass2_contacts_mean: float = 0.0
    mission_duration_s_mean: float = 0.0
    # Cell echo (handy for filtering the CSV without re-parsing cell_id).
    n_devices: int = 0
    rf_range_m: float = 0.0
    n_missions_target: int = 0

    # EX-4.1 — real-model convergence on the held-out set. None/0 on the
    # EX-4.0 stub path (no ``model_eval`` events). ``init_*`` is the seeded
    # baseline (round 0); ``final_*`` is the last aggregated model;
    # ``t_at_tau_round`` is the first round to reach accuracy >= ``tau``.
    init_auc: Optional[float] = None
    init_accuracy: Optional[float] = None
    init_loss: Optional[float] = None
    final_auc: Optional[float] = None
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    best_auc: Optional[float] = None
    delta_auc: Optional[float] = None
    rounds_evaluated: int = 0
    t_at_tau_round: Optional[int] = None
    tau: Optional[float] = None

    # FeRRy Phase 3 — the mission clock (module docstring). None (blank) on
    # the wall clock. Per-mission means over the trial's missions, in
    # simulated seconds; the counts are the trial's totals.
    sim_mission_duration_s_mean: Optional[float] = None
    sim_transit_s_mean: Optional[float] = None
    sim_dwell_s_mean: Optional[float] = None
    sim_listen_s_mean: Optional[float] = None
    sim_return_s_mean: Optional[float] = None
    sim_upload_s_mean: Optional[float] = None
    sim_turnaround_s_mean: Optional[float] = None
    sim_dock_wait_s_mean: Optional[float] = None
    sim_energy_j_mean: Optional[float] = None       # SIMULATED (energy_status)
    energy_status: Optional[str] = None
    #: Over the missions flown with a budget; blank without one.
    sim_budget_overrun_s_mean: Optional[float] = None
    sim_budget_overrun_rate: Optional[float] = None
    sim_replans: Optional[int] = None
    sim_aborts: Optional[int] = None
    sim_inserts: Optional[int] = None

    def to_row(self) -> Dict[str, object]:
        return {
            "update_yield": self.update_yield,
            "round_close_rate_kmin1": self.round_close_rate_kmin1,
            "round_close_rate_kmin2": self.round_close_rate_kmin2,
            "round_close_rate_kminhalf": self.round_close_rate_kminhalf,
            "round_close_rate_kminN": self.round_close_rate_kminN,
            "coverage": self.coverage,
            "jains_fairness": self.jains_fairness,
            "participation_entropy": self.participation_entropy,
            "mission_completion_rate": self.mission_completion_rate,
            "completion_fairness": self.completion_fairness,
            "pass2_coverage": _blank(self.pass2_coverage),
            "rho_contact": _blank(self.rho_contact),
            "rounds_closed": self.rounds_closed,
            "missions_completed": self.missions_completed,
            "mission_failures": self.mission_failures,
            "pass1_contacts_mean": self.pass1_contacts_mean,
            "pass2_contacts_mean": self.pass2_contacts_mean,
            "mission_duration_s_mean": self.mission_duration_s_mean,
            "n_devices": self.n_devices,
            "rf_range_m": self.rf_range_m,
            "n_missions_target": self.n_missions_target,
            "init_auc": _blank(self.init_auc),
            "init_accuracy": _blank(self.init_accuracy),
            "init_loss": _blank(self.init_loss),
            "final_auc": _blank(self.final_auc),
            "final_accuracy": _blank(self.final_accuracy),
            "final_loss": _blank(self.final_loss),
            "best_auc": _blank(self.best_auc),
            "delta_auc": _blank(self.delta_auc),
            "rounds_evaluated": self.rounds_evaluated,
            "t_at_tau_round": _blank(self.t_at_tau_round),
            "tau": _blank(self.tau),
            **{col: _blank(getattr(self, col)) for col in SIM_COLUMNS},
        }

    @staticmethod
    def csv_columns() -> List[str]:
        return [
            "update_yield",
            "round_close_rate_kmin1",
            "round_close_rate_kmin2",
            "round_close_rate_kminhalf",
            "round_close_rate_kminN",
            "coverage",
            "jains_fairness",
            "participation_entropy",
            "mission_completion_rate",
            "completion_fairness",
            "pass2_coverage",
            "rho_contact",
            "rounds_closed",
            "missions_completed",
            "mission_failures",
            "pass1_contacts_mean",
            "pass2_contacts_mean",
            "mission_duration_s_mean",
            "n_devices",
            "rf_range_m",
            "n_missions_target",
            "init_auc",
            "init_accuracy",
            "init_loss",
            "final_auc",
            "final_accuracy",
            "final_loss",
            "best_auc",
            "delta_auc",
            "rounds_evaluated",
            "t_at_tau_round",
            "tau",
            *SIM_COLUMNS,
        ]


#: The FeRRy Phase 3 columns, in row order, after every earlier one (so each
#: earlier column keeps its place relative to the others).
SIM_COLUMNS = (
    "sim_mission_duration_s_mean",
    *(col for _kind, col in SIM_LEDGER_COLUMNS),
    "sim_energy_j_mean",
    "energy_status",
    "sim_budget_overrun_s_mean",
    "sim_budget_overrun_rate",
    "sim_replans",
    "sim_aborts",
    "sim_inserts",
)


def summarise_observation(
    obs: Exp4Observation,
    *,
    n_devices: int,
    rf_range_m: float,
    n_missions_target: int,
    tau: float = 0.9,
) -> Exp4MetricSummary:
    """Roll one trial's :class:`Exp4Observation` up to the reportables."""

    # ---- Per-round log → update yield + round-close(quorum) rates ---- #
    # FeRRy audit #3: an update the mule's merge excluded (past its age
    # cutoff) was collected but never reached the model, so it is not a
    # yield. Traces without the merged fields merged every CLEAN update.
    # Keyed by (mule, round): every mule numbers its missions from 1.
    own_updates: Dict[MissionKey, int] = {}
    for m in obs.missions:
        if m.pass_1_merged_updates is not None:
            n_up = m.pass_1_merged_updates
        elif m.pass_1_merged_devices is not None:
            n_up = len(m.pass_1_merged_devices)
        elif m.pass_1_updates is not None:
            n_up = m.pass_1_updates
        else:
            n_up = len(m.pass_1_clean_devices)
        own_updates[obs.mission_key(m)] = int(n_up)

    def _reached(key: MissionKey) -> bool:
        return (
            key not in obs.backhaul_lost_keys
            and key not in obs.expired_keys
            and key not in obs.unmerged_keys
        )

    rounds: List[Exp3RoundLog] = []
    yields: List[int] = []
    for i, m in enumerate(obs.missions):
        key = obs.mission_key(m)
        n_up = own_updates[key]
        yields.append(n_up)
        # Pass 1 scheduled nobody: the whole slice is the target, and with
        # several mules that is the mission's own mule's slice.
        n_target = (
            m.pass_1_scheduled if m.pass_1_scheduled
            else math.ceil(_slice_size(obs, m, n_devices))
        )
        # EX-4.2 honesty fix: a round "closes" only if it actually produced
        # a cross-mule aggregate at the cluster — i.e. it had >=1 update AND
        # its mule->BS backhaul upload was not dropped. Empty rounds (no
        # uplink succeeded) and backhaul-dropped rounds do NOT close, so H1's
        # jittery penalty is visible in round_close_rate (previously this was
        # hard-coded True, masking H1 non-closure). FeRRy audit #4: nor does
        # an upload the cluster buffered without merging, or one whose merge
        # expired. A FedBuff flush closes the flushing mission's round with
        # every update it releases — its own and those of the rounds it
        # flushed, other mules' included — so the quorum thresholds count
        # them all there. Under agg:plain nothing is deferred, flushed or
        # expired. With several mules, nor does an upload the cluster logged
        # but never folded (refused as a duplicate, or still waiting for its
        # quorum when the trial ended; see ``Exp4Observation.unmerged_keys``).
        own = 0 if key in obs.deferred_keys else n_up
        n_merged = own + sum(
            own_updates.get(k, 0) for k, at in obs.flush_of_keys.items()
            if at == key and _reached(k)
        )
        closed = (
            n_merged > 0
            and _reached(key)
            and key not in obs.deferred_keys
        )
        rounds.append(
            Exp3RoundLog(
                round_index=i,
                n_updates=int(n_merged),
                n_target=int(n_target),
                deadline_met=closed,
            )
        )
    _, close_by_k = aggregate_round_logs(rounds)
    # Yield stays per mission: the updates each mission's merge used,
    # whether or not the cluster later applied them (a FedBuff buffer still
    # filling at the end never does; the trace scorer's merged_total counts
    # only updates that reached θ).
    yield_mean = (sum(yields) / len(yields)) if yields else 0.0
    n_target_max = max((r.n_target for r in rounds), default=n_devices)
    k_half = max(1, n_target_max // 2)
    k_full = max(1, n_target_max)

    # ---- Coverage + fairness over the device population ---- #
    visits = dict(obs.per_device_serves)
    cov = coverage(visits, scheduled_count=n_devices)
    jf = jains_fairness(visits)
    pe = participation_entropy(visits)

    # ---- Completion counts (Pass-1 CLEAN contributions per device) ---- #
    # Sessions completed, so CLEAN rather than merged: a device that finished
    # its session did its part even if the merge later excluded the update.
    completions: Dict[str, int] = {}
    for m in obs.missions:
        for did in m.pass_1_clean_devices:
            completions[did] = completions.get(did, 0) + 1
    mcr = mission_completion_rate(completions, n_devices=n_devices)
    cf = completion_fairness(completions, n_devices=n_devices)

    # ---- Two-pass / contact structure ---- #
    # Pass 2 delivers to the mission's own mule's slice, so that is what each
    # mission's coverage is a share of; against every device it would top out
    # near 1/K with K mules.
    if obs.missions and n_devices > 0:
        pass2 = sum(
            min(1.0, (m.delivered or 0) / _slice_size(obs, m, n_devices))
            for m in obs.missions
        ) / len(obs.missions)
    else:
        pass2 = 0.0

    tot_scheduled = sum((m.pass_1_scheduled or 0) for m in obs.missions)
    tot_contacts = sum(m.pass_1_contacts for m in obs.missions)
    rho = (tot_scheduled / tot_contacts) if tot_contacts > 0 else 0.0

    p1c = _mean(m.pass_1_contacts for m in obs.missions)
    p2c = _mean(m.pass_2_contacts for m in obs.missions)
    dur = _mean(
        m.duration_s for m in obs.missions if m.duration_s is not None
    )

    conv = _convergence_from_evals(obs.model_evals, tau)

    return Exp4MetricSummary(
        update_yield=yield_mean,
        round_close_rate_kmin1=close_by_k.get(1, 0.0),
        round_close_rate_kmin2=close_by_k.get(2, 0.0),
        round_close_rate_kminhalf=close_by_k.get(k_half, 0.0),
        round_close_rate_kminN=close_by_k.get(k_full, 0.0),
        coverage=cov,
        jains_fairness=jf,
        participation_entropy=pe,
        mission_completion_rate=mcr,
        completion_fairness=cf,
        pass2_coverage=pass2,
        rho_contact=rho,
        rounds_closed=obs.cluster_rounds_closed,
        missions_completed=obs.missions_completed,
        mission_failures=obs.mission_failures,
        pass1_contacts_mean=p1c,
        pass2_contacts_mean=p2c,
        mission_duration_s_mean=dur,
        n_devices=int(n_devices),
        rf_range_m=float(rf_range_m),
        n_missions_target=int(n_missions_target),
        **conv,
        **_sim_summary(obs),
    )


def _sim_summary(obs: Exp4Observation) -> Dict[str, object]:
    """The FeRRy Phase 3 columns of a trial on the mission clock; {} on the wall clock.

    Means are per mission, over the missions that record the figure (on the
    mission clock every one does, empty missions included: an empty mission
    still flies, returns and turns around). The budget overrun counts only
    missions flown under a budget, and a mission overran when it ended past
    it by any amount. Re-plans, aborts and inserts are the trial's totals.
    """
    if obs.mission_clock != CLOCK_SIM:
        return {}
    missions = obs.missions

    def mean(values) -> Optional[float]:
        values = [float(v) for v in values if v is not None]
        return sum(values) / len(values) if values else None

    ledgers = [m.ledger() for m in missions if m.sim_ledger is not None]
    overruns = [m.budget_overrun_s for m in missions if m.budget_overrun_s is not None]
    out: Dict[str, object] = {
        "sim_mission_duration_s_mean": mean(m.sim_duration_s for m in missions),
        "sim_energy_j_mean": mean(m.energy_j for m in missions),
        "energy_status": ENERGY_SIMULATED,
        "sim_budget_overrun_s_mean": mean(overruns),
        "sim_budget_overrun_rate": (
            sum(1 for v in overruns if v > 0.0) / len(overruns) if overruns else None
        ),
        "sim_replans": sum(m.replans or 0 for m in missions),
        "sim_aborts": sum(m.aborts or 0 for m in missions),
        "sim_inserts": sum(m.inserts or 0 for m in missions),
    }
    for kind, col in SIM_LEDGER_COLUMNS:
        out[col] = mean(ledger.get(kind, 0.0) for ledger in ledgers)
    return out


def summarise_flat_fl(
    *,
    model_evals: List[ModelEvalPoint],
    round_logs: List[Exp3RoundLog],
    per_client_participation: Dict[object, int],
    n_devices: int,
    rf_range_m: float,
    n_missions_target: int,
    tau: float = 0.9,
) -> Exp4MetricSummary:
    """Roll up a traditional flat-FL (H0) trial.

    H0 has no mule, so the mule-only metrics (``pass2_coverage``,
    ``rho_contact``) are N/A (None -> blank), matching the paper's A1
    convention. The federation-side metrics and the convergence trace use
    the **same** definitions as the mule arms, so H0 and H1 rows are
    directly comparable at a paired seed.
    """
    yield_mean, close_by_k = aggregate_round_logs(round_logs)
    n_target_max = max((r.n_target for r in round_logs), default=n_devices)
    k_half = max(1, n_target_max // 2)
    k_full = max(1, n_target_max)

    visits = dict(per_client_participation)
    cov = coverage(visits, scheduled_count=n_devices)
    jf = jains_fairness(visits)
    pe = participation_entropy(visits)
    # In flat FL every sampled client contributes a completed update, so
    # completion counts == participation counts.
    mcr = mission_completion_rate(visits, n_devices=n_devices)
    cf = completion_fairness(visits, n_devices=n_devices)

    conv = _convergence_from_evals(model_evals, tau)
    return Exp4MetricSummary(
        update_yield=yield_mean,
        round_close_rate_kmin1=close_by_k.get(1, 0.0),
        round_close_rate_kmin2=close_by_k.get(2, 0.0),
        round_close_rate_kminhalf=close_by_k.get(k_half, 0.0),
        round_close_rate_kminN=close_by_k.get(k_full, 0.0),
        coverage=cov,
        jains_fairness=jf,
        participation_entropy=pe,
        mission_completion_rate=mcr,
        completion_fairness=cf,
        pass2_coverage=None,   # no Pass 2 in flat FL
        rho_contact=None,      # no contact events in flat FL
        rounds_closed=len(round_logs),
        missions_completed=0,  # no mule missions
        mission_failures=0,
        pass1_contacts_mean=0.0,
        pass2_contacts_mean=0.0,
        mission_duration_s_mean=0.0,
        n_devices=int(n_devices),
        rf_range_m=float(rf_range_m),
        n_missions_target=int(n_missions_target),
        **conv,
    )


def _convergence_from_evals(model_evals: List[ModelEvalPoint], tau: float) -> Dict[str, object]:
    """Init/final/best AUC, ΔAUC, and T@τ from a held-out eval trace.

    Shared by the mule-arm (:func:`summarise_observation`) and flat-FL
    (:func:`summarise_flat_fl`) paths so the convergence numbers mean the
    same thing across arms. All fields are None (blank) when the trace is
    empty (the EX-4.0 stub path).
    """
    if not model_evals:
        return dict(
            init_auc=None, init_accuracy=None, init_loss=None,
            final_auc=None, final_accuracy=None, final_loss=None,
            best_auc=None, delta_auc=None, rounds_evaluated=0,
            t_at_tau_round=None, tau=None,
        )
    init_e, final_e = model_evals[0], model_evals[-1]
    t_at_tau = next(
        (e.cluster_round for e in model_evals
         if e.cluster_round > 0 and e.accuracy >= tau),
        None,
    )
    return dict(
        init_auc=init_e.auc,
        init_accuracy=init_e.accuracy,
        init_loss=init_e.loss,
        final_auc=final_e.auc,
        final_accuracy=final_e.accuracy,
        final_loss=final_e.loss,
        best_auc=max(e.auc for e in model_evals),
        delta_auc=final_e.auc - init_e.auc,
        rounds_evaluated=len(model_evals),
        t_at_tau_round=t_at_tau,
        tau=float(tau),
    )


def _slice_size(obs: Exp4Observation, mission: MissionRecord, n_devices: int) -> float:
    """How many devices the mission's mule serves.

    Every device with one mule. With several, the slice its config lists,
    else an even share of the devices (the configs name the slices on every
    trace the orchestrator wrote, so the share only covers hand-built rows).
    """
    if obs.n_mules <= 1:
        return n_devices
    members = obs.mule_slices.get(obs.mule_key(mission.mule_id))
    return len(members) if members else n_devices / obs.n_mules


def _mean(xs) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _blank(v):
    """CSV cell for an optional metric — empty string when absent."""
    return v if v is not None else ""
