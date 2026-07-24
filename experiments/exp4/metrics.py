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
"""

from __future__ import annotations

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

from .events_consumer import Exp4Observation, ModelEvalPoint


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
        ]


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
    rounds: List[Exp3RoundLog] = []
    for i, m in enumerate(obs.missions):
        if m.pass_1_updates is not None:
            n_up = m.pass_1_updates
        else:
            n_up = len(m.pass_1_clean_devices)
        n_target = m.pass_1_scheduled if m.pass_1_scheduled else n_devices
        rounds.append(
            Exp3RoundLog(
                round_index=i,
                n_updates=int(n_up),
                n_target=int(n_target),
                # No deadline model in the loopback stack yet — a
                # completed mission is a closed round (EX-4.2 replaces
                # this with the shaped-link deadline).
                deadline_met=True,
            )
        )
    yield_mean, close_by_k = aggregate_round_logs(rounds)
    n_target_max = max((r.n_target for r in rounds), default=n_devices)
    k_half = max(1, n_target_max // 2)
    k_full = max(1, n_target_max)

    # ---- Coverage + fairness over the device population ---- #
    visits = dict(obs.per_device_serves)
    cov = coverage(visits, scheduled_count=n_devices)
    jf = jains_fairness(visits)
    pe = participation_entropy(visits)

    # ---- Completion counts (Pass-1 CLEAN contributions per device) ---- #
    completions: Dict[str, int] = {}
    for m in obs.missions:
        for did in m.pass_1_clean_devices:
            completions[did] = completions.get(did, 0) + 1
    mcr = mission_completion_rate(completions, n_devices=n_devices)
    cf = completion_fairness(completions, n_devices=n_devices)

    # ---- Two-pass / contact structure ---- #
    if obs.missions and n_devices > 0:
        pass2 = sum(
            min(1.0, (m.delivered or 0) / n_devices) for m in obs.missions
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
    )


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


def _mean(xs) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _blank(v):
    """CSV cell for an optional metric — empty string when absent."""
    return v if v is not None else ""
