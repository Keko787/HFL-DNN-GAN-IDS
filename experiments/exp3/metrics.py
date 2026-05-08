"""Experiment-3 metrics — the 9 reportables in §IV-D of the paper.

Five derive directly from the per-step accounting in
:class:`~experiments.exp3.sim_env.Exp3EpisodeMetrics`:

* **Round close rate** at multiple ``kmin`` thresholds — fraction of
  rounds hitting at least ``kmin`` aggregated updates within deadline.
* **Update yield** — mean count of updates aggregated per round.
* **Coverage** — fraction of scheduled devices serviced ≥ once.
* **ρ_contact** — Σ |c.devices| / |contacts| (mean devices per
  Pass-1 contact event).
* **Pass-2 coverage** — fraction of devices reached by the second
  greedy pass (delivered fresh weights).

Three are derived from per-device service distributions:

* **Jain's fairness index** — ``J = (Σx)² / (N · Σx²)`` on per-device
  service counts.
* **Participation entropy** — Shannon entropy of the per-device
  service share distribution.

One is the post-hoc energy proxy from the calibration TOML:

* **Propulsion energy (Eq. 5)** —
  ``E = T_mission · P_idle + B_tx · ε_bit + L_path · ε_prop``.

Each function below is a pure roll-up over an
:class:`~experiments.exp3.sim_env.Exp3EpisodeMetrics` instance (or the
per-trial CSV row form) so the analysis notebook can apply them
post-hoc without re-running the sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from experiments.calibration import (
    Exp3Calibration,
    Exp3EnergyDecomposition,
    exp3_energy_proxy,
)

from .sim_env import Exp3EpisodeMetrics


# --------------------------------------------------------------------------- #
# Per-trial summary — one row per (arm, cell, trial)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Exp3MetricSummary:
    """All 9 reportables for one (arm, cell, trial) row.

    The driver writes this dict into the trial-grid CSV; the analysis
    module reads it back. Mule-less arms (A1) emit the mule-only fields
    as None — the CSV layer encodes them as the empty string.
    """

    # Federation-side metrics — all four arms.
    update_yield: float
    coverage: float
    jains_fairness: float
    participation_entropy: float
    round_close_rate_kmin1: float
    round_close_rate_kminhalf: float
    round_close_rate_kminN: float
    # Round-count-invariant yield: fraction of admitted devices with
    # ≥1 completed Δθ contribution to the mission. Bounded [0, 1] for
    # both A1 and the mule arms — unaffected by the survivorship bias
    # that distorts ``update_yield`` when upload pressure truncates
    # Pass 1 to fewer rounds. See ``mission_completion_rate`` in
    # ``metrics.py`` for the docstring.
    mission_completion_rate: float

    # Mule-only — A1 emits None.
    rho_contact: Optional[float]
    pass2_coverage: Optional[float]
    propulsion_energy_J: Optional[float]
    propulsion_idle_J: Optional[float]
    propulsion_tx_J: Optional[float]
    propulsion_prop_J: Optional[float]
    mission_completion_s: Optional[float]
    path_length_m: Optional[float]

    def to_row(self) -> Dict[str, object]:
        """Flatten to a dict the CSVTrialLog can write."""
        out: Dict[str, object] = {
            "update_yield": self.update_yield,
            "coverage": self.coverage,
            "jains_fairness": self.jains_fairness,
            "participation_entropy": self.participation_entropy,
            "round_close_rate_kmin1": self.round_close_rate_kmin1,
            "round_close_rate_kminhalf": self.round_close_rate_kminhalf,
            "round_close_rate_kminN": self.round_close_rate_kminN,
            "mission_completion_rate": self.mission_completion_rate,
        }
        for k, v in (
            ("rho_contact", self.rho_contact),
            ("pass2_coverage", self.pass2_coverage),
            ("propulsion_energy_J", self.propulsion_energy_J),
            ("propulsion_idle_J", self.propulsion_idle_J),
            ("propulsion_tx_J", self.propulsion_tx_J),
            ("propulsion_prop_J", self.propulsion_prop_J),
            ("mission_completion_s", self.mission_completion_s),
            ("path_length_m", self.path_length_m),
        ):
            out[k] = v if v is not None else ""
        return out

    @staticmethod
    def csv_columns() -> List[str]:
        return [
            "update_yield",
            "coverage",
            "jains_fairness",
            "participation_entropy",
            "round_close_rate_kmin1",
            "round_close_rate_kminhalf",
            "round_close_rate_kminN",
            "mission_completion_rate",
            "rho_contact",
            "pass2_coverage",
            "propulsion_energy_J",
            "propulsion_idle_J",
            "propulsion_tx_J",
            "propulsion_prop_J",
            "mission_completion_s",
            "path_length_m",
        ]


# --------------------------------------------------------------------------- #
# Per-round log — feeds round-close-rate at variable kmin
# --------------------------------------------------------------------------- #

@dataclass
class Exp3RoundLog:
    """Per-round update count + deadline-met flag.

    Both A1 (centralized) and the mule arms produce one of these per
    aggregation round; the analysis aggregates them via
    :func:`round_close_rate` at the requested ``kmin`` threshold.
    """

    round_index: int
    n_updates: int
    n_target: int
    deadline_met: bool


def aggregate_round_logs(
    rounds: Sequence[Exp3RoundLog],
) -> Tuple[float, Dict[int, float]]:
    """Compute (update_yield, {kmin: round_close_rate}) over a trial.

    Returns the trial-mean update-yield + a dict mapping the three
    canonical kmin thresholds (1, ⌈N/2⌉, N) to their round-close-rate.
    """
    if not rounds:
        return 0.0, {}
    yield_mean = sum(r.n_updates for r in rounds) / len(rounds)

    n_target = max((r.n_target for r in rounds), default=0)
    thresholds = (1, max(1, n_target // 2), max(1, n_target))
    out: Dict[int, float] = {}
    for k in thresholds:
        hits = sum(1 for r in rounds if r.deadline_met and r.n_updates >= k)
        out[k] = hits / len(rounds)
    return yield_mean, out


# --------------------------------------------------------------------------- #
# Federation-side metrics
# --------------------------------------------------------------------------- #

def update_yield(rounds: Sequence[Exp3RoundLog]) -> float:
    """Mean count of updates aggregated per round."""
    if not rounds:
        return 0.0
    return sum(r.n_updates for r in rounds) / len(rounds)


def round_close_rate(
    rounds: Sequence[Exp3RoundLog], *, kmin: int,
) -> float:
    """Fraction of rounds with ≥ ``kmin`` updates within deadline.

    ``kmin`` thresholds the paper reports: 1, ⌈N/2⌉, N. The metric
    rewards a scheduler that consistently hits the threshold rather
    than averaging over a few high-yield + many empty rounds.
    """
    if kmin < 1:
        raise ValueError(f"kmin must be >= 1, got {kmin}")
    if not rounds:
        return 0.0
    hits = sum(1 for r in rounds if r.deadline_met and r.n_updates >= kmin)
    return hits / len(rounds)


def coverage(
    per_device_visits: Mapping[object, int], *, scheduled_count: Optional[int] = None,
) -> float:
    """Fraction of scheduled devices serviced at least once.

    ``scheduled_count`` defaults to ``len(per_device_visits)`` — if the
    caller has additional devices that were scheduled but never even
    appeared in the per-device map, pass the larger total to keep the
    denominator honest.
    """
    n_total = scheduled_count if scheduled_count is not None else len(per_device_visits)
    if n_total <= 0:
        return 0.0
    n_seen = sum(1 for v in per_device_visits.values() if v > 0)
    return n_seen / n_total


def mission_completion_rate(
    per_device_completions: Mapping[object, int], *, n_devices: int,
) -> float:
    """Fraction of admitted devices with ≥1 completed Δθ contribution.

    Round-count-invariant yield metric, bounded ``[0, 1]`` for every
    arm. Use this in cross-regime / cross-arm comparisons where
    ``update_yield`` (mean updates per round) is biased by survivorship:
    when upload pressure truncates Pass 1 to fewer rounds, the mean
    is taken over a non-random sample of contacts (early/dense
    clusters), and a regime that visits *fewer* contacts can post a
    *higher* per-round average even though fewer devices contribute
    overall.

    ``mission_completion_rate`` measures *what fraction of the device
    population produced any usable Δθ in this mission*, which is what
    the FL aggregator actually consumes. It is the single most honest
    one-number yield for cross-regime plots.
    """
    if n_devices <= 0:
        return 0.0
    n_completed = sum(1 for v in per_device_completions.values() if v > 0)
    return n_completed / n_devices


# --------------------------------------------------------------------------- #
# Fairness — Jain's index + participation entropy
# --------------------------------------------------------------------------- #

def jains_fairness(per_device_visits: Mapping[object, int]) -> float:
    """``J = (Σxᵢ)² / (N · Σxᵢ²)`` over per-device service counts.

    Range ``[1/N, 1]``; 1 = perfectly fair (equal counts), 1/N = one
    device hogs everything.

    Returns 1.0 for an empty input — degenerate, but better than NaN
    for a trial-grid CSV cell.
    """
    counts = list(per_device_visits.values())
    if not counts:
        return 1.0
    n = len(counts)
    s = sum(counts)
    if s == 0:
        return 1.0
    sq = sum(c * c for c in counts)
    if sq == 0:
        return 1.0
    return (s * s) / (n * sq)


def participation_entropy(per_device_visits: Mapping[object, int]) -> float:
    """Shannon entropy (in bits) of the per-device service share.

    Range ``[0, log₂(N)]``; 0 = one device gets every service, log₂(N)
    = uniform service across all devices.
    """
    counts = list(per_device_visits.values())
    if not counts:
        return 0.0
    s = sum(counts)
    if s == 0:
        return 0.0
    H = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / s
        H -= p * math.log2(p)
    return H


# --------------------------------------------------------------------------- #
# ρ_contact + Pass-2 coverage (mule arms only)
# --------------------------------------------------------------------------- #

def rho_contact(metrics: Exp3EpisodeMetrics) -> float:
    """``ρ_contact = devices_visited / contacts_visited`` (Pass 1).

    A trial that exits before visiting any contact reports 0.0 — the
    schedule didn't get to demonstrate any contact-event aggregation.
    """
    if metrics.contacts_visited > 0:
        return metrics.devices_visited / metrics.contacts_visited
    return 0.0


def pass2_coverage(metrics: Exp3EpisodeMetrics, *, n_devices: int) -> float:
    """Fraction of devices that received fresh weights in Pass 2."""
    if n_devices <= 0:
        return 0.0
    return min(1.0, metrics.pass2_devices_reached / n_devices)


# --------------------------------------------------------------------------- #
# Propulsion energy — Eq. 5 wrapper
# --------------------------------------------------------------------------- #

def propulsion_energy(
    metrics: Exp3EpisodeMetrics, cal: Exp3Calibration,
) -> Exp3EnergyDecomposition:
    """Eq. 5 applied to one trial's :class:`Exp3EpisodeMetrics`."""
    return exp3_energy_proxy(
        T_mission_s=metrics.time_total_s,
        B_tx_bytes=metrics.upload_bytes,
        L_path_m=metrics.path_length_m,
        cal=cal,
    )


# --------------------------------------------------------------------------- #
# Convenience: roll the whole trial summary at once
# --------------------------------------------------------------------------- #

def summarise_trial(
    *,
    rounds: Sequence[Exp3RoundLog],
    metrics: Optional[Exp3EpisodeMetrics],
    cal: Optional[Exp3Calibration],
    n_devices: int,
    is_mule_arm: bool,
) -> Exp3MetricSummary:
    """One-call summary used by both the A1 and the mule-arm drivers.

    ``metrics`` and ``cal`` are required for the mule arms (the
    propulsion / ρ_contact / Pass-2 reportables come from them). For
    A1 they are passed as ``None`` and the mule-only fields are set to
    ``None`` in the summary.
    """
    yield_mean, close_rate_by_k = aggregate_round_logs(rounds)
    n_target = max((r.n_target for r in rounds), default=n_devices)
    k_half = max(1, n_target // 2)
    k_full = max(1, n_target)

    if is_mule_arm:
        if metrics is None:
            raise ValueError("mule-arm summary requires metrics")
        device_visits = metrics.per_device_visits
        cov = coverage(device_visits, scheduled_count=n_devices)
        jf = jains_fairness(device_visits)
        pe = participation_entropy(device_visits)
        mcr = mission_completion_rate(
            metrics.per_device_completions, n_devices=n_devices,
        )
        rc = (
            metrics.devices_visited / metrics.contacts_visited
            if metrics.contacts_visited > 0
            else 0.0
        )
        p2 = pass2_coverage(metrics, n_devices=n_devices)
        if cal is not None:
            energy = propulsion_energy(metrics, cal)
            energy_total: Optional[float] = energy.total_J
            energy_idle: Optional[float] = energy.idle_J
            energy_tx: Optional[float] = energy.tx_J
            energy_prop: Optional[float] = energy.prop_J
        else:
            energy_total = energy_idle = energy_tx = energy_prop = None
        mission_s: Optional[float] = metrics.time_total_s
        path_m: Optional[float] = metrics.path_length_m
    else:
        # A1 — centralized FL, no mule. Fairness / coverage are still
        # meaningful (they describe sampling across clients), so derive
        # them from the per-round log directly.
        per_client_visits: Dict[object, int] = {}
        for r in rounds:
            for cid in getattr(r, "client_ids", ()):  # optional extension
                per_client_visits[cid] = per_client_visits.get(cid, 0) + 1
        if not per_client_visits:
            # Fall back: distribute n_updates evenly so the metric is
            # non-degenerate. Round-close already captured anomalies.
            for r in rounds:
                for k in range(r.n_updates):
                    cid = f"client-{k}"
                    per_client_visits[cid] = per_client_visits.get(cid, 0) + 1
        cov = coverage(per_client_visits, scheduled_count=n_devices)
        jf = jains_fairness(per_client_visits)
        pe = participation_entropy(per_client_visits)
        # A1's ``client_ids`` log is the *completed* set per round —
        # ``per_client_visits`` is therefore semantically completion
        # counts. ``mission_completion_rate`` here is the fraction of
        # clients that completed at least one round across the trial.
        # arm_a1.py overrides this in its post-processing using the
        # canonical per-client maps it tracks during the Monte Carlo;
        # the value computed here is a sensible fallback.
        mcr = mission_completion_rate(
            per_client_visits, n_devices=n_devices,
        )
        rc = None
        p2 = None
        energy_total = energy_idle = energy_tx = energy_prop = None
        mission_s = None
        path_m = None

    return Exp3MetricSummary(
        update_yield=yield_mean,
        coverage=cov,
        jains_fairness=jf,
        participation_entropy=pe,
        round_close_rate_kmin1=close_rate_by_k.get(1, 0.0),
        round_close_rate_kminhalf=close_rate_by_k.get(k_half, 0.0),
        round_close_rate_kminN=close_rate_by_k.get(k_full, 0.0),
        mission_completion_rate=mcr,
        rho_contact=rc,
        pass2_coverage=p2,
        propulsion_energy_J=energy_total,
        propulsion_idle_J=energy_idle,
        propulsion_tx_J=energy_tx,
        propulsion_prop_J=energy_prop,
        mission_completion_s=mission_s,
        path_length_m=path_m,
    )
