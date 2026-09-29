"""Score retained Exp 4 traces after the fact (Phase 0 of the FeRRy build plan).

``--keep-event-traces`` keeps each trial's JSONL streams under
``<csv>_traces/<cell_id>__<arm>__t<i>__s<seed>/``. Everything below is a
function of those streams, so a new metric needs neither a re-run nor a
change to the trial CSV's schema. Per trial it reports:

* **The standard Exp 4 summary**, from :func:`summarise_observation`, with
  round closure corrected. Traces recorded before Freeze Amendment 5 carry no
  mission round on ``backhaul_upload_lost``, so the recorded CSVs counted
  backhaul-dropped rounds as closed; the consumer now places each loss in the
  mission whose time window contains it. Under an age-aware merge rule an
  upload the cluster deferred (``agg:fedbuff``) or expired does not close its
  round either.
* **Time to τ** for any list of thresholds: the mission, the cluster round
  and the wall-clock seconds at which accuracy first reached τ. Wall-clock
  time is dominated by local training, not flight, until the Phase 3 mission
  clock exists; missions are the fairer unit for comparing arms. The mission
  count is the reaching mule's: how many missions the mule whose upload
  closed the reaching round had flown, that mission included. Mules fly in
  parallel, so this counts mission periods and compares across fleet sizes,
  where cluster rounds (about K per period with K mules) do not; with one
  mule it is the trial's mission count, as before.
* **Update age.** After each mission, a device's age is the number of
  missions since its last update was merged into the global model — Age of
  Updates, after Cui et al. (TMC 2024). Network AoU is the weighted mean age
  (uniform weights unless given), reported as its mean over the trial and at
  the end, with the worst and 95th-percentile per-device age. "Merged" means
  the update survived the mule's merge and reached θ: an update excluded past
  its age cutoff never does, and a deferred one counts at the mission whose
  merge flushed it. With several mules a device's age counts its own mule's
  missions, and Network AoU is sampled after every mission of any mule, in
  completion order.
* **Deadline misses**: the share of devices admitted to Pass 1 whose update
  was not collected by the device's own Deadline(j). Traces that record only
  each contact's tightest-member deadline fall back to it, and
  ``deadline_basis`` says which was used. Only traces that record the Pass-1
  plan and outcomes (from 2026-09-28) carry this; older ones report it blank.

Each row also carries the trial's **status** — from the ``trial_status.json``
marker the driver writes beside the trace, joined with the trial CSV
(``--status-csv``, one per trace root) when given: the CSV is the only record
for traces from before the marker, and has the final word on a failure,
since the runner relabels a late trial ``timeout`` after the marker is
written — and its **provenance**, the scheduler configuration read from the
trace's own process configs, in the driver's CSV columns so the two files
join. Trials whose status is not ``ok`` are left out unless
``--include-failed`` is given, as the analysis of the trial CSV leaves them
out.

Traces of several mules (FeRRy Phase 2) score like single-mule ones, with
each mission told apart by ``(mule_id, mission_round)`` (every mule numbers
its own missions from 1), and each row reports the fleet size in the
provenance column ``n_mules``. Two things only several mules produce are
scored too. An upload the cluster logged but never folded into θ — a second
upload from a mule that stopped waiting for its quorum and flew on, refused
as a duplicate, or a partial still waiting for quorum when the trial ended —
neither closes its round nor merges its updates (``unmerged_missions``). A
mission that docked with nothing to upload (``dock_on_empty``) counts in
``missions_empty`` only, not among the deferred or expired uploads, though
the cluster's fold lists it with them. Every recorded Exp 4 trial used one
mule, and on those every column is what it was before multi-mule scoring
existed; ``unmerged_missions`` is 0 on all of them.

Usage::

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp4_matrix/C_traces --tau 0.82 0.85 --csv scored.csv

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp4_s3c/off_traces results/exp4_s3c/on_traces \\
        --status-csv results/exp4_s3c/off.csv results/exp4_s3c/on.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from experiments.exp3.metrics import jains_fairness
from experiments.exp4.driver import PROVENANCE_COLUMNS, TRIAL_STATUS_FILE
from experiments.exp4.events_consumer import (
    Exp4Observation,
    MissionKey,
    MissionRecord,
    consume_run_dir,
)
from experiments.exp4.metrics import Exp4MetricSummary, summarise_observation
from hermes.mission.aggregation_rules import AGG_PLAIN, AggregationSpec
from hermes.scheduler.stages.s3_deadline import LAW_ADDITIVE, DeadlineLaw


# --------------------------------------------------------------------------- #
# Trial identity
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrialKey:
    """What a trace directory's name says about its trial."""

    cell_id: str
    arm: str
    trial_index: int
    seed: int


def parse_trial_dir(name: str) -> TrialKey:
    """``<cell_id>__<arm>__t<i>__s<seed>``, as ``--keep-event-traces`` writes it."""
    parts = name.split("__")
    if (
        len(parts) < 4
        or not parts[-2].startswith("t") or not parts[-2][1:].isdigit()
        or not parts[-1].startswith("s") or not parts[-1][1:].isdigit()
    ):
        raise ValueError(f"not a trace directory name: {name!r}")
    return TrialKey(
        cell_id="__".join(parts[:-3]),
        arm=parts[-3],
        trial_index=int(parts[-2][1:]),
        seed=int(parts[-1][1:]),
    )


# --------------------------------------------------------------------------- #
# Trial status
# --------------------------------------------------------------------------- #

#: A trial CSV's statuses by ``(arm, trial_index, seed)``. Not keyed on
#: ``cell_id``: the trace directory's copy of it is sanitised for the
#: filesystem, and the seed already identifies the cell (it is derived from
#: the cell id and the trial index). The key is unique only within one sweep —
#: paired sweeps (window adaptation off and on, say) re-run the same cells
#: under the same seeds — so an index belongs to one trace root, never to a
#: merge of several trial CSVs.
StatusIndex = Mapping[Tuple[str, int, int], str]
StatusSource = Union[None, str, Path, StatusIndex]


@dataclass(frozen=True)
class TrialStatus:
    """A trial's outcome as the harness recorded it, and where that came from."""

    key: TrialKey
    path: Path
    status: str
    source: str                         # "marker", "csv", "soft_cap" or "default"
    csv_status: Optional[str] = None    # the trial CSV's, whenever one was given
    marker_status: Optional[str] = None # the marker's, whenever there is one


def load_status_csv(path) -> Dict[Tuple[str, int, int], str]:
    """``status`` of every row of a trial CSV; a later row wins over an earlier."""
    index: Dict[Tuple[str, int, int], str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (str(row["arm"]), int(row["trial_index"]), int(row["seed"]))
            except (KeyError, TypeError, ValueError):
                continue
            index[key] = (row.get("status") or "").strip() or "ok"
    return index


def trial_status(trace_dir, status_csv: StatusSource = None) -> TrialStatus:
    """The trial's status, from its marker and its trial CSV row, else ``ok``.

    Only the driver's marker (or the CSV row) knows a trial timed out or never
    evaluated a model; the events alone do not. The marker holds the status
    the driver handed the runner, but the runner writes the CSV row after it
    and relabels a trial that returned past its soft cap ``timeout``, so:

    * a CSV row that records a failure is the final word;
    * without a CSV row, a marker's ``ok`` past the cap is relabelled
      ``timeout`` (source ``soft_cap``) by the runner's default rule — the
      cap is the trial budget unless ``--timeout-s`` overrode it, and then
      only the CSV knows;
    * a CSV ``ok`` never clears a marker's failure: the runner never turns a
      failed trial ok, so that row is some other run's.

    Traces recorded before the marker, scored without their CSV, are taken as
    ``ok``, as they always were.
    """
    trace_dir = Path(trace_dir)
    key = parse_trial_dir(trace_dir.name)
    index = _status_index(status_csv)
    csv_status = (
        index.get((key.arm, key.trial_index, key.seed)) if index is not None else None
    )
    marker_path = trace_dir / TRIAL_STATUS_FILE
    marker = _read_json(marker_path) if marker_path.exists() else {}
    marker_status = str(marker["status"]) if marker.get("status") else None

    def found(status: str, source: str) -> TrialStatus:
        return TrialStatus(key, trace_dir, status, source, csv_status, marker_status)

    if csv_status is not None and csv_status != "ok":
        return found(csv_status, "csv")
    if marker_status is not None:
        if marker_status == "ok" and csv_status is None and _past_soft_cap(marker):
            return found("timeout", "soft_cap")
        return found(marker_status, "marker")
    if csv_status is not None:
        return found(csv_status, "csv")
    return found("ok", "default")


def trial_statuses(
    trace_root,
    *,
    arms: Optional[Iterable[str]] = None,
    status_csv: StatusSource = None,
) -> List[TrialStatus]:
    """:func:`trial_status` of every trial directory under ``trace_root``."""
    index = _status_index(status_csv)
    return [trial_status(d, index) for d, _ in _trial_dirs(trace_root, arms)]


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def trial_provenance(trace_dir) -> Dict[str, object]:
    """The trial's scheduler configuration, as the driver's CSV columns hold it.

    Read from the trace's own process configs (the mule's, and a device's for
    ``fedprox_rho``) rather than the ``*_ready`` events, because the configs
    are what the processes were started with and are kept for every trace.
    Each value is formatted exactly as :class:`Exp4Driver` formats its
    :data:`PROVENANCE_COLUMNS`, so a scored row joins its trial CSV row. A key
    a legacy config lacks takes the recorded default. The fleet columns
    (Phase 2) come from the number of mule configs and the cluster's config;
    the quorum is blank for the recorded topology, one mule with a quorum of 1.
    """
    trace_dir = Path(trace_dir)
    mule = _first_json(trace_dir, "mule-*.json")
    device = _first_json(trace_dir, "device-*.json")
    cluster = _first_json(trace_dir, "cluster*.json")
    aggregation, aggregation_params = _format_aggregation(
        mule.get("aggregation") or AGG_PLAIN, mule.get("aggregation_params") or {},
    )
    deadline_law, deadline_params = _format_deadline_law(
        mule.get("deadline_law") or LAW_ADDITIVE, mule.get("deadline_params") or {},
    )
    budget = mule.get("mission_budget_s")
    provenance = {
        "mission_budget_s": "" if budget is None else float(budget),
        "mission_window_adaptation": int(bool(mule.get("mission_window_adaptation"))),
        "aggregation": aggregation,
        "aggregation_params": aggregation_params,
        "fedprox_rho": float(device.get("fedprox_rho") or 0.0),
        "pass_2_budget": int(bool(mule.get("pass_2_budget"))),
        "deadline_law": deadline_law,
        "deadline_params": deadline_params,
        "miss_priority": int(bool(mule.get("miss_priority"))),
        "dock_params": _format_dock_params(mule),
        "policy_params": _format_policy_params(mule),
    }
    n_mules = len(list(trace_dir.glob("mule-*.json"))) or 1
    quorum = int(cluster.get("min_participation") or 1)
    provenance["n_mules"] = n_mules
    provenance["min_participation"] = "" if (n_mules, quorum) == (1, 1) else quorum
    return {col: provenance.get(col, "") for col in PROVENANCE_COLUMNS}


def _format_aggregation(rule, params) -> Tuple[str, str]:
    """``aggregation`` and ``aggregation_params`` as the driver writes them."""
    try:
        spec = AggregationSpec.from_config(rule, params)
    except (TypeError, ValueError):
        # A rule or parameter this checkout does not know: keep the record.
        return str(rule), ("" if rule == AGG_PLAIN else json.dumps(params, sort_keys=True))
    return spec.rule, ("" if spec.is_plain else json.dumps(spec.to_params(), sort_keys=True))


def _format_dock_params(mule: Mapping[str, object]) -> str:
    """``dock_params`` as the driver writes it: blank for the recorded dock."""
    dock_on_empty = bool(mule.get("dock_on_empty"))
    down_wait_s = mule.get("down_wait_s")
    if not dock_on_empty and down_wait_s is None:
        return ""
    return json.dumps(
        {"dock_on_empty": dock_on_empty,
         "down_wait_s": None if down_wait_s is None else float(down_wait_s)},
        sort_keys=True,
    )


def _format_policy_params(mule: Mapping[str, object]) -> str:
    """``policy_params`` as the driver writes it: the options of arm D3
    (``whittle``) or D5 (``fedcs``), blank for every other policy."""
    policy = mule.get("contact_policy")
    if policy == "whittle":
        params = {"variant": mule.get("whittle_variant") or "expected",
                  "weights": mule.get("whittle_weights") or "uniform"}
    elif policy == "fedcs":
        params = {"value": mule.get("fedcs_value") or "unit"}
    else:
        return ""
    return json.dumps(params, sort_keys=True)


def _format_deadline_law(form, params) -> Tuple[str, str]:
    """``deadline_law`` and ``deadline_params`` as the driver writes them."""
    try:
        law = DeadlineLaw.from_config(form, params)
    except (TypeError, ValueError):
        return str(form), ("" if not params else json.dumps(params, sort_keys=True))
    return law.form, ("" if law.is_recorded else json.dumps(law.to_params(), sort_keys=True))


# --------------------------------------------------------------------------- #
# Time to τ
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TauReach:
    """When a trial's global model first reached accuracy τ, if it did."""

    tau: float
    reached: bool
    #: 1-based, in completion order among the reaching mule's missions — with
    #: one mule, among all of them.
    mission: Optional[int] = None
    cluster_round: Optional[int] = None
    wall_s: Optional[float] = None       # from the first mission's start
    mule_id: Optional[str] = None        # the reaching mission's mule


def tau_reach(obs: Exp4Observation, tau: float) -> TauReach:
    """First aggregated model with accuracy ≥ τ.

    The seeded model (cluster round 0) does not count, matching
    ``t_at_tau_round`` in :mod:`experiments.exp4.metrics`: reaching τ takes at
    least one aggregation. The evaluation runs at the inter-pass dock, inside
    the window of the mission that produced it; if it lands just after that
    window closes, it belongs to the latest mission that had started. With
    several mules the windows overlap, so only the missions of the mule whose
    upload closed the evaluated round are searched (all of them when the
    trace does not say whose it was), and ``mission`` is the reaching
    mission's place among its own mule's missions (see the module
    docstring). Wall-clock time runs from the fleet's first mission start.
    """
    first = next(
        (e for e in obs.model_evals if e.cluster_round > 0 and e.accuracy >= tau),
        None,
    )
    if first is None:
        return TauReach(tau=float(tau), reached=False)
    missions = _ordered(obs.missions)
    start = _fleet_start(obs, missions)
    candidates = missions
    closer = obs.closed_by_mule.get(first.cluster_round)
    if closer is not None:
        # With one mule every mission's key names it, so this keeps them all.
        candidates = [m for m in missions if obs.mule_key(m.mule_id) == obs.mule_key(closer)]
    reaching = next((m for m in candidates if m.contains(first.ts)), None)
    if reaching is None and first.ts is not None:
        started = [
            m for m in candidates
            if m.started_ts is not None and m.started_ts <= first.ts
        ]
        reaching = started[-1] if started else None
    mission = None
    if reaching is not None:
        own = obs.mule_key(reaching.mule_id)
        mission = next(
            i for i, m in enumerate(
                (m for m in missions if obs.mule_key(m.mule_id) == own), start=1,
            ) if m is reaching
        )
    wall_s = (first.ts - start) if (first.ts is not None and start is not None) else None
    return TauReach(
        tau=float(tau), reached=True, mission=mission,
        cluster_round=first.cluster_round, wall_s=wall_s,
        mule_id=None if reaching is None else reaching.mule_id,
    )


# --------------------------------------------------------------------------- #
# Update age and Network AoU
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AgeProfile:
    """Per-device update ages after each mission, rolled up.

    The age figures are None for a trial with no missions, which has no ages
    to summarise; ``jain_merged`` is None when nothing was merged, where
    Jain's degenerate 1.0 would read as perfectly fair.
    """

    n_missions: int
    network_aou_mean: Optional[float]    # weighted mean age, averaged over missions
    network_aou_final: Optional[float]   # weighted mean age after the last mission
    age_max: Optional[int]               # worst age any device reached
    age_p95: Optional[float]             # 95th percentile over every (device, mission)
    merged_updates: Dict[str, int]
    jain_merged: Optional[float]         # Jain's index over merged-update counts

    @property
    def merged_total(self) -> int:
        return sum(self.merged_updates.values())


def merged_devices(obs: Exp4Observation, mission: MissionRecord) -> Tuple[str, ...]:
    """Devices whose update reached the global model at this mission.

    A mission's own updates are the ones the mule's merge kept — its Pass-1
    CLEAN devices on traces from before the mule recorded that. They reach the
    model at this mission only if its upload was neither lost on the backhaul
    nor deferred, expired or left unmerged at the cluster. A deferred upload
    reaches it at the mission whose merge flushed the buffer, so that mission
    also credits the deferral's updates (a device can then appear once per
    merged update), whichever mule flew the deferred mission; one never
    flushed, and an expired one, reach it never.
    """
    key = obs.mission_key(mission)
    credited: List[str] = []
    if _merged_on_arrival(obs, key):
        credited.extend(_kept_by_mule(mission))
    for other in _ordered(obs.missions):
        other_key = obs.mission_key(other)
        if (
            other_key in obs.deferred_keys
            and obs.flush_of_keys.get(other_key) == key
            and not _dropped(obs, other_key)
        ):
            credited.extend(_kept_by_mule(other))
    return tuple(credited)


def age_profile(
    obs: Exp4Observation,
    devices: Sequence[str],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> AgeProfile:
    """Age of Updates for every device, after every mission.

    ``age_i(m) = m − U_i(m)``, where ``U_i(m)`` is the last mission up to
    ``m`` that merged device ``i``'s update, and 0 if none has, so a device
    never merged ages from the start of the trial. Empty missions count: a
    mission that merged nothing still makes every device one mission older.
    Network AoU is ``Σ_i ω_i · age_i(m)`` with the weights normalised to sum
    to 1; uniform weights make it the mean age.

    With several mules, ``m`` and ``U_i`` count the missions of device
    ``i``'s own mule (see :func:`_home_mules`), and every mule's missions are
    the sampling points, in completion order. An update merged at another
    mule's mission — a FedBuff flush — resets the device's age to 0 there.
    """
    devices = list(dict.fromkeys(str(d) for d in devices))
    if not devices:
        raise ValueError("age_profile needs at least one device")
    w = _normalised_weights(devices, weights)

    home = _home_mules(obs, devices)
    flown: Dict[Optional[str], int] = {}    # missions completed, per mule key
    fleet = 0                               # ... and by the whole fleet

    def clock(d: str) -> int:
        """Missions completed that count toward device ``d``'s age."""
        return fleet if home[d] is None else flown.get(home[d], 0)

    last_merged = {d: 0 for d in devices}
    merged_counts = {d: 0 for d in devices}
    network_aou: List[float] = []
    ages_seen: List[int] = []
    missions = _ordered(obs.missions)
    for mission in missions:
        fleet += 1
        mule = obs.mule_key(mission.mule_id)
        flown[mule] = flown.get(mule, 0) + 1
        for d in merged_devices(obs, mission):
            if d in last_merged:
                last_merged[d] = clock(d)
                merged_counts[d] += 1
        ages = {d: clock(d) - last_merged[d] for d in devices}
        network_aou.append(sum(w[d] * ages[d] for d in devices))
        ages_seen.extend(ages.values())

    jain = jains_fairness(merged_counts) if sum(merged_counts.values()) > 0 else None
    if not missions:
        return AgeProfile(
            n_missions=0, network_aou_mean=None, network_aou_final=None,
            age_max=None, age_p95=None, merged_updates=merged_counts,
            jain_merged=jain,
        )
    return AgeProfile(
        n_missions=len(missions),
        network_aou_mean=float(np.mean(network_aou)),
        network_aou_final=float(network_aou[-1]),
        age_max=int(max(ages_seen)),
        age_p95=float(np.percentile(ages_seen, 95)),
        merged_updates=merged_counts,
        jain_merged=jain,
    )


# --------------------------------------------------------------------------- #
# Deadline misses
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DeadlineMisses:
    """Admitted devices whose update was not collected by their deadline."""

    missions_with_plan: int
    admitted: int
    missed: int
    #: Which deadline the devices were held to: ``"device"`` (each one's own
    #: Deadline(j)), ``"contact"`` (the contact's tightest-member deadline, all
    #: a legacy trace records), ``"mixed"`` across missions, or None when no
    #: plan admitted anyone.
    basis: Optional[str] = None

    @property
    def rate(self) -> Optional[float]:
        return self.missed / self.admitted if self.admitted else None


def deadline_misses(obs: Exp4Observation) -> DeadlineMisses:
    """Score every mission whose trace records its Pass-1 plan.

    A device admitted to the plan is on time only if the mule recorded a
    CLEAN session with it no later than its own Deadline(j). Everything else
    is a miss: a CLEAN that finished after the device's deadline, a failed
    session, a device the mule aborted before reaching, and every device of
    an empty mission that recorded no sessions. A mission reported empty
    because the mule's merge excluded every update it collected is scored on
    its recorded sessions like any other: this metric is about collection by
    the deadline, and the exclusion shows in the merged-update figures
    instead. A trace from before the plan recorded each member's
    deadline holds every member to its contact's deadline — the tightest
    member's — so a later member can count late there that is on time under
    its own; ``basis`` reports which was used.
    """
    with_plan = admitted = missed = 0
    bases = set()
    for mission in obs.missions:
        if mission.pass_1_deadlines is None:
            continue
        with_plan += 1
        if mission.deadline_basis is not None:
            bases.add(mission.deadline_basis)
        deadline = dict(mission.pass_1_deadlines)
        on_time = {
            device
            for device, outcome, contact_ts in (mission.pass_1_outcomes or ())
            if outcome == "clean" and device in deadline and contact_ts <= deadline[device]
        }
        admitted += len(deadline)
        missed += sum(1 for device in deadline if device not in on_time)
    basis = None if not bases else (bases.pop() if len(bases) == 1 else "mixed")
    return DeadlineMisses(
        missions_with_plan=with_plan, admitted=admitted, missed=missed, basis=basis,
    )


# --------------------------------------------------------------------------- #
# One trial, and a whole trace root
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrialScore:
    key: TrialKey
    n_devices: int
    summary: Exp4MetricSummary
    missions_empty: int
    backhaul_lost_missions: int
    tau: Tuple[TauReach, ...]
    ages: AgeProfile
    deadlines: DeadlineMisses
    status: str = "ok"
    #: :data:`PROVENANCE_COLUMNS` → value, formatted as the driver's CSV.
    provenance: Mapping[str, object] = field(default_factory=dict)
    trace_root: str = ""
    #: Uploads the cluster deferred, and uploads whose updates it cut or
    #: dropped as expired; an empty upload (``dock_on_empty``) is in neither.
    deferred_missions: int = 0
    expired_missions: int = 0
    n_mules: int = 1
    #: Uploads the cluster logged but never folded (several mules only; see
    #: ``Exp4Observation.unmerged_keys``).
    unmerged_missions: int = 0

    def provenance_key(self) -> Tuple[Tuple[str, object], ...]:
        """The provenance as a hashable, column-ordered tuple."""
        return tuple((col, self.provenance.get(col, "")) for col in PROVENANCE_COLUMNS)

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "cell_id": self.key.cell_id,
            "arm": self.key.arm,
            "trial_index": self.key.trial_index,
            "seed": self.key.seed,
            "trace_root": self.trace_root,
        }
        # The fleet size is the provenance column ``n_mules`` (the number of
        # mule configs), as the driver's rows carry it.
        row.update(self.provenance_key())
        row["status"] = self.status
        row.update(self.summary.to_row())
        row.update({
            "missions_empty": self.missions_empty,
            "backhaul_lost_missions": self.backhaul_lost_missions,
            "deferred_missions": self.deferred_missions,
            "expired_missions": self.expired_missions,
            "unmerged_missions": self.unmerged_missions,
            "network_aou_mean": _blank(self.ages.network_aou_mean),
            "network_aou_final": _blank(self.ages.network_aou_final),
            "age_max": _blank(self.ages.age_max),
            "age_p95": _blank(self.ages.age_p95),
            "merged_total": self.ages.merged_total,
            "jain_merged": _blank(self.ages.jain_merged),
            "deadline_admitted": self.deadlines.admitted,
            "deadline_missed": self.deadlines.missed,
            "deadline_miss_rate": _blank(self.deadlines.rate),
            "deadline_basis": _blank(self.deadlines.basis),
        })
        for reach in self.tau:
            tag = _tau_tag(reach.tau)
            row[f"reached_{tag}"] = int(reach.reached)
            # The reaching mule's missions (mission periods), comparable across
            # fleet sizes; rounds_to_ counts cluster rounds, about K per period.
            row[f"missions_to_{tag}"] = _blank(reach.mission)
            row[f"rounds_to_{tag}"] = _blank(reach.cluster_round)
            row[f"wall_s_to_{tag}"] = _blank(reach.wall_s)
        return row


def score_trial(
    trace_dir,
    *,
    taus: Sequence[float] = (0.82,),
    weights: Optional[Mapping[str, float]] = None,
    status_csv: StatusSource = None,
) -> TrialScore:
    """Score one retained trial directory.

    ``status_csv`` (a trial CSV's path, or :func:`load_status_csv` of one)
    supplies the status of a trace that carries no marker, and overrides the
    marker's ``ok`` when it records a failure (see :func:`trial_status`).
    """
    trace_dir = Path(trace_dir)
    if not taus:
        raise ValueError("score_trial needs at least one tau")
    key = parse_trial_dir(trace_dir.name)
    devices = _device_ids(trace_dir)
    obs = consume_run_dir(trace_dir, n_devices=len(devices))
    if not devices:
        devices = sorted(obs.per_device_serves)
        obs.n_devices = len(devices)

    # Every mule of a trial runs the same scheduler settings, so the first
    # config speaks for the fleet, as it does for the provenance.
    mule_cfg = _first_json(trace_dir, "mule-*.json")
    summary = summarise_observation(
        obs,
        n_devices=len(devices),
        rf_range_m=float(mule_cfg.get("rf_range_m") or 0.0),
        n_missions_target=int(mule_cfg.get("n_missions") or len(obs.missions)),
        tau=float(taus[0]),
    )
    return TrialScore(
        key=key,
        n_devices=len(devices),
        summary=summary,
        missions_empty=obs.missions_empty,
        backhaul_lost_missions=len(obs.backhaul_lost_keys),
        tau=tuple(tau_reach(obs, t) for t in taus),
        ages=age_profile(obs, devices, weights=weights),
        deadlines=deadline_misses(obs),
        status=trial_status(trace_dir, status_csv).status,
        provenance=trial_provenance(trace_dir),
        trace_root=str(trace_dir.parent),
        deferred_missions=len(obs.deferred_keys),
        expired_missions=len(obs.expired_keys),
        n_mules=obs.n_mules,
        unmerged_missions=len(obs.unmerged_keys),
    )


def score_traces(
    trace_root,
    *,
    taus: Sequence[float] = (0.82,),
    arms: Optional[Iterable[str]] = None,
    include_failed: bool = False,
    status_csv: StatusSource = None,
) -> List[TrialScore]:
    """Score every trial directory under ``trace_root``, in name order.

    Directories whose names are not trial names are skipped, and so, unless
    ``include_failed``, are trials whose status is not ``ok`` (see
    :func:`trial_status`): a timed-out or ``no_eval`` trial is not a valid
    observation, and the trial CSV's analysis drops it too.
    """
    index = _status_index(status_csv)
    scores: List[TrialScore] = []
    for d, _ in _trial_dirs(trace_root, arms):
        if not include_failed and trial_status(d, index).status != "ok":
            continue
        scores.append(score_trial(d, taus=taus, status_csv=index))
    return scores


def write_scores_csv(scores: Sequence[TrialScore], path) -> None:
    """One row per trial. The column set is fixed by the first score's τ list."""
    if not scores:
        raise ValueError("no scores to write")
    rows = [s.to_row() for s in scores]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="traces_scorer", description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", required=True, nargs="+",
                    help="One or more trace roots, e.g. results/exp4_matrix/C_traces")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.82],
                    help="Accuracy thresholds for time to tau.")
    ap.add_argument("--arms", nargs="+", default=None, help="Only these arms.")
    ap.add_argument("--csv", default=None, help="Write one row per trial here.")
    ap.add_argument("--status-csv", nargs="+", default=None,
                    help="The trial CSV of each --traces root, in the same "
                         "order ('-' for a root without one). Gives the status "
                         "of traces recorded without a trial_status.json "
                         "marker, and of trials the runner relabelled after it.")
    ap.add_argument("--include-failed", action="store_true",
                    help="Also score trials whose status is not ok.")
    args = ap.parse_args(argv)
    # One CSV per root, never merged: a trial's key is unique only within its
    # own sweep (see StatusIndex), so paired sweeps would overwrite each
    # other's statuses.
    status_csvs = args.status_csv or ["-"] * len(args.traces)
    if len(status_csvs) != len(args.traces):
        ap.error(f"--status-csv takes one trial CSV per --traces root, '-' for "
                 f"none: {len(args.traces)} root(s), {len(status_csvs)} CSV(s)")

    scores: List[TrialScore] = []
    statuses: List[TrialStatus] = []
    for root, status_csv in zip(args.traces, status_csvs):
        index = None if status_csv == "-" else load_status_csv(status_csv)
        scores.extend(score_traces(
            root, taus=args.tau, arms=args.arms,
            include_failed=args.include_failed, status_csv=index,
        ))
        statuses.extend(trial_statuses(root, arms=args.arms, status_csv=index))

    for s in statuses:
        if (s.marker_status is not None and s.csv_status is not None
                and s.marker_status != s.csv_status):
            used = "the trial CSV" if s.source == "csv" else "the marker"
            print(f"warning: {s.path.name}: marker says {s.marker_status!r}, "
                  f"trial CSV says {s.csv_status!r}; using {used}")
    if not args.include_failed:
        excluded: Dict[str, int] = {}
        for s in statuses:
            excluded.setdefault(s.key.arm, 0)
            if s.status != "ok":
                excluded[s.key.arm] += 1
        if excluded:
            print("excluded (status not ok): " + ", ".join(
                f"{arm} {n}" for arm, n in sorted(excluded.items())))
    if not scores:
        print("no trial directories found" if not statuses else "no trials left to score")
        return 1
    _warn_provenance_conflicts(scores)
    if args.csv:
        write_scores_csv(scores, args.csv)
        print(f"wrote {len(scores)} rows to {args.csv}")

    groups: Dict[Tuple[Tuple[Tuple[str, object], ...], str], List[TrialScore]] = {}
    for s in scores:
        groups.setdefault((s.provenance_key(), s.key.arm), []).append(s)
    reach_hdr = "".join(f"{'reach@' + _tau_tag(t):>16}" for t in args.tau)
    for prov in sorted({p for p, _ in groups}, key=repr):
        print("\n" + " ".join(f"{col}={value}" for col, value in prov if value != ""))
        print(f"{'arm':<6}{'trials':>7}{reach_hdr}{'rcr_k1':>9}{'lost':>6}"
              f"{'NAoU':>7}{'age_max':>9}{'miss':>7}")
        for arm in sorted(a for p, a in groups if p == prov):
            group = groups[(prov, arm)]
            reach = "".join(
                f"{sum(s.tau[i].reached for s in group):>12}/{len(group):<3}"
                for i in range(len(args.tau))
            )
            print(
                f"{arm:<6}{len(group):>7}{reach}"
                f"{_fmt_mean([s.summary.round_close_rate_kmin1 for s in group], 9, 3)}"
                f"{sum(s.backhaul_lost_missions for s in group):>6}"
                f"{_fmt_mean([s.ages.network_aou_mean for s in group], 7, 2)}"
                f"{_fmt_mean([s.ages.age_max for s in group], 9, 2)}"
                f"{_fmt_mean([s.deadlines.rate for s in group], 7, 3)}"
            )
    print("\nrcr_k1: round closure with backhaul-lost, deferred and expired rounds "
          "excluded, and with several mules uploads the cluster never folded. "
          "lost: missions whose upload was lost. NAoU: mean age of the "
          "devices' last merged updates, in missions. miss: deadline-miss rate "
          "(blank on traces without the Pass-1 plan). Means skip blank values.")
    return 0


def _warn_provenance_conflicts(scores: Sequence[TrialScore]) -> None:
    """Flag one trial scored under two configurations (e.g. two trace roots).

    Summaries are grouped by provenance, so such trials are never pooled, but
    a paired comparison that joins on the trial key alone would mix them.
    """
    seen: Dict[TrialKey, set] = {}
    for s in scores:
        seen.setdefault(s.key, set()).add(s.provenance_key())
    for key, provs in seen.items():
        if len(provs) > 1:
            print(f"warning: {key.cell_id} {key.arm} t{key.trial_index} s{key.seed} "
                  f"was scored under {len(provs)} different provenances")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _merged_on_arrival(obs: Exp4Observation, key: MissionKey) -> bool:
    """Whether the cluster merged this mission's upload when it arrived."""
    return not (key in obs.deferred_keys or _dropped(obs, key))


def _dropped(obs: Exp4Observation, key: MissionKey) -> bool:
    """Whether this mission's upload can never reach θ: lost on the backhaul,
    expired at the cluster, or logged there and never folded."""
    return (
        key in obs.backhaul_lost_keys
        or key in obs.expired_keys
        or key in obs.unmerged_keys
    )


def _home_mules(obs: Exp4Observation, devices: Sequence[str]) -> Dict[str, Optional[str]]:
    """Each device's mule (as a mission-key mule), whose missions its age counts.

    With one mule every mission is the device's mule's, so every device maps
    to None: its age counts every mission. With several, the mule whose config
    lists the device in its slice, else the first mule whose mission names it
    (collected, merged, planned or visited in Pass 1); a device that neither
    places also maps to None and is aged by every mission of the fleet.
    """
    if obs.n_mules <= 1:
        return {d: None for d in devices}
    home: Dict[str, Optional[str]] = {}
    for mule_id, members in obs.mule_slices.items():
        for d in members:
            home.setdefault(d, obs.mule_key(mule_id))
    for m in _ordered(obs.missions):
        named = [
            *m.pass_1_clean_devices,
            *(m.pass_1_merged_devices or ()),
            *(d for d, _ in m.pass_1_deadlines or ()),
            *(d for d, _, _ in m.pass_1_outcomes or ()),
        ]
        for d in named:
            home.setdefault(d, obs.mule_key(m.mule_id))
    return {d: home.get(d) for d in devices}


def _fleet_start(obs: Exp4Observation, missions: Sequence[MissionRecord]) -> Optional[float]:
    """When the fleet began flying: the earliest start among each mule's first
    completed mission (``missions`` in completion order). With one mule, its
    first mission's start, None if that has none."""
    firsts: Dict[Optional[str], MissionRecord] = {}
    for m in missions:
        firsts.setdefault(obs.mule_key(m.mule_id), m)
    starts = [m.started_ts for m in firsts.values() if m.started_ts is not None]
    return min(starts) if starts else None


def _kept_by_mule(mission: MissionRecord) -> Tuple[str, ...]:
    """The mission's updates its mule merged; every CLEAN one on traces older
    than Phase 1 (later ones record the exclusions, see MissionRecord)."""
    if mission.pass_1_merged_devices is not None:
        return mission.pass_1_merged_devices
    return mission.pass_1_clean_devices


def _trial_dirs(trace_root, arms: Optional[Iterable[str]]) -> List[Tuple[Path, TrialKey]]:
    """Trial directories under ``trace_root`` in name order, optionally by arm."""
    wanted = set(arms) if arms is not None else None
    found: List[Tuple[Path, TrialKey]] = []
    for d in sorted(Path(trace_root).iterdir()):
        if not d.is_dir():
            continue
        try:
            key = parse_trial_dir(d.name)
        except ValueError:
            continue
        if wanted is not None and key.arm not in wanted:
            continue
        found.append((d, key))
    return found


def _status_index(status_csv: StatusSource) -> Optional[StatusIndex]:
    if status_csv is None:
        return None
    if isinstance(status_csv, Mapping):
        return status_csv
    return load_status_csv(status_csv)


def _past_soft_cap(marker: Mapping[str, object]) -> bool:
    """Whether the marker's run time exceeds its trial budget, the runner's
    default soft cap. False for a marker that records neither."""
    try:
        return float(marker["run_s"]) > float(marker["trial_budget_s"])
    except (KeyError, TypeError, ValueError):
        return False


def _ordered(missions: Sequence[MissionRecord]) -> List[MissionRecord]:
    """Missions in completion order; the recorded order when timestamps are absent."""
    if all(m.completed_ts is not None for m in missions):
        return sorted(missions, key=lambda m: m.completed_ts)
    return list(missions)


def _normalised_weights(
    devices: Sequence[str], weights: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    if weights is None:
        return {d: 1.0 / len(devices) for d in devices}
    missing = [d for d in devices if d not in weights]
    if missing:
        raise ValueError(f"weights missing for devices {missing}")
    raw = {d: float(weights[d]) for d in devices}
    if any(v < 0 for v in raw.values()) or sum(raw.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    total = sum(raw.values())
    return {d: v / total for d, v in raw.items()}


def _device_ids(trace_dir: Path) -> List[str]:
    """The trial's devices, from the cluster's seed list or the device configs."""
    cluster_cfg = _first_json(trace_dir, "cluster*.json")
    seeded = [
        str(s["device_id"]) for s in cluster_cfg.get("seed_devices") or ()
        if isinstance(s, dict) and "device_id" in s
    ]
    if seeded:
        return seeded
    return sorted(
        str(_read_json(p).get("device_id") or p.stem[len("device-"):])
        for p in trace_dir.glob("device-*.json")
    )


def _first_json(trace_dir: Path, pattern: str) -> dict:
    for p in sorted(trace_dir.glob(pattern)):
        return _read_json(p)
    return {}


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _fmt_mean(values, width: int, places: int) -> str:
    """Right-aligned mean of the values that are not None; a dash if none are."""
    present = [v for v in values if v is not None]
    if not present:
        return f"{'—':>{width}}"
    return f"{np.mean(present):>{width}.{places}f}"


def _tau_tag(tau: float) -> str:
    return f"tau{tau:g}"


def _blank(v):
    return "" if v is None else v


if __name__ == "__main__":
    raise SystemExit(main())
