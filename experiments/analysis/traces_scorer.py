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
  and the wall-clock seconds at which accuracy first reached τ, and on the
  simulated mission clock the simulated seconds too. Wall-clock time is
  dominated by local training, not flight; missions, and on the mission
  clock simulated seconds, are the fairer units for comparing arms. The mission
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

Traces on the simulated mission clock (FeRRy Phase 3; ``mule_ready`` says
``mission_clock: "sim"``, and a trace that does not say so is wall-clock, as
every recorded one is) are scored in simulated time where time matters. Their
missions complete, and so are ordered for the ages and Network AoU, by their
simulated ends (``mission_completed.sim_end_s``); the envelope timestamps
stay wall time and still place cluster events in the missions' windows. Time
to τ adds the simulated seconds (``sim_s_to_τ``, from ``model_eval.sim_ts``)
beside the wall ones, and the deadline misses compare simulated contact
times with simulated deadlines. The summary adds the simulated mission
duration, the clock's ledger, the SIMULATED energy, the budget overrun and
the re-plans, aborts and inserts (:mod:`experiments.exp4.metrics`), and the
provenance the Phase 3 driver columns. A trace whose clocks disagree is
refused, never scored (:class:`~experiments.exp4.events_consumer.ClockDomainError`).

FeRRy Phase 4 (the plan clock; the Phase 4 spec, other choices 6 and 12) adds
:data:`PHASE_4_COLUMNS` after ``deadline_basis``, each blank where the trace
cannot say, so every trace recorded before Phase 4 scores blank in all of them
at the defaults and keeps every other column:

* **Cap violations** at an age cap S (the user's decision 1: a device's age is
  its own mule's missions since its last merged update, the unit of the ages
  above): ``cap_s``; ``cap_violations``, the (device, mission) pairs whose age
  after the mission is at least S; ``cap_violation_devices``, the devices with
  any; and ``cap_violation_events``, the mule's own log of the capped devices
  it failed, by cause (``unplannable``, ``crowded``, ``dropped_in_flight``,
  ``not_merged``; JSON), at the S the mule ran. S is ``--age-cap-s`` when
  given, which scores every arm of a study at one S, the H and D arms and F-cap
  included; else the trace's own (the S its plan-mode mule ran); else the four
  are blank. The pair count and the mule's log differ by what the mule cannot
  see (a lost backhaul upload, a merge the cluster deferred), and with a
  lookahead L > 0 the mule also logs devices aged S − L to S − 1.
* **The plan** (plan-mode missions only): ``plan_served_share_mean``, the mean
  over missions of the share of the demand the committed plan serves, counted
  in devices (|served| / |demand|, not the coverage weights, which differ
  between F, F-prio and the uniform option), over missions with a demand; and
  ``plan_v_mean``, the mean of the plans' V.
* On simulated-clock traces a Phase 4 build recorded (its mule config carries
  ``plan_mode``, as every Phase 4 mule's JSON does, at the defaults too):
  ``band_shares``, the share of Pass-1 stops flown on each band class
  (``pass_1_flown[].band``, so FX's per-stop switches count; JSON);
  ``far_served_share``, the merged updates of the devices beyond ``rf_range_m``
  of the dock over those devices' own missions; and, for a whole-scheduler
  baseline (D1-D5), ``policy_drops``, the devices it left out before takeoff
  (``pass_1_policy_drops``, summed over missions: 0 when it left nothing out,
  which the mule then does not write). Older traces record these facts only in
  part, or not at all (D arms reported no drops before Phase 4), so there they
  are blank.

FeRRy Phase 5 (the flight clock's learned fillings; the Phase 5 spec, other
choices 6 and 9). The provenance names a learned arm's checkpoint as the
driver does, by tag and sha only: ``pair_tag`` and ``pair_sha256`` in
``ferry_params`` for an FQ arm, and ``policy_tag`` and ``policy_sha256`` in
``policy_params`` for E3 (the driver's own ``plan_ferry_params`` and
``learned_policy_params``), so their rows join their trial CSV rows too. With
``pair_columns`` (``--pair-columns``) :data:`PHASE_5_COLUMNS` follow the τ
columns; without it the row is the Phase 4 one, column for column, so no row
of a recorded trace changes and no Phase 4 pin moves (:func:`pair_report` says
what each holds):

* **The pair slot's decisions** (``pass_1_pairs``, pooled over the trial's
  missions): ``pair_decisions``, how many it made, one per Pass-1 stop flown;
  ``pair_feasible_mean``, the mean number of pairs its mask admitted;
  ``pair_mask_empty``, how many found none and fell back on FX's pair;
  ``pair_fx_agree_share``, the share whose chosen pair was FX's by FX's own
  rule at that arrival (not the FX arm's flight, which under ``replan``
  re-plans first; an empty mask's decision agrees, since its fallback is
  FX's pair); ``pair_band_off_bbar_share``, the share served on a class other
  than the committed b̄; ``pair_reorder_share``, the share whose pair chose a
  stop other than the remainder's head. The agreement and re-order shares
  count the choice: the stop is served on the chosen band at once, but the
  departure check after it may re-plan the order the pair set
  (``trimmed_next``) and the beacon hook may insert a stop ahead of it, so
  the stop flown next can differ. Blank on a trial that flew no pair slot; on
  one that made no decision the two counts are 0 and the rest blank.
* **E3's unvisited stops**: ``e3_unvisited_mean``, the mean over missions of
  the stops its pass left when none was admissible any more
  (``pass_1_e3_unvisited``, never widened); blank for every other arm.

Study 5.6's "stops served per mission" is ``pass1_contacts_mean``.

Exp 5 addendum, Study 5.11 (a): the decision cost. With ``cost_columns``
(``--cost-columns``) :data:`COST_COLUMNS` follow the τ columns (and the pair
columns, when both are asked for); without it the row is the one above,
column for column. Every one is a wall time or derived from wall times, so
none enters a determinism comparison (:func:`cost_report` says what each
holds):

* **The planner** (plan-mode missions, ``plan_wall_s``): ``plan_wall_s_mean``
  and ``plan_wall_s_p95`` over the trial's missions, and
  ``plan_search_shares``, the share of the missions whose committed class
  ran each search mode (JSON, every mode listed).
* **The flight clock** (``pass_1_pairs_wall``, ``pass_1_e3_wall``, pooled
  over the trial's decisions): ``pair_wall_s_mean``, ``pair_wall_s_p95`` and
  ``pair_mask_wall_s_mean`` for the pair slot, the same three for E3's calls,
  and ``flight_decisions_per_mission``, the timed decisions per mission.
* **The footprint** (``footprint.json``, which the driver's footprint probe
  writes beside the kept trace): ``trial_processes`` and the peak resident
  memory in MiB, concurrent (``peak_rss_mib_total``) and per role
  (``peak_rss_mib_cluster``, ``_mule``, ``_device``: the largest process).

Exp 5 addendum, Study 5.13: the data and the detector. With
``detection_columns`` (``--detection-columns``) :data:`DETECTION_COLUMNS`
follow the τ, pair and cost columns (:func:`detection_report`): the trial's
partition and alpha and the Network AoU weighted by each device's shard size
(from the status marker's ``data``, which a non-IID or family-labelled trial
writes), and the final evaluation's TPR, FPR, precision, F1 and recall per
attack family (``model_eval.detection``, which the cluster writes when the
test set carries the families).

Usage::

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp4_matrix/C_traces --tau 0.82 0.85 --csv scored.csv

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp4_s3c/off_traces results/exp4_s3c/on_traces \\
        --status-csv results/exp4_s3c/off.csv results/exp4_s3c/on.csv

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp4_p4/s58_traces --age-cap-s 3 --csv scored.csv

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp5/s55_traces --pair-columns --csv scored.csv

    python -m experiments.analysis.traces_scorer \\
        --traces results/exp5/s511_traces --cost-columns --csv scored.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from experiments.exp3.metrics import jains_fairness
from experiments.exp4.driver import (
    PROVENANCE_COLUMNS,
    TRIAL_STATUS_FILE,
    contact_band_column,
    learned_policy_params,
    plan_ferry_params,
)
from experiments.exp4.events_consumer import (
    ClockDomainError,
    Exp4Observation,
    MissionKey,
    MissionRecord,
    completion_order,
    consume_run_dir,
)
from experiments.exp4.footprint import read_footprint
from experiments.exp4.metrics import Exp4MetricSummary, summarise_observation
from experiments.exp4.topology_builder import SESSION_TTL_S, device_positions, device_spread_m
from hermes.l1.mission_clock import DOCK_POSE
from hermes.mission.aggregation_rules import AGG_PLAIN, AggregationSpec
from hermes.processes.config import (
    BACKHAUL_SECONDS,
    CLOCK_SIM,
    CLOCK_WALL,
    CONTACT_POLICY_CHEN_DQN,
    FERRY_PARAMS_OMITTED_AT_NONE,
    FERRY_SPEC_FIELDS,
    FLIGHT_SLOT_PAIR_Q,
    PLAN_MODE_FERRY,
    MuleConfig,
)
from hermes.scheduler.stages.s3_deadline import LAW_ADDITIVE, DeadlineLaw, time_scale_for_period
from hermes.types.scheduler import CAP_REASONS, SEARCH_MODES


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
      ``timeout`` (source ``soft_cap``) by the runner's rule. The cap is the
      marker's ``soft_cap_s`` when it records one: a mission-clock trial run
      by ``runner_main`` records the cap the runner applied (the largest
      re-costed budget over the grid, or ``--timeout-s``), which can exceed
      the trial's own budget. Otherwise it is the marker's ``trial_budget_s``,
      the trial's own budget and, on the wall clock, the runner's default
      cap; a ``--timeout-s`` that overrode it there is known only to the CSV;
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
    The clock columns (Phase 3) are :func:`_clock_provenance`'s; on a trace
    recorded before them every one is blank at the recorded settings, as the
    driver's are, but the L1 channel, realism and the model width, which are
    read off the configs as the trial ran them.
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
    provenance.update(_clock_provenance(
        mule, cluster,
        devices=[_read_json(p) for p in sorted(trace_dir.glob("device-*.json"))],
        marker=_read_json(trace_dir / TRIAL_STATUS_FILE),
        seed=_trial_seed(trace_dir, mule),
    ))
    return {col: provenance.get(col, "") for col in PROVENANCE_COLUMNS}


#: ``MuleConfig`` ferry fields ``ferry_params`` leaves out, as the driver does:
#: those with a column of their own, and the ground-truth availability, which
#: no record shows (critic B16).
_FERRY_PARAMS_OMITTED = (
    "contact_band", "in_flight_response", "backhaul_model",
    "contact_reliability_source", "device_availability", "t_nom_s",
)


def _clock_provenance(
    mule: Mapping[str, object],
    cluster: Mapping[str, object],
    *,
    devices: Sequence[Mapping[str, object]],
    marker: Mapping[str, object],
    seed: Optional[int],
) -> Dict[str, object]:
    """The FeRRy Phase 3 provenance columns, as ``Exp4Driver._clock_provenance``
    writes them, from the trace's own configs.

    Each is blank at its recorded value. The switches, the deadline time
    unit, Φ₀, T_nom and the session TTL are the mule config's fields;
    ``ferry_params`` (simulated clock only) is every other ferry field of the
    mule config, as :func:`_format_ferry_params` writes it. Three settings no
    CSV recorded before are read off the other configs: ``l1_channel`` from
    the cluster's per-mission loss schedule, ``input_dim`` from the cluster's
    model width, and ``realism`` from the devices' own contact reliability
    (:func:`_realism`). On a trace recorded before Phase 3 (every kept one)
    that leaves ``l1_channel``, ``realism`` and ``input_dim`` as recorded and
    the other ten blank: all of them ran the recorded 3 s TTL.

    FeRRy Phase 4 (other choices 12; unit_U3b.md section 5.5): ``contact_band``
    and the plan keys of ``ferry_params`` come from the driver's own functions
    (:func:`~experiments.exp4.driver.contact_band_column`,
    :func:`~experiments.exp4.driver.plan_ferry_params`) applied to the mule
    config, so the two files agree by construction: ``search`` for a plan arm
    that searches the band classes, and the plan fields in plan mode, or an H
    or D arm's ``member_admission`` when it is ``subset``. A config without the
    plan fields (every trace recorded before Phase 4) reads as their defaults,
    so its strings are unchanged.
    """
    sim = mule.get("mission_clock") == CLOCK_SIM

    def unless(value, recorded):
        return "" if value is None or value == recorded else value

    return {
        "mission_clock": CLOCK_SIM if sim else "",
        "contact_band": contact_band_column(mule),
        "in_flight_response": unless(mule.get("in_flight_response"), "abort"),
        "backhaul_model": unless(mule.get("backhaul_model"), "mission"),
        "contact_reliability_source": unless(mule.get("contact_reliability_source"), "origin"),
        "deadline_time_scale": unless(_float_or_none(mule.get("deadline_time_scale")), 1.0),
        "initial_window_s": _blank(_float_or_none(mule.get("initial_window_s"))),
        "t_nom_s": _blank(_float_or_none(mule.get("t_nom_s"))),
        "session_ttl_s": unless(_float_or_none(mule.get("session_ttl_s")), float(SESSION_TTL_S)),
        "ferry_params": _format_ferry_params(mule, marker) if sim else "",
        "l1_channel": 1 if cluster.get("backhaul_loss_schedule") is not None else "",
        "realism": 1 if _realism(mule, devices, seed) else "",
        "input_dim": "" if cluster.get("input_dim") is None else int(cluster["input_dim"]),
    }


def _format_ferry_params(mule: Mapping[str, object], marker: Mapping[str, object]) -> str:
    """``ferry_params`` as the driver writes it: JSON (sorted keys) of every
    ferry field of the mule config but those :data:`_FERRY_PARAMS_OMITTED`
    (a field the config lacks takes ``MuleConfig``'s default), the backhaul
    period resolved to ``n_missions * T_nom`` when the seconds model computes
    it, ``t_nom_computed`` (:func:`_t_nom_computed`) and the FeRRy Phase 4
    plan keys (:func:`~experiments.exp4.driver.plan_ferry_params`: none at the
    defaults)."""
    from hermes.l1.channel_model import backhaul_period_s

    defaults = {f.name: f for f in fields(MuleConfig)}
    shown: Dict[str, object] = {}
    for name in FERRY_SPEC_FIELDS:
        if name in _FERRY_PARAMS_OMITTED:
            continue
        if name in mule:
            shown[name] = mule[name]
        else:
            default = defaults[name]
            shown[name] = (default.default_factory() if default.default_factory is not MISSING
                           else default.default)
        if name in FERRY_PARAMS_OMITTED_AT_NONE and shown[name] is None:
            # The Exp 5 addendum's ferry fields, left out at None as the driver
            # leaves them (a recorded trace keeps its string).
            del shown[name]
    t_nom = _float_or_none(mule.get("t_nom_s"))
    if (mule.get("backhaul_model") == BACKHAUL_SECONDS and mule.get("backhaul_period_s") is None
            and t_nom is not None):
        shown["backhaul_period_s"] = backhaul_period_s(int(mule.get("n_missions") or 0), t_nom)
    shown["t_nom_computed"] = _t_nom_computed(mule, marker)
    shown.update(plan_ferry_params(mule))
    return json.dumps(shown, sort_keys=True, default=str)


def _t_nom_computed(mule: Mapping[str, object], marker: Mapping[str, object]) -> bool:
    """Whether the driver computed the trial's T_nom rather than being given it.

    The per-role configs record T_nom (``t_nom_s``) but not where it came
    from, so the trial's status marker is read first, should it record
    ``t_nom_computed``. Otherwise it is inferred. The driver computes T_nom
    exactly when a setting needs one and none was given (``--t-nom-s``), and
    the configs show each such setting: the seconds backhaul without a
    period, the deadline time unit at T_nom / 10 s, D5's merge period at
    T_nom, and Φ₀ in missions, which they record as its window in seconds
    (``initial_window_s``). A T_nom recorded with none of them was given; one
    recorded with any of them is taken as computed. Without ``--t-nom-s``
    the driver records a T_nom only when it computed one, so that is exact
    for every such trial. A T_nom given beside one of those settings reads
    as computed, and so does one given beside a Φ₀ given in seconds, which
    the configs record as they record a Φ₀ given in missions. FeRRy Phase 4:
    plan mode (``plan_mode == "ferry"``) needs T_nom too, as T in its score
    (decision 2 (b)), so it counts as such a setting.
    """
    recorded = marker.get("t_nom_computed")
    if isinstance(recorded, bool):
        return recorded
    t_nom = _float_or_none(mule.get("t_nom_s"))
    if t_nom is None or not t_nom > 0.0:
        return False
    params = mule.get("aggregation_params")
    period = _float_or_none(params.get("period_s")) if isinstance(params, Mapping) else None
    return (
        (mule.get("backhaul_model") == BACKHAUL_SECONDS and mule.get("backhaul_period_s") is None)
        or _float_or_none(mule.get("deadline_time_scale")) == time_scale_for_period(t_nom)
        or period == t_nom
        or _float_or_none(mule.get("initial_window_s")) is not None
        or mule.get("plan_mode") == PLAN_MODE_FERRY
    )


def _realism(
    mule: Mapping[str, object], devices: Sequence[Mapping[str, object]], seed: Optional[int],
) -> bool:
    """Whether the trial ran with the driver's ``realism`` (EX-4.2).

    Realism gives every device its own contact reliability, so a device config
    that carries one says so; that decides every trace recorded before Phase 3.
    Under the channel reliability source (simulated clock) the devices carry
    none, the draw being the mule's ground-truth availability, and the
    topology builder fills that availability with or without realism. There
    the layout decides: realism spreads the devices over its field, and
    without it they are drawn from the trial seed inside the tight cluster
    (``device_spread_m`` of the RF range), which is reproduced exactly.
    """
    if any(d.get("contact_reliability") is not None for d in devices):
        return True
    if mule.get("mission_clock") != CLOCK_SIM or not mule.get("device_availability"):
        return False
    xy = sorted(
        (float(d["position"][0]), float(d["position"][1])) for d in devices
        if isinstance(d.get("position"), (list, tuple)) and len(d["position"]) >= 2
    )
    if not xy or len(xy) != len(devices) or seed is None:
        return False
    tight = sorted(device_positions(len(xy), int(seed), device_spread_m(
        float(mule.get("rf_range_m") or 0.0),
    )))
    return not all(
        abs(a - c) <= 1e-9 and abs(b - d) <= 1e-9 for (a, b), (c, d) in zip(xy, tight)
    )


def _trial_seed(trace_dir: Path, mule: Mapping[str, object]) -> Optional[int]:
    """The trial's seed: the mule config's ``trial_seed`` (simulated clock),
    else the trace directory's name."""
    if mule.get("trial_seed") is not None:
        return int(mule["trial_seed"])
    try:
        return parse_trial_dir(trace_dir.name).seed
    except ValueError:
        return None


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
    (``whittle``) or D5 (``fedcs``), and FeRRy Phase 5's arm E3 (``chen_dqn``)
    by its checkpoint's tag and sha, from the driver's own
    :func:`~experiments.exp4.driver.learned_policy_params` so the two columns
    agree; blank for every other policy."""
    policy = mule.get("contact_policy")
    if policy == "whittle":
        params = {"variant": mule.get("whittle_variant") or "expected",
                  "weights": mule.get("whittle_weights") or "uniform"}
    elif policy == "fedcs":
        params = {"value": mule.get("fedcs_value") or "unit"}
    else:
        params = {}
    params.update(learned_policy_params(mule))
    return json.dumps(params, sort_keys=True) if params else ""


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
    #: On the simulated mission clock (FeRRy Phase 3): simulated seconds from
    #: the fleet's first takeoff to the evaluated model's simulated time; None
    #: on the wall clock.
    sim_s: Optional[float] = None


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

    On the simulated mission clock, ``sim_s`` is the same span on that clock:
    from the fleet's first simulated takeoff (``sim_start_s``) to the
    evaluation's ``model_eval.sim_ts``, the simulated completion of the latest
    upload the cluster had ingested, which with one mule is the reaching
    mission's own upload. The windows that place the evaluation stay wall
    ones, as the event envelopes are.
    """
    first = next(
        (e for e in obs.model_evals if e.cluster_round > 0 and e.accuracy >= tau),
        None,
    )
    if first is None:
        return TauReach(tau=float(tau), reached=False)
    missions = _ordered(obs.missions, obs.mission_clock)
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
    sim_s = None
    if obs.mission_clock == CLOCK_SIM and first.sim_ts is not None:
        sim_start = _fleet_sim_start(obs, missions)
        if sim_start is not None:
            sim_s = first.sim_ts - sim_start
    return TauReach(
        tau=float(tau), reached=True, mission=mission,
        cluster_round=first.cluster_round, wall_s=wall_s,
        mule_id=None if reaching is None else reaching.mule_id,
        sim_s=sim_s,
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
    for other in _ordered(obs.missions, obs.mission_clock):
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
    On the simulated mission clock the completion order is that of the
    missions' simulated ends (FeRRy Phase 3); with one mule either order is
    the mission rounds'.
    """
    devices = list(dict.fromkeys(str(d) for d in devices))
    if not devices:
        raise ValueError("age_profile needs at least one device")
    w = _normalised_weights(devices, weights)

    merged_counts = {d: 0 for d in devices}
    network_aou: List[float] = []
    ages_seen: List[int] = []
    missions = _ordered(obs.missions, obs.mission_clock)
    for step in _age_walk(obs, devices):
        for d in step.merged:
            merged_counts[d] += 1
        network_aou.append(sum(w[d] * step.ages[d] for d in devices))
        ages_seen.extend(step.ages.values())

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


@dataclass(frozen=True)
class _AgeStep:
    """One mission of :func:`_age_walk`: the mission, the devices whose age
    counts it (their own mule flew it; with one mule every device), each
    device's age after it, and the devices whose updates reached the model at
    it (a device appears once per merged update)."""

    mission: MissionRecord
    own: Tuple[str, ...]
    ages: Dict[str, int]
    merged: Tuple[str, ...]


def _age_walk(obs: Exp4Observation, devices: Sequence[str]) -> Iterator[_AgeStep]:
    """The missions in completion order with every device's age after each.

    The one definition of a device's age (:func:`age_profile`), shared by the
    cap violations (FeRRy Phase 4, decision 1, which counts the cap in exactly
    this unit) and the far devices' service, so they can never drift apart.
    ``devices`` are distinct ids.
    """
    home = _home_mules(obs, devices)
    flown: Dict[Optional[str], int] = {}    # missions completed, per mule key
    fleet = 0                               # ... and by the whole fleet

    def clock(d: str) -> int:
        """Missions completed that count toward device ``d``'s age."""
        return fleet if home[d] is None else flown.get(home[d], 0)

    last_merged = {d: 0 for d in devices}
    for mission in _ordered(obs.missions, obs.mission_clock):
        fleet += 1
        mule = obs.mule_key(mission.mule_id)
        flown[mule] = flown.get(mule, 0) + 1
        merged = []
        for d in merged_devices(obs, mission):
            if d in last_merged:
                last_merged[d] = clock(d)
                merged.append(d)
        yield _AgeStep(
            mission=mission,
            own=tuple(d for d in devices if home[d] is None or home[d] == mule),
            ages={d: clock(d) - last_merged[d] for d in devices},
            merged=tuple(merged),
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

    On the simulated mission clock the deadlines and the contact times are
    both simulated seconds: the two switch clocks together, and the consumer
    refuses a trace in which they do not
    (:class:`~experiments.exp4.events_consumer.ClockDomainError`).
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
# FeRRy Phase 4: the plan clock
# --------------------------------------------------------------------------- #

#: The scorer's FeRRy Phase 4 columns, in row order after ``deadline_basis``
#: (the Phase 4 spec, other choices 12; the module docstring says what each
#: holds and when it is blank). Scorer-only: the trial CSV's header is
#: unchanged.
PHASE_4_COLUMNS = (
    "cap_s", "cap_violations", "cap_violation_devices", "cap_violation_events",
    "band_shares", "plan_served_share_mean", "plan_v_mean", "far_served_share",
    "policy_drops",
)


@dataclass(frozen=True)
class CapViolations:
    """The age cap's violations in one trial at S (decision 1; other choices 6).

    ``pairs`` counts the (device, mission) pairs, over each device's own
    mule's missions, whose age after the mission is at least ``s``: a device
    the mission merged has age 0, any other ``m − U`` missions since its last
    merged update ``U`` (never merged: ``m``). That is exactly a device capped
    when mission m was planned (age ≥ S, lookahead 0; unit U1's
    ``device_age``) that m did not merge, so the count needs no plan and
    scores every arm alike. ``devices`` counts the devices with any.
    """

    s: int
    pairs: int
    devices: int


def cap_violations(obs: Exp4Observation, devices: Sequence[str], s: int) -> CapViolations:
    """:class:`CapViolations` of ``devices`` at the cap ``s`` (an int ≥ 1)."""
    if isinstance(s, bool) or not isinstance(s, int) or s < 1:
        raise ValueError(f"the age cap S is an int >= 1, got {s!r}")
    devices = list(dict.fromkeys(str(d) for d in devices))
    pairs = 0
    late: set = set()
    for step in _age_walk(obs, devices):
        for d in step.own:
            if step.ages[d] >= s:
                pairs += 1
                late.add(d)
    return CapViolations(s=int(s), pairs=pairs, devices=len(late))


def trace_cap_s(obs: Exp4Observation, mule_cfg: Mapping[str, object]) -> Optional[int]:
    """The age cap S the trace's own mules ran, None when they ran none.

    Read from the missions' plans (``plan.cap.s``), else, for a plan-mode trial
    that recorded no plan (it flew no mission), from its mule config's
    ``age_cap_missions``. Every mule of a trial runs one S, so plans that
    disagree are refused rather than scored at either.
    """
    ran = {m.cap_s for m in obs.missions if m.cap_s is not None}
    if len(ran) > 1:
        raise ValueError(f"the trace's plans ran different age caps: {sorted(ran)}")
    if ran:
        return ran.pop()
    if mule_cfg.get("plan_mode") == PLAN_MODE_FERRY:
        cap = mule_cfg.get("age_cap_missions")
        if isinstance(cap, int) and not isinstance(cap, bool):
            return int(cap)
    return None


def cap_violation_events(obs: Exp4Observation) -> Optional[Dict[str, int]]:
    """The mule's own log of the capped devices it failed, by cause.

    ``plan.cap.violations`` summed over the missions whose plan ran a cap,
    every cause in ``CAP_REASONS`` listed (0 included); None when no mission
    did. At the S the mule ran (``plan.cap.s``), with its lookahead.
    """
    capped = [m for m in obs.missions if m.cap_s is not None]
    if not capped:
        return None
    counts = {reason: 0 for reason in CAP_REASONS}
    for m in capped:
        for _device, _age, reason in m.cap_violations or ():
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def band_shares(obs: Exp4Observation) -> Optional[Dict[str, float]]:
    """The share of Pass-1 stops flown on each band class, over the trial.

    Per stop (``pass_1_flown[].band``), so FX's switches at a stop show, not
    only each mission's committed class (decision 5; Pass 2 always flies the
    committed class b̄). None when no stop was flown on a band.
    """
    counts: Dict[str, int] = {}
    for m in obs.missions:
        for band in m.flown_bands or ():
            if band is not None:
                counts[band] = counts.get(band, 0) + 1
    total = sum(counts.values())
    if not total:
        return None
    return {band: n / total for band, n in sorted(counts.items())}


def plan_means(obs: Exp4Observation) -> Tuple[Optional[float], Optional[float]]:
    """``(served share, V)`` of the committed plans, each averaged over missions.

    The served share is |served| / |demand| in devices (a mission whose
    demand is empty is left out), not the plan's weighted coverage (1 −
    ``score.coverage``): the weights differ between F, F-prio and the uniform
    option, the device count does not. None where no mission recorded a plan.
    """
    shares = [
        len(m.plan_served or ()) / len(m.plan_demand)
        for m in obs.missions if m.has_plan and m.plan_demand
    ]
    values = [m.plan_v for m in obs.missions if m.has_plan and m.plan_v is not None]
    return (
        float(np.mean(shares)) if shares else None,
        float(np.mean(values)) if values else None,
    )


def far_devices(
    positions: Mapping[str, Sequence[float]], rf_range_m: float,
    dock: Sequence[float] = DOCK_POSE,
) -> Tuple[str, ...]:
    """The devices beyond ``rf_range_m`` of the dock, in the plane (Study 5.4's
    far devices: R_planar(wide) is the run's ``rf_range_m``)."""
    return tuple(
        d for d, p in positions.items()
        if math.hypot(float(p[0]) - float(dock[0]), float(p[1]) - float(dock[1]))
        > float(rf_range_m)
    )


def far_served_share(
    obs: Exp4Observation, devices: Sequence[str], far: Iterable[str],
) -> Optional[float]:
    """The far devices' merged updates over their own missions (Study 5.4).

    Each mission of a far device's own mule can collect one update of it at
    most, so the share, Σ merged updates / Σ own missions over the far
    devices, is at most 1. Merged as the ages count it (:func:`_age_walk`):
    an update lost on the backhaul or cut by the merge never counts, and a
    deferred one counts when the merge that flushed it runs. None without a
    far device or a mission.
    """
    devices = list(dict.fromkeys(str(d) for d in devices))
    wanted = {str(d) for d in far} & set(devices)
    merged = missions = 0
    for step in _age_walk(obs, devices):
        missions += sum(1 for d in step.own if d in wanted)
        merged += sum(1 for d in step.merged if d in wanted)
    return merged / missions if missions else None


def policy_drops(obs: Exp4Observation) -> int:
    """The devices a whole-scheduler baseline left out before takeoff, summed
    over missions (``pass_1_policy_drops``; a mission without it left none)."""
    return sum(len(devices) for m in obs.missions for devices, _ in m.policy_drops or ())


@dataclass(frozen=True)
class PlanReport:
    """One trial's :data:`PHASE_4_COLUMNS`; None is a blank column."""

    cap: Optional[CapViolations] = None
    cap_events: Optional[Dict[str, int]] = None
    band_shares: Optional[Dict[str, float]] = None
    plan_served_share_mean: Optional[float] = None
    plan_v_mean: Optional[float] = None
    far_served_share: Optional[float] = None
    policy_drops: Optional[int] = None

    def to_row(self) -> Dict[str, object]:
        cap = self.cap
        return {
            "cap_s": "" if cap is None else cap.s,
            "cap_violations": "" if cap is None else cap.pairs,
            "cap_violation_devices": "" if cap is None else cap.devices,
            "cap_violation_events": _json_or_blank(self.cap_events),
            "band_shares": _json_or_blank(self.band_shares),
            "plan_served_share_mean": _blank(self.plan_served_share_mean),
            "plan_v_mean": _blank(self.plan_v_mean),
            "far_served_share": _blank(self.far_served_share),
            "policy_drops": _blank(self.policy_drops),
        }


def plan_report(
    obs: Exp4Observation,
    devices: Sequence[str],
    *,
    mule_cfg: Mapping[str, object],
    positions: Mapping[str, Sequence[float]],
    age_cap_s: Optional[int] = None,
) -> PlanReport:
    """One trial's FeRRy Phase 4 columns, blank where the trace cannot say.

    The cap columns at ``age_cap_s`` when given (every arm of a study scored
    at one S), else at the trace's own S (:func:`trace_cap_s`), else blank;
    the mule's log by cause wherever its plans ran a cap. A trace whose plans
    ran two caps is refused whether or not ``age_cap_s`` is given: the log
    would add up missions taken at two S. The plan means
    wherever a mission recorded a plan. The band shares, the far devices'
    service and a baseline's drops only on a simulated-clock trace a Phase 4
    build recorded, recognised by its mule config's ``plan_mode`` (every Phase
    4 mule's JSON holds the plan fields, at their defaults too): a trace
    recorded before cannot say whether a baseline dropped anything (D arms
    reported nothing then), and would otherwise gain columns its recorded rows
    never had. ``devices`` and ``positions`` are the trial's (the cluster's
    seed list); the far devices are those beyond the mule config's
    ``rf_range_m`` of the dock.
    """
    # Read even when a study S is given, so its refusal of two caps holds.
    ran = trace_cap_s(obs, mule_cfg)
    s = age_cap_s if age_cap_s is not None else ran
    served, v = plan_means(obs)
    phase_4 = "plan_mode" in mule_cfg and obs.mission_clock == CLOCK_SIM
    far = None
    rf_range_m = _float_or_none(mule_cfg.get("rf_range_m"))
    if phase_4 and rf_range_m is not None and positions:
        far = far_served_share(obs, devices, far_devices(positions, rf_range_m))
    return PlanReport(
        cap=None if s is None else cap_violations(obs, devices, s),
        cap_events=cap_violation_events(obs),
        band_shares=band_shares(obs) if phase_4 else None,
        plan_served_share_mean=served,
        plan_v_mean=v,
        far_served_share=far,
        policy_drops=(policy_drops(obs) if phase_4 and mule_cfg.get("contact_policy")
                      else None),
    )


# --------------------------------------------------------------------------- #
# FeRRy Phase 5: the learned fillings
# --------------------------------------------------------------------------- #

#: The scorer's FeRRy Phase 5 columns (the Phase 5 spec, other choices 9), in
#: row order after the τ columns, and only with ``pair_columns``
#: (``--pair-columns``): without it the row is the Phase 4 one, so no Phase 4
#: pin moves (critic B1). Scorer-only: the trial CSV's header is unchanged.
PHASE_5_COLUMNS = (
    "pair_decisions", "pair_feasible_mean", "pair_mask_empty", "pair_fx_agree_share",
    "pair_band_off_bbar_share", "pair_reorder_share", "e3_unvisited_mean",
)


@dataclass(frozen=True)
class PairReport:
    """One trial's :data:`PHASE_5_COLUMNS`; None is a blank column."""

    pair_decisions: Optional[int] = None
    pair_feasible_mean: Optional[float] = None
    pair_mask_empty: Optional[int] = None
    pair_fx_agree_share: Optional[float] = None
    pair_band_off_bbar_share: Optional[float] = None
    pair_reorder_share: Optional[float] = None
    e3_unvisited_mean: Optional[float] = None

    def to_row(self) -> Dict[str, object]:
        return {col: _blank(getattr(self, col)) for col in PHASE_5_COLUMNS}


def pair_report(obs: Exp4Observation, *, mule_cfg: Mapping[str, object]) -> PairReport:
    """One trial's FeRRy Phase 5 columns, blank where it flew no learned filling.

    The pair columns pool every decision of every mission (one per Pass-1 stop
    the pair slot flew; with several mules, every mule's): their count; the
    mean number of pairs the mask admitted; how many found none, so that the
    slot fell back on FX's pair (``fallback``); and the shares whose chosen
    pair was FX's by FX's own rule at that arrival (``agrees_fx``: FX's rule,
    not the FX arm's flight, which under ``replan`` re-plans first; unit U3;
    a decision whose mask was empty fell back on FX's pair, so it agrees),
    that served the stop on a class other than the committed b̄, and whose
    pair chose a stop other than the remainder's head (``next_index``
    neither 0 nor home). The agreement and re-order shares count the choice,
    not the flight: the stop is served on the chosen band at once, but the
    departure check after it may re-plan the order the pair set
    (``trimmed_next``) and the beacon hook may insert a stop ahead of it. A
    trial flew the pair slot when its mule config names it or any of its
    missions recorded a decision: FerrySim installs its slots on FX's
    configuration (``MuleSupervisor.install_flight_slot``), so its traces
    name FX's slot.
    The mule writes ``pass_1_pairs`` only for a mission that decided
    something, so on such a trial a mission without it made no decision: with
    none at all the two counts are 0 and the means and shares, with nothing to
    average, blank. A share or mean leaves out a decision that does not hold
    its field in the form the mule writes, and a record whose choice is not
    in that form is no decision (the consumer's ``PairDecision``); no record
    the mule writes is either. On any other trial every pair column is blank.

    ``e3_unvisited_mean`` is the mean over the trial's missions of the stops
    arm E3's pass left when no stop was admissible any more
    (``pass_1_e3_unvisited``, reported and never widened; a mission without
    it left none), on a trial whose mule config names E3's policy or whose
    missions recorded E3's calls; blank on any other, and with no mission.
    """
    missions = obs.missions
    unvisited = None
    if missions and (mule_cfg.get("contact_policy") == CONTACT_POLICY_CHEN_DQN or any(
            m.e3_calls is not None or m.e3_unvisited is not None for m in missions)):
        unvisited = float(np.mean([len(m.e3_unvisited or ()) for m in missions]))
    if not (mule_cfg.get("flight_slot") == FLIGHT_SLOT_PAIR_Q
            or any(m.pair_decisions is not None for m in missions)):
        return PairReport(e3_unvisited_mean=unvisited)
    decisions = [d for m in missions for d in m.pair_decisions or ()]
    return PairReport(
        pair_decisions=len(decisions),
        pair_feasible_mean=_mean_present(d.feasible for d in decisions),
        pair_mask_empty=sum(1 for d in decisions if d.mask_empty),
        pair_fx_agree_share=_mean_present(d.agrees_fx for d in decisions),
        pair_band_off_bbar_share=_mean_present(
            None if d.committed is None else d.band != d.committed for d in decisions),
        pair_reorder_share=_mean_present(d.reorders for d in decisions),
        e3_unvisited_mean=unvisited,
    )


# --------------------------------------------------------------------------- #
# Exp 5 addendum, Study 5.11 (a): the decision cost
# --------------------------------------------------------------------------- #

#: The scorer's Study 5.11 columns (the build plan's addendum of 2 Oct 2026,
#: metric "Decision cost"), in row order after the τ columns and the pair
#: columns, and only with ``cost_columns`` (``--cost-columns``): without it the
#: row is the Phase 5 one, so no pin moves. Wall times, so none enters a
#: determinism comparison. Scorer-only: the trial CSV's header is unchanged.
COST_COLUMNS = (
    "plan_wall_s_mean", "plan_wall_s_p95", "plan_search_shares",
    "pair_wall_s_mean", "pair_wall_s_p95", "pair_mask_wall_s_mean",
    "e3_wall_s_mean", "e3_wall_s_p95", "e3_mask_wall_s_mean",
    "flight_decisions_per_mission",
    "trial_processes", "peak_rss_mib_total",
    "peak_rss_mib_cluster", "peak_rss_mib_mule", "peak_rss_mib_device",
)

_MIB = float(1 << 20)

#: The scorer's Study 5.13 columns (the build plan's addendum, metric
#: "Detection quality"), in row order after the τ, pair and cost columns, and
#: only with ``detection_columns`` (``--detection-columns``). Scorer-only.
DETECTION_COLUMNS = (
    "data_partition", "data_alpha", "network_aou_shard_weighted_mean",
    "tpr_final", "fpr_final", "precision_final", "f1_final",
    "recall_family_min", "recall_by_family_final",
)


@dataclass(frozen=True)
class CostReport:
    """One trial's :data:`COST_COLUMNS`; None is a blank column."""

    plan_wall_s_mean: Optional[float] = None
    plan_wall_s_p95: Optional[float] = None
    plan_search_shares: Optional[Dict[str, float]] = None
    pair_wall_s_mean: Optional[float] = None
    pair_wall_s_p95: Optional[float] = None
    pair_mask_wall_s_mean: Optional[float] = None
    e3_wall_s_mean: Optional[float] = None
    e3_wall_s_p95: Optional[float] = None
    e3_mask_wall_s_mean: Optional[float] = None
    flight_decisions_per_mission: Optional[float] = None
    trial_processes: Optional[int] = None
    peak_rss_mib_total: Optional[float] = None
    peak_rss_mib_cluster: Optional[float] = None
    peak_rss_mib_mule: Optional[float] = None
    peak_rss_mib_device: Optional[float] = None

    def to_row(self) -> Dict[str, object]:
        row = {col: _blank(getattr(self, col)) for col in COST_COLUMNS}
        row["plan_search_shares"] = _json_or_blank(self.plan_search_shares)
        return row


def _p95(values: Sequence[float]) -> Optional[float]:
    """The 95th percentile (numpy's linear rule, as ``age_p95``); None if empty."""
    return float(np.percentile(values, 95)) if values else None


def _mib(value) -> Optional[float]:
    """A byte count from ``footprint.json`` in MiB; None unless an int >= 0."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value / _MIB


def cost_report(obs: Exp4Observation, *, mule_cfg: Mapping[str, object],
                footprint: Optional[Mapping[str, object]] = None) -> CostReport:
    """One trial's decision cost (Exp 5 addendum, Study 5.11 (a)), blank where it has none.

    **The planner.** ``plan_wall_s_mean`` and ``plan_wall_s_p95`` are over the
    trial's missions that recorded ``plan_wall_s`` (plan mode: the wall
    seconds ``FLScheduler.build_ferry_plan`` took at the dock, every class's
    search included), with several mules every mule's; blank with none.
    ``plan_search_shares`` is the share of the missions that recorded a plan
    whose committed class ran each mode (``plan.search``: ``exact``,
    ``stop_subsets`` or ``local``, every one listed, 0 included; JSON), so a
    sweep that forces a mode through ``--plan-search-params`` can check it
    held; blank with no plan.

    **The flight clock.** The pair slot's decisions and E3's calls, pooled
    over the trial's missions and mules (``pass_1_pairs_wall``,
    ``pass_1_e3_wall``): the mean and 95th percentile of each decision's
    ``decide_s`` (the mask's predicate, the scorer or policy and the pick)
    and the mean of its ``mask_s`` (the predicate's share), each blank where
    the trial recorded none; a time the mule did not write as a number >= 0
    is left out. ``flight_decisions_per_mission`` is the mean over the
    trial's missions of the decisions timed in flight, both kinds together,
    on a trial whose mule config names the pair slot or E3's policy or whose
    missions recorded a wall (a mission without one made none); blank on any
    other trial, and with no mission. The view or observation each decision
    reads is built before it and is not timed; FX's and F's fixed fillings
    record no decision and so no wall.

    **The footprint** (``footprint``, the trial's ``footprint.json`` as
    :func:`experiments.exp4.footprint.read_footprint` returns it; None for a
    trial run without the probe, when all five are blank):
    ``trial_processes``, the processes the trial started (the cluster, the
    mules, the devices); ``peak_rss_mib_total``, the largest summed resident
    memory over one sample, the concurrent peak; and per role the largest
    process's own peak (``peak_rss_mib_cluster``, ``_mule``, ``_device``).
    In MiB; a value the probe did not write as a byte count is blank.
    """
    missions = obs.missions
    plan_walls = [m.plan_wall_s for m in missions if m.plan_wall_s is not None]
    searched = [m.plan_search for m in missions if m.has_plan]
    shares = None
    if searched:
        shares = {mode: sum(1 for s in searched if s == mode) / len(searched)
                  for mode in SEARCH_MODES}

    def walls(kind: str) -> List:
        return [w for m in missions for w in getattr(m, kind) or ()]

    pair, e3 = walls("pair_walls"), walls("e3_walls")

    def decided(ws) -> List[float]:
        return [w.decide_s for w in ws if w.decide_s is not None]

    per_mission = None
    if missions and (mule_cfg.get("flight_slot") == FLIGHT_SLOT_PAIR_Q
                     or mule_cfg.get("contact_policy") == CONTACT_POLICY_CHEN_DQN
                     or pair or e3):
        per_mission = float(np.mean([len(m.pair_walls or ()) + len(m.e3_walls or ())
                                     for m in missions]))
    return CostReport(
        plan_wall_s_mean=_mean_present(plan_walls),
        plan_wall_s_p95=_p95(plan_walls),
        plan_search_shares=shares,
        pair_wall_s_mean=_mean_present(decided(pair)),
        pair_wall_s_p95=_p95(decided(pair)),
        pair_mask_wall_s_mean=_mean_present(w.mask_s for w in pair),
        e3_wall_s_mean=_mean_present(decided(e3)),
        e3_wall_s_p95=_p95(decided(e3)),
        e3_mask_wall_s_mean=_mean_present(w.mask_s for w in e3),
        flight_decisions_per_mission=per_mission,
        **_footprint_fields(footprint),
    )


def _footprint_fields(footprint: Optional[Mapping[str, object]]) -> Dict[str, object]:
    """:func:`cost_report`'s five footprint fields from ``footprint.json``."""
    if not isinstance(footprint, Mapping):
        return {}
    processes = footprint.get("processes")
    by_role = footprint.get("peak_rss_bytes_by_role")
    by_role = by_role if isinstance(by_role, Mapping) else {}
    return {
        "trial_processes": (processes if isinstance(processes, int)
                            and not isinstance(processes, bool) and processes >= 0 else None),
        "peak_rss_mib_total": _mib(footprint.get("peak_rss_bytes_total")),
        "peak_rss_mib_cluster": _mib(by_role.get("cluster")),
        "peak_rss_mib_mule": _mib(by_role.get("mule")),
        "peak_rss_mib_device": _mib(by_role.get("device")),
    }


@dataclass(frozen=True)
class DetectionReport:
    """One trial's :data:`DETECTION_COLUMNS`; None is a blank column."""

    data_partition: Optional[str] = None
    data_alpha: Optional[float] = None
    network_aou_shard_weighted_mean: Optional[float] = None
    tpr_final: Optional[float] = None
    fpr_final: Optional[float] = None
    precision_final: Optional[float] = None
    f1_final: Optional[float] = None
    recall_family_min: Optional[float] = None
    recall_by_family_final: Optional[Dict[str, float]] = None

    def to_row(self) -> Dict[str, object]:
        row = {col: _blank(getattr(self, col)) for col in DETECTION_COLUMNS}
        row["recall_by_family_final"] = _json_or_blank(self.recall_by_family_final)
        return row


def detection_report(obs: Exp4Observation, devices: Sequence[str],
                     marker: Mapping[str, object]) -> DetectionReport:
    """One trial's data and detector columns (Exp 5 addendum, Study 5.13).

    **The data** (the status marker's ``data``, written only by a trial whose
    partition or family labels were not the recorded ones; blank otherwise):
    ``data_partition`` and ``data_alpha``, and
    ``network_aou_shard_weighted_mean``, the Network AoU with each device
    weighted by its shard's rows (``data.shard_rows``; :func:`age_profile`),
    so under quantity skew a large shard left stale counts for more. Blank
    when the marker does not name every device's rows.

    **The detector** (the last ``model_eval`` that carries ``detection``,
    which the cluster writes when the test set carries the attack families;
    blank without one): ``tpr_final``, ``fpr_final``, ``precision_final``,
    ``f1_final``; ``recall_by_family_final`` (JSON, every family the test set
    holds, Benign's being 1 - FPR) and ``recall_family_min``, the worst attack
    family's (Benign left out). The final accuracy and AUC are the summary's
    own columns.
    """
    data = marker.get("data") if isinstance(marker.get("data"), Mapping) else {}
    partition = data.get("partition") if isinstance(data.get("partition"), str) else None
    alpha = _float_or_none(data.get("dirichlet_alpha"))
    weighted = None
    rows = data.get("shard_rows")
    if isinstance(rows, Mapping) and devices and all(
            isinstance(rows.get(d), int) and not isinstance(rows.get(d), bool)
            for d in devices) and sum(rows[d] for d in devices) > 0:
        weighted = age_profile(obs, devices, weights={d: float(rows[d]) for d in devices}
                               ).network_aou_mean
    final = next((e.detection for e in reversed(obs.model_evals)
                  if e.detection is not None), None)
    if final is None:
        return DetectionReport(data_partition=partition, data_alpha=alpha,
                               network_aou_shard_weighted_mean=weighted)
    attacks = [v for k, v in final.recall_by_family.items() if k != "Benign"]
    return DetectionReport(
        data_partition=partition, data_alpha=alpha,
        network_aou_shard_weighted_mean=weighted,
        tpr_final=final.tpr, fpr_final=final.fpr, precision_final=final.precision,
        f1_final=final.f1, recall_family_min=min(attacks) if attacks else None,
        recall_by_family_final=dict(final.recall_by_family) or None,
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
    #: FeRRy Phase 4: :data:`PHASE_4_COLUMNS` (:func:`plan_report`), blank by
    #: default.
    plan: PlanReport = field(default_factory=PlanReport)
    #: FeRRy Phase 5: :data:`PHASE_5_COLUMNS` (:func:`pair_report`), after the
    #: τ columns, when scored with ``pair_columns``; None leaves them out, so
    #: the default row is the Phase 4 one.
    pairs: Optional[PairReport] = None
    #: Exp 5 addendum, Study 5.11 (a): :data:`COST_COLUMNS`
    #: (:func:`cost_report`), after the τ and pair columns, when scored with
    #: ``cost_columns``; None leaves them out.
    costs: Optional[CostReport] = None
    #: Exp 5 addendum, Study 5.13: :data:`DETECTION_COLUMNS`
    #: (:func:`detection_report`), last, when scored with
    #: ``detection_columns``; None leaves them out.
    detection: Optional[DetectionReport] = None

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
        row.update(self.plan.to_row())
        for reach in self.tau:
            tag = _tau_tag(reach.tau)
            row[f"reached_{tag}"] = int(reach.reached)
            # The reaching mule's missions (mission periods), comparable across
            # fleet sizes; rounds_to_ counts cluster rounds, about K per period.
            row[f"missions_to_{tag}"] = _blank(reach.mission)
            row[f"rounds_to_{tag}"] = _blank(reach.cluster_round)
            row[f"wall_s_to_{tag}"] = _blank(reach.wall_s)
            # FeRRy Phase 3: simulated seconds, on the mission clock only.
            row[f"sim_s_to_{tag}"] = _blank(reach.sim_s)
        if self.pairs is not None:
            row.update(self.pairs.to_row())
        if self.costs is not None:
            row.update(self.costs.to_row())
        if self.detection is not None:
            row.update(self.detection.to_row())
        return row


def score_trial(
    trace_dir,
    *,
    taus: Sequence[float] = (0.82,),
    weights: Optional[Mapping[str, float]] = None,
    status_csv: StatusSource = None,
    age_cap_s: Optional[int] = None,
    pair_columns: bool = False,
    cost_columns: bool = False,
    detection_columns: bool = False,
) -> TrialScore:
    """Score one retained trial directory.

    ``status_csv`` (a trial CSV's path, or :func:`load_status_csv` of one)
    supplies the status of a trace that carries no marker, and overrides the
    marker's ``ok`` when it records a failure (see :func:`trial_status`).
    ``age_cap_s`` is the S the cap violations are counted at; None counts
    them at the trace's own S, if its mules ran one (:func:`plan_report`).
    ``pair_columns`` adds :data:`PHASE_5_COLUMNS` after the τ columns
    (:func:`pair_report`); without it the row has none of them.
    ``cost_columns`` adds :data:`COST_COLUMNS` after those
    (:func:`cost_report`), and ``detection_columns`` :data:`DETECTION_COLUMNS`
    last (:func:`detection_report`); without them the row has none of them.

    Raises :class:`~experiments.exp4.events_consumer.ClockDomainError`,
    naming the trial, for a trace whose clocks disagree: in its events (see
    :func:`~experiments.exp4.events_consumer.trace_clock_domain`), or between
    the clock its mules announced and the one their configs set, and
    ``ValueError``, naming it too, for an ``age_cap_s`` that is not an int
    >= 1 or plans that ran two caps (with or without ``age_cap_s``).
    """
    trace_dir = Path(trace_dir)
    if not taus:
        raise ValueError("score_trial needs at least one tau")
    key = parse_trial_dir(trace_dir.name)
    devices = _device_ids(trace_dir)
    try:
        obs = consume_run_dir(trace_dir, n_devices=len(devices))
    except ClockDomainError as e:
        raise ClockDomainError(f"{trace_dir.name}: {e}") from e
    if not devices:
        devices = sorted(obs.per_device_serves)
        obs.n_devices = len(devices)

    # Every mule of a trial runs the same scheduler settings, so the first
    # config speaks for the fleet, as it does for the provenance.
    mule_cfg = _first_json(trace_dir, "mule-*.json")
    configured = str(mule_cfg.get("mission_clock") or CLOCK_WALL)
    if obs.mule_ready and configured != obs.mission_clock:
        raise ClockDomainError(
            f"{trace_dir.name}: clock domains disagree: the mule config sets "
            f"mission_clock {configured!r}, but its mule_ready announced "
            f"{obs.mission_clock!r}"
        )
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
        plan=_trial_plan_report(trace_dir, obs, devices, mule_cfg, age_cap_s),
        pairs=pair_report(obs, mule_cfg=mule_cfg) if pair_columns else None,
        costs=(cost_report(obs, mule_cfg=mule_cfg, footprint=read_footprint(trace_dir))
               if cost_columns else None),
        detection=(detection_report(obs, devices, _read_json(trace_dir / TRIAL_STATUS_FILE))
                   if detection_columns else None),
    )


def _trial_plan_report(
    trace_dir: Path, obs: Exp4Observation, devices: Sequence[str],
    mule_cfg: Mapping[str, object], age_cap_s: Optional[int],
) -> PlanReport:
    """:func:`plan_report` of one trial directory; a trace it refuses (plans
    that ran two caps) is named, as a clock-domain refusal is."""
    try:
        return plan_report(obs, devices, mule_cfg=mule_cfg,
                           positions=_device_positions(trace_dir), age_cap_s=age_cap_s)
    except ValueError as e:
        raise ValueError(f"{trace_dir.name}: {e}") from e


def score_traces(
    trace_root,
    *,
    taus: Sequence[float] = (0.82,),
    arms: Optional[Iterable[str]] = None,
    include_failed: bool = False,
    status_csv: StatusSource = None,
    age_cap_s: Optional[int] = None,
    pair_columns: bool = False,
    cost_columns: bool = False,
    detection_columns: bool = False,
) -> List[TrialScore]:
    """Score every trial directory under ``trace_root``, in name order.

    Directories whose names are not trial names are skipped, and so, unless
    ``include_failed``, are trials whose status is not ``ok`` (see
    :func:`trial_status`): a timed-out or ``no_eval`` trial is not a valid
    observation, and the trial CSV's analysis drops it too. ``age_cap_s``,
    ``pair_columns``, ``cost_columns`` and ``detection_columns`` as in
    :func:`score_trial`.
    """
    index = _status_index(status_csv)
    scores: List[TrialScore] = []
    for d, _ in _trial_dirs(trace_root, arms):
        if not include_failed and trial_status(d, index).status != "ok":
            continue
        scores.append(score_trial(d, taus=taus, status_csv=index, age_cap_s=age_cap_s,
                                  pair_columns=pair_columns, cost_columns=cost_columns,
                                  detection_columns=detection_columns))
    return scores


def write_scores_csv(scores: Sequence[TrialScore], path) -> None:
    """One row per trial. The column set is fixed by the first score's τ list
    and whether it carries the pair, cost and detection columns."""
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
    ap.add_argument("--age-cap-s", type=int, default=None,
                    help="Count cap violations at this age cap S (missions since a "
                         "device's last merged update) for every arm; default: each "
                         "trace's own S, and blank for a trace whose mules ran none.")
    ap.add_argument("--pair-columns", action="store_true",
                    help="Add the FeRRy Phase 5 columns after the tau columns: the pair "
                         "slot's decisions (how many, the pairs its mask admitted, empty "
                         "masks, and the shares that chose FX's pair, served off the "
                         "committed class and chose a stop other than the remainder's "
                         "head) and E3's unvisited stops per mission. Off by default, "
                         "when the row is the Phase 4 one.")
    ap.add_argument("--cost-columns", action="store_true",
                    help="Add Study 5.11's decision-cost columns after the tau and pair "
                         "columns: the planner's wall time per mission (mean, p95) and "
                         "the share of missions in each search mode, and the pair "
                         "slot's and E3's wall time per decision (mean, p95, the mask's "
                         "mean) with the timed decisions per mission; and the trial's "
                         "processes and peak memory where the runner's --footprint-probe "
                         "recorded them. Wall times: off by default, when the row is the "
                         "one without them.")
    ap.add_argument("--detection-columns", action="store_true",
                    help="Add Study 5.13's columns last: the trial's partition and alpha "
                         "and the Network AoU weighted by shard size (from the status "
                         "marker of a non-IID or family-labelled trial), and the final "
                         "TPR, FPR, precision, F1 and recall per attack family (from "
                         "model_eval's detection). Off by default.")
    args = ap.parse_args(argv)
    if args.age_cap_s is not None and args.age_cap_s < 1:
        ap.error(f"--age-cap-s must be >= 1, got {args.age_cap_s}")
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
            age_cap_s=args.age_cap_s, pair_columns=args.pair_columns,
            cost_columns=args.cost_columns, detection_columns=args.detection_columns,
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
    if args.age_cap_s is not None:
        _warn_other_caps(scores, args.age_cap_s)
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


def _warn_other_caps(scores: Sequence[TrialScore], age_cap_s: int) -> None:
    """Flag trials whose mules ran another cap than ``--age-cap-s``.

    Their ``cap_violations`` count at ``--age-cap-s``, but
    ``cap_violation_events`` is the mule's own log at the S it ran, so the two
    columns of such a row do not compare.
    """
    for s in scores:
        ran = _cap_in_provenance(s.provenance)
        if ran is not None and ran != age_cap_s:
            k = s.key
            print(f"warning: {k.cell_id} {k.arm} t{k.trial_index} s{k.seed} ran the age "
                  f"cap S = {ran}; its violations are counted at --age-cap-s {age_cap_s}, "
                  f"its cap_violation_events at S = {ran}")


def _cap_in_provenance(provenance: Mapping[str, object]) -> Optional[int]:
    """The ``age_cap_missions`` a plan-mode row's ``ferry_params`` records."""
    try:
        params = json.loads(str(provenance.get("ferry_params") or "{}"))
    except json.JSONDecodeError:
        return None
    cap = params.get("age_cap_missions") if isinstance(params, dict) else None
    return cap if isinstance(cap, int) and not isinstance(cap, bool) else None


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
    for m in _ordered(obs.missions, obs.mission_clock):
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
    starts = [m.started_ts for m in _first_missions(obs, missions) if m.started_ts is not None]
    return min(starts) if starts else None


def _fleet_sim_start(obs: Exp4Observation, missions: Sequence[MissionRecord]) -> Optional[float]:
    """:func:`_fleet_start` on the simulated mission clock: the earliest
    simulated takeoff among each mule's first mission."""
    starts = [m.sim_start_s for m in _first_missions(obs, missions) if m.sim_start_s is not None]
    return min(starts) if starts else None


def _first_missions(
    obs: Exp4Observation, missions: Sequence[MissionRecord],
) -> List[MissionRecord]:
    """Each mule's first mission of ``missions`` (in completion order)."""
    firsts: Dict[Optional[str], MissionRecord] = {}
    for m in missions:
        firsts.setdefault(obs.mule_key(m.mule_id), m)
    return list(firsts.values())


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
    """Whether the marker's run time exceeds the runner's soft cap: the
    ``soft_cap_s`` it records (mission-clock trials run by ``runner_main``),
    else its trial budget, the runner's default cap on the wall clock. False
    for a marker that records no run time or no cap."""
    try:
        cap = marker["soft_cap_s"] if "soft_cap_s" in marker else marker["trial_budget_s"]
        return float(marker["run_s"]) > float(cap)
    except (KeyError, TypeError, ValueError):
        return False


def _ordered(
    missions: Sequence[MissionRecord], clock: str = CLOCK_WALL,
) -> List[MissionRecord]:
    """Missions in completion order on ``clock``; the recorded order when
    timestamps are absent (:func:`~experiments.exp4.events_consumer.completion_order`:
    on the simulated clock, the missions' simulated ends)."""
    return completion_order(missions, clock)


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


def _device_positions(trace_dir: Path) -> Dict[str, Tuple[float, ...]]:
    """Each device's position, from the cluster's seed list, else the device
    configs (the same sources as :func:`_device_ids`); a device whose position
    neither records is left out."""
    positions: Dict[str, Tuple[float, ...]] = {}
    for s in _first_json(trace_dir, "cluster*.json").get("seed_devices") or ():
        if isinstance(s, dict) and "device_id" in s and _is_position(s.get("position")):
            positions[str(s["device_id"])] = tuple(float(c) for c in s["position"])
    if positions:
        return positions
    for p in sorted(trace_dir.glob("device-*.json")):
        cfg = _read_json(p)
        if _is_position(cfg.get("position")):
            did = str(cfg.get("device_id") or p.stem[len("device-"):])
            positions[did] = tuple(float(c) for c in cfg["position"])
    return positions


def _is_position(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) >= 2
            and all(_float_or_none(c) is not None for c in v))


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


def _mean_present(values: Iterable[object]) -> Optional[float]:
    """The mean of the values that are not None (a bool as 1 or 0); None if none are."""
    present = [float(v) for v in values if v is not None]
    return float(np.mean(present)) if present else None


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


def _json_or_blank(v) -> str:
    """A mapping column as JSON with sorted keys, as the driver writes its own;
    blank for None."""
    return "" if v is None else json.dumps(v, sort_keys=True)


def _float_or_none(v) -> Optional[float]:
    """A config value as a float; None when it is absent, null or not a number."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
