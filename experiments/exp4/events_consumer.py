"""JSONL event-stream consumer for Experiment 4 (chunk EX-4.0).

The multi-process orchestrator writes one JSONL file per role under the
run dir (``cluster-<id>.jsonl``, ``mule-<id>.jsonl``,
``device-<id>.jsonl``). Each line is one event envelope::

    {"ts": ..., "schema_version": 1, "role": "mule", "id": "...",
     "event": "mission_completed", ...payload}

This module folds those three streams into one :class:`Exp4Observation`
— the structured, per-trial view the metric layer rolls up. It reads
only *already-flushed per-event lines* (the emitter is line-buffered),
never the end-of-run ``metrics_snapshot``, so it is robust to a hard
``TerminateProcess`` shutdown on Windows that skips the cluster's
``finally`` block.

The parsing is split into a pure :func:`observation_from_rows` (row
dicts → observation, unit-testable without spawning anything) and a thin
:func:`consume_run_dir` that reads the files first.

Several mules (FeRRy Phase 2): each mule numbers its own missions from 1,
so a mission round alone no longer names a mission. The observation keys
its per-mission ledger by :data:`MissionKey` ``(mule_id, mission_round)``;
the round-keyed fields every single-mule caller reads are that ledger's
projection onto rounds, which is exact while one mule flies. Several mules
also let an upload reach the cluster and still never reach θ: a mule that
gave up waiting for its quorum's DOWN uploads again while its first partial
still sits in the open round, and the cluster refuses the second as a
duplicate, though it logs its ingest like any other. The ledger replays the
cluster's open round to find those (see :func:`_unmerged_uploads`).

FeRRy Phase 3 — the simulated mission clock. A mule on it says so in
``mule_ready.mission_clock`` ("sim"); the field is absent on the wall clock,
so every trace recorded before Phase 3 reads as wall-clock. The envelope
``ts`` stays wall time on either clock, so it still places cluster events in
the missions' (wall) windows exactly as before. On the simulated clock each
``mission_completed`` also carries the mission's simulated record — its start
and end (``sim_start_s``, ``sim_end_s``), the clock's per-kind ledger, the
SIMULATED energy, the budget overrun, and the re-plans, aborts and beacon
inserts — and each ``model_eval`` the cluster's simulated time (``sim_ts``,
the latest simulated upload it has ingested). The Pass-1 plan's deadlines and
the sessions' ``contact_ts`` keep their names and switch to simulated seconds
together. A trace whose clocks disagree is refused with
:class:`ClockDomainError` rather than scored (:func:`trace_clock_domain`): a
simulated ``contact_ts`` held to a wall-clock deadline would score every
device on time, and the reverse every device late. The boundary is the
mission clock's ceiling, ``SIM_CEILING_S`` = 1e9 s, which no simulated clock
reaches and every wall stamp since 2001 exceeds.

Targeted solicits and the device-side counts (critic B13). ``per_device_serves``
counts ``device_served`` events, one per solicit a device answered and then
saw through to an outcome. On the wall clock every solicit is a broadcast:
every eligible device answers it, and one that is not a member of the contact
times out waiting for a push, which still counts as a serve. On the
simulated clock the mule solicits only the contact's members, so those
non-member timeouts never happen, and the serve counts (and the coverage and
Jain's index computed from them) count member contacts only. Ferry and
wall-clock rows do not compare on those figures.

FeRRy Phase 4 — the plan clock (the Phase 4 spec, other choices 12). A mission
in plan mode records its closed commit in ``mission_completed.plan`` (the
class b̄ it committed to, the demand, the devices it serves, V and the age
cap's S, capped set and violations) and the plan's wall time beside it
(``plan_wall_s``); every simulated-clock mission records the class each
Pass-1 stop was flown on (``pass_1_flown[].band``), and a whole-scheduler
baseline what it left out before takeoff (``pass_1_policy_drops``, only when
it left something out). :class:`MissionRecord` carries them all, None where a
mission does not record them, so a trace recorded before Phase 4 reads as it
did.

FeRRy Phase 5 — the flight clock's learned fillings (the Phase 5 spec, other
choices 9). A mission flown with the pair slot (``flight_slot = pair_q``)
records its closed decisions in ``mission_completed.pass_1_pairs``, one per
Pass-1 stop flown: the (band, next stop) pair chosen at the arrival, what the
mask admitted, FX's pair at that arrival and what the stop collected, with no
wall time (:class:`PairDecision`). Arm E3 (Chen et al.'s DQN) records each of
its next-stop calls (``pass_1_e3``, :class:`E3Call`) and the stops its pass
left when none was admissible any more (``pass_1_e3_unvisited``). The mule
writes each only when it is not empty, so :class:`MissionRecord` reads them as
None where a mission does not record them, and a trace recorded before Phase 5
reads as it did.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from hermes.l1.mission_clock import SIM_CEILING_S
from hermes.processes.config import CLOCK_SIM, CLOCK_WALL


#: A mission's identity in a trial: ``(mule_id, mission_round)``. The mule half
#: is normalised by :meth:`Exp4Observation.mule_key`, so with one mule every
#: key carries the same id (or None) whatever a row called it.
MissionKey = Tuple[Optional[str], int]


class ClockDomainError(ValueError):
    """A trace whose mission-time stamps come from two clocks (FeRRy Phase 3).

    Raised instead of scoring it: every comparison of a simulated stamp with a
    wall one (a contact against its deadline, a mission against another) is
    meaningless, and would read as 0 % or 100 % rather than as an error.
    """


# --------------------------------------------------------------------------- #
# Structured observation
# --------------------------------------------------------------------------- #

#: ``pass_1_pairs[].fallback`` when no pair fitted and the slot chose FX's pair
#: (``hermes.scheduler.plan.types.PAIR_FALLBACK_MASK_EMPTY``), restated here
#: because scoring loads no plan module.
PAIR_MASK_EMPTY = "mask_empty"


@dataclass(frozen=True)
class PairDecision:
    """One closed decision of the pair slot (FeRRy Phase 5, ``flight_slot = pair_q``).

    Read from ``mission_completed.pass_1_pairs[]``, the record the mule closes
    at the mission's end (``hermes.scheduler.policies.pair_slot``,
    ``DECISION_KEYS`` and ``CLOSE_KEYS``): at a Pass-1 arrival the slot chose a
    pair, the class the stop is served on, at once, and the stop to fly next,
    which the supervisor then moves to the front of the remainder. That is
    the choice, not always the flight: the departure check after the stop
    folds the new order and may re-plan it (``trimmed_next``), and the beacon
    hook may insert a stop ahead of it.

    Each field is read only in the JSON form the mule writes it in: a number
    is a finite int or float, an index or a count an int >= 0, a flag a bool,
    a name a string, an id list a list of strings. A field in any other form
    (a bool or a float as an index, a numeric string, a bool as a number)
    reads as None, a list as empty, so none is read as a value of another
    kind. A record whose choice is in another form (``band`` not a string, or
    ``next_index`` neither an int >= 0 nor None for home) holds no decision
    and is skipped, so an index that cannot be read never reads as home; for
    the same reason FX's pair is read whole, and an ``admitted_pairs`` entry
    that cannot be read is left out. Values are not checked against each
    other: the slot checks its records as it writes them (``closed_record``).
    """

    #: The arrival, simulated seconds.
    t_s: Optional[float]
    #: The stop's members, and b̄, the class the mission committed to.
    devices: Tuple[str, ...]
    committed: Optional[str]
    #: The pair chosen: the class the stop was served on, and the next stop,
    #: by its index in the remainder as it stood (0 keeps that order; None is
    #: home, offered only once no stop is left) and by its members (None for
    #: home).
    band: str
    next_index: Optional[int]
    next_devices: Optional[Tuple[str, ...]]
    #: How many pairs were offered and how many the mask admitted, and those as
    #: ``(band, index)`` in row order (index None for home).
    pairs: Optional[int]
    feasible: Optional[int]
    admitted_pairs: Tuple[Tuple[str, Optional[int]], ...]
    #: :data:`PAIR_MASK_EMPTY` when the mask admitted no pair, else None.
    fallback: Optional[str]
    #: FX's pair by FX's own rule at this arrival (both halves None when the
    #: record does not hold it so), and whether the pair chosen is it.
    fx_band: Optional[str]
    fx_next: Optional[int]
    agrees_fx: Optional[bool]
    #: The scorer that ranked the pairs (``pair_v1`` for the learned score, else
    #: a scripted reference's name), and its Q for the pair chosen and for FX's
    #: (None for a scorer without Q values).
    scorer: Optional[str]
    q: Optional[float]
    q_fx: Optional[float]
    #: Closed at the mission's end: the members collected CLEAN at the stop,
    #: their raw L3 weights in that order, those collected after their own
    #: deadline, the next decision's arrival (the end of the sortie for its
    #: last decision), whether it was the sortie's last decision, and whether
    #: the departure check after the stop re-planned the order the pair set
    #: (under ``abort``, gave up the pass).
    collected: Tuple[str, ...]
    w: Tuple[float, ...]
    late: Tuple[str, ...]
    t_next_s: Optional[float]
    terminal: Optional[bool]
    trimmed_next: Optional[bool]

    @property
    def mask_empty(self) -> bool:
        """Whether the mask admitted no pair, so the slot fell back on FX's pair
        (FX's band, with no reorder)."""
        return self.fallback == PAIR_MASK_EMPTY

    @property
    def reorders(self) -> bool:
        """Whether the pair chose a stop other than the remainder's head
        (``next_index`` neither 0 nor home); what flew next can differ
        (``trimmed_next``, a beacon insert)."""
        return self.next_index not in (0, None)


@dataclass(frozen=True)
class E3Call:
    """One next-stop call of arm E3 (FeRRy Phase 5), from ``mission_completed.pass_1_e3[]``.

    At takeoff and at every Pass-1 departure the mule asks Chen et al.'s DQN
    which stop of those left to fly next, among the ones S3b's single-contact
    check admits (the return and the upload included, no deadline); None flies
    home and ends the pass. Read as :class:`PairDecision` is: each field only
    in the JSON form the mule writes it in, else None (a list empty), and a
    call whose choice (``next_index``: an int >= 0, or None for home) is in
    another form is skipped.
    """

    #: When the call was made (simulated seconds), and whether after a stop
    #: (False at takeoff).
    t_s: Optional[float]
    after_stop: Optional[bool]
    #: The stops left, each as its members, and S3b's verdict on each.
    stops: Tuple[Tuple[str, ...], ...]
    admissible: Tuple[bool, ...]
    #: The stop flown next, by its index in ``stops``; None flew home.
    next_index: Optional[int]


@dataclass(frozen=True)
class MissionRecord:
    """One mule mission (= one FL round in the integrated stack).

    Sourced from a mule ``mission_completed`` event. ``pass_1_updates`` /
    ``pass_1_scheduled`` / ``pass_1_clean_devices`` are the EX-4.0
    instrumentation fields added to that event; older logs without them
    leave the optionals ``None`` and the metric layer falls back to
    ``len(pass_1_clean_devices)`` / ``n_devices``. Where the trace records
    what the mule's merge kept (``pass_1_merged_*``), the metric layer counts
    that instead of the CLEAN collections.
    """

    mission_round: int
    pass_1_contacts: int
    pass_2_contacts: int
    pass_1_updates: Optional[int]
    pass_1_scheduled: Optional[int]
    pass_1_clean_devices: Tuple[str, ...]
    delivered: Optional[int]
    undelivered: Optional[int]
    duration_s: Optional[float]
    # Trace-scorer additions (Phase 0). The time window comes from the
    # envelope timestamps of this mission's ``mission_started`` and
    # ``mission_completed`` events; hand-built rows without ``ts`` leave it
    # None. The plan and outcomes are None on traces recorded before the mule
    # emitted them — distinct from an empty tuple, which is a recorded plan
    # (or outcome list) with nothing in it.
    mule_id: Optional[str] = None
    started_ts: Optional[float] = None
    completed_ts: Optional[float] = None
    #: ``(device, deadline_ts)`` for every device admitted to Pass 1.
    pass_1_deadlines: Optional[Tuple[Tuple[str, float], ...]] = None
    #: ``(device, outcome, contact_ts)`` for every Pass-1 session recorded.
    pass_1_outcomes: Optional[Tuple[Tuple[str, str, float], ...]] = None
    # FeRRy audit #3. An age-aware merge can exclude a CLEAN update that is
    # past its age cutoff, so "collected" and "merged" differ: these are the
    # CLEAN devices minus the excluded ones, and their count. Traces recorded
    # between Phase 1 and these fields carry the exclusions only in
    # ``pass_1_merge.excluded``, from which the same subtraction recovers
    # them. None on traces older than Phase 1, where every CLEAN update was
    # merged; an empty tuple is a recorded merge that kept nothing.
    pass_1_merged_devices: Optional[Tuple[str, ...]] = None
    pass_1_merged_updates: Optional[int] = None
    #: Where ``pass_1_deadlines`` came from: ``"device"`` when the plan carried
    #: each member's own Deadline(j), ``"contact"`` when every member was held
    #: to its contact's (tightest-member) deadline, as traces recorded before
    #: ``device_deadlines`` existed are. None without a plan, or with a plan
    #: that admitted nobody.
    deadline_basis: Optional[str] = None
    # FeRRy Phase 3 — the mission on the simulated mission clock, from
    # ``mission_completed``'s simulated fields (design section 2.5). All None
    # on a wall-clock trace. Times are simulated seconds; the energy is
    # SIMULATED (the Zeng-Xu-Zhang 2019 rotary-wing model, not a measurement).
    sim_start_s: Optional[float] = None
    sim_end_s: Optional[float] = None
    #: The clock's charges for this mission, ``(kind, seconds)`` in the order
    #: the trace lists them (transit, dwell, listen, return, upload,
    #: turnaround, dock_wait); they sum to ``sim_end_s - sim_start_s``.
    sim_ledger: Optional[Tuple[Tuple[str, float], ...]] = None
    energy_j: Optional[float] = None
    #: How far the Pass-1 upload (or the landing, when nothing was uploaded)
    #: ended past the mission budget; 0 within it, None without a budget.
    budget_overrun_s: Optional[float] = None
    #: How many in-flight re-plans and aborts the mission made, and how many
    #: beacon offers it inserted.
    replans: Optional[int] = None
    aborts: Optional[int] = None
    inserts: Optional[int] = None
    # FeRRy Phase 4 (the Phase 4 spec, other choices 12): the band flown, the
    # plan clock's commit and the whole-scheduler baselines' drop report, from
    # ``mission_completed``'s fields. Each is None where the mission does not
    # record it: every field on a wall-clock mission; the plan fields outside
    # plan mode (they come from ``plan``, which only a plan-mode mission
    # writes); ``policy_drops`` on every trace recorded before Phase 4, and on
    # every mission a baseline left nothing out of, since the mule writes
    # ``pass_1_policy_drops`` only when it is not empty (critic D2). Which of
    # those an absence means is the scorer's to say (``traces_scorer``).
    #: The class the mission flew (``mission_completed.band``): the committed
    #: class b̄ in plan mode, the configured band otherwise, None without one.
    band: Optional[str] = None
    #: ``pass_1_flown[].band`` in flight order: the class each Pass-1 stop was
    #: flown on, which under FX can differ from b̄ (decision 5; Pass 2 always
    #: flies b̄). None without ``pass_1_flown`` (the wall clock); an entry is
    #: None for a stop flown without a band (the channel-free control).
    flown_bands: Optional[Tuple[Optional[str], ...]] = None
    #: The committed plan (``mission_completed.plan``, the closed
    #: ``PlanCommit.describe()``): its class and search mode, the demand (the
    #: devices left after S1 and S3) and the devices its stops serve, V and the
    #: predicted whole mission (``score.mission_s``: Pass 1, the turnaround and
    #: Pass 2, decision 2 (b); never ``delta_s``, which leaves the dwell out
    #: under F-dwell), and the age cap: S (``cap.s``, None when the arm runs
    #: none), the devices capped at planning and every violation the mule
    #: logged as ``(device, planning age, reason)``.
    plan_band: Optional[str] = None
    plan_search: Optional[str] = None
    plan_demand: Optional[Tuple[str, ...]] = None
    plan_served: Optional[Tuple[str, ...]] = None
    plan_v: Optional[float] = None
    plan_mission_s: Optional[float] = None
    cap_s: Optional[int] = None
    cap_capped: Optional[Tuple[str, ...]] = None
    cap_violations: Optional[Tuple[Tuple[str, int, str], ...]] = None
    #: The wall seconds the plan took (``plan_wall_s``): a wall time, so it is
    #: left out of every determinism comparison (critic B12).
    plan_wall_s: Optional[float] = None
    #: What a whole-scheduler baseline (D1-D5) left out before takeoff on the
    #: simulated clock (``pass_1_policy_drops``, decision 6), as ``(devices,
    #: reason)`` per dropped contact; reported by the mule, never widened.
    policy_drops: Optional[Tuple[Tuple[Tuple[str, ...], str], ...]] = None
    # FeRRy Phase 5 (the Phase 5 spec, other choices 9): the learned fillings'
    # records, from ``mission_completed``'s fields. The mule writes each only
    # when it is not empty, so None is both "this arm records none" and "this
    # mission made none"; which one an absence means is the scorer's to say
    # (``traces_scorer.pair_report``), as for ``policy_drops``.
    #: ``pass_1_pairs``: the pair slot's closed decisions in flight order, one
    #: per Pass-1 stop flown.
    pair_decisions: Optional[Tuple[PairDecision, ...]] = None
    #: ``pass_1_e3``: arm E3's next-stop calls, in order.
    e3_calls: Optional[Tuple[E3Call, ...]] = None
    #: ``pass_1_e3_unvisited``: the members of each stop E3's pass left when no
    #: stop was admissible any more (reported, never widened).
    e3_unvisited: Optional[Tuple[Tuple[str, ...], ...]] = None

    @property
    def has_plan(self) -> bool:
        """Whether the mission recorded a plan-clock commit (plan mode only)."""
        return self.plan_demand is not None

    def contains(self, ts: Optional[float]) -> bool:
        """Whether ``ts`` falls inside this mission's time window."""
        return (
            ts is not None
            and self.started_ts is not None
            and self.completed_ts is not None
            and self.started_ts <= ts <= self.completed_ts
        )

    @property
    def sim_duration_s(self) -> Optional[float]:
        """The mission's simulated duration, takeoff to the Pass-2 landing."""
        if self.sim_start_s is None or self.sim_end_s is None:
            return None
        return self.sim_end_s - self.sim_start_s

    def ledger(self) -> Optional[Dict[str, float]]:
        """``sim_ledger`` as a dict, kind -> seconds; None off the mission clock."""
        return None if self.sim_ledger is None else dict(self.sim_ledger)


@dataclass(frozen=True)
class ModelEvalPoint:
    """One held-out convergence point (EX-4.1 ``model_eval`` event).

    ``cluster_round`` 0 is the seeded init θ baseline; 1..R are the
    aggregated models after each cross-mule FedAvg.
    """

    cluster_round: int
    accuracy: float
    auc: float
    loss: float
    n_test: int
    ts: Optional[float] = None
    #: FeRRy Phase 3: the cluster's simulated time at the evaluation, the
    #: latest simulated upload it had ingested (None on the wall clock, and
    #: for the seed model, evaluated before any upload).
    sim_ts: Optional[float] = None


@dataclass
class Exp4Observation:
    """Everything the metric layer needs from one finished trial."""

    n_devices: int
    cluster_rounds_closed: int
    up_bundles_ingested: int
    missions: List[MissionRecord] = field(default_factory=list)
    mission_failures: int = 0
    missions_empty: int = 0
    # EX-4.2 — mission_rounds whose mule->BS backhaul upload was dropped
    # (round did not close; recoverable). Used to mark those rounds as
    # not-closed in round_close_rate, so H1's jittery penalty is visible.
    backhaul_lost_rounds: Set[int] = field(default_factory=set)
    backhaul_losses: int = 0
    # FeRRy audit #4 — the cluster's merge ledger under an age-aware rule
    # (empty under agg:plain, which emits none of these events). agg:fedbuff
    # buffers an upload without changing θ (``cluster_merge_deferred``) until
    # a later upload fills the buffer and flushes it, so a deferred mission's
    # updates reach the model at the flushing mission, not their own; an
    # upload that is never flushed never reaches it. ``flush_of`` maps each
    # flushed deferred round to the mission round whose applied merge flushed
    # it. An expired round (``cluster_merge_expired``) merged nothing at all.
    # Neither a deferred nor an expired round closes.
    deferred_rounds: Set[int] = field(default_factory=set)
    expired_rounds: Set[int] = field(default_factory=set)
    flush_of: Dict[int, int] = field(default_factory=dict)
    # EX-4.1 real-model convergence trace (empty on the stub path).
    model_evals: List[ModelEvalPoint] = field(default_factory=list)
    # Per-device Pass-1+Pass-2 serve counts, padded to every device that
    # announced itself (``device_ready``) so zero-serve devices still
    # count toward fairness / entropy denominators. On the simulated clock
    # they count member contacts only: solicits are targeted, so no
    # non-member times out on a broadcast (module docstring, critic B13).
    per_device_serves: Dict[str, int] = field(default_factory=dict)
    device_serve_failures: int = 0
    # Sanity flags harvested from the streams, surfaced for debugging.
    cluster_ready: bool = False
    mule_ready: bool = False
    dock_bootstrapped: bool = False
    # FeRRy Phase 2 — several mules. ``mule_ids`` are the mules the trial ran
    # (every id on a mule row, and every configured mule), and ``mule_slices``
    # the devices each mule's config assigned it, where the config lists them.
    # The ``*_keys`` fields are the per-mission ledger above keyed by
    # :data:`MissionKey`; the metrics read these, because at K > 1 the
    # round-keyed sets collide (a lost round 2 on one mule would mark every
    # mule's round 2). The round-keyed fields are their projection onto
    # rounds, kept for the callers that read them: exact with one mule, and
    # ambiguous across mules with several. ``__post_init__`` fills whichever
    # half a caller leaves out, so an observation built from the round-keyed
    # fields alone, as before the keys existed, scores as it did then.
    mule_ids: Tuple[str, ...] = ()
    mule_slices: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    backhaul_lost_keys: Set[MissionKey] = field(default_factory=set)
    deferred_keys: Set[MissionKey] = field(default_factory=set)
    expired_keys: Set[MissionKey] = field(default_factory=set)
    flush_of_keys: Dict[MissionKey, MissionKey] = field(default_factory=dict)
    #: Cluster round → the mule whose upload closed it, which is the mule whose
    #: inter-pass dock the round's evaluation ran in.
    closed_by_mule: Dict[int, str] = field(default_factory=dict)
    #: Missions whose upload the cluster logged as ingested but never folded
    #: into θ: a second upload from a mule whose partial was still in the open
    #: round (refused as a duplicate), or a partial still waiting for its
    #: quorum when the trial ended. Only several mules produce them; with one,
    #: every ingest is answered at once and this stays empty.
    unmerged_keys: Set[MissionKey] = field(default_factory=set)
    #: FeRRy Phase 3: the clock the trial's mission time ran on, "wall" or
    #: "sim" (:func:`trace_clock_domain`; "wall" for every recorded trace).
    #: On "sim" the missions complete in the order of their simulated ends.
    mission_clock: str = CLOCK_WALL

    def __post_init__(self) -> None:
        # Whichever half of the ledger the caller gave, the other follows, so
        # the metrics (which read the keyed half) never ignore a set that was
        # given. A round-keyed set is lifted onto the trial's one mule, where a
        # round names exactly one mission; with several mules it names one of
        # each, so it is refused rather than guessed at. Given both halves, both
        # are kept as given. Fields assigned after construction are not synced.
        lifted = _mule_key(self.mule_ids, None)
        for keyed, by_round in (
            ("backhaul_lost_keys", "backhaul_lost_rounds"),
            ("deferred_keys", "deferred_rounds"),
            ("expired_keys", "expired_rounds"),
        ):
            keys, rounds = getattr(self, keyed), getattr(self, by_round)
            if rounds and not keys:
                self._refuse_round_keyed(by_round)
                setattr(self, keyed, {(lifted, r) for r in rounds})
            elif keys and not rounds:
                setattr(self, by_round, {r for _, r in keys})
        if self.flush_of and not self.flush_of_keys:
            self._refuse_round_keyed("flush_of")
            self.flush_of_keys = {
                (lifted, p): (lifted, at) for p, at in self.flush_of.items()
            }
        elif self.flush_of_keys and not self.flush_of:
            self.flush_of = {p[1]: at[1] for p, at in self.flush_of_keys.items()}

    def _refuse_round_keyed(self, name: str) -> None:
        if self.n_mules > 1:
            raise ValueError(
                f"{name} is keyed by mission round alone, which names one mission "
                f"per mule with {self.n_mules} mules; give its (mule_id, "
                f"mission_round)-keyed counterpart instead"
            )

    @property
    def missions_completed(self) -> int:
        return len(self.missions)

    @property
    def n_mules(self) -> int:
        return len(self.mule_ids)

    def mule_key(self, mule_id: Optional[str]) -> Optional[str]:
        """The mule half of a :data:`MissionKey` for a row that names ``mule_id``."""
        return _mule_key(self.mule_ids, mule_id)

    def mission_key(self, mission: MissionRecord) -> MissionKey:
        return (self.mule_key(mission.mule_id), mission.mission_round)

    def ordered_missions(self) -> List[MissionRecord]:
        """The missions in completion order on the trial's own clock
        (:func:`completion_order`)."""
        return completion_order(self.missions, self.mission_clock)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def _events(rows: Sequence[dict], name: str) -> List[dict]:
    return [r for r in rows if r.get("event") == name]


def observation_from_rows(
    *,
    cluster_rows: Sequence[dict],
    mule_rows: Sequence[dict],
    device_rows: Sequence[dict],
    n_devices: int,
    mule_slices: Optional[Mapping[str, Sequence[str]]] = None,
) -> Exp4Observation:
    """Fold three role event streams into one :class:`Exp4Observation`.

    ``cluster_rows`` / ``mule_rows`` / ``device_rows`` are the parsed
    JSONL envelopes for, respectively, all cluster / mule / device
    processes in the run (already concatenated if there were several of
    a role). ``mule_slices`` maps each configured mule to the devices its
    config assigns it (empty where the config names none); without it the
    trial's mules are the ones its mule rows name.

    Raises :class:`ClockDomainError` for a trace whose clocks disagree
    (:func:`trace_clock_domain`); no recorded trace does.
    """
    mission_clock = trace_clock_domain(cluster_rows, mule_rows)

    # ------------------------------- mule -------------------------------- #
    # Parsed first so the cluster section can place its events in a
    # mission's time window. Rows are walked in order per mule: each
    # ``mission_started`` opens the window its ``mission_completed`` closes.
    missions: List[MissionRecord] = []
    started_at: Dict[Optional[str], Optional[float]] = {}
    for r in mule_rows:
        event = r.get("event")
        mule_id = _opt_str(r.get("id"))
        if event == "mission_started":
            started_at[mule_id] = _opt_float(r.get("ts"))
            continue
        if event != "mission_completed":
            continue
        clean = r.get("pass_1_clean_devices")
        clean_tuple: Tuple[str, ...] = (
            tuple(str(d) for d in clean) if isinstance(clean, (list, tuple)) else ()
        )
        merged = r.get("pass_1_merged_devices")
        merged_n = _opt_int(r.get("pass_1_merged_updates"))
        if not isinstance(merged, (list, tuple)):
            merged = _merged_from_merge_record(clean_tuple, r.get("pass_1_merge"))
            if merged is not None:
                merged_n = len(merged)
        deadlines, deadline_basis = _plan_deadlines(r.get("pass_1_plan"))
        missions.append(
            MissionRecord(
                mission_round=int(r.get("mission_round", 0)),
                pass_1_contacts=int(r.get("pass_1_contacts", 0) or 0),
                pass_2_contacts=int(r.get("pass_2_contacts", 0) or 0),
                pass_1_updates=_opt_int(r.get("pass_1_updates")),
                pass_1_scheduled=_opt_int(r.get("pass_1_scheduled")),
                pass_1_clean_devices=clean_tuple,
                delivered=_opt_int(r.get("delivered")),
                undelivered=_opt_int(r.get("undelivered")),
                duration_s=_opt_float(r.get("duration_s")),
                mule_id=mule_id,
                started_ts=started_at.pop(mule_id, None),
                completed_ts=_opt_float(r.get("ts")),
                pass_1_deadlines=deadlines,
                pass_1_outcomes=_session_outcomes(r.get("pass_1_outcomes")),
                pass_1_merged_devices=(
                    tuple(str(d) for d in merged)
                    if isinstance(merged, (list, tuple)) else None
                ),
                pass_1_merged_updates=merged_n,
                deadline_basis=deadline_basis,
                sim_start_s=_opt_float(r.get("sim_start_s")),
                sim_end_s=_opt_float(r.get("sim_end_s")),
                sim_ledger=_sim_ledger(r.get("sim_ledger")),
                energy_j=_opt_float(r.get("energy_j")),
                budget_overrun_s=_opt_float(r.get("budget_overrun_s")),
                replans=_opt_len(r.get("replans")),
                aborts=_opt_len(r.get("aborts")),
                inserts=_opt_len(r.get("inserts")),
                band=_opt_str(r.get("band")),
                flown_bands=_flown_bands(r.get("pass_1_flown")),
                plan_wall_s=_opt_float(r.get("plan_wall_s")),
                policy_drops=_policy_drops(r.get("pass_1_policy_drops")),
                pair_decisions=_pair_decisions(r.get("pass_1_pairs")),
                e3_calls=_e3_calls(r.get("pass_1_e3")),
                e3_unvisited=_e3_unvisited(r.get("pass_1_e3_unvisited")),
                **_plan_fields(r.get("plan")),
            )
        )

    # The trial's mules: every id a mule row carries, and every configured
    # mule (one may have flown nothing).
    mule_ids = tuple(sorted(
        {str(r["id"]) for r in mule_rows if r.get("id") is not None}
        | {str(m) for m in (mule_slices or {})}
    ))
    place = _mission_placer(missions, mule_ids)

    # ------------------------------ cluster ------------------------------ #
    cluster_rounds_closed = len(_events(cluster_rows, "cluster_round_closed"))
    # A refused partial (Phase 2, several mules) is still logged as ingested,
    # for its reports; it is not an accepted upload.
    up_bundles_ingested = sum(
        1 for r in _events(cluster_rows, "up_bundle_ingested")
        if not r.get("partial_refused")
    )
    cluster_ready = bool(_events(cluster_rows, "cluster_ready"))

    # Traces recorded before Freeze Amendment 5 carry no mission round on
    # ``backhaul_upload_lost`` (the cluster read it from the wrong object).
    # The loss happens at the inter-pass dock, inside the mission's window,
    # so the round is recovered from the mule's timestamps instead.
    backhaul_lost_keys: Set[MissionKey] = set()
    for r in _events(cluster_rows, "backhaul_upload_lost"):
        key = place(
            _opt_str(r.get("mule_id")), _opt_int(r.get("mission_round")),
            _opt_float(r.get("ts")),
        )
        if key is not None:
            backhaul_lost_keys.add(key)
    backhaul_losses = len(_events(cluster_rows, "backhaul_upload_lost"))
    deferred_keys, expired_keys, flush_of_keys = _merge_ledger(
        cluster_rows, place, mule_ids,
    )
    unmerged_keys = _unmerged_uploads(cluster_rows, place, mule_ids)
    # A mule with dock_on_empty (Phase 2) docks after a mission with nothing
    # to upload, and its empty partial goes through the cluster's fold like
    # any other: an age-aware fold cuts it to zero weight and lists it with the
    # expired partials, FedBuff reports it deferred without buffering it, and
    # it holds its mule's place in the open round. These ledgers count updates
    # that failed to reach θ, and it carried none, so it is left out of them;
    # ``missions_empty`` counts it. No trace without dock_on_empty has one.
    empty_uploads = _empty_uploads(mule_rows, mule_ids)
    # Likewise the empty partial a cluster holds in a mule's place when its
    # upload was lost under a quorum above 1 (``awaits_quorum``): the fold
    # lists it with the expired partials, but its mission is counted where it
    # belongs, as a backhaul loss, and carried no update to lose again.
    no_update = empty_uploads | _held_places(cluster_rows, place)
    deferred_keys -= no_update
    expired_keys -= no_update
    unmerged_keys -= no_update
    flush_of_keys = {p: at for p, at in flush_of_keys.items() if p not in no_update}

    model_evals: List[ModelEvalPoint] = []
    for r in _events(cluster_rows, "model_eval"):
        model_evals.append(
            ModelEvalPoint(
                cluster_round=int(r.get("cluster_round", 0) or 0),
                accuracy=float(r.get("accuracy", 0.0) or 0.0),
                auc=float(r.get("auc", 0.0) or 0.0),
                loss=float(r.get("loss", 0.0) or 0.0),
                n_test=int(r.get("n_test", 0) or 0),
                ts=_opt_float(r.get("ts")),
                sim_ts=_opt_float(r.get("sim_ts")),
            )
        )
    model_evals.sort(key=lambda p: p.cluster_round)

    mission_failures = len(_events(mule_rows, "mission_failed"))
    missions_empty = len(_events(mule_rows, "mission_empty"))
    mule_ready = bool(_events(mule_rows, "mule_ready"))
    dock_bootstrapped = bool(_events(mule_rows, "dock_bootstrapped"))

    # ------------------------------ device ------------------------------- #
    # Seed the visit map with every device that announced itself so a
    # device that never served still occupies a (zero) slot — otherwise
    # Jain's index / entropy would be computed over the served subset only
    # and overstate fairness.
    per_device_serves: Dict[str, int] = {}
    for r in _events(device_rows, "device_ready"):
        did = r.get("id")
        if did is not None:
            per_device_serves.setdefault(str(did), 0)
    for r in _events(device_rows, "device_served"):
        did = r.get("id")
        if did is None:
            continue
        did = str(did)
        per_device_serves[did] = per_device_serves.get(did, 0) + 1
    device_serve_failures = len(_events(device_rows, "device_serve_failed"))

    return Exp4Observation(
        n_devices=int(n_devices),
        cluster_rounds_closed=cluster_rounds_closed,
        up_bundles_ingested=up_bundles_ingested,
        missions=missions,
        mission_failures=mission_failures,
        missions_empty=missions_empty,
        # The round-keyed fields are left to __post_init__, which projects
        # the keyed ledger below onto rounds.
        backhaul_losses=backhaul_losses,
        model_evals=model_evals,
        per_device_serves=per_device_serves,
        device_serve_failures=device_serve_failures,
        cluster_ready=cluster_ready,
        mule_ready=mule_ready,
        dock_bootstrapped=dock_bootstrapped,
        mule_ids=mule_ids,
        mule_slices={
            str(m): tuple(str(d) for d in devices)
            for m, devices in (mule_slices or {}).items() if devices
        },
        backhaul_lost_keys=backhaul_lost_keys,
        deferred_keys=deferred_keys,
        expired_keys=expired_keys,
        flush_of_keys=flush_of_keys,
        closed_by_mule=_round_closers(cluster_rows),
        unmerged_keys=unmerged_keys,
        mission_clock=mission_clock,
    )


def completion_order(
    missions: Sequence[MissionRecord], clock: str = CLOCK_WALL,
) -> List[MissionRecord]:
    """``missions`` in completion order on ``clock``.

    On the wall clock by the envelope time of each ``mission_completed``, as
    always. On the simulated clock by the simulated end (``sim_end_s``): the
    envelope time stays wall, and with several mules the order in which their
    processes happen to finish is not the order in which their missions end
    (FeRRy Phase 3). Either way, the recorded order when any mission lacks
    the stamp, and ties keep it (the sort is stable). With one mule both
    orders are its mission rounds.
    """
    key = (
        (lambda m: m.sim_end_s) if clock == CLOCK_SIM
        else (lambda m: m.completed_ts)
    )
    if all(key(m) is not None for m in missions):
        return sorted(missions, key=key)
    return list(missions)


def trace_clock_domain(cluster_rows: Sequence[dict], mule_rows: Sequence[dict]) -> str:
    """The clock a trace's mission time ran on, "wall" or "sim" (FeRRy Phase 3).

    Read from the mules' ``mule_ready.mission_clock``, else the cluster's
    ``cluster_ready.mission_clock``; absent means "wall", so every trace
    recorded before Phase 3 is wall-clock. Raises :class:`ClockDomainError`
    when the trace's clocks disagree:

    * the mules disagree with each other, or the cluster with them, or one
      names a clock that does not exist;
    * the domain is "sim", and a ``mission_completed`` lacks its simulated
      start or end, or a mission-time stamp is a wall one (at or above
      ``SIM_CEILING_S``): a Pass-1 plan deadline, a session's ``contact_ts``,
      a mission's simulated start or end, or the cluster's simulated times;
    * the domain is "wall", and the trace carries simulated fields (a
      mission's ``sim_start_s`` / ``sim_end_s``, the cluster's ``sim_ts`` /
      ``sim_upload_ts``), or its Pass-1 stamps lie on both sides of the
      ceiling. The wall clock is held to one side rather than to wall time:
      hand-built traces stamp everything in small numbers.

    Only the stamps the analysis compares or orders by are checked.
    """
    mule_tags = _clock_tags(mule_rows, "mule_ready", "mule")
    cluster_tags = _clock_tags(cluster_rows, "cluster_ready", "cluster")
    for tag, who in mule_tags + cluster_tags:
        if tag not in (CLOCK_WALL, CLOCK_SIM):
            raise ClockDomainError(
                f"{who} names mission clock {tag!r}; a trace's clock is "
                f"{CLOCK_WALL!r} or {CLOCK_SIM!r}"
            )
    for tags, role in ((mule_tags, "mules"), (cluster_tags, "cluster")):
        if len({tag for tag, _ in tags}) > 1:
            raise ClockDomainError(
                f"clock domains disagree: the {role} ran on different clocks ("
                + ", ".join(f"{who}: {tag!r}" for tag, who in tags) + ")"
            )
    domain = (mule_tags or cluster_tags or [(CLOCK_WALL, "")])[0][0]
    if cluster_tags and cluster_tags[0][0] != domain:
        raise ClockDomainError(
            f"clock domains disagree: the mules ran on the {domain!r} clock and "
            f"the cluster on the {cluster_tags[0][0]!r} one"
        )

    stamps: List[Tuple[str, float]] = []        # Pass-1 deadlines and contacts
    sim_fields: List[Tuple[str, float]] = []    # fields only the sim clock writes
    for r in mule_rows:
        event = r.get("event")
        if event == "mission_started":
            v = _opt_float(r.get("sim_start_s"))
            if v is not None:
                sim_fields.append((f"mission_started (mule {r.get('id')}) sim_start_s", v))
            continue
        if event != "mission_completed":
            continue
        where = f"mission_completed (mule {r.get('id')}, round {r.get('mission_round')})"
        for name in ("sim_start_s", "sim_end_s"):
            v = _opt_float(r.get(name))
            if v is not None:
                sim_fields.append((f"{where} {name}", v))
            elif domain == CLOCK_SIM:
                raise ClockDomainError(
                    f"{where} has no {name}, but the trace is on the simulated "
                    f"clock, where every mission records its simulated start and end"
                )
        for i, contact in enumerate(r.get("pass_1_plan") or ()):
            if not isinstance(contact, dict):
                continue
            v = _opt_float(contact.get("deadline_ts"))
            if v is not None:
                stamps.append((f"{where} pass_1_plan[{i}].deadline_ts", v))
            own = contact.get("device_deadlines")
            for device, raw in (own.items() if isinstance(own, dict) else ()):
                v = _opt_float(raw)
                if v is not None:
                    stamps.append((f"{where} pass_1_plan[{i}].device_deadlines[{device}]", v))
        for s in r.get("pass_1_outcomes") or ():
            v = _opt_float(s.get("contact_ts")) if isinstance(s, dict) else None
            if v is not None:
                stamps.append((f"{where} pass_1_outcomes[{s.get('device')}].contact_ts", v))
    for r in cluster_rows:
        for name in ("sim_ts", "sim_upload_ts"):
            v = _opt_float(r.get(name))
            if v is not None:
                sim_fields.append((f"cluster {r.get('event')} {name}", v))

    if domain == CLOCK_SIM:
        for where, v in stamps + sim_fields:
            if not v < SIM_CEILING_S:
                raise ClockDomainError(
                    f"clock domains disagree: the trace is on the simulated mission "
                    f"clock (mission_clock {CLOCK_SIM!r}), but {where} = {v!r} is not "
                    f"a simulated time (simulated stamps stay below "
                    f"{SIM_CEILING_S:g} s; wall ones are above it)"
                )
        return domain
    if sim_fields:
        where, v = sim_fields[0]
        raise ClockDomainError(
            f"clock domains disagree: the trace is on the wall clock (no "
            f"mule_ready or cluster_ready names mission_clock {CLOCK_SIM!r}), but "
            f"{where} = {v!r} is a field only the simulated clock writes"
        )
    wall = next(((w, v) for w, v in stamps if not v < SIM_CEILING_S), None)
    sim = next(((w, v) for w, v in stamps if v < SIM_CEILING_S), None)
    if wall is not None and sim is not None:
        raise ClockDomainError(
            f"clock domains disagree: {wall[0]} = {wall[1]!r} is a wall-clock "
            f"stamp and {sim[0]} = {sim[1]!r} a simulated one (the domains meet "
            f"at {SIM_CEILING_S:g} s); a deadline miss compares the two and would "
            f"read 0 % or 100 %"
        )
    return domain


def consume_run_dir(run_dir, *, n_devices: int) -> Exp4Observation:
    """Read every ``{cluster,mule,device}-*.jsonl`` under ``run_dir``.

    Globs by role prefix so it is agnostic to the exact node ids (and
    tolerant of multi-mule / multi-device topologies). Missing files are
    treated as empty streams — a trial where the mule never started still
    produces a (zeroed) observation rather than raising. Each mule's config
    (``mule-<id>.json``, written by the orchestrator) supplies its slice.
    """
    run_dir = Path(run_dir)
    cluster_rows = _read_role(run_dir, "cluster")
    mule_rows = _read_role(run_dir, "mule")
    device_rows = _read_role(run_dir, "device")
    return observation_from_rows(
        cluster_rows=cluster_rows,
        mule_rows=mule_rows,
        device_rows=device_rows,
        n_devices=n_devices,
        mule_slices=_read_mule_slices(run_dir),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _read_role(run_dir: Path, prefix: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(run_dir.glob(f"{prefix}-*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _read_mule_slices(run_dir: Path) -> Dict[str, Tuple[str, ...]]:
    """Each ``mule-<id>.json``'s ``expected_devices``, by mule id.

    The id is the config's ``mule_id``, else the file name's. A config that
    lists no devices maps to an empty tuple: the mule exists, but its slice
    was assigned elsewhere (round-robin in ``TopologyConfig.validate``).
    """
    slices: Dict[str, Tuple[str, ...]] = {}
    for path in sorted(run_dir.glob("mule-*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(cfg, dict):
            continue
        mule_id = str(cfg.get("mule_id") or path.stem[len("mule-"):])
        devices = cfg.get("expected_devices")
        slices[mule_id] = (
            tuple(str(d) for d in devices) if isinstance(devices, (list, tuple)) else ()
        )
    return slices


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A crash mid-write can leave a torn final line; the
                    # completed lines above it are still valid.
                    continue
    except OSError:
        return rows
    return rows


def _opt_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_str(v) -> Optional[str]:
    return None if v is None else str(v)


def _opt_len(v) -> Optional[int]:
    """How many entries a recorded list has; None when the field is absent."""
    return len(v) if isinstance(v, (list, tuple)) else None


def _sim_ledger(raw) -> Optional[Tuple[Tuple[str, float], ...]]:
    """``mission_completed.sim_ledger`` ({kind: seconds}) as ``(kind, seconds)``
    pairs in the recorded order; None when absent (a wall-clock mission)."""
    if not isinstance(raw, dict):
        return None
    pairs: List[Tuple[str, float]] = []
    for kind, seconds in raw.items():
        v = _opt_float(seconds)
        if v is not None:
            pairs.append((str(kind), v))
    return tuple(pairs)


def _clock_tags(rows: Sequence[dict], event: str, role: str) -> List[Tuple[str, str]]:
    """``(mission_clock, who)`` of every ``event`` row, "wall" where the row
    names none (every row written before FeRRy Phase 3, and every wall-clock
    one since)."""
    return [
        (str(r.get("mission_clock") or CLOCK_WALL), f"{role} {r.get('id')}")
        for r in rows if r.get("event") == event
    ]


def _mule_key(mule_ids: Sequence[str], mule_id: Optional[str]) -> Optional[str]:
    """The mule half of a :data:`MissionKey` (see :meth:`Exp4Observation.mule_key`).

    With one mule, or none named, every mission is that mule's, whatever a row
    calls it (hand-built rows often leave the id out), so every key takes the
    trial's one id: the keyed ledger is then the round-keyed one exactly. With
    several, each row's own id.
    """
    if len(mule_ids) > 1:
        return mule_id
    return mule_ids[0] if mule_ids else None


def _mission_round_at(
    missions: Sequence[MissionRecord],
    mule_id: Optional[str],
    ts: Optional[float],
) -> Optional[int]:
    """Round of the mission (flown by ``mule_id``) whose window contains ``ts``."""
    for m in missions:
        same_mule = mule_id is None or m.mule_id is None or m.mule_id == mule_id
        if same_mule and m.contains(ts):
            return m.mission_round
    return None


def _mission_placer(
    missions: Sequence[MissionRecord], mule_ids: Sequence[str],
) -> Callable[..., Optional[MissionKey]]:
    """``place(mule_id, mission_round, ts, hint=None)`` → the event's mission key.

    With one mule an event is placed exactly as before multi-mule keys
    existed: by its recorded round, else by the mission window its timestamp
    falls in. With several, the mule matters. An event that names no mule
    takes ``hint`` (the uploader, when the caller knows it), then the one mule
    that flew that round, then the one whose window at that round contains
    the event; an event without a round is placed by the window of its mule's
    missions, and by any mule's only while those windows name a single mule.
    None when the event cannot be placed.
    """
    multi = len(mule_ids) > 1

    def place(
        mule_id: Optional[str], rnd: Optional[int], ts: Optional[float],
        hint: Optional[str] = None,
    ) -> Optional[MissionKey]:
        if not multi:
            if rnd is None:
                rnd = _mission_round_at(missions, mule_id, ts)
            return None if rnd is None else (_mule_key(mule_ids, mule_id), rnd)
        mule_id = mule_id if mule_id is not None else hint
        if rnd is None:
            return _only_key(
                m for m in missions
                if (mule_id is None or m.mule_id == mule_id) and m.contains(ts)
            )
        if mule_id is None:
            same_round = [m for m in missions if m.mission_round == rnd]
            key = _only_key(same_round)
            if key is None:
                key = _only_key(m for m in same_round if m.contains(ts))
            return key
        return (mule_id, rnd)

    return place


def _only_key(missions) -> Optional[MissionKey]:
    """The key of the first mission, if every one given is the same mule's."""
    missions = list(missions)
    if not missions or len({m.mule_id for m in missions}) != 1:
        return None
    return (missions[0].mule_id, missions[0].mission_round)


def _merge_ledger(
    cluster_rows: Sequence[dict],
    place: Callable[..., Optional[MissionKey]],
    mule_ids: Sequence[str],
) -> Tuple[Set[MissionKey], Set[MissionKey], Dict[MissionKey, MissionKey]]:
    """Deferred missions, expired missions, and which mission flushed each deferral.

    Walks the cluster's merge events in order. Each is placed (see
    :func:`_mission_placer`) by its recorded ``mule_id`` and ``mission_round``
    (the uploading mission) or, on traces from before the cluster recorded
    the round, by the mission window its timestamp falls in — the merge runs
    at the inter-pass dock, as a backhaul loss does. An applied
    ``cluster_merge`` names no mule; its uploader is the mule of the
    ``up_bundle_ingested`` just before it, whose ingest ran the fold. An
    applied merge that lists the ``partials`` it flushed is taken at its
    word; one that does not flushes every mission its mule has had deferred
    since the previous flush, which is what a FedBuff buffer does. Only
    missions seen deferred enter the flush map. A merge whose own mission
    cannot be placed still empties the buffer, but its flushed missions get
    no flush mission and so are never credited.
    """
    deferred: Set[MissionKey] = set()
    expired: Set[MissionKey] = set()
    flush_of: Dict[MissionKey, MissionKey] = {}
    # Deferred missions awaiting a flush, per mule as each event names it,
    # for the fallback above.
    pending: Dict[Optional[str], List[MissionKey]] = {}
    last_up: Tuple[Optional[str], Optional[int]] = (None, None)
    for r in cluster_rows:
        event = r.get("event")
        if _runs_a_fold(r):
            last_up = (_opt_str(r.get("mule_id")), _opt_int(r.get("mission_round")))
            continue
        if event not in ("cluster_merge_deferred", "cluster_merge_expired", "cluster_merge"):
            continue
        mule_id = _opt_str(r.get("mule_id"))
        rnd = _opt_int(r.get("mission_round"))
        partials = _partial_keys(r.get("partials"), mule_ids)
        key = place(
            mule_id, rnd, _opt_float(r.get("ts")),
            hint=_uploader(last_up, rnd, partials),
        )
        if event == "cluster_merge_deferred":
            if key is not None:
                deferred.add(key)
                pending.setdefault(mule_id, []).append(key)
            continue
        if event == "cluster_merge_expired":
            # An expired fold discards every pending partial, not only the
            # uploader's; its ``partials`` names them all.
            if key is not None:
                expired.add(key)
            expired.update(partials or ())
            continue
        if not r.get("applied", True):
            continue
        # A partial cut to zero weight inside an applied fold never reached θ.
        expired.update(_partial_keys(r.get("expired_partials"), mule_ids) or ())
        if partials is None:
            if mule_id is None:
                flushed = [p for keys in pending.values() for p in keys]
                pending.clear()
            else:
                flushed = pending.pop(mule_id, [])
        else:
            flushed = [p for p in partials if p != key]
            for keys in pending.values():
                keys[:] = [p for p in keys if p not in flushed]
        if key is not None:
            for p in flushed:
                if p in deferred:
                    flush_of[p] = key
    return deferred, expired, flush_of


def _unmerged_uploads(
    cluster_rows: Sequence[dict],
    place: Callable[..., Optional[MissionKey]],
    mule_ids: Sequence[str],
) -> Set[MissionKey]:
    """Missions the cluster logged as ingested but never folded into θ.

    Replays the cluster's open round from its event log, which one service
    thread writes in order. ``HFLHostCluster.ingest_up_bundle`` keeps one
    partial per mule per open round and refuses a second as a duplicate, yet
    the service logs ``up_bundle_ingested`` for it all the same: a mule that
    stopped waiting for its quorum's DOWN (``down_wait_s``) flies on and
    uploads again while its first partial still waits. So an ingest from a
    mule that already holds a partial in the open round never reached θ,
    and nor did a partial still open when the log ends. The round empties at
    ``cluster_round_closed``, at a deferral (FedBuff moves every pending
    partial into its buffer) and at an expiry; a backhaul loss never reached
    the round and leaves it as it was.

    Where the cluster says so, the replay follows it: an ingest marked
    ``partial_refused`` is a refused duplicate whatever the replay holds, and
    a ``backhaul_upload_lost`` marked ``awaits_quorum`` put an empty partial
    in the lost mule's place (unless the round already held one of its),
    which holds that place and carries no update, so it is never itself
    counted here. An applied fold that lists its ``partials`` has the last
    word: an open partial it neither merged nor cut (``expired_partials``)
    did not reach θ, and any mission it merged did, whatever the replay said
    — the fold is what the cluster did, the replay only a model of it.

    Only with several mules: with one, each ingest is answered by a merge,
    deferral or expiry before the next, so no upload is refused, and the
    ledger stays exactly what it was before this existed.
    """
    if len(mule_ids) <= 1:
        return set()
    unmerged: Set[MissionKey] = set()
    folded: Set[MissionKey] = set()
    # The mule of each partial in the open round → its mission (None when it
    # cannot be placed, which still holds the mule's place).
    open_round: Dict[str, Optional[MissionKey]] = {}
    for r in cluster_rows:
        event = r.get("event")
        if event == "up_bundle_ingested":
            mule_id = _opt_str(r.get("mule_id"))
            if mule_id is None:
                continue          # whose upload it was is unknown
            rnd = _opt_int(r.get("mission_round"))
            key = place(mule_id, rnd, _opt_float(r.get("ts")))
            if r.get("partial_refused"):
                # The round kept the partial it already held for this mule;
                # a resend of that same mission changes nothing.
                held = _opt_int(r.get("held_mission_round"))
                if key is not None and (held is None or held != rnd):
                    unmerged.add(key)
            elif mule_id in open_round:
                if key is not None:
                    unmerged.add(key)
            else:
                open_round[mule_id] = key
        elif event == "backhaul_upload_lost":
            mule_id = _opt_str(r.get("mule_id"))
            if r.get("awaits_quorum") and mule_id is not None:
                open_round.setdefault(mule_id, None)
        elif event == "cluster_merge":
            partials = _partial_keys(r.get("partials"), mule_ids)
            if not r.get("applied", True) or partials is None:
                continue
            folded.update(partials)
            named = set(partials) | set(_partial_keys(r.get("expired_partials"), mule_ids) or ())
            unmerged.update(
                k for k in open_round.values() if k is not None and k not in named
            )
            open_round.clear()
        elif event in ("cluster_round_closed", "cluster_merge_deferred", "cluster_merge_expired"):
            open_round.clear()
    unmerged.update(k for k in open_round.values() if k is not None)
    return unmerged - folded


def _runs_a_fold(r: dict) -> bool:
    """Whether a cluster row is an upload whose partial joined the open round
    and ran the fold: an accepted ingest, or the empty partial held for a
    lost upload under a quorum (a refused ingest runs none)."""
    event = r.get("event")
    if event == "up_bundle_ingested":
        return not r.get("partial_refused")
    return event == "backhaul_upload_lost" and bool(r.get("awaits_quorum"))


def _held_places(
    cluster_rows: Sequence[dict], place: Callable[..., Optional[MissionKey]],
) -> Set[MissionKey]:
    """Missions whose upload was lost under a quorum above 1, where the cluster
    held an empty partial in the mule's place (``awaits_quorum``)."""
    keys: Set[MissionKey] = set()
    for r in _events(cluster_rows, "backhaul_upload_lost"):
        if not r.get("awaits_quorum"):
            continue
        key = place(
            _opt_str(r.get("mule_id")), _opt_int(r.get("mission_round")),
            _opt_float(r.get("ts")),
        )
        if key is not None:
            keys.add(key)
    return keys


def _empty_uploads(mule_rows: Sequence[dict], mule_ids: Sequence[str]) -> Set[MissionKey]:
    """Missions that docked with nothing to upload: the ``mission_empty`` rows
    marked ``docked``, which only a mule with ``dock_on_empty`` writes."""
    keys: Set[MissionKey] = set()
    for r in _events(mule_rows, "mission_empty"):
        rnd = _opt_int(r.get("mission_round"))
        if r.get("docked") and rnd is not None:
            keys.add((_mule_key(mule_ids, _opt_str(r.get("id"))), rnd))
    return keys


def _uploader(
    last_up: Tuple[Optional[str], Optional[int]],
    rnd: Optional[int],
    partials: Optional[Sequence[MissionKey]],
) -> Optional[str]:
    """The mule whose upload ran a merge event that names none.

    The cluster emits ``up_bundle_ingested`` and then any merge event its
    fold produced, so the last ingested upload is the uploader when the rounds
    agree. Without one, the only partial at the merge's round is the
    uploader's own.
    """
    up_mule, up_round = last_up
    if up_mule is not None and (rnd is None or up_round is None or up_round == rnd):
        return up_mule
    if rnd is not None and partials:
        own = {p[0] for p in partials if p[1] == rnd}
        if len(own) == 1:
            return own.pop()
    return None


def _round_closers(cluster_rows: Sequence[dict]) -> Dict[int, str]:
    """Cluster round → the mule whose upload closed it.

    ``cluster_round_closed`` follows the ingest of the upload that completed
    the merge (and any merge event between them), so the round belongs to the
    last ingested upload's mule unless the event names its own.
    """
    closers: Dict[int, str] = {}
    last_mule: Optional[str] = None
    for r in cluster_rows:
        event = r.get("event")
        if _runs_a_fold(r):
            last_mule = _opt_str(r.get("mule_id"))
        elif event == "cluster_round_closed":
            rnd = _opt_int(r.get("cluster_round"))
            mule = _opt_str(r.get("mule_id")) or last_mule
            if rnd is not None and mule is not None:
                closers[rnd] = mule
    return closers


def _merged_from_merge_record(
    clean: Tuple[str, ...], merge,
) -> Optional[Tuple[str, ...]]:
    """CLEAN devices minus ``pass_1_merge.excluded``, for traces recorded
    after Phase 1 added the merge record but before the mule emitted
    ``pass_1_merged_devices``. None when there is no merge record."""
    if not isinstance(merge, dict) or not isinstance(merge.get("excluded"), (list, tuple)):
        return None
    excluded = {str(d) for d in merge["excluded"]}
    return tuple(d for d in clean if d not in excluded)


def _partial_keys(raw, mule_ids: Sequence[str]) -> Optional[List[MissionKey]]:
    """Mission keys of a merge's ``partials`` (``[[mule_id, round], ...]``)."""
    if not isinstance(raw, (list, tuple)):
        return None
    keys: List[MissionKey] = []
    for p in raw:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            rnd = _opt_int(p[1])
            if rnd is not None:
                keys.append((_mule_key(mule_ids, _opt_str(p[0])), rnd))
    return keys


def _plan_deadlines(
    raw,
) -> Tuple[Optional[Tuple[Tuple[str, float], ...]], Optional[str]]:
    """``pass_1_plan`` → ``(device, deadline_ts)`` pairs and their basis.

    Each member is held to its own Deadline(j) when its contact records one
    (``device_deadlines``). A contact also carries its tightest member's
    deadline (S3a), which is all a trace recorded before ``device_deadlines``
    has, so there every member falls back to it — and a later member that
    finished after the contact's deadline but before its own counts as late.
    The basis says which was used: ``"device"`` when every member had its own,
    ``"contact"`` when any fell back. Both are None when the plan is absent;
    the basis is also None for a plan that admitted nobody.
    """
    if not isinstance(raw, list):
        return None, None
    pairs: List[Tuple[str, float]] = []
    fell_back = False
    for contact in raw:
        if not isinstance(contact, dict):
            continue
        contact_deadline = _opt_float(contact.get("deadline_ts"))
        own = contact.get("device_deadlines")
        if not isinstance(own, dict):
            own = {}
        for device in contact.get("devices") or ():
            deadline = _opt_float(own.get(str(device)))
            if deadline is None:
                if contact_deadline is None:
                    continue
                deadline = contact_deadline
                fell_back = True
            pairs.append((str(device), deadline))
    if not pairs:
        return (), None
    return tuple(pairs), ("contact" if fell_back else "device")


def _session_outcomes(raw) -> Optional[Tuple[Tuple[str, str, float], ...]]:
    """``pass_1_outcomes`` → ``(device, outcome, contact_ts)``; None when absent."""
    if not isinstance(raw, list):
        return None
    sessions: List[Tuple[str, str, float]] = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        contact_ts = _opt_float(s.get("contact_ts"))
        if s.get("device") is None or s.get("outcome") is None or contact_ts is None:
            continue
        sessions.append((str(s["device"]), str(s["outcome"]), contact_ts))
    return tuple(sessions)


def _ids_or_none(raw) -> Optional[Tuple[str, ...]]:
    """A recorded list of device ids as a tuple of strings; None when absent."""
    return tuple(str(d) for d in raw) if isinstance(raw, (list, tuple)) else None


def _flown_bands(raw) -> Optional[Tuple[Optional[str], ...]]:
    """``pass_1_flown`` → each stop's ``band`` in flight order; None when absent."""
    if not isinstance(raw, list):
        return None
    return tuple(_opt_str(s.get("band")) for s in raw if isinstance(s, dict))


def _policy_drops(raw) -> Optional[Tuple[Tuple[Tuple[str, ...], str], ...]]:
    """``pass_1_policy_drops`` → ``(devices, reason)`` per entry; None when absent.

    The mule writes the field only when a baseline left something out, so an
    absent field means "nothing dropped" on a trace whose build reports drops
    and "not known" on one recorded before; the scorer decides which
    (``traces_scorer.plan_report``).
    """
    if not isinstance(raw, list):
        return None
    drops: List[Tuple[Tuple[str, ...], str]] = []
    for entry in raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("devices"), (list, tuple)):
            continue
        drops.append((tuple(str(d) for d in entry["devices"]), str(entry.get("reason"))))
    return tuple(drops)


def _plan_fields(raw) -> Dict[str, object]:
    """``mission_completed.plan`` → :class:`MissionRecord`'s plan and cap fields.

    The plan is the closed ``PlanCommit.describe()`` (``hermes.types.scheduler``):
    ``band``, ``search``, ``demand``, ``served``, ``score`` (``v`` and, for
    every plan arm, ``mission_s``) and ``cap`` (``s``, ``capped`` and
    ``violations``, each ``{device, age, reason}``). Every field is None when
    the mission recorded no plan, and each one the plan lacks is None too.
    """
    if not isinstance(raw, dict):
        return {}
    score = raw.get("score") if isinstance(raw.get("score"), dict) else {}
    cap = raw.get("cap") if isinstance(raw.get("cap"), dict) else {}
    violations = None
    if isinstance(cap.get("violations"), list):
        violations = tuple(
            (str(v["device"]), int(v["age"]), str(v["reason"]))
            for v in cap["violations"]
            if isinstance(v, dict) and v.get("device") is not None
            and _opt_int(v.get("age")) is not None and v.get("reason") is not None
        )
    return {
        "plan_band": _opt_str(raw.get("band")),
        "plan_search": _opt_str(raw.get("search")),
        # A plan always records its demand (possibly empty): has_plan reads it.
        "plan_demand": _ids_or_none(raw.get("demand")) or (),
        "plan_served": _ids_or_none(raw.get("served")),
        "plan_v": _opt_float(score.get("v")),
        "plan_mission_s": _opt_float(score.get("mission_s")),
        "cap_s": _opt_int(cap.get("s")),
        "cap_capped": _ids_or_none(cap.get("capped")),
        "cap_violations": violations,
    }


# FeRRy Phase 5's records are read strictly (PairDecision): each field only in
# the JSON form the mule writes it in, never coerced as the legacy fields'
# _opt_int, _opt_float, _opt_str and _ids_or_none coerce theirs, which would
# read a bool as index 1, truncate a float index or parse a numeric string.

def _opt_bool(v) -> Optional[bool]:
    """A recorded flag; None when absent or not a bool (the mule writes no 0 or 1)."""
    return v if isinstance(v, bool) else None


def _opt_number(v) -> Optional[float]:
    """A recorded number as a float: a finite int or float, as the mule writes
    every number; None for anything else, a bool or a numeric string included."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return None
    try:
        out = float(v)
    except OverflowError:
        return None
    return out if math.isfinite(out) else None


def _opt_count(v) -> Optional[int]:
    """A recorded count or index: an int >= 0; None for anything else, so a bool
    never reads as 1, a float is never truncated and a numeric string never parsed."""
    if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
        return None
    return int(v)


#: A key the record lacks, told apart from a recorded None (home) by :func:`_stop_index`.
_MISSING = object()


def _stop_index(v) -> Tuple[bool, Optional[int]]:
    """A recorded next stop as ``(read, index)``: an int >= 0, or None for home.
    ``read`` is False for anything else, a missing key included, which must
    therefore never read as home."""
    if v is None:
        return True, None
    index = _opt_count(v)
    return index is not None, index


def _opt_name(v) -> Optional[str]:
    """A recorded name (a class, the fallback, a scorer): a string; else None."""
    return v if isinstance(v, str) else None


def _id_list(raw) -> Optional[Tuple[str, ...]]:
    """A recorded list of device ids: a list of strings; None for anything else."""
    if isinstance(raw, (list, tuple)) and all(isinstance(d, str) for d in raw):
        return tuple(raw)
    return None


def _float_tuple(raw) -> Tuple[float, ...]:
    """A recorded list of numbers as floats; empty when absent, or when any entry
    is not a number, which would leave the rest out of step with their ids."""
    if not isinstance(raw, (list, tuple)):
        return ()
    values = tuple(_opt_number(v) for v in raw)
    return () if any(v is None for v in values) else values


def _pair_decisions(raw) -> Optional[Tuple[PairDecision, ...]]:
    """``pass_1_pairs`` → one :class:`PairDecision` per record, in flight order;
    None when absent. An entry that is not a record, or whose choice is in
    another form (:func:`_pair_decision`), is skipped."""
    if not isinstance(raw, list):
        return None
    read = (_pair_decision(r) for r in raw if isinstance(r, dict))
    return tuple(d for d in read if d is not None)


def _pair_decision(r: Mapping[str, object]) -> Optional[PairDecision]:
    """One closed record of ``pass_1_pairs`` (the slot's ``DECISION_KEYS`` and
    ``CLOSE_KEYS``); None when its choice is in another form: ``band`` not a
    string, or ``next_index`` neither an int >= 0 nor None for home."""
    band = _opt_name(r.get("band"))
    read, next_index = _stop_index(r.get("next_index", _MISSING))
    if band is None or not read:
        return None
    # FX's pair is read whole, so a half that cannot be read never pairs FX's
    # band with home.
    fx_band = _opt_name(r.get("fx_band"))
    fx_read, fx_next = _stop_index(r.get("fx_next", _MISSING))
    if fx_band is None or not fx_read:
        fx_band, fx_next = None, None
    return PairDecision(
        t_s=_opt_number(r.get("t_s")),
        devices=_id_list(r.get("devices")) or (),
        committed=_opt_name(r.get("committed")),
        band=band,
        next_index=next_index,
        # "home" for the dock, else the next stop's members.
        next_devices=_id_list(r.get("next")),
        pairs=_opt_count(r.get("pairs")),
        feasible=_opt_count(r.get("feasible")),
        admitted_pairs=_admitted_pairs(r.get("admitted_pairs")),
        fallback=_opt_name(r.get("fallback")),
        fx_band=fx_band,
        fx_next=fx_next,
        agrees_fx=_opt_bool(r.get("agrees_fx")),
        scorer=_opt_name(r.get("scorer")),
        q=_opt_number(r.get("q")),
        q_fx=_opt_number(r.get("q_fx")),
        collected=_id_list(r.get("collected")) or (),
        w=_float_tuple(r.get("w")),
        late=_id_list(r.get("late")) or (),
        t_next_s=_opt_number(r.get("t_next_s")),
        terminal=_opt_bool(r.get("terminal")),
        trimmed_next=_opt_bool(r.get("trimmed_next")),
    )


def _admitted_pairs(raw) -> Tuple[Tuple[str, Optional[int]], ...]:
    """``admitted_pairs`` (``[[band, index], ...]``, index None for home) as
    ``(band, index)`` tuples. An entry in another form is left out, so an
    index that cannot be read never reads as home."""
    if not isinstance(raw, (list, tuple)):
        return ()
    out: List[Tuple[str, Optional[int]]] = []
    for p in raw:
        if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], str):
            read, index = _stop_index(p[1])
            if read:
                out.append((p[0], index))
    return tuple(out)


def _stop_lists(raw) -> Tuple[Tuple[str, ...], ...]:
    """A recorded list of stops, each its members; empty when absent, or when
    any stop is in another form, so the stops stay index for index with their
    verdicts."""
    if not isinstance(raw, (list, tuple)):
        return ()
    stops = tuple(_id_list(s) for s in raw)
    return () if any(s is None for s in stops) else stops


def _e3_calls(raw) -> Optional[Tuple[E3Call, ...]]:
    """``pass_1_e3`` → one :class:`E3Call` per call, in order; None when absent.
    An entry that is not a record, or whose choice (``next_index``) is in
    another form, is skipped."""
    if not isinstance(raw, list):
        return None
    calls: List[E3Call] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        read, next_index = _stop_index(r.get("next_index", _MISSING))
        if not read:
            continue
        verdicts = r.get("admissible")
        calls.append(E3Call(
            t_s=_opt_number(r.get("t_s")),
            after_stop=_opt_bool(r.get("after_stop")),
            stops=_stop_lists(r.get("stops")),
            # All or nothing, as the stops are.
            admissible=(tuple(verdicts) if isinstance(verdicts, (list, tuple))
                        and all(isinstance(v, bool) for v in verdicts) else ()),
            next_index=next_index,
        ))
    return tuple(calls)


def _e3_unvisited(raw) -> Optional[Tuple[Tuple[str, ...], ...]]:
    """``pass_1_e3_unvisited`` → each stop's members, in order; None when absent.
    An entry that is not a record holding its members (a list of ids) is skipped."""
    if not isinstance(raw, list):
        return None
    members = (_id_list(entry.get("devices")) for entry in raw if isinstance(entry, dict))
    return tuple(m for m in members if m is not None)
