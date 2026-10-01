"""``MuleSupervisor`` — wires the four mule-NUC programs into one runnable.

Phase 6 / Sprint 1 deliverable per the implementation plan: a single
object that owns the four mule-side programs and runs them as a
coherent system.

Programs wired here, per design §4 + §5:

    L1 ChannelDDQN              — RF band selector (read-only here)
    FLScheduler (+ optional
        TargetSelectorRL)       — per-mission visit-queue producer
    HFLHostMission              — runs FL sessions over the RF link
    ClientCluster               — handles dock UP/DOWN + bundle distribution

Intra-NUC bus wiring (design §3 information flow rules):

    HFLHostMission.scheduler_bus  -> FLScheduler.ingest_round_close_delta
                                     (fast-phase deadline)
    BundleDistributor.on_slice_*  -> FLScheduler.ingest_slice
                                     (slow-phase / cluster amendments)
    BundleDistributor.on_next_*   -> MuleSupervisor._stash_next_round_model
                                     (theta_disc + synth for next round)

Sprint 1 scope (in-process loopback only):
    * Same RF/DOCK loopback transports the Phase 2/3 demos use.
    * No real targeted solicitation — the scheduler produces a queue of
      length N and we run N sessions; loopback devices answer FCFS. The
      design's "mule flies to specific device, then solicits" requires
      physical separation that loopback can't model. Sprint 2 / real
      radio adds true targeting.
    * L1 ChannelDDQN is consulted per visit but its choice is recorded,
      not actuated — the loopback radio has no concept of band.

The supervisor is deliberately framework-free: no Flower, no asyncio,
no docker. Sprint 2 wraps it in a process boundary; the supervisor's
contract doesn't change.

FeRRy Phase 3 (the mission clock, design sections 2.2-2.3 and 3.3-3.7): with
``mission_clock=MissionClock()`` the supervisor flies every mission on
simulated time. Flight legs, contact airtime, missed replies, the backhaul
upload and the dock turnaround are charged to the clock; the pose returns to
the dock after each pass; plans, deadlines, the S3b budget and every contact
outcome are stamped from the clock; the in-flight check is priced on it
(``abort``) or checks and repairs the whole remainder (``replan``); a beacon
hook can insert contacts that fit. The glue lives in ``mule/ferry.py``.
Without a clock every path is exactly the recorded one (Freeze Rule 1).

FeRRy Phase 4 (the plan clock, "reach as a decision"; build plan L822-847): with
``plan_mode="ferry"`` the mule commits each mission at the dock to the band class
b̄ and the Pass-1 route as one decision (``FLScheduler.build_ferry_plan``), flies
b̄ in both passes (``FerryRuntime.set_band``), fills the flight slot with the
committed order or arm FX's cross-heuristic, holds the stops whose members the
age cap binds exempt from their own deadline at every departure, and closes the
plan with its visited set once the merge is known. ``member_admission`` passes
member-subset admission through to the scheduler. At the defaults
(``plan_mode="legacy"``, ``member_admission="whole"``) every path is the
recorded one, and the plan package is never loaded.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from hermes.l1.channel_ddqn import ChannelDDQN, L1_STATE_DIM
from hermes.mission import HFLHostMission, MissionSessionError
from hermes.mission.aggregation_rules import AggregationSpec, age_cap
from hermes.mule import BundleDistributor, ClientCluster
from hermes.mule.client_cluster import DownTimeout
from hermes.scheduler import FLScheduler
from hermes.transport import DockLink, RFLink
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    ContactWaypoint,
    DeviceID,
    MissionDeliveryReport,
    MissionPass,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    TargetWaypoint,
    Weights,
)


log = logging.getLogger(__name__)


MulePose = Tuple[float, float, float]

#: The default ``now_fn``: the wall clock, as bound when this module loads.
#: A supervisor on the mission clock refuses any other ``now_fn``.
_WALL_NOW = time.time

#: ``plan_mode``'s values (FeRRy Phase 4), the recorded one first. Restated from
#: ``hermes.scheduler.plan.types.PLAN_MODES`` so that a legacy mule never loads
#: the plan package (Freeze Rule 1); a test pins the two equal.
_PLAN_MODE_LEGACY = "legacy"
_PLAN_MODE_FERRY = "ferry"
_PLAN_MODES: Tuple[str, ...] = (_PLAN_MODE_LEGACY, _PLAN_MODE_FERRY)


class MuleSupervisorError(RuntimeError):
    """Raised when the supervisor's invariants are violated."""


# --------------------------------------------------------------------------- #
# S3c mission accounting — pure, so the ratio that drives window adaptation
# can be tested without a live process tree.
# --------------------------------------------------------------------------- #

def mission_planned_devices(queue, feasibility=None) -> int:
    """Devices this mission INTENDED to serve.

    The queue *plus* whatever S3b already dropped before take-off. Counting
    only the surviving queue would let the gate flatter itself: drop nine
    contacts, serve the tenth, report 100 % success, and never widen — the
    starvation loop, hidden inside its own success metric.

    FeRRy Phase 3 (critic B10): the drops of the simulated energy clause count
    too, and so do the drops of the route-level on-board clause
    (``deadline_bounds="delivery"``). ``dropped_energy`` and
    ``dropped_delivery`` are empty in every legacy plan, so nothing recorded
    moves.

    FeRRy Phase 4 (critic C6): a plan-mode plan's ``dropped_plan``, the demand
    it left out by choice although no clause refused it, is not counted. Those
    devices are no shortfall of the deadline windows, so counting them would
    make S3c widen every window for the plan's own choices. It is empty in
    every legacy plan.
    """
    planned = sum(len(c.devices) for c in queue)
    for wp in (
        list(getattr(feasibility, "dropped_overdue", ()))
        + list(getattr(feasibility, "dropped_budget", ()))
        + list(getattr(feasibility, "dropped_energy", ()))
        + list(getattr(feasibility, "dropped_delivery", ()))
    ):
        planned += len(getattr(wp, "devices", ()))
    return planned


def mission_served_devices(queue, aborted=()) -> int:
    """Devices actually reached: the queue minus any tail the abort abandoned."""
    n_aborted = len(list(aborted))
    return sum(len(c.devices) for c in queue[: len(queue) - n_aborted])


@dataclass
class MissionRunResult:
    """One mission's run, summarised for callers / tests.

    Single-pass (legacy) and two-pass (Sprint 1.5) results both populate
    this struct. Single-pass populates ``queue`` (per-device
    ``TargetWaypoint``s); two-pass populates ``pass_1_queue`` and
    ``pass_2_queue`` (per-contact ``ContactWaypoint``s) plus
    ``delivery_report``. Empty lists / None on the unused fields make
    callers' assertions easy to write either way.

    Sprint 1.5 M5: ``channel_choices`` is the legacy single-pass list;
    two-pass paths populate ``pass_1_channel_choices`` and
    ``pass_2_channel_choices`` separately so callers can attribute L1
    decisions to the right pass without counting against queue lengths.
    """

    mission_round: int
    queue: List[TargetWaypoint] = field(default_factory=list)
    aggregate: Optional[PartialAggregate] = None
    report: Optional[MissionRoundCloseReport] = None
    contacts: Optional[ContactHistory] = None
    channel_choices: List[int] = field(default_factory=list)

    # Sprint 1.5 — two-pass additions
    pass_1_queue: List[ContactWaypoint] = field(default_factory=list)
    pass_2_queue: List[ContactWaypoint] = field(default_factory=list)
    delivery_report: Optional[MissionDeliveryReport] = None
    pass_1_channel_choices: List[int] = field(default_factory=list)
    pass_2_channel_choices: List[int] = field(default_factory=list)
    # EX-4.2 — True when Pass 1 collected zero updates (a recoverable empty
    # round: no dock, no Pass 2, θ carried forward). Still a counted round
    # (0 updates, does not close), but flagged so it isn't mistaken for a
    # productive mission.
    empty: bool = False
    # An empty mission whose sessions DID complete, but every update was past
    # its age cutoff (age-aware rules only): the round report, kept so the
    # trace still shows those on-time sessions instead of scoring them as
    # misses. None otherwise; ``report`` stays None for every empty mission.
    unmerged_report: Optional[MissionRoundCloseReport] = None
    # Each planned device's own Deadline(j) from this mission's Pass-1 plan.
    pass_1_device_deadlines: Dict[DeviceID, float] = field(default_factory=dict)
    # FeRRy Phase 2 (``down_wait_s`` set): the inter-pass dock uploaded but no
    # DOWN came within the wait, so Pass 2 was skipped and the mission's own
    # θ was restaged for the next one. Always False without ``down_wait_s``,
    # where a missing DOWN still ends the mission with an error.
    down_timeout: bool = False
    # FeRRy Phase 2 (``dock_on_empty`` set): this empty mission still docked,
    # uploading an empty partial so a quorum can close without it. True once
    # the upload is sent, whether or not its DOWN then came in time
    # (``down_timeout`` says which): the cluster holds the partial either way.
    docked_empty: bool = False

    # FeRRy Phase 3 — the mission on the simulated mission clock (design
    # section 2.5, ``mission_completed`` sim fields). Every field is None on a
    # wall-clock mission, so legacy results are unchanged. Values are
    # JSON-ready (device ids as str, poses as lists). Times are simulated
    # seconds; energy is SIMULATED (the Zeng 2019 model, not a measurement).
    #
    # ``sim_ledger`` is the clock's per-kind charges for this mission and sums
    # to ``sim_end_s - sim_start_s``. ``pass_1_flown`` / ``pass_2_flown`` hold
    # one entry per stop flown: position, devices, the departure state
    # (``depart_s``, ``depart_pose``, ``depart_energy_j``), ``transit_s``,
    # ``arrival_s``, ``end_s``, ``band``, ``targets``, ``unreachable``,
    # ``snr_db`` and ``rate_bps`` per member at arrival (None without a band),
    # the contact's ``dwell_s`` and ``listen_s``, ``missing``,
    # ``uplink_dropped`` and ``l1_choice``. ``replans`` and ``aborts`` record
    # the in-flight responses, ``inserts`` / ``offers_refused`` the beacon
    # hook. ``budget_overrun_s`` is how far the Pass-1 upload (or the landing,
    # with nothing uploaded) ended past the budget, None without one;
    # ``pass_2_budget_overrun_s`` the same for a budgeted Pass 2. ``backhaul``
    # is the upload (carrier, snr_db, p_loss, t_upload_s = its completion,
    # upload_s, ...), None when nothing was priced — and under the recorded
    # ``mission`` backhaul model, whose upload is still charged to the clock
    # (at the fixed carrier's noise-free mean SNR) but reports no outcome.
    # ``pass_1_preflight_drops`` is a diagnostic: the contacts S3b (and the
    # pre-flight order check) refused before takeoff, which the mule widened
    # then, one entry each with ``position``, ``devices``, ``deadline_ts``
    # and ``reason`` (``overdue``, ``budget``, ``energy`` or, under
    # ``deadline_bounds="delivery"``, ``delivery``, in that order), so a
    # trace shows what emptied or shrank a mission (finding E2E1-01).
    # No predicted home: S3b keeps none, and the record never re-prices.
    # Empty for the whole-scheduler baselines, whose walks report no drops
    # (their report is ``pass_1_policy_drops``, below). In plan mode a fifth
    # reason follows the four, ``plan``: demand the plan left out by choice.
    # ``delivery_overrun_s`` (``deadline_bounds="delivery"`` only, None in
    # every other mode) is how far the Pass-1 upload (or the landing, with
    # nothing uploaded) ended past ``deliver_by``, the earliest Deadline(j)
    # of the updates on board as the in-flight check counted them; 0 when
    # every one was on time or nothing was on board. The route-level bound is
    # re-checked at departures only, so a contact that runs longer than priced
    # at the stop where Pass 1 ends can still land an update late; this says
    # by how much, as ``budget_overrun_s`` does for the budget.
    sim_start_s: Optional[float] = None
    sim_end_s: Optional[float] = None
    sim_ledger: Optional[Dict[str, float]] = None
    sim_pass_2_start_s: Optional[float] = None
    pass_1_flown: Optional[List[Dict[str, Any]]] = None
    pass_2_flown: Optional[List[Dict[str, Any]]] = None
    replans: Optional[List[Dict[str, Any]]] = None
    aborts: Optional[List[Dict[str, Any]]] = None
    inserts: Optional[List[Dict[str, Any]]] = None
    offers_refused: Optional[List[Dict[str, Any]]] = None
    budget_overrun_s: Optional[float] = None
    pass_2_budget_overrun_s: Optional[float] = None
    energy_j: Optional[float] = None
    band: Optional[str] = None
    backhaul: Optional[Dict[str, Any]] = None
    pass_1_preflight_drops: Optional[List[Dict[str, Any]]] = None
    delivery_overrun_s: Optional[float] = None

    # FeRRy Phase 4 — plan mode and the baselines' drop report (the Phase 4
    # spec, other choices 12; the user's decision 6). Each is None at the
    # defaults, so a recorded mission's result keeps its values. In plan mode
    # ``band`` is the mission's committed class b̄, and ``pass_1_flown[].band``
    # the class each stop was flown on (FX may fly another class than b̄ at a
    # Pass-1 stop; Pass 2 flies b̄). ``plan`` is the mission's closed
    # ``PlanCommit.describe()``: JSON-ready and free of wall time, so a
    # repeated trial describes the same plan. ``plan_wall_s`` is the wall time
    # the plan took, the only wall time a result holds, and so left out of
    # every determinism comparison (critic B12). ``pass_1_policy_drops`` is
    # what a whole-scheduler baseline (D1-D5) left out before takeoff on the
    # simulated clock, one entry per contact as in ``pass_1_preflight_drops``
    # plus ``"widened": false``: reported, never widened, since a synthetic
    # miss would move the fields the baselines' round inference reads. None
    # when the baseline left nothing out, so no other trace gains the key
    # (critic D2).
    plan: Optional[Dict[str, Any]] = None
    plan_wall_s: Optional[float] = None
    pass_1_policy_drops: Optional[List[Dict[str, Any]]] = None


def _merged_device_ids(report, agg) -> List[DeviceID]:
    """CLEAN devices of a round report minus those its merge excluded."""
    excluded = set(getattr(agg, "excluded_devices", ()) or ())
    return [
        line.device_id for line in report.lines
        if line.outcome.is_on_time() and line.device_id not in excluded
    ]


def _ids(devices) -> List[str]:
    return [str(d) for d in devices]


class _FerryLog:
    """What one mission on the mission clock did, gathered for its result."""

    def __init__(self) -> None:
        self.flown: Dict[MissionPass, List[Dict[str, Any]]] = {
            MissionPass.COLLECT: [], MissionPass.DELIVER: [],
        }
        self.choices: Dict[MissionPass, List[int]] = {
            MissionPass.COLLECT: [], MissionPass.DELIVER: [],
        }
        self.replans: List[Dict[str, Any]] = []
        self.aborts: List[Dict[str, Any]] = []
        self.inserts: List[Dict[str, Any]] = []
        self.offers_refused: List[Dict[str, Any]] = []
        #: What S3b refused before takeoff (``pass_1_preflight_drops``).
        self.preflight_drops: List[Dict[str, Any]] = []
        #: Devices of accepted inserts: they count in S3c's ``planned``.
        self.inserted_devices: int = 0
        #: Devices the beacon hook must not insert this mission: the Pass-1
        #: plan, its pre-flight drops and everything inserted since.
        self.no_insert: set = set()
        #: Each Pass-1 member's own Deadline(j), as its stop's ``deadline_ts``
        #: was the minimum of: the plan's (``last_plan_deadlines``), and an
        #: inserted member's from its insertion. Read only for the in-flight
        #: ``deliver_by`` of ``deadline_bounds="delivery"``.
        self.deadlines: Dict[DeviceID, float] = {}
        #: The Pass-1 ``deliver_by`` when the mule turned home (``inf``:
        #: nothing on board, or not ``delivery``): ``delivery_overrun_s``.
        self.deliver_by: float = math.inf
        #: What a whole-scheduler baseline left out before takeoff
        #: (``pass_1_policy_drops``, FeRRy Phase 4 decision 6): never widened.
        self.policy_drops: List[Dict[str, Any]] = []
        #: Plan mode's capped devices (FeRRy Phase 4): their updates do not
        #: lower the in-flight ``deliver_by``, as a protected stop's deadline
        #: does not in the predicate. Empty in legacy mode.
        self.capped: FrozenSet[DeviceID] = frozenset()


class MuleSupervisor:
    """Per-mule supervisor: one instance per mobile AVN.

    The supervisor doesn't own the cluster — it speaks to ``DockLink``.
    In Sprint 1 the cluster runs in-process and shares the link; in
    Sprint 2 they're separate processes and the link is a TCP socket.

    FeRRy Phase 3 (design section 2.2). ``mission_clock`` (a
    ``hermes.l1.mission_clock.MissionClock``) puts the mule on simulated
    mission time, configured by ``ferry`` (a ``hermes.mule.ferry.FerrySpec``;
    None is the channel-free control, critic A1). The clock becomes ``_now``
    (the attribute keeps its name, and nothing is stored as ``_clock``: tests
    bind supervisor methods onto stand-ins that carry their own ``_clock``,
    critic B7), the scheduler's ``now_fn`` and the mission server's
    ``now_fn``; the scheduler prices with the ferry model, refuses the
    cluster's wall-clock deadline overrides, and validates the flown order
    under ``replan``. ``feasibility_model`` gives the planner's cruise speed
    and session time (the defaults when None); in sim mode its cruise speed
    must be the flight model's. ``deadline_time_scale`` and
    ``initial_window_s`` restate the deadline law's time unit (spec Q1; 1.0
    and None are the recorded law). Refused: a clock together with a
    ``now_fn``, a ``FerrySpec`` (or a ferry-physics model) without the clock,
    and the clock without ``rf_range_m`` (the single-pass path is not ported).
    Without a clock every path is the recorded one.

    FeRRy Phase 4, on the mission clock only. ``member_admission`` (``whole``,
    recorded, or ``subset``; the user's decision 4 (b)) is passed through to
    the scheduler, its one runtime switch, and only when it is not the
    recorded value. ``plan_mode="ferry"`` plans each mission with the plan
    clock: ``plan_options`` (a ``hermes.scheduler.plan.PlanOptions``, the
    ``MuleConfig`` plan fields) and ``t_nom_s`` (T in the plan score) are
    required then and refused in legacy mode, where the plan package is never
    loaded (:meth:`_init_plan`). ``plan_mode="legacy"``, the default, is every
    recorded path.
    """

    def __init__(
        self,
        *,
        mule_id: MuleID,
        rf: RFLink,
        dock: DockLink,
        target_selector=None,
        channel_actor: Optional[ChannelDDQN] = None,
        mule_pose: MulePose = (0.0, 0.0, 0.0),
        mule_energy: float = 1.0,
        rf_prior_snr_db: float = 20.0,
        beacon_window_s: float = 30.0,
        session_ttl_s: float = 5.0,
        rf_range_m: Optional[float] = None,
        mission_budget_s: Optional[float] = None,
        mission_window_adapter=None,
        now_fn=time.time,
        aggregation: Optional[AggregationSpec] = None,
        pass_2_budget: bool = False,
        deadline_law=None,
        miss_priority: bool = False,
        down_wait_s: Optional[float] = None,
        dock_on_empty: bool = False,
        should_stop: Optional[Callable[[], bool]] = None,
        deadline_time_scale: float = 1.0,
        initial_window_s: Optional[float] = None,
        feasibility_model=None,
        mission_clock=None,
        ferry=None,
        member_admission: str = "whole",
        plan_mode: str = _PLAN_MODE_LEGACY,
        plan_options=None,
        t_nom_s: Optional[float] = None,
    ) -> None:
        self.mule_id = mule_id
        self.mule_pose = mule_pose
        self.mule_energy = mule_energy
        self.rf_prior_snr_db = rf_prior_snr_db
        self.rf_range_m = rf_range_m
        # FeRRy Phase 3 — the mission clock (design section 2.2). ``ferry`` is
        # the FerrySpec and ``_ferry_run`` its runtime; both stay None on the
        # wall clock, and every legacy branch reads them with getattr, since
        # tests bind supervisor methods onto stand-ins (critic B7).
        self.ferry = None
        self._ferry_run = None
        # A DOWN whose slice the sim-mode scheduler refused (see
        # _on_slice_and_amendment); raised once the dock returns.
        self._ferry_slice_error: Optional[BaseException] = None
        # The beacon hook's queued offers (design section 3.6); inert unless
        # something calls offer_contact.
        self._offers: List[ContactWaypoint] = []
        self._offers_lock = threading.Lock()
        sched_extra: Dict[str, Any] = {}
        # FeRRy Phase 3 (spec Q1) — the deadline law's time unit. Passed only
        # when it is not the recorded one, so a recorded mule builds its
        # scheduler with exactly the arguments it always did.
        if isinstance(deadline_time_scale, bool) or deadline_time_scale != 1.0:
            sched_extra["deadline_time_scale"] = deadline_time_scale
        if initial_window_s is not None:
            sched_extra["initial_window_s"] = initial_window_s
        # FeRRy Phase 4 — the flight slot, which only plan mode fills
        # (_init_plan). None keeps every recorded path, and each branch reads
        # it with getattr, since tests bind supervisor methods onto stand-ins
        # (critic B7).
        self._flight_slot = None
        if plan_mode not in _PLAN_MODES:
            raise MuleSupervisorError(f"plan_mode must be one of {_PLAN_MODES}, got {plan_mode!r}")
        if plan_mode == _PLAN_MODE_LEGACY and (plan_options is not None or t_nom_s is not None):
            raise MuleSupervisorError(
                "plan_options and t_nom_s configure plan mode (plan_mode='ferry'); a legacy "
                "mule plans with build_contact_queue"
            )
        host_now = None
        if mission_clock is None:
            if ferry is not None:
                raise MuleSupervisorError(
                    "a FerrySpec (contact band, replan, ...) runs on the mission "
                    "clock: pass mission_clock=MissionClock() as well"
                )
            if getattr(feasibility_model, "ferry", None) is not None:
                raise MuleSupervisorError(
                    "a feasibility model with ferry physics prices simulated seconds; "
                    "it needs the mission clock"
                )
            # FeRRy Phase 4 (unit_U3b.md section 1.4): member subsets price each
            # member with the ferry physics, and the plan clock plans with it.
            from hermes.scheduler.stages.s3b_feasibility import MEMBER_ADMISSION_WHOLE

            if member_admission != MEMBER_ADMISSION_WHOLE:
                raise MuleSupervisorError(
                    f"member_admission={member_admission!r} prices each member with the "
                    "ferry physics: it runs on the mission clock (mission_clock=MissionClock())"
                )
            if plan_mode != _PLAN_MODE_LEGACY:
                raise MuleSupervisorError(
                    "plan_mode='ferry' plans on the mission clock: pass "
                    "mission_clock=MissionClock() and a FerrySpec with a contact band as well"
                )
            if feasibility_model is not None:
                sched_extra["feasibility_model"] = feasibility_model
        else:
            self._init_ferry(
                mission_clock, ferry, feasibility_model,
                now_fn=now_fn, rf_range_m=rf_range_m, sched_extra=sched_extra,
                member_admission=member_admission, plan_mode=plan_mode,
                plan_options=plan_options, t_nom_s=t_nom_s, pass_2_budget=pass_2_budget,
            )
            now_fn = mission_clock
            host_now = mission_clock
        self._now = now_fn
        # FeRRy Phase 2 — several mules share one cluster. ``down_wait_s``
        # bounds how long the inter-pass dock waits for its DOWN, and makes
        # running out survivable (Pass 2 skipped, the mission's θ kept): with a
        # quorum the answer can take as long as the slowest other mule's
        # mission. None keeps the recorded single 10 s wait, whose timeout ends
        # the mission loop. ``dock_on_empty`` makes a mission that collected
        # nothing dock anyway, with an empty partial, so a quorum that counts
        # this mule can still close. Both are off for one mule.
        if down_wait_s is not None and not float(down_wait_s) > 0.0:
            raise MuleSupervisorError(f"down_wait_s must be > 0, got {down_wait_s}")
        self.down_wait_s: Optional[float] = (
            None if down_wait_s is None else float(down_wait_s)
        )
        self.dock_on_empty = bool(dock_on_empty)
        self._should_stop: Callable[[], bool] = should_stop or (lambda: False)
        # FeRRy Phase 1. The merge rule (agg:plain, the recorded merge, by
        # default) and whether Pass 2 is walked against the budget. Off, Pass 2
        # delivers to the whole slice, so every basis is current and ages never
        # spread; on, devices it cannot reach keep their older basis.
        self.aggregation: AggregationSpec = aggregation or AggregationSpec()
        self.pass_2_budget = bool(pass_2_budget)

        # Scheduler — slow-phase amendments + fast-phase round-close deltas
        # are wired through here.
        self.scheduler = FLScheduler(
            beacon_window_s=beacon_window_s,
            now_fn=now_fn,
            target_selector=target_selector,
            # S3b — when set, the deadline stops being a sort key and becomes
            # an enforced constraint (see stages/s3b_feasibility.py).
            mission_budget_s=mission_budget_s,
            # S3c — when an *enabled* adapter is supplied, systemic mission
            # shortfall widens every device's window together
            # (see stages/s3c_mission_window.py).
            mission_window_adapter=mission_window_adapter,
            # FeRRy Phase 1 — the deadline law (None = the recorded additive
            # law) and the miss-streak priority key in S3b.
            deadline_law=deadline_law,
            miss_priority=miss_priority,
            # FeRRy Phase 3 — a time unit other than the recorded one, and on
            # the mission clock the ferry model, the override refusal and the
            # flown-order check (see _init_ferry). Empty for a recorded mule.
            **sched_extra,
        )

        # Mission server — emits one RoundCloseDelta per session into the
        # scheduler bus, which folds it into Deadline(j) immediately.
        self.mission = HFLHostMission(
            mule_id=mule_id,
            rf=rf,
            scheduler_bus=self.scheduler.ingest_round_close_delta,
            session_ttl_s=session_ttl_s,
            aggregation=self.aggregation,
            # A budgeted Pass 2 may not come back to a device, so Pass 1 asks
            # it to train ahead on the basis it adopts (audit #0).
            train_ahead=self.pass_2_budget,
            # FeRRy Phase 3 — None keeps the host's wall stamps; on the
            # mission clock it stamps its reports, and every contact needs a
            # ContactPlan on the same clock (critic A1).
            now_fn=host_now,
        )

        # ClientCluster owns the dock lifecycle. The distributor fans the
        # DOWN bundle out to: scheduler (slice + amendment) and supervisor
        # (theta + synth, stashed for the next open_round).
        self._next_theta: Optional[Weights] = None
        self._next_synth = None
        # Version of ``_next_theta``: the cluster round that produced it. It
        # rides every push so each update comes back with an age.
        self._next_theta_version: Optional[int] = None
        self._incoming_theta_version: Optional[int] = None
        # H3 — Pass-2 delivery report from the most recent mission, held
        # locally until the next mission's Pass-1 dock UPs it as
        # ``UpBundle.prev_mission_delivery_report``.
        self._pending_delivery_report: Optional[MissionDeliveryReport] = None
        self.distributor = BundleDistributor(
            on_slice_and_amendment=self._on_slice_and_amendment,
            on_next_round_model=self._on_next_round_model,
            on_model_version=self._on_model_version,
        )
        self.client_cluster = ClientCluster(
            mule_id=mule_id,
            dock=dock,
            distributor=self.distributor,
            # A DOWN that arrives after a survived wait is stale by the next
            # dock; only then is there anything to drain before an upload.
            recoverable_down_wait=self.down_wait_s is not None,
        )

        # L1 channel actor — optional. When None, channel choice is not
        # logged at all (no actuation in loopback either way).
        self.channel_actor = channel_actor

    def _init_ferry(
        self,
        mission_clock,
        ferry,
        feasibility_model,
        *,
        now_fn,
        rf_range_m: Optional[float],
        sched_extra: Dict[str, Any],
        member_admission: str = "whole",
        plan_mode: str = _PLAN_MODE_LEGACY,
        plan_options=None,
        t_nom_s: Optional[float] = None,
        pass_2_budget: bool = False,
    ) -> None:
        """Wire the mission clock and the ferry runtime (design section 2.2).

        FeRRy Phase 4: in plan mode, also the planner's setup and the flight
        slot (:meth:`_init_plan`); in legacy mode ``member_admission`` reaches
        the scheduler only when it is not the recorded ``whole`` (unit_U3b.md
        section 5.3), so a recorded mule builds its scheduler with exactly the
        arguments it always did.
        """
        from hermes.mule.ferry import RESPONSE_REPLAN, FerryRuntime, FerrySpec
        from hermes.scheduler.stages.s3b_feasibility import (
            MEMBER_ADMISSION_WHOLE,
            FeasibilityModel,
        )

        if now_fn is not _WALL_NOW:
            raise MuleSupervisorError(
                "mission_clock and now_fn are exclusive: on the mission clock every "
                "mission-time read comes from the clock"
            )
        if rf_range_m is None:
            raise MuleSupervisorError(
                "the mission clock runs the two-pass contact path: set rf_range_m "
                "(the single-pass path is not ported)"
            )
        if not callable(mission_clock) or not all(
            callable(getattr(mission_clock, attr, None))
            for attr in ("advance", "advance_to", "ledger", "reset_ledger")
        ):
            raise MuleSupervisorError(
                "mission_clock must be a MissionClock (callable, with advance, "
                f"advance_to, ledger and reset_ledger), got {type(mission_clock).__name__}"
            )
        spec = FerrySpec() if ferry is None else ferry
        if not isinstance(spec, FerrySpec):
            raise MuleSupervisorError(f"ferry must be a FerrySpec, got {type(spec).__name__}")
        base = (feasibility_model if feasibility_model is not None
                else FeasibilityModel(cruise_speed_m_s=spec.flight.cruise_speed_m_s))
        try:
            run = FerryRuntime(spec, mission_clock, rf_range_m=float(rf_range_m),
                               session_time_s=base.session_time_s)
            model = run.feasibility_model(base)
        except ValueError as e:
            raise MuleSupervisorError(str(e)) from e
        if tuple(float(c) for c in self.mule_pose) != spec.flight.dock:
            raise MuleSupervisorError(
                f"a mule on the mission clock starts at the dock {spec.flight.dock!r}, "
                f"not at {tuple(self.mule_pose)!r}"
            )
        self.mule_pose = spec.flight.dock
        self.ferry = spec
        self._ferry_run = run
        sched_extra.update(
            feasibility_model=model,
            validate_flown_order=spec.in_flight_response == RESPONSE_REPLAN,
            refuse_deadline_overrides=True,
            replan_fallback=spec.replan_fallback,
        )
        if plan_mode == _PLAN_MODE_FERRY:
            self._flight_slot = self._init_plan(
                spec, run, base, plan_options=plan_options, t_nom_s=t_nom_s,
                member_admission=member_admission, pass_2_budget=pass_2_budget,
                sched_extra=sched_extra,
            )
        elif member_admission != MEMBER_ADMISSION_WHOLE:
            sched_extra["member_admission"] = member_admission

    def _init_plan(
        self,
        spec,
        run,
        base,
        *,
        plan_options,
        t_nom_s: Optional[float],
        member_admission: str,
        pass_2_budget: bool,
        sched_extra: Dict[str, Any],
    ):
        """Plan mode's wiring (FeRRy Phase 4; the Phase 4 spec, other choices 1, 2 and 10).

        The planner's setup is built once: one class per class of the contact
        link, each model bound to its own class (``FerryRuntime.plan_classes``,
        critic B3) on the cost model's ``base``; the run's ``contact_band`` as
        the reference class (under ``fixed:<c>``, the class pinned); T =
        ``t_nom_s``, since the score measures the whole mission against the
        cell's nominal period (the user's decision 2 (b)); and the dock
        turnaround that mission counts between the passes. The scheduler gets
        ``plan_mode="ferry"``, the setup and ``member_admission``, which is the
        one switch: the scheduler refuses options that say otherwise
        (unit_U3b.md section 1.1; R5), as it refuses the ``reorder`` fallback
        (critic B11) and a target selector. Returns the flight slot the
        options name (``policies.cross_heuristic.flight_slot_policy``).

        Refused here, each for its own reason: options that are not
        ``PlanOptions``; the channel-free control, which has no band classes to
        choose among; no ``t_nom_s``; ``pass_2_budget``, whose walk admits whole
        stops, so a narrow Pass 2 at 1 MB (45 s at the median) would deliver
        nothing under a short budget (critic B8); and ``abort`` together with a
        cap, since abort gives up the whole tail, capped stops that would fit
        alone included (critic A10). The plan package is imported only here
        and on plan mode's paths, so a legacy mule never loads it.
        """
        from hermes.mule.ferry import RESPONSE_ABORT
        from hermes.scheduler.plan.types import PLAN_MODE_FERRY, PlanOptions, PlanSetup
        from hermes.scheduler.policies.cross_heuristic import flight_slot_policy

        if not isinstance(plan_options, PlanOptions):
            raise MuleSupervisorError(
                "plan_mode='ferry' needs plan_options, a hermes.scheduler.plan.PlanOptions "
                f"(the MuleConfig plan fields), got {plan_options!r}"
            )
        if not spec.banded:
            raise MuleSupervisorError(
                "plan_mode='ferry' chooses the band class each mission: it needs a contact "
                "band (the channel-free control has no band classes)"
            )
        if t_nom_s is None:
            raise MuleSupervisorError(
                "plan_mode='ferry' scores the whole mission against T_nom: set t_nom_s"
            )
        if pass_2_budget:
            raise MuleSupervisorError(
                "plan_mode='ferry' refuses pass_2_budget: the budgeted Pass-2 walk admits "
                "whole stops, so a narrow Pass 2 would deliver nothing under a short budget "
                "(critic B8)"
            )
        if plan_options.cap.enabled and spec.in_flight_response == RESPONSE_ABORT:
            raise MuleSupervisorError(
                "in_flight_response='abort' gives up the whole tail, capped stops that would "
                "fit alone included: it is refused with an age cap (critic A10); use 'replan'"
            )
        try:
            setup = PlanSetup(
                options=plan_options, classes=run.plan_classes(base), reference=spec.band,
                t_ref_s=t_nom_s, turnaround_s=spec.flight.turnaround_s,
            )
        except (TypeError, ValueError) as e:
            raise MuleSupervisorError(f"plan_mode='ferry': {e}") from e
        sched_extra.update(
            plan_mode=PLAN_MODE_FERRY, plan=setup, member_admission=member_admission,
        )
        return flight_slot_policy(plan_options.flight_slot)

    @property
    def mission_clock(self):
        """The mission clock on a sim-mode mule, None on the wall clock."""
        return self._now if getattr(self, "_ferry_run", None) is not None else None

    # ------------------------------------------------------------------ #
    # Distribution callbacks
    # ------------------------------------------------------------------ #

    def _on_slice_and_amendment(
        self,
        mission_slice: MissionSlice,
        amendment: Optional[ClusterAmendment],
    ) -> None:
        # Sprint 1.5 H7: positions and delivery_priority arrive on the
        # mule via ``ClusterAmendment.registry_deltas`` — the cluster's
        # ``dispatch_down_bundle`` enriches each delta with the device's
        # ``last_known_position`` from the registry, and the scheduler's
        # ``fold_cluster_amendment`` writes them into ``DeviceSchedulerState``.
        # The MissionSlice itself doesn't carry full DeviceRecord rows
        # through the distributor today; if you ever disable the H7
        # cluster-side fold, position resets to (0,0,0) and S3a clusters
        # everything at the origin — which the two-pass test would catch
        # but only at the integration level.
        if getattr(self, "_ferry_run", None) is None:
            self.scheduler.ingest_slice(mission_slice, amendment=amendment)
            return
        # FeRRy Phase 3 (critic A1/B3): on the mission clock the scheduler
        # refuses a DOWN whose amendment carries deadline overrides (they are
        # wall-clock stamps), and with it the whole slice. ClientCluster logs a
        # failing sink and still stages θ, so the mule would fly on without
        # its slice; the failure is kept and raised by the supervisor once the
        # dock returns (_ferry_check_slice).
        try:
            self.scheduler.ingest_slice(mission_slice, amendment=amendment)
        except Exception as e:
            self._ferry_slice_error = e
            raise

    def _on_model_version(self, version: int) -> None:
        # Arrives just before the model itself, from the same DOWN bundle.
        self._incoming_theta_version = int(version)

    def _on_next_round_model(self, theta: Weights, synth_batch) -> None:
        self._next_theta = theta
        self._next_synth = synth_batch
        # Keep the version the DOWN bundle issued with θ (FeRRy Phase 1); it
        # used to be dropped here, so no update could be aged.
        self._next_theta_version = self._incoming_theta_version
        self._incoming_theta_version = None

    # ------------------------------------------------------------------ #
    # Mission cycle
    # ------------------------------------------------------------------ #

    def wait_for_initial_dock(self, timeout: Optional[float] = None) -> bool:
        """Block until the first DOWN bundle arrives + is distributed.

        Required before the first ``run_one_mission`` call so the mule
        knows what slice it owns. Uses the DOWN-only bootstrap path
        because the mule has no aggregate to upload yet.

        ``timeout`` also bounds the DOWN wait (FeRRy Phase 2): a bootstrap not
        yet sent returns False rather than raising, so a caller polling in
        ticks keeps its own window and its timeout branch. The DOWN wait used
        to be a fixed 10 s whose expiry raised past every caller.
        """
        if not self.client_cluster.wait_for_dock(timeout=timeout):
            return False
        wait_s = (
            None if timeout is None
            else min(float(timeout), self.client_cluster.down_timeout_s)
        )
        down = self.client_cluster.bootstrap_down_only(timeout=wait_s)
        if down is not None and getattr(self, "_ferry_run", None) is not None:
            # FeRRy Phase 3: a slice the scheduler refused ends the bootstrap
            # (MuleSupervisorError) rather than leaving the mule with none.
            # Critic B8: a mule that joins a running cluster, or was restarted
            # with a fresh clock at the epoch, adopts the cluster's simulated
            # time from its bootstrap DOWN.
            self._ferry_check_slice()
            self._ferry_sync(down)
        return down is not None

    def run_one_mission(self) -> MissionRunResult:
        """One end-to-end mission.

        If ``rf_range_m`` was set at construction time, runs the Sprint 1.5
        two-pass + per-contact path:
            Pass 1 (collect) → dock UP/DOWN → Pass 2 (deliver) → stash
            delivery report for the next mission's UP.

        Otherwise runs the legacy single-pass + per-device path:
            One circuit of run_session calls → dock UP/DOWN.

        Postcondition: a fresh DOWN bundle has been distributed
        intra-NUC, so ``self._next_theta`` is staged for the next call.
        """
        if self._next_theta is None:
            raise MuleSupervisorError(
                "run_one_mission called without a next-round model staged; "
                "did you call wait_for_initial_dock first?"
            )

        # FeRRy Phase 3 — on the mission clock the takeoff itself stamps the
        # budget, after the pose check and the ledger reset.
        ferry_run = getattr(self, "_ferry_run", None)
        if ferry_run is not None:
            return self._run_ferry_mission(ferry_run)

        # Freeze Amendment 6 — each mission's budget runs from its own start.
        # The scheduler also stamps on every DOWN bundle, but a DOWN arrives
        # mid-mission (the inter-pass dock) and not at all after an empty
        # mission, which left the next mission planning against a stale stamp.
        self.scheduler.start_mission()

        if self.rf_range_m is not None:
            return self._run_two_pass_mission()
        return self._run_single_pass_mission()

    def _run_single_pass_mission(self) -> MissionRunResult:
        """Legacy Sprint-1A path: per-device queue + run_session FCFS."""

        theta = self._next_theta
        synth = self._next_synth
        theta_version = self._next_theta_version
        # Consumed — the next dock cycle restages.
        self._next_theta = None
        self._next_synth = None
        self._next_theta_version = None

        # 1. Open round on the mission server.
        mission_round = self.mission.open_round(theta, theta_version=theta_version)
        self.scheduler.set_mission_round(mission_round)

        # 2. Build the visit queue from the scheduler.
        queue = self.scheduler.build_target_queue(
            mule_pose=self.mule_pose,
            mule_energy=self.mule_energy,
            rf_prior_snr_db=self.rf_prior_snr_db,
        )
        # The merge cutoff reads the window each device was planned under,
        # before this mission's sessions fold their outcomes into it.
        planned_caps = self._age_caps()
        log.info(
            "mule=%s round=%d queue_size=%d",
            self.mule_id, mission_round, len(queue),
        )

        # 3. Visit each waypoint. In loopback, run_session() answers FCFS;
        #    in Sprint 2 / real RF this becomes a targeted solicitation.
        channel_choices: List[int] = []
        for wp in queue:
            ch_idx = self._pick_channel(wp)
            if ch_idx is not None:
                channel_choices.append(ch_idx)
            try:
                self.mission.run_session(synth_batch=synth)
            except MissionSessionError as e:
                # One bad session shouldn't kill the round — log + carry on.
                log.warning(
                    "mule=%s round=%d session error on %s: %s",
                    self.mule_id, mission_round, wp.device_id, e,
                )

        # 4. Close round → partial FedAvg + report + contacts.
        try:
            agg, report, contacts = self.mission.close_round(
                age_caps=planned_caps,
            )
        except MissionSessionError as e:
            # No clean gradients — abort the dock cycle and let the
            # caller decide whether to skip the UP.
            log.error(
                "mule=%s round=%d close_round failed: %s",
                self.mule_id, mission_round, e,
            )
            raise

        self.scheduler.record_merged(
            _merged_device_ids(report, agg), mission_round,
        )

        # 5. Hand off to ClientCluster, dock cycle (UP + DOWN).
        self.client_cluster.collect(
            partial_aggregate=agg, report=report, contacts=contacts,
        )
        if not self.client_cluster.wait_for_dock(timeout=None):
            raise MuleSupervisorError("dock did not become available")
        # run_dock_cycle distributes DOWN through the BundleDistributor,
        # which restages _next_theta + _next_synth via the callbacks.
        docked = self._dock_and_await_down()
        if not docked:
            # FeRRy Phase 2: no DOWN within down_wait_s. Fly the next mission
            # on this one's θ rather than stopping.
            self._restage(theta, synth, theta_version)

        return MissionRunResult(
            mission_round=mission_round,
            queue=list(queue),
            aggregate=agg,
            report=report,
            contacts=contacts,
            channel_choices=channel_choices,
            down_timeout=not docked,
        )

    # ------------------------------------------------------------------ #
    # Sprint 1.5 — two-pass mission
    # ------------------------------------------------------------------ #

    def _run_two_pass_mission(self) -> MissionRunResult:
        """Sprint 1.5 path: Pass 1 (collect) → dock → Pass 2 (deliver).

        Mission timeline:
            1. open_round(theta) — Pass 1 starts in COLLECT mode
            2. build_contact_queue — slice → S3a clustering → bucket order
            3. for each contact: pick_channel + run_contact (parallel
               exchange-only sessions)
            4. close_round — partial-FedAvg → mission_aggregate +
               MissionRoundCloseReport + ContactHistory
            5. inter-pass dock — UP the aggregate, DOWN the cluster's
               freshly-aggregated θ' for Pass 2
            6. open_pass_2(theta_new) — switch HFLHostMission to DELIVER
            7. build_pass_2_queue — every slice contact, nearest-first
            8. for each contact: pick_channel + deliver_contact (push θ',
               collect DeliveryAck)
            9. close_pass_2 — MissionDeliveryReport
            10. stash delivery report for cluster ingest at next dock
                (Chunk F wires the UP-bundle field).
        """
        assert self.rf_range_m is not None
        rf_range_m = self.rf_range_m

        theta_pass_1 = self._next_theta
        synth_pass_1 = self._next_synth
        version_pass_1 = self._next_theta_version
        self._next_theta = None
        self._next_synth = None
        self._next_theta_version = None

        # ============================ Pass 1 ============================
        mission_round = self.mission.open_round(
            theta_pass_1, theta_version=version_pass_1,
        )
        self.scheduler.set_mission_round(mission_round)
        pass_1_queue = self.scheduler.build_contact_queue(
            rf_range_m=rf_range_m,
            mule_pose=self.mule_pose,
            mule_energy=self.mule_energy,
            rf_prior_snr_db=self.rf_prior_snr_db,
        )
        # Snapshot what the plan was made with before any session folds its
        # outcome into Φ or S3c moves its scale: the merge cutoff a_max_j must
        # read the window the device was admitted under (audit #2), and the
        # trace scores each device against its own deadline (audit #13).
        planned_caps = self._age_caps()
        planned_deadlines = dict(
            getattr(self.scheduler, "last_plan_deadlines", None) or {}
        )
        log.info(
            "mule=%s round=%d pass=1 contacts=%d devices_total=%d",
            self.mule_id, mission_round, len(pass_1_queue),
            sum(len(c.devices) for c in pass_1_queue),
        )
        # Close the other half of the starvation loop: devices S3b dropped
        # PRE-flight also never get a contact, so they too would never widen.
        _feas = getattr(self.scheduler, "last_feasibility", None)
        if _feas is not None and getattr(_feas, "n_dropped", 0):
            self._widen_abandoned(
                list(_feas.dropped_overdue) + list(_feas.dropped_budget),
                mission_round=mission_round,
            )

        planned_devices = mission_planned_devices(pass_1_queue, _feas)

        # M5 — split channel choices per pass for clean attribution.
        pass_1_channel_choices: List[int] = []
        aborted_wps: List[ContactWaypoint] = []
        for idx, wp in enumerate(pass_1_queue):
            # S3b in-flight — re-check feasibility from where the mule ACTUALLY
            # is and what the clock ACTUALLY says, not from the pre-flight plan.
            #
            # The queue was filtered before take-off, but contacts take real
            # time and can fail, so the mule can fall behind its own plan. Once
            # the remainder is unreachable there is nothing to gain by flying
            # the rest of it: continuing burns budget and delays delivery of the
            # updates already aboard. Break, and let close_round + the dock
            # deliver what was collected.
            #
            # Note this can only foresee running out of TIME — a deterministic
            # function of clock and geometry. It cannot foresee a random link
            # failure, which is stochastic by construction.
            if not self._remaining_is_feasible(pass_1_queue[idx:]):
                aborted_wps = list(pass_1_queue[idx:])
                log.info(
                    "mule=%s round=%d Pass 1 ABORTING at contact %d/%d: "
                    "remaining queue is no longer reachable in time; "
                    "returning with %d contact(s) already collected",
                    self.mule_id, mission_round, idx, len(pass_1_queue), idx,
                )
                # Close the feedback loop for everyone we are abandoning.
                # Without this they receive NO RoundCloseDelta at all (deltas
                # are only emitted from inside a contact session), so their
                # fulfilment window never widens and S3b is free to drop them
                # again next mission — a starvation loop introduced by the gate
                # itself. Feeding a TIMEOUT widens Φ, which is exactly the
                # signal "we could not get to you in time".
                self._widen_abandoned(aborted_wps, mission_round=mission_round)
                break
            ch_idx = self._pick_channel_contact(wp)
            if ch_idx is not None:
                pass_1_channel_choices.append(ch_idx)
            try:
                self.mission.run_contact(
                    contact_devices=list(wp.devices),
                    synth_batch=synth_pass_1,
                )
            except MissionSessionError as e:
                log.warning(
                    "mule=%s round=%d Pass 1 run_contact failed pos=%s: %s",
                    self.mule_id, mission_round, wp.position, e,
                )
            # H5 — advance the mule's tracked pose to this contact's
            # stop position so the next contact's selector inputs and
            # the Pass-2 nearest-first ordering both plan from the
            # *current* location, not from origin.
            self.mule_pose = wp.position

        # S3c — hand this mission's outcome to the mission-level adapter.
        # No-op unless an enabled adapter is attached.
        _served_devices = mission_served_devices(pass_1_queue, aborted_wps)
        self.scheduler.record_mission_outcome(
            served=_served_devices, planned=planned_devices,
        )
        if getattr(self.scheduler, "_window_adapter", None) is not None:
            log.info(
                "mule=%s round=%d S3c mission outcome served=%d/%d -> %s",
                self.mule_id, mission_round, _served_devices, planned_devices,
                self.scheduler._window_adapter.describe(),
            )

        try:
            agg, report, contacts = self.mission.close_round(
                age_caps=planned_caps,
            )
        except MissionSessionError as e:
            # EX-4.2: no device uplink succeeded this Pass 1 (e.g. under lossy
            # short-range links). This is a recoverable outcome, not a fatal
            # error: per the design principle that FL never stalls on absent
            # devices, we record a zero-update round, skip the inter-pass dock
            # and Pass 2 (nothing to aggregate or deliver), restage the same θ,
            # and let the sortie continue to the next mission. Under agg:cutoff
            # the same holds when every update collected was past its cutoff.
            log.warning(
                "mule=%s round=%d Pass 1 collected no updates; recording an "
                "empty round and continuing: %s",
                self.mule_id, mission_round, e,
            )
            # ``answered``: the DOWN staged the next θ. An upload whose DOWN
            # did not come in time still docked: the cluster holds its empty
            # partial, so the trace must say it uploaded.
            answered = False
            if self.dock_on_empty:
                # FeRRy Phase 2: dock anyway. The empty partial counts toward
                # the cluster's quorum and adds nothing to θ, and the DOWN
                # brings the θ the other mules have moved on to. Pass 2 is
                # still skipped: that θ reaches the devices with the next
                # mission's Pass-1 push.
                answered = self._dock_empty(mission_round, version_pass_1)
            if not answered:
                self._restage(theta_pass_1, synth_pass_1, version_pass_1)
            return MissionRunResult(
                mission_round=mission_round,
                pass_1_queue=list(pass_1_queue),
                pass_1_channel_choices=pass_1_channel_choices,
                empty=True,
                unmerged_report=self._excluded_only_report(),
                pass_1_device_deadlines=planned_deadlines,
                down_timeout=self.dock_on_empty and not answered,
                docked_empty=self.dock_on_empty,
            )

        # Age-of-Update anchor for the devices this merge used (FeRRy Phase 2):
        # the CLEAN sessions minus any update the age cutoff excluded.
        self.scheduler.record_merged(
            _merged_device_ids(report, agg), mission_round,
        )

        # ===================== Inter-pass dock =====================
        # H3 — ride the *previous* mission's Pass-2 delivery report up
        # in this mission's Pass-1 UP bundle. The cluster will bump
        # DeviceRecord.delivery_priority on undelivered rows.
        self.client_cluster.collect(
            partial_aggregate=agg,
            report=report,
            contacts=contacts,
            delivery_report=self._pending_delivery_report,
        )
        # Clear the pending stash; whether or not the UP succeeds, the
        # report is now ClientCluster's responsibility to ship/retry.
        self._pending_delivery_report = None
        if not self.client_cluster.wait_for_dock(timeout=None):
            raise MuleSupervisorError("dock did not become available between passes")
        if not self._dock_and_await_down():
            # FeRRy Phase 2 (down_wait_s set): uploaded, but no DOWN within the
            # wait. Without θ' there is nothing to deliver, so Pass 2 is
            # skipped and this mission's θ carries the next one, exactly as an
            # empty mission does. The upload itself stands.
            log.warning(
                "mule=%s round=%d no DOWN within %.1fs of the upload; skipping "
                "Pass 2 and flying the next mission on this mission's θ",
                self.mule_id, mission_round, self.down_wait_s,
            )
            self._restage(theta_pass_1, synth_pass_1, version_pass_1)
            return MissionRunResult(
                mission_round=mission_round,
                aggregate=agg,
                report=report,
                contacts=contacts,
                channel_choices=list(pass_1_channel_choices),
                pass_1_queue=list(pass_1_queue),
                pass_1_channel_choices=pass_1_channel_choices,
                pass_1_device_deadlines=planned_deadlines,
                down_timeout=True,
            )
        # The DOWN-bundle distribution staged self._next_theta /
        # self._next_synth — those are Pass-2's payload.
        if self._next_theta is None:
            raise MuleSupervisorError(
                "inter-pass dock did not stage a Pass-2 model — cluster "
                "must dispatch a fresh θ' after ingesting Pass-1's UP"
            )
        theta_pass_2 = self._next_theta
        synth_pass_2 = self._next_synth
        version_pass_2 = self._next_theta_version
        # We deliberately DO NOT clear _next_theta here; Pass 2 itself
        # re-uses the same θ as the basis for next mission's training,
        # so the next run_one_mission call will see it staged. This
        # matches the design's "Pass 2's θ becomes mission n+1's basis"
        # — no extra dock cycle needed at end of mission.

        # ============================ Pass 2 ============================
        self.mission.open_pass_2(theta_pass_2, theta_version=version_pass_2)
        # H5 — Pass-2 plans from the mule's *current* pose (advanced by
        # Pass-1 contacts above), not from the constructor-time origin.
        pass_2_queue = self.scheduler.build_pass_2_queue(
            rf_range_m=rf_range_m,
            mule_pose=self.mule_pose,
        )
        # FeRRy Phase 1 — a budgeted Pass 2 flies only what fits; the devices
        # it skips keep their older basis, which is what lets ages spread.
        skipped_pass_2: List[ContactWaypoint] = []
        if self.pass_2_budget:
            pass_2_queue, skipped_pass_2 = self._budget_pass_2(pass_2_queue)
            if skipped_pass_2:
                self.mission.record_skipped_delivery(
                    [d for wp in skipped_pass_2 for d in wp.devices]
                )
        log.info(
            "mule=%s round=%d pass=2 contacts=%d devices_total=%d skipped=%d",
            self.mule_id, mission_round, len(pass_2_queue),
            sum(len(c.devices) for c in pass_2_queue),
            sum(len(c.devices) for c in skipped_pass_2),
        )

        # M5 — Pass-2 channel choices accumulate separately.
        pass_2_channel_choices: List[int] = []
        for wp in pass_2_queue:
            ch_idx = self._pick_channel_contact(wp)
            if ch_idx is not None:
                pass_2_channel_choices.append(ch_idx)
            try:
                self.mission.deliver_contact(
                    contact_devices=list(wp.devices),
                    synth_batch=synth_pass_2,
                )
            except MissionSessionError as e:
                log.warning(
                    "mule=%s round=%d Pass 2 deliver_contact failed pos=%s: %s",
                    self.mule_id, mission_round, wp.position, e,
                )
            # H5 — advance mule pose during Pass 2 too.
            self.mule_pose = wp.position

        delivery_report = self.mission.close_pass_2()
        delivered, undelivered = delivery_report.counts()
        log.info(
            "mule=%s round=%d Pass 2 closed delivered=%d undelivered=%d",
            self.mule_id, mission_round, delivered, undelivered,
        )

        # H3 — stash the delivery report locally; the *next* mission's
        # Pass-1 dock will ride it up in the UP bundle.
        self._pending_delivery_report = delivery_report

        # M5 — combined channel_choices retained for backward compat.
        all_channel_choices = pass_1_channel_choices + pass_2_channel_choices

        return MissionRunResult(
            mission_round=mission_round,
            aggregate=agg,
            report=report,
            contacts=contacts,
            channel_choices=all_channel_choices,
            pass_1_queue=list(pass_1_queue),
            pass_2_queue=list(pass_2_queue),
            delivery_report=delivery_report,
            pass_1_channel_choices=pass_1_channel_choices,
            pass_2_channel_choices=pass_2_channel_choices,
            pass_1_device_deadlines=planned_deadlines,
        )

    def _excluded_only_report(self) -> Optional[MissionRoundCloseReport]:
        """The report of a mission whose every collected update was cut off.

        Only the age-aware rules can refuse an update that arrived on time, so
        only they can leave CLEAN sessions behind an empty round. Returning the
        report keeps those sessions in the trace, where the deadline scorer
        would otherwise count them as misses (audit #3). None for agg:plain,
        whose empty rounds stay exactly as recorded, and for a mission that
        collected nothing.
        """
        if self.aggregation.is_plain:
            return None
        unmerged = getattr(self.mission, "last_unmerged", None)
        if unmerged is None:
            return None
        report = unmerged[0]
        if not any(line.outcome.is_on_time() for line in report.lines):
            return None
        return report

    # ------------------------------------------------------------------ #
    # FeRRy Phase 3 — one mission on the mission clock
    # ------------------------------------------------------------------ #

    def _run_ferry_mission(self, fx) -> MissionRunResult:
        """One two-pass mission on the mission clock (design section 2.3).

        1. **Takeoff.** The mule is at the dock (checked); the clock's ledger,
           and with it the mission's simulated energy, restarts; the S3b
           budget is stamped now (``start_mission``).
        2. **Plan** from the dock with S3a's radius R_planar(b); annotate the
           plan; widen every pre-flight drop, energy drops (critic B10) and
           on-board ``delivery`` drops included, at the simulated takeoff
           time, and record each with its reason (``pass_1_preflight_drops``).
        3. **Each stop:** the departure check (``abort``: the next stop with
           its return-and-upload tail, priced on the clock, today's rule;
           ``replan``: the whole remainder, re-planned when it fails), the
           beacon hook, the leg (transit), the contact plan at arrival, the
           contact.
        4. **Return, close and dock:** the return leg (also after an abort, a
           re-plan to nothing or no stops), S3c from the contacts flown,
           ``close_round``, the upload charge, the dock (a wall wait), the
           turnaround, and the Lamport sync to the DOWN's ``cluster_sim_ts``.
        5. **Pass 2** from ``t2 = clock()``, its budget origin: plan from the
           dock, the budgeted fold when ``pass_2_budget`` is on (skips become
           SKIPPED lines), fly it with ``deliver_contact`` (checked in flight
           only under ``replan``), return, close.

        The mission's result carries the sim fields of design section 2.5.
        Nothing sleeps for simulated time.

        FeRRy Phase 4, plan mode (the Phase 4 spec, other choices 1, 2, 6, 8
        and 9). Step 2 is the commit: ``build_ferry_plan`` chooses b̄ and the
        route at the dock, and the runtime flies b̄ from then on
        (``set_band``), so the annotations, Pass 2's radius and the beacon
        hook's range follow it. The demand the plan left out by choice
        (``plan``) is widened and recorded like every pre-flight drop, and
        stays out of S3c's planned count. The plan is closed after
        ``record_merged`` with the Pass-1 stops actually flown and exactly the
        devices merged, on the empty path with none merged. The D arms' drop
        report (decision 6) is read from the scheduler in either mode.
        """
        from hermes.mule.ferry import RESPONSE_REPLAN
        from hermes.scheduler.stages.s3b_feasibility import (
            REASON_BUDGET,
            REASON_DELIVERY,
            REASON_ENERGY,
            REASON_OVERDUE,
            RULE_BUDGET,
            FlightState,
        )
        from hermes.types import weights_byte_count

        clock = self._now
        dock = fx.spec.flight.dock
        collect_pass, deliver_pass = MissionPass.COLLECT, MissionPass.DELIVER

        # ---------------------------------------------------------- takeoff
        if tuple(float(c) for c in self.mule_pose) != dock:
            raise MuleSupervisorError(
                f"takeoff away from the dock: pose {tuple(self.mule_pose)!r}, dock {dock!r}"
            )
        clock.reset_ledger()
        sim_start = clock()
        self.scheduler.start_mission()
        rec = _FerryLog()

        theta_pass_1 = self._next_theta
        synth_pass_1 = self._next_synth
        version_pass_1 = self._next_theta_version
        self._next_theta = None
        self._next_synth = None
        self._next_theta_version = None
        fx.observe_payload(theta_pass_1, synth_pass_1)

        # ============================ Pass 1 ============================
        mission_round = self.mission.open_round(theta_pass_1, theta_version=version_pass_1)
        self.scheduler.set_mission_round(mission_round)
        plan_mode = getattr(self, "_flight_slot", None) is not None
        if plan_mode:
            # FeRRy Phase 4 — the commit (build plan Fig. 1): b̄ and the route
            # as one decision at the dock, then b̄ in both passes.
            pass_1_queue = self.scheduler.build_ferry_plan(mule_pose=dock)
            commit = self.scheduler.last_plan
            fx.set_band(commit.band)
            rec.capped = frozenset(commit.capped)
        else:
            pass_1_queue = self.scheduler.build_contact_queue(
                rf_range_m=fx.range_planar_m,
                mule_pose=dock,
                mule_energy=self.mule_energy,
                rf_prior_snr_db=self.rf_prior_snr_db,
            )
        pass_1_queue = fx.annotate(pass_1_queue, self._ferry_positions(pass_1_queue))
        planned_caps = self._age_caps()
        planned_deadlines = dict(
            getattr(self.scheduler, "last_plan_deadlines", None) or {}
        )
        log.info(
            "mule=%s round=%d pass=1 contacts=%d devices_total=%d sim_t=%.3f band=%s",
            self.mule_id, mission_round, len(pass_1_queue),
            sum(len(c.devices) for c in pass_1_queue), sim_start, fx.band,
        )
        _feas = getattr(self.scheduler, "last_feasibility", None)
        # A contact the on-board clause refused (deadline_bounds="delivery")
        # is widened like a budget drop: it was not late itself, but it got
        # no contact either. The list is empty in every other mode.
        dropped_pre = (
            list(getattr(_feas, "dropped_overdue", ()))
            + list(getattr(_feas, "dropped_budget", ()))
            + list(getattr(_feas, "dropped_energy", ()))
            + list(getattr(_feas, "dropped_delivery", ()))
        )
        drop_reasons = [(REASON_OVERDUE, "dropped_overdue"),
                        (REASON_BUDGET, "dropped_budget"),
                        (REASON_ENERGY, "dropped_energy"),
                        (REASON_DELIVERY, "dropped_delivery")]
        if plan_mode:
            # FeRRy Phase 4 (the Phase 4 spec, other choices 8): the demand the
            # plan left out by choice got no contact either, so it is widened
            # and recorded like any drop, under the plan-level reason ``plan``.
            # mission_planned_devices leaves it out of S3c's count (critic C6).
            from hermes.scheduler.plan.types import REASON_PLAN

            dropped_pre += list(getattr(_feas, "dropped_plan", ()))
            drop_reasons.append((REASON_PLAN, "dropped_plan"))
        if dropped_pre:
            self._widen_abandoned(dropped_pre, mission_round=mission_round)
        # Finding E2E1-01: without this record an empty mission's trace does
        # not say what emptied it (e.g. one field-wide contact over budget).
        rec.preflight_drops = [
            {"position": [float(c) for c in wp.position], "devices": _ids(wp.devices),
             "deadline_ts": float(wp.deadline_ts), "reason": reason}
            for reason, name in drop_reasons
            for wp in getattr(_feas, name, ())
        ]
        # FeRRy Phase 4 (the user's decision 6): what a whole-scheduler
        # baseline left out, reported and never widened (a synthetic miss
        # would move the fields its round inference reads). The scheduler
        # fills it for the D arms on this clock only; empty for every other arm.
        rec.policy_drops = [
            {"position": [float(c) for c in wp.position], "devices": _ids(wp.devices),
             "deadline_ts": float(wp.deadline_ts), "reason": reason, "widened": False}
            for wp, reason in getattr(self.scheduler, "last_policy_drops", None) or ()
        ]
        planned_devices = mission_planned_devices(pass_1_queue, _feas)
        rec.no_insert = {d for wp in list(pass_1_queue) + dropped_pre for d in wp.devices}
        rec.deadlines = dict(planned_deadlines)
        budget = self.scheduler.mission_budget_s
        start = self.scheduler.mission_start_ts
        budget_end_1 = None if budget is None else float(start) + float(budget)

        flown_1 = self._ferry_fly_pass(
            fx, pass_1_queue, pass_kind=collect_pass, mission_round=mission_round,
            synth=synth_pass_1, budget_end=budget_end_1, check=True,
            energy_origin_j=0.0, rec=rec,
        )
        self._ferry_home(fx)

        # S3c — served counts the devices of the contacts flown (design
        # section 3.4); planned is the committed plan, its pre-flight drops
        # and the beacon hook's inserts.
        planned_devices += rec.inserted_devices
        served_devices = sum(len(wp.devices) for wp in flown_1)
        self.scheduler.record_mission_outcome(
            served=served_devices, planned=planned_devices,
        )
        if getattr(self.scheduler, "_window_adapter", None) is not None:
            log.info(
                "mule=%s round=%d S3c mission outcome served=%d/%d -> %s",
                self.mule_id, mission_round, served_devices, planned_devices,
                self.scheduler._window_adapter.describe(),
            )

        common = dict(
            fx=fx, rec=rec, mission_round=mission_round, sim_start=sim_start,
            pass_1_queue=pass_1_queue, planned_deadlines=planned_deadlines,
            budget_end_1=budget_end_1,
        )
        try:
            agg, report, contacts = self.mission.close_round(age_caps=planned_caps)
        except MissionSessionError as e:
            # EX-4.2 on the mission clock: a recoverable empty round. The
            # clock still charges the return (above), the turnaround and, if
            # the mule docks, the upload of its empty partial.
            log.warning(
                "mule=%s round=%d Pass 1 collected no updates; recording an "
                "empty round and continuing: %s",
                self.mule_id, mission_round, e,
            )
            if plan_mode:
                # FeRRy Phase 4 (the Phase 4 spec, other choices 6): nothing
                # merged, so every capped device the plan served and a stop
                # flown visited is ``not_merged``.
                self.scheduler.close_plan(flown_1, ())
            answered = False
            up = None
            before = self.client_cluster.last_down()
            if self.dock_on_empty:
                up = fx.charge_upload(0)
                self._ferry_observed_upload(fx, up)
                answered = self._dock_empty(
                    mission_round, version_pass_1, sim_upload_ts=clock(), backhaul=up,
                )
            t_pass_1_end = clock()
            self._ferry_turnaround(fx)
            if answered:
                self._ferry_sync_after(before)
            else:
                self._restage(theta_pass_1, synth_pass_1, version_pass_1)
            return self._ferry_result(
                **common, t_pass_1_end=t_pass_1_end, up=up,
                pass_1_channel_choices=list(rec.choices[collect_pass]),
                empty=True,
                unmerged_report=self._excluded_only_report(),
                down_timeout=self.dock_on_empty and not answered,
                docked_empty=self.dock_on_empty,
            )

        merged = _merged_device_ids(report, agg)
        self.scheduler.record_merged(merged, mission_round)
        if plan_mode:
            # FeRRy Phase 4 (the Phase 4 spec, other choices 6): the visited set
            # is the members of the Pass-1 stops actually flown (after any trim,
            # with the beacon hook's inserts), and the close-time violations
            # read exactly the devices record_merged was given.
            self.scheduler.close_plan(flown_1, merged)

        # ===================== Inter-pass dock =====================
        up = fx.charge_upload(weights_byte_count(agg.weights))
        self._ferry_observed_upload(fx, up)
        t_pass_1_end = clock()
        self.client_cluster.collect(
            partial_aggregate=agg,
            report=report,
            contacts=contacts,
            delivery_report=self._pending_delivery_report,
            sim_upload_ts=t_pass_1_end,
            backhaul=up,
        )
        self._pending_delivery_report = None
        if not self.client_cluster.wait_for_dock(timeout=None):
            raise MuleSupervisorError("dock did not become available between passes")
        before = self.client_cluster.last_down()
        docked = self._dock_and_await_down()
        self._ferry_turnaround(fx)
        if not docked:
            log.warning(
                "mule=%s round=%d no DOWN within %.1fs of the upload; skipping "
                "Pass 2 and flying the next mission on this mission's θ",
                self.mule_id, mission_round, self.down_wait_s,
            )
            self._restage(theta_pass_1, synth_pass_1, version_pass_1)
            return self._ferry_result(
                **common, t_pass_1_end=t_pass_1_end, up=up,
                aggregate=agg, report=report, contacts=contacts,
                channel_choices=list(rec.choices[collect_pass]),
                pass_1_channel_choices=list(rec.choices[collect_pass]),
                down_timeout=True,
            )
        self._ferry_sync_after(before)
        if self._next_theta is None:
            raise MuleSupervisorError(
                "inter-pass dock did not stage a Pass-2 model — cluster "
                "must dispatch a fresh θ' after ingesting Pass-1's UP"
            )
        theta_pass_2 = self._next_theta
        synth_pass_2 = self._next_synth
        version_pass_2 = self._next_theta_version
        # As in the recorded path, Pass 2's θ stays staged: it is the next
        # mission's basis.

        # ============================ Pass 2 ============================
        t2 = clock()
        energy_at_t2 = fx.energy_j()
        self.mission.open_pass_2(theta_pass_2, theta_version=version_pass_2)
        fx.observe_payload(theta_pass_2, synth_pass_2)
        pass_2_queue = self.scheduler.build_pass_2_queue(
            rf_range_m=fx.range_planar_m, mule_pose=dock,
        )
        pass_2_queue = fx.annotate(pass_2_queue, self._ferry_positions(pass_2_queue))
        budget_end_2 = (
            t2 + float(budget) if (self.pass_2_budget and budget is not None) else None
        )
        skipped_pass_2: List[ContactWaypoint] = []
        if budget_end_2 is not None and pass_2_queue:
            # Design section 3.5: the Pass-2 walk from (dock, t2), DELIVER
            # bytes and no upload tail; what it skips keeps its older basis.
            walk = self.scheduler.feasibility_model.fold(
                pass_2_queue, FlightState(dock, t2), rule=RULE_BUDGET,
                budget_end=budget_end_2, pass_kind=deliver_pass, skip=True,
            )
            pass_2_queue = list(walk.route)
            skipped_pass_2 = [wp for wp, _ in walk.rejected]
            if skipped_pass_2:
                self.mission.record_skipped_delivery(
                    [d for wp in skipped_pass_2 for d in wp.devices]
                )
        log.info(
            "mule=%s round=%d pass=2 contacts=%d devices_total=%d skipped=%d sim_t=%.3f",
            self.mule_id, mission_round, len(pass_2_queue),
            sum(len(c.devices) for c in pass_2_queue),
            sum(len(c.devices) for c in skipped_pass_2), t2,
        )
        self._ferry_fly_pass(
            fx, pass_2_queue, pass_kind=deliver_pass, mission_round=mission_round,
            synth=synth_pass_2, budget_end=budget_end_2,
            check=(fx.spec.in_flight_response == RESPONSE_REPLAN
                   and budget_end_2 is not None),
            energy_origin_j=energy_at_t2, rec=rec,
        )
        self._ferry_home(fx)
        delivery_report = self.mission.close_pass_2()
        delivered, undelivered = delivery_report.counts()
        log.info(
            "mule=%s round=%d Pass 2 closed delivered=%d undelivered=%d sim_t=%.3f",
            self.mule_id, mission_round, delivered, undelivered, clock(),
        )
        # H3 — the next mission's Pass-1 dock rides it up.
        self._pending_delivery_report = delivery_report
        return self._ferry_result(
            **common, t_pass_1_end=t_pass_1_end, up=up,
            aggregate=agg, report=report, contacts=contacts,
            channel_choices=list(rec.choices[collect_pass]) + list(rec.choices[deliver_pass]),
            pass_2_queue=list(pass_2_queue),
            delivery_report=delivery_report,
            pass_1_channel_choices=list(rec.choices[collect_pass]),
            pass_2_channel_choices=list(rec.choices[deliver_pass]),
            t2=t2, budget_end_2=budget_end_2,
        )

    def _ferry_fly_pass(
        self,
        fx,
        queue: List[ContactWaypoint],
        *,
        pass_kind: MissionPass,
        mission_round: int,
        synth,
        budget_end: Optional[float],
        check: bool,
        energy_origin_j: float,
        rec: _FerryLog,
    ) -> List[ContactWaypoint]:
        """Fly one pass stop by stop; return the stops flown, in order.

        At every departure, takeoff included, the mule holds its flight state
        ``(pose, clock(), energy spent this pass, deliver_by)``; with
        ``check`` it runs the departure check (:meth:`_ferry_departure`), and
        in Pass 1 the beacon hook may insert an offered contact. The pass
        ends when nothing is left or an abort gave up the rest.

        ``deliver_by`` is ``inf`` (nothing on board) except in Pass 1 under
        ``deadline_bounds="delivery"``, where it is the running minimum of
        the own Deadline(j) (``rec.deadlines``) of every member whose update
        was actually collected, CLEAN, at a stop flown so far: the check, the
        re-plan and the beacon hook then hold the rest of the route to the
        updates really on board. S3b's walk before takeoff assumes every
        planned member answers and folds the stop's ``deadline_ts``, the
        minimum of the same members' deadlines, so the two agree when every
        member answers, and the in-flight bound is never the tighter: a
        member that did not answer carries no update to be late. Pass 2
        carries nothing to the cluster and has no deadline clause.

        The bound is checked at departures only. When the contact at the stop
        where the pass ends (the last one, or the one after which an abort or
        a re-plan gave up the rest) runs longer than priced, a silent
        member's listen window or a noisy band, nothing is left to decide:
        the updates are on board and the mule flies home, so the landing can
        pass ``deliver_by``. The pass leaves its final ``deliver_by`` in
        ``rec.deliver_by`` for ``delivery_overrun_s``, which records that.

        FeRRy Phase 4, plan mode. The next stop is the flight slot's pick
        (:meth:`_ferry_next_stop`), made after the departure check and the
        beacon hook: the committed slot (F) always picks the plan's next
        stop, as legacy mode does; FX's cross-heuristic may pick another after
        a Pass-1 stop. A capped member's update does not lower ``deliver_by``
        (the Phase 4 spec, other choices 9), as the predicate does not lower
        it for a stop exempt from its own deadline: a stop whose members the
        cap binds is flown for them whatever their deadlines, so holding the
        rest of the route to those deadlines would refuse every later stop
        for updates that may be late anyway (``s3b_feasibility``, ``admit``).
        """
        from hermes.scheduler.stages.s3b_feasibility import (
            DEADLINE_BOUNDS_DELIVERY,
            FlightState,
        )
        from hermes.types import MissionOutcome

        clock = self._now
        remainder = list(queue)
        flown: List[ContactWaypoint] = []
        collect = pass_kind is MissionPass.COLLECT
        route_level = collect and fx.spec.deadline_bounds == DEADLINE_BOUNDS_DELIVERY
        capped = getattr(rec, "capped", frozenset())
        deliver_by = math.inf
        while True:
            state = FlightState(
                tuple(self.mule_pose), clock(), fx.energy_j() - energy_origin_j, deliver_by,
            )
            if remainder and check:
                kept = self._ferry_departure(
                    fx, remainder, state, pass_kind=pass_kind,
                    mission_round=mission_round, budget_end=budget_end, rec=rec,
                )
                if kept is None:
                    break
                remainder = kept
            if collect:
                remainder = self._take_offers(fx, remainder, state, budget_end=budget_end, rec=rec)
            if not remainder:
                break
            slot = getattr(self, "_flight_slot", None)
            index = 0 if slot is None else self._ferry_next_stop(
                slot, remainder, state, pass_kind=pass_kind, budget_end=budget_end,
                after_stop=bool(flown),
            )
            wp = remainder.pop(index)
            outcomes = self._ferry_stop(
                fx, wp, state, pass_kind=pass_kind, mission_round=mission_round,
                synth=synth, energy_origin_j=energy_origin_j, rec=rec,
            )
            if route_level:
                for did, outcome in outcomes.items():
                    if outcome is MissionOutcome.CLEAN and did not in capped:
                        deliver_by = min(deliver_by, rec.deadlines.get(did, wp.deadline_ts))
            flown.append(wp)
        if route_level:
            rec.deliver_by = deliver_by
        return flown

    def _ferry_protected(
        self, stops: List[ContactWaypoint], pass_kind: MissionPass,
    ) -> Dict[str, Any]:
        """Plan mode's exempt stops among ``stops``, as the fold's keyword; else none.

        The Phase 4 spec, other choices 9: in plan mode the departure check,
        the re-plan, the beacon hook's fold and FX's ``fits`` hold a stop
        whose every member the plan capped exempt from its own deadline
        clause, as the plan and its guard priced it
        (``FLScheduler.plan_protected``). The set is recomputed on ``stops``
        as they stand, since a trim makes new waypoints (critic B1). Pass 1
        only: the cap is about collecting updates, and Pass 2 has no deadline
        clause. Legacy mode passes nothing, so every recorded call keeps its
        arguments (Phase 3 gave none of them a protected set).
        """
        if getattr(self, "_flight_slot", None) is None or pass_kind is not MissionPass.COLLECT:
            return {}
        return {"protected": self.scheduler.plan_protected(stops)}

    def _ferry_next_stop(
        self,
        slot,
        remainder: List[ContactWaypoint],
        state,
        *,
        pass_kind: MissionPass,
        budget_end: Optional[float],
        after_stop: bool,
    ) -> int:
        """The index of the stop to fly next: the flight slot's pick (plan mode).

        The user's decision 5 and U6's order: the slot chooses after the
        departure check and the beacon hook have settled the remainder, so it
        reads the state the service left. The committed slot (F, FB+c)
        returns 0, the plan's next stop, without calling ``fits``. FX's
        cross-heuristic, in Pass 1 after a stop (``after_stop``: a stop of
        this pass has been flown; not at takeoff, not in Pass 2), returns the
        nearest remaining stop whose move to the front still passes the
        departure check's own fold (``fits``: the whole reordered remainder
        from ``state`` under the arm's in-flight rule, the plan's exempt stops
        protected), else 0. The slot applies those rules itself.
        """
        def fits(order: List[ContactWaypoint]) -> bool:
            return self.scheduler.fold_remainder(
                order, state=state, budget_end=budget_end, pass_kind=pass_kind,
                **self._ferry_protected(list(order), pass_kind),
            ).ok

        return slot.next_stop(remainder, state, fits=fits, pass_kind=pass_kind,
                              after_stop=after_stop)

    def _ferry_departure(
        self,
        fx,
        remainder: List[ContactWaypoint],
        state,
        *,
        pass_kind: MissionPass,
        mission_round: int,
        budget_end: Optional[float],
        rec: _FerryLog,
    ) -> Optional[List[ContactWaypoint]]:
        """The in-flight check at a departure (design section 3.4); None = abort.

        ``abort``: today's rule (Freeze Amendment 8) on the mission clock: the
        next stop is admitted from the current state under the arm's in-flight
        rule, its return leg and (Pass 1) upload included; if it is not, the
        rest of the pass is abandoned and, in Pass 1, widened at the abort
        time. ``replan``: the whole remainder is folded as it would be flown;
        if any stop fails, ``FLScheduler.replan_remainder`` repairs it (any
        ``order_used``, ``arm_trimmed`` included), and every stop it drops is
        final for the mission: widened in Pass 1, a SKIPPED delivery in Pass 2,
        stamped at the drop time. δ_obs is 0 (critic C1). Both price with the
        scheduler's model, so the mule never re-implements S3b.

        FeRRy Phase 4, plan mode (the Phase 4 spec, other choices 9): in Pass 1
        both folds and the re-plan hold the plan's exempt stops protected
        (:meth:`_ferry_protected`), and the re-plan is the scheduler's trim of
        the committed plan (never ``routing.replan.replan_route``, whose
        identity check forbids the reduced stops a member trim makes), given
        each member's own deadline from the mule's record so that a stop the
        beacon hook inserted is dated as it was priced (``rec.deadlines``).
        ``abort`` flies only without a cap there (critic A10).
        """
        from hermes.mule.ferry import RESPONSE_ABORT

        collect = pass_kind is MissionPass.COLLECT
        if fx.spec.in_flight_response == RESPONSE_ABORT:
            head = self.scheduler.fold_remainder(
                remainder[:1], state=state, budget_end=budget_end, pass_kind=pass_kind,
                **self._ferry_protected(remainder[:1], pass_kind),
            )
            if head.ok:
                return remainder
            abandoned = list(remainder)
            log.info(
                "mule=%s round=%d Pass %d ABORTING at sim t=%.3f: the next contact "
                "is no longer reachable in time (%s); abandoning %d contact(s)",
                self.mule_id, mission_round, 1 if collect else 2, state.clock,
                head.rejected[0][1], len(abandoned),
            )
            if collect:
                self._widen_abandoned(abandoned, mission_round=mission_round)
            else:
                self.mission.record_skipped_delivery(
                    [d for wp in abandoned for d in wp.devices]
                )
            rec.aborts.append({
                "t_s": state.clock,
                "pass": pass_kind.value,
                "reason": head.rejected[0][1],
                "abandoned": [_ids(wp.devices) for wp in abandoned],
            })
            return None

        plan_kw = self._ferry_protected(remainder, pass_kind)
        fold = self.scheduler.fold_remainder(
            remainder, state=state, budget_end=budget_end, pass_kind=pass_kind, **plan_kw,
        )
        if fold.ok:
            return remainder
        if plan_kw:
            plan_kw["deadlines"] = rec.deadlines
        res = self.scheduler.replan_remainder(
            remainder, state=state, budget_end=budget_end, pass_kind=pass_kind, **plan_kw,
        )
        dropped = res.dropped_contacts
        rec.replans.append({
            "t_s": state.clock,
            "pass": pass_kind.value,
            "order_used": res.order_used,
            "rejected": [{"devices": _ids(wp.devices), "reason": why}
                         for wp, why in fold.rejected],
            "before": [_ids(wp.devices) for wp in remainder],
            "route": [_ids(wp.devices) for wp in res.route],
            "dropped": [{"devices": _ids(wp.devices), "reason": why}
                        for wp, why in res.dropped],
            "delta_obs_db": 0.0,
        })
        log.info(
            "mule=%s round=%d Pass %d re-plan at sim t=%.3f: %s order, %d/%d "
            "contacts kept (%d dropped)",
            self.mule_id, mission_round, 1 if collect else 2, state.clock,
            res.order_used, len(res.route), len(remainder), len(dropped),
        )
        if dropped:
            if collect:
                self._widen_abandoned(dropped, mission_round=mission_round)
            else:
                self.mission.record_skipped_delivery(
                    [d for wp in dropped for d in wp.devices]
                )
        return list(res.route)

    def _ferry_stop(
        self,
        fx,
        wp: ContactWaypoint,
        state,
        *,
        pass_kind: MissionPass,
        mission_round: int,
        synth,
        energy_origin_j: float,
        rec: _FerryLog,
    ) -> Dict[DeviceID, Any]:
        """Fly to ``wp`` and serve it: the leg, the contact plan, the contact.

        The leg is charged as ``transit`` and the pose jumps to the stop; the
        contact plan is built on arrival, and the host's commit charges the
        contact's airtime and listen window. With a channel actor wired, the
        L1 choice is recorded, not acted on: from the state observed at
        arrival with a band (design section 4.6), from the recorded state
        before the leg without one. The observed state's energy is this
        sortie's, counted from ``energy_origin_j`` (the pass's takeoff) as
        the departure state and the energy clause count it: Pass 2 restarts
        at 0, so the L1 state and the energy clause agree on the battery.

        Returns the contact's per-device outcome map (empty when the contact
        failed), from which the caller learns whose updates are on board.

        FeRRy Phase 4, plan mode: the contact plan is built on the class the
        flight slot names at the arrival (:meth:`_ferry_band_at_arrival`), and
        the record's ``band`` and rates are that plan's class. In legacy mode,
        and under the committed slot, the plan's class is the runtime's band,
        so the record holds what it always did.
        """
        clock = self._now
        collect = pass_kind is MissionPass.COLLECT
        positions = self._ferry_positions([wp])
        l1_choice: Optional[int] = None
        if self.channel_actor is not None and not fx.banded:
            l1_choice = self._pick_channel_contact(wp)
        transit = fx.spec.flight.leg_s(self.mule_pose, wp.position)
        clock.advance(transit, "transit")
        self.mule_pose = wp.position
        if self.channel_actor is not None and fx.banded:
            obs = fx.observe(wp, positions, clock())
            l1_choice = int(self.channel_actor.argmax(fx.l1_state(
                obs, pose=self.mule_pose, energy_j=fx.energy_j() - energy_origin_j,
                budget_s=self.scheduler.mission_budget_s,
            )))
        slot = getattr(self, "_flight_slot", None)
        if slot is None:
            plan = fx.contact_plan(wp, positions, pass_kind=pass_kind, mission_round=mission_round)
        else:
            plan = fx.contact_plan(
                wp, positions, pass_kind=pass_kind, mission_round=mission_round,
                band=self._ferry_band_at_arrival(slot, fx, wp, positions, pass_kind),
            )
        before = self.mission.last_contact
        outcomes: Dict[DeviceID, Any] = {}
        try:
            if collect:
                served = self.mission.run_contact(list(wp.devices), synth, plan=plan)
            else:
                served = self.mission.deliver_contact(list(wp.devices), synth, plan=plan)
            if isinstance(served, dict):
                outcomes = served
        except MissionSessionError as e:
            log.warning(
                "mule=%s round=%d Pass %d %s failed pos=%s: %s",
                self.mule_id, mission_round, 1 if collect else 2,
                "run_contact" if collect else "deliver_contact", wp.position, e,
            )
        commit = self.mission.last_contact
        if commit is before:
            commit = None
        snr = rate = None
        if fx.banded:
            arrival_snr = [float(plan.snr_db[d]) for d in wp.devices]
            snr = {str(d): s for d, s in zip(wp.devices, arrival_snr)}
            rate = {str(d): r for d, r in zip(wp.devices,
                                              fx.rates_bps(arrival_snr, band=plan.band))}
        rec.flown[pass_kind].append({
            "position": [float(c) for c in wp.position],
            "devices": _ids(wp.devices),
            "deadline_ts": float(wp.deadline_ts),
            "depart_s": state.clock,
            "depart_pose": [float(c) for c in state.pose],
            "depart_energy_j": state.energy_j,
            "transit_s": transit,
            "arrival_s": plan.arrival_ts,
            "end_s": commit.end_ts if commit is not None else clock(),
            "band": plan.band,
            "targets": _ids(plan.targets),
            "unreachable": _ids(plan.unreachable),
            "snr_db": snr,
            "rate_bps": rate,
            "dwell_s": commit.dwell_s if commit is not None else 0.0,
            "listen_s": commit.listen_s if commit is not None else 0.0,
            "missing": _ids(commit.missing) if commit is not None else [],
            "uplink_dropped": _ids(commit.uplink_dropped) if commit is not None else [],
            "l1_choice": l1_choice,
        })
        if l1_choice is not None:
            rec.choices[pass_kind].append(l1_choice)
        return outcomes

    def _ferry_band_at_arrival(self, slot, fx, wp: ContactWaypoint, positions,
                               pass_kind: MissionPass) -> Optional[str]:
        """The class to build this stop's contact plan on (plan mode); None: b̄.

        The user's decision 5 with critic A7's rule: FX's slot reads what each
        class would reach here now (``FerryRuntime.arrival_view``, at the
        arrival: the transit is charged and nothing else is until the contact
        plan, so the view and the plan see one instant) and names the fastest
        class that still reaches every device the committed class reaches.
        At the arrival SNR it never dwells longer than b̄ would, so the
        remainder the departure check priced on b̄ stays priced
        conservatively, at the last stop too, after which nothing re-checks.
        The committed slot reads no view and keeps b̄; in Pass 2 every slot
        keeps b̄, and the runtime refuses a Pass-2 plan on another class.
        """
        view = (fx.arrival_view(wp, positions, self._now(), pass_kind=pass_kind)
                if slot.reads_arrival_view(pass_kind) else None)
        return slot.band_at_arrival(view, pass_kind=pass_kind)

    def _ferry_home(self, fx) -> None:
        """The return leg to the dock, charged as ``return``; the pose is the dock."""
        dock = fx.spec.flight.dock
        self._now.advance(fx.spec.flight.leg_s(self.mule_pose, dock), "return")
        self.mule_pose = dock

    def _ferry_turnaround(self, fx) -> None:
        """The dock turnaround, once per mission, upload or not (design section 2.3)."""
        self._now.advance(fx.spec.flight.turnaround_s, "turnaround")

    def _ferry_observed_upload(self, fx, up) -> None:
        """Feed the causal RF prior (critic B4) from the upload just priced.

        With a seconds-axis backhaul the planner's ``rf_prior_snr_db`` becomes
        the SNR the mule last observed on the carrier it holds; before the
        first upload it keeps its configured value (20 dB by default).
        """
        if up is not None and fx.rf_prior is not None:
            self.rf_prior_snr_db = fx.rf_prior.prior_snr_db(carrier=up.carrier)

    def _ferry_sync(self, down) -> None:
        """The Lamport sync at the dock (design section 2.3, critic B8).

        The mule adopts the cluster's simulated time when the cluster is ahead
        (``advance_to`` is a max, charged as ``dock_wait``); a DOWN without
        ``cluster_sim_ts`` makes no sync.
        """
        ts = getattr(down, "cluster_sim_ts", None)
        if ts is None:
            return
        try:
            self._now.advance_to(float(ts), "dock_wait")
        except ValueError as e:
            raise MuleSupervisorError(f"cannot sync to the DOWN's cluster_sim_ts: {e}") from e

    def _ferry_sync_after(self, before) -> None:
        """Take the DOWN the dock just delivered, if one came: check its slice
        was ingested (:meth:`_ferry_check_slice`), then sync to it."""
        down = self.client_cluster.last_down()
        if down is not None and down is not before:
            self._ferry_check_slice()
            self._ferry_sync(down)

    def _ferry_check_slice(self) -> None:
        """Raise if the scheduler refused the slice of the DOWN just distributed.

        On the mission clock the scheduler refuses an amendment carrying the
        cluster's ``deadline_overrides``, wall-clock stamps (critic A1/B3), and
        with it the slice. ``ClientCluster`` only logs a failing sink and
        stages θ anyway, so without this check the mule would report the dock
        as a success and fly on a stale slice (or, at the bootstrap, none).
        """
        err = getattr(self, "_ferry_slice_error", None)
        if err is None:
            return
        self._ferry_slice_error = None
        raise MuleSupervisorError(
            f"the DOWN's slice was not ingested on the mission clock: {err}"
        ) from err

    def _ferry_positions(self, queue) -> Dict[DeviceID, Tuple[float, ...]]:
        """Each member's last-known position: the ones S3a clustered with."""
        states = self.scheduler.device_states
        return {
            d: tuple(float(c) for c in states[d].last_known_position)
            for wp in queue for d in wp.devices
        }

    def _ferry_result(
        self,
        *,
        fx,
        rec: _FerryLog,
        mission_round: int,
        sim_start: float,
        pass_1_queue: List[ContactWaypoint],
        planned_deadlines: Dict[DeviceID, float],
        budget_end_1: Optional[float],
        t_pass_1_end: float,
        up,
        t2: Optional[float] = None,
        budget_end_2: Optional[float] = None,
        **legacy,
    ) -> MissionRunResult:
        """The mission's result: the recorded fields as the legacy path fills
        them, plus the sim fields (design section 2.5).

        FeRRy Phase 4: in plan mode ``band`` is b̄, the class the runtime flew,
        and ``plan`` the closed commit's description with the plan's wall time
        beside it (critic B12); ``pass_1_policy_drops`` only when a baseline
        left something out (critic D2). All three stay None otherwise.
        """
        from hermes.mule.ferry import backhaul_record
        from hermes.scheduler.stages.s3b_feasibility import DEADLINE_BOUNDS_DELIVERY

        clock = self._now
        sim_end = clock()
        ledger = clock.ledger()
        plan = plan_wall_s = None
        if getattr(self, "_flight_slot", None) is not None:
            plan = self.scheduler.last_plan.describe()
            plan_wall_s = self.scheduler.last_plan_wall_s
        return MissionRunResult(
            mission_round=mission_round,
            pass_1_queue=list(pass_1_queue),
            pass_1_device_deadlines=planned_deadlines,
            sim_start_s=sim_start,
            sim_end_s=sim_end,
            sim_ledger=ledger,
            sim_pass_2_start_s=t2,
            pass_1_flown=list(rec.flown[MissionPass.COLLECT]),
            pass_2_flown=list(rec.flown[MissionPass.DELIVER]),
            replans=list(rec.replans),
            aborts=list(rec.aborts),
            inserts=list(rec.inserts),
            offers_refused=list(rec.offers_refused),
            budget_overrun_s=(None if budget_end_1 is None
                              else max(0.0, t_pass_1_end - budget_end_1)),
            pass_2_budget_overrun_s=(None if budget_end_2 is None
                                     else max(0.0, sim_end - budget_end_2)),
            energy_j=fx.spec.flight.energy.energy_j(ledger),
            band=fx.band,
            backhaul=backhaul_record(up),
            pass_1_preflight_drops=list(rec.preflight_drops),
            delivery_overrun_s=(max(0.0, t_pass_1_end - rec.deliver_by)
                                if fx.spec.deadline_bounds == DEADLINE_BOUNDS_DELIVERY
                                else None),
            plan=plan,
            plan_wall_s=plan_wall_s,
            pass_1_policy_drops=list(getattr(rec, "policy_drops", ())) or None,
            **legacy,
        )

    # ------------------------------------------------------------------ #
    # FeRRy Phase 3 — the beacon hook (design section 3.6)
    # ------------------------------------------------------------------ #

    def offer_contact(self, wp: ContactWaypoint) -> None:
        """Offer an opportunistic contact; it is considered at the next Pass-1 departure.

        The offer names the devices (``wp.devices``, with ``wp.bucket``);
        the mule rebuilds the stop itself: at its first member's last-known
        position, with the members' tightest deadline. At the next departure
        it is inserted at its cheapest place in the remainder, and accepted
        only if the fold of the edited remainder passes without skipping
        under the arm's in-flight rule; it never evicts a planned stop. Only
        devices the scheduler already tracks, not already planned (or
        dropped) this mission, whose members lie within R_planar(b) of the
        first, can be inserted; the hook never touches ``is_in_slice``.
        Accepted inserts are recorded (``MissionRunResult.inserts``) and count
        in S3c's planned and served; refused ones are recorded with a reason
        (an offer that names a device twice among them). Offers queued during
        Pass 2 wait for the next mission's first departure. Thread-safe; inert
        with no source. Needs the mission clock.
        """
        if getattr(self, "_ferry_run", None) is None:
            raise MuleSupervisorError(
                "the beacon hook runs on the mission clock (mission_clock=...)"
            )
        if not isinstance(wp, ContactWaypoint):
            raise TypeError(f"offer_contact takes a ContactWaypoint, got {type(wp).__name__}")
        with self._offers_lock:
            self._offers.append(wp)

    def _take_offers(
        self,
        fx,
        remainder: List[ContactWaypoint],
        state,
        *,
        budget_end: Optional[float],
        rec: _FerryLog,
    ) -> List[ContactWaypoint]:
        """Try every queued offer against the remainder; return the new remainder."""
        with self._offers_lock:
            offers, self._offers = list(self._offers), []
        for offer in offers:
            remainder = self._ferry_try_insert(
                fx, offer, remainder, state, budget_end=budget_end, rec=rec,
            )
        return remainder

    def _ferry_try_insert(
        self,
        fx,
        offer: ContactWaypoint,
        remainder: List[ContactWaypoint],
        state,
        *,
        budget_end: Optional[float],
        rec: _FerryLog,
    ) -> List[ContactWaypoint]:
        """One offer: the edited remainder if it fits, else the remainder unchanged."""
        from hermes.mission.contact_plan import planar_distance_m
        from hermes.scheduler.stages.s3_deadline import compute_deadline

        states = self.scheduler.device_states
        devices = tuple(offer.devices)
        t = state.clock

        def refuse(reason: str) -> List[ContactWaypoint]:
            rec.offers_refused.append({"t_s": t, "devices": _ids(devices), "reason": reason})
            log.info("mule=%s beacon offer %s refused: %s", self.mule_id, _ids(devices), reason)
            return remainder

        if len(set(devices)) != len(devices):
            # A stop solicits each member once: its contact plan refuses
            # repeats, and by then the leg to it has been charged, so the
            # mission would end away from the dock.
            return refuse("members repeat")
        if any(d not in states for d in devices):
            return refuse("unknown device")
        if any(d in rec.no_insert for d in devices):
            return refuse("already planned this mission")
        positions = {d: tuple(float(c) for c in states[d].last_known_position) for d in devices}
        if any(not states[d].is_in_slice and positions[d] == (0.0, 0.0, 0.0) for d in devices):
            # A device outside the slice whose position the cluster never
            # sent still holds the state's (0, 0, 0) default: pricing it
            # there would plan a stop at the dock (design section 3.6).
            return refuse("position unknown")
        anchor = positions[devices[0]]
        if any(planar_distance_m(anchor, positions[d]) > fx.range_planar_m for d in devices):
            return refuse("members not within range of one stop")
        member_deadlines = {
            d: compute_deadline(states[d], t, self.scheduler.window_scale,
                                law=self.scheduler.deadline_law)
            for d in devices
        }
        deadline = min(member_deadlines.values())
        stop = fx.annotate([ContactWaypoint(
            position=anchor, devices=devices, bucket=offer.bucket, deadline_ts=deadline,
        )], positions)[0]
        best = None
        for i in range(len(remainder) + 1):
            edited = list(remainder[:i]) + [stop] + list(remainder[i:])
            # FeRRy Phase 4: in plan mode the plan's exempt stops stay protected,
            # so an insert is priced as the departure check will price it.
            fold = self.scheduler.fold_remainder(
                edited, state=state, budget_end=budget_end, pass_kind=MissionPass.COLLECT,
                **self._ferry_protected(edited, MissionPass.COLLECT),
            )
            if fold.ok and (best is None or fold.home < best[0]):
                best = (fold.home, i, edited)
        if best is None:
            return refuse("does not fit")
        home, index, edited = best
        rec.inserts.append({
            "t_s": t,
            "devices": _ids(devices),
            "position": list(anchor),
            "index": index,
            "home_s": home,
        })
        rec.inserted_devices += len(devices)
        rec.no_insert.update(devices)
        # The inserted stop was priced with these deadlines, so its members'
        # updates are held to them once on board (deadline_bounds="delivery").
        rec.deadlines.update(member_deadlines)
        log.info(
            "mule=%s beacon offer %s inserted at %d/%d (predicted home %.3f)",
            self.mule_id, _ids(devices), index, len(remainder), home,
        )
        return edited

    # ------------------------------------------------------------------ #
    # FeRRy Phase 2 — docking with other mules on the same cluster
    # ------------------------------------------------------------------ #

    #: Re-arm interval of a ``down_wait_s`` wait: short, so a stop request is
    #: honoured within about a second even during a long quorum wait.
    _DOWN_WAIT_TICK_S: float = 1.0

    def _restage(self, theta, synth, version) -> None:
        """Stage ``theta`` again for the next mission, with its batch and version."""
        self._next_theta = theta
        self._next_synth = synth
        self._next_theta_version = version

    def _dock_and_await_down(self) -> bool:
        """Upload what is staged and wait for the DOWN; True once one arrived.

        Without ``down_wait_s`` this is the recorded single ``run_dock_cycle``:
        a 10 s wait whose expiry raises and ends the mission loop. With it the
        wait is re-armed in ticks until ``down_wait_s`` has passed or a stop is
        requested, and running out returns False instead of raising: with
        several mules the answer can wait on another mule's mission. An upload
        that did not land returns True either way, and the caller's check for
        a staged θ reports it as before.
        """
        if self.down_wait_s is None:
            self.client_cluster.run_dock_cycle()
            return True
        deadline = time.monotonic() + self.down_wait_s
        tick = min(self._DOWN_WAIT_TICK_S, self.down_wait_s)
        try:
            self.client_cluster.run_dock_cycle(down_timeout_s=tick)
            return True
        except DownTimeout:
            pass
        while not self._should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                self.client_cluster.await_down(timeout=min(tick, remaining))
                return True
            except DownTimeout:
                continue
        return False

    def _dock_empty(
        self,
        mission_round: int,
        base_version: Optional[int],
        *,
        sim_upload_ts: Optional[float] = None,
        backhaul=None,
    ) -> bool:
        """Dock after a mission that collected nothing (``dock_on_empty``).

        Uploads an empty partial, tagged with this mule's rule so the cluster
        accepts its form, and the mission's round report (its sessions still
        happened), with the previous mission's Pass-2 ledger riding along as
        in any upload. The cluster counts the partial toward its quorum and
        merges nothing from it. Returns True once the DOWN has staged the next
        mission's θ, False when ``down_wait_s`` ran out first.

        FeRRy Phase 3: on the mission clock the UP carries the upload's
        simulated completion time and backhaul pricing, and a report built
        here is stamped from the clock (critic B3); both None otherwise.
        """
        unmerged = getattr(self.mission, "last_unmerged", None)
        if unmerged is not None:
            report, contacts = unmerged
        else:
            now = (time.time() if getattr(self, "_ferry_run", None) is None
                   else self._now())
            report = MissionRoundCloseReport(
                mule_id=self.mule_id, mission_round=mission_round,
                started_at=now, finished_at=now,
            )
            contacts = ContactHistory(mule_id=self.mule_id, mission_round=mission_round)
        spec = self.aggregation
        empty = PartialAggregate(
            mule_id=self.mule_id,
            mission_round=mission_round,
            weights=[],
            num_examples=0,
            rule=spec.rule,
            update_form=spec.update_form,
            base_version=base_version,
        )
        self.client_cluster.collect(
            partial_aggregate=empty,
            report=report,
            contacts=contacts,
            delivery_report=self._pending_delivery_report,
            sim_upload_ts=sim_upload_ts,
            backhaul=backhaul,
        )
        self._pending_delivery_report = None
        if not self.client_cluster.wait_for_dock(timeout=None):
            raise MuleSupervisorError("dock did not become available after an empty mission")
        if not self._dock_and_await_down():
            return False
        if self._next_theta is None:
            raise MuleSupervisorError(
                "empty-mission dock did not stage a model — the cluster must "
                "answer every upload with a DOWN"
            )
        return True

    # ------------------------------------------------------------------ #
    # L1 channel pick — features per design §2.6 state vector
    # ------------------------------------------------------------------ #

    def _pick_channel(self, wp: TargetWaypoint) -> Optional[int]:
        """Run L1 inference for one waypoint. Returns the chosen band index.

        Returns ``None`` when no channel actor is wired. The choice is
        not actuated on the loopback radio — it's recorded for trace.
        """
        if self.channel_actor is None:
            return None
        return self._pick_channel_at_position(wp.position)

    def _remaining_is_feasible(self, remaining: List[ContactWaypoint]) -> bool:
        """S3b in-flight — can the rest of the queue still be served in time?

        Re-runs the same feasibility check the scheduler applied before
        take-off, but from the mule's **current** pose and clock. Returns True
        (fly on) when no mission budget is configured, so this is inert unless
        deadline enforcement is switched on — matching S3b's opt-in contract.

        Only the *first* remaining contact needs to be reachable for the mission
        to be worth continuing: if it survives, we fly it and re-check at the
        next stop. That keeps the decision incremental rather than committing to
        a whole-queue prediction that the next contact's outcome may invalidate.
        """
        if not remaining:
            return False
        budget = self.scheduler.mission_budget_s
        if budget is None:
            return True
        start = self.scheduler.mission_start_ts
        if start is None:
            return True

        # Freeze Amendment 8 — a whole-scheduler baseline (D1/D2) replaces S3b,
        # so its route must not be held to S3b's per-device deadline in flight
        # either: MAX-AoI puts the most overdue devices first by design, and
        # the deadline test refused exactly those. Such a policy declares what
        # it re-checks; the default is the mission budget alone.
        policy = getattr(self.scheduler, "target_selector", None)
        if policy is not None and hasattr(policy, "admit_and_order"):
            from hermes.scheduler.policies.budget_walk import (
                IN_FLIGHT_BUDGET, IN_FLIGHT_NONE, greedy_budget_walk,
            )

            rule = getattr(policy, "in_flight_check", IN_FLIGHT_BUDGET)
            if rule == IN_FLIGHT_NONE:
                return True
            return bool(greedy_budget_walk(
                remaining[:1],
                key=lambda wp: (0,),
                mule_pose=self.mule_pose,
                now=self._now(),
                mission_deadline_ts=start + budget,
                model=self.scheduler.feasibility_model,
            ))

        from hermes.scheduler.stages.s3b_feasibility import filter_feasible

        feas = filter_feasible(
            remaining[:1],
            now=self._now(),
            mule_pose=self.mule_pose,
            mission_deadline_ts=start + budget,
            model=self.scheduler.feasibility_model,
        )
        return bool(feas.kept)

    def _age_caps(self) -> Optional[Dict[DeviceID, Optional[int]]]:
        """Each slice device's merge cutoff in cluster rounds (``agg:cutoff``).

        Decision D5: a device's deadline window Φ_j, scaled by S3c exactly as
        ``compute_deadline`` scales it, converts to rounds with the mission
        period. None when the rule has no cutoff.

        The mission calls this once, right after planning, and merges with
        that snapshot: by close time every session has folded its outcome into
        Φ_j (a CLEAN tightens it) and S3c may have moved its scale, so a later
        read would cut an update with a window it was never planned under.
        """
        spec = self.aggregation
        if not spec.uses_age_cap:
            return None
        from hermes.scheduler.stages.s3_deadline import effective_window

        scale = self.scheduler.window_scale
        law = self.scheduler.deadline_law
        return {
            did: age_cap(spec, effective_window(st, law=law) * scale)
            for did, st in self.scheduler.device_states.items()
        }

    def _budget_pass_2(
        self, queue: List[ContactWaypoint],
    ) -> Tuple[List[ContactWaypoint], List[ContactWaypoint]]:
        """Walk the Pass-2 queue against the budget; return (fly, skip).

        Pass 2 is a second sortie, so it gets the mission budget afresh,
        priced with the S3b cost model (transit at cruise speed plus the
        session time) from the mule's tracked pose — its last Pass-1 stop,
        since the pose returns to the dock only once FeRRy Phase 3 lands the
        mission clock. The walk keeps the queue's nearest-first order and
        skips, rather than stops at, a contact that does not fit — the greedy
        walk the D1/D2 arms use. Planned seconds only, no wall clock, so the
        cut is reproducible. With no budget configured nothing is skipped.
        """
        budget = self.scheduler.mission_budget_s
        if budget is None or not queue:
            return list(queue), []
        from hermes.scheduler.policies.budget_walk import greedy_budget_walk

        order = {wp: i for i, wp in enumerate(queue)}
        fly = greedy_budget_walk(
            queue,
            key=lambda wp: (order[wp],),
            mule_pose=self.mule_pose,
            now=0.0,
            mission_deadline_ts=float(budget),
            model=self.scheduler.feasibility_model,
        )
        kept = set(fly)
        return fly, [wp for wp in queue if wp not in kept]

    def _widen_abandoned(
        self, abandoned: List[ContactWaypoint], *, mission_round: int,
    ) -> None:
        """Feed a TIMEOUT delta for every device we could not reach.

        Deltas are normally emitted only from inside a contact session, so a
        device that is never visited gets no feedback and its fulfilment window
        never adapts. That matters most for devices S3b drops or an in-flight
        abort abandons: without a signal they stay just as un-serveable next
        mission. Widening Φ is the same response the scheduler already gives a
        missed contact. Every reason is treated alike: a contact refused by
        the on-board clause (``delivery``) is widened as a budget drop is,
        since neither device was late itself.
        """
        from hermes.types import MissionOutcome, RoundCloseDelta

        now = self._now()
        for wp in abandoned:
            for did in wp.devices:
                try:
                    self.scheduler.ingest_round_close_delta(
                        RoundCloseDelta(
                            device_id=did,
                            mule_id=self.mule_id,
                            mission_round=mission_round,
                            outcome=MissionOutcome.TIMEOUT,
                            utility=0.0,
                            contact_ts=now,
                            # Never attempted: not a reachability observation.
                            answered=False,
                            synthetic=True,
                        )
                    )
                except Exception:  # never let bookkeeping kill the sortie
                    log.exception("failed to widen deadline for %s", did)

    def _pick_channel_contact(self, wp: ContactWaypoint) -> Optional[int]:
        """Same as :meth:`_pick_channel` but for a Sprint-1.5 contact event."""
        if self.channel_actor is None:
            return None
        return self._pick_channel_at_position(wp.position)

    def _pick_channel_at_position(self, position: MulePose) -> int:
        """Shared L1 state-vector builder for both per-device and per-contact paths."""
        dist = float(np.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.mule_pose, position)
        )))
        # State per design §2.6: SNR per band, distance, mule pose, energy.
        # Sprint 1 has no real per-band SNR; we fan the rf_prior_snr_db
        # out across all three bands as a uniform prior. Real telemetry
        # arrives in Sprint 2.
        state = np.zeros(L1_STATE_DIM, dtype=np.float32)
        prior = float(self.rf_prior_snr_db) / 30.0
        state[0] = prior
        state[1] = prior
        state[2] = prior
        state[3] = dist / 100.0
        state[4] = float(self.mule_pose[0]) / 100.0
        state[5] = float(self.mule_pose[1]) / 100.0
        state[6] = float(self.mule_pose[2]) / 100.0
        state[7] = float(self.mule_energy)
        return int(self.channel_actor.argmax(state))
