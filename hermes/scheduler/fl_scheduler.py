"""FLScheduler — L2 Scheduler on the mule NUC.

Ties the pure stage functions into one per-mission object that:

* ingests slow-phase inputs at dock (``MissionSlice`` + ``ClusterAmendment``)
* ingests fast-phase inputs in-field (``RoundCloseDelta``, ``BeaconObservation``)
* pipelines S1 → S3 bucket classify → S3.5 order → visit queue

The class is deliberately I/O-free. Glue code on the mule binds it to:

    * ``ClientCluster.on_slice_and_amendment`` -> :meth:`ingest_slice`
    * ``HFLHostMission.scheduler_bus`` -> :meth:`ingest_round_close_delta`
    * L1 RF listener -> :meth:`ingest_beacon`
    * Supervisor main loop -> :meth:`build_target_queue`

Design refs:
    * HERMES_FL_Scheduler_Design.md §5.1 FLScheduler loop
    * HERMES_FL_Scheduler_Design.md §6.2 FLScheduler state
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hermes.types import (
    BUCKET_PRIORITY,
    BeaconObservation,
    Bucket,
    ClusterAmendment,
    ContactWaypoint,
    DeviceID,
    DeviceRecord,
    DeviceSchedulerState,
    FLReadyAdv,
    MissionPass,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
    TargetWaypoint,
)
from hermes.types.scheduler import DEFAULT_FULFILMENT_WINDOW_S

from .stages import (
    classify_bucket,
    cluster_by_rf_range,
    compute_deadline,
    filter_eligible,
    fold_cluster_amendment,
    fold_round_close_delta,
    is_on_contact_ready,
    order_pass_2_greedy,
    passes_fl_threshold,
    select_order,
)
from .stages.s2b_flag import DEFAULT_FL_THRESHOLD
from .stages.s3_deadline import DeadlineLaw

log = logging.getLogger(__name__)


class FLSchedulerError(RuntimeError):
    """Raised for scheduler-level invariant violations."""


MulePose = Tuple[float, float, float]


class FLScheduler:
    """Per-mule, per-mission scheduler instance.

    Not thread-safe — bind one lock at the caller if the in-field bus and
    the dock bus can race. In the current wiring the supervisor serialises
    these callbacks.
    """

    def __init__(
        self,
        *,
        fl_threshold: float = DEFAULT_FL_THRESHOLD,
        beacon_window_s: float = 30.0,
        now_fn=time.time,
        target_selector=None,
        mission_budget_s: Optional[float] = None,
        feasibility_model=None,
        mission_window_adapter=None,
        deadline_law: Optional[DeadlineLaw] = None,
        miss_priority: bool = False,
        deadline_time_scale: float = 1.0,
        initial_window_s: Optional[float] = None,
        refuse_deadline_overrides: bool = False,
        validate_flown_order: bool = False,
        replan_fallback: str = "reorder",
    ):
        self._device_states: Dict[DeviceID, DeviceSchedulerState] = {}
        # FeRRy Phase 3 (spec Q1) — the deadline law's time unit. 1.0 is the
        # recorded unit and leaves ``deadline_law`` exactly as given (None
        # stays None). Any other value is folded into the law as its
        # ``time_scale``, so every reader of ``deadline_law`` (the mule's merge
        # cutoff included) sees the same scaled Φ.
        if isinstance(deadline_time_scale, bool):
            raise FLSchedulerError(
                f"deadline_time_scale must be a number, got {deadline_time_scale!r}"
            )
        scale = float(deadline_time_scale)
        if not (math.isfinite(scale) and scale > 0.0):
            raise FLSchedulerError(
                f"deadline_time_scale must be finite and > 0, got {deadline_time_scale!r}"
            )
        if scale != 1.0:
            if (deadline_law is not None and deadline_law.time_scale != 1.0
                    and deadline_law.time_scale != scale):
                raise FLSchedulerError(
                    f"deadline_time_scale={scale} conflicts with the law's own "
                    f"time_scale={deadline_law.time_scale}; set one of them"
                )
            deadline_law = (deadline_law or DeadlineLaw()).with_time_scale(scale)
        # FeRRy Phase 3 (spec Q1) — a new device's window Φ₀, stated like every
        # other constant of the deadline law: in the law's recorded unit, and
        # multiplied by ``deadline_time_scale`` when a row is created. None is
        # the recorded 60 s, so None and an explicit 60 are the same Φ₀ at any
        # time scale (60 * scale s on the scheduler's clock); at the recorded
        # unit new rows are built exactly as before. For a Φ₀ counted in
        # missions (critic A7) pass what s3_deadline.initial_window_for_missions
        # returns: it divides this scale back out. A window already in clock
        # seconds must not be passed here unless the scale is 1.0.
        if initial_window_s is not None:
            phi0 = float(initial_window_s)
            if not (math.isfinite(phi0) and phi0 > 0.0):
                raise FLSchedulerError(
                    f"initial_window_s must be finite and > 0, got {initial_window_s!r}"
                )
            initial_window_s = phi0
        self._initial_window_s = initial_window_s
        # FeRRy Phase 3 — on the simulated mission clock the mule refuses the
        # cluster's absolute (wall-clock) deadline overrides (critic B3).
        self._refuse_deadline_overrides = bool(refuse_deadline_overrides)
        # FeRRy Phase 3 (design §3.3) — under the ``replan`` response, check
        # the order our arms will actually fly before takeoff. Needs the ferry
        # model and a budget; inert otherwise.
        self._validate_flown_order = bool(validate_flown_order)
        # FeRRy Phase 3 — what our arms' Pass-1 re-plans (the pre-flight check
        # included) do when the arm's own order over the admitted stops does
        # not fit: ``reorder`` (design §3.4 with critic C3: 2-OPT, then the
        # admission order) or ``trim`` (keep the arm's order, drop what it
        # cannot serve). See routing/replan.py for why the choice matters
        # before takeoff; it is made at the pilot (critic B5). Inert unless
        # the mule re-plans.
        from .routing.replan import FALLBACKS  # noqa: WPS433

        if replan_fallback not in FALLBACKS:
            raise FLSchedulerError(
                f"replan_fallback must be one of {FALLBACKS}, got {replan_fallback!r}"
            )
        self._replan_fallback = replan_fallback
        # FeRRy Phase 1. ``deadline_law`` None is the recorded additive law,
        # run with its original arithmetic; see stages/s3_deadline.py. With
        # ``miss_priority`` S3b admits contacts by their members' miss streak
        # before their deadline, so a device the mule missed is not also sent
        # to the back of the queue by the wider window the miss gave it.
        self._deadline_law = deadline_law
        self._miss_priority = bool(miss_priority)
        self._current_slice: Optional[MissionSlice] = None
        self._fl_threshold = fl_threshold
        self._beacon_window_s = beacon_window_s
        self._now = now_fn
        # Phase-5 S3.5 — optional learned selector. If None, the
        # deterministic distance placeholder in :func:`select_order`
        # runs.
        self._target_selector = target_selector
        # S3b — deadline feasibility gate. ``None`` (the default) leaves the
        # deadline as a sort key only, which is the historical behaviour every
        # recorded result was produced under. Supplying a per-mission budget
        # turns the deadline into an enforced constraint: contacts that cannot
        # be reached before their own deadline, or that would overrun the
        # budget, are dropped BEFORE ordering — so the learned selector still
        # cannot resurrect them.
        self._mission_budget_s = (
            None if mission_budget_s is None else float(mission_budget_s)
        )
        # FeRRy Phase 3 — a ferry model prices each contact's dwell from its
        # members' positions (critic B11); unless the caller bound its own map,
        # the model looks them up in this scheduler's device states, a live
        # view. A legacy model is kept as the very object passed.
        ferry = getattr(feasibility_model, "ferry", None)
        if ferry is not None and ferry.device_states is None:
            feasibility_model = dataclasses.replace(
                feasibility_model, ferry=ferry.bind(self._device_states),
            )
        self._feasibility_model = feasibility_model
        # The budget is measured from this stamp. The mule sets it at the start
        # of every mission (start_mission); ingest_slice also sets it, for
        # callers that drive the scheduler without a mule.
        self._mission_start_ts: Optional[float] = None
        self.last_feasibility: Optional[object] = None
        # FeRRy Phase 3 — the pre-flight order check of the latest Pass-1
        # plan (a routing.replan.ReplanResult), or None when it did not run.
        self.last_order_check: Optional[object] = None
        # Each device's own Deadline(j) from the most recent Pass-1 plan, so
        # the trace can score a miss against the device's deadline rather than
        # the tightest one in its contact.
        self.last_plan_deadlines: Dict[DeviceID, float] = {}
        # FeRRy Phase 2 — the mission being planned, when the mule says so;
        # handed to whole-scheduler policies through SelectorEnv.
        self._mission_round: Optional[int] = None
        # S3c — mission-level window adaptation. ``None`` (the default) means no
        # global widening at all: the scale is 1.0 and every deadline is exactly
        # what the per-device rule produced, which is how every recorded sweep
        # ran. Supply an *enabled* adapter to let systemic mission shortfall
        # widen all windows together — see s3c_mission_window for why the
        # per-device rule cannot see that signal on its own.
        self._window_adapter = mission_window_adapter

    # ------------------------------------------------------------------ #
    # Introspection — tests & observability
    # ------------------------------------------------------------------ #

    @property
    def device_states(self) -> Dict[DeviceID, DeviceSchedulerState]:
        """Read-only view; callers must not mutate directly."""
        return self._device_states

    @property
    def current_slice(self) -> Optional[MissionSlice]:
        return self._current_slice

    def get_state(self, device_id: DeviceID) -> Optional[DeviceSchedulerState]:
        return self._device_states.get(device_id)

    @property
    def window_scale(self) -> float:
        """S3c multiplier applied to every fulfilment window this round.

        Exactly 1.0 when no adapter is attached or the attached one is
        disabled — which is the configuration every recorded sweep ran under.
        """
        if self._window_adapter is None:
            return 1.0
        return float(self._window_adapter.scale)

    # Freeze Amendment 7 — read-only views of the S3b budget, its start stamp
    # and the cost model, so the mule stops reading the private fields.

    @property
    def mission_budget_s(self) -> Optional[float]:
        """Per-mission time budget in seconds; None = no enforcement."""
        return self._mission_budget_s

    @property
    def mission_start_ts(self) -> Optional[float]:
        """Stamp the current mission's budget is measured from."""
        return self._mission_start_ts

    @property
    def feasibility_model(self):
        """The S3b cost model, or None for its defaults."""
        return self._feasibility_model

    @property
    def deadline_law(self) -> Optional[DeadlineLaw]:
        """The fast-phase law; None = the recorded additive law."""
        return self._deadline_law

    @property
    def deadline_time_scale(self) -> float:
        """The deadline law's time unit (FeRRy Phase 3); 1.0 = recorded."""
        return 1.0 if self._deadline_law is None else self._deadline_law.time_scale

    @property
    def initial_window_s(self) -> float:
        """Φ₀ as configured, in the deadline law's recorded unit (60 s recorded).

        :attr:`effective_initial_window_s` is what a new row starts with.
        """
        if self._initial_window_s is not None:
            return self._initial_window_s
        return DEFAULT_FULFILMENT_WINDOW_S

    @property
    def effective_initial_window_s(self) -> float:
        """Φ₀ on the scheduler's clock: ``initial_window_s * deadline_time_scale``.

        The window, in seconds, a newly tracked device starts with (exactly
        the configured value at the recorded time unit).
        """
        scale = self.deadline_time_scale
        phi0 = self.initial_window_s
        return phi0 if scale == 1.0 else phi0 * scale

    @property
    def refuses_deadline_overrides(self) -> bool:
        return self._refuse_deadline_overrides

    @property
    def validates_flown_order(self) -> bool:
        return self._validate_flown_order

    @property
    def replan_fallback(self) -> str:
        """``reorder`` or ``trim``: see the constructor and routing/replan.py."""
        return self._replan_fallback

    def _new_state(self, device_id: DeviceID, **fields) -> DeviceSchedulerState:
        """A state row for a newly tracked device, starting at Φ₀.

        At the recorded settings (no ``initial_window_s``, time scale 1.0) the
        row is built exactly as before, with the dataclass's own 60 s.
        """
        st = DeviceSchedulerState(device_id=device_id, **fields)
        if self._initial_window_s is not None or self.deadline_time_scale != 1.0:
            st.deadline_fulfilment_s = self.effective_initial_window_s
        return st

    @property
    def miss_priority(self) -> bool:
        return self._miss_priority

    @property
    def target_selector(self):
        """The S3.5 selector or whole-scheduler policy; None = the placeholder."""
        return self._target_selector

    def _contact_miss_priority(self, wp: ContactWaypoint) -> int:
        """A contact's priority: the longest miss streak among its members."""
        return max(
            (self._device_states[d].miss_streak for d in wp.devices
             if d in self._device_states),
            default=0,
        )

    def record_mission_outcome(self, *, served: int, planned: int) -> None:
        """Feed one mission's served/planned into S3c. No-op without an adapter.

        Called by the mule once per mission, after the collection pass — the
        only place that knows how much of the plan actually happened.
        """
        if self._window_adapter is None:
            return
        try:
            self._window_adapter.record(served, planned)
        except Exception:  # bookkeeping must never kill a mission
            log.warning("scheduler: mission-outcome record failed", exc_info=True)

    def start_mission(self) -> float:
        """Start the S3b budget clock for a new mission; returns the stamp.

        Freeze Amendment 6. The mule calls this at the start of every mission,
        before Pass 1 is planned. Stamping only in ``ingest_slice`` tied the
        clock to DOWN bundles, which arrive mid-mission (the inter-pass dock)
        and not at all after an empty mission (no updates, so no dock). The
        next mission then planned against a stale stamp, and its budget shrank
        by however long the previous missions took.
        """
        self._mission_start_ts = self._now()
        return self._mission_start_ts

    @property
    def mission_round(self) -> Optional[int]:
        """The mission being planned, as the mule last set it; None = unknown."""
        return self._mission_round

    def set_mission_round(self, mission_round: Optional[int]) -> None:
        """Record which mission is being planned (FeRRy Phase 2).

        Whole-scheduler policies that age devices in missions (the Whittle
        baseline, arm D3) read it from ``SelectorEnv.mission_round`` instead of
        inferring it from the devices' last outcomes. Nothing else reads it.
        """
        self._mission_round = None if mission_round is None else int(mission_round)

    def record_merged(self, device_ids, mission_round: int) -> None:
        """Mark the devices whose update this mission's merge used.

        Sets ``last_merged_round``, the Age-of-Update anchor, for every tracked
        device given; unknown ids are ignored. A CLEAN whose update the age
        cutoff excluded is not passed here, so it does not reset the age.
        """
        for did in device_ids:
            st = self._device_states.get(did)
            if st is not None:
                st.last_merged_round = int(mission_round)

    # ------------------------------------------------------------------ #
    # Slow-phase ingest — dock
    # ------------------------------------------------------------------ #

    def ingest_slice(
        self,
        mission_slice: MissionSlice,
        amendment: Optional[ClusterAmendment] = None,
        *,
        registry_records: Optional[Iterable[DeviceRecord]] = None,
    ) -> None:
        """Handoff from ``ClientCluster`` after a successful dock DOWN.

        * Creates scheduler state rows for any new slice members.
        * Flips ``is_in_slice`` correctly (members in / members out).
        * Optionally pulls ``last_known_position`` + ``is_new`` from the
          registry records so the first round after dock can bucket and
          sort without waiting for a beacon.
        * Folds the amendment (deadline overrides + registry_deltas).

        With ``refuse_deadline_overrides`` (the simulated mission clock) an
        amendment carrying deadline overrides is refused before anything is
        ingested: they are wall-clock stamps (critic B3).
        """
        if (self._refuse_deadline_overrides and amendment is not None
                and amendment.deadline_overrides):
            raise FLSchedulerError(
                f"ingest_slice: {len(amendment.deadline_overrides)} cluster deadline "
                "override(s) refused: they are wall-clock stamps and this "
                "scheduler runs on the simulated mission clock"
            )
        self._current_slice = mission_slice
        # Fallback budget stamp for callers that drive the scheduler without a
        # mule. The mule re-stamps at the start of every mission
        # (start_mission): a DOWN also arrives mid-mission, at the inter-pass
        # dock, and not at all after an empty mission.
        self._mission_start_ts = self._now()
        slice_ids = set(mission_slice.device_ids)

        # Pre-seed from registry if the caller handed it over.
        # H4 — copy delivery_priority alongside last_known_position so
        # S3a clustering's tie-breaker reads the *current* cluster-side
        # value, not a stale 0. Without this, the cluster's bumped
        # delivery_priority on undelivered devices never reaches the
        # mule and S3a never pulls high-priority devices to anchors.
        if registry_records is not None:
            for rec in registry_records:
                st = self._device_states.get(rec.device_id)
                if st is None:
                    st = self._new_state(
                        rec.device_id,
                        is_new=rec.is_new,
                        last_known_position=rec.last_known_position,
                        delivery_priority=rec.delivery_priority,
                    )
                    self._device_states[rec.device_id] = st
                else:
                    st.last_known_position = rec.last_known_position
                    st.delivery_priority = rec.delivery_priority

        # Admit every slice member that isn't already tracked.
        for did in mission_slice.device_ids:
            if did not in self._device_states:
                self._device_states[did] = self._new_state(did)

        # Refresh slice membership flags.
        for did, st in self._device_states.items():
            st.is_in_slice = did in slice_ids

        # Slow-phase deadline fold. (Overrides were refused above, before
        # anything was ingested, when this scheduler must refuse them.)
        if amendment is not None:
            fold_cluster_amendment(
                self._device_states, amendment, law=self._deadline_law,
            )

        log.info(
            "scheduler ingest_slice round=%d slice_size=%d amendments=%s",
            mission_slice.issued_round,
            len(mission_slice.device_ids),
            len(amendment.deadline_overrides) if amendment else 0,
        )

    # ------------------------------------------------------------------ #
    # Fast-phase ingest — in-field bus
    # ------------------------------------------------------------------ #

    def ingest_round_close_delta(self, delta: RoundCloseDelta) -> None:
        """Apply one in-mission delta from ``HFLHostMission``."""
        st = self._device_states.get(delta.device_id)
        if st is None:
            # We track only slice + beacon-heard devices; an untracked ID
            # is a programming error, not a silent miss.
            raise FLSchedulerError(
                f"RoundCloseDelta for untracked device {delta.device_id!r}"
            )
        fold_round_close_delta(st, delta, law=self._deadline_law)

    def ingest_beacon(self, obs: BeaconObservation) -> None:
        """Opportunistic RF beacon (design §4 step 16).

        If we've never seen the device, we stand up a minimal state row so
        S1 can admit it. Position is unknown — the selector will treat it
        as infinitely far, which is deliberately conservative.
        """
        st = self._device_states.get(obs.device_id)
        if st is None:
            st = self._new_state(obs.device_id, is_new=True)
            self._device_states[obs.device_id] = st
        st.last_beacon_ts = obs.observed_at

    def ingest_ready_adv(self, adv: FLReadyAdv, *, now: Optional[float] = None) -> bool:
        """S2A + S2B verification at contact.

        Returns True if the advert passes both gates — the caller
        (``HFLHostMission``) then proceeds with ``push_model``. A False
        return means the mule should mark the device timed-out / skipped
        for this attempt.
        """
        _now = self._now() if now is None else now
        if not is_on_contact_ready(adv, now=_now):
            return False
        if not passes_fl_threshold(adv, fl_threshold=self._fl_threshold):
            return False
        return True

    # ------------------------------------------------------------------ #
    # Plan / build — what the supervisor calls per pipeline pass
    # ------------------------------------------------------------------ #

    def build_target_queue(
        self,
        *,
        now: Optional[float] = None,
        mule_pose: MulePose = (0.0, 0.0, 0.0),
        mule_energy: float = 1.0,
        rf_prior_snr_db: float = 20.0,
    ) -> List[TargetWaypoint]:
        """Run the full S1 → S3 → S3.5 pipeline and return the visit queue.

        The queue walks buckets in :data:`BUCKET_PRIORITY` order. Inside
        each bucket:

        * If a ``target_selector`` was injected at construction time
          (Phase 5 :class:`TargetSelectorRL`), it ranks the bucket.
        * Otherwise the deterministic distance-sorted placeholder is
          used (Phase 4 fallback).

        ``mule_energy`` and ``rf_prior_snr_db`` are consumed only by the
        learned selector; they're ignored by the placeholder.
        """
        _now = self._now() if now is None else now
        _wscale = self.window_scale  # S3c; 1.0 unless enabled

        # S1 — eligibility.
        eligible_ids = filter_eligible(
            self._device_states, now=_now, beacon_window_s=self._beacon_window_s
        )

        # S3 — bucket-classify each eligible device and cache their deadlines.
        by_bucket: Dict[Bucket, List[DeviceID]] = {b: [] for b in BUCKET_PRIORITY}
        deadlines: Dict[DeviceID, float] = {}
        for did in eligible_ids:
            st = self._device_states[did]
            try:
                bucket = classify_bucket(
                    st, now=_now, beacon_window_s=self._beacon_window_s
                )
            except ValueError:
                # Slipped past S1 due to stale state; drop it and log.
                log.warning("scheduler: S3 refused to bucket %s", did)
                continue
            st.bucket = bucket
            by_bucket[bucket].append(did)
            deadlines[did] = compute_deadline(
                st, now=_now, window_scale=_wscale, law=self._deadline_law,
            )

        # S3.5 — intra-bucket order.
        selector_env = None
        if self._target_selector is not None:
            # Lazy import so the selector package stays optional.
            from .selector import SelectorEnv  # noqa: WPS433
            selector_env = SelectorEnv(
                mule_pose=mule_pose,
                mule_energy=mule_energy,
                rf_prior_snr_db=rf_prior_snr_db,
                beacon_window_s=self._beacon_window_s,
                now=_now,
            )

        queue: List[TargetWaypoint] = []
        for bucket in BUCKET_PRIORITY:
            members = by_bucket[bucket]
            if not members:
                continue
            if self._target_selector is not None and selector_env is not None:
                ordered = self._target_selector.rank(
                    members,
                    self._device_states,
                    bucket=bucket,
                    env=selector_env,
                )
            else:
                ordered = select_order(
                    members, self._device_states, mule_pose=mule_pose
                )
            for did in ordered:
                st = self._device_states[did]
                queue.append(
                    TargetWaypoint(
                        device_id=did,
                        position=st.last_known_position,
                        bucket=bucket,
                        deadline_ts=deadlines[did],
                    )
                )

        return queue

    # ------------------------------------------------------------------ #
    # Sprint 1.5 — contact-aware queue builders
    # ------------------------------------------------------------------ #

    def build_contact_queue(
        self,
        *,
        rf_range_m: float,
        now: Optional[float] = None,
        mule_pose: MulePose = (0.0, 0.0, 0.0),
        mule_energy: float = 1.0,
        rf_prior_snr_db: float = 20.0,
    ) -> List[ContactWaypoint]:
        """Pass-1 contact-event queue: S1 → S3 deadline + bucket → S3a → S3.5.

        Sprint 1.5 design §7 principle 15. Pipeline:

        1. S1 — eligibility filter (existing).
        2. S3 — per-device deadline math + bucket classify (existing).
        3. S3a — group eligible devices into ContactWaypoints by
           ``rf_range_m``; each contact inherits the worst (highest-
           priority) bucket among its members.
        4. Walk :data:`BUCKET_PRIORITY` over the contacts. Inside each
           bucket: if a learned selector is wired, call ``rank_contacts``;
           otherwise sort contacts by distance from ``mule_pose``.

        Pass 2 has its own ordering (``build_pass_2_queue``); the
        selector is bypassed there.
        """
        if rf_range_m <= 0.0:
            raise FLSchedulerError(
                f"build_contact_queue requires rf_range_m > 0, got {rf_range_m}"
            )
        _now = self._now() if now is None else now
        _wscale = self.window_scale  # S3c; 1.0 unless enabled
        # Freeze Amendment 8 — this plan's diagnostics only. An early return
        # below used to leave the previous mission's gate result in place, and
        # the mule would widen that mission's dropped devices a second time.
        self.last_feasibility = None
        self.last_plan_deadlines = {}
        self.last_order_check = None

        eligible_ids = filter_eligible(
            self._device_states, now=_now, beacon_window_s=self._beacon_window_s
        )
        if not eligible_ids:
            return []

        # S3 — bucket + deadline per eligible device.
        deadlines: Dict[DeviceID, float] = {}
        for did in eligible_ids:
            st = self._device_states[did]
            try:
                bucket = classify_bucket(
                    st, now=_now, beacon_window_s=self._beacon_window_s
                )
            except ValueError:
                log.warning("build_contact_queue: S3 refused to bucket %s", did)
                continue
            st.bucket = bucket
            deadlines[did] = compute_deadline(
                st, now=_now, window_scale=_wscale, law=self._deadline_law,
            )

        self.last_plan_deadlines = dict(deadlines)

        # Filter out anyone S3 couldn't bucket (kept simple — drop them).
        bucketed = [d for d in eligible_ids if self._device_states[d].bucket is not None]
        if not bucketed:
            return []

        # S3a — cluster into ContactWaypoints.
        contacts = cluster_by_rf_range(
            eligible_device_ids=bucketed,
            device_states=self._device_states,
            deadlines=deadlines,
            rf_range_m=rf_range_m,
        )
        if not contacts:
            return []

        # Freeze Amendment 4 — whole-scheduler baseline delegation (arms D1/D2).
        #
        # A policy that exposes `admit_and_order` owns BOTH decisions: it
        # replaces S3's deadline ordering, S3b's admission gate and S3.5's
        # tie-break, and returns the route directly. This exists because the
        # earlier ordering-only comparison was vacuous — S3b fixes *who* is
        # served before any ranking policy runs, so a baseline confined to the
        # selector slot could only permute a list our gate had already decided,
        # and every arm produced byte-identical results.
        #
        # S1 and S3a still run above: slice membership and RF clustering are
        # physics, not policy, and every arm must face the same ones. The budget
        # is passed through unchanged, and the baseline is handed OUR feasibility
        # model so both arms price travel identically — otherwise the experiment
        # would measure whose cost model is cheaper, not whose policy is better.
        #
        # Inert for every arm that does not implement the method (H0-H3), so no
        # recorded result is affected.
        if self._target_selector is not None and hasattr(
            self._target_selector, "admit_and_order"
        ):
            from .selector import SelectorEnv  # noqa: WPS433

            start = (self._mission_start_ts
                     if self._mission_start_ts is not None else _now)
            route = self._target_selector.admit_and_order(
                contacts,
                self._device_states,
                SelectorEnv(
                    mule_pose=mule_pose,
                    mule_energy=mule_energy,
                    rf_prior_snr_db=rf_prior_snr_db,
                    beacon_window_s=self._beacon_window_s,
                    now=_now,
                    mission_round=self._mission_round,
                ),
                mission_deadline_ts=(
                    None if self._mission_budget_s is None
                    else start + self._mission_budget_s
                ),
                feasibility_model=self._feasibility_model,
            )
            log.info(
                "whole-scheduler policy %s admitted %d/%d contacts",
                getattr(self._target_selector, "name", "?"),
                len(route), len(contacts),
            )
            return route

        # S3b — deadline feasibility gate (hard, and BEFORE ordering, so the
        # learned selector cannot resurrect anything it drops). No-op unless a
        # mission budget is configured.
        if self._mission_budget_s is not None:
            from .stages.s3b_feasibility import filter_feasible  # noqa: WPS433

            start = self._mission_start_ts if self._mission_start_ts is not None else _now
            feas = filter_feasible(
                contacts,
                now=_now,
                mule_pose=mule_pose,
                mission_deadline_ts=start + self._mission_budget_s,
                model=self._feasibility_model,
                priority=(
                    self._contact_miss_priority if self._miss_priority else None
                ),
            )
            self.last_feasibility = feas
            if feas.n_dropped:
                log.info(
                    "S3b feasibility gate dropped %d/%d contacts "
                    "(%d unreachable before deadline, %d over mission budget)",
                    feas.n_dropped, len(contacts),
                    len(feas.dropped_overdue), len(feas.dropped_budget),
                )
            contacts = feas.kept
            if not contacts:
                return []

        # Group contacts by their inherited bucket and walk priority order.
        by_bucket: Dict[Bucket, List[ContactWaypoint]] = {b: [] for b in BUCKET_PRIORITY}
        for c in contacts:
            by_bucket[c.bucket].append(c)

        selector_env = None
        if self._target_selector is not None:
            from .selector import SelectorEnv  # noqa: WPS433
            selector_env = SelectorEnv(
                mule_pose=mule_pose,
                mule_energy=mule_energy,
                rf_prior_snr_db=rf_prior_snr_db,
                beacon_window_s=self._beacon_window_s,
                now=_now,
            )

        # Distance-from-mule sort — the deterministic fallback, also used
        # for single-candidate buckets (see below).
        def _dist_key(wp: ContactWaypoint) -> float:
            return sum(
                (a - b) ** 2 for a, b in zip(mule_pose, wp.position)
            ) ** 0.5

        queue: List[ContactWaypoint] = []
        for bucket in BUCKET_PRIORITY:
            members = by_bucket[bucket]
            if not members:
                continue
            # Design §2.7: the selector is only consulted when a bucket
            # has ≥2 candidate positions. With one candidate there is
            # nothing to choose between, so we skip the DDQN forward
            # pass and emit the lone contact directly. ``argmax`` over a
            # 1-row matrix would give the same result, but the design
            # text explicitly carves out this short-circuit and the code
            # should match.
            use_selector = (
                self._target_selector is not None
                and selector_env is not None
                and len(members) >= 2
            )
            if use_selector:
                # M2 — pass the upstream-admitted set (= every eligible
                # device this round) so the selector's scope guard can
                # actually fire if a bucket leaks a gated-out device.
                ordered = self._target_selector.rank_contacts(
                    members,
                    self._device_states,
                    env=selector_env,
                    pass_kind=MissionPass.COLLECT,
                    admitted=bucketed,
                )
            else:
                ordered = sorted(members, key=_dist_key)
            queue.extend(ordered)

        # FeRRy Phase 3 (design §3.3) — S3b admitted the contacts in EDF order,
        # but the H arms fly them in bucket/distance or learned order, which
        # the predicate never saw (probe P1: a 70 s walk flown as 150 s). Under
        # the ``replan`` response, fold the order about to be flown and repair
        # it if it fails; drops join ``last_feasibility`` by reason, so the
        # mule's pre-flight widening covers them. Needs the ferry model and a
        # budget: no budget, no gate (the opt-in contract).
        if (self._validate_flown_order and self._mission_budget_s is not None
                and queue
                and getattr(self._feasibility_model, "ferry", None) is not None):
            queue = self._validate_order(queue, now=_now, mule_pose=mule_pose)

        return queue

    def _validate_order(
        self, queue: List[ContactWaypoint], *, now: float, mule_pose: MulePose,
    ) -> List[ContactWaypoint]:
        """The pre-flight order check (design §3.3): keep, repair or trim ``queue``.

        ``queue`` is S3b's kept set in the order the arm will fly it, and the
        re-plan starts from the very state and budget S3b just walked, so S3b
        re-admits every stop and the arm's own order over them is ``queue``
        itself. What happens when that order does not fit therefore depends
        only on :attr:`replan_fallback`:

        * ``reorder`` (default): the route becomes 2-OPT's path or S3b's EDF
          order, both functions of the kept *set*. Arms that differ only in
          their order (H1, H2, H3) fly the same route whenever this check
          fires, and the check never drops a stop (the EDF order passes by
          construction), so ``last_feasibility`` keeps S3b's drops;
        * ``trim``: the arm's order is kept and the stops it cannot serve in
          that order are dropped; they join ``last_feasibility`` under their
          reason (energy included, critic B10, and the on-board ``delivery``
          clause), so the mule's pre-flight widening covers them.

        Critic C3 asked that the H arms not collapse onto one route; under
        ``reorder`` they still do here. The alternatives (``trim``, or the
        ``abort`` response for the H arms) are chosen at the pilot.
        """
        from .routing.replan import ORDER_CURRENT  # noqa: WPS433
        from .stages.s3b_feasibility import (  # noqa: WPS433
            REASON_BUDGET,
            REASON_DELIVERY,
            REASON_ENERGY,
            REASON_OVERDUE,
            FeasibilityResult,
            FlightState,
        )

        start = self._mission_start_ts if self._mission_start_ts is not None else now
        res = self.replan_remainder(
            queue,
            state=FlightState(tuple(mule_pose), float(now)),  # type: ignore[arg-type]
            budget_end=start + self._mission_budget_s,  # type: ignore[operator]
            pass_kind=MissionPass.COLLECT,
        )
        self.last_order_check = res
        if res.order_used == ORDER_CURRENT:
            return queue
        log.info(
            "flown-order check: %s order, %d/%d contacts kept (%d dropped)",
            res.order_used, len(res.route), len(queue), len(res.dropped),
        )
        feas = self.last_feasibility
        self.last_feasibility = FeasibilityResult(
            list(res.route),
            list(getattr(feas, "dropped_overdue", ())) + res.dropped_by(REASON_OVERDUE),
            list(getattr(feas, "dropped_budget", ())) + res.dropped_by(REASON_BUDGET),
            list(getattr(feas, "dropped_energy", ())) + res.dropped_by(REASON_ENERGY),
            list(getattr(feas, "dropped_delivery", ())) + res.dropped_by(REASON_DELIVERY),
        )
        return list(res.route)

    # ------------------------------------------------------------------ #
    # FeRRy Phase 3 — the in-flight rule and the re-plan (design §3.4)
    # ------------------------------------------------------------------ #

    def in_flight_rule(self, pass_kind: MissionPass = MissionPass.COLLECT) -> str:
        """The predicate rule this mule's remainder is held to in flight.

        Pass 2 is walked the same way by every arm: the budget only. In
        Pass 1 our arms keep S3b's deadline and budget; a whole-scheduler
        baseline gets what it declares as ``in_flight_check`` (Freeze
        Amendment 8): the budget only, or no check at all (D4).
        """
        from .stages.s3b_feasibility import (  # noqa: WPS433
            RULE_BUDGET,
            RULE_DEADLINE_BUDGET,
            RULE_NONE,
        )

        if MissionPass(pass_kind) is MissionPass.DELIVER:
            return RULE_BUDGET
        policy = self._target_selector
        if policy is not None and hasattr(policy, "admit_and_order"):
            from .policies.budget_walk import (  # noqa: WPS433
                IN_FLIGHT_BUDGET,
                IN_FLIGHT_NONE,
            )

            check = getattr(policy, "in_flight_check", IN_FLIGHT_BUDGET)
            return RULE_NONE if check == IN_FLIGHT_NONE else RULE_BUDGET
        return RULE_DEADLINE_BUDGET

    def fold_remainder(
        self,
        remainder: Sequence[ContactWaypoint],
        *,
        state,
        budget_end: Optional[float],
        pass_kind: MissionPass = MissionPass.COLLECT,
        protected: Collection[ContactWaypoint] = (),
        snr_offset_db: float = 0.0,
    ):
        """The whole-remainder check at a departure (design §3.4).

        ``remainder`` folded as it would be flown, without skipping, from
        ``state`` (an ``s3b_feasibility.FlightState``) under this arm's
        :meth:`in_flight_rule`, priced with the scheduler's model. Returns the
        ``s3b_feasibility.FoldResult``: ``ok`` says whether the rest of the
        pass still fits, ``rejected`` which stops do not and why, and ``home``
        when the mule would be back at the dock (D4's overrun: its rule never
        rejects but still reports it). The mule calls this, and
        :meth:`replan_remainder` when it fails, rather than re-implementing
        S3b (design principle 1). ``budget_end`` None: no gate, so it passes.
        """
        from .stages.s3b_feasibility import FeasibilityModel  # noqa: WPS433

        pass_kind = MissionPass(pass_kind)
        model = self._feasibility_model or FeasibilityModel()
        return model.fold(
            list(remainder), state, rule=self.in_flight_rule(pass_kind),
            budget_end=budget_end, pass_kind=pass_kind, skip=False,
            protected=protected, snr_offset_db=snr_offset_db,
        )

    def replan_remainder(
        self,
        remainder: Sequence[ContactWaypoint],
        *,
        state,
        budget_end: Optional[float],
        pass_kind: MissionPass = MissionPass.COLLECT,
        protected: Collection[ContactWaypoint] = (),
        snr_offset_db: float = 0.0,
        two_opt_fallback: Optional[bool] = None,
    ):
        """Re-plan the rest of a pass from a flight state (design §3.4, critic C3).

        ``state`` is an ``s3b_feasibility.FlightState`` (pose, clock, energy
        spent); ``budget_end`` the absolute end of this pass's budget (None:
        no gate, so the remainder is kept). Returns a
        ``routing.replan.ReplanResult``. The arm decides admission and keeps
        its own order when that passes:

        * our arms (no whole-scheduler policy): S3b's ``filter_feasible`` from
          ``state``, by miss priority when that is on, else EDF; when their
          own order over the admitted stops does not fit, the 2-OPT path to
          the dock and then the admission order (``replan_fallback`` =
          ``reorder``), or their own order with what it cannot serve dropped
          (``trim``);
        * D1-D3 and D5: the policy's own ``admit_and_order`` with
          ``SelectorEnv(mule_pose=pose, now=clock, mission_round=...)`` and
          the model's capacity reduced by the energy spent; no 2-OPT (the
          baseline keeps its own order);
        * D4 (``IN_FLIGHT_NONE``): no re-plan; the remainder is flown as is;
        * Pass 2, every arm: a nearest-first budget walk, 2-OPT fallback.

        Drops are final for the mission (spec Q10). ``protected`` stops are
        admitted first and dropped only if they cannot all fit on their own
        (empty in Phase 3). ``snr_offset_db`` is δ_obs (0 by default, spec);
        a baseline's own admission cannot take it and prices at 0 dB, and the
        final fold still holds its route to δ_obs. ``two_opt_fallback``
        overrides the per-arm default above (None keeps it): 2-OPT would give
        a baseline a FeRRy mechanism (critic B5), so it is off for D1-D3/D5
        unless asked for. It has no effect under ``trim``, which never
        re-orders.

        A baseline's re-admission is a call to its own ``admit_and_order``
        over the remainder, so what that call does to the policy happens
        here too: Whittle (D3) overwrites ``last_device_inputs`` with the
        re-plan's inputs, and Oort (D2) infers its current round from the
        remainder's members only. A baseline's drop is labelled with the
        clause that refuses the stop on its own from ``state``, else
        ``budget`` (the policy's walk does not report which clause bound).
        """
        from .routing.replan import (  # noqa: WPS433
            FALLBACK_REORDER,
            ORDER_NONE,
            ReplanResult,
            replan_route,
        )
        from .stages.s3b_feasibility import (  # noqa: WPS433
            REASON_BUDGET,
            REASON_DELIVERY,
            REASON_ENERGY,
            REASON_OVERDUE,
            RULE_BUDGET,
            RULE_NONE,
            FeasibilityModel,
            filter_feasible,
        )

        pass_kind = MissionPass(pass_kind)
        stops = list(remainder)
        rule = self.in_flight_rule(pass_kind)
        if rule == RULE_NONE:
            return ReplanResult(tuple(stops), (), ORDER_NONE)
        model = self._feasibility_model or FeasibilityModel()
        ferry = getattr(model, "ferry", None)
        dock = None if ferry is None else ferry.dock
        policy = self._target_selector
        two_opt = True
        # ``replan_fallback`` is for our arms' Pass 1 only: a baseline's
        # admission order is already its own policy's order, and every arm
        # flies the same nearest-first Pass 2, so neither can collapse there.
        fallback = FALLBACK_REORDER

        if pass_kind is MissionPass.DELIVER:
            def admission(contacts, st):
                chain = order_pass_2_greedy(contacts, mule_pose=st.pose)
                walk = model.fold(
                    chain, st, rule=RULE_BUDGET, budget_end=budget_end,
                    pass_kind=MissionPass.DELIVER, skip=True,
                    snr_offset_db=snr_offset_db,
                )
                return list(walk.route), list(walk.rejected)
        elif policy is not None and hasattr(policy, "admit_and_order"):
            two_opt = False
            from .selector import SelectorEnv  # noqa: WPS433

            def admission(contacts, st):
                route = policy.admit_and_order(
                    list(contacts),
                    self._device_states,
                    SelectorEnv(
                        mule_pose=tuple(st.pose),
                        beacon_window_s=self._beacon_window_s,
                        now=st.clock,
                        mission_round=self._mission_round,
                    ),
                    mission_deadline_ts=budget_end,
                    feasibility_model=model.with_energy_spent(st.energy_j),
                )
                kept = {id(wp) for wp in route}
                reasons = []
                for wp in contacts:
                    if id(wp) in kept:
                        continue
                    # The clause that refuses the stop on its own from here;
                    # otherwise the budget the policy's higher-ranked stops
                    # took (the policy's walk does not say which).
                    v = model.admit(st, wp, rule=rule, budget_end=budget_end,
                                    pass_kind=pass_kind, snr_offset_db=snr_offset_db)
                    reasons.append((wp, v.reason if not v.ok else REASON_BUDGET))
                return route, reasons
        else:
            priority = self._contact_miss_priority if self._miss_priority else None
            fallback = self._replan_fallback

            def admission(contacts, st):
                feas = filter_feasible(
                    contacts, now=st.clock, mule_pose=st.pose,
                    mission_deadline_ts=budget_end, model=model, priority=priority,
                    state=st, snr_offset_db=snr_offset_db,
                )
                reasons = (
                    [(wp, REASON_OVERDUE) for wp in feas.dropped_overdue]
                    + [(wp, REASON_BUDGET) for wp in feas.dropped_budget]
                    + [(wp, REASON_ENERGY) for wp in feas.dropped_energy]
                    + [(wp, REASON_DELIVERY) for wp in feas.dropped_delivery]
                )
                return feas.kept, reasons

        if two_opt_fallback is not None:
            two_opt = bool(two_opt_fallback)
        return replan_route(
            stops, state=state, model=model, rule=rule, budget_end=budget_end,
            pass_kind=pass_kind, admission=admission, protected=protected,
            dock=dock, two_opt_fallback=two_opt, snr_offset_db=snr_offset_db,
            fallback=fallback,
        )

    def build_pass_2_queue(
        self,
        *,
        rf_range_m: float,
        now: Optional[float] = None,
        mule_pose: MulePose = (0.0, 0.0, 0.0),
    ) -> List[ContactWaypoint]:
        """Pass-2 delivery queue: every slice contact, nearest-first greedy.

        Sprint 1.5 design §7 principle 13 + Implementation Plan §3.6.2
        task 6: Pass 2 walks every contact in the slice — no skipping,
        no selector, no bucket priority. Order is greedy nearest-first
        from the post-Pass-1 ``mule_pose`` so propulsion energy on the
        return-leg is minimised.

        Caller is expected to advance ``mule_pose`` to the contact's
        position after each visit; this method computes one ordering
        in one shot, not an interactive policy.
        """
        if rf_range_m <= 0.0:
            raise FLSchedulerError(
                f"build_pass_2_queue requires rf_range_m > 0, got {rf_range_m}"
            )
        _now = self._now() if now is None else now
        _wscale = self.window_scale  # S3c; 1.0 unless enabled

        # Pass 2 must reach every slice member regardless of S1's
        # eligibility gate — even devices whose deadlines have passed
        # need the new θ. So we cluster the ENTIRE slice, not just
        # eligible_ids.
        slice_ids: List[DeviceID] = [
            did for did, st in self._device_states.items() if st.is_in_slice
        ]
        if not slice_ids:
            return []

        # M3 — DO NOT mutate scheduler state from Pass-2 ordering.
        # Earlier code force-set ``st.bucket = SCHEDULED_THIS_ROUND``
        # whenever a state had no bucket yet, which leaked Pass-2's
        # synthetic bucket into the *next* Pass-1's S3 classification.
        # Instead, build a shadow state map for the clustering call:
        # any state without a bucket gets a transient SCHEDULED tag
        # that lives only for the duration of this method.
        deadlines: Dict[DeviceID, float] = {}
        shadow_states: Dict[DeviceID, DeviceSchedulerState] = {}
        for did in slice_ids:
            st = self._device_states[did]
            deadlines[did] = compute_deadline(
                st, now=_now, window_scale=_wscale, law=self._deadline_law,
            )
            if st.bucket is None:
                # Build a shallow copy with a transient bucket — the
                # original state row is left untouched.
                shadow = DeviceSchedulerState(
                    device_id=st.device_id,
                    is_in_slice=st.is_in_slice,
                    is_new=st.is_new,
                    last_outcome=st.last_outcome,
                    last_contact_ts=st.last_contact_ts,
                    last_utility=st.last_utility,
                    on_time_count=st.on_time_count,
                    missed_count=st.missed_count,
                    delivery_priority=st.delivery_priority,
                    deadline_fulfilment_s=st.deadline_fulfilment_s,
                    idle_time_ref_ts=st.idle_time_ref_ts,
                    deadline_override_ts=st.deadline_override_ts,
                    last_beacon_ts=st.last_beacon_ts,
                    last_known_position=st.last_known_position,
                )
                shadow.bucket = Bucket.SCHEDULED_THIS_ROUND
                shadow_states[did] = shadow
            else:
                shadow_states[did] = st

        contacts = cluster_by_rf_range(
            eligible_device_ids=slice_ids,
            device_states=shadow_states,
            deadlines=deadlines,
            rf_range_m=rf_range_m,
        )

        return order_pass_2_greedy(contacts, mule_pose=mule_pose)


# --------------------------------------------------------------------------- #
# FeRRy Phase 3 — the nominal mission period T_nom (spec Q1)
# --------------------------------------------------------------------------- #

def nominal_mission_period_s(
    device_positions: Mapping[DeviceID, Sequence[float]],
    *,
    rf_range_m: float,
    feasibility_model,
    turnaround_s: float,
) -> float:
    """One layout's nominal two-pass mission period on the mission clock.

    Spec Q1: the recorded deadline constants were set against missions of
    about 10 s of wall clock, and a two-pass mission on the simulated clock
    runs minutes (design §0 finding 2), so Phase 3 restates them in units of
    ``T_nom``, the cell's median of this period over its layouts
    (:func:`median_nominal_mission_period_s`): ``deadline_time_scale = T_nom
    / 10 s`` (``s3_deadline.time_scale_for_period``), Φ₀ in missions
    (``s3_deadline.initial_window_for_missions``), D5's ``period_s`` and the
    backhaul period ``P_bh = n_missions * T_nom``.

    "Nominal" means every slice device is served and nothing fails (no
    listen charge), priced with the physics ``feasibility_model`` carries
    (the wide class in Phase 3, the declared payload, the predicted upload),
    and one plan for every arm: the scheduler's own plan with no budget and
    no selector, from the ferry dock (S1, S3, S3a with ``rf_range_m``, bucket
    then distance order; Pass 2 nearest-first). The period therefore does
    not depend on the arm being run, and it is a pure function of the layout.

    It is Pass 1 (each leg's transit and predicted dwell, the return leg and
    the upload) + ``turnaround_s`` + Pass 2 (each leg's transit and Pass-2
    dwell, the return leg): the design §0 probe's sum, with the ferry model's
    dwell and upload in place of its fixed 0.03 s per device.
    ``feasibility_model`` must carry the ferry physics; its members are looked
    up in this layout, whatever device map it was bound to.
    """
    from .stages.s3b_feasibility import RULE_NONE, FlightState  # noqa: WPS433

    ferry = getattr(feasibility_model, "ferry", None)
    if ferry is None:
        raise FLSchedulerError(
            "nominal_mission_period_s prices the mission on the mission clock: it "
            "needs a FeasibilityModel carrying the ferry physics"
        )
    turn = float(turnaround_s)
    if not (math.isfinite(turn) and turn >= 0.0):
        raise FLSchedulerError(f"turnaround_s must be finite and >= 0, got {turnaround_s!r}")
    positions = {}
    for did, pos in device_positions.items():
        pose = tuple(float(c) for c in pos)
        if len(pose) != 3:
            raise FLSchedulerError(f"position of {did!r} must be (x, y, z), got {pos!r}")
        positions[did] = pose

    planner = FLScheduler(now_fn=lambda: 0.0)
    planner.ingest_slice(MissionSlice(
        mule_id=MuleID("t_nom"), device_ids=tuple(positions),
        issued_round=0, issued_at=0.0,
    ))
    for did, pose in positions.items():
        planner.device_states[did].last_known_position = pose  # type: ignore[assignment]
    model = dataclasses.replace(feasibility_model, ferry=ferry.bind(planner.device_states))
    dock = model.ferry.dock
    pass_1 = planner.build_contact_queue(rf_range_m=rf_range_m, mule_pose=dock)
    pass_2 = planner.build_pass_2_queue(rf_range_m=rf_range_m, mule_pose=dock)
    t_1 = model.fold(pass_1, FlightState(dock, 0.0), rule=RULE_NONE, budget_end=None,
                     pass_kind=MissionPass.COLLECT, skip=False).home
    t_2 = model.fold(pass_2, FlightState(dock, 0.0), rule=RULE_NONE, budget_end=None,
                     pass_kind=MissionPass.DELIVER, skip=False).home
    return t_1 + turn + t_2


def median_nominal_mission_period_s(
    layouts: Iterable[Mapping[DeviceID, Sequence[float]]],
    *,
    rf_range_m: float,
    feasibility_model,
    turnaround_s: float,
) -> float:
    """T_nom: the median of :func:`nominal_mission_period_s` over a cell's layouts.

    One value per cell, the same for every arm (spec Q1). Deterministic: the
    layouts come from the cell's seeds, and the period is a pure function of
    each layout.
    """
    import statistics  # noqa: WPS433

    periods = [
        nominal_mission_period_s(
            layout, rf_range_m=rf_range_m, feasibility_model=feasibility_model,
            turnaround_s=turnaround_s,
        )
        for layout in layouts
    ]
    if not periods:
        raise FLSchedulerError("median_nominal_mission_period_s needs at least one layout")
    return float(statistics.median(periods))
