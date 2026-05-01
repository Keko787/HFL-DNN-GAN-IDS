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

import logging
import time
from typing import Dict, Iterable, List, Optional, Tuple

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
    RoundCloseDelta,
    TargetWaypoint,
)

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
    ):
        self._device_states: Dict[DeviceID, DeviceSchedulerState] = {}
        self._current_slice: Optional[MissionSlice] = None
        self._fl_threshold = fl_threshold
        self._beacon_window_s = beacon_window_s
        self._now = now_fn
        # Phase-5 S3.5 — optional learned selector. If None, the
        # deterministic distance placeholder in :func:`select_order`
        # runs.
        self._target_selector = target_selector

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
        """
        self._current_slice = mission_slice
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
                    st = DeviceSchedulerState(
                        device_id=rec.device_id,
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
                self._device_states[did] = DeviceSchedulerState(device_id=did)

        # Refresh slice membership flags.
        for did, st in self._device_states.items():
            st.is_in_slice = did in slice_ids

        # Slow-phase deadline fold.
        if amendment is not None:
            fold_cluster_amendment(self._device_states, amendment)

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
        fold_round_close_delta(st, delta)

    def ingest_beacon(self, obs: BeaconObservation) -> None:
        """Opportunistic RF beacon (design §4 step 16).

        If we've never seen the device, we stand up a minimal state row so
        S1 can admit it. Position is unknown — the selector will treat it
        as infinitely far, which is deliberately conservative.
        """
        st = self._device_states.get(obs.device_id)
        if st is None:
            st = DeviceSchedulerState(device_id=obs.device_id, is_new=True)
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
            deadlines[did] = compute_deadline(st, now=_now)

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
            deadlines[did] = compute_deadline(st, now=_now)

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

        return queue

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
            deadlines[did] = compute_deadline(st, now=_now)
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
