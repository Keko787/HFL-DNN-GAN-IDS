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
    DeviceID,
    DeviceRecord,
    DeviceSchedulerState,
    FLReadyAdv,
    MissionSlice,
    RoundCloseDelta,
    TargetWaypoint,
)

from .stages import (
    classify_bucket,
    compute_deadline,
    filter_eligible,
    fold_cluster_amendment,
    fold_round_close_delta,
    is_on_contact_ready,
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
        if registry_records is not None:
            for rec in registry_records:
                st = self._device_states.get(rec.device_id)
                if st is None:
                    st = DeviceSchedulerState(
                        device_id=rec.device_id,
                        is_new=rec.is_new,
                        last_known_position=rec.last_known_position,
                    )
                    self._device_states[rec.device_id] = st
                else:
                    st.last_known_position = rec.last_known_position

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
