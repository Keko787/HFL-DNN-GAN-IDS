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
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hermes.l1.channel_ddqn import ChannelDDQN, L1_STATE_DIM
from hermes.mission import HFLHostMission, MissionSessionError
from hermes.mule import BundleDistributor, ClientCluster
from hermes.scheduler import FLScheduler
from hermes.transport import DockLink, RFLink
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    ContactWaypoint,
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


class MuleSupervisorError(RuntimeError):
    """Raised when the supervisor's invariants are violated."""


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


class MuleSupervisor:
    """Per-mule supervisor: one instance per mobile AVN.

    The supervisor doesn't own the cluster — it speaks to ``DockLink``.
    In Sprint 1 the cluster runs in-process and shares the link; in
    Sprint 2 they're separate processes and the link is a TCP socket.
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
        now_fn=time.time,
    ) -> None:
        self.mule_id = mule_id
        self.mule_pose = mule_pose
        self.mule_energy = mule_energy
        self.rf_prior_snr_db = rf_prior_snr_db
        self.rf_range_m = rf_range_m
        self._now = now_fn

        # Scheduler — slow-phase amendments + fast-phase round-close deltas
        # are wired through here.
        self.scheduler = FLScheduler(
            beacon_window_s=beacon_window_s,
            now_fn=now_fn,
            target_selector=target_selector,
        )

        # Mission server — emits one RoundCloseDelta per session into the
        # scheduler bus, which folds it into Deadline(j) immediately.
        self.mission = HFLHostMission(
            mule_id=mule_id,
            rf=rf,
            scheduler_bus=self.scheduler.ingest_round_close_delta,
            session_ttl_s=session_ttl_s,
        )

        # ClientCluster owns the dock lifecycle. The distributor fans the
        # DOWN bundle out to: scheduler (slice + amendment) and supervisor
        # (theta + synth, stashed for the next open_round).
        self._next_theta: Optional[Weights] = None
        self._next_synth = None
        # H3 — Pass-2 delivery report from the most recent mission, held
        # locally until the next mission's Pass-1 dock UPs it as
        # ``UpBundle.prev_mission_delivery_report``.
        self._pending_delivery_report: Optional[MissionDeliveryReport] = None
        self.distributor = BundleDistributor(
            on_slice_and_amendment=self._on_slice_and_amendment,
            on_next_round_model=self._on_next_round_model,
        )
        self.client_cluster = ClientCluster(
            mule_id=mule_id,
            dock=dock,
            distributor=self.distributor,
        )

        # L1 channel actor — optional. When None, channel choice is not
        # logged at all (no actuation in loopback either way).
        self.channel_actor = channel_actor

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
        self.scheduler.ingest_slice(mission_slice, amendment=amendment)

    def _on_next_round_model(self, theta: Weights, synth_batch) -> None:
        self._next_theta = theta
        self._next_synth = synth_batch

    # ------------------------------------------------------------------ #
    # Mission cycle
    # ------------------------------------------------------------------ #

    def wait_for_initial_dock(self, timeout: Optional[float] = None) -> bool:
        """Block until the first DOWN bundle arrives + is distributed.

        Required before the first ``run_one_mission`` call so the mule
        knows what slice it owns. Uses the DOWN-only bootstrap path
        because the mule has no aggregate to upload yet.
        """
        if not self.client_cluster.wait_for_dock(timeout=timeout):
            return False
        down = self.client_cluster.bootstrap_down_only()
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

        if self.rf_range_m is not None:
            return self._run_two_pass_mission()
        return self._run_single_pass_mission()

    def _run_single_pass_mission(self) -> MissionRunResult:
        """Legacy Sprint-1A path: per-device queue + run_session FCFS."""

        theta = self._next_theta
        synth = self._next_synth
        # Consumed — the next dock cycle restages.
        self._next_theta = None
        self._next_synth = None

        # 1. Open round on the mission server.
        mission_round = self.mission.open_round(theta)

        # 2. Build the visit queue from the scheduler.
        queue = self.scheduler.build_target_queue(
            mule_pose=self.mule_pose,
            mule_energy=self.mule_energy,
            rf_prior_snr_db=self.rf_prior_snr_db,
        )
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
            agg, report, contacts = self.mission.close_round()
        except MissionSessionError as e:
            # No clean gradients — abort the dock cycle and let the
            # caller decide whether to skip the UP.
            log.error(
                "mule=%s round=%d close_round failed: %s",
                self.mule_id, mission_round, e,
            )
            raise

        # 5. Hand off to ClientCluster, dock cycle (UP + DOWN).
        self.client_cluster.collect(
            partial_aggregate=agg, report=report, contacts=contacts,
        )
        if not self.client_cluster.wait_for_dock(timeout=None):
            raise MuleSupervisorError("dock did not become available")
        # run_dock_cycle distributes DOWN through the BundleDistributor,
        # which restages _next_theta + _next_synth via the callbacks.
        self.client_cluster.run_dock_cycle()

        return MissionRunResult(
            mission_round=mission_round,
            queue=list(queue),
            aggregate=agg,
            report=report,
            contacts=contacts,
            channel_choices=channel_choices,
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
        self._next_theta = None
        self._next_synth = None

        # ============================ Pass 1 ============================
        mission_round = self.mission.open_round(theta_pass_1)
        pass_1_queue = self.scheduler.build_contact_queue(
            rf_range_m=rf_range_m,
            mule_pose=self.mule_pose,
            mule_energy=self.mule_energy,
            rf_prior_snr_db=self.rf_prior_snr_db,
        )
        log.info(
            "mule=%s round=%d pass=1 contacts=%d devices_total=%d",
            self.mule_id, mission_round, len(pass_1_queue),
            sum(len(c.devices) for c in pass_1_queue),
        )

        # M5 — split channel choices per pass for clean attribution.
        pass_1_channel_choices: List[int] = []
        for wp in pass_1_queue:
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

        try:
            agg, report, contacts = self.mission.close_round()
        except MissionSessionError as e:
            log.error(
                "mule=%s round=%d Pass 1 close_round failed: %s",
                self.mule_id, mission_round, e,
            )
            raise

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
        self.client_cluster.run_dock_cycle()
        # The DOWN-bundle distribution staged self._next_theta /
        # self._next_synth — those are Pass-2's payload.
        if self._next_theta is None:
            raise MuleSupervisorError(
                "inter-pass dock did not stage a Pass-2 model — cluster "
                "must dispatch a fresh θ' after ingesting Pass-1's UP"
            )
        theta_pass_2 = self._next_theta
        synth_pass_2 = self._next_synth
        # We deliberately DO NOT clear _next_theta here; Pass 2 itself
        # re-uses the same θ as the basis for next mission's training,
        # so the next run_one_mission call will see it staged. This
        # matches the design's "Pass 2's θ becomes mission n+1's basis"
        # — no extra dock cycle needed at end of mission.

        # ============================ Pass 2 ============================
        self.mission.open_pass_2(theta_pass_2)
        # H5 — Pass-2 plans from the mule's *current* pose (advanced by
        # Pass-1 contacts above), not from the constructor-time origin.
        pass_2_queue = self.scheduler.build_pass_2_queue(
            rf_range_m=rf_range_m,
            mule_pose=self.mule_pose,
        )
        log.info(
            "mule=%s round=%d pass=2 contacts=%d devices_total=%d",
            self.mule_id, mission_round, len(pass_2_queue),
            sum(len(c.devices) for c in pass_2_queue),
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
        )

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
