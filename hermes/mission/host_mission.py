"""``HFLHostMission`` — mule-side FL server for one mission round.

Responsibilities per design §6.3:

* Open FL solicitation on the RF link, collect FL_READY_ADV replies.
* Push θ_disc + synth batch to eligible devices.
* Verify gradient receipts (checksum, byte count, mission_round, TTL).
* Maintain a partial FedAvg accumulator over the mission round.
* Emit one ``RoundCloseDelta`` per device onto the intra-NUC scheduler bus.
* Write the authoritative ``MissionRoundCloseReport`` shipped at dock.
* Hold a TTL-bounded device busy-flag for cross-mule race arbitration.

Flower is deliberately *not* imported here — the design keeps the mule
FL plumbing behind the ``RFLink`` ABC so Phase 2 can run under a pure
loopback. The real Flower wiring arrives in Phase 6 as an adapter.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from hermes.transport import RFLink, RFLinkError
from hermes.types import (
    ContactHistory,
    ContactRecord,
    DeliveryAck,
    DeliveryOutcome,
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    GradientSubmission,
    MissionDeliveryLine,
    MissionDeliveryReport,
    MissionOutcome,
    MissionPass,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MuleID,
    PartialAggregate,
    RoundCloseDelta,
    Weights,
    weights_byte_count,
    weights_signature,
)

from .partial_fedavg import PartialFedAvgError, partial_fedavg

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Small value types (program-local)
# --------------------------------------------------------------------------- #

class MissionSessionError(RuntimeError):
    """Raised when an FL session must be aborted (bad gradient, TTL, etc.)."""


@dataclass
class _BusyFlag:
    """TTL-bounded busy-flag for one device.

    Cross-mule race arbitration: if another mule asks and the flag is
    still live, this device is being handled here; back off.
    """

    until_ts: float

    def is_live(self, now: float) -> bool:
        return now < self.until_ts


# Callable emitted into the intra-NUC scheduler bus; in tests this is a
# list-append lambda, in Phase 4+ it is a real pub/sub handle.
SchedulerBus = Callable[[RoundCloseDelta], None]


# --------------------------------------------------------------------------- #
# HFLHostMission
# --------------------------------------------------------------------------- #

class HFLHostMission:
    """Mule-side FL server. One instance per mule, one round at a time."""

    def __init__(
        self,
        *,
        mule_id: MuleID,
        rf: RFLink,
        scheduler_bus: Optional[SchedulerBus] = None,
        session_ttl_s: float = 30.0,
        busy_ttl_s: float = 45.0,
    ) -> None:
        self.mule_id = mule_id
        self.rf = rf
        self.scheduler_bus = scheduler_bus or (lambda _delta: None)
        self.session_ttl_s = session_ttl_s
        self.busy_ttl_s = busy_ttl_s

        self._lock = threading.RLock()
        self._mission_round: int = 0

        # Round-scoped state (reset between rounds)
        self._current_theta: Optional[Weights] = None
        self._accepted: List[GradientSubmission] = []
        self._report: Optional[MissionRoundCloseReport] = None
        self._contacts: Optional[ContactHistory] = None
        self._round_started_at: float = 0.0

        # Sprint 1.5 — two-pass mission state.
        self._current_pass: MissionPass = MissionPass.COLLECT
        self._delivery_report: Optional[MissionDeliveryReport] = None
        # Sprint 1.5 H1 — internal stash for FL_READY_ADV replies that
        # arrive from devices outside the *current* contact event. The
        # next run_contact / deliver_contact picks them up first before
        # blocking on the link's recv_ready_adv. Replaces the previous
        # rf.send_ready_adv re-queue path which crashed on TCP (the
        # mule-side TCP server doesn't implement the device→mule
        # direction).
        self._misrouted_advs: List[FLReadyAdv] = []

        # Cross-round state
        self._busy: Dict[DeviceID, _BusyFlag] = {}

    # -------------------------------------------------------------- round API

    def open_round(self, theta_disc: Weights) -> int:
        """Start a new mission round with a fresh copy of θ_disc.

        Returns the new ``mission_round`` integer. Pass-1 (COLLECT) is
        the default mode — call :meth:`open_pass_2` to switch to
        DELIVER mode after the inter-pass dock.
        """
        with self._lock:
            self._mission_round += 1
            self._current_theta = [w.copy() for w in theta_disc]
            self._accepted = []
            self._round_started_at = time.time()
            self._report = MissionRoundCloseReport(
                mule_id=self.mule_id,
                mission_round=self._mission_round,
                started_at=self._round_started_at,
                finished_at=0.0,
            )
            self._contacts = ContactHistory(
                mule_id=self.mule_id,
                mission_round=self._mission_round,
            )
            self._current_pass = MissionPass.COLLECT
            self._delivery_report = None
            log.info(
                "open_round mule=%s round=%d theta_layers=%d bytes=%d pass=%s",
                self.mule_id,
                self._mission_round,
                len(self._current_theta),
                weights_byte_count(self._current_theta),
                self._current_pass.value,
            )
            return self._mission_round

    def close_round(
        self,
    ) -> Tuple[PartialAggregate, MissionRoundCloseReport, ContactHistory]:
        """End the mission round: run partial FedAvg, finalize report.

        Raises if no gradients were accepted (all timeouts / all partial).
        """
        with self._lock:
            if self._report is None or self._contacts is None:
                raise MissionSessionError("close_round called before open_round")
            self._report.finished_at = time.time()

            try:
                aggregate = partial_fedavg(
                    mule_id=self.mule_id,
                    mission_round=self._mission_round,
                    submissions=self._accepted,
                )
            except PartialFedAvgError as e:
                log.warning(
                    "close_round mule=%s round=%d: partial_fedavg failed: %s",
                    self.mule_id,
                    self._mission_round,
                    e,
                )
                raise MissionSessionError(str(e)) from e

            report = self._report
            contacts = self._contacts

            log.info(
                "close_round mule=%s round=%d on_time=%d missed=%d",
                self.mule_id,
                self._mission_round,
                *report.counts(),
            )

            # Leave _mission_round as-is so it monotonically increases.
            self._current_theta = None
            self._accepted = []
            self._report = None
            self._contacts = None

            return aggregate, report, contacts

    # ---------------------------------------------- per-device session driver

    def run_session(
        self,
        synth_batch,
        *,
        min_utility: float = 0.0,
    ) -> Optional[MissionOutcome]:
        """Run one end-to-end FL session with whichever device answers next.

        Returns the outcome tag, or ``None`` if no device answered before
        the RF recv timeout.

        This is the single-threaded happy-path driver used by the Phase 2
        demo + tests. A real Phase 6 mule would fan out sessions across a
        thread pool but keep the same per-session contract.
        """
        with self._lock:
            self._require_open_round()
            theta = [w.copy() for w in self._current_theta]  # type: ignore[arg-type]
            mission_round = self._mission_round

        # Solicit + wait for a reply (outside the lock so long blocks don't stall
        # cross-thread inspection of busy flags etc.)
        solicit = FLOpenSolicit(
            mule_id=self.mule_id,
            mission_round=mission_round,
            issued_at=time.time(),
        )
        try:
            self.rf.broadcast_open_solicit(solicit)
            adv = self.rf.recv_ready_adv(timeout=self.session_ttl_s)
        except RFLinkError:
            log.debug("run_session: no device answered in %.1fs", self.session_ttl_s)
            return None

        # S2B gate on arrival (scheduler ran S2B pre-contact; the mule
        # re-checks here because remote state is never trusted blind).
        if not adv.is_eligible() or adv.utility < min_utility:
            self._record_contact(adv, in_session=False)
            self._record_outcome(
                device_id=adv.device_id,
                outcome=MissionOutcome.PARTIAL,
                contact_ts=time.time(),
                utility=adv.utility,
                bytes_received=0,
                bytes_sent=0,
            )
            log.info(
                "session refused device=%s state=%s utility=%.3f",
                adv.device_id, adv.state.value, adv.utility,
            )
            return MissionOutcome.PARTIAL

        # Claim busy slot for cross-mule arbitration
        self._claim_busy(adv.device_id)

        # Push θ_disc + synth batch
        push = DiscPush(
            mule_id=self.mule_id,
            mission_round=mission_round,
            theta_disc=theta,
            synth_batch=synth_batch,
        )
        self.rf.push_disc(adv.device_id, push)

        # Await gradient
        try:
            grad = self.rf.recv_gradient(
                adv.device_id, timeout=self.session_ttl_s
            )
        except RFLinkError:
            outcome = MissionOutcome.TIMEOUT
            self._record_contact(adv, in_session=True)
            self._record_outcome(
                device_id=adv.device_id,
                outcome=outcome,
                contact_ts=time.time(),
                utility=adv.utility,
                bytes_received=0,
                bytes_sent=push_byte_count(push),
            )
            self._release_busy(adv.device_id)
            log.warning(
                "session TIMEOUT device=%s round=%d", adv.device_id, mission_round
            )
            return outcome

        # Verify receipt
        outcome = self._verify_receipt(grad, mission_round)
        if outcome is MissionOutcome.CLEAN:
            with self._lock:
                self._accepted.append(grad)

        self._record_contact(adv, in_session=True)
        self._record_outcome(
            device_id=adv.device_id,
            outcome=outcome,
            contact_ts=grad.submitted_at,
            utility=adv.utility,
            bytes_received=grad.byte_count,
            bytes_sent=push_byte_count(push),
        )
        self._release_busy(adv.device_id)
        return outcome

    # ---------------------------------------------- Sprint 1.5: two-pass API
    #
    # ``run_contact`` is the Pass-1 successor to ``run_session``: instead
    # of accepting whichever device replies first, it serves a *target
    # set* of in-range devices in parallel — one contact event covers
    # N≥1 devices. ``open_pass_2`` flips the mode flag, swaps in the
    # cluster-aggregated θ', and starts the Pass-2 ledger.
    # ``deliver_contact`` is Pass-2's push-only counterpart: no Δθ is
    # requested, just a DeliveryAck per device.
    # ``close_pass_2`` finalises the delivery report.
    #
    # Pass-1 aggregation still flows through the legacy ``_accepted``
    # list — ``close_round`` continues to do partial-FedAvg over it.
    # The "per-contact merge" (design §7 principle 15) is a logical
    # structure: each ``run_contact`` invocation appends N deltas to
    # ``_accepted`` and ``close_round`` does one weighted merge at the
    # end, which is mathematically equivalent to fold-as-you-go because
    # weighted averaging is associative. Memory-efficient streaming
    # merge can replace this in Sprint 2 without an API change.

    @property
    def current_pass(self) -> MissionPass:
        with self._lock:
            return self._current_pass

    def run_contact(
        self,
        contact_devices: Sequence[DeviceID],
        synth_batch,
        *,
        min_utility: float = 0.0,
    ) -> Dict[DeviceID, MissionOutcome]:
        """Pass-1 parallel exchange-only sessions for one contact event.

        Sprint 1.5 design §7 principle 15: at one stop, every device in
        ``contact_devices`` is served in parallel — broadcast solicit,
        gather replies, push θ + synth in parallel threads, collect Δθ
        in parallel. The N=1 case (a one-device contact) collapses to
        the same flow as ``run_session`` for that device.

        Returns a per-device outcome map. Failure modes (TTL, bad
        receipt, FL_READY=False) are recorded individually so other
        in-range devices in the same contact remain unaffected.
        """
        if self._current_pass is not MissionPass.COLLECT:
            raise MissionSessionError(
                f"run_contact called in pass={self._current_pass.value}; "
                f"call open_round (or open_pass_2 first if intentional)"
            )
        if not contact_devices:
            raise ValueError("run_contact requires at least one device")

        with self._lock:
            self._require_open_round()
            theta = [w.copy() for w in self._current_theta]  # type: ignore[arg-type]
            mission_round = self._mission_round

        # 1. Broadcast solicit (Pass 1).
        solicit = FLOpenSolicit(
            mule_id=self.mule_id,
            mission_round=mission_round,
            issued_at=time.time(),
            pass_kind=MissionPass.COLLECT,
        )
        try:
            self.rf.broadcast_open_solicit(solicit)
        except RFLinkError as e:
            log.warning("run_contact: broadcast failed: %s", e)
            return {did: MissionOutcome.TIMEOUT for did in contact_devices}

        # 2. Gather replies. Drain the internal misrouted stash first
        #    (replies that arrived during a prior contact and weren't
        #    consumed), then pop fresh FLReadyAdv off the link until every
        #    expected device is accounted for OR the per-contact deadline
        #    elapses. Replies from outside this contact's expected set
        #    go back into the stash for the next contact's drain.
        expected: Dict[DeviceID, FLReadyAdv] = {}
        deadline = time.time() + self.session_ttl_s
        expected_set = set(contact_devices)

        leftover: List[FLReadyAdv] = []
        with self._lock:
            for adv in self._misrouted_advs:
                if adv.device_id in expected_set and adv.device_id not in expected:
                    expected[adv.device_id] = adv
                else:
                    leftover.append(adv)
            self._misrouted_advs = leftover

        while expected_set - expected.keys():
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                adv = self.rf.recv_ready_adv(timeout=remaining)
            except RFLinkError:
                break
            if adv.device_id in expected_set and adv.device_id not in expected:
                expected[adv.device_id] = adv
            else:
                # Reply from outside this contact — stash for the next
                # contact's drain. No direct re-queue on the link
                # (TCP servers don't implement device→mule send).
                with self._lock:
                    self._misrouted_advs.append(adv)

        # 3. Per-device session in parallel — push, recv, verify.
        outcomes: Dict[DeviceID, MissionOutcome] = {}
        outcomes_lock = threading.Lock()

        def _device_worker(did: DeviceID, adv: Optional[FLReadyAdv]) -> None:
            if adv is None:
                with outcomes_lock:
                    outcomes[did] = MissionOutcome.TIMEOUT
                self._record_outcome(
                    device_id=did,
                    outcome=MissionOutcome.TIMEOUT,
                    contact_ts=time.time(),
                    utility=0.0,
                    bytes_received=0,
                    bytes_sent=0,
                )
                return

            # S2B gate on arrival.
            if not adv.is_eligible() or adv.utility < min_utility:
                self._record_contact(adv, in_session=False)
                self._record_outcome(
                    device_id=adv.device_id,
                    outcome=MissionOutcome.PARTIAL,
                    contact_ts=time.time(),
                    utility=adv.utility,
                    bytes_received=0,
                    bytes_sent=0,
                )
                with outcomes_lock:
                    outcomes[did] = MissionOutcome.PARTIAL
                return

            self._claim_busy(adv.device_id)

            push = DiscPush(
                mule_id=self.mule_id,
                mission_round=mission_round,
                theta_disc=theta,
                synth_batch=synth_batch,
                pass_kind=MissionPass.COLLECT,
            )
            try:
                self.rf.push_disc(adv.device_id, push)
            except RFLinkError as e:
                log.warning("run_contact push failed device=%s: %s", did, e)
                self._record_outcome(
                    device_id=did,
                    outcome=MissionOutcome.TIMEOUT,
                    contact_ts=time.time(),
                    utility=adv.utility,
                    bytes_received=0,
                    bytes_sent=0,
                )
                self._release_busy(did)
                with outcomes_lock:
                    outcomes[did] = MissionOutcome.TIMEOUT
                return

            try:
                grad = self.rf.recv_gradient(
                    adv.device_id, timeout=self.session_ttl_s
                )
            except RFLinkError:
                self._record_contact(adv, in_session=True)
                self._record_outcome(
                    device_id=adv.device_id,
                    outcome=MissionOutcome.TIMEOUT,
                    contact_ts=time.time(),
                    utility=adv.utility,
                    bytes_received=0,
                    bytes_sent=push_byte_count(push),
                )
                self._release_busy(adv.device_id)
                with outcomes_lock:
                    outcomes[did] = MissionOutcome.TIMEOUT
                return

            outcome = self._verify_receipt(grad, mission_round)
            if outcome is MissionOutcome.CLEAN:
                with self._lock:
                    self._accepted.append(grad)

            self._record_contact(adv, in_session=True)
            self._record_outcome(
                device_id=adv.device_id,
                outcome=outcome,
                contact_ts=grad.submitted_at,
                utility=adv.utility,
                bytes_received=grad.byte_count,
                bytes_sent=push_byte_count(push),
            )
            self._release_busy(adv.device_id)
            with outcomes_lock:
                outcomes[did] = outcome

        threads: List[threading.Thread] = []
        for did in contact_devices:
            adv = expected.get(did)
            t = threading.Thread(target=_device_worker, args=(did, adv), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=self.session_ttl_s * 2.0)

        return outcomes

    def open_pass_2(self, theta_disc_new: Weights) -> None:
        """Switch the mission to Pass-2 DELIVER mode.

        Called between Pass 1's UP and Pass 2's outbound flight (after
        ``close_round`` has finalised Pass 1 and the cluster has
        aggregated + dispatched fresh θ'). Stages the new θ in the
        mission server and initializes the Pass-2 delivery report.

        Pre-condition: ``open_round`` has been called at some point in
        this mission (i.e. ``mission_round > 0``). Pass 1's per-pass
        state (``_current_theta``, ``_report``, ``_contacts``) has
        legitimately been cleared by ``close_round``, so we don't
        require those — we just need a valid round counter to tag the
        Pass-2 ledger with.
        """
        with self._lock:
            if self._mission_round <= 0:
                raise MissionSessionError(
                    "open_pass_2 called before any open_round; "
                    "call open_round first to establish a mission_round."
                )
            self._current_pass = MissionPass.DELIVER
            self._current_theta = [w.copy() for w in theta_disc_new]
            self._delivery_report = MissionDeliveryReport(
                mule_id=self.mule_id,
                mission_round=self._mission_round,
                started_at=time.time(),
                finished_at=0.0,
            )
            log.info(
                "open_pass_2 mule=%s round=%d theta_layers=%d",
                self.mule_id,
                self._mission_round,
                len(theta_disc_new),
            )

    def deliver_contact(
        self,
        contact_devices: Sequence[DeviceID],
        synth_batch,
    ) -> Dict[DeviceID, DeliveryOutcome]:
        """Pass-2 push-only delivery for one contact event.

        Pushes the staged θ' + ``synth_batch`` to every device in
        ``contact_devices`` in parallel. Each device is expected to ack
        receipt with a ``DeliveryAck``; absence of an ack within TTL
        marks the device as UNDELIVERED. The cluster bumps
        ``DeviceRecord.delivery_priority`` for undelivered rows in the
        next slice, so they get pulled toward cluster anchors.
        """
        if self._current_pass is not MissionPass.DELIVER:
            raise MissionSessionError(
                f"deliver_contact called in pass={self._current_pass.value}; "
                f"call open_pass_2 first"
            )
        if not contact_devices:
            raise ValueError("deliver_contact requires at least one device")

        with self._lock:
            # Pass-2 doesn't need Pass-1's report/contacts (those were
            # legitimately cleared by close_round). It only needs the
            # current θ' (set by open_pass_2) and the mission_round.
            if self._current_theta is None or self._mission_round <= 0:
                raise MissionSessionError(
                    "deliver_contact called without open_pass_2 staging θ'"
                )
            theta = [w.copy() for w in self._current_theta]
            mission_round = self._mission_round

        solicit = FLOpenSolicit(
            mule_id=self.mule_id,
            mission_round=mission_round,
            issued_at=time.time(),
            pass_kind=MissionPass.DELIVER,
        )
        try:
            self.rf.broadcast_open_solicit(solicit)
        except RFLinkError as e:
            log.warning("deliver_contact: broadcast failed: %s", e)
            return {did: DeliveryOutcome.UNDELIVERED for did in contact_devices}

        # Gather Pass-2 FL_READY_ADV replies — same misrouted-stash
        # discipline as run_contact (see H1 fix).
        expected: Dict[DeviceID, FLReadyAdv] = {}
        deadline = time.time() + self.session_ttl_s
        expected_set = set(contact_devices)

        leftover: List[FLReadyAdv] = []
        with self._lock:
            for adv in self._misrouted_advs:
                if adv.device_id in expected_set and adv.device_id not in expected:
                    expected[adv.device_id] = adv
                else:
                    leftover.append(adv)
            self._misrouted_advs = leftover

        while expected_set - expected.keys():
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                adv = self.rf.recv_ready_adv(timeout=remaining)
            except RFLinkError:
                break
            if adv.device_id in expected_set and adv.device_id not in expected:
                expected[adv.device_id] = adv
            else:
                with self._lock:
                    self._misrouted_advs.append(adv)

        outcomes: Dict[DeviceID, DeliveryOutcome] = {}
        outcomes_lock = threading.Lock()

        def _device_worker(did: DeviceID, adv: Optional[FLReadyAdv]) -> None:
            if adv is None:
                self._record_delivery_line(
                    device_id=did,
                    outcome=DeliveryOutcome.UNDELIVERED,
                    contact_ts=time.time(),
                )
                with outcomes_lock:
                    outcomes[did] = DeliveryOutcome.UNDELIVERED
                return

            push = DiscPush(
                mule_id=self.mule_id,
                mission_round=mission_round,
                theta_disc=theta,
                synth_batch=synth_batch,
                pass_kind=MissionPass.DELIVER,
            )
            try:
                self.rf.push_disc(adv.device_id, push)
            except RFLinkError as e:
                log.warning("deliver_contact push failed device=%s: %s", did, e)
                self._record_delivery_line(
                    device_id=did,
                    outcome=DeliveryOutcome.UNDELIVERED,
                    contact_ts=time.time(),
                )
                with outcomes_lock:
                    outcomes[did] = DeliveryOutcome.UNDELIVERED
                return

            try:
                ack = self.rf.recv_delivery_ack(
                    adv.device_id, timeout=self.session_ttl_s
                )
            except RFLinkError:
                self._record_delivery_line(
                    device_id=did,
                    outcome=DeliveryOutcome.UNDELIVERED,
                    contact_ts=time.time(),
                )
                with outcomes_lock:
                    outcomes[did] = DeliveryOutcome.UNDELIVERED
                return

            self._record_delivery_line(
                device_id=did,
                outcome=DeliveryOutcome.DELIVERED,
                contact_ts=ack.received_at,
            )
            with outcomes_lock:
                outcomes[did] = DeliveryOutcome.DELIVERED

        threads: List[threading.Thread] = []
        for did in contact_devices:
            adv = expected.get(did)
            t = threading.Thread(target=_device_worker, args=(did, adv), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=self.session_ttl_s * 2.0)

        return outcomes

    def close_pass_2(self) -> MissionDeliveryReport:
        """Finalise and return the Pass-2 delivery report.

        Companion to :meth:`close_round` for Pass 1. The supervisor
        ships this report up at the post-Pass-2 dock so the cluster can
        bump ``DeviceRecord.delivery_priority`` on undelivered rows.
        """
        with self._lock:
            if self._delivery_report is None:
                raise MissionSessionError(
                    "close_pass_2 called without open_pass_2 — no Pass-2 ledger"
                )
            self._delivery_report.finished_at = time.time()
            report = self._delivery_report
            self._delivery_report = None
            log.info(
                "close_pass_2 mule=%s round=%d delivered=%d undelivered=%d",
                self.mule_id, self._mission_round, *report.counts(),
            )
            return report

    def _record_delivery_line(
        self,
        *,
        device_id: DeviceID,
        outcome: DeliveryOutcome,
        contact_ts: float,
    ) -> None:
        with self._lock:
            if self._delivery_report is None:
                return
            self._delivery_report.append(
                MissionDeliveryLine(
                    device_id=device_id,
                    outcome=outcome,
                    contact_ts=contact_ts,
                )
            )

    # ---------------------------------------------- busy-flag API (§6.3)

    def is_busy(self, device_id: DeviceID) -> bool:
        """True iff another mule should back off on this device right now."""
        now = time.time()
        with self._lock:
            # lazy sweep
            self._busy = {d: b for d, b in self._busy.items() if b.is_live(now)}
            flag = self._busy.get(device_id)
            return bool(flag and flag.is_live(now))

    # ---------------------------------------------- introspection

    @property
    def mission_round(self) -> int:
        with self._lock:
            return self._mission_round

    def accepted_count(self) -> int:
        with self._lock:
            return len(self._accepted)

    # ---------------------------------------------- internal

    def _require_open_round(self) -> None:
        if self._report is None or self._current_theta is None:
            raise MissionSessionError("no mission round is open; call open_round first")

    def _verify_receipt(
        self, grad: GradientSubmission, mission_round: int
    ) -> MissionOutcome:
        """Three-way verifier: round / byte_count / checksum. TTL done upstream."""
        if grad.mission_round != mission_round:
            log.warning(
                "verify: mission_round mismatch device=%s got=%d expected=%d",
                grad.device_id, grad.mission_round, mission_round,
            )
            return MissionOutcome.PARTIAL

        expected_bytes = weights_byte_count(grad.delta_theta)
        if grad.byte_count != expected_bytes:
            log.warning(
                "verify: byte_count mismatch device=%s got=%d expected=%d",
                grad.device_id, grad.byte_count, expected_bytes,
            )
            return MissionOutcome.PARTIAL

        expected_sig = weights_signature(grad.delta_theta)
        if grad.checksum != expected_sig:
            log.warning(
                "verify: checksum mismatch device=%s", grad.device_id
            )
            return MissionOutcome.PARTIAL

        # TTL: fudge factor of 2x session_ttl for clock skew tolerance.
        if time.time() - grad.submitted_at > 2 * self.session_ttl_s:
            log.warning(
                "verify: TTL expired device=%s age=%.1fs",
                grad.device_id, time.time() - grad.submitted_at,
            )
            return MissionOutcome.PARTIAL

        return MissionOutcome.CLEAN

    def _record_outcome(
        self,
        *,
        device_id: DeviceID,
        outcome: MissionOutcome,
        contact_ts: float,
        utility: float,
        bytes_received: int,
        bytes_sent: int,
    ) -> None:
        with self._lock:
            if self._report is None:
                return
            self._report.append(
                MissionRoundCloseLine(
                    device_id=device_id,
                    outcome=outcome,
                    contact_ts=contact_ts,
                    bytes_received=bytes_received,
                    bytes_sent=bytes_sent,
                )
            )
            mission_round = self._mission_round

        # Fast-phase fan-out (outside the lock so a slow subscriber can't
        # stall the session thread).
        try:
            self.scheduler_bus(
                RoundCloseDelta(
                    device_id=device_id,
                    mule_id=self.mule_id,
                    mission_round=mission_round,
                    outcome=outcome,
                    utility=utility,
                    contact_ts=contact_ts,
                )
            )
        except Exception:  # pragma: no cover — bus faults must not kill the mule
            log.exception("scheduler_bus raised; dropping delta")

    def _record_contact(self, adv: FLReadyAdv, *, in_session: bool) -> None:
        with self._lock:
            if self._contacts is None:
                return
            self._contacts.add(
                ContactRecord(
                    device_id=adv.device_id,
                    contact_ts=adv.issued_at or time.time(),
                    in_session=in_session,
                )
            )

    def _claim_busy(self, device_id: DeviceID) -> None:
        with self._lock:
            self._busy[device_id] = _BusyFlag(until_ts=time.time() + self.busy_ttl_s)

    def _release_busy(self, device_id: DeviceID) -> None:
        with self._lock:
            self._busy.pop(device_id, None)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def push_byte_count(push: DiscPush) -> int:
    """Tally bytes shipped to a device for one session (for the round report)."""
    return weights_byte_count(push.theta_disc) + sum(
        int(a.nbytes) for a in push.synth_batch
    )
