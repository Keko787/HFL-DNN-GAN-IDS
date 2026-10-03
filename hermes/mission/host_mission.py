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

FeRRy Phase 3 (design section 4) gives the two contact routines one body,
``_serve_contact``, with two paths:

* **Legacy** (no contact plan): the afa9526 exchange, byte for byte: broadcast
  solicit, the misrouted-advert stash, workers joined one after the other, and
  every ledger write made by the worker when it happens, stamped with wall time
  (P-01 defects 2 and 3 kept, pinned by ``tests/golden``).
* **Ferry** (``plan=ContactPlan``, on the mission clock): a numbered solicit to
  the stop's reachable members only, stale adverts, gradients and acks
  rejected, workers joined under one deadline, and a commit that writes the
  ledger in device order, stamped in simulated seconds from the contact's
  arrival plus its airtime, and charges the mission clock once.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

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

from .aggregation_rules import AggregationSpec, merge_on_mule, update_age
from .contact_plan import KIND_DWELL, KIND_LISTEN, ContactCommit, ContactPlan
from .partial_fedavg import PartialFedAvgError

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
# One contact routine for both passes (FeRRy Phase 3, finding D-01)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _PassSpec:
    """What differs between ``run_contact`` (Pass 1) and ``deliver_contact`` (Pass 2)."""

    kind: MissionPass
    api: str                  # the public name, used in errors and logs
    wrong_pass_hint: str
    broadcast_failed: Any     # the outcome every device gets when the broadcast fails


_COLLECT = _PassSpec(
    MissionPass.COLLECT, "run_contact",
    "call open_round (or open_pass_2 first if intentional)",
    MissionOutcome.TIMEOUT,
)
_DELIVER = _PassSpec(
    MissionPass.DELIVER, "deliver_contact",
    "call open_pass_2 first",
    DeliveryOutcome.UNDELIVERED,
)


class _LiveSink:
    """Legacy path: every worker write reaches the host the moment it is made.

    As at afa9526, each write reads the host's state when it happens. A worker
    that outlives its join therefore writes into whatever round is open by
    then: its gradient goes to the ``_accepted`` list of that round, read at
    append time (critic A2), and its lines to that round's ledgers. That is
    P-01 defect 3, kept on purpose in legacy mode (``tests/golden``,
    ``late_writer``).
    """

    __slots__ = ("_host",)

    def __init__(self, host: "HFLHostMission") -> None:
        self._host = host

    def outcome(self, **kwargs) -> None:
        self._host._record_outcome(**kwargs)

    def contact(self, adv: FLReadyAdv, *, in_session: bool) -> None:
        self._host._record_contact(adv, in_session=in_session)

    def delivery(self, **kwargs) -> None:
        self._host._record_delivery_line(**kwargs)

    def accept(self, grad: GradientSubmission) -> None:
        # The caller holds the host's lock; the list is looked up now.
        self._host._accepted.append(grad)


# How a ferry worker's session ended. ``REPLIED``: a matching gradient or ack
# came back. ``NO_REPLY``: the push went out and no matching reply came in the
# TTL, or the uplink was dropped by the availability draw. ``REFUSED``: Pass 1's
# S2B gate turned the advert down, nothing was pushed. ``PUSH_FAILED``: the link
# refused the push. ``NOT_READY`` (Exp 5 addendum, Study 5.12): the plan names
# the target as having no update ready on the fit clock; its advert came,
# nothing was pushed.
_REPLIED = "replied"
_NO_REPLY = "no_reply"
_REFUSED = "refused"
_PUSH_FAILED = "push_failed"
_NOT_READY = "not_ready"
# How the commit classes the other members: never solicited (the gate), or
# solicited with no advert answering this contact's solicit.
_UNREACHABLE = "unreachable"
_SILENT = "silent"


@dataclass(frozen=True)
class _FerrySession:
    """One answered target's session, as its worker finished it."""

    kind: str
    push_bytes: int = 0
    grad: Optional[GradientSubmission] = None
    verdict: Optional[MissionOutcome] = None     # the receipt check of ``grad``
    uplink_drop: bool = False


@dataclass(frozen=True)
class _FerryContext:
    """What every worker of one ferry contact shares (read-only)."""

    spec: _PassSpec
    mission_round: int
    solicit_id: int
    theta: Weights
    theta_version: Optional[int]
    synth_batch: Any
    min_utility: float
    drop_uplink: FrozenSet[DeviceID]
    not_ready: FrozenSet[DeviceID] = frozenset()


class _BufferSink:
    """Ferry path: each worker's result is held until the commit.

    A worker records its push (``pushed``) as soon as it is out, and its
    session once (``put``) when it ends. The commit closes the sink at the
    join deadline and works from what it holds, in device order; a worker still
    running then is committed from what it had done, and anything it reports
    later is dropped and logged, so no late write reaches a ledger, the
    accepted list or the returned map (P-01 defect 3, closed in ferry mode).
    The stale-message counters live here because workers update them.
    """

    def __init__(self, api: str) -> None:
        self._api = api
        self._lock = threading.Lock()
        self._closed = False
        self._sessions: Dict[DeviceID, _FerrySession] = {}
        self._pushed: Dict[DeviceID, int] = {}
        self._stale: Dict[str, int] = {"adv": 0, "gradient": 0, "ack": 0}
        self.late_dropped = 0

    def _late(self, did: DeviceID, what: str) -> None:
        self.late_dropped += 1
        log.warning(
            "%s: late %s from device=%s dropped: its contact is already committed",
            self._api, what, did,
        )

    def pushed(self, did: DeviceID, nbytes: int) -> None:
        with self._lock:
            if self._closed:
                self._late(did, "push")
                return
            self._pushed[did] = int(nbytes)

    def put(self, did: DeviceID, session: _FerrySession) -> None:
        with self._lock:
            if self._closed:
                self._late(did, f"{session.kind} session")
                return
            self._sessions[did] = session

    def stale(self, what: str, n: int = 1) -> None:
        with self._lock:
            self._stale[what] = self._stale.get(what, 0) + int(n)

    def close(self) -> Tuple[Dict[DeviceID, _FerrySession], Dict[DeviceID, int], Dict[str, int]]:
        with self._lock:
            self._closed = True
            return dict(self._sessions), dict(self._pushed), dict(self._stale)


#: Upper bound on the items one non-blocking drain takes from one queue: a link
#: whose receive never runs dry must not hang the mule before its solicit.
_DRAIN_LIMIT = 10_000

#: How far, in units in the last place of the contact's end time, the charged
#: clock may land from it through float rounding alone. The two charges are
#: exact differences while the airtime stays below the clock's own reading
#: (Sterbenz); past that each charge rounds twice, by at most one ulp each. A
#: gap wider than this is a clock the charges did not move: a wiring bug.
_CHARGE_ROUNDING_ULPS = 8


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
        aggregation: Optional[AggregationSpec] = None,
        train_ahead: bool = False,
        now_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.mule_id = mule_id
        self.rf = rf
        self.scheduler_bus = scheduler_bus or (lambda _delta: None)
        self.session_ttl_s = session_ttl_s
        self.busy_ttl_s = busy_ttl_s
        # FeRRy Phase 1 — the merge rule decides which form devices answer in
        # (full weights for agg:plain, a delta against their basis otherwise)
        # and how close_round merges.
        self.aggregation: AggregationSpec = aggregation or AggregationSpec()
        # Ask Pass-1 devices to train ahead on the basis they are pushed; the
        # mule sets this when its Pass 2 is budgeted (FeRRy Phase 1).
        self.train_ahead = bool(train_ahead)
        # FeRRy Phase 3 — the mission clock (design section 2.2). None keeps
        # every mission-time stamp on the wall clock, with the same literal
        # time.time() calls in the same order as before. Set, it stamps the
        # round and delivery reports and the skipped deliveries, and every
        # contact must come with a ContactPlan on the same clock, whose commit
        # stamps the sessions (critic A1). Transport timers (TTL gathers and
        # receives, joins, the receipt TTL, busy flags) stay on the wall clock
        # either way (design section 2.4).
        self.now_fn: Optional[Callable[[], float]] = now_fn
        # The ledger of the last round whose merge failed (nothing merged),
        # kept so the caller can still report its sessions.
        self.last_unmerged: Optional[Tuple[MissionRoundCloseReport, ContactHistory]] = None
        # FeRRy Phase 3 — what the last ferry contact charged and stamped, for
        # the supervisor's trace (None until a ferry contact has run).
        self.last_contact: Optional[ContactCommit] = None

        self._lock = threading.RLock()
        self._mission_round: int = 0
        # Numbers the ferry path's targeted solicits (0 = unnumbered, legacy).
        self._solicit_seq: int = 0

        # Round-scoped state (reset between rounds)
        self._current_theta: Optional[Weights] = None
        # Version of ``_current_theta`` (the cluster round that produced it),
        # pushed with it so every update can be aged. None = unknown.
        self._current_theta_version: Optional[int] = None
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

    def open_round(
        self, theta_disc: Weights, *, theta_version: Optional[int] = None,
    ) -> int:
        """Start a new mission round with a fresh copy of θ_disc.

        Returns the new ``mission_round`` integer. Pass-1 (COLLECT) is
        the default mode — call :meth:`open_pass_2` to switch to
        DELIVER mode after the inter-pass dock. ``theta_version`` is the
        cluster round that produced θ_disc; it rides every push.
        """
        with self._lock:
            self._mission_round += 1
            self._current_theta = [w.copy() for w in theta_disc]
            self._current_theta_version = theta_version
            self._accepted = []
            self.last_unmerged = None
            self._round_started_at = time.time() if self.now_fn is None else self.now_fn()
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
        *,
        age_caps: Optional[Mapping[DeviceID, Optional[int]]] = None,
    ) -> Tuple[PartialAggregate, MissionRoundCloseReport, ContactHistory]:
        """End the mission round: run partial FedAvg, finalize report.

        The merge follows ``self.aggregation``: ``agg:plain`` is the original
        num_examples-weighted mean; the age-aware rules merge deltas, weighted
        by age, with ``age_caps`` giving each device's cutoff in cluster rounds.

        Raises if no gradients were accepted (all timeouts / all partial), or
        if the rule gave every accepted update zero weight.
        """
        with self._lock:
            if self._report is None or self._contacts is None:
                raise MissionSessionError("close_round called before open_round")
            self._report.finished_at = time.time() if self.now_fn is None else self.now_fn()

            try:
                aggregate = merge_on_mule(
                    self.aggregation,
                    mule_id=self.mule_id,
                    mission_round=self._mission_round,
                    submissions=self._accepted,
                    base_version=self._current_theta_version,
                    age_caps=age_caps,
                )
            except PartialFedAvgError as e:
                log.warning(
                    "close_round mule=%s round=%d: partial_fedavg failed: %s",
                    self.mule_id,
                    self._mission_round,
                    e,
                )
                # Keep the ledger: under an age cutoff the round can hold
                # on-time sessions whose updates were all past their cutoff.
                self.last_unmerged = (self._report, self._contacts)
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
            theta_version = self._current_theta_version
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
                local_loss=adv.local_loss,
                num_examples=adv.num_examples,
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
            basis_version=theta_version,
            update_form=self.aggregation.update_form,
            train_ahead=self.train_ahead,
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
                local_loss=adv.local_loss,
                num_examples=adv.num_examples,
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
            # Prefer the GRADIENT's values over the advertisement's. The adv was
            # built before this session trained, so its loss describes the
            # PREVIOUS round; the gradient's describes the update we just took.
            # Using the adv here would lag Oort's ranking signal a full mission
            # behind the training it summarises.
            local_loss=(grad.local_loss if grad.local_loss is not None
                        else adv.local_loss),
            num_examples=(grad.num_examples or adv.num_examples),
            bytes_received=grad.byte_count,
            bytes_sent=push_byte_count(push),
            basis_version=grad.basis_version,
            line_num_examples=grad.num_examples,
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
    #
    # FeRRy Phase 3 (finding D-01): both routines are one-line wrappers over
    # ``_serve_contact``, which holds the shared guards, snapshot, solicit,
    # gather and joins once; the per-pass session bodies are
    # ``_collect_session`` and ``_deliver_session``.

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
        plan: Optional[ContactPlan] = None,
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

        FeRRy Phase 3: with ``plan`` (a :class:`ContactPlan` built at arrival
        on the mission clock) the contact runs the ferry path described in
        :meth:`_ferry_contact`, and the map comes back complete, in
        ``contact_devices`` order.
        """
        return self._serve_contact(
            _COLLECT, contact_devices, synth_batch, min_utility=min_utility, plan=plan,
        )

    def open_pass_2(
        self, theta_disc_new: Weights, *, theta_version: Optional[int] = None,
    ) -> None:
        """Switch the mission to Pass-2 DELIVER mode.

        Called between Pass 1's UP and Pass 2's outbound flight (after
        ``close_round`` has finalised Pass 1 and the cluster has
        aggregated + dispatched fresh θ'). Stages the new θ in the
        mission server and initializes the Pass-2 delivery report.
        ``theta_version`` rides each delivery so devices know their basis.

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
            self._current_theta_version = theta_version
            self._delivery_report = MissionDeliveryReport(
                mule_id=self.mule_id,
                mission_round=self._mission_round,
                started_at=time.time() if self.now_fn is None else self.now_fn(),
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
        *,
        plan: Optional[ContactPlan] = None,
    ) -> Dict[DeviceID, DeliveryOutcome]:
        """Pass-2 push-only delivery for one contact event.

        Pushes the staged θ' + ``synth_batch`` to every device in
        ``contact_devices`` in parallel. Each device is expected to ack
        receipt with a ``DeliveryAck``; absence of an ack within TTL
        marks the device as UNDELIVERED. The cluster bumps
        ``DeviceRecord.delivery_priority`` for undelivered rows in the
        next slice, so they get pulled toward cluster anchors.

        FeRRy Phase 3: with ``plan`` the delivery runs the ferry path
        (:meth:`_ferry_contact`); Pass 2 faces the SNR gate but no
        availability draw, so the plan's ``drop_uplink`` must be empty.
        """
        return self._serve_contact(_DELIVER, contact_devices, synth_batch, plan=plan)

    def record_skipped_delivery(self, devices: Sequence[DeviceID]) -> None:
        """Log devices a budgeted Pass 2 will not fly to (FeRRy Phase 1).

        Each gets a SKIPPED line, which the cluster counts as not delivered,
        so the device keeps its older basis and its delivery priority rises.
        """
        now = time.time() if self.now_fn is None else self.now_fn()
        for did in devices:
            self._record_delivery_line(
                device_id=did, outcome=DeliveryOutcome.SKIPPED, contact_ts=now,
            )

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
            self._delivery_report.finished_at = (
                time.time() if self.now_fn is None else self.now_fn()
            )
            report = self._delivery_report
            self._delivery_report = None
            log.info(
                "close_pass_2 mule=%s round=%d delivered=%d undelivered=%d",
                self.mule_id, self._mission_round, *report.counts(),
            )
            return report

    # ---------------------------------------------- the shared contact routine

    def _serve_contact(
        self,
        spec: _PassSpec,
        contact_devices: Sequence[DeviceID],
        synth_batch,
        *,
        min_utility: float = 0.0,
        plan: Optional[ContactPlan] = None,
    ) -> Dict[DeviceID, Any]:
        """Guards and snapshot, then the legacy exchange or the ferry path."""
        if self._current_pass is not spec.kind:
            raise MissionSessionError(
                f"{spec.api} called in pass={self._current_pass.value}; "
                f"{spec.wrong_pass_hint}"
            )
        if not contact_devices:
            raise ValueError(f"{spec.api} requires at least one device")

        with self._lock:
            if spec.kind is MissionPass.COLLECT:
                self._require_open_round()
            elif self._current_theta is None or self._mission_round <= 0:
                # Pass-2 doesn't need Pass-1's report/contacts (those were
                # legitimately cleared by close_round). It only needs the
                # current θ' (set by open_pass_2) and the mission_round.
                raise MissionSessionError(
                    "deliver_contact called without open_pass_2 staging θ'"
                )
            theta = [w.copy() for w in self._current_theta]  # type: ignore[union-attr]
            theta_version = self._current_theta_version
            mission_round = self._mission_round

        if plan is None:
            if self.now_fn is not None:
                # On the mission clock the legacy exchange would stamp lines
                # and deltas with device wall times (critic A1, B3).
                raise ValueError(
                    f"{spec.api}: this mission server runs on a mission clock "
                    "(now_fn), so every contact needs a ContactPlan (critic A1)"
                )
            return self._legacy_contact(
                spec, contact_devices, synth_batch, min_utility=min_utility,
                theta=theta, theta_version=theta_version, mission_round=mission_round,
            )
        return self._ferry_contact(
            spec, contact_devices, synth_batch, plan=plan, min_utility=min_utility,
            theta=theta, theta_version=theta_version, mission_round=mission_round,
        )

    # ---------------------------------------------- legacy path (afa9526)

    def _legacy_contact(
        self,
        spec: _PassSpec,
        contact_devices: Sequence[DeviceID],
        synth_batch,
        *,
        min_utility: float,
        theta: Weights,
        theta_version: Optional[int],
        mission_round: int,
    ) -> Dict[DeviceID, Any]:
        """The afa9526 exchange of both routines, statement for statement.

        Broadcast solicit, stash-aware gather, one worker per device writing
        through :class:`_LiveSink`, and each worker joined in turn for up to
        2 x TTL (P-01 defect 2, kept).
        """
        # 1. Broadcast solicit.
        solicit = FLOpenSolicit(
            mule_id=self.mule_id,
            mission_round=mission_round,
            issued_at=time.time(),
            pass_kind=spec.kind,
        )
        try:
            self.rf.broadcast_open_solicit(solicit)
        except RFLinkError as e:
            # Literal strings, so each LogRecord's msg is the one it always was.
            if spec.kind is MissionPass.COLLECT:
                log.warning("run_contact: broadcast failed: %s", e)
            else:
                log.warning("deliver_contact: broadcast failed: %s", e)
            return {did: spec.broadcast_failed for did in contact_devices}

        # 2. Gather replies (drains the misrouted stash first).
        expected = self._gather_ready_advs(contact_devices)

        # 3. Per-device session in parallel.
        outcomes: Dict[DeviceID, Any] = {}
        outcomes_lock = threading.Lock()
        session = (self._collect_session if spec.kind is MissionPass.COLLECT
                   else self._deliver_session)
        sink = _LiveSink(self)

        def _device_worker(did: DeviceID, adv: Optional[FLReadyAdv]) -> None:
            session(
                did, adv, theta=theta, theta_version=theta_version,
                mission_round=mission_round, synth_batch=synth_batch,
                min_utility=min_utility, outcomes=outcomes,
                outcomes_lock=outcomes_lock, sink=sink,
            )

        threads: List[threading.Thread] = []
        for did in contact_devices:
            adv = expected.get(did)
            t = threading.Thread(target=_device_worker, args=(did, adv), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=self.session_ttl_s * 2.0)

        return outcomes

    def _gather_ready_advs(
        self, contact_devices: Sequence[DeviceID],
    ) -> Dict[DeviceID, FLReadyAdv]:
        """Legacy gather: the stash first, then the link, until all answer or TTL.

        Drain the internal misrouted stash first (replies that arrived during
        a prior contact and weren't consumed), then pop fresh FLReadyAdv off
        the link until every expected device is accounted for OR the
        per-contact deadline elapses. Replies from outside this contact's
        expected set go back into the stash for the next contact's drain
        (Sprint 1.5 H1).
        """
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
        return expected

    def _collect_session(
        self,
        did: DeviceID,
        adv: Optional[FLReadyAdv],
        *,
        theta: Weights,
        theta_version: Optional[int],
        mission_round: int,
        synth_batch,
        min_utility: float,
        outcomes: Dict[DeviceID, Any],
        outcomes_lock,
        sink: _LiveSink,
    ) -> None:
        """Legacy Pass-1 worker body (afa9526 ``run_contact``'s closure)."""
        if adv is None:
            with outcomes_lock:
                outcomes[did] = MissionOutcome.TIMEOUT
            sink.outcome(
                device_id=did,
                outcome=MissionOutcome.TIMEOUT,
                contact_ts=time.time(),
                utility=0.0,
                bytes_received=0,
                bytes_sent=0,
                answered=False,
            )
            return

        # S2B gate on arrival.
        if not adv.is_eligible() or adv.utility < min_utility:
            sink.contact(adv, in_session=False)
            sink.outcome(
                device_id=adv.device_id,
                outcome=MissionOutcome.PARTIAL,
                contact_ts=time.time(),
                utility=adv.utility,
                local_loss=adv.local_loss,
                num_examples=adv.num_examples,
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
            basis_version=theta_version,
            update_form=self.aggregation.update_form,
            train_ahead=self.train_ahead,
        )
        try:
            self.rf.push_disc(adv.device_id, push)
        except RFLinkError as e:
            log.warning("run_contact push failed device=%s: %s", did, e)
            sink.outcome(
                device_id=did,
                outcome=MissionOutcome.TIMEOUT,
                contact_ts=time.time(),
                utility=adv.utility,
                local_loss=adv.local_loss,
                num_examples=adv.num_examples,
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
            sink.contact(adv, in_session=True)
            sink.outcome(
                device_id=adv.device_id,
                outcome=MissionOutcome.TIMEOUT,
                contact_ts=time.time(),
                utility=adv.utility,
                local_loss=adv.local_loss,
                num_examples=adv.num_examples,
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
                sink.accept(grad)

        sink.contact(adv, in_session=True)
        sink.outcome(
            device_id=adv.device_id,
            outcome=outcome,
            contact_ts=grad.submitted_at,
            utility=adv.utility,
            local_loss=adv.local_loss,
            num_examples=adv.num_examples,
            bytes_received=grad.byte_count,
            bytes_sent=push_byte_count(push),
            basis_version=grad.basis_version,
            line_num_examples=grad.num_examples,
        )
        self._release_busy(adv.device_id)
        with outcomes_lock:
            outcomes[did] = outcome

    def _deliver_session(
        self,
        did: DeviceID,
        adv: Optional[FLReadyAdv],
        *,
        theta: Weights,
        theta_version: Optional[int],
        mission_round: int,
        synth_batch,
        min_utility: float,
        outcomes: Dict[DeviceID, Any],
        outcomes_lock,
        sink: _LiveSink,
    ) -> None:
        """Legacy Pass-2 worker body (afa9526 ``deliver_contact``'s closure).

        ``min_utility`` is accepted and ignored: Pass 2 has no S2B gate.
        """
        if adv is None:
            sink.delivery(
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
            basis_version=theta_version,
        )
        try:
            self.rf.push_disc(adv.device_id, push)
        except RFLinkError as e:
            log.warning("deliver_contact push failed device=%s: %s", did, e)
            sink.delivery(
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
            sink.delivery(
                device_id=did,
                outcome=DeliveryOutcome.UNDELIVERED,
                contact_ts=time.time(),
            )
            with outcomes_lock:
                outcomes[did] = DeliveryOutcome.UNDELIVERED
            return

        sink.delivery(
            device_id=did,
            outcome=DeliveryOutcome.DELIVERED,
            contact_ts=ack.received_at,
        )
        with outcomes_lock:
            outcomes[did] = DeliveryOutcome.DELIVERED

    # ---------------------------------------------- ferry path (FeRRy Phase 3)

    def _ferry_contact(
        self,
        spec: _PassSpec,
        contact_devices: Sequence[DeviceID],
        synth_batch,
        *,
        plan: ContactPlan,
        min_utility: float,
        theta: Weights,
        theta_version: Optional[int],
        mission_round: int,
    ) -> Dict[DeviceID, Any]:
        """One contact on the mission clock (design section 4.3).

        1. The plan is checked against the contact and the clock (the clock
           must still read the plan's arrival time).
        2. The misrouted stash is emptied, and the link's stale adverts and
           the targets' stale gradients and acks are drained without blocking
           (critic B2).
        3. One numbered solicit goes to the plan's targets only
           (``RFLink.solicit``); unreachable members are never solicited.
        4. Adverts are gathered under the wall TTL (those already queued when
           it runs out are still taken, without waiting); an advert counts only
           if it comes from a target the solicit reached and answers this
           solicit's number (critic B1). Anything else is discarded, never
           stashed. A worker starts as soon as its advert is in, so no target
           waits for its push on a silent one. A Pass-1 target in
           ``drop_uplink`` gets a push marked ``uplink_drop`` and is not
           waited for.
        5. All workers are joined under one wall deadline of 2 x TTL after the
           gather (P-01 defect 2 closed in this mode).
        6. :meth:`_ferry_commit` writes the ledger in ``contact_devices`` order,
           stamped in simulated seconds, and charges the clock.

        The receipt TTL check, the busy flags and each advert's loss and
        example count are the legacy ones.
        """
        members = list(contact_devices)
        self._check_plan(spec, plan, members)
        with self._lock:
            self._solicit_seq += 1
            solicit_id = self._solicit_seq
        sink = _BufferSink(spec.api)
        ctx = _FerryContext(
            spec=spec, mission_round=mission_round, solicit_id=solicit_id,
            theta=theta, theta_version=theta_version, synth_batch=synth_batch,
            min_utility=min_utility,
            drop_uplink=frozenset(plan.drop_uplink) if spec.kind is MissionPass.COLLECT
            else frozenset(),
            not_ready=frozenset(plan.not_ready),
        )
        target_set = set(plan.targets)
        targets = [did for did in members if did in target_set]

        # 2. Nothing that belongs to another contact may be read as this one's.
        with self._lock:
            self._misrouted_advs = []
        reached: List[DeviceID] = []
        answered: Dict[DeviceID, FLReadyAdv] = {}
        threads: List[threading.Thread] = []
        if targets:
            sink.stale("adv", self._drain(self.rf.recv_ready_adv))
            for did in targets:
                sink.stale("gradient", self._drain(
                    lambda timeout, _d=did: self.rf.recv_gradient(_d, timeout=timeout)))
                sink.stale("ack", self._drain(
                    lambda timeout, _d=did: self.rf.recv_delivery_ack(_d, timeout=timeout)))

            # 3. Targeted, numbered solicit. Exp 5 addendum (Study 5.12): the
            #    targets with no update ready are named in it, so they wait for
            #    no push; the field is left at its default otherwise.
            not_ready = tuple(did for did in targets if did in ctx.not_ready)
            solicit = FLOpenSolicit(
                mule_id=self.mule_id,
                mission_round=mission_round,
                issued_at=plan.arrival_ts,
                pass_kind=spec.kind,
                solicit_id=solicit_id,
                **({"not_ready": not_ready} if not_ready else {}),
            )
            try:
                reached = list(self.rf.solicit(solicit, targets))
            except RFLinkError as e:
                # Nobody was reached: every target is silent, and the commit
                # still writes every member's line and charges the listen.
                log.warning("%s: targeted solicit failed: %s", spec.api, e)
                reached = []

            # 4. Gather, starting each target's worker as its advert arrives.
            worker = (self._ferry_collect_worker if spec.kind is MissionPass.COLLECT
                      else self._ferry_deliver_worker)
            waiting = set(reached)
            deadline = time.time() + self.session_ttl_s
            late_polls = 0
            while waiting - answered.keys():
                # Wait for adverts until the deadline. Past it, still take the
                # ones already queued, without waiting: an advert that arrived
                # in time must not be lost because starting the workers of the
                # adverts before it used up a short TTL on a loaded host. The
                # cap ends the loop on a link that never runs dry.
                remaining = deadline - time.time()
                if remaining <= 0:
                    late_polls += 1
                    if late_polls > _DRAIN_LIMIT:
                        break
                try:
                    adv = self.rf.recv_ready_adv(timeout=max(0.0, remaining))
                except RFLinkError:
                    break
                did = adv.device_id
                if did in waiting and did not in answered and adv.in_reply_to == solicit_id:
                    answered[did] = adv
                    t = threading.Thread(
                        target=worker, args=(did, adv, ctx, sink), daemon=True,
                        name=f"ferry-{spec.kind.value}-{did}",
                    )
                    t.start()
                    threads.append(t)
                else:
                    sink.stale("adv")
                    log.debug(
                        "%s: discarded an advert from device=%s answering solicit %d "
                        "(this contact's is %d)",
                        spec.api, did, adv.in_reply_to, solicit_id,
                    )

        # 5. One join deadline for the whole contact.
        join_deadline = time.time() + 2.0 * self.session_ttl_s
        for t in threads:
            t.join(timeout=max(0.0, join_deadline - time.time()))
        sessions, pushed, stale = sink.close()

        # 6. Commit in device order on the mission clock.
        return self._ferry_commit(
            spec, plan, members, ctx=ctx, reached=reached, answered=answered,
            sessions=sessions, pushed=pushed, stale=stale,
        )

    def _check_plan(
        self, spec: _PassSpec, plan: ContactPlan, members: List[DeviceID],
    ) -> None:
        """Refuse a plan that does not describe this contact on this clock.

        The clock test compares readings, so it refuses a plan made at another
        time but cannot tell two clocks apart that happen to read the same
        (two fresh clocks at the epoch). The commit finishes the check once it
        has charged: the plan's clock and the host's ``now_fn`` must then both
        read the contact's end, or it raises before anything is written.
        """
        if not isinstance(plan, ContactPlan):
            raise TypeError(f"plan must be a ContactPlan, got {type(plan).__name__}")
        if self.now_fn is None:
            raise ValueError(
                f"{spec.api}: a ContactPlan runs on the mission clock; build "
                "HFLHostMission(now_fn=<that clock>) so the reports share it"
            )
        if len(set(members)) != len(members):
            raise ValueError(f"{spec.api}: contact devices repeat: {members!r}")
        if set(members) != set(plan.members):
            raise ValueError(
                f"{spec.api}: the plan's members {sorted(plan.members)} are not "
                f"the contact's devices {sorted(members)}"
            )
        if spec.kind is MissionPass.DELIVER and plan.drop_uplink:
            raise ValueError(
                "deliver_contact: drop_uplink is a Pass-1 availability draw; "
                "Pass 2 faces the SNR gate only"
            )
        if spec.kind is MissionPass.DELIVER and plan.not_ready:
            raise ValueError(
                "deliver_contact: not_ready is a Pass-1 mark (no update to collect); "
                "a delivery needs none"
            )
        clock_now = plan.clock()
        if clock_now != plan.arrival_ts or self.now_fn() != plan.arrival_ts:
            raise ValueError(
                f"{spec.api}: the plan was made at t={plan.arrival_ts!r} but the "
                f"mission clock reads {clock_now!r} (host clock {self.now_fn()!r}); "
                "build the plan at arrival and run the contact at once"
            )

    @staticmethod
    def _drain(recv: Callable[..., Any]) -> int:
        """Take everything already queued on ``recv`` without waiting; return how many."""
        n = 0
        while n < _DRAIN_LIMIT:
            try:
                recv(timeout=0.0)
            except RFLinkError:
                break
            n += 1
        return n

    def _await_reply(
        self,
        recv: Callable[..., Any],
        did: DeviceID,
        matches: Callable[[Any], bool],
        what: str,
        sink: _BufferSink,
    ) -> Optional[Any]:
        """The device's matching reply within the wall TTL, or None.

        A reply that belongs to another contact (another round, solicit or
        push) is discarded and the wait goes on to the same deadline: a stale
        gradient never becomes this contact's PARTIAL, and a stale ack is
        never counted as DELIVERED (critic B2). Past the deadline, replies
        already queued are still read, without waiting, as the gather does:
        a matching one behind stale ones counts.
        """
        deadline = time.time() + self.session_ttl_s
        late_polls = 0
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                late_polls += 1
                if late_polls > _DRAIN_LIMIT:
                    return None
            try:
                msg = recv(did, timeout=max(0.0, remaining))
            except RFLinkError:
                return None
            if matches(msg):
                return msg
            sink.stale(what)
            log.info(
                "ferry contact mule=%s: discarded a stale %s from device=%s",
                self.mule_id, what, did,
            )

    def _ferry_collect_worker(
        self, did: DeviceID, adv: FLReadyAdv, ctx: _FerryContext, sink: _BufferSink,
    ) -> None:
        """Pass-1 ferry session with one answered target; records nothing itself."""
        if did in ctx.not_ready:
            # Exp 5 addendum (Study 5.12): no update ready on the fit clock.
            # The device was told so in the solicit and waits for no push.
            sink.put(did, _FerrySession(_NOT_READY))
            return
        # S2B gate on arrival, as in legacy.
        if not adv.is_eligible() or adv.utility < ctx.min_utility:
            sink.put(did, _FerrySession(_REFUSED))
            return

        self._claim_busy(did)
        uplink_drop = did in ctx.drop_uplink
        push = DiscPush(
            mule_id=self.mule_id,
            mission_round=ctx.mission_round,
            theta_disc=ctx.theta,
            synth_batch=ctx.synth_batch,
            pass_kind=MissionPass.COLLECT,
            basis_version=ctx.theta_version,
            update_form=self.aggregation.update_form,
            train_ahead=self.train_ahead,
            uplink_drop=uplink_drop,
        )
        try:
            self.rf.push_disc(did, push)
        except RFLinkError as e:
            log.warning("run_contact push failed device=%s: %s", did, e)
            self._release_busy(did)
            sink.put(did, _FerrySession(_PUSH_FAILED))
            return
        nbytes = push_byte_count(push)
        sink.pushed(did, nbytes)

        if uplink_drop:
            # The device adopts the basis and sends nothing back: there is no
            # reply to wait for, and the TTL is not waited out.
            self._release_busy(did)
            sink.put(did, _FerrySession(_NO_REPLY, push_bytes=nbytes, uplink_drop=True))
            return

        def _matches(g: Any) -> bool:
            return (
                isinstance(g, GradientSubmission)
                and g.device_id == did
                and g.mule_id == self.mule_id
                and g.mission_round == ctx.mission_round
                and g.in_reply_to == ctx.solicit_id
            )

        grad = self._await_reply(self.rf.recv_gradient, did, _matches, "gradient", sink)
        if grad is None:
            self._release_busy(did)
            sink.put(did, _FerrySession(_NO_REPLY, push_bytes=nbytes))
            return
        # The receipt check keeps its wall-clock TTL: it compares the mule's
        # wall time with the device's (design section 2.4).
        verdict = self._verify_receipt(grad, ctx.mission_round)
        self._release_busy(did)
        sink.put(did, _FerrySession(_REPLIED, push_bytes=nbytes, grad=grad, verdict=verdict))

    def _ferry_deliver_worker(
        self, did: DeviceID, adv: FLReadyAdv, ctx: _FerryContext, sink: _BufferSink,
    ) -> None:
        """Pass-2 ferry delivery to one answered target; records nothing itself."""
        push = DiscPush(
            mule_id=self.mule_id,
            mission_round=ctx.mission_round,
            theta_disc=ctx.theta,
            synth_batch=ctx.synth_batch,
            pass_kind=MissionPass.DELIVER,
            basis_version=ctx.theta_version,
        )
        try:
            self.rf.push_disc(did, push)
        except RFLinkError as e:
            log.warning("deliver_contact push failed device=%s: %s", did, e)
            sink.put(did, _FerrySession(_PUSH_FAILED))
            return
        nbytes = push_byte_count(push)
        sink.pushed(did, nbytes)

        def _matches(a: Any) -> bool:
            return (
                isinstance(a, DeliveryAck)
                and a.device_id == did
                and a.mule_id == self.mule_id
                and a.mission_round == ctx.mission_round
                and a.weights_sig == push.weights_sig
                and a.in_reply_to == ctx.solicit_id
            )

        ack = self._await_reply(self.rf.recv_delivery_ack, did, _matches, "ack", sink)
        if ack is None:
            sink.put(did, _FerrySession(_NO_REPLY, push_bytes=nbytes))
            return
        sink.put(did, _FerrySession(_REPLIED, push_bytes=nbytes))

    def _ferry_commit(
        self,
        spec: _PassSpec,
        plan: ContactPlan,
        members: List[DeviceID],
        *,
        ctx: _FerryContext,
        reached: List[DeviceID],
        answered: Dict[DeviceID, FLReadyAdv],
        sessions: Dict[DeviceID, _FerrySession],
        pushed: Dict[DeviceID, int],
        stale: Dict[str, int],
    ) -> Dict[DeviceID, Any]:
        """Stamp, charge and write one ferry contact, in ``members`` order.

        Time runs from ``t = plan.arrival_ts``:

        * an unreachable member is stamped at arrival, with no airtime;
        * with a band, each answered target's session costs
          ``dwell(bytes, SNR at its own session start)`` (critic C2) and ends
          at ``t += dwell``; ``bytes`` is the ledger's push plus update (Pass 2:
          the push), or the declared payload per direction;
        * a session refused by S2B or whose push failed costs nothing and is
          stamped at the current ``t``, with no listen; so is a worker still
          inside its push at the join deadline: the mule gave up on that push,
          as on a failed one;
        * a target whose reply is missing costs its push's airtime if a push
          went out, and is stamped after the listen window, which is charged
          once for the contact (critic C7). Missing means silent (no advert
          answered this contact's solicit, no push), uplink dropped, or pushed
          with no matching reply by the TTL or by the join deadline;
        * a target the plan marks ``not_ready`` (Exp 5 addendum, Study 5.12)
          costs nothing and is stamped at the current ``t``, as a refused one:
          its advert came and nothing was pushed;
        * without a band the contact costs ``session_time_s`` once (critic A1)
          and every answered session ends there.

        The clock is charged once for the dwell and, separately, once for the
        listen, then the lines, deltas and contact records are written with
        those stamps, the band and the SNR, and CLEAN updates join the accepted
        list in device order. Afterwards ``clock() == max(contact_ts)``, and
        the host's ``now_fn`` reads the same: a plan whose charges did not move
        its own clock, or whose clock is not the host's, is a wiring bug and
        raises ``RuntimeError`` after the charge, before any write.
        """
        collect = spec.kind is MissionPass.COLLECT
        banded = plan.band is not None
        unreachable = set(plan.unreachable)
        t_arr = plan.arrival_ts
        if plan.clock() != t_arr:
            # Something else charged the clock while the contact ran: a wiring
            # bug, not a session failure, so not a MissionSessionError (the
            # supervisor logs and skips those). Nothing is charged or written.
            raise RuntimeError(
                f"{spec.api}: the mission clock moved during the contact "
                f"({t_arr!r} -> {plan.clock()!r}); only the commit may charge it"
            )

        # 1. Classify every member once. A worker still running at the join
        #    deadline is committed from what it had done: with its push out it
        #    is a missing reply, without one the mule gave up on the push, as
        #    on a failed one.
        klass: Dict[DeviceID, str] = {}
        push_bytes: Dict[DeviceID, int] = {}
        for did in members:
            sess = sessions.get(did)
            if did in unreachable:
                klass[did] = _UNREACHABLE
            elif did not in answered:
                klass[did] = _SILENT
            elif sess is not None:
                klass[did] = sess.kind
                push_bytes[did] = sess.push_bytes
            elif did in pushed:
                klass[did] = _NO_REPLY
                push_bytes[did] = pushed[did]
            else:
                klass[did] = _PUSH_FAILED

        # 2. Stamps and airtime, in device order from the arrival.
        stamps: Dict[DeviceID, float] = {}
        snrs: Dict[DeviceID, Optional[float]] = {}
        dwell_by_device: Dict[DeviceID, float] = {}
        uplink_by_device: Dict[DeviceID, float] = {}
        missing: List[DeviceID] = []

        def charge(did: DeviceID, t: float, nbytes: int) -> float:
            """Price one session starting at ``t``; return when it ends."""
            snr = plan.snr_at(did, t)
            snrs[did] = snr
            if not banded:
                return t
            d = plan.session_dwell_s(nbytes, snr)
            if d > 0.0:
                dwell_by_device[did] = d
            return t + d

        t = t_arr if banded else t_arr + plan.session_time_s
        for did in members:
            kind = klass[did]
            if kind == _UNREACHABLE:
                stamps[did] = t_arr
                snrs[did] = plan.snr_db[did]
            elif kind == _SILENT:
                snrs[did] = plan.snr_db[did] if banded else None
                missing.append(did)
            elif kind in (_REFUSED, _PUSH_FAILED, _NOT_READY):
                snrs[did] = plan.snr_at(did, t)
                stamps[did] = t
            elif kind == _REPLIED:
                if collect:
                    grad = sessions[did].grad
                    assert grad is not None
                    nbytes = plan.session_bytes(push_bytes[did] + grad.byte_count, 2)
                else:
                    nbytes = plan.session_bytes(push_bytes[did], 1)
                t = charge(did, t, nbytes)
                stamps[did] = t
                if collect and banded:
                    # Study 5.12: the update's own share, for the device's
                    # transmit energy; read only, nothing is charged for it.
                    uplink_by_device[did] = plan.session_dwell_s(
                        plan.session_bytes(grad.byte_count, 1), snrs[did])
            else:  # _NO_REPLY: the push's airtime was spent, the reply never came
                t = charge(did, t, plan.session_bytes(push_bytes[did], 1))
                missing.append(did)
        t_dwell = t
        t_end = t_dwell + plan.listen_s if missing else t_dwell
        for did in missing:
            stamps[did] = t_end

        # 3. Charge the clock: once for the dwell, once more for the listen.
        #    The charges are differences of the stamps, exact while the
        #    contact's airtime stays below the clock's own reading (Sterbenz),
        #    so the clock lands on ``t_end`` itself; longer, it can land an ulp
        #    off. It is charged before the writes, so no delta is stamped ahead
        #    of it.
        plan.advance(t_dwell - t_arr, KIND_DWELL)
        if missing:
            plan.advance(t_end - t_dwell, KIND_LISTEN)
        t_clock = plan.clock()
        if t_clock != t_end:
            if abs(t_clock - t_end) > _CHARGE_ROUNDING_ULPS * math.ulp(t_end):
                # The plan's advance charged some other clock (or none). Its
                # clock was not moved as charged, so no stamp can be trusted:
                # nothing is written.
                raise RuntimeError(
                    f"{spec.api}: charging the contact should take the plan's "
                    f"clock from {t_arr!r} to {t_end!r}, but it reads {t_clock!r}: "
                    "plan.advance must charge plan.clock"
                )
            # Float rounding of a very long contact: move the stamps that ended
            # it onto the clock, so clock() == max(contact_ts) stays exact.
            stamps = {d: (t_clock if s == t_end else s) for d, s in stamps.items()}
            t_end = t_clock
        t_host = self.now_fn()  # type: ignore[misc]  # _check_plan: not None
        if t_host != t_clock:
            # Two clocks that read the same at arrival pass _check_plan; only
            # a charge tells them apart. The host stamps its reports with
            # now_fn, so they would fall behind these lines (a report finished
            # before its own sessions ended). A wiring bug: nothing is written.
            raise RuntimeError(
                f"{spec.api}: the contact charged the plan's clock to {t_clock!r} "
                f"but the host's clock (now_fn) reads {t_host!r}: build the "
                "ContactPlan and HFLHostMission(now_fn=...) on the same mission clock"
            )

        # 4. The ledger, in device order.
        band_index = plan.band_index
        outcomes: Dict[DeviceID, Any] = {}
        for did in members:
            kind = klass[did]
            ts = stamps[did]
            snr = snrs.get(did)
            if not collect:
                outcome = (DeliveryOutcome.DELIVERED if kind == _REPLIED
                           else DeliveryOutcome.UNDELIVERED)
                self._record_delivery_line(
                    device_id=did, outcome=outcome, contact_ts=ts,
                    band=band_index, bytes_sent=push_bytes.get(did, 0),
                )
                outcomes[did] = outcome
                continue

            if kind in (_UNREACHABLE, _SILENT):
                outcome = MissionOutcome.TIMEOUT
                self._record_outcome(
                    device_id=did, outcome=outcome, contact_ts=ts, utility=0.0,
                    bytes_received=0, bytes_sent=0, answered=False,
                    band=band_index, snr_db=snr,
                )
                outcomes[did] = outcome
                continue
            adv = answered[did]
            rec_snr = 0.0 if snr is None else snr
            common = dict(
                device_id=did, contact_ts=ts, utility=adv.utility,
                local_loss=adv.local_loss, num_examples=adv.num_examples,
                band=band_index, snr_db=snr,
            )
            if kind == _REFUSED:
                outcome = MissionOutcome.PARTIAL
                self._record_contact(adv, in_session=False, contact_ts=ts,
                                     snr_at_contact=rec_snr)
                self._record_outcome(outcome=outcome, bytes_received=0,
                                     bytes_sent=0, **common)
            elif kind == _PUSH_FAILED:
                outcome = MissionOutcome.TIMEOUT
                self._record_outcome(outcome=outcome, bytes_received=0,
                                     bytes_sent=0, **common)
            elif kind == _NOT_READY:
                # Exp 5 addendum (Study 5.12): the advert came, no update was
                # ready, nothing was pushed. Recorded as a contact out of
                # session and a TIMEOUT: the round got no update from it.
                outcome = MissionOutcome.TIMEOUT
                self._record_contact(adv, in_session=False, contact_ts=ts,
                                     snr_at_contact=rec_snr)
                self._record_outcome(outcome=outcome, bytes_received=0,
                                     bytes_sent=0, **common)
            elif kind == _NO_REPLY:
                outcome = MissionOutcome.TIMEOUT
                self._record_contact(adv, in_session=True, contact_ts=ts,
                                     snr_at_contact=rec_snr)
                self._record_outcome(outcome=outcome, bytes_received=0,
                                     bytes_sent=push_bytes[did], **common)
            else:  # _REPLIED
                sess = sessions[did]
                grad, outcome = sess.grad, sess.verdict
                assert grad is not None and outcome is not None
                if outcome is MissionOutcome.CLEAN:
                    with self._lock:
                        self._accepted.append(grad)
                self._record_contact(adv, in_session=True, contact_ts=ts,
                                     snr_at_contact=rec_snr)
                self._record_outcome(
                    outcome=outcome,
                    bytes_received=grad.byte_count,
                    bytes_sent=push_bytes[did],
                    basis_version=grad.basis_version,
                    line_num_examples=grad.num_examples,
                    **common,
                )
            outcomes[did] = outcome

        answered_ids = tuple(did for did in members if did in answered)
        self.last_contact = ContactCommit(
            arrival_ts=t_arr,
            end_ts=t_end,
            dwell_s=t_dwell - t_arr,
            listen_s=t_end - t_dwell,
            band=plan.band,
            band_index=band_index,
            targets=tuple(did for did in members if did not in unreachable),
            unreachable=tuple(did for did in members if did in unreachable),
            solicited=tuple(reached),
            answered=answered_ids,
            missing=tuple(missing),
            uplink_dropped=tuple(
                did for did in members
                if did in sessions and sessions[did].uplink_drop
            ),
            contact_ts=dict(stamps),
            snr_db=dict(snrs),
            session_dwell_s=dict(dwell_by_device),
            stale_discarded=dict(stale),
            solicit_id=ctx.solicit_id,
            not_ready=tuple(did for did in members if klass[did] == _NOT_READY),
            pushed=tuple(did for did in members if klass[did] in (_REPLIED, _NO_REPLY)),
            uplink_dwell_s=dict(uplink_by_device),
        )
        log.info(
            "ferry %s mule=%s round=%d band=%s members=%d targets=%d answered=%d "
            "missing=%d dwell=%.6fs listen=%.3fs t=%.6f->%.6f stale=%s",
            spec.api, self.mule_id, ctx.mission_round, plan.band, len(members),
            len(members) - len(unreachable), len(answered_ids), len(missing),
            t_dwell - t_arr, t_end - t_dwell, t_arr, t_end, dict(stale),
        )
        return outcomes

    def _record_delivery_line(
        self,
        *,
        device_id: DeviceID,
        outcome: DeliveryOutcome,
        contact_ts: float,
        band: Optional[int] = None,
        bytes_sent: int = 0,
    ) -> None:
        with self._lock:
            if self._delivery_report is None:
                return
            self._delivery_report.append(
                MissionDeliveryLine(
                    device_id=device_id,
                    outcome=outcome,
                    contact_ts=contact_ts,
                    band=band,
                    bytes_sent=bytes_sent,
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

        # A full model and a delta are not interchangeable: merging one as the
        # other would corrupt θ silently, so a form the rule did not ask for is
        # refused like any other bad receipt.
        if grad.update_form != self.aggregation.update_form:
            log.warning(
                "verify: update form mismatch device=%s got=%s expected=%s",
                grad.device_id, grad.update_form, self.aggregation.update_form,
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
        # Freeze Amendment 3 — Oort baseline inputs, forwarded from the
        # device's advertisement. Optional so callers that do not carry them
        # (and every pre-B2 arm) stay valid and fold to a no-op.
        local_loss: Optional[float] = None,
        num_examples: int = 0,
        # FeRRy Phase 1 — for the report line only: the collected update's
        # basis version and example count. The scheduler delta above keeps
        # the values it always had.
        basis_version: Optional[int] = None,
        line_num_examples: Optional[int] = None,
        # FeRRy: whether the device's advert arrived. Every caller holds the
        # advert except the one branch that never heard from the device.
        answered: bool = True,
        # FeRRy Phase 3 — the report line's contact band index and SNR; only
        # the ferry commit passes them.
        band: Optional[int] = None,
        snr_db: Optional[float] = None,
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
                    num_examples=int(
                        num_examples if line_num_examples is None
                        else line_num_examples
                    ),
                    basis_version=basis_version,
                    age=update_age(self._current_theta_version, basis_version),
                    band=band,
                    snr_db=snr_db,
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
                    local_loss=local_loss,
                    num_examples=num_examples,
                    answered=bool(answered),
                )
            )
        except Exception:  # pragma: no cover — bus faults must not kill the mule
            log.exception("scheduler_bus raised; dropping delta")

    def _record_contact(
        self,
        adv: FLReadyAdv,
        *,
        in_session: bool,
        contact_ts: Optional[float] = None,
        snr_at_contact: float = 0.0,
    ) -> None:
        # ``contact_ts`` None is the legacy stamp, the device's advert time
        # (or the wall clock); the ferry commit passes its simulated stamp and
        # the SNR so no device wall time reaches a record (critic B3).
        with self._lock:
            if self._contacts is None:
                return
            self._contacts.add(
                ContactRecord(
                    device_id=adv.device_id,
                    contact_ts=(adv.issued_at or time.time())
                    if contact_ts is None else contact_ts,
                    in_session=in_session,
                    snr_at_contact=snr_at_contact,
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
