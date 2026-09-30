"""``ClientCluster`` — Mule-NUC dock handler (Phase 3).

Mirror of ``HFLHostCluster`` on the mule side. Drives one dock cycle:

    AWAIT_DOCK -> COLLECT -> UP -> DOWN -> VERIFY -> DISTRIBUTE -> AWAIT_DOCK

Design refs:
* HERMES_FL_Scheduler_Design.md §5.4 (ClientCluster state machine)
* HERMES_FL_Scheduler_Implementation_Plan.md §3 Phase 3

Responsibilities:
1. Poll ``DockLink.is_available()`` until the mule is docked.
2. Collect the most recent mission outputs from ``HFLHostMission``.
3. Upload the ``UpBundle``; retry across dock attempts on failure.
4. Await the ``DownBundle``; verify its signature.
5. Fan out intra-NUC:
     * ``MissionSlice`` + ``ClusterAmendment`` -> FLScheduler (slow-phase)
     * ``theta_disc`` + ``synth_batch`` -> HFLHostMission (next-round model)

The two fan-out sinks are plain callables so Phase 3 can be tested
without a real scheduler or a live mission server. Phase 4 wires real
intra-NUC pub/sub behind the same callable signature.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from hermes.transport import DockLink, DockLinkError
from hermes.transport.dock_link import DockLinkTimeout
from hermes.types.bundles import BackhaulUpload
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    DownBundle,
    MissionDeliveryReport,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    UpBundle,
    Weights,
    sign_up_bundle,
    verify_down_bundle,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors + enums
# --------------------------------------------------------------------------- #

class ClientClusterError(RuntimeError):
    """Raised on unrecoverable dock-cycle failures."""


class DownTimeout(ClientClusterError):
    """No DOWN arrived within the wait, and the dock link is still up.

    Raised only where the caller asked to survive the wait
    (``recoverable_down_wait``, :meth:`ClientCluster.await_down`, or a bounded
    bootstrap wait): the cluster may simply not have answered yet, for
    example while it waits for other mules to reach a quorum. A closed link
    still raises a plain :class:`ClientClusterError`.
    """


class ClientClusterState(str, Enum):
    """States of the Phase 3 state machine."""

    AWAIT_DOCK = "await_dock"
    COLLECT = "collect"
    UP = "up"
    DOWN = "down"
    VERIFY = "verify"
    DISTRIBUTE = "distribute"


# --------------------------------------------------------------------------- #
# Intra-NUC fan-out contract
# --------------------------------------------------------------------------- #

SchedulerSlowPhaseSink = Callable[[MissionSlice, ClusterAmendment], None]
MissionModelSink = Callable[[Weights, List], None]  # (theta_disc, synth_batch)
ModelVersionSink = Callable[[int], None]  # version of the θ about to be staged


@dataclass
class BundleDistributor:
    """Holds the intra-NUC callables the dock fan-out delivers into.

    Default sinks are no-ops so a ``ClientCluster`` constructed in tests
    doesn't need both callables wired up. Production builds (Phase 6)
    pass real scheduler / mission-server handles.

    ``on_model_version`` receives the version of the DOWN bundle's θ (the
    cluster round that produced it, ``mission_slice.issued_round``) just
    before ``on_next_round_model`` receives θ itself (FeRRy Phase 1).
    """

    on_slice_and_amendment: SchedulerSlowPhaseSink = field(
        default=lambda _s, _a: None
    )
    on_next_round_model: MissionModelSink = field(
        default=lambda _w, _b: None
    )
    on_model_version: ModelVersionSink = field(default=lambda _v: None)


# --------------------------------------------------------------------------- #
# Retry queue entry
# --------------------------------------------------------------------------- #

@dataclass
class _PendingUp:
    """One UP that failed to land and must be retried next dock."""

    bundle: UpBundle
    first_attempt_at: float
    attempts: int = 0


# --------------------------------------------------------------------------- #
# ClientCluster
# --------------------------------------------------------------------------- #

class ClientCluster:
    """Mule-NUC dock handler. One instance per mule.

    ``recoverable_down_wait`` (FeRRy Phase 2, off by default) is for a mule
    that may outlive a DOWN wait: a timeout while the link is up raises
    :class:`DownTimeout`, which the caller can survive, and every DOWN already
    queued when the mule docks is thrown away before it uploads. Those can
    only be late answers to an upload the mule stopped waiting for — the
    cluster answers a mule only after ingesting its UP — so reading one as the
    reply to the new upload would hand the mule a θ that leaves its own update
    out. Off, a timeout is the plain :class:`ClientClusterError` it always was
    and nothing is drained.

    Whatever the mode, when several DOWNs are queued at a read the newest (by
    ``issued_round``) wins and the rest are dropped. A lone mule is answered
    exactly once per upload, so there is never more than one to choose from.
    """

    def __init__(
        self,
        *,
        mule_id: MuleID,
        dock: DockLink,
        distributor: Optional[BundleDistributor] = None,
        dock_poll_interval_s: float = 0.5,
        up_timeout_s: float = 10.0,
        down_timeout_s: float = 10.0,
        max_retry_attempts: int = 5,
        recoverable_down_wait: bool = False,
    ) -> None:
        self.mule_id = mule_id
        self.dock = dock
        self.distributor = distributor or BundleDistributor()
        self.dock_poll_interval_s = dock_poll_interval_s
        self.up_timeout_s = up_timeout_s
        self.down_timeout_s = down_timeout_s
        self.max_retry_attempts = max_retry_attempts
        self.recoverable_down_wait = bool(recoverable_down_wait)
        #: DOWNs thrown away: drained before an upload, or superseded by a
        #: newer one at a read. Introspection only.
        self.stale_downs_dropped: int = 0

        self._lock = threading.RLock()
        self._state: ClientClusterState = ClientClusterState.AWAIT_DOCK

        # Pending round-output snapshot. ``collect()`` pushes into this;
        # ``run_dock_cycle`` pulls from it.
        self._staged_aggregate: Optional[PartialAggregate] = None
        self._staged_report: Optional[MissionRoundCloseReport] = None
        self._staged_contacts: Optional[ContactHistory] = None
        # Sprint 1.5 H3 — Pass-2 delivery report from the *previous*
        # mission, ridden up in the *next* mission's Pass-1 UP bundle
        # so the cluster can bump DeviceRecord.delivery_priority on
        # undelivered devices. Cleared after a successful UP send.
        self._staged_delivery_report: Optional[MissionDeliveryReport] = None
        # FeRRy Phase 3 — the staged upload's simulated completion time and
        # its backhaul pricing (the mission clock only; None otherwise).
        self._staged_sim_upload_ts: Optional[float] = None
        self._staged_backhaul: Optional[BackhaulUpload] = None

        # Retry queue — oldest first, drained on each successful dock.
        self._retry_queue: List[_PendingUp] = []

        # Last-seen down bundle (for introspection in tests/demos).
        self._last_down: Optional[DownBundle] = None

    # ---------------------------------------------- state API

    @property
    def state(self) -> ClientClusterState:
        with self._lock:
            return self._state

    def retry_queue_depth(self) -> int:
        with self._lock:
            return len(self._retry_queue)

    def last_down(self) -> Optional[DownBundle]:
        with self._lock:
            return self._last_down

    def last_cluster_sim_ts(self) -> Optional[float]:
        """The latest DOWN's ``cluster_sim_ts`` (FeRRy Phase 3), or None.

        None before any DOWN and for every DOWN that does not carry one (the
        cluster echoes its simulated time only in sim mode): the mule then
        makes no Lamport sync.
        """
        with self._lock:
            return getattr(self._last_down, "cluster_sim_ts", None)

    # ---------------------------------------------- COLLECT

    def collect(
        self,
        *,
        partial_aggregate: PartialAggregate,
        report: MissionRoundCloseReport,
        contacts: ContactHistory,
        delivery_report: Optional[MissionDeliveryReport] = None,
        sim_upload_ts: Optional[float] = None,
        backhaul: Optional[BackhaulUpload] = None,
    ) -> None:
        """Stage the latest mission output. Overwrites any prior stage.

        Called by the mule supervisor (or the demo) whenever
        ``HFLHostMission.close_round`` yields a fresh triple.

        Sprint 1.5 H3: ``delivery_report`` is the *previous mission's*
        Pass-2 ledger, attached to this mission's Pass-1 UP bundle so
        the cluster can carry over undelivered devices into the next
        slice. ``None`` for the very first mission (or for legacy
        single-pass missions).

        FeRRy Phase 3: ``sim_upload_ts`` (the upload's simulated completion
        time, critic B8) and ``backhaul`` (how the mule priced it) ride the UP
        built from this stage, and stay with it if it has to be retried.
        None, the default, is every wall-clock mule.
        """
        if partial_aggregate.mule_id != self.mule_id:
            raise ClientClusterError(
                f"mule_id mismatch in collect(): bundle={partial_aggregate.mule_id} "
                f"self={self.mule_id}"
            )
        with self._lock:
            self._staged_aggregate = partial_aggregate
            self._staged_report = report
            self._staged_contacts = contacts
            self._staged_delivery_report = delivery_report
            self._staged_sim_upload_ts = sim_upload_ts
            self._staged_backhaul = backhaul
            self._set_state(ClientClusterState.COLLECT)
            log.info(
                "collect: mule=%s mission_round=%d accepted=%d lines=%d "
                "delivery_report=%s",
                self.mule_id,
                partial_aggregate.mission_round,
                partial_aggregate.num_examples,
                len(report.lines),
                "yes" if delivery_report is not None else "no",
            )

    # ---------------------------------------------- AWAIT_DOCK

    def wait_for_dock(self, *, timeout: Optional[float] = None) -> bool:
        """Poll ``dock.is_available()`` until True or ``timeout`` elapses."""
        with self._lock:
            self._set_state(ClientClusterState.AWAIT_DOCK)

        deadline = None if timeout is None else time.time() + timeout
        while True:
            if self.dock.is_available():
                return True
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(self.dock_poll_interval_s)

    # ---------------------------------------------- bootstrap dock

    def bootstrap_down_only(
        self, timeout: Optional[float] = None,
    ) -> Optional[DownBundle]:
        """Initial dock: receive + distribute a DOWN bundle without sending UP.

        Used at supervisor startup, before the mule has run any missions
        and therefore has no aggregate to upload. The cluster must have
        already pre-dispatched a DOWN for this mule (registry slice +
        initial θ_disc + synth + amendments).

        Returns the verified ``DownBundle`` on success, ``None`` if the
        dock is unavailable. Raises ``ClientClusterError`` on a verify
        or routing failure (same semantics as the DOWN leg of the full
        cycle).

        ``timeout`` bounds the wait and makes it survivable: if nothing has
        arrived by then and the link is still up, this returns ``None`` so the
        caller can try again or give up on its own schedule. Without it the
        wait is ``down_timeout_s`` and running out raises, as it always has.
        """
        if not self.dock.is_available():
            return None
        if timeout is None:
            down = self._recv_and_verify_down()
        else:
            try:
                down = self._recv_and_verify_down(timeout, survivable=True)
            except DownTimeout:
                return None
        self._distribute(down)
        return down

    def await_down(self, timeout: float) -> DownBundle:
        """Receive, verify and distribute a DOWN without uploading anything.

        For a mule that already uploaded and is still owed an answer: a second
        :meth:`run_dock_cycle` would find nothing staged. Raises
        :class:`DownTimeout` if nothing arrives within ``timeout`` while the
        link is up, and :class:`ClientClusterError` otherwise.
        """
        down = self._recv_and_verify_down(timeout, survivable=True)
        self._distribute(down)
        return down

    # ---------------------------------------------- full dock cycle

    def run_dock_cycle(
        self, *, down_timeout_s: Optional[float] = None,
    ) -> Optional[DownBundle]:
        """Drive COLLECT -> UP -> DOWN -> VERIFY -> DISTRIBUTE.

        Requires a prior ``collect()`` unless the retry queue is non-empty.
        Returns the verified ``DownBundle`` on success, or ``None`` if the
        upload succeeded but the server sent nothing (degenerate case).
        Raises ``ClientClusterError`` on unrecoverable failure, and
        :class:`DownTimeout` on a DOWN wait that ran out under
        ``recoverable_down_wait``. ``down_timeout_s`` overrides the DOWN wait
        for this cycle (default ``self.down_timeout_s``).
        """
        if not self.dock.is_available():
            raise ClientClusterError("dock not available at run_dock_cycle entry")

        if self.recoverable_down_wait:
            # Anything queued before this upload answers an older one.
            self._drop_queued_downs("before UP")

        # ---- Build / gather the outbound queue (retries first, then fresh) ----
        bundles_to_send: List[UpBundle] = []
        with self._lock:
            for pend in list(self._retry_queue):
                bundles_to_send.append(pend.bundle)
            fresh = self._build_up_bundle_locked()
            if fresh is not None:
                bundles_to_send.append(fresh)

        if not bundles_to_send:
            raise ClientClusterError(
                "run_dock_cycle: nothing staged and retry queue empty"
            )

        # ---- UP: ship every outbound bundle --------------------------------
        sent_ok = self._send_bundles(bundles_to_send)
        if not sent_ok:
            # Nothing got through — leave retry queue in place, bail out.
            return None

        # ---- DOWN + VERIFY + DISTRIBUTE ------------------------------------
        down = self._recv_and_verify_down(
            down_timeout_s, survivable=self.recoverable_down_wait,
        )
        self._distribute(down)
        return down

    # ---------------------------------------------- internals

    def _build_up_bundle_locked(self) -> Optional[UpBundle]:
        """Build the fresh UpBundle (if anything is staged). Holds the lock."""
        if self._staged_aggregate is None:
            return None
        assert self._staged_report is not None
        assert self._staged_contacts is not None
        bundle = UpBundle(
            mule_id=self.mule_id,
            partial_aggregate=self._staged_aggregate,
            round_close_report=self._staged_report,
            contact_history=self._staged_contacts,
            prev_mission_delivery_report=self._staged_delivery_report,
            sim_upload_ts=self._staged_sim_upload_ts,
            backhaul=self._staged_backhaul,
        )
        sign_up_bundle(bundle)

        # Clear staged values so a re-run without a new collect won't resend.
        self._staged_aggregate = None
        self._staged_report = None
        self._staged_contacts = None
        self._staged_delivery_report = None
        self._staged_sim_upload_ts = None
        self._staged_backhaul = None
        return bundle

    def _send_bundles(self, bundles: List[UpBundle]) -> bool:
        """Ship bundles in order. Failures go to / stay in the retry queue.

        Returns True iff at least one bundle made it across.
        """
        with self._lock:
            self._set_state(ClientClusterState.UP)

        landed = 0
        new_retry: List[_PendingUp] = []

        for bundle in bundles:
            try:
                self.dock.client_send_up(bundle)
                landed += 1
                log.info(
                    "UP ok: mule=%s round=%d bytes=%d",
                    bundle.mule_id,
                    bundle.partial_aggregate.mission_round,
                    sum(int(w.nbytes) for w in bundle.partial_aggregate.weights),
                )
            except DockLinkError as e:
                log.warning(
                    "UP failed for mule=%s round=%d: %s",
                    bundle.mule_id,
                    bundle.partial_aggregate.mission_round,
                    e,
                )
                new_retry.append(self._bump_retry_entry(bundle))

        with self._lock:
            # Drop the successfully-sent entries from the retry queue by
            # replacing it with what we built. Anything not seen falls off.
            self._retry_queue = new_retry

        # Any bundle that hit max attempts raises, surface to caller
        for pend in new_retry:
            if pend.attempts >= self.max_retry_attempts:
                raise ClientClusterError(
                    f"UP bundle hit max_retry_attempts "
                    f"({self.max_retry_attempts}); mule={pend.bundle.mule_id} "
                    f"round={pend.bundle.partial_aggregate.mission_round}"
                )

        return landed > 0

    def _bump_retry_entry(self, bundle: UpBundle) -> _PendingUp:
        """Look up an existing retry entry for this bundle or create one."""
        with self._lock:
            for pend in self._retry_queue:
                if (
                    pend.bundle.mule_id == bundle.mule_id
                    and pend.bundle.partial_aggregate.mission_round
                    == bundle.partial_aggregate.mission_round
                ):
                    pend.attempts += 1
                    return pend
            return _PendingUp(
                bundle=bundle,
                first_attempt_at=time.time(),
                attempts=1,
            )

    def _recv_and_verify_down(
        self, timeout: Optional[float] = None, *, survivable: bool = False,
    ) -> DownBundle:
        """Block for this mule's DOWN, keep the newest queued, and verify it.

        ``timeout`` defaults to ``down_timeout_s``. With ``survivable`` a wait
        that runs out while the link is up raises :class:`DownTimeout`; every
        other failure, and any timeout without it, raises a plain
        :class:`ClientClusterError` with the message it always had.
        """
        with self._lock:
            self._set_state(ClientClusterState.DOWN)
        wait_s = self.down_timeout_s if timeout is None else timeout
        try:
            down = self.dock.client_recv_down(self.mule_id, timeout=wait_s)
        except DockLinkError as e:
            if (
                survivable
                and isinstance(e, DockLinkTimeout)
                and self.dock.is_available()
            ):
                raise DownTimeout(f"DOWN recv failed: {e}") from e
            raise ClientClusterError(f"DOWN recv failed: {e}") from e
        down = self._newest_of(down)

        with self._lock:
            self._set_state(ClientClusterState.VERIFY)
        if not verify_down_bundle(down):
            log.warning(
                "DOWN verify FAILED mule=%s round=%d — refusing handoff",
                down.mule_id,
                down.mission_slice.issued_round,
            )
            raise ClientClusterError("DOWN bundle signature verification failed")
        if down.mule_id != self.mule_id:
            raise ClientClusterError(
                f"DOWN routed to wrong mule: got {down.mule_id} expected "
                f"{self.mule_id}"
            )
        log.info(
            "DOWN verified mule=%s round=%d slice_size=%d",
            down.mule_id,
            down.mission_slice.issued_round,
            len(down.mission_slice.device_ids),
        )
        with self._lock:
            self._last_down = down
        return down

    def _queued_downs(self) -> List[DownBundle]:
        """DOWNs already queued for this mule; a failing drain counts as none."""
        try:
            return list(self.dock.client_drain_down(self.mule_id))
        except DockLinkError:
            return []

    def _newest_of(self, first: DownBundle) -> DownBundle:
        """``first`` or a DOWN queued behind it, whichever is newest.

        Newest is the highest ``issued_round``, the later arrival on a tie: a
        mule acts on one θ per dock, and an older one would carry a basis the
        cluster has already moved past. A lone mule never has a second DOWN
        queued, so this returns ``first`` for it.
        """
        queued = self._queued_downs()
        if not queued:
            return first
        candidates = [first] + queued
        best = max(
            range(len(candidates)),
            key=lambda i: (int(candidates[i].mission_slice.issued_round), i),
        )
        self.stale_downs_dropped += len(candidates) - 1
        log.info(
            "DOWN: mule=%s kept round=%d, dropped %d older queued bundle(s)",
            self.mule_id,
            candidates[best].mission_slice.issued_round,
            len(candidates) - 1,
        )
        return candidates[best]

    def _drop_queued_downs(self, when: str) -> int:
        """Throw away every DOWN queued now; returns how many."""
        stale = self._queued_downs()
        if stale:
            self.stale_downs_dropped += len(stale)
            log.info(
                "DOWN: mule=%s dropped %d stale bundle(s) %s (rounds %s)",
                self.mule_id, len(stale), when,
                [d.mission_slice.issued_round for d in stale],
            )
        return len(stale)

    def _distribute(self, down: DownBundle) -> None:
        with self._lock:
            self._set_state(ClientClusterState.DISTRIBUTE)

        # Slow-phase scheduler trigger
        try:
            self.distributor.on_slice_and_amendment(
                down.mission_slice, down.cluster_amendments
            )
        except Exception:
            log.exception("scheduler slow-phase sink raised; continuing")

        # Next-round model state into HFLHostMission. Its version first, so the
        # model sink can stage the two together.
        try:
            self.distributor.on_model_version(int(down.mission_slice.issued_round))
        except Exception:
            log.exception("model-version sink raised; continuing")
        try:
            self.distributor.on_next_round_model(down.theta_disc, down.synth_batch)
        except Exception:
            log.exception("mission-model sink raised; continuing")

        with self._lock:
            self._set_state(ClientClusterState.AWAIT_DOCK)

    def _set_state(self, new: ClientClusterState) -> None:
        if self._state != new:
            log.debug(
                "ClientCluster mule=%s state %s -> %s",
                self.mule_id,
                self._state.value,
                new.value,
            )
            self._state = new
