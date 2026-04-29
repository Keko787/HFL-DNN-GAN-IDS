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
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    DownBundle,
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


@dataclass
class BundleDistributor:
    """Holds the two intra-NUC callables the dock fan-out delivers into.

    Default sinks are no-ops so a ``ClientCluster`` constructed in tests
    doesn't need both callables wired up. Production builds (Phase 6)
    pass real scheduler / mission-server handles.
    """

    on_slice_and_amendment: SchedulerSlowPhaseSink = field(
        default=lambda _s, _a: None
    )
    on_next_round_model: MissionModelSink = field(
        default=lambda _w, _b: None
    )


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
    """Mule-NUC dock handler. One instance per mule."""

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
    ) -> None:
        self.mule_id = mule_id
        self.dock = dock
        self.distributor = distributor or BundleDistributor()
        self.dock_poll_interval_s = dock_poll_interval_s
        self.up_timeout_s = up_timeout_s
        self.down_timeout_s = down_timeout_s
        self.max_retry_attempts = max_retry_attempts

        self._lock = threading.RLock()
        self._state: ClientClusterState = ClientClusterState.AWAIT_DOCK

        # Pending round-output snapshot. ``collect()`` pushes into this;
        # ``run_dock_cycle`` pulls from it.
        self._staged_aggregate: Optional[PartialAggregate] = None
        self._staged_report: Optional[MissionRoundCloseReport] = None
        self._staged_contacts: Optional[ContactHistory] = None

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

    # ---------------------------------------------- COLLECT

    def collect(
        self,
        *,
        partial_aggregate: PartialAggregate,
        report: MissionRoundCloseReport,
        contacts: ContactHistory,
    ) -> None:
        """Stage the latest mission output. Overwrites any prior stage.

        Called by the mule supervisor (or the demo) whenever
        ``HFLHostMission.close_round`` yields a fresh triple.
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
            self._set_state(ClientClusterState.COLLECT)
            log.info(
                "collect: mule=%s mission_round=%d accepted=%d lines=%d",
                self.mule_id,
                partial_aggregate.mission_round,
                partial_aggregate.num_examples,
                len(report.lines),
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

    # ---------------------------------------------- full dock cycle

    def run_dock_cycle(self) -> Optional[DownBundle]:
        """Drive COLLECT -> UP -> DOWN -> VERIFY -> DISTRIBUTE.

        Requires a prior ``collect()`` unless the retry queue is non-empty.
        Returns the verified ``DownBundle`` on success, or ``None`` if the
        upload succeeded but the server sent nothing (degenerate case).
        Raises ``ClientClusterError`` on unrecoverable failure.
        """
        if not self.dock.is_available():
            raise ClientClusterError("dock not available at run_dock_cycle entry")

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
        down = self._recv_and_verify_down()
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
        )
        sign_up_bundle(bundle)

        # Clear staged values so a re-run without a new collect won't resend.
        self._staged_aggregate = None
        self._staged_report = None
        self._staged_contacts = None
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

    def _recv_and_verify_down(self) -> DownBundle:
        with self._lock:
            self._set_state(ClientClusterState.DOWN)
        try:
            down = self.dock.client_recv_down(
                self.mule_id, timeout=self.down_timeout_s
            )
        except DockLinkError as e:
            raise ClientClusterError(f"DOWN recv failed: {e}") from e

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

        # Next-round model state into HFLHostMission
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
