"""Dock link — Mule (ClientCluster) <-> Edge Server (HFLHostCluster).

This is the *only* transport Phase 1 needs. The cluster server side calls
``recv_up`` to ingest a mule's mission output and ``send_down`` to dispatch
the next-mission bundle. The mule client side (Phase 3) calls the inverse.

Phase 0 ships an in-process loopback (``LoopbackDockLink``). Phase 6
swaps in a real wired/high-bw transport behind the same ``DockLink`` ABC.

Design refs:
* HERMES_FL_Scheduler_Design.md §6.9 (interface contracts)
* HERMES_FL_Scheduler_Implementation_Plan.md §3 Phase 0 / Phase 1

FeRRy Phase 3, unit U9: :class:`DockClockMarker` is the one dock message that
is not a bundle. With several mules on the simulated mission clock the
cluster folds uploads in simulated-time order and needs to know, for every
mule, how early an upload it may still send; see the class docstring. The
recorded dock neither sends nor receives one.
"""

from __future__ import annotations

import math
import numbers
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

from hermes.types import DownBundle, MuleID, UpBundle


#: :class:`DockClockMarker` kinds a mule sends (``DockLink.client_send_clock``).
MARKER_CLOCK = "clock"
MARKER_DONE = "done"
#: Kinds only a dock server makes, in its own receive queue.
MARKER_REGISTERED = "registered"
MARKER_DEPARTED = "departed"
MULE_MARKER_KINDS = (MARKER_CLOCK, MARKER_DONE)
MARKER_KINDS = (MARKER_CLOCK, MARKER_DONE, MARKER_REGISTERED, MARKER_DEPARTED)


@dataclass(frozen=True)
class DockClockMarker:
    """What the dock knows of a mule's simulated time, without a bundle (FeRRy Phase 3, U9).

    With several mules on the simulated mission clock the cluster folds their
    uploads in simulated-time order: it holds each UP until no other mule can
    still send one that completed earlier, the conservative rule of
    conservative parallel discrete-event simulation (Chandy and Misra 1979;
    Fujimoto, *Parallel and Distributed Simulation Systems*, 2000, ch. 3). For
    that it needs, per mule, a lower bound on the ``sim_upload_ts`` of every
    UP the mule may still send. An UP carries one itself (a mule's clock never
    goes back, so its later UPs complete no earlier); a marker carries one
    without a bundle:

    * ``clock`` (from the mule): every later UP of this mule completes at or
      after ``sim_ts``;
    * ``done`` (from the mule): it will send no more UPs; ``sim_ts``, if
      given, is its final clock, for the trace;
    * ``registered`` (made by the TCP dock server): the mule registered
      connection number ``session``; nothing is known yet of its time;
    * ``departed`` (made by the TCP dock server): connection ``session``
      ended and no newer one of the mule replaced it, so the mule can send
      nothing more. A mule that finishes its run, fails, crashes or is
      killed departs this way, so the cluster never waits for a mule that
      has exited, whether or not the mule said ``done``.

    A server built with ``sim_markers=True`` puts every marker in the queue
    its UPs go through, so one mule's markers and UPs reach the cluster in
    the order they happened. ``sim_ts`` is simulated seconds on the mission
    clock; ``session`` is set by the server, never trusted from a mule.
    """

    mule_id: MuleID
    kind: str
    sim_ts: Optional[float] = None
    session: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in MARKER_KINDS:
            raise ValueError(f"marker kind must be one of {MARKER_KINDS}, got {self.kind!r}")
        ts = self.sim_ts
        if ts is not None:
            if isinstance(ts, bool) or not isinstance(ts, numbers.Real):
                raise TypeError(f"marker sim_ts must be a number or None, got {ts!r}")
            if not math.isfinite(float(ts)):
                raise ValueError(f"marker sim_ts must be finite, got {ts!r}")
            object.__setattr__(self, "sim_ts", float(ts))
        if self.kind == MARKER_CLOCK and self.sim_ts is None:
            raise ValueError("a clock marker needs its sim_ts")


#: What ``DockLink.recv_dock_event`` returns.
DockEvent = Union[UpBundle, DockClockMarker]


class DockLinkError(RuntimeError):
    """Raised when a dock-link operation fails (timeout, drop, etc.)."""


class DockLinkTimeout(DockLinkError):
    """A blocking receive ran out of time; the link itself may still be up.

    A subclass, so every caller that catches :class:`DockLinkError` keeps
    working. It exists so a mule can tell "the cluster has not answered yet"
    (worth waiting on, FeRRy Phase 2) from "the link is gone" (not).
    """


def _drain(q: "queue.Queue[DownBundle]") -> List[DownBundle]:
    """Everything queued right now, oldest first, without blocking."""
    out: List[DownBundle] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


class DockLink(ABC):
    """Symmetric dock-link ABC.

    The cluster (``HFLHostCluster``) is the *server* — Phase 1 used only
    ``recv_up`` / ``send_down``. The mule (``ClientCluster``) is the
    *client* and uses ``client_send_up`` / ``client_recv_down``.

    Kept as a single ABC so tests and demos can drive both sides through
    one loopback instance; real transports may subclass and route the
    client vs server calls onto different underlying sockets.
    """

    # ---- cluster (server) side ---------------------------------------------

    @abstractmethod
    def recv_up(self, timeout: Optional[float] = None) -> UpBundle:
        """Cluster-side: block until an UP bundle arrives."""

    @abstractmethod
    def send_down(self, bundle: DownBundle) -> None:
        """Cluster-side: dispatch a DOWN bundle to the awaiting mule."""

    # ---- mule (client) side -------------------------------------------------

    @abstractmethod
    def client_send_up(self, bundle: UpBundle) -> None:
        """Mule-side: push an UP bundle to the cluster."""

    @abstractmethod
    def client_recv_down(
        self, mule_id: MuleID, timeout: Optional[float] = None
    ) -> DownBundle:
        """Mule-side: block until this mule's DOWN bundle arrives."""

    def client_drain_down(self, mule_id: MuleID) -> List[DownBundle]:
        """Mule-side: take every DOWN already queued for this mule, oldest first.

        Never blocks. A mule calls it to throw away a DOWN that answered an
        upload it stopped waiting for, and to pick the newest of several
        (FeRRy Phase 2). Not abstract: a transport that cannot queue more than
        one bundle has nothing to drain, so the default returns nothing.
        """
        return []

    def client_send_clock(
        self, mule_id: MuleID, sim_ts: Optional[float], *, done: bool = False,
    ) -> bool:
        """Mule-side: report this mule's simulated time without a bundle (FeRRy Phase 3, U9).

        Sends a :class:`DockClockMarker`: ``clock`` (every later UP of this
        mule completes at or after ``sim_ts``) or, with ``done``, ``done``
        (no more UPs). Returns True when it was sent. Not abstract: a
        transport that carries no markers sends nothing and returns False, and
        a cluster then learns a mule's time from its UPs and its departure
        only.
        """
        return False

    def recv_dock_event(self, timeout: Optional[float] = None) -> "DockEvent":
        """Cluster-side: the next UP bundle or :class:`DockClockMarker`, in arrival order.

        FeRRy Phase 3, U9. Not abstract: a transport without markers has only
        UPs, so the default is :meth:`recv_up`.
        """
        return self.recv_up(timeout)

    # ---- shared -------------------------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """True iff the link is currently usable (dock detector).

        Loopback: always True until ``close``. Real transport: reflects
        the physical dock state (connector seated, carrier detect, etc.).
        """

    @abstractmethod
    def close(self) -> None:
        """Release any underlying resources."""


class LoopbackDockLink(DockLink):
    """Thread-safe in-process loopback. **Tests + demos only.**

    Phase 7 retirement: this class is allowed in ``tests/`` and the
    ``hermes.{cluster,mule,mission}.__main__`` pedagogical demos, but
    must **not** be imported from any production runtime path
    (``hermes.processes.*`` and the supervised cluster / mule / device
    services). Production uses :class:`TCPDockLinkServer` +
    :class:`TCPDockLinkClient` — the real transport that survives
    multi-process deployment. The ``test_loopback_retirement.py``
    import-graph test pins this invariant.

    Both sides share one instance. The mule-side test helpers
    (``client_send_up`` / ``client_recv_down``) sit alongside the cluster
    methods so a single object plays both roles.

    Per-mule queues keep traffic isolated when several mules dock against
    one cluster in tests.
    """

    def __init__(self) -> None:
        self._up: "queue.Queue[UpBundle]" = queue.Queue()
        self._down_per_mule: dict[MuleID, "queue.Queue[DownBundle]"] = {}
        self._lock = threading.Lock()
        self._closed = False

    # ---- cluster (server) side ------------------------------------------------

    def recv_up(self, timeout: Optional[float] = None) -> UpBundle:
        if self._closed:
            raise DockLinkError("dock link closed")
        try:
            return self._up.get(timeout=timeout)
        except queue.Empty as e:
            raise DockLinkTimeout(f"recv_up timed out after {timeout}s") from e

    def send_down(self, bundle: DownBundle) -> None:
        if self._closed:
            raise DockLinkError("dock link closed")
        q = self._ensure_down_queue(bundle.mule_id)
        q.put(bundle)

    # ---- mule (client) side --- used by Phase 3 ClientCluster + tests --------

    def client_send_up(self, bundle: UpBundle) -> None:
        if self._closed:
            raise DockLinkError("dock link closed")
        self._up.put(bundle)

    def client_recv_down(
        self, mule_id: MuleID, timeout: Optional[float] = None
    ) -> DownBundle:
        if self._closed:
            raise DockLinkError("dock link closed")
        q = self._ensure_down_queue(mule_id)
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise DockLinkTimeout(
                f"client_recv_down for {mule_id!r} timed out after {timeout}s"
            ) from e

    def client_drain_down(self, mule_id: MuleID) -> List[DownBundle]:
        return _drain(self._ensure_down_queue(mule_id))

    # ---- shared --------------------------------------------------------------

    def is_available(self) -> bool:
        return not self._closed

    def close(self) -> None:
        self._closed = True

    def _ensure_down_queue(self, mule_id: MuleID) -> "queue.Queue[DownBundle]":
        with self._lock:
            q = self._down_per_mule.get(mule_id)
            if q is None:
                q = queue.Queue()
                self._down_per_mule[mule_id] = q
            return q
