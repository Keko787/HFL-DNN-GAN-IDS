"""Dock link — Mule (ClientCluster) <-> Edge Server (HFLHostCluster).

This is the *only* transport Phase 1 needs. The cluster server side calls
``recv_up`` to ingest a mule's mission output and ``send_down`` to dispatch
the next-mission bundle. The mule client side (Phase 3) calls the inverse.

Phase 0 ships an in-process loopback (``LoopbackDockLink``). Phase 6
swaps in a real wired/high-bw transport behind the same ``DockLink`` ABC.

Design refs:
* HERMES_FL_Scheduler_Design.md §6.9 (interface contracts)
* HERMES_FL_Scheduler_Implementation_Plan.md §3 Phase 0 / Phase 1
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

from hermes.types import DownBundle, MuleID, UpBundle


class DockLinkError(RuntimeError):
    """Raised when a dock-link operation fails (timeout, drop, etc.)."""


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
            raise DockLinkError(f"recv_up timed out after {timeout}s") from e

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
            raise DockLinkError(
                f"client_recv_down for {mule_id!r} timed out after {timeout}s"
            ) from e

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
