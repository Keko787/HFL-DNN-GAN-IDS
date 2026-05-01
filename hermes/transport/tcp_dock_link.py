"""Sprint 2 — TCP-backed DockLink for the multi-process AVN topology.

Replaces the in-process ``LoopbackDockLink`` with a real TCP transport
between each mule (client) and the cluster (server). Designed for
bursty large-bundle traffic — UpBundles ship at end of Pass 1 + after
Pass 2; DownBundles ship between Pass 1 and Pass 2 + as the cluster
reshuffles slices.

Wire model:

* Cluster binds + listens on a port. Each mule's ``ClientCluster``
  connects as a TCP client and sends a ``_MuleRegistrationMessage``
  first so the cluster can map socket → MuleID and route DOWN bundles
  back to the right mule.
* The cluster's ``send_down`` writes a frame on the matching mule's
  socket. ``recv_up`` reads from a shared queue that every per-mule
  reader thread populates.
* The mule's ``client_send_up`` writes synchronously on its single
  socket. ``client_recv_down`` reads from a queue populated by its
  reader thread.

Compared to :class:`TCPRFLinkServer`:

* No channel emulator — the dock link models a wired/high-bandwidth
  hop (mule docks at the edge server). Loss + jitter are not part
  of the design assumption here.
* Bundles can be very large (hundreds of MB for a real model). The
  framing in :mod:`wire` handles up to 256 MiB, which is enough for
  the Sprint 2 demos; larger bundles want a streaming variant later.
"""

from __future__ import annotations

import logging
import queue
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from hermes.types import DownBundle, MuleID, UpBundle

from .dock_link import DockLink, DockLinkError
from .wire import WireError, recv_message, send_message

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Registration handshake — first frame on every mule socket
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _MuleRegistrationMessage:
    """First frame sent by a connecting mule's ClientCluster.

    Wire-internal — the cluster reads this to populate its
    socket→MuleID map; without it we couldn't route DOWN bundles back.
    """

    mule_id: MuleID


# --------------------------------------------------------------------------- #
# Server side — runs on the edge-server (HFLHostCluster) AVN
# --------------------------------------------------------------------------- #

class TCPDockLinkServer(DockLink):
    """Cluster-side TCP DockLink. Binds on construction; accept loop on start.

    Lifecycle:

    1. ``__init__`` binds a listener socket on ``(host, port)``.
    2. ``start()`` spawns the accept loop. Each accepted connection
       reads a ``_MuleRegistrationMessage`` and then pumps inbound
       UpBundles into a shared queue.
    3. ``recv_up`` blocks on the shared queue. ``send_down`` looks up
       the registered socket for the bundle's mule_id and writes.
    4. ``close()`` shuts down the listener + every mule socket.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        accept_timeout_s: float = 0.25,
        send_timeout_s: float = 60.0,
    ) -> None:
        self._host = host
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(8)
        self._listener.settimeout(accept_timeout_s)
        self._port: int = self._listener.getsockname()[1]

        # S2-M3: dock bundles are bigger than RF messages (whole-model
        # uploads); 60 s is the dock send-timeout default.
        self._send_timeout_s = send_timeout_s

        self._lock = threading.RLock()
        self._closed = threading.Event()

        self._sockets: Dict[MuleID, socket.socket] = {}
        self._reader_threads: Dict[MuleID, threading.Thread] = {}
        self._up_q: "queue.Queue[UpBundle]" = queue.Queue()

        self._accept_thread: Optional[threading.Thread] = None
        # S2-M4 / S2-H3 — same pattern as TCPRFLinkServer.
        self._registration_cv = threading.Condition(self._lock)
        self._last_accept_error: Optional[BaseException] = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        if self._accept_thread is not None:
            return
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="TCPDockLinkServer-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def wait_for_mules(self, mule_ids: List[MuleID], timeout: float = 5.0) -> bool:
        """Block until every named mule has registered, or timeout.

        Returns True iff all expected mules are connected.

        S2-M4: notified by the registration handler instead of polling.
        """
        import time as _time
        wanted = set(mule_ids)
        deadline = _time.time() + timeout
        with self._registration_cv:
            while True:
                got = set(self._sockets.keys())
                if wanted.issubset(got):
                    return True
                remaining = deadline - _time.time()
                if remaining <= 0:
                    return False
                self._registration_cv.wait(timeout=remaining)

    @property
    def last_accept_error(self) -> Optional[BaseException]:
        """S2-H3: most recent fault from the accept loop (None if clean)."""
        with self._lock:
            return self._last_accept_error

    def registered_mules(self) -> List[MuleID]:
        """L-H2: snapshot of mules currently holding a docked socket.

        The set may grow (new mule docks) or shrink (existing mule's
        reader loop ended on WireError). Cluster services use this to
        detect mid-flight reconnects and re-dispatch DOWN bundles.
        """
        with self._lock:
            return list(self._sockets.keys())

    # ------------------------------------------------------------------ #
    # Cluster (server) interface
    # ------------------------------------------------------------------ #

    def recv_up(self, timeout: Optional[float] = None) -> UpBundle:
        self._raise_if_closed()
        try:
            return self._up_q.get(timeout=timeout)
        except queue.Empty as e:
            raise DockLinkError(f"recv_up timed out after {timeout}s") from e

    def send_down(self, bundle: DownBundle) -> None:
        self._raise_if_closed()
        sock = self._socket_for(bundle.mule_id)
        try:
            send_message(sock, bundle)
        except WireError as e:
            self._drop_mule(bundle.mule_id)
            raise DockLinkError(
                f"send_down to {bundle.mule_id!r} failed: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Mule-side methods — not implemented on the server
    # ------------------------------------------------------------------ #

    def client_send_up(self, *_a, **_kw):
        raise NotImplementedError(
            "TCPDockLinkServer is the cluster side; use TCPDockLinkClient on mules"
        )

    def client_recv_down(self, *_a, **_kw):
        raise NotImplementedError(
            "TCPDockLinkServer is the cluster side; use TCPDockLinkClient on mules"
        )

    # ------------------------------------------------------------------ #
    # Shared
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        # S2-M6: "available" means the listener is up and accepting
        # connections — NOT that any specific mule is currently docked.
        # Per-mule connectivity is exposed via wait_for_mules. This
        # mirrors LoopbackDockLink's "always True until close" semantics
        # so callers don't need to special-case the transport.
        return not self._closed.is_set()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._listener.close()
        except OSError:
            pass
        with self._lock:
            for sock in self._sockets.values():
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
            self._sockets.clear()
            self._reader_threads.clear()

    # ------------------------------------------------------------------ #
    # Internal — accept + per-mule reader
    # ------------------------------------------------------------------ #

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError as e:
                # S2-H3: surface, don't silently exit.
                if not self._closed.is_set():
                    log.error(
                        "TCPDockLinkServer accept loop dying: %s", e,
                    )
                    with self._lock:
                        self._last_accept_error = e
                break
            try:
                self._spawn_reader(conn)
            except Exception as e:  # pragma: no cover — defensive
                log.exception("TCPDockLinkServer _spawn_reader raised")
                with self._lock:
                    self._last_accept_error = e

    def _spawn_reader(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(2.0)
            reg = recv_message(conn)
        except (WireError, OSError) as e:
            log.warning("rejected unregistered mule: %s", e)
            try:
                conn.close()
            except OSError:
                pass
            return

        if not isinstance(reg, _MuleRegistrationMessage):
            log.warning("rejected non-registration first frame: %r", reg)
            conn.close()
            return

        mid = reg.mule_id
        # Reader stays blocking-forever — mules can sit idle between
        # missions on a long-lived dock connection. Peer-vanish
        # surfaces via WireError on the next frame.
        conn.settimeout(None)
        # S2-M3: bound the dock SEND-timeout via SO_SNDTIMEO so a
        # stuck recipient doesn't hang the cluster's send_down. Linux
        # uses SO_SNDTIMEO seconds; Windows uses milliseconds — we set
        # a struct-format value compatible with both via socket.timeval.
        try:
            tv = struct.pack(
                "ll",
                int(self._send_timeout_s),
                int((self._send_timeout_s % 1) * 1_000_000),
            )
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, tv)
        except OSError:
            # SO_SNDTIMEO can fail on platforms that ignore the option;
            # fall back to relying on the peer to drain.
            log.debug("SO_SNDTIMEO not honoured on this platform")
        with self._registration_cv:
            self._sockets[mid] = conn
            t = threading.Thread(
                target=self._reader_loop,
                args=(mid, conn),
                name=f"TCPDockLinkServer-reader-{mid}",
                daemon=True,
            )
            self._reader_threads[mid] = t
            # S2-M4: wake any wait_for_mules caller blocked on this mule.
            self._registration_cv.notify_all()
        t.start()
        log.info("TCPDockLinkServer registered mule %s", mid)

    def _reader_loop(self, mule_id: MuleID, conn: socket.socket) -> None:
        while not self._closed.is_set():
            try:
                msg = recv_message(conn)
            except WireError:
                break

            if isinstance(msg, UpBundle):
                self._up_q.put(msg)
            else:
                log.warning(
                    "TCPDockLinkServer ignored unexpected message %s from %s",
                    type(msg).__name__, mule_id,
                )
        self._drop_mule(mule_id)
        log.info("TCPDockLinkServer reader for %s exiting", mule_id)

    def _drop_mule(self, mule_id: MuleID) -> None:
        with self._lock:
            sock = self._sockets.pop(mule_id, None)
            self._reader_threads.pop(mule_id, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _socket_for(self, mule_id: MuleID) -> socket.socket:
        with self._lock:
            sock = self._sockets.get(mule_id)
        if sock is None:
            raise DockLinkError(
                f"no registered socket for {mule_id!r} on this cluster"
            )
        return sock

    def _raise_if_closed(self) -> None:
        if self._closed.is_set():
            raise DockLinkError("dock link closed")


# --------------------------------------------------------------------------- #
# Client side — runs on each mule's NUC (ClientCluster)
# --------------------------------------------------------------------------- #

class TCPDockLinkClient(DockLink):
    """Mule-side TCP DockLink.

    Lifecycle:

    1. ``__init__(mule_id, host, port)`` opens a TCP connection,
       sends the registration message, and starts a reader thread
       that pumps inbound DownBundles into a queue.
    2. ``client_send_up`` writes synchronously. ``client_recv_down``
       blocks on the queue.
    3. ``close()`` tears the connection down.
    """

    def __init__(
        self,
        mule_id: MuleID,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self._mule_id = mule_id
        self._closed = threading.Event()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(connect_timeout_s)
        self._sock.connect((host, port))
        self._sock.settimeout(None)  # bundles can be large

        send_message(self._sock, _MuleRegistrationMessage(mule_id=mule_id))

        self._down_q: "queue.Queue[DownBundle]" = queue.Queue()
        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"TCPDockLinkClient-{mule_id}",
            daemon=True,
        )
        self._reader.start()

    @property
    def mule_id(self) -> MuleID:
        return self._mule_id

    # ------------------------------------------------------------------ #
    # Mule (client) interface
    # ------------------------------------------------------------------ #

    def client_send_up(self, bundle: UpBundle) -> None:
        self._raise_if_closed()
        if bundle.mule_id != self._mule_id:
            raise DockLinkError(
                f"client_send_up: bundle.mule_id={bundle.mule_id!r} "
                f"!= client.mule_id={self._mule_id!r}"
            )
        try:
            send_message(self._sock, bundle)
        except WireError as e:
            self._closed.set()
            raise DockLinkError(f"client_send_up failed: {e}") from e

    def client_recv_down(
        self, mule_id: MuleID, timeout: Optional[float] = None
    ) -> DownBundle:
        self._raise_if_closed()
        if mule_id != self._mule_id:
            raise DockLinkError(
                f"client_recv_down for {mule_id!r} on client {self._mule_id!r}"
            )
        try:
            return self._down_q.get(timeout=timeout)
        except queue.Empty as e:
            raise DockLinkError(
                f"client_recv_down for {mule_id!r} timed out after {timeout}s"
            ) from e

    # ------------------------------------------------------------------ #
    # Cluster-side methods — not implemented on the client
    # ------------------------------------------------------------------ #

    def recv_up(self, *_a, **_kw):
        raise NotImplementedError(
            "TCPDockLinkClient is the mule side; use TCPDockLinkServer on the cluster"
        )

    def send_down(self, *_a, **_kw):
        raise NotImplementedError(
            "TCPDockLinkClient is the mule side; use TCPDockLinkServer on the cluster"
        )

    # ------------------------------------------------------------------ #
    # Shared
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        return not self._closed.is_set()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _reader_loop(self) -> None:
        while not self._closed.is_set():
            try:
                msg = recv_message(self._sock)
            except WireError:
                break
            if isinstance(msg, DownBundle):
                self._down_q.put(msg)
            else:
                log.warning(
                    "TCPDockLinkClient %s: unexpected message %s",
                    self._mule_id, type(msg).__name__,
                )
        self._closed.set()
        log.info("TCPDockLinkClient %s reader exiting", self._mule_id)

    def _raise_if_closed(self) -> None:
        if self._closed.is_set():
            raise DockLinkError(f"dock link to {self._mule_id!r} closed")
