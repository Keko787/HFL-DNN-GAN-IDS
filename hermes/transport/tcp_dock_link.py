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

FeRRy Phase 3, unit U9 (``TCPDockLinkServer(sim_markers=True)``, off by
default). The server also queues :class:`~hermes.transport.dock_link.DockClockMarker`
records, in the queue the UPs go through: ``registered`` when a mule's
connection registers, the ``clock`` and ``done`` markers a mule sends
(``TCPDockLinkClient.client_send_clock``), and ``departed`` when a mule's
connection ends without a newer one replacing it. Each is queued by the
thread that orders it against the mule's UPs (the accept thread under the
registration lock, or the mule's own reader), so the cluster reads a mule's
registration, UPs, markers and departure in the order they happened, and can
never take a mule for gone while one of its UPs is still queued. Off, no
marker is made, and a marker a mule sends is ignored like any unexpected
frame, as before.
"""

from __future__ import annotations

import logging
import math
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from hermes.types import DownBundle, MuleID, UpBundle

from .dock_link import (
    MARKER_CLOCK,
    MARKER_DEPARTED,
    MARKER_DONE,
    MARKER_REGISTERED,
    MULE_MARKER_KINDS,
    DockClockMarker,
    DockLink,
    DockLinkError,
    DockLinkTimeout,
    _drain,
)
from .wire import WireError, recv_message, send_message

log = logging.getLogger(__name__)


def _close_socket(sock: socket.socket) -> None:
    """Shut down and close ``sock``, ignoring a socket that is already gone.

    The shutdown matters for a socket another thread is blocked reading with
    no timeout: on Linux, ``close`` alone does not wake that ``recv``;
    ``shutdown`` does, on every platform.
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


# Largest SO_SNDTIMEO Winsock accepts: a DWORD of milliseconds (~49.7 days).
_WINSOCK_MAX_MS = 0xFFFFFFFF


def sndtimeo_optval(timeout_s: Optional[float], *, platform: Optional[str] = None) -> bytes:
    """The ``SO_SNDTIMEO`` option value that bounds one blocking send.

    The option's type differs by OS. Winsock reads a DWORD of milliseconds
    (Microsoft, "SOL_SOCKET socket options"); POSIX reads a ``struct
    timeval`` of seconds and microseconds (socket(7)), whose layout two
    native longs match on Linux and 64-bit macOS. The dock link used to pack
    a timeval everywhere, so Windows read ``tv_sec`` as milliseconds: its
    60 s bound was 60 ms, and a bound under 1 s packed ``tv_sec = 0``, which
    Winsock reads as no bound at all (Freeze Amendment 10).

    ``None`` or 0 packs 0, which both APIs read as "block forever". A
    positive bound is rounded to the API's unit but never below one unit,
    so it never packs as 0. ``platform`` defaults to ``sys.platform``;
    tests pass it to check both encodings on one host.
    """
    plat = sys.platform if platform is None else platform
    t = 0.0 if timeout_s is None else float(timeout_s)
    if math.isnan(t) or t < 0.0 or math.isinf(t):
        raise ValueError(f"send timeout must be a finite value >= 0, got {timeout_s!r}")
    if plat.startswith("win"):
        ms = min(int(round(t * 1000.0)), _WINSOCK_MAX_MS)
        if t > 0.0 and ms == 0:
            ms = 1
        return struct.pack("=I", ms)
    sec = int(t)
    usec = int(round((t - sec) * 1_000_000))
    if usec >= 1_000_000:
        sec, usec = sec + 1, 0
    if t > 0.0 and sec == 0 and usec == 0:
        usec = 1
    return struct.pack("ll", sec, usec)


def set_send_timeout(sock: socket.socket, timeout_s: Optional[float]) -> bool:
    """Bound every blocking send on ``sock`` to ``timeout_s`` (SO_SNDTIMEO).

    Used on sockets whose reads block with no timeout (``settimeout(None)``),
    so a reader can sit idle indefinitely while a send to a stuck peer still
    fails, with an ``OSError`` that ``wire.send_message`` turns into a
    ``WireError``. The option does nothing on a socket with a Python-level
    timeout, which is non-blocking underneath.

    The bound is per send call, not per ``sendall``: Winsock fails a call
    that has not finished in time, while Linux returns the bytes a call got
    out and ``sendall`` carries on, so there only a peer that takes nothing
    for ``timeout_s`` fails the frame. Either way a stuck peer cannot hold a
    sender for long; a slow one is not cut off mid-frame on Linux.

    Returns False when the platform refuses the option; the send is then
    bounded only by the peer draining it, as before.
    """
    try:
        sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDTIMEO, sndtimeo_optval(timeout_s),
        )
    except OSError:
        return False
    return True


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

    ``sim_markers`` (FeRRy Phase 3, unit U9; off by default): queue the
    mules' :class:`DockClockMarker` records with their UPs (module
    docstring); read them with :meth:`recv_dock_event`. :meth:`recv_up` still
    returns UP bundles only, skipping any marker.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        accept_timeout_s: float = 0.25,
        send_timeout_s: float = 60.0,
        sim_markers: bool = False,
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

        # FeRRy Phase 3, unit U9: the in-band markers. Each registration is a
        # numbered session; ``_sessions`` holds each mule's latest, so the end
        # of an older connection is not taken for the mule's departure.
        self._sim_markers = bool(sim_markers)
        self._session_seq = 0
        self._sessions: Dict[MuleID, int] = {}

    @property
    def sim_markers(self) -> bool:
        """True when the server queues :class:`DockClockMarker` records (FeRRy Phase 3, U9)."""
        return self._sim_markers

    def session_of(self, mule_id: MuleID) -> Optional[int]:
        """The session number of ``mule_id``'s current connection (FeRRy Phase 3, U9).

        None without ``sim_markers``, or when the mule has no connection.
        """
        with self._lock:
            return self._sessions.get(mule_id)

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
        if self._sim_markers:
            return self._recv_up_past_markers(timeout)
        try:
            return self._up_q.get(timeout=timeout)
        except queue.Empty as e:
            raise DockLinkTimeout(f"recv_up timed out after {timeout}s") from e

    def recv_dock_event(self, timeout: Optional[float] = None):
        """The next UP bundle or :class:`DockClockMarker`, in the order queued (FeRRy Phase 3, U9).

        Without ``sim_markers`` the queue holds UPs only, so this is
        :meth:`recv_up`.
        """
        self._raise_if_closed()
        try:
            return self._up_q.get(timeout=timeout)
        except queue.Empty as e:
            raise DockLinkTimeout(f"recv_dock_event timed out after {timeout}s") from e

    def _recv_up_past_markers(self, timeout: Optional[float]) -> UpBundle:
        """:meth:`recv_up` on a server that queues markers: the next UP, markers dropped.

        A caller of ``recv_up`` asked for bundles only; the markers belong to
        :meth:`recv_dock_event`'s caller, so one reading them here has picked
        the wrong call, and they are dropped with a debug line. ``timeout``
        bounds the whole wait, markers included.
        """
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                item = self._up_q.get(timeout=remaining)
            except queue.Empty as e:
                raise DockLinkTimeout(f"recv_up timed out after {timeout}s") from e
            if isinstance(item, UpBundle):
                return item
            log.debug("recv_up skipped a dock marker: %r", item)

    def send_down(self, bundle: DownBundle) -> None:
        self._raise_if_closed()
        sock = self._socket_for(bundle.mule_id)
        try:
            send_message(sock, bundle)
        except WireError as e:
            self._drop_mule(bundle.mule_id, sock)
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
        # stuck recipient doesn't hang the cluster's send_down. Amendment
        # 10: packed per OS by sndtimeo_optval (Windows reads a DWORD of
        # milliseconds, POSIX a struct timeval); the timeval packed here
        # before gave Windows a 60 ms bound instead of 60 s.
        if not set_send_timeout(conn, self._send_timeout_s):
            # SO_SNDTIMEO can fail on platforms that ignore the option;
            # fall back to relying on the peer to drain.
            log.debug("SO_SNDTIMEO not honoured on this platform")
        with self._registration_cv:
            # A mule re-registering under the same id (restarted, re-docked)
            # replaces its old connection. The old one is closed below, so its
            # reader ends; that reader then leaves the new entry alone, because
            # _drop_mule only removes the socket it was given (FeRRy Phase 2).
            previous = self._sockets.get(mid)
            self._sockets[mid] = conn
            # FeRRy Phase 3, U9: number the session and queue its
            # ``registered`` marker before its reader can queue any UP.
            reader_kwargs = {}
            if self._sim_markers:
                reader_kwargs["session"] = self._open_session(mid)
            t = threading.Thread(
                target=self._reader_loop,
                args=(mid, conn),
                kwargs=reader_kwargs,
                name=f"TCPDockLinkServer-reader-{mid}",
                daemon=True,
            )
            self._reader_threads[mid] = t
            # S2-M4: wake any wait_for_mules caller blocked on this mule.
            self._registration_cv.notify_all()
        if previous is not None and previous is not conn:
            log.info("TCPDockLinkServer: mule %s re-registered; closing its old socket", mid)
            _close_socket(previous)
        t.start()
        log.info("TCPDockLinkServer registered mule %s", mid)

    def _reader_loop(
        self, mule_id: MuleID, conn: socket.socket, session: Optional[int] = None,
    ) -> None:
        while not self._closed.is_set():
            try:
                msg = recv_message(conn)
            except WireError:
                break

            if isinstance(msg, UpBundle):
                self._up_q.put(msg)
            elif self._sim_markers and isinstance(msg, DockClockMarker):
                self._queue_mule_marker(mule_id, msg, session)
            else:
                log.warning(
                    "TCPDockLinkServer ignored unexpected message %s from %s",
                    type(msg).__name__, mule_id,
                )
        self._drop_mule(mule_id, conn)
        if self._sim_markers:
            self._close_session(mule_id, session)
        log.info("TCPDockLinkServer reader for %s exiting", mule_id)

    # ------------------------------------------------------------------ #
    # FeRRy Phase 3, unit U9 — the in-band markers (``sim_markers``)
    # ------------------------------------------------------------------ #

    def _open_session(self, mule_id: MuleID) -> int:
        """Number a new connection of ``mule_id`` and queue its ``registered`` marker.

        The caller holds the lock, so the marker is queued in the order the
        registrations happened and before the session's reader starts.
        """
        self._session_seq += 1
        session = self._session_seq
        self._sessions[mule_id] = session
        self._up_q.put(DockClockMarker(mule_id=mule_id, kind=MARKER_REGISTERED, session=session))
        return session

    def _queue_mule_marker(
        self, mule_id: MuleID, marker: DockClockMarker, session: Optional[int],
    ) -> None:
        """Queue a marker a mule sent, stamped with its connection's session.

        Only ``clock`` and ``done`` come from a mule, and only for itself: a
        ``registered``/``departed`` marker or one naming another mule is
        dropped with a warning, so a mule cannot report for another one.
        """
        if marker.kind not in MULE_MARKER_KINDS or marker.mule_id != mule_id:
            log.warning(
                "TCPDockLinkServer refused a %r marker for %s on %s's connection",
                marker.kind, marker.mule_id, mule_id,
            )
            return
        self._up_q.put(DockClockMarker(
            mule_id=mule_id, kind=marker.kind, sim_ts=marker.sim_ts, session=session,
        ))

    def _close_session(self, mule_id: MuleID, session: Optional[int]) -> None:
        """Queue ``departed`` for a connection that ended, unless a newer one replaced it.

        Called by the session's own reader after its last UP is queued, so the
        departure can never overtake one of the mule's UPs. Under the lock, so
        it is ordered against a concurrent registration of the same id: either
        it is queued first, or the newer session is already the mule's and the
        old connection's end is no departure.
        """
        with self._lock:
            if session is None or self._sessions.get(mule_id) != session:
                return
            del self._sessions[mule_id]
            self._up_q.put(DockClockMarker(mule_id=mule_id, kind=MARKER_DEPARTED, session=session))

    def _drop_mule(self, mule_id: MuleID, conn: Optional[socket.socket] = None) -> None:
        """Forget ``mule_id``'s connection and close it.

        With ``conn``, only that connection is dropped: if the mule has since
        re-registered on a new socket, the entry is the new one and stays, so
        an old reader ending late cannot disconnect the mule's new session.
        ``conn`` itself is closed either way.
        """
        with self._lock:
            current = self._sockets.get(mule_id)
            if conn is None or current is conn:
                sock = self._sockets.pop(mule_id, None)
                self._reader_threads.pop(mule_id, None)
            else:
                sock = conn
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

    def client_send_clock(
        self, mule_id: MuleID, sim_ts: Optional[float], *, done: bool = False,
    ) -> bool:
        """Report this mule's simulated time without a bundle (FeRRy Phase 3, U9).

        Sends a ``clock`` marker (every later UP of this mule completes at or
        after ``sim_ts``) or, with ``done``, a ``done`` marker (no more UPs;
        ``sim_ts`` optional). A cluster server built with ``sim_markers``
        queues it with this mule's UPs; any other server ignores it. Call it
        from the thread that sends the UPs: sends on the one socket are not
        interleaved otherwise. Raises :class:`DockLinkError` like
        :meth:`client_send_up`; returns True once sent.
        """
        self._raise_if_closed()
        if mule_id != self._mule_id:
            raise DockLinkError(
                f"client_send_clock for {mule_id!r} on client {self._mule_id!r}"
            )
        marker = DockClockMarker(
            mule_id=mule_id, kind=MARKER_DONE if done else MARKER_CLOCK, sim_ts=sim_ts,
        )
        try:
            send_message(self._sock, marker)
        except WireError as e:
            self._closed.set()
            raise DockLinkError(f"client_send_clock failed: {e}") from e
        return True

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
            raise DockLinkTimeout(
                f"client_recv_down for {mule_id!r} timed out after {timeout}s"
            ) from e

    def client_drain_down(self, mule_id: MuleID) -> List[DownBundle]:
        """Every DOWN the reader thread has queued so far, oldest first.

        Works on a closed link too: bundles that arrived before it closed are
        still returned, and an empty list means nothing is waiting.
        """
        if mule_id != self._mule_id:
            raise DockLinkError(
                f"client_drain_down for {mule_id!r} on client {self._mule_id!r}"
            )
        return _drain(self._down_q)

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
