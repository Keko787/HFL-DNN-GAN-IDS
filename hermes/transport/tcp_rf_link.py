"""Sprint 2 — TCP-backed RFLink for the multi-process AVN topology.

Replaces the in-process ``LoopbackRFLink`` with a real TCP transport
between the mule (server) and each edge device (client). Designed so
that when AERPAW returns, only the IPs change — the protocol shape and
the supervisor wiring stay the same.

Wire model:

* The mule binds + listens on a port. Each ``ClientMission`` connects
  as a TCP client and sends a ``_DeviceRegistrationMessage`` first so
  the mule can map socket → DeviceID. After that the connection is
  long-lived; both sides exchange length-prefix-framed pickled messages.
* The mule's ``broadcast_open_solicit`` writes the same frame on every
  registered socket. ``recv_ready_adv`` reads from a shared queue that
  every per-device reader thread populates.
* Per-device ``push_disc`` / ``recv_gradient`` / ``recv_delivery_ack``
  are unicast on the matching socket and pulled from per-device queues.
* The ``ChannelEmulator`` rolls a drop / delay decision on every
  outbound + inbound message — symmetric, applied at this layer.

Asymmetry vs the abstract ``RFLink`` ABC:

* :class:`TCPRFLinkServer` implements the mule-side methods. The
  device-side methods raise ``NotImplementedError`` if called on the
  server instance — wrong-side wiring is a programming bug, not a
  runtime case to handle.
* :class:`TCPRFLinkClient` mirrors the inverse.

This split keeps each class single-responsibility while still
satisfying the abstract base class for type-checkers and the tests
that use ``isinstance(..., RFLink)``.

Tunables exposed on the constructor (S2-L3):

* ``accept_timeout_s`` — listener-side ``select`` quantum; lower is
  more responsive shutdown but higher CPU on idle. Default 0.25s.
* ``send_timeout_s`` — bound on per-message ``sendall`` so a stuck
  peer can't block the supervisor. Default 30s (RF) / 60s (dock).

Freeze Amendment 10 (finding P-02): reads block with no timeout on both
sides, as the dock link's reader does, so a device may stay silent for
any stretch of wall time (a quorum wait at the dock, a long local fit, a
stop it is not solicited at) without being dropped. ``send_timeout_s``
bounds sends only, through ``SO_SNDTIMEO``. A reader thread ends when its
socket is closed or its peer goes away. A device that registers again
under the same id replaces its old socket, and the device side can
re-dial (:meth:`TCPRFLinkClient.reconnect`).

A re-dial counts only once the mule acknowledges it: the re-registration
asks for a :class:`_DeviceRegistrationAck`, which the server sends before
it publishes the socket. Whatever else holds the port by then (another
process's listener, once the device's mule has exited) refuses or ignores
the registration, and the re-dial fails. The initial registration asks for
no acknowledgement and gets none, as before. ``link_token`` (default None,
unchecked) goes further: a server started with one refuses a registration
carrying any other, so a device whose mule has gone cannot register with a
mule of another trial that later binds the same port.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from hermes.types import (
    DeliveryAck,
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    GradientSubmission,
    MuleID,
)

from .channel_emulator import ChannelEmulator, no_op_emulator
from .rf_link import RFLink, RFLinkError, _target_ids, _take_newest
from .tcp_dock_link import _close_socket, set_send_timeout
from .wire import WireError, recv_message, send_message

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Registration handshake — first frame on every client socket
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _DeviceRegistrationMessage:
    """First frame sent by a connecting device to identify itself.

    Wire-internal — not part of the design doc's message catalogue.
    The mule reads this to populate its socket→DeviceID map; without
    it we'd have no way to route unicast pushes.

    Amendment 10 adds two fields, both off by default (the initial
    registration of every recorded run): ``confirm`` asks the server to
    reply with a :class:`_DeviceRegistrationAck` (a re-dial sets it, and
    counts as done only on that reply); ``link_token`` is checked by a
    server started with a token of its own.
    """

    device_id: DeviceID
    confirm: bool = False
    link_token: Optional[str] = None


@dataclass(frozen=True)
class _DeviceRegistrationAck:
    """The server's reply to a registration that set ``confirm`` (Amendment 10).

    Sent before the socket is published, so it is the first frame the
    device reads on that connection: no broadcast can get ahead of it.
    """

    device_id: DeviceID


# --------------------------------------------------------------------------- #
# Server side — runs on the mule NUC
# --------------------------------------------------------------------------- #

class TCPRFLinkServer(RFLink):
    """Mule-side TCP RFLink. Binds on construction; accept loop on start.

    Lifecycle:

    1. ``__init__`` binds a listener socket on (host, port).
    2. ``start()`` spawns the accept loop. Each accepted connection
       spawns a reader thread that pulls the registration message and
       then pumps inbound frames into the right queues.
    3. ``broadcast_open_solicit`` / ``push_disc`` send synchronously on
       the relevant socket(s). ``recv_*`` block on the matching queue.
    4. ``close()`` shuts down all connections + the listener.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        emulator: Optional[ChannelEmulator] = None,
        accept_timeout_s: float = 0.25,
        send_timeout_s: float = 30.0,
        link_token: Optional[str] = None,
    ) -> None:
        self._host = host
        # Amendment 10: None (the default) registers any device, as before;
        # a token makes this server refuse registrations that carry another.
        self._link_token = link_token
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(32)
        self._listener.settimeout(accept_timeout_s)
        self._port: int = self._listener.getsockname()[1]

        self._emulator = emulator or no_op_emulator()
        # S2-M3: registered sockets get this timeout for sendall, so a
        # stuck peer can't block the supervisor indefinitely. 30s is a
        # generous default — large DiscPush blobs over slow loopback
        # finish well within it.
        self._send_timeout_s = send_timeout_s

        self._lock = threading.RLock()
        self._closed = threading.Event()

        # Per-device sockets and per-device queues.
        self._sockets: Dict[DeviceID, socket.socket] = {}
        self._reader_threads: Dict[DeviceID, threading.Thread] = {}
        self._gradient_q: Dict[DeviceID, "queue.Queue[GradientSubmission]"] = {}
        self._delivery_ack_q: Dict[DeviceID, "queue.Queue[DeliveryAck]"] = {}
        self._ready_q: "queue.Queue[FLReadyAdv]" = queue.Queue()

        self._accept_thread: Optional[threading.Thread] = None
        # S2-M4: registration uses a Condition so wait_for_devices wakes
        # the moment the last expected device shows up — no polling.
        self._registration_cv = threading.Condition(self._lock)
        # S2-H3: surface accept-loop / handler exceptions to the
        # supervisor instead of silently exiting the daemon thread.
        self._last_accept_error: Optional[BaseException] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        """Spawn the accept loop. Idempotent."""
        if self._accept_thread is not None:
            return
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="TCPRFLinkServer-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def wait_for_devices(
        self, device_ids: List[DeviceID], timeout: float = 5.0
    ) -> bool:
        """Block until every named device has registered, or timeout.

        Returns True iff all expected devices are connected.

        S2-M4: uses a ``Condition`` notified at registration time, so we
        wake instantly when the last expected device registers (no 50ms
        polling jitter).
        """
        wanted = set(device_ids)
        deadline = time.time() + timeout
        with self._registration_cv:
            while True:
                got = set(self._sockets.keys())
                if wanted.issubset(got):
                    return True
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                # Condition.wait returns True on notify, False on timeout.
                self._registration_cv.wait(timeout=remaining)

    @property
    def last_accept_error(self) -> Optional[BaseException]:
        """S2-H3: most recent fault from the accept loop / per-handler.

        Useful for tests + the supervisor to surface a hidden faulty
        listener that would otherwise leave ``wait_for_devices`` hanging
        without explanation. ``None`` when nothing's gone wrong.
        """
        with self._lock:
            return self._last_accept_error

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._listener.close()
        except OSError:
            pass
        with self._lock:
            for s in self._sockets.values():
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass
            self._sockets.clear()
            self._reader_threads.clear()

    # ------------------------------------------------------------------ #
    # Mule-side (server) interface
    # ------------------------------------------------------------------ #

    def broadcast_open_solicit(self, msg: FLOpenSolicit) -> None:
        # S2-L2: "atomic to all live devices" is a fuzzy contract — we
        # snapshot the registered set under the lock, then release, then
        # send to each. If a device disconnects between the snapshot and
        # its send, we get a WireError on that one socket and drop it
        # cleanly via _drop_device. Devices that connect after
        # the snapshot won't see THIS broadcast; they'll get the next
        # one. That's intentional — broadcast semantics are best-effort.
        self._raise_if_closed()
        with self._lock:
            sockets = list(self._sockets.items())

        # Apply the channel emulator once per recipient — drops are per-link.
        for did, sock in sockets:
            drop, delay = self._emulator.apply()
            if drop:
                log.debug("TCPRFLinkServer broadcast: dropped to %s", did)
                continue
            if delay > 0.0:
                time.sleep(delay)
            try:
                send_message(sock, msg)
            except WireError as e:
                log.warning("broadcast send to %s failed: %s", did, e)
                # Only this socket: if the device has re-registered since the
                # snapshot, its new socket stays (Amendment 10).
                self._drop_device(did, sock)

    def solicit(
        self, msg: FLOpenSolicit, device_ids: Iterable[DeviceID]
    ) -> List[DeviceID]:
        """Send ``msg`` to the listed devices only (FeRRy Phase 3).

        Same per-recipient path as :meth:`broadcast_open_solicit` (one
        channel-emulator draw per recipient, in the given order), restricted
        to the listed ids that hold a socket. Unknown or dropped ids are
        skipped. The returned ids include recipients the emulator dropped:
        the mule cannot observe a channel loss, only a missing advert. A
        recipient whose send fails is dropped, as in a broadcast, and left
        out of the result.
        """
        self._raise_if_closed()
        wanted = _target_ids(device_ids)
        with self._lock:
            targets = [
                (did, self._sockets[did]) for did in wanted if did in self._sockets
            ]
        if len(targets) < len(wanted):
            known = {did for did, _sock in targets}
            log.debug(
                "TCPRFLinkServer solicit: skipped unknown %s",
                [did for did in wanted if did not in known],
            )
        sent: List[DeviceID] = []
        for did, sock in targets:
            drop, delay = self._emulator.apply()
            if drop:
                log.debug("TCPRFLinkServer solicit: dropped to %s", did)
                sent.append(did)
                continue
            if delay > 0.0:
                time.sleep(delay)
            try:
                send_message(sock, msg)
            except WireError as e:
                log.warning("solicit send to %s failed: %s", did, e)
                self._drop_device(did, sock)
                continue
            sent.append(did)
        return sent

    def recv_ready_adv(self, timeout: Optional[float] = None) -> FLReadyAdv:
        self._raise_if_closed()
        try:
            return self._ready_q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(f"recv_ready_adv timed out after {timeout}s") from e

    def push_disc(self, device_id: DeviceID, msg: DiscPush) -> None:
        self._raise_if_closed()
        sock = self._socket_for(device_id)
        drop, delay = self._emulator.apply()
        if drop:
            log.debug("TCPRFLinkServer push_disc: dropped to %s", device_id)
            return
        if delay > 0.0:
            time.sleep(delay)
        try:
            send_message(sock, msg)
        except WireError as e:
            self._drop_device(device_id, sock)
            raise RFLinkError(
                f"push_disc to {device_id!r} failed: {e}"
            ) from e

    def recv_gradient(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> GradientSubmission:
        self._raise_if_closed()
        q = self._ensure_queue(self._gradient_q, device_id)
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_gradient for {device_id!r} timed out after {timeout}s"
            ) from e

    def recv_delivery_ack(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> DeliveryAck:
        self._raise_if_closed()
        q = self._ensure_queue(self._delivery_ack_q, device_id)
        try:
            return q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_delivery_ack for {device_id!r} timed out after {timeout}s"
            ) from e

    # ------------------------------------------------------------------ #
    # Device-side methods — not implemented on the server
    # ------------------------------------------------------------------ #

    def recv_open_solicit(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkServer is the mule side; use TCPRFLinkClient on devices")

    def send_ready_adv(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkServer is the mule side; use TCPRFLinkClient on devices")

    def recv_disc_push(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkServer is the mule side; use TCPRFLinkClient on devices")

    def send_gradient(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkServer is the mule side; use TCPRFLinkClient on devices")

    def send_delivery_ack(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkServer is the mule side; use TCPRFLinkClient on devices")

    # ------------------------------------------------------------------ #
    # Internal — accept loop + per-device readers
    # ------------------------------------------------------------------ #

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError as e:
                # S2-H3: not silently exiting. If we got here without
                # being closed, the listener died unexpectedly — record
                # so wait_for_devices / supervisor can surface a clear
                # cause instead of a mystery hang.
                if not self._closed.is_set():
                    log.error(
                        "TCPRFLinkServer accept loop dying: %s", e,
                    )
                    with self._lock:
                        self._last_accept_error = e
                break
            try:
                self._spawn_reader(conn)
            except Exception as e:  # pragma: no cover — defensive
                log.exception("TCPRFLinkServer _spawn_reader raised")
                with self._lock:
                    self._last_accept_error = e

    def _spawn_reader(self, conn: socket.socket) -> None:
        # Read the registration message synchronously — bounded timeout
        # so a stuck client can't pin an accept worker forever.
        try:
            conn.settimeout(2.0)
            reg = recv_message(conn)
        except (WireError, OSError) as e:
            log.warning("rejected unregistered client: %s", e)
            try:
                conn.close()
            except OSError:
                pass
            return

        if not isinstance(reg, _DeviceRegistrationMessage):
            log.warning("rejected non-registration first frame: %r", reg)
            conn.close()
            return

        did = reg.device_id
        if self._link_token is not None and reg.link_token != self._link_token:
            # Amendment 10: most likely a device of another trial whose own
            # mule has exited, re-dialling the port this server now holds.
            # Registering it would evict this mule's device of the same id.
            log.warning(
                "TCPRFLinkServer rejected device %s: its link token is not this mule's",
                did,
            )
            _close_socket(conn)
            return
        if reg.confirm:
            # A re-dial: the device counts it as done only on this reply.
            # Sent before the socket is published below, so nothing else can
            # be sent on it first; still under the registration read's 2 s
            # timeout, so a peer that stopped reading cannot hold this loop.
            try:
                send_message(conn, _DeviceRegistrationAck(device_id=did))
            except WireError as e:
                log.warning("registration ack to %s failed: %s", did, e)
                _close_socket(conn)
                return
        # Amendment 10 (P-02): the reader blocks until a frame arrives or
        # the socket is closed, as the dock link's does. The 30 s socket
        # timeout set here before also bounded reads, so any device silent
        # for 30 s of wall time (a quorum wait at the dock, a long fit, a
        # stop it is not solicited at) was dropped, and nothing brought it
        # back. S2-M3's bound now applies to sends only, via SO_SNDTIMEO,
        # still long enough for big DiscPush blobs.
        conn.settimeout(None)
        if not set_send_timeout(conn, self._send_timeout_s):
            log.debug("SO_SNDTIMEO not honoured on this platform")
        with self._registration_cv:  # implicitly takes self._lock
            # A device registering again under the same id (it re-dialled
            # after losing its link) replaces its old connection. The old
            # one is closed below, so its reader ends; that reader then
            # leaves the new entry alone, because _drop_device only removes
            # the socket it was given (the dock server's Phase 2 pattern).
            previous = self._sockets.get(did)
            self._sockets[did] = conn
            self._gradient_q.setdefault(did, queue.Queue())
            self._delivery_ack_q.setdefault(did, queue.Queue())
            t = threading.Thread(
                target=self._reader_loop,
                args=(did, conn),
                name=f"TCPRFLinkServer-reader-{did}",
                daemon=True,
            )
            self._reader_threads[did] = t
            # S2-M4: wake any wait_for_devices caller that was blocked
            # waiting for *this* device to register.
            self._registration_cv.notify_all()
        if previous is not None and previous is not conn:
            log.info(
                "TCPRFLinkServer: device %s re-registered; closing its old socket", did,
            )
            _close_socket(previous)
        t.start()
        log.info("TCPRFLinkServer registered device %s", did)

    def _reader_loop(self, device_id: DeviceID, conn: socket.socket) -> None:
        while not self._closed.is_set():
            try:
                msg = recv_message(conn)
            except WireError:
                # peer closed, socket closed by us, or framing failure
                break

            drop, delay = self._emulator.apply()
            if drop:
                log.debug(
                    "TCPRFLinkServer recv: dropped %s from %s",
                    type(msg).__name__, device_id,
                )
                continue
            if delay > 0.0:
                time.sleep(delay)

            if isinstance(msg, FLReadyAdv):
                self._ready_q.put(msg)
            elif isinstance(msg, GradientSubmission):
                q = self._ensure_queue(self._gradient_q, device_id)
                q.put(msg)
            elif isinstance(msg, DeliveryAck):
                q = self._ensure_queue(self._delivery_ack_q, device_id)
                q.put(msg)
            else:
                log.warning(
                    "TCPRFLinkServer received unknown message type %s from %s",
                    type(msg).__name__, device_id,
                )

        self._drop_device(device_id, conn)
        log.info("TCPRFLinkServer reader for %s exiting", device_id)

    def _drop_device(
        self, device_id: DeviceID, conn: Optional[socket.socket] = None
    ) -> None:
        """Forget ``device_id``'s connection and close it.

        With ``conn``, only that connection is dropped: if the device has
        since re-registered on a new socket, the entry is the new one and
        stays, so an old reader ending late, or a failed send on the old
        socket, cannot disconnect the new session. ``conn`` itself is closed
        either way. Closing shuts the socket down first, which is what ends
        a reader blocked on it now that reads have no timeout.
        """
        with self._lock:
            current = self._sockets.get(device_id)
            if conn is None or current is conn:
                sock = self._sockets.pop(device_id, None)
                self._reader_threads.pop(device_id, None)
            else:
                sock = conn
        if sock is not None:
            _close_socket(sock)

    def _socket_for(self, device_id: DeviceID) -> socket.socket:
        with self._lock:
            sock = self._sockets.get(device_id)
        if sock is None:
            raise RFLinkError(
                f"no registered socket for {device_id!r} on this mule"
            )
        return sock

    def _ensure_queue(self, store: dict, device_id: DeviceID):
        with self._lock:
            q = store.get(device_id)
            if q is None:
                q = queue.Queue()
                store[device_id] = q
            return q

    def _raise_if_closed(self) -> None:
        if self._closed.is_set():
            raise RFLinkError("rf link closed")


# --------------------------------------------------------------------------- #
# Client side — runs on each edge device process
# --------------------------------------------------------------------------- #

class TCPRFLinkClient(RFLink):
    """Device-side TCP RFLink. Connects on construction; one socket per device.

    Lifecycle:

    1. ``__init__(host, port, device_id)`` opens a TCP connection,
       sends the registration message, and starts a single reader
       thread that fans inbound frames into per-message queues.
    2. ``recv_open_solicit`` / ``recv_disc_push`` block on those
       queues. ``send_*`` write synchronously on the socket.
    3. When the mule side goes away (or a send fails) the link is down:
       ``connected`` turns False and the device-side calls raise
       ``RFLinkError`` at once. ``reconnect()`` re-dials the same mule and
       registers again, and succeeds only once the mule acknowledges it
       (Amendment 10); queued frames are kept.
    4. ``close()`` tears the connection down for good.

    ``newest_solicit_only`` (FeRRy Phase 3, critic B1; default off, the
    FIFO every recorded run used) makes ``recv_open_solicit`` answer only
    the newest queued solicit and drop the older ones. A device that missed
    a gather otherwise answers that gather's stale solicit first, which a
    strict (``solicit_id``-matching) mule discards, and then waits out its
    push timeout through the next gather too.

    ``link_token`` (Amendment 10; default None, as in every recorded run)
    goes out with every registration; a server started with a different
    token refuses it (see the module docstring).
    """

    def __init__(
        self,
        device_id: DeviceID,
        host: str,
        port: int,
        *,
        emulator: Optional[ChannelEmulator] = None,
        connect_timeout_s: float = 5.0,
        send_timeout_s: float = 60.0,
        newest_solicit_only: bool = False,
        link_token: Optional[str] = None,
    ) -> None:
        self._device_id = device_id
        self._host = host
        self._port = port
        self._emulator = emulator or no_op_emulator()
        self._link_token = link_token
        self._connect_timeout_s = connect_timeout_s
        # Amendment 10 (P-02): the 60 s socket timeout set here before also
        # bounded reads, so a device the mule left alone for 60 s of wall time
        # lost its link. It now bounds sends only, via SO_SNDTIMEO.
        self._send_timeout_s = send_timeout_s
        self._newest_solicit_only = bool(newest_solicit_only)
        #: Solicits dropped because a newer one was already queued
        #: (``newest_solicit_only``); always 0 with the option off.
        self.stale_solicits_dropped = 0
        # ``_closed``: the link is down (peer gone, a send failed, or close());
        # reconnect() clears it. ``_shut``: close() was called; final.
        self._closed = threading.Event()
        self._shut = threading.Event()
        self._lock = threading.RLock()

        self._solicit_q: "queue.Queue[FLOpenSolicit]" = queue.Queue()
        self._disc_q: "queue.Queue[DiscPush]" = queue.Queue()

        # Connect and register first; a failure raises from here, as before.
        self._sock = self._dial()
        self._reader = self._start_reader(self._sock)

    @property
    def device_id(self) -> DeviceID:
        return self._device_id

    @property
    def connected(self) -> bool:
        """False while the link is down: the mule side went away, a send
        failed, or :meth:`close` was called."""
        return not self._closed.is_set()

    def reconnect(self) -> None:
        """Replace the connection with a fresh one to the same mule.

        Amendment 10: before it, a device whose link dropped could never
        get it back (and its service loop spun on the dead link). The old
        socket is closed first, which ends its reader; the new one
        registers under the same device id, and the mule's server swaps it
        in for the old entry. Frames already queued stay queued.

        The re-dial is done only when the mule acknowledges the
        registration, within ``connect_timeout_s``. A TCP connect alone
        proves nothing: once the device's mule has exited, any process may
        bind its port, and a listener that is not this device's mule (or,
        with ``link_token``, is a mule of another trial) refuses or ignores
        the registration. Raises ``RFLinkError`` when the mule cannot be
        reached or does not acknowledge (the link stays down, so the caller
        can retry with backoff), or after :meth:`close`.
        """
        with self._lock:
            if self._shut.is_set():
                raise RFLinkError(
                    f"rf link to {self._device_id!r} closed; not reconnecting"
                )
            self._closed.set()
            _close_socket(self._sock)
            try:
                sock = self._dial(confirm=True)
            except (OSError, WireError) as e:
                raise RFLinkError(
                    f"reconnect of {self._device_id!r} to "
                    f"{self._host}:{self._port} failed: {e}"
                ) from e
            self._sock = sock
            self._closed.clear()
            self._reader = self._start_reader(sock)
        log.info(
            "TCPRFLinkClient %s reconnected to %s:%d",
            self._device_id, self._host, self._port,
        )

    def close(self) -> None:
        # Holding the lock orders close() after a reconnect() in progress,
        # so the socket that reconnect() installs is the one closed here.
        with self._lock:
            if self._shut.is_set():
                return
            self._shut.set()
            self._closed.set()
            sock = self._sock
        _close_socket(sock)

    # ------------------------------------------------------------------ #
    # Device-side (client) interface
    # ------------------------------------------------------------------ #

    def recv_open_solicit(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> FLOpenSolicit:
        self._raise_if_closed()
        if device_id != self._device_id:
            raise RFLinkError(
                f"recv_open_solicit for {device_id!r} on client {self._device_id!r}"
            )
        try:
            if not self._newest_solicit_only:
                return self._solicit_q.get(timeout=timeout)
            msg, dropped = _take_newest(self._solicit_q, timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_open_solicit for {device_id!r} timed out after {timeout}s"
            ) from e
        if dropped:
            with self._lock:
                self.stale_solicits_dropped += dropped
            log.debug(
                "TCPRFLinkClient %s: answering the newest solicit, dropped %d older",
                self._device_id, dropped,
            )
        return msg

    def send_ready_adv(self, msg: FLReadyAdv) -> None:
        self._raise_if_closed()
        self._send_with_emulator(msg, label="ready_adv")

    def recv_disc_push(
        self, device_id: DeviceID, timeout: Optional[float] = None
    ) -> DiscPush:
        self._raise_if_closed()
        if device_id != self._device_id:
            raise RFLinkError(
                f"recv_disc_push for {device_id!r} on client {self._device_id!r}"
            )
        try:
            return self._disc_q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_disc_push for {device_id!r} timed out after {timeout}s"
            ) from e

    def send_gradient(self, msg: GradientSubmission) -> None:
        self._raise_if_closed()
        self._send_with_emulator(msg, label="gradient")

    def send_delivery_ack(self, msg: DeliveryAck) -> None:
        self._raise_if_closed()
        self._send_with_emulator(msg, label="delivery_ack")

    # ------------------------------------------------------------------ #
    # Mule-side methods — not implemented on the client
    # ------------------------------------------------------------------ #

    def broadcast_open_solicit(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkClient is the device side; use TCPRFLinkServer on the mule")

    def solicit(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkClient is the device side; use TCPRFLinkServer on the mule")

    def recv_ready_adv(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkClient is the device side; use TCPRFLinkServer on the mule")

    def push_disc(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkClient is the device side; use TCPRFLinkServer on the mule")

    def recv_gradient(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkClient is the device side; use TCPRFLinkServer on the mule")

    def recv_delivery_ack(self, *_a, **_kw):
        raise NotImplementedError("TCPRFLinkClient is the device side; use TCPRFLinkServer on the mule")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _dial(self, *, confirm: bool = False) -> socket.socket:
        """Open one connection to the mule, configure it, and register.

        With ``confirm`` (a re-dial), also wait up to ``connect_timeout_s``
        for the mule's :class:`_DeviceRegistrationAck`, which it sends
        before any other frame; anything else (the peer closing, silence,
        another frame) fails the dial. Without it (the initial connect) the
        registration goes out unacknowledged, as it always has.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(self._connect_timeout_s)
            sock.connect((self._host, self._port))
            # Amendment 10: reads block until a frame arrives or the socket
            # is closed; sends are bounded by SO_SNDTIMEO.
            sock.settimeout(None)
            if not set_send_timeout(sock, self._send_timeout_s):
                log.debug("SO_SNDTIMEO not honoured on this platform")
            # Register first.
            send_message(sock, _DeviceRegistrationMessage(
                device_id=self._device_id,
                confirm=confirm,
                link_token=self._link_token,
            ))
            if confirm:
                self._await_ack(sock)
        except BaseException:
            _close_socket(sock)
            raise
        return sock

    def _await_ack(self, sock: socket.socket) -> None:
        """Read the mule's registration acknowledgement off ``sock`` (a re-dial).

        Runs before the reader starts, so the reply never reaches the
        channel emulator or the message queues. The timeout applies to this
        read only; the socket goes back to blocking reads afterwards.
        """
        where = f"{self._host}:{self._port}"
        try:
            reply = recv_message(sock, timeout=self._connect_timeout_s)
        except WireError as e:
            raise WireError(
                f"no registration acknowledgement from {where}: {e}"
            ) from e
        if isinstance(reply, _DeviceRegistrationAck) and reply.device_id == self._device_id:
            return
        got = (
            f"an acknowledgement for {reply.device_id!r}"
            if isinstance(reply, _DeviceRegistrationAck)
            else type(reply).__name__
        )
        raise WireError(
            f"{where} did not acknowledge the registration of "
            f"{self._device_id!r}; its first frame was {got}"
        )

    def _start_reader(self, sock: socket.socket) -> threading.Thread:
        t = threading.Thread(
            target=self._reader_loop,
            args=(sock,),
            name=f"TCPRFLinkClient-{self._device_id}",
            daemon=True,
        )
        t.start()
        return t

    def _reader_loop(self, sock: socket.socket) -> None:
        while not self._shut.is_set():
            try:
                msg = recv_message(sock)
            except WireError:
                break

            drop, delay = self._emulator.apply()
            if drop:
                log.debug(
                    "TCPRFLinkClient %s: dropped inbound %s",
                    self._device_id, type(msg).__name__,
                )
                continue
            if delay > 0.0:
                time.sleep(delay)

            if isinstance(msg, FLOpenSolicit):
                self._solicit_q.put(msg)
            elif isinstance(msg, DiscPush):
                self._disc_q.put(msg)
            else:
                log.warning(
                    "TCPRFLinkClient %s: unknown message %s",
                    self._device_id, type(msg).__name__,
                )

        self._link_down(sock)
        log.info("TCPRFLinkClient %s reader exiting", self._device_id)

    def _link_down(self, sock: socket.socket) -> None:
        """Mark the link down if ``sock`` is still its connection.

        An old connection's reader ending, or a failed send on it, after
        reconnect() installed a new one must not take the new one down.
        """
        with self._lock:
            if self._sock is sock:
                self._closed.set()

    def _send_with_emulator(self, msg, *, label: str) -> None:
        drop, delay = self._emulator.apply()
        if drop:
            log.debug("TCPRFLinkClient %s: dropped outbound %s", self._device_id, label)
            return
        if delay > 0.0:
            time.sleep(delay)
        sock = self._sock
        try:
            send_message(sock, msg)
        except WireError as e:
            self._link_down(sock)
            # A failed or timed-out send leaves the stream mid-frame (Winsock
            # says the connection "should be closed"); closing also ends the
            # reader, which no read timeout would end now.
            _close_socket(sock)
            raise RFLinkError(
                f"send {label} from {self._device_id!r} failed: {e}"
            ) from e

    def _raise_if_closed(self) -> None:
        if self._closed.is_set():
            raise RFLinkError(f"rf link to {self._device_id!r} closed")
