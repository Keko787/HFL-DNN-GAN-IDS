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
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

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
from .rf_link import RFLink, RFLinkError
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
    ) -> None:
        self._host = host
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
        # cleanly via _drop_device_locked. Devices that connect after
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
                self._drop_device_locked(did)

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
            self._drop_device(device_id)
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
        # Long reads on the worker — use a sane timeout so peer-vanish
        # is detected within a few seconds.
        # S2-M3: keep this socket's timeout long enough for big DiscPush
        # blobs but bounded so a stuck peer doesn't hang the supervisor
        # forever. Sends use the same setting via sendall.
        conn.settimeout(self._send_timeout_s)
        with self._registration_cv:  # implicitly takes self._lock
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
        t.start()
        log.info("TCPRFLinkServer registered device %s", did)

    def _reader_loop(self, device_id: DeviceID, conn: socket.socket) -> None:
        while not self._closed.is_set():
            try:
                msg = recv_message(conn)
            except WireError:
                # peer closed or framing failure — drop cleanly
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

        self._drop_device(device_id)
        log.info("TCPRFLinkServer reader for %s exiting", device_id)

    def _drop_device(self, device_id: DeviceID) -> None:
        with self._lock:
            self._drop_device_locked(device_id)

    def _drop_device_locked(self, device_id: DeviceID) -> None:
        sock = self._sockets.pop(device_id, None)
        self._reader_threads.pop(device_id, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

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
    3. ``close()`` tears the connection down.
    """

    def __init__(
        self,
        device_id: DeviceID,
        host: str,
        port: int,
        *,
        emulator: Optional[ChannelEmulator] = None,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self._device_id = device_id
        self._emulator = emulator or no_op_emulator()
        self._closed = threading.Event()
        self._lock = threading.RLock()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(connect_timeout_s)
        self._sock.connect((host, port))
        self._sock.settimeout(60.0)

        # Register first.
        send_message(self._sock, _DeviceRegistrationMessage(device_id=device_id))

        self._solicit_q: "queue.Queue[FLOpenSolicit]" = queue.Queue()
        self._disc_q: "queue.Queue[DiscPush]" = queue.Queue()

        self._reader = threading.Thread(
            target=self._reader_loop,
            name=f"TCPRFLinkClient-{device_id}",
            daemon=True,
        )
        self._reader.start()

    @property
    def device_id(self) -> DeviceID:
        return self._device_id

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
            return self._solicit_q.get(timeout=timeout)
        except queue.Empty as e:
            raise RFLinkError(
                f"recv_open_solicit for {device_id!r} timed out after {timeout}s"
            ) from e

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

    def _reader_loop(self) -> None:
        while not self._closed.is_set():
            try:
                msg = recv_message(self._sock)
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

        self._closed.set()
        log.info("TCPRFLinkClient %s reader exiting", self._device_id)

    def _send_with_emulator(self, msg, *, label: str) -> None:
        drop, delay = self._emulator.apply()
        if drop:
            log.debug("TCPRFLinkClient %s: dropped outbound %s", self._device_id, label)
            return
        if delay > 0.0:
            time.sleep(delay)
        try:
            send_message(self._sock, msg)
        except WireError as e:
            self._closed.set()
            raise RFLinkError(
                f"send {label} from {self._device_id!r} failed: {e}"
            ) from e

    def _raise_if_closed(self) -> None:
        if self._closed.is_set():
            raise RFLinkError(f"rf link to {self._device_id!r} closed")
