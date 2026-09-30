"""Freeze Amendment 10 (finding P-02) and targeted solicits on real TCP sockets.

Before the amendment, the RF server's ``send_timeout_s`` (30 s) was the socket
timeout, so it bounded READS too: a device silent for 30 s of wall time was
dropped (and the device side did the same at 60 s). Here the timeouts are cut
to fractions of a second, so the old behaviour would show within the test.

Covered:
* an idle device stays registered and connected past the (shortened) old
  timeout, and the link still works both ways;
* sends stay bounded (SO_SNDTIMEO) against a peer that stops reading, on both
  sides, and a client whose send failed closes its socket and ends its reader;
* a reader with no read timeout still ends when its peer or its server closes;
* a device that registers again under its id replaces its old socket, and
  neither the old reader ending nor a failed push or broadcast on the old
  socket tears down the new entry;
* the device side's ``reconnect`` (same server, and a restarted one), which
  closes its old socket first and counts only once the mule acknowledges the
  registration; a listener that is not the mule fails it, as does a mule with
  another link token;
* ``solicit`` reaches only the listed ids and skips unknown ones; broadcast is
  unchanged;
* the newest-solicit client option;
* the send bound each socket actually holds, and the dock link waiting for a
  slow reader instead of failing after milliseconds (the Windows packing bug).

Tests that push into a peer that is not reading run the pushes on a helper
thread with a deadline (``_run_within``): the suite has no per-test timeout,
and a send left unbounded by a regression must fail the test, not hang it.
"""

from __future__ import annotations

import socket
import struct
import sys
import threading
import time

import numpy as np
import pytest

from hermes.transport import (
    ChannelEmulator,
    RFLinkError,
    TCPDockLinkServer,
    TCPRFLinkClient,
    TCPRFLinkServer,
)
from hermes.transport import tcp_rf_link
from hermes.transport.tcp_dock_link import _MuleRegistrationMessage
from hermes.transport.tcp_rf_link import _DeviceRegistrationAck, _DeviceRegistrationMessage
from hermes.transport.wire import recv_message, send_message
from hermes.types import (
    ClusterAmendment,
    DeviceID,
    DiscPush,
    DownBundle,
    FLOpenSolicit,
    FLReadyAdv,
    FLState,
    GradientSubmission,
    MissionSlice,
    MuleID,
)

MULE = MuleID("mule-a10")
D0, D1, D2 = DeviceID("d0"), DeviceID("d1"), DeviceID("d2")


def _server(**kwargs) -> TCPRFLinkServer:
    s = TCPRFLinkServer(host="127.0.0.1", port=kwargs.pop("port", 0), **kwargs)
    s.start()
    return s


def _client(server: TCPRFLinkServer, did: DeviceID, **kwargs) -> TCPRFLinkClient:
    return TCPRFLinkClient(device_id=did, host=server.host, port=server.port, **kwargs)


def _solicit(mission_round: int) -> FLOpenSolicit:
    return FLOpenSolicit(mule_id=MULE, mission_round=mission_round, issued_at=time.time())


def _adv(did: DeviceID) -> FLReadyAdv:
    return FLReadyAdv(
        device_id=did, state=FLState.FL_OPEN, performance_score=0.5,
        diversity_adjusted=0.5, utility=0.5, issued_at=time.time(),
    )


def _registered(server: TCPRFLinkServer, did: DeviceID) -> bool:
    return server.wait_for_devices([did], timeout=0.0)


def _wait_until(pred, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _reader_of(server: TCPRFLinkServer, did: DeviceID) -> threading.Thread:
    """The reader thread ``server`` runs for ``did``'s current socket.

    Taken from this server's own table, not by thread name: a reader of the
    same name left over from an earlier test (or server) must not count.
    """
    with server._lock:
        return server._reader_threads[did]


def _running_reader_of(server: TCPRFLinkServer, did: DeviceID) -> threading.Thread:
    """``_reader_of`` once that thread runs: the server publishes a device's
    entry, then starts its reader, so a test must not join it before that."""
    assert _wait_until(lambda: _reader_of(server, did).is_alive())
    return _reader_of(server, did)


def _read_sndtimeo_s(sock: socket.socket) -> float:
    if sys.platform.startswith("win"):
        raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, 4)
        return struct.unpack("=I", raw[:4])[0] / 1000.0
    raw = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, struct.calcsize("ll"))
    sec, usec = struct.unpack("ll", raw)
    return sec + usec / 1e6


def _round_trip(server: TCPRFLinkServer, client: TCPRFLinkClient, mission_round: int) -> None:
    """One solicit down and one advert up: the link works both ways."""
    server.broadcast_open_solicit(_solicit(mission_round))
    assert client.recv_open_solicit(client.device_id, timeout=2.0).mission_round == mission_round
    client.send_ready_adv(_adv(client.device_id))
    assert server.recv_ready_adv(timeout=2.0).device_id == client.device_id


def _run_within(seconds: float, fn, what: str):
    """Run ``fn`` on a daemon thread; fail the test if it is still running after ``seconds``.

    Returns what ``fn`` returned, or raises what it raised. A send that a
    regression left unbounded then fails the test instead of blocking the
    run; the caller's ``finally`` closes the sockets, which ends that send.
    """
    box: dict = {}

    def _target() -> None:
        try:
            box["value"] = fn()
        except BaseException as e:  # handed to the test's thread below
            box["error"] = e

    t = threading.Thread(target=_target, name=f"bounded: {what}", daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        pytest.fail(f"{what}: still blocked after {seconds} s; the send is not bounded")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _push_until_one_fails(push, attempts: int = 64):
    """Call ``push()`` until it raises; return (the error, how long that call blocked).

    Windows takes one send whole into an idle pipe and blocks on the next,
    so a single send is not enough to reach the bound.
    """
    for _ in range(attempts):
        t0 = time.monotonic()
        try:
            push()
        except Exception as e:
            return e, time.monotonic() - t0
    return None, 0.0


def _big_gradient(did: DeviceID) -> GradientSubmission:
    return GradientSubmission(
        device_id=did, mule_id=MULE, mission_round=1,
        delta_theta=[np.zeros(1_000_000, dtype=np.float32)],
        num_examples=8, submitted_at=time.time(),
    )


# --------------------------------------------------------------------------- #
# P-02: silence no longer drops a device
# --------------------------------------------------------------------------- #

def test_idle_device_stays_registered_past_the_old_read_timeout():
    # 0.3 s stands in for the server's old 30 s socket timeout, which bounded
    # reads as well as sends. (The client is built exactly as before.)
    server = _server(send_timeout_s=0.3)
    client = _client(server, D0)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        time.sleep(1.2)  # four times the old timeout, no traffic either way
        assert _registered(server, D0)
        _round_trip(server, client, 1)
    finally:
        client.close()
        server.close()


def test_idle_client_stays_connected_past_its_old_read_timeout():
    # The client's old 60 s socket timeout bounded its reads too; the value
    # now bounds sends only, and is cut to 0.3 s here.
    server = _server()
    client = _client(server, D0, send_timeout_s=0.3)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        time.sleep(1.2)
        assert client.connected
        _round_trip(server, client, 1)
    finally:
        client.close()
        server.close()


def test_send_to_a_peer_that_stops_reading_is_still_bounded():
    server = _server(send_timeout_s=0.3)
    # A device that registers and then never reads.
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    raw.connect((server.host, server.port))
    send_message(raw, _DeviceRegistrationMessage(device_id=D0))
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        # A small server-side send buffer, so the sends back up quickly
        # instead of the loopback stack buffering megabytes (test-only poke).
        server._sockets[D0].setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        big = DiscPush(
            mule_id=MULE, mission_round=1,
            theta_disc=[np.zeros(1_000_000, dtype=np.float32)], synth_batch=[],
        )
        err, blocked_s = _run_within(
            10.0, lambda: _push_until_one_fails(lambda: server.push_disc(D0, big)),
            "push_disc to a device that stopped reading",
        )
        assert isinstance(err, RFLinkError) and "push_disc" in str(err), err
        # The push that failed waited for the bound, and no longer than that.
        assert 0.2 <= blocked_s < 5.0
        assert not _registered(server, D0)  # a failed send drops the device
    finally:
        raw.close()
        server.close()


def test_a_failed_client_send_closes_its_socket_and_ends_its_reader():
    # The mule side accepts the device and then never reads. The client's
    # send times out (its SO_SNDTIMEO), and a timed-out send leaves the stream
    # mid-frame, so the client closes the socket, which also ends its reader
    # (no read timeout would end it now).
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = TCPRFLinkClient(D0, "127.0.0.1", listener.getsockname()[1], send_timeout_s=0.3)
    mule_side, _ = listener.accept()
    try:
        sock, reader = client._sock, client._reader
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)  # test-only poke
        err, blocked_s = _run_within(
            10.0, lambda: _push_until_one_fails(lambda: client.send_gradient(_big_gradient(D0))),
            "send_gradient to a mule that stopped reading",
        )
        assert isinstance(err, RFLinkError) and "gradient" in str(err), err
        assert 0.2 <= blocked_s < 5.0
        assert not client.connected
        assert sock.fileno() == -1  # closed, not just marked down
        reader.join(timeout=3.0)
        assert not reader.is_alive()
    finally:
        client.close()
        mule_side.close()
        listener.close()


# --------------------------------------------------------------------------- #
# Re-registration and reconnect
# --------------------------------------------------------------------------- #

def test_reregistration_replaces_the_old_socket():
    server = _server()
    old = _client(server, D0)
    new = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        old_sock, old_reader = server._sockets[D0], _running_reader_of(server, D0)
        new = _client(server, D0)  # the same id on a second connection
        assert _wait_until(lambda: server._sockets.get(D0) is not old_sock)
        # The server closes the old connection...
        assert _wait_until(lambda: not old.connected)
        # ...whose reader then ends without dropping the new entry.
        old_reader.join(timeout=3.0)
        assert not old_reader.is_alive()
        assert _registered(server, D0)
        assert _reader_of(server, D0) is not old_reader
        # (The new reader is started just after its entry is published.)
        assert _wait_until(lambda: _reader_of(server, D0).is_alive())
        _round_trip(server, new, 1)
    finally:
        old.close()
        if new is not None:
            new.close()
        server.close()


def test_a_failed_send_on_a_replaced_socket_keeps_the_new_entry():
    # A broadcast or push that snapshotted the old socket and fails on it
    # after the device re-registered drops that socket only.
    server = _server()
    old = _client(server, D0)
    new = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        old_sock = server._sockets[D0]
        new = _client(server, D0)
        assert _wait_until(lambda: server._sockets.get(D0) is not old_sock)
        server._drop_device(D0, old_sock)  # what the failed send does
        assert _registered(server, D0)
        _round_trip(server, new, 1)
    finally:
        old.close()
        if new is not None:
            new.close()
        server.close()


def _replace(server: TCPRFLinkServer, did: DeviceID):
    """Register ``did`` on a second connection; return (the new client, the
    server's old socket) once the server has swapped the entry and closed the
    old socket."""
    old_sock = server._sockets[did]
    new = _client(server, did)
    assert _wait_until(lambda: old_sock.fileno() == -1)
    assert server._sockets[did] is not old_sock
    return new, old_sock


def test_a_push_that_fails_on_a_replaced_socket_keeps_the_new_entry(monkeypatch):
    # push_disc looked the device's socket up just before it re-registered;
    # the send then fails on the old socket, which the server has closed.
    server = _server()
    old = _client(server, D0)
    new = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        new, old_sock = _replace(server, D0)
        new_sock = server._sockets[D0]
        monkeypatch.setattr(server, "_socket_for", lambda _did: old_sock)
        push = DiscPush(
            mule_id=MULE, mission_round=1,
            theta_disc=[np.ones((4,), dtype=np.float32)], synth_batch=[],
        )
        with pytest.raises(RFLinkError, match="push_disc"):
            server.push_disc(D0, push)
        monkeypatch.undo()
        assert server._sockets.get(D0) is new_sock
        _round_trip(server, new, 1)
    finally:
        old.close()
        if new is not None:
            new.close()
        server.close()


def test_a_broadcast_that_fails_on_a_replaced_socket_keeps_the_new_entry(monkeypatch):
    # The device re-registers after the broadcast took its snapshot and
    # before it sends to the device; that send fails on the old socket.
    server = _server()
    old = _client(server, D0)
    replaced: list = []
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        old_sock = server._sockets[D0]
        real_send = tcp_rf_link.send_message

        def _send(sock, msg):
            if sock is old_sock and not replaced:
                replaced.append(_replace(server, D0))
            return real_send(sock, msg)

        monkeypatch.setattr(tcp_rf_link, "send_message", _send)
        server.broadcast_open_solicit(_solicit(1))  # the failed send is logged
        monkeypatch.undo()
        new, _old_sock = replaced[0]
        assert _registered(server, D0)
        assert server._sockets[D0] is not old_sock
        _round_trip(server, new, 2)
    finally:
        old.close()
        for new, _old_sock in replaced:
            new.close()
        server.close()


def test_client_reconnect_registers_again_on_the_same_server():
    server = _server()
    client = _client(server, D0)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        old_sock, old_reader = server._sockets[D0], _running_reader_of(server, D0)
        own_sock, own_reader = client._sock, client._reader
        client.reconnect()
        assert client.connected
        # The client closed its old socket before re-dialling, which ended
        # its reader there.
        assert own_sock.fileno() == -1
        own_reader.join(timeout=3.0)
        assert not own_reader.is_alive()
        # The server drops the old connection (its reader ends on the close)
        # and registers the new one, in either order.
        old_reader.join(timeout=3.0)
        assert not old_reader.is_alive()
        assert server.wait_for_devices([D0], timeout=2.0)
        assert server._sockets[D0] is not old_sock
        _round_trip(server, client, 1)
    finally:
        client.close()
        server.close()


def test_client_link_goes_down_with_its_server_and_comes_back_after_a_restart():
    server = _server()
    port = server.port
    client = TCPRFLinkClient(D0, "127.0.0.1", port, connect_timeout_s=0.5)
    server2 = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        server.close()
        assert _wait_until(lambda: not client.connected)
        # Down: device-side calls fail at once instead of waiting their timeout.
        t0 = time.monotonic()
        with pytest.raises(RFLinkError, match="closed"):
            client.recv_open_solicit(D0, timeout=5.0)
        assert time.monotonic() - t0 < 1.0
        # Nobody listening: a re-dial fails and the link stays down.
        with pytest.raises(RFLinkError, match="reconnect"):
            client.reconnect()
        assert not client.connected

        server2 = _server(port=port)
        client.reconnect()
        assert client.connected
        assert server2.wait_for_devices([D0], timeout=2.0)
        _round_trip(server2, client, 2)
    finally:
        client.close()
        server.close()
        if server2 is not None:
            server2.close()


def test_reconnect_after_close_is_refused():
    server = _server()
    client = _client(server, D0)
    try:
        client.close()
        client.close()  # idempotent
        with pytest.raises(RFLinkError, match="not reconnecting"):
            client.reconnect()
        assert not client.connected
    finally:
        server.close()


def test_a_device_that_disconnects_is_dropped_by_the_server():
    # With no read timeout, the peer's close is what ends the reader.
    server = _server()
    client = _client(server, D0)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        reader = _running_reader_of(server, D0)
        client.close()
        assert _wait_until(lambda: not _registered(server, D0))
        reader.join(timeout=3.0)
        assert not reader.is_alive()
    finally:
        server.close()


def test_closing_the_server_ends_its_blocked_readers():
    server = _server()
    clients = [_client(server, d) for d in (D0, D1)]
    try:
        assert server.wait_for_devices([D0, D1], timeout=2.0)
        readers = [_running_reader_of(server, d) for d in (D0, D1)]
        server.close()
        for t in readers:
            t.join(timeout=3.0)
            assert not t.is_alive()
        assert _wait_until(lambda: not any(c.connected for c in clients))
    finally:
        for c in clients:
            c.close()
        server.close()


# --------------------------------------------------------------------------- #
# A re-dial counts only once the mule acknowledges it
# --------------------------------------------------------------------------- #

def test_only_a_registration_that_asks_is_acknowledged():
    # The initial registration gets no reply, as in every recorded run; a
    # re-dial's gets the acknowledgement as its first frame, ahead of any
    # broadcast.
    server = _server()
    plain = socket.create_connection((server.host, server.port), timeout=2.0)
    asking = socket.create_connection((server.host, server.port), timeout=2.0)
    try:
        send_message(plain, _DeviceRegistrationMessage(device_id=D0))
        send_message(asking, _DeviceRegistrationMessage(device_id=D1, confirm=True))
        assert server.wait_for_devices([D0, D1], timeout=2.0)
        server.broadcast_open_solicit(_solicit(1))

        ack = recv_message(asking, timeout=2.0)
        assert isinstance(ack, _DeviceRegistrationAck) and ack.device_id == D1
        assert recv_message(asking, timeout=2.0).mission_round == 1
        first = recv_message(plain, timeout=2.0)
        assert isinstance(first, FLOpenSolicit) and first.mission_round == 1
    finally:
        plain.close()
        asking.close()
        server.close()


class _Impostor:
    """A listener on a port a device's mule used to hold, that is not that mule.

    ``kind``: ``dock`` is a real cluster dock server, which rejects the
    device's registration and closes; ``silent`` accepts and never answers;
    ``chatty`` answers the registration with a solicit instead of an
    acknowledgement; ``other_ack`` acknowledges a different device.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._dock = None
        self._raw = None
        self._conns: list = []
        if kind == "dock":
            self._dock = TCPDockLinkServer(host="127.0.0.1", port=0)
            self._dock.start()
            self.port = self._dock.port
            return
        self._raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._raw.bind(("127.0.0.1", 0))
        self._raw.listen(8)
        self.port = self._raw.getsockname()[1]
        if kind == "chatty":
            threading.Thread(target=self._answer, args=(_solicit(9),), daemon=True).start()
        elif kind == "other_ack":
            reply = _DeviceRegistrationAck(device_id=D1)
            threading.Thread(target=self._answer, args=(reply,), daemon=True).start()

    def _answer(self, reply) -> None:
        """Answer every registration with ``reply``."""
        while True:
            try:
                conn, _ = self._raw.accept()
            except OSError:
                return
            self._conns.append(conn)
            try:
                recv_message(conn, timeout=2.0)
                send_message(conn, reply)
            except Exception:
                pass

    def close(self) -> None:
        if self._dock is not None:
            self._dock.close()
        if self._raw is not None:
            self._raw.close()
        for c in self._conns:
            c.close()


@pytest.mark.parametrize("kind", ["dock", "silent", "chatty", "other_ack"])
def test_reconnect_to_a_listener_that_is_not_the_mule_fails(kind):
    impostor = _Impostor(kind)
    # The initial registration asks for no acknowledgement, so the client
    # connects to anything listening (as it always has); the re-dial must not.
    client = TCPRFLinkClient(D0, "127.0.0.1", impostor.port, connect_timeout_s=0.5)
    try:
        t0 = time.monotonic()
        with pytest.raises(RFLinkError, match="reconnect .*acknowledge"):
            client.reconnect()
        assert time.monotonic() - t0 < 3.0  # bounded by connect_timeout_s
        assert not client.connected
        assert client._sock.fileno() == -1
    finally:
        client.close()
        impostor.close()


def test_a_mule_with_another_link_token_refuses_the_device_and_keeps_its_own():
    # A device of trial A whose mule has exited re-dials the port a mule of
    # trial B now holds; trial B's device has the same id.
    server = _server(link_token="trial-B")
    own = _client(server, D0, link_token="trial-B")
    stale = None
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        own_sock = server._sockets[D0]
        stale = _client(server, D0, link_token="trial-A", connect_timeout_s=0.5)
        assert _wait_until(lambda: not stale.connected)  # refused and closed
        with pytest.raises(RFLinkError, match="reconnect"):
            stale.reconnect()
        assert server._sockets[D0] is own_sock
        assert own.connected
        _round_trip(server, own, 1)
    finally:
        own.close()
        if stale is not None:
            stale.close()
        server.close()


def test_a_link_token_is_checked_only_by_a_server_that_has_one():
    tokened = _server(link_token="t1")
    open_server = _server()  # no token: every recorded run
    clients = []
    try:
        clients.append(_client(tokened, D0))  # carries no token: refused
        clients.append(_client(tokened, D1, link_token="t1"))
        clients.append(_client(open_server, D0, link_token="anything"))
        clients.append(_client(open_server, D1))
        assert tokened.wait_for_devices([D1], timeout=2.0)
        assert open_server.wait_for_devices([D0, D1], timeout=2.0)
        assert _wait_until(lambda: not clients[0].connected)
        assert not _registered(tokened, D0)
        # A device with the right token re-dials its mule as any other does.
        tokened._drop_device(D1)
        assert _wait_until(lambda: not clients[1].connected)
        clients[1].reconnect()
        assert tokened.wait_for_devices([D1], timeout=2.0)
        _round_trip(tokened, clients[1], 1)
    finally:
        for c in clients:
            c.close()
        tokened.close()
        open_server.close()


# --------------------------------------------------------------------------- #
# Targeted solicit (FeRRy Phase 3)
# --------------------------------------------------------------------------- #

def test_solicit_reaches_only_the_listed_devices_and_skips_unknown_ids():
    server = _server()
    clients = [_client(server, d) for d in (D0, D1, D2)]
    try:
        assert server.wait_for_devices([D0, D1, D2], timeout=2.0)
        sent = server.solicit(_solicit(3), [D2, DeviceID("ghost"), D0, D2])

        assert sent == [D2, D0]  # given order, unknown skipped, no repeats
        for c in (clients[0], clients[2]):
            assert c.recv_open_solicit(c.device_id, timeout=2.0).mission_round == 3
        with pytest.raises(RFLinkError):  # not listed
            clients[1].recv_open_solicit(D1, timeout=0.3)
        with pytest.raises(RFLinkError):  # the repeated id got one copy
            clients[2].recv_open_solicit(D2, timeout=0.2)
    finally:
        for c in clients:
            c.close()
        server.close()


def test_solicit_skips_a_device_that_has_gone():
    server = _server()
    c0, c1 = _client(server, D0), _client(server, D1)
    try:
        assert server.wait_for_devices([D0, D1], timeout=2.0)
        c1.close()
        assert _wait_until(lambda: not _registered(server, D1))
        assert server.solicit(_solicit(1), [D0, D1]) == [D0]
        assert c0.recv_open_solicit(D0, timeout=2.0).mission_round == 1
    finally:
        c0.close()
        server.close()


def test_solicit_counts_a_channel_drop_as_sent():
    # The mule cannot observe a channel loss, only the missing advert.
    server = _server(emulator=ChannelEmulator(drop_prob=1.0, seed=0))
    client = _client(server, D0)
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        assert server.solicit(_solicit(1), [D0]) == [D0]
        with pytest.raises(RFLinkError):
            client.recv_open_solicit(D0, timeout=0.3)
    finally:
        client.close()
        server.close()


def test_solicit_refuses_a_bare_id_and_a_closed_link():
    server = _server()
    try:
        with pytest.raises(TypeError):
            server.solicit(_solicit(1), D0)
    finally:
        server.close()
    with pytest.raises(RFLinkError):
        server.solicit(_solicit(1), [D0])


def test_broadcast_is_unchanged_next_to_targeted_solicits():
    server = _server()
    clients = [_client(server, d) for d in (D0, D1)]
    try:
        assert server.wait_for_devices([D0, D1], timeout=2.0)
        server.solicit(_solicit(1), [D0])
        server.broadcast_open_solicit(_solicit(2))
        assert clients[0].recv_open_solicit(D0, timeout=2.0).mission_round == 1
        assert clients[0].recv_open_solicit(D0, timeout=2.0).mission_round == 2
        assert clients[1].recv_open_solicit(D1, timeout=2.0).mission_round == 2
    finally:
        for c in clients:
            c.close()
        server.close()


def test_client_side_refuses_solicit():
    server = _server()
    client = _client(server, D0)
    try:
        with pytest.raises(NotImplementedError):
            client.solicit(_solicit(1), [D0])
    finally:
        client.close()
        server.close()


# --------------------------------------------------------------------------- #
# Newest-solicit option (critic B1)
# --------------------------------------------------------------------------- #

def test_newest_solicit_only_client_answers_the_newest_queued_solicit():
    server = _server()
    newest = _client(server, D0, newest_solicit_only=True)
    fifo = _client(server, D1)
    try:
        assert server.wait_for_devices([D0, D1], timeout=2.0)
        for r in (1, 2, 3):
            server.broadcast_open_solicit(_solicit(r))
        # Wait until the reader threads have queued all three.
        assert _wait_until(
            lambda: newest._solicit_q.qsize() == 3 and fifo._solicit_q.qsize() == 3
        )

        assert newest.recv_open_solicit(D0, timeout=1.0).mission_round == 3
        assert newest.stale_solicits_dropped == 2
        with pytest.raises(RFLinkError, match="timed out"):
            newest.recv_open_solicit(D0, timeout=0.2)

        # Off (the default): arrival order, nothing dropped.
        got = [fifo.recv_open_solicit(D1, timeout=1.0).mission_round for _ in range(3)]
        assert got == [1, 2, 3]
        assert fifo.stale_solicits_dropped == 0
    finally:
        newest.close()
        fifo.close()
        server.close()


# --------------------------------------------------------------------------- #
# The send bound each socket holds
# --------------------------------------------------------------------------- #

def test_rf_sockets_hold_their_send_bound_in_the_platform_unit():
    server = _server()  # send_timeout_s default 30 s
    client = _client(server, D0)  # send_timeout_s default 60 s
    try:
        assert server.wait_for_devices([D0], timeout=2.0)
        assert _read_sndtimeo_s(server._sockets[D0]) == pytest.approx(30.0, abs=0.02)
        assert _read_sndtimeo_s(client._sock) == pytest.approx(60.0, abs=0.02)
        # Reads have no timeout at the Python level either.
        assert server._sockets[D0].gettimeout() is None
        assert client._sock.gettimeout() is None
        # A re-dial reads the acknowledgement under the connect timeout, then
        # goes back to blocking reads with the same send bound.
        old_sock = server._sockets[D0]
        client.reconnect()
        assert _wait_until(lambda: server._sockets.get(D0) not in (None, old_sock))
        assert _read_sndtimeo_s(server._sockets[D0]) == pytest.approx(30.0, abs=0.02)
        assert _read_sndtimeo_s(client._sock) == pytest.approx(60.0, abs=0.02)
        assert server._sockets[D0].gettimeout() is None
        assert client._sock.gettimeout() is None
    finally:
        client.close()
        server.close()


def test_dock_socket_holds_60_s_not_60_ms():
    server = TCPDockLinkServer(host="127.0.0.1", port=0)  # send_timeout_s 60 s
    server.start()
    raw = socket.create_connection((server.host, server.port), timeout=2.0)
    try:
        send_message(raw, _MuleRegistrationMessage(mule_id=MuleID("m1")))
        assert server.wait_for_mules([MuleID("m1")], timeout=2.0)
        assert _read_sndtimeo_s(server._sockets[MuleID("m1")]) == pytest.approx(60.0, abs=0.02)
    finally:
        raw.close()
        server.close()


def test_dock_send_waits_for_a_slow_reader():
    """The cluster's send_down outlasts a mule that is slow to read.

    With the old packing, Windows read the 5 s bound as 5 ms, so the second
    send below failed as soon as the buffers were full and the cluster
    dropped the mule. (Windows takes one send whole into an idle pipe; the
    second has to wait for the reader.)
    """
    server = TCPDockLinkServer(host="127.0.0.1", port=0, send_timeout_s=5.0)
    server.start()
    mule = MuleID("m1")
    raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    raw.connect((server.host, server.port))
    got: list = []
    try:
        send_message(raw, _MuleRegistrationMessage(mule_id=mule))
        assert server.wait_for_mules([mule], timeout=2.0)
        server._sockets[mule].setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)

        def _slow_reader():
            time.sleep(0.5)  # the mule is busy; the sends have to wait for it
            got.append(recv_message(raw))
            got.append(recv_message(raw))

        reader = threading.Thread(target=_slow_reader, daemon=True)
        reader.start()

        def _down(cluster_round: int) -> DownBundle:
            return DownBundle(
                mule_id=mule,
                mission_slice=MissionSlice(
                    mule_id=mule, device_ids=(D0,), issued_round=cluster_round,
                    issued_at=0.0,
                ),
                theta_disc=[np.zeros(1_000_000, dtype=np.float32)],
                synth_batch=[],
                cluster_amendments=ClusterAmendment(cluster_round=cluster_round),
            )

        def _send_both() -> float:
            t0 = time.monotonic()
            server.send_down(_down(1))
            server.send_down(_down(2))
            return time.monotonic() - t0

        assert _run_within(10.0, _send_both, "send_down to a slow mule") < 5.0
        reader.join(timeout=10.0)
        assert [d.cluster_amendments.cluster_round for d in got] == [1, 2]
        assert server.registered_mules() == [mule]
    finally:
        raw.close()
        server.close()
