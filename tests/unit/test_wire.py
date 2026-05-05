"""Sprint 2 — wire-format unit tests.

Pins down the framing + serialization contract that ``TCPRFLink`` and
``TCPDockLink`` ride on. Real socket I/O is exercised in the link-level
integration tests; here we verify that round-tripping arbitrary HERMES
messages works and that the framing rejects pathological inputs.
"""

from __future__ import annotations

import io
import socket
import threading
from contextlib import closing
from typing import Tuple

import numpy as np
import pytest

from hermes.transport.wire import (
    MAX_FRAME_BYTES,
    WireError,
    decode_message,
    encode_message,
    recv_message,
    send_message,
)
from hermes.types import (
    DeliveryAck,
    DiscPush,
    DeviceID,
    FLOpenSolicit,
    MissionPass,
    MuleID,
)


# --------------------------------------------------------------------------- #
# Pure encode/decode round-trips
# --------------------------------------------------------------------------- #

# S2-L1: framing is now [4-byte magic 'HRMS'][u8 version][u32 length][body].
# Body starts at offset 9 = _HEADER_SIZE.
_HEADER_OFFSET = 9


def test_encode_decode_round_trip_simple_dict():
    msg = {"hello": "world", "n": 42}
    frame = encode_message(msg)
    # Header (9 bytes) + body
    assert frame[:4] == b"HRMS"
    assert frame[4] == 1  # version byte
    assert decode_message(frame[_HEADER_OFFSET:]) == msg


def test_round_trip_fl_open_solicit():
    msg = FLOpenSolicit(
        mule_id=MuleID("m1"),
        mission_round=3,
        issued_at=1000.0,
        pass_kind=MissionPass.DELIVER,
    )
    frame = encode_message(msg)
    decoded = decode_message(frame[_HEADER_OFFSET:])
    assert decoded == msg
    assert decoded.pass_kind is MissionPass.DELIVER


def test_round_trip_disc_push_with_numpy_arrays():
    """Numpy arrays survive pickling intact; this is the hot-path message."""
    theta = [
        np.zeros((4,), dtype=np.float32),
        np.full((3, 3), 7.0, dtype=np.float32),
    ]
    synth = [np.arange(8, dtype=np.float32)]
    msg = DiscPush(
        mule_id=MuleID("m1"),
        mission_round=1,
        theta_disc=theta,
        synth_batch=synth,
        pass_kind=MissionPass.COLLECT,
    )
    frame = encode_message(msg)
    decoded = decode_message(frame[_HEADER_OFFSET:])
    assert decoded.mission_round == 1
    assert decoded.pass_kind is MissionPass.COLLECT
    assert len(decoded.theta_disc) == 2
    np.testing.assert_array_equal(decoded.theta_disc[1], np.full((3, 3), 7.0, dtype=np.float32))
    np.testing.assert_array_equal(decoded.synth_batch[0], np.arange(8, dtype=np.float32))


def test_round_trip_delivery_ack():
    msg = DeliveryAck(
        device_id=DeviceID("d0"),
        mule_id=MuleID("m1"),
        mission_round=2,
        weights_sig="sha256:abc",
        received_at=42.0,
    )
    decoded = decode_message(encode_message(msg)[_HEADER_OFFSET:])
    assert decoded == msg


# --------------------------------------------------------------------------- #
# Frame-size guard
# --------------------------------------------------------------------------- #

def test_encode_rejects_oversize():
    """A message that would serialise > MAX_FRAME_BYTES must be rejected."""
    # Build a payload larger than MAX_FRAME_BYTES via numpy; using bytes
    # directly keeps memory pressure manageable.
    big = bytes(MAX_FRAME_BYTES + 1)
    with pytest.raises(WireError, match="too large"):
        encode_message(big)


# --------------------------------------------------------------------------- #
# socket-level send/recv via a localhost pair
# --------------------------------------------------------------------------- #

def _make_socket_pair() -> Tuple[socket.socket, socket.socket]:
    """Return (client, server-accepted) connected TCP sockets on localhost."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    accepted: list[socket.socket] = []
    accept_done = threading.Event()

    def _accept():
        s, _ = listener.accept()
        accepted.append(s)
        accept_done.set()

    t = threading.Thread(target=_accept, daemon=True)
    t.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    accept_done.wait(timeout=2.0)
    listener.close()
    return client, accepted[0]


def test_socket_round_trip_tiny_message():
    client, server = _make_socket_pair()
    try:
        send_message(client, {"x": 1})
        out = recv_message(server, timeout=1.0)
        assert out == {"x": 1}
    finally:
        client.close()
        server.close()


def test_socket_round_trip_disc_push():
    client, server = _make_socket_pair()
    try:
        msg = DiscPush(
            mule_id=MuleID("m1"),
            mission_round=1,
            theta_disc=[np.ones((10,), dtype=np.float32)],
            synth_batch=[np.zeros((4, 4), dtype=np.float32)],
        )
        send_message(client, msg)
        out = recv_message(server, timeout=1.0)
        assert out.mission_round == 1
        np.testing.assert_array_equal(out.theta_disc[0], np.ones((10,), dtype=np.float32))
    finally:
        client.close()
        server.close()


def test_recv_message_raises_on_peer_close_mid_frame():
    """A peer that closes mid-body (after a full header) must raise WireError.

    Earlier this test wrote only 4 bytes — the pre-Sprint-2 header size.
    With the 9-byte ``HRMS`` + version + length-prefix header it
    accidentally exercised "header truncated" instead of "body
    truncated"; both raise the same error so the test stayed green for
    the wrong reason. We now send the *full* 9-byte header announcing
    100 body bytes and close before any body is sent — the receiver
    completes the header read, then peer-closes while waiting on body.
    """
    client, server = _make_socket_pair()
    try:
        full_header = b"HRMS" + bytes([1]) + (100).to_bytes(4, "big")
        client.sendall(full_header)
        client.close()
        with pytest.raises(WireError, match="peer closed mid-frame"):
            recv_message(server, timeout=1.0)
    finally:
        server.close()


def test_recv_message_rejects_pathological_size():
    client, server = _make_socket_pair()
    try:
        # Send a complete header (magic + version + bad length).
        # Length claims 1 GiB — well under uint32 max but over MAX_FRAME_BYTES.
        bad_header = b"HRMS" + bytes([1]) + (MAX_FRAME_BYTES + 1).to_bytes(4, "big")
        client.sendall(bad_header)
        with pytest.raises(WireError, match="refusing pathological"):
            recv_message(server, timeout=1.0)
    finally:
        client.close()
        server.close()


def test_recv_message_rejects_bad_magic():
    """S2-L1: a non-hermes peer (e.g. curl POSTing) gets a clear error."""
    client, server = _make_socket_pair()
    try:
        # 9 bytes that don't start with the magic.
        client.sendall(b"GET / HTT")
        with pytest.raises(WireError, match="bad frame magic"):
            recv_message(server, timeout=1.0)
    finally:
        client.close()
        server.close()


def test_recv_message_rejects_unsupported_version():
    """S2-L1: future-version peer gets rejected before we trust the body."""
    client, server = _make_socket_pair()
    try:
        # Magic OK, but version 99 is in the future.
        bad = b"HRMS" + bytes([99]) + (4).to_bytes(4, "big") + b"junk"
        client.sendall(bad)
        with pytest.raises(WireError, match="unsupported wire version"):
            recv_message(server, timeout=1.0)
    finally:
        client.close()
        server.close()


def test_recv_message_restores_socket_timeout():
    """S2-H1: ``timeout=`` argument doesn't leak into the next blocking call."""
    client, server = _make_socket_pair()
    try:
        server.settimeout(5.0)
        # Send + recv one message with a tight per-call timeout.
        send_message(client, {"x": 1})
        recv_message(server, timeout=0.5)
        # The previous timeout (5.0) must be restored on the way out.
        assert server.gettimeout() == 5.0
    finally:
        client.close()
        server.close()


def test_recv_message_restores_none_timeout():
    """S2-H1: a None-timeout (blocking) socket isn't sticky-changed by recv_message."""
    client, server = _make_socket_pair()
    try:
        server.settimeout(None)  # blocking forever
        send_message(client, {"x": 1})
        recv_message(server, timeout=0.5)
        assert server.gettimeout() is None
    finally:
        client.close()
        server.close()


def test_recv_message_timeout():
    client, server = _make_socket_pair()
    try:
        # No data sent — server.recv should time out.
        with pytest.raises(WireError):
            recv_message(server, timeout=0.1)
    finally:
        client.close()
        server.close()
