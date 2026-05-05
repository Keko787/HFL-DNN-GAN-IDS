"""EX-1.1 protocol unit tests — frame round-trip + control / bulk distinctions."""

from __future__ import annotations

import socket
import threading

import pytest

from experiments.exp1 import protocol as proto


def _make_socket_pair() -> tuple[socket.socket, socket.socket]:
    """Return a (client, server) connected pair on a free local port."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    server, _ = listener.accept()
    listener.close()
    return client, server


def test_control_round_trip():
    c, s = _make_socket_pair()
    try:
        proto.send_control(c, {"type": "REGISTER", "client_id": "d1",
                                "data_partition": 0})
        msg = proto.recv_control(s)
        assert msg == {"type": "REGISTER", "client_id": "d1", "data_partition": 0}
    finally:
        c.close(); s.close()


def test_bulk_round_trip():
    c, s = _make_socket_pair()
    try:
        payload = bytes(1024)
        proto.send_bulk(c, payload)
        body = proto.recv_bulk(s)
        assert len(body) == 1024
    finally:
        c.close(); s.close()


def test_recv_control_rejects_bulk_frame():
    c, s = _make_socket_pair()
    try:
        proto.send_bulk(c, b"hello")
        with pytest.raises(proto.WireError, match="expected CONTROL"):
            proto.recv_control(s)
    finally:
        c.close(); s.close()


def test_recv_bulk_rejects_control_frame():
    c, s = _make_socket_pair()
    try:
        proto.send_control(c, {"hi": 1})
        with pytest.raises(proto.WireError, match="expected BULK"):
            proto.recv_bulk(s)
    finally:
        c.close(); s.close()


def test_send_control_oversized_rejected():
    c, s = _make_socket_pair()
    try:
        big = {"k": "x" * (proto.MAX_FRAME_BYTES + 1)}
        with pytest.raises(proto.WireError, match="too large"):
            proto.send_control(c, big)
    finally:
        c.close(); s.close()


def test_recv_exactly_raises_on_peer_close_mid_frame():
    """A peer that closes after only the type byte must raise."""
    c, s = _make_socket_pair()
    try:
        c.sendall(bytes([proto.FrameType.CONTROL]))  # only 1 byte; need 5
        c.close()
        with pytest.raises(proto.WireError, match="peer closed"):
            proto.recv_control(s)
    finally:
        s.close()


def test_recv_control_rejects_pathological_length():
    """A claimed length > MAX_FRAME_BYTES must surface, not block forever."""
    c, s = _make_socket_pair()
    try:
        bad = bytes([proto.FrameType.CONTROL]) + (
            (proto.MAX_FRAME_BYTES + 1).to_bytes(4, "big")
        )
        c.sendall(bad)
        with pytest.raises(proto.WireError, match="pathological"):
            proto.recv_control(s)
    finally:
        c.close(); s.close()


def test_recv_control_rejects_unknown_frame_type():
    c, s = _make_socket_pair()
    try:
        bad = bytes([0xFF]) + (0).to_bytes(4, "big")
        c.sendall(bad)
        with pytest.raises(proto.WireError, match="unknown frame type"):
            proto.recv_control(s)
    finally:
        c.close(); s.close()


def test_recv_control_rejects_non_dict_json():
    """JSON that decodes to a list is a protocol violation."""
    c, s = _make_socket_pair()
    try:
        body = b"[1,2,3]"
        header = bytes([proto.FrameType.CONTROL]) + len(body).to_bytes(4, "big")
        c.sendall(header + body)
        with pytest.raises(proto.WireError, match="must decode to a dict"):
            proto.recv_control(s)
    finally:
        c.close(); s.close()


def test_message_constructors_round_trip_through_send():
    """All make_* constructors produce valid CONTROL bodies."""
    c, s = _make_socket_pair()
    try:
        proto.send_control(c, proto.make_register("d1", 0))
        proto.send_control(c, proto.make_accepted("d1"))
        proto.send_control(c, proto.make_rejected("nope"))
        proto.send_control(c, proto.make_trial_begin(
            arm="FL", cell_id="x", trial_index=0, seed=1,
            arm_params={"theta_bytes": 1024, "n_rounds": 5},
        ))
        proto.send_control(c, proto.make_round_begin(0))
        proto.send_control(c, proto.make_ack(0))
        proto.send_control(c, proto.make_trial_end())
        proto.send_control(c, proto.make_shutdown())

        for expected_type in (
            proto.MSG_REGISTER, proto.MSG_ACCEPTED, proto.MSG_REJECTED,
            proto.MSG_TRIAL_BEGIN, proto.MSG_ROUND_BEGIN, proto.MSG_ACK,
            proto.MSG_TRIAL_END, proto.MSG_SHUTDOWN,
        ):
            msg = proto.recv_control(s)
            assert msg["type"] == expected_type
    finally:
        c.close(); s.close()
