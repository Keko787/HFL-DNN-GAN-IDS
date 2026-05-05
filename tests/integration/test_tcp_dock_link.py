"""Sprint 2 — TCPDockLink end-to-end (real localhost sockets).

Drives the cluster (server) + N mule clients over real TCP, exercising
every entry point of the DockLink ABC: registration, client_send_up,
recv_up, send_down, client_recv_down. Also covers the multi-mule
routing case (cluster sends DOWN to the right mule by mule_id).
"""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from hermes.transport import (
    DockLinkError,
    TCPDockLinkClient,
    TCPDockLinkServer,
)
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    DeviceID,
    DownBundle,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    UpBundle,
)


def _open_server() -> TCPDockLinkServer:
    s = TCPDockLinkServer(host="127.0.0.1", port=0)
    s.start()
    return s


def _connect(server: TCPDockLinkServer, mule_ids: List[str]) -> List[TCPDockLinkClient]:
    clients = []
    for m in mule_ids:
        c = TCPDockLinkClient(MuleID(m), host=server.host, port=server.port)
        clients.append(c)
    assert server.wait_for_mules(
        [MuleID(m) for m in mule_ids], timeout=2.0
    ), "server did not register all mules in time"
    return clients


def _aggregate(mule: MuleID, mission_round: int = 1) -> PartialAggregate:
    return PartialAggregate(
        mule_id=mule,
        mission_round=mission_round,
        weights=[np.zeros((4,), dtype=np.float32)],
        num_examples=10,
    )


def _round_report(mule: MuleID, mission_round: int = 1) -> MissionRoundCloseReport:
    return MissionRoundCloseReport(
        mule_id=mule,
        mission_round=mission_round,
        started_at=0.0,
        finished_at=1.0,
    )


def _mk_up(mule: MuleID, *, mission_round: int = 1) -> UpBundle:
    return UpBundle(
        mule_id=mule,
        partial_aggregate=_aggregate(mule, mission_round),
        round_close_report=_round_report(mule, mission_round),
        contact_history=ContactHistory(mule_id=mule, mission_round=mission_round),
    )


def _mk_down(mule: MuleID) -> DownBundle:
    return DownBundle(
        mule_id=mule,
        mission_slice=MissionSlice(
            mule_id=mule,
            device_ids=(DeviceID("d0"),),
            issued_round=1,
            issued_at=0.0,
        ),
        theta_disc=[np.zeros((4,), dtype=np.float32)],
        synth_batch=[np.zeros((2, 2), dtype=np.float32)],
        cluster_amendments=ClusterAmendment(cluster_round=1),
    )


# --------------------------------------------------------------------------- #
# Connection + registration
# --------------------------------------------------------------------------- #

def test_server_binds_and_mules_register():
    server = _open_server()
    try:
        clients = _connect(server, ["m1", "m2"])
        assert server.port > 0
        assert server.is_available()
        for c in clients:
            assert c.is_available()
    finally:
        for c in clients:
            c.close()
        server.close()


def test_server_rejects_non_registration_first_frame():
    import socket as _sock
    from hermes.transport.wire import send_message
    server = _open_server()
    raw = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    try:
        raw.connect((server.host, server.port))
        send_message(raw, {"hello": "wrong"})
        time.sleep(0.2)
        # Server should not have any registered mules.
        assert not server.wait_for_mules([MuleID("anyone")], timeout=0.2)
    finally:
        try:
            raw.close()
        except OSError:
            pass
        server.close()


# --------------------------------------------------------------------------- #
# UP / DOWN round trips
# --------------------------------------------------------------------------- #

def test_single_mule_up_and_down_round_trip():
    server = _open_server()
    try:
        clients = _connect(server, ["m1"])
        cm = clients[0]

        # Mule -> Cluster: UP
        cm.client_send_up(_mk_up(MuleID("m1")))
        up = server.recv_up(timeout=1.0)
        assert up.mule_id == MuleID("m1")
        assert up.partial_aggregate.mission_round == 1

        # Cluster -> Mule: DOWN
        server.send_down(_mk_down(MuleID("m1")))
        down = cm.client_recv_down(MuleID("m1"), timeout=1.0)
        assert down.mule_id == MuleID("m1")
    finally:
        for c in clients:
            c.close()
        server.close()


def test_multi_mule_down_routing():
    """Cluster sends DOWN to mule m2; mule m1 must NOT receive it."""
    server = _open_server()
    try:
        clients = _connect(server, ["m1", "m2"])
        c1, c2 = clients

        server.send_down(_mk_down(MuleID("m2")))

        # m2 sees its DOWN.
        d2 = c2.client_recv_down(MuleID("m2"), timeout=1.0)
        assert d2.mule_id == MuleID("m2")

        # m1 sees nothing.
        with pytest.raises(DockLinkError):
            c1.client_recv_down(MuleID("m1"), timeout=0.2)
    finally:
        for c in clients:
            c.close()
        server.close()


def test_multi_mule_up_collected_in_one_queue():
    """recv_up returns ANY mule's UP (loopback semantics preserved)."""
    server = _open_server()
    try:
        clients = _connect(server, ["m1", "m2"])
        c1, c2 = clients

        c1.client_send_up(_mk_up(MuleID("m1")))
        c2.client_send_up(_mk_up(MuleID("m2")))

        seen = set()
        for _ in range(2):
            up = server.recv_up(timeout=1.0)
            seen.add(up.mule_id)
        assert seen == {MuleID("m1"), MuleID("m2")}
    finally:
        for c in clients:
            c.close()
        server.close()


def test_send_down_to_unknown_mule_raises():
    server = _open_server()
    try:
        with pytest.raises(DockLinkError, match="no registered socket"):
            server.send_down(_mk_down(MuleID("ghost")))
    finally:
        server.close()


def test_client_send_up_with_wrong_mule_id_raises():
    server = _open_server()
    try:
        clients = _connect(server, ["m1"])
        c = clients[0]
        with pytest.raises(DockLinkError, match="bundle.mule_id"):
            c.client_send_up(_mk_up(MuleID("m2")))  # bundle for m2 sent on m1 client
    finally:
        for c in clients:
            c.close()
        server.close()


def test_client_recv_down_with_wrong_mule_id_raises():
    server = _open_server()
    try:
        clients = _connect(server, ["m1"])
        c = clients[0]
        with pytest.raises(DockLinkError, match="for 'm2' on client 'm1'"):
            c.client_recv_down(MuleID("m2"), timeout=0.1)
    finally:
        for c in clients:
            c.close()
        server.close()


# --------------------------------------------------------------------------- #
# Wrong-side calls
# --------------------------------------------------------------------------- #

def test_server_raises_on_client_side_calls():
    server = _open_server()
    try:
        with pytest.raises(NotImplementedError):
            server.client_send_up(_mk_up(MuleID("m1")))
    finally:
        server.close()


def test_client_raises_on_server_side_calls():
    server = _open_server()
    try:
        clients = _connect(server, ["m1"])
        c = clients[0]
        with pytest.raises(NotImplementedError):
            c.recv_up(timeout=0.1)
    finally:
        for c in clients:
            c.close()
        server.close()


# --------------------------------------------------------------------------- #
# Close + timeout semantics
# --------------------------------------------------------------------------- #

def test_close_propagates_to_recv_up():
    server = _open_server()
    server.close()
    with pytest.raises(DockLinkError, match="dock link closed"):
        server.recv_up(timeout=0.1)


def test_recv_up_timeout():
    server = _open_server()
    try:
        with pytest.raises(DockLinkError, match="recv_up timed out"):
            server.recv_up(timeout=0.1)
    finally:
        server.close()
