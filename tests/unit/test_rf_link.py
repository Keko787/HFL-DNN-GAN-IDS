"""Phase 2 tests for the RF-link loopback."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from hermes.transport import LoopbackRFLink, RFLinkError
from hermes.types import (
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    FLState,
    GradientSubmission,
    MuleID,
)


def _dev(i: str) -> DeviceID:
    return DeviceID(i)


def _mule(i: str) -> MuleID:
    return MuleID(i)


def test_register_device_creates_queues():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    assert _dev("d1") in rf.known_devices()


def test_broadcast_solicit_reaches_all_registered_devices():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    rf.register_device(_dev("d2"))

    solicit = FLOpenSolicit(mule_id=_mule("m1"), mission_round=1, issued_at=0.0)
    rf.broadcast_open_solicit(solicit)

    assert rf.recv_open_solicit(_dev("d1"), timeout=0.1) == solicit
    assert rf.recv_open_solicit(_dev("d2"), timeout=0.1) == solicit


def test_ready_adv_is_fifo_across_devices():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    rf.register_device(_dev("d2"))

    for d in ("d1", "d2"):
        rf.send_ready_adv(
            FLReadyAdv(
                device_id=_dev(d),
                state=FLState.FL_OPEN,
                performance_score=0.5,
                diversity_adjusted=0.1,
                utility=0.35,
            )
        )
    first = rf.recv_ready_adv(timeout=0.1)
    second = rf.recv_ready_adv(timeout=0.1)
    assert {first.device_id, second.device_id} == {_dev("d1"), _dev("d2")}


def test_push_disc_is_unicast_per_device():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    rf.register_device(_dev("d2"))

    push = DiscPush(
        mule_id=_mule("m1"),
        mission_round=1,
        theta_disc=[np.zeros((2, 2), dtype=np.float32)],
        synth_batch=[],
    )
    rf.push_disc(_dev("d1"), push)
    # only d1 should see it
    received = rf.recv_disc_push(_dev("d1"), timeout=0.1)
    assert received.mule_id == _mule("m1")
    with pytest.raises(RFLinkError):
        rf.recv_disc_push(_dev("d2"), timeout=0.05)


def test_recv_times_out_cleanly():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    with pytest.raises(RFLinkError):
        rf.recv_open_solicit(_dev("d1"), timeout=0.05)
    with pytest.raises(RFLinkError):
        rf.recv_ready_adv(timeout=0.05)
    with pytest.raises(RFLinkError):
        rf.recv_gradient(_dev("d1"), timeout=0.05)


def test_send_gradient_routes_to_source_device_queue():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    grad = GradientSubmission(
        device_id=_dev("d1"),
        mule_id=_mule("m1"),
        mission_round=1,
        delta_theta=[np.zeros((3,), dtype=np.float32)],
        num_examples=4,
        submitted_at=time.time(),
    )
    rf.send_gradient(grad)
    got = rf.recv_gradient(_dev("d1"), timeout=0.1)
    assert got.device_id == _dev("d1")
    assert got.mission_round == 1


def test_closed_link_refuses_operations():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    rf.close()
    with pytest.raises(RFLinkError):
        rf.recv_open_solicit(_dev("d1"), timeout=0.01)
    with pytest.raises(RFLinkError):
        rf.send_ready_adv(
            FLReadyAdv(
                device_id=_dev("d1"),
                state=FLState.FL_OPEN,
                performance_score=0.0,
                diversity_adjusted=0.0,
                utility=0.0,
            )
        )


def test_concurrent_push_and_recv():
    rf = LoopbackRFLink()
    rf.register_device(_dev("d1"))
    received: list = []

    def reader():
        received.append(rf.recv_disc_push(_dev("d1"), timeout=1.0))

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)  # give reader a chance to block
    push = DiscPush(
        mule_id=_mule("m1"),
        mission_round=7,
        theta_disc=[np.zeros((1,), dtype=np.float32)],
        synth_batch=[],
    )
    rf.push_disc(_dev("d1"), push)
    t.join(timeout=1.0)
    assert len(received) == 1
    assert received[0].mission_round == 7
