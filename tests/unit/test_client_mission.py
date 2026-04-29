"""Phase 2 ClientMission tests — device side."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from hermes.mission import ClientMission, LocalTrainResult
from hermes.transport import LoopbackRFLink
from hermes.types import (
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLState,
    MissionOutcome,
    MuleID,
)


def _dev(n: str) -> DeviceID:
    return DeviceID(n)


def _mule() -> MuleID:
    return MuleID("mA")


def _fake_local_train(theta, synth):
    # pretend the device learned something — small non-zero delta, decent metrics
    delta = [w + 0.1 for w in theta]
    return LocalTrainResult(
        delta_theta=delta,
        num_examples=8,
        accuracy=0.85,
        auc=0.80,
        loss=0.20,
        theta_after=delta,
    )


def test_default_state_is_unavailable_and_utility_is_zero():
    rf = LoopbackRFLink()
    cm = ClientMission(
        device_id=_dev("d1"),
        rf=rf,
        local_train=_fake_local_train,
    )
    assert cm.state is FLState.UNAVAILABLE
    assert cm.last_utility == 0.0


def test_set_state_updates_adv_payload():
    rf = LoopbackRFLink()
    cm = ClientMission(
        device_id=_dev("d1"),
        rf=rf,
        local_train=_fake_local_train,
    )
    cm.set_state(FLState.FL_OPEN)
    adv = cm.build_ready_adv()
    assert adv.state is FLState.FL_OPEN
    assert adv.device_id == _dev("d1")


def test_beacon_fn_is_called():
    rf = LoopbackRFLink()
    bursts = []
    cm = ClientMission(
        device_id=_dev("d1"),
        rf=rf,
        local_train=_fake_local_train,
        beacon_fn=bursts.append,
    )
    cm.set_state(FLState.FL_OPEN)
    adv = cm.emit_beacon()
    assert len(bursts) == 1
    assert bursts[0].device_id == adv.device_id


def test_serve_once_returns_none_on_solicit_timeout():
    rf = LoopbackRFLink()
    cm = ClientMission(
        device_id=_dev("d1"),
        rf=rf,
        local_train=_fake_local_train,
        solicit_timeout_s=0.05,
    )
    cm.set_state(FLState.FL_OPEN)
    assert cm.serve_once() is None


def test_serve_once_refuses_when_unavailable():
    rf = LoopbackRFLink()
    cm = ClientMission(
        device_id=_dev("d1"),
        rf=rf,
        local_train=_fake_local_train,
        solicit_timeout_s=0.5,
    )
    cm.set_state(FLState.UNAVAILABLE)

    # Queue a solicit so we don't block forever
    rf.broadcast_open_solicit(
        FLOpenSolicit(mule_id=_mule(), mission_round=1, issued_at=time.time())
    )
    outcome = cm.serve_once()
    assert outcome is None

    # But the device still *replied* (so mule knows the state) — check the queue
    adv = rf.recv_ready_adv(timeout=0.1)
    assert adv.state is FLState.UNAVAILABLE


def test_serve_once_happy_path_sends_gradient_and_updates_utility():
    rf = LoopbackRFLink()
    cm = ClientMission(
        device_id=_dev("d1"),
        rf=rf,
        local_train=_fake_local_train,
        solicit_timeout_s=1.0,
        disc_push_timeout_s=1.0,
    )
    cm.set_state(FLState.FL_OPEN)

    # Orchestrate the mule side from a helper thread: send solicit, read adv,
    # send disc push, read gradient.
    done = threading.Event()
    seen = {}

    def mule_side():
        rf.broadcast_open_solicit(
            FLOpenSolicit(mule_id=_mule(), mission_round=1, issued_at=time.time())
        )
        seen["adv"] = rf.recv_ready_adv(timeout=1.0)
        push = DiscPush(
            mule_id=_mule(),
            mission_round=1,
            theta_disc=[np.zeros((3,), dtype=np.float32)],
            synth_batch=[],
        )
        rf.push_disc(_dev("d1"), push)
        seen["grad"] = rf.recv_gradient(_dev("d1"), timeout=1.0)
        done.set()

    t = threading.Thread(target=mule_side)
    t.start()

    outcome = cm.serve_once()
    t.join(timeout=2.0)

    assert outcome is MissionOutcome.CLEAN
    assert done.is_set()
    assert cm.last_utility > 0.0  # utility got updated from the metrics
    grad = seen["grad"]
    assert grad.device_id == _dev("d1")
    assert grad.num_examples == 8
    # checksum auto-populated by __post_init__
    assert len(grad.checksum) == 64


def test_local_train_exception_yields_partial():
    def explode(theta, synth):
        raise RuntimeError("boom")

    rf = LoopbackRFLink()
    cm = ClientMission(
        device_id=_dev("d1"),
        rf=rf,
        local_train=explode,
        solicit_timeout_s=1.0,
        disc_push_timeout_s=1.0,
    )
    cm.set_state(FLState.FL_OPEN)

    def mule_side():
        rf.broadcast_open_solicit(
            FLOpenSolicit(mule_id=_mule(), mission_round=1, issued_at=time.time())
        )
        rf.recv_ready_adv(timeout=1.0)
        rf.push_disc(
            _dev("d1"),
            DiscPush(
                mule_id=_mule(),
                mission_round=1,
                theta_disc=[np.zeros((3,), dtype=np.float32)],
                synth_batch=[],
            ),
        )

    t = threading.Thread(target=mule_side)
    t.start()
    outcome = cm.serve_once()
    t.join(timeout=2.0)
    assert outcome is MissionOutcome.PARTIAL
