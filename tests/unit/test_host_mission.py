"""Phase 2 HFLHostMission tests.

Cover: open/close round, gradient verify paths (CLEAN / PARTIAL / TIMEOUT),
RoundCloseDelta bus fan-out, busy-flag TTL.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from hermes.mission import HFLHostMission, MissionSessionError
from hermes.transport import LoopbackRFLink, RFLinkError
from hermes.types import (
    DeviceID,
    FLOpenSolicit,
    FLReadyAdv,
    FLState,
    GradientSubmission,
    MissionOutcome,
    MuleID,
    RoundCloseDelta,
    weights_signature,
)


def _dev(n: str) -> DeviceID:
    return DeviceID(n)


def _mule() -> MuleID:
    return MuleID("mA")


def _zero_theta():
    return [np.zeros((3,), dtype=np.float32)]


def _adv(
    device: str,
    *,
    state: FLState = FLState.FL_OPEN,
    utility: float = 0.5,
) -> FLReadyAdv:
    return FLReadyAdv(
        device_id=_dev(device),
        state=state,
        performance_score=utility,
        diversity_adjusted=0.0,
        utility=utility,
        issued_at=time.time(),
    )


def _good_grad(device: str, mission_round: int) -> GradientSubmission:
    delta = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
    return GradientSubmission(
        device_id=_dev(device),
        mule_id=_mule(),
        mission_round=mission_round,
        delta_theta=delta,
        num_examples=4,
        submitted_at=time.time(),
    )


def _host(rf, bus_sink=None):
    return HFLHostMission(
        mule_id=_mule(),
        rf=rf,
        scheduler_bus=bus_sink or (lambda d: None),
        session_ttl_s=0.25,  # tight timeouts for tests
        busy_ttl_s=0.5,
    )


def test_close_round_without_open_raises():
    host = _host(LoopbackRFLink())
    with pytest.raises(MissionSessionError):
        host.close_round()


def test_open_round_monotonically_increments():
    host = _host(LoopbackRFLink())
    assert host.open_round(_zero_theta()) == 1
    # simulate failed round that raises on close; still bumps the counter
    try:
        host.close_round()
    except MissionSessionError:
        pass
    assert host.open_round(_zero_theta()) == 2


def test_happy_path_accepts_clean_gradient():
    rf = LoopbackRFLink()
    deltas = []
    host = _host(rf, bus_sink=deltas.append)
    round_ = host.open_round(_zero_theta())

    # stage the device-side reply and gradient ahead of time
    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1"))
    # device must submit gradient after solicit+push — we pre-queue it
    rf.send_gradient(_good_grad("d1", round_))

    outcome = host.run_session(synth_batch=[])
    assert outcome is MissionOutcome.CLEAN
    assert host.accepted_count() == 1

    # RoundCloseDelta fired onto the bus
    assert len(deltas) == 1
    d = deltas[0]
    assert isinstance(d, RoundCloseDelta)
    assert d.outcome is MissionOutcome.CLEAN
    assert d.mission_round == round_


def test_bad_checksum_yields_partial():
    rf = LoopbackRFLink()
    host = _host(rf)
    round_ = host.open_round(_zero_theta())

    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1"))

    # Corrupt checksum
    bad = _good_grad("d1", round_)
    bad.checksum = "00" * 32
    rf.send_gradient(bad)

    outcome = host.run_session(synth_batch=[])
    assert outcome is MissionOutcome.PARTIAL
    assert host.accepted_count() == 0


def test_bad_byte_count_yields_partial():
    rf = LoopbackRFLink()
    host = _host(rf)
    round_ = host.open_round(_zero_theta())

    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1"))

    bad = _good_grad("d1", round_)
    bad.byte_count = 1  # obvious mismatch
    rf.send_gradient(bad)

    outcome = host.run_session(synth_batch=[])
    assert outcome is MissionOutcome.PARTIAL


def test_wrong_round_yields_partial():
    rf = LoopbackRFLink()
    host = _host(rf)
    host.open_round(_zero_theta())  # round 1

    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1"))
    rf.send_gradient(_good_grad("d1", mission_round=99))

    outcome = host.run_session(synth_batch=[])
    assert outcome is MissionOutcome.PARTIAL


def test_gradient_timeout_yields_timeout():
    rf = LoopbackRFLink()
    host = _host(rf)
    host.open_round(_zero_theta())

    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1"))
    # do NOT enqueue a gradient; recv_gradient should time out

    outcome = host.run_session(synth_batch=[])
    assert outcome is MissionOutcome.TIMEOUT


def test_refuses_session_when_device_unavailable():
    rf = LoopbackRFLink()
    host = _host(rf)
    host.open_round(_zero_theta())
    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1", state=FLState.UNAVAILABLE))

    outcome = host.run_session(synth_batch=[])
    assert outcome is MissionOutcome.PARTIAL  # refused counts as partial contact
    assert host.accepted_count() == 0


def test_refuses_session_when_below_min_utility():
    rf = LoopbackRFLink()
    host = _host(rf)
    host.open_round(_zero_theta())
    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1", utility=0.01))

    outcome = host.run_session(synth_batch=[], min_utility=0.5)
    assert outcome is MissionOutcome.PARTIAL


def test_no_device_answers_returns_none():
    rf = LoopbackRFLink()
    host = _host(rf)
    host.open_round(_zero_theta())

    outcome = host.run_session(synth_batch=[])
    assert outcome is None


def test_close_round_runs_partial_fedavg():
    rf = LoopbackRFLink()
    host = _host(rf)
    round_ = host.open_round(_zero_theta())
    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1"))
    rf.send_gradient(_good_grad("d1", round_))
    host.run_session(synth_batch=[])

    agg, report, contacts = host.close_round()
    assert agg.mule_id == _mule()
    assert agg.mission_round == round_
    assert agg.num_examples == 4
    assert report.mule_id == _mule()
    on_time, missed = report.counts()
    assert (on_time, missed) == (1, 0)
    assert len(contacts.records) == 1


def test_busy_flag_expires_after_ttl():
    rf = LoopbackRFLink()
    host = _host(rf)  # busy_ttl_s = 0.5
    host.open_round(_zero_theta())
    rf.register_device(_dev("d1"))
    rf.send_ready_adv(_adv("d1"))
    # do not send gradient so session hits TIMEOUT, releasing the busy flag
    host.run_session(synth_batch=[])
    # after timeout, is_busy should be False (released in the timeout branch)
    assert host.is_busy(_dev("d1")) is False
