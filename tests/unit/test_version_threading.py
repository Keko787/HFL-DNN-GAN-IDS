"""FeRRy Phase 1 — every update carries the version of the θ it was trained on.

The cluster round that produced θ rides the push; the device keeps it with
its training basis and echoes it with the update it later trains from that
basis; the mule turns it into an age on the report line and the partial.
"""

from __future__ import annotations

import socket
import threading
from typing import List

import numpy as np
import pytest

from hermes.mission import ClientMission, HFLHostMission, LocalTrainResult, MissionSessionError
from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec
from hermes.transport import LoopbackRFLink
from hermes.transport.wire import recv_message, send_message
from hermes.types import (
    DeviceID,
    DiscPush,
    FLState,
    GradientSubmission,
    MissionOutcome,
    MissionPass,
    MuleID,
)
from hermes.types.fl_messages import UPDATE_FORM_DELTA, UPDATE_FORM_WEIGHTS

MULE = MuleID("mule-v")
DEV = DeviceID("dev-v")


def _theta(value: float):
    return [np.full((3,), value, dtype=np.float32), np.full((2, 2), value, dtype=np.float32)]


def _plus_one(theta, synth):
    after = [w + 1.0 for w in theta]
    return LocalTrainResult(delta_theta=after, num_examples=8, loss=0.3, theta_after=after)


class _RecordingRF:
    """Just enough of an RF link for the device-side push handlers."""

    def __init__(self) -> None:
        self.gradients: List[GradientSubmission] = []
        self.acks = []

    def send_gradient(self, grad) -> None:
        self.gradients.append(grad)

    def send_delivery_ack(self, ack) -> None:
        self.acks.append(ack)


def _push(theta, version, *, form=UPDATE_FORM_WEIGHTS, kind=MissionPass.COLLECT):
    return DiscPush(
        mule_id=MULE, mission_round=1, theta_disc=theta, synth_batch=[],
        pass_kind=kind, basis_version=version, update_form=form,
    )


# --------------------------------------------------------------------------- #
# Over the wire
# --------------------------------------------------------------------------- #

def test_version_and_form_round_trip_over_a_socket():
    a, b = socket.socketpair()
    try:
        push = _push(_theta(1.0), 7, form=UPDATE_FORM_DELTA)
        grad = GradientSubmission(
            device_id=DEV, mule_id=MULE, mission_round=1,
            delta_theta=_theta(0.5), num_examples=4, submitted_at=0.0,
            basis_version=6, update_form=UPDATE_FORM_DELTA,
        )
        send_message(a, push)
        send_message(a, grad)
        got_push = recv_message(b, timeout=2.0)
        got_grad = recv_message(b, timeout=2.0)
    finally:
        a.close()
        b.close()
    assert got_push.basis_version == 7 and got_push.update_form == UPDATE_FORM_DELTA
    assert got_grad.basis_version == 6 and got_grad.update_form == UPDATE_FORM_DELTA
    assert got_grad.checksum == grad.checksum


def test_defaults_describe_every_recorded_run():
    push = DiscPush(mule_id=MULE, mission_round=1, theta_disc=_theta(0.0), synth_batch=[])
    assert push.basis_version is None and push.update_form == UPDATE_FORM_WEIGHTS


# --------------------------------------------------------------------------- #
# Device side
# --------------------------------------------------------------------------- #

def _client(rf, **kw) -> ClientMission:
    return ClientMission(device_id=DEV, rf=rf, local_train=_plus_one, **kw)


def test_fallback_update_echoes_the_pushed_version_in_full_weights():
    rf = _RecordingRF()
    cm = _client(rf)
    assert cm._handle_collect_push(_push(_theta(2.0), 4)) is MissionOutcome.CLEAN
    grad = rf.gradients[-1]
    assert grad.basis_version == 4 and grad.update_form == UPDATE_FORM_WEIGHTS
    assert all(np.array_equal(w, np.full_like(w, 3.0)) for w in grad.delta_theta)


def test_prepared_update_keeps_the_basis_it_was_trained_on():
    rf = _RecordingRF()
    cm = _client(rf)
    # Pass 2 delivers θ at version 5; the device trains ahead on it.
    cm._handle_delivery_push(_push(_theta(5.0), 5, kind=MissionPass.DELIVER))
    # The next Pass 1 carries a newer θ (version 6) and asks for a delta.
    assert cm._handle_collect_push(
        _push(_theta(6.0), 6, form=UPDATE_FORM_DELTA)
    ) is MissionOutcome.CLEAN
    grad = rf.gradients[-1]
    assert grad.basis_version == 5, "must name the basis trained on, not the newest"
    assert grad.update_form == UPDATE_FORM_DELTA
    # trained = θ5 + 1, so the delta against θ5 is exactly +1 everywhere
    assert all(np.array_equal(w, np.ones_like(w)) for w in grad.delta_theta)
    # ...and the device adopted θ6 as its next basis
    assert cm._theta_basis_version == 6


def test_dropped_contact_adopts_the_new_basis_but_ships_the_old_update_later():
    rf = _RecordingRF()
    cm = _client(rf, contact_reliability=0.0)
    cm._handle_delivery_push(_push(_theta(5.0), 5, kind=MissionPass.DELIVER))
    assert cm._handle_collect_push(_push(_theta(6.0), 6)) is MissionOutcome.TIMEOUT
    assert cm._theta_basis_version == 6 and not rf.gradients
    cm._contact_reliability = None
    cm._handle_collect_push(_push(_theta(7.0), 7))
    assert rf.gradients[-1].basis_version == 5


def test_delta_with_no_known_basis_sends_nothing():
    rf = _RecordingRF()
    cm = _client(rf)
    cm._prepared_delta = _plus_one(_theta(1.0), [])   # prepared outside train_offline
    out = cm._handle_collect_push(_push(_theta(1.0), 1, form=UPDATE_FORM_DELTA))
    assert out is MissionOutcome.PARTIAL and not rf.gradients


# --------------------------------------------------------------------------- #
# Mule side
# --------------------------------------------------------------------------- #

def _serve_one(cm: ClientMission) -> threading.Thread:
    t = threading.Thread(target=cm.serve_once, daemon=True)
    t.start()
    return t


def _mission_with_stale_device(spec: AggregationSpec):
    rf = LoopbackRFLink()
    rf.register_device(DEV)
    cm = ClientMission(
        device_id=DEV, rf=rf, local_train=_plus_one,
        solicit_timeout_s=2.0, disc_push_timeout_s=2.0,
    )
    cm.set_state(FLState.FL_OPEN)
    # The device trained ahead on θ at version 3 ...
    cm._set_theta_basis(_theta(3.0), [], version=3)
    cm.train_offline()
    # ... and the mule now carries θ at version 5.
    host = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=2.0, aggregation=spec)
    host.open_round(_theta(5.0), theta_version=5)
    t = _serve_one(cm)
    outcomes = host.run_contact([DEV], synth_batch=[])
    t.join(timeout=5.0)
    return host, outcomes


def test_report_line_and_partial_carry_the_age():
    host, outcomes = _mission_with_stale_device(AggregationSpec())
    assert outcomes[DEV] is MissionOutcome.CLEAN
    agg, report, _ = host.close_round()
    (line,) = report.lines
    assert line.basis_version == 3 and line.age == 2 and line.num_examples == 8
    assert agg.base_version == 5
    assert agg.device_ages == (2,) and agg.device_basis_versions == (3,)


def test_age_aware_rule_asks_for_a_delta_and_cuts_off_past_a_max():
    host, outcomes = _mission_with_stale_device(AggregationSpec(rule=AGG_CUTOFF, a_max=1))
    assert outcomes[DEV] is MissionOutcome.CLEAN
    with pytest.raises(MissionSessionError, match="past their age cutoff"):
        host.close_round(age_caps={DEV: 1})


def test_mule_refuses_an_update_in_the_wrong_form():
    rf = LoopbackRFLink()
    host = HFLHostMission(
        mule_id=MULE, rf=rf, session_ttl_s=2.0,
        aggregation=AggregationSpec(rule=AGG_CUTOFF),
    )
    host.open_round(_theta(0.0), theta_version=0)
    grad = GradientSubmission(
        device_id=DEV, mule_id=MULE, mission_round=host.mission_round,
        delta_theta=_theta(1.0), num_examples=4, submitted_at=__import__("time").time(),
        update_form=UPDATE_FORM_WEIGHTS,
    )
    assert host._verify_receipt(grad, host.mission_round) is MissionOutcome.PARTIAL
