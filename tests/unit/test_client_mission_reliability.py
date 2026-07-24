"""EX-4.2 — ClientMission short-range contact-reliability gate.

Pins the device<->mule uplink completion model in isolation: with
``contact_reliability`` set, a Pass-1 collect completes (CLEAN, gradient
sent) or drops (TIMEOUT, no gradient) per the Bernoulli draw. Default
(None) always completes — the EX-4.0/4.1 behaviour.
"""

from __future__ import annotations

import numpy as np

from hermes.mission import ClientMission, LocalTrainResult
from hermes.types import DeviceID, DiscPush, MissionOutcome, MissionPass, MuleID


class _FakeRF:
    def __init__(self):
        self.grads = []

    def register_device(self, _d):
        pass

    def send_gradient(self, g):
        self.grads.append(g)


def _local_train(theta, synth):
    return LocalTrainResult(
        delta_theta=[np.zeros(2, np.float32)],
        num_examples=5,
        theta_after=[np.zeros(2, np.float32)],
    )


def _collect_push():
    return DiscPush(
        mule_id=MuleID("m"),
        mission_round=0,
        theta_disc=[np.zeros(2, np.float32)],
        synth_batch=[],
        pass_kind=MissionPass.COLLECT,
    )


def test_reliability_zero_drops_uplink():
    rf = _FakeRF()
    cm = ClientMission(
        device_id=DeviceID("d"), rf=rf, local_train=_local_train,
        contact_reliability=0.0, contact_rng_seed=1,
    )
    outcome = cm._handle_collect_push(_collect_push())
    assert outcome is MissionOutcome.TIMEOUT
    assert rf.grads == []          # a failed uplink sends nothing


def test_reliability_one_always_completes():
    rf = _FakeRF()
    cm = ClientMission(
        device_id=DeviceID("d"), rf=rf, local_train=_local_train,
        contact_reliability=1.0, contact_rng_seed=1,
    )
    outcome = cm._handle_collect_push(_collect_push())
    assert outcome is MissionOutcome.CLEAN
    assert len(rf.grads) == 1


def test_reliability_none_is_backward_compatible():
    rf = _FakeRF()
    cm = ClientMission(device_id=DeviceID("d"), rf=rf, local_train=_local_train)
    assert cm._handle_collect_push(_collect_push()) is MissionOutcome.CLEAN
    assert len(rf.grads) == 1


def test_reliability_partial_is_stochastic_but_bounded():
    # Over many draws at p=0.5, a meaningful fraction both complete and drop.
    n_clean = 0
    for seed in range(40):
        rf = _FakeRF()
        cm = ClientMission(
            device_id=DeviceID("d"), rf=rf, local_train=_local_train,
            contact_reliability=0.5, contact_rng_seed=seed,
        )
        if cm._handle_collect_push(_collect_push()) is MissionOutcome.CLEAN:
            n_clean += 1
    assert 5 <= n_clean <= 35   # neither always-clean nor always-dropped
