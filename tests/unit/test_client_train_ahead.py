"""Device-side train-ahead (FeRRy Phase 1 audit #0) under concurrency.

With a budgeted Pass 2 the mule marks its Pass-1 push ``train_ahead``, and the
device trains on the basis it just adopted on a background thread. Three paths
matter and none is reachable when the harness waits for the thread first:

* a Pass-1 push arriving while that fit still runs must wait for it and ship
  its result (trained on the older basis), not fall back to in-session
  training on the new θ, which would report age 0 again;
* a Pass-1 contact whose uplink drops still adopts the pushed basis and must
  train ahead on it;
* a fit overtaken by a newer basis (a delivery) is stale and is discarded.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List

import numpy as np

from hermes.mission import ClientMission, LocalTrainResult
from hermes.types import DeviceID, DiscPush, MissionOutcome, MuleID
from hermes.types.fl_messages import UPDATE_FORM_DELTA

DEV = DeviceID("dev-ta")
MULE = MuleID("mule-ta")
FALLBACK = "falling back to in-session training"


class _RF:
    """Just enough of an RF link to capture what the device sends."""

    def __init__(self) -> None:
        self.gradients = []

    def send_gradient(self, grad) -> None:
        self.gradients.append(grad)

    def send_delivery_ack(self, ack) -> None:  # pragma: no cover - unused
        pass


def _theta(v: float) -> List[np.ndarray]:
    return [np.full((3,), v, dtype=np.float32)]


def _slow_train(started: threading.Event, release: threading.Event):
    def _train(theta, synth):
        started.set()
        release.wait(timeout=5.0)
        after = [w + 0.5 for w in theta]
        return LocalTrainResult(
            delta_theta=after, num_examples=8, accuracy=0.8, auc=0.8,
            loss=0.2, theta_after=after,
        )
    return _train


def _push(version: int, *, train_ahead: bool = False) -> DiscPush:
    return DiscPush(
        mule_id=MULE, mission_round=version + 1, theta_disc=_theta(float(version)),
        synth_batch=[], basis_version=version, update_form=UPDATE_FORM_DELTA,
        train_ahead=train_ahead,
    )


def test_a_push_during_a_train_ahead_waits_and_ships_the_older_basis(caplog):
    started, release = threading.Event(), threading.Event()
    rf = _RF()
    cm = ClientMission(device_id=DEV, rf=rf, local_train=_slow_train(started, release))
    cm._set_theta_basis(_theta(0.0), [], version=0)
    cm._start_train_ahead()
    assert started.wait(timeout=5.0)            # the fit on basis 0 is running

    result = {}
    with caplog.at_level(logging.WARNING):
        t = threading.Thread(
            target=lambda: result.setdefault("oc", cm._handle_collect_push(_push(1))),
        )
        t.start()
        time.sleep(0.1)
        assert not rf.gradients                 # blocked behind the fit
        release.set()
        t.join(timeout=5.0)

    assert result["oc"] is MissionOutcome.CLEAN
    (grad,) = rf.gradients
    assert grad.basis_version == 0              # age 1 against the mule's θ_1
    # Delta against basis 0: after − basis = +0.5 everywhere.
    np.testing.assert_allclose(grad.delta_theta[0], np.full((3,), 0.5))
    assert not [r for r in caplog.records if FALLBACK in r.getMessage()]


def test_a_dropped_uplink_still_trains_ahead_on_the_pushed_basis():
    started, release = threading.Event(), threading.Event()
    release.set()
    cm = ClientMission(
        device_id=DEV, rf=_RF(), local_train=_slow_train(started, release),
        contact_reliability=0.0,
    )
    assert cm._handle_collect_push(_push(3, train_ahead=True)) is MissionOutcome.TIMEOUT
    cm._train_thread.join(timeout=5.0)
    assert cm._prepared_basis_version == 3
    assert cm._prepared_delta is not None


def test_no_train_ahead_without_the_flag():
    started, release = threading.Event(), threading.Event()
    release.set()
    cm = ClientMission(
        device_id=DEV, rf=_RF(), local_train=_slow_train(started, release),
        contact_reliability=0.0,
    )
    cm._handle_collect_push(_push(3))
    assert cm._train_thread is None and cm._prepared_delta is None


def test_a_fit_overtaken_by_a_newer_basis_is_discarded():
    started, release = threading.Event(), threading.Event()
    cm = ClientMission(device_id=DEV, rf=_RF(), local_train=_slow_train(started, release))
    cm._set_theta_basis(_theta(0.0), [], version=0)
    cm._start_train_ahead()
    assert started.wait(timeout=5.0)
    cm._set_theta_basis(_theta(1.0), [], version=1)   # a delivery lands mid-fit
    release.set()
    cm._train_thread.join(timeout=5.0)
    assert cm._prepared_delta is None and cm._prepared_basis_version is None
