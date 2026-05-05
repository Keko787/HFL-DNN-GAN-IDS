"""Sprint 1.5 — two-pass + per-contact integration smoke tests.

Drives ``HFLHostMission.run_contact`` (Pass 1 parallel exchange) and
``HFLHostMission.deliver_contact`` (Pass 2 push-only) against multiple
``ClientMission`` instances over the loopback RF link.

What this pins down:
* `run_contact(devices, synth)` serves N≥1 devices in parallel, each
  device's outcome recorded individually.
* The N=1 case (a one-device contact) is the same code path as N=K.
* `train_offline()` populates `_prepared_delta`; `serve_once` ships it
  without running `local_train` inline (no compute during the contact).
* `open_pass_2` + `deliver_contact` push θ' to every device and collect
  DeliveryAck per device; `MissionDeliveryReport` records DELIVERED /
  UNDELIVERED outcomes correctly.
* Backward compat: `serve_once` falls back to inline `local_train` when
  no prepared delta is staged (so existing Phase 3 demo + Sprint 1A
  tests keep working).
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List

import numpy as np
import pytest

from hermes.mission import ClientMission, HFLHostMission, LocalTrainResult
from hermes.transport import LoopbackRFLink
from hermes.types import (
    DeliveryOutcome,
    DeviceID,
    FLState,
    MissionOutcome,
    MissionPass,
    MuleID,
)


MULE = MuleID("mule-test")


def _train_factory(seed: int):
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        noise = [rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        delta = [w + n for w, n in zip(theta, noise)]
        return LocalTrainResult(
            delta_theta=delta,
            num_examples=int(rng.integers(4, 16)),
            accuracy=float(rng.uniform(0.7, 0.9)),
            auc=float(rng.uniform(0.7, 0.9)),
            loss=float(rng.uniform(0.1, 0.3)),
            theta_after=delta,
        )
    return _train


def _toy_theta() -> List[np.ndarray]:
    return [
        np.zeros((4,), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32) * 0.01,
    ]


def _toy_synth() -> List[np.ndarray]:
    return [np.zeros((2, 4), dtype=np.float32)]


def _make_devices(n: int, rf: LoopbackRFLink) -> List[ClientMission]:
    devices = []
    for i in range(n):
        did = DeviceID(f"dev-{i:02d}")
        rf.register_device(did)
        cm = ClientMission(
            device_id=did,
            rf=rf,
            local_train=_train_factory(seed=100 + i),
            solicit_timeout_s=2.0,
            disc_push_timeout_s=2.0,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)
    return devices


def _drive_serve_once(devices: List[ClientMission]) -> List[threading.Thread]:
    """Spawn one worker thread per device, each calling serve_once()."""
    workers = []
    for cm in devices:
        t = threading.Thread(target=cm.serve_once, daemon=True)
        t.start()
        workers.append(t)
    return workers


def _drive_serve_delivery(devices: List[ClientMission]) -> List[threading.Thread]:
    workers = []
    for cm in devices:
        t = threading.Thread(target=cm.serve_delivery, daemon=True)
        t.start()
        workers.append(t)
    return workers


# --------------------------------------------------------------------------- #
# Pass-1 run_contact smoke
# --------------------------------------------------------------------------- #

def test_run_contact_parallel_serves_all_in_range_devices():
    """N=3 contact event — every device's outcome is recorded clean."""
    rf = LoopbackRFLink()
    devices = _make_devices(3, rf)

    mission = HFLHostMission(
        mule_id=MULE, rf=rf, scheduler_bus=lambda d: None, session_ttl_s=2.0,
    )
    mission.open_round(_toy_theta())

    workers = _drive_serve_once(devices)
    outcomes = mission.run_contact(
        contact_devices=[d.device_id for d in devices],
        synth_batch=_toy_synth(),
    )
    for w in workers:
        w.join(timeout=3.0)

    assert len(outcomes) == 3
    assert all(o is MissionOutcome.CLEAN for o in outcomes.values()), (
        f"expected all clean, got {outcomes}"
    )
    assert mission.accepted_count() == 3


def test_run_contact_n_equals_one_works():
    """The N=1 degenerate case uses the same code path with no special branch."""
    rf = LoopbackRFLink()
    devices = _make_devices(1, rf)

    mission = HFLHostMission(
        mule_id=MULE, rf=rf, scheduler_bus=lambda d: None, session_ttl_s=2.0,
    )
    mission.open_round(_toy_theta())

    workers = _drive_serve_once(devices)
    outcomes = mission.run_contact(
        contact_devices=[devices[0].device_id], synth_batch=_toy_synth(),
    )
    for w in workers:
        w.join(timeout=3.0)

    assert outcomes == {devices[0].device_id: MissionOutcome.CLEAN}
    assert mission.accepted_count() == 1


def test_run_contact_in_wrong_pass_raises():
    """Calling run_contact while in DELIVER mode is a programming error."""
    rf = LoopbackRFLink()
    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.1)
    mission.open_round(_toy_theta())
    mission.open_pass_2(_toy_theta())

    with pytest.raises(Exception, match="run_contact called in pass=deliver"):
        mission.run_contact(
            contact_devices=[DeviceID("d0")], synth_batch=_toy_synth(),
        )


# --------------------------------------------------------------------------- #
# Pass-2 deliver_contact + DeliveryAck
# --------------------------------------------------------------------------- #

def test_deliver_contact_pushes_theta_and_records_delivery():
    """Pass-2 push-only flow: every device acks; report records DELIVERED."""
    rf = LoopbackRFLink()
    devices = _make_devices(2, rf)

    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=2.0)
    mission.open_round(_toy_theta())
    new_theta = [
        np.full((4,), 7.0, dtype=np.float32),
        np.full((3, 3), 0.5, dtype=np.float32),
    ]
    mission.open_pass_2(new_theta)

    workers = _drive_serve_delivery(devices)
    outcomes = mission.deliver_contact(
        contact_devices=[d.device_id for d in devices],
        synth_batch=_toy_synth(),
    )
    for w in workers:
        w.join(timeout=3.0)

    assert all(o is DeliveryOutcome.DELIVERED for o in outcomes.values())

    report = mission.close_pass_2()
    delivered, undelivered = report.counts()
    assert delivered == 2
    assert undelivered == 0
    assert sorted(report.delivered()) == sorted(d.device_id for d in devices)


def test_deliver_contact_records_undelivered_when_device_silent():
    """A silent device (no worker) ends up as UNDELIVERED in the report."""
    rf = LoopbackRFLink()
    devices = _make_devices(2, rf)

    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.5)
    mission.open_round(_toy_theta())
    mission.open_pass_2(_toy_theta())

    # Only ONE device's worker runs — the other is silent.
    workers = _drive_serve_delivery([devices[0]])
    outcomes = mission.deliver_contact(
        contact_devices=[d.device_id for d in devices],
        synth_batch=_toy_synth(),
    )
    for w in workers:
        w.join(timeout=2.0)

    assert outcomes[devices[0].device_id] is DeliveryOutcome.DELIVERED
    assert outcomes[devices[1].device_id] is DeliveryOutcome.UNDELIVERED

    report = mission.close_pass_2()
    delivered, undelivered = report.counts()
    assert delivered == 1
    assert undelivered == 1


def test_deliver_contact_in_wrong_pass_raises():
    rf = LoopbackRFLink()
    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.1)
    mission.open_round(_toy_theta())
    # Stays in COLLECT mode without open_pass_2.

    with pytest.raises(Exception, match="deliver_contact called in pass=collect"):
        mission.deliver_contact(
            contact_devices=[DeviceID("d0")], synth_batch=_toy_synth(),
        )


def test_close_pass_2_without_open_raises():
    rf = LoopbackRFLink()
    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.1)
    mission.open_round(_toy_theta())

    with pytest.raises(Exception, match="close_pass_2 called without open_pass_2"):
        mission.close_pass_2()


# --------------------------------------------------------------------------- #
# train_offline + serve_once exchange-only
# --------------------------------------------------------------------------- #

def test_serve_once_falls_back_to_inline_train_without_prepared_delta():
    """Backward compat: pre-Sprint-1.5 tests don't call train_offline first."""
    rf = LoopbackRFLink()
    devices = _make_devices(1, rf)
    cm = devices[0]
    # No train_offline() call — _prepared_delta is None.

    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=2.0)
    mission.open_round(_toy_theta())

    workers = _drive_serve_once([cm])
    outcomes = mission.run_contact(
        contact_devices=[cm.device_id], synth_batch=_toy_synth(),
    )
    for w in workers:
        w.join(timeout=3.0)

    assert outcomes[cm.device_id] is MissionOutcome.CLEAN
    assert mission.accepted_count() == 1


def test_train_offline_populates_prepared_delta_and_serve_once_uses_it():
    """The Sprint-1.5 path: train_offline() between visits, serve_once uses it."""
    rf = LoopbackRFLink()

    # Track the local_train call count to prove inline training was NOT
    # called during the contact.
    call_count = {"n": 0}
    rng = np.random.default_rng(0)

    def counting_train(theta, synth):
        call_count["n"] += 1
        delta = [w + rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        return LocalTrainResult(
            delta_theta=delta,
            num_examples=8,
            accuracy=0.8, auc=0.8, loss=0.2,
            theta_after=delta,
        )

    did = DeviceID("dev-x")
    rf.register_device(did)
    cm = ClientMission(
        device_id=did, rf=rf, local_train=counting_train,
        solicit_timeout_s=2.0, disc_push_timeout_s=2.0,
    )
    cm.set_state(FLState.FL_OPEN)

    # Step 1: simulate a previous Pass-2 delivery so the device has a θ basis.
    cm._set_theta_basis(_toy_theta(), _toy_synth())  # internal helper

    # Step 2: device runs offline training between visits — this must fire
    # local_train exactly once.
    result = cm.train_offline()
    assert result is not None
    assert call_count["n"] == 1, "train_offline should call local_train"
    assert cm._prepared_delta is not None  # internal: prepared slot populated

    # Step 3: now run a Pass-1 contact. The contact itself MUST NOT call
    # local_train — it should ship the prepared Δθ.
    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=2.0)
    mission.open_round(_toy_theta())

    worker = threading.Thread(target=cm.serve_once, daemon=True)
    worker.start()

    outcomes = mission.run_contact(
        contact_devices=[did], synth_batch=_toy_synth(),
    )
    worker.join(timeout=3.0)

    assert outcomes[did] is MissionOutcome.CLEAN
    # local_train was called ONCE total — during train_offline, not during serve_once.
    assert call_count["n"] == 1, (
        f"local_train called during contact (expected 0 in-session calls), "
        f"total calls={call_count['n']}"
    )
    # Prepared slot was consumed.
    assert cm._prepared_delta is None
