"""Phase 2 demo — one mule, three devices, one mission round.

Run with:
    python -m hermes.mission

Exercises:
  * solicit -> FL_READY_ADV handshake over RF loopback
  * disc push -> local train -> gradient submission
  * receipt verification (checksum + byte count + round + TTL)
  * partial FedAvg across device gradients
  * per-device RoundCloseDelta onto the scheduler bus
  * MissionRoundCloseReport summary

No external radio / server is required. Prints a short summary + the
round-close report lines + the scheduler-bus deltas.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import List

import numpy as np

from hermes.mission import ClientMission, HFLHostMission, LocalTrainResult
from hermes.transport import LoopbackRFLink
from hermes.types import (
    DeviceID,
    FLState,
    MuleID,
    RoundCloseDelta,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("hermes.phase2_demo")


def fake_local_train_factory(seed: int):
    """Build a deterministic local-train callable for one device."""
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        noise = [rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        delta = [w + n for w, n in zip(theta, noise)]
        return LocalTrainResult(
            delta_theta=delta,
            num_examples=int(rng.integers(4, 32)),
            accuracy=float(rng.uniform(0.7, 0.95)),
            auc=float(rng.uniform(0.7, 0.95)),
            loss=float(rng.uniform(0.05, 0.30)),
            theta_after=delta,
        )

    return _train


def run_demo() -> int:
    rf = LoopbackRFLink()
    bus: List[RoundCloseDelta] = []

    # ---- initial θ_disc (small, deterministic) ----------------------------
    theta_disc = [
        np.zeros((4,), dtype=np.float32),
        np.ones((3, 3), dtype=np.float32) * 0.01,
    ]
    synth_batch = [np.zeros((2,), dtype=np.float32) for _ in range(4)]

    # ---- mule ----
    mule = HFLHostMission(
        mule_id=MuleID("mule-01"),
        rf=rf,
        scheduler_bus=bus.append,
        session_ttl_s=2.0,
    )
    mission_round = mule.open_round(theta_disc)
    log.info("opened mission_round=%d", mission_round)

    # ---- devices ----
    devices: List[ClientMission] = []
    device_ids = [DeviceID(f"dev-{i:02d}") for i in range(3)]
    for i, did in enumerate(device_ids):
        cm = ClientMission(
            device_id=did,
            rf=rf,
            local_train=fake_local_train_factory(seed=100 + i),
            solicit_timeout_s=2.0,
            disc_push_timeout_s=2.0,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)

    # Run each device's serve loop in its own thread.
    device_outcomes = [None] * len(devices)

    def device_worker(idx: int) -> None:
        device_outcomes[idx] = devices[idx].serve_once()

    workers = [
        threading.Thread(target=device_worker, args=(i,), daemon=True)
        for i in range(len(devices))
    ]
    for w in workers:
        w.start()

    # Drive one session at a time from the mule. Each run_session grabs
    # whichever device replies first; the other devices keep waiting.
    for _ in range(len(devices)):
        outcome = mule.run_session(synth_batch=synth_batch)
        log.info("mule run_session -> %s", outcome)

    for w in workers:
        w.join(timeout=3.0)

    # ---- close the round and inspect -------------------------------------
    aggregate, report, contacts = mule.close_round()

    print("\n=== Phase 2 demo summary ===")
    print(f"mule_id              = {aggregate.mule_id}")
    print(f"mission_round        = {aggregate.mission_round}")
    print(f"contributing_devices = {aggregate.contributing_devices}")
    print(f"aggregate.num_examples = {aggregate.num_examples}")
    print(f"aggregate layers     = {[w.shape for w in aggregate.weights]}")
    on_time, missed = report.counts()
    print(f"round_close: on_time={on_time} missed={missed}")
    for line in report.lines:
        print(
            f"  - dev={line.device_id} outcome={line.outcome.value} "
            f"bytes_in={line.bytes_received} bytes_out={line.bytes_sent}"
        )
    print(f"contact records: {len(contacts.records)}")
    print(f"scheduler-bus deltas: {len(bus)}")
    for d in bus:
        print(
            f"  - delta dev={d.device_id} outcome={d.outcome.value} "
            f"util={d.utility:.3f}"
        )

    # Invariant check: every CLEAN line matched a device in the aggregate
    clean_devs = {
        l.device_id for l in report.lines if l.outcome.is_on_time()
    }
    assert set(aggregate.contributing_devices) == clean_devs, (
        f"invariant broken: aggregate contributors {aggregate.contributing_devices}"
        f" vs clean lines {clean_devs}"
    )

    print("\nOK — invariants satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
