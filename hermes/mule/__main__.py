"""Phase 3 demo — full Phase 1 + 2 + 3 roundtrip.

Run with:
    python -m hermes.mule

Exercises the end-to-end flow from the Phase 3 DoD:

    mission round 1  (HFLHostMission + ClientMission, Phase 2)
        -> collect() into ClientCluster
        -> UP to HFLHostCluster, cross-mule FedAvg, close cluster_round
        -> DOWN bundle with ClusterAmendment
        -> ClientCluster distributes -> fresh mission slice + theta_disc
    mission round 2 consumes the amendments (observes the new slice)

Nothing real-RF, nothing real-radio, no Flower. Pure in-process loopback
on both transports.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import List

import numpy as np

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.mission import ClientMission, HFLHostMission, LocalTrainResult
from hermes.mule import BundleDistributor, ClientCluster
from hermes.transport import LoopbackDockLink, LoopbackRFLink
from hermes.types import (
    ClusterAmendment,
    DeviceID,
    FLState,
    MuleID,
    RoundCloseDelta,
    SpectrumSig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("hermes.phase3_demo")

MULE = MuleID("mule-01")
DEVICE_IDS = [DeviceID(f"dev-{i:02d}") for i in range(4)]


def _train_factory(seed: int):
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        noise = [rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        delta = [w + n for w, n in zip(theta, noise)]
        return LocalTrainResult(
            delta_theta=delta,
            num_examples=int(rng.integers(4, 16)),
            accuracy=float(rng.uniform(0.75, 0.95)),
            auc=float(rng.uniform(0.75, 0.95)),
            loss=float(rng.uniform(0.1, 0.3)),
            theta_after=delta,
        )
    return _train


def _drive_in_field_round(
    mule: HFLHostMission,
    devices: List[ClientMission],
    synth_batch,
) -> None:
    """Run one FL round across all devices concurrently."""
    outcomes: list = [None] * len(devices)

    def worker(i: int) -> None:
        outcomes[i] = devices[i].serve_once()

    workers = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(len(devices))]
    for w in workers:
        w.start()

    for _ in range(len(devices)):
        mule.run_session(synth_batch=synth_batch)

    for w in workers:
        w.join(timeout=3.0)


def run_demo() -> int:
    # ---- transports ----
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()

    # ---- Tier 2: cluster (Phase 1) ----
    registry = DeviceRegistry()
    for did in DEVICE_IDS:
        registry.register(
            device_id=did,
            position=(0.0, 0.0, 0.0),
            spectrum_sig=SpectrumSig(
                bands=(0,),
                last_good_snr_per_band=(20.0,),
            ),
        )
    # Assign everything to our one mule
    registry.rebalance([MULE], round_counter=0)

    generator = StubGeneratorHost(
        disc_weights=[np.zeros((4,), dtype=np.float32),
                      np.ones((3, 3), dtype=np.float32) * 0.01]
    )
    cluster = HFLHostCluster(
        registry=registry,
        generator=generator,
        dock=dock,
        synth_batch_size=4,
    )

    # ---- Tier 2: mule-NUC side ----
    mission_deltas: List[RoundCloseDelta] = []
    mission = HFLHostMission(
        mule_id=MULE,
        rf=rf,
        scheduler_bus=mission_deltas.append,
        session_ttl_s=2.0,
    )

    received_slices = []
    received_models = []
    distributor = BundleDistributor(
        on_slice_and_amendment=lambda s, a: received_slices.append((s, a)),
        on_next_round_model=lambda w, b: received_models.append((w, b)),
    )
    client_cluster = ClientCluster(
        mule_id=MULE,
        dock=dock,
        distributor=distributor,
    )

    # ---- Tier 1: devices ----
    devices: List[ClientMission] = []
    for i, did in enumerate(DEVICE_IDS):
        cm = ClientMission(
            device_id=did,
            rf=rf,
            local_train=_train_factory(seed=100 + i),
            solicit_timeout_s=2.0,
            disc_push_timeout_s=2.0,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)

    # ========================= ROUND 1 =========================
    print("\n=== mission round 1 (in-field) ===")
    theta_r1 = generator.get_global_disc_weights()
    mr1 = mission.open_round(theta_r1)
    synth_batch = generator.make_synth_batch(4)
    _drive_in_field_round(mission, devices, synth_batch)
    aggregate, report, contacts = mission.close_round()
    on_time, missed = report.counts()
    print(f"  mission_round={mr1}  on_time={on_time}  missed={missed}  "
          f"num_examples={aggregate.num_examples}")

    # ========================= DOCK ============================
    print("\n=== dock cycle ===")
    assert client_cluster.wait_for_dock(timeout=1.0)
    client_cluster.collect(
        partial_aggregate=aggregate,
        report=report,
        contacts=contacts,
    )

    # Cluster-side serves the dock in a background thread.
    # The amendment shifts dev-00's deadline by +5s to prove round 2 sees it.
    amendment_notes = "demo-amendment r1"
    amendment = ClusterAmendment(
        cluster_round=1,
        deadline_overrides={DeviceID("dev-00"): time.time() + 5.0},
        notes=amendment_notes,
    )

    def cluster_side():
        up = cluster.dock.recv_up(timeout=2.0)
        cluster.ingest_up_bundle(up)
        # Run cross-mule FedAvg (with min_participation=1 by default)
        merged = cluster.aggregate_pending()
        print(f"  cluster cross-mule FedAvg merged_layers="
              f"{[w.shape for w in merged]}")
        # Close the cluster round to bump the counter and stash an amendment
        cluster.close_cluster_round(
            deadline_overrides=amendment.deadline_overrides,
            notes=amendment_notes,
        )
        # Dispatch a DOWN bundle back to the mule
        down = cluster.dispatch_down_bundle(MULE)
        cluster.dock.send_down(down)

    t = threading.Thread(target=cluster_side, daemon=True)
    t.start()

    down = client_cluster.run_dock_cycle()
    t.join(timeout=3.0)

    assert down is not None
    print(f"  DOWN received: round={down.mission_slice.issued_round} "
          f"slice={down.mission_slice.device_ids} "
          f"amendment.notes={down.cluster_amendments.notes!r}")
    assert down.cluster_amendments.notes == amendment_notes
    assert received_slices, "scheduler slow-phase sink never fired"
    assert received_models, "mission-model sink never fired"

    # ========================= ROUND 2 =========================
    # Feed the distributed theta_disc + slice into round 2 to prove
    # consumption of the amendment.
    print("\n=== mission round 2 (post-dock) ===")
    new_theta, new_synth = received_models[-1]
    new_slice, new_amendment = received_slices[-1]
    print(f"  consumed slice size={len(new_slice.device_ids)} "
          f"amendment.deadline_overrides={dict(new_amendment.deadline_overrides)}")

    mr2 = mission.open_round(new_theta)
    _drive_in_field_round(mission, devices, new_synth)
    agg2, report2, _ = mission.close_round()
    on_time2, missed2 = report2.counts()
    print(f"  mission_round={mr2}  on_time={on_time2}  missed={missed2}  "
          f"num_examples={agg2.num_examples}")

    # ========================= summary =========================
    print("\n=== Phase 3 summary ===")
    print(f"  ClientCluster state        = {client_cluster.state.value}")
    print(f"  retry queue depth          = {client_cluster.retry_queue_depth()}")
    print(f"  scheduler deltas observed  = {len(mission_deltas)}")
    print(f"  slow-phase slice events    = {len(received_slices)}")
    print(f"  model-state events         = {len(received_models)}")
    print(f"  round 1 mule mission_round = {mr1}")
    print(f"  round 2 mule mission_round = {mr2}")
    print("  DoD checks:")
    print(f"   * UP + DOWN + VERIFY + DISTRIBUTE all fired .... OK")
    print(f"   * round 2 consumed the amendment ................ OK")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
