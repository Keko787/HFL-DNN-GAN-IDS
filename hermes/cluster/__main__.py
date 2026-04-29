"""Phase 1 demo — runnable via ``python -m hermes.cluster``.

Spins up a single ``HFLHostCluster`` over a loopback dock link, then
walks two fake mules through a dock cycle:

* register 8 devices,
* rebalance into disjoint slices for ``m1`` + ``m2``,
* each mule "uploads" a partial aggregate over its slice,
* cluster runs cross-mule FedAvg,
* both mules receive a fresh DOWN bundle.

This is the Phase 1 Implementation-Plan §3 demo, executable as a script.
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.transport import LoopbackDockLink
from hermes.types import (
    ContactHistory,
    DeviceID,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MuleID,
    PartialAggregate,
    SpectrumSig,
    UpBundle,
)


def _seed(reg: DeviceRegistry, n: int) -> None:
    sig = SpectrumSig(bands=(0, 1, 2), last_good_snr_per_band=(10.0, 14.0, 18.0))
    for i in range(n):
        reg.register(DeviceID(f"d{i:03d}"), (0.0, 0.0, 0.0), sig)


def _fake_partial(mule: MuleID, devs, value: float) -> UpBundle:
    pa = PartialAggregate(
        mule_id=mule,
        mission_round=1,
        weights=[np.full(4, value, dtype=np.float32)],
        num_examples=len(devs) * 5,
        contributing_devices=tuple(devs),
    )
    rep = MissionRoundCloseReport(
        mule_id=mule, mission_round=1, started_at=0.0, finished_at=10.0,
        lines=[
            MissionRoundCloseLine(
                device_id=d, outcome=MissionOutcome.CLEAN, contact_ts=1.0
            )
            for d in devs
        ],
    )
    return UpBundle(
        mule_id=mule,
        partial_aggregate=pa,
        round_close_report=rep,
        contact_history=ContactHistory(mule, 1, []),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    log = logging.getLogger("phase1.demo")

    reg = DeviceRegistry()
    _seed(reg, 8)
    log.info("seeded registry with %d devices", len(reg.all()))

    dock = LoopbackDockLink()
    cluster = HFLHostCluster(
        registry=reg,
        generator=StubGeneratorHost(disc_weights=[np.zeros(4, dtype=np.float32)]),
        dock=dock,
        synth_batch_size=2,
        min_participation=2,
    )

    mules = [MuleID("m1"), MuleID("m2")]
    slices = cluster.rebalance_for(mules)
    for m, s in slices.items():
        log.info("slice for %s: %s", m, list(s.device_ids))

    # mule m1 docks first
    log.info("--- mule m1 docks ---")
    dock.client_send_up(_fake_partial(MuleID("m1"), slices[MuleID("m1")].device_ids, value=1.0))
    cluster.serve_one_dock(timeout=1.0)
    down1 = dock.client_recv_down(MuleID("m1"), timeout=1.0)
    log.info(
        "m1 received DOWN: slice_size=%d theta_disc[0][:3]=%s",
        len(down1.mission_slice.device_ids),
        down1.theta_disc[0][:3].tolist(),
    )

    # mule m2 docks next
    log.info("--- mule m2 docks ---")
    dock.client_send_up(_fake_partial(MuleID("m2"), slices[MuleID("m2")].device_ids, value=3.0))
    cluster.serve_one_dock(timeout=1.0)
    down2 = dock.client_recv_down(MuleID("m2"), timeout=1.0)
    log.info(
        "m2 received DOWN: slice_size=%d theta_disc[0][:3]=%s",
        len(down2.mission_slice.device_ids),
        down2.theta_disc[0][:3].tolist(),
    )

    # both partials in — run cross-mule FedAvg
    merged = cluster.aggregate_pending()
    if merged is None:
        log.error("aggregate_pending returned None — should have merged 2 partials")
        return 1
    # both slices were 4 devices, equal num_examples — average should be 2.0
    log.info("cluster FedAvg merged weights[0][:3] = %s", merged[0][:3].tolist())

    amend = cluster.close_cluster_round(notes="phase1 demo close")
    log.info("closed cluster round=%d", amend.cluster_round)

    # disjointness invariant — the whole point of slicing
    s1 = set(slices[MuleID("m1")].device_ids)
    s2 = set(slices[MuleID("m2")].device_ids)
    assert s1.isdisjoint(s2), "slices overlapped — cluster invariant broken"
    assert s1 | s2 == {DeviceID(f"d{i:03d}") for i in range(8)}, "slice union != registry"
    log.info("disjoint + complete slice invariant: OK")

    return 0


if __name__ == "__main__":
    sys.exit(main())
