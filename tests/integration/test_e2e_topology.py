"""Sprint 2 chunk N — full §4 happy-path end-to-end on the multi-process topology.

This is the chunk-N acceptance gate. The loopback two-pass test
(``test_mule_supervisor_two_pass.py``) already confirms the supervisor
mechanics are correct in-process; the orchestrator lifecycle test
(``test_orchestrator.py``) confirms the subprocesses come up and shut
down cleanly. What's missing — and what this file owns — is a real
mission cycle running through real TCP transports between real
subprocesses, end-to-end.

Topology: 1 cluster + 1 mule + 1 device. The mule is capped at
``n_missions=1`` so it exits as soon as a Pass-1 + dock + Pass-2 cycle
completes; the test then asserts the JSONL event logs each process
wrote contain the §4 happy-path transition events.

We don't re-verify in-process correctness here (that's other suites'
job). We pin the multi-process *integration* surface:

* The mule walks one full mission and exits naturally.
* The cluster ingests at least one UP bundle and closes a round.
* The device serves at least once during the mule's visit.
* Every process emits the §4 milestone events into its JSONL log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pytest

from hermes.processes import (
    ClusterConfig,
    DeviceConfig,
    MuleConfig,
    MultiProcessOrchestrator,
    TopologyConfig,
)


def _e2e_topology() -> TopologyConfig:
    """1c + 1m + 1d, mule capped to one mission for natural exit.

    ``n_serves=None`` on the device — the orchestrator's shutdown_all
    kills the device after the mule has finished. We don't try to cap
    n_serves precisely because the exact serve count depends on which
    Pass-2 contact ordering the supervisor picks and how many parallel
    sessions land in each contact (N=1 here, but the count would change
    in larger topologies).
    """
    return TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="cluster-e2e",
            dock_host="127.0.0.1",
            dock_port=0,
            synth_batch_size=2,
            min_participation=1,
        ),
        mules=[
            MuleConfig(
                mule_id="mule-e2e-1",
                rf_host="127.0.0.1",
                rf_port=0,
                rf_range_m=60.0,
                session_ttl_s=3.0,
                n_missions=1,  # natural exit after one mission cycle
            ),
        ],
        devices=[
            DeviceConfig(
                device_id="dev-e2e-1",
                position=(0.0, 0.0, 0.0),
            ),
        ],
    )


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _events_of(rows: List[dict], name: str) -> List[dict]:
    return [r for r in rows if r["event"] == name]


@pytest.mark.slow
def test_e2e_one_mission_runs_pass_1_dock_pass_2():
    """Full §4 happy path on real subprocesses + real TCP."""
    topo = _e2e_topology()
    orch = MultiProcessOrchestrator(topo, capture_output=True)

    try:
        orch.start_all(timeout=20.0)

        # The mule has n_missions=1 so it will exit on its own once the
        # full Pass-1 → dock → Pass-2 cycle completes. We block on its
        # process handle so we know the mission really finished before
        # we read the logs. Generous timeout — the cycle includes a
        # bootstrap-DOWN wait, two RF round-trips per device per pass,
        # and one dock round-trip in the middle.
        mule_handle = orch.mule_handles["mule-e2e-1"]
        try:
            mule_handle.proc.wait(timeout=30.0)
        except Exception as e:
            stderr_tail = mule_handle.stderr_tail(80)
            pytest.fail(
                f"mule did not complete one mission within 30s: {e}\n"
                f"--- mule stderr tail ---\n{stderr_tail}"
            )

        assert mule_handle.returncode() == 0, (
            f"mule exited with code {mule_handle.returncode()}\n"
            f"--- stderr tail ---\n{mule_handle.stderr_tail(80)}"
        )
    finally:
        # cleanup_tmpdir=False so we can read the JSONL files below.
        orch.shutdown_all(timeout=10.0, cleanup_tmpdir=False)

    try:
        run_dir = orch.tmpdir
        cluster_rows = _read_jsonl(run_dir / "cluster-cluster-e2e.jsonl")
        mule_rows = _read_jsonl(run_dir / "mule-mule-e2e-1.jsonl")
        device_rows = _read_jsonl(run_dir / "device-dev-e2e-1.jsonl")

        # ---------------------------- Cluster ---------------------------- #
        # cluster_ready, mule_bootstrapped, ≥1 up_bundle_ingested,
        # ≥1 cluster_round_closed (since min_participation=1, every
        # ingest closes a round).
        assert _events_of(cluster_rows, "cluster_ready"), \
            f"cluster never emitted cluster_ready: {cluster_rows!r}"
        assert _events_of(cluster_rows, "mule_bootstrapped"), \
            "cluster never bootstrapped the mule"
        assert _events_of(cluster_rows, "up_bundle_ingested"), \
            "cluster never ingested an UP bundle"
        assert _events_of(cluster_rows, "cluster_round_closed"), \
            "cluster never closed a round (FedAvg never ran)"

        # ----------------------------- Mule ------------------------------ #
        # mule_ready, dock_bootstrapped, mission_started, mission_completed.
        assert _events_of(mule_rows, "mule_ready")
        assert _events_of(mule_rows, "dock_bootstrapped"), \
            "mule never received bootstrap DOWN from cluster"
        assert _events_of(mule_rows, "mission_started")
        completed = _events_of(mule_rows, "mission_completed")
        assert completed, (
            "mule never emitted mission_completed; mission did not finish "
            "the Pass-1 → dock → Pass-2 cycle"
        )

        # The mission_completed event carries the delivery counts. With
        # one device and a successful Pass 2, we expect ≥1 delivered,
        # 0 undelivered.
        m = completed[0]
        assert m["pass_1_contacts"] >= 1
        assert m["pass_2_contacts"] >= 1
        assert m["delivered"] is not None
        assert m["delivered"] + (m["undelivered"] or 0) >= 1

        # ---------------------------- Device ----------------------------- #
        # device_ready, ≥2 device_served (one per pass).
        assert _events_of(device_rows, "device_ready")
        served = _events_of(device_rows, "device_served")
        assert len(served) >= 2, (
            f"device served {len(served)} time(s); expected ≥2 "
            f"(one per pass): events={[r['event'] for r in device_rows]}"
        )
    finally:
        orch.cleanup()
