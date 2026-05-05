"""Sprint 2 chunk M — JSONL events fire end-to-end through the orchestrator.

Spawns the smallest viable topology (1 cluster + 1 mule + 1 device) under
the multi-process orchestrator, then asserts each process emitted its
``*_ready`` event into its own JSONL file under the orchestrator's tmpdir.

We deliberately don't assert on mission-cycle events here — that's chunk
N's e2e happy path. Chunk M only owns the wiring contract: every spawned
process sees ``--run-dir`` and writes a JSONL file with at least the
canonical envelope fields.

We also don't assert on the ``metrics_snapshot`` / ``service_stopped``
events that fire from each process's ``shutdown()``. On Windows
``Popen.terminate()`` maps to ``TerminateProcess``, which is uncatchable
hard-kill — Python's ``finally`` blocks (where the snapshot is emitted)
never run. The snapshot mechanism is unit-tested separately; this test
is for the wiring contract.
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


def _smallest_topology() -> TopologyConfig:
    return TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="cluster-obs",
            dock_host="127.0.0.1",
            dock_port=0,
            synth_batch_size=2,
            min_participation=1,
        ),
        mules=[
            MuleConfig(
                mule_id="mule-obs-1",
                rf_host="127.0.0.1",
                rf_port=0,
                rf_range_m=60.0,
                session_ttl_s=3.0,
            ),
        ],
        devices=[
            DeviceConfig(
                device_id="dev-obs-1",
                position=(0.0, 0.0, 0.0),
            ),
        ],
    )


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.mark.slow
def test_each_process_writes_jsonl_with_ready_event():
    topo = _smallest_topology()
    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=15.0)
        # Give each process a moment to flush its ready event before
        # we tear down. Without this the cluster's ready event may
        # still be in the Python logging buffer when SIGTERM lands.
        time.sleep(0.5)
    finally:
        # Defer tmpdir cleanup so we can read the JSONL files.
        orch.shutdown_all(timeout=5.0, cleanup_tmpdir=False)

    try:
        run_dir = orch.tmpdir
        cluster_log = run_dir / "cluster-cluster-obs.jsonl"
        mule_log = run_dir / "mule-mule-obs-1.jsonl"
        device_log = run_dir / "device-dev-obs-1.jsonl"

        assert cluster_log.exists(), f"missing {cluster_log}"
        assert mule_log.exists(), f"missing {mule_log}"
        assert device_log.exists(), f"missing {device_log}"

        # 1. Each file's first event is the role-specific ready event.
        cluster_rows = _read_jsonl(cluster_log)
        mule_rows = _read_jsonl(mule_log)
        device_rows = _read_jsonl(device_log)

        assert cluster_rows[0]["event"] == "cluster_ready"
        assert cluster_rows[0]["role"] == "cluster"
        assert cluster_rows[0]["id"] == "cluster-obs"
        assert cluster_rows[0]["dock_port"] > 0

        assert mule_rows[0]["event"] == "mule_ready"
        assert mule_rows[0]["role"] == "mule"
        assert mule_rows[0]["id"] == "mule-obs-1"
        assert mule_rows[0]["rf_port"] > 0

        assert device_rows[0]["event"] == "device_ready"
        assert device_rows[0]["role"] == "device"
        assert device_rows[0]["id"] == "dev-obs-1"

        # 2. Every row carries the canonical envelope fields.
        for rows in (cluster_rows, mule_rows, device_rows):
            for row in rows:
                assert "ts" in row and isinstance(row["ts"], (int, float))
                assert row["schema_version"] >= 1
                assert "role" in row and "id" in row and "event" in row
    finally:
        orch.cleanup()
