"""Sprint 2 chunk O — fault injection covering design doc §4.1.

§4.1 enumerates 11 exception paths. Not all of them are meaningfully
testable through the multi-process + TCP harness this chunk owns;
this file lands the rows that catch real regressions when something
breaks in Sprint-2 land. The rows that aren't here are documented
below with the rationale.

Covered here
------------

* **Pass 2 device unreachable (§4.1 row 5).**
  Force the device subprocess to exit after Pass 1 (``n_serves=1``).
  Assert the mule's ``mission_completed`` event reports
  ``undelivered >= 1, delivered == 0`` and that the supervisor still
  exits cleanly with rc=0 (the dropped device is *expected* fault
  behavior, not a crash).

* **Mule crash mid-flight does not kill the cluster (§4.1 row 3).**
  Spawn the topology, ``proc.kill()`` the mule, give the cluster's
  dock-reader thread a moment to surface the WireError, then assert
  the cluster process is still alive. Verifies the cluster survives a
  peer's hard exit instead of cascading.

* **Mission slice collision rejected at validation (§4.1 row 10).**
  Put one device in ``expected_devices`` of two different mules.
  ``TopologyConfig.validate()`` must raise
  ``TopologyValidationError`` — disjoint slicing is enforced before
  the orchestrator ever launches subprocesses.

* **Orchestrator's ``all_alive()`` detects a crashed peer.**
  Belt-and-suspenders for the production health check: kill any
  subprocess and ``orch.all_alive()`` returns False within one tick.

Documented as not in scope
--------------------------

* **Wire corruption — gradient checksum / TTL / DOWN verification
  failure (§4.1 rows 1, 8).** Would need to inject malformed pickle
  frames mid-stream. The wire-format invariants (bad magic, oversize,
  peer-close-mid-frame, unsupported version) are already covered by
  ``tests/unit/test_wire.py``. Re-testing through subprocess + TCP
  duplicates that coverage at higher cost.

* **Cross-mule race on same device (§4.1 row 6).** Needs >1 mule
  docking and precise timing on the in-RF-range overlap. The
  *prevention* (disjoint slicing) is pinned by the validate() test
  here; the runtime busy-flag is defense-in-depth and lives in
  scheduler unit tests.

* **Dock-link drop mid-UP (§4.1 row 7).** The subprocess-level timing
  window for "kill the cluster between UP frame send and ack" is too
  narrow to hit reliably from a Python test. The TCP-link unit tests
  (``test_tcp_dock_link.py``) cover the equivalent behavior at the
  transport layer.

* **Stale Δθ (§4.1 row 11).** Design notes the two-pass mission model
  makes this structurally impossible — no fault injection by design.

* **Pass 1 partial / TTL / per-device timeout (§4.1 rows 1, 2).** The
  per-device outcome semantics live in HFLHostMission's per-contact
  merge, covered by ``test_two_pass_contact.py`` at the in-process
  level. The multi-process layer doesn't change those semantics.
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
    TopologyValidationError,
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


# --------------------------------------------------------------------------- #
# §4.1 row 5 — Pass 2 device unreachable
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_pass_2_unreachable_device_marked_undelivered():
    """A device that exits after Pass 1 must be reported as undelivered.

    Setup: ``n_serves=1`` makes the device exit after handling one
    solicit (which covers Pass 1's collect). When the mule swings
    around for Pass 2, the RF link to that device is gone, so the
    delivery push cannot complete.
    """
    topo = TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="cluster-fault5",
            dock_host="127.0.0.1",
            dock_port=0,
            synth_batch_size=2,
            min_participation=1,
        ),
        mules=[
            MuleConfig(
                mule_id="mule-fault5",
                rf_host="127.0.0.1",
                rf_port=0,
                rf_range_m=60.0,
                session_ttl_s=2.0,
                n_missions=1,
            ),
        ],
        devices=[
            DeviceConfig(
                device_id="dev-fault5",
                position=(0.0, 0.0, 0.0),
                n_serves=1,  # exits after Pass 1
            ),
        ],
    )
    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=15.0)
        mule_handle = orch.mule_handles["mule-fault5"]
        try:
            mule_handle.proc.wait(timeout=30.0)
        except Exception as e:
            pytest.fail(
                f"mule did not finish mission: {e}\n"
                f"mule stderr:\n{mule_handle.stderr_tail(60)}"
            )
        assert mule_handle.returncode() == 0
    finally:
        orch.shutdown_all(timeout=5.0, cleanup_tmpdir=False)

    try:
        mule_rows = _read_jsonl(orch.tmpdir / "mule-mule-fault5.jsonl")
        completed = _events_of(mule_rows, "mission_completed")
        assert completed, "mule didn't emit mission_completed"
        m = completed[0]
        # The device was reachable in Pass 1 → delivered counts ought to
        # be 0 and undelivered counts ought to be ≥1.
        assert m["delivered"] == 0, (
            f"expected 0 deliveries, got {m['delivered']}; mission state: {m!r}"
        )
        assert m["undelivered"] is not None and m["undelivered"] >= 1, (
            f"expected ≥1 undelivered, got {m['undelivered']}; "
            f"mission state: {m!r}"
        )
    finally:
        orch.cleanup()


# --------------------------------------------------------------------------- #
# §4.1 row 3 — Mule crash mid-flight does not kill the cluster
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_mule_kill_does_not_crash_cluster():
    """SIGKILL-equivalent on the mule must leave the cluster running.

    The cluster's dock-reader thread receives a ``WireError`` on the
    abruptly-closed socket, drops the mule from ``registered_mules``,
    and continues serving. Without this property a single mule crash
    would cascade and take the whole topology down.
    """
    topo = TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="cluster-fault3",
            dock_host="127.0.0.1",
            dock_port=0,
            synth_batch_size=2,
            min_participation=1,
        ),
        mules=[
            MuleConfig(
                mule_id="mule-fault3",
                rf_host="127.0.0.1",
                rf_port=0,
                rf_range_m=60.0,
                session_ttl_s=3.0,
            ),
        ],
        devices=[
            DeviceConfig(
                device_id="dev-fault3",
                position=(0.0, 0.0, 0.0),
            ),
        ],
    )
    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=15.0)

        # Wait for the mule to actually dock so the cluster has a
        # registered socket to drop. Without this delay we might kill
        # the mule before the dock handshake, which exercises a
        # different code path.
        time.sleep(1.0)
        cluster_h = orch.cluster_handle
        assert cluster_h is not None and cluster_h.is_alive()

        mule_h = orch.mule_handles["mule-fault3"]
        mule_h.proc.kill()
        try:
            mule_h.proc.wait(timeout=5.0)
        except Exception:
            pytest.fail("mule did not die after SIGKILL")

        # Give the cluster's reader thread a tick to see the closed
        # socket and drop the mule.
        time.sleep(1.0)

        assert cluster_h.is_alive(), (
            "cluster crashed after mule kill\n"
            f"cluster stderr:\n{cluster_h.stderr_tail(60)}"
        )
    finally:
        orch.shutdown_all(timeout=5.0)


# --------------------------------------------------------------------------- #
# §4.1 row 10 — Mission slice collision rejected at validation
# --------------------------------------------------------------------------- #

def test_mission_slice_collision_rejected():
    """One device claimed by two mules' expected_devices must fail validate()."""
    topo = TopologyConfig(
        cluster=ClusterConfig(cluster_id="c0"),
        mules=[
            MuleConfig(
                mule_id="m1",
                expected_devices=["d_shared"],
            ),
            MuleConfig(
                mule_id="m2",
                expected_devices=["d_shared"],  # collision
            ),
        ],
        devices=[
            DeviceConfig(device_id="d_shared"),
        ],
    )
    with pytest.raises(TopologyValidationError, match="multiple mules"):
        topo.validate()


def test_validate_rejects_mule_pointing_at_unknown_device():
    """A mule's expected_devices must reference a real DeviceConfig."""
    topo = TopologyConfig(
        cluster=ClusterConfig(cluster_id="c0"),
        mules=[
            MuleConfig(mule_id="m1", expected_devices=["d_ghost"]),
        ],
        devices=[
            DeviceConfig(device_id="d_real"),
        ],
    )
    with pytest.raises(TopologyValidationError, match="unknown device"):
        topo.validate()


# --------------------------------------------------------------------------- #
# Orchestrator health check — all_alive() reflects subprocess crashes
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_all_alive_returns_false_after_subprocess_crash():
    """Belt-and-suspenders: the production health check must detect a kill."""
    topo = TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="cluster-health",
            dock_host="127.0.0.1",
            dock_port=0,
            synth_batch_size=2,
            min_participation=1,
        ),
        mules=[
            MuleConfig(
                mule_id="mule-health",
                rf_host="127.0.0.1",
                rf_port=0,
                rf_range_m=60.0,
                session_ttl_s=3.0,
            ),
        ],
        devices=[
            DeviceConfig(device_id="dev-health", position=(0.0, 0.0, 0.0)),
        ],
    )
    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=15.0)
        time.sleep(0.5)
        assert orch.all_alive()

        orch.device_handles["dev-health"].proc.kill()
        try:
            orch.device_handles["dev-health"].proc.wait(timeout=3.0)
        except Exception:
            pytest.fail("device did not die after SIGKILL")

        # Once Popen.poll() registers the exit, all_alive() must flip.
        assert orch.all_alive() is False
    finally:
        orch.shutdown_all(timeout=5.0)
