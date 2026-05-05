"""Sprint 2 — multi-process orchestrator smoke tests.

Brings up a real subprocess topology (1 cluster + 1 mule + 1 device)
and verifies:

* All processes start in dependency order.
* Each process binds its expected port.
* The orchestrator can shut everything down cleanly.

This is the chunk-L acceptance test. The fuller end-to-end mission
(multiple devices, multiple mules, full Pass-1 + Pass-2) lives in
chunk N's test_e2e_topology.py.

These tests are *real* subprocess tests — they are slower than the
in-process integration tests (~3-5 seconds each). Marked ``slow`` so
they can be excluded from quick local runs but still gate CI.
"""

from __future__ import annotations

import time

import pytest

from hermes.processes import (
    ClusterConfig,
    DeviceConfig,
    MuleConfig,
    MultiProcessOrchestrator,
    TopologyConfig,
)


def _smallest_topology(*, n_missions=None, n_serves=None) -> TopologyConfig:
    """1 cluster, 1 mule, 1 device — smallest viable HERMES deployment.

    By default the mule and device run-forever; tests pass caps when
    they need the topology to converge to natural exit.
    """
    return TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="cluster-test",
            dock_host="127.0.0.1",
            dock_port=0,  # ephemeral
            synth_batch_size=2,
            min_participation=1,
        ),
        mules=[
            MuleConfig(
                mule_id="mule-test-1",
                rf_host="127.0.0.1",
                rf_port=0,
                # dock_port filled in by orchestrator after cluster starts
                rf_range_m=60.0,
                session_ttl_s=3.0,
                n_missions=n_missions,
            )
        ],
        devices=[
            DeviceConfig(
                device_id="dev-test-1",
                # mule_rf_port filled in by orchestrator after mule starts
                position=(0.0, 0.0, 0.0),
                n_serves=n_serves,
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_orchestrator_starts_and_stops_topology():
    """Cluster + mule + device all spawn, bind ports, then shut down clean."""
    topo = _smallest_topology()
    orch = MultiProcessOrchestrator(topo, capture_output=True)

    try:
        orch.start_cluster(timeout=10.0)
        assert orch.cluster_handle is not None
        assert orch.cluster_handle.is_alive()
        assert orch.cluster_handle.actual_port is not None
        assert orch.cluster_handle.actual_port > 0

        ports = orch.start_mules(timeout=10.0)
        assert "mule-test-1" in ports
        assert ports["mule-test-1"] > 0

        orch.start_devices(timeout=10.0)
        assert "dev-test-1" in orch.device_handles
        assert orch.device_handles["dev-test-1"].is_alive()

        # Brief settling — every process registered with its peers.
        time.sleep(0.5)
        assert orch.all_alive(), "a process exited before shutdown"
    finally:
        orch.shutdown_all(timeout=5.0)

    # After shutdown, nothing is alive.
    assert orch.cluster_handle is not None
    assert not orch.cluster_handle.is_alive()
    for h in orch.mule_handles.values():
        assert not h.is_alive()
    for h in orch.device_handles.values():
        assert not h.is_alive()


@pytest.mark.slow
def test_orchestrator_start_all_convenience():
    topo = _smallest_topology()
    orch = MultiProcessOrchestrator(topo, capture_output=True)

    try:
        orch.start_all(timeout=15.0)
        time.sleep(0.5)
        assert orch.all_alive()
    finally:
        orch.shutdown_all(timeout=5.0)


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #

def test_orchestrator_start_mules_before_cluster_raises():
    topo = _smallest_topology()
    orch = MultiProcessOrchestrator(topo)
    try:
        with pytest.raises(Exception, match="start_cluster must succeed"):
            orch.start_mules(timeout=1.0)
    finally:
        orch.shutdown_all(timeout=2.0)


def test_orchestrator_start_devices_before_mules_raises():
    topo = _smallest_topology()
    orch = MultiProcessOrchestrator(topo)
    try:
        with pytest.raises(Exception, match="start_mules must succeed"):
            orch.start_devices(timeout=1.0)
    finally:
        orch.shutdown_all(timeout=2.0)


# --------------------------------------------------------------------------- #
# Topology config JSON round-trip
# --------------------------------------------------------------------------- #

def test_topology_config_json_round_trip():
    topo = _smallest_topology()
    payload = topo.to_json()
    restored = TopologyConfig.from_json(payload)
    assert restored.cluster.cluster_id == topo.cluster.cluster_id
    assert len(restored.mules) == len(topo.mules)
    assert restored.mules[0].mule_id == topo.mules[0].mule_id
    assert len(restored.devices) == len(topo.devices)
    assert restored.devices[0].device_id == topo.devices[0].device_id


def test_device_config_position_round_trip_as_tuple():
    """Position must come back as a tuple, not a list."""
    from hermes.processes.config import (
        device_config_from_json,
        device_config_to_json,
    )
    cfg = DeviceConfig(device_id="d0", position=(1.0, 2.0, 3.0))
    restored = device_config_from_json(device_config_to_json(cfg))
    assert isinstance(restored.position, tuple)
    assert restored.position == (1.0, 2.0, 3.0)
