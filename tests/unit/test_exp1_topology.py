"""EX-1.1 topology unit tests — three arg-parsing modes + JSON round-trip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.exp1.topology import (
    ClientSlot,
    Exp1Topology,
    add_topology_arguments,
    build_topology_from_args,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_topology_arguments(p)
    return p


# --------------------------------------------------------------------------- #
# Explicit JSON
# --------------------------------------------------------------------------- #

def test_explicit_json_round_trip(tmp_path):
    topo = Exp1Topology(
        bind_host="0.0.0.0", bind_port=9000,
        clients=[
            ClientSlot(client_id="d1", data_partition=0, host="192.168.1.11"),
            ClientSlot(client_id="d2", data_partition=1, host="192.168.1.12"),
        ],
    )
    path = tmp_path / "topo.json"
    path.write_text(topo.to_json(), encoding="utf-8")
    restored = Exp1Topology.from_file(path)
    restored.validate()

    assert restored.bind_host == topo.bind_host
    assert restored.bind_port == topo.bind_port
    assert [s.client_id for s in restored.clients] == ["d1", "d2"]
    assert [s.data_partition for s in restored.clients] == [0, 1]
    assert [s.host for s in restored.clients] == ["192.168.1.11", "192.168.1.12"]


def test_explicit_json_via_args(tmp_path):
    topo_path = tmp_path / "topo.json"
    topo_path.write_text(
        json.dumps({
            "server": {"host": "0.0.0.0", "port": 9000},
            "clients": [
                {"client_id": "d1", "host": "10.0.0.1", "data_partition": 0},
                {"client_id": "d2", "host": "10.0.0.2", "data_partition": 1},
                {"client_id": "d3", "host": "10.0.0.3", "data_partition": 2},
                {"client_id": "d4", "host": "10.0.0.4", "data_partition": 3},
            ],
        }),
        encoding="utf-8",
    )
    args = _parser().parse_args(["--topology", str(topo_path)])
    topo = build_topology_from_args(args)
    assert topo.n_clients == 4
    assert topo.expected_partitions() == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Explicit CLI
# --------------------------------------------------------------------------- #

def test_explicit_cli_clients_with_partitions():
    args = _parser().parse_args([
        "--client", "d1@192.168.1.11:partition=0",
        "--client", "d2@192.168.1.12:partition=1",
        "--client", "d3@192.168.1.13:partition=2",
        "--client", "d4@192.168.1.14:partition=3",
    ])
    topo = build_topology_from_args(args)
    assert topo.n_clients == 4
    assert topo.expected_client_ids() == ["d1", "d2", "d3", "d4"]
    assert topo.expected_partitions() == [0, 1, 2, 3]
    assert [s.host for s in topo.clients] == [
        "192.168.1.11", "192.168.1.12", "192.168.1.13", "192.168.1.14",
    ]


def test_explicit_cli_clients_auto_partitioned_in_flag_order():
    args = _parser().parse_args([
        "--client", "d1@h1",
        "--client", "d2@h2",
    ])
    topo = build_topology_from_args(args)
    assert topo.expected_partitions() == [0, 1]


def test_explicit_cli_malformed_value_rejected():
    with pytest.raises(argparse.ArgumentTypeError, match="doesn't match"):
        build_topology_from_args(
            _parser().parse_args(["--client", "this-is-not-the-right-shape"])
        )


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def test_discovery_mode_creates_placeholder_slots():
    args = _parser().parse_args(["--discover", "--n-clients", "4"])
    topo = build_topology_from_args(args)
    assert topo.discover is True
    assert topo.n_clients == 4
    assert topo.expected_partitions() == [0, 1, 2, 3]
    assert all(s.client_id == "" and s.host is None for s in topo.clients)


def test_discovery_requires_n_clients():
    with pytest.raises(argparse.ArgumentTypeError, match="--n-clients"):
        build_topology_from_args(_parser().parse_args(["--discover"]))


def test_discovery_rejects_zero_n_clients():
    with pytest.raises(argparse.ArgumentTypeError):
        build_topology_from_args(
            _parser().parse_args(["--discover", "--n-clients", "0"])
        )


# --------------------------------------------------------------------------- #
# Mutual exclusivity + defaults
# --------------------------------------------------------------------------- #

def test_no_mode_specified_raises():
    """Default behavior must be 'pick one mode' — no implicit defaults."""
    with pytest.raises(argparse.ArgumentTypeError, match="must specify exactly one"):
        build_topology_from_args(_parser().parse_args([]))


def test_combining_modes_raises(tmp_path):
    topo_path = tmp_path / "x.json"
    topo_path.write_text("{}", encoding="utf-8")
    with pytest.raises(argparse.ArgumentTypeError, match="mutually exclusive"):
        build_topology_from_args(_parser().parse_args([
            "--topology", str(topo_path),
            "--client", "d1@h1",
        ]))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validate_rejects_partition_gap():
    """0,1,3 (skipping 2) must fail."""
    topo = Exp1Topology(clients=[
        ClientSlot(client_id="d1", data_partition=0),
        ClientSlot(client_id="d2", data_partition=1),
        ClientSlot(client_id="d3", data_partition=3),
    ])
    with pytest.raises(ValueError, match="must cover"):
        topo.validate()


def test_validate_rejects_duplicate_client_ids():
    topo = Exp1Topology(clients=[
        ClientSlot(client_id="d1", data_partition=0),
        ClientSlot(client_id="d1", data_partition=1),
    ])
    with pytest.raises(ValueError, match="duplicate client_id"):
        topo.validate()


def test_validate_rejects_explicit_mode_with_no_clients():
    topo = Exp1Topology(clients=[])
    with pytest.raises(ValueError, match="explicit-mode topology"):
        topo.validate()


def test_strict_ip_flag_carries_through(tmp_path):
    topo_path = tmp_path / "topo.json"
    topo_path.write_text(
        json.dumps({
            "server": {"host": "0.0.0.0", "port": 9000},
            "clients": [
                {"client_id": "d1", "host": "10.0.0.1", "data_partition": 0},
            ],
        }),
        encoding="utf-8",
    )
    args = _parser().parse_args([
        "--topology", str(topo_path),
        "--strict-ip",
    ])
    topo = build_topology_from_args(args)
    assert topo.strict_ip is True
