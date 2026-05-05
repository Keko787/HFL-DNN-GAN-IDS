"""EX-1.1 integration — server + N clients on 127.0.0.1, small grid.

Verifies the full pipeline:

* Server binds, accepts N clients, validates REGISTER per topology mode.
* Each client connects, registers, processes TRIAL_BEGIN messages.
* For Centralized arm: client uploads Dpd_bytes, server times to last byte.
* For FL arm: client uploads + receives + ACKs each round.
* CSV gets a row per (cell, arm, trial_index) with the right shape.
* SHUTDOWN cleans up every socket.

This is the chunk-EX-1.1 acceptance gate: the whole architecture works
end-to-end through real TCP between processes (here, threads — the
same protocol works across process boundaries because it's all sockets).
"""

from __future__ import annotations

import csv
import socket
import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from experiments.exp1.client import ExpClient
from experiments.exp1.server import (
    Exp1ServerDriver,
    _RegistrationDispatcher,
    accept_clients,
)
from experiments.exp1.topology import ClientSlot, Exp1Topology
from experiments.runner import Cell, TrialGrid, TrialRunner


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_topology(n_clients: int, port: int) -> Exp1Topology:
    return Exp1Topology(
        bind_host="127.0.0.1",
        bind_port=port,
        clients=[
            ClientSlot(
                client_id=f"d{i+1}",
                data_partition=i,
                host="127.0.0.1",
            )
            for i in range(n_clients)
        ],
    )


def _start_clients(
    n_clients: int, port: int, server_ready: threading.Event,
) -> tuple[List[threading.Thread], List[BaseException]]:
    """Spin one thread per client; they wait on ``server_ready`` then connect.

    Each thread runs the full client event loop (including the run()
    handler) until SHUTDOWN. Returns ``(threads, errors_list)`` —
    callers join the threads then assert the errors list is empty.
    """
    threads: List[threading.Thread] = []
    errors: List[BaseException] = []

    def _client_main(client_id: str, partition: int):
        try:
            server_ready.wait(timeout=5.0)
            c = ExpClient(
                client_id=client_id,
                data_partition=partition,
                server_host="127.0.0.1",
                server_port=port,
                connect_timeout_s=10.0,
            )
            c.connect_and_register()
            c.run()
        except BaseException as e:  # pragma: no cover - surfaces in test
            errors.append(e)

    for i in range(n_clients):
        t = threading.Thread(
            target=_client_main, args=(f"d{i+1}", i), daemon=True,
            name=f"exp1-client-d{i+1}",
        )
        t.start()
        threads.append(t)

    return threads, errors


def _read_csv(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Smallest viable grid (1 cell × 2 arms × 2 trials = 4 trials)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_exp1_server_runs_small_grid_against_4_clients(tmp_path):
    port = _free_port()
    n_clients = 4
    topo = _make_topology(n_clients, port)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((topo.bind_host, topo.bind_port))
    listener.listen(8)

    server_ready = threading.Event()
    server_ready.set()  # listener is up

    client_threads, client_errors = _start_clients(n_clients, port, server_ready)

    # Run the server-side accept + trial-grid loop on this test thread.
    try:
        dispatcher = _RegistrationDispatcher(topo)
        conns = accept_clients(listener, dispatcher, timeout_s=5.0)
        assert len(conns) == n_clients

        # Smallest grid: 1 Dpd × 1 alpha × 1 R × 2 arms × 2 trials = 4.
        grid = TrialGrid(
            independent_vars={
                "Dpd": ["10KB"],
                "alpha": [1.0],
                "R": [2],
            },
            arms=["FL", "Centralized"],
            n_trials=2,
            base_seed=42,
        )

        Dpd_bytes_map = {"10KB": 10 * 1024}
        theta_bytes = 4096
        metric_columns = [
            "Tproc_s", "Ttx_s", "Bpw_bytes", "eta",
            "deadline_s", "Pcomplete", "n_rounds", "n_clients",
        ]
        out_csv = tmp_path / "exp1_test.csv"
        runner = TrialRunner(grid, out_csv, metric_columns=metric_columns)
        driver = Exp1ServerDriver(conns)

        def driver_with_byte_resolution(cell: Cell):
            params = dict(cell.params)
            params["Dpd_bytes"] = Dpd_bytes_map[params["Dpd"]]
            params["theta_bytes"] = theta_bytes
            resolved = Cell(
                cell_id=cell.cell_id, arm=cell.arm,
                trial_index=cell.trial_index, seed=cell.seed,
                params=params,
            )
            return driver.run_trial(resolved)

        runner.run(driver_with_byte_resolution)

        # Read + verify the CSV.
        rows = _read_csv(out_csv)
        assert len(rows) == 4, f"expected 4 trials, got {len(rows)}"
        assert all(r["status"] == "ok" for r in rows), \
            f"unexpected non-ok status: {[r['status'] for r in rows]!r}"

        # Per-arm sanity: Bpw matches the formula.
        for r in rows:
            arm = r["arm"]
            if arm == "Centralized":
                assert int(r["Bpw_bytes"]) == 10 * 1024
                assert int(r["n_rounds"]) == 1
            elif arm == "FL":
                assert int(r["Bpw_bytes"]) == 2 * 2 * theta_bytes  # 2|θ| · R
                assert int(r["n_rounds"]) == 2
            assert int(r["n_clients"]) == n_clients
            assert float(r["Tproc_s"]) > 0.0

    finally:
        # SHUTDOWN every client; close listener.
        from experiments.exp1 import protocol as proto
        for c in conns:
            try:
                proto.send_control(c.sock, proto.make_shutdown())
            except OSError:
                pass
            c.close()
        listener.close()
        for t in client_threads:
            t.join(timeout=5.0)
        # Surface any client-thread exceptions.
        assert not client_errors, f"client thread(s) raised: {client_errors!r}"


# --------------------------------------------------------------------------- #
# Registration paths
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_exp1_server_rejects_unexpected_client_id(tmp_path):
    """Explicit-mode topology rejects a client whose ID isn't in the allowlist."""
    port = _free_port()
    topo = Exp1Topology(
        bind_host="127.0.0.1", bind_port=port,
        clients=[
            ClientSlot(client_id="d1", data_partition=0, host="127.0.0.1"),
        ],
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((topo.bind_host, topo.bind_port))
    listener.listen(8)

    intruder_error: List[BaseException] = []
    legit_done = threading.Event()

    def _intruder():
        try:
            c = ExpClient(
                client_id="d_unknown", data_partition=0,
                server_host="127.0.0.1", server_port=port,
                connect_timeout_s=5.0,
            )
            c.connect_and_register()
        except BaseException as e:
            intruder_error.append(e)

    def _legit():
        try:
            c = ExpClient(
                client_id="d1", data_partition=0,
                server_host="127.0.0.1", server_port=port,
                connect_timeout_s=5.0,
            )
            c.connect_and_register()
            legit_done.set()
            c.run()
        except BaseException:
            legit_done.set()
            raise

    intruder_t = threading.Thread(target=_intruder, daemon=True)
    legit_t = threading.Thread(target=_legit, daemon=True)
    intruder_t.start()
    # Tiny delay so the intruder reaches the server first.
    time.sleep(0.1)
    legit_t.start()

    try:
        dispatcher = _RegistrationDispatcher(topo)
        conns = accept_clients(listener, dispatcher, timeout_s=5.0)
        assert len(conns) == 1
        assert conns[0].slot.client_id == "d1"

        intruder_t.join(timeout=2.0)
        assert intruder_error, "intruder should have been rejected"
        assert "rejected REGISTER" in str(intruder_error[0]) \
            or "is not in the topology" in str(intruder_error[0])

    finally:
        from experiments.exp1 import protocol as proto
        for c in conns:
            try:
                proto.send_control(c.sock, proto.make_shutdown())
            except OSError:
                pass
            c.close()
        listener.close()
        legit_t.join(timeout=2.0)


@pytest.mark.slow
def test_exp1_server_discovery_accepts_first_n_clients(tmp_path):
    """Discovery mode with n_clients=2 takes the first 2 unique partitions."""
    port = _free_port()
    topo = Exp1Topology(
        bind_host="127.0.0.1", bind_port=port,
        clients=[
            ClientSlot(client_id="", data_partition=0, host=None),
            ClientSlot(client_id="", data_partition=1, host=None),
        ],
        discover=True,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((topo.bind_host, topo.bind_port))
    listener.listen(8)

    threads: List[threading.Thread] = []
    errors: List[BaseException] = []

    def _client(cid, partition):
        try:
            c = ExpClient(
                client_id=cid, data_partition=partition,
                server_host="127.0.0.1", server_port=port,
                connect_timeout_s=5.0,
            )
            c.connect_and_register()
            c.run()
        except BaseException as e:
            errors.append(e)

    for cid, partition in [("alpha", 0), ("beta", 1)]:
        t = threading.Thread(target=_client, args=(cid, partition), daemon=True)
        t.start()
        threads.append(t)

    try:
        dispatcher = _RegistrationDispatcher(topo)
        conns = accept_clients(listener, dispatcher, timeout_s=5.0)
        assert len(conns) == 2
        connected_ids = sorted(c.slot.client_id for c in conns)
        assert connected_ids == ["alpha", "beta"]

    finally:
        from experiments.exp1 import protocol as proto
        for c in conns:
            try:
                proto.send_control(c.sock, proto.make_shutdown())
            except OSError:
                pass
            c.close()
        listener.close()
        for t in threads:
            t.join(timeout=2.0)
        assert not errors, f"client errors: {errors!r}"
