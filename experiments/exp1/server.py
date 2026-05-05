"""Experiment-1 server — runs the trial grid as the single clock authority.

Lifecycle:

1. Parse args; build the :class:`Exp1Topology` (explicit JSON, explicit
   CLI, or discovery).
2. Bind + listen on ``(bind_host, bind_port)``.
3. Accept incoming connections; for each, read one ``REGISTER`` control
   frame and either ACCEPT (fills a slot) or REJECT.
4. When all slots are filled, walk the trial grid via the EX-0 harness.
   For each ``(cell, arm, trial_index)`` from the grid:
     a. Broadcast ``TRIAL_BEGIN`` to all clients.
     b. Drain the per-client uplink streams in parallel via
        ``ThreadPoolExecutor``. Server's ``time.perf_counter()`` brackets
        the trial start (just before broadcast) and trial end (when
        every client's stream has fully arrived).
     c. For FL arm: per-round uplink + downlink; for Centralized: one
        bulk uplink per client.
     d. Compute per-row metrics; ``CSVTrialLog.append`` writes the row.
5. Broadcast ``SHUTDOWN`` and exit.

One-sided timing: the server's clock is the sole source of truth.
Clients don't report their own durations.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from experiments.runner import Cell, TrialGrid, TrialRunner

from . import protocol as proto
from .topology import (
    ClientSlot,
    Exp1Topology,
    add_topology_arguments,
    build_topology_from_args,
)

log = logging.getLogger("experiments.exp1.server")


# --------------------------------------------------------------------------- #
# Connection bookkeeping
# --------------------------------------------------------------------------- #

@dataclass
class ClientConnection:
    """One registered client + its socket. Owned by the server main thread."""

    slot: ClientSlot
    sock: socket.socket
    peer_host: str

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

class _RegistrationDispatcher:
    """Validates incoming REGISTER frames against the topology.

    Fills :attr:`Exp1Topology.clients` slots in place when in discovery
    mode; matches connecting clients to their pre-declared slots in
    explicit modes.
    """

    def __init__(self, topo: Exp1Topology) -> None:
        self._topo = topo
        # Track which slots are claimed (True) so we can reject duplicates.
        self._claimed = [False] * topo.n_clients

    @property
    def topo(self) -> Exp1Topology:
        return self._topo

    @property
    def claimed_count(self) -> int:
        return sum(1 for c in self._claimed if c)

    def try_claim(
        self,
        client_id: str,
        data_partition: int,
        peer_host: str,
    ) -> tuple[Optional[int], Optional[str]]:
        """Return ``(slot_index, None)`` on success or ``(None, reason)``."""
        for i, slot in enumerate(self._topo.clients):
            if self._claimed[i]:
                continue
            if self._topo.discover:
                if slot.data_partition == data_partition:
                    self._topo.clients[i] = ClientSlot(
                        client_id=client_id,
                        data_partition=data_partition,
                        host=peer_host,
                    )
                    self._claimed[i] = True
                    return i, None
            else:
                if slot.client_id != client_id:
                    continue
                if slot.data_partition != data_partition:
                    return None, (
                        f"client_id {client_id!r} expected partition "
                        f"{slot.data_partition}, got {data_partition}"
                    )
                if slot.host is not None and slot.host != peer_host:
                    msg = (
                        f"client_id {client_id!r} expected host "
                        f"{slot.host!r}, connected from {peer_host!r}"
                    )
                    if self._topo.strict_ip:
                        return None, msg
                    log.warning(msg + " (warn-only; pass --strict-ip to reject)")
                self._claimed[i] = True
                return i, None
        if self._topo.discover:
            return None, (
                f"discovery mode: no free slot for partition "
                f"{data_partition} (already claimed or out of range)"
            )
        return None, f"client_id {client_id!r} is not in the topology"

    @property
    def all_claimed(self) -> bool:
        return all(self._claimed)


def accept_clients(
    listener: socket.socket,
    dispatcher: _RegistrationDispatcher,
    timeout_s: float,
) -> List[ClientConnection]:
    """Block until every slot is claimed or ``timeout_s`` elapses.

    Returns the list of connected + registered clients in slot order.
    """
    deadline = time.perf_counter() + timeout_s
    listener.settimeout(min(0.5, timeout_s))

    n_slots = len(dispatcher.topo.clients)
    conns: List[Optional[ClientConnection]] = [None] * n_slots

    while not dispatcher.all_claimed:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            unclaimed_ids = []
            for i, claimed in enumerate(dispatcher._claimed):
                if not claimed:
                    s = dispatcher.topo.clients[i]
                    unclaimed_ids.append(
                        s.client_id if s.client_id else f"slot[{i}]"
                    )
            raise TimeoutError(
                f"registration timed out after {timeout_s}s; "
                f"missing slots: {unclaimed_ids!r}"
            )

        try:
            sock, addr = listener.accept()
        except socket.timeout:
            continue
        peer_host = addr[0]

        try:
            sock.settimeout(5.0)
            msg = proto.recv_control(sock)
        except (proto.WireError, OSError) as e:
            log.warning("rejected unregistered connection from %s: %s", peer_host, e)
            sock.close()
            continue

        if msg.get("type") != proto.MSG_REGISTER:
            log.warning(
                "rejected non-REGISTER first frame from %s: %r", peer_host, msg
            )
            try:
                proto.send_control(
                    sock, proto.make_rejected("first frame must be REGISTER"),
                )
            except OSError:
                pass
            sock.close()
            continue

        cid = str(msg.get("client_id", ""))
        partition = int(msg.get("data_partition", -1))

        slot_idx, reason = dispatcher.try_claim(cid, partition, peer_host)
        if slot_idx is None:
            log.warning(
                "rejected REGISTER from %s (cid=%s partition=%d): %s",
                peer_host, cid, partition, reason,
            )
            try:
                proto.send_control(sock, proto.make_rejected(reason or "rejected"))
            except OSError:
                pass
            sock.close()
            continue

        try:
            proto.send_control(sock, proto.make_accepted(cid))
        except OSError as e:
            log.warning("ACCEPTED send failed for %s: %s", cid, e)
            sock.close()
            continue

        # Long-lived data socket — clear the registration timeout.
        sock.settimeout(None)
        conns[slot_idx] = ClientConnection(
            slot=dispatcher.topo.clients[slot_idx],
            sock=sock,
            peer_host=peer_host,
        )
        log.info(
            "REGISTERED %s (partition=%d) from %s [%d/%d]",
            cid, partition, peer_host,
            dispatcher.claimed_count, n_slots,
        )

    return [c for c in conns if c is not None]


# --------------------------------------------------------------------------- #
# Trial driver
# --------------------------------------------------------------------------- #

class Exp1ServerDriver:
    """Runs one trial against the connected clients; one-sided timing."""

    def __init__(
        self,
        conns: List[ClientConnection],
        *,
        downlink_filler_size: int = 64 * 1024,
    ) -> None:
        self._conns = conns
        # One reusable filler buffer for downlink BULK frames. The actual
        # bytes don't matter — only the count does, so we reuse the same
        # bytes object every time and let TCP do the hard work.
        self._downlink_filler = bytes(downlink_filler_size)

    def run_trial(self, cell: Cell) -> Mapping[str, Any]:
        if cell.arm == "Centralized":
            return self._run_centralized(cell, cell.params)
        if cell.arm == "FL":
            return self._run_fl(cell, cell.params)
        raise ValueError(f"unknown arm {cell.arm!r}; expected FL or Centralized")

    # ---------------------------- Centralized -------------------------- #

    def _run_centralized(
        self, cell: Cell, params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        Dpd_bytes = int(params["Dpd_bytes"])
        alpha = float(params["alpha"])

        begin_msg = proto.make_trial_begin(
            arm="Centralized",
            cell_id=cell.cell_id,
            trial_index=cell.trial_index,
            seed=cell.seed,
            arm_params={"Dpd_bytes": Dpd_bytes, "alpha": alpha},
        )
        t0 = time.perf_counter()
        for conn in self._conns:
            proto.send_control(conn.sock, begin_msg)

        # Drain all client uplinks in parallel.
        per_client_finish: List[float] = [0.0] * len(self._conns)
        with ThreadPoolExecutor(max_workers=len(self._conns)) as pool:
            futs = {
                pool.submit(self._recv_full_payload, c.sock, Dpd_bytes): i
                for i, c in enumerate(self._conns)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                fut.result()  # may raise
                per_client_finish[i] = time.perf_counter()

        Tproc = max(per_client_finish) - t0
        Bpw = Dpd_bytes
        eta = 1.0  # Centralized has minimal protocol overhead
        D = alpha * self._centralized_baseline_s(Dpd_bytes)
        Pcomplete = 1 if Tproc <= D else 0

        for conn in self._conns:
            try:
                proto.send_control(conn.sock, proto.make_trial_end())
            except OSError:
                pass

        return {
            "Tproc_s": Tproc,
            "Ttx_s": Tproc,
            "Bpw_bytes": Bpw,
            "eta": eta,
            "deadline_s": D,
            "Pcomplete": Pcomplete,
            "n_rounds": 1,
            "n_clients": len(self._conns),
        }

    # ------------------------------ FL --------------------------------- #

    def _run_fl(
        self, cell: Cell, params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        theta_bytes = int(params["theta_bytes"])
        # Grid var is named ``R`` to match the paper; some callers may
        # also use ``n_rounds``. Accept either.
        R = int(params.get("R", params.get("n_rounds", 0)))
        if R <= 0:
            raise KeyError("FL arm requires R (or n_rounds) > 0 in cell.params")
        alpha = float(params["alpha"])
        Dpd_bytes = int(params.get("Dpd_bytes", 0))

        begin_msg = proto.make_trial_begin(
            arm="FL",
            cell_id=cell.cell_id,
            trial_index=cell.trial_index,
            seed=cell.seed,
            arm_params={
                "theta_bytes": theta_bytes,
                "n_rounds": R,
                "alpha": alpha,
            },
        )
        t0 = time.perf_counter()
        for conn in self._conns:
            proto.send_control(conn.sock, begin_msg)

        round_durations: List[float] = []
        for r in range(R):
            t_round_start = time.perf_counter()

            for conn in self._conns:
                proto.send_control(conn.sock, proto.make_round_begin(r))

            # Phase A: drain per-client uplinks in parallel.
            with ThreadPoolExecutor(max_workers=len(self._conns)) as pool:
                futs = [
                    pool.submit(self._recv_full_payload, c.sock, theta_bytes)
                    for c in self._conns
                ]
                for fut in as_completed(futs):
                    fut.result()

            # Phase B: send downlinks + collect ACKs in parallel.
            downlink = self._make_downlink_bytes(theta_bytes)
            with ThreadPoolExecutor(max_workers=len(self._conns)) as pool:
                futs = [
                    pool.submit(self._send_downlink_then_ack, c.sock, downlink, r)
                    for c in self._conns
                ]
                for fut in as_completed(futs):
                    fut.result()

            round_durations.append(time.perf_counter() - t_round_start)

        Tproc = time.perf_counter() - t0
        Bpw = R * 2 * theta_bytes
        eta = (R * theta_bytes) / Bpw if Bpw > 0 else 0.0
        D = (
            alpha * self._centralized_baseline_s(Dpd_bytes)
            if Dpd_bytes > 0 else float("inf")
        )
        Pcomplete = 1 if Tproc <= D else 0

        for conn in self._conns:
            try:
                proto.send_control(conn.sock, proto.make_trial_end())
            except OSError:
                pass

        return {
            "Tproc_s": Tproc,
            "Ttx_s": sum(round_durations),
            "Bpw_bytes": Bpw,
            "eta": eta,
            "deadline_s": D,
            "Pcomplete": Pcomplete,
            "n_rounds": R,
            "n_clients": len(self._conns),
        }

    # ------------------------- helpers ---------------------------------- #

    @staticmethod
    def _recv_full_payload(sock: socket.socket, expected: int) -> int:
        body = proto.recv_bulk(sock)
        if len(body) != expected:
            raise proto.WireError(
                f"BULK length mismatch: got {len(body)}, expected {expected}"
            )
        return len(body)

    @staticmethod
    def _send_downlink_then_ack(
        sock: socket.socket, payload: bytes, round_index: int,
    ) -> None:
        proto.send_bulk(sock, payload)
        ack = proto.recv_control(sock)
        if ack.get("type") != proto.MSG_ACK:
            raise proto.WireError(
                f"expected ACK after downlink, got {ack!r}"
            )

    def _make_downlink_bytes(self, n: int) -> bytes:
        if n <= len(self._downlink_filler):
            return self._downlink_filler[:n]
        return bytes(n)

    @staticmethod
    def _centralized_baseline_s(Dpd_bytes: int) -> float:
        """Notional centralized transmission time at the paper's
        ``B_nominal = 10 Mbps`` rate. Used only to compute the
        alpha-scaled deadline for ``Pcomplete`` bookkeeping; the actual
        wire time is whatever the real link delivers."""
        if Dpd_bytes <= 0:
            return 0.0
        B_nominal_bps = 10_000_000
        return Dpd_bytes * 8 / B_nominal_bps


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #

def _parse_size(s: str) -> int:
    """Tiny size-string parser: '10MB' → 10 * 1024 * 1024."""
    s = s.strip().upper()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suf in ("GB", "MB", "KB", "B"):
        if s.endswith(suf):
            return int(float(s[: -len(suf)]) * units[suf])
    return int(s)


def _parse_filter(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    out: Dict[str, Any] = {}
    for piece in s.split(","):
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        k, v = k.strip(), v.strip()
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def _build_grid(
    *,
    n_trials: int,
    Dpd_choices: Sequence[str],
    alpha_choices: Sequence[float],
    R_choices: Sequence[int],
    base_seed: int,
    filter_params: Dict[str, Any],
) -> TrialGrid:
    return TrialGrid(
        independent_vars={
            "Dpd": list(Dpd_choices),
            "alpha": list(alpha_choices),
            "R": list(R_choices),
        },
        arms=["FL", "Centralized"],
        n_trials=n_trials,
        base_seed=base_seed,
        filter_params=filter_params,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp1.server")
    add_topology_arguments(parser)
    parser.add_argument("--output", required=True, type=Path,
                        help="CSV path for trial results.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--Dpd", nargs="+", default=["10MB", "100MB", "1GB"])
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--R", nargs="+", type=int, default=[5, 20, 50])
    parser.add_argument(
        "--theta-bytes", type=int, default=200_000,
        help="Per-round model size for the FL arm (bytes per uplink/downlink).",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Restrict the grid to cells matching k=v[,k=v]. "
             "Example: --filter 'Dpd=10MB,R=5'.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Wipe the output CSV and start over.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    topo = build_topology_from_args(args)
    log.info(
        "topology: bind=%s:%d clients=%d mode=%s",
        topo.bind_host, topo.bind_port, topo.n_clients,
        "discover" if topo.discover else "explicit",
    )

    if args.no_resume and args.output.exists():
        args.output.unlink()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((topo.bind_host, topo.bind_port))
    listener.listen(max(8, topo.n_clients))
    log.info("listening on %s:%d", topo.bind_host, topo.bind_port)

    conns: List[ClientConnection] = []
    try:
        dispatcher = _RegistrationDispatcher(topo)
        conns = accept_clients(
            listener, dispatcher, timeout_s=topo.registration_timeout_s,
        )
        log.info("all %d clients registered; starting trial grid", len(conns))

        Dpd_bytes_map = {label: _parse_size(label) for label in args.Dpd}
        grid = _build_grid(
            n_trials=args.n_trials,
            Dpd_choices=args.Dpd,
            alpha_choices=args.alpha,
            R_choices=args.R,
            base_seed=args.base_seed,
            filter_params=_parse_filter(args.filter),
        )
        log.info("grid: %d total trials", grid.total())

        metric_columns = [
            "Tproc_s", "Ttx_s", "Bpw_bytes", "eta",
            "deadline_s", "Pcomplete", "n_rounds", "n_clients",
        ]
        runner = TrialRunner(grid, args.output, metric_columns=metric_columns)
        driver = Exp1ServerDriver(conns)

        def driver_with_byte_resolution(cell: Cell) -> Mapping[str, Any]:
            params = dict(cell.params)
            params["Dpd_bytes"] = Dpd_bytes_map[params["Dpd"]]
            params["theta_bytes"] = args.theta_bytes
            resolved = Cell(
                cell_id=cell.cell_id, arm=cell.arm,
                trial_index=cell.trial_index, seed=cell.seed,
                params=params,
            )
            return driver.run_trial(resolved)

        runner.run(driver_with_byte_resolution)

    finally:
        for c in conns:
            try:
                proto.send_control(c.sock, proto.make_shutdown())
            except OSError:
                pass
            c.close()
        try:
            listener.close()
        except OSError:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
