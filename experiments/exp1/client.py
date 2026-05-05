"""Experiment-1 client — connects to the server, ships bytes when told.

The client is intentionally dumb. It loads its data partition, connects
to the server, sends ``REGISTER``, waits for ``ACCEPTED`` / ``REJECTED``,
then loops on incoming control frames:

* ``TRIAL_BEGIN(arm=Centralized, Dpd_bytes=N)`` → upload N bytes as
  one BULK frame, then wait for the next message.
* ``TRIAL_BEGIN(arm=FL, theta_bytes=N, n_rounds=R)`` → for r in 0..R:
    - wait for ``ROUND_BEGIN(round_index=r)``
    - upload N bytes (uplink)
    - receive N bytes (downlink)
    - send ``ACK(round_index=r)``
* ``TRIAL_END`` → loop back to wait for the next ``TRIAL_BEGIN``.
* ``SHUTDOWN`` → close the socket and exit.

All timing is measured by the server; the client's job is to ship
bytes as fast as the link allows. The bytes themselves are a filler
buffer matching the partition's size — we're measuring transport, not
training.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from . import protocol as proto

log = logging.getLogger("experiments.exp1.client")


class ExpClient:
    """One Experiment-1 client; one socket; one event loop."""

    def __init__(
        self,
        client_id: str,
        data_partition: int,
        server_host: str,
        server_port: int,
        *,
        connect_timeout_s: float = 30.0,
    ) -> None:
        self.client_id = client_id
        self.data_partition = data_partition
        self.server_host = server_host
        self.server_port = server_port
        self._connect_timeout_s = connect_timeout_s
        self._sock: Optional[socket.socket] = None
        # Single reusable filler buffer for uplink BULK payloads. Real
        # data shaping is the operator's job (tc/netem); we only care
        # that the byte count matches the protocol contract.
        self._uplink_filler = bytes(64 * 1024)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def connect_and_register(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._connect_timeout_s)
        sock.connect((self.server_host, self.server_port))
        sock.settimeout(None)

        proto.send_control(
            sock,
            proto.make_register(self.client_id, self.data_partition),
        )
        sock.settimeout(self._connect_timeout_s)
        reply = proto.recv_control(sock)
        sock.settimeout(None)

        if reply.get("type") == proto.MSG_REJECTED:
            raise RuntimeError(
                f"server rejected REGISTER: {reply.get('reason', '?')}"
            )
        if reply.get("type") != proto.MSG_ACCEPTED:
            raise RuntimeError(
                f"unexpected reply to REGISTER: {reply!r}"
            )
        log.info(
            "registered as %s (partition=%d) at %s:%d",
            self.client_id, self.data_partition,
            self.server_host, self.server_port,
        )
        self._sock = sock

    def run(self) -> None:
        """Event loop. Returns when the server sends SHUTDOWN."""
        if self._sock is None:
            raise RuntimeError("call connect_and_register() before run()")
        sock = self._sock
        try:
            while True:
                msg = proto.recv_control(sock)
                kind = msg.get("type")
                if kind == proto.MSG_SHUTDOWN:
                    log.info("client %s: SHUTDOWN received", self.client_id)
                    return
                if kind == proto.MSG_TRIAL_BEGIN:
                    self._handle_trial_begin(msg)
                    continue
                if kind == proto.MSG_TRIAL_END:
                    # Stale TRIAL_END (e.g., trial completed but we haven't
                    # caught up yet); ignore harmlessly.
                    continue
                log.warning(
                    "client %s: unexpected top-level message %r",
                    self.client_id, msg,
                )
        finally:
            self.close()

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None

    # ------------------------------------------------------------------ #
    # Per-arm handlers
    # ------------------------------------------------------------------ #

    def _handle_trial_begin(self, msg: Dict[str, Any]) -> None:
        arm = msg.get("arm")
        params = msg.get("arm_params", {})
        if arm == "Centralized":
            self._run_centralized(params)
        elif arm == "FL":
            self._run_fl(params)
        else:
            log.warning("client %s: unknown arm %r; skipping",
                        self.client_id, arm)

    def _run_centralized(self, params: Dict[str, Any]) -> None:
        Dpd_bytes = int(params["Dpd_bytes"])
        payload = self._make_payload(Dpd_bytes)
        proto.send_bulk(self._sock, payload)
        # Wait for TRIAL_END (may follow immediately, or after the server
        # finishes draining other clients).
        ack = proto.recv_control(self._sock)
        if ack.get("type") != proto.MSG_TRIAL_END:
            log.warning("client %s: expected TRIAL_END, got %r",
                        self.client_id, ack)

    def _run_fl(self, params: Dict[str, Any]) -> None:
        theta_bytes = int(params["theta_bytes"])
        R = int(params["n_rounds"])
        uplink = self._make_payload(theta_bytes)

        for r in range(R):
            rb = proto.recv_control(self._sock)
            if rb.get("type") != proto.MSG_ROUND_BEGIN:
                raise proto.WireError(
                    f"client {self.client_id}: expected ROUND_BEGIN, got {rb!r}"
                )
            # Phase A: uplink.
            proto.send_bulk(self._sock, uplink)
            # Phase B: receive downlink, ack.
            downlink = proto.recv_bulk(self._sock)
            if len(downlink) != theta_bytes:
                raise proto.WireError(
                    f"client {self.client_id}: downlink length mismatch "
                    f"({len(downlink)} != {theta_bytes})"
                )
            proto.send_control(self._sock, proto.make_ack(r))

        # Server sends TRIAL_END after all rounds.
        end = proto.recv_control(self._sock)
        if end.get("type") != proto.MSG_TRIAL_END:
            log.warning("client %s: expected TRIAL_END, got %r",
                        self.client_id, end)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_payload(self, n: int) -> bytes:
        """Reuse the filler buffer when possible; else allocate."""
        if n <= len(self._uplink_filler):
            return self._uplink_filler[:n]
        return bytes(n)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _parse_server_addr(s: str) -> tuple[str, int]:
    """Parse 'host:port' → (host, port)."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(
            f"--server expects 'host:port', got {s!r}"
        )
    host, port = s.rsplit(":", 1)
    return host, int(port)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp1.client")
    parser.add_argument("--client-id", required=True,
                        help="Unique identifier within the topology (e.g. d1).")
    parser.add_argument("--server", required=True, type=_parse_server_addr,
                        help="Server address as host:port.")
    parser.add_argument("--data-partition", required=True, type=int,
                        help="Which CICIOT shard this client owns (0..N-1).")
    parser.add_argument("--connect-timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    host, port = args.server
    client = ExpClient(
        client_id=args.client_id,
        data_partition=args.data_partition,
        server_host=host,
        server_port=port,
        connect_timeout_s=args.connect_timeout_s,
    )
    client.connect_and_register()
    client.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
