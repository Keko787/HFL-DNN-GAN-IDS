"""Wire protocol for the Experiment-1 server/client pair.

Two frame types share one length-prefixed envelope:

  +---------+-------------+-------------------+
  | type 1B | length 4B BE| payload           |
  +---------+-------------+-------------------+

* ``type=0x01 CONTROL``: payload is UTF-8 JSON. Carries protocol
  messages (REGISTER, TRIAL_BEGIN, ROUND_BEGIN, ACK, SHUTDOWN, ...).
* ``type=0x02 BULK``: payload is raw bytes. The "bytes on wire" the
  experiment actually measures.

We don't reuse ``hermes.transport.wire`` here — that module pickles
its body, which would force every BULK frame through pickle's encoder
and warp the timing. The Experiment-1 protocol is small enough to
keep self-contained.

Length cap is 256 MiB to match the rest of the system; any single
``|D|pd`` ≤ 1 GiB needs to be chunked into multiple BULK frames at
the application layer (the server's centralized-arm reader already
loops over ``recv_bulk`` until it has the expected total).
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional


MAX_FRAME_BYTES: int = 256 * 1024 * 1024  # 256 MiB
_HEADER = struct.Struct(">BI")  # 1B type + 4B big-endian length
_HEADER_SIZE = _HEADER.size  # 5


class FrameType(IntEnum):
    CONTROL = 0x01
    BULK = 0x02


class WireError(IOError):
    """Raised on any framing-level fault (bad type, bad length, peer close)."""


# --------------------------------------------------------------------------- #
# Low-level send/recv
# --------------------------------------------------------------------------- #

def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise WireError."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise WireError(
                f"recv_exactly: peer closed mid-frame after "
                f"{n - remaining}/{n} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, frame_type: int, body: bytes) -> None:
    if len(body) > MAX_FRAME_BYTES:
        raise WireError(
            f"frame body too large: {len(body)} > {MAX_FRAME_BYTES}"
        )
    header = _HEADER.pack(frame_type, len(body))
    sock.sendall(header + body)


def _recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    """Block until one full frame arrives. Returns ``(type, body)``."""
    header = _recv_exactly(sock, _HEADER_SIZE)
    frame_type, length = _HEADER.unpack(header)
    if frame_type not in (FrameType.CONTROL, FrameType.BULK):
        raise WireError(f"unknown frame type 0x{frame_type:02x}")
    if length > MAX_FRAME_BYTES:
        raise WireError(
            f"refusing pathological length {length} > {MAX_FRAME_BYTES}"
        )
    body = _recv_exactly(sock, length)
    return int(frame_type), body


# --------------------------------------------------------------------------- #
# Control frames (JSON)
# --------------------------------------------------------------------------- #

def send_control(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Encode + send one control message as a UTF-8 JSON CONTROL frame."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    _send_frame(sock, FrameType.CONTROL, body)


def recv_control(sock: socket.socket) -> Dict[str, Any]:
    """Receive one CONTROL frame; return the decoded dict.

    Raises :class:`WireError` if the next frame is BULK (caller protocol
    bug) or if JSON decoding fails.
    """
    frame_type, body = _recv_frame(sock)
    if frame_type != FrameType.CONTROL:
        raise WireError(
            f"expected CONTROL frame, got type 0x{frame_type:02x}"
        )
    try:
        msg = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise WireError(f"control frame JSON decode failed: {e}") from e
    if not isinstance(msg, dict):
        raise WireError(
            f"control frame must decode to a dict, got {type(msg).__name__}"
        )
    return msg


# --------------------------------------------------------------------------- #
# Bulk frames (raw bytes)
# --------------------------------------------------------------------------- #

def send_bulk(sock: socket.socket, body: bytes) -> None:
    """Send raw bytes as one BULK frame."""
    _send_frame(sock, FrameType.BULK, body)


def recv_bulk(sock: socket.socket) -> bytes:
    """Receive one BULK frame; return the raw bytes."""
    frame_type, body = _recv_frame(sock)
    if frame_type != FrameType.BULK:
        raise WireError(
            f"expected BULK frame, got type 0x{frame_type:02x}"
        )
    return body


# --------------------------------------------------------------------------- #
# Protocol message constructors — keeps the wire shape in one place
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProtocolError(Exception):
    """Application-layer protocol violation (distinct from WireError)."""

    message: str

    def __str__(self) -> str:
        return self.message


# Message names — one source of truth for the string keys.
MSG_REGISTER = "REGISTER"
MSG_ACCEPTED = "ACCEPTED"
MSG_REJECTED = "REJECTED"
MSG_TRIAL_BEGIN = "TRIAL_BEGIN"
MSG_ROUND_BEGIN = "ROUND_BEGIN"
MSG_ACK = "ACK"
MSG_TRIAL_END = "TRIAL_END"
MSG_SHUTDOWN = "SHUTDOWN"


def make_register(client_id: str, data_partition: int) -> Dict[str, Any]:
    return {
        "type": MSG_REGISTER,
        "client_id": client_id,
        "data_partition": int(data_partition),
    }


def make_accepted(client_id: str) -> Dict[str, Any]:
    return {"type": MSG_ACCEPTED, "client_id": client_id}


def make_rejected(reason: str) -> Dict[str, Any]:
    return {"type": MSG_REJECTED, "reason": str(reason)}


def make_trial_begin(
    *,
    arm: str,
    cell_id: str,
    trial_index: int,
    seed: int,
    arm_params: Dict[str, Any],
) -> Dict[str, Any]:
    """One TRIAL_BEGIN message.

    ``arm_params`` is arm-specific:

    * Centralized: ``{"Dpd_bytes": int}`` — total bytes the client
      uploads in one shot.
    * FL: ``{"theta_bytes": int, "n_rounds": int}`` — bytes per uplink
      and per downlink within each round; total Bpw = ``2 * theta_bytes
      * n_rounds`` per worker.
    """
    return {
        "type": MSG_TRIAL_BEGIN,
        "arm": arm,
        "cell_id": cell_id,
        "trial_index": int(trial_index),
        "seed": int(seed),
        "arm_params": dict(arm_params),
    }


def make_round_begin(round_index: int) -> Dict[str, Any]:
    return {"type": MSG_ROUND_BEGIN, "round_index": int(round_index)}


def make_ack(round_index: Optional[int] = None) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"type": MSG_ACK}
    if round_index is not None:
        msg["round_index"] = int(round_index)
    return msg


def make_trial_end(*, status: str = "ok") -> Dict[str, Any]:
    return {"type": MSG_TRIAL_END, "status": status}


def make_shutdown() -> Dict[str, Any]:
    return {"type": MSG_SHUTDOWN}
