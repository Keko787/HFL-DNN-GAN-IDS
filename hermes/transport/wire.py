"""Sprint 2 — wire format helpers (framed serialization over TCP).

Both ``TCPRFLink`` and ``TCPDockLink`` ride on top of these primitives.
Single source of truth for:

* **Framing** — every message on the wire is
  ``[4-byte magic 'HRMS'][u8 version][u32 length][payload]``. Magic
  catches accidental cross-protocol misconnects (e.g. a curl request
  hitting the RF port); version lets a future receiver reject older
  senders cleanly instead of garbage-decoding them.
* **Serialization** — ``pickle.HIGHEST_PROTOCOL``. We control both ends
  (the AVN binaries are the same wheel), numpy arrays serialise natively,
  and the existing dataclasses/enums round-trip without custom encoders.
  Pickle's "untrusted-data" risk is bounded because the only senders
  are co-deployed mule / cluster / device processes; for an actual
  hostile-network deployment this would swap to msgpack + a hand-rolled
  numpy adapter (S2-M1, Sprint 3+ backlog).

S2-H1: ``recv_message`` saves and restores the socket's timeout so the
sticky behaviour of ``socket.settimeout`` doesn't leak across calls.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any, Optional


# Frame layout:
#
#   offset  size  field
#   ──────────────────────────────────────────────
#     0      4    magic = b'HRMS'  (S2-L1)
#     4      1    version          (S2-L1)
#     5      4    length (network-byte-order uint32)
#     9      N    pickled payload
#
# Big-endian throughout so wireshark / tcpdump captures are readable.
_MAGIC: bytes = b"HRMS"
_VERSION: int = 1
_HEADER_FMT = struct.Struct("!4sBI")
_HEADER_SIZE = _HEADER_FMT.size

# Hard cap to refuse pathological / malicious frames before allocating
# a multi-GB buffer. 256 MiB is far above any expected DiscPush.
MAX_FRAME_BYTES: int = 256 * 1024 * 1024


class WireError(IOError):
    """Raised on framing / serialization failure."""


def encode_message(msg: Any) -> bytes:
    """Pickle ``msg`` and prepend the framing header (magic + version + length).

    Returns the complete on-wire frame ready for ``sock.sendall``.
    """
    body = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    if len(body) > MAX_FRAME_BYTES:
        raise WireError(
            f"encode_message: payload too large ({len(body)} > {MAX_FRAME_BYTES})"
        )
    return _HEADER_FMT.pack(_MAGIC, _VERSION, len(body)) + body


def decode_message(payload: bytes) -> Any:
    """Inverse of :func:`encode_message`'s body (header already consumed).

    Caller is expected to have already validated magic + version + length;
    this fn just pickle-decodes the body.
    """
    try:
        return pickle.loads(payload)
    except Exception as e:  # pragma: no cover — pickle errors are hard to mock
        raise WireError(f"decode_message: pickle failed: {e}") from e


def send_message(sock: socket.socket, msg: Any) -> None:
    """Encode + sendall over a connected TCP socket.

    Caller owns the socket lifecycle. Raises :class:`WireError` on any
    framing failure; callers translate to the link-specific error type.
    """
    frame = encode_message(msg)
    try:
        sock.sendall(frame)
    except OSError as e:
        raise WireError(f"send_message: sendall failed: {e}") from e


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Block until exactly ``n`` bytes are read, or raise on close/timeout.

    ``socket.recv(n)`` may return fewer bytes; we loop until we have
    the full length. Returns immediately if ``n == 0``.
    """
    if n == 0:
        return b""
    chunks = bytearray()
    while len(chunks) < n:
        try:
            buf = sock.recv(n - len(chunks))
        except OSError as e:
            raise WireError(f"recv_exactly: socket error after {len(chunks)}/{n} bytes: {e}") from e
        if not buf:
            raise WireError(
                f"recv_exactly: peer closed mid-frame after {len(chunks)}/{n} bytes"
            )
        chunks.extend(buf)
    return bytes(chunks)


def recv_message(sock: socket.socket, timeout: Optional[float] = None) -> Any:
    """Read one framed message off the wire.

    Validates the magic + version + length header, then decodes the body.

    S2-H1: ``timeout`` (when set) is applied for the duration of *this*
    call only — the socket's previous timeout is restored on the way
    out, so callers don't accidentally inherit the timeout for later
    blocking operations on the same socket.

    A bad peer can still hold us up to ~2× ``timeout`` (header read +
    body read) in the worst case. For a tight bound, set the socket's
    timeout explicitly on the caller side and pass ``timeout=None`` here.
    """
    prev_timeout = sock.gettimeout() if timeout is not None else None
    if timeout is not None:
        sock.settimeout(timeout)
    try:
        header = recv_exactly(sock, _HEADER_SIZE)
        magic, version, length = _HEADER_FMT.unpack(header)
        if magic != _MAGIC:
            raise WireError(
                f"recv_message: bad frame magic {magic!r} (expected {_MAGIC!r}) "
                f"— is this a hermes peer?"
            )
        if version != _VERSION:
            raise WireError(
                f"recv_message: unsupported wire version {version} "
                f"(this build expects {_VERSION})"
            )
        if length > MAX_FRAME_BYTES:
            raise WireError(
                f"recv_message: refusing pathological frame size {length} "
                f"> {MAX_FRAME_BYTES}"
            )
        body = recv_exactly(sock, length)
        return decode_message(body)
    finally:
        if timeout is not None:
            sock.settimeout(prev_timeout)
