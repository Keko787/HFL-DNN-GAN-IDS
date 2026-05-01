"""JSONL event emitter for the multi-process topology.

Design constraints:

* **One JSONL file per process** — written under the orchestrator's run dir
  so a test or a tail tool can pick the events for a single AVN out without
  parsing interleaved streams.
* **Append-only**. No rotation here — Sprint 2 missions are short enough that
  one file per run is fine. AERPAW deployment may layer rotation later.
* **Synchronous flush** so a crashed subprocess still leaves a complete tail
  on disk for the orchestrator's post-mortem (chunk-L's stderr_tail is the
  human-readable counterpart; this is the structured one).
* **Best-effort emit**. A write that fails (disk full, file closed) logs at
  ``DEBUG`` and drops the event. Observability must never crash the process
  it's instrumenting.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, TextIO

log = logging.getLogger("hermes.observability")

# Bump this when adding/removing required envelope keys. Adding *optional*
# fields under a specific event name is NOT a schema break.
SCHEMA_VERSION: int = 1


class JsonEventEmitter:
    """Per-process JSONL event writer.

    Each call to :meth:`emit` writes one line:

    .. code-block:: json

        {"ts": 1714694400.123, "schema_version": 1, "role": "mule",
         "id": "mule-test-1", "event": "pass_1_started",
         "round": 0, "contacts": 2, "devices_total": 3}

    ``ts`` is wall-clock seconds-since-epoch as a float. ``role`` and ``id``
    identify the emitter (``cluster|mule|device|orchestrator`` and the
    config-supplied id). ``event`` is the transition name, free-form
    snake_case. Everything past those is the per-event payload.
    """

    def __init__(
        self,
        path: Path,
        *,
        role: str,
        node_id: str,
        clock=None,
    ) -> None:
        self._path = Path(path)
        self._role = role
        self._id = node_id
        self._clock = clock or time.time
        self._lock = threading.Lock()
        # Open in line-buffered append mode. ``buffering=1`` only line-buffers
        # text streams, which is exactly what we want — every JSONL line is
        # one write.
        self._fp: Optional[TextIO] = open(
            self._path, "a", buffering=1, encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def path(self) -> Path:
        return self._path

    @property
    def role(self) -> str:
        return self._role

    @property
    def node_id(self) -> str:
        return self._id

    def emit(self, event: str, **fields: Any) -> None:
        """Append one event line to the JSONL file.

        Reserved field names — ``ts``, ``schema_version``, ``role``, ``id``,
        ``event`` — silently override the caller's value to keep the
        envelope canonical. Pass any extra context as kwargs.
        """
        record = {
            "ts": float(self._clock()),
            "schema_version": SCHEMA_VERSION,
            "role": self._role,
            "id": self._id,
            "event": event,
        }
        # Caller fields go on top, then we re-stamp the reserved keys so
        # they win even if a confused caller passed e.g. ``ts=...``.
        for k, v in fields.items():
            if k in record:
                continue
            record[k] = _coerce(v)

        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            fp = self._fp
            if fp is None:
                return
            try:
                fp.write(line + "\n")
            except Exception:
                # Per the design constraint above — best-effort. A failed
                # write must not propagate into the instrumented code path.
                log.debug("event emit failed (event=%s)", event, exc_info=True)

    def close(self) -> None:
        with self._lock:
            fp = self._fp
            self._fp = None
        if fp is not None:
            try:
                fp.flush()
                fp.close()
            except Exception:
                log.debug("event emitter close failed", exc_info=True)

    # Context-manager + del so a forgetful caller doesn't leak the FD.
    def __enter__(self) -> "JsonEventEmitter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover — best-effort GC fallback
        try:
            self.close()
        except Exception:
            pass


class NullEventEmitter:
    """No-op emitter used when ``--run-dir`` isn't supplied.

    Tests that want to exercise a service-layer object without setting up a
    JSONL file pass this in. The interface matches :class:`JsonEventEmitter`
    so call sites don't branch on ``emitter is not None``.
    """

    def __init__(self, role: str = "test", node_id: str = "test") -> None:
        self._role = role
        self._id = node_id

    @property
    def path(self) -> Optional[Path]:
        return None

    @property
    def role(self) -> str:
        return self._role

    @property
    def node_id(self) -> str:
        return self._id

    def emit(self, event: str, **fields: Any) -> None:
        return

    def close(self) -> None:
        return

    def __enter__(self) -> "NullEventEmitter":
        return self

    def __exit__(self, *_a) -> None:
        return


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _coerce(v: Any) -> Any:
    """Best-effort JSON-friendly coercion of common HERMES value types.

    Tuples → lists (positions, mostly), Path → str, anything else passes
    through. We deliberately don't try to handle numpy arrays here — those
    don't belong in event lines (they go on the wire format instead). If a
    caller hands us a numpy scalar, ``json.dumps`` will raise and the emit
    drops via the exception handler — visible in DEBUG logs.
    """
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, Path):
        return str(v)
    return v
