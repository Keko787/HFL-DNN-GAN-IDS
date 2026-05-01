"""Sprint 2 chunk M — structured observability.

Two pieces, deliberately small:

* :class:`JsonEventEmitter` — writes one JSON object per line to a per-process
  JSONL file. Every state-transition the multi-process topology cares about
  (mule docked, mission round closed, contact opened, etc.) gets one line.
  Plain text logging through :mod:`logging` continues alongside; the JSONL
  stream is for downstream tooling (Grafana / Loki / OEO ingest) that wants
  machine-parseable events.

* :class:`MetricsRegistry` — counters, gauges, timers. ``snapshot()`` returns
  a flat dict suitable for emitting as one ``metrics_snapshot`` event at
  shutdown (or on a periodic tick if a future chunk wants it).

The taxonomy is intentionally not enforced as an enum — real ops shows that
event names drift faster than enums can keep up, and Sprint 2's downstream
consumers (test assertions, e2e log scrapers) want strings, not enum values.
What IS enforced is the **envelope shape**: every event carries
``{ts, schema_version, role, id, event}`` plus its own payload fields.
"""

from __future__ import annotations

from .events import JsonEventEmitter, NullEventEmitter
from .metrics import MetricsRegistry

__all__ = [
    "JsonEventEmitter",
    "MetricsRegistry",
    "NullEventEmitter",
]
