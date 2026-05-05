"""Sprint 2 chunk M — observability unit tests.

Covers:

* :class:`JsonEventEmitter` envelope shape and reserved-key handling.
* :class:`NullEventEmitter` no-op behavior.
* :class:`MetricsRegistry` counter/gauge/timer math + snapshot shape.
* Cross-kind name clash rejection.
"""

from __future__ import annotations

import json

import pytest

from hermes.observability import (
    JsonEventEmitter,
    MetricsRegistry,
    NullEventEmitter,
)
from hermes.observability.events import SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# JsonEventEmitter
# --------------------------------------------------------------------------- #

def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_emit_writes_one_line_per_event(tmp_path):
    emitter = JsonEventEmitter(
        tmp_path / "a.jsonl", role="cluster", node_id="c0",
        clock=lambda: 100.0,
    )
    emitter.emit("ready", port=8000)
    emitter.emit("round_closed", round=1)
    emitter.close()

    rows = _read_jsonl(tmp_path / "a.jsonl")
    assert len(rows) == 2
    assert rows[0]["event"] == "ready"
    assert rows[1]["event"] == "round_closed"


def test_emit_envelope_has_canonical_fields(tmp_path):
    emitter = JsonEventEmitter(
        tmp_path / "b.jsonl", role="mule", node_id="mule-7",
        clock=lambda: 42.5,
    )
    emitter.emit("contact_opened", devices=("d0", "d1"))
    emitter.close()

    row = _read_jsonl(tmp_path / "b.jsonl")[0]
    assert row["ts"] == 42.5
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["role"] == "mule"
    assert row["id"] == "mule-7"
    assert row["event"] == "contact_opened"
    # Tuples coerce to lists for JSON cleanliness.
    assert row["devices"] == ["d0", "d1"]


def test_emit_reserved_keys_cannot_be_overridden(tmp_path):
    """A confused caller passing ``ts=0`` must not corrupt the envelope.

    (``event`` is the positional arg of :meth:`emit` and Python's own
    signature check rejects passing it twice — no need to test that
    here. The reserved fields we explicitly defend against in the body
    are ``ts``, ``schema_version``, ``role``, and ``id``.)
    """
    emitter = JsonEventEmitter(
        tmp_path / "c.jsonl", role="cluster", node_id="c0",
        clock=lambda: 99.0,
    )
    emitter.emit(
        "ready",
        ts=0,
        role="HACKED",
        id="ALSO_HACKED",
        schema_version=999,
    )
    emitter.close()

    row = _read_jsonl(tmp_path / "c.jsonl")[0]
    assert row["ts"] == 99.0
    assert row["role"] == "cluster"
    assert row["id"] == "c0"
    assert row["event"] == "ready"
    assert row["schema_version"] == SCHEMA_VERSION


def test_emit_after_close_is_safe(tmp_path):
    emitter = JsonEventEmitter(
        tmp_path / "d.jsonl", role="device", node_id="d0",
        clock=lambda: 0.0,
    )
    emitter.emit("ready")
    emitter.close()
    # Must not raise.
    emitter.emit("ignored")
    rows = _read_jsonl(tmp_path / "d.jsonl")
    assert len(rows) == 1


def test_emitter_context_manager_closes_file(tmp_path):
    path = tmp_path / "e.jsonl"
    with JsonEventEmitter(path, role="cluster", node_id="c0", clock=lambda: 1.0) as e:
        e.emit("ready")
    rows = _read_jsonl(path)
    assert len(rows) == 1


def test_null_emitter_drops_silently():
    null = NullEventEmitter(role="device", node_id="d0")
    null.emit("anything", x=1)  # must not raise
    null.close()
    assert null.role == "device"
    assert null.node_id == "d0"
    assert null.path is None


def test_emit_handles_path_value(tmp_path):
    emitter = JsonEventEmitter(
        tmp_path / "f.jsonl", role="cluster", node_id="c0",
        clock=lambda: 1.0,
    )
    cfg_path = tmp_path / "cfg.json"
    emitter.emit("config_loaded", path=cfg_path)
    emitter.close()
    row = _read_jsonl(tmp_path / "f.jsonl")[0]
    assert row["path"] == str(cfg_path)


# --------------------------------------------------------------------------- #
# MetricsRegistry
# --------------------------------------------------------------------------- #

def test_counter_increment_default_step():
    m = MetricsRegistry()
    m.increment("frames_dropped")
    m.increment("frames_dropped")
    assert m.counter_value("frames_dropped") == 2


def test_counter_increment_custom_step():
    m = MetricsRegistry()
    m.increment("bytes_in", by=1024)
    m.increment("bytes_in", by=512)
    assert m.counter_value("bytes_in") == 1536


def test_gauge_records_latest():
    m = MetricsRegistry()
    m.set_gauge("mule_energy", 1.0)
    m.set_gauge("mule_energy", 0.7)
    assert m.gauge_value("mule_energy") == pytest.approx(0.7)


def test_timer_observes_count_sum_min_max():
    m = MetricsRegistry()
    m.observe("pass_1_duration", 1.5)
    m.observe("pass_1_duration", 0.5)
    m.observe("pass_1_duration", 2.0)
    state = m.timer_state("pass_1_duration")
    assert state.count == 3
    assert state.sum_s == pytest.approx(4.0)
    assert state.min_s == pytest.approx(0.5)
    assert state.max_s == pytest.approx(2.0)


def test_timer_clips_negative_durations():
    m = MetricsRegistry()
    m.observe("weird", -1.0)
    state = m.timer_state("weird")
    assert state.min_s == 0.0
    assert state.max_s == 0.0


def test_snapshot_shape():
    m = MetricsRegistry()
    m.increment("contacts_visited", by=3)
    m.set_gauge("queue_depth", 5.0)
    m.observe("dock_round_trip", 0.25)
    m.observe("dock_round_trip", 0.75)

    snap = m.snapshot()
    assert snap["counter.contacts_visited"] == 3
    assert snap["gauge.queue_depth"] == 5.0

    timer = snap["timer.dock_round_trip"]
    assert timer["count"] == 2
    assert timer["sum_s"] == pytest.approx(1.0)
    assert timer["min_s"] == pytest.approx(0.25)
    assert timer["max_s"] == pytest.approx(0.75)
    assert timer["mean_s"] == pytest.approx(0.5)


def test_snapshot_omits_empty_timers():
    m = MetricsRegistry()
    m.increment("x")
    snap = m.snapshot()
    assert "counter.x" in snap
    # No timers were observed, so no timer keys appear.
    assert not any(k.startswith("timer.") for k in snap)


def test_kind_clash_rejected():
    m = MetricsRegistry()
    m.increment("conflicted")
    with pytest.raises(ValueError, match="already in use"):
        m.set_gauge("conflicted", 1.0)
    with pytest.raises(ValueError, match="already in use"):
        m.observe("conflicted", 0.1)
