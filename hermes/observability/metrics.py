"""Lightweight in-process metrics registry.

Three metric kinds — all the multi-process topology needs in Sprint 2:

* **Counter** (monotonic int) — events you tally: contacts visited,
  devices delivered, frames dropped.
* **Gauge** (latest float) — point-in-time values: mule pose components,
  mule energy, queue depth.
* **Timer** (sum + count + min + max) — durations: pass_1_duration,
  pass_2_duration, dock_round_trip. ``observe(value_s)`` records one sample.

The registry is process-local. Cross-process aggregation is the
orchestrator's job (chunk N reads each process's JSONL) — keeping this
in-process means no IPC / no Prometheus dep / nothing to scrape.

``snapshot()`` returns a flat dict keyed by ``"<kind>.<name>"`` so the
output is one envelope-friendly object that can ride directly inside an
event line as ``metrics_snapshot``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class _CounterState:
    value: int = 0


@dataclass
class _GaugeState:
    value: float = 0.0
    set_at: Optional[float] = None


@dataclass
class _TimerState:
    count: int = 0
    sum_s: float = 0.0
    min_s: Optional[float] = None
    max_s: Optional[float] = None


class MetricsRegistry:
    """Thread-safe registry for counters, gauges, and timers.

    Metric names are ``snake_case`` strings. Reusing a name with a
    different kind raises — surfaces the typo as a hard error rather than
    silently overwriting prior samples with a wrong-shaped record.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, _CounterState] = {}
        self._gauges: Dict[str, _GaugeState] = {}
        self._timers: Dict[str, _TimerState] = {}

    # ----------------------------- Counters ---------------------------- #

    def increment(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._reject_clash(name, "counter")
            self._counters.setdefault(name, _CounterState()).value += int(by)

    def counter_value(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, _CounterState()).value

    # ------------------------------- Gauges ---------------------------- #

    def set_gauge(self, name: str, value: float, *, ts: Optional[float] = None) -> None:
        with self._lock:
            self._reject_clash(name, "gauge")
            g = self._gauges.setdefault(name, _GaugeState())
            g.value = float(value)
            g.set_at = ts

    def gauge_value(self, name: str) -> float:
        with self._lock:
            return self._gauges.get(name, _GaugeState()).value

    # ------------------------------- Timers ---------------------------- #

    def observe(self, name: str, duration_s: float) -> None:
        if duration_s < 0.0:
            # Negative duration would corrupt min/max; defensive — clip.
            duration_s = 0.0
        with self._lock:
            self._reject_clash(name, "timer")
            t = self._timers.setdefault(name, _TimerState())
            t.count += 1
            t.sum_s += float(duration_s)
            if t.min_s is None or duration_s < t.min_s:
                t.min_s = float(duration_s)
            if t.max_s is None or duration_s > t.max_s:
                t.max_s = float(duration_s)

    def timer_state(self, name: str) -> _TimerState:
        with self._lock:
            t = self._timers.get(name)
            if t is None:
                return _TimerState()
            # Return a copy so callers can't mutate registry state.
            return _TimerState(
                count=t.count, sum_s=t.sum_s, min_s=t.min_s, max_s=t.max_s,
            )

    # ----------------------------- Snapshot ---------------------------- #

    def snapshot(self) -> Dict[str, object]:
        """Flat ``{kind.name: value}`` dict for emitting as one event.

        Counters → int, gauges → float, timers → ``{count, sum_s, min_s,
        max_s, mean_s}``. Empty timers (count=0) are omitted; a zero-count
        timer carries no information.
        """
        out: Dict[str, object] = {}
        with self._lock:
            for name, c in self._counters.items():
                out[f"counter.{name}"] = c.value
            for name, g in self._gauges.items():
                out[f"gauge.{name}"] = g.value
            for name, t in self._timers.items():
                if t.count == 0:
                    continue
                out[f"timer.{name}"] = {
                    "count": t.count,
                    "sum_s": t.sum_s,
                    "min_s": t.min_s,
                    "max_s": t.max_s,
                    "mean_s": t.sum_s / t.count if t.count > 0 else 0.0,
                }
        return out

    # ----------------------------- Internals --------------------------- #

    def _reject_clash(self, name: str, kind: str) -> None:
        # Caller already holds the lock.
        for other_kind, store in (
            ("counter", self._counters),
            ("gauge", self._gauges),
            ("timer", self._timers),
        ):
            if other_kind != kind and name in store:
                raise ValueError(
                    f"metric name {name!r} already in use as a "
                    f"{other_kind}; cannot re-register as {kind}"
                )
