"""Experiment 4 — integrated end-to-end (Algorithm 2) measurement.

Experiment 4 is the *measured realization of Algorithm 2*: L1 + L2 + L3
composed in a single trial, against traditional FL. See
``DeveloperDocs/HERMES_Experiment4_Integrated_Design_and_Plan.md``.

**Chunk EX-4.0 (this module) — instrumentation, no new physics.**
It drives the *real* multi-process orchestrator (``MultiProcessOrchestrator``)
to natural exit and computes the federation-side scheduling metrics from
the real JSONL event stream — the first genuinely-integrated L2+L3
measurement, distinct from Experiment 3's abstracted sim. It ships arm
**H1** (mule + Four-Stage Gated Scheduler + two-pass HFL, deterministic
distance ranking, no RL selector, no L1). Later chunks add the real
DNN-IDS (EX-4.1), the RL selector + shaped radio (EX-4.2), and real L1
(EX-4.3).

Public surface:

* :class:`~experiments.exp4.topology_builder.build_exp4_topology` — a
  finite :class:`~hermes.processes.TopologyConfig` from a trial cell.
* :func:`~experiments.exp4.events_consumer.consume_run_dir` — parse a
  finished run's per-process JSONL into an :class:`Exp4Observation`.
* :class:`~experiments.exp4.metrics.Exp4MetricSummary` /
  :func:`~experiments.exp4.metrics.summarise_observation` — the
  federation-side reportables, reusing the Experiment-3 metric
  definitions verbatim.
* :class:`~experiments.exp4.driver.Exp4Driver` — the
  ``run_trial(cell) -> dict`` slot for the shared trial harness.
"""

from .events_consumer import (
    Exp4Observation,
    MissionRecord,
    consume_run_dir,
    observation_from_rows,
)
from .metrics import Exp4MetricSummary, summarise_observation
from .topology_builder import build_exp4_topology

__all__ = [
    "Exp4Observation",
    "MissionRecord",
    "consume_run_dir",
    "observation_from_rows",
    "Exp4MetricSummary",
    "summarise_observation",
    "build_exp4_topology",
]
