"""Shared trial-grid harness used by Experiment 1 and Experiment 3.

Three small pieces:

* :class:`TrialGrid` — cartesian product of independent variables ×
  arms × trial indices, with paired-seed logic so the same trial index
  uses the same seed across arms (paired statistical tests).
* :class:`CSVTrialLog` — append-only CSV with an idempotent append +
  `(cell_id, arm, trial_index)` unique key. Resume = read existing
  rows, skip already-done.
* :class:`TrialRunner` — drives a per-experiment ``run_trial(cell)``
  callable through the grid, captures timing + exceptions, writes one
  row per trial.

Per-experiment drivers (``experiments.exp1.*`` / ``experiments.exp3.*``)
plug into the runner via the ``run_trial`` callable; the runner is
generic and doesn't know what's inside the cells.
"""

from __future__ import annotations

from .csv_log import CSVTrialLog
from .grid import Cell, TrialGrid
from .runner import TrialOutcome, TrialRunner

__all__ = [
    "Cell",
    "CSVTrialLog",
    "TrialGrid",
    "TrialOutcome",
    "TrialRunner",
]
