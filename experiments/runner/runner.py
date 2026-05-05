"""Drives a per-experiment ``run_trial(cell)`` callable across a grid.

Responsibilities (Chunk EX-0):

* Walk the :class:`TrialGrid`, skipping cells already in the
  :class:`CSVTrialLog`.
* Time each trial wall-clock; capture exceptions; mark status as
  ``ok`` / ``error`` / ``timeout``.
* Optional soft timeout — wall-clock budget per trial. The runner
  doesn't hard-kill the driver (cross-platform process management is
  the driver's responsibility); a soft timeout records the elapsed
  time and a warning. Drivers that spawn subprocesses must respect
  their own timeout knobs to actually stop.
* Failing one trial doesn't break subsequent trials — the row is
  written with status=error and the loop continues.
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from .csv_log import CSVTrialLog
from .grid import Cell, TrialGrid

log = logging.getLogger("experiments.runner")


TrialDriver = Callable[[Cell], Mapping[str, Any]]


@dataclass
class TrialOutcome:
    """In-memory summary of one trial — the runner returns these for
    downstream reporting (progress bars, immediate stats)."""

    cell: Cell
    status: str  # "ok" | "skipped" | "error" | "timeout"
    duration_s: float
    error: Optional[str] = None
    row: Optional[Dict[str, Any]] = None


class TrialRunner:
    """Walks a :class:`TrialGrid` and writes results to a :class:`CSVTrialLog`.

    ``run_trial`` is the per-experiment driver. Its return value is a
    dict of metric columns; the runner merges that with the cell's
    ``to_row_prefix()`` (cell_id, arm, trial_index, seed, param_*) and
    its own status/duration columns to produce the CSV row.

    The runner stamps three columns on every row:

    * ``status``: ``ok`` / ``error`` / ``timeout``
    * ``duration_s``: wall-clock seconds
    * ``error``: traceback's last line on ``error``, message on
      ``timeout``, empty string on ``ok``
    """

    _STATUS_COLS = ("status", "duration_s", "error")

    def __init__(
        self,
        grid: TrialGrid,
        log_path,
        *,
        metric_columns,
        timeout_s: Optional[float] = None,
        clock=None,
    ) -> None:
        # Fieldnames = key prefix + param_* + metric columns + status.
        # We collect param_* dynamically from the grid's first cell.
        try:
            sample_cell = next(iter(grid))
        except StopIteration:
            sample_cell = None

        prefix = list(sample_cell.to_row_prefix().keys()) if sample_cell else [
            "cell_id", "arm", "trial_index", "seed",
        ]
        all_cols = list(prefix) + [
            c for c in metric_columns if c not in prefix
        ] + [c for c in self._STATUS_COLS if c not in prefix and c not in metric_columns]
        self._csv = CSVTrialLog(log_path, fieldnames=all_cols)
        self._grid = grid
        self._timeout_s = timeout_s
        self._clock = clock or time.time

    @property
    def csv(self) -> CSVTrialLog:
        return self._csv

    def run(self, run_trial: TrialDriver) -> int:
        """Execute the grid. Returns the number of trials newly written.

        Already-done cells are skipped. Errors and timeouts are
        recorded with their respective status; the loop never aborts.
        """
        new_count = 0
        for cell in self._grid:
            outcome = self._run_one(cell, run_trial)
            if outcome.status == "skipped":
                continue
            new_count += 1
        return new_count

    def iter_outcomes(self, run_trial: TrialDriver):
        """Same as :meth:`run` but yields each :class:`TrialOutcome`.

        Useful for progress reporting + per-trial logging.
        """
        for cell in self._grid:
            outcome = self._run_one(cell, run_trial)
            yield outcome

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _run_one(self, cell: Cell, run_trial: TrialDriver) -> TrialOutcome:
        if self._csv.already_done(cell):
            return TrialOutcome(cell=cell, status="skipped", duration_s=0.0)

        start = self._clock()
        try:
            row = dict(run_trial(cell))
            duration = self._clock() - start

            if self._timeout_s is not None and duration > self._timeout_s:
                # Soft timeout — the driver returned, but late. Record
                # so the analysis can filter / refit on a retry.
                row[self._status_col("status")] = "timeout"
                row[self._status_col("error")] = (
                    f"trial took {duration:.1f}s (soft cap {self._timeout_s:.1f}s)"
                )
                outcome_status = "timeout"
                outcome_error = row[self._status_col("error")]
            else:
                row.setdefault(self._status_col("status"), "ok")
                row.setdefault(self._status_col("error"), "")
                outcome_status = str(row[self._status_col("status")])
                outcome_error = row[self._status_col("error")] or None

            row[self._status_col("duration_s")] = duration
            self._csv.append(cell, row)
            log.info(
                "trial done cell=%s arm=%s trial=%d status=%s dur=%.2fs",
                cell.cell_id, cell.arm, cell.trial_index,
                outcome_status, duration,
            )
            return TrialOutcome(
                cell=cell, status=outcome_status, duration_s=duration,
                error=outcome_error, row=row,
            )

        except Exception as e:
            duration = self._clock() - start
            tb_last = traceback.format_exception_only(type(e), e)[-1].strip()
            err_row: Dict[str, Any] = {
                self._status_col("status"): "error",
                self._status_col("duration_s"): duration,
                self._status_col("error"): tb_last,
            }
            try:
                self._csv.append(cell, err_row)
            except Exception:
                # If even the error row can't be written, log and
                # continue; the next trial's append still gets a chance.
                log.exception("failed to write error row for %s", cell)
            log.warning(
                "trial FAILED cell=%s arm=%s trial=%d in %.2fs: %s",
                cell.cell_id, cell.arm, cell.trial_index, duration, tb_last,
            )
            return TrialOutcome(
                cell=cell, status="error", duration_s=duration,
                error=tb_last, row=err_row,
            )

    @staticmethod
    def _status_col(name: str) -> str:
        return name
