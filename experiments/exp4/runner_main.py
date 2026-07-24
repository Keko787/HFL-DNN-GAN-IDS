"""Experiment-4 CLI entry point (chunk EX-4.0).

Drives the integrated two-pass orchestrator over a small grid via the
shared :class:`~experiments.runner.TrialRunner` and writes a resumable
per-trial CSV. This is the first *genuinely integrated* L2+L3
measurement — real subprocesses, real TCP, real cross-mule FedAvg —
distinct from Experiment 3's abstracted sim.

Usage::

    python -m experiments.exp4.runner_main \\
        --csv results/exp4_smoke.csv \\
        --N 2 --rrf 60 --n-missions 1 --n-trials 1

Defaults are a tiny smoke grid (each trial spawns a real process tree,
so keep the grid small until the paper run). Only arm **H1** exists in
EX-4.0; H0/H2/H3 arrive in later chunks.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from experiments.runner import TrialGrid, TrialRunner

from .driver import ARMS, Exp4Driver
from .metrics import Exp4MetricSummary

log = logging.getLogger("experiments.exp4.runner_main")


def _build_grid(
    *,
    arms: Sequence[str],
    Ns: Sequence[int],
    rrfs: Sequence[float],
    n_missions_values: Sequence[int],
    n_trials: int,
    base_seed: int = 42,
) -> TrialGrid:
    return TrialGrid(
        independent_vars={
            "N": list(Ns),
            "rrf": list(rrfs),
            "n_missions": list(n_missions_values),
        },
        arms=list(arms),
        n_trials=n_trials,
        base_seed=base_seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp4.runner_main")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Per-trial CSV path (created if missing; resumable).")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Trials per cell (paired across arms).")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        help=f"Which arms to run; EX-4.0 ships {list(ARMS)}.")
    parser.add_argument("--N", nargs="+", type=int, default=[2],
                        help="Device-population sweep.")
    parser.add_argument("--rrf", nargs="+", type=float, default=[60.0],
                        help="rf_range_m sweep.")
    parser.add_argument("--n-missions", nargs="+", type=int, default=[2],
                        help="Missions (FL rounds) per trial.")
    parser.add_argument(
        "--trial-budget-s", type=float, default=120.0,
        help="Hard per-trial wall-clock budget; the process tree is "
             "killed on overrun and the row recorded as an error.",
    )
    parser.add_argument(
        "--startup-timeout-s", type=float, default=30.0,
        help="Timeout for the topology to come up (all ports bound).",
    )
    parser.add_argument(
        "--timeout-s", type=float, default=None,
        help="Soft harness timeout (warning-only label). Defaults to "
             "trial-budget-s so a killed trial is also labelled.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    grid = _build_grid(
        arms=args.arms,
        Ns=args.N,
        rrfs=args.rrf,
        n_missions_values=args.n_missions,
        n_trials=args.n_trials,
        base_seed=args.base_seed,
    )

    driver = Exp4Driver(
        trial_budget_s=float(args.trial_budget_s),
        startup_timeout_s=float(args.startup_timeout_s),
    )
    runner = TrialRunner(
        grid=grid,
        log_path=args.csv,
        metric_columns=Exp4MetricSummary.csv_columns(),
        timeout_s=(args.timeout_s if args.timeout_s is not None else args.trial_budget_s),
    )
    log.info(
        "exp4 grid: arms=%s N=%s rrf=%s n_missions=%s trials=%d (%d cells)",
        args.arms, args.N, args.rrf, args.n_missions, args.n_trials,
        grid.total(),
    )
    n = runner.run(driver.run_trial)
    print(f"wrote {n} new trial rows to {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
