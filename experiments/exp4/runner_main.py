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
    # ---- EX-4.1 real-model flags ---- #
    parser.add_argument(
        "--real-model", action="store_true",
        help="Run the real canonical DNN-IDS in the loop (EX-4.1): real "
             "training on each device + per-round held-out convergence. "
             "Omit for the EX-4.0 stub (federation metrics only).",
    )
    parser.add_argument(
        "--data-source", choices=["canonical", "synthetic"], default="canonical",
        help="Real-model data: 'canonical' = the production CICIOT pipeline "
             "(balanced, 21 features, paper-faithful); 'synthetic' = a "
             "real-shaped separable task (fast, no dataset needed).",
    )
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--tau", type=float, default=0.9,
                        help="Target accuracy for the T@tau metric.")
    parser.add_argument("--train-files", type=int, default=3,
                        help="canonical: CICIOT csv parts to draw train from.")
    parser.add_argument("--test-files", type=int, default=1,
                        help="canonical: CICIOT csv parts to draw test from.")
    parser.add_argument("--train-dataset-size", type=int, default=20000,
                        help="canonical: total balanced train rows (50/50).")
    parser.add_argument("--test-dataset-size", type=int, default=8000,
                        help="canonical: total balanced test rows before attack reduction.")
    parser.add_argument("--attack-eval-ratio", type=float, default=0.5,
                        help="canonical: attack fraction kept in the test set.")
    parser.add_argument("--synth-rows-per-device", type=int, default=512)
    parser.add_argument("--synth-test-rows", type=int, default=512)
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # H0 (traditional flat FL) is a real-model convergence baseline; drop it
    # from a stub run rather than erroring every H0 trial.
    arms = list(args.arms)
    if not args.real_model and "H0" in arms:
        log.warning("H0 requires --real-model; dropping it from this stub run")
        arms = [a for a in arms if a != "H0"]
    if not arms:
        parser.error("no runnable arms left (H0 needs --real-model)")

    grid = _build_grid(
        arms=arms,
        Ns=args.N,
        rrfs=args.rrf,
        n_missions_values=args.n_missions,
        n_trials=args.n_trials,
        base_seed=args.base_seed,
    )

    driver = Exp4Driver(
        trial_budget_s=float(args.trial_budget_s),
        startup_timeout_s=float(args.startup_timeout_s),
        real_model=bool(args.real_model),
        data_source=args.data_source,
        local_epochs=int(args.local_epochs),
        local_batch_size=int(args.local_batch_size),
        tau=float(args.tau),
        train_files=int(args.train_files),
        test_files=int(args.test_files),
        train_dataset_size=int(args.train_dataset_size),
        test_dataset_size=int(args.test_dataset_size),
        attack_eval_ratio=float(args.attack_eval_ratio),
        synth_rows_per_device=int(args.synth_rows_per_device),
        synth_test_rows=int(args.synth_test_rows),
    )
    if args.real_model:
        log.info(
            "EX-4.1 real-model run: source=%s epochs=%d batch=%d tau=%.2f",
            args.data_source, args.local_epochs, args.local_batch_size, args.tau,
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
