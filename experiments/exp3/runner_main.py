"""Experiment-3 CLI entry point.

Walks the A1×A2×A3×A4 grid via the EX-0 :class:`TrialRunner` and writes
a per-trial CSV. Resumable across sessions via the CSV log's
already-done set.

Usage::

    python -m experiments.exp3.runner_main \\
        --csv results/exp3.csv \\
        --n-trials 20

The grid axes are the four sweep knobs from §IV-D plus the four arms
(A1, A2, A3, A4). The defaults mirror the implementation plan §4.2
sweep — ``N ∈ {5, 10, 20}``, ``β ∈ {0.5, 1.0, 2.0}``,
``deadline_het ∈ {False, True}``, ``rrf ∈ {30, 60, 120}``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from experiments.runner import TrialGrid, TrialRunner

from hermes.scheduler.selector import TargetSelectorRL
from hermes.scheduler.selector.ddqn import DDQN

from .driver import ARMS, Exp3Driver
from .metrics import Exp3MetricSummary


log = logging.getLogger("experiments.exp3.runner_main")


def _build_grid(
    *,
    arms: Sequence[str],
    Ns: Sequence[int],
    betas: Sequence[float],
    rrfs: Sequence[float],
    deadline_het_modes: Sequence[bool],
    jittery_modes: Sequence[bool],
    n_trials: int,
    base_seed: int = 42,
) -> TrialGrid:
    return TrialGrid(
        independent_vars={
            "N": list(Ns),
            "beta": list(betas),
            "rrf": list(rrfs),
            "deadline_het": list(deadline_het_modes),
            "jittery": list(jittery_modes),
        },
        arms=list(arms),
        n_trials=n_trials,
        base_seed=base_seed,
    )


def _metric_columns() -> list[str]:
    """Metric + extra param columns the CSV declares.

    The runner adds ``cell_id``, ``arm``, ``trial_index``, ``seed``,
    and ``param_*`` automatically; we own the rest.
    """
    return Exp3MetricSummary.csv_columns() + [
        "n_devices",
        "beta",
        "deadline_het",
        "rf_range_m",
        "jittery",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.exp3.runner_main")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Per-trial CSV path (created if missing).")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Trials per cell (paired across arms).")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        help="Which arms to run; subset of A1 A2 A3 A4.")
    parser.add_argument("--N", nargs="+", type=int, default=[5, 10, 20],
                        help="Bucket-size sweep.")
    parser.add_argument(
        "--beta", nargs="+", type=float,
        default=[0.25, 0.5, 1.0, 2.0],
        help=(
            "Deadline-tightness sweep. β scales the mission budget "
            "(mission_budget * β). β=0.25 yields a budget-tight regime "
            "where A3's feasibility filter actually fires; the relaxed "
            "values (β >= 1.0) are the budget-rich regime where the "
            "filter never triggers."
        ),
    )
    parser.add_argument("--rrf", nargs="+", type=float, default=[30.0, 60.0, 120.0],
                        help="rf_range_m sweep.")
    parser.add_argument("--deadline-het", nargs="+", type=int,
                        default=[0, 1],
                        help="0 = uniform deadlines; 1 = heterogeneous.")
    parser.add_argument(
        "--jittery", nargs="+", type=int, default=[0, 1],
        help=(
            "Network-link jitter axis. 0 = clean (no packet loss, no "
            "latency jitter); 1 = jittery (2%% packet loss + 30%% "
            "latency jitter, matching Exp.\\ 1's --jittery cell). "
            "Default sweeps both so the paper can compare the "
            "scheduling ablation across clean and jittery conditions. "
            "Pass `--jittery 0` to skip the jittery cells."
        ),
    )
    parser.add_argument(
        "--clean-upload-bytes", type=float, default=1.0e6,
        help=(
            "Per-contact upload payload in CLEAN cells. Default 1 MB "
            "represents a small FL update; combined with --clean-"
            "upload-bps this yields upload_s << collect_s, so A3's "
            "feasibility filter rarely fires in clean cells."
        ),
    )
    parser.add_argument(
        "--clean-upload-bps", type=float, default=1.0e7,
        help=(
            "Nominal upload rate (bits/sec) in CLEAN cells. Default "
            "10 Mbps is a healthy mule-edge link. Combined with the "
            "1 MB default payload this gives upload_s ≈ 0.8 s per "
            "contact — minimal network pressure."
        ),
    )
    parser.add_argument(
        "--jittery-upload-bytes", type=float, default=1.0e7,
        help=(
            "Per-contact upload payload in JITTERY cells. Default 10 MB "
            "matches a typical FL gradient payload. Combined with "
            "--jittery-upload-bps this yields upload_s ~ 80 s per "
            "contact, putting genuine pressure on A3's filter."
        ),
    )
    parser.add_argument(
        "--jittery-upload-bps", type=float, default=1.0e6,
        help=(
            "Nominal upload rate (bits/sec) in JITTERY cells. Default "
            "1 Mbps is a stressed mule-edge link. The simulator also "
            "applies inverse-distance falloff and ±20%% rate jitter "
            "on top, plus 2%% packet loss and 30%% latency jitter "
            "(the Exp.\\ 1 ``--jittery`` parity)."
        ),
    )
    parser.add_argument(
        "--jittery-a1-dead-zone-pct", type=float, default=60.0,
        help=(
            "Fraction (0-100) of A1 clients marked PERSISTENTLY "
            "long-range-unreachable in JITTERY cells. Models terrain "
            "blockage, range-edge SNR collapse, and similar correlated "
            "(not i.i.d.) failure modes that the per-round "
            "link_quality multiplier alone can't capture: with N=20 "
            "FedAvg rounds, an i.i.d. 40%% per-round failure unions "
            "to ~100%% mission_completion_rate, but a 60%% dead zone "
            "caps it at 40%%. Tune lower for a milder jittery story; "
            "tune higher (e.g., 80) to put A1 below the mule arms in "
            "jittery on cumulative metrics. Mule arms have no "
            "analogous knob — short-range device↔mule RF is treated "
            "as reliable in the design."
        ),
    )
    parser.add_argument(
        "--clean-a1-dead-zone-pct", type=float, default=0.0,
        help=(
            "Fraction of A1 clients marked unreachable in CLEAN "
            "cells. Default 0 — the long-range link works for "
            "everyone when network conditions are good."
        ),
    )
    parser.add_argument("--timeout-s", type=float, default=300.0,
                        help="Soft per-trial timeout (warning only).")
    parser.add_argument(
        "--selector-weights", type=Path, default=None,
        help=(
            "Path to a trained DDQN .npz produced by "
            "experiments.exp3.train_a4. If omitted, A4 uses a "
            "RANDOM-INIT network (smoke-test only — not paper-grade)."
        ),
    )
    parser.add_argument(
        "--require-trained-a4", action="store_true",
        help=(
            "Refuse to run when A4 would use random-init weights. "
            "Set this on every paper-grade run."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    grid = _build_grid(
        arms=args.arms,
        Ns=args.N,
        betas=args.beta,
        rrfs=args.rrf,
        deadline_het_modes=[bool(b) for b in args.deadline_het],
        jittery_modes=[bool(b) for b in args.jittery],
        n_trials=args.n_trials,
        base_seed=args.base_seed,
    )

    selector_a4: Optional[TargetSelectorRL] = None
    if args.selector_weights is not None:
        ddqn = DDQN.load(args.selector_weights)
        selector_a4 = TargetSelectorRL(ddqn=ddqn, epsilon=0.0)
        log.info("loaded A4 selector weights from %s", args.selector_weights)
    elif "A4" in args.arms:
        msg = (
            "A4 is in --arms but --selector-weights was not provided. "
            "A4 will use random-init DDQN weights — results are not "
            "paper-grade. Train weights first with "
            "`python -m experiments.exp3.train_a4 --output PATH` and "
            "re-run with --selector-weights PATH."
        )
        if args.require_trained_a4:
            parser.error(msg)
        log.warning(msg)

    driver = Exp3Driver(
        selector_a4=selector_a4,
        clean_upload_bytes=float(args.clean_upload_bytes),
        clean_upload_bps=float(args.clean_upload_bps),
        jittery_upload_bytes=float(args.jittery_upload_bytes),
        jittery_upload_bps=float(args.jittery_upload_bps),
        clean_a1_dead_zone_pct=float(args.clean_a1_dead_zone_pct),
        jittery_a1_dead_zone_pct=float(args.jittery_a1_dead_zone_pct),
    )
    log.info(
        "clean upload model: %g bytes / contact at %g bps",
        args.clean_upload_bytes, args.clean_upload_bps,
    )
    log.info(
        "jittery upload model: %g bytes / contact at %g bps "
        "+ 2%% packet loss + 30%% latency jitter",
        args.jittery_upload_bytes, args.jittery_upload_bps,
    )
    runner = TrialRunner(
        grid=grid,
        log_path=args.csv,
        metric_columns=_metric_columns(),
        timeout_s=args.timeout_s,
    )
    n = runner.run(driver.run_trial)
    print(f"wrote {n} new trial rows to {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
