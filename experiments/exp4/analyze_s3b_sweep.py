"""Read out the S3b budget sweep: what does deadline enforcement cost?

Each shard of the sweep is the same cell at the same 20 paired seeds, differing
only in ``--mission-budget-s``. The control shard runs with the gate off, so it
reproduces the committed configuration and every other shard can be compared
against it **paired by seed**.

Reports, per budget: the mean of each participation metric, the paired
difference against the control, and a bootstrap 95% CI — the same estimator the
methodology tables use, so this can be quoted directly.

    python -m experiments.exp4.analyze_s3b_sweep
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.analysis.exp4 import load_exp4
from experiments.analysis.stats import bootstrap_ci, paired_wilcoxon_with_cliffs_delta

METRICS = (
    "update_yield",
    "mission_completion_rate",
    "round_close_rate_kmin2",
    "final_auc",
)

# Shard basename -> budget label, in ladder order.
LADDER = [
    ("s3b_control", "gate OFF"),
    ("s3b_b120", "120 s"),
    ("s3b_b060", "60 s (knee)"),
    ("s3b_b030", "30 s"),
    ("s3b_b015", "15 s"),
]


def _paired(a: pd.DataFrame, b: pd.DataFrame, metric: str):
    """Join two shards on trial_index and return matched value arrays."""
    ka = a.set_index("trial_index")[metric].apply(pd.to_numeric, errors="coerce")
    kb = b.set_index("trial_index")[metric].apply(pd.to_numeric, errors="coerce")
    j = pd.concat([ka.rename("a"), kb.rename("b")], axis=1).dropna()
    return j["a"].to_numpy(float), j["b"].to_numpy(float)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.exp4.analyze_s3b_sweep")
    p.add_argument("--dir", default="results/exp4_s3b")
    args = p.parse_args(argv)

    frames = {}
    for base, label in LADDER:
        path = os.path.join(args.dir, base + ".csv")
        if os.path.exists(path):
            df = load_exp4(path)
            if len(df):
                frames[label] = df

    if "gate OFF" not in frames:
        print(f"No control shard in {args.dir} — run experiments/exp4/"
              f"run_s3b_budget_sweep.sh first.")
        return 1

    control = frames["gate OFF"]
    print(f"S3b budget sweep — {args.dir}")
    print(f"control: {len(control)} ok rows\n")

    print(f"{'budget':>13} {'metric':>24} {'n':>3} {'value':>8} {'vs control':>11} "
          f"{'95% CI':>20} {'verdict':>10}")
    print("-" * 96)

    for label in [l for _, l in LADDER if l in frames]:
        df = frames[label]
        for m in METRICS:
            if m not in df.columns:
                continue
            vals = pd.to_numeric(df[m], errors="coerce").dropna()
            if label == "gate OFF":
                print(f"{label:>13} {m:>24} {len(vals):3d} {vals.mean():8.4f} "
                      f"{'—':>11} {'—':>20} {'baseline':>10}")
                continue
            a, b = _paired(df, control, m)
            if a.size < 2:
                continue
            diff = a - b
            _, lo, hi = bootstrap_ci(diff, np.mean, n_bootstraps=5000)
            t = paired_wilcoxon_with_cliffs_delta(a, b)
            claimed = (lo > 0 or hi < 0) and t.p_value < 0.05
            verdict = ("lower*" if diff.mean() < 0 else "higher*") if claimed else "tie"
            print(f"{label:>13} {m:>24} {a.size:3d} {a.mean():8.4f} "
                  f"{diff.mean():+11.4f} [{lo:+.4f},{hi:+.4f}] {verdict:>10}")
        print()

    print("* = CI excludes 0 AND paired Wilcoxon p<0.05 (the protocol's claim rule).")
    print()
    print("Reading it: the budget at which participation first departs the control is")
    print("where deadline enforcement starts costing real coverage. Above that the gate")
    print("is slack; below it the mule is refusing work it previously (wrongly) accepted.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
