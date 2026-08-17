"""Recompute time-to-accuracy (T@τ) from retained traces, at any τ.

**Why this exists.** The committed `t_at_tau_round` column is baked at whatever τ
the run used, and the original default (0.9) sat *above* the p90 of
`final_accuracy` — only 5.9 % of trials ever reached it, so the metric measured
almost nothing. Changing τ would normally mean re-running everything.

It does not, because `--keep-event-traces` preserved the per-round `model_eval`
events (`cluster_round`, `accuracy`, `auc`, `loss`). T@τ is a *function of that
history*, so any τ can be evaluated after the fact. This is the first payoff of
trace retention, and the concrete case for keeping it on every run.

**One asymmetry that matters.** Only the **mule arms** are traced. H0 runs
in-process with no orchestrator, so it emits no run-dir JSONL — meaning T@τ can be
recomputed for H1/H2/H3/B1/B2 but **not** for H0. An H0-vs-H1 time-to-accuracy
comparison still needs a re-run; the L1 comparison (H2 vs H3) does not.

Usage::

    python -m experiments.exp4.tau_from_traces \\
        --traces results/exp4_matrix/C_traces --tau 0.82 --treatment H3 --baseline H2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.analysis.stats import (
    bootstrap_ci,
    paired_wilcoxon_with_cliffs_delta,
)


def round_history(trace_dir: str, metric: str = "accuracy") -> List[Tuple[int, float]]:
    """`(cluster_round, value)` pairs from one trial's `model_eval` events."""
    hist: List[Tuple[int, float]] = []
    for f in glob.glob(os.path.join(trace_dir, "*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") == "model_eval" and metric in r:
                    hist.append((int(r["cluster_round"]), float(r[metric])))
    return sorted(set(hist))


def t_at_tau(hist: List[Tuple[int, float]], tau: float) -> Optional[int]:
    """First round whose value reaches τ, or None if never reached.

    Round 0 is the *initial* evaluation before any aggregation; it is a legal
    answer only in the degenerate case where the seed model already clears τ.
    """
    for rnd, val in hist:
        if val >= tau:
            return rnd
    return None


def _parse(trace_dir: str) -> Tuple[str, str]:
    """(arm, trial_index) from the directory name written by `trace_dir_name`."""
    base = os.path.basename(trace_dir.rstrip("/\\"))
    parts = base.split("__")
    if len(parts) < 4:
        return ("?", base)
    return (parts[-3], parts[-2].lstrip("t"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tau_from_traces")
    ap.add_argument("--traces", required=True,
                    help="Trace root, e.g. results/exp4_matrix/C_traces")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.82],
                    help="One or more thresholds to evaluate.")
    ap.add_argument("--treatment", default="H3")
    ap.add_argument("--baseline", default="H2")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "auc"])
    args = ap.parse_args(argv)

    hists: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for d in sorted(glob.glob(os.path.join(args.traces, "*"))):
        if not os.path.isdir(d):
            continue
        arm, trial = _parse(d)
        h = round_history(d, args.metric)
        if h:
            hists.setdefault(arm, {})[trial] = h

    if not hists:
        print(f"no model_eval history under {args.traces}")
        return 1
    print(f"traces: {args.traces}   metric={args.metric}")
    for arm in sorted(hists):
        print(f"  {arm}: {len(hists[arm])} trials with round history")

    t, b = args.treatment, args.baseline
    if t not in hists or b not in hists:
        print(f"\nneed both {t} and {b}; H0 is never traced (runs in-process).")
        return 1
    shared = sorted(set(hists[t]) & set(hists[b]))
    print(f"  paired: {len(shared)}\n")

    hdr = (f"{'tau':>6} {'reach_'+t:>9} {'reach_'+b:>9} {'both':>6} "
           f"{t+' rnds':>9} {b+' rnds':>9} {'diff':>8} {'CI95':>18} {'p':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for tau in args.tau:
        tt = {k: t_at_tau(hists[t][k], tau) for k in shared}
        bb = {k: t_at_tau(hists[b][k], tau) for k in shared}
        rt = sum(1 for v in tt.values() if v is not None)
        rb = sum(1 for v in bb.values() if v is not None)
        both = [k for k in shared if tt[k] is not None and bb[k] is not None]
        if len(both) < 2:
            print(f"{tau:>6.2f} {rt:>9} {rb:>9} {len(both):>6}   "
                  f"(too few pairs reach tau to test)")
            continue
        a = [float(tt[k]) for k in both]
        c = [float(bb[k]) for k in both]
        diffs = [x - y for x, y in zip(a, c)]
        md, lo, hi = bootstrap_ci(diffs, lambda v: float(np.mean(v)))
        test = paired_wilcoxon_with_cliffs_delta(a, c)
        # FEWER rounds is better, so a negative difference favours the treatment.
        if hi < 0 and test.p_value < 0.05:
            verdict = f"{t} FASTER"
        elif lo > 0 and test.p_value < 0.05:
            verdict = f"{b} faster"
        else:
            verdict = "tie"
        print(f"{tau:>6.2f} {rt:>9} {rb:>9} {len(both):>6} "
              f"{np.mean(a):>9.2f} {np.mean(c):>9.2f} {md:>+8.3f} "
              f"[{lo:>+6.2f},{hi:>+6.2f}] {test.p_value:>8.4f}  {verdict}")

    print("\nFewer rounds = faster. A tie on this metric with a win on final "
          "accuracy means the arm converges to a better model, not a quicker one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
