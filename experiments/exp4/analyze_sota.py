"""Whole-scheduler SOTA comparison — reach-rate across a budget axis.

**Primary metric: reach-rate.** The fraction of trials whose global model reaches
τ within the mission budget. This is the metric the architecture is actually
about — availability under a constraint — and it is what differs. *Training time
to accuracy is a measured tie in this system* (conditional on reaching τ, arms
take identical rounds), so it is reported alongside as the honest null rather
than led with.

Reach-rate is a **binary outcome per paired seed**, so the correct test is
**McNemar on the discordant pairs** — the seeds where one arm reached τ and the
other did not. A t-test on the rates would ignore the pairing and overstate
significance.

Everything is recomputed from the retained traces, so τ is an analysis choice.

Usage::

    python -m experiments.exp4.analyze_sota
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from math import comb
from typing import Dict, List, Optional, Sequence

import numpy as np

from experiments.analysis.stats import (
    bootstrap_ci,
    paired_wilcoxon_with_cliffs_delta,
)
from experiments.exp4.tau_from_traces import _parse, round_history, t_at_tau


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar over the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def load_hists(trace_root: str) -> Dict[str, Dict[str, list]]:
    out: Dict[str, Dict[str, list]] = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(trace_root, "*"))):
        if not os.path.isdir(d):
            continue
        arm, trial = _parse(d)
        h = round_history(d, "accuracy")
        if h:
            out[arm][trial] = h
    return out


def report_point(label: str, trace_root: str, csv_path: str,
                 taus: Sequence[float], baseline: str,
                 treatments: Sequence[str]) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    if not os.path.isdir(trace_root):
        print(f"  no traces at {trace_root}")
        return
    hists = load_hists(trace_root)
    arms = sorted(hists)
    print(f"  arms: {arms}   trials each: "
          f"{ {a: len(hists[a]) for a in arms} }")

    for t in treatments:
        if t not in hists or baseline not in hists:
            continue
        shared = sorted(set(hists[t]) & set(hists[baseline]))
        print(f"\n  --- {baseline} (ours) vs {t} --- paired n={len(shared)}")
        hdr = (f"  {'tau':>6} {'ours':>7} {t:>7} {'ours-only':>10} "
               f"{t+'-only':>10} {'McNemar p':>10}  verdict")
        print(hdr)
        for tau in taus:
            ours = {k: t_at_tau(hists[baseline][k], tau) is not None for k in shared}
            them = {k: t_at_tau(hists[t][k], tau) is not None for k in shared}
            b = sum(1 for k in shared if ours[k] and not them[k])
            c = sum(1 for k in shared if them[k] and not ours[k])
            p = exact_mcnemar(b, c)
            ro = sum(ours.values()) / len(shared) if shared else 0
            rt = sum(them.values()) / len(shared) if shared else 0
            if p < 0.05:
                verdict = f"{baseline} REACHES MORE" if b > c else f"{t} REACHES MORE"
            else:
                verdict = "tie"
            print(f"  {tau:>6.2f} {ro:>7.2f} {rt:>7.2f} {b:>10} {c:>10} "
                  f"{p:>10.4f}  {verdict}")

        # The honest null: rounds to tau among pairs where BOTH reached it.
        for tau in taus:
            both = [k for k in shared
                    if t_at_tau(hists[baseline][k], tau) is not None
                    and t_at_tau(hists[t][k], tau) is not None]
            if len(both) < 2:
                continue
            a = [float(t_at_tau(hists[baseline][k], tau)) for k in both]
            d = [float(t_at_tau(hists[t][k], tau)) for k in both]
            test = paired_wilcoxon_with_cliffs_delta(a, d)
            md, lo, hi = bootstrap_ci([x - y for x, y in zip(a, d)],
                                      lambda v: float(np.mean(v)))
            print(f"  {'':>6} conditional rounds@{tau:.2f}: ours {np.mean(a):.2f} "
                  f"vs {np.mean(d):.2f}, diff {md:+.3f} [{lo:+.2f},{hi:+.2f}] "
                  f"p={test.p_value:.4f}  (fewer = faster)")

    # Outcome context straight from the csv.
    if os.path.exists(csv_path):
        rows = [r for r in csv.DictReader(open(csv_path)) if r["status"] == "ok"]
        by = defaultdict(list)
        for r in rows:
            by[r["arm"]].append(r)
        print(f"\n  outcome means (n={len(rows)} rows):")
        print(f"  {'arm':<4}{'auc':>9}{'acc':>9}{'compl':>9}{'yield':>9}{'cover':>9}")
        for a in sorted(by):
            def m(col):
                v = [float(x[col]) for x in by[a] if x.get(col) not in (None, "")]
                return sum(v) / len(v) if v else float("nan")
            print(f"  {a:<4}{m('final_auc'):>9.4f}{m('final_accuracy'):>9.4f}"
                  f"{m('mission_completion_rate'):>9.3f}{m('update_yield'):>9.3f}"
                  f"{m('coverage'):>9.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analyze_sota")
    ap.add_argument("--root", default="results/exp4_sota")
    ap.add_argument("--tau", type=float, nargs="+", default=[0.85, 0.82, 0.88])
    ap.add_argument("--baseline", default="H1", help="our arm")
    ap.add_argument("--treatments", nargs="+", default=["D1", "D2"])
    args = ap.parse_args(argv)

    print("Whole-scheduler SOTA comparison — reach-rate at tau")
    print("Primary: reach-rate (McNemar, paired). Secondary: conditional "
          "rounds-to-tau, expected to tie.")
    for label, traces, csvp in (
        ("BUDGET 120 s — the loose constraint",
         f"{args.root}/pilot_traces", f"{args.root}/pilot.csv"),
        ("BUDGET 60 s — the tight constraint",
         f"{args.root}/b60_traces", f"{args.root}/b60.csv"),
    ):
        report_point(label, traces, csvp, args.tau, args.baseline,
                     args.treatments)
    print("\nReading: if our arm wins on reach-rate only at the TIGHT budget, the "
          "gate earns its keep exactly where the constraint binds — which is the "
          "claim the architecture supports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
