"""S3c pilot analysis — checklist §5.0.

Answers one question: **does mission-level window adaptation change anything when
the deadline gate is binding?** The two arms differ by exactly one flag, so any
difference is attributable.

Applies the project's standing claim rule: a difference is claimed only when the
bootstrap CI excludes 0 **and** the paired Wilcoxon p < 0.05. Anything else is
reported as a tie, however suggestive the means look.

Usage::

    python -m experiments.exp4.analyze_s3c_pilot --dir results/exp4_s3c
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse the project's statistics rather than re-deriving them here. The claim
# rule has to mean the same thing in this pilot as it does in every other Exp-4
# analysis, and a second implementation is a second thing that can disagree.
from experiments.analysis.stats import (
    bootstrap_ci,
    paired_wilcoxon_with_cliffs_delta,
)

# Metrics that answer "who got served", which is what S3c acts on.
METRICS = (
    "mission_completion_rate",
    "coverage",
    "update_yield",
    "jains_fairness",
    "participation_entropy",
    "completion_fairness",
    "rounds_closed",
    "missions_completed",
)

PAIR_KEY = ("cell_id", "trial_index")


def _load(path: Path) -> Dict[Tuple[str, str], dict]:
    rows: Dict[Tuple[str, str], dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            rows[tuple(r[k] for k in PAIR_KEY)] = r
    return rows


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analyze_s3c_pilot")
    ap.add_argument("--dir", type=Path, default=Path("results/exp4_s3c"))
    ap.add_argument("--baseline", default="off")
    ap.add_argument("--treatment", default="on")
    args = ap.parse_args(argv)

    base = _load(args.dir / f"{args.baseline}.csv")
    treat = _load(args.dir / f"{args.treatment}.csv")
    shared = sorted(set(base) & set(treat))

    print(f"S3c pilot — {args.treatment} vs {args.baseline}")
    print(f"  rows: {args.baseline}={len(base)}  {args.treatment}={len(treat)}  "
          f"paired={len(shared)}")

    # Pairing integrity: the seeds must match, or this is not a paired test.
    mismatched = [k for k in shared if base[k].get("seed") != treat[k].get("seed")]
    if mismatched:
        print(f"  !! {len(mismatched)} pair(s) have MISMATCHED SEEDS — not paired")
    else:
        print("  seeds match on every pair (comparison is genuinely paired)")

    # Provenance: confirm the arms differ in the one flag and agree on the other.
    def _uniq(rows, col):
        return sorted({r.get(col, "") for r in rows.values()})
    print(f"  mission_budget_s: {args.baseline}={_uniq(base,'mission_budget_s')} "
          f"{args.treatment}={_uniq(treat,'mission_budget_s')}")
    print(f"  window_adaptation: {args.baseline}={_uniq(base,'mission_window_adaptation')} "
          f"{args.treatment}={_uniq(treat,'mission_window_adaptation')}")
    print()

    if not shared:
        print("  no paired rows — nothing to test")
        return 1

    hdr = f"{'metric':<26} {'base':>8} {'treat':>8} {'diff':>9} {'CI95':>20} {'p':>8} {'delta':>7}  verdict"
    print(hdr)
    print("-" * len(hdr))

    claims: List[str] = []
    for m in METRICS:
        pairs = [(_f(base[k], m), _f(treat[k], m)) for k in shared]
        pairs = [(b, t) for b, t in pairs if b is not None and t is not None]
        if not pairs:
            print(f"{m:<26} {'-':>8} {'-':>8}   (not recorded)")
            continue
        bs = [b for b, _ in pairs]
        ts = [t for _, t in pairs]
        diffs = [t - b for b, t in pairs]
        mb, mt = sum(bs) / len(bs), sum(ts) / len(ts)

        if len(pairs) < 2:
            print(f"{m:<26} {mb:8.4f} {mt:8.4f}   (need >=2 pairs)")
            continue
        test = paired_wilcoxon_with_cliffs_delta(ts, bs)
        md, lo, hi = bootstrap_ci(diffs, lambda a: float(np.mean(a)))

        # The project's standing rule: CI excludes 0 AND p < 0.05.
        claimable = (lo > 0 or hi < 0) and test.p_value < 0.05
        verdict = "CLAIM" if claimable else "tie"
        if claimable:
            claims.append(f"{m} {md:+.4f} (p={test.p_value:.4f}, "
                          f"delta={test.cliffs_delta:+.3f} {test.delta_magnitude})")
        print(f"{m:<26} {mb:8.4f} {mt:8.4f} {md:+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {test.p_value:8.4f} "
              f"{test.cliffs_delta:+7.3f}  {verdict}")

    print()
    n_ties = 0
    for m in METRICS:
        pairs = [(_f(base[k], m), _f(treat[k], m)) for k in shared]
        pairs = [(b, t) for b, t in pairs if b is not None and t is not None]
        if pairs and all(b == t for b, t in pairs):
            n_ties += 1
    print(f"metrics identical on EVERY pair: {n_ties}/{len(METRICS)}")

    if claims:
        print(f"\nOUTCOME: non-null — {len(claims)} metric(s) meet the claim rule:")
        for c in claims:
            print(f"  - {c}")
        print("  -> S3c earns a place in the matrix; decide axis-vs-pinned (checklist §5.0).")
    else:
        print("\nOUTCOME: NULL at this operating point — no metric meets the claim rule")
        print("  -> Drop S3c from the matrix; report as a negative result and save the")
        print("     doubling. Note the operating point: this is one cell, not all budgets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
