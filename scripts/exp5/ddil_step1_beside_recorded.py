"""DDIL plan, decision 3: step 1's numbers beside the recorded ones (read-only).

Each of P1's eight arms in the three mission-backhaul cells P1 flew (default, 17 dB and
n = 2.7) is paired with itself in step 1's matching cell, on the same seeds (step 1 flew on
P1's), from both cells' ``_scored_ddil.csv`` files (DDIL build D0's scorer): its mean on each
physics, and step 1 minus recorded with a 95% percentile bootstrap CI. Then F and FX on the
recorded 17 dB channel against each of step 1's switches alone (its check cells) and every
switch together, the same way. Descriptive: no Holm, no claims. The probe reports
(``loss_probes.py DDIL1``) hold the paired, Holm-corrected comparisons against F.

Usage::

    py -3.11 scripts/exp5/ddil_step1_beside_recorded.py           # writes OUT
    py -3.11 scripts/exp5/ddil_step1_beside_recorded.py --check   # exit 1 if OUT would change

Added 17 Nov 2026; OUT was first written that day from the same files.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
RECORDED = REPO / "results/exp5_p1mbh/b3/s515"      # P1, mission backhaul
STEP1 = REPO / "results/exp5_ddil1/b3/s515"         # the DDIL1 probe
OUT = REPO / "results/exp5/scores/ddil/step1_beside_recorded.md"
TAU = 0.71                                          # loss_probes.TAU
ARMS = ("F", "FX", "H1", "D1", "D2", "D3", "D4", "D5")
#: (recorded cell, step 1's cell, title)
CELLS = (("n6k1_knee_base", "n6k1_knee_d1base", "default"),
         ("n6k1_knee_ampharsh", "n6k1_knee_d1ampharsh", "17 dB"),
         ("n6k1_knee_npl", "n6k1_knee_d1npl", "n = 2.7"))
#: step 1's cells for each switch alone on the recorded 17 dB channel, then every switch
SWITCHES = (("a1sess10", "session model, t_out 10 s"), ("a1sess2", "session model, t_out 2 s"),
            ("a1repoll", "re-poll"), ("a1est", "causal estimate"), ("a1burst40", "40% bursts"),
            ("d1ampharshpc", "every switch, per-class phases"),
            ("d1ampharsh", "every switch (step 1's 17 dB cell)"))
TIME = f"sim_s_to_tau{TAU:g}_or_end"
#: (column, label, decimals) of the two tables
BESIDE = ((TIME, "time to τ, miss = end (s)", 0), (f"reached_tau{TAU:g}", "reach", 2),
          ("sim_dwell_s_mean", "hover (s)", 0), ("update_yield", "updates", 2),
          ("sim_budget_overrun_s_mean", "overrun (s)", 1))
ALONE = ((TIME, "time to τ (s)", 0), ("sim_dwell_s_mean", "hover (s)", 0),
         ("update_yield", "updates", 2), ("sim_budget_overrun_s_mean", "overrun (s)", 1))
N_BOOT = 4000
SEED_BESIDE, SEED_ALONE = 11, 3                     # as on 17 Nov

Rows = Dict[Tuple[str, str], Dict[str, str]]


def load(path: Path) -> Rows:
    """A scored file's ok rows by (seed, trial_index)."""
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["seed"], r["trial_index"]): r for r in csv.DictReader(fh)
                if r.get("status", "ok") == "ok"}


def num(row: Dict[str, str], col: str) -> Optional[float]:
    try:
        return float(row[col])
    except (KeyError, ValueError):
        return None


def boot(d: Sequence[float], seed: int, n: int = N_BOOT) -> Tuple[float, float]:
    """The 95% percentile bootstrap CI of the mean of ``d``."""
    rng = random.Random(seed)
    means = sorted(st.fmean(rng.choices(d, k=len(d))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def paired(before: Rows, after: Rows, col: str, seed: int
           ) -> Optional[Tuple[float, float, float, float, float, int]]:
    """(mean before, mean after, mean of after - before, CI low, CI high, pairs) over the
    trials both have, on their shared seeds; None with no pair."""
    xs = [(num(before[k], col), num(after[k], col)) for k in sorted(set(before) & set(after))]
    xs = [(x, y) for x, y in xs if x is not None and y is not None]
    if not xs:
        return None
    d = [y - x for x, y in xs]
    lo, hi = boot(d, seed)
    return (st.fmean(x for x, _ in xs), st.fmean(y for _, y in xs), st.fmean(d), lo, hi, len(xs))


def _scored(root: Path, cell: str, arm: str) -> Path:
    return root / f"{cell}__{arm}_scored_ddil.csv"


def report() -> str:
    lines: List[str] = [
        "# Step 1 beside the recorded numbers (decision 3)", "",
        "*Written by `scripts/exp5/ddil_step1_beside_recorded.py` from the DDIL1 and P1 "
        "mission-backhaul trials' `_scored_ddil.csv` files (DDIL build D0's scorer; step 1 flown "
        "from 5a63b61d on 16 Nov, P1 from its recorded commit). Descriptive: each arm paired "
        "with itself on the same seeds, recorded physics against step 1's, with a 95% "
        "percentile bootstrap CI; no Holm, no claims. The probe reports "
        "(`results/exp5/scores/probes/DDIL1.md`) hold the paired, Holm-corrected comparisons "
        f"against F. Time to tau is at tau = {TAU:g}, with a missed trial given the trial's "
        "end.*", ""]
    for rec, s1, title in CELLS:
        lines += ["", f"## {title}: recorded -> step 1 (step 1 - recorded [95% CI])",
                  "| arm | pairs | " + " | ".join(c[1] for c in BESIDE) + " |",
                  "|---|---|" + "---|" * len(BESIDE)]
        for arm in ARMS:
            a, b = load(_scored(RECORDED, rec, arm)), load(_scored(STEP1, s1, arm))
            cells = []
            for col, _, nd in BESIDE:
                p = paired(a, b, col, SEED_BESIDE)
                cells.append("—" if p is None else
                             f"{p[0]:.{nd}f} → {p[1]:.{nd}f} ({p[2]:+.{nd}f} "
                             f"[{p[3]:+.{nd}f}, {p[4]:+.{nd}f}])")
            lines.append(f"| {arm} | {len(set(a) & set(b))} | " + " | ".join(cells) + " |")
    lines += ["", "## Which of step 1's switches does it: each alone on the recorded 17 dB channel",
              "", "Per-class phases and the recorded pricing, except the switch named (step 1's "
              "check cells). Each arm's step-1 value (step-1 cell minus recorded 17 dB, paired "
              "by seed [95% CI]).", ""]
    for arm in ("F", "FX"):
        rec = load(_scored(RECORDED, "n6k1_knee_ampharsh", arm))
        lines += ["", f"**{arm}**, recorded 17 dB: " + ", ".join(
            f"{lab} {st.fmean(num(r, col) for r in rec.values()):.{nd}f}"
            for col, lab, nd in ALONE), "",
            "| alone | " + " | ".join(c[1] for c in ALONE) + " |", "|---|" + "---|" * len(ALONE)]
        for tag, title in SWITCHES:
            s1 = load(_scored(STEP1, f"n6k1_knee_{tag}", arm))
            cells = []
            for col, _, nd in ALONE:
                p = paired(rec, s1, col, SEED_ALONE)
                cells.append("—" if p is None else
                             f"{p[1]:.{nd}f} ({p[2]:+.{nd}f} [{p[3]:+.{nd}f}, {p[4]:+.{nd}f}])")
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help=f"exit 1 if {OUT.relative_to(REPO)} would change; write nothing")
    a = ap.parse_args(argv)
    text = report()
    if a.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"{OUT.relative_to(REPO)}: {'current' if same else 'would change'}")
        return 0 if same else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
