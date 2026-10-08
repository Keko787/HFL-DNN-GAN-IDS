"""Exploratory: FeRRy's learned per-stop score (FQ, gamma = 0.75) against Chen et al.'s
learned scheduler (E3), paired by seed in Study 5.3's N = 6, one-mule cells.

    py -3.11 scripts/exp5/fq_vs_e3.py

Not pre-registered: added on 8 Oct 2026, after Study 5.5's verdict dropped FQ from
batch 2, to compare a learner inside FeRRy's plan and gates with a published learner
that has neither (M1, the planned monolithic agent, was not built). FQ flies the
verdict's pick (results/exp5/checkpoints/5.5-jittery56/g75/g0.75_s3.npz) in the same
cells and seeds as batch 1's arms and batch 2's E3, from the run

    launch.py run batch2 --study s53x --trials 20 --set "s53x.arms=['FQ-g75']"
      --set "s53x.K=[1]" --set "s53x.budgets=['knee','stress']"
      --set s53x.d4_faithful=false --set s53x.h0=false --set rl.keep_learned=true

It scores the two FQ trial CSVs with 5.3's scorer arguments and compares FQ with E3, F,
FX and D4 on time to tau, paired by seed, Holm across these comparisons only, so the
pre-registered 5.3 scores (results/exp5/scores/b2/s53x.*) are left as they are. It
writes results/exp5/scores/b2/exploratory_fq_vs_e3.{md,csv}.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

SETS = ["s53x.arms=['F','FX','D4','E3','FQ-g75']", "s53x.K=[1]",
        "s53x.budgets=['knee','stress']", "s53x.d4_faithful=false", "s53x.h0=false",
        "rl.keep_learned=true"]
CELLS = ["n6k1_knee", "n6k1_stress"]
REFS = ["E3", "F", "FX", "D4"]
VARIANT = "FQ-g75"
COL = "sim_s_to_tau0.71"


def _launcher():
    spec = importlib.util.spec_from_file_location("exp5_launch", HERE / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp5_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    L = _launcher()
    SC = L._scoring()
    s, _ = L.load_settings(L.PARAMS)
    ns = argparse.Namespace(trials=20, set=SETS, dataset=None,
                            **{o: None for o, _, _ in L.LEVERS})
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build("batch2", s, ["s53x"], "results/exp5")
    runs, entries, _, plan = L.score_plan("batch2", s, jobs)
    # Score FQ's trial CSVs (the others were scored with 5.3 already).
    for j in runs:
        if VARIANT in j.name and not L.done(j):
            print(f"scoring {j.out}")
            subprocess.run([sys.executable] + j.args, cwd=REPO, check=True)
    spec = plan.specs.get("s53x")
    by = {}
    for e in entries["s53x"]:
        SC.load(e)
        by[(e.cell, e.variant)] = e

    comps = []
    for cell in CELLS:
        var = by[(cell, VARIANT)]
        for ref in REFS:
            comps.append(SC.compare_pair(by[(cell, ref)], var, COL, spec, plan))
    for c in comps:
        c.study = "fq_vs_e3"            # one exploratory family, Holm across it alone
    SC.holm(comps, plan)

    def mean(e, col):
        xs = [SC._num(r.get(col)) for r in e.scored_rows]
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else float("nan")

    def bands(e):
        acc, n = {}, 0
        for r in e.scored_rows:
            d = json.loads(r.get("band_shares") or "{}")
            if d:
                n += 1
                for k, v in d.items():
                    acc[k] = acc.get(k, 0.0) + v
        return {k: round(v / n, 2) for k, v in acc.items()} if n else {}

    out_dir = REPO / "results/exp5/scores/b2"
    lines = ["# Exploratory: FQ (learned per-stop score inside FeRRy's plan) vs E3 (Chen et al.)",
             "",
             "Not pre-registered (added 8 Oct 2026). N = 6, one mule, 20 paired seeds per cell; "
             "time to tau = 0.71 on the simulated clock over complete pairs; difference = "
             "reference - FQ (positive: FQ faster); Holm across the eight comparisons below. "
             "FQ flies Study 5.5's gamma = 0.75 pick. The 5.3 scores are unchanged.",
             "", "## Comparisons", "",
             "| cell | reference | n pairs | reference mean | FQ mean | difference [95% CI] | "
             "p | Holm p | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in comps:
        verdict = c.favours if c.claim else "no claim"
        diff = (f"{c.mean_diff:.1f} [{c.ci_low:.1f}, {c.ci_high:.1f}]"
                if c.mean_diff is not None else c.note)
        lines.append(f"| {c.cell} | {c.reference} | {c.n_pairs} | "
                     f"{'' if c.ref_mean is None else f'{c.ref_mean:.1f}'} | "
                     f"{'' if c.variant_mean is None else f'{c.variant_mean:.1f}'} | {diff} | "
                     f"{'' if c.p_value is None else f'{c.p_value:.4f}'} | "
                     f"{'' if c.p_holm is None else f'{c.p_holm:.4f}'} | {verdict} |")
    lines += ["", "## Arms (descriptive)", "",
              "| cell | arm | trials ok | reach tau | mean time to tau | updates/round | "
              "mission (s) | transit (s) | dwell (s) | band shares |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for cell in CELLS:
        for arm in [VARIANT] + REFS:
            e = by[(cell, arm)]
            n = len(e.scored_rows)
            reached = sum(1 for r in e.scored_rows if SC._num(r.get(COL)) is not None)
            lines.append(
                f"| {cell} | {arm} | {n} | {reached / n:.2f} | {mean(e, COL):.1f} | "
                f"{mean(e, 'update_yield'):.2f} | {mean(e, 'sim_mission_duration_s_mean'):.0f} | "
                f"{mean(e, 'sim_transit_s_mean'):.0f} | {mean(e, 'sim_dwell_s_mean'):.0f} | "
                f"{bands(e) or '-'} |")
    (out_dir / "exploratory_fq_vs_e3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "exploratory_fq_vs_e3.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comps[0].row()))
        w.writeheader()
        for c in comps:
            w.writerow(c.row())
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
