"""DDIL1N: Deadline-Heuristic and FedEx pinned to narrow under step 1's contact model, beside
step 1's FeRRy-Plan, FeRRy-Adapt and the same baselines on wide (read-only).

Every baseline in step 1 flew the wide band, while F and FX choose the band (narrow, at N = 6),
so step 1's lead over the baselines mixes the band with the plan. P10 separated the two on the
recorded physics; this does the same under step 1's model, in its default and 17 dB cells, on
step 1's seeds (they follow the cell and trial, not the arm). Three families, each measure in a
Holm family of its own (time to tau over the trials both reach, reach at tau, and DDIL build
D0's hover, updates per mission and time with a miss at the trial's end):

- F against each baseline on narrow: does the plan still beat the baselines on F's band?
- FX against each baseline on narrow.
- each baseline on narrow against itself on wide: what the band gives the baseline.

Usage::

    py -3.11 scripts/exp5/ddil_score_probes.py DDIL1N    # score the flight first
    py -3.11 scripts/exp5/ddil_narrow_baselines.py       # writes OUT_MD and OUT_CSV

Added 17 Nov 2026 with the DDIL1N flight.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import loss_probes as LP  # noqa: E402

STEP1 = REPO / "results/exp5_ddil1/b3/s515"
NARROW = REPO / "results/exp5_ddil1_bnarrow/b3/s515"
OUT_MD = REPO / "results/exp5/scores/probes/DDIL1N.md"
OUT_CSV = REPO / "results/exp5/scores/probes/DDIL1N.csv"
CELLS = (("n6k1_knee_d1base", "default channel"), ("n6k1_knee_d1ampharsh", "17 dB"))
BASELINES = ("H1", "D4")
TRIALS = 20
NAMES = {"F": "FeRRy-Plan", "FX": "FeRRy-Adapt", "H1": "Deadline-Heuristic", "D4": "FedEx"}


def _entry(SC, root: Path, cell: str, arm: str, variant: str):
    e = SC.Entry("s515", cell, variant, arm, root / f"{cell}__{arm}.csv",
                 root / f"{cell}__{arm}_scored_ddil.csv", TRIALS)
    SC.load(e)
    return e


def _name(variant: str) -> str:
    arm, _, band = variant.partition("@")
    return NAMES.get(arm, arm) + (f" on {band}" if band else "")


def _mean(entry, col: str, SC):
    vals = [SC._num(r.get(col)) for r in entry.scored_rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    L = LP._launcher()
    SC = L._scoring()
    s, _ = L.load_settings(L.PARAMS)
    plan, problems = SC.plan_from(s.data)
    if problems:
        sys.exit("params.toml [score]: " + "; ".join(problems))
    tau = LP.TAU
    comps: List[Any] = []
    tables: Dict[str, Dict[str, Any]] = {}
    for cell, _ in CELLS:
        arms = {"F": _entry(SC, STEP1, cell, "F", "F"), "FX": _entry(SC, STEP1, cell, "FX", "FX")}
        for b in BASELINES:
            arms[f"{b}@wide"] = _entry(SC, STEP1, cell, b, f"{b}@wide")
            arms[f"{b}@narrow"] = _entry(SC, NARROW, cell, b, f"{b}@narrow")
        missing = [v for v, e in arms.items() if len(e.scored_rows) < TRIALS]
        if missing:
            sys.exit(f"{cell}: not scored in full: {missing} (run ddil_score_probes.py DDIL1N)")
        tables[cell] = arms
        todo = ([(arms["F"], arms[f"{b}@narrow"], "F-vs-narrow") for b in BASELINES]
                + [(arms["FX"], arms[f"{b}@narrow"], "FX-vs-narrow") for b in BASELINES]
                + [(arms[f"{b}@wide"], arms[f"{b}@narrow"], "narrow-vs-wide") for b in BASELINES])
        for ref, var, family in todo:
            for c in LP._compare_all(ref, var, tau, f"DDIL1N/{family}", plan, SC, True):
                c.cell = cell
                comps.append(c)
    SC.holm(comps, plan)

    cols = (("reached_tau", "reach τ", 2), ("sim_s_to_tau", "time to τ (s)", 0),
            ("sim_s_to_tau_or_end", "time to τ, miss at the end (s)", 0),
            ("sim_dwell_s_mean", "hover per mission (s)", 0),
            ("update_yield", "updates per mission", 2))
    col = {"reached_tau": f"reached_tau{tau:g}", "sim_s_to_tau": f"sim_s_to_tau{tau:g}",
           "sim_s_to_tau_or_end": f"sim_s_to_tau{tau:g}_or_end"}
    lines = ["# DDIL1N: the baselines pinned to narrow under step 1's contact model", "",
             "Exploratory, not pre-registered. Deadline-Heuristic and FedEx pinned to the narrow "
             "band (campaign.baseline_contact_band), beside step 1's FeRRy-Plan, FeRRy-Adapt and "
             f"the same baselines on wide, N = 6, knee budget, {TRIALS} trials on step 1's seeds, "
             f"τ = {tau:g}. Paired by trial and seed; each measure in a Holm family of its own per "
             "comparison group. Difference = the first arm − the second.", ""]
    for cell, title in CELLS:
        lines += [f"## {title} ({cell})", "",
                  "| arm | " + " | ".join(c[1] for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for v, e in tables[cell].items():
            vals = []
            for key, _, nd in cols:
                m = _mean(e, col.get(key, key), SC)
                vals.append("" if m is None else f"{m:.{nd}f}")
            lines.append(f"| {_name(v)} | " + " | ".join(vals) + " |")
        lines += ["", "| first | second | measure | first − second [95% CI] | pairs | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|"]
        for c in comps:
            if c.cell != cell:
                continue
            kind = LP.kind_of(c.metric)
            nd = 2 if kind in ("reach", "updates") else 1
            lines.append(f"| {_name(c.reference)} | {_name(c.variant)} | {LP.KIND_LABELS[kind]} | "
                         f"{LP._diff(c, nd)} | {c.n_pairs} | {LP._holm(c)} | "
                         f"{_name(c.favours) if c.claim else 'no claim'} |")
        lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LP._write_csv(OUT_CSV, [c.row() for c in comps])
    print(f"wrote {OUT_MD.relative_to(REPO)} and {OUT_CSV.relative_to(REPO)}: {len(comps)} comparisons, "
          f"{sum(1 for c in comps if c.claim)} claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
