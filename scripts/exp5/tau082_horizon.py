"""Exploratory: the training-time cells at the second tau, 0.82, on reach and at equal
flight time.

    py -3.11 scripts/exp5/tau082_horizon.py

Added 15 Nov 2026; not pre-registered, and no flights. P9 found two reach claims against
F at tau = 0.82, both in 5.12's training-time cells at 4 missions (the FedEx route on
spread, H1 with stragglers). P11 flew those cells to 10 missions and tested tau = 0.71
only. This reads both runs and, with F and then FX as the reference against H1, D3, D4
and D5:

- reach at 0.82 over the whole run: the probe scorer's exact McNemar test
  (loss_probes.compare_reach);
- reach at 0.82 by the horizon, the shortest flight of any arm in any trial of the cell,
  so every arm has flown to it in every trial (loss_probes.as_of, P11's horizon rule);
- the restricted mean time to 0.82 at the horizon, a miss counted as the horizon (the
  scorer's paired comparison).

A trial's flight is missions_completed x sim_mission_duration_s_mean from its scored row.
P11 measured it from the mule traces (first takeoff to last landing), which this checkout
does not hold; the two agree on P11's cells (F 925 and 896 s, H1 2,159 and 2,119 s;
horizons 436 and 427 s). Holm per run, reference and metric, across the run's two cells.
Output: results/exp5/scores/probes/P11_tau082.{md,csv}.
"""

from __future__ import annotations

import csv
import sys
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import loss_probes as LP  # noqa: E402
import scoring as SC  # noqa: E402

OUT = REPO / "results/exp5/scores/probes"
TAU = 0.82
ARMS = ("F", "FX", "H1", "D3", "D4", "D5")
REFERENCES = ("F", "FX")
NAMES = {"F": "F", "FX": "FX", "H1": "Deadline-Heuristic (H1)", "D3": "Value-Aware FL (D3)",
         "D4": "FedEx (D4)", "D5": "FedCS (D5)"}
RUNS = [  # (label, folder under results/, cells)
    ("10 missions (P11)", "exp5_rd10/b3/s512",
     ["n6k1_knee_1mb_spread", "n6k1_knee_1mb_stragglers"]),
    ("4 missions (5.12, as P9 read it)", "exp5/b3/s512",
     ["n6k1_knee_1mb_spread", "n6k1_knee_1mb_stragglers"]),
]


def load_plan() -> SC.Plan:
    plan, _ = SC.plan_from(tomllib.loads((HERE / "params.toml").read_text(encoding="utf-8")))
    return plan


def entry(folder: str, cell: str, arm: str) -> SC.Entry:
    base = REPO / "results" / folder / f"{cell}__{arm}"
    e = SC.Entry("tau082", cell, arm, arm, Path(str(base) + ".csv"),
                 Path(str(base) + "_scored_comp.csv"), 10 ** 9)
    SC.load(e)
    if not e.scored_rows:
        raise SystemExit(f"no scored rows in {e.scored}")
    return e


def flight_s(row: Dict[str, str]) -> float:
    return float(row["missions_completed"]) * float(row["sim_mission_duration_s_mean"])


def _diff(c: Optional[SC.Comparison], places: int = 1) -> str:
    if c is None or c.mean_diff is None:
        return ""
    return f"{c.mean_diff:.{places}f} [{c.ci_low:.{places}f}, {c.ci_high:.{places}f}]"


def _p(c: Optional[SC.Comparison]) -> str:
    return "" if c is None or c.p_holm is None else f"{c.p_holm:.4f}"


def _verdict(c: Optional[SC.Comparison]) -> str:
    if c is None or c.p_holm is None:
        return ""
    return NAMES.get(c.favours, c.favours) if c.claim else "no claim"


def _note(c: Optional[SC.Comparison]) -> str:
    return "" if c is None else c.note.split("; ")[-1]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    plan = load_plan()
    lines = [
        "# 5.12's training-time cells at tau = 0.82: reach, and reach at equal flight time",
        "",
        "Exploratory, not pre-registered (15 Nov 2026; `scripts/exp5/tau082_horizon.py`). No "
        "flights. P9 found two reach claims against F at tau = 0.82, both in these cells at 4 "
        "missions; P11 flew them to 10 missions and tested tau = 0.71 only. F, then FX, is the "
        "reference against each baseline: reach at 0.82 over the whole run (exact McNemar), "
        "reach at 0.82 by the horizon, and the restricted mean time to 0.82 at the horizon "
        "(a miss counted as the horizon; the scorer's paired comparison). The horizon is the "
        "shortest flight of any arm in any trial of the cell, P11's rule, so every arm has "
        "flown to it in every trial. A trial's flight is missions_completed × "
        "sim_mission_duration_s_mean, which reproduces P11's trace-based flights (F 925 and "
        "896 s, H1 2,159 and 2,119 s; horizons 436 and 427 s). Holm per run, reference and "
        f"metric, across the run's two cells; a claim needs the CI to exclude 0 and Holm p < "
        f"{plan.alpha:g}. Difference = reference − arm: reach, positive when the reference "
        "reaches 0.82 more often; time, negative when the reference is faster.",
    ]
    rows = []
    for label, folder, cells in RUNS:
        data = {c: {a: entry(folder, c, a) for a in ARMS} for c in cells}
        horizon = {c: min(flight_s(r) for e in data[c].values() for r in e.scored_rows)
                   for c in cells}
        seen = {c: {a: LP.as_of(e, horizon[c], SC, tau=TAU) for a, e in data[c].items()}
                for c in cells}
        comps: List[SC.Comparison] = []
        by: Dict[tuple, SC.Comparison] = {}
        for ref in REFERENCES:
            for c in cells:
                for arm in ARMS:
                    if arm in REFERENCES:
                        continue
                    fam = f"tau082/{label}/{ref}"
                    for kind, x in (
                            ("reach", LP.compare_reach(data[c][ref], data[c][arm], TAU,
                                                       f"{fam}/reach", plan, SC)),
                            ("horizon_reach", LP.compare_reach(seen[c][ref], seen[c][arm], TAU,
                                                               f"{fam}/horizon_reach", plan, SC)),
                            ("horizon_time", LP.compare_time(seen[c][ref], seen[c][arm], TAU,
                                                             f"{fam}/horizon_time", plan, SC))):
                        x.cell = c
                        comps.append(x)
                        by[(ref, c, arm, kind)] = x
        SC.holm(comps, plan)
        rows += [{"run": label, "horizon_s": round(horizon[c.cell], 1), **c.row()} for c in comps]
        for c in cells:
            lines += ["", f"## {label}: {c}", "",
                      f"The horizon is {horizon[c]:.0f} s.", "",
                      "| arm | trials | flight mean (s) | reach 0.82, whole run | reached 0.82 by "
                      "the horizon | restricted mean time to 0.82 (s) |",
                      "|---|---|---|---|---|---|"]
            for a in ARMS:
                e, s = data[c][a], seen[c][a]
                n = len(e.scored_rows)
                whole = sum(1 for r in e.scored_rows if SC._num(r.get(f"sim_s_to_tau{TAU:g}")) is not None)
                by_h = sum(1 for r in s.scored_rows if r[f"reached_tau{TAU:g}"] == 1.0)
                rmean = sum(r[f"sim_s_to_tau{TAU:g}"] for r in s.scored_rows) / n
                fmean = sum(flight_s(r) for r in e.scored_rows) / n
                lines.append(f"| {NAMES[a]} | {n} | {fmean:.0f} | {whole}/{n} | {by_h}/{n} | "
                             f"{rmean:.1f} |")
            for ref in REFERENCES:
                lines += ["", f"{ref} against each baseline:", "",
                          f"| arm | reach, whole run: {ref} − arm [95% CI] | discordant | Holm p | "
                          f"verdict | by the horizon: {ref} − arm [95% CI] | discordant | Holm p | "
                          f"verdict | restricted time: {ref} − arm [95% CI] | Holm p | verdict |",
                          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
                for a in ARMS:
                    if a in REFERENCES:
                        continue
                    w, h, t = (by[(ref, c, a, k)] for k in ("reach", "horizon_reach", "horizon_time"))
                    lines.append(f"| {NAMES[a]} | {_diff(w, 2)} | {_note(w)} | {_p(w)} | {_verdict(w)} | "
                                 f"{_diff(h, 2)} | {_note(h)} | {_p(h)} | {_verdict(h)} | "
                                 f"{_diff(t)} | {_p(t)} | {_verdict(t)} |")
    lines += ["", "## Reading it", "",
              "- Over the whole run the baselines reach 0.82 more often, and most of those "
              "differences are claims, but the baselines fly 1.8 to 2.6 times as long: F's and "
              "FX's missions are shorter, and the run is capped by missions, not by time. P11 "
              "found the same cap behind F's reach at 0.71.",
              "- By the horizon, when every arm has flown as long, F and FX reach 0.82 at "
              "least as often as every baseline at 10 missions. FX's lead is a claim against "
              "H1 and D3 in both cells and against D4 with stragglers, and its restricted time "
              "in six of eight comparisons; F's lead is a claim nowhere (Holm 0.06 to 0.43).",
              "- At 4 missions the horizon (209 and 189 s) is too short for almost any arm to "
              "reach 0.82, so P9's two claims there show no loss at equal time, and no gain.",
              "- At 10 missions F and FX merge fewer updates per mission (P11.md), since the "
              "mule can arrive before training finishes."]
    (OUT / "P11_tau082.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / "P11_tau082.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))
    print(f"\nwritten: {OUT / 'P11_tau082.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
