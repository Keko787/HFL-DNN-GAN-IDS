"""Exploratory: the baselines head to head, FeRRy's arms left out.

    py -3.11 scripts/exp5/baseline_h2h.py

Added 15 Nov 2026; not pre-registered, and no flights: it reads the scored trial files
the campaign and the loss probes already wrote. A scenario is one cell, gathered across
the stage folders its baselines flew in (a cell's trials pair by trial index and seed,
whatever folder they are in). In each scenario the fastest baseline (the lowest mean
time to tau = 0.71 over its own reached trials) is compared with every other baseline:

- time to tau: the scorer's paired comparison (scoring.compare_pair), over the trials
  in which both reach tau;
- reach at tau: the probe scorer's exact McNemar test (loss_probes.compare_reach).

Holm runs within each scenario, one family per metric. A claim needs the CI to exclude 0
and Holm p < alpha ([score] in params.toml), as in the scorer. When a time claim favours
another arm over the one with the lowest mean (its mean is over fewer reached trials),
that arm becomes the fastest and the scenario is compared again. Output:
results/exp5/scores/exploratory/baselines_h2h.{md,csv}.
"""

from __future__ import annotations

import csv
import math
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import loss_probes as LP  # noqa: E402
import scoring as SC  # noqa: E402

OUT = REPO / "results/exp5/scores/exploratory"
RESULTS = REPO / "results"
TAU = 0.71
BASELINES = ("H1", "D1", "D2", "D3", "D4", "D4fedex", "D5", "E3")
#: The paper's names, with the runner's arm.
NAMES = {"H1": "Deadline-Heuristic (H1)", "D1": "MAX-AoI (D1)", "D2": "Oort (D2)",
         "D3": "Value-Aware FL (D3)", "D4": "FedEx (D4)", "D4fedex": "FedEx-Merge (D4fedex)",
         "D5": "FedCS (D5)", "E3": "Model-Aided RL (E3)"}
SCORED_SUFFIXES = ("_scored.csv", "_scored_comp.csv", "_scored_det.csv")

#: (group, scenario, [(folder under results/, cell), ...]); the first folder holding an
#: arm's scored file supplies it.
SCENARIOS: List[Tuple[str, str, List[Tuple[str, str]]]] = [
    ("Budget, N = 6, 1 mule", "knee 150 s", [("exp5/b1/s53", "n6k1_knee"), ("exp5/b2/s53x", "n6k1_knee")]),
    ("Budget, N = 6, 1 mule", "stress 75 s", [("exp5/b1/s53", "n6k1_stress"), ("exp5/b2/s53x", "n6k1_stress")]),
    ("Budget, N = 6, 1 mule", "60 s", [("exp5_probe_tight40/b2/s53x", "n6k1_relaxed")]),
    ("Budget, N = 6, 1 mule", "45 s", [("exp5_probe_tight30/b2/s53x", "n6k1_relaxed")]),
    ("Budget, N = 6, 1 mule", "300 s", [("exp5/b2/s53x", "n6k1_relaxed")]),
    ("Budget, N = 6, 1 mule", "1800 s", [("exp5_relaxed1800/b2/s53x", "n6k1_relaxed")]),
    ("Mules", "N = 6, 3 mules, knee", [("exp5/b1/s53", "n6k3_knee"), ("exp5/b2/s53x", "n6k3_knee")]),
    ("Mules", "N = 6, 3 mules, stress", [("exp5/b1/s53", "n6k3_stress"), ("exp5/b2/s53x", "n6k3_stress")]),
    ("Mules", "N = 6, 2 mules, knee", [("exp5/b2/s59x", "n6k2_knee")]),
    ("Mules", "N = 12, 2 mules, knee", [("exp5/b1/s511b", "n12k2_knee"), ("exp5/b2/s59x", "n12k2_knee")]),
    ("Mules", "N = 12, 3 mules, knee", [("exp5/b1/s511b", "n12k3_knee"), ("exp5/b2/s59x", "n12k3_knee")]),
    ("Mules", "N = 24, 2 mules, knee", [("exp5/b2/s59x", "n24k2_knee")]),
    ("Mules", "N = 24, 3 mules, knee", [("exp5/b2/s59x", "n24k3_knee")]),
    ("Scale, 1 mule", "N = 12, knee", [("exp5/b1/s59", "n12k1_knee"), ("exp5/b2/s59x", "n12k1_knee")]),
    ("Scale, 1 mule", "N = 12, stress", [("exp5/b2/s59x", "n12k1_stress"), ("exp5/b2/s57", "n12k1_stress")]),
    ("Scale, 1 mule", "N = 24, knee", [("exp5/b1/s59", "n24k1_knee"), ("exp5/b2/s59x", "n24k1_knee")]),
    ("Scale, 1 mule", "N = 24, stress", [("exp5/b2/s59x", "n24k1_stress")]),
    ("Radio, time-varying backhaul", "default 5 dB", [("exp5/b3/s515", "n6k1_knee_base")]),
    ("Radio, time-varying backhaul", "11 dB", [("exp5/b3/s515", "n6k1_knee_amp11")]),
    ("Radio, time-varying backhaul", "17 dB", [("exp5/b3/s515", "n6k1_knee_ampharsh")]),
    ("Radio, time-varying backhaul", "path loss 2.7", [("exp5/b3/s515", "n6k1_knee_npl")]),
    ("Radio, time-varying backhaul", "shadowing 8 dB", [("exp5/b3/s515", "n6k1_knee_sigma8")]),
    ("Radio, mission backhaul", "default 5 dB", [("exp5_p1mbh/b3/s515", "n6k1_knee_base")]),
    ("Radio, mission backhaul", "17 dB", [("exp5_p1mbh/b3/s515", "n6k1_knee_ampharsh")]),
    ("Radio, mission backhaul", "path loss 2.7", [("exp5_p1mbh/b3/s515", "n6k1_knee_npl")]),
    ("Payload, N = 6 knee", "2 MB", [("exp5_probe_2mb/b2/s53x", "n6k1_knee")]),
    ("Payload, N = 6 knee", "5 MB", [("exp5_probe_5mb/b2/s53x", "n6k1_knee")]),
    ("Payload, N = 6 knee", "10 MB", [("exp5_probe_10mb/b2/s53x", "n6k1_knee")]),
    ("Layout, N = 6 knee", "25 % beyond 60 m", [("exp5/b2/s54", "n6k1_knee_1mb_far25")]),
    ("Layout, N = 6 knee", "50 % beyond 60 m", [("exp5/b2/s54", "n6k1_knee_1mb_far50")]),
    ("Layout, N = 6 knee", "75 % beyond 60 m", [("exp5/b2/s54", "n6k1_knee_1mb_far75")]),
    ("Field, N = 12 knee", "±200 m, 1 MB", [("exp5_probe_w200_1mb/b1/s59", "n12k1_knee"),
                                             ("exp5_probe_w200_1mb/b2/s59x", "n12k1_knee")]),
    ("Field, N = 12 knee", "±300 m, 1 MB", [("exp5_probe_w300r/b1/s59", "n12k1_knee"),
                                             ("exp5_probe_w300r/b2/s59x", "n12k1_knee")]),
    ("Field, N = 12 knee", "±200 m, 10 MB", [("exp5_probe_w200_10mb/b2/s59x", "n12k1_knee")]),
    ("Field, N = 12 knee", "±300 m, 10 MB", [("exp5_probe_w300_10mb/b2/s59x", "n12k1_knee")]),
    ("Field", "N = 24, ±300 m, 10 MB, 600 s", [("exp5_probe_w300_n24/b2/s59x", "n24k1_knee")]),
    ("Training time, 10 missions", "spread", [("exp5_rd10/b3/s512", "n6k1_knee_1mb_spread")]),
    ("Training time, 10 missions", "stragglers", [("exp5_rd10/b3/s512", "n6k1_knee_1mb_stragglers")]),
    ("Data, N = 6 knee", "IID", [("exp5/b2/s513", "n6k1_knee_iid")]),
    ("Data, N = 6 knee", "skewed shard sizes", [("exp5/b2/s513", "n6k1_knee_quantity1")]),
    ("Data, N = 6 knee", "Dirichlet 1", [("exp5/b2/s513", "n6k1_knee_dirichlet1")]),
    ("Data, N = 6 knee", "Dirichlet 0.1", [("exp5/b2/s513", "n6k1_knee_dirichlet0.1")]),
]


def load_plan() -> SC.Plan:
    data = tomllib.loads((HERE / "params.toml").read_text(encoding="utf-8"))
    plan, _ = SC.plan_from(data)
    return plan


def entry(folder: str, cell: str, arm: str, study: str) -> Optional[SC.Entry]:
    """The arm's scored file in ``folder`` as a scorer entry, or None."""
    base = RESULTS / folder / f"{cell}__{arm}"
    for suffix in SCORED_SUFFIXES:
        scored = Path(str(base) + suffix)
        if scored.exists():
            e = SC.Entry(study, cell, arm, arm, Path(str(base) + ".csv"), scored, 10 ** 9)
            SC.load(e)
            return e if e.scored_rows else None
    return None


def arm_stats(e: SC.Entry) -> Dict[str, Any]:
    times = [SC._num(r.get(f"sim_s_to_tau{TAU:g}")) for r in e.scored_rows]
    reached = [t for t in times if t is not None]
    return {"trials": len(times), "reach": len(reached) / len(times),
            "mean": sum(reached) / len(reached) if reached else math.inf}


def compare(entries: Dict[str, SC.Entry], fastest: str, key: str, plan) -> List[SC.Comparison]:
    comps = []
    for arm, e in entries.items():
        if arm == fastest:
            continue
        for kind, fn in (("time", LP.compare_time), ("reach", LP.compare_reach)):
            c = fn(entries[fastest], e, TAU, f"h2h/{key}/{kind}", plan, SC)
            comps.append(c)
    SC.holm(comps, plan)
    return comps


def scenario(group: str, label: str, sources, plan) -> Optional[Dict[str, Any]]:
    key = f"{group}: {label}"
    entries: Dict[str, SC.Entry] = {}
    for folder, cell in sources:
        for arm in BASELINES:
            if arm not in entries:
                e = entry(folder, cell, arm, key)
                if e is not None:
                    entries[arm] = e
    if len(entries) < 2:
        return None
    stats = {a: arm_stats(e) for a, e in entries.items()}
    order = sorted(entries, key=lambda a: stats[a]["mean"])
    fastest = order[0]
    for _ in range(len(order)):
        comps = compare(entries, fastest, key, plan)
        ahead = [c for c in comps if c.metric.startswith("sim_s") and c.claim
                 and c.favours == c.variant]
        if not ahead:
            break
        fastest = max(ahead, key=lambda c: c.mean_diff).variant
    time = {c.variant: c for c in comps if c.metric.startswith("sim_s")}
    reach = {c.variant: c for c in comps if c.metric.startswith("reached")}
    slower = [a for a, c in time.items() if c.claim and c.favours == fastest]
    return {"group": group, "label": label, "key": key, "stats": stats, "order": order,
            "fastest": fastest, "slower": slower,
            "level": [a for a in order if a != fastest and a not in slower],
            "time": time, "reach": reach, "comps": comps}


def _f(x: Optional[float], places: int = 1) -> str:
    return "" if x is None else f"{x:.{places}f}"


def _diff(c: Optional[SC.Comparison], places: int = 1) -> str:
    if c is None or c.mean_diff is None:
        return ""
    return f"{c.mean_diff:.{places}f} [{c.ci_low:.{places}f}, {c.ci_high:.{places}f}]"


def _verdict(c: Optional[SC.Comparison]) -> str:
    if c is None or c.p_holm is None:
        return ""
    return NAMES.get(c.favours, c.favours) if c.claim else "no claim"


def report(results: List[Dict[str, Any]], plan) -> List[str]:
    tally = {a: dict(flown=0, fastest=0, level=0, slower=0, last=0) for a in BASELINES}
    for s in results:
        for a in s["stats"]:
            tally[a]["flown"] += 1
        tally[s["fastest"]]["fastest"] += 1
        for a in s["level"]:
            tally[a]["level"] += 1
        for a in s["slower"]:
            tally[a]["slower"] += 1
        tally[s["order"][-1]]["last"] += 1
    lines = [
        "# The baselines head to head",
        "",
        "Exploratory, not pre-registered (15 Nov 2026; `scripts/exp5/baseline_h2h.py`). No "
        "flights: the scored trial files of the campaign and the loss probes. FeRRy's arms "
        "are left out. In each scenario the fastest baseline (the lowest mean time to tau = "
        f"{TAU:g} over its own reached trials) is compared with every other baseline: time to "
        "tau by the scorer's paired comparison over the trials both reach, and reach at tau "
        "by exact McNemar, each with a 95% bootstrap CI. Holm within each scenario, one "
        f"family per metric. A claim needs the CI to exclude 0 and Holm p < {plan.alpha:g}. "
        "Difference = fastest − arm: time, negative when the fastest is faster; reach, "
        "positive when the fastest reaches tau more often. When a time claim favoured "
        "another arm over the one with the lowest mean, that arm is the fastest.",
        "",
        "## Each baseline",
        "",
        "Over the scenarios it flew in: fastest; level with the fastest (no time claim "
        "against it); significantly slower than the fastest; last by mean.",
        "",
        "| baseline | scenarios flown | fastest | level | significantly slower | last |",
        "|---|---|---|---|---|---|",
    ]
    for a in sorted(tally, key=lambda a: -(tally[a]["fastest"] + tally[a]["level"])
                    / max(1, tally[a]["flown"])):
        t = tally[a]
        lines.append(f"| {NAMES[a]} | {t['flown']} | {t['fastest']} | {t['level']} | "
                     f"{t['slower']} | {t['last']} |")
    lines += ["", "## The fastest baseline in each scenario", "",
              "| group | scenario | trials | fastest: mean time to tau (s), reach | "
              "significantly slower than it (fastest − arm, s) |", "|---|---|---|---|---|"]
    for s in results:
        st, f = s["stats"], s["fastest"]
        slow = ", ".join(f"{NAMES[a]} ({s['time'][a].mean_diff:.0f})" for a in s["slower"])
        lines.append(f"| {s['group']} | {s['label']} | {st[f]['trials']} | {NAMES[f]}: "
                     f"{st[f]['mean']:.0f}, {st[f]['reach']:.2f} | {slow or 'none'} |")
    for s in results:
        st = s["stats"]
        lines += ["", f"## {s['key']}", "",
                  f"Fastest: {NAMES[s['fastest']]}.", "",
                  "| baseline | trials | reach τ | mean time to τ (s) | time: fastest − arm "
                  "[95% CI] | pairs | Holm p | verdict | reach: fastest − arm [95% CI] | "
                  "discordant | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in [s["fastest"]] + [x for x in s["order"] if x != s["fastest"]]:
            t, r = s["time"].get(a), s["reach"].get(a)
            lines.append(
                f"| {NAMES[a]} | {st[a]['trials']} | {st[a]['reach']:.2f} | "
                f"{_f(st[a]['mean'])} | {_diff(t)} | {'' if t is None else t.n_pairs} | "
                f"{'' if t is None or t.p_holm is None else f'{t.p_holm:.4f}'} | {_verdict(t)} | "
                f"{_diff(r, 2)} | {'' if r is None else r.note.split('; ')[-1]} | "
                f"{'' if r is None or r.p_holm is None else f'{r.p_holm:.4f}'} | {_verdict(r)} |")
    lines += ["", "## Reading it", "",
              "- Means are over each arm's own reached trials, so an arm that reached tau less "
              "often can show a lower mean; the paired time test and the reach test are the "
              "check on that.",
              "- Cells of 20 trials and of 40 are not the same trials: FedEx's route flies "
              "the same at every budget, and its 264 s in the 20-trial budget cells and 232 s in "
              "the 40-trial ones come from different trial sets.",
              "- Holm runs within a scenario only, not across scenarios."]
    return lines


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    plan = load_plan()
    results = [r for r in (scenario(g, l, src, plan) for g, l, src in SCENARIOS) if r]
    OUT.mkdir(parents=True, exist_ok=True)
    lines = report(results, plan)
    (OUT / "baselines_h2h.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = []
    for s in results:
        for c in s["comps"]:
            rows.append({"group": s["group"], "scenario": s["label"], "fastest": s["fastest"],
                         **c.row()})
    with (OUT / "baselines_h2h.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines[:40]))
    print(f"\nwritten: {OUT / 'baselines_h2h.md'} ({len(results)} scenarios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
