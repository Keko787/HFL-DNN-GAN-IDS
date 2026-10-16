"""Exploratory: energy cost and the time budget in Study 5.3's N = 6 cells.

    py -3.11 scripts/exp5/energy_budget.py

Added 16 Oct 2026, after the deadline; not pre-registered. Two questions:

1. Energy. How much simulated drone energy does each arm spend to reach tau = 0.71,
   and per merged update? Cells n6k1_knee, n6k1_stress and n6k3_knee; FeRRy's F, FX
   and FQ (gamma = 0.75) against the seven baselines (H1, D1-D5, E3), paired by seed,
   Holm across each FeRRy arm's seven comparisons per cell and metric. Energy comes
   from the mule traces (mission_completed: energy_j), so the kept traces must be on
   disk (`launch.py unpack` for an archived stage).
   - energy to tau: the energy of the missions that started before the model first
     reached tau (the mission in flight when it does counts in full; with three mules,
     every mule's missions);
   - energy per merged update: a trial's total energy over its merged updates;
   - energy per mission: the scorer's sim_energy_j_mean.

2. Budget. Time to tau at the stress (75 s), knee (150 s), relaxed (300 s, the
   pre-registered 2x knee) and 1800 s (12x knee) budgets, N = 6, one mule. The relaxed
   cell is the run committed in 5ca8f3de,

     launch.py run batch2 --study s53x --trials 20
       --set "s53x.arms=['F','FX','H1','D1','D2','D3','D4','D5','E3','FQ-g75']"
       --set "s53x.K=[1]" --set "s53x.budgets=['relaxed']"
       --set s53x.d4_faithful=true --set s53x.h0=false --set rl.keep_learned=true

   and 1800 s is the same command with --out-root results/exp5_relaxed1800 and
   --set batch2.relaxed_factor=12. Each relaxed job is scored with 5.3's scorer
   arguments (its *_scored.csv); F is compared with each arm, Holm across each cell's
   ten. Stress and knee are read as scored; their pre-registered comparisons stay in
   results/exp5/scores/b2/s53x.*, which this script leaves as they are.

It writes results/exp5/scores/b2/exploratory_{energy,budget}.{md,csv}.
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
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

COL = "sim_s_to_tau0.71"
FERRY = ["F", "FX", "FQ-g75"]
BASELINES = ["H1", "D1", "D2", "D3", "D4", "D5", "E3"]
NAMES = {"F": "F", "FX": "FX", "FQ-g75": "FQ", "H1": "H1", "D1": "MAX-AoI", "D2": "Oort",
         "D3": "Whittle (Cui)", "D4": "FedEx route", "D5": "FedCS", "E3": "DQN (Chen)",
         "D4fedex": "FedEx route + merge"}
ENERGY_CELLS = ["n6k1_knee", "n6k1_stress", "n6k3_knee"]
ENERGY_COLS = {"energy_to_tau_j": "energy to tau", "energy_per_update_j": "energy per merged update",
               "sim_energy_j_mean": "energy per mission"}
ARM_SET = "s53x.arms=['F','FX','H1','D1','D2','D3','D4','D5','E3','FQ-g75']"
BASE_SETS = [ARM_SET, "s53x.d4_faithful=true", "s53x.h0=false", "rl.keep_learned=true"]
RELAXED_SETS = BASE_SETS + ["s53x.K=[1]", "s53x.budgets=['relaxed']"]
RELAXED = {"300 s": ("results/exp5", []),
           "1800 s": ("results/exp5_relaxed1800", ["batch2.relaxed_factor=12"])}
OUT = REPO / "results/exp5/scores/b2"


def _launcher():
    spec = importlib.util.spec_from_file_location("exp5_launch", HERE / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp5_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- energy from traces
def missions_from_trace(trace_dir: Path) -> List[Dict[str, float]]:
    """A trial's completed missions (every mule), in start order."""
    out = []
    for p in sorted(Path(trace_dir).glob("mule-*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("event") != "mission_completed":
                    continue
                out.append({"start": float(r["sim_start_s"]), "end": float(r["sim_end_s"]),
                            "energy_j": float(r.get("energy_j") or 0.0),
                            "updates": int(r.get("pass_1_merged_updates") or 0)})
    return sorted(out, key=lambda m: m["start"])


def energy_to_tau(missions: List[Dict[str, float]], t_tau: Optional[float]) -> Optional[float]:
    """Energy of the missions that started before tau was first reached (None if never)."""
    if t_tau is None or not missions:
        return None
    t0 = missions[0]["start"]
    return sum(m["energy_j"] for m in missions if m["start"] - t0 < t_tau)


def energy_per_update(missions: List[Dict[str, float]]) -> Optional[float]:
    updates = sum(m["updates"] for m in missions)
    return sum(m["energy_j"] for m in missions) / updates if updates else None


def trace_dir(row: Dict[str, str], repo: Path = REPO) -> Optional[Path]:
    root = repo / str(row.get("trace_root", "")).replace("\\", "/")
    if not row.get("trace_root") or not root.is_dir():
        return None
    hits = sorted(root.glob(f"*__t{row['trial_index']}__s{row['seed']}"))
    return hits[0] if hits else None


def attach_energy(entry, SC, repo: Path = REPO) -> int:
    """Add energy_to_tau_j / energy_per_update_j (and totals) to an entry's scored rows.
    Returns how many rows had no trace."""
    missing = 0
    for r in entry.scored_rows:
        d = trace_dir(r, repo)
        ms = missions_from_trace(d) if d else []
        if not ms:
            missing += 1
        e2t = energy_to_tau(ms, SC._num(r.get(COL)))
        epu = energy_per_update(ms)
        r["energy_to_tau_j"] = "" if e2t is None else repr(e2t)
        r["energy_per_update_j"] = "" if epu is None else repr(epu)
        r["_energy_total_j"] = sum(m["energy_j"] for m in ms)
        r["_updates_total"] = sum(m["updates"] for m in ms)
    return missing


# --------------------------------------------------------------------- launcher entries
def entries(L, SC, sets: List[str], out_root: str, *, score: bool = False):
    s, _ = L.load_settings(L.PARAMS)
    ns = argparse.Namespace(trials=20, set=sets, dataset=None,
                            **{o: None for o, _, _ in L.LEVERS})
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build("batch2", s, ["s53x"], out_root)
    runs, ents, _, plan = L.score_plan("batch2", s, jobs)
    if score:
        for j in runs:
            if not L.done(j):
                print(f"scoring {j.out}")
                subprocess.run([sys.executable] + j.args, cwd=REPO, check=True)
    by = {}
    for e in ents["s53x"]:
        if Path(e.scored).exists():
            SC.load(e)
            by[(e.cell, e.variant)] = e
    return by, plan.specs.get("s53x"), plan


def _vals(e, col: str) -> List[float]:
    return [v for v in (_f(r.get(col)) for r in e.scored_rows) if v is not None]


def _f(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def lower_count(ref, var, col: str) -> Tuple[int, int]:
    """Trials (paired by index and seed) in which var's value is below ref's."""
    a = {int(r["trial_index"]): r for r in ref.scored_rows}
    n = lower = 0
    for r in var.scored_rows:
        q = a.get(int(r["trial_index"]))
        if q is None or q.get("seed") != r.get("seed"):
            continue
        x, y = _f(q.get(col)), _f(r.get(col))
        if x is None or y is None:
            continue
        n += 1
        lower += y < x
    return lower, n


def _cmp_line(c, unit: float = 1.0, nd: int = 1) -> str:
    verdict = c.favours if c.claim else "no claim"
    if c.mean_diff is None:
        return f"| {NAMES.get(c.reference, c.reference)} | {NAMES.get(c.variant, c.variant)} | {c.n_pairs} | | | {c.note} | | | |"
    return (f"| {NAMES.get(c.reference, c.reference)} | {NAMES.get(c.variant, c.variant)} | {c.n_pairs} | "
            f"{c.ref_mean / unit:.{nd}f} | {c.variant_mean / unit:.{nd}f} | "
            f"{c.mean_diff / unit:.{nd}f} [{c.ci_low / unit:.{nd}f}, {c.ci_high / unit:.{nd}f}] | "
            f"{c.p_value:.4f} | {c.p_holm:.4f} | {NAMES.get(verdict, verdict)} |")


def _write_csv(path: Path, comps) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comps[0].row()))
        w.writeheader()
        for c in comps:
            w.writerow(c.row())


# --------------------------------------------------------------------- the two reports
def energy_report(L, SC) -> List[str]:
    by, spec, plan = entries(L, SC, BASE_SETS, "results/exp5")
    missing = {k: attach_energy(e, SC) for k, e in by.items() if k[0] in ENERGY_CELLS}
    comps = []
    for cell in ENERGY_CELLS:
        for col in ENERGY_COLS:
            for fa in FERRY:
                if (cell, fa) not in by:
                    continue
                for b in BASELINES:
                    if (cell, b) in by:
                        c = SC.compare_pair(by[(cell, fa)], by[(cell, b)], col, spec, plan)
                        c.study = f"energy/{cell}/{col}/{fa}"   # Holm: one FeRRy arm's seven
                        comps.append(c)
            if (cell, "FQ-g75") in by:
                c = SC.compare_pair(by[(cell, "FX")], by[(cell, "FQ-g75")], col, spec, plan)
                c.study = f"energy/{cell}/fq_vs_fx"                # Holm: the three metrics
                comps.append(c)
    SC.holm(comps, plan)

    lines = ["# Exploratory: energy cost of FeRRy (F, FX, FQ) against the seven baselines",
             "",
             "Not pre-registered (16 Oct 2026). Study 5.3's N = 6 cells, 20 paired seeds. "
             "Simulated rotary-wing energy from the mule traces. Energy to tau = the missions "
             "started before the model first reached tau = 0.71; energy per merged update = a "
             "trial's energy over its merged updates; energy per mission = sim_energy_j_mean. "
             "Difference = FeRRy arm - baseline (negative: FeRRy uses less). Holm across each "
             "FeRRy arm's seven comparisons per cell and metric (FQ vs FX: across its three "
             "metrics). A claim needs the bootstrap CI to exclude 0 and Holm p < 0.05.", ""]
    if any(missing.values()):
        lines += ["Rows without traces: " + ", ".join(f"{c}/{v} {n}" for (c, v), n in missing.items() if n), ""]
    lines += ["## Arms (descriptive)", "",
              "| cell | arm | trials | energy per mission (kJ) | mean power (W) | transit (s) | dwell (s) | "
              "energy per merged update (kJ, all energy / all updates) | energy to tau (kJ) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for cell in ENERGY_CELLS:
        for arm in FERRY + BASELINES:
            e = by.get((cell, arm))
            if e is None:
                continue
            rows = e.scored_rows
            power = _mean([_f(r["sim_energy_j_mean"]) / _f(r["sim_mission_duration_s_mean"])
                           for r in rows if _f(r.get("sim_mission_duration_s_mean"))])
            etot = sum(r["_energy_total_j"] for r in rows)
            utot = sum(r["_updates_total"] for r in rows)
            lines.append(
                f"| {cell} | {NAMES[arm]} | {len(rows)} | {_mean(_vals(e, 'sim_energy_j_mean')) / 1000:.2f} | "
                f"{power:.0f} | {_mean(_vals(e, 'sim_transit_s_mean')):.0f} | {_mean(_vals(e, 'sim_dwell_s_mean')):.0f} | "
                f"{(etot / utot / 1000) if utot else float('nan'):.2f} | {_mean(_vals(e, 'energy_to_tau_j')) / 1000:.1f} |")
    for cell in ENERGY_CELLS:
        lines += ["", f"## {cell}", ""]
        for col, label in ENERGY_COLS.items():
            fam = [c for c in comps if c.cell == cell and c.metric == col and not c.study.endswith("fq_vs_fx")]
            if not fam:
                continue
            claims = {fa: sum(c.claim for c in fam if c.reference == fa) for fa in FERRY}
            lines += [f"### {label} (kJ): claims " + ", ".join(f"{NAMES[k]} {v} of 7" for k, v in claims.items()
                                                                if any(c.reference == k for c in fam)), "",
                      "| FeRRy arm | baseline | n pairs | FeRRy mean | baseline mean | difference [95% CI] | p | Holm p | verdict |",
                      "|---|---|---|---|---|---|---|---|---|"]
            lines += [_cmp_line(c, 1000.0, 2) for c in fam]
            lines.append("")
        fq = [c for c in comps if c.cell == cell and c.study.endswith("fq_vs_fx")]
        if fq:
            lines += ["### FQ against FX (kJ; difference = FX - FQ, negative: FQ uses more)", "",
                      "| reference | variant | n pairs | FX mean | FQ mean | difference [95% CI] | p | Holm p | verdict | FQ lower in |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for c in fq:
                low, n = lower_count(by[(cell, "FX")], by[(cell, "FQ-g75")], c.metric)
                lines.append(_cmp_line(c, 1000.0, 2) + f" {low} of {n} ({ENERGY_COLS[c.metric]}) |")
            lines.append("")
    (OUT / "exploratory_energy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / "exploratory_energy.csv", comps)
    return lines


def budget_report(L, SC) -> List[str]:
    main, spec, plan = entries(L, SC, BASE_SETS, "results/exp5")
    cells = {"75 s (stress)": main, "150 s (knee)": main}
    keys = {"75 s (stress)": "n6k1_stress", "150 s (knee)": "n6k1_knee"}
    comps = []
    relaxed = {}
    for label, (root, extra) in RELAXED.items():
        by, rspec, rplan = entries(L, SC, RELAXED_SETS + extra, root, score=True)
        relaxed[label] = by
        cells[f"{label} (relaxed)"] = by
        keys[f"{label} (relaxed)"] = "n6k1_relaxed"
        for v in ["FX", "FQ-g75"] + BASELINES + ["D4fedex"]:
            if ("n6k1_relaxed", "F") in by and ("n6k1_relaxed", v) in by:
                c = SC.compare_pair(by[("n6k1_relaxed", "F")], by[("n6k1_relaxed", v)], COL, rspec, rplan)
                c.study = f"budget/{label}"
                c.cell = f"n6k1_relaxed_{label.split()[0]}s"
                comps.append(c)
    SC.holm(comps, plan)

    lines = ["# Exploratory: time to tau as the mission budget grows (N = 6, one mule)", "",
             "Not pre-registered as a sweep (16 Oct 2026). Stress and knee as scored for 5.3 "
             "(their pre-registered comparisons are in s53x.md); relaxed = 300 s (5.3's third "
             "budget, 2x knee) and 1800 s (12x knee), flown on 16 Oct on the same 20 seeds. "
             "Mean time to tau = 0.71 over the trials that reached it (simulated seconds).", "",
             "## Time to tau by budget", "",
             "| arm | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
    for arm in FERRY + BASELINES + ["D4fedex"]:
        cols = []
        for label, by in cells.items():
            e = by.get((keys[label], arm))
            if e is None:
                cols.append("")
                continue
            xs = _vals(e, COL)
            cols.append(f"{_mean(xs):.1f} ({len(xs)}/{len(e.scored_rows)})")
        lines.append(f"| {NAMES[arm]} | " + " | ".join(cols) + " |")
    lines += ["", "## F against each arm at the relaxed budgets", "",
              "Difference = F - arm (negative: F faster); Holm across each budget's ten.", ""]
    for label in RELAXED:
        fam = [c for c in comps if c.study == f"budget/{label}"]
        lines += [f"### {label}: F faster than {sum(c.claim and c.favours == 'F' for c in fam if c.variant in BASELINES)} "
                  f"of {sum(c.variant in BASELINES for c in fam)} baselines", "",
                  "| reference | arm | n pairs | F mean | arm mean | difference [95% CI] | p | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        lines += [_cmp_line(c) for c in fam] + [""]
    lines += ["## Same seeds, same result?", "",
              "Trials whose time to tau is identical between two budgets (same seeds per cell and trial).", "",
              "| arm | 150 s = 300 s | 300 s = 1800 s |", "|---|---|---|"]
    for arm in FERRY + BASELINES + ["D4fedex"]:
        k, r3, r18 = (main.get(("n6k1_knee", arm)), relaxed["300 s"].get(("n6k1_relaxed", arm)),
                      relaxed["1800 s"].get(("n6k1_relaxed", arm)))
        lines.append(f"| {NAMES[arm]} | {_same(k, r3)} | {_same(r3, r18)} |")
    lines.append("")
    (OUT / "exploratory_budget.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / "exploratory_budget.csv", comps)
    return lines


def _same(a, b) -> str:
    if a is None or b is None:
        return ""
    x = {(r["trial_index"], r["seed"]): r.get(COL) for r in a.scored_rows}
    pairs = [(x.get((r["trial_index"], r["seed"])), r.get(COL)) for r in b.scored_rows]
    pairs = [p for p in pairs if p[0] is not None]
    return f"{sum(p == q for p, q in pairs)} of {len(pairs)}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    L = _launcher()
    SC = L._scoring()
    print("\n".join(energy_report(L, SC)))
    print("\n".join(budget_report(L, SC)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
