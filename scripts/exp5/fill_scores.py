"""The paper tables' fill cells: arms a study never flew, scored outside its Holm family.

`launch.py score` folds every listed cell into its study's family, which would change the
pre-registered results, so these flights never go through it. Each cell is scored next to
its CSV and compared with F on the same seeds; the p-values are raw, and no claim is made
from them. The writes go to results/exp5/scores/fills/.

    py -3.11 scripts/exp5/fill_scores.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "exp5"))
OUT = REPO / "results" / "exp5" / "scores" / "fills"
ROOT = "results/exp5"
TIME = "sim_s_to_tau"

FQ_SETS = ["s53x.arms=['FQ-g75']", "s53x.K=[1,3]",
           "s53x.budgets=['knee','stress','relaxed']",
           "s53x.d4_faithful=false", "s53x.h0=false", "rl.keep_learned=true"]
EIGHT = "['F','H1','D1','D2','D3','D4','D5','E3']"


@dataclass
class Fill:
    """One flight's cells: its table, how it was built, and who it is compared with."""

    table: str
    stage: str
    study: str
    sets: List[str]
    arms: Sequence[str]
    ref_of: Callable[[str], Tuple[str, str]]     # cell -> (stage, study) holding F
    reference: str = "F"
    ref_sets: Tuple[str, ...] = ()


def same(stage: str, study: str) -> Callable[[str], Tuple[str, str]]:
    return lambda cell: (stage, study)


def scale_ref(cell: str) -> Tuple[str, str]:
    return ("batch1", "s59") if cell.endswith("knee") else ("batch2", "s59x")


FILLS = [
    Fill("mules", "batch2", "s53x", FQ_SETS, ["FQ-g75"], same("batch2", "s53x")),
    Fill("scale", "batch2", "s55", ["s55.N=24", "s55.n_trials=20"], ["FQ-g75"], scale_ref),
    Fill("robustness", "batch2", "s513",
         ["s513.route_arms=['D4','D5']", "s513.partitions=[['dirichlet',0.1]]"], ["D4", "D5"],
         same("batch2", "s513")),
    Fill("robustness", "batch3", "s512", ["s512.arms=['D3']", "s512.payloads=[1000000]"], ["D3"],
         same("batch3", "s512")),
    Fill("robustness", "batch3", "s515",
         [f"s515.arms={EIGHT}", "s515.amps_db=[11.0]", "s515.mission_backhaul_arms=[]"],
         ["D3", "D4", "D5"], same("batch3", "s515"),
         ref_sets=(f"s515.arms={EIGHT}", "s515.amps_db=[11.0]",
                   "s515.mission_backhaul_arms=[]")),
    Fill("pins", "batch2", "s54",
         ["s54.payloads=[1000000]", "s54.narrow_range_ratios=[]", "s54.far_shares=[]",
          "s54.o1=false", "s54.n_trials=40"], ["FBpnarrow", "FBpmedium", "FBpwide"],
         same("batch2", "s54"),
         ref_sets=("s54.payloads=[1000000]", "s54.narrow_range_ratios=[]", "s54.far_shares=[]",
                   "s54.o1=false", "s54.n_trials=40")),
]
FERRYSIM = [("jit-n6-150", 6), ("jit-n12-180", 12)]


def _module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def entries_of(L, stage: str, study: str, sets: Sequence[str], whole: bool):
    """(runs, the study's entries, plan) as `launch.py score` would build them. `whole`
    builds the stage, so a cell the study shares reads the CSV its primary study flew."""
    s, _ = L.load_settings(L.PARAMS)
    ns = argparse.Namespace(trials=None, set=list(sets), dataset=None,
                            **{o: None for o, _, _ in L.LEVERS})
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build(stage, s, None if whole else [study], ROOT)
    runs, entries, _, plan = L.score_plan(stage, s, jobs)
    return runs, entries.get(study, []), plan


def scored_rows_of(entries, runs, L) -> None:
    wanted = {str(e.scored) for e in entries}
    for j in runs:
        if str(REPO / j.out) in wanted and not L.done(j):
            print(f"scoring {j.out}", flush=True)
            subprocess.run([sys.executable] + j.args, cwd=REPO, check=True)


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def row_of(fill: Fill, SC, plan, ref, var, col: str) -> Dict[str, Any]:
    spec = plan.specs[fill.study]
    c = SC.compare_pair(ref, var, col, spec, plan)
    both = [(e, [v for _, v in SC._values(e, col).values() if v is not None]) for e in (ref, var)]
    reach = [_mean([v for _, v in SC._values(e, SC.column("reached_tau", plan.tau)).values()
                    if v is not None]) for e in (ref, var)]
    return {"table": fill.table, "study": fill.study, "cell": var.cell, "variant": var.variant,
            "reference": ref.variant, "metric": col, "trials": var.trials,
            "rows": len(var.scored_rows), "ref_rows": len(ref.scored_rows),
            "n_pairs": c.n_pairs, "pairing_errors": c.pairing_errors,
            "ref_mean": _mean(both[0][1]), "variant_mean": _mean(both[1][1]),
            "reach_ref": reach[0], "reach_variant": reach[1],
            "paired_diff": c.mean_diff, "ci_low": c.ci_low, "ci_high": c.ci_high,
            "p_raw": c.p_value, "note": c.note}


def _cell(x: Any, places: int = 3) -> str:
    return "" if x is None else (f"{x:.{places}f}" if isinstance(x, float) else str(x))


def main() -> int:
    L = _module("exp5_launch", "scripts/exp5/launch.py")
    SC = L._scoring()
    rows: List[Dict[str, Any]] = []
    for fill in FILLS:
        runs, mine, plan = entries_of(L, fill.stage, fill.study, fill.sets, whole=False)
        mine = [e for e in mine if e.arm in fill.arms or e.variant in fill.arms]
        scored_rows_of(mine, runs, L)
        spec = plan.specs[fill.study]
        cols = [TIME] if spec.metric == TIME else [spec.metric, TIME]
        for e in mine:
            SC.load(e)
            stage, study = fill.ref_of(e.cell)
            _, others, ref_plan = entries_of(L, stage, study, list(fill.ref_sets),
                                             whole=not fill.ref_sets)
            ref = next((o for o in others if o.cell == e.cell and o.variant == fill.reference),
                       None)
            if ref is None:
                raise SystemExit(f"no {fill.reference} for {e.cell} in {study}")
            # The reference is read at the variant's seeds, however many its study flew.
            ref.trials = e.trials
            SC.load(ref)
            for col in cols:
                rows.append(row_of(fill, SC, plan, ref, e, SC.column(col, plan.tau)))
    sim = []
    for cell, n in FERRYSIM:
        rep = json.loads((REPO / ROOT / "b3" / "s511c" / f"{cell}.json")
                         .read_text(encoding="utf-8"))
        for r in rep["table"]:
            sim.append({"cell": cell, "N": n, "policy": r["policy"], "budget_s": r["budget_s"],
                        "served_share_mean": r["served_share_mean"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "fills.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "ferrysim.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sim[0]))
        w.writeheader()
        w.writerows(sim)
    lines = ["# Fill cells, outside the studies' Holm families", "",
             "Each cell against F on the same seeds; p is raw and no claim is made from it. "
             "Means take the trials that reached tau; the paired difference (F - variant) "
             "takes the seeds both reached.", ""]
    head = ["table", "cell", "variant", "metric", "trials", "n_pairs", "ref_mean",
            "variant_mean", "reach_ref", "reach_variant", "paired_diff", "ci_low", "ci_high",
            "p_raw"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(_cell(r[k]) for k in head) + " |" for r in rows]
    lines += ["", "## FerrySim", "", "| N | budget | policy | served share |", "|---|---|---|---|"]
    lines += [f"| {r['N']} | {r['budget_s']:.0f} | {r['policy']} | {r['served_share_mean']:.3f} |"
              for r in sim]
    (OUT / "fills.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows, {len(sim)} FerrySim rows in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
