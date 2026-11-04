"""Exploratory: the loss probes, each scored against FeRRy's F, and P9, the reach test
on every recorded comparison.

    py -3.11 scripts/exp5/loss_probes.py p9               # the recorded cells; no flights
    py -3.11 scripts/exp5/loss_probes.py P1 P4 [--score]  # a probe's cells, once flown
    py -3.11 scripts/exp5/loss_probes.py all [--score]

Added 3 Nov 2026; not pre-registered. The probes, their cells and the commands that fly
them are in DeveloperDocs/Experiment_5_Loss_Probes_Plan.md. Each probe is its own
family. F is the reference, and every other arm in a cell is compared with it on

- time to tau: the scorer's paired comparison (scoring.compare_pair), over the trials
  in which both arms reach tau;
- reach at tau: the exact McNemar test on the paired reached / not-reached outcomes,
  with a percentile-bootstrap CI on the paired difference in reach share.

Holm runs across each probe's comparisons, one family per metric, with FeRRy's other
arms (FX, FQ) against F in families of their own. A claim needs the CI to exclude 0 and
Holm p < alpha, as in the scorer. Time to tau leaves out every trial in which either arm
never reached tau, so it can favour the arm that reached tau less often; the reach test
is the check on that.

P9 runs the reach test at tau = 0.71, and both tests at the second tau, 0.82, on every
comparison the pre-registered studies make (batch 1, sens, batch 2 and batch 3), from
the scored files the campaign wrote; Holm across each study and test. It lists each
recorded time-to-tau claim whose favoured arm reached tau less often.

--score runs the trace scorer on a probe's trial CSVs that have no scored file yet, as
launch.py score does. Never run launch.py score with a probe's --set changes: it would
overwrite the pre-registered score files. P6-P8 join the registry once their builds (B1,
B2, B3) exist. Outputs: results/exp5/scores/probes/<name>.{md,csv}.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

OUT = REPO / "results/exp5/scores/probes"
TAU = 0.71
SECOND_TAU = 0.82
REFERENCE = "F"
P9_STAGES = ("batch1", "sens", "batch2", "batch3")
NAMES = {"F": "F", "FX": "FX", "H1": "H1", "D1": "MAX-AoI", "D2": "Oort", "D3": "Whittle",
         "D4": "FedEx route", "D5": "FedCS", "E3": "DQN (Chen)", "D4fedex": "FedEx route + merge"}
EIGHT = "['F','FX','H1','D1','D2','D3','D4','D5']"


def is_ferry(variant: str) -> bool:
    """FeRRy's own arms besides F: compared with F apart from the baselines."""
    return variant.startswith("FX") or variant.startswith("FQ")


# --------------------------------------------------------------------- the probes
@dataclass
class Flight:
    """One launcher run of a probe: its stage, out-root and --set changes (always at 20
    trials), the study whose cells it scores, and optionally the arms it keeps."""

    label: str
    stage: str
    out_root: str
    sets: List[str]
    study: str
    arms: Optional[List[str]] = None


@dataclass
class Probe:
    name: str
    title: str
    flights: List[Flight]
    #: cell -> the cell whose baselines stand in where this one has none flown (the
    #: baselines fly wide, so a setting that moves only the narrow band leaves them as is)
    ref_cells: Dict[str, str] = field(default_factory=dict)


def _s53x(extra: Sequence[str]) -> List[str]:
    return ["s53x.K=[1]", "s53x.h0=false", *extra]


PROBES: Dict[str, Probe] = {p.name: p for p in [
    Probe("P1", "Harsh radio, every baseline", [
        Flight("seconds backhaul", "batch3", "results/exp5",
               [f"s515.arms={EIGHT}", "s515.amps_db=[11.0]", "s515.mission_backhaul_arms=[]"],
               "s515"),
        Flight("mission backhaul", "batch3", "results/exp5_p1mbh",
               ["s515.backhaul='mission'", f"s515.arms={EIGHT}", "s515.amps_db=[]",
                "s515.shadow_sigmas_db=[]", "s515.mission_backhaul_arms=[]"], "s515"),
    ]),
    Probe("P2", "Budgets tighter than stress", [
        Flight("60 s", "batch2", "results/exp5_probe_tight40",
               _s53x(["batch2.relaxed_factor=0.4", "s53x.budgets=['relaxed']"]), "s53x"),
        Flight("45 s", "batch2", "results/exp5_probe_tight30",
               _s53x(["batch2.relaxed_factor=0.3", "s53x.budgets=['relaxed']"]), "s53x"),
    ]),
    Probe("P3", "MAX-AoI, Oort and FedCS at scale", [
        Flight("", "batch2", "results/exp5",
               ["s59x.arms=['F','D1','D2','D5']", "s59x.N=[12,24]", "s59x.K=[1]",
                "s59x.contact_regimes=['jittery']"], "s59x"),
    ]),
    Probe("P4", "Baselines in the far-device and short-reach cells", [
        Flight("", "batch2", "results/exp5",
               ["s54.arms=['F','H1','D1','D2','D3','D4','D5']", "s54.payloads=[1000000]",
                "s54.o1=false"], "s54"),
    ], ref_cells={"n6k1_knee_1mb_ratio2": "n6k1_knee_1mb",
                  "n6k1_knee_1mb_ratio3": "n6k1_knee_1mb"}),
    Probe("P5", "Bigger uploads at the N = 6 cell", [
        Flight(f"{mb} MB", "batch2", f"results/exp5_probe_{mb}mb",
               _s53x([f"s53x.payload_bytes={mb}000000", "s53x.budgets=['knee']",
                      "s53x.d4_faithful=false"]), "s53x",
               arms=["F", "FX", "H1", "D1", "D2", "D3", "D4", "D5"])
        for mb in (2, 5, 10)
    ]),
]}


# --------------------------------------------------------------------- the tests
def mcnemar_exact(only_ref: int, only_var: int) -> float:
    """Two-sided exact McNemar p: the discordant pairs (only the reference reached tau,
    only the variant did) against Binomial(their sum, 1/2)."""
    n = only_ref + only_var
    if n == 0:
        return 1.0
    k = min(only_ref, only_var)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def compare_reach(ref, var, tau: float, family: str, plan, SC):
    """F (or a study's reference) against one arm on reaching tau, paired by trial and
    seed; mean_diff = reference share - variant share, so a claim favours the arm that
    reached tau more often."""
    import numpy as np
    from experiments.analysis.stats import bootstrap_ci

    col = f"reached_tau{tau:g}"
    c = SC.Comparison(family, var.cell, var.variant, ref.variant, col, "higher", 0)
    a, b = SC._values(ref, col), SC._values(var, col)
    xs: List[float] = []
    ys: List[float] = []
    for t in sorted(set(a) & set(b)):
        (seed_a, x), (seed_b, y) = a[t], b[t]
        if seed_a != seed_b:
            c.pairing_errors += 1
            continue
        if x is None or y is None:
            continue
        xs.append(1.0 if x else 0.0)
        ys.append(1.0 if y else 0.0)
    c.n_pairs = len(xs)
    notes = [f"{c.pairing_errors} trials with different seeds left out (pairing error)"] \
        if c.pairing_errors else []
    if c.n_pairs < 2:
        c.note = "; ".join(notes + ["fewer than 2 pairs"])
        return c
    only_ref = sum(1 for x, y in zip(xs, ys) if x > y)
    only_var = sum(1 for x, y in zip(xs, ys) if y > x)
    c.ref_mean = sum(xs) / len(xs)
    c.variant_mean = sum(ys) / len(ys)
    c.mean_diff, c.ci_low, c.ci_high = bootstrap_ci(
        [x - y for x, y in zip(xs, ys)], np.mean, n_bootstraps=plan.n_bootstraps)
    c.p_value = mcnemar_exact(only_ref, only_var)
    c.note = "; ".join(notes + [f"{only_ref} reached by {ref.variant} only, "
                                f"{only_var} by {var.variant} only"])
    return c


def compare_time(ref, var, tau: float, family: str, plan, SC):
    """The scorer's paired time-to-tau comparison (lower is better), in ``family``."""
    spec = SC.Spec(family, "sim_s_to_tau", "lower", reference=ref.variant)
    return SC.compare_pair(ref, var, f"sim_s_to_tau{tau:g}", spec, plan)


def reach_contradicts(favoured: str, reach) -> bool:
    """Whether the arm a time claim favours reached tau less often than the other arm
    (the reach comparison's point estimate)."""
    if reach is None or reach.ref_mean is None:
        return False
    mine, other = ((reach.ref_mean, reach.variant_mean) if favoured == reach.reference
                   else (reach.variant_mean, reach.ref_mean))
    return mine < other


# --------------------------------------------------------------------- probe scoring
def probe_comparisons(probe: Probe, cells: Dict[str, Dict[str, Any]], plan, SC,
                      tau: float = TAU) -> List[Any]:
    """Every arm against F in each cell, on time and on reach; Holm per family."""
    comps = []
    for cell, variants in cells.items():
        ref = variants.get(REFERENCE)
        if ref is None or not ref.scored_rows:
            continue
        pool = dict(variants)
        prefix, _, bare = cell.rpartition(": ")
        prefix = f"{prefix}: " if prefix else ""
        stand_in = probe.ref_cells.get(bare)
        if stand_in:
            for v, e in cells.get(prefix + stand_in, {}).items():
                if v != REFERENCE and (v not in pool or not pool[v].scored_rows):
                    pool[v] = e
        for v, e in pool.items():
            if v == REFERENCE or not e.scored_rows:
                continue
            group = "ferry" if is_ferry(v) else "baselines"
            for kind, fn in (("time", compare_time), ("reach", compare_reach)):
                c = fn(ref, e, tau, f"{probe.name}/{group}/{kind}", plan, SC)
                c.cell = cell
                comps.append(c)
    SC.holm(comps, plan)
    return comps


def _launcher():
    spec = importlib.util.spec_from_file_location("exp5_launch", HERE / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp5_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


def flight_cells(L, SC, flight: Flight, *, score: bool = False):
    """A flight's entries by cell (prefixed with the flight's label) and variant, and the
    plan. The whole stage is built, so a cell its study shares with another reads that
    study's CSV, as launch.py score reads it."""
    s, _ = L.load_settings(L.PARAMS)
    ns = argparse.Namespace(trials=20, set=list(flight.sets), dataset=None,
                            **{o: None for o, _, _ in L.LEVERS})
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build(flight.stage, s, None, flight.out_root)
    runs, entries, _, plan = L.score_plan(flight.stage, s, jobs)
    mine = [e for e in entries.get(flight.study, [])
            if flight.arms is None or e.arm in flight.arms]
    if score:
        wanted = {str(e.scored) for e in mine}
        for j in runs:
            if str(REPO / j.out) in wanted and not L.done(j):
                print(f"scoring {j.out}")
                subprocess.run([sys.executable] + j.args, cwd=REPO, check=True)
    cells: Dict[str, Dict[str, Any]] = {}
    for e in mine:
        SC.load(e)
        key = f"{flight.label}: {e.cell}" if flight.label else e.cell
        cells.setdefault(key, {})[e.variant] = e
    return cells, plan


def _f(x: Optional[float], nd: int = 1) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def _diff(c, nd: int = 1, scale: float = 1.0) -> str:
    if c is None:
        return ""
    if c.mean_diff is None:
        return c.note
    return (f"{c.mean_diff * scale:.{nd}f} [{c.ci_low * scale:.{nd}f}, "
            f"{c.ci_high * scale:.{nd}f}]")


def _verdict(c) -> str:
    if c is None or c.p_holm is None:
        return ""
    return NAMES.get(c.favours, c.favours) if c.claim else "no claim"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    names: List[str] = []
    for r in rows:
        names += [k for k in r if k not in names]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def probe_report(L, SC, probe: Probe, *, score: bool = False) -> List[str]:
    cells: Dict[str, Dict[str, Any]] = {}
    plan = None
    for flight in probe.flights:
        got, plan = flight_cells(L, SC, flight, score=score)
        cells.update(got)
    comps = probe_comparisons(probe, cells, plan, SC)
    by = {(c.cell, c.variant, c.metric.startswith("reached")): c for c in comps}
    arms = [(cell, v, e) for cell, vs in cells.items() for v, e in vs.items()]
    flown = sum(1 for *_, e in arms if e.scored_rows)
    losses = [c for c in comps if c.claim and c.favours != REFERENCE and not is_ferry(c.variant)]
    lines = [f"# {probe.name}: {probe.title}", "",
             "Exploratory, not pre-registered (DeveloperDocs/Experiment_5_Loss_Probes_Plan.md). "
             f"Reference F. Time to tau = {TAU:g} over the trials both arms reach it, and reach "
             "at tau (exact McNemar on the paired outcomes), each with a 95% bootstrap CI. "
             "Holm per family: the baselines on time, on reach, and FX on each apart. "
             "Difference = F - arm: time, negative when F is faster; reach, positive when F "
             "reaches tau more often.", "",
             f"Flown and scored: {flown} of {len(arms)} arm-cells.", ""]
    lines += ["## Where a baseline beats F", ""]
    lines += ([f"- {c.cell}: {NAMES.get(c.variant, c.variant)} "
               f"({'reach' if c.metric.startswith('reached') else 'time to tau'}; "
               f"Holm p {c.p_holm:.4f})" for c in losses] or ["- none (so far)"])
    for cell, variants in cells.items():
        prefix, _, bare = cell.rpartition(": ")
        prefix = f"{prefix}: " if prefix else ""
        stand_in = probe.ref_cells.get(bare)
        borrowed = cells.get(prefix + stand_in, {}) if stand_in else {}
        lines += ["", f"## {cell}", ""]
        if stand_in:
            lines += [f"An arm not flown here is compared on its {stand_in} trials (same seeds), "
                      "marked with that cell's name.", ""]
        lines += ["| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | "
                  "Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v, e in variants.items():
            name = NAMES.get(v, v)
            if not e.scored_rows and v != REFERENCE and v in borrowed:
                e, name = borrowed[v], f"{name} ({stand_in})"
            n = len(e.scored_rows)
            if not n:
                lines.append(f"| {name} | not flown | | | | | | | | | |")
                continue
            reach = [SC._num(r.get(f"reached_tau{TAU:g}")) for r in e.scored_rows]
            reach = [x for x in reach if x is not None]
            times = [SC._num(r.get(f"sim_s_to_tau{TAU:g}")) for r in e.scored_rows]
            times = [x for x in times if x is not None]
            t, r = by.get((cell, v, False)), by.get((cell, v, True))
            lines.append(
                f"| {name} | {n} | {_f(sum(reach) / len(reach) if reach else None, 2)} | "
                f"{_f(sum(times) / len(times) if times else None)} | {_diff(t)} | "
                f"{'' if t is None or t.p_holm is None else f'{t.p_holm:.4f}'} | {_verdict(t)} | "
                f"{_diff(r, 2)} | {'' if r is None else r.note.split('; ')[-1]} | "
                f"{'' if r is None or r.p_holm is None else f'{r.p_holm:.4f}'} | {_verdict(r)} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{probe.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / f"{probe.name}.csv", [c.row() for c in comps])
    return lines


# --------------------------------------------------------------------- P9
def recorded_claims(stage_dir: Path, study: str) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """The pre-registered time-to-tau claims of one study, by (cell, variant, reference)."""
    path = stage_dir / f"{study}_comparisons.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["cell"], r["variant"], r["reference"]): r for r in rows
            if r.get("claim") == "True" and r.get("metric", "").startswith("sim_s_to_tau")}


def p9_report(L, SC) -> List[str]:
    comps: List[Any] = []
    stage_of: Dict[int, str] = {}
    claims: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    plan = None
    for stage in P9_STAGES:
        s, _ = L.load_settings(L.PARAMS)
        with contextlib.redirect_stdout(io.StringIO()):
            jobs = L.build(stage, s, None, "results/exp5")
        _, entries, _, plan = L.score_plan(stage, s, jobs)
        stage_dir = REPO / "results/exp5/scores" / L.STAGE_DIR[stage]
        for study, members in entries.items():
            spec = plan.specs.get(study)
            if spec is None or not spec.metric:
                continue
            for key, row in recorded_claims(stage_dir, study).items():
                claims[(stage, study, *key)] = row
            by_cell: Dict[str, Dict[str, Any]] = {}
            for e in members:
                SC.load(e)
                if e.scored_rows:
                    by_cell.setdefault(e.cell, {})[e.variant] = e
            for cell, variants in by_cell.items():
                for name, e in variants.items():
                    ref = spec.reference_of(name)
                    if ref is None or ref not in variants:
                        continue
                    for kind, tau, fn in (("reach", TAU, compare_reach),
                                          ("time", SECOND_TAU, compare_time),
                                          ("reach", SECOND_TAU, compare_reach)):
                        c = fn(variants[ref], e, tau, f"p9/{stage}/{study}/{kind}{tau:g}",
                               plan, SC)
                        stage_of[id(c)] = stage
                        comps.append(c)
    SC.holm(comps, plan)

    def study_of(c) -> str:
        return c.study.split("/")[2]

    def kind_of(c) -> str:
        return c.study.split("/")[3]

    reach71 = {(stage_of[id(c)], study_of(c), c.cell, c.variant, c.reference): c
               for c in comps if kind_of(c) == f"reach{TAU:g}"}
    lines = ["# P9: reach at tau, and the second tau, on every recorded comparison", "",
             "Exploratory, not pre-registered (3 Nov 2026). Every comparison the pre-registered "
             "studies make, from the scored files of batch 1, sens, batch 2 and batch 3 "
             "(results/exp5). Reach: exact McNemar on the paired reached / not-reached outcomes, "
             "with a 95% bootstrap CI on the paired difference in reach share (reference - "
             "variant). Time: the scorer's paired comparison at the second tau. Holm across "
             "each study and test. The studies' own comparisons (scores/<batch>/) are unchanged.",
             "", "## Summary", "",
             f"| batch | study | comparisons | reach at {TAU:g}: claims for the reference / the variant | "
             f"time to {SECOND_TAU:g}: comparisons with 2+ pairs | claims | "
             f"reach at {SECOND_TAU:g}: claims |",
             "|---|---|---|---|---|---|---|"]
    keys = sorted({(stage_of[id(c)], study_of(c)) for c in comps},
                  key=lambda k: (P9_STAGES.index(k[0]), k[1]))
    for stage, study in keys:
        mine = [c for c in comps if stage_of[id(c)] == stage and study_of(c) == study]
        r71 = [c for c in mine if kind_of(c) == f"reach{TAU:g}"]
        t82 = [c for c in mine if kind_of(c) == f"time{SECOND_TAU:g}"]
        r82 = [c for c in mine if kind_of(c) == f"reach{SECOND_TAU:g}"]
        lines.append(
            f"| {L.STAGE_DIR[stage]} | {study} | {len(r71)} | "
            f"{sum(c.claim and c.favours == c.reference for c in r71)} / "
            f"{sum(c.claim and c.favours == c.variant for c in r71)} | "
            f"{sum(c.p_value is not None for c in t82)} | {sum(c.claim for c in t82)} | "
            f"{sum(c.claim for c in r82)} |")

    lines += ["", "## Recorded time-to-tau claims whose favoured arm reached tau less often", "",
              "A recorded claim (the study's own score file) whose favoured arm reached "
              f"tau = {TAU:g} in fewer of the paired trials than the other arm.", "",
              "| batch | study | cell | favoured | other | time diff (ref − variant) | "
              "reach favoured | reach other | reach Holm p | reach verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    flagged = 0
    for (stage, study, cell, variant, ref), row in sorted(claims.items()):
        r = reach71.get((stage, study, cell, variant, ref))
        fav = row["favours"]
        if not reach_contradicts(fav, r):
            continue
        flagged += 1
        mine, other = ((r.ref_mean, r.variant_mean) if fav == r.reference
                       else (r.variant_mean, r.ref_mean))
        lines.append(f"| {L.STAGE_DIR[stage]} | {study} | {cell} | {fav} | "
                     f"{variant if fav == ref else ref} | {float(row['mean_diff']):.1f} | "
                     f"{mine:.2f} | {other:.2f} | {r.p_holm:.4f} | {_verdict(r)} |")
    if not flagged:
        lines.append("| | | none | | | | | | | |")

    for kind, title in ((f"reach{TAU:g}", f"Reach claims at tau = {TAU:g}"),
                        (f"time{SECOND_TAU:g}", f"Time-to-tau claims at tau = {SECOND_TAU:g}"),
                        (f"reach{SECOND_TAU:g}", f"Reach claims at tau = {SECOND_TAU:g}")):
        rows = [c for c in comps if kind_of(c) == kind and c.claim]
        scale, nd = (1.0, 1) if kind.startswith("time") else (1.0, 2)
        lines += ["", f"## {title}", "",
                  "| batch | study | cell | reference | variant | n pairs | reference | variant | "
                  "difference [95% CI] | Holm p | favours | note |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in rows:
            lines.append(f"| {L.STAGE_DIR[stage_of[id(c)]]} | {study_of(c)} | {c.cell} | "
                         f"{c.reference} | {c.variant} | {c.n_pairs} | {_f(c.ref_mean, nd)} | "
                         f"{_f(c.variant_mean, nd)} | {_diff(c, nd, scale)} | {c.p_holm:.4f} | "
                         f"{NAMES.get(c.favours, c.favours)} | {c.note} |")
        if not rows:
            lines.append("| | | none | | | | | | | | | |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p9.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / "p9.csv", [{"batch": L.STAGE_DIR[stage_of[id(c)]], "test": kind_of(c),
                                 **c.row()} for c in comps])
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("which", nargs="+", help="p9, a probe (" + ", ".join(PROBES) + ") or all")
    ap.add_argument("--score", action="store_true",
                    help="score a probe's trial CSVs that have no scored file yet")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    which = [w.upper() if w.lower() != "p9" else "p9" for w in a.which]
    if "ALL" in which:
        which = ["p9"] + list(PROBES)
    unknown = [w for w in which if w != "p9" and w not in PROBES]
    if unknown:
        sys.exit(f"unknown probe(s) {', '.join(unknown)}: p9, {', '.join(PROBES)} or all")
    L = _launcher()
    SC = L._scoring()
    for w in which:
        lines = p9_report(L, SC) if w == "p9" else probe_report(L, SC, PROBES[w], score=a.score)
        print("\n".join(lines[:40]) + ("\n…" if len(lines) > 40 else ""))
        print(f"\nwritten: {OUT / (w + '.md')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
