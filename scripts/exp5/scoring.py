"""Exp 5 scoring: each job's kept traces through the trace scorer, then each
study's paired comparisons, as params.toml [score] sets them.

launch.py's `score` command runs both steps for a stage:

1. Score. Each runner job with rows gets one run of
   experiments.analysis.traces_scorer over its kept traces (<csv stem>_traces),
   with its trial CSV as the status source, at [score] tau, written beside the
   CSV as <csv stem>_scored<suffix>.csv (the suffix names what the study adds:
   _cap<S> for an age cap, _det for 5.13's detection columns, _comp for 5.12's
   compute columns), with the scorer's arguments in a .argv.json sidecar. A
   scored file older than its CSV, or scored under other arguments, is scored
   again; the rest are kept.
2. Compare. Per study, the scored rows are grouped by cell (the job's tag
   before "__") and variant (after it: the arm, or the study's own label, such
   as 5.14's switch or 5.1's rule). Within a cell each variant is compared with
   the study's reference variant on the study's primary metric, trial by trial
   on the paired seeds: the mean paired difference (reference - variant) with a
   bootstrap CI and a paired Wilcoxon with Cliff's delta
   (experiments.analysis.stats.compare_to_reference), then Holm's adjustment
   across the study's family (every comparison of the study, or each cell's,
   per [score] family). A claim needs the CI to exclude zero and the
   Holm-adjusted p below [score] alpha.

Pairs are complete cases, as in Exp 4's analysis (experiments.analysis.exp4):
a trial whose status is not ok, or whose metric is blank (time to tau in a run
that never reached tau), leaves that comparison, and each comparison reports
its n_pairs. The arms table gives each variant's reach rate at tau beside its
time to tau, so a time read on few pairs shows as such. A trial index whose
seed differs between the two sides is a pairing error: it is left out and
counted, never compared.

Outputs, under <out_root>/scores/<stage dir>/: <study>_arms.csv (per cell and
variant: trials asked, rows, rows not ok, the primary metric's n, mean, median
and sd, the reach rate at tau, and the means of the study's other columns),
<study>_comparisons.csv, <study>.md (both, readable) and index.md.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCORER = ["-m", "experiments.analysis.traces_scorer"]
#: Every job is scored with these; each adds columns only (blank where a trace
#: has nothing to say), so a scored row keeps every other column.
DEFAULT_SCORER_FLAGS = ("--cost-columns", "--pair-columns")
#: What a study may add, and the scored file's suffix for it.
STUDY_SCORER_FLAGS = {"--detection-columns": "_det", "--compute-columns": "_comp"}


# --------------------------------------------------------------------------- #
# The study's analysis plan
# --------------------------------------------------------------------------- #

@dataclass
class Spec:
    """One study's [score.<study>] table, with [score]'s defaults filled in."""

    study: str
    metric: Optional[str]                 # None: the study is described, not compared
    better: str = "lower"
    reference: Optional[str] = None
    versus: Dict[str, Optional[str]] = field(default_factory=dict)
    also: List[str] = field(default_factory=list)
    scorer_flags: List[str] = field(default_factory=list)
    age_cap: bool = False
    note: str = ""

    def reference_of(self, variant: str) -> Optional[str]:
        """The variant ``variant`` is compared with (None: not compared)."""
        if variant in self.versus:
            return self.versus[variant]
        if variant == self.reference:
            return None
        return self.reference


@dataclass
class Plan:
    """[score]: the thresholds, the family, alpha and the per-study specs."""

    taus: List[float]
    alpha: float
    family: str
    n_bootstraps: int
    specs: Dict[str, Spec]

    @property
    def tau(self) -> float:
        return self.taus[0]


def plan_from(data: Dict[str, Any]) -> Tuple[Plan, List[str]]:
    """The scoring plan from params.toml's data, and the problems found in it."""
    problems: List[str] = []
    top = dict(data.get("score") or {})
    taus = top.get("tau", [0.82])
    taus = [float(t) for t in (taus if isinstance(taus, list) else [taus])]
    family = str(top.get("family", "study"))
    if family not in ("study", "cell"):
        problems.append(f"score.family must be \"study\" or \"cell\", not {family!r}")
    default_also = list(top.get("also", []))
    specs: Dict[str, Spec] = {}
    for study, t in top.items():
        if not isinstance(t, dict):
            continue
        flags = list(t.get("scorer_flags", []))
        for f in flags:
            if f not in STUDY_SCORER_FLAGS:
                problems.append(f"score.{study}.scorer_flags: {f} is not one of "
                                f"{', '.join(STUDY_SCORER_FLAGS)}")
        better = str(t.get("better", "lower"))
        if better not in ("lower", "higher"):
            problems.append(f"score.{study}.better must be \"lower\" or \"higher\"")
        versus = {str(k): (None if v in ("", None) else str(v))
                  for k, v in dict(t.get("versus", {})).items()}
        specs[study] = Spec(
            study=study, metric=t.get("metric"), better=better,
            reference=t.get("reference"), versus=versus,
            also=list(t.get("also", default_also)), scorer_flags=flags,
            age_cap=bool(t.get("age_cap", False)), note=str(t.get("note", "")))
    plan = Plan(taus=taus, alpha=float(top.get("alpha", 0.05)), family=family,
                n_bootstraps=int(top.get("n_bootstraps", 2000)), specs=specs)
    return plan, problems


def column(name: str, tau: float) -> str:
    """A metric's scored column: a name ending in "tau" takes the primary tau
    (sim_s_to_tau -> sim_s_to_tau0.82, as the scorer names it)."""
    return f"{name}{tau:g}" if name.endswith("tau") else name


# --------------------------------------------------------------------------- #
# Step 1: the scorer, one run per trial CSV
# --------------------------------------------------------------------------- #

def traces_dir(csv_path: str) -> str:
    return csv_path[:-4] + "_traces"


def scored_path(csv_path: str, suffix: str) -> str:
    return csv_path[:-4] + f"_scored{suffix}.csv"


def scorer_call(csv_path: str, plan: Plan, spec: Optional[Spec],
                s_star: Optional[int]) -> Tuple[List[str], str, List[str]]:
    """(the scorer's arguments, the scored file, settings it still needs) for
    one trial CSV under one study's spec."""
    flags = list(DEFAULT_SCORER_FLAGS)
    suffix = ""
    missing: List[str] = []
    if spec is not None:
        if spec.age_cap:
            if s_star is None:
                missing.append("pilot_outputs.s_star")
            else:
                flags += ["--age-cap-s", str(int(s_star))]
                suffix += f"_cap{int(s_star)}"
        for f in spec.scorer_flags:
            flags.append(f)
            suffix += STUDY_SCORER_FLAGS[f]
    out = scored_path(csv_path, suffix)
    args = SCORER + ["--traces", traces_dir(csv_path), "--status-csv", csv_path,
                     "--tau", *[f"{t:g}" for t in plan.taus], *flags, "--csv", out]
    return args, out, missing


# --------------------------------------------------------------------------- #
# Step 2: the comparisons
# --------------------------------------------------------------------------- #

@dataclass
class Entry:
    """One job's place in its study: its cell and variant, and its rows."""

    study: str
    cell: str
    variant: str
    arm: str
    csv_path: Path
    scored: Path
    trials: int
    rows: int = 0                  # trial-CSV rows within the first `trials`
    rows_not_ok: int = 0
    scored_rows: List[Dict[str, str]] = field(default_factory=list)


def labels(job_name: str, arm: str) -> Tuple[str, str]:
    """(cell, variant) from a job's name, <study>/<cell>__<variant>."""
    tag = job_name.split("/", 1)[1] if "/" in job_name else job_name
    if "__" in tag:
        cell, variant = tag.split("__", 1)
        return cell, variant
    return tag, arm


def _read(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _first(rows: List[Dict[str, str]], trials: int) -> List[Dict[str, str]]:
    """The rows of the first ``trials`` trials (a CSV a later batch extended
    holds more; the study reads the trials it asked for)."""
    out = []
    for r in rows:
        try:
            if int(r["trial_index"]) < trials:
                out.append(r)
        except (KeyError, ValueError):
            continue
    return out


def load(entry: Entry) -> None:
    trial_rows = _first(_read(entry.csv_path), entry.trials)
    entry.rows = len(trial_rows)
    entry.rows_not_ok = sum(1 for r in trial_rows if r.get("status", "ok") != "ok")
    entry.scored_rows = [r for r in _first(_read(entry.scored), entry.trials)
                         if r.get("status", "ok") == "ok"]


def _num(v: Optional[str]) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _values(entry: Entry, col: str) -> Dict[int, Tuple[str, Optional[float]]]:
    """trial_index -> (seed, value) of the entry's ok rows."""
    return {int(r["trial_index"]): (r.get("seed", ""), _num(r.get(col)))
            for r in entry.scored_rows}


@dataclass
class Comparison:
    study: str
    cell: str
    variant: str
    reference: str
    metric: str
    better: str
    n_pairs: int
    ref_mean: Optional[float] = None
    variant_mean: Optional[float] = None
    mean_diff: Optional[float] = None       # reference - variant
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_value: Optional[float] = None
    p_holm: Optional[float] = None
    cliffs_delta: Optional[float] = None
    claim: bool = False
    favours: str = ""
    pairing_errors: int = 0
    note: str = ""

    def row(self) -> Dict[str, Any]:
        return {k: ("" if v is None else v) for k, v in self.__dict__.items()}


def compare_pair(ref: Entry, var: Entry, col: str, spec: Spec, plan: Plan) -> Comparison:
    from experiments.analysis.stats import compare_to_reference

    c = Comparison(spec.study, var.cell, var.variant, ref.variant, col, spec.better, 0)
    a, b = _values(ref, col), _values(var, col)
    xs: List[float] = []
    ys: List[float] = []
    for t in sorted(set(a) & set(b)):
        (seed_a, x), (seed_b, y) = a[t], b[t]
        if seed_a != seed_b:
            c.pairing_errors += 1
            continue
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    c.n_pairs = len(xs)
    if c.pairing_errors:
        c.note = f"{c.pairing_errors} trials with different seeds left out (pairing error)"
    if c.n_pairs < 2:
        c.note = (c.note + "; " if c.note else "") + "fewer than 2 complete pairs"
        return c
    r = compare_to_reference({"ref": xs, "var": ys}, "ref", alpha=plan.alpha,
                             n_bootstraps=plan.n_bootstraps)["var"]
    c.ref_mean = sum(xs) / len(xs)
    c.variant_mean = sum(ys) / len(ys)
    c.mean_diff, c.ci_low, c.ci_high = r.mean_diff, r.ci_low, r.ci_high
    c.p_value, c.cliffs_delta = r.p_value, r.cliffs_delta
    return c


def holm(comparisons: List[Comparison], plan: Plan) -> None:
    """Holm across each family; then the claim rule and which side it favours."""
    from experiments.analysis.stats import holm_bonferroni

    families: Dict[str, List[Comparison]] = {}
    for c in comparisons:
        if c.p_value is None:
            continue
        key = c.study if plan.family == "study" else f"{c.study}/{c.cell}"
        families.setdefault(key, []).append(c)
    for members in families.values():
        adjusted = holm_bonferroni([c.p_value for c in members], alpha=plan.alpha).adjusted
        for c, p in zip(members, adjusted):
            c.p_holm = float(p)
            excludes_zero = c.ci_low > 0.0 or c.ci_high < 0.0
            c.claim = bool(excludes_zero and c.p_holm < plan.alpha)
            if not c.claim:
                c.favours = "no claim"
                continue
            ref_larger = c.mean_diff > 0
            ref_better = ref_larger == (c.better == "higher")
            c.favours = c.reference if ref_better else c.variant


def describe(entry: Entry, spec: Spec, plan: Plan) -> Dict[str, Any]:
    """The arms table's row for one entry."""
    out: Dict[str, Any] = {
        "study": entry.study, "cell": entry.cell, "variant": entry.variant, "arm": entry.arm,
        "trials": entry.trials, "rows": entry.rows, "rows_not_ok": entry.rows_not_ok,
        "scored": len(entry.scored_rows),
    }
    if spec.metric:
        col = column(spec.metric, plan.tau)
        vals = [v for _, v in _values(entry, col).values() if v is not None]
        out["metric"] = col
        out["metric_n"] = len(vals)
        out["metric_mean"] = _r(sum(vals) / len(vals)) if vals else ""
        out["metric_median"] = _r(statistics.median(vals)) if vals else ""
        out["metric_sd"] = _r(statistics.stdev(vals)) if len(vals) > 1 else ""
    reach = [v for _, v in _values(entry, column("reached_tau", plan.tau)).values()
             if v is not None]
    out[f"reach_rate_tau{plan.tau:g}"] = _r(sum(reach) / len(reach)) if reach else ""
    for name in spec.also:
        col = column(name, plan.tau)
        vals = [v for _, v in _values(entry, col).values() if v is not None]
        out[f"mean_{col}"] = _r(sum(vals) / len(vals)) if vals else ""
    return out


def _r(x: float) -> float:
    return float(f"{x:.6g}")


def study_results(study: str, entries: List[Entry], spec: Spec, plan: Plan
                  ) -> Tuple[List[Dict[str, Any]], List[Comparison]]:
    """The arms rows and the comparisons of one study (Holm applied later)."""
    for e in entries:
        load(e)
    arms = [describe(e, spec, plan) for e in entries]
    comparisons: List[Comparison] = []
    if not spec.metric:
        return arms, comparisons
    col = column(spec.metric, plan.tau)
    by_cell: Dict[str, Dict[str, Entry]] = {}
    for e in entries:
        by_cell.setdefault(e.cell, {})[e.variant] = e
    for cell, variants in by_cell.items():
        for name, e in variants.items():
            ref_name = spec.reference_of(name)
            if ref_name is None or ref_name not in variants:
                continue
            comparisons.append(compare_pair(variants[ref_name], e, col, spec, plan))
    return arms, comparisons


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    names: List[str] = []
    for r in rows:
        names += [k for k in r if k not in names]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def _fmt(x: Any, places: int = 3) -> str:
    if x in (None, ""):
        return "–"
    if isinstance(x, float):
        return f"{x:.{places}g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.2e}"
    return str(x)


def study_markdown(study: str, spec: Spec, plan: Plan, arms: List[Dict[str, Any]],
                   comparisons: List[Comparison]) -> str:
    lines = [f"# {study}", ""]
    if spec.metric:
        lines.append(
            f"Primary metric `{column(spec.metric, plan.tau)}` ({spec.better} is better); "
            f"each variant against `{spec.reference}` in its cell"
            + ("".join(f", `{v}` against `{r}`" for v, r in spec.versus.items() if r)) + ". "
            f"Paired seeds, complete pairs; Holm across "
            f"{'the study' if plan.family == 'study' else 'each cell'}; a claim needs the "
            f"bootstrap CI to exclude 0 and Holm p < {plan.alpha:g}.")
    else:
        lines.append("Described only: [score] gives this study no primary metric.")
    if spec.note:
        lines += ["", spec.note]
    incomplete = [a for a in arms if a["rows"] < a["trials"]]
    if incomplete:
        lines += ["", f"**Incomplete:** {len(incomplete)} of {len(arms)} variants have fewer "
                      f"rows than trials asked (a stage still to run or finish)."]
    by_cell: Dict[str, List[Dict[str, Any]]] = {}
    for a in arms:
        by_cell.setdefault(a["cell"], []).append(a)
    cmp_by = {(c.cell, c.variant): c for c in comparisons}
    reach_col = f"reach_rate_tau{plan.tau:g}"
    for cell, members in by_cell.items():
        lines += ["", f"## {cell}", "",
                  "| variant | trials (not ok) | n | mean | median | reach τ | vs | "
                  "diff (ref − variant) [95% CI] | p Holm | δ | claim |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in members:
            c = cmp_by.get((cell, a["variant"]))
            ci = (f"{_fmt(c.mean_diff)} [{_fmt(c.ci_low)}, {_fmt(c.ci_high)}]"
                  if c and c.mean_diff is not None else (c.note if c else ""))
            lines.append(
                f"| {a['variant']} | {a['rows']}/{a['trials']} ({a['rows_not_ok']}) | "
                f"{a.get('metric_n', '–')} | {_fmt(a.get('metric_mean'))} | "
                f"{_fmt(a.get('metric_median'))} | {_fmt(a.get(reach_col))} | "
                f"{c.reference if c else ''} | {ci} | "
                f"{_fmt(c.p_holm) if c else ''} | {_fmt(c.cliffs_delta) if c else ''} | "
                f"{(c.favours if c.claim else 'no') if c and c.p_holm is not None else ''} |")
    return "\n".join(lines) + "\n"


def write_study(out_dir: Path, study: str, spec: Spec, plan: Plan,
                arms: List[Dict[str, Any]], comparisons: List[Comparison]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / f"{study}_arms.csv", arms)
    _write_csv(out_dir / f"{study}_comparisons.csv", [c.row() for c in comparisons])
    (out_dir / f"{study}.md").write_text(
        study_markdown(study, spec, plan, arms, comparisons), encoding="utf-8")


def index_markdown(stage: str, rows: List[Dict[str, Any]], tools: List[Tuple[str, str, bool]],
                   plan: Plan) -> str:
    lines = [f"# Exp 5 scores: {stage}", "",
             f"τ = {', '.join(f'{t:g}' for t in plan.taus)} (the first is the primary); "
             f"Holm per {plan.family}; alpha {plan.alpha:g}. Each study's page has its "
             f"tables; the CSVs beside it hold every column.", "",
             "| study | primary metric | variants | comparisons | claims | trials scored |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| [{r['study']}]({r['study']}.md) | {r['metric'] or '–'} | "
                     f"{r['variants']} | {r['comparisons']} | {r['claims']} | "
                     f"{r['scored']}/{r['trials']} |")
    if tools:
        lines += ["", "Outputs the score step does not read (FerrySim and tool jobs; "
                      "each is a JSON report of its own):", ""]
        for name, out, written in tools:
            lines.append(f"- `{name}`: `{out}`" + ("" if written else " (not written yet)"))
    return "\n".join(lines) + "\n"


def write_index(out_dir: Path, stage: str, rows: List[Dict[str, Any]],
                tools: List[Tuple[str, str, bool]], plan: Plan) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.md"
    path.write_text(index_markdown(stage, rows, tools, plan), encoding="utf-8")
    (out_dir / "plan.json").write_text(json.dumps({
        "taus": plan.taus, "alpha": plan.alpha, "family": plan.family,
        "n_bootstraps": plan.n_bootstraps,
        "specs": {k: v.__dict__ for k, v in plan.specs.items()}}, indent=1), encoding="utf-8")
    return path
