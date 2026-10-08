"""Exp 5's results as figures for the paper, read from the scored studies and traces.

    py -3.11 scripts/exp5/paper_figures.py [--out results/exp5/paper/figures]

Writes a PDF (vector, for LaTeX) and a 300-dpi PNG of each figure:

* ``fig_exp5_convergence``: test accuracy over simulated time at N = 6, one mule,
  the knee and stress budgets (Study 5.3): F and FX against every baseline.
* ``fig_exp5_tau``: time to tau by arm with 95% bootstrap CIs (Study 5.3).
* ``fig_exp5_scale``: time to tau as N grows, both budgets (Study 5.9).
* ``fig_exp5_systems``: planner time against N (5.11 a), FerrySim's served share
  at each scale's knee (5.11 c) and time to tau as mules are added (5.11 b).
* ``fig_exp5_claims``: every claim test as a relative effect with its CI.
* ``fig_exp5_compute``: update yield and reach rate as devices take time to
  train (Study 5.12), the limitation the paper reports.
* ``fig_exp5_mechanism``: why F wins at N = 6: mission time by component
  (transit, dwell, the rest) and time to tau, with F's band pinned to each
  class beside the baselines, which all fly the wide band.
* ``fig_exp5_bands``: F's band class by cell, as N and the budget change.
* ``fig_exp5_paired``: the share of paired seeds in which F reaches tau first.
* ``fig_exp5_budget``: updates per round against budget overruns (stress).

Numbers come from the same entries the scorer compared (``launch.py``'s
score plan under each stage's run flags, the first ``trials`` rows of each
cell), so a figure and its table never disagree. Colors follow the entity in
every figure (F blue, FX orange, H1 aqua, D3 yellow, D4 magenta, greedy-1
violet; other baselines gray), each arm also has its own marker, and the
palettes were checked with the dataviz validator (adjacent pairs; all pairs
for FerrySim's three lines).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
TAU = 0.71
TAU_COL = "sim_s_to_tau0.71"

# The run flags of each stage (the deadline campaign of 7 Oct 2026), so the score
# plan names the same cells and trials the scorer compared. 5.3 is read with its
# full arm set, as it was scored.
STAGE_SETS: Dict[str, Tuple[Optional[int], List[str], Optional[List[str]]]] = {
    "batch1": (None, [], None),
    "batch2": (20, ["s51.pass_2=['budgeted']",
                    "s53x.arms=['F','FX','H1','D1','D2','D3','D4','D5','E3']",
                    "s53x.budgets=['knee','stress']", "s53x.d4_faithful=true",
                    "s53x.h0=true", "s54.narrow_range_ratios=[]", "s54.far_shares=[]",
                    "s58.missions=[4]", "s59x.N=[12,24]", "s59x.K=[1]",
                    "s59x.budgets=['stress']", "s59x.contact_regimes=['jittery']",
                    "s513.partitions=[['dirichlet',0.1]]"],
               ["s51", "s52", "s53x", "s54", "s55", "s57", "s58", "s59x", "s513"]),
    "batch3": (None, ["s512.payloads=[1000000]", "s515.arms=['F','FX','H1']",
                      "s515.mission_backhaul_arms=[]"], None),
    # The exploratory FQ run of 8 Oct 2026 (scripts/exp5/fq_vs_e3.py): FQ in 5.3's
    # N = 6, one-mule cells beside F, FX, D4 and E3.
    "batch2fq": (20, ["s53x.arms=['F','FX','D4','E3','FQ-g75']", "s53x.K=[1]",
                      "s53x.budgets=['knee','stress']", "s53x.d4_faithful=false",
                      "s53x.h0=false", "rl.keep_learned=true"], ["s53x"]),
}
LAUNCH_STAGE = {"batch2fq": "batch2"}

# --------------------------------------------------------------------------- #
# Style (the dataviz reference palette and chrome, light, for print)
# --------------------------------------------------------------------------- #

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
CONTEXT = "#b9b7ae"                         # de-emphasis gray for context series
ARM_COLOR = {"F": "#2a78d6", "FX": "#eb6834", "H1": "#1baf7a", "D3": "#eda100",
             "D4": "#e87ba4", "greedy_1": "#4a3aa7", "D5": "#008300"}
ARM_MARKER = {"F": "o", "FX": "s", "H1": "^", "D3": "D", "D4": "v", "greedy_1": "P",
              "D5": "p", "D1": "<", "D2": ">", "E3": "h", "D4fedex": "X"}
ARM_LABEL = {"F": "F", "FX": "FX", "H1": "H1", "D1": "D1 (MAX-AoI)", "D2": "D2 (Oort)",
             "D3": "D3 (Cui)", "D4": "D4 (FedEx route)", "D4fedex": "D4 + FedEx merge",
             "D5": "D5 (FedCS)", "E3": "E3 (Chen)", "greedy_1": "greedy-1"}
SEQ = ["#86b6ef", "#2a78d6", "#104281"]     # ordinal blue ramp: steps 250, 450, 650
COL1, COL2 = 3.5, 7.16                      # IEEE column and page widths (in)


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "legend.frameon": False, "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def save(fig, out: Path, name: str) -> None:
    """Write each format to a temporary file, then swap it into place: on Windows a
    viewer or file watcher holding the old figure open would otherwise fail the write."""
    import os
    import time
    for ext in ("pdf", "png"):
        target = out / f"{name}.{ext}"
        tmp = out / f".{name}.tmp.{ext}"
        fig.savefig(tmp, format=ext, bbox_inches="tight", pad_inches=0.02)
        for attempt in range(20):
            try:
                os.replace(tmp, target)
                break
            except OSError:
                if attempt == 19:
                    raise
                time.sleep(0.25)
    plt.close(fig)
    print(f"  {name}.pdf/.png")


# --------------------------------------------------------------------------- #
# Reading the entries the scorer compared
# --------------------------------------------------------------------------- #

def _launcher():
    spec = importlib.util.spec_from_file_location("exp5_launch", HERE / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp5_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _launcher()
SC = L._scoring()


def entries(stage: str) -> Dict[Tuple[str, str, str], object]:
    trials, sets, studies = STAGE_SETS[stage]
    s, _ = L.load_settings(L.PARAMS)
    ns = argparse.Namespace(trials=trials, set=sets, dataset=None,
                            **{o: None for o, _, _ in L.LEVERS})
    launch_stage = LAUNCH_STAGE.get(stage, stage)
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build(launch_stage, s, studies, "results/exp5")
    _, by_study, _, _ = L.score_plan(launch_stage, s, jobs)
    out = {}
    for study, members in by_study.items():
        for e in members:
            SC.load(e)
            out[(study, e.cell, e.variant)] = e
    return out


E: Dict[str, Dict] = {}


def entry(stage: str, study: str, cell: str, variant: str):
    if stage not in E:
        E[stage] = entries(stage)
    return E[stage][(study, cell, variant)]


def values(e, col: str) -> List[float]:
    out = []
    for r in e.scored_rows:
        v = SC._num(r.get(col))
        if v is not None:
            out.append(v)
    return out


def reach_rate(e, col: str = TAU_COL) -> float:
    n = len(e.scored_rows)
    return len(values(e, col)) / n if n else float("nan")


def mean_ci(xs: Sequence[float], n_boot: int = 4000, seed: int = 7) -> Tuple[float, float, float]:
    """Mean and its 95% percentile-bootstrap CI."""
    if not xs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    m = statistics.fmean(xs)
    boots = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n_boot))
    return m, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def comparison(stage_dir: str, study: str, cell: str, variant: str) -> Dict[str, str]:
    import csv
    path = REPO / "results/exp5/scores" / stage_dir / f"{study}_comparisons.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cell"] == cell and r["variant"] == variant:
                return r
    raise KeyError((study, cell, variant))


def is_claim(c: Dict[str, str]) -> bool:
    return c.get("claim", "").lower() in ("true", "1")


# --------------------------------------------------------------------------- #
# Accuracy curves from the kept traces
# --------------------------------------------------------------------------- #

def curves(e, n_devices: int) -> List[List[Tuple[float, float]]]:
    """Each trial's (simulated s since the first takeoff, accuracy) steps, round 0 at t = 0."""
    from experiments.analysis import traces_scorer as TS
    from experiments.exp4.events_consumer import consume_run_dir
    root = REPO / SC.traces_dir(str(e.csv_path.relative_to(REPO)))
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            key = TS.parse_trial_dir(d.name)
        except ValueError:
            continue
        if key.trial_index >= e.trials:
            continue
        obs = consume_run_dir(d, n_devices=n_devices)
        start = TS._fleet_sim_start(obs, TS._ordered(obs.missions, obs.mission_clock))
        if start is None:
            continue
        pts = []
        for ev in sorted(obs.model_evals, key=lambda v: v.cluster_round):
            if ev.cluster_round == 0:
                pts.append((0.0, ev.accuracy))
            elif ev.sim_ts is not None:
                pts.append((ev.sim_ts - start, ev.accuracy))
        if pts:
            out.append(pts)
    return out


def mean_curve(trials: List[List[Tuple[float, float]]], grid: Sequence[float]) -> List[float]:
    """The mean over trials of each trial's step function (last evaluation at or before t)."""
    rows = []
    for pts in trials:
        vals, j, cur = [], 0, pts[0][1]
        for t in grid:
            while j < len(pts) and pts[j][0] <= t:
                cur = pts[j][1]
                j += 1
            vals.append(cur)
        rows.append(vals)
    return [statistics.fmean(col) for col in zip(*rows)]


# --------------------------------------------------------------------------- #
# The figures
# --------------------------------------------------------------------------- #

BASELINES = ["H1", "D1", "D2", "D3", "D4", "D5", "E3"]


def fig_convergence(out: Path) -> None:
    cells = [("n6k1_knee", "knee budget (150 s)"), ("n6k1_stress", "stress budget (75 s)")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.15), sharey=True)
    grid = [i * 2.0 for i in range(0, 451)]          # 0..900 s
    for ax, (cell, title) in zip(axes, cells):
        for arm in BASELINES:
            e = entry("batch2", "s53x", cell, arm)
            ax.plot(grid, mean_curve(curves(e, 6), grid), color=CONTEXT, lw=0.9, zorder=2)
        for arm in ("FX", "F"):
            e = entry("batch2", "s53x", cell, arm)
            ax.plot(grid, mean_curve(curves(e, 6), grid), color=ARM_COLOR[arm], lw=1.8,
                    zorder=4)
        ax.axhline(TAU, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.text(grid[-1], TAU + 0.004, r"$\tau$ = 0.71", ha="right", va="bottom",
                color=INK2, fontsize=6.5)
        ax.set_title(title, color=INK2, loc="left")
        ax.set_xlabel("simulated time (s)")
        ax.set_xlim(0, grid[-1])
    axes[0].set_ylabel("test accuracy (mean of 20 trials)")
    axes[0].set_ylim(0.35, 0.88)
    handles = [Line2D([], [], color=ARM_COLOR["F"], lw=1.8, label="F"),
               Line2D([], [], color=ARM_COLOR["FX"], lw=1.8, label="FX"),
               Line2D([], [], color=CONTEXT, lw=0.9,
                      label="baselines: H1, D1 (MAX-AoI), D2 (Oort), D3 (Cui), "
                            "D4 (FedEx), D5 (FedCS), E3 (Chen)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.27, wspace=0.08)
    save(fig, out, "fig_exp5_convergence")


def fig_tau(out: Path) -> None:
    arms = ["F", "FX", "D4", "D2", "D5", "D3", "E3", "D1", "H1"]
    cells = [("n6k1_knee", "knee budget"), ("n6k1_stress", "stress budget")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True)
    for ax, (cell, title) in zip(axes, cells):
        for i, arm in enumerate(arms):
            y = len(arms) - 1 - i
            e = entry("batch2", "s53x", cell, arm)
            xs = values(e, TAU_COL)
            rng = random.Random(i)
            ax.scatter(xs, [y + rng.uniform(-0.18, 0.18) for _ in xs], s=4, color=CONTEXT,
                       alpha=0.7, lw=0, zorder=2)
            m, lo, hi = mean_ci(xs)
            color = ARM_COLOR.get(arm, INK2) if arm in ("F", "FX") else INK2
            ax.plot([lo, hi], [y, y], color=color, lw=1.4, zorder=3)
            ax.scatter([m], [y], s=22, color=color, marker=ARM_MARKER.get(arm, "o"),
                       edgecolor=SURFACE, linewidth=1.0, zorder=4)
            mark = ""
            if arm != "F":
                c = comparison("b2", "s53x", cell, arm)
                if is_claim(c) and c.get("favours") == "F":
                    mark = r"$^{\dagger}$"
            rr = reach_rate(e)
            note = f"  {m:.0f}{mark}" + (f"  (reach {rr:.2f})" if rr < 0.995 else "")
            ax.text(505, y, note.strip(), va="center", fontsize=6.3, color=INK2,
                    clip_on=False)
        ax.set_title(title, color=INK2, loc="left")
        ax.set_xlabel(r"simulated time to $\tau$ = 0.71 (s)")
        ax.set_xlim(0, 500)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels([ARM_LABEL[a] for a in reversed(arms)])
    axes[0].tick_params(axis="y", labelcolor=INK2)
    for lbl in axes[0].get_yticklabels():
        if lbl.get_text() in ("F", "FX"):
            lbl.set_color(INK)
            lbl.set_fontweight("bold")
    fig.subplots_adjust(wspace=0.42)
    save(fig, out, "fig_exp5_tau")


SCALE_ARMS = ["F", "FX", "H1", "D3", "D4"]


def fig_scale(out: Path) -> None:
    ns = [6, 12, 24]
    src = {"knee": {6: ("batch1", "s59", "n6k1_knee"), 12: ("batch1", "s59", "n12k1_knee"),
                    24: ("batch1", "s59", "n24k1_knee")},
           "stress": {6: ("batch2", "s53x", "n6k1_stress"),
                      12: ("batch2", "s59x", "n12k1_stress"),
                      24: ("batch2", "s59x", "n24k1_stress")}}
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True)
    dodge = {a: (i - 2) * 0.07 for i, a in enumerate(SCALE_ARMS)}
    for ax, budget in zip(axes, ("knee", "stress")):
        for arm in SCALE_ARMS:
            xs, ms, los, his = [], [], [], []
            for k, n in enumerate(ns):
                stage, study, cell = src[budget][n]
                m, lo, hi = mean_ci(values(entry(stage, study, cell, arm), TAU_COL))
                xs.append(k + dodge[arm])
                ms.append(m)
                los.append(m - lo)
                his.append(hi - m)
            lw = 1.8 if arm == "F" else 1.2
            ax.errorbar(xs, ms, yerr=[los, his], color=ARM_COLOR[arm], lw=lw,
                        marker=ARM_MARKER[arm], ms=4.2, mec=SURFACE, mew=0.8,
                        elinewidth=0.8, capsize=0, zorder=4 if arm == "F" else 3,
                        label=ARM_LABEL[arm])
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"N = {n}" for n in ns])
        ax.set_title(f"{budget} budget", color=INK2, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(r"simulated time to $\tau$ (s)")
    axes[0].set_ylim(0, None)
    axes[1].legend(loc="upper left", ncol=1, handlelength=1.6)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, "fig_exp5_scale")


def fig_systems(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2))
    # (a) planner wall time per plan, alone on the host (the 7 Oct rerun)
    quiet = json.loads((REPO / "results/exp5/s511a_quiet/b1/s511a/auto.json")
                       .read_text(encoding="utf-8"))
    rows = {r["cell"]: r for r in quiet["table"]}
    cells = [("jit-n6-150", 6), ("jit-n12-90", 12), ("scl-n24-350", 24),
             ("scl-n48-680", 48), ("scl-n96-1330", 96)]
    ax = axes[0]
    ns = [n for _, n in cells]
    mean = [rows[c]["plan_wall_s_mean"] for c, _ in cells]
    p95 = [rows[c]["plan_wall_s_p95"] for c, _ in cells]
    ax.plot(ns, mean, color=ARM_COLOR["F"], marker="o", ms=4, mec=SURFACE, mew=0.8)
    ax.plot(ns, p95, color=ARM_COLOR["F"], lw=0.9, ls=(0, (3, 2)), marker="o", ms=3,
            mfc=SURFACE, mec=ARM_COLOR["F"], mew=0.8)
    ax.text(ns[-1], mean[-1], " mean", va="center", fontsize=6.3, color=INK2)
    ax.text(ns[-1], p95[-1], " p95", va="center", fontsize=6.3, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    yt = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.set_xlabel("devices N")
    ax.set_ylabel("planner time per plan (s)")
    ax.set_title("(a) decision cost", color=INK2, loc="left")
    ax.set_xlim(5, 140)
    # (b) FerrySim at each scale's knee
    ax = axes[1]
    scale = [("scl-n24-350", 24), ("scl-n48-680", 48), ("scl-n96-1330", 96)]
    rep = {c: {r["policy"]: r for r in json.loads(
        (REPO / f"results/exp5/b3/s511c/{c}.json").read_text(encoding="utf-8"))["table"]}
        for c, _ in scale}
    for pol in ("F", "FX", "greedy_1"):
        ys = [rep[c][pol]["served_share_mean"] for c, _ in scale]
        ax.plot([n for _, n in scale], ys, color=ARM_COLOR[pol], marker=ARM_MARKER[pol],
                ms=4, mec=SURFACE, mew=0.8, label=ARM_LABEL[pol])
    ax.set_xscale("log")
    ax.set_xticks([24, 48, 96])
    ax.set_xticklabels(["24", "48", "96"])
    ax.minorticks_off()
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("devices N (FerrySim, knee budget)")
    ax.set_ylabel("share of devices served")
    ax.set_title("(b) plan quality at scale", color=INK2, loc="left")
    ax.legend(loc="lower right")
    # (c) strong scaling: mules at N = 12
    ax = axes[2]
    ks = [1, 2, 3]
    cells_k = {1: "n12k1_knee", 2: "n12k2_knee", 3: "n12k3_knee"}
    for i, arm in enumerate(("F", "FX", "H1", "D4")):
        xs, ms, los, his = [], [], [], []
        for k in ks:
            m, lo, hi = mean_ci(values(entry("batch1", "s511b", cells_k[k], arm), TAU_COL))
            xs.append(k + (i - 1.5) * 0.06)
            ms.append(m)
            los.append(m - lo)
            his.append(hi - m)
        ax.errorbar(xs, ms, yerr=[los, his], color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                    ms=4, mec=SURFACE, mew=0.8, elinewidth=0.8, capsize=0,
                    lw=1.8 if arm == "F" else 1.2, label=ARM_LABEL[arm])
    ax.set_xticks(ks)
    ax.set_xticklabels([f"K = {k}" for k in ks])
    ax.set_ylim(0, None)
    ax.set_xlabel("mules (N = 12, knee budget)")
    ax.set_ylabel(r"simulated time to $\tau$ (s)")
    ax.set_title("(c) scaling out with mules", color=INK2, loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out, "fig_exp5_systems")


# (claim, test label, study dir, study, cell, variant, lower is better, flip)
# flip: the variant is FeRRy's own choice, so the sign turns to keep
# "right = FeRRy's design better".
CLAIM_ROWS = [
    ("C1", "band pinned: wide", "b2", "s54", "n6k1_knee_1mb", "FBpwide", True, False),
    ("C1", "band pinned: narrow", "b2", "s54", "n6k1_knee_1mb", "FBpnarrow", True, False),
    ("C2", "coverage term off", "b2", "s57", "n12k1_knee", "FX-cov", False, False),
    ("C2", "dwell term off", "b2", "s57", "n12k1_knee", "FX-dwell", False, False),
    ("C3", "merge: plain mean", "b2", "s51", "n6k1_knee_F_unbud", "plain", True, False),
    ("C3", "merge: Async-HFL", "b2", "s51", "n6k1_knee_F_unbud", "asynchfl", True, False),
    ("C3", "merge: FedBuff", "b2", "s51", "n6k1_knee_F_unbud", "fedbuff", True, False),
    ("C3", "merge: + FedProx", "b2", "s51", "n6k1_knee_F_unbud", "cutoff_fedprox", True, False),
    ("C3", "deadline: round (FedCS)", "b2", "s52", "n6k1_stress", "F-round", True, False),
    ("C3", "deadline: preferred (Oort)", "b2", "s52", "n6k1_stress", "F-pref", True, False),
    ("C4", "learned score (5.5)", "b2", "s55", "n12k1_knee", "FQ-g75", True, False),
    ("C5", "MAX-AoI (D1)", "b2", "s58", "n6k1_stress_m4", "D1", True, False),
    ("C5", "Cui's Whittle (D3)", "b2", "s58", "n6k1_stress_m4", "D3", True, False),
    ("C5", "coverage term off", "b2", "s58", "n6k1_stress_m4", "F-cov", True, False),
    ("C5", "age cap off", "b2", "s58", "n6k1_stress_m4", "F-cap", True, False),
    ("5.14", "whole stops only", "b1", "s514", "n6k1_stress", "whole", True, False),
    ("5.14", "no adaptive backhaul", "b1", "s514", "n6k1_knee", "secFL1", True, True),
]


def fig_claims(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL1, 3.6))
    xmax, xmin = 60.0, -35.0
    y = 0
    ticks, labels, groups = [], [], []
    last = None
    for claim, label, sdir, study, cell, variant, lower, flip in CLAIM_ROWS:
        if last is not None and claim != last:
            y -= 0.6
        if claim != last:
            groups.append((y, claim))
        last = claim
        c = comparison(sdir, study, cell, variant)
        ref = float(c["ref_mean"])
        d, lo, hi = float(c["mean_diff"]), float(c["ci_low"]), float(c["ci_high"])
        # right = the reference (FeRRy's design) better
        sgn = -1.0 if lower else 1.0
        vals = sorted(sgn * v / ref * 100 for v in (d, lo, hi))
        point = sgn * d / ref * 100
        if flip:
            point, vals = -point, sorted(-v for v in vals)
        claim_on = is_claim(c)
        color = ARM_COLOR["F"] if claim_on else MUTED
        lo_c, hi_c = max(vals[0], xmin), min(vals[2], xmax)
        ax.plot([lo_c, hi_c], [y, y], color=color, lw=1.3, zorder=3)
        off = point > xmax or point < xmin
        px = min(max(point, xmin), xmax)
        if off:
            ax.scatter([px], [y], s=26, zorder=4, marker=">" if point > xmax else "<",
                       color=color if claim_on else SURFACE, edgecolor=color, linewidth=1.1)
            ax.text(px + (2 if point > xmax else -2), y, f"{point:+.0f}%", va="center",
                    ha="left" if point > xmax else "right", fontsize=6, color=INK2)
        else:
            ax.scatter([px], [y], s=20, zorder=4, color=color if claim_on else SURFACE,
                       edgecolor=color, linewidth=1.1)
        ticks.append(y)
        labels.append(label)
        y -= 1
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(xmin - 6, xmax + 12)
    ax.set_ylim(y + 0.4, 1.35)                 # room for the first group label
    ax.grid(axis="y", visible=False)
    for gy, g in groups:
        ax.text(xmin - 5, gy + 0.5, g, fontsize=6.6, color=INK, fontweight="bold",
                ha="left", va="bottom")
    ax.set_xlabel("how much better FeRRy's design does (% difference)")
    # Direction cues at both ends of the axis, above the plot.
    ax.text(0.0, 1.0, "← alternative better", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=6.4, color=INK2)
    ax.text(1.0, 1.0, "FeRRy's design better →", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.4, color=INK2)
    handles = [Line2D([], [], color=ARM_COLOR["F"], marker="o", lw=1.3, ms=4.5,
                      label="claim (CI excludes 0, Holm p < 0.05)"),
               Line2D([], [], color=MUTED, marker="o", mfc=SURFACE, lw=1.3, ms=4.5,
                      label="no claim")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, -0.32), ncol=1)
    save(fig, out, "fig_exp5_claims")


def fig_compute(out: Path) -> None:
    arms = ["F", "FX", "H1", "D5", "D4"]
    levels = [("none", "no training time"), ("spread", "training spread (median 68 s)"),
              ("stragglers", "+ 20% stragglers at 5x")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 1.9), sharey=True)
    for ax, metric in zip(axes, ("yield", "reach")):
        for j, (lvl, name) in enumerate(levels):
            for i, arm in enumerate(arms):
                e = entry("batch3", "s512", f"n6k1_knee_1mb_{lvl}", arm)
                if metric == "yield":
                    v = statistics.fmean(values(e, "update_yield"))
                else:
                    v = reach_rate(e)
                yy = len(arms) - 1 - i + (1 - j) * 0.22
                ax.scatter([v], [yy], s=20, color=SEQ[j], edgecolor=SURFACE, linewidth=0.8,
                           zorder=3, label=name if i == 0 else None)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels([ARM_LABEL[a] for a in reversed(arms)])
    axes[0].tick_params(axis="y", labelcolor=INK2)
    axes[0].set_xlabel("updates merged per round")
    axes[0].set_title("(a) update yield", color=INK2, loc="left")
    axes[1].set_xlabel(r"share of trials reaching $\tau$")
    axes[1].set_xlim(0.5, 1.03)
    axes[1].set_title(r"(b) reach rate", color=INK2, loc="left")
    axes[0].set_xlim(0, 4)
    axes[1].legend(loc="lower left", bbox_to_anchor=(-1.15, -0.62), ncol=3)
    fig.subplots_adjust(wspace=0.06, bottom=0.3)
    save(fig, out, "fig_exp5_compute")


# --------------------------------------------------------------------------- #
# Why F wins: the mechanism figures
# --------------------------------------------------------------------------- #

TRANSIT, DWELL = "#e34948", "#4a3aa7"      # mission-time components (validated pair)
BAND_RAMP = {"narrow": "#86b6ef", "medium": "#2a78d6", "wide": "#104281"}   # ordinal


def _mean(e, col: str) -> float:
    xs = values(e, col)
    return statistics.fmean(xs) if xs else float("nan")


def _band_shares(e) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    n = 0
    for r in e.scored_rows:
        try:
            d = json.loads(r.get("band_shares") or "{}")
        except json.JSONDecodeError:
            continue
        if d:
            n += 1
            for k, v in d.items():
                acc[k] = acc.get(k, 0.0) + float(v)
    return {k: v / n for k, v in acc.items()} if n else {}


MECH_ROWS = [  # (label, stage, study, cell, variant, kind)
    ("F (band chosen per mission)", "batch2", "s53x", "n6k1_knee", "F", "F"),
    ("FX", "batch2", "s53x", "n6k1_knee", "FX", "FX"),
    ("F, band pinned narrow", "batch2", "s54", "n6k1_knee_1mb", "FBpnarrow", "pin"),
    ("F, band pinned medium", "batch2", "s54", "n6k1_knee_1mb", "FBpmedium", "pin"),
    ("F, band pinned wide", "batch2", "s54", "n6k1_knee_1mb", "FBpwide", "pin"),
    ("D4 (FedEx route)", "batch2", "s53x", "n6k1_knee", "D4", "base"),
    ("D2 (Oort)", "batch2", "s53x", "n6k1_knee", "D2", "base"),
    ("D5 (FedCS)", "batch2", "s53x", "n6k1_knee", "D5", "base"),
    ("D3 (Cui)", "batch2", "s53x", "n6k1_knee", "D3", "base"),
    ("E3 (Chen)", "batch2", "s53x", "n6k1_knee", "E3", "base"),
    ("D1 (MAX-AoI)", "batch2", "s53x", "n6k1_knee", "D1", "base"),
    ("H1", "batch2", "s53x", "n6k1_knee", "H1", "base"),
]


def fig_mechanism(out: Path) -> None:
    """Where F's time goes: mission-time anatomy and time to tau, with F's band
    pinned to each class, against the baselines (all fly the wide band)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, 2.9), sharey=True,
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    n = len(MECH_ROWS)
    for i, (label, stage, study, cell, var, kind) in enumerate(MECH_ROWS):
        y = n - 1 - i
        e = entry(stage, study, cell, var)
        tr, dw = _mean(e, "sim_transit_s_mean"), _mean(e, "sim_dwell_s_mean")
        total = _mean(e, "sim_mission_duration_s_mean")
        other = max(total - tr - dw, 0.0)
        left = 0.0
        for width, color in ((tr, TRANSIT), (dw, DWELL), (other, CONTEXT)):
            ax.barh(y, width, left=left, height=0.62, color=color, edgecolor=SURFACE,
                    linewidth=1.0, zorder=3)
            left += width
        ax.text(total + 4, y, f"{total:.0f}", va="center", fontsize=6.3, color=INK2)
        m, lo, hi = mean_ci(values(e, TAU_COL))
        color = ARM_COLOR.get(kind, INK2) if kind in ("F", "FX") else INK2
        bx.plot([lo, hi], [y, y], color=color, lw=1.4, zorder=3)
        bx.scatter([m], [y], s=22, color=color, zorder=4, edgecolor=SURFACE, linewidth=1.0,
                   marker=ARM_MARKER.get(var, "o") if kind != "pin" else "o")
        bx.text(505, y, f"{m:.0f}", va="center", fontsize=6.3, color=INK2, clip_on=False)
    for k in (1.5, 4.5):                       # separate F/FX, the pinned F, the baselines
        ax.axhline(n - 1 - k, color=GRID, lw=0.8)
        bx.axhline(n - 1 - k, color=GRID, lw=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(MECH_ROWS)])
    ax.tick_params(axis="y", labelcolor=INK2)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("F (") or lbl.get_text() == "FX":
            lbl.set_color(INK)
            lbl.set_fontweight("bold")
    ax.set_xlabel("simulated mission time (s), mean")
    ax.set_title("(a) where a mission's time goes", color=INK2, loc="left")
    ax.set_xlim(0, 250)
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TRANSIT, label="transit to stops"),
               plt.Rectangle((0, 0), 1, 1, color=DWELL, label="dwell (serving devices)"),
               plt.Rectangle((0, 0), 1, 1, color=CONTEXT,
                             label="return, upload, dock turnaround")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, -0.03), ncol=3,
               fontsize=6.5, handlelength=1.0)
    bx.set_xlabel(r"simulated time to $\tau$ = 0.71 (s)")
    bx.set_title(r"(b) time to $\tau$, mean and 95% CI", color=INK2, loc="left")
    bx.set_xlim(0, 500)
    bx.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.12, bottom=0.22)
    save(fig, out, "fig_exp5_mechanism")


BAND_CELLS = [  # (label, stage, study, cell)
    ("N = 6, knee", "batch2", "s53x", "n6k1_knee"),
    ("N = 6, stress", "batch2", "s53x", "n6k1_stress"),
    ("N = 12, knee", "batch1", "s59", "n12k1_knee"),
    ("N = 12, stress", "batch2", "s59x", "n12k1_stress"),
    ("N = 24, knee", "batch1", "s59", "n24k1_knee"),
    ("N = 24, stress", "batch2", "s59x", "n24k1_stress"),
    ("N = 12, K = 3, knee", "batch1", "s511b", "n12k3_knee"),
]


def fig_bands(out: Path) -> None:
    """F's band class by cell: the reach decision shifts as the field and budget change."""
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    n = len(BAND_CELLS)
    for i, (label, stage, study, cell) in enumerate(BAND_CELLS):
        y = n - 1 - i
        shares = _band_shares(entry(stage, study, cell, "F"))
        left = 0.0
        for band in ("narrow", "medium", "wide"):
            w = shares.get(band, 0.0)
            if w <= 0:
                continue
            ax.barh(y, w, left=left, height=0.62, color=BAND_RAMP[band], edgecolor=SURFACE,
                    linewidth=1.0, zorder=3)
            if w >= 0.16:
                ink = INK if band == "narrow" else SURFACE
                ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=6,
                        color=ink, zorder=4)
            left += w
    ax.set_yticks(range(n))
    ax.set_yticklabels([c[0] for c in reversed(BAND_CELLS)])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of F's plans flying each band class")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_RAMP[b], label=f"{b} ({bw})")
               for b, bw in (("narrow", "1.4 MHz"), ("medium", "5 MHz"), ("wide", "20 MHz"))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.36, -0.42), ncol=3,
              fontsize=6.2, handlelength=1.0, columnspacing=1.0)
    save(fig, out, "fig_exp5_bands")


def _win_share(cell: str, arm: str) -> Tuple[float, int]:
    """Share of seeds where F reaches tau first (ties half), over seeds where either reaches."""
    def by_trial(e):
        return {int(r["trial_index"]): SC._num(r.get(TAU_COL)) for r in e.scored_rows}
    f = by_trial(entry("batch2", "s53x", cell, "F"))
    b = by_trial(entry("batch2", "s53x", cell, arm))
    score, n = 0.0, 0
    for t in set(f) & set(b):
        fv, bv = f[t], b[t]
        if fv is None and bv is None:
            continue
        n += 1
        if bv is None or (fv is not None and fv < bv - 1e-9):
            score += 1
        elif fv is not None and bv is not None and abs(fv - bv) <= 1e-9:
            score += 0.5
    return (score / n if n else float("nan")), n


def fig_paired(out: Path) -> None:
    """Seed by seed: how often F reaches tau before each baseline on the same layout."""
    arms = ["D4", "D2", "D5", "D3", "E3", "D1", "H1"]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    for i, arm in enumerate(arms):
        y = len(arms) - 1 - i
        knee, _ = _win_share("n6k1_knee", arm)
        stress, _ = _win_share("n6k1_stress", arm)
        ax.plot([min(knee, stress), max(knee, stress)], [y, y], color=GRID, lw=1.6, zorder=2)
        ax.scatter([knee], [y], s=26, color=ARM_COLOR["F"], edgecolor=SURFACE, linewidth=1.0,
                   zorder=4, label="knee budget" if i == 0 else None)
        ax.scatter([stress], [y], s=26, color=SURFACE, edgecolor=ARM_COLOR["F"], linewidth=1.3,
                   zorder=4, label="stress budget" if i == 0 else None)
    ax.axvline(0.5, color=AXIS, lw=0.8, zorder=1)
    ax.text(0.51, -0.75, "even", ha="left", va="center", fontsize=6, color=MUTED)
    ax.set_ylim(-1.0, len(arms) - 0.5)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ARM_LABEL[a] for a in reversed(arms)])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(0.3, 1.0)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0])
    ax.set_xticklabels(["30%", "50%", "70%", "90%", "100%"])
    ax.set_xlabel(r"share of paired seeds where F reaches $\tau$ first")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=6.3)
    save(fig, out, "fig_exp5_paired")


def fig_budget(out: Path) -> None:
    """Under the stress budget: updates kept per round against the share of missions
    that overran the budget (N = 6, one mule)."""
    arms = ["F", "FX", "H1", "D1", "D2", "D3", "D4", "D5", "E3"]
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    pts = {}
    for arm in arms:
        e = entry("batch2", "s53x", "n6k1_stress", arm)
        pts[arm] = (_mean(e, "sim_budget_overrun_rate"), _mean(e, "update_yield"))
        color = ARM_COLOR.get(arm) if arm in ("F", "FX", "D4") else MUTED
        ax.scatter([pts[arm][0]], [pts[arm][1]], s=30, color=color,
                   marker=ARM_MARKER.get(arm, "o"), edgecolor=SURFACE, linewidth=1.0, zorder=4)
    # F, FX and D4 stand apart: label them beside their marks.
    for arm, (dx, dy, ha) in {"F": (7, 5, "left"), "FX": (-7, 5, "right"),
                              "D4": (-7, -9, "right")}.items():
        ax.annotate(ARM_LABEL[arm], pts[arm], xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.2, color=INK2, ha=ha, va="center")
    # The baselines cluster near zero overrun: a label column with leader lines.
    cluster = sorted((a for a in arms if a not in ("F", "FX", "D4")), key=lambda a: -pts[a][1])
    top, step = 2.95, 0.17
    for i, arm in enumerate(cluster):
        ly = top - i * step
        ax.annotate(ARM_LABEL[arm], pts[arm], xytext=(0.22, ly), textcoords="data",
                    fontsize=6.2, color=INK2, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6,
                                    shrinkA=0, shrinkB=3))
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(1.5, 4.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("missions that overran the budget")
    ax.set_ylabel("updates merged per round")
    save(fig, out, "fig_exp5_budget")


LEARN_ROWS = [  # (label, variant, has FeRRy's plan)
    ("F (plan, committed in flight)", "F", True),
    ("FX (plan + fixed per-stop rule)", "FX", True),
    ("FQ (plan + learned per-stop score)", "FQ-g75", True),
    ("E3 (learned scheduler, no plan)", "E3", False),
]


def fig_learning(out: Path) -> None:
    """Learning inside FeRRy's plan against learning without one (N = 6, one mule):
    time to tau at both budgets, and where a knee mission's time goes."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL2, 2.25), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1]})
    n = len(LEARN_ROWS)
    # The exploratory family's claims (scripts/exp5/fq_vs_e3.py): arms slower than FQ.
    import csv as _csv
    slower = set()
    path = REPO / "results/exp5/scores/b2/exploratory_fq_vs_e3.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r["claim"].lower() in ("true", "1") and r["favours"] == "FQ-g75":
                slower.add((r["cell"], r["reference"]))
    for i, (label, var, planned) in enumerate(LEARN_ROWS):
        y = n - 1 - i
        color = ARM_COLOR["F"] if planned else INK2
        for dy, cell, filled in ((0.14, "n6k1_knee", True), (-0.14, "n6k1_stress", False)):
            e = entry("batch2fq", "s53x", cell, var)
            m, lo, hi = mean_ci(values(e, TAU_COL))
            ax.plot([lo, hi], [y + dy, y + dy], color=color, lw=1.3, zorder=3)
            ax.scatter([m], [y + dy], s=22, zorder=4, color=color if filled else SURFACE,
                       edgecolor=color, linewidth=1.2)
            rr = reach_rate(e)
            mark = r"$^{\ddagger}$" if (cell, var) in slower else ""
            ax.text(505, y + dy, f"{m:.0f}{mark}" + (f" ({rr:.2f})" if rr < 0.995 else ""),
                    va="center", fontsize=6.0, color=INK2, clip_on=False)
        e = entry("batch2fq", "s53x", "n6k1_knee", var)
        tr, dw = _mean(e, "sim_transit_s_mean"), _mean(e, "sim_dwell_s_mean")
        total = _mean(e, "sim_mission_duration_s_mean")
        left = 0.0
        for width, c in ((tr, TRANSIT), (dw, DWELL), (max(total - tr - dw, 0.0), CONTEXT)):
            bx.barh(y, width, left=left, height=0.56, color=c, edgecolor=SURFACE,
                    linewidth=1.0, zorder=3)
            left += width
        shares = _band_shares(e)
        band = (f"narrow {shares.get('narrow', 0):.0%}" if planned else "wide (fixed)")
        bx.text(total + 4, y, f"{total:.0f} s, {band}", va="center", fontsize=6.0, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(LEARN_ROWS)])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_xlim(0, 500)
    ax.set_xlabel(r"simulated time to $\tau$ = 0.71 (s)")
    ax.set_title(r"(a) time to $\tau$, mean and 95% CI", color=INK2, loc="left")
    ax.grid(axis="y", visible=False)
    bx.set_xlim(0, 330)
    bx.set_xlabel("simulated mission time at the knee (s), mean")
    bx.set_title("(b) where a mission's time goes", color=INK2, loc="left")
    bx.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=INK2, marker="o", lw=1.3, ms=4.5, label="knee budget"),
               Line2D([], [], color=INK2, marker="o", mfc=SURFACE, lw=1.3, ms=4.5,
                      label="stress budget"),
               plt.Rectangle((0, 0), 1, 1, color=TRANSIT, label="transit"),
               plt.Rectangle((0, 0), 1, 1, color=DWELL, label="dwell"),
               plt.Rectangle((0, 0), 1, 1, color=CONTEXT, label="return, upload, dock")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=5,
               fontsize=6.4, handlelength=1.3, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.22, bottom=0.27)
    save(fig, out, "fig_exp5_learning")


FIGURE_TEX = r"""% Experiment 5 figures, generated by scripts/exp5/paper_figures.py (the PDFs) and
% written here with their captions. Copy results/exp5/paper/figures/*.pdf into the
% paper's Figures/ folder.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/fig_exp5_convergence.pdf}
\caption{Test accuracy over simulated time at $N=6$ with one mule (Study~5.3), the mean
of 20 paired trials per arm, at the knee (150\,s) and stress (75\,s) budgets. F and FX
cross $\tau=0.71$ first; gray lines are the seven baselines (H1, D1--D5, E3).}
\label{fig:exp5_convergence}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/fig_exp5_tau.pdf}
\caption{Time to $\tau$ at $N=6$, one mule (Study~5.3): mean and 95\% bootstrap CI over
the trials that reach $\tau$; dots are single trials; the reach rate is given where it
is below 1. $^{\dagger}$Slower than F (paired by seed, Holm $p<0.05$).}
\label{fig:exp5_tau}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/fig_exp5_scale.pdf}
\caption{Time to $\tau$ as the device population grows (Study~5.9), one mule, each $N$
at its own knee and stress budget: mean and 95\% bootstrap CI over the trials that reach
$\tau$, 20 paired trials per arm. Markers are offset horizontally for legibility.}
\label{fig:exp5_scale}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/fig_exp5_systems.pdf}
\caption{Systems scaling (Study~5.11). (a)~Planner wall time per plan against $N$, mean
and p95, one host running alone. (b)~FerrySim's share of devices served at each $N$'s
knee budget under F, FX and greedy-1 (30 episodes). (c)~Time to $\tau$ at $N=12$ as mules
are added, mean and 95\% bootstrap CI.}
\label{fig:exp5_systems}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{Figures/fig_exp5_claims.pdf}
\caption{Each design-claim test compares FeRRy's design with one alternative (a baseline,
a component removed, or another rule): the paired difference as a percentage of FeRRy's
value, so that right of zero means FeRRy's design does better, with its 95\% bootstrap CI.
Filled: a claim
(CI excludes 0 and Holm $p<0.05$ within the study). Tests and metrics as in
Table~\ref{tab:exp5_claims}.}
\label{fig:exp5_claims}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/fig_exp5_mechanism.pdf}
\caption{Where F's advantage comes from ($N=6$, one mule, knee budget). (a)~Mean simulated
mission time by component. Every baseline flies the wide band and spends most of a mission
in transit between stops; F mostly plans the narrow band, whose range reaches the field from
about one stop, and trades transit for dwell. Pinning F to the wide band removes its
advantage. (b)~Time to $\tau$ for the same rows: mean and 95\% bootstrap CI over the trials
that reach $\tau$. The pinned-band rows come from Study~5.4, whose pre-registered metric is
the network age of updates; their time to $\tau$ is descriptive.}
\label{fig:exp5_mechanism}
\end{figure*}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{Figures/fig_exp5_bands.pdf}
\caption{Share of F's plans flying each band class, by cell. With few devices F plans the
long-reach narrow band; as the field fills and the budget tightens, it shifts to the faster
medium and wide classes.}
\label{fig:exp5_bands}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{Figures/fig_exp5_paired.pdf}
\caption{Seed-paired comparison at $N=6$, one mule: the share of the 20 paired seeds (same
layout, channel and data) in which F reaches $\tau$ before each baseline. A seed in which
only F reaches $\tau$ counts as a win, and a tie as half.}
\label{fig:exp5_paired}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{Figures/fig_exp5_budget.pdf}
\caption{Updates merged per round against the share of missions that overran the budget,
at $N=6$ under the stress budget. The baselines stay within the budget by leaving devices
out. FedEx's visit-all tour (D4) keeps its yield by overrunning the budget in @D4_OVER@\%
of missions. F keeps nearly D4's yield by flying fewer stops, but its plans, priced at the
mean SNR, overrun in @F_OVER@\% of missions, by @F_OVER_S@\,s on average.}
\label{fig:exp5_budget}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/fig_exp5_learning.pdf}
\caption{Learning inside FeRRy's plan against learning without one ($N=6$, one mule, 20
paired seeds; exploratory, not pre-registered). F commits the plan in flight, FX adds the
fixed per-stop rule, FQ the learned per-stop score (Study~5.5's $\gamma=0.75$ pick), and E3
is Chen et al.'s learned scheduler with no plan. (a)~Time to $\tau$ at the knee (filled) and
stress (hollow) budgets: mean and 95\% bootstrap CI over the trials that reach $\tau$, with
the reach rate in brackets when it is below 1; $^{\ddagger}$slower than FQ (paired by seed,
Holm $p<0.05$ across the exploratory comparisons of FQ with E3, F, FX and D4 at both
budgets). (b)~Mean mission time at the knee by
component, and the band each arm flies: the three FeRRy variants share the plan's
narrow-band reach, while E3 flies the wide band.}
\label{fig:exp5_learning}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/fig_exp5_compute.pdf}
\caption{Device training time (Study~5.12, $N=6$, knee budget): updates merged per round
and the share of trials reaching $\tau$ with no training time, a seeded spread (median
68\,s) and 20\% stragglers at $5\times$. F's plan does not price training time: its yield
falls to @F_YIELD@ updates per round and it reaches $\tau$ in @F_REACH@\% of
trials, while H1, D4 and D5 keep 85--100\%.}
\label{fig:exp5_compute}
\end{figure*}
"""


def write_tex(out: Path) -> None:
    yields = [statistics.fmean(values(entry("batch3", "s512", f"n6k1_knee_1mb_{lvl}", "F"),
                                      "update_yield")) for lvl in ("spread", "stragglers")]
    reaches = [reach_rate(entry("batch3", "s512", f"n6k1_knee_1mb_{lvl}", "F")) * 100
               for lvl in ("spread", "stragglers")]
    stress_f = entry("batch2", "s53x", "n6k1_stress", "F")
    stress_d4 = entry("batch2", "s53x", "n6k1_stress", "D4")
    text = (FIGURE_TEX.replace("@F_YIELD@", f"{max(yields):.1f}")
            .replace("@F_REACH@", f"{min(reaches):.0f}--{max(reaches):.0f}")
            .replace("@D4_OVER@", f"{_mean(stress_d4, 'sim_budget_overrun_rate') * 100:.0f}")
            .replace("@F_OVER@", f"{_mean(stress_f, 'sim_budget_overrun_rate') * 100:.0f}")
            .replace("@F_OVER_S@", f"{_mean(stress_f, 'sim_budget_overrun_s_mean'):.1f}"))
    path = out.parent / "exp5_figures.tex"
    path.write_text(text, encoding="utf-8")
    print(f"  {path.relative_to(REPO)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/exp5/paper/figures")
    ap.add_argument("--only", nargs="*", default=None, help="figure function names")
    a = ap.parse_args(argv)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    style()
    figs = {"convergence": fig_convergence, "tau": fig_tau, "scale": fig_scale,
            "systems": fig_systems, "claims": fig_claims, "compute": fig_compute,
            "mechanism": fig_mechanism, "bands": fig_bands, "paired": fig_paired,
            "budget": fig_budget, "learning": fig_learning}
    for name, fn in figs.items():
        if a.only and name not in a.only:
            continue
        fn(out)
    if not a.only:
        write_tex(out)
    print(f"figures in {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
