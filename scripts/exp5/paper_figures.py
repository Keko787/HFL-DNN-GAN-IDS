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
}

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
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
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
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build(stage, s, studies, "results/exp5")
    _, by_study, _, _ = L.score_plan(stage, s, jobs)
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
    ax.set_ylim(y + 0.4, 0.8)
    ax.grid(axis="y", visible=False)
    for gy, g in groups:
        ax.text(xmin - 5, gy + 0.5, g, fontsize=6.6, color=INK, fontweight="bold",
                ha="left", va="bottom")
    ax.set_xlabel("effect, % of reference  (right: FeRRy's design better)")
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
\caption{Each design-claim test as the paired effect relative to its reference, oriented
so that right means FeRRy's design is better, with its 95\% bootstrap CI. Filled: a claim
(CI excludes 0 and Holm $p<0.05$ within the study). Tests and metrics as in
Table~\ref{tab:exp5_claims}.}
\label{fig:exp5_claims}
\end{figure}

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
    text = (FIGURE_TEX.replace("@F_YIELD@", f"{max(yields):.1f}")
            .replace("@F_REACH@", f"{min(reaches):.0f}--{max(reaches):.0f}"))
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
            "systems": fig_systems, "claims": fig_claims, "compute": fig_compute}
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
