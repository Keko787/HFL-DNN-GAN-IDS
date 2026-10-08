"""Exp 5's results as LaTeX tables for the paper, read from the scored studies.

    py -3.11 scripts/exp5/paper_tables.py [--out results/exp5/paper/exp5_tables.tex]

Every number comes from results/exp5/scores/<batch>/<study>_{arms,comparisons}.csv
(``launch.py score``), the 5.11 (a) rerun's and 5.11 (c)'s FerrySim reports, O1's
report and H0's trial CSV, so the tables regenerate whenever a study is rescored.
Time to tau is the simulated seconds to tau = 0.71 (``sim_s_to_tau0.71``) over the
trials that reach it; a claim is the scorer's (paired seeds, bootstrap CI excludes
0, Holm p < 0.05). The tables use booktabs and \\resizebox (graphicx), as the paper
does, and cite the paper's own keys; two baselines have no key there yet
(MAX-AoI, FedAsync), marked TODO.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
SCORES = REPO / "results" / "exp5" / "scores"
TAU = "0.71"
DAG = r"$^{\dagger}$"          # significantly worse than the reference (Holm claim)
STAR = r"$^{\ast}$"            # significantly better than the reference (Holm claim)


# --------------------------------------------------------------------------- #
# Reading the scores
# --------------------------------------------------------------------------- #

def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Study:
    """One scored study: its variants per cell, and each one's comparison."""

    def __init__(self, batch: str, study: str):
        self.name = study
        self.arms = {(r["cell"], r["variant"]): r
                     for r in _rows(SCORES / batch / f"{study}_arms.csv")}
        comp = SCORES / batch / f"{study}_comparisons.csv"
        self.comps = ({(r["cell"], r["variant"]): r for r in _rows(comp)}
                      if comp.exists() else {})

    def arm(self, cell: str, variant: str) -> Dict[str, str]:
        return self.arms[(cell, variant)]

    def comp(self, cell: str, variant: str) -> Optional[Dict[str, str]]:
        return self.comps.get((cell, variant))


def _f(v, digits: int = 0) -> Optional[float]:
    try:
        return round(float(v), digits) if digits else float(v)
    except (TypeError, ValueError):
        return None


def claim_mark(c: Optional[Dict[str, str]], reference: str) -> str:
    if not c or c.get("claim", "").lower() not in ("true", "1"):
        return ""
    return DAG if c.get("favours") == reference else STAR


def tau_cell(st: Study, cell: str, variant: str, reference: Optional[str] = None,
             bold: bool = False) -> str:
    """Mean time to tau (s), the reach rate in brackets when below 1, and the claim mark."""
    a = st.arm(cell, variant)
    mean = _f(a.get("metric_mean"))
    reach = _f(a.get(f"reach_rate_tau{TAU}"))
    if mean is None:
        return "--"
    text = f"{mean:.0f}"
    if bold:
        text = rf"\textbf{{{text}}}"
    if reach is not None and reach < 0.995:
        text += rf"\,{{\scriptsize({reach:.2f})}}"
    if reference:
        text += claim_mark(st.comp(cell, variant), reference)
    return text


def cmp_row(st: Study, cell: str, variant: str, digits: int = 2) -> Tuple[str, str, str, str, str]:
    """(reference mean, variant mean, diff [CI], Holm p, verdict) of one comparison."""
    c = st.comp(cell, variant)
    if c is None:
        raise KeyError((st.name, cell, variant))
    fmt = (lambda v: f"{float(v):.{digits}f}")
    verdict = "--"
    if c.get("claim", "").lower() in ("true", "1"):
        verdict = c["favours"]
    diff = (f"{fmt(c['mean_diff'])} [{fmt(c['ci_low'])}, {fmt(c['ci_high'])}]"
            if c.get("mean_diff") not in ("", None) else "--")
    p = c.get("p_holm")
    return (fmt(c["ref_mean"]), fmt(c["variant_mean"]), diff,
            "--" if p in ("", None) else f"{float(p):.3f}", verdict)


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #

ARMS_TABLE = r"""
\begin{table}[t]
\centering
\caption{Arms compared in Experiment~5. FeRRy's arms share the dock-time plan; the
baselines are whole schedulers flown on the same simulator, budgets and paired seeds.}
\label{tab:exp5_arms}
\small
\begin{tabularx}{\columnwidth}{l X}
\toprule
\textbf{Arm} & \textbf{Description} \\
\midrule
F & FeRRy: dock-time plan (band class, route, deadline-gated admission, coverage term, age cap $S$), committed in flight. \\
FX & F with the cross-layer in-flight rule: at each stop, the nearest stop that keeps the plan feasible, at the fastest band that still reaches the planned targets. \\
H1 & HERMES's staged heuristic: deadline-tiered contact regions, no dock-time plan. \\
H0 & Live-link reference: FL over the degraded infrastructure link, no mule. \\
D1 & MAX-AoI: visit the stalest device next \cite{TODO-maxaoi}. \\
D2 & Oort-style statistical utility \cite{lai2021oort}. \\
D3 & Whittle index on value-weighted age of updates \cite{cui2023data}, our expected-connectivity variant. \\
D4 & FedEx's visit-all 2-OPT tour \cite{bian2025indirect}, one transporter, never skips; D4$_{\text{FedEx}}$ adds its $1/N$ delta merge. \\
D5 & FedCS, degraded: round-deadline selection without its resource-request phase \cite{nishio2019client}. \\
E3 & Single-agent next-stop DQN after Chen et al.\ \cite{chen2023model}, trained in FerrySim. \\
\bottomrule
\end{tabularx}
\end{table}
""".strip("\n")

HEADLINE_ARMS = [("F", "F"), ("FX", "FX"), ("H1", "H1"), ("D1", "D1 (MAX-AoI)"),
                 ("D2", "D2 (Oort)"), ("D3", "D3 (Cui)"), ("D4", "D4 (FedEx route)"),
                 ("D4fedex", r"D4$_{\text{FedEx}}$"), ("D5", "D5 (FedCS)"), ("E3", "E3 (Chen)")]
HEADLINE_CELLS = [("n6k1_knee", "knee"), ("n6k1_stress", "stress"),
                  ("n6k3_knee", "knee"), ("n6k3_stress", "stress")]


def headline_table() -> str:
    st = Study("b2", "s53x")
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Whole-scheduler comparison at $N=6$ (Study~5.3): mean simulated "
             r"time to $\tau=0.71$ in seconds (lower is better), 20 paired trials per cell; "
             r"reach rate in brackets when below 1. " + DAG + r" significantly slower than F "
             r"(paired bootstrap CI excludes 0, Holm $p<0.05$). D4 flies its whole tour "
             r"whatever the budget, so its two budget columns coincide.}",
             r"\label{tab:exp5_headline}",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l rr rr}",
             r"\toprule",
             r" & \multicolumn{2}{c}{$K=1$ mule} & \multicolumn{2}{c}{$K=3$ mules} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"\textbf{Arm} & knee & stress & knee & stress \\",
             r"\midrule"]
    for variant, label in HEADLINE_ARMS:
        cells = [tau_cell(st, cell, variant, reference="F", bold=(variant == "F"))
                 for cell, _ in HEADLINE_CELLS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if variant == "FX":
            lines.append(r"\midrule")
    # H0 flies the wall clock: described beside F, never paired.
    h0 = _rows(REPO / "results/exp5/b2/s53x/n6_wall__H0.csv")
    h0_ok = [r for r in h0 if r.get("status", "ok") == "ok"]
    h0_acc = statistics.mean(float(r["final_accuracy"]) for r in h0_ok)
    h0_yield = statistics.mean(float(r["update_yield"]) for r in h0_ok)
    f = st.arm("n6k1_knee", "F")
    lines += [r"\midrule",
              rf"\multicolumn{{5}}{{l}}{{\footnotesize H0 (no mule, live link): "
              rf"{h0_yield:.1f} updates/round, final accuracy {h0_acc:.3f}; "
              rf"F at the $K=1$ knee: {float(f['mean_update_yield']):.1f} updates/round, "
              rf"{float(f['mean_final_accuracy']):.3f}.}} \\",
              r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


SCALE_ARMS = [("F", "F"), ("FX", "FX"), ("H1", "H1"), ("D3", "D3 (Cui)"),
              ("D4", "D4 (FedEx route)")]


def scale_table() -> str:
    core = Study("b1", "s59")          # the knee, N = 6, 12, 24
    ext = Study("b2", "s59x")          # the stress budget, N = 12, 24
    h53 = Study("b2", "s53x")          # N = 6 stress
    cols = [(core, "n6k1_knee"), (core, "n12k1_knee"), (core, "n24k1_knee"),
            (h53, "n6k1_stress"), (ext, "n12k1_stress"), (ext, "n24k1_stress")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Scale (Study~5.9): mean time to $\tau$ (s) with one mule as the "
             r"device population grows, at each $N$'s knee and stress budgets. Notation as in "
             r"Table~\ref{tab:exp5_headline}.}",
             r"\label{tab:exp5_scale}",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l rrr rrr}",
             r"\toprule",
             r" & \multicolumn{3}{c}{knee budget} & \multicolumn{3}{c}{stress budget} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"\textbf{Arm} & $N=6$ & $12$ & $24$ & $N=6$ & $12$ & $24$ \\",
             r"\midrule"]
    for variant, label in SCALE_ARMS:
        cells = [tau_cell(stx, cell, variant, reference="F", bold=(variant == "F"))
                 for stx, cell in cols]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def decision_cost_table() -> str:
    """5.11 (a): the planner's time per plan, alone on the host (the 7 Oct rerun);
    5.11 (c): FerrySim at each scale cell's knee, every in-flight rule."""
    quiet = json.loads((REPO / "results/exp5/s511a_quiet/b1/s511a/auto.json")
                       .read_text(encoding="utf-8"))
    plan = {r["cell"]: r for r in quiet["table"]}
    cells = [("jit-n6-150", 6), ("jit-n12-90", 12), ("scl-n24-350", 24),
             ("scl-n48-680", 48), ("scl-n96-1330", 96)]
    ferry = {}
    for cell in ("scl-n24-350", "scl-n48-680", "scl-n96-1330"):
        rep = json.loads((REPO / f"results/exp5/b3/s511c/{cell}.json").read_text(encoding="utf-8"))
        ferry[cell] = {r["policy"]: r for r in rep["table"]}
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Decision cost and scale (Study~5.11). Planner wall time per plan "
             r"(mean / p95, s; one host, alone); for $N\ge 24$, FerrySim's served share of "
             r"devices at each $N$'s knee budget under F and FX, 30 episodes.}",
             r"\label{tab:exp5_cost}",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{r rr rrr}",
             r"\toprule",
             r" & \multicolumn{2}{c}{plan time (s)} & \multicolumn{3}{c}{FerrySim at the knee} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
             r"$N$ & mean & p95 & budget (s) & F served & FX served \\",
             r"\midrule"]
    for cell, n in cells:
        p = plan[cell]
        row = [str(n), f"{p['plan_wall_s_mean']:.3f}", f"{p['plan_wall_s_p95']:.3f}"]
        if cell in ferry:
            fr = ferry[cell]
            row += [f"{fr['F']['budget_s']:.0f}", f"{fr['F']['served_share_mean']:.2f}",
                    f"{fr['FX']['served_share_mean']:.2f}"]
        else:
            row += ["--", "--", "--"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def mules_table() -> str:
    """5.11 (b): weak scaling (six devices a mule) and strong scaling (N = 12)."""
    st = Study("b1", "s511b")
    cols = [("n6k1_knee", "6/1"), ("n12k2_knee", "12/2"), ("n18k3_knee", "18/3"),
            ("n12k1_knee", "12/1"), ("n12k3_knee", "12/3")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Scaling out with mules (Study~5.11b), knee budget: mean time to $\tau$ "
             r"(s). Weak scaling holds six devices per mule ($N/K$ = 6/1, 12/2, 18/3); strong "
             r"scaling adds mules at $N=12$. Notation as in Table~\ref{tab:exp5_headline}.}",
             r"\label{tab:exp5_mules}",
             r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{l rrr rr}",
             r"\toprule",
             r" & \multicolumn{3}{c}{weak scaling} & \multicolumn{2}{c}{strong scaling} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"\textbf{Arm} & " + " & ".join(f"$N/K$={c}" for _, c in cols) + r" \\",
             r"\midrule"]
    for variant, label in [("F", "F"), ("FX", "FX"), ("H1", "H1"), ("D4", "D4 (FedEx route)")]:
        cells = [tau_cell(st, cell, variant, reference="F", bold=(variant == "F"))
                 for cell, _ in cols]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


def claims_table() -> str:
    """One row per design claim's test: the reference, the variant, both means,
    the paired difference (reference - variant) with its CI, Holm p and verdict."""
    s51, s52 = Study("b2", "s51"), Study("b2", "s52")
    s54, s57, s58 = Study("b2", "s54"), Study("b2", "s57"), Study("b2", "s58")
    s55, s514 = Study("b2", "s55"), Study("b1", "s514")
    o1 = json.loads((REPO / "results/exp5/b2/s54/o1_base.json").read_text(encoding="utf-8"))
    rows = [
        # (claim, test, metric, study, cell, variant, digits)
        ("C1", "band class pinned wide vs F", "AoU", s54, "n6k1_knee_1mb", "FBpwide", 3),
        ("C1", "band class pinned narrow vs F", "AoU", s54, "n6k1_knee_1mb", "FBpnarrow", 3),
        ("C2", "coverage term off (FX-cov)", "close", s57, "n12k1_knee", "FX-cov", 2),
        ("C2", "dwell term off (FX-dwell)", "close", s57, "n12k1_knee", "FX-dwell", 2),
        ("C3", "plain mean vs cutoff, F route", r"$t_\tau$", s51, "n6k1_knee_F_unbud", "plain", 0),
        ("C3", "Async-HFL vs cutoff, F route", r"$t_\tau$", s51, "n6k1_knee_F_unbud", "asynchfl", 0),
        ("C3", "FedBuff vs cutoff, F route", r"$t_\tau$", s51, "n6k1_knee_F_unbud", "fedbuff", 0),
        ("C3", "+FedProx vs cutoff, F route", r"$t_\tau$", s51, "n6k1_knee_F_unbud", "cutoff_fedprox", 0),
        ("C3", "round deadline (FedCS) vs F", "miss", s52, "n6k1_stress", "F-round", 3),
        ("C3", "preferred duration (Oort) vs F", "miss", s52, "n6k1_stress", "F-pref", 3),
        ("C4", "learned score ($\\gamma{=}0.75$) vs FX", r"$t_\tau$", s55, "n12k1_knee", "FQ-g75", 0),
        ("C5", "MAX-AoI (D1) vs F, stress", "AoU", s58, "n6k1_stress_m4", "D1", 3),
        ("C5", "Cui's Whittle (D3) vs F, stress", "AoU", s58, "n6k1_stress_m4", "D3", 3),
        ("C5", "coverage term off (F-cov), stress", "AoU", s58, "n6k1_stress_m4", "F-cov", 3),
        ("C5", "age cap off (F-cap), stress", "AoU", s58, "n6k1_stress_m4", "F-cap", 3),
        ("--", "whole stops vs member subsets, stress", "AoU", s514, "n6k1_stress", "whole", 3),
        ("--", "adaptive backhaul (F+L1) vs F", "AoU", s514, "n6k1_knee", "secFL1", 3),
    ]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Tests of FeRRy's design claims ($N=6$ unless noted; 5.7 and 5.5 at "
             r"$N=12$). Difference is reference $-$ variant, paired by seed, with its 95\% "
             r"bootstrap CI; AoU = mean network age of updates (lower is better), close = "
             r"round-close rate, miss = deadline-miss rate, $t_\tau$ = time to $\tau$ (s). "
             r"Means are over complete pairs. A claim needs the CI to exclude 0 and Holm "
             r"$p<0.05$ within the study.}",
             r"\label{tab:exp5_claims}",
             r"\small",
             r"\begin{tabular}{l l l rr r r l}",
             r"\toprule",
             r"\textbf{Claim} & \textbf{Test} & \textbf{Metric} & \textbf{Ref.} & "
             r"\textbf{Variant} & \textbf{Difference [95\% CI]} & \textbf{Holm $p$} & "
             r"\textbf{Verdict} \\",
             r"\midrule"]
    last = None
    for claim, test, metric, st, cell, variant, digits in rows:
        ref_m, var_m, diff, p, verdict = cmp_row(st, cell, variant, digits)
        if last is not None and claim != last:
            lines.append(r"\midrule")
        last = claim
        verdict = {"--": "no difference"}.get(verdict, f"favours {verdict}")
        verdict = (verdict.replace("FBpwide", "FB+wide").replace("secFL1", "F+L1")
                   .replace("cutoff", "cutoff").replace("capS", "F"))
        lines.append(f"{claim} & {test} & {metric} & {ref_m} & {var_m} & {diff} & {p} & "
                     f"{verdict}" + r" \\")
    v = json.loads((REPO / "results/exp5/rl/s55/verdict.json")
                   .read_text(encoding="utf-8"))["verdict"]
    best = max(v["means"].values())
    s = o1["summary"]
    lines += [r"\midrule",
              rf"\multicolumn{{8}}{{l}}{{\footnotesize C4, pre-registered FerrySim sweep "
              rf"(Study~5.5; 60 trainings, six $\gamma$ from 0 to 0.99): {v['outcome']}, no "
              rf"$\gamma$ beats $\gamma=0$ by $\epsilon={v['epsilon']:g}$. Held-out return: best "
              rf"learned {best:.4f}, FX {v['references']['FX']:.4f}, greedy-1 "
              rf"{v['references']['greedy_1']:.4f}.}} \\",
              rf"\multicolumn{{8}}{{l}}{{\footnotesize C1, optimality: against an exhaustive "
              rf"oracle (O1) over band class, clustering, route and per-stop band, F's plan "
              rf"serves the same share of devices at the knee (gap "
              rf"{s['jit-n6-150']['gap_share_at_key_mean']:.2f}) and "
              rf"{s['jit-n6-75']['gap_share_at_key_mean']:.2f} less at the stress budget.}} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def robustness_table() -> str:
    s512, s513, s515 = Study("b3", "s512"), Study("b2", "s513"), Study("b3", "s515")
    cols = [
        (s512, "n6k1_knee_1mb_spread", "train spread"),
        (s512, "n6k1_knee_1mb_stragglers", "stragglers"),
        (s513, "n6k1_knee_dirichlet0.1", r"Dir(0.1)"),
        (s515, "n6k1_knee_base", "default"),
        (s515, "n6k1_knee_ampharsh", "17 dB interf."),
        (s515, "n6k1_knee_npl", r"$n_{\text{pl}}{=}2.7$"),
        (s515, "n6k1_knee_sigma8", r"$\sigma{=}8$ dB"),
    ]
    arms = [("F", "F"), ("FX", "FX"), ("H1", "H1"), ("D3", "D3 (Cui)"),
            ("D4", "D4 (FedEx route)"), ("D5", "D5 (FedCS)")]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Robustness at $N=6$, the knee budget: device training time (Study~5.12; "
             r"median 68\,s, and with 20\% of devices $5\times$ slower), non-IID data "
             r"(Study~5.13, Dirichlet $\alpha=0.1$) and harsher channels (Study~5.15). Mean time "
             r"to $\tau$ (s), reach rate in brackets when below 1; notation as in "
             r"Table~\ref{tab:exp5_headline}. Means are over trials that reach $\tau$: under "
             r"training time F reaches it in fewer trials.}",
             r"\label{tab:exp5_robust}",
             r"\small",
             r"\begin{tabular}{l rr r rrrr}",
             r"\toprule",
             r" & \multicolumn{2}{c}{compute (5.12)} & data (5.13) & "
             r"\multicolumn{4}{c}{channel (5.15)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-8}",
             r"\textbf{Arm} & " + " & ".join(c[2] for c in cols) + r" \\",
             r"\midrule"]
    for variant, label in arms:
        cells = []
        for st, cell, _ in cols:
            if (cell, variant) in st.arms:
                cells.append(tau_cell(st, cell, variant, reference="F", bold=(variant == "F")))
            else:
                cells.append("--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    yields = {v: float(s512.arm("n6k1_knee_1mb_spread", v)["mean_update_yield"])
              for v in ("F", "H1", "D5")}
    lines += [r"\midrule",
              rf"\multicolumn{{8}}{{l}}{{\footnotesize Updates merged per round under the "
              rf"training spread: F {yields['F']:.2f}, H1 {yields['H1']:.2f}, "
              rf"D5 {yields['D5']:.2f} (F 3.54 without training time).}} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/exp5/paper/exp5_tables.tex")
    a = ap.parse_args(argv)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("% Experiment 5 tables, generated by scripts/exp5/paper_tables.py from\n"
              "% results/exp5/scores (do not edit by hand; rerun the script).\n"
              "% Needs booktabs, tabularx, graphicx (all loaded by the paper).\n"
              "% TODO keys: MAX-AoI (D1) has no entry in biblio.bib yet.\n")
    parts = [ARMS_TABLE, headline_table(), scale_table(), mules_table(),
             decision_cost_table(), claims_table(), robustness_table()]
    out.write_text(header + "\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)} ({len(parts)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
