"""Experiment-4 statistical analysis — paired treatment vs baseline arm.

The trial grid runs both arms at **paired seeds** (same cell + trial_index
→ same device reliabilities, same shards, same init θ), so per-trial
differences are meaningful. For each regime × metric this module forms the
paired (treatment − baseline) differences across all seeds and reports:

* the **paired Wilcoxon** signed-rank p-value + **Cliff's δ** effect size
  (reusing :mod:`experiments.analysis.stats`), and
* a **bootstrap 95% CI on the mean difference**.

A claim is only made when the CI excludes 0 (and Wilcoxon agrees):
``<treat> > <base>`` if the whole CI is positive, ``<base> > <treat>`` if
negative, else ``tie``. This is what turns the single-run illustration into a
defensible result — the crossover sign must survive at the seed level.

Default pair is **H1 (mule) vs H0 (flat FL)** — the participation claim.
Pass ``--treatment H3 --baseline H2`` for the **L1 adaptive-channel** claim.

Usage::

    python -m experiments.analysis.exp4 --csv results/exp4_sweep.csv
    python -m experiments.analysis.exp4 --csv h2h3_l1.csv --treatment H3 --baseline H2
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.analysis.stats import bootstrap_ci, paired_wilcoxon_with_cliffs_delta

# Higher-is-better metrics: a positive (treatment − baseline) diff favours
# the treatment arm.
DEFAULT_METRICS = (
    "final_auc",
    "final_accuracy",
    "mission_completion_rate",
    "update_yield",
    "round_close_rate_kmin2",
)
# Within a regime, these + trial_index identify a paired trial. dead_zone and
# link_quality MUST be here: without them a merged multi-cell surface CSV would
# mispair a treatment row from one cell with a baseline row from another (same
# N/rrf/n_missions/seed). Absent columns are filtered out per-CSV, so a
# single-cell CSV still pairs on whatever keys it has.
PAIR_KEYS = (
    "param_N", "param_rrf", "param_n_missions",
    "param_dead_zone", "param_link_quality", "trial_index",
)


@dataclass(frozen=True)
class PairedMetricResult:
    regime: str
    metric: str
    n_pairs: int
    treatment: str
    baseline: str
    mean_treat: float
    mean_base: float
    mean_diff: float   # treatment − baseline
    ci_lo: float
    ci_hi: float
    p_value: float
    cliffs_delta: float
    magnitude: str
    verdict: str       # "<treat> > <base>" | "<base> > <treat>" | "tie"


def load_exp4(csv_path) -> pd.DataFrame:
    """Load a trial CSV, keeping only status=ok rows."""
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "ok"].copy()
    return df


def _paired_arrays(
    df: pd.DataFrame, regime: str, metric: str,
    *, treatment: str = "H1", baseline: str = "H0",
):
    """Return matched (treatment, baseline) value arrays for a regime × metric."""
    d = df[df["param_regime"].astype(str) == regime]
    keys = [k for k in PAIR_KEYS if k in d.columns]
    t = (
        d[d["arm"] == treatment].set_index(keys)[metric]
        .apply(pd.to_numeric, errors="coerce")
    )
    b = (
        d[d["arm"] == baseline].set_index(keys)[metric]
        .apply(pd.to_numeric, errors="coerce")
    )
    joined = pd.concat([t.rename("t"), b.rename("b")], axis=1).dropna()
    return joined["t"].to_numpy(dtype=float), joined["b"].to_numpy(dtype=float)


def analyze_metric(
    df: pd.DataFrame, regime: str, metric: str,
    *, treatment: str = "H1", baseline: str = "H0", n_bootstraps: int = 2000,
) -> Optional[PairedMetricResult]:
    t, b = _paired_arrays(df, regime, metric, treatment=treatment, baseline=baseline)
    if t.size < 2:
        return None
    diff = t - b
    test = paired_wilcoxon_with_cliffs_delta(t, b)
    _, lo, hi = bootstrap_ci(diff, np.mean, n_bootstraps=n_bootstraps)
    if lo > 0.0:
        verdict = f"{treatment} > {baseline}"
    elif hi < 0.0:
        verdict = f"{baseline} > {treatment}"
    else:
        verdict = "tie"
    return PairedMetricResult(
        regime=regime, metric=metric, n_pairs=int(t.size),
        treatment=treatment, baseline=baseline,
        mean_treat=float(t.mean()), mean_base=float(b.mean()),
        mean_diff=float(diff.mean()), ci_lo=float(lo), ci_hi=float(hi),
        p_value=test.p_value, cliffs_delta=test.cliffs_delta,
        magnitude=test.delta_magnitude, verdict=verdict,
    )


def analyze(
    df: pd.DataFrame, metrics: Sequence[str] = DEFAULT_METRICS,
    *, treatment: str = "H1", baseline: str = "H0",
) -> List[PairedMetricResult]:
    results: List[PairedMetricResult] = []
    for regime in sorted(df["param_regime"].astype(str).unique()):
        for metric in metrics:
            if metric not in df.columns:
                continue
            r = analyze_metric(
                df, regime, metric, treatment=treatment, baseline=baseline,
            )
            if r is not None:
                results.append(r)
    return results


def analyze_surface(
    df: pd.DataFrame,
    *,
    metric: str = "final_auc",
    regime: str = "jittery",
    keys: Sequence[str] = ("param_dead_zone", "param_link_quality"),
    treatment: str = "H1",
    baseline: str = "H0",
) -> List[PairedMetricResult]:
    """Paired verdict at each (dead_zone x link_quality) point — the B2 surface.

    Shows where the mule's jittery advantage holds, is a tie, or flips, so
    the headline operating point is not a single tuned value.
    """
    d = df[df["param_regime"].astype(str) == regime]
    keys = [k for k in keys if k in d.columns]
    if not keys:
        r = analyze_metric(d, regime, metric, treatment=treatment, baseline=baseline)
        return [r] if r else []
    results: List[PairedMetricResult] = []
    for vals, grp in d.groupby(keys):
        r = analyze_metric(grp, regime, metric, treatment=treatment, baseline=baseline)
        if r is None:
            continue
        vals_t = vals if isinstance(vals, tuple) else (vals,)
        label = ",".join(
            f"{k.replace('param_', '')}={v}" for k, v in zip(keys, vals_t)
        )
        results.append(PairedMetricResult(
            regime=f"{regime}[{label}]", metric=r.metric, n_pairs=r.n_pairs,
            treatment=r.treatment, baseline=r.baseline,
            mean_treat=r.mean_treat, mean_base=r.mean_base, mean_diff=r.mean_diff,
            ci_lo=r.ci_lo, ci_hi=r.ci_hi, p_value=r.p_value,
            cliffs_delta=r.cliffs_delta, magnitude=r.magnitude, verdict=r.verdict,
        ))
    return results


def format_report(results: Sequence[PairedMetricResult]) -> str:
    if not results:
        return "(no paired results — need >=2 paired seeds per regime)"
    treat = results[0].treatment
    base = results[0].baseline
    diff_lbl = f"{treat}-{base}"
    hdr = (
        f"{'regime':8s} {'metric':24s} {'n':>3s} {treat:>7s} {base:>7s} "
        f"{diff_lbl:>8s} {'95% CI (mean diff)':>20s} {'wilcox_p':>8s} "
        f"{'cliff':>6s} {'verdict':>9s}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        ci = f"[{r.ci_lo:+.4f},{r.ci_hi:+.4f}]"
        star = "*" if r.verdict != "tie" and r.p_value < 0.05 else " "
        lines.append(
            f"{r.regime:8s} {r.metric:24s} {r.n_pairs:>3d} {r.mean_treat:>7.3f} "
            f"{r.mean_base:>7.3f} {r.mean_diff:>+8.3f} {ci:>20s} {r.p_value:>8.4f} "
            f"{r.cliffs_delta:>+6.2f} {r.verdict:>8s}{star}"
        )
    lines.append("")
    lines.append("* = CI excludes 0 AND paired Wilcoxon p<0.05 (a real difference).")
    lines.append("A 'tie' whose CI straddles 0 means no detectable difference at this N.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.analysis.exp4")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument(
        "--metrics", nargs="+", default=list(DEFAULT_METRICS),
        help="Higher-is-better metrics to test (treatment - baseline).",
    )
    parser.add_argument(
        "--treatment", default="H1",
        help="Treatment arm (default H1). Use H3 for the L1 claim.",
    )
    parser.add_argument(
        "--baseline", default="H0",
        help="Baseline arm (default H0). Use H2 for the L1 claim.",
    )
    parser.add_argument(
        "--surface", action="store_true",
        help="Also report the jittery verdict at each dead_zone x "
             "link_quality point (the B2 sensitivity surface).",
    )
    parser.add_argument(
        "--surface-metric", default="final_auc",
        help="Metric for the sensitivity surface (default final_auc).",
    )
    args = parser.parse_args(argv)
    df = load_exp4(args.csv)
    n_seeds = df["trial_index"].nunique() if "trial_index" in df.columns else 0
    print(f"loaded {len(df)} ok rows; ~{n_seeds} seeds per cell "
          f"({args.treatment} vs {args.baseline})\n")
    print(format_report(analyze(
        df, metrics=args.metrics, treatment=args.treatment, baseline=args.baseline,
    )))
    if args.surface:
        print("\n=== jittery sensitivity surface (metric=%s) ===" % args.surface_metric)
        print(format_report(analyze_surface(
            df, metric=args.surface_metric,
            treatment=args.treatment, baseline=args.baseline,
        )))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
