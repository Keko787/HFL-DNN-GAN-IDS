"""Shared plotting style + normalization for the Experiment-4 figures.

Both Exp-4 figures import from here so their scales, estimators and styling are
**guaranteed** identical rather than coincidentally similar — a reader can put
them side by side and compare bar heights directly.

Normalization rules (the reason this module exists):

* **Every plotted metric is normalized to [0,1] and drawn on a [0,1] axis.**
  `final_auc`, `mission_completion_rate` and `round_close_rate` are already
  fractions; `update_yield` is a raw count of updates per round, so it is
  divided by the device count to become a *participation fraction*. Mixing a
  [0,1] metric with a 0–5 count on adjacent panels makes the panels
  incomparable and invites misreading.
* **Uncertainty is a percentile bootstrap CI of the mean** — the same estimator
  `experiments.analysis.exp4` reports in the tables, and bounded inside the
  data range *by construction*, so a bounded metric can never be drawn outside
  its bounds. Never use a symmetric ±SD whisker here: `final_auc` is bimodal
  (a session either trains or stays at its untrained init), so ±SD implies a
  spread that does not exist and renders above AUC = 1.0.
* **The paired-difference panel shares one fixed y-range across both figures**
  (`DIFF_YLIM`), so the size of the mule effect and the size of the L1 effect
  are directly comparable. Auto-scaling each panel to its own data would make a
  null look as dramatic as a large effect.

Determinism: bootstrap and jitter use fixed seeds so figures are byte-stable
across runs.
"""

from __future__ import annotations

import random
import statistics as st
from typing import Callable, List, Optional, Sequence

import numpy as np

# ---- Shared scales ------------------------------------------------------- #
# A metric bounded in [0,1] gets a [0,1] axis. No headroom for legends: put the
# legend outside the axes instead of inflating the axis past the bound.
METRIC_YLIM = (0.0, 1.0)
# Zoomed window for the AUC panels, shared by both figures so they stay
# comparable side by side. On a full [0,1] axis the AUC differences (0.01-0.12)
# are visually indistinguishable; this is 2.5x the vertical resolution.
#
# 0.60 is the tightest honest floor: it contains every mean AND every bootstrap
# CI bound in both datasets (the lowest is 0.6572 — fig-2 jittery dz=0.2 H2).
# A 0.80 floor was requested but would clip real results, including a bar mean
# of 0.7997 and four CI bounds. Do not raise it without re-checking
# `assert_within_zoom` for every condition.
AUC_ZOOM_YLIM = (0.60, 1.00)
# Fixed across BOTH figures. Sized to contain every bootstrap CI bound in
# either dataset (global span [-0.429, +0.206]) with a little headroom.
DIFF_YLIM = (-0.46, 0.25)

BOOT_N = 10000
BOOT_SEED = 20240611
JITTER_SEED = 7

# (fill, hatch, marker, marker colour) — greyscale-safe, print-safe.
BASELINE_STYLE = ("white", "///", "o", "0.20")
TREATMENT_STYLE = ("0.55", "...", "x", "black")

CHANCE_LINE = 0.5


def normalize_yield(values: Sequence[float], n_devices: int) -> List[float]:
    """Raw updates/round → participation fraction in [0,1]."""
    if n_devices <= 0:
        raise ValueError(f"n_devices must be positive, got {n_devices}")
    return [float(v) / float(n_devices) for v in values]


def boot_ci(values: Sequence[float], n: int = BOOT_N, seed: int = BOOT_SEED):
    """Percentile bootstrap 95% CI of the mean.

    Every resample mean lies in [min, max] of the data, so for a metric bounded
    in [0,1] the interval is bounded in [0,1] BY CONSTRUCTION — not by clipping.
    """
    v = list(values)
    if not v:
        return float("nan"), float("nan"), float("nan")
    m = st.mean(v)
    if len(v) < 2:
        return m, m, m
    rng = random.Random(seed)
    k = len(v)
    means = sorted(sum(v[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return m, means[int(0.025 * n)], means[int(0.975 * n)]


def draw_metric_panel(
    ax,
    x: np.ndarray,
    labels: Sequence[str],
    series_by_arm: Sequence[Sequence[Sequence[float]]],
    arm_labels: Sequence[str],
    *,
    ylabel: str,
    title: str,
    width: float = 0.36,
    show_chance: bool = False,
    collapse_below: Optional[float] = None,
    divider_after: int = 0,
) -> float:
    """Grouped bars = mean, whiskers = bootstrap CI, points = per-seed values.

    ``series_by_arm[a][c]`` is the list of per-seed values for arm ``a`` in
    condition ``c``. Returns the highest y any artist reaches, so the caller can
    assert nothing was drawn outside the metric's bounds.
    """
    styles = [BASELINE_STYLE, TREATMENT_STYLE]
    top = 0.0
    for ai, (series, label) in enumerate(zip(series_by_arm, arm_labels)):
        color, hatch, marker, mcol = styles[ai % len(styles)]
        off = (ai - (len(series_by_arm) - 1) / 2) * width
        stats = [boot_ci(v) for v in series]
        means = [s[0] for s in stats]
        lo = [s[0] - s[1] if s[0] == s[0] else 0.0 for s in stats]
        hi = [s[2] - s[0] if s[0] == s[0] else 0.0 for s in stats]
        ax.bar(x + off, means, width, yerr=[lo, hi], capsize=3, label=label,
               color=color, edgecolor="black", hatch=hatch, linewidth=1.1,
               error_kw=dict(ecolor="0.15", lw=1.2), zorder=2)
        rng = random.Random(JITTER_SEED)
        for xi, v in zip(x, series):
            jx = [xi + off + (rng.random() - 0.5) * width * 0.78 for _ in v]
            kw = (dict(facecolors="none", edgecolors=mcol) if marker == "o"
                  else dict(color=mcol))
            ax.scatter(jx, v, s=8, marker=marker, linewidths=0.65, alpha=0.5,
                       zorder=3, **kw)
        # Name the failure mode in numbers instead of hiding it in a whisker.
        if collapse_below is not None:
            for xi, v in zip(x, series):
                c = sum(1 for y in v if y < collapse_below)
                if c:
                    ax.annotate(f"{c}/{len(v)}", xy=(xi + off, 0.045), ha="center",
                                fontsize=6.8, color="0.10",
                                bbox=dict(fc="white", ec="none", pad=0.7, alpha=0.85))
        vals = [s[2] for s in stats if s[2] == s[2]] + [max(v) for v in series if v]
        top = max([top] + vals)

    if divider_after:
        ax.axvline(divider_after - 0.5, color="black", ls="--", lw=1.4, zorder=1)
    if show_chance:
        ax.axhline(CHANCE_LINE, color="0.45", ls=(0, (1, 3)), lw=1.0, zorder=1)
        ax.annotate("chance (0.50)", xy=(len(labels) - 0.62, CHANCE_LINE + 0.014),
                    fontsize=7.2, color="0.35", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*METRIC_YLIM)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", ls="--", alpha=0.45)
    ax.set_axisbelow(True)
    return top


def draw_zoomed_point_panel(
    ax,
    x: np.ndarray,
    labels: Sequence[str],
    series_by_arm: Sequence[Sequence[Sequence[float]]],
    arm_labels: Sequence[str],
    *,
    ylabel: str,
    title: str,
    ylim=AUC_ZOOM_YLIM,
    width: float = 0.36,
    below_axis_note: str = "collapsed",
    divider_after: int = 0,
) -> dict:
    """Mean + bootstrap CI as a DOT plot on a zoomed (non-zero-baseline) axis.

    Deliberately **not** bars. A bar encodes magnitude by length from zero, so
    on a truncated axis its length is meaningless and the chart exaggerates
    differences — the classic misleading-graph pattern, and exactly what a
    reviewer who already caught an axis problem will look for next. A marker
    with an interval encodes a point estimate and its uncertainty honestly at
    any baseline.

    Values that fall below the zoomed window (the untrained-init collapse mode)
    are NOT silently dropped: they are counted per condition and reported back
    to the caller, and a downward marker is drawn at the axis floor so the
    reader can see that something lies beyond it.

    Returns ``{"below": {(arm_index, cond_index): count}, "min": lowest CI
    bound drawn, "max": highest}``.
    """
    styles = [BASELINE_STYLE, TREATMENT_STYLE]
    out = {"below": {}, "min": float("inf"), "max": float("-inf")}
    lo_b, hi_b = ylim
    for ai, (series, label) in enumerate(zip(series_by_arm, arm_labels)):
        _, _, marker, mcol = styles[ai % len(styles)]
        off = (ai - (len(series_by_arm) - 1) / 2) * width
        for ci_, (xi, v) in enumerate(zip(x, series)):
            if not v:
                continue
            m, lo, hi = boot_ci(v)
            out["min"] = min(out["min"], lo)
            out["max"] = max(out["max"], hi)
            # CI as a vertical line with caps, mean as a filled marker.
            ax.plot([xi + off, xi + off], [lo, hi], color="0.15", lw=1.4, zorder=3)
            for y in (lo, hi):
                ax.plot([xi + off - width * 0.16, xi + off + width * 0.16], [y, y],
                        color="0.15", lw=1.4, zorder=3)
            ax.plot([xi + off], [m], marker="o" if ai == 0 else "s",
                    ms=7.5, mfc="white" if ai == 0 else "0.35",
                    mec="black", mew=1.4, zorder=4,
                    label=label if ci_ == 0 else None)
            # Per-seed points, only those inside the window.
            rng = random.Random(JITTER_SEED + ci_)
            inside = [y for y in v if lo_b <= y <= hi_b]
            jx = [xi + off + (rng.random() - 0.5) * width * 0.7 for _ in inside]
            kw = (dict(facecolors="none", edgecolors=mcol) if marker == "o"
                  else dict(color=mcol))
            ax.scatter(jx, inside, s=7, marker=marker, linewidths=0.6,
                       alpha=0.38, zorder=2, **kw)
            n_below = sum(1 for y in v if y < lo_b)
            if n_below:
                out["below"][(ai, ci_)] = n_below
                # Mark, don't hide: an arrow at the floor + the count.
                ax.annotate(f"↓{n_below}", xy=(xi + off, lo_b + (hi_b - lo_b) * 0.021),
                            ha="center", fontsize=6.9, color="0.10",
                            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    if divider_after:
        ax.axvline(divider_after - 0.5, color="black", ls="--", lw=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", ls="--", alpha=0.45)
    ax.set_axisbelow(True)
    # The meaning of the ↓n markers is stated in the figure caption rather than
    # in-axis: an in-axis note collides with the per-condition counts.
    return out


def assert_within_zoom(results_min: float, results_max: float, ylim=AUC_ZOOM_YLIM) -> None:
    """Fail loudly if any mean/CI falls outside the zoomed window.

    A zoomed axis is only honest while it contains every estimate it claims to
    show. If this trips, widen ``AUC_ZOOM_YLIM`` rather than letting an
    interval be cropped.
    """
    lo, hi = ylim
    if results_min < lo or results_max > hi:
        raise AssertionError(
            f"a mean/CI at [{results_min:.4f},{results_max:.4f}] falls outside the "
            f"zoomed axis {ylim} — widen AUC_ZOOM_YLIM in figstyle.py; do NOT crop")


def draw_diff_panel(
    ax,
    x: np.ndarray,
    labels: Sequence[str],
    results_by_metric: Sequence[Sequence[object]],
    metric_labels: Sequence[str],
    *,
    ylabel: str,
    title: str,
    width: float = 0.36,
    divider_after: int = 0,
) -> None:
    """Paired (treatment − baseline) differences with bootstrap CIs.

    ``results_by_metric[m][c]`` is a ``PairedMetricResult`` (or None) from
    ``experiments.analysis.exp4.analyze_metric`` — so the figure plots exactly
    the numbers the tables report. A CI crossing the zero line is a tie, and is
    visibly one.
    """
    fills = [("0.55", "..."), ("white", "\\\\\\")]
    for mi, (results, mlabel) in enumerate(zip(results_by_metric, metric_labels)):
        color, hatch = fills[mi % len(fills)]
        off = (mi - (len(results_by_metric) - 1) / 2) * width
        d = [(r.mean_diff if r else float("nan")) for r in results]
        lo = [(r.mean_diff - r.ci_lo if r else 0.0) for r in results]
        hi = [(r.ci_hi - r.mean_diff if r else 0.0) for r in results]
        ax.bar(x + off, d, width, yerr=[lo, hi], capsize=3, label=mlabel,
               color=color, edgecolor="black", hatch=hatch, linewidth=1.1,
               error_kw=dict(ecolor="0.15", lw=1.2), zorder=2)
        # Mark only what the protocol lets us claim: CI excludes 0 AND p<0.05.
        for xi, r in zip(x, results):
            if r is None:
                continue
            claimed = (r.ci_lo > 0 or r.ci_hi < 0) and r.p_value < 0.05
            if claimed:
                y = r.ci_hi + 0.012 if r.mean_diff >= 0 else r.ci_lo - 0.032
                ax.annotate("*", xy=(xi + off, y), ha="center", fontsize=11,
                            color="black")
    ax.axhline(0.0, color="black", lw=1.3, zorder=1)
    if divider_after:
        ax.axvline(divider_after - 0.5, color="black", ls="--", lw=1.4, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(*DIFF_YLIM)          # fixed across both figures — comparable
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="y", ls="--", alpha=0.45)
    ax.set_axisbelow(True)
