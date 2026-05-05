"""Statistical helpers shared by Experiment-1 and Experiment-3 analyses.

Three primitives the paper plan calls for:

* :func:`paired_wilcoxon_with_cliffs_delta` — paired Wilcoxon
  signed-rank test + Cliff's δ effect-size estimator. The paired-trial
  seed contract from the trial-grid harness (EX-0) is what makes
  paired tests valid here.
* :func:`bootstrap_ci` — non-parametric bootstrap CI on an arbitrary
  scalar statistic.
* :func:`solve_crossover_round` — the cumulative-crossover R* recovery
  via linear regression of T_proc against R.

Kept in a separate module so the experiment-specific modules
(:mod:`experiments.analysis.exp1`, etc.) only worry about the
domain-specific layout of their CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Paired Wilcoxon + Cliff's δ
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairedTestResult:
    """Output of :func:`paired_wilcoxon_with_cliffs_delta`."""

    n_pairs: int
    statistic: float  # Wilcoxon W
    p_value: float
    cliffs_delta: float  # ∈ [-1, 1]
    delta_magnitude: str  # "negligible" | "small" | "medium" | "large"

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect-size estimator: P(a > b) - P(a < b).

    Range [-1, +1]. +1 = a always larger; -1 = b always larger;
    0 = stochastically equivalent. Romano et al.'s thresholds:

    * |δ| < 0.147: negligible
    * 0.147 ≤ |δ| < 0.33: small
    * 0.33 ≤ |δ| < 0.474: medium
    * |δ| ≥ 0.474: large
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("cliffs_delta requires non-empty inputs")

    # Pairwise comparison via broadcasting; O(n*m) memory but fine for
    # n, m ≤ 1000 (typical paper-experiment trial counts).
    cmp = np.sign(a_arr[:, None] - b_arr[None, :])
    return float(cmp.mean())


def _delta_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def paired_wilcoxon_with_cliffs_delta(
    a: Sequence[float], b: Sequence[float],
) -> PairedTestResult:
    """Paired Wilcoxon signed-rank on ``a - b`` plus Cliff's δ.

    Inputs must be the same length; index ``i`` of ``a`` is paired
    with index ``i`` of ``b`` (by the trial-grid's paired-seed
    contract — same cell + same trial_index across arms).

    A pair where ``a[i] == b[i]`` is dropped from the Wilcoxon test
    (scipy's default behaviour with ``zero_method='wilcox'``); the
    effect-size estimator uses every pair.
    """
    from scipy import stats

    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"paired-test inputs must have the same shape, got "
            f"{a_arr.shape} vs {b_arr.shape}"
        )
    if a_arr.size < 2:
        raise ValueError("paired Wilcoxon needs at least 2 pairs")

    diff = a_arr - b_arr
    if np.all(diff == 0):
        # All-zero differences make the test undefined; report a
        # neutral result rather than letting scipy raise.
        return PairedTestResult(
            n_pairs=int(a_arr.size),
            statistic=0.0,
            p_value=1.0,
            cliffs_delta=0.0,
            delta_magnitude="negligible",
        )

    res = stats.wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative="two-sided")
    delta = cliffs_delta(a_arr, b_arr)
    return PairedTestResult(
        n_pairs=int(a_arr.size),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        cliffs_delta=delta,
        delta_magnitude=_delta_magnitude(delta),
    )


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    samples: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Non-parametric percentile bootstrap CI.

    Returns ``(point_estimate, lower, upper)`` at the requested
    confidence level. Default 2000 resamples is enough for 95% CIs
    at the trial counts we run (≤ 1000 trials per cell).
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("bootstrap_ci requires non-empty samples")
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    boot = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        resample = rng.choice(arr, size=arr.size, replace=True)
        boot[i] = statistic(resample)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    return point, lo, hi


# --------------------------------------------------------------------------- #
# R* — cumulative crossover round count
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CrossoverEstimate:
    """The round count where FL's cumulative bytes meet centralized's."""

    R_star: Optional[float]   # None when the regression rejects the cell
    slope_per_round_s: float  # FL's seconds-per-round
    intercept_s: float
    centralized_baseline_s: float

    @property
    def is_well_defined(self) -> bool:
        return self.R_star is not None and self.R_star > 0.0


def solve_crossover_round(
    fl_R_values: Sequence[int],
    fl_Tproc_seconds: Sequence[float],
    centralized_Tproc_seconds: Sequence[float],
) -> CrossoverEstimate:
    """Recover R* from one (|D|pd, alpha) cell.

    Linear-regress FL's ``Tproc`` against ``R`` to estimate
    ``T_proc_FL(R) = slope · R + intercept``. Solve
    ``slope · R + intercept = mean(centralized_Tproc)`` for R.

    A degenerate cell (all FL trials at one R, or zero variance)
    returns ``R_star=None``.
    """
    R = np.asarray(fl_R_values, dtype=np.float64)
    T = np.asarray(fl_Tproc_seconds, dtype=np.float64)
    if R.size != T.size:
        raise ValueError(
            f"R / T size mismatch: {R.size} vs {T.size}"
        )
    if R.size < 2 or np.unique(R).size < 2:
        return CrossoverEstimate(
            R_star=None,
            slope_per_round_s=0.0,
            intercept_s=float(T.mean()) if T.size else 0.0,
            centralized_baseline_s=float(np.mean(centralized_Tproc_seconds)),
        )

    # Closed-form OLS (no scipy dep needed).
    slope, intercept = np.polyfit(R, T, deg=1)
    cent_baseline = float(np.mean(centralized_Tproc_seconds))

    if slope <= 0.0:
        # FL doesn't cost more per round; no crossover by Tproc.
        return CrossoverEstimate(
            R_star=None,
            slope_per_round_s=float(slope),
            intercept_s=float(intercept),
            centralized_baseline_s=cent_baseline,
        )

    R_star = (cent_baseline - intercept) / slope
    return CrossoverEstimate(
        R_star=float(R_star) if R_star > 0.0 else None,
        slope_per_round_s=float(slope),
        intercept_s=float(intercept),
        centralized_baseline_s=cent_baseline,
    )


def bootstrap_R_star_ci(
    fl_R_values: Sequence[int],
    fl_Tproc_seconds: Sequence[float],
    centralized_Tproc_seconds: Sequence[float],
    *,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """CI on R* via paired bootstrap of the FL trials.

    Returns ``(R_star_point, lo, hi)``; any element is ``None`` when
    the corresponding regression is degenerate.
    """
    R = np.asarray(fl_R_values, dtype=np.float64)
    T = np.asarray(fl_Tproc_seconds, dtype=np.float64)
    cent = np.asarray(centralized_Tproc_seconds, dtype=np.float64)
    if R.size != T.size or R.size < 2:
        return None, None, None
    if np.unique(R).size < 2:
        return None, None, None

    point = solve_crossover_round(R, T, cent).R_star
    rng = np.random.default_rng(seed)

    boot: list[float] = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, R.size, size=R.size)
        cent_idx = rng.integers(0, cent.size, size=cent.size)
        est = solve_crossover_round(R[idx], T[idx], cent[cent_idx])
        if est.R_star is not None:
            boot.append(est.R_star)

    if not boot:
        return point, None, None
    arr = np.asarray(boot, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))
