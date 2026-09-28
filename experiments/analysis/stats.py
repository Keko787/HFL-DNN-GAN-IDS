"""Statistical helpers shared by the experiment analyses.

Three primitives the paper plan calls for:

* :func:`paired_wilcoxon_with_cliffs_delta` — paired Wilcoxon
  signed-rank test + Cliff's δ effect-size estimator. The paired-trial
  seed contract from the trial-grid harness (EX-0) is what makes
  paired tests valid here.
* :func:`bootstrap_ci` — non-parametric bootstrap CI on an arbitrary
  scalar statistic.
* :func:`solve_crossover_round` — the cumulative-crossover R* recovery
  via linear regression of T_proc against R.

Families of comparisons, added for Phase 0 of the FeRRy build plan:

* :func:`holm_bonferroni` — Holm's step-down adjustment over one family
  of comparisons. The claim rule (CI excludes zero and adjusted p < 0.05)
  reads its adjusted p-values.
* :func:`friedman_test` — three or more arms ranked on paired seeds, with
  Kendall's W and each arm's mean rank.
* :func:`factorial_2x2` / :func:`difference_of_differences` — the main
  effects and the interaction of a 2×2 design on paired seeds (the RL
  decision memo's §6.2 interaction test).
* :func:`compare_to_reference` — every arm against one reference arm as a
  Holm-adjusted family, with the claim rule applied.

Kept in a separate module so the experiment-specific modules
(:mod:`experiments.analysis.exp1`, etc.) only worry about the
domain-specific layout of their CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

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


def _paired_matrix(columns: Sequence[Sequence[float]], *, what: str) -> np.ndarray:
    """Stack index-aligned per-seed samples into an ``(n_seeds, n_arms)`` matrix.

    Index ``i`` of every column must be the same cell and trial (the
    trial-grid's paired-seed contract). Incomplete seeds have to be dropped
    by the caller: a NaN would silently turn every downstream p-value into
    NaN, so it is rejected here.
    """
    arrays = [np.asarray(c, dtype=np.float64) for c in columns]
    if any(a.ndim != 1 for a in arrays) or len({a.size for a in arrays}) != 1:
        raise ValueError(
            f"{what} inputs must be 1-D and the same length, got shapes "
            f"{[a.shape for a in arrays]}"
        )
    if arrays[0].size < 2:
        raise ValueError(f"{what} needs at least 2 paired seeds")
    matrix = np.column_stack(arrays)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"{what} inputs must be finite; drop incomplete seeds before testing"
        )
    return matrix


# --------------------------------------------------------------------------- #
# Multiplicity — Holm–Bonferroni
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HolmResult:
    """Output of :func:`holm_bonferroni`, index-aligned with its input."""

    p_values: tuple[float, ...]
    adjusted: tuple[float, ...]
    reject: tuple[bool, ...]
    alpha: float

    @property
    def n_rejected(self) -> int:
        return int(sum(self.reject))


def holm_bonferroni(
    p_values: Sequence[float], *, alpha: float = 0.05,
) -> HolmResult:
    """Holm's step-down adjustment over one family of comparisons.

    Sorts the p-values ascending, multiplies the i-th smallest (from 1) by
    ``m − i + 1``, takes a running maximum so the adjusted values stay in
    order, and caps them at 1. A comparison is rejected when its adjusted
    p-value is below ``alpha``: strict, like the claim rule and
    :attr:`PairedTestResult.significant`. Holm controls the family-wise
    error rate as Bonferroni does and is never more conservative.

    The result is index-aligned with the input, so labels pair up by
    zipping: ``dict(zip(labels, holm_bonferroni(ps).adjusted))``. An empty
    family returns an empty result.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"holm_bonferroni expects a 1-D sequence, got shape {p.shape}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if p.size == 0:
        return HolmResult(p_values=(), adjusted=(), reject=(), alpha=float(alpha))
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"p-values must be finite and in [0, 1], got {p.tolist()}")

    m = p.size
    order = np.argsort(p, kind="mergesort")        # stable, so ties keep input order
    stepped = (m - np.arange(m)) * p[order]
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(stepped))
    return HolmResult(
        p_values=tuple(float(x) for x in p),
        adjusted=tuple(float(x) for x in adjusted),
        reject=tuple(bool(x < alpha) for x in adjusted),
        alpha=float(alpha),
    )


# --------------------------------------------------------------------------- #
# Multi-arm — Friedman
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FriedmanResult:
    """Output of :func:`friedman_test`."""

    n_blocks: int                 # paired seeds
    k: int                        # arms
    statistic: float              # Friedman chi-square, tie-corrected
    p_value: float
    kendalls_w: float             # statistic / (n_blocks·(k − 1)), in [0, 1]
    mean_ranks: Dict[str, float]  # 1 = best arm on a seed; ties share the average

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def friedman_test(
    samples_by_arm: Mapping[str, Sequence[float]],
    *,
    higher_is_better: bool = True,
) -> FriedmanResult:
    """Friedman test across three or more arms on paired seeds.

    ``samples_by_arm`` maps each arm to its per-seed values, index-aligned
    across arms. Each seed is a block: the arms are ranked within it, and the
    test asks whether their mean ranks differ by more than chance. It says
    *that* the arms differ, not which ones; follow a significant result with
    paired comparisons under :func:`holm_bonferroni`, for example through
    :func:`compare_to_reference`. Compare two arms with
    :func:`paired_wilcoxon_with_cliffs_delta` instead.

    Kendall's W is the effect size: 1 means every seed ranks the arms the
    same way, 0 means no agreement. ``mean_ranks`` puts the best arm at 1 —
    the highest value when ``higher_is_better`` (accuracy, AUC, round
    closure), the lowest otherwise (time to τ).
    """
    from scipy import stats

    names = list(samples_by_arm)
    if len(names) < 3:
        raise ValueError(
            "friedman_test needs at least 3 arms; compare two with "
            "paired_wilcoxon_with_cliffs_delta"
        )
    data = _paired_matrix([samples_by_arm[a] for a in names], what="friedman_test")
    n, k = data.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, -data if higher_is_better else data)
    mean_ranks = {name: float(r) for name, r in zip(names, ranks.mean(axis=0))}

    if np.all(data == data[:, :1]):
        # Every seed ties every arm: there is nothing to rank, and the
        # tie-corrected statistic would be 0/0. Report a neutral result, as
        # the paired test does for all-zero differences.
        return FriedmanResult(
            n_blocks=n, k=k, statistic=0.0, p_value=1.0, kendalls_w=0.0,
            mean_ranks=mean_ranks,
        )

    res = stats.friedmanchisquare(*data.T)
    statistic = float(res.statistic)
    return FriedmanResult(
        n_blocks=n,
        k=k,
        statistic=statistic,
        p_value=float(res.pvalue),
        kendalls_w=statistic / (n * (k - 1)),
        mean_ranks=mean_ranks,
    )


# --------------------------------------------------------------------------- #
# 2×2 factorial — main effects and difference of differences
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairedEffect:
    """A per-seed effect tested against zero."""

    n_pairs: int
    estimate: float   # mean of the per-seed effects
    ci_low: float     # percentile bootstrap CI on that mean
    ci_high: float
    statistic: float  # Wilcoxon W on the per-seed effects
    p_value: float

    @property
    def ci_excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def _paired_effect(
    effects: np.ndarray,
    *,
    scale: float,
    n_bootstraps: int,
    confidence: float,
    seed: int,
) -> PairedEffect:
    """Wilcoxon signed-rank against zero plus a bootstrap CI on the mean."""
    from scipy import stats

    # A four-term combination of floats leaves rounding residue where the
    # true effect is exactly zero (10.1 + 13.1 − 11.1 − 12.1 is not 0.0 in
    # binary). A rank test would read that residue as a consistent sign, so
    # anything at the inputs' rounding scale counts as zero.
    effects = np.where(np.abs(effects) <= 1e-12 * max(1.0, scale), 0.0, effects)
    if np.all(effects == 0):
        return PairedEffect(
            n_pairs=int(effects.size), estimate=0.0, ci_low=0.0, ci_high=0.0,
            statistic=0.0, p_value=1.0,
        )
    res = stats.wilcoxon(effects, zero_method="wilcox", alternative="two-sided")
    point, lo, hi = bootstrap_ci(
        effects, np.mean,
        n_bootstraps=n_bootstraps, confidence=confidence, seed=seed,
    )
    return PairedEffect(
        n_pairs=int(effects.size),
        estimate=point,
        ci_low=lo,
        ci_high=hi,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
    )


@dataclass(frozen=True)
class Factorial2x2Result:
    """Output of :func:`factorial_2x2`."""

    main_a: PairedEffect
    main_b: PairedEffect
    interaction: PairedEffect


def factorial_2x2(
    base: Sequence[float],
    a_only: Sequence[float],
    b_only: Sequence[float],
    both: Sequence[float],
    *,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Factorial2x2Result:
    """Main effects and interaction of a 2×2 design on paired seeds.

    The four arguments are the per-seed values of the four cells,
    index-aligned across cells: ``base`` has neither factor, ``a_only`` has
    factor A only, ``b_only`` factor B only, and ``both`` has both. Per seed:

    * main effect of A = ((a_only − base) + (both − b_only)) / 2
    * main effect of B = ((b_only − base) + (both − a_only)) / 2
    * interaction = (both − b_only) − (a_only − base), the difference of
      differences

    Each is tested against zero with a Wilcoxon signed-rank test and a
    percentile bootstrap CI on its mean. A positive interaction means the two
    factors together add more than the sum of their separate effects.

    For the RL decision memo's 2×2 (§6.2), with A = the policy chooses the
    trajectory as well as the selection and B = the objective is in FL
    units: ``base`` = D1 (MAX-AoI), ``a_only`` = E3 (DQN after Chen et al.),
    ``b_only`` = D2 (Oort), and ``both`` = the cross-heuristic or FeRRy. The
    interaction is the headline.
    """
    cells = _paired_matrix([base, a_only, b_only, both], what="factorial_2x2")
    b0, a1, b1, ab = cells.T
    kw = dict(
        scale=float(np.max(np.abs(cells))),
        n_bootstraps=n_bootstraps, confidence=confidence, seed=seed,
    )
    return Factorial2x2Result(
        main_a=_paired_effect(((a1 - b0) + (ab - b1)) / 2.0, **kw),
        main_b=_paired_effect(((b1 - b0) + (ab - a1)) / 2.0, **kw),
        interaction=_paired_effect((ab - b1) - (a1 - b0), **kw),
    )


def difference_of_differences(
    base: Sequence[float],
    a_only: Sequence[float],
    b_only: Sequence[float],
    both: Sequence[float],
    *,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedEffect:
    """The interaction of :func:`factorial_2x2` on its own.

    Per seed, ``(both − b_only) − (a_only − base)``: how much more factor A
    helps when factor B is present than when it is absent.
    """
    cells = _paired_matrix(
        [base, a_only, b_only, both], what="difference_of_differences",
    )
    b0, a1, b1, ab = cells.T
    return _paired_effect(
        (ab - b1) - (a1 - b0),
        scale=float(np.max(np.abs(cells))),
        n_bootstraps=n_bootstraps, confidence=confidence, seed=seed,
    )


# --------------------------------------------------------------------------- #
# A family against one reference arm
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReferenceComparison:
    """One arm against the reference arm, inside a Holm-adjusted family."""

    arm: str
    n_pairs: int
    mean_diff: float     # mean of (reference − arm) over paired seeds
    ci_low: float
    ci_high: float
    p_value: float       # paired Wilcoxon, unadjusted
    p_holm: float        # Holm-adjusted across the whole family
    cliffs_delta: float  # reference against arm, in [-1, 1]
    alpha: float

    @property
    def claim(self) -> bool:
        """The claim rule: the CI excludes zero and the Holm-adjusted p < alpha."""
        return (self.ci_low > 0.0 or self.ci_high < 0.0) and self.p_holm < self.alpha


def compare_to_reference(
    samples_by_arm: Mapping[str, Sequence[float]],
    reference: str,
    *,
    alpha: float = 0.05,
    n_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, ReferenceComparison]:
    """Every other arm against ``reference``, as one Holm-adjusted family.

    The usual shape of a study: one system against each baseline, on paired
    seeds. Each comparison is a paired Wilcoxon on ``reference − arm`` with
    Cliff's δ and a bootstrap CI on the mean paired difference; the family's
    p-values are then Holm-adjusted together. Results keep the input order of
    the non-reference arms. ``claim`` applies the project's rule; the sign of
    ``mean_diff`` says which way the difference runs.
    """
    if reference not in samples_by_arm:
        raise ValueError(
            f"reference arm {reference!r} is not among {sorted(samples_by_arm)}"
        )
    others = [arm for arm in samples_by_arm if arm != reference]
    if not others:
        raise ValueError("compare_to_reference needs at least one arm besides the reference")

    tests: Dict[str, PairedTestResult] = {}
    cis: Dict[str, tuple[float, float, float]] = {}
    for arm in others:
        pair = _paired_matrix(
            [samples_by_arm[reference], samples_by_arm[arm]],
            what="compare_to_reference",
        )
        ref, other = pair.T
        tests[arm] = paired_wilcoxon_with_cliffs_delta(ref, other)
        cis[arm] = bootstrap_ci(
            ref - other, np.mean,
            n_bootstraps=n_bootstraps, confidence=confidence, seed=seed,
        )

    holm = holm_bonferroni([tests[arm].p_value for arm in others], alpha=alpha)
    return {
        arm: ReferenceComparison(
            arm=arm,
            n_pairs=tests[arm].n_pairs,
            mean_diff=cis[arm][0],
            ci_low=cis[arm][1],
            ci_high=cis[arm][2],
            p_value=tests[arm].p_value,
            p_holm=holm.adjusted[i],
            cliffs_delta=tests[arm].cliffs_delta,
            alpha=float(alpha),
        )
        for i, arm in enumerate(others)
    }


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
