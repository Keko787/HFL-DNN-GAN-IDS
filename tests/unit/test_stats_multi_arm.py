"""Phase 0 — statistics for families of comparisons.

Pins the four additions to :mod:`experiments.analysis.stats`:

* :func:`holm_bonferroni` — against a hand-worked step-down example and a
  direct implementation of the step-down procedure.
* :func:`friedman_test` — against the closed-form statistic for a perfectly
  consistent ranking, and against scipy on random data.
* :func:`factorial_2x2` / :func:`difference_of_differences` — against a design
  with a known interaction, and an additive design that must read as zero.
* :func:`compare_to_reference` — the claim rule applied across a family,
  including the case where every comparison is significant on its own and
  none survives Holm.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.analysis.stats import (
    compare_to_reference,
    difference_of_differences,
    factorial_2x2,
    friedman_test,
    holm_bonferroni,
)


# --------------------------------------------------------------------------- #
# Holm–Bonferroni
# --------------------------------------------------------------------------- #

def test_holm_matches_the_hand_worked_step_down():
    # Sorted: 0.005·4 = 0.02, 0.01·3 = 0.03, 0.03·2 = 0.06, 0.04·1 -> 0.06
    # after the running maximum. Mapped back to input order:
    res = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
    assert res.adjusted == pytest.approx((0.03, 0.06, 0.06, 0.02))
    assert res.reject == (True, False, False, True)
    assert res.n_rejected == 2


def test_holm_agrees_with_the_step_down_procedure():
    rng = np.random.default_rng(3)
    for _ in range(200):
        m = int(rng.integers(1, 12))
        p = rng.uniform(0.0, 0.2, size=m)
        # Direct procedure: walk the sorted p-values and stop at the first
        # one above alpha / (m - i).
        expected = [False] * m
        for i, idx in enumerate(np.argsort(p)):
            if p[idx] >= 0.05 / (m - i):
                break
            expected[idx] = True
        assert list(holm_bonferroni(p).reject) == expected


def test_holm_is_never_more_conservative_than_bonferroni():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.0, 0.3, size=9)
    adjusted = np.array(holm_bonferroni(p).adjusted)
    assert np.all(adjusted <= np.minimum(1.0, 9 * p) + 1e-15)
    assert np.all(adjusted >= p)


def test_holm_caps_at_one():
    assert holm_bonferroni([0.5, 0.6]).adjusted == (1.0, 1.0)


def test_a_single_comparison_is_unchanged():
    res = holm_bonferroni([0.03])
    assert res.adjusted == (0.03,) and res.reject == (True,)


def test_tied_p_values_share_one_adjusted_value():
    assert holm_bonferroni([0.02, 0.02, 0.02]).adjusted == pytest.approx((0.06,) * 3)


def test_rejection_is_strict_at_alpha():
    """Adjusted p exactly at alpha is not a rejection, matching the claim rule."""
    res = holm_bonferroni([0.025, 0.025])
    assert res.adjusted == pytest.approx((0.05, 0.05))
    assert res.reject == (False, False)


def test_an_empty_family_is_empty():
    res = holm_bonferroni([])
    assert res.adjusted == () and res.n_rejected == 0


@pytest.mark.parametrize("bad", [[0.1, float("nan")], [1.2], [-0.1]])
def test_invalid_p_values_are_refused(bad):
    with pytest.raises(ValueError, match="p-values"):
        holm_bonferroni(bad)


# --------------------------------------------------------------------------- #
# Friedman
# --------------------------------------------------------------------------- #

def _ordered_arms():
    """A < B < C on every one of 4 seeds, with the seeds on different levels."""
    return {
        "A": [1.0, 5.0, 2.0, 7.0],
        "B": [2.0, 6.0, 3.0, 8.0],
        "C": [3.0, 9.0, 4.0, 9.5],
    }


def test_friedman_matches_the_closed_form_for_perfect_agreement():
    # Rank sums 4, 8, 12 over n = 4 seeds and k = 3 arms:
    # 12 / (n·k·(k + 1)) · (16 + 64 + 144) − 3·n·(k + 1) = 56 − 48 = 8.
    res = friedman_test(_ordered_arms())
    assert (res.n_blocks, res.k) == (4, 3)
    assert res.statistic == pytest.approx(8.0)
    assert res.p_value == pytest.approx(math.exp(-4.0))   # chi-square, 2 df
    assert res.kendalls_w == pytest.approx(1.0)
    assert res.mean_ranks == {"A": 3.0, "B": 2.0, "C": 1.0}


def test_mean_ranks_follow_the_metric_direction():
    res = friedman_test(_ordered_arms(), higher_is_better=False)   # e.g. time to tau
    assert res.mean_ranks == {"A": 1.0, "B": 2.0, "C": 3.0}
    assert res.statistic == pytest.approx(8.0)                    # the test is symmetric


def test_friedman_matches_scipy_on_random_data():
    from scipy import stats

    rng = np.random.default_rng(11)
    arms = {f"arm{j}": rng.normal(loc=0.1 * j, size=12) for j in range(4)}
    res = friedman_test(arms)
    ref = stats.friedmanchisquare(*arms.values())
    assert res.statistic == pytest.approx(float(ref.statistic))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.kendalls_w == pytest.approx(res.statistic / (12 * 3))


def test_all_tied_seeds_give_a_neutral_result():
    res = friedman_test({"A": [1.0, 2.0], "B": [1.0, 2.0], "C": [1.0, 2.0]})
    assert (res.statistic, res.p_value, res.kendalls_w) == (0.0, 1.0, 0.0)
    assert res.mean_ranks == {"A": 2.0, "B": 2.0, "C": 2.0}


def test_two_arms_are_refused():
    with pytest.raises(ValueError, match="paired_wilcoxon"):
        friedman_test({"A": [1.0, 2.0], "B": [2.0, 3.0]})


def test_friedman_refuses_ragged_or_incomplete_seeds():
    with pytest.raises(ValueError, match="same length"):
        friedman_test({"A": [1.0, 2.0], "B": [1.0, 2.0], "C": [1.0]})
    with pytest.raises(ValueError, match="finite"):
        friedman_test({"A": [1.0, 2.0], "B": [1.0, float("nan")], "C": [0.0, 1.0]})


# --------------------------------------------------------------------------- #
# 2×2 factorial and difference of differences
# --------------------------------------------------------------------------- #

BASE = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
DELTA = np.array([4.0, 4.4, 4.8, 5.2, 5.6, 6.0])   # the built-in interaction, mean 5


def _cells(delta=DELTA):
    """A adds 1, B adds 2, and together they add 3 + delta."""
    return BASE, BASE + 1.0, BASE + 2.0, BASE + 3.0 + delta


def test_the_interaction_is_the_difference_of_differences():
    res = factorial_2x2(*_cells())
    assert res.interaction.n_pairs == 6
    assert res.interaction.estimate == pytest.approx(5.0)
    assert res.interaction.ci_excludes_zero
    # Six same-sign effects with distinct sizes: exact two-sided p = 2 / 2^6.
    assert res.interaction.p_value == pytest.approx(2 / 2 ** 6)


def test_the_main_effects_average_over_the_other_factor():
    res = factorial_2x2(*_cells())
    assert res.main_a.estimate == pytest.approx(1.0 + 5.0 / 2)   # (1 + (1 + delta)) / 2
    assert res.main_b.estimate == pytest.approx(2.0 + 5.0 / 2)   # (2 + (2 + delta)) / 2


def test_difference_of_differences_is_the_factorial_interaction():
    assert difference_of_differences(*_cells()) == factorial_2x2(*_cells()).interaction


def test_an_additive_design_reads_as_no_interaction():
    """0.1-style decimals leave rounding residue in a four-term sum; it must
    not turn into a consistent sign that a rank test calls significant."""
    base = np.array([0.1, 0.2, 0.3, 0.7, 1.1, 1.3])
    res = difference_of_differences(base, base + 0.1, base + 0.2, base + 0.3)
    assert (res.estimate, res.p_value) == (0.0, 1.0)
    assert not res.ci_excludes_zero


def test_swapping_the_factors_swaps_the_main_effects():
    base, a_only, b_only, both = _cells()
    fwd = factorial_2x2(base, a_only, b_only, both)
    rev = factorial_2x2(base, b_only, a_only, both)
    assert rev.interaction.estimate == pytest.approx(fwd.interaction.estimate)
    assert rev.main_a.estimate == pytest.approx(fwd.main_b.estimate)
    assert rev.main_b.estimate == pytest.approx(fwd.main_a.estimate)


def test_factorial_refuses_misaligned_or_incomplete_cells():
    base, a_only, b_only, both = _cells()
    with pytest.raises(ValueError, match="same length"):
        factorial_2x2(base, a_only, b_only, both[:-1])
    with pytest.raises(ValueError, match="finite"):
        factorial_2x2(base, a_only, b_only, np.where(both > 20, np.nan, both))
    with pytest.raises(ValueError, match="at least 2"):
        difference_of_differences([1.0], [2.0], [3.0], [4.0])


# --------------------------------------------------------------------------- #
# A family against one reference arm
# --------------------------------------------------------------------------- #

def test_a_clear_win_is_claimed_and_a_tie_is_not():
    rng = np.random.default_rng(0)
    ref = rng.normal(10.0, 1.0, size=30)
    family = {
        "F": ref,
        "D1": ref - 2.0 - rng.uniform(0.0, 0.5, size=30),
        "D2": ref.copy(),
    }
    res = compare_to_reference(family, "F")
    assert list(res) == ["D1", "D2"]

    win, tie = res["D1"], res["D2"]
    assert win.mean_diff > 2.0 and win.ci_low > 0.0
    assert win.p_holm == pytest.approx(min(1.0, 2 * win.p_value))
    assert win.claim
    assert (tie.p_value, tie.p_holm, tie.mean_diff) == (1.0, 1.0, 0.0)
    assert not tie.claim


def test_holm_can_withhold_a_claim_each_test_would_make_alone():
    """Seven arms, each beaten on all 8 seeds: every unadjusted p is 2/2^8,
    but seven of them together adjust to 7·2/2^8 > 0.05."""
    ref = np.arange(10.0, 18.0)
    gaps = 1.0 + 0.01 * np.arange(8)                     # distinct, all positive
    family = {"F": ref, **{f"D{j}": ref - gaps for j in range(7)}}
    res = compare_to_reference(family, "F")
    for comparison in res.values():
        assert comparison.p_value == pytest.approx(2 / 2 ** 8)
        assert comparison.ci_low > 0.0
        assert comparison.p_holm == pytest.approx(7 * 2 / 2 ** 8)
        assert not comparison.claim


def test_compare_to_reference_refuses_a_missing_or_lonely_reference():
    with pytest.raises(ValueError, match="not among"):
        compare_to_reference({"A": [1.0, 2.0], "B": [2.0, 3.0]}, "F")
    with pytest.raises(ValueError, match="besides the reference"):
        compare_to_reference({"F": [1.0, 2.0]}, "F")
