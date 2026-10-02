"""Study 5.5's report: the user's decision 5 (a), the rule fixed in advance, applied.

FeRRy Phase 5, unit U8b (the Phase 5 spec, other choices 12; critic A1, A2,
A7 and B5; orchestrator resolutions R6 and R10). Study 5.5 trains the pair
score at γ ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99} from 10 training seeds each and
asks whether looking ahead helps; the exit gate keeps FX if it does not (build
plan L935). The rule is read on N = 12, the decision-rich cell at both its
budgets (decision 5 (a); N = 6 is reported as the control where looking ahead
cannot matter): Study 5.5's two cells, jit-n12-120 and jit-n12-180
(``cells.STUDY_5_5_CELLS``, the command line's default), whichever family
trained the score; Study 5.6's N = 12 cells are read only when named (the
orchestrator's resolution R22). It is read on one shared set of held-out
episodes per cell (common random numbers), and the training seed is the unit:
a seed's score is its checkpoint's mean undiscounted held-out return,
averaged over the cells read (:class:`SweepTable`). ε is fixed before the
sweep, max(0.01, 0.1 × the validation headroom)
(``headroom.epsilon_from_headroom``; :func:`epsilon_from_headroom_report`),
the headroom flown on the plan the evaluation's policies flew (resolution
R23).
:func:`decide` applies the rule in this order, and nothing is looked at twice:

1. **The sanity check** (other choices 12; critic A2). The γ = 0 score must
   reach the best one-step fixed rule within ε: its mean over the seeds is at
   least max(FX, ``greedy_1``) − ε, FX being the FX arm itself (resolution R6)
   and ``greedy_1`` the slot's "most devices, then least time" (the critic's
   one-step reference, not FX's agreement rate). When it fails the curve is not
   read: the outcome is ``sanity-failed`` and FX stays (the learner may be
   revised once at most, before the sweep, other choices 12; this report
   refuses a sweep trained by two revisions, or by a revision past the first).
2. **"Rising"**: the best γ > 0, picked on the validation runs (the mean over
   its seeds of each kept checkpoint's validation score on the cells read;
   ties to the lower γ), beats γ = 0 by at least ε, with the bootstrap CI of
   the gain excluding 0 and the Holm-adjusted p below 0.05: an exact paired
   Wilcoxon of every γ > 0 against γ = 0 on the seed means, Holm over those
   contrasts (``stats.compare_to_reference``; with 10 seeds the exact floor is
   0.002, 0.0098 after Holm over five, critic A7).
3. **"Flat"**: every γ > 0 is within ±ε of γ = 0 by Schuirmann's TOST on the
   seed means (``stats.tost_paired``), an intersection-union test, so no Holm
   (Berger 1982).
4. **"Inconclusive"**: anything else. There is no second look.

**Replacing FX.** The learned score replaces FX only if the outcome is rising
AND the best γ beats the best fixed rule by ε under the same claim rule (its
gain over that rule's mean at least ε, the CI excluding 0, p < 0.05). The fixed
rules are decision 5's: the FX and F arms (resolution R6), FX's band with the
planned order (``hyb``) and "most devices, then least time" (``greedy_1``),
and the best is the one with the highest held-out mean, the hardest to beat.
The slot's own versions of FX and F (``fx_pair``, ``committed_pair``) are
reported beside them, not in the rule. Otherwise FX stays FeRRy's filling and
the null is published.

**greedy_1 against FX** (critic A2, accepted in part). Phase 4 rejected "most
devices first" as a filling, so the rule never adopts it; but if ``greedy_1``
beats the FX arm by ε on the held-out runs under the study's claim rule (here
with the held-out episode as the unit, since neither has a training seed), the
report says so and the user decides.

**Reported alongside, never decided on:** Page's trend test across γ with
Spearman's ρ and a bootstrap over the seeds (``stats.trend_test``), the
learning curves (the mean validation score over the seeds at each validation),
each cell's share of sorties with two or more decisions (critic A1), every
cell's means (the N = 6 control included) and the stack check's picks: the best
γ and γ = 0, each from its median-validation seed (other choices 12).

**The pre-registered grid** (the orchestrator's resolution R26). The rule is
fixed in advance on decision 5 (a)'s grid: γ ∈ :data:`GAMMAS`,
:data:`SEEDS_PER_GAMMA` training seeds per γ and :data:`HELD_OUT_EPISODES`
shared held-out episodes per cell (the spec, other choices 12).
:func:`preregistration` compares a sweep with it, and the verdict carries the
label (``preregistered``, and ``preregistration`` with the reasons) and prints
it on its second line. A sweep off the grid is labelled, never refused: the
calibration (γ ∈ {0, 0.9} x 3 seeds) is read by this same code, and the label
changes no step of the rule. It warns that the outcome is not the
pre-registered one: with fewer seeds the exact floor rises, so below 8 seeds
Holm over five contrasts leaves "rising" out of reach while "flat" is not.

:func:`sweep_table` builds the table from an evaluation file
(:data:`EVALUATION_FORMAT`, written by ``python -m experiments.ferrysim
evaluate``), and refuses a sweep that is not one: episodes of another stream
than the held-out one, a checkpoint not ``trained``, two learner revisions or a
revision past the one allowed (resolution R2: the revision is in the header,
so the sha binds it), two families, rewards or training specs, a missing
γ = 0, uneven seeds, or references that did not fly the same held-out
episodes.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments.analysis.stats import (
    bootstrap_ci,
    compare_to_reference,
    paired_wilcoxon_with_cliffs_delta,
    tost_paired,
    trend_test,
)
from experiments.ferrysim.cells import HELDOUT_STREAM, VAL_STREAM

#: Study 5.5's γ grid (decision 5 (a)).
GAMMAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)
#: Decision 5 (a)'s training seeds per γ: with 10 the exact floor is 0.002, 0.0098
#: after Holm over five contrasts (critic A7).
SEEDS_PER_GAMMA = 10
#: Decision 5 (a)'s shared held-out runs: 1,000 held-out episodes per cell (the
#: spec, other choices 12; the evaluate command's default ``--episodes``).
HELD_OUT_EPISODES = 1000
#: Decision 5's fixed rules: the FX and F arms (resolution R6), FX's band with
#: the planned order (the slot's ``hyb``) and "most devices, then least time"
#: (``greedy_1``).
FIXED_RULES: Tuple[str, ...] = ("FX", "F", "hyb", "greedy_1")
#: The one-step fixed rules the γ = 0 score must reach (other choices 12).
ONE_STEP_RULES: Tuple[str, ...] = ("FX", "greedy_1")
#: The slot's own versions of FX and F, reported beside the fixed rules.
SLOT_REFERENCES: Tuple[str, ...] = ("fx_pair", "committed_pair")
#: The study's claim level.
ALPHA = 0.05
#: The most learner revisions a sweep may come from: the spec allows one
#: revision before the sweep (other choices 12), so ``pair_q.LEARNER_REVISION``
#: is 0 or 1.
MAX_LEARNER_REVISION = 1

OUTCOME_RISING = "rising"
OUTCOME_FLAT = "flat"
OUTCOME_INCONCLUSIVE = "inconclusive"
OUTCOME_SANITY_FAILED = "sanity-failed"
OUTCOMES: Tuple[str, ...] = (OUTCOME_RISING, OUTCOME_FLAT, OUTCOME_INCONCLUSIVE,
                             OUTCOME_SANITY_FAILED)

#: The evaluation file's format (``python -m experiments.ferrysim evaluate``).
EVALUATION_FORMAT = "ferrysim-evaluation-1"


def gamma_label(gamma: float) -> str:
    """A γ as the report names it: ``f"{γ:g}"`` (``0``, ``0.25``, ..., ``0.99``)."""
    return f"{float(gamma):g}"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} is a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} is finite, got {value!r}")
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class SweepTable:
    """What the rule reads, on the cells it is read on.

    ``held_out[γ][seed]`` is the seed's score: its checkpoint's mean held-out
    return, averaged over the cells read; ``validation[γ][seed]`` the kept
    checkpoint's validation score on those cells, which picks the best γ;
    ``references[label]`` each fixed policy's held-out returns, one per
    episode, aligned across policies (common random numbers). ``epsilon`` is
    the margin fixed before the sweep. The rest is reported, never decided on:
    ``curves[γ]`` the mean validation score over the seeds at each
    validation, ``decisions[cell][label]`` the share of sorties with two or
    more decisions (critic A1), ``per_cell`` every cell's means, and
    ``held_out_episodes`` the held-out episodes every policy flew per cell
    (the evaluation's ``episodes``; None when not known), which only the
    grid's label reads (:func:`preregistration`).
    """

    epsilon: float
    held_out: Mapping[float, Mapping[int, float]]
    validation: Mapping[float, Mapping[int, float]]
    references: Mapping[str, Sequence[float]]
    cells: Tuple[str, ...] = ()
    curves: Mapping[float, Sequence[Tuple[int, float, int]]] = dataclasses.field(
        default_factory=dict)
    decisions: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)
    per_cell: Mapping[str, Mapping[str, float]] = dataclasses.field(default_factory=dict)
    held_out_episodes: Optional[int] = None

    def __post_init__(self) -> None:
        eps = _finite(self.epsilon, "epsilon")
        if eps <= 0.0:
            raise ValueError(f"epsilon is a margin > 0, got {self.epsilon!r}")
        object.__setattr__(self, "epsilon", eps)
        held = {float(g): {int(s): _finite(v, f"held_out[{g}][{s}]") for s, v in by.items()}
                for g, by in self.held_out.items()}
        val = {float(g): {int(s): _finite(v, f"validation[{g}][{s}]") for s, v in by.items()}
               for g, by in self.validation.items()}
        if 0.0 not in held:
            raise ValueError("the sweep has no γ = 0: the rule reads every γ > 0 against it")
        if len(held) < 2:
            raise ValueError("the sweep needs at least one γ > 0")
        for gamma in held:
            if not 0.0 <= gamma <= 1.0:
                raise ValueError(f"γ is in [0, 1], got {gamma}")
        seeds = set(held[0.0])
        for gamma, by in held.items():
            if set(by) != seeds:
                raise ValueError(f"γ = {gamma_label(gamma)} has the seeds {sorted(by)}, γ = 0 "
                                 f"{sorted(seeds)}: the seed is the paired unit")
        if len(seeds) < 2:
            raise ValueError("the rule needs at least two training seeds per γ")
        if set(val) != set(held) or any(set(val[g]) != seeds for g in val):
            raise ValueError("every (γ, seed) needs a validation score, and only those")
        refs = {str(k): tuple(_finite(x, f"references[{k}]") for x in v)
                for k, v in self.references.items()}
        if len({len(v) for v in refs.values()}) > 1:
            raise ValueError("the references flew different numbers of held-out episodes: "
                             "they are paired episode by episode")
        episodes = self.held_out_episodes
        if episodes is not None:
            if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
                raise ValueError(f"held_out_episodes is a count per cell >= 1, got "
                                 f"{episodes!r}")
            cells = tuple(self.cells)
            flown = sorted({len(v) for v in refs.values()})
            if cells and flown and flown != [episodes * len(cells)]:
                raise ValueError(f"the references flew {flown[0]} held-out episode(s), not "
                                 f"{episodes} on each of the {len(cells)} cell(s) read")
        object.__setattr__(self, "held_out", held)
        object.__setattr__(self, "validation", val)
        object.__setattr__(self, "references", refs)
        object.__setattr__(self, "cells", tuple(self.cells))

    @property
    def gammas(self) -> Tuple[float, ...]:
        return tuple(sorted(self.held_out))

    @property
    def seeds(self) -> Tuple[int, ...]:
        return tuple(sorted(self.held_out[0.0]))

    def column(self, gamma: float) -> List[float]:
        """γ's seed scores in seed order."""
        return [self.held_out[gamma][s] for s in self.seeds]

    def mean(self, gamma: float) -> float:
        return statistics.fmean(self.column(gamma))

    def reference_mean(self, label: str) -> float:
        try:
            return statistics.fmean(self.references[label])
        except KeyError:
            raise ValueError(f"no held-out returns of {label!r}; the references are "
                             f"{sorted(self.references)}") from None


def _gamma_set(gammas: Sequence[float]) -> str:
    """A γ set as a label names it: ``{0, 0.25, ...}``, a γ off the grid in full."""
    return "{" + ", ".join(gamma_label(g) if g in GAMMAS else repr(float(g))
                           for g in sorted(gammas)) + "}"


def preregistration(table: SweepTable) -> Dict[str, Any]:
    """Whether ``table`` is decision 5 (a)'s pre-registered grid, and if not, why
    (the orchestrator's resolution R26).

    The grid is γ ∈ :data:`GAMMAS`, :data:`SEEDS_PER_GAMMA` training seeds per
    γ and :data:`HELD_OUT_EPISODES` shared held-out episodes per cell; a table
    whose held-out count is not known (``held_out_episodes`` None) is not
    shown to be on it. ``reasons`` names each setting that differs (none on
    the grid); ``grid`` and ``sweep`` give the three settings of each. A
    label, never a refusal, and :func:`decide` reads none of it: the rule
    reads a sweep off the grid (the calibration's) as it reads the grid.
    """
    reasons: List[str] = []
    gammas, seeds, episodes = table.gammas, len(table.seeds), table.held_out_episodes
    if set(gammas) != set(GAMMAS):
        reasons.append(f"γ ∈ {_gamma_set(gammas)}, not {_gamma_set(GAMMAS)}")
    if seeds != SEEDS_PER_GAMMA:
        reasons.append(f"{seeds} seeds per γ, not {SEEDS_PER_GAMMA}")
    if episodes is None:
        reasons.append(f"held-out episodes per cell not recorded (the grid has "
                       f"{HELD_OUT_EPISODES})")
    elif episodes != HELD_OUT_EPISODES:
        reasons.append(f"{episodes} held-out episodes per cell, not {HELD_OUT_EPISODES}")
    return {
        "reasons": reasons,
        "grid": {"gammas": list(GAMMAS), "seeds_per_gamma": SEEDS_PER_GAMMA,
                 "held_out_episodes": HELD_OUT_EPISODES},
        "sweep": {"gammas": list(gammas), "seeds_per_gamma": seeds,
                  "held_out_episodes": episodes},
    }


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class Verdict:
    """:func:`decide`'s reading of a :class:`SweepTable` (JSON-ready by :meth:`to_json`).

    ``outcome`` is :data:`OUTCOMES`'; ``replace_fx`` is True only when the
    outcome is rising and the best γ beats the best fixed rule by ε;
    ``greedy_1_flag`` is True when ``greedy_1`` beats the FX arm by ε on the
    held-out runs (the user decides, critic A2). The rest are the numbers each
    step read, and what is reported alongside; last, the grid's label
    (resolution R26): ``preregistered`` is True only on decision 5 (a)'s
    grid, and ``preregistration`` is :func:`preregistration`'s reading, the
    reasons included. No step reads the label.
    """

    outcome: str
    replace_fx: bool
    greedy_1_flag: bool
    epsilon: float
    cells: Tuple[str, ...]
    gammas: Tuple[float, ...]
    seeds: Tuple[int, ...]
    best_gamma: float
    sanity: Mapping[str, Any]
    rising: Optional[Mapping[str, Any]]
    flat: Optional[Mapping[str, Any]]
    fixed_rule: Optional[Mapping[str, Any]]
    greedy_1: Mapping[str, Any]
    trend: Optional[Mapping[str, Any]]
    stack_check: Mapping[str, Any]
    means: Mapping[str, float]
    references: Mapping[str, float]
    curves: Mapping[str, Any]
    decisions: Mapping[str, Mapping[str, float]]
    per_cell: Mapping[str, Mapping[str, float]]
    preregistered: bool
    preregistration: Mapping[str, Any]

    def to_json(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["cells"] = list(self.cells)
        out["gammas"] = list(self.gammas)
        out["seeds"] = list(self.seeds)
        return _plain(out)


def _plain(value: Any) -> Any:
    """``value`` with tuples as lists and numpy scalars as Python's (JSON-ready)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanity(table: SweepTable, one_step_rules: Sequence[str]) -> Dict[str, Any]:
    rules = {label: table.reference_mean(label) for label in one_step_rules}
    best = max(rules, key=lambda label: (rules[label], -one_step_rules.index(label)))
    gamma0 = table.mean(0.0)
    floor = rules[best] - table.epsilon
    return {
        "passed": gamma0 >= floor,
        "gamma0_mean": gamma0,
        "best_rule": best,
        "best_rule_mean": rules[best],
        "floor": floor,
        "shortfall": rules[best] - gamma0,
        "seeds_below_floor": sum(1 for x in table.column(0.0) if x < floor),
        "rules": rules,
    }


def _best_gamma(table: SweepTable) -> float:
    """The γ > 0 with the highest mean validation score over its seeds; ties to the
    lower γ (decision 5 (a): picked on the validation runs, never the held-out)."""
    positive = [g for g in table.gammas if g > 0.0]
    means = {g: statistics.fmean(table.validation[g][s] for s in table.seeds) for g in positive}
    return max(positive, key=lambda g: (means[g], -g))


def _claim(gain: float, ci_low: float, p: float, epsilon: float, alpha: float) -> bool:
    """The study's claim rule for "beats by ε": the gain at least ε, its CI above 0,
    p below alpha."""
    return gain >= epsilon and ci_low > 0.0 and p < alpha


def _rising(table: SweepTable, best: float, alpha: float, n_bootstraps: int,
            seed: int) -> Dict[str, Any]:
    samples = {gamma_label(g): table.column(g) for g in table.gammas}
    family = compare_to_reference(samples, gamma_label(0.0), alpha=alpha,
                                  n_bootstraps=n_bootstraps, seed=seed)
    contrasts = {}
    for g in table.gammas:
        if g == 0.0:
            continue
        c = family[gamma_label(g)]
        # compare_to_reference's difference is the reference minus the arm.
        gain, low, high = -c.mean_diff, -c.ci_high, -c.ci_low
        contrasts[gamma_label(g)] = {
            "gamma": g, "mean": table.mean(g), "gain": gain, "ci_low": low, "ci_high": high,
            "p_value": c.p_value, "p_holm": c.p_holm,
            "beats_gamma0": _claim(gain, low, c.p_holm, table.epsilon, alpha),
        }
    chosen = contrasts[gamma_label(best)]
    return {"best_gamma": best, "rising": bool(chosen["beats_gamma0"]), "contrasts": contrasts}


def _flat(table: SweepTable, alpha: float) -> Dict[str, Any]:
    contrasts = {}
    for g in table.gammas:
        if g == 0.0:
            continue
        t = tost_paired(table.column(g), table.column(0.0), margin=table.epsilon, alpha=alpha)
        contrasts[gamma_label(g)] = {
            "gamma": g, "mean_diff": t.mean_diff, "ci_low": t.ci_low, "ci_high": t.ci_high,
            "p_value": t.p_value, "equivalent": t.equivalent,
        }
    return {"flat": all(c["equivalent"] for c in contrasts.values()), "contrasts": contrasts}


def _fixed_rule(table: SweepTable, best: float, fixed_rules: Sequence[str], alpha: float,
                n_bootstraps: int, seed: int) -> Dict[str, Any]:
    rules = {label: table.reference_mean(label) for label in fixed_rules}
    rule = max(rules, key=lambda label: (rules[label], -fixed_rules.index(label)))
    learned = table.column(best)
    family = compare_to_reference({"rule": [rules[rule]] * len(learned), "learned": learned},
                                  "rule", alpha=alpha, n_bootstraps=n_bootstraps, seed=seed)
    c = family["learned"]
    gain, low, high = -c.mean_diff, -c.ci_high, -c.ci_low
    return {
        "gamma": best, "rule": rule, "rule_mean": rules[rule], "rules": rules,
        "learned_mean": statistics.fmean(learned), "gain": gain, "ci_low": low,
        "ci_high": high, "p_value": c.p_value,
        "beats": _claim(gain, low, c.p_value, table.epsilon, alpha),
    }


def _greedy_1(table: SweepTable, alpha: float, n_bootstraps: int, seed: int) -> Dict[str, Any]:
    greedy = np.asarray(table.references["greedy_1"], dtype=np.float64)
    fx = np.asarray(table.references["FX"], dtype=np.float64)
    if greedy.size < 2:
        raise ValueError("greedy_1 against FX needs at least two held-out episodes")
    gain, low, high = bootstrap_ci(greedy - fx, np.mean, n_bootstraps=n_bootstraps,
                                   seed=seed)
    test = paired_wilcoxon_with_cliffs_delta(greedy, fx)
    return {
        "episodes": int(greedy.size), "gain": gain, "ci_low": low, "ci_high": high,
        "p_value": test.p_value, "cliffs_delta": test.cliffs_delta,
        "flag": _claim(gain, low, test.p_value, table.epsilon, alpha),
    }


def _trend(table: SweepTable, n_bootstraps: int, seed: int) -> Optional[Dict[str, Any]]:
    if len(table.gammas) < 3:
        return None
    t = trend_test({g: table.column(g) for g in table.gammas}, n_bootstraps=n_bootstraps,
                   seed=seed)
    return {"levels": t.levels, "means": t.means, "statistic": t.statistic, "rho": t.rho,
            "ci_low": t.ci_low, "ci_high": t.ci_high, "p_value": t.p_value,
            "alternative": t.alternative, "method": t.method}


def median_seed(table: SweepTable, gamma: float) -> int:
    """γ's median-validation seed, the stack check's (other choices 12): the seeds
    ranked by their validation score (then by seed), the lower median."""
    ranked = sorted(table.seeds, key=lambda s: (table.validation[gamma][s], s))
    return ranked[(len(ranked) - 1) // 2]


def decide(table: SweepTable, *, alpha: float = ALPHA,
           fixed_rules: Sequence[str] = FIXED_RULES,
           one_step_rules: Sequence[str] = ONE_STEP_RULES,
           n_bootstraps: int = 2000, seed: int = 42) -> Verdict:
    """Decision 5 (a)'s rule on ``table`` (module docstring): the sanity check, then
    rising, flat or inconclusive, then whether FX is replaced; ``greedy_1``
    against FX beside it. Deterministic: every bootstrap draws from ``seed``.
    A table without the held-out returns of every rule the steps read is
    refused before any step, whichever steps it would have reached. The
    verdict carries the grid's label (:func:`preregistration`), which no step
    reads: a sweep off decision 5 (a)'s grid is decided as the grid is."""
    if not isinstance(table, SweepTable):
        raise TypeError(f"table is a SweepTable, got {table!r}")
    fixed_rules, one_step_rules = tuple(fixed_rules), tuple(one_step_rules)
    read = list(dict.fromkeys(fixed_rules + one_step_rules + ("FX", "greedy_1")))
    missing = [label for label in read if label not in table.references]
    if missing:
        # Every refusal comes before the first number is read: the rule reads
        # all of its references or none.
        raise ValueError(f"no held-out returns of {missing}: the rule reads {read} (the "
                         f"evaluate command flies them all unless --references narrows it)")
    best = _best_gamma(table)
    sanity = _sanity(table, one_step_rules)
    rising = flat = fixed = None
    if not sanity["passed"]:
        outcome = OUTCOME_SANITY_FAILED
    else:
        rising = _rising(table, best, alpha, n_bootstraps, seed)
        flat = _flat(table, alpha)
        fixed = _fixed_rule(table, best, fixed_rules, alpha, n_bootstraps, seed)
        if rising["rising"]:
            outcome = OUTCOME_RISING
        elif flat["flat"]:
            outcome = OUTCOME_FLAT
        else:
            outcome = OUTCOME_INCONCLUSIVE
    greedy = _greedy_1(table, alpha, n_bootstraps, seed)
    shown = [label for label in dict.fromkeys(fixed_rules + one_step_rules + SLOT_REFERENCES)
             if label in table.references]
    curves = {gamma_label(g): [list(point) for point in table.curves[g]]
              for g in table.gammas if g in table.curves}
    grid = preregistration(table)
    return Verdict(
        outcome=outcome,
        replace_fx=bool(outcome == OUTCOME_RISING and fixed is not None and fixed["beats"]),
        greedy_1_flag=bool(greedy["flag"]),
        epsilon=table.epsilon,
        cells=table.cells,
        gammas=table.gammas,
        seeds=table.seeds,
        best_gamma=best,
        sanity=sanity,
        rising=rising,
        flat=flat,
        fixed_rule=fixed,
        greedy_1=greedy,
        trend=_trend(table, n_bootstraps, seed),
        stack_check={"best_gamma": best, "best_gamma_seed": median_seed(table, best),
                     "gamma0_seed": median_seed(table, 0.0)},
        means={gamma_label(g): table.mean(g) for g in table.gammas},
        references={label: table.reference_mean(label) for label in shown},
        curves=curves,
        decisions={cell: dict(v) for cell, v in table.decisions.items()},
        per_cell={cell: dict(v) for cell, v in table.per_cell.items()},
        preregistered=not grid["reasons"],
        preregistration=grid,
    )


# --------------------------------------------------------------------------- #
# From an evaluation file
# --------------------------------------------------------------------------- #

def epsilon_from_headroom_report(report: Mapping[str, Any], cells: Sequence[str], *,
                                 plan_score_params: Optional[Mapping[str, Any]] = None
                                 ) -> float:
    """ε of a score read on ``cells``: max(0.01, 0.1 × their mean validation headroom).

    The seed's score is the mean over the cells, so its headroom is the mean of
    the cells' (``headroom.headroom_report``'s ``cells[...]["headroom"]``, read
    on the validation stream, critic B14). ε applies that headroom to returns
    flown on the evaluation's plan, ``plan_score_params`` (the evaluation
    file's; None or ``{}``: the cells' own), so the headroom report must have
    flown the same plan (its ``plan_score_params``, none recorded being the
    cells' own), compared with ``PlanScoreParams``' defaults filled in
    (``checkpoints.plan_differences``; the orchestrator's resolution R23, and
    the Phase 5 repair round's A-1); another plan is refused.
    """
    from experiments.ferrysim.headroom import epsilon_from_headroom

    if report.get("stream") != VAL_STREAM:
        raise ValueError(f"ε reads the validation headroom ({VAL_STREAM!r}), got the stream "
                         f"{report.get('stream')!r}")
    missing = [c for c in cells if c not in report.get("cells", {})]
    if missing:
        raise ValueError(f"the headroom report has no cells {missing}")
    flown, read = report.get("plan_score_params") or {}, plan_score_params or {}
    if flown or read:
        from experiments.ferrysim.checkpoints import plan_differences

        differ = plan_differences(flown, read)
        if differ:
            raise ValueError(
                "the headroom report flew other plan score settings than the evaluation's "
                "policies (resolution R23): " + ", ".join(
                    f"{name} (headroom {mine!r}, evaluation {theirs!r})"
                    for name, (mine, theirs) in differ.items())
                + "; read ε from a headroom report flown on the evaluation's plan (headroom "
                  "--plan-score-params)")
    return epsilon_from_headroom(statistics.fmean(float(report["cells"][c]["headroom"])
                                                  for c in cells))


def _training_without(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """A training spec with what a sweep varies taken out: the seed and γ."""
    out = dict(spec)
    out.pop("seed", None)
    network = dict(out.get("network") or {})
    network.pop("gamma", None)
    out["network"] = network
    return out


def _kept_validation(manifest: Mapping[str, Any], cells: Sequence[str], where: str) -> float:
    kept = [p for p in manifest["validation"] if p.get("episode") == manifest["episodes_trained"]]
    if len(kept) != 1:
        raise ValueError(f"{where}: no validation entry at its kept episode "
                         f"{manifest['episodes_trained']}")
    scores = kept[0].get("cells", {})
    missing = [c for c in cells if c not in scores]
    if missing:
        raise ValueError(f"{where}: its validation did not fly the cells {missing}")
    return statistics.fmean(float(scores[c]) for c in cells)


def _share_2_or_more(decisions: Sequence[Sequence[int]]) -> float:
    sorties = [d for episode in decisions for d in episode]
    return sum(1 for d in sorties if d >= 2) / len(sorties) if sorties else 0.0


def sweep_table(evaluation: Mapping[str, Any], *, cells: Sequence[str],
                epsilon: float) -> SweepTable:
    """The rule's table from an evaluation file, on ``cells`` (refusals above).

    ``evaluation`` is :data:`EVALUATION_FORMAT`'s: the held-out stream, the
    cells, and per policy (each reference, each checkpoint with its verified
    manifest) each cell's returns and decisions per sortie, episode by episode.
    Every policy flew the evaluation's ``episodes`` on each cell, the table's
    ``held_out_episodes`` (the grid's label reads it, resolution R26).
    """
    if evaluation.get("format") != EVALUATION_FORMAT:
        raise ValueError(f"not a {EVALUATION_FORMAT} file (format {evaluation.get('format')!r})")
    if evaluation.get("stream") != HELDOUT_STREAM:
        raise ValueError(f"Study 5.5 is read on the held-out runs ({HELDOUT_STREAM!r}), not on "
                         f"{evaluation.get('stream')!r} (critic B14)")
    cells = tuple(cells)
    if not cells:
        raise ValueError("name the cells the rule is read on")
    flown = list(evaluation["cells"])
    missing = [c for c in cells if c not in flown]
    if missing:
        raise ValueError(f"the evaluation flew {flown}, not {missing}")
    episodes = int(evaluation["episodes"])

    def returns_of(entry: Mapping[str, Any], cell: str, where: str) -> List[float]:
        got = entry["returns"].get(cell)
        if got is None or len(got) != episodes:
            raise ValueError(f"{where}: {0 if got is None else len(got)} held-out episode(s) "
                             f"of {cell}, not the evaluation's {episodes}")
        return [float(x) for x in got]

    references = {label: [x for c in cells for x in returns_of(entry, c, label)]
                  for label, entry in evaluation["references"].items()}
    held: Dict[float, Dict[int, float]] = {}
    val: Dict[float, Dict[int, float]] = {}
    revisions, families, rewards, specs = set(), set(), set(), set()
    curves: Dict[float, Dict[int, List[float]]] = {}
    per_cell: Dict[str, Dict[str, List[float]]] = {}
    for entry in evaluation["checkpoints"]:
        manifest = entry["manifest"]
        where = str(entry.get("path") or manifest.get("sha256"))
        if manifest.get("kind") != "pair_q":
            raise ValueError(f"{where}: a {manifest.get('kind')!r} checkpoint; Study 5.5 sweeps "
                             f"the pair score")
        if manifest.get("purpose") != "trained":
            raise ValueError(f"{where}: its purpose is {manifest.get('purpose')!r}, not trained")
        revision = manifest.get("learner_revision")
        if not isinstance(revision, int) or not 0 <= revision <= MAX_LEARNER_REVISION:
            raise ValueError(f"{where}: learner revision {revision!r}; the spec allows one "
                             f"revision at most (0 or 1)")
        revisions.add(revision)
        families.add((manifest.get("cell_family"), manifest.get("cell_family_sha256")))
        rewards.add(repr(sorted(manifest["reward"].items())))
        specs.add(repr(_training_without(manifest["training"]["spec"])))
        gamma, seed = float(manifest["gamma"]), int(manifest["seeds"]["run"])
        if seed in held.get(gamma, {}):
            raise ValueError(f"{where}: two checkpoints of γ = {gamma_label(gamma)}, seed {seed}")
        means = {c: statistics.fmean(returns_of(entry, c, where)) for c in flown}
        held.setdefault(gamma, {})[seed] = statistics.fmean(means[c] for c in cells)
        val.setdefault(gamma, {})[seed] = _kept_validation(manifest, cells, where)
        for point in manifest["validation"]:
            curves.setdefault(gamma, {}).setdefault(int(point["episode"]), []).append(
                statistics.fmean(float(point["cells"][c]) for c in cells))
        for c in flown:
            per_cell.setdefault(c, {}).setdefault(gamma_label(gamma), []).append(means[c])
    for what, values in (("learner revisions", revisions), ("cell families", families),
                         ("rewards", rewards), ("training specs (but γ and the seed)", specs)):
        if len(values) > 1:
            raise ValueError(f"the sweep mixes {len(values)} {what}: one learner trains a sweep")
    decisions = {c: {label: _share_2_or_more(entry["decisions"][c])
                     for label, entry in evaluation["references"].items()} for c in flown}
    table_cells = {c: {label: statistics.fmean(v) for label, v in by.items()}
                   for c, by in per_cell.items()}
    for c in flown:
        for label, entry in evaluation["references"].items():
            table_cells.setdefault(c, {})[label] = statistics.fmean(
                returns_of(entry, c, label))
    return SweepTable(
        epsilon=epsilon, held_out=held, validation=val, references=references, cells=cells,
        curves={g: tuple((e, statistics.fmean(v), len(v)) for e, v in sorted(by.items()))
                for g, by in curves.items()},
        decisions=decisions, per_cell=table_cells, held_out_episodes=episodes)


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #

def format_verdict(verdict: Verdict) -> str:
    """The verdict as text, the rule's steps in order, after the grid's label."""
    v = verdict
    lines = [f"Study 5.5 on {', '.join(v.cells) or 'the cells read'}: "
             f"{len(v.gammas)} γ x {len(v.seeds)} seeds, ε = {v.epsilon:.4f}"]
    grid = v.preregistration["grid"]
    if v.preregistered:
        lines.append(f"Grid: pre-registered (decision 5 (a)): γ ∈ {_gamma_set(grid['gammas'])}, "
                     f"{grid['seeds_per_gamma']} seeds per γ, {grid['held_out_episodes']} "
                     f"held-out episodes per cell")
    else:
        lines.append("Grid: NOT pre-registered (decision 5 (a)): "
                     + "; ".join(v.preregistration["reasons"]))
    s = v.sanity
    lines.append(
        f"1. sanity: γ = 0 mean {s['gamma0_mean']:+.4f} against {s['best_rule']} "
        f"{s['best_rule_mean']:+.4f} - ε = {s['floor']:+.4f}: "
        f"{'passed' if s['passed'] else 'FAILED (the curve is not read)'}")
    if v.rising is not None:
        for label, c in v.rising["contrasts"].items():
            lines.append(f"   γ = {label}: gain over γ = 0 {c['gain']:+.4f} "
                         f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}], p {c['p_value']:.4f}, "
                         f"Holm {c['p_holm']:.4f}")
        lines.append(f"2. rising (best γ on validation {gamma_label(v.best_gamma)}): "
                     f"{'yes' if v.rising['rising'] else 'no'}")
    if v.flat is not None:
        lines.append(f"3. flat (TOST at ±ε for every γ > 0): "
                     f"{'yes' if v.flat['flat'] else 'no'}")
    lines.append(f"Outcome: {v.outcome}")
    if v.fixed_rule is not None:
        f = v.fixed_rule
        lines.append(f"Best fixed rule {f['rule']} {f['rule_mean']:+.4f}; γ = "
                     f"{gamma_label(f['gamma'])} gains {f['gain']:+.4f} "
                     f"[{f['ci_low']:+.4f}, {f['ci_high']:+.4f}], p {f['p_value']:.4f}")
    lines.append(f"Replace FX: {'YES' if v.replace_fx else 'no'}")
    g = v.greedy_1
    lines.append(f"greedy_1 over FX on {g['episodes']} held-out episodes: {g['gain']:+.4f} "
                 f"[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}], p {g['p_value']:.4f}"
                 + (" -- greedy_1 beats FX by ε: for the user to decide (critic A2)"
                    if v.greedy_1_flag else ""))
    if v.trend is not None:
        t = v.trend
        lines.append(f"Trend across γ (Page): ρ {t['rho']:+.3f} [{t['ci_low']:+.3f}, "
                     f"{t['ci_high']:+.3f}], p {t['p_value']:.4f}")
    lines.append("Means: " + ", ".join(f"γ {k} {m:+.4f}" for k, m in v.means.items()))
    lines.append("References: " + ", ".join(f"{k} {m:+.4f}" for k, m in v.references.items()))
    lines.append(f"Stack check: γ = {gamma_label(v.stack_check['best_gamma'])} seed "
                 f"{v.stack_check['best_gamma_seed']}, γ = 0 seed "
                 f"{v.stack_check['gamma0_seed']}")
    for cell, shares in v.decisions.items():
        lines.append(f"Sorties with 2 or more decisions, {cell}: " + ", ".join(
            f"{k} {x:.2f}" for k, x in shares.items()))
    return "\n".join(lines)


__all__ = [
    "ALPHA",
    "EVALUATION_FORMAT",
    "FIXED_RULES",
    "GAMMAS",
    "HELD_OUT_EPISODES",
    "MAX_LEARNER_REVISION",
    "ONE_STEP_RULES",
    "OUTCOMES",
    "OUTCOME_FLAT",
    "OUTCOME_INCONCLUSIVE",
    "OUTCOME_RISING",
    "OUTCOME_SANITY_FAILED",
    "SEEDS_PER_GAMMA",
    "SLOT_REFERENCES",
    "SweepTable",
    "Verdict",
    "decide",
    "epsilon_from_headroom_report",
    "format_verdict",
    "gamma_label",
    "median_seed",
    "preregistration",
    "sweep_table",
]
