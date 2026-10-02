"""The headroom oracle and report: the most any pair choice could gain over FX.

FeRRy Phase 5, unit U8a (the user's decision 10 (i)(a); the spec, other
choices 8 and 12; critic B14; orchestrator resolution R8: the oracle's replay
scorer is this unit's). Before any training, scripted practice runs show how
much any choice of (band, next stop) could gain over FX in each cell, on
FerrySim's validation stream; the build pauses and reports only if that gain
is below 0.01 per practice run in every cell, and Study 5.5's margin is
ε = max(0.01, 0.1 × the headroom) (decision 5).

**The oracle, per sortie.** FerrySim cannot fork a mission, so the oracle
replays: for sortie j of an episode, a depth-first search over the decision
indices of j enumerates every sequence of admitted pairs (each decision's
admitted pairs in FX's own ranking, so the first leaf is FX's choices), and
each leaf is a deterministic replay of the episode with FX's pair rule
(``fx_pair``, the slot's version of FX) flying every other sortie and the mule
stopped once sortie j has closed (nothing later is read). The replay scorer
(:class:`ReplayScorer`) holds the slot to the leaf's path. A sortie's best
return is exact when its leaves are exhausted; at most :data:`MAX_LEAVES` are
flown, and beyond that the sortie is flagged ``truncated`` and its value is a
lower bound. The search is clairvoyant: each leaf flies the realized channel
and availability draws.

**The headroom of an episode.** ``V_dfs`` is the sum over its sorties of their
best returns (each from the state FX's pair rule left). ``V`` is the largest
of ``V_dfs``, the FX arm's return and the returns of U3's four scripted
references flown whole in the slot (:func:`value_references`), so it is at
least every scripted policy's return, and the gain ``V - R(FX)``, never
negative, at least each one's gain over FX ("it must be >= every scripted
policy"). The FX of the gain is the FX arm itself (resolution R6). Apart from
FX's own return, which only keeps the gain from going negative, ``V`` is made
of the slot's own flights, pair choices within decision 1 (a)'s limits, which
is what a learned score could gain. The F arm is not in ``V``: it flies b̄
even at a stop where the slot's mask refuses it, so no pair choice reaches its
flight, and its gain over FX, ``f_gain``, is reported beside the headroom (the
design D-N: the best return "against FX's and F's"), not in it.
``slot_gain = V_dfs - R(fx_pair)``, the oracle over the slot's version of FX,
is reported too and is never negative.

**The report** (:func:`headroom_report`), per cell over the first
``episodes`` validation episodes: the mean gain with its standard error, ε,
the slot gain, F's gain and the episodes in which F's return exceeds ``V``,
every reference's mean return, the oracle's leaves and truncations, and the
share of sorties with two or more decisions (critic A1: at N = 6 a sortie
mostly has one, so looking ahead cannot matter there). The pause rule reads
the mean gain of every cell.

**The plan** (the orchestrator's resolution R23; the Phase 5 repair round's
A-1). Every flight, the references' and each oracle leaf's, flies the cells'
own plan score settings, or ``plan_score_params`` (``--plan-score-params``, as
the runner's flag takes them) as a driver override, so a sweep trained and
evaluated under a pilot's plan reads its ε on that plan; the report records
the plan, and Study 5.5's report refuses a headroom report of another plan
than its evaluation's (``report.epsilon_from_headroom_report``). On the
cells' own plan nothing is overridden or recorded, so that report is the one
it always was.

    python -m experiments.ferrysim.headroom --episodes 200 --workers 8 --out headroom.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from experiments.ferrysim import cells as C
from experiments.ferrysim import inprocess
from experiments.ferrysim.checkpoints import plan_score_settings
from experiments.ferrysim.episode import ARM_F, ARM_FX, Policy, reference_policies, run_episode
from experiments.ferrysim.evaluate import parallel_map
from experiments.ferrysim.reward import DERIVED, RewardSpec

#: The most leaves the oracle flies per sortie (the spec, other choices 8).
MAX_LEAVES = 512
#: Decision 5's floor on ε, and decision 10's pause threshold, per practice run.
EPSILON_FLOOR = 0.01
EPSILON_SHARE = 0.1
PAUSE_BELOW = 0.01

ORACLE_LABEL = "oracle"


def epsilon_from_headroom(headroom: float) -> float:
    """ε = max(0.01, 0.1 × the validation headroom) (decision 5 (a))."""
    if not isinstance(headroom, (int, float)) or isinstance(headroom, bool) \
            or not math.isfinite(headroom):
        raise ValueError(f"headroom is a finite number, got {headroom!r}")
    return max(EPSILON_FLOOR, EPSILON_SHARE * float(headroom))


def value_references() -> Tuple[str, ...]:
    """The references whose whole-episode returns count in ``V``: the FX arm and
    every reference flown in the pair slot (U3's scripted ones). The F arm's
    return is reported beside ``V``, not in it (module docstring)."""
    return tuple(p.label for p in reference_policies() if p.pair_slot or p.label == ARM_FX)


def _on_plan(plan_score_params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``run_episode``'s keywords that fly a plan (module docstring): the plan as a
    driver override, and none on the cells' own plan, so its flights are the ones
    they always were."""
    plan = dict(plan_score_params or {})
    return {"driver_overrides": {"plan_score_params": plan}} if plan else {}


# --------------------------------------------------------------------------- #
# The oracle's scorers
# --------------------------------------------------------------------------- #

def _fx_scorer():
    from hermes.scheduler.policies.pair_slot import FX_PAIR, scripted_scorer

    return scripted_scorer(FX_PAIR)


class ReplayScorer:
    """The headroom oracle's scorer (resolution R8): one leaf of a sortie's search.

    In mission ``sortie`` (0-based; FerrySim's mission tap calls
    :meth:`begin_mission` before each), decision d takes the admitted pair at
    position ``path[d]`` (0 past the path's end) of the admitted pairs in the
    ``base`` scorer's ranking, FX's pair rule by default, so the empty path
    is FX's choices; every other mission is the base ranking. It records each
    decision's number of admitted pairs (``branching``, 0 when the mask was
    empty and the slot fell back) and the pair taken (``chosen``), from which
    the search finds its next leaf. Returns one number per pair, 1 for the
    pair taken and 0 elsewhere, so the slot's masked argmax takes it; it
    never ranks a refused pair first, and the slot's own guards still hold it
    to the admitted pairs (Freeze principle 12).
    """

    name = "oracle_replay"
    q_values = False

    def __init__(self, sortie: int, path: Sequence[int] = (), *, base: Any = None) -> None:
        if isinstance(sortie, bool) or not isinstance(sortie, int) or sortie < 0:
            raise ValueError(f"sortie is a mission index >= 0, got {sortie!r}")
        self.sortie = sortie
        self.path = tuple(int(i) for i in path)
        if any(i < 0 for i in self.path):
            raise ValueError(f"a path holds positions >= 0, got {self.path}")
        self._base = base if base is not None else _fx_scorer()
        self._mission: Optional[int] = None
        self.branching: List[int] = []
        self.chosen: List[Optional[Tuple[str, Optional[int]]]] = []

    def begin_mission(self, index: int) -> None:
        self._mission = int(index)

    def score(self, view, *, mask: Tuple[bool, ...]) -> Tuple[float, ...]:
        scores = tuple(float(x) for x in self._base.score(view, mask=mask))
        if self._mission is None:
            raise RuntimeError("the replay scorer was never told which mission flies "
                               "(FerrySim's mission tap calls begin_mission)")
        if self._mission != self.sortie:
            return scores
        decision = len(self.branching)
        admitted = [row for row, ok in enumerate(mask) if ok]
        if not admitted:
            self.branching.append(0)
            self.chosen.append(None)
            return scores
        order = sorted(admitted, key=lambda row: (-scores[row], row))
        position = self.path[decision] if decision < len(self.path) else 0
        if position >= len(order):
            raise ValueError(
                f"the path {self.path} takes admitted pair {position} of {len(order)} at "
                f"decision {decision}: the replay left the flight it was built from")
        row = order[position]
        self.branching.append(len(order))
        self.chosen.append(tuple(view.pairs[row]))
        return tuple(1.0 if r == row else 0.0 for r in range(len(mask)))


class SortieScorer:
    """One scorer in one sortie and another elsewhere: a reference flying a
    single sortie of an otherwise FX-flown episode, as the oracle's leaves
    do, against which the oracle's sortie value is checked."""

    q_values = False

    def __init__(self, sortie: int, inside: Any, outside: Any = None) -> None:
        self.sortie = int(sortie)
        self._inside = inside
        self._outside = outside if outside is not None else _fx_scorer()
        self._mission: Optional[int] = None
        self.name = f"sortie{self.sortie}:{getattr(inside, 'name', 'scorer')}"

    def begin_mission(self, index: int) -> None:
        self._mission = int(index)

    def score(self, view, *, mask: Tuple[bool, ...]):
        if self._mission is None:
            raise RuntimeError("the sortie scorer was never told which mission flies")
        scorer = self._inside if self._mission == self.sortie else self._outside
        return scorer.score(view, mask=mask)


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class SortieOracle:
    """The oracle's value of one sortie: its best return over every leaf.

    ``fx_return`` is the first leaf's, FX's pair rule; ``leaves`` the leaves
    flown; ``truncated`` True when :data:`MAX_LEAVES` stopped the search
    before it was exhausted (``best_return`` is then a lower bound);
    ``best_path`` the best leaf's positions and ``branching`` the first
    leaf's admitted pairs per decision.
    """

    sortie: int
    best_return: float
    fx_return: float
    leaves: int
    truncated: bool
    best_path: Tuple[int, ...]
    branching: Tuple[int, ...]

    def to_json(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["best_path"] = list(self.best_path)
        out["branching"] = list(self.branching)
        return out


def _next_path(path: Sequence[int], branching: Sequence[int]) -> Optional[Tuple[int, ...]]:
    """The leaf after the one flown with ``path`` (variable-radix order); None
    when every sequence has been flown. ``branching`` is that leaf's admitted
    pairs per decision of the sortie."""
    full = [path[d] if d < len(path) else 0 for d in range(len(branching))]
    d = len(full) - 1
    while d >= 0 and full[d] + 1 >= max(branching[d], 1):
        d -= 1
    if d < 0:
        return None
    return tuple(full[:d] + [full[d] + 1])


def sortie_oracle(cell: C.FerryCell, seed: int, sortie: int, *, trial_index: int = 0,
                  reward: RewardSpec = DERIVED, max_leaves: int = MAX_LEAVES,
                  device_model: str = inprocess.DEVICE_MODEL_EQUAL,
                  plan_score_params: Optional[Mapping[str, Any]] = None) -> SortieOracle:
    """The best return of sortie ``sortie`` of the episode, every other sortie
    flown by FX's pair rule (module docstring), each leaf on the plan
    ``plan_score_params`` (None or ``{}``: the cells' own)."""
    if isinstance(max_leaves, bool) or not isinstance(max_leaves, int) or max_leaves < 1:
        raise ValueError(f"max_leaves is an int >= 1, got {max_leaves!r}")
    on_plan = _on_plan(plan_score_params)
    path: Optional[Tuple[int, ...]] = ()
    best = None
    best_path: Tuple[int, ...] = ()
    first: Optional[Tuple[float, Tuple[int, ...]]] = None
    leaves = 0
    while path is not None and leaves < max_leaves:
        scorer = ReplayScorer(sortie, path)
        policy = Policy(label=ORACLE_LABEL, arm=ARM_FX, scorer=lambda s=scorer: s)
        result = run_episode(cell, seed, policy, trial_index=trial_index, reward=reward,
                             device_model=device_model, stop_after=sortie, **on_plan)
        if len(result.sorties) <= sortie:
            raise ValueError(f"the episode flew {len(result.sorties)} mission(s); sortie "
                             f"{sortie} is not among them")
        value = result.sortie_returns[sortie]
        leaves += 1
        if first is None:
            first = (value, tuple(scorer.branching))
        if best is None or value > best:
            best, best_path = value, tuple(
                path[d] if d < len(path) else 0 for d in range(len(scorer.branching)))
        path = _next_path(path, scorer.branching)
    assert first is not None and best is not None
    return SortieOracle(sortie=sortie, best_return=best, fx_return=first[0], leaves=leaves,
                        truncated=path is not None, best_path=best_path, branching=first[1])


@dataclasses.dataclass(frozen=True)
class EpisodeHeadroom:
    """One validation episode's oracle, its references and its gain over FX.

    ``references`` holds every reference's whole-episode return by label
    (:func:`~experiments.ferrysim.episode.reference_policies`), and
    ``decisions`` the decisions per sortie of the ``fx_pair`` episode, the
    flight the oracle's search branches from.
    """

    cell: str
    seed: int
    index: int
    sorties: Tuple[SortieOracle, ...]
    references: Mapping[str, float]
    reference_sorties: Mapping[str, Tuple[float, ...]]
    decisions: Tuple[int, ...]

    @property
    def v_dfs(self) -> float:
        return sum(s.best_return for s in self.sorties)

    @property
    def value(self) -> float:
        """V: the oracle's sum, FX's return, or a scripted reference flown whole in
        the slot, whichever is largest (:func:`value_references`; not F's)."""
        return max([self.v_dfs] + [self.references[label] for label in value_references()])

    @property
    def gain(self) -> float:
        """V - R(FX arm): the most any pair choice could gain over FX (resolution R6)."""
        return self.value - self.references[ARM_FX]

    @property
    def slot_gain(self) -> float:
        """V_dfs - R(fx_pair): the oracle over the slot's version of FX (>= 0)."""
        from hermes.scheduler.policies.pair_slot import FX_PAIR

        return self.v_dfs - self.references[FX_PAIR]

    @property
    def f_gain(self) -> float:
        """R(F arm) - R(FX arm): F's gain over FX, reported beside the headroom
        (F's flight is outside the slot's choices, so it is not in V)."""
        return self.references[ARM_F] - self.references[ARM_FX]

    def to_json(self) -> Dict[str, Any]:
        return {
            "cell": self.cell, "seed": int(self.seed), "index": int(self.index),
            "sorties": [s.to_json() for s in self.sorties],
            "references": dict(self.references),
            "reference_sorties": {k: list(v) for k, v in self.reference_sorties.items()},
            "decisions": list(self.decisions),
            "v_dfs": self.v_dfs, "value": self.value, "gain": self.gain,
            "slot_gain": self.slot_gain, "f_gain": self.f_gain,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "EpisodeHeadroom":
        """The episode :meth:`to_json` wrote; its derived values are recomputed,
        so a report re-reads under this module's definitions."""
        sorties = tuple(SortieOracle(
            sortie=int(s["sortie"]), best_return=float(s["best_return"]),
            fx_return=float(s["fx_return"]), leaves=int(s["leaves"]),
            truncated=bool(s["truncated"]), best_path=tuple(int(i) for i in s["best_path"]),
            branching=tuple(int(b) for b in s["branching"])) for s in data["sorties"])
        return cls(cell=str(data["cell"]), seed=int(data["seed"]), index=int(data["index"]),
                   sorties=sorties,
                   references={str(k): float(v) for k, v in data["references"].items()},
                   reference_sorties={str(k): tuple(float(x) for x in v)
                                      for k, v in data["reference_sorties"].items()},
                   decisions=tuple(int(d) for d in data["decisions"]))


def episode_headroom(cell: C.FerryCell, seed: int, *, index: int = 0,
                     reward: RewardSpec = DERIVED, max_leaves: int = MAX_LEAVES,
                     device_model: str = inprocess.DEVICE_MODEL_EQUAL,
                     plan_score_params: Optional[Mapping[str, Any]] = None) -> EpisodeHeadroom:
    """The oracle of every sortie of one episode, and every reference flown whole,
    each flight on the plan ``plan_score_params`` (None or ``{}``: the cells' own).

    Checks that each sortie's first leaf is the ``fx_pair`` episode's own
    sortie (the replay flies what that policy flies), and raises otherwise.
    """
    from hermes.scheduler.policies.pair_slot import FX_PAIR

    on_plan = _on_plan(plan_score_params)
    references: Dict[str, float] = {}
    by_sortie: Dict[str, Tuple[float, ...]] = {}
    decisions: Tuple[int, ...] = ()
    for policy in reference_policies():
        result = run_episode(cell, seed, policy, trial_index=index, reward=reward,
                             device_model=device_model, **on_plan)
        references[policy.label] = result.ret
        by_sortie[policy.label] = tuple(result.sortie_returns)
        if policy.label == FX_PAIR:
            decisions = tuple(len(s.stops) for s in result.sorties)
    oracles = tuple(
        sortie_oracle(cell, seed, j, trial_index=index, reward=reward, max_leaves=max_leaves,
                      device_model=device_model, plan_score_params=plan_score_params)
        for j in range(len(by_sortie[FX_PAIR])))
    for oracle, fx in zip(oracles, by_sortie[FX_PAIR]):
        if oracle.fx_return != fx:
            raise AssertionError(
                f"{cell.name} seed {seed} sortie {oracle.sortie}: the oracle's first leaf "
                f"returned {oracle.fx_return}, the fx_pair episode {fx}")
    return EpisodeHeadroom(cell=cell.name, seed=int(seed), index=int(index), sorties=oracles,
                           references=references, reference_sorties=by_sortie,
                           decisions=decisions)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class HeadroomTask:
    """One validation episode's headroom (a worker's unit of work), on the plan
    ``plan_score_params`` (none: the cells' own)."""

    cell: C.FerryCell
    index: int
    seed: int
    reward: RewardSpec = DERIVED
    max_leaves: int = MAX_LEAVES
    device_model: str = inprocess.DEVICE_MODEL_EQUAL
    plan_score_params: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def run_headroom_task(task: HeadroomTask) -> Dict[str, Any]:
    return episode_headroom(task.cell, task.seed, index=task.index, reward=task.reward,
                            max_leaves=task.max_leaves, device_model=task.device_model,
                            plan_score_params=task.plan_score_params).to_json()


def _mean_se(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    return mean, (statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.0)


def cell_headroom(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """One cell's headroom from its episodes' :meth:`EpisodeHeadroom.to_json`."""
    if not episodes:
        raise ValueError("a cell's headroom needs at least one episode")
    gains = [float(e["gain"]) for e in episodes]
    slot = [float(e["slot_gain"]) for e in episodes]
    f_gains = [float(e["f_gain"]) for e in episodes]
    gain, gain_se = _mean_se(gains)
    slot_gain, slot_se = _mean_se(slot)
    f_gain, f_se = _mean_se(f_gains)
    refs = sorted(episodes[0]["references"])
    sorties = [s for e in episodes for s in e["sorties"]]
    decisions = [d for e in episodes for d in e["decisions"]]
    return {
        "cell": episodes[0]["cell"],
        "episodes": len(episodes),
        "headroom": gain,
        "headroom_se": gain_se,
        "epsilon": epsilon_from_headroom(gain),
        "slot_headroom": slot_gain,
        "slot_headroom_se": slot_se,
        "f_gain": f_gain,
        "f_gain_se": f_se,
        "episodes_f_above_value": sum(
            1 for e in episodes if float(e["references"][ARM_F]) > float(e["value"]) + 1e-12),
        "v_dfs_mean": statistics.fmean(float(e["v_dfs"]) for e in episodes),
        "value_mean": statistics.fmean(float(e["value"]) for e in episodes),
        "references_mean": {r: statistics.fmean(float(e["references"][r]) for e in episodes)
                            for r in refs},
        "episodes_with_gain": sum(1 for g in gains if g > 1e-12),
        "sorties": len(sorties),
        "sorties_with_2_or_more_decisions": (sum(1 for d in decisions if d >= 2)
                                             / len(decisions) if decisions else 0.0),
        "decisions_per_sortie_mean": statistics.fmean(decisions) if decisions else 0.0,
        "leaves_mean": statistics.fmean(int(s["leaves"]) for s in sorties) if sorties else 0.0,
        "leaves_max": max((int(s["leaves"]) for s in sorties), default=0),
        "truncated_sorties": sum(1 for s in sorties if s["truncated"]),
    }


def pause_rule(per_cell: Mapping[str, Mapping[str, Any]]) -> bool:
    """Decision 10 (i)(a): pause only if the headroom is below 0.01 per
    practice run in every cell."""
    if not per_cell:
        raise ValueError("the pause rule reads at least one cell")
    return all(float(c["headroom"]) < PAUSE_BELOW for c in per_cell.values())


def headroom_report(cells: Iterable[Any], *, episodes: int = 200, start: int = 0,
                    reward: RewardSpec = DERIVED, max_leaves: int = MAX_LEAVES,
                    workers: int = 1,
                    device_model: str = inprocess.DEVICE_MODEL_EQUAL,
                    plan_score_params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Decision 10 (i)'s report on the validation stream, per cell (module docstring).

    ``cells`` are FerrySim cells or their names. ``pause`` is
    :func:`pause_rule`'s verdict. ``plan_score_params`` are the plan score
    settings every flight flies (``PlanScoreParams`` fields, read as
    ``checkpoints.plan_score_settings`` reads them; None or ``{}``: the cells'
    own plan), recorded as ``plan_score_params`` only when set (resolution
    R23).
    """
    plan = plan_score_settings(plan_score_params or {})
    resolved = [C.cell_named(cell) for cell in cells]
    tasks = [HeadroomTask(cell=cell, index=int(start) + e, seed=seed, reward=reward,
                          max_leaves=max_leaves, device_model=device_model,
                          plan_score_params=plan)
             for cell in resolved
             for e, seed in enumerate(C.stream_seeds(C.VAL_STREAM, cell.name, int(episodes),
                                                     start=int(start)))]
    results = parallel_map(run_headroom_task, tasks, workers=workers)
    per_cell = {cell.name: cell_headroom([r for r in results if r["cell"] == cell.name])
                for cell in resolved}
    report = {
        "stream": C.VAL_STREAM,
        "episodes_per_cell": int(episodes),
        "first_index": int(start),
        "reward": reward.to_json(),
        "max_leaves": int(max_leaves),
        "device_model": device_model,
        "pause_below": PAUSE_BELOW,
        "pause": pause_rule(per_cell),
        "cells": per_cell,
        "episodes": results,
    }
    if plan:
        report["plan_score_params"] = plan
    return report


def format_report(report: Mapping[str, Any]) -> str:
    plan = report.get("plan_score_params")
    lines = [f"Headroom over FX on {report['stream']}, {report['episodes_per_cell']} episodes "
             f"per cell (reward {report['reward']['kind']}, c_t {report['reward']['c_t']}, "
             f"c_cov {report['reward']['c_cov']}; at most {report['max_leaves']} leaves per "
             f"sortie" + (f"; plan score settings {json.dumps(plan, sort_keys=True)}"
                          if plan else "") + ")"]
    for name, c in report["cells"].items():
        refs = ", ".join(f"{k} {v:+.4f}" for k, v in sorted(c["references_mean"].items()))
        lines.append(
            f"  {name:12s} headroom {c['headroom']:+.4f} +- {c['headroom_se']:.4f} "
            f"(eps {c['epsilon']:.4f}); slot {c['slot_headroom']:+.4f} +- "
            f"{c['slot_headroom_se']:.4f}; gain in {c['episodes_with_gain']}/{c['episodes']}; "
            f"sorties >= 2 decisions {c['sorties_with_2_or_more_decisions']:.2f}; leaves mean "
            f"{c['leaves_mean']:.1f} max {c['leaves_max']}, truncated "
            f"{c['truncated_sorties']}/{c['sorties']}")
        lines.append(f"  {'':12s} beside it: F over FX {c['f_gain']:+.4f} +- "
                     f"{c['f_gain_se']:.4f}, F above V in "
                     f"{c['episodes_f_above_value']}/{c['episodes']}")
        lines.append(f"  {'':12s} mean returns: V {c['value_mean']:+.4f} "
                     f"(sum of sorties {c['v_dfs_mean']:+.4f}); {refs}")
    lines.append(f"Pause (headroom below {report['pause_below']} in every cell): "
                 f"{'YES' if report['pause'] else 'no'}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.ferrysim.headroom",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=[c.name for c in C.CELLS])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-leaves", type=int, default=MAX_LEAVES)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--plan-score-params", default=None, metavar="JSON",
                    help="The plan score's settings every flight flies, as the runner's "
                         "--plan-score-params takes them (default: the cells' own): a sweep "
                         "trained under a pilot's plan reads its ε on that plan.")
    ap.add_argument("--out", default=None, help="Write the full report here (JSON).")
    args = ap.parse_args(argv)
    plan: Dict[str, Any] = {}
    if args.plan_score_params is not None:
        # read by the runner's own parser, and refused before any episode flies
        from experiments.exp4.runner_main import _json_object

        plan = _json_object(args.plan_score_params, "--plan-score-params", ap)
        try:
            plan_score_settings(plan, "--plan-score-params")
        except (TypeError, ValueError) as e:
            ap.error(f"--plan-score-params: {e}")
    if args.out:
        # the report's folder is made before any episode flies, as the command
        # line makes its files' folders: a long run never ends on a missing one
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # logs below errors are off while the episodes fly, and put back after them
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    t0 = time.perf_counter()
    try:
        report = headroom_report(args.cells, episodes=args.episodes, start=args.start,
                                 max_leaves=args.max_leaves, workers=args.workers,
                                 plan_score_params=plan)
    except ValueError as e:
        ap.error(str(e))
    finally:
        logging.disable(previous)
    print(format_report(report))
    print(f"({time.perf_counter() - t0:.0f} s)")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
