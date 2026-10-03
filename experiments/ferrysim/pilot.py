"""Budget pilots and decision cost in FerrySim (Exp 5 addendum, Study 5.11 (a) and (c)).

Study 5.11 (c) flies FerrySim beyond the stack (N = 24, 48 and 96, the scale
family, ``cells.SCALE_CELLS``), and its cells fly stand-in budgets until a
budget pilot measures each size's knee; its first step is to measure what an
episode costs at N = 96, which nothing has. Study 5.11 (a) needs the
planner's and the flight slot's wall time against N, in process, with the
search forced into each mode. This module does both, on the validation stream
only (``ferrysim-val``: a pilot never reads Study 5.5's held-out episodes):

* :func:`pilot` flies every (cell, budget, policy) over the first ``episodes``
  episodes of each cell's validation stream (common random numbers: every
  budget and policy meets the same layouts), each budget overriding the
  cell's own (``mission_budget_s``), and summarises each episode
  (:func:`run_pilot_task`): the evaluator's summary (``evaluate.episode_summary``),
  the served share per mission (updates collected over the cell's N, and
  over the plan's demand), and the wall times: the episode's own, the
  planner's per mission (``plan_wall_s``) and each flight decision's
  (``decide_s``, ``mask_s``: the mule's ``pass_1_pairs_wall`` and
  ``pass_1_e3_wall``);
* :func:`pilot_table` folds them per (cell, budget, policy): the means, and
  the 95th percentiles of the wall times. The knee is read off the table, as
  the stack's pilot reads it (where the served share stops rising;
  Experiment_4_Run_Guide.md section 2.6): this module names no knee.

Wall times are measured, never decided on: nothing here enters a decision,
a reward, a checkpoint or a determinism comparison (``EpisodeResult.walls``
is left out of equality). ``plan_search_params`` (``--plan-search-params``)
forces the planner's mode as the runner's flag does, and ``trace_root``
keeps each episode's traces, per budget, for the trace scorer's cost columns
(``traces_scorer --cost-columns``).

Nothing here has run as a study; each pilot waits for the user's go-ahead.

    python -m experiments.ferrysim pilot --cells scl-n96-1330 --budgets 1000 1330 1700 \\
        --policies FX greedy_1 --episodes 20 --workers 4 --out pilot.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from experiments.ferrysim import cells as C
from experiments.ferrysim import inprocess
from experiments.ferrysim.episode import Policy, reference_policies, run_episode
from experiments.ferrysim.evaluate import episode_summary, parallel_map

__all__ = [
    "PilotTask",
    "main",
    "pilot",
    "pilot_table",
    "run_pilot_task",
]


@dataclasses.dataclass(frozen=True)
class PilotTask:
    """One pilot episode: the cell, its validation episode, the policy, the
    budget that overrides the cell's (None: the cell's own), the device
    model and the driver settings every episode adds (``run_episode``'s
    ``driver_overrides``)."""

    cell: C.FerryCell
    index: int
    seed: int
    policy: Policy
    budget_s: Optional[float] = None
    device_model: str = inprocess.DEVICE_MODEL_EQUAL
    overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    trace_root: Optional[str] = None

    @property
    def budget(self) -> float:
        """The budget the episode flies."""
        return float(self.cell.budget_s if self.budget_s is None else self.budget_s)


def _budget_dir(budget_s: float) -> str:
    return f"budget={budget_s:g}"


def run_pilot_task(task: PilotTask) -> Dict[str, Any]:
    """Fly one :class:`PilotTask`; its summary (a worker's unit of work)."""
    overrides = dict(task.overrides)
    if task.budget_s is not None:
        overrides["mission_budget_s"] = float(task.budget_s)
    if task.trace_root is not None:
        overrides["trace_root"] = str(Path(task.trace_root) / _budget_dir(task.budget))
    started = time.perf_counter()
    result = run_episode(task.cell, task.seed, task.policy, trial_index=task.index,
                         device_model=task.device_model,
                         driver_overrides=overrides or None)
    wall_s = time.perf_counter() - started
    out = episode_summary(result)
    n = int(task.cell.n_devices)
    out.update({
        "stream": C.VAL_STREAM,
        "budget_s": task.budget,
        "n_devices": n,
        "served_share": [len(s.collected) / n for s in result.sorties],
        "served_of_demand": [len(s.collected) / s.n_demand if s.n_demand else None
                             for s in result.sorties],
        "wall_s": wall_s,
        "plan_wall_s": [w.get("plan_wall_s") for w in result.walls],
        "decide_s": [d["decide_s"] for w in result.walls for d in w["pairs"] + w["e3"]],
        "mask_s": [d["mask_s"] for w in result.walls for d in w["pairs"] + w["e3"]],
    })
    return out


def pilot(
    cells: Iterable[Any],
    policies: Sequence[Policy],
    *,
    budgets: Optional[Sequence[float]] = None,
    episodes: int = 20,
    start: int = 0,
    device_model: str = inprocess.DEVICE_MODEL_EQUAL,
    workers: int = 1,
    driver_overrides: Optional[Mapping[str, Any]] = None,
    trace_root: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Every (cell, budget, policy) on episodes ``start`` .. ``start + episodes - 1``
    of each cell's validation stream; one summary each, cell-major, then budget,
    then episode, then policy. ``budgets`` None flies each cell's own; each
    budget must be > 0. ``trace_root`` keeps the traces under
    ``<trace_root>/budget=<b>/<cell>/<policy>/``."""
    resolved = [C.cell_named(cell) for cell in cells]
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError(f"episodes is an int >= 1, got {episodes!r}")
    if budgets is not None:
        budgets = [float(b) for b in budgets]
        bad = [b for b in budgets if not (math.isfinite(b) and b > 0.0)]
        if bad or not budgets:
            raise ValueError(f"budgets are seconds > 0, got {list(budgets)!r}")
    if not policies:
        raise ValueError("give at least one policy")
    overrides = dict(driver_overrides or {})
    for key in ("mission_budget_s", "trace_root"):
        if key in overrides:
            raise ValueError(f"{key} is the pilot's own setting (budgets, trace_root)")
    tasks = []
    for cell in resolved:
        seeds = C.stream_seeds(C.VAL_STREAM, cell.name, int(episodes), start=int(start))
        for budget in (budgets if budgets is not None else [None]):
            for e, seed in enumerate(seeds):
                for policy in policies:
                    tasks.append(PilotTask(cell=cell, index=int(start) + e, seed=seed,
                                           policy=policy, budget_s=budget,
                                           device_model=device_model, overrides=overrides,
                                           trace_root=trace_root))
    return parallel_map(run_pilot_task, tasks, workers=workers)


def _p95(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    # numpy's linear rule, as the trace scorer's p95 columns
    k = (len(ordered) - 1) * 0.95
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    return statistics.fmean(present) if present else None


def pilot_table(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (cell, budget, policy), in the summaries' first-seen order.

    ``served_share_mean`` is the mean over every mission of every episode of
    the updates collected over the cell's N (``served_of_demand_mean`` over
    the plan's demand instead); ``return_mean`` the mean undiscounted return;
    ``decisions_per_sortie_mean`` the Pass-1 stops flown per mission; and the
    wall times' means and 95th percentiles (numpy's linear rule): the
    episode's (``episode_wall_s``), the planner's per mission and the flight
    decisions' (pooled; blank, None, where none was timed).
    """
    groups: Dict[Tuple[str, float, str], List[Mapping[str, Any]]] = {}
    for s in summaries:
        groups.setdefault((s["cell"], float(s["budget_s"]), s["policy"]), []).append(s)
    rows = []
    for (cell, budget, policy), group in groups.items():
        plan = [float(v) for s in group for v in s["plan_wall_s"] if v is not None]
        decide = [float(v) for s in group for v in s["decide_s"]]
        walls = [float(s["wall_s"]) for s in group]
        rows.append({
            "cell": cell,
            "budget_s": budget,
            "policy": policy,
            "n_devices": int(group[0]["n_devices"]),
            "episodes": len(group),
            "served_share_mean": _mean(v for s in group for v in s["served_share"]),
            "served_of_demand_mean": _mean(v for s in group for v in s["served_of_demand"]),
            "return_mean": _mean(s["return"] for s in group),
            "decisions_per_sortie_mean": _mean(d for s in group for d in s["decisions"]),
            "episode_wall_s_mean": _mean(walls),
            "episode_wall_s_p95": _p95(walls),
            "episode_wall_s_max": max(walls),
            "plan_wall_s_mean": _mean(plan),
            "plan_wall_s_p95": _p95(plan),
            "decide_s_mean": _mean(decide),
            "decide_s_p95": _p95(decide),
            "mask_s_mean": _mean(float(v) for s in group for v in s["mask_s"]),
        })
    return rows


def _policies(names: Optional[Sequence[str]]) -> Tuple[Policy, ...]:
    """Reference labels (FX, F and the scripted ones); any other name flies
    that driver arm's own configuration (``Policy.of_arm``)."""
    refs = {p.label: p for p in reference_policies()}
    if not names:
        return (refs["FX"],)
    return tuple(refs.get(n) or Policy.of_arm(n) for n in names)


def _fmt(value: Optional[float], spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.ferrysim pilot",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", required=True,
                    help="FerrySim cells by name (e.g. scl-n96-1330).")
    ap.add_argument("--budgets", nargs="+", type=float, default=None,
                    help="Mission budgets (s) that override each cell's own; default the "
                         "cell's own budget.")
    ap.add_argument("--policies", nargs="+", default=None,
                    help="Reference labels (FX, F, fx_pair, committed_pair, hyb, greedy_1) or "
                         "driver arms; default FX.")
    ap.add_argument("--episodes", type=int, default=20,
                    help="Validation episodes per (cell, budget, policy); default 20.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device-model", choices=inprocess.DEVICE_MODELS,
                    default=inprocess.DEVICE_MODEL_EQUAL)
    ap.add_argument("--plan-search-params", default=None,
                    help="JSON PlanSearchParams, as the runner's flag (Study 5.11 (a) forces "
                         "the planner's mode with it).")
    ap.add_argument("--trace-root", default=None,
                    help="Keep each episode's traces here, per budget, for traces_scorer "
                         "--cost-columns.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="Write the summaries and the table here (JSON).")
    args = ap.parse_args(argv)
    overrides: Dict[str, Any] = {}
    if args.plan_search_params is not None:
        try:
            params = json.loads(args.plan_search_params)
        except json.JSONDecodeError as e:
            ap.error(f"--plan-search-params: {e}")
        if not isinstance(params, dict):
            ap.error("--plan-search-params takes a JSON object")
        overrides["plan_search_params"] = params
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        summaries = pilot(args.cells, _policies(args.policies), budgets=args.budgets,
                          episodes=args.episodes, start=args.start,
                          device_model=args.device_model, workers=args.workers,
                          driver_overrides=overrides, trace_root=args.trace_root)
    except ValueError as e:
        ap.error(str(e))
    finally:
        logging.disable(previous)
    table = pilot_table(summaries)
    print(f"{'cell':14s} {'budget':>7s} {'policy':15s} {'n':>4s} {'served':>7s} "
          f"{'of dem.':>7s} {'stops':>6s} {'episode s':>10s} {'plan s':>8s} {'decide ms':>10s}")
    for r in table:
        print(f"{r['cell']:14s} {r['budget_s']:7g} {r['policy']:15s} {r['episodes']:4d} "
              f"{_fmt(r['served_share_mean'], '7.3f')} {_fmt(r['served_of_demand_mean'], '7.3f')} "
              f"{_fmt(r['decisions_per_sortie_mean'], '6.2f')} "
              f"{_fmt(r['episode_wall_s_mean'], '10.2f')} {_fmt(r['plan_wall_s_mean'], '8.3f')} "
              f"{_fmt(None if r['decide_s_mean'] is None else 1e3 * r['decide_s_mean'], '10.3f')}")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summaries": summaries, "table": table,
                       "settings": {"cells": list(args.cells), "budgets": args.budgets,
                                    "policies": [p.label for p in _policies(args.policies)],
                                    "episodes": args.episodes, "start": args.start,
                                    "device_model": args.device_model,
                                    "overrides": overrides}},
                      fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
