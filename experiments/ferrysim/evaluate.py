"""Held-out and validation returns of FerrySim policies (FeRRy Phase 5, unit U8a).

Study 5.5's judgement (the user's decision 5 (a); the spec, other choices 12)
scores every checkpoint and every reference on one shared set of episodes per
cell, its held-out stream (common random numbers), by their undiscounted
return; the headroom report and ε read the validation stream instead (critic
B14). This module runs those episodes and summarises them:

* :func:`evaluate` runs every (cell, policy, episode) of a stream, in this
  process or in worker processes (:func:`parallel_map`), each episode as a
  compact JSON summary (:func:`episode_summary`: the return and its terms,
  the decisions per sortie);
* :func:`score_summary` is the held-out score a checkpoint's manifest records
  (``pair_q.HELD_OUT_KEYS``: ``episodes`` and ``return_mean``, plus the rest
  of the summary), and :func:`record_held_out_score` writes it there
  (``pair_q.record_held_out``);
* the policies are :class:`~experiments.ferrysim.episode.Policy` values: the
  FX and F arms (resolution R6), U3's scripted references in the pair slot,
  or any scorer factory, a checkpoint's included (unit U8b builds that
  factory; this module does not load checkpoints itself);
* :func:`fx_lags` and :func:`fx_lag_median` are Study 5.6's lag measurement
  (the orchestrator's resolution R22; critic A3): FX's arrival-to-arrival lags
  on the validation stream, whose pooled median, rounded, pins
  ``cells.STUDY_5_6_LAGS_S`` and is measured again when the cells are
  re-pinned.

**Workers** (the spec, other choices 8; risk R10). The in-process trial
patches process-wide names, so a process runs one episode at a time; parallel
runs use worker processes started with Windows' ``spawn``, each with
``OPENBLAS_NUM_THREADS=1`` (set before any worker starts, so each worker's
numpy reads it) and its own drivers, so T_nom is computed once per cell per
worker. Results come back in task order, so a run's output does not depend on
the number of workers.

Command line (validation stream, the references, four workers)::

    python -m experiments.ferrysim.evaluate --cells jit-n12-90 --stream val \\
        --episodes 50 --workers 4 --out returns.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import statistics
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from experiments.ferrysim import cells as C
from experiments.ferrysim import inprocess
from experiments.ferrysim.episode import ARM_FX, Policy, reference_policies, run_episode
from experiments.ferrysim.reward import DERIVED, RewardSpec

STREAMS = {C.STREAM_VAL: C.VAL_STREAM, C.STREAM_HELDOUT: C.HELDOUT_STREAM}
#: Study 5.6's lag sample (resolution R22): the first 200 episodes of the
#: validation stream of the Study 5.5 cell the lag is measured on.
LAG_EPISODES = 200


# --------------------------------------------------------------------------- #
# Worker processes
# --------------------------------------------------------------------------- #

def _init_worker() -> None:
    """A FerrySim worker: one BLAS thread, and HERMES' logs at errors only."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    logging.disable(logging.WARNING)


def parallel_map(fn: Callable[[Any], Any], tasks: Sequence[Any], *, workers: int = 1) -> List[Any]:
    """``[fn(t) for t in tasks]``, in order, with ``workers`` spawned processes.

    ``workers`` 1 runs in this process. Otherwise ``fn`` and each task must be
    importable values (module-level functions, frozen dataclasses,
    ``functools.partial`` of module-level functions), since each worker is a
    fresh interpreter (``spawn``). ``OPENBLAS_NUM_THREADS`` is set to 1 in this
    process's environment before the workers start, so they inherit it.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers is an int >= 1, got {workers!r}")
    if workers == 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        return list(pool.map(fn, tasks, chunksize=1))


# --------------------------------------------------------------------------- #
# Episodes as summaries
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class EpisodeTask:
    """One episode to run: the cell, the stream, its seed and index, the policy,
    the reward, the device model, and the driver settings it adds to the cell's
    (``run_episode``'s ``driver_overrides``: a plan's score settings, the
    orchestrator's resolution R23; none by default)."""

    cell: C.FerryCell
    stream: str
    index: int
    seed: int
    policy: Policy
    reward: RewardSpec = DERIVED
    device_model: str = inprocess.DEVICE_MODEL_EQUAL
    overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def episode_summary(result) -> Dict[str, Any]:
    """A compact JSON summary of an episode: the return, its terms, and per
    sortie its decisions and return (the evaluator's unit of record)."""
    return {
        "cell": result.cell,
        "seed": int(result.seed),
        "index": int(result.trial_index),
        "policy": result.policy,
        "arm": result.arm,
        "return": result.ret,
        "terms": result.terms.to_json(),
        "missions": len(result.sorties),
        "decisions": [len(s.stops) for s in result.sorties],
        "sortie_returns": list(result.sortie_returns),
        "collected": [len(s.collected) for s in result.sorties],
        "demand": [int(s.n_demand) for s in result.sorties],
        "undecided_shortfall": [s.undecided_shortfall for s in result.sorties],
    }


def run_task(task: EpisodeTask) -> Dict[str, Any]:
    """Run one :class:`EpisodeTask`; its summary (a worker's unit of work)."""
    extra = {"driver_overrides": dict(task.overrides)} if task.overrides else {}
    result = run_episode(task.cell, task.seed, task.policy, trial_index=task.index,
                         reward=task.reward, device_model=task.device_model, **extra)
    out = episode_summary(result)
    out["stream"] = task.stream
    return out


def evaluate(
    cells: Iterable[Any],
    policies: Sequence[Policy],
    *,
    stream: str = C.HELDOUT_STREAM,
    episodes: int = 1000,
    start: int = 0,
    reward: RewardSpec = DERIVED,
    device_model: str = inprocess.DEVICE_MODEL_EQUAL,
    workers: int = 1,
    driver_overrides: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Every policy on episodes ``start`` .. ``start + episodes - 1`` of each cell's
    ``stream``: one summary per (cell, episode, policy), cell-major, then
    episode, then policy, so every policy meets the same episodes (common
    random numbers). ``cells`` are FerrySim cells or their names;
    ``driver_overrides`` are settings every episode adds to its cell's (the
    plan the policies fly, resolution R23)."""
    resolved = [C.cell_named(cell) for cell in cells]
    if C.stream_kind(stream) == C.STREAM_TRAIN:
        raise ValueError("evaluation reads the validation or held-out stream, never a "
                         "training stream (critic B14)")
    overrides = dict(driver_overrides or {})
    tasks = []
    for cell in resolved:
        seeds = C.stream_seeds(stream, cell.name, int(episodes), start=int(start))
        for e, seed in enumerate(seeds):
            for policy in policies:
                tasks.append(EpisodeTask(cell=cell, stream=stream, index=int(start) + e,
                                         seed=seed, policy=policy, reward=reward,
                                         device_model=device_model, overrides=overrides))
    return parallel_map(run_task, tasks, workers=workers)


# --------------------------------------------------------------------------- #
# Study 5.6's lag (critic A3; the orchestrator's resolution R22)
# --------------------------------------------------------------------------- #

def lags_task(task: EpisodeTask) -> List[float]:
    """One episode's arrival-to-arrival lags (``cells.arrival_lags``; a worker's
    unit of work)."""
    result = run_episode(task.cell, task.seed, task.policy, trial_index=task.index,
                         reward=task.reward, device_model=task.device_model)
    return C.arrival_lags(result.sorties)


def fx_lags(cell: Any, *, episodes: int = LAG_EPISODES, workers: int = 1) -> List[float]:
    """FX's arrival-to-arrival lags over the first ``episodes`` episodes of ``cell``'s
    validation stream, pooled in episode order (critic A3's lag; resolution R22).

    The FX arm itself flies each episode (resolution R6) at the cell's own
    interference period, as :func:`evaluate` flies it: the equal-shard device
    model, ``trial_index`` the episode's index in the stream, nothing trained.
    A lag runs from one Pass-1 arrival to the next in the same sortie
    (``cells.arrival_lags``). ``cell`` is a FerrySim cell or its name;
    ``workers`` as :func:`parallel_map`, which keeps the episodes' order.
    """
    resolved = C.cell_named(cell)
    fx = Policy.of_arm(ARM_FX)
    tasks = [EpisodeTask(cell=resolved, stream=C.VAL_STREAM, index=e, seed=seed, policy=fx)
             for e, seed in enumerate(C.stream_seeds(C.VAL_STREAM, resolved.name,
                                                     int(episodes)))]
    return [lag for lags in parallel_map(lags_task, tasks, workers=workers) for lag in lags]


def fx_lag_median(cell: Any, *, episodes: int = LAG_EPISODES, workers: int = 1) -> float:
    """R22's lag before it is rounded: the median of :func:`fx_lags`, pooled over
    every lag of the sample, as critic A3's probe pooled them (the median of
    each episode's median is another statistic, 27.99 s and 37.94 s on the
    pinned sample). Rounded to the nearest second, it is the cell's entry of
    ``cells.STUDY_5_6_LAGS_S``; ValueError when the episodes flew no lag."""
    lags = fx_lags(cell, episodes=episodes, workers=workers)
    if not lags:
        raise ValueError(f"FX flew no lag in the first {episodes} validation episode(s) of "
                         f"{C.cell_named(cell).name}: no sortie made two Pass-1 stops")
    return statistics.median(lags)


# --------------------------------------------------------------------------- #
# Summaries and the held-out score
# --------------------------------------------------------------------------- #

def _mean_sd_se(values: Sequence[float]) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return math.nan, math.nan, math.nan
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return mean, sd, sd / math.sqrt(n)


def score_summary(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The held-out score of one policy on one cell, from its episode summaries.

    ``episodes`` and ``return_mean`` (the mean undiscounted return, the spec's
    score; ``pair_q.HELD_OUT_KEYS``), with the standard deviation and error,
    the terms' means, the decisions per sortie, the cell, the stream and the
    policy. Refuses summaries of more than one cell, stream or policy.
    """
    if not summaries:
        raise ValueError("a score needs at least one episode")
    for key in ("cell", "stream", "policy"):
        values = {s.get(key) for s in summaries}
        if len(values) != 1:
            raise ValueError(f"a score is of one {key}: got {sorted(map(str, values))}")
    returns = [float(s["return"]) for s in summaries]
    mean, sd, se = _mean_sd_se(returns)
    terms = {}
    for name in ("gain", "time", "energy", "distance", "coverage"):
        terms[name] = statistics.fmean(float(s["terms"][name]) for s in summaries)
    sorties = [d for s in summaries for d in s["decisions"]]
    return {
        "episodes": len(summaries),
        "return_mean": mean,
        "return_sd": sd,
        "return_se": se,
        "terms_mean": terms,
        "decisions_per_sortie_mean": statistics.fmean(sorties) if sorties else 0.0,
        "sorties_with_2_or_more": (sum(1 for d in sorties if d >= 2) / len(sorties)
                                   if sorties else 0.0),
        "cell": summaries[0]["cell"],
        "stream": summaries[0]["stream"],
        "policy": summaries[0]["policy"],
        "first_index": min(int(s["index"]) for s in summaries),
    }


def scores_by(summaries: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """:func:`score_summary` per (cell, policy)."""
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for s in summaries:
        groups.setdefault((s["cell"], s["policy"]), []).append(s)
    return {key: score_summary(group) for key, group in groups.items()}


def record_held_out_score(checkpoint_path: str, summaries: Sequence[Mapping[str, Any]]):
    """Write a checkpoint's held-out score into its manifest; the manifest.

    ``pair_q.record_held_out`` (unit U2) with :func:`score_summary` of the
    checkpoint's held-out episodes (one cell, the held-out stream, one
    policy), which carries ``pair_q.HELD_OUT_KEYS`` and no wall time.
    """
    score = score_summary(summaries)
    if C.stream_kind(score["stream"]) != C.STREAM_HELDOUT:
        raise ValueError(f"a held-out score reads {C.HELDOUT_STREAM!r}, got {score['stream']!r}")
    from hermes.scheduler.selector.pair_q import record_held_out

    return record_held_out(checkpoint_path, score)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _policies(names: Optional[Sequence[str]]) -> Tuple[Policy, ...]:
    refs = {p.label: p for p in reference_policies()}
    if not names:
        return tuple(refs.values())
    unknown = [n for n in names if n not in refs]
    if unknown:
        raise ValueError(f"unknown policies {unknown}; the references are {sorted(refs)}")
    return tuple(refs[n] for n in names)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.ferrysim.evaluate",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=[c.name for c in C.STUDY_5_5_CELLS])
    ap.add_argument("--stream", choices=sorted(STREAMS), default=C.STREAM_HELDOUT)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--policies", nargs="+", default=None,
                    help="Reference labels (default: FX, F and the four scripted ones).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=None, help="Write the summaries and scores here (JSON).")
    args = ap.parse_args(argv)
    if args.out:
        # the file's folder is made before any episode flies, as the command
        # line makes its files' folders: a long run never ends on a missing one
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # logs below errors are off while the episodes fly, and put back after them
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        summaries = evaluate(args.cells, _policies(args.policies), stream=STREAMS[args.stream],
                             episodes=args.episodes, start=args.start, workers=args.workers)
    except ValueError as e:
        ap.error(str(e))
    finally:
        logging.disable(previous)
    scores = scores_by(summaries)
    for (cell, policy), score in sorted(scores.items()):
        print(f"{cell:12s} {policy:15s} n={score['episodes']:5d} "
              f"return {score['return_mean']:+.4f} +- {score['return_se']:.4f} "
              f"decisions/sortie {score['decisions_per_sortie_mean']:.2f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summaries": summaries,
                       "scores": [dict(score) for score in scores.values()]},
                      fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
