"""O1: the offline oracle behind Study 5.4's optimality gap (after Zhai et al., TWC 2025).

The build plan's O1 (Baselines; decided 2026-10-05): for N <= 6, search every
committed band class x grouping of devices into stops x stop order x band per
stop, and report how far F's plan falls short, ``V_O1 - V_F >= 0``. It is a
planning-level tool, never flown.

**Where the missions come from.** F flies FerrySim episodes in process
(``experiments.ferrysim.episode.run_episode``), and an ``on_mule`` hook wraps
the scheduler's ``build_ferry_plan``: each mission F plans as it always does,
then the oracle searches on exactly the inputs that plan had (the demand, the
coverage weights, S3's deadlines, the age cap, the start state, the budget's
end, each class's physics and its Pass 2, the score's settings). So the gap is
read on F's own mission states, ages and caps included, not on fresh layouts.

**What the oracle searches.** A plan is an ordered sequence of disjoint groups
of demanded devices. Each group is served at one position, on one class whose
planar reach covers every member from it (the gate's range; a member beyond
it would count as served at outage 1 with no dwell, so it is refused). The
positions offered to a group are its centroid and each member's position (S3a's
two rules), and every stop F was offered on any class (S3a's and the hover
rule's), so the oracle's family contains F's: its best is never worse than F's
under F's own key. Each stop is admitted by the class's predicate from the
flight state the route has reached, as F's search admits it (S3b, the
deadline-and-budget rule, an exempt stop protected); the classes share the
dock, the speed and the hover power, so one state threads through them. The
committed class b̄ prices Pass 2 and nothing else, as F's plan does.

**How it scores.** Exactly as F's search: ``plan_score.score`` with each served
member's outage on its own stop's class, Pass 1's home, dwell and energy (the
return leg included), the turnaround and b̄'s Pass 2, and ``cap_key``. Two
optima are reported: the best under F's own plan key (cap key, then the
served share under the default lexicographic rank, then V), which is the gap
the arm could close, and the best V alone. A branch is pruned only when an
earlier one served the same devices, stands at the same position, and is no
later, on no more energy, with no more link loss (and, where Delta leaves dwell
out, no more dwell): every extension of the pruned branch is then no better.

    python -m experiments.analysis.o1_oracle --cells jit-n6-75 jit-n6-150 --episodes 30 --out o1.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from hermes.scheduler.plan.member_subset import pass_energy_j
from hermes.scheduler.plan.plan_score import applied_rank, score, served_share
from hermes.scheduler.plan.plan_search import price_pass_2
from hermes.scheduler.plan.types import COVERAGE_RANK_WEIGHTED
from hermes.scheduler.stages.s3a_cluster import _centroid, _worst_bucket
from hermes.scheduler.stages.s3b_feasibility import RULE_DEADLINE_BUDGET, FlightState
from hermes.scheduler.stages.s3d_age_cap import (
    cap_key, evaluate_cap, is_exempt, with_cap_deadlines,
)
from hermes.types import DeviceID
from hermes.types.scheduler import ContactWaypoint, MissionPass

__all__ = [
    "MAX_DEVICES",
    "MissionGap",
    "OracleSearch",
    "mission_gap",
    "oracle_hook",
    "main",
]

#: The build plan's bound on the oracle (N <= 6).
MAX_DEVICES = 6
_COLLECT = MissionPass.COLLECT
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# One mission's inputs, as F's plan had them
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MissionInputs:
    """What ``build_ferry_plan`` planned on, recomputed exactly as it computes them."""

    mission_round: int
    demand: Tuple[DeviceID, ...]
    weights: Mapping[DeviceID, float]
    deadlines: Mapping[DeviceID, float]
    cap: Any                                   # s3d_age_cap.CapState
    start: FlightState
    budget_end: Optional[float]
    classes: Tuple[Any, ...]                   # plan.types.PlanClass, the arm's searched
    pass_2: Mapping[str, Any]                  # class name -> plan_search.PassTwo
    offered: Mapping[str, Tuple[ContactWaypoint, ...]]   # F's offered stops per class
    positions: Mapping[DeviceID, Tuple[float, float, float]]
    buckets: Mapping[DeviceID, Any]
    params: Any                                # plan.types.PlanScoreParams
    turnaround_s: float
    t_ref_s: float
    p_hover_w: float
    f_band: str
    f_route: Tuple[ContactWaypoint, ...]
    f_v: float


def capture(sch, *, now: float) -> MissionInputs:
    """``sch``'s last plan's inputs, right after ``build_ferry_plan`` committed it."""
    from hermes.scheduler.plan.hover import offer_hover_stops
    from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range

    setup = sch._plan
    commit = sch.last_plan
    options = setup.options
    states = sch.device_states
    demand = tuple(commit.demand)
    deadlines = dict(sch.last_plan_deadlines)
    cap = evaluate_cap(list(demand), states, mission_round=commit.mission_round,
                       spec=options.cap)
    dock = tuple(setup.classes[0].model.ferry.dock)
    start = FlightState(dock, float(now))
    pass_2, offered = {}, {}
    for c in setup.searched:
        stops = cluster_by_rf_range(eligible_device_ids=list(demand), device_states=states,
                                    deadlines=deadlines, rf_range_m=c.radius_m)
        if options.search.hover_stops:
            stops = offer_hover_stops(
                stops, model=c.model, reach_m=c.radius_m, movable=cap.capped, start=start,
                budget_end=commit.budget_end, deadlines=deadlines, device_states=states,
                capped=cap.capped)
        offered[c.name] = tuple(stops)
        pass_2[c.name] = price_pass_2(c.model, sch.build_pass_2_queue(
            rf_range_m=c.radius_m, now=now, mule_pose=dock))
    return MissionInputs(
        mission_round=int(commit.mission_round), demand=demand, weights=dict(commit.weights),
        deadlines=deadlines, cap=cap, start=start, budget_end=commit.budget_end,
        classes=tuple(setup.searched), pass_2=pass_2, offered=offered,
        positions={d: tuple(states[d].last_known_position) for d in demand},
        buckets={d: states[d].bucket for d in demand}, params=options.score,
        turnaround_s=float(setup.turnaround_s), t_ref_s=float(setup.t_ref_s),
        p_hover_w=float(setup.p_hover_w), f_band=commit.band, f_route=tuple(commit.queue),
        f_v=float(commit.score["v"]),
    )


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #

@dataclass
class _Label:
    state: FlightState
    home: float
    link: float          # sum of w * outage over the served
    dwell: float         # Pass 1's predicted dwell
    route: Tuple[Tuple[ContactWaypoint, str], ...]
    outage: Mapping[DeviceID, float]


@dataclass(frozen=True)
class Scored:
    """One plan as the oracle prices it."""

    band: str
    route: Tuple[Tuple[ContactWaypoint, str], ...]   # each stop and the class it is flown on
    served: FrozenSet[DeviceID]
    v: float
    share: float
    cap_key: Tuple[int, ...]
    key: Tuple[Any, ...]


class OracleSearch:
    """The exhaustive search over one mission's inputs."""

    def __init__(self, inp: MissionInputs, *, max_devices: int = MAX_DEVICES,
                 max_expansions: int = 20_000_000) -> None:
        if len(inp.demand) > max_devices:
            raise ValueError(f"O1 is bounded to N <= {max_devices}; this mission demands "
                             f"{len(inp.demand)} devices")
        self.inp = inp
        self.max_expansions = int(max_expansions)
        self.by_name = {c.name: c for c in inp.classes}
        self.capped = frozenset(inp.cap.capped)
        self.dwell_in_delta = bool(inp.params.dwell_in_delta)
        self.lexicographic = applied_rank(inp.params) != COVERAGE_RANK_WEIGHTED
        self.options = self._options()
        self.expansions = 0
        self.plans = 0
        self.pruned = 0

    # -- the stops a group may be served at ----------------------------------- #

    def _options(self) -> List[Tuple[FrozenSet[DeviceID], ContactWaypoint, str, bool, Dict]]:
        inp = self.inp
        demand = list(inp.demand)
        # F flies an offered stop, or a subset of its members, at the stop's own
        # position: offering that position to every subset of its members keeps
        # F's whole family inside the oracle's.
        offered = [(frozenset(wp.devices), tuple(wp.position))
                   for stops in inp.offered.values() for wp in stops]
        out = []
        for k in range(1, len(demand) + 1):
            for group in itertools.combinations(demand, k):
                pts = [inp.positions[d] for d in group]
                members = frozenset(group)
                candidates = ({tuple(_centroid(pts))} | {tuple(p) for p in pts}
                              | {pos for devs, pos in offered if members <= devs})
                deadline = min(inp.deadlines.get(d, math.inf) for d in group)
                bucket = _worst_bucket([inp.buckets[d] for d in group])
                for pos in sorted(candidates):
                    wp = ContactWaypoint(position=pos, devices=tuple(group), bucket=bucket,
                                         deadline_ts=deadline)
                    (wp,) = with_cap_deadlines([wp], deadlines=inp.deadlines,
                                               capped=self.capped)
                    for cls in inp.classes:
                        dists = cls.model.ferry.member_distances_m(wp)
                        if any(dd > cls.radius_m + _EPS for dd in dists):
                            continue            # a member beyond the class's reach
                        outage = {d: float(cls.outage(dd)) for d, dd in zip(group, dists)}
                        out.append((frozenset(group), wp, cls.name,
                                    is_exempt(wp, self.capped), outage))
        return out

    # -- pricing ----------------------------------------------------------------- #

    def _score(self, label: _Label, band: str) -> Scored:
        inp = self.inp
        bar = self.by_name[band]
        served = frozenset(label.outage)
        p2 = inp.pass_2[band]
        energy = pass_energy_j(bar.model, label.state)
        terms = score(
            inp.params, weights=inp.weights, served_outage=dict(label.outage),
            pass_1_s=label.home - inp.start.clock, pass_1_dwell_s=label.dwell,
            pass_1_energy_j=energy, turnaround_s=inp.turnaround_s,
            pass_2_s=p2.time_s, pass_2_dwell_s=p2.dwell_s, pass_2_energy_j=p2.energy_j,
            t_ref_s=inp.t_ref_s, p_hover_w=inp.p_hover_w,
        )
        ck = cap_key(served, inp.cap)
        share = served_share(terms)
        v = float(terms.v)
        key = ((ck, -round(share, 9), -round(v, 9)) if self.lexicographic
               else (ck, -round(v, 9)))
        return Scored(band=band, route=label.route, served=served, v=v, share=float(share),
                      cap_key=ck, key=key)

    def price_route(self, route: Sequence[Tuple[ContactWaypoint, str]], band: str) -> Scored:
        """One given route (each stop on its class) priced as the search prices it;
        a stop the predicate refuses raises."""
        inp = self.inp
        state = inp.start
        label = _Label(state, self.by_name[band].model.home_at(state), 0.0, 0.0, (), {})
        for wp, cname in route:
            cls = self.by_name[cname]
            label = self._extend(label, wp, cls, is_exempt(wp, self.capped),
                                 {d: float(cls.outage(dd)) for d, dd in
                                  zip(wp.devices, cls.model.ferry.member_distances_m(wp))})
            if label is None:
                raise ValueError(f"the predicate refuses {list(wp.devices)} on {cname}")
        return self._score(label, band)

    def _extend(self, label: _Label, wp: ContactWaypoint, cls, protected: bool,
                outage: Mapping[DeviceID, float]) -> Optional[_Label]:
        v = cls.model.admit(label.state, wp, rule=RULE_DEADLINE_BUDGET,
                            budget_end=self.inp.budget_end, pass_kind=_COLLECT,
                            protected=protected)
        if not v.ok:
            return None
        w = self.inp.weights
        return _Label(
            state=v.next_state, home=v.home,
            link=label.link + math.fsum(w[d] * p for d, p in outage.items()),
            dwell=label.dwell + cls.model.ferry.dwell_s(wp, _COLLECT),
            route=label.route + ((wp, cls.name),),
            outage={**label.outage, **outage},
        )

    def _v_upper(self, label: _Label) -> float:
        """A bound on V over every extension of ``label`` (one stop or more).

        Along an extension the home (back at the dock with the upload done),
        Pass 1's energy with the return leg, and the link loss can only grow:
        the triangle inequality bounds the detour, and the upload grows with
        what is on board. Coverage can at best reach every device. So V is at
        most the score of the label's own home, energy and link with nothing
        left uncovered, on the cheapest Pass 2 of any class."""
        inp = self.inp
        c1, _c2, c3, c4 = inp.params.constants(len(inp.demand))
        p2_s = min(p.time_s for p in inp.pass_2.values())
        p2_e = min(p.energy_j for p in inp.pass_2.values())
        delta = (label.home - inp.start.clock) + inp.turnaround_s + p2_s
        energy = pass_energy_j(inp.classes[0].model, label.state) + p2_e
        demand_weight = math.fsum(inp.weights[d] for d in inp.demand)
        link = label.link / demand_weight if demand_weight > 0.0 else 0.0
        e_term = energy / (inp.p_hover_w * inp.t_ref_s) if inp.p_hover_w > 0.0 else 0.0
        return 0.0 - (c1 * (delta / inp.t_ref_s) ** 2 + c3 * link) - c4 * e_term

    def _hopeless(self, label: _Label, best_key: Scored, best_v: Scored) -> bool:
        """No extension of ``label`` can beat either best (F's default Delta only)."""
        if not self.dwell_in_delta or not label.route:
            return False
        ub = self._v_upper(label) + 1e-9
        if ub > best_v.v:
            return False
        lower = ((), -1.0, -round(ub, 9)) if self.lexicographic else ((), -round(ub, 9))
        return not lower < best_key.key

    @staticmethod
    def _dominates(a: _Label, b: _Label, with_dwell: bool) -> bool:
        return (a.state.clock <= b.state.clock + _EPS
                and a.state.energy_j <= b.state.energy_j + _EPS
                and a.state.deliver_by >= b.state.deliver_by - _EPS
                and a.link <= b.link + _EPS
                and (not with_dwell or a.dwell <= b.dwell + _EPS))

    def run(self) -> Tuple[Scored, Scored]:
        """(the best under F's plan key, the best V)."""
        inp = self.inp
        start = _Label(inp.start, self.by_name[inp.classes[0].name].model.home_at(inp.start),
                       0.0, 0.0, (), {})
        with_dwell = not self.dwell_in_delta
        fronts: Dict[Tuple[FrozenSet[DeviceID], Tuple[float, ...]], List[_Label]] = {
            (frozenset(), tuple(inp.start.pose)): [start]}
        layer = [start]
        best_key: Optional[Scored] = None
        best_v: Optional[Scored] = None

        def consider(label: _Label) -> None:
            nonlocal best_key, best_v
            for cls in inp.classes:
                if not label.route:
                    # The empty plan: home from takeoff, as F's search prices it.
                    label = dataclasses.replace(label, home=cls.model.home_at(inp.start))
                s = self._score(label, cls.name)
                self.plans += 1
                if best_key is None or s.key < best_key.key:
                    best_key = s
                if best_v is None or s.v > best_v.v + _EPS:
                    best_v = s

        consider(start)
        while layer:
            nxt: List[_Label] = []
            for label in layer:
                if self._hopeless(label, best_key, best_v):
                    self.pruned += 1
                    continue
                served = frozenset(label.outage)
                for group, wp, cname, protected, outage in self.options:
                    if group & served:
                        continue
                    self.expansions += 1
                    if self.expansions > self.max_expansions:
                        raise RuntimeError(f"O1 exceeded {self.max_expansions} expansions")
                    got = self._extend(label, wp, self.by_name[cname], protected, outage)
                    if got is None:
                        continue
                    key = (served | group, tuple(wp.position))
                    front = fronts.setdefault(key, [])
                    if any(self._dominates(o, got, with_dwell) for o in front):
                        continue
                    front[:] = [o for o in front if not self._dominates(got, o, with_dwell)]
                    front.append(got)
                    nxt.append(got)
                    consider(got)
            layer = nxt
        assert best_key is not None and best_v is not None
        return best_key, best_v


# --------------------------------------------------------------------------- #
# One mission's gap
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MissionGap:
    mission_round: int
    n_demand: int
    f_band: str
    f_v: float
    f_v_repriced: float
    f_share: float
    o1_key_band: str
    o1_key_v: float
    o1_key_share: float
    o1_key_stops: int
    o1_v_band: str
    o1_v: float
    o1_v_share: float
    o1_v_stops: int
    gap_v: float             # O1's best V - F's V (>= 0)
    gap_v_at_key: float      # V of O1's best under F's key - F's V
    gap_share_at_key: float  # served share of O1's best under F's key - F's
    per_stop_bands: bool     # O1's key-best flies a stop off its committed class
    expansions: int
    plans: int
    wall_s: float

    def to_json(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def mission_gap(inp: MissionInputs, *, max_devices: int = MAX_DEVICES) -> MissionGap:
    t0 = time.perf_counter()
    search = OracleSearch(inp, max_devices=max_devices)
    f = search.price_route([(wp, inp.f_band) for wp in inp.f_route], inp.f_band)
    if abs(f.v - inp.f_v) > 1e-6:
        raise RuntimeError(f"O1 prices F's plan at V={f.v!r}, F priced it at {inp.f_v!r}: "
                           "the oracle and the search disagree on pricing")
    by_key, by_v = search.run()
    if by_key.key > f.key:
        raise RuntimeError("O1's best is worse than F's under F's key: its family should "
                           "contain F's plans")
    return MissionGap(
        mission_round=inp.mission_round, n_demand=len(inp.demand), f_band=inp.f_band,
        f_v=inp.f_v, f_v_repriced=f.v, f_share=f.share,
        o1_key_band=by_key.band, o1_key_v=by_key.v, o1_key_share=by_key.share,
        o1_key_stops=len(by_key.route), o1_v_band=by_v.band, o1_v=by_v.v,
        o1_v_share=by_v.share, o1_v_stops=len(by_v.route),
        gap_v=by_v.v - f.v, gap_v_at_key=by_key.v - f.v, gap_share_at_key=by_key.share - f.share,
        per_stop_bands=any(c != by_key.band for _, c in by_key.route),
        expansions=search.expansions, plans=search.plans,
        wall_s=round(time.perf_counter() - t0, 3),
    )


def oracle_hook(sink: List[MissionGap], *, max_devices: int = MAX_DEVICES
                ) -> Callable[[Any], None]:
    """An ``on_mule`` hook: each mission F plans, the oracle reads the gap into ``sink``."""

    def install(service) -> None:
        sch = service.supervisor.scheduler
        original = sch.build_ferry_plan

        def wrapped(*args, **kw):
            now = kw.get("now")
            now = sch._now() if now is None else now
            route = original(*args, **kw)
            sink.append(mission_gap(capture(sch, now=float(now)), max_devices=max_devices))
            return route

        sch.build_ferry_plan = wrapped

    return install


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #

def _episode_gaps(task: Tuple[str, int, int, str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    cell_name, index, seed, arm, overrides = task
    from experiments.ferrysim import cells as C
    from experiments.ferrysim.episode import Policy, run_episode

    logging.disable(logging.WARNING)
    sink: List[MissionGap] = []
    run_episode(C.cell_named(cell_name), seed, Policy.of_arm(arm), hooks=(oracle_hook(sink),),
                driver_overrides=dict(overrides) or None)
    return [dict(g.to_json(), cell=cell_name, episode=index, seed=seed) for g in sink]


def summarise(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cell in sorted({r["cell"] for r in rows}):
        rs = [r for r in rows if r["cell"] == cell]
        gv = [r["gap_v"] for r in rs]
        gk = [r["gap_v_at_key"] for r in rs]
        gs = [r["gap_share_at_key"] for r in rs]
        out[cell] = {
            "missions": len(rs),
            "gap_v_mean": statistics.fmean(gv), "gap_v_median": statistics.median(gv),
            "gap_v_max": max(gv), "share_at_zero_gap_v": sum(g <= 1e-9 for g in gv) / len(gv),
            "gap_share_at_key_mean": statistics.fmean(gs),
            "gap_v_at_key_mean": statistics.fmean(gk),
            "per_stop_band_share": sum(bool(r["per_stop_bands"]) for r in rs) / len(rs),
            "wall_s_mean": statistics.fmean(r["wall_s"] for r in rs),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="o1_oracle", description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=["jit-n6-75", "jit-n6-150"],
                    help="FerrySim cells at N <= 6 (default the N = 6 controls).")
    ap.add_argument("--episodes", type=int, default=30, help="Episodes per cell (default 30).")
    ap.add_argument("--stream", default="ferrysim-val",
                    help="The episode stream (default the validation stream).")
    ap.add_argument("--arm", default="F", help="The arm whose plans are judged (default F).")
    ap.add_argument("--physics", default=None, metavar="JSON",
                    help="ferry_physics overrides on the cells (e.g. Study 5.4's "
                         "'{\"narrow_range_ratio\": 2}').")
    ap.add_argument("--driver", default=None, metavar="JSON",
                    help="Other driver overrides (e.g. '{\"far_share\": 0.5}').")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True, help="Write every mission's gap and the summary here.")
    a = ap.parse_args(argv)

    from experiments.ferrysim import cells as C
    from experiments.ferrysim.evaluate import parallel_map

    tasks = []
    for name in a.cells:
        cell = C.cell_named(name)
        if cell.n_devices > MAX_DEVICES:
            ap.error(f"{name}: O1 is bounded to N <= {MAX_DEVICES}")
        overrides: Dict[str, Any] = dict(json.loads(a.driver)) if a.driver else {}
        if a.physics:
            physics = dict(cell.driver_settings().get("ferry_physics") or {})
            physics.update(json.loads(a.physics))
            overrides["ferry_physics"] = physics
        for i, seed in enumerate(C.stream_seeds(a.stream, cell.name, int(a.episodes))):
            tasks.append((cell.name, i, int(seed), a.arm, overrides))
    rows = [r for got in parallel_map(_episode_gaps, tasks, workers=int(a.workers)) for r in got]
    report = {"arm": a.arm, "cells": a.cells, "episodes": a.episodes, "stream": a.stream,
              "physics": a.physics, "driver": a.driver, "missions": rows,
              "summary": summarise(rows)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for cell, s in report["summary"].items():
        print(f"{cell}: {s['missions']} missions, gap V mean {s['gap_v_mean']:.4f} "
              f"(median {s['gap_v_median']:.4f}, max {s['gap_v_max']:.4f}, zero in "
              f"{s['share_at_zero_gap_v']:.0%}); share gap at F's key {s['gap_share_at_key_mean']:.3f}; "
              f"{s['wall_s_mean']:.1f} s per mission")
    return 0


if __name__ == "__main__":
    sys.exit(main())
