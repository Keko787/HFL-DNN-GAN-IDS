"""FeRRy Phase 4 (unit U4): the plan search, ``hermes.scheduler.plan.plan_search``.

The search picks the mission's band class and Pass-1 route together, as the
smallest plan key over the candidates of every class the arm may fly (Phase 4
spec, other choices 3; critic A6 and C7). Pinned here:

* **exact** (a demand of 6 devices or fewer): equal to an independent brute
  force (``tests/unit/_p4_brute.py``: ``itertools`` over class × device subset
  × orders of the touched stops, priced with a plain ``FeasibilityModel.fold``
  without skipping and its own V), with the cap key equal, V within 1e-9 and
  then the tie-breaks equal, on random instances with caps, ties, every
  deadline mode, energy, F-cov, F-dwell and ``whole``; it scores exactly the
  admitted plans of that family, and each class's summary describes the
  brute force's best of that class. Hand-built cases pin ties (the stops, the
  class index), the cap key (a capped device served against V; critic C5's
  (4, 4, 4) before (5, 3)) and critic A6's own failure, a stop that fits whole
  but should shed a member;
* **stop_subsets** (more devices, at most 6 stops): equal to ``itertools`` over
  its own family, the ordered stop subsets each walked whole if it fits, else
  greedily in the F member order, by an independent walk, the summaries too;
* **local** (more stops): the start is U3's trim of the 2-OPT tour (closed at
  the dock), and under ``whole`` that tour with its priority stops first and
  its exempt stops protected (built independently); the result is a local
  optimum for the drop, insert, reverse and drop-member moves (every
  neighbour walked independently), under both admissions and also when forced
  on small problems with stops of several members; the search does exactly
  what the brute force's replay of its documented scans does (move order,
  first improvement, the route as flown, seen neighbours skipped; under
  ``lexicographic`` weighted's own scan first, then the plan key's from the
  best plan met and from the start, an earlier scan's walks reused): the
  same plan, passes, walks, bound and candidates, also when a bound cuts it
  short and on mirror-image layouts whose flight states differ only in the
  deadlines on board;
  a search whose start is already a local optimum walks exactly its
  neighbourhood and scores exactly the neighbours in which every listed stop
  admits someone; an inserted stop is walked whole or reduced greedily; one
  pass sheds the least-worth member; and the search stops within
  ``heuristic_max_passes`` and ``heuristic_max_evaluations``, counts only;
* FB+c (``fixed:c``) commits exactly class c's best, and F's key is never above
  any FB+c's (critic A3); exactly the classes the policy allows are searched;
* Pass 2 is priced once per class as the T_nom helper prices it, and V counts
  it only when the plan serves someone (the user's decision 2 (b));
* every result passes the scheduler's guard fold, is deterministic and does not
  depend on the order the stops are listed in; the per-class summaries are
  what the commit accepts (one per class, JSON-ready, no wall time) and
  record the rank applied and the best's served weight share; on the mule's
  own physics the search still equals the brute force;
* the inputs the scheduler hands over are checked, and the module keeps the
  layering (numpy-free, nothing from ``hermes.l1``, the mule, the policies,
  the scheduler or ``experiments``; the recorded pipeline never loads it).

The rank (the orchestrator's resolution R11): every mode ranks by U2's
``plan_key`` under the arm's ``coverage_rank``, and the brute force builds
the same key on its own, so each equality above holds under both ranks (the
exact and depth-first families and the mule's physics under each; the local
searches under the rank each random problem draws). Under ``lexicographic``
the empty plan wins only when no plan that serves anyone is admitted; the
local search never ends below the plan ``weighted`` commits, under its own
key, nor, unless a bound ends it, below the plan key's scan alone (the
review's problem 30125, on which that scan alone served 14 of 15 devices,
now serves all 15); F-cov is cap-only service under either setting; and in
U7's probe world (u and v 60 m either side of the dock, z 400 m out, FB+wide
at 1 MB, the cap off) the search serves a device under ``lexicographic`` and
flies empty under ``weighted``, alone and on the mule's supervisor over six
missions.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes.scheduler.plan import (
    SEARCH_EXACT,
    SEARCH_LOCAL,
    SEARCH_STOP_SUBSETS,
    AgeCapSpec,
    CapState,
    PlanClass,
    PlanOptions,
    PlanScoreParams,
    PlanSearchParams,
    PlanSetup,
    SearchResult,
)
from hermes.scheduler.plan.member_subset import fold_members, pass_energy_j, trim_members
from hermes.scheduler.plan.plan_score import (
    COVERAGE_RANKS,
    applied_rank,
    demand_weights,
    plan_key,
    served_share,
)
from hermes.scheduler.plan.plan_search import (
    ClassInput,
    ClassResult,
    PassTwo,
    plan_search,
    price_pass_2,
    search_class,
    search_mode,
)
from hermes.scheduler.routing.two_opt import order_contacts
from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range, order_pass_2_greedy
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
)
from hermes.scheduler.stages.s3d_age_cap import cap_stops, with_cap_deadlines
from hermes.types import BUCKET_PRIORITY, Bucket, DeviceID, DeviceSchedulerState
from hermes.types.scheduler import MissionPass, plan_class_summaries
from tests.unit import _p4_brute as brute

REPO = Path(__file__).resolve().parents[2]
DOCK = (0.0, 0.0, 0.0)
BANDS = ("wide", "medium", "narrow")
COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
P_MOVE, P_HOVER = 143.6, 168.5


# --------------------------------------------------------------------------- #
# Instances: synthetic class physics on S3a's own stops
# --------------------------------------------------------------------------- #

class _Dwell:
    """A member's dwell: ``a + b*d`` seconds collecting, half that delivering,
    ``d`` its planar distance to the stop."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, d, pass_kind, offset):
        t = self.a + self.b * d
        return t if pass_kind is COLLECT else 0.5 * t


class _Step:
    """A member's dwell: ``near`` seconds within ``edge_m`` of its stop,
    ``far`` beyond (collecting; half that delivering)."""

    def __init__(self, edge_m, near, far):
        self.edge_m, self.near, self.far = edge_m, near, far

    def __call__(self, d, pass_kind, offset):
        t = self.near if d < self.edge_m else self.far
        return t if pass_kind is COLLECT else 0.5 * t


class _Upload:
    def __init__(self, s):
        self.s = s

    def __call__(self):
        return self.s


class _Outage:
    """A member's outage on a class, growing with its distance to the stop."""

    def __init__(self, radius, base):
        self.radius, self.base = radius, base

    def __call__(self, d):
        return min(1.0, self.base + 0.8 * (d / self.radius) ** 2)


def _build(positions, *, classes, budget, now=0.0, deadlines=None, ages=None, weights=None,
           s_missions=None, lookahead=0, buckets=None, score=None, search=None, whole=False,
           speed=5.0, upload=0.0, capacity=None, bounds="collection", turnaround=30.0,
           t_ref=None, policy="search", dwell=None):
    """One planning problem, for the search and for the brute force alike.

    ``positions`` maps a device id to its (x, y); ``classes`` lists ``(name,
    radius, a, b, outage base)`` in link order, a member's dwell being ``a +
    b*d`` unless ``dwell`` gives one for every class. Each class's stops are
    S3a's own (``cluster_by_rf_range`` at the class's radius) and its Pass-2
    queue is those stops nearest first, as the scheduler builds them.
    ``budget`` is in seconds after ``now`` (None: no budget).
    """
    ids = [DeviceID(d) for d in positions]
    buckets = buckets or {}
    states = {
        d: DeviceSchedulerState(device_id=d, last_known_position=(x, y, 0.0),
                                bucket=buckets.get(d, Bucket.NEW))
        for d, (x, y) in zip(ids, positions.values())
    }
    deadlines = dict(deadlines) if deadlines is not None else {d: now + 1e6 for d in ids}
    ages = dict(ages) if ages is not None else {d: 1 for d in ids}
    weights = dict(weights) if weights is not None else {d: 1.0 for d in ids}
    plan_classes, entries, brute_classes = [], [], []
    for index, (name, radius, a, b, base) in enumerate(classes):
        physics = FerryPhysics(dock=DOCK, member_dwell_s=dwell or _Dwell(a, b),
                               upload_s=_Upload(upload),
                               p_move_w=P_MOVE, p_hover_w=P_HOVER, energy_capacity_j=capacity,
                               deadline_bounds=bounds, range_m=radius, device_states=states)
        model = FeasibilityModel(cruise_speed_m_s=speed, session_time_s=1.0, ferry=physics)
        pc = PlanClass(name=name, index=index, radius_m=radius, model=model,
                       outage=_Outage(radius, base))
        stops = cluster_by_rf_range(ids, states, deadlines, radius) if ids else []
        queue = order_pass_2_greedy(stops, DOCK)
        plan_classes.append(pc)
        entries.append(ClassInput(pc, tuple(stops), price_pass_2(model, queue)))
        brute_classes.append(brute.BruteClass(name, index, model, pc.outage, stops, queue))
    if t_ref is None:
        t_ref = 100.0 if budget is None else max(budget, 1.0) + turnaround
    score = score or PlanScoreParams()
    options = PlanOptions(band_class_policy=policy,
                          member_admission="whole" if whole else "subset",
                          score=score, search=search or PlanSearchParams())
    reference = plan_classes[0].name if policy == "search" else policy.split(":", 1)[1]
    setup = PlanSetup(options=options, classes=tuple(plan_classes), reference=reference,
                      t_ref_s=t_ref, turnaround_s=turnaround)
    cap = CapState(AgeCapSpec(s_missions=s_missions, lookahead=lookahead), ages)
    start = FlightState(DOCK, now)
    budget_end = None if budget is None else now + budget
    kw = dict(start=start, budget_end=budget_end, deadlines=deadlines, device_states=states,
              cap=cap, weights=weights)
    world = brute.World(
        classes=brute_classes, start=start, budget_end=budget_end, deadlines=deadlines,
        device_states=states, ages=cap.ages, capped=cap.capped, weights=weights,
        c_time=score.c_time, kappa=score.c_cov_per_device, c_link=score.c_link,
        c_energy=score.c_energy, dwell_in_delta=score.dwell_in_delta, t_ref_s=t_ref,
        turnaround_s=turnaround, p_hover_w=P_HOVER, whole=whole,
        coverage_rank=score.coverage_rank,
    )
    return SimpleNamespace(setup=setup, entries=entries, kw=kw, world=world, states=states,
                           ids=ids, cap=cap, classes=plan_classes, score=score)


def _drawn_rank(seed, n):
    """The coverage rank a random problem flies unless one is asked for: drawn
    from a stream of its own, so the problem itself is drawn as before R11."""
    return COVERAGE_RANKS[random.Random(7919 * seed + 104_729 * n + 31).random() < 0.5]


def _random(seed, n, *, few_stops=False, many_stops=False, whole=None, rank=None):
    """A random problem with ``n`` devices: 1-3 classes (now and then
    identical, to tie), and the budget, deadlines, energy capacity and deadline
    mode drawn so that each binds sometimes; random ages and cap, weights and
    score settings (F, F-cov, F-dwell, other κ and c₄), and the coverage rank
    (``rank``, else drawn).

    ``many_stops``: radii so small that nearly every device is a stop of its
    own (the local search). ``few_stops``: radii grown until no class has more
    than 6 stops (the depth-first search above 6 devices).
    """
    rank = _drawn_rank(seed, n) if rank is None else rank
    rng = random.Random(7919 * seed + 104_729 * n + 17)
    span = rng.choice((10.0, 30.0, 80.0))
    factor = rng.choice((0.02, 0.05)) if many_stops else rng.choice((0.3, 0.6, 1.2))
    speed = rng.choice((1.0, 3.0, 8.0))
    now = rng.choice((0.0, 5000.0))
    positions = {f"d{i}": (rng.uniform(-span, span), rng.uniform(-span, span)) for i in range(n)}
    same = rng.random() < 0.12
    a0, b0 = rng.uniform(0.3, 3.0), rng.uniform(0.0, 0.15)
    shapes = []
    for i in range(rng.choice((1, 2, 3, 3))):
        if same:
            shapes.append((1.0, a0, b0, 0.05))
        else:
            shapes.append((2.0 ** i, a0 * (1.0 + i * rng.uniform(0.3, 1.5)),
                           rng.uniform(0.0, 0.15), rng.choice((0.0, 0.05, 0.3))))
    # Roughly how long serving everyone takes: the scale of the budget, the
    # deadlines, the energy and T.
    full = 4.0 * span * max(1.0, n ** 0.5) / speed + n * (a0 + b0 * span) + 30.0
    ids = [DeviceID(d) for d in positions]
    s = rng.choice((None, None, 1, 2, 3, 4))
    kappa = rng.choice((1.0, 1.0, 1.0, 0.25, 0.0))
    c_link = 0.0 if kappa == 0.0 else rng.choice((None, None, None, 0.5 * kappa * max(n, 1)))
    rest = dict(
        now=now, budget=None if rng.random() < 0.08 else rng.uniform(0.05, 1.4) * full,
        deadlines={d: now + rng.uniform(-0.1, 1.5) * full for d in ids},
        ages={d: rng.randint(0, 6) for d in ids},
        weights={d: rng.choice((0.5, 1.0, 1.0, 2.0, 4.0, 9.0, 16.0)) for d in ids},
        s_missions=s, lookahead=rng.choice((0, 0, 1)) if s is not None and s > 1 else 0,
        buckets={d: rng.choice(BUCKET_PRIORITY) for d in ids},
        score=PlanScoreParams(c_cov_per_device=kappa, c_link=c_link,
                              c_energy=rng.choice((0.0, 0.1)),
                              dwell_in_delta=rng.random() < 0.8, coverage_rank=rank),
        whole=(rng.random() < 0.15) if whole is None else whole,
        speed=speed, upload=rng.choice((0.0, 2.0)),
        capacity=None if rng.random() < 0.7 else rng.uniform(0.2, 1.5) * full * P_HOVER,
        bounds=rng.choice(DEADLINE_BOUNDS), turnaround=rng.choice((0.0, 30.0)),
        t_ref=rng.uniform(0.8, 3.0) * full,
    )

    def build(f):
        classes = [(BANDS[i], f * span * m, a, b, base) for i, (m, a, b, base) in enumerate(shapes)]
        return _build(positions, classes=classes, **rest)

    inst = build(factor)
    for grown in (0.8, 1.2, 2.0, 4.0):
        if not few_stops or max(len(e.stops) for e in inst.entries) <= 6:
            break
        inst = build(grown)
    return inst


def _search(inst, **kw):
    return plan_search(inst.setup, inst.entries, **{**inst.kw, **kw})


def _class(inst, index=0):
    return search_class(inst.setup, inst.entries[index], **inst.kw)


def _stops(route):
    return tuple((tuple(wp.position), tuple(wp.devices)) for wp in route)


def _key(inst, cand):
    """``cand``'s plan key under the problem's score settings (U2's ``plan_key``)."""
    return plan_key(inst.setup.options.score, cand)


def _agree(got, want):
    """The spec's comparison: cap key equal, the served weight share (the
    lexicographic rank's second place) and V within 1e-9, then the tie-breaks
    (class index, each stop's position and devices) equal."""
    assert got.cap_key == want.cap_key
    assert served_share(got.terms) == pytest.approx(want.share, abs=1e-9)
    assert got.terms.v == pytest.approx(want.v, abs=1e-9)
    assert (got.cls.index, _stops(got.fold.route)) == (want.index, want.stops)
    assert got.served == want.served


def _guard(inst, cand):
    """The scheduler's guard fold (spec, other choices 1): the route passes the
    predicate without skipping, its exempt stops recomputed on it (critic B1)."""
    route = cand.fold.route
    fold = cand.cls.model.fold(route, inst.kw["start"], rule=RULE_DEADLINE_BUDGET,
                               budget_end=inst.kw["budget_end"], skip=False,
                               protected=cap_stops(route, inst.cap.capped).exempt)
    assert fold.ok, fold.rejected
    if route:
        assert fold.home == cand.fold.home and fold.state == cand.fold.state


def _source(inst, index, wp):
    """The S3a stop of class ``index`` a flown stop was cut from."""
    return next(s for s in inst.entries[index].stops if wp.devices[0] in s.devices)


def _summaries_agree(inst, result, plans):
    """Each class's summary describes that class's own search (the trace's
    ``plan.per_class``): its band and index, the number of S3a stops, and its
    best plan's V, served count, cap key and served weight share, here those
    of the brute force's best plan of the class, found among ``plans``; and
    the rank the search applied."""
    assert len(result.per_class) == len(inst.setup.searched) == len(inst.entries)
    for entry, c, s in zip(inst.entries, inst.setup.searched, result.per_class):
        assert entry.cls.name == c.name
        own = brute.best(p for p in plans if p.band == c.name)
        assert (s["band"], s["index"], s["stops"]) == (c.name, c.index, len(entry.stops))
        assert (s["served"], s["cap_key"]) == (len(own.served), list(own.cap_key))
        assert s["v"] == pytest.approx(own.v, abs=1e-9)
        assert s["served_share"] == pytest.approx(own.share, abs=1e-9)
        assert s["coverage_rank"] == applied_rank(inst.score)


# --------------------------------------------------------------------------- #
# exact: equal to the independent brute force (critic A6)
# --------------------------------------------------------------------------- #

EXACT_SEEDS = range(96)


@pytest.mark.parametrize("rank", COVERAGE_RANKS)
@pytest.mark.parametrize("seed", EXACT_SEEDS)
def test_the_exact_search_equals_an_independent_brute_force(seed, rank):
    inst = _random(seed, 1 + seed % 6, rank=rank)
    result = _search(inst)
    plans = list(brute.exact_plans(inst.world))
    _agree(result.best, brute.best(iter(plans)))
    assert result.mode == SEARCH_EXACT
    _guard(inst, result.best)
    # The exact family chooses each stop's members; it refuses none of them.
    assert result.best.fold.dropped == ()
    # It scores exactly the admitted plans of the family, class by class.
    per_class = Counter(p.band for p in plans)
    assert [s["candidates"] for s in result.per_class] == [
        per_class[c.name] for c in inst.setup.searched]
    assert result.n_candidates == len(plans)
    _summaries_agree(inst, result, plans)


def test_the_random_instances_reach_what_they_are_meant_to():
    """The brute-force instances differ where it matters: several classes and
    identical ones, a cap and a cap key that decides, empty plans, stops
    reduced to part of their members, ``whole``, no budget; and the summaries
    of the classes are not all alike (a class after the first, one that
    serves part of the demand, one that leaves a capped device out, one whose
    best flies fewer stops than S3a made). Each rank is drawn often, and on
    some problems the two ranks choose different plans: the same cap key, the
    lexicographic one serving more weight at no higher V."""
    seen = Counter()
    for seed in EXACT_SEEDS:
        inst = _random(seed, 1 + seed % 6)
        result = _search(inst)
        best = result.best
        world = inst.world
        rank = inst.score.coverage_rank
        seen[rank] += 1
        other_rank = "weighted" if rank == "lexicographic" else "lexicographic"
        other = _search(_random(seed, 1 + seed % 6, rank=other_rank)).best
        if _stops(other.fold.route) != _stops(best.fold.route) or other.band != best.band:
            seen["ranks differ"] += 1
            lex, wtd = (best, other) if rank == "lexicographic" else (other, best)
            # Both keep the smallest cap key; beyond it each wins on its own terms.
            assert lex.cap_key == wtd.cap_key
            assert round(served_share(lex.terms), 9) > round(served_share(wtd.terms), 9)
            assert round(lex.terms.v, 9) <= round(wtd.terms.v, 9)
        for index, s in enumerate(result.per_class):
            own = _class(inst, index).best
            seen["summary_index"] += s["index"] > 0
            seen["summary_part"] += s["served"] < len(world.weights)
            seen["summary_cap_key"] += bool(s["cap_key"])
            seen["summary_fewer_stops"] += len(own.fold.route) < s["stops"]
        seen["capped"] += bool(world.capped)
        seen["cap_key"] += bool(best.cap_key)
        seen["empty"] += not best.fold.route
        seen["whole"] += world.whole
        seen["classes"] += len(world.classes) > 1
        seen["identical"] += len(world.classes) > 1 and len(
            {(c.model.ferry.member_dwell_s.a, c.model.ferry.range_m) for c in world.classes}) == 1
        seen["reduced"] += any(len(wp.devices) < len(_source(inst, best.cls.index, wp).devices)
                               for wp in best.fold.route)
        seen["no_budget"] += world.budget_end is None
        seen["not_first"] += best.cls.index > 0
    for what in ("capped", "cap_key", "empty", "whole", "classes", "identical", "reduced",
                 "no_budget", "not_first", "summary_index", "summary_part", "summary_cap_key",
                 "summary_fewer_stops", "ranks differ"):
        assert seen[what] >= 3, (what, seen)
    assert seen["lexicographic"] >= 30 and seen["weighted"] >= 30, seen


def test_a_tie_in_v_falls_to_the_stops():
    """Two devices that each fit alone and cost exactly the same, as mirror
    images: V ties bit for bit, and the key's stops (position, then devices)
    decide, the same way in the search and in the brute force."""
    inst = _build({"a": (20.0, 5.0), "b": (20.0, -5.0)}, classes=[("wide", 1.0, 2.0, 0.0, 0.0)],
                  budget=12.0, t_ref=200.0)
    result = _search(inst)
    plans = list(brute.exact_plans(inst.world))
    one = [p for p in plans if len(p.served) == 1]
    assert len(one) == 2 and one[0].v == one[1].v and len(plans) == 3   # both do not fit
    _agree(result.best, brute.best(iter(plans)))
    assert result.best.served == {DeviceID("b")}          # (20, -5) sorts before (20, 5)


def test_a_tie_between_identical_classes_falls_to_the_class_index():
    inst = _build({"a": (10.0, 0.0), "b": (-10.0, 0.0)},
                  classes=[("wide", 30.0, 1.0, 0.0, 0.1), ("medium", 30.0, 1.0, 0.0, 0.1)],
                  budget=60.0)
    result = _search(inst)
    _agree(result.best, brute.best(brute.exact_plans(inst.world)))
    assert result.best.band == "wide"
    assert result.per_class[0]["v"] == result.per_class[1]["v"]


def test_the_cap_key_serves_a_capped_device_that_v_would_leave_out():
    """V prefers the two near devices; the capped far one must come first."""
    positions = {"far": (60.0, 0.0), "n1": (-5.0, 0.0), "n2": (-6.0, 0.0)}
    kw = dict(classes=[("wide", 2.0, 1.0, 0.0, 0.0)], budget=26.0,
              ages={DeviceID("far"): 3, DeviceID("n1"): 1, DeviceID("n2"): 1})
    free = _search(_build(positions, **kw)).best
    assert free.served == {DeviceID("n1"), DeviceID("n2")}
    inst = _build(positions, s_missions=3, **kw)
    result = _search(inst)
    _agree(result.best, brute.best(brute.exact_plans(inst.world)))
    assert result.best.served == {DeviceID("far")} and result.best.cap_key == ()
    assert result.best.terms.v < free.terms.v


def test_the_cap_key_leaves_the_oldest_out_last():
    """Critic C5: leaving (4, 4, 4) out beats leaving (5, 3) out, although it
    leaves more capped devices out."""
    positions = {"old": (30.0, 0.0), "young": (31.0, 0.0),
                 "m1": (-30.0, 0.0), "m2": (-31.0, 0.0), "m3": (-32.0, 0.0)}
    ages = {DeviceID("old"): 5, DeviceID("young"): 3, DeviceID("m1"): 4, DeviceID("m2"): 4,
            DeviceID("m3"): 4}
    # One side of the dock per mission: 2 x 32 m at 5 m/s, a second per member.
    inst = _build(positions, classes=[("wide", 5.0, 1.0, 0.0, 0.0)], budget=16.0, ages=ages,
                  s_missions=3)
    result = _search(inst)
    _agree(result.best, brute.best(brute.exact_plans(inst.world)))
    assert result.best.served == {DeviceID("old"), DeviceID("young")}
    assert result.best.cap_key == (4, 4, 4)


def test_the_exact_search_sheds_a_member_from_a_stop_that_fits_whole():
    """Critic A6's failure of the design's search, under the ``weighted`` rank
    (V alone): every member fits, but one worth little costs a lot of dwell,
    so the best plan leaves it out. The depth-first family (whole whenever it
    fits) cannot; the exact search does. (Under ``lexicographic`` serving all
    three wins, since it serves more weight: the next test.)"""
    positions = {"a": (20.0, 0.0), "b": (20.0, 0.1), "c": (24.0, 0.0)}
    weights = {DeviceID("a"): 9.0, DeviceID("b"): 9.0, DeviceID("c"): 0.5}
    # One stop at the members' centroid, 10 s of dwell per metre from it: c,
    # twice as far as a and b, costs twice their dwell and weighs 1/18 of it.
    kw = dict(classes=[("wide", 5.0, 1.0, 10.0, 0.0)], budget=None, weights=weights,
              t_ref=200.0)
    inst = _build(positions, score=PlanScoreParams(c_energy=0.0, coverage_rank="weighted"), **kw)
    (stop,) = inst.entries[0].stops
    assert len(stop.devices) == 3
    best = _search(inst).best
    _agree(best, brute.best(brute.exact_plans(inst.world)))
    assert best.served == {DeviceID("a"), DeviceID("b")}
    dfs = brute.best(brute.stop_subset_plans(inst.world, inst.world.classes[0]))
    assert dfs.served == {DeviceID("a"), DeviceID("b"), DeviceID("c")} and dfs.v < best.terms.v
    everyone = _search(_build(positions, score=PlanScoreParams(c_energy=0.0), **kw)).best
    assert everyone.served == dfs.served and everyone.terms.v == pytest.approx(dfs.v, abs=1e-9)


def test_under_lexicographic_the_exact_search_sheds_a_member_to_serve_more_weight():
    """Critic A6 under the ``lexicographic`` rank: shedding a member never
    serves more weight by itself, but it can make room for a heavier stop.
    S1 = {x, y} fits whole from takeoff; S1 whole and then S2 = {z} (weighing
    three times x) runs over the 38 s budget; flying S2 first makes S1 late
    for its 25 s deadline. So the best plan is S1 cut to x, then S2 (share
    0.8): the depth-first family, which takes S1 whole whenever it fits,
    reaches only {z} (share 0.6); the exact search finds it."""
    x, y, z = (DeviceID(k) for k in "xyz")
    inst = _build({"x": (20.0, 0.0), "y": (20.0, 2.0), "z": (-40.0, 0.0)},
                  classes=[("wide", 5.0, 5.0, 0.0, 0.0)], budget=38.0, t_ref=200.0,
                  weights={x: 1.0, y: 1.0, z: 3.0}, deadlines={x: 25.0, y: 25.0, z: 1e6})
    world, cls = inst.world, inst.world.classes[0]
    s1, s2 = sorted(inst.entries[0].stops, key=lambda wp: len(wp.devices), reverse=True)
    assert (s1.devices, s2.devices) == ((x, y), (z,))
    whole = brute.greedy(world, cls, world.start, brute.stop(s1, s1.devices, world))
    assert whole is not None and whole[0].devices == (x, y)          # S1 fits whole
    # The depth-first family flies no two-stop route: S2 no longer fits after
    # S1 whole, and S1 admits nobody after S2.
    assert brute.walk(world, cls, [(s1, s1.devices), (s2, s2.devices)]) is None
    assert brute.walk(world, cls, [(s2, s2.devices), (s1, s1.devices)]) is None
    best = _search(inst).best
    _agree(best, brute.best(brute.exact_plans(world)))
    assert [wp.devices for wp in best.fold.route] == [(x,), (z,)]
    assert served_share(best.terms) == 0.8
    dfs = brute.best(brute.stop_subset_plans(world, cls))
    assert dfs.served == {z} and dfs.share == 0.6


def test_the_empty_plan_pays_the_turnaround_and_no_pass_2():
    """Nothing fits: the plan is empty, flies neither pass and still pays the
    dock turnaround (decision 2 (b))."""
    inst = _build({"a": (50.0, 0.0)}, classes=[("wide", 5.0, 1.0, 0.0, 0.0)], budget=5.0,
                  t_ref=60.0)
    result = _search(inst)
    best = result.best
    assert best.fold.route == () and best.fold.feasible and best.served == frozenset()
    assert best.fold.home == 0.0 and best.fold.energy_j == 0.0
    assert best.terms.delta_s == 30.0
    c1, c2, _, _ = inst.score.constants(1)
    assert best.terms.v == -(c1 * (30.0 / 60.0) ** 2 + c2 * 1.0)
    assert result.n_candidates == 1


def test_a_plan_that_serves_pays_both_passes_on_its_class():
    inst = _build({"a": (10.0, 0.0), "b": (-10.0, 0.0)},
                  classes=[("wide", 3.0, 2.0, 0.0, 0.0), ("medium", 30.0, 4.0, 0.0, 0.0)],
                  budget=None, upload=1.5)
    result = _search(inst)
    start = inst.kw["start"]
    assert inst.entries[0].pass_2 != inst.entries[1].pass_2
    for index, (entry, summary) in enumerate(zip(inst.entries, result.per_class)):
        best = _class(inst, index).best
        assert best.served == {DeviceID("a"), DeviceID("b")}       # no budget: everyone
        assert best.terms.delta_s == pytest.approx(
            best.fold.home - start.clock + 30.0 + entry.pass_2.time_s, abs=1e-12)
        assert best.terms.energy_j == pytest.approx(best.fold.energy_j + entry.pass_2.energy_j)
        assert summary["v"] == best.terms.v


def test_an_empty_demand_is_searched_exactly_and_flies_nothing():
    inst = _build({}, classes=[("wide", 5.0, 1.0, 0.0, 0.0), ("medium", 9.0, 1.0, 0.0, 0.0)],
                  budget=30.0)
    result = _search(inst)
    assert result.mode == SEARCH_EXACT and result.n_candidates == 2
    assert result.best.fold.route == () and result.best.band == "wide"
    assert result.best.terms.coverage == result.best.terms.link == 0.0


def test_member_admission_whole_flies_whole_stops_only():
    """``whole`` keeps the cliff for comparison (design D-D): a stop that does
    not fit whole is not flown at all, where ``subset`` flies part of it."""
    positions = {f"d{i}": (40.0 + i, 0.0) for i in range(4)}
    kw = dict(classes=[("narrow", 20.0, 5.0, 0.0, 0.0)], budget=35.0)
    subset = _search(_build(positions, **kw)).best
    inst = _build(positions, whole=True, **kw)
    whole = _search(inst).best
    assert 0 < len(subset.served) < 4
    assert whole.fold.route == ()
    _agree(whole, brute.best(brute.exact_plans(inst.world)))


# --------------------------------------------------------------------------- #
# stop_subsets: equal to itertools over its own family
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rank", COVERAGE_RANKS)
@pytest.mark.parametrize("seed", range(40))
def test_the_depth_first_search_above_six_devices_equals_itertools_over_its_family(seed, rank):
    inst = _random(1000 + seed, 7 + seed % 6, few_stops=True, rank=rank)
    result = _search(inst)
    assert result.mode == SEARCH_STOP_SUBSETS
    assert all(len(e.stops) <= 6 for e in inst.entries)
    plans = [p for cls in inst.world.classes for p in brute.stop_subset_plans(inst.world, cls)]
    _agree(result.best, brute.best(iter(plans)))
    _guard(inst, result.best)
    per_class = Counter(p.band for p in plans)
    assert [s["candidates"] for s in result.per_class] == [
        per_class[c.name] for c in inst.setup.searched]
    _summaries_agree(inst, result, plans)
    # The candidate is U3's fold of its stops, the complements it drops included.
    best, index = result.best, result.best.cls.index
    stops = with_cap_deadlines([_source(inst, index, wp) for wp in best.fold.route],
                               deadlines=inst.kw["deadlines"], capped=inst.cap.capped)
    fold = fold_members(stops, inst.kw["start"], model=best.cls.model, rule=RULE_DEADLINE_BUDGET,
                        budget_end=inst.kw["budget_end"], deadlines=inst.kw["deadlines"],
                        device_states=inst.states, require_all=True, capped=inst.cap.capped,
                        weights=None if inst.world.whole else inst.kw["weights"])
    if not inst.world.whole:
        assert fold.route == best.fold.route and fold.dropped == best.fold.dropped
        assert fold.home == best.fold.home and fold.energy_j == best.fold.energy_j
    else:
        assert best.fold.dropped == ()


def test_the_depth_first_instances_reduce_stops_and_prune():
    reduced = pruned = several = 0
    for seed in range(40):
        inst = _random(1000 + seed, 7 + seed % 6, few_stops=True)
        result = _search(inst)
        best = result.best
        reduced += any(len(wp.devices) < len(_source(inst, best.cls.index, wp).devices)
                       for wp in best.fold.route)
        several += max(len(e.stops) for e in inst.entries) > 2
        # A stop that admits nobody prunes: fewer candidates than ordered subsets.
        pruned += any(s["candidates"] < sum(math.perm(s["stops"], r) for r in range(s["stops"] + 1))
                      for s in result.per_class)
    assert reduced >= 5 and pruned >= 5 and several >= 5, (reduced, pruned, several)


def test_the_modes_follow_the_demand_and_the_stops():
    bounds = PlanSearchParams()
    assert search_mode(0, 0, bounds) == search_mode(6, 6, bounds) == SEARCH_EXACT
    assert search_mode(6, 50, bounds) == SEARCH_EXACT                  # the demand decides first
    assert search_mode(7, 6, bounds) == SEARCH_STOP_SUBSETS
    assert search_mode(7, 7, bounds) == SEARCH_LOCAL
    tight = PlanSearchParams(exact_max_devices=2, exhaustive_max_stops=1)
    assert (search_mode(2, 9, tight), search_mode(3, 1, tight), search_mode(3, 2, tight)) == (
        SEARCH_EXACT, SEARCH_STOP_SUBSETS, SEARCH_LOCAL)
    # Per class: a class with 8 stops searches locally, one with 1 stop its subsets.
    positions = {f"d{i}": (12.0 * i + 12.0, 0.0) for i in range(8)}
    inst = _build(positions, budget=None,
                  classes=[("wide", 1.0, 1.0, 0.0, 0.0), ("medium", 200.0, 1.0, 0.0, 0.0)])
    result = _search(inst)
    assert [(s["mode"], s["stops"]) for s in result.per_class] == [
        (SEARCH_LOCAL, 8), (SEARCH_STOP_SUBSETS, 1)]
    assert result.mode == result.per_class[result.best.cls.index]["mode"]


# --------------------------------------------------------------------------- #
# local: the 2-OPT tour, the trim, a bounded local search
# --------------------------------------------------------------------------- #

def _entries_of(inst, index, route):
    return [(_source(inst, index, wp), tuple(wp.devices)) for wp in route]


def _neighbours(inst, index, route):
    """Every drop, insert, reverse and drop-member neighbour of ``route`` (the
    local search's route as flown), as ``brute.walk`` entries; under ``whole``
    only those that keep every stop whole (no drop-member moves)."""
    stops = inst.entries[index].stops
    entries = _entries_of(inst, index, route)
    routed = {id(s) for s, _ in entries}
    k = len(entries)
    for p in range(k):
        yield entries[:p] + entries[p + 1:]
    for wp in stops:
        if id(wp) not in routed:
            for p in range(k + 1):
                yield entries[:p] + [(wp, wp.devices)] + entries[p:]
    for a in range(k - 1):
        for b in range(a + 1, k):
            yield entries[:a] + entries[a:b + 1][::-1] + entries[b + 1:]
    if inst.world.whole:
        return
    for p, (wp, members) in enumerate(entries):
        for d in members if len(members) > 1 else ():
            yield entries[:p] + [(wp, tuple(m for m in members if m != d))] + entries[p + 1:]


def _at_local_optimum(inst, index, best, world=None):
    """Every neighbour of ``best``'s route, walked by the brute force, has a
    larger plan key, and the brute force flies that route as the search did
    (priced in ``world``, the problem's own unless given)."""
    world = inst.world if world is None else world
    cls = world.classes[index]
    mine = brute.walk(world, cls, _entries_of(inst, index, best.fold.route))
    assert mine is not None and mine.stops == _stops(best.fold.route)
    assert mine.v == pytest.approx(best.terms.v, abs=1e-9)
    for entries in _neighbours(inst, index, best.fold.route):
        other = brute.walk(world, cls, entries)
        assert other is None or other.key > mine.key, (entries, other.key, mine.key)


LOCAL_SEEDS = range(24)


@pytest.mark.parametrize("whole", [False, True], ids=["subset", "whole"])
@pytest.mark.parametrize("seed", LOCAL_SEEDS)
def test_the_local_search_ends_at_a_local_optimum_of_its_moves(seed, whole):
    """Every drop, insert, reverse and (under ``subset``) drop-member neighbour
    of the result, walked by the brute force, has a larger plan key; under
    ``whole`` with the cap's priority and exempt stops too. The class's
    summary describes that result."""
    inst = _random(2000 + seed, 10 + seed % 7, many_stops=True, whole=whole)
    for index, cls in enumerate(inst.world.classes):
        got = _class(inst, index)
        if got.mode != SEARCH_LOCAL:
            continue
        s = got.summary
        assert s["bounded"] is False
        assert (s["index"], s["stops"]) == (cls.index, len(inst.entries[index].stops))
        assert (s["served"], s["cap_key"]) == (len(got.best.served), list(got.best.cap_key))
        # One candidate per walk that passed, and the empty plan once.
        assert 1 < got.n_candidates <= s["evaluations"] + 1
        best = got.best
        _guard(inst, best)
        if not best.fold.route:
            continue                  # the empty plan beat the local search's route
        assert not whole or all(wp.devices == _source(inst, index, wp).devices
                                for wp in best.fold.route)
        _at_local_optimum(inst, index, best)


def test_the_local_optimum_test_is_not_vacuous():
    """Most instances above end on a non-empty route of a class searched
    locally, after more than one pass, under each admission; under ``whole``
    many of them with a cap."""
    seen = Counter()
    for whole in (False, True):
        for seed in LOCAL_SEEDS:
            inst = _random(2000 + seed, 10 + seed % 7, many_stops=True, whole=whole)
            for index in range(len(inst.entries)):
                got = _class(inst, index)
                if got.mode == SEARCH_LOCAL and got.best.fold.route:
                    seen[whole, "routes"] += 1
                    seen[whole, "passes"] += got.summary["passes"] > 1
                    seen[whole, "capped"] += bool(inst.world.capped)
    for whole in (False, True):
        assert seen[whole, "routes"] >= 20 and seen[whole, "passes"] >= 10, seen
    assert seen[True, "capped"] >= 10, seen


RING = {f"d{i}": (30.0 * math.cos(i), 30.0 * math.sin(i)) for i in range(9)}
# The same ring 60 m east of the dock: there the closed tour and an open path
# from the dock differ, so the start below shows which one it is built on.
OFF_RING = {f"d{i}": (60.0 + 30.0 * math.cos(i), 30.0 * math.sin(i)) for i in range(9)}


def _start(inst, index=0):
    """The local search's start, built independently: U3's member trim of the
    2-OPT tour of the class's stops (B2's deadlines on them), from the dock
    back to the dock."""
    entry = inst.entries[index]
    stops = with_cap_deadlines(entry.stops, deadlines=inst.kw["deadlines"],
                               capped=inst.cap.capped)
    tour = order_contacts(stops, DOCK, end=DOCK)
    return trim_members(tour, inst.kw["start"], model=entry.cls.model,
                        budget_end=inst.kw["budget_end"], deadlines=inst.kw["deadlines"],
                        device_states=inst.states, capped=inst.cap.capped,
                        weights=inst.kw["weights"]).route


def test_the_local_search_starts_from_the_trim_of_the_2_opt_tour():
    """With one walk allowed, the class's plan is its start: U3's member trim
    of the 2-OPT tour of the stops, from the dock back to the dock."""
    ages = {DeviceID(f"d{i}"): 1 + i % 3 for i in range(9)}
    for layout, budgets in ((RING, (None, 60.0, 25.0)), (OFF_RING, (None, 70.0, 30.0))):
        for budget in budgets:
            inst = _build(layout, classes=[("wide", 1.0, 2.0, 0.0, 0.0)], budget=budget,
                          search=PlanSearchParams(heuristic_max_evaluations=1), ages=ages,
                          s_missions=3)
            got = _class(inst)
            assert got.mode == SEARCH_LOCAL and got.summary["evaluations"] == 1
            trimmed = _start(inst)
            assert trimmed and got.best.fold.route == tuple(trimmed), budget
            assert got.n_candidates == 2                        # the empty plan and the start
    stops = _build(OFF_RING, classes=[("wide", 1.0, 2.0, 0.0, 0.0)], budget=None).entries[0].stops
    assert order_contacts(stops, DOCK) != order_contacts(stops, DOCK, end=DOCK)


FORCED = PlanSearchParams(exact_max_devices=0, exhaustive_max_stops=0)


@pytest.mark.parametrize("seed", range(9000, 9120))
def test_the_local_search_forced_on_small_problems_ends_at_a_local_optimum(seed):
    """The local search forced on small problems (bounds 0 and 0), where stops
    have several members: the result is a local optimum of every drop, insert,
    reverse and drop-member move, ties included (a reversed route of the same
    V has a larger key). When the start is already one for the first scan's
    key, that scan walks exactly its neighbourhood once, and, when no later
    scan walks anything more, the search scores exactly the neighbours in
    which every listed stop admits someone: a move that leaves a stop unserved
    is no candidate (design D-H: each move is checked by a fold that does not
    skip)."""
    inst = _random(seed, 3 + seed % 5, whole=False)
    setup = dataclasses.replace(inst.setup,
                                options=dataclasses.replace(inst.setup.options, search=FORCED))
    for index, cls in enumerate(inst.world.classes):
        got = search_class(setup, inst.entries[index], **inst.kw)
        assert got.mode == SEARCH_LOCAL and got.summary["bounded"] is False
        _guard(inst, got.best)
        route = got.best.fold.route
        if route:
            mine = brute.walk(inst.world, cls, _entries_of(inst, index, route))
            assert mine.stops == _stops(route)
            for entries in _neighbours(inst, index, route):
                other = brute.walk(inst.world, cls, entries)
                assert other is None or other.key > mine.key, (entries, other.key, mine.key)
        s = got.summary
        if s.get("weighted_passes", s["passes"]) == 1:
            start = _start(inst, index)
            around = list(_neighbours(inst, index, start))
            walks = s.get("weighted_evaluations", s["evaluations"])
            assert walks == 1 + len(around)
            if s["evaluations"] == walks:
                fits = sum(1 for n in around if n and brute.walk(inst.world, cls, n) is not None)
                assert got.n_candidates == 1 + bool(start) + fits


def test_an_inserted_stop_is_walked_whole_or_reduced_greedily():
    """The insert move puts an unrouted stop in with all its members and walks
    it like the depth-first search: whole if it fits, else its members in the
    F order, skip not stop. Here the start leaves the three-member stop out:
    after the other stop its two cheap members are overdue. Inserted first, it
    serves them; its first member (the anchor S3a put first) costs 50 s and
    fits nowhere, so a move limited to that member would change nothing."""
    ids = {k: DeviceID(k) for k in ("x", "e", "c1", "c2")}
    positions = {"x": (-10.0, 0.0), "e": (13.0, 0.0), "c1": (10.0, 0.0), "c2": (10.1, 0.0)}
    kw = dict(classes=[("wide", 3.1, 0.0, 0.0, 0.0)], budget=30.0, t_ref=100.0,
              deadlines={ids["x"]: 1e6, ids["e"]: 1e6, ids["c1"]: 4.0, ids["c2"]: 4.0},
              buckets={ids["e"]: Bucket.NEW, ids["c1"]: Bucket.SCHEDULED_THIS_ROUND,
                       ids["c2"]: Bucket.SCHEDULED_THIS_ROUND,
                       ids["x"]: Bucket.SCHEDULED_THIS_ROUND},
              dwell=_Step(1.5, 0.3, 50.0))
    inst = _build(positions, search=FORCED, **kw)
    (trio,) = [wp for wp in inst.entries[0].stops if len(wp.devices) == 3]
    assert trio.devices[0] == ids["e"]
    start = _class(_build(positions, search=dataclasses.replace(FORCED,
                                                                heuristic_max_evaluations=1), **kw))
    assert start.best.served == {ids["x"]}
    got = _class(inst)
    # Under the default rank three scans run: weighted's own inserts the trio in
    # its first pass and finds nothing better in its second; the plan key's
    # from the best plan met finds nothing in one pass; the plan key's from
    # the start makes the same insert, then finds nothing: 2 + 1 + 2 passes.
    assert got.mode == SEARCH_LOCAL and got.summary["weighted_passes"] == 2
    assert got.summary["passes"] == 5
    assert [wp.devices for wp in got.best.fold.route] == [(ids["c1"], ids["c2"]), (ids["x"],)]
    # e, tried beside c1 and c2, is held to their deadline (B2) and misses it.
    assert [(wp.devices, why) for wp, why in got.best.fold.dropped] == [((ids["e"],), "overdue")]
    exact = _build(positions, **kw)
    assert _key(inst, got.best) == _key(exact, _class(exact).best)       # the exact optimum


def test_the_forced_local_problems_are_not_trivial():
    """The problems above end on routes with stops of several members, some
    searches take more than one pass, and many single-pass ones have a
    neighbour whose walk leaves a stop unserved."""
    seen = Counter()
    for seed in range(9000, 9120):
        inst = _random(seed, 3 + seed % 5, whole=False)
        setup = dataclasses.replace(inst.setup,
                                    options=dataclasses.replace(inst.setup.options, search=FORCED))
        for index, cls in enumerate(inst.world.classes):
            got = search_class(setup, inst.entries[index], **inst.kw)
            s = got.summary
            seen["members"] += any(len(wp.devices) > 1 for wp in got.best.fold.route)
            seen["passes"] += s.get("weighted_passes", s["passes"]) > 1
            if s.get("weighted_passes", s["passes"]) == 1:
                seen["one pass", s["coverage_rank"]] += 1
                seen["no later walk"] += s["evaluations"] == s.get("weighted_evaluations",
                                                                   s["evaluations"])
                around = _neighbours(inst, index, _start(inst, index))
                seen["unserved"] += any(n and brute.walk(inst.world, cls, n) is None
                                        for n in around)
    assert seen["members"] >= 50 and seen["passes"] >= 20 and seen["unserved"] >= 20, seen
    assert seen["one pass", "lexicographic"] >= 10 and seen["one pass", "weighted"] >= 10, seen
    assert seen["no later walk"] >= 20, seen


def _bounded(inst, index, **bounds):
    """Class ``index`` of ``inst`` searched again under other search bounds."""
    options = dataclasses.replace(inst.setup.options, search=PlanSearchParams(**bounds))
    setup = dataclasses.replace(inst.setup, options=options)
    return search_class(setup, inst.entries[index], **inst.kw)


def _improving():
    """(instance, class index) pairs of the local-optimum instances whose local
    search improved on its start more than once."""
    for seed in LOCAL_SEEDS:
        inst = _random(2000 + seed, 10 + seed % 7, many_stops=True, whole=False)
        for index in range(len(inst.entries)):
            got = _class(inst, index)
            if got.mode == SEARCH_LOCAL and got.summary["passes"] >= 3:
                yield inst, index, got


def test_the_local_search_improves_on_its_start():
    better = Counter()
    for inst, index, full in _improving():
        start = _bounded(inst, index, heuristic_max_evaluations=1)
        assert start.summary["evaluations"] == 1 and start.n_candidates == 2
        assert _key(inst, full.best) <= _key(inst, start.best)
        better[inst.score.coverage_rank] += _key(inst, full.best) < _key(inst, start.best)
    assert better["lexicographic"] >= 5 and better["weighted"] >= 2, better


def test_the_local_search_is_bounded_by_walks_and_passes_never_by_time():
    inst, index, free = next(_improving())
    assert free.summary["bounded"] is False
    n, walks = free.summary["passes"], free.summary["evaluations"]
    for cap in (1, 2, walks // 2, walks - 1):
        got = _bounded(inst, index, heuristic_max_evaluations=cap)
        assert got.summary["evaluations"] == cap and got.summary["bounded"] is True
        assert _key(inst, got.best) >= _key(inst, free.best)
    # Exactly enough walks: the same search, ended by its local optimum.
    enough = _bounded(inst, index, heuristic_max_evaluations=walks)
    assert dict(enough.summary) == dict(free.summary)
    assert _key(inst, enough.best) == _key(inst, free.best)
    one = _bounded(inst, index, heuristic_max_passes=1)
    assert one.summary["passes"] == 1 and one.summary["bounded"] is True
    # The last pass improves nothing: with exactly that many passes the search
    # ends at the same local optimum, not at a bound; with one fewer, at a bound.
    last = _bounded(inst, index, heuristic_max_passes=n)
    assert dict(last.summary) == dict(free.summary)
    assert _key(inst, last.best) == _key(inst, free.best)
    fewer = _bounded(inst, index, heuristic_max_passes=n - 1)
    assert fewer.summary["bounded"] is True and fewer.summary["passes"] == n - 1
    assert fewer.summary["evaluations"] < walks


def test_the_local_search_in_whole_mode_moves_whole_stops():
    """Eight pairs of devices, a stop per pair: under ``whole`` every stop flown
    is an S3a stop whole, and the result is still a local optimum of the moves
    that keep stops whole (drop, insert, reverse)."""
    pairs = {}
    for k in range(8):
        x, y = 30.0 * math.cos(k * math.pi / 4), 30.0 * math.sin(k * math.pi / 4)
        pairs[f"p{k}a"], pairs[f"p{k}b"] = (x, y), (x + 1.0, y)
    inst = _build(pairs, classes=[("wide", 2.0, 2.0, 0.0, 0.0)], budget=60.0, whole=True,
                  t_ref=200.0)
    assert len(inst.entries[0].stops) == 8
    got = _class(inst)
    assert got.mode == SEARCH_LOCAL and got.best.fold.route and got.summary["bounded"] is False
    assert all(wp in inst.entries[0].stops for wp in got.best.fold.route)
    assert got.best.fold.dropped == ()
    _guard(inst, got.best)
    _at_local_optimum(inst, 0, got.best)


def _with_search(setup, search):
    """``setup`` with other search bounds."""
    return dataclasses.replace(setup, options=dataclasses.replace(setup.options, search=search))


def _whole_start(inst, index):
    """The local search's start under ``whole``, built independently: the 2-OPT
    tour, from the dock back to the dock, of the class's stops as the brute
    force rebuilds them (B2's deadlines), its priority stops first, folded
    with skipping and its exempt stops protected (``brute.priority_start``)."""
    world, cls = inst.world, inst.world.classes[index]
    stops = [brute.stop(wp, wp.devices, world) for wp in cls.stops]
    return brute.priority_start(world, cls, order_contacts(stops, DOCK, end=DOCK))


def test_the_whole_mode_start_flies_capped_stops_first_and_protects_exempt_ones():
    """Under ``whole`` the local search starts as U3's trim does, with the cap's
    rules on whole stops (spec, other choices 5): the stops that hold a capped
    member first, and those whose members are all capped exempt from their
    own deadline. Eight one-device stops on a ring, a 30 s budget: the tour's
    third stop is capped and already overdue at takeoff, its last stop is
    capped. Flown in tour order the budget runs out before the last one;
    unprotected, the overdue one is refused. The start serves both, and the
    unbounded search keeps them and ends at a local optimum of whole moves."""
    ring = {f"r{k}": (30.0 * math.cos(k * math.pi / 4), 30.0 * math.sin(k * math.pi / 4))
            for k in range(8)}
    kw = dict(classes=[("wide", 1.0, 2.0, 0.0, 0.0)], budget=30.0, whole=True, t_ref=200.0)
    tour = order_contacts(_build(ring, **kw).entries[0].stops, DOCK, end=DOCK)
    (past,), (late,) = tour[2].devices, tour[-1].devices
    ages = {DeviceID(d): 1 for d in ring}
    ages.update({past: 3, late: 4})
    deadlines = {DeviceID(d): 1e6 for d in ring}
    deadlines[past] = -1.0
    kw.update(ages=ages, deadlines=deadlines, s_missions=3)
    inst = _build(ring, search=PlanSearchParams(heuristic_max_evaluations=1), **kw)
    got = _class(inst)
    assert got.mode == SEARCH_LOCAL and got.summary["bounded"] is True
    start = _whole_start(inst, 0)
    assert got.best.fold.route == start and got.n_candidates == 2
    assert [wp.devices for wp in start] == [(past,), (late,)] and got.best.cap_key == ()
    # Why each rule matters here.
    world, model = inst.world, inst.world.classes[0].model
    stops = order_contacts([brute.stop(wp, wp.devices, world) for wp in world.classes[0].stops],
                           DOCK, end=DOCK)
    exempt = frozenset(wp for wp in stops if brute.exempt(wp, world.capped))
    assert exempt == frozenset(start)
    alone = model.admit(world.start, start[0], rule=RULE_DEADLINE_BUDGET,
                        budget_end=world.budget_end)
    assert alone.reason == "overdue"                    # flown only because it is protected
    in_order = model.fold(stops, world.start, rule=RULE_DEADLINE_BUDGET,
                          budget_end=world.budget_end, skip=True, protected=exempt).route
    assert [wp.devices for wp in in_order] == [tour[0].devices, tour[1].devices, (past,)]
    full = _class(_build(ring, **kw))
    assert full.summary["bounded"] is False and full.best.cap_key == ()
    assert {past, late} <= full.best.served
    _guard(inst, full.best)
    _at_local_optimum(inst, 0, full.best)


def test_the_whole_mode_start_is_the_capped_first_protected_fold():
    """On random problems under ``whole`` (natural and forced local searches),
    with one walk allowed, the class's plan is that start or the empty plan,
    whichever the brute force prices lower; and on many capped ones putting
    the priority stops first, or protecting the exempt ones, changes the
    start."""
    seen = Counter()
    one = PlanSearchParams(heuristic_max_evaluations=1)
    for seed in range(60):
        if seed % 2:
            inst = _random(2000 + seed, 10 + seed % 7, many_stops=True, whole=True)
            search = one
        else:
            inst = _random(6000 + seed, 3 + seed % 5, whole=True)
            search = dataclasses.replace(FORCED, heuristic_max_evaluations=1)
        setup, world = _with_search(inst.setup, search), inst.world
        for index, cls in enumerate(world.classes):
            got = search_class(setup, inst.entries[index], **inst.kw)
            if got.mode != SEARCH_LOCAL:
                continue
            assert got.summary["bounded"] is True and got.summary["evaluations"] == 1
            start = _whole_start(inst, index)
            flown = brute.walk(world, cls, [(_source(inst, index, wp), wp.devices) for wp in start])
            want = min(flown, brute.walk(world, cls, []), key=lambda p: p.key)
            assert got.best.fold.route == (start if want is flown else ())
            assert got.best.cap_key == want.cap_key
            assert got.best.terms.v == pytest.approx(want.v, abs=1e-9)
            # The start the other way: in tour order, or without protection.
            tour = order_contacts([brute.stop(wp, wp.devices, world) for wp in cls.stops],
                                  DOCK, end=DOCK)
            first = sorted(tour, key=lambda wp: not world.capped & set(wp.devices))
            exempt = frozenset(wp for wp in tour if brute.exempt(wp, world.capped))

            def fold(route, protected):
                return cls.model.fold(route, world.start, rule=RULE_DEADLINE_BUDGET,
                                      budget_end=world.budget_end, skip=True,
                                      protected=protected).route

            seen["capped"] += bool(world.capped)
            seen["start"] += want is flown and bool(start)
            seen["priority first changes it"] += fold(tour, exempt) != start
            seen["protection changes it"] += fold(first, frozenset()) != start
    assert seen["capped"] >= 30 and seen["start"] >= 30, seen
    assert seen["priority first changes it"] >= 5 and seen["protection changes it"] >= 5, seen


# The problems the local search's scan is replayed on (``_scan_problem``), by
# family, with how many of each; the small ones are also replayed cut short.
SCAN = {"natural": 24, "forced": 120, "whole": 24, "whole_forced": 60, "mirror": 200}
SMALL = ("forced", "whole_forced", "mirror")


def _mirror(seed):
    """A small problem of mirror-image twins, a device at (x, y) and one at
    (x, -y), maybe one more on the axis, one-device stops, under
    ``deadline_bounds = "delivery"``, searched locally by force. Two routes
    that differ only by a twin reach the axis stop at the same clock with the
    same energy, bit for bit, but with different deadlines on board
    (``deliver_by``), so the local search's walk memo must tell such states
    apart."""
    rng = random.Random(seed)
    positions = {}
    for k in range(rng.choice((2, 3, 4))):
        x, y = rng.choice((10.0, 20.0, 30.0)) * (k + 1) / 2, rng.choice((4.0, 8.0, 12.0))
        positions[f"u{k}"], positions[f"l{k}"] = (x, y), (x, -y)
    if rng.random() < 0.7:
        positions["m"] = (rng.choice((15.0, 25.0, 40.0)), 0.0)
    deadlines = {DeviceID(d): rng.choice((15.0, 25.0, 40.0, 60.0, 1e6)) for d in positions}
    return _build(positions, classes=[("wide", 0.5, rng.choice((1.0, 2.0)), 0.0, 0.0)],
                  budget=rng.choice((60.0, 90.0, 150.0)), deadlines=deadlines,
                  bounds="delivery", t_ref=200.0, search=FORCED,
                  capacity=None if rng.random() < 0.5 else rng.choice((8000.0, 12000.0, 20000.0)))


def _twin_states(inst):
    """How many pairs of two-stop routes, a twin and then the same other stop,
    both admitted, end in states equal but for ``deliver_by``."""
    world, cls = inst.world, inst.world.classes[0]
    stops = {wp.devices[0]: wp for wp in cls.stops}
    found = 0
    for k in range(4):
        pair = (stops.get(DeviceID(f"u{k}")), stops.get(DeviceID(f"l{k}")))
        if None in pair:
            break
        for other in cls.stops:
            if other in pair:
                continue
            a, b = (cls.model.fold([twin, other], world.start, rule=RULE_DEADLINE_BUDGET,
                                   budget_end=world.budget_end, skip=False).state for twin in pair)
            found += ((a.pose, a.clock, a.energy_j) == (b.pose, b.clock, b.energy_j)
                      and a.deliver_by != b.deliver_by)
    return found


def _scan_problem(family, seed):
    """``natural`` and ``whole`` are the local-optimum problems (many stops,
    mostly of one device), ``forced`` and ``whole_forced`` small ones searched
    locally by force, with stops of several members (``forced`` is the forced
    local-optimum problems'), ``mirror`` the twins of :func:`_mirror`. Returns
    the problem and its setup."""
    if family in ("natural", "whole"):
        inst = _random(2000 + seed, 10 + seed % 7, many_stops=True, whole=family == "whole")
        return inst, inst.setup
    if family == "mirror":
        inst = _mirror(seed)
        return inst, inst.setup
    if family == "forced":
        inst = _random(9000 + seed, 3 + seed % 5, whole=False)
    else:
        inst = _random(6000 + seed, 3 + seed % 5, whole=True)
    return inst, _with_search(inst.setup, FORCED)


def _replay(inst, setup, index, *, world=None, single=False):
    """``brute.local_search`` of class ``index`` under ``setup``'s bounds, from
    the search's start (U3's trim of the tour; the capped-first protected fold
    under ``whole``), priced in ``world`` (the problem's own unless given);
    ``single``: the plan key's scan alone."""
    bounds = setup.options.search
    world = inst.world if world is None else world
    start = _whole_start(inst, index) if world.whole else _start(inst, index)
    return brute.local_search(world, world.classes[index], start,
                              max_passes=bounds.heuristic_max_passes,
                              max_evaluations=bounds.heuristic_max_evaluations, single=single)


def _follows(got, ref, where):
    """The search did what the replay did: the same passes, walks, bound and
    candidates (and under ``lexicographic`` the same passes and walks in its
    first scan, on the weighted key), and the same plan, which its summary
    describes."""
    s = got.summary
    assert (s["passes"], s["evaluations"], s["bounded"]) == (
        ref.passes, ref.evaluations, ref.bounded), (where, ref.taken)
    first = (s.get("weighted_passes"), s.get("weighted_evaluations"))
    assert first == (ref.first or (None, None)), (where, ref.taken)
    assert got.n_candidates == s["candidates"] == ref.candidates, where
    assert got.best.cap_key == ref.plan.cap_key and got.best.served == ref.plan.served, where
    assert got.best.terms.v == pytest.approx(ref.plan.v, abs=1e-9), where
    assert _stops(got.best.fold.route) == ref.plan.stops, where
    assert (s["served"], s["cap_key"]) == (len(ref.plan.served), list(ref.plan.cap_key)), where


@pytest.mark.parametrize("family", SCAN)
def test_the_local_search_follows_its_documented_scan(family):
    """The module documents the local search's scan: drop, insert, reverse,
    then drop-member moves, each kind in a fixed order; the first improvement
    taken; the route as flown; a neighbour walked before skipped. Under
    ``lexicographic`` it runs weighted's own scan first, then the plan key's
    from the best plan met so far and, when that is another route, from the
    start, each taking an earlier scan's walks rather than walking again, all
    within one pair of bounds. The scans decide which local optimum the
    search reaches and how many walks the trace records, so the search must do
    exactly what the brute force's own replay of them does
    (``brute.local_search``: its walks, its V): the same plan, passes, walks,
    bound and candidates, and the first scan's passes and walks; on the small
    problems also when either bound cuts it short. The replays take every
    kind of move (no drop-member move under ``whole``), run several passes
    and skip neighbours walked before; under ``lexicographic`` they reuse
    walks and run two scans and three; the mirror problems reach states the
    walk memo must tell apart by their ``deliver_by`` alone."""
    seen = Counter()
    for seed in range(SCAN[family]):
        inst, setup = _scan_problem(family, seed)
        if family == "mirror":
            seen["twin states"] += _twin_states(inst)
        for index in range(len(inst.entries)):
            got = search_class(setup, inst.entries[index], **inst.kw)
            if got.mode != SEARCH_LOCAL:
                continue
            where = f"{family} seed {seed} class {index}"
            assert (got.summary["index"], got.summary["stops"]) == (
                index, len(inst.entries[index].stops)), where
            ref = _replay(inst, setup, index)
            _follows(got, ref, where)
            seen.update(ref.taken)
            seen["passes"] += ref.passes > 1
            seen["skipped"] += ref.skipped > 0
            seen["reused"] += ref.reused > 0
            seen[f"{len(ref.starts)} scans"] += 1
            if family in SMALL and ref.passes > 1:
                for bounds in ({"heuristic_max_passes": ref.passes - 1},
                               {"heuristic_max_evaluations": max(1, ref.evaluations // 2)}):
                    cut = _with_search(setup, dataclasses.replace(setup.options.search, **bounds))
                    again = _replay(inst, cut, index)
                    assert again.bounded, where
                    _follows(search_class(cut, inst.entries[index], **inst.kw), again,
                             f"{where} cut to {bounds}")
                    seen["cut"] += 1
    kinds = {"forced": ("drop", "insert", "reverse", "member"),
             "mirror": ()}.get(family, ("drop", "insert", "reverse"))
    assert all(seen[kind] >= 5 for kind in kinds), seen
    assert seen["passes"] >= 10 and seen["skipped"] >= 10, seen
    assert seen["reused"] >= 5 and seen["2 scans"] >= 5 and seen["3 scans"] >= 5, seen
    assert family == "mirror" or seen["1 scans"] >= 10, seen
    assert not family.startswith("whole") or seen["member"] == 0, seen
    assert family not in SMALL or seen["cut"] >= 20, seen
    assert family != "mirror" or seen["twin states"] >= 20, seen


def test_the_drop_member_move_sheds_the_least_worth_member_first():
    """One stop of three members: a (weight 9, beside the stop), b and c
    (weight 1, 6.7 and 7.3 m off it, 10 s of dwell per metre). Dropping b or
    c each raises V, dropping a lowers it. The drop-member moves are scanned
    least worth first in the F order (a, b, c), so one pass sheds c, not b;
    the next sheds b. Dropping the stop, walked in the first pass, is skipped
    as seen in the later ones: 4 walks in 3 passes. V alone ranks here
    (``weighted``). Under ``lexicographic`` shedding a member never serves
    more weight: that scan runs first, walk for walk, and the plan key's scan
    from the best plan met, the whole stop, then keeps it whole."""
    three = {"a": (20.0, 0.0), "b": (20.0, 6.0), "c": (20.0, -8.0)}
    a, b, c = (DeviceID(k) for k in three)
    kw = dict(classes=[("wide", 10.0, 1.0, 10.0, 0.0)], budget=None, t_ref=300.0,
              weights={a: 9.0, b: 1.0, c: 1.0},
              score=PlanScoreParams(coverage_rank="weighted"))
    inst = _build(three, search=FORCED, **kw)
    world, cls = inst.world, inst.world.classes[0]
    (stop,) = inst.entries[0].stops
    assert brute.f_order(world, cls, stop) == [a, b, c]
    v = {m: brute.walk(world, cls, [(stop, m)]).v for m in ((a, b, c), (a, b), (a, c), (b, c))}
    assert v[a, b] > v[a, c] > v[a, b, c] > v[b, c]
    one = _class(_build(three, search=dataclasses.replace(FORCED, heuristic_max_passes=1), **kw))
    assert [wp.devices for wp in one.best.fold.route] == [(a, b)]
    got = _class(inst)
    assert [wp.devices for wp in got.best.fold.route] == [(a,)]
    assert (got.summary["passes"], got.summary["evaluations"], got.n_candidates) == (3, 4, 4)
    kept = _class(_build(three, search=FORCED, **dict(kw, score=PlanScoreParams())))
    assert [wp.devices for wp in kept.best.fold.route] == [(a, b, c)]
    s = kept.summary
    assert s["coverage_rank"] == "lexicographic"
    assert (s["weighted_passes"], s["weighted_evaluations"]) == (3, 4)
    # The best plan met is the start, so no third scan runs. The plan key's one
    # pass takes the drop and c's shedding from the first scan's walks and
    # walks b's and a's: 6 walks in 4 passes.
    assert (s["passes"], s["evaluations"], kept.n_candidates) == (4, 6, 6)


# --------------------------------------------------------------------------- #
# FB+c and F (critic A3), and the classes searched
# --------------------------------------------------------------------------- #

def _pinned(inst, name):
    """The same problem under ``fixed:<name>``: FB+c."""
    setup = PlanSetup(options=dataclasses.replace(inst.setup.options,
                                                  band_class_policy=f"fixed:{name}"),
                      classes=inst.setup.classes, reference=name, t_ref_s=inst.setup.t_ref_s,
                      turnaround_s=inst.setup.turnaround_s)
    entry = next(e for e in inst.entries if e.cls.name == name)
    return plan_search(setup, [entry], **inst.kw)


@pytest.mark.parametrize("rank", COVERAGE_RANKS)
@pytest.mark.parametrize("kind", ["exact", "stop_subsets", "local"])
def test_fb_c_commits_class_cs_best_and_f_is_never_worse(kind, rank):
    """Critic A3 under either rank: FB+c commits class c's own best, and F's
    plan key (U2's ``plan_key`` under the arm's rank) is the smallest of the
    FB+c keys, so never above any of them."""
    for seed in range(12):
        if kind == "exact":
            inst = _random(3000 + seed, 1 + seed % 6, rank=rank)
        elif kind == "stop_subsets":
            inst = _random(3000 + seed, 7 + seed % 4, few_stops=True, rank=rank)
        else:
            inst = _random(3000 + seed, 10 + seed % 4, many_stops=True, rank=rank)
        f = _search(inst)
        fixed = []
        for index, c in enumerate(inst.setup.classes):
            fb = _pinned(inst, c.name)
            own = _class(inst, index)
            assert fb.best.band == c.name and _key(inst, fb.best) == _key(inst, own.best)
            assert fb.best.fold.route == own.best.fold.route
            assert fb.best.terms == own.best.terms
            assert [dict(s) for s in fb.per_class] == [dict(f.per_class[index])]
            assert _key(inst, f.best) <= _key(inst, fb.best)
            fixed.append(_key(inst, fb.best))
        assert _key(inst, f.best) == min(fixed)


def test_the_search_takes_exactly_the_classes_the_policy_allows():
    inst = _build({"a": (5.0, 0.0), "b": (-5.0, 0.0)}, budget=30.0,
                  classes=[("wide", 3.0, 1.0, 0.0, 0.0), ("medium", 6.0, 2.0, 0.0, 0.0),
                           ("narrow", 12.0, 3.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="searches"):
        plan_search(inst.setup, inst.entries[:1], **inst.kw)          # one class missing
    with pytest.raises(ValueError, match="searches"):
        plan_search(inst.setup, inst.entries[::-1], **inst.kw)       # out of link order
    name = inst.entries[0].cls.name
    setup = PlanSetup(options=dataclasses.replace(inst.setup.options,
                                                  band_class_policy=f"fixed:{name}"),
                      classes=inst.setup.classes, reference=name, t_ref_s=inst.setup.t_ref_s,
                      turnaround_s=inst.setup.turnaround_s)
    with pytest.raises(ValueError, match="searches"):
        plan_search(setup, inst.entries, **inst.kw)                  # FB+c flies only c
    with pytest.raises(TypeError):
        plan_search(inst.setup, [inst.entries[0].cls], **inst.kw)


# --------------------------------------------------------------------------- #
# Pass 2 (decision 2 (b))
# --------------------------------------------------------------------------- #

def test_pass_2_is_priced_as_the_t_nom_helper_prices_it():
    """``fl_scheduler.nominal_mission_period_s``: the queue folded from the dock
    at clock 0, no gate, no budget, delivering, without skipping; plus its
    dwell and its energy with the return leg. The brute force's own pricing
    agrees."""
    inst = _random(7, 6)
    for entry, cls in zip(inst.entries, inst.world.classes):
        model = entry.cls.model
        queue = cls.pass_2_queue
        walk = model.fold(queue, FlightState(DOCK, 0.0), rule=RULE_NONE, budget_end=None,
                          pass_kind=DELIVER, skip=False)
        p2 = price_pass_2(model, queue)
        assert p2.time_s == walk.home
        assert p2.energy_j == pass_energy_j(model, walk.state)
        assert p2.dwell_s == pytest.approx(sum(model.ferry.dwell_s(wp, DELIVER) for wp in queue))
        assert p2.dwell_s == pytest.approx(0.5 * sum(model.ferry.dwell_s(wp, COLLECT)
                                                     for wp in queue))
        mine = cls.pass_2
        assert (p2.time_s, p2.dwell_s, p2.energy_j) == pytest.approx(
            (mine.time_s, mine.dwell_s, mine.energy_j), abs=1e-9)
    empty = price_pass_2(inst.entries[0].cls.model, [])
    assert (empty.time_s, empty.dwell_s, empty.energy_j) == (0.0, 0.0, 0.0)


def test_pass_2_needs_ferry_physics_and_sane_numbers():
    with pytest.raises(ValueError, match="ferry"):
        price_pass_2(FeasibilityModel(), [])
    for bad in ({"time_s": -1.0}, {"dwell_s": math.inf}, {"energy_j": math.nan}):
        with pytest.raises(ValueError):
            PassTwo(**{"time_s": 1.0, "dwell_s": 0.5, "energy_j": 2.0, **bad})
    with pytest.raises(TypeError):
        PassTwo(time_s=True, dwell_s=0.0, energy_j=0.0)


# --------------------------------------------------------------------------- #
# The result: guard, determinism, summaries
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", ["exact", "stop_subsets", "local"])
def test_the_result_does_not_depend_on_the_order_of_the_stops(kind):
    for seed in range(8):
        if kind == "exact":
            inst = _random(4000 + seed, 1 + seed % 6)
        elif kind == "stop_subsets":
            inst = _random(4000 + seed, 7 + seed % 4, few_stops=True)
        else:
            inst = _random(4000 + seed, 10 + seed % 4, many_stops=True)
        first = _search(inst)
        again = _search(inst)
        rng = random.Random(seed)
        shuffled = []
        for entry in inst.entries:
            stops = list(entry.stops)
            rng.shuffle(stops)
            shuffled.append(dataclasses.replace(entry, stops=tuple(stops)))
        other = plan_search(inst.setup, shuffled, **inst.kw)
        for res in (again, other):
            assert _key(inst, res.best) == _key(inst, first.best)
            assert res.best.terms == first.best.terms
            assert res.best.fold == first.best.fold
            assert [dict(s) for s in res.per_class] == [dict(s) for s in first.per_class]
            assert res.n_candidates == first.n_candidates
        _guard(inst, first.best)


@pytest.mark.parametrize("rank", COVERAGE_RANKS)
def test_the_summaries_are_what_the_commit_accepts(rank):
    """One JSON-ready summary per class, no wall time, each its own class's
    search: the counts, the best's V, served count, cap key and served weight
    share, and the rank the search applied (R11: a trace shows which rank
    chose the plan; F-cov reads ``weighted`` under either setting)."""
    inst = _random(9, 11, many_stops=True, rank=rank)
    assert inst.score.c_cov_per_device > 0.0
    result = _search(inst)
    assert isinstance(result, SearchResult)
    assert [s["band"] for s in result.per_class] == [c.name for c in inst.setup.searched]
    assert [dict(s) for s in plan_class_summaries(result.per_class, result.best.band)] == [
        dict(s) for s in result.per_class]
    for s in result.per_class:
        assert not any("wall" in key for key in s)
        base = {"band", "index", "mode", "stops", "candidates", "v", "served", "cap_key",
                "served_share", "coverage_rank"}
        extra = {"evaluations", "passes", "bounded"} if s["mode"] == SEARCH_LOCAL else set()
        if extra and rank == "lexicographic":
            # The first of the local search's scans, on the weighted key.
            extra |= {"weighted_passes", "weighted_evaluations"}
            assert 1 <= s["weighted_passes"] < s["passes"]
            assert 1 <= s["weighted_evaluations"] <= s["evaluations"]
        assert set(s) == base | extra
        assert s["coverage_rank"] == rank
    best = result.per_class[result.best.cls.index]
    assert (best["v"], best["served"], best["cap_key"], best["served_share"]) == (
        result.best.terms.v, len(result.best.served), list(result.best.cap_key),
        served_share(result.best.terms))
    assert result.n_candidates == sum(s["candidates"] for s in result.per_class)
    got = _class(inst, result.best.cls.index)
    assert isinstance(got, ClassResult) and _key(inst, got.best) == _key(inst, result.best)
    # Every class's summary is that class's own search's, not the best's.
    for index, (entry, s) in enumerate(zip(inst.entries, result.per_class)):
        own = _class(inst, index)
        assert dict(s) == dict(own.summary)
        assert (s["index"], s["stops"], s["candidates"]) == (
            entry.cls.index, len(entry.stops), own.n_candidates)
        assert (s["v"], s["served"], s["cap_key"], s["served_share"]) == (
            own.best.terms.v, len(own.best.served), list(own.best.cap_key),
            served_share(own.best.terms))
    # F-cov (κ = 0) records the rank it applies; the link term off alone does not.
    for score, applied in ((dataclasses.replace(inst.score, c_cov_per_device=0.0, c_link=0.0),
                            "weighted"),
                           (dataclasses.replace(inst.score, c_link=0.0), rank)):
        setup = dataclasses.replace(inst.setup, options=dataclasses.replace(inst.setup.options,
                                                                             score=score))
        summaries = plan_search(setup, inst.entries, **inst.kw).per_class
        assert {s["coverage_rank"] for s in summaries} == {applied}


def test_a_share_that_differs_by_rounding_alone_ties_and_v_decides():
    """The share is rounded to 9 decimals, as V is, so float noise in it never
    decides: a and b each fit alone, not together; a weighs 1e-12 more but
    lies farther out, so the shares tie and V serves the nearer b."""
    a, b = DeviceID("a"), DeviceID("b")
    inst = _build({"a": (40.0, 0.0), "b": (-20.0, 0.0)}, classes=[("wide", 1.0, 1.0, 0.0, 0.0)],
                  budget=20.0, t_ref=200.0, weights={a: 1.0 + 1e-12, b: 1.0})
    plans = list(brute.exact_plans(inst.world))
    assert sorted(len(p.served) for p in plans) == [0, 1, 1]                 # not both
    best = _search(inst).best
    _agree(best, brute.best(iter(plans)))
    alone = {p.served: p for p in plans if p.served}
    assert alone[frozenset({a})].share > alone[frozenset({b})].share
    assert alone[frozenset({a})].v < alone[frozenset({b})].v
    assert best.served == {b}


# --------------------------------------------------------------------------- #
# The mule's own physics (U6's plan classes)
# --------------------------------------------------------------------------- #

def _world_problem(spec, positions, budget, *, options, ages, s_missions, theta_bytes,
                   deadlines=None):
    """A problem on the mule runtime's own classes (U6) for ``spec``: the
    devices at ``positions`` (id to (x, y, z)), S3a and Pass 2 from a
    scheduler, as the scheduler will hand them over, each class the options
    search, from the dock at 0 s to ``budget``; S3's deadlines unless
    ``deadlines`` are given. The mule is imported here only."""
    from hermes.mule.ferry import FerryRuntime
    from hermes.scheduler import FLScheduler
    from hermes.types import MissionSlice, MuleID

    rt = FerryRuntime(spec, None, rf_range_m=60.0)
    rt.set_payload(theta_bytes=theta_bytes, synth_bytes=0)
    pos = {DeviceID(d): p for d, p in positions.items()}
    sch = FLScheduler(now_fn=lambda: 0.0)
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=tuple(pos), issued_round=0,
                                  issued_at=0.0))
    for d, p in pos.items():
        sch.device_states[d].last_known_position = p
    setup = PlanSetup(options=options, classes=rt.plan_classes(), reference=spec.band,
                      t_ref_s=200.0, turnaround_s=30.0)
    entries, brutes, found = [], [], {}
    for c in setup.searched:
        model = dataclasses.replace(c.model, ferry=c.model.ferry.bind(sch.device_states))
        bound = dataclasses.replace(c, model=model)
        stops = sch.build_contact_queue(rf_range_m=c.radius_m, mule_pose=model.ferry.dock)
        found = dict(sch.last_plan_deadlines)
        queue = sch.build_pass_2_queue(rf_range_m=c.radius_m, mule_pose=model.ferry.dock)
        entries.append(ClassInput(bound, tuple(stops), price_pass_2(model, queue)))
        brutes.append(brute.BruteClass(c.name, c.index, model, c.outage, stops, queue))
    deadlines = found if deadlines is None else deadlines
    cap = CapState(AgeCapSpec(s_missions=s_missions), ages)
    weights = demand_weights(tuple(pos), sch.device_states, ages=ages, miss_priority=True,
                             mode="age")
    start = FlightState(entries[0].cls.model.ferry.dock, 0.0)
    kw = dict(start=start, budget_end=budget, deadlines=deadlines,
              device_states=sch.device_states, cap=cap, weights=weights)
    score = options.score
    world = brute.World(classes=brutes, start=start, budget_end=budget, deadlines=deadlines,
                        device_states=sch.device_states, ages=cap.ages, capped=cap.capped,
                        weights=weights, c_time=score.c_time, kappa=score.c_cov_per_device,
                        c_link=score.c_link, c_energy=score.c_energy,
                        dwell_in_delta=score.dwell_in_delta, t_ref_s=200.0, turnaround_s=30.0,
                        p_hover_w=setup.p_hover_w, coverage_rank=score.coverage_rank)
    return setup, entries, kw, world


def _mule_problem(layout_seed, budget, rank):
    """A realistic problem: N = 6 devices on a realism layout, the mule
    runtime's own classes at 1 MB (U6), arm F under ``rank``."""
    from experiments.exp4.topology_builder import device_positions
    from hermes.mule.ferry import FerrySpec

    spec = FerrySpec.from_config(rf_range_m=60.0, seed=layout_seed, contact_band="wide",
                                 payload_bytes=1_000_000, contact_regime="jittery")
    xy = device_positions(6, layout_seed, 100.0)
    positions = {f"d{i}": (x, y, 0.0) for i, (x, y) in enumerate(xy)}
    ages = {DeviceID(d): 1 + i % 3 for i, d in enumerate(positions)}
    options = PlanOptions(score=PlanScoreParams(coverage_rank=rank))
    return _world_problem(spec, positions, budget, options=options, ages=ages, s_missions=3,
                          theta_bytes=18_756)


@pytest.mark.parametrize("rank", COVERAGE_RANKS)
@pytest.mark.parametrize("budget", [30.0, 60.0])
@pytest.mark.parametrize("layout_seed", [3, 17, 41])
def test_on_the_mules_own_physics_the_search_equals_the_brute_force(layout_seed, budget, rank):
    setup, entries, kw, world = _mule_problem(layout_seed, budget, rank)
    result = plan_search(setup, entries, **kw)
    _agree(result.best, brute.best(brute.exact_plans(world)))
    assert result.mode == SEARCH_EXACT
    assert {s["coverage_rank"] for s in result.per_class} == {rank}


# --------------------------------------------------------------------------- #
# The rank (the orchestrator's resolution R11)
# --------------------------------------------------------------------------- #

def _serves_anyone_from_takeoff(world):
    """Whether some stop of some class admits anyone from takeoff (whole under
    ``whole``, else any one member), priced by the brute force alone: exactly
    when a plan that serves anyone is admitted, in every search's family (a
    prefix that fits flies its first stop from takeoff, and the predicate is
    monotone in a stop's members)."""
    return any(brute.greedy(world, cls, world.start, brute.stop(wp, wp.devices, world))
               is not None for cls in world.classes for wp in cls.stops)


def test_under_lexicographic_the_empty_plan_wins_only_when_no_plan_that_serves_fits():
    """R11: every demanded device weighs more than 0, and a plan's cap key is
    never above the empty plan's, so under ``lexicographic`` (κ > 0) the
    empty plan is chosen exactly when no plan that serves anyone is admitted,
    in every mode. (Under ``weighted`` the empty plan can win although a
    device fits, as Pass 2 is paid only by plans that serve: U7's probe world
    below.)"""
    seen = Counter()
    families = [(seed, 1 + seed % 6, {}) for seed in range(60)]
    families += [(1000 + seed, 7 + seed % 6, {"few_stops": True}) for seed in range(20)]
    families += [(2000 + seed, 10 + seed % 7, {"many_stops": True}) for seed in range(20)]
    for seed, n, kw in families:
        inst = _random(seed, n, rank="lexicographic", **kw)
        if inst.score.c_cov_per_device == 0.0:
            continue                                       # F-cov: the next test
        result = _search(inst)
        fits = _serves_anyone_from_takeoff(inst.world)
        assert (not result.best.fold.route) is (not fits), (seed, n)
        seen["empty"] += not result.best.fold.route
        seen["searched"] += 1
        seen[result.mode] += 1
    assert seen["searched"] >= 60 and seen["empty"] >= 3, seen
    assert all(seen[mode] >= 10 for mode in (SEARCH_EXACT, SEARCH_STOP_SUBSETS, SEARCH_LOCAL)), seen


def _arm_f(inst, rank):
    """``inst`` scored as arm F (κ = 1, c₃ = c₂; its c₄ and F-dwell as drawn)
    under ``rank``: the search's setup and the brute force's world."""
    score = PlanScoreParams(c_cov_per_device=1.0, c_link=None, c_energy=inst.score.c_energy,
                            dwell_in_delta=inst.score.dwell_in_delta, coverage_rank=rank)
    setup = dataclasses.replace(inst.setup, options=dataclasses.replace(inst.setup.options,
                                                                         score=score))
    world = dataclasses.replace(inst.world, kappa=1.0, c_link=None, coverage_rank=rank)
    return setup, world


def test_the_reviews_local_problem_serves_every_device_under_lexicographic():
    """The review's problem (seed 30125: 15 devices, a stop each, 9 capped, arm
    F's score). The plan key's scan alone, the search before the fix, takes
    whatever insert raises the share first and cannot drop a stop to make
    room: it left d8 out (14 of 15 served, share 0.987), while ``weighted``
    serves all 15. The lexicographic search now runs weighted's scan first,
    walk for walk (13 passes and 527 walks on wide), and commits its plan."""
    inst = _random(30125, 15, many_stops=True, rank="lexicographic")
    got = {rank: plan_search(_arm_f(inst, rank)[0], inst.entries, **inst.kw)
           for rank in COVERAGE_RANKS}
    lex, wtd = got["lexicographic"], got["weighted"]
    assert lex.mode == wtd.mode == SEARCH_LOCAL and lex.best.band == "wide"
    assert len(lex.best.served) == 15 and served_share(lex.best.terms) == 1.0
    assert lex.best.fold == wtd.best.fold and lex.best.terms == wtd.best.terms
    first, own = lex.per_class[0], wtd.per_class[0]
    assert (first["weighted_passes"], first["weighted_evaluations"]) == (
        own["passes"], own["evaluations"]) == (13, 527)
    setup, world = _arm_f(inst, "lexicographic")
    _follows(search_class(setup, inst.entries[0], **inst.kw),
             _replay(inst, setup, 0, world=world), "30125 wide")
    alone = _replay(inst, setup, 0, world=world, single=True)
    assert len(alone.plan.served) == 14 and DeviceID("d8") not in alone.plan.served
    assert round(alone.plan.share, 3) == 0.987 and not alone.bounded
    assert set(alone.taken) == {"insert", "reverse"}


# Small random local problems at arm F's score on which the lexicographic
# local search's scans matter (a Phase 4 build probe over 400 seeds): the
# plan key's scan alone ends below weighted's plan for some class, or the best
# plan is the third scan's, strictly better than the second's.
SCANS_MATTER = (50092, 50119, 50219, 50265, 50283, 50289, 50375, 50380)


def test_under_lexicographic_the_local_search_never_ends_below_weighted_or_its_scan_alone():
    """The review of R11's local search: under ``lexicographic`` each class
    searched locally ends on a plan whose plan key is never above that of the
    plan ``weighted`` commits for the class (the first scan is weighted's own,
    walk for walk, with the whole bound: its passes and walks are those of
    weighted's summary), and, unless a bound ends it, never above the plan
    key's scan alone (the brute force's replay of the search before the fix)
    and a local optimum of every move under the plan key. On the pinned
    problems the scan alone ends below weighted's plan, or the third scan
    (from the start) beats the second (from the best plan met)."""
    seen = Counter()
    problems = [(seed, 7 + (seed - 50000) % 10) for seed in SCANS_MATTER]
    problems += [(50000 + s, 7 + s % 10) for s in range(24)]
    for seed, n in problems:
        inst = _random(seed, n, many_stops=True, rank="lexicographic")
        setup, world = _arm_f(inst, "lexicographic")
        w_setup, _ = _arm_f(inst, "weighted")
        params = setup.options.score
        lex = plan_search(setup, inst.entries, **inst.kw)
        assert plan_key(params, lex.best) <= plan_key(
            params, plan_search(w_setup, inst.entries, **inst.kw).best), seed
        for index, entry in enumerate(inst.entries):
            got = search_class(setup, entry, **inst.kw)
            if got.mode != SEARCH_LOCAL:
                continue
            where = f"seed {seed} class {index}"
            wtd = search_class(w_setup, entry, **inst.kw)
            s, w = got.summary, wtd.summary
            assert (s["weighted_passes"], s["weighted_evaluations"]) == (
                w["passes"], w["evaluations"]), where
            assert not w["bounded"] or s["bounded"], where
            assert plan_key(params, got.best) <= plan_key(params, wtd.best), where
            ref = _replay(inst, setup, index, world=world)
            _follows(got, ref, where)
            alone = _replay(inst, setup, index, world=world, single=True)
            # Each drop serves less weight: the plan key's scan never takes one.
            assert not {"drop", "member"} & set(alone.taken), where
            seen["classes"] += 1
            if s["bounded"] or alone.bounded:
                seen["bounded"] += 1
                continue
            assert ref.plan.key <= alone.plan.key, where
            _at_local_optimum(inst, index, got.best, world)
            seen["alone below weighted"] += alone.plan.key > plan_key(params, wtd.best)
            seen["better than alone"] += ref.plan.key < alone.plan.key
            seen["third scan wins"] += len(ref.ends) == 3 and ref.ends[2].key < ref.ends[1].key
    assert seen["alone below weighted"] >= 5 and seen["third scan wins"] >= 5, seen
    assert seen["classes"] >= 60, seen


@pytest.mark.parametrize("kind", ["exact", "stop_subsets", "local"])
def test_f_cov_is_cap_only_service_under_either_setting(kind):
    """F-cov (κ = c₃ = 0) ranks by V alone whatever ``coverage_rank`` says, so
    both settings choose the same plan, bit for bit, summaries included; with
    the cap off it flies empty, and every stop it flies holds a capped member
    (the exact search, choosing members, serves capped devices only)."""
    seen = Counter()
    for seed in range(16):
        if kind == "exact":
            inst = _random(5000 + seed, 1 + seed % 6)
        elif kind == "stop_subsets":
            inst = _random(5000 + seed, 7 + seed % 4, few_stops=True)
        else:
            inst = _random(5000 + seed, 10 + seed % 4, many_stops=True)
        results = {}
        for rank in COVERAGE_RANKS:
            f_cov = PlanScoreParams(c_cov_per_device=0.0, c_link=0.0,
                                    c_energy=inst.score.c_energy,
                                    dwell_in_delta=inst.score.dwell_in_delta, coverage_rank=rank)
            setup = dataclasses.replace(inst.setup, options=dataclasses.replace(
                inst.setup.options, score=f_cov))
            results[rank] = plan_search(setup, inst.entries, **inst.kw)
        lex, weighted = results["lexicographic"], results["weighted"]
        assert lex.best == weighted.best and lex.n_candidates == weighted.n_candidates
        assert [dict(s) for s in lex.per_class] == [dict(s) for s in weighted.per_class]
        assert {s["coverage_rank"] for s in lex.per_class} == {"weighted"}
        best, capped = lex.best, inst.cap.capped
        assert all(capped & set(wp.devices) for wp in best.fold.route)
        if not capped:
            assert best.fold.route == ()
        if kind == "exact" and not inst.world.whole:
            assert best.served <= capped
        seen["served"] += bool(best.served)
        seen["capped"] += bool(capped)
    assert seen["served"] >= 3 and seen["capped"] >= 5, seen


def _probe_world():
    """U7's empty-plan probe world (Phase 4 build), with U7's helpers:
    u and v 60 m either side of the dock each fit alone within the budget, not
    together; z 400 m out fits nowhere; FB+wide at 1 MB on critic B4's
    deterministic channel, T = 200 s, every deadline far off."""
    from tests.integration import test_p4_plan_missions as mission

    layout = (("u", (60.0, 0.0, 0.0)), ("v", (-60.0, 0.0, 0.0)), ("z", (400.0, 0.0, 0.0)))
    spec = mission.det_spec(7, layout)
    one, two = mission.priced_home(spec, layout, "u"), mission.priced_home(spec, layout, "uv")
    return mission, layout, spec, (one + two) / 2


def _probe_kw(mission, budget, rank, *, s=None, **score):
    """FB+wide's supervisor keywords (U7's ``plan_kw``) under ``rank``."""
    kw = mission.plan_kw(budget=budget, s=s, policy="fixed:wide")
    kw["plan_options"] = dataclasses.replace(
        kw["plan_options"], score=PlanScoreParams(coverage_rank=rank, **score))
    return kw


def test_in_u7s_probe_world_the_search_serves_v_or_flies_empty_by_rank():
    """U7's probe at the search: V alone prefers the empty plan (−3.02, its
    30 s turnaround) to serving v (−3.85: Pass 2 flies out to z), so
    ``weighted`` flies empty at κ = 1 although u and v each fit;
    ``lexicographic`` serves v (the key's stops break the tie with u). The
    brute force agrees under both."""
    mission, layout, spec, budget = _probe_world()
    one, two = mission.priced_home(spec, layout, "u"), mission.priced_home(spec, layout, "uv")
    assert one == mission.priced_home(spec, layout, "v") < budget < two
    assert mission.priced_home(spec, layout, "z") > budget
    got = {}
    for rank in COVERAGE_RANKS:
        options = PlanOptions(band_class_policy="fixed:wide",
                              score=PlanScoreParams(coverage_rank=rank))
        setup, entries, kw, world = _world_problem(
            spec, dict(layout), budget, options=options,
            ages={DeviceID(d): 1 for d, _ in layout}, s_missions=None, theta_bytes=52,
            deadlines={DeviceID(d): 1e6 for d, _ in layout})
        result = plan_search(setup, entries, **kw)
        _agree(result.best, brute.best(brute.exact_plans(world)))
        assert _serves_anyone_from_takeoff(world)
        got[rank] = result.best
    lex, weighted = got["lexicographic"], got["weighted"]
    assert lex.served == {DeviceID("v")} and served_share(lex.terms) == 1 / 3
    assert weighted.fold.route == () and served_share(weighted.terms) == 0.0
    assert weighted.terms.v == pytest.approx(-(30.0 / 200.0) ** 2 - 3.0, abs=1e-12)
    assert round(lex.terms.v, 4) == -3.8475 and lex.terms.v < weighted.terms.v


def test_in_u7s_probe_world_a_mission_serves_a_device_under_lexicographic_only():
    """U7's probe on the mule's supervisor, one mission each: ``lexicographic``
    flies v and Pass 2, ``weighted`` flies nothing and closes on the empty
    path; each commit records the rank that chose it and the served share."""
    mission, layout, spec, budget = _probe_world()
    flown = {}
    for rank in COVERAGE_RANKS:
        _, _, (rec,) = mission.fly(spec=spec, layout=layout, **_probe_kw(mission, budget, rank))
        r = rec.result
        (summary,) = r.plan["per_class"]
        assert (summary["band"], summary["coverage_rank"]) == ("wide", rank)
        assert summary["served_share"] == len(r.plan["served"]) / 3
        flown[rank] = r
    lex, weighted = flown["lexicographic"], flown["weighted"]
    assert lex.plan["served"] == ["v"] and not lex.empty
    assert [s["devices"] for s in lex.pass_1_flown] == [["v"]] and lex.pass_2_flown
    assert weighted.plan["served"] == [] and weighted.empty and weighted.pass_1_flown == []
    assert weighted.plan["score"]["mission_s"] == 30.0 < lex.plan["score"]["mission_s"]
    assert lex.plan["score"]["v"] < weighted.plan["score"]["v"]


def test_over_six_missions_lexicographic_never_flies_empty_while_a_device_fits():
    """U7's probe_empty_streak.py on the supervisor. u and v each fit alone at
    every mission, so ``lexicographic`` serves one every mission, the one
    missed longer (v first, the key's stops breaking mission 1's tie), and
    with S = 3 only z, which fits nowhere, violates the cap (``unplannable``).
    ``weighted`` at κ = 1 flies empty every mission with the cap off (the
    weights grow together), and with S = 3 flies empty at missions 1, 2 and 5
    and crowds u at mission 3."""
    mission, layout, spec, budget = _probe_world()
    runs = {}
    for rank in COVERAGE_RANKS:
        for s in (None, 3):
            _, _, recs = mission.fly(spec=spec, layout=layout, missions=6,
                                     **_probe_kw(mission, budget, rank, s=s))
            runs[rank, s] = [(rec.result.plan["served"], rec.result.empty,
                              mission.violations(rec.result)) for rec in recs]
    alternating = [["v"], ["u"]] * 3
    for s in (None, 3):
        assert [served for served, _, _ in runs["lexicographic", s]] == alternating
        assert not any(empty for _, empty, _ in runs["lexicographic", s])
    assert [v for _, _, v in runs["lexicographic", None]] == [[]] * 6
    assert [v for _, _, v in runs["lexicographic", 3]] == [[], []] + [
        [("z", age, "unplannable")] for age in (3, 4, 5, 6)]
    assert runs["weighted", None] == [([], True, [])] * 6
    assert [served for served, _, _ in runs["weighted", 3]] == [[], [], ["v"], ["u"], [], ["v"]]
    assert runs["weighted", 3][2][2] == [("z", 3, "unplannable"), ("u", 3, "crowded")]


def test_in_u7s_probe_world_f_cov_serves_only_what_the_cap_forces_under_either_setting():
    """F-cov (κ = c₃ = 0) on the supervisor: with the cap off it flies empty,
    and with every device capped (S = 1) it serves one capped device a
    mission, under either setting, whose commits both record ``weighted``."""
    mission, layout, spec, budget = _probe_world()
    for rank in COVERAGE_RANKS:
        for s, want in ((None, [[], []]), (1, [["v"], ["u"]])):
            _, _, recs = mission.fly(spec=spec, layout=layout, missions=2,
                                     **_probe_kw(mission, budget, rank, s=s,
                                                 c_cov_per_device=0.0, c_link=0.0))
            assert [rec.result.plan["served"] for rec in recs] == want, (rank, s)
            assert {rec.result.plan["per_class"][0]["coverage_rank"] for rec in recs} == {
                "weighted"}


# --------------------------------------------------------------------------- #
# The inputs the scheduler hands over
# --------------------------------------------------------------------------- #

def test_a_class_model_must_be_bound_to_the_device_states():
    inst = _random(3, 4)
    entry = inst.entries[0]
    loose = dataclasses.replace(entry.cls.model,
                                ferry=dataclasses.replace(entry.cls.model.ferry,
                                                          device_states=None))
    with pytest.raises(ValueError, match="bind"):
        ClassInput(dataclasses.replace(entry.cls, model=loose), entry.stops, entry.pass_2)


def test_a_class_input_refuses_what_s3a_never_gives():
    inst = _random(3, 4)
    entry = inst.entries[0]
    with pytest.raises(ValueError, match="one stop"):
        ClassInput(entry.cls, entry.stops + entry.stops[:1], entry.pass_2)
    with pytest.raises(TypeError):
        ClassInput(entry.cls, [entry.stops[0].devices], entry.pass_2)
    with pytest.raises(TypeError):
        ClassInput(entry.cls, entry.stops, (1.0, 0.0, 0.0))
    with pytest.raises(TypeError):
        ClassInput(entry.stops, entry.stops, entry.pass_2)


def test_the_search_refuses_inputs_that_do_not_describe_one_plan():
    inst = _build({"a": (5.0, 0.0), "b": (-5.0, 0.0)}, classes=[("wide", 3.0, 1.0, 0.0, 0.0)],
                  budget=30.0, s_missions=1)
    kw = inst.kw
    entry = inst.entries[0]
    A, B = DeviceID("a"), DeviceID("b")
    cases = [
        (dict(weights={A: 1.0}), ValueError, "cover exactly"),                 # b not demanded
        (dict(weights={A: 1.0, B: 1.0, DeviceID("c"): 1.0}), ValueError, "cover exactly"),
        (dict(deadlines={A: 100.0}), ValueError, "deadline"),
        (dict(cap=CapState(AgeCapSpec(s_missions=1), {A: 1, B: 1, DeviceID("z"): 4})),
         ValueError, "demanded"),
        (dict(cap={"a": 1}), TypeError, "CapState"),
        (dict(start=(0.0, 0.0, 0.0)), TypeError, "FlightState"),
        (dict(budget_end=math.inf), ValueError, "budget_end"),
        (dict(budget_end=True), TypeError, "budget_end"),
        (dict(budget_end="60"), TypeError, "budget_end"),
    ]
    for change, error, match in cases:
        with pytest.raises(error, match=match):
            search_class(inst.setup, entry, **{**kw, **change})
    with pytest.raises(TypeError, match="PlanSetup"):
        search_class(inst.setup.options, entry, **kw)
    with pytest.raises(TypeError, match="ClassInput"):
        search_class(inst.setup, entry.cls, **kw)
    with pytest.raises(TypeError, match="PlanSetup"):
        plan_search(inst.setup.options, inst.entries, **kw)
    other = PlanClass(name="wide", index=1, radius_m=3.0, model=entry.cls.model,
                      outage=entry.cls.outage)
    with pytest.raises(ValueError, match="class 0"):
        search_class(inst.setup, ClassInput(other, entry.stops, entry.pass_2), **kw)
    with pytest.raises(KeyError):
        search_class(inst.setup, ClassInput(dataclasses.replace(entry.cls, name="narrow"),
                                            entry.stops, entry.pass_2), **kw)
    # The weights themselves are U2's to check: a device weighing nothing is refused there.
    with pytest.raises(ValueError, match="> 0"):
        search_class(inst.setup, entry, **{**kw, "weights": {A: 0.0, B: 1.0}})


# --------------------------------------------------------------------------- #
# Layering and the recorded pipeline
# --------------------------------------------------------------------------- #

_STDLIB = {"__future__", "dataclasses", "math", "numbers", "types", "typing"}
_MODULE_LEVEL = {"hermes.types.ids", "hermes.types.scheduler",
                 "hermes.scheduler.routing.two_opt",
                 "hermes.scheduler.stages.s3b_feasibility",
                 "hermes.scheduler.stages.s3d_age_cap",
                 ".member_subset", ".plan_score", ".types"}
_FORBIDDEN = ("numpy", "hermes.l1", "hermes.mule", "hermes.mission", "experiments",
              "hermes.scheduler.policies", "hermes.scheduler.fl_scheduler")


def test_the_search_imports_the_plan_units_s3b_the_cap_stage_and_the_router_only():
    tree = ast.parse((REPO / "hermes/scheduler/plan/plan_search.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    assert names
    for name in names:
        assert not name.startswith(_FORBIDDEN), name
        assert name.split(".")[0] in _STDLIB or name in _MODULE_LEVEL, name
    # Every import is at module level: nothing is loaded lazily behind a call.
    kinds = (ast.Import, ast.ImportFrom)
    top = [n for n in tree.body if isinstance(n, kinds)]
    assert len(top) == len([n for n in ast.walk(tree) if isinstance(n, kinds)])


def _python(code: str) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_recorded_pipeline_never_loads_the_search():
    """Freeze Rule 1: the scheduler, S3b, the router, the policies, the mule and
    its process load no plan module; the search then imports cleanly, and on
    its own in a fresh interpreter."""
    _python(
        "import sys\n"
        "import hermes.scheduler\n"
        "import hermes.scheduler.fl_scheduler\n"
        "import hermes.scheduler.stages.s3b_feasibility\n"
        "import hermes.scheduler.routing.replan\n"
        "import hermes.scheduler.policies\n"
        "import hermes.mule.mule_main\n"
        "import hermes.processes.mule\n"
        "plan = sorted(m for m in sys.modules if m.startswith('hermes.scheduler.plan'))\n"
        "assert not plan, plan\n"
        "import hermes.scheduler.plan.plan_search as ps\n"
        "assert ps.plan_search\n"
    )
    # (numpy comes with hermes.types, not with the search: the AST test above
    # pins that the search imports none.)
    _python(
        "import sys\n"
        "import hermes.scheduler.plan.plan_search as ps\n"
        "loaded = [m for m in sys.modules if m.startswith(('hermes.l1', 'hermes.mule', "
        "'experiments', 'hermes.scheduler.policies'))]\n"
        "assert not loaded, loaded\n"
    )
