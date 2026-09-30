"""Capture of the legacy feasibility walks at afa9526.

Phase 3 (unit U4) rebuilds S3b, the D-arm budget walks, FedCS, the FedEx tour
diagnostics and the mule's in-flight check as folds over one predicate, behind
switches whose defaults must reproduce today exactly. Pinned here, over seeded
random instances and the existing exact-boundary cases:

* ``filter_feasible`` (S3b): kept, overdue and over-budget lists, with and
  without the miss-streak priority key;
* ``greedy_budget_walk`` with an EDF key and with the queue order (the Pass-2
  walk's key), and the D-arm admissions over it: MAX-AoI (D1), Oort (D2)
  and Whittle (D3, both variants, with the per-device inputs it ranked on);
* ``fedcs_greedy_select`` (arm D5), both value keys;
* ``FedExCarpPolicy.admit_and_order`` (arm D4), closed tour and dock depot, with
  every ``last_tour_*`` / ``last_*_without_return`` diagnostic;
* ``MuleSupervisor._remaining_is_feasible`` for our arms (the S3b rule), a
  budget-rule policy and a no-check policy, at every position of the queue;
* ``MuleSupervisor._budget_pass_2``: flown and skipped.

The supervisor methods are bound onto a stand-in, the pattern the existing
tests use (``test_mule_inflight_abort.py``; critic B7 keeps it working).

Results are stored per instance as a digest of their canonical form; the first
``FULL_DETAIL`` instances are also stored in full so a failure can be read.
Each instance is also stored as a digest (``in``), which proves the generator
still builds the instance the pins were recorded on. That digest hashes only
the field values ``ContactWaypoint`` and ``FeasibilityModel`` had at afa9526
(:data:`AFA9526_FIELDS`), so the default-valued fields unit U4 adds to them
(``band``, ``range_m``, ``pred_snr_db``, ``ferry``) do not move it (critic A3).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

from hermes.mule.mule_main import MuleSupervisor
from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.policies.budget_walk import greedy_budget_walk
from hermes.scheduler.policies.fedcs_degraded import (
    VALUE_DEVICES,
    VALUE_UNIT,
    FedCSDegradedPolicy,
    fedcs_greedy_select,
)
from hermes.scheduler.policies.fedex_carp import FedExCarpPolicy
from hermes.scheduler.policies.max_aoi import MaxAoIPolicy
from hermes.scheduler.policies.oort import OortPolicy, OortUnusableError
from hermes.scheduler.policies.whittle import WhittlePolicy
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, filter_feasible
from hermes.types import Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState, MuleID

from tests.golden._canon import TYPE_KEY, canon, digest

N_INSTANCES = 2400
FULL_DETAIL = 120
DOCK = (0.0, 0.0, 0.0)

#: The fields the instance's dataclasses had at afa9526, in their order then.
#: Phase 3 adds default-valued fields to both (design sections 3.1 and 4.5);
#: the instance digest must not see them.
AFA9526_FIELDS: Dict[str, Tuple[str, ...]] = {
    "ContactWaypoint": ("position", "devices", "bucket", "deadline_ts"),
    "FeasibilityModel": ("cruise_speed_m_s", "session_time_s"),
}


class _Stand:
    """The two supervisor methods under test, on a stand-in (critic B7)."""

    _remaining_is_feasible = MuleSupervisor._remaining_is_feasible
    _budget_pass_2 = MuleSupervisor._budget_pass_2

    def __init__(self, scheduler, *, pose, now):
        self.scheduler = scheduler
        self.mule_pose = pose
        self.mule_id = MuleID("m-golden")
        self._now = lambda: now


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #

def make_instance(seed: int) -> Dict[str, Any]:
    """One seeded instance: contacts, pose, clock, budget, model, priorities."""
    rng = random.Random(1_000_003 * seed + 17)
    n = rng.choice((0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9))
    scale = rng.choice((10.0, 40.0, 80.0, 150.0, 400.0))
    now = rng.choice((0.0, 1000.0, 1_000_000.0, 1.7e9 + rng.uniform(0.0, 1e5)))
    if rng.random() < 0.4:
        pose = DOCK
    else:
        pose = (rng.uniform(-scale, scale), rng.uniform(-scale, scale),
                0.0 if rng.random() < 0.7 else rng.uniform(0.0, 30.0))
    if rng.random() < 0.35:
        model = None
    else:
        model = FeasibilityModel(
            cruise_speed_m_s=rng.choice((1.0, 2.5, 5.0, 12.0)),
            session_time_s=rng.choice((0.0, 0.5, 1.0, 3.0)),
        )
    speed = (model or FeasibilityModel()).cruise_speed_m_s
    grid = rng.random() < 0.3                      # integer positions: exact ties
    contacts: List[ContactWaypoint] = []
    deadlines_pool: List[float] = []
    for i in range(n):
        if contacts and rng.random() < 0.1:
            pos = contacts[rng.randrange(len(contacts))].position    # co-located
        elif grid:
            pos = (float(rng.randint(-4, 4) * scale / 4), float(rng.randint(-4, 4) * scale / 4), 0.0)
        else:
            pos = (rng.uniform(-scale, scale), rng.uniform(-scale, scale),
                   0.0 if rng.random() < 0.8 else rng.uniform(0.0, 40.0))
        if deadlines_pool and rng.random() < 0.2:
            deadline = rng.choice(deadlines_pool)                     # deadline tie
        else:
            deadline = now + rng.uniform(-60.0, 3.0 * scale / speed + 60.0)
        deadlines_pool.append(deadline)
        devices = tuple(DeviceID(f"c{i}d{k}") for k in range(rng.choice((1, 1, 1, 2, 3))))
        contacts.append(ContactWaypoint(
            position=tuple(float(c) for c in pos), devices=devices,
            bucket=rng.choice(list(Bucket)), deadline_ts=float(deadline),
        ))
    if rng.random() < 0.15:
        budget = None
    else:
        budget = rng.choice((0.0, 5.0, 12.0, rng.uniform(1.0, 4.0 * scale / speed + 20.0)))
    start = now - (0.0 if rng.random() < 0.5 else rng.uniform(0.0, 0.5 * (budget or 10.0)))
    priority = {c.devices: rng.choice((0, 0, 1, 2, 3)) for c in contacts}
    ages = {d: (0.0 if rng.random() < 0.25 else now - rng.uniform(0.0, 500.0))
            for c in contacts for d in c.devices}
    return dict(seed=seed, contacts=contacts, pose=tuple(pose), now=float(now),
                budget=budget, start=float(start), model=model,
                priority=priority, last_clean=ages)


def canon_afa9526(obj: Any) -> Dict[str, Any]:
    """``canon(obj)`` cut down to the fields its type had at afa9526.

    At afa9526 this is exactly ``canon(obj)``, so the recorded instance
    digests still hold; a field added since is left out, and a field removed
    or renamed since raises ``AttributeError`` (a real change to report).
    """
    name = type(obj).__name__
    out: Dict[str, Any] = {TYPE_KEY: name}
    for fld in AFA9526_FIELDS[name]:
        out[fld] = canon(getattr(obj, fld))
    return out


def instance_digest(inst: Dict[str, Any]) -> str:
    """Digest of an instance's afa9526 field values (see :func:`canon_afa9526`)."""
    model = inst["model"]
    return digest(canon({
        "contacts": [canon_afa9526(c) for c in inst["contacts"]],
        "pose": inst["pose"], "now": inst["now"],
        "budget": inst["budget"], "start": inst["start"],
        "model": None if model is None else canon_afa9526(model),
        "priority": [[list(k), v] for k, v in inst["priority"].items()],
        "last_clean": inst["last_clean"],
    }), 10)


# --------------------------------------------------------------------------- #
# Walks
# --------------------------------------------------------------------------- #

def _idx(contacts: Sequence[ContactWaypoint], route: Sequence[ContactWaypoint]) -> List[int]:
    """Positions of ``route``'s waypoints in ``contacts`` (by identity)."""
    ids = {id(c): i for i, c in enumerate(contacts)}
    out = []
    for wp in route:
        if id(wp) not in ids:
            raise AssertionError("a walk returned a waypoint it was not given")
        out.append(ids[id(wp)])
    return out


def _fedex(policy: FedExCarpPolicy, contacts, pose, now, mdl, model) -> List[Any]:
    route = policy.admit_and_order(
        contacts, {}, SelectorEnv(mule_pose=pose, now=now),
        mission_deadline_ts=mdl, feasibility_model=model,
    )
    return [
        _idx(contacts, route),
        canon(policy.last_tour_cost_s), canon(policy.last_tour_fits),
        canon(policy.last_tour_overrun_s), canon(policy.last_return_leg_s),
        canon(policy.last_fits_without_return), canon(policy.last_overrun_without_return_s),
    ]


def rich_states(inst: Dict[str, Any]) -> Tuple[Dict[DeviceID, DeviceSchedulerState], Any]:
    """Device histories for the D2/D3 keys, and the mission round the plan is for.

    Drawn from their own stream, so the instance's other draws are unchanged.
    """
    rng = random.Random(7919 * inst["seed"] + 3)
    states: Dict[DeviceID, DeviceSchedulerState] = {}
    for c in inst["contacts"]:
        for d in c.devices:
            attempts = rng.randint(0, 6)
            states[d] = DeviceSchedulerState(
                device_id=d,
                last_clean_ts=float(inst["last_clean"][d]),
                last_loss=None if rng.random() < 0.3 else rng.uniform(0.05, 2.0),
                last_num_examples=rng.choice((0, 5, 17, 40)),
                on_time_count=rng.randint(0, 4),
                missed_count=rng.randint(0, 3),
                last_served_round=rng.randint(0, 5),
                last_clean_round=rng.randint(0, 5),
                reach_attempts=attempts,
                reach_answered=rng.randint(0, attempts),
                last_merged_round=None if rng.random() < 0.3 else rng.randint(0, 5),
                miss_streak=rng.randint(0, 3),
            )
    return states, rng.choice((None, 1, 2, 3, 6))


def _scheduler(budget, model, selector, start) -> FLScheduler:
    clock = [start]
    sch = FLScheduler(now_fn=lambda: clock[0], mission_budget_s=budget,
                      feasibility_model=model, target_selector=selector)
    sch.start_mission()
    return sch


def run_instance(inst: Dict[str, Any]) -> Dict[str, Any]:
    contacts = inst["contacts"]
    pose, now, budget, model = inst["pose"], inst["now"], inst["budget"], inst["model"]
    mdl = None if budget is None else inst["start"] + budget
    prio = inst["priority"]
    out: Dict[str, Any] = {}

    res = filter_feasible(contacts, now=now, mule_pose=pose, mission_deadline_ts=mdl, model=model)
    out["s3b"] = [_idx(contacts, res.kept), _idx(contacts, res.dropped_overdue),
                  _idx(contacts, res.dropped_budget), res.n_dropped]
    res = filter_feasible(contacts, now=now, mule_pose=pose, mission_deadline_ts=mdl,
                          model=model, priority=lambda c: prio[c.devices])
    out["s3b_prio"] = [_idx(contacts, res.kept), _idx(contacts, res.dropped_overdue),
                       _idx(contacts, res.dropped_budget), res.n_dropped]

    edf_key = lambda c: (c.deadline_ts, c.position, c.devices)   # noqa: E731
    out["greedy_edf"] = _idx(contacts, greedy_budget_walk(
        contacts, key=edf_key, mule_pose=pose, now=now, mission_deadline_ts=mdl, model=model))
    order = {id(c): i for i, c in enumerate(contacts)}
    out["greedy_order"] = _idx(contacts, greedy_budget_walk(
        contacts, key=lambda c: (order[id(c)],), mule_pose=pose, now=now,
        mission_deadline_ts=mdl, model=model))

    states = {
        d: DeviceSchedulerState(device_id=d, last_clean_ts=float(ts))
        for d, ts in inst["last_clean"].items()
    }
    env = SelectorEnv(mule_pose=pose, now=now)
    out["d1_max_aoi"] = _idx(contacts, MaxAoIPolicy().admit_and_order(
        contacts, states, env, mission_deadline_ts=mdl, feasibility_model=model))
    out["fedcs_unit"] = _idx(contacts, fedcs_greedy_select(
        contacts, value=VALUE_UNIT, mule_pose=pose, now=now, mission_deadline_ts=mdl, model=model))
    out["fedcs_devices"] = _idx(contacts, FedCSDegradedPolicy(VALUE_DEVICES).admit_and_order(
        contacts, states, env, mission_deadline_ts=mdl, feasibility_model=model))

    rich, mission_round = rich_states(inst)
    env_r = SelectorEnv(mule_pose=pose, now=now, mission_round=mission_round)
    try:
        out["d2_oort"] = _idx(contacts, OortPolicy().admit_and_order(
            contacts, rich, env_r, mission_deadline_ts=mdl, feasibility_model=model))
    except OortUnusableError as e:
        out["d2_oort"] = ["raised", type(e).__name__]
    for variant in ("expected", "literal"):
        pol = WhittlePolicy(variant)
        route = pol.admit_and_order(contacts, rich, env_r, mission_deadline_ts=mdl,
                                    feasibility_model=model)
        out[f"d3_whittle_{variant}"] = [
            _idx(contacts, route),
            canon({str(k): v for k, v in sorted(pol.last_device_inputs.items())}),
        ]

    out["fedex_closed"] = _fedex(FedExCarpPolicy(), contacts, pose, now, mdl, model)
    out["fedex_dock"] = _fedex(FedExCarpPolicy(depot=DOCK, restarts=3, seed=5),
                               contacts, pose, now, mdl, model)

    inflight: Dict[str, List[bool]] = {}
    for name, selector in (("s3b", None), ("budget_rule", MaxAoIPolicy()),
                           ("none_rule", FedExCarpPolicy())):
        stand = _Stand(_scheduler(budget, model, selector, inst["start"]), pose=pose, now=now)
        inflight[name] = [bool(stand._remaining_is_feasible(contacts[k:]))
                          for k in range(len(contacts) + 1)]
    out["inflight"] = inflight

    stand = _Stand(_scheduler(budget, model, None, inst["start"]), pose=pose, now=now)
    fly, skip = stand._budget_pass_2(list(contacts))
    out["pass_2"] = [_idx(contacts, fly), _idx(contacts, skip)]
    return out


def build_random_cases() -> Dict[str, Any]:
    cases: Dict[str, Any] = {}
    for seed in range(N_INSTANCES):
        inst = make_instance(seed)
        result = run_instance(inst)
        entry: Dict[str, Any] = {"in": instance_digest(inst), "out": digest(result)}
        if seed < FULL_DETAIL:
            entry["detail"] = result
        cases[f"random:{seed:04d}"] = entry
    return cases


# --------------------------------------------------------------------------- #
# Exact boundaries (the cases the existing tests pin, gathered in one place)
# --------------------------------------------------------------------------- #

def _wp(x: float, deadline: float, *devs: str, y: float = 0.0) -> ContactWaypoint:
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=float(deadline))


def build_boundary_cases() -> Dict[str, Any]:
    up = lambda x: math.nextafter(x, math.inf)     # noqa: E731
    down = lambda x: math.nextafter(x, -math.inf)  # noqa: E731
    cases: Dict[str, Any] = {}
    m4 = FeasibilityModel(cruise_speed_m_s=4.0, session_time_s=7.0)
    m1 = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)

    # FeasibilityModel.cost itself.
    cases["cost"] = [canon(m.cost(a, b)) for m in (FeasibilityModel(), m4, m1)
                     for a, b in (((0, 0, 0), (40, 0, 0)), ((3, 4, 0), (0, 0, 0)),
                                  ((1.5, -2.25, 7.0), (-9.0, 11.0, 0.5)), ((5, 5, 5), (5, 5, 5)))]

    # S3b: arrival == deadline is kept, one ulp later is overdue; clock + total
    # == mission deadline is kept, one ulp less budget drops it.
    # 40 m at 4 m/s = 10 s of transit, then a 7 s session.
    for label, dl, mdl in (("arrive_eq", 10.0, 100.0), ("arrive_ulp_late", down(10.0), 100.0),
                           ("budget_eq", 1e9, 17.0), ("budget_ulp_short", 1e9, down(17.0))):
        wp = _wp(40.0, dl, "a")
        r = filter_feasible([wp], now=0.0, mission_deadline_ts=mdl, model=m4)
        cases[f"s3b:{label}"] = [len(r.kept), len(r.dropped_overdue), len(r.dropped_budget)]
    # No budget is a strict no-op: same objects, same order, even hopeless ones.
    hopeless = [_wp(0.0, -1e9, "x"), _wp(1e6, -1e9, "y")]
    r = filter_feasible(hopeless, now=0.0, mission_deadline_ts=None)
    cases["s3b:no_budget_noop"] = [r.kept == hopeless and all(p is q for p, q in zip(r.kept, hopeless)),
                                   r.n_dropped]
    # EDF order, and the walk's cumulative cost (test_s3b_feasibility).
    cs = [_wp(1.0, 30.0, "late"), _wp(2.0, 10.0, "early"), _wp(3.0, 20.0, "mid")]
    r = filter_feasible(cs, now=0.0, mission_deadline_ts=1e9,
                        model=FeasibilityModel(cruise_speed_m_s=1e6, session_time_s=0.0))
    cases["s3b:edf_order"] = [c.devices[0] for c in r.kept]
    near, far = _wp(3.0, 1e9, "near"), _wp(4.0, 1e9, "far")
    cases["s3b:cumulative"] = [
        [c.devices[0] for c in filter_feasible([near, far], now=0.0, mission_deadline_ts=m,
                                               model=m1).kept]
        for m in (5.0, 3.5, 4.0, down(4.0))
    ]
    # Ties: equal deadline and position fall back to the devices tuple.
    tie = [_wp(10.0, 50.0, "b"), _wp(10.0, 50.0, "a"), _wp(10.0, 50.0, "a", "z")]
    cases["s3b:tie_break"] = [list(c.devices) for c in
                              filter_feasible(tie, now=0.0, mission_deadline_ts=1e9, model=m1).kept]
    cases["s3b:expired_budget"] = len(filter_feasible([_wp(0.0, 1e9, "a")], now=100.0,
                                                      mission_deadline_ts=50.0,
                                                      model=FeasibilityModel(1.0, 1.0)).kept)

    # Greedy walk: <= admits, and it skips rather than stops.
    for label, mdl in (("eq", 17.0), ("ulp_short", down(17.0))):
        cases[f"greedy:{label}"] = len(greedy_budget_walk(
            [_wp(40.0, 0.0, "a")], key=lambda c: (0,), mule_pose=DOCK, now=0.0,
            mission_deadline_ts=mdl, model=m4))
    # The Pass-2 walk example of test_phase1_two_pass_ages: a (20 m) fits, b
    # (60 m more) does not, c (30 m) still does.
    q = [_wp(20.0, 0.0, "a"), _wp(60.0, 0.0, "b"), _wp(30.0, 0.0, "c")]
    stand = _Stand(_scheduler(10.0, None, None, 0.0), pose=DOCK, now=0.0)
    fly, skip = stand._budget_pass_2(q)
    cases["pass_2:skip_not_stop"] = [[w.devices[0] for w in fly], [w.devices[0] for w in skip]]
    for label, budget in (("eq", 10.0), ("ulp_short", down(10.0))):
        # 45 m at the default 5 m/s is 9 s, plus the 1 s session: exactly 10 s.
        stand = _Stand(_scheduler(budget, None, None, 0.0), pose=DOCK, now=0.0)
        fly, skip = stand._budget_pass_2([_wp(45.0, 0.0, "e")])
        cases[f"pass_2:{label}"] = [len(fly), len(skip)]
    stand = _Stand(_scheduler(None, None, None, 0.0), pose=DOCK, now=0.0)
    cases["pass_2:no_budget"] = [len(x) for x in stand._budget_pass_2(q)]

    # FedCS: 55 m at 5 m/s + 1 s = 12 s exactly is admitted (deviation 5).
    wp = _wp(55.0, 0.0, "a")
    for label, mdl in (("eq", 12.0), ("minus_1e-9", 12.0 - 1e-9), ("ulp_short", down(12.0))):
        for value in (VALUE_UNIT, VALUE_DEVICES):
            cases[f"fedcs:{label}:{value}"] = len(fedcs_greedy_select(
                [wp], value=value, mule_pose=DOCK, now=0.0, mission_deadline_ts=mdl, model=None))
    cases["fedcs:zero_budget"] = len(fedcs_greedy_select(
        [_wp(1.0, 0.0, "a")], mule_pose=DOCK, now=0.0, mission_deadline_ts=0.0))
    # The worked example: unit key routes P then R, devices key routes Q only.
    P, Q, R = _wp(10.0, 0.0, "p"), _wp(0.0, 0.0, "q1", "q2", "q3", y=30.0), _wp(20.0, 0.0, "r")
    for value in (VALUE_UNIT, VALUE_DEVICES):
        cases[f"fedcs:worked:{value}"] = [list(w.devices) for w in fedcs_greedy_select(
            [P, Q, R], value=value, mule_pose=DOCK, now=0.0, mission_deadline_ts=12.0)]

    # FedEx: 40 + 30 + 50 m at 4 m/s plus two 7 s sessions = 44 s exactly.
    NOW = 1000.0
    two = [_wp(40.0, 0.0, "a"), _wp(40.0, 0.0, "b", y=30.0)]
    for label, mdl in (("fits_eq", NOW + 44.0), ("overrun_half", NOW + 43.5),
                       ("fits_wo_return_eq", NOW + 31.5), ("ulp_short", down(NOW + 44.0)),
                       ("no_deadline", None)):
        for depot_label, pol in (("closed", FedExCarpPolicy()), ("dock", FedExCarpPolicy(depot=DOCK))):
            cases[f"fedex:{label}:{depot_label}"] = _fedex(pol, two, DOCK, NOW, mdl, m4)
    # A mule away from the dock: the depot, not the start, closes the tour.
    away = (100.0, 0.0, 0.0)
    ring = [_wp(0.0, 0.0, "a", y=50.0), _wp(-50.0, 0.0, "b"), _wp(50.0, 0.0, "c", y=-40.0)]
    for depot_label, pol in (("closed", FedExCarpPolicy()), ("dock", FedExCarpPolicy(depot=DOCK))):
        cases[f"fedex:away:{depot_label}"] = _fedex(pol, ring, away, NOW, NOW + 200.0, None)
    cases["fedex:empty"] = _fedex(FedExCarpPolicy(), [], DOCK, NOW, NOW + 1.0, None)

    # In flight (test_mule_inflight_abort / test_amendment8_mule_fixes).
    def infl(selector, budget, wps, *, now=NOW, pose=DOCK, start=NOW):
        sch = _scheduler(budget, m1, selector, start)
        return _Stand(sch, pose=pose, now=now)._remaining_is_feasible(wps)

    cases["inflight:no_budget"] = infl(None, None, [_wp(1e6, -1e9, "a")])
    cases["inflight:empty"] = infl(None, 100.0, [])
    cases["inflight:unreachable_by_deadline"] = infl(None, 1000.0, [_wp(500.0, NOW + 10.0, "a")])
    cases["inflight:reachable"] = infl(None, 1000.0, [_wp(5.0, 1e9, "a")])
    cases["inflight:max_aoi_overdue_fits"] = infl(MaxAoIPolicy(), 60.0, [_wp(10.0, NOW - 140.0, "s")])
    cases["inflight:ours_overdue"] = infl(None, 60.0, [_wp(10.0, NOW - 140.0, "s")])
    cases["inflight:max_aoi_budget_spent"] = infl(MaxAoIPolicy(), 60.0, [_wp(100.0, 1e9, "f")])
    cases["inflight:fedex_no_check"] = infl(FedExCarpPolicy(), 60.0, [_wp(100.0, NOW - 500.0, "f")])
    cases["inflight:deadline_eq"] = infl(None, 1000.0, [_wp(10.0, NOW + 10.0, "a")])
    cases["inflight:deadline_ulp_late"] = infl(None, 1000.0, [_wp(10.0, down(NOW + 10.0), "a")])
    # Budget boundaries at t = 0, where one ulp of budget is not rounded away.
    cases["inflight:budget_eq"] = infl(None, 10.0, [_wp(10.0, 1e9, "a")], now=0.0, start=0.0)
    cases["inflight:budget_ulp_short"] = infl(None, down(10.0), [_wp(10.0, 1e9, "a")],
                                              now=0.0, start=0.0)
    cases["inflight:from_pose"] = [
        infl(None, 1000.0, [_wp(500.0, NOW + 10.0, "a")], pose=(495.0, 0.0, 0.0)),
        infl(None, 1000.0, [_wp(500.0, NOW + 10.0, "a")], pose=(0.0, 0.0, 0.0)),
    ]
    cases["inflight:budget_rule_eq"] = infl(MaxAoIPolicy(), 10.0, [_wp(10.0, 0.0, "a")],
                                            now=0.0, start=0.0)
    cases["inflight:budget_rule_ulp"] = infl(MaxAoIPolicy(), down(10.0), [_wp(10.0, 0.0, "a")],
                                             now=0.0, start=0.0)
    # The clock has moved on since the mission started: the budget runs from
    # the start stamp, not from now.
    # (10 + 2**-40 is exact, and so is its sum with the 10 s leg; one ulp of
    # 10 would round back to 20.0 in that sum.)
    cases["inflight:elapsed"] = [
        infl(None, 20.0, [_wp(10.0, 1e9, "a")], now=10.0, start=0.0),
        infl(None, 20.0, [_wp(10.0, 1e9, "a")], now=10.0 + 2.0 ** -40, start=0.0),
    ]
    cases["inflight:budget_rule_elapsed"] = [
        infl(MaxAoIPolicy(), 20.0, [_wp(10.0, 0.0, "a")], now=10.0, start=0.0),
        infl(MaxAoIPolicy(), 20.0, [_wp(10.0, 0.0, "a")], now=up(10.0 + 2.0 ** -40), start=0.0),
    ]
    return {k: canon(v) for k, v in cases.items()}


def build_cases() -> Dict[str, Any]:
    cases = build_boundary_cases()
    cases = {f"boundary:{k}": v for k, v in cases.items()}
    cases.update(build_random_cases())
    return cases
