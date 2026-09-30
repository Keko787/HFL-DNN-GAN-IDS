"""FeRRy Phase 3 (unit U4): the re-plan and the pre-flight order check.

``FLScheduler.replan_remainder`` (design §3.4, critic C3 binding) and the
flown-order validation at the end of ``build_contact_queue`` (design §3.3).

Property tests over random instances, for every arm (ours with and without
the miss priority and with the ``trim`` fallback, D1 MAX-AoI, D2 Oort, D3
Whittle, D5 FedCS with both keys, D4 FedEx, and Pass 2) and on both the ferry
and the legacy model:

* the re-plan never admits a stop the predicate rejects: the returned route
  folds without skipping from the given state under the arm's rule;
* the route is a subset of the remainder, each stop once, and every other
  stop is dropped with a reason, in the remainder's order;
* a protected stop is dropped only if the protected-only route fails;
* no budget, no gate; D4 never re-plans; the baselines never get 2-OPT;
  ``trim`` never re-orders.

Then one deterministic case per branch of the order step (keep the current
order, the arm's own order over the admitted stops, the 2-OPT fallback, the
admission order and its guard fold, the ``trim`` fallback), each input of the
re-admission (the miss priority, δ_obs, the energy already spent, the mission
round, Pass 2's nearest-first walk), protected-first, the in-flight rule per
arm, and the pre-flight check's contract, including what critic C3's order
preference can and cannot do before takeoff.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pytest

from hermes.scheduler import FLScheduler, FLSchedulerError
from hermes.scheduler.policies import (
    FedCSDegradedPolicy,
    FedExCarpPolicy,
    MaxAoIPolicy,
    OortPolicy,
    WhittlePolicy,
)
from hermes.scheduler.policies.fedcs_degraded import VALUE_DEVICES, VALUE_UNIT
from hermes.scheduler.routing.replan import (
    FALLBACK_REORDER,
    FALLBACK_TRIM,
    FALLBACKS,
    ORDER_ADMISSION,
    ORDER_ARM,
    ORDER_ARM_TRIMMED,
    ORDER_CURRENT,
    ORDER_NONE,
    ORDER_TWO_OPT,
    ORDERS,
    ReplanResult,
    replan_route,
)
from hermes.scheduler.routing.two_opt import order_contacts
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    REASON_BUDGET,
    REASON_ENERGY,
    REASON_OVERDUE,
    REASONS,
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
)
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
    MissionSlice,
    MuleID,
)

COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
DOCK = (0.0, 0.0, 0.0)
N_CASES = 700


# --------------------------------------------------------------------------- #
# Random instances
# --------------------------------------------------------------------------- #

class _Dwell:
    """a + b*d seconds per member (half in Pass 2), unreachable beyond ``floor_m``;
    each dB of offset saves ``per_db`` seconds."""

    def __init__(self, a, b, floor_m, per_db):
        self.a, self.b, self.floor_m, self.per_db = a, b, floor_m, per_db

    def __call__(self, d, pass_kind, offset):
        if d > self.floor_m:
            return None
        t = self.a + self.b * d
        if pass_kind is DELIVER:
            t *= 0.5
        return max(0.0, t - self.per_db * offset)


class _Upload:
    def __init__(self, s):
        self.s = s

    def __call__(self):
        return self.s


def make_case(seed: int) -> Dict:
    rng = random.Random(90_001 * seed + 7)
    n = rng.choice((0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8))
    scale = rng.choice((20.0, 60.0, 120.0, 300.0))
    speed = rng.choice((1.0, 5.0, 12.0))
    now = rng.choice((0.0, 1e6, 1e6 + rng.uniform(0.0, 500.0)))
    grid = rng.random() < 0.3
    positions: Dict[DeviceID, Tuple[float, float, float]] = {}
    contacts: List[ContactWaypoint] = []
    deadlines: List[float] = []
    for i in range(n):
        if contacts and rng.random() < 0.1:
            stop = contacts[rng.randrange(len(contacts))].position
        elif grid:
            stop = (float(rng.randint(-4, 4)) * scale / 4, float(rng.randint(-4, 4)) * scale / 4, 0.0)
        else:
            stop = (rng.uniform(-scale, scale), rng.uniform(-scale, scale), 0.0)
        devs = []
        for j in range(rng.choice((1, 1, 2, 3))):
            did = DeviceID(f"c{i}d{j}")
            r, ang = rng.uniform(0.0, 45.0), rng.uniform(0.0, 2 * math.pi)
            positions[did] = (stop[0] + r * math.cos(ang), stop[1] + r * math.sin(ang), 0.0)
            devs.append(did)
        if deadlines and rng.random() < 0.2:
            deadline = rng.choice(deadlines)
        else:
            deadline = now + rng.uniform(-40.0, 3.0 * scale / speed + 90.0)
        deadlines.append(deadline)
        contacts.append(ContactWaypoint(position=tuple(float(c) for c in stop),
                                        devices=tuple(devs),
                                        bucket=rng.choice(list(Bucket)),
                                        deadline_ts=float(deadline)))
    if rng.random() < 0.2:
        model = FeasibilityModel(cruise_speed_m_s=speed,
                                 session_time_s=rng.choice((0.0, 1.0, 3.0)))
    else:
        dwell = _Dwell(rng.choice((0.0, 0.5, 2.0)), rng.choice((0.0, 0.02, 0.1)),
                       rng.choice((25.0, 35.0, 1e9)), rng.choice((0.0, 0.05)))
        # Some cases run the clock without a band (one session per contact,
        # critic A1); drawn apart so the other instances do not move.
        no_band = random.Random(31 * seed + 1).random() < 0.15
        physics = FerryPhysics(
            dock=DOCK,
            member_dwell_s=None if no_band else dwell,
            upload_s=_Upload(rng.choice((0.0, 3.0, 12.0))),
            p_move_w=143.6, p_hover_w=168.5,
            energy_capacity_j=(None if rng.random() < 0.6
                               else rng.uniform(0.0, 300.0 * scale / speed + 3000.0)),
            deadline_bounds=rng.choice(DEADLINE_BOUNDS),
            range_m=rng.choice((None, None, 40.0, 30.0)),
        )
        model = FeasibilityModel(cruise_speed_m_s=speed, session_time_s=1.0, ferry=physics)
    if rng.random() < 0.4:
        pose = DOCK
    else:
        pose = (rng.uniform(-scale, scale), rng.uniform(-scale, scale), 0.0)
    clock = now + rng.uniform(0.0, 60.0)
    energy = 0.0 if rng.random() < 0.5 else rng.uniform(0.0, 2000.0)
    state = FlightState(pose, clock, energy)
    if rng.random() < 0.1:
        budget_end = None
    else:
        budget_end = clock + rng.uniform(0.0, 4.0 * scale / speed + 60.0)
    remainder = list(contacts)
    rng.shuffle(remainder)
    protected = [wp for wp in remainder if rng.random() < 0.25] if rng.random() < 0.3 else []
    states = {
        did: DeviceSchedulerState(
            device_id=did, is_in_slice=True, last_known_position=pos,
            last_clean_ts=0.0 if rng.random() < 0.3 else now - rng.uniform(0.0, 300.0),
            miss_streak=rng.choice((0, 0, 1, 3)),
            reach_attempts=rng.randint(0, 5),
        )
        for did, pos in positions.items()
    }
    for st in states.values():
        st.reach_answered = min(st.reach_attempts, rng.randint(0, 5))
    # What Oort (D2) ranks on, and the rounds it and Whittle infer from;
    # drawn apart so the other draws of an instance do not move.
    oort = random.Random(13 * seed + 3)
    for st in states.values():
        if oort.random() < 0.75:
            st.last_loss = oort.uniform(0.01, 3.0)
            st.last_num_examples = oort.randint(1, 500)
        st.last_served_round = oort.randint(0, 5)
        st.last_clean_round = min(st.last_served_round, oort.randint(0, 5))
    # δ_obs (0 by default, spec) on some instances; drawn apart so the other
    # draws of an instance do not move.
    extra = random.Random(77 * seed + 5)
    offset = extra.choice((-6.0, -2.0, 3.0)) if extra.random() < 0.25 else 0.0
    return dict(remainder=remainder, state=state, budget_end=budget_end, model=model,
                protected=protected, states=states, mission_round=rng.choice((None, 1, 4)),
                snr_offset_db=offset)


ARMS = {
    "ours": dict(selector=None, miss_priority=False),
    "ours_prio": dict(selector=None, miss_priority=True),
    "ours_trim": dict(selector=None, miss_priority=False, replan_fallback=FALLBACK_TRIM),
    "d1_max_aoi": dict(selector=MaxAoIPolicy, miss_priority=False),
    "d2_oort": dict(selector=OortPolicy, miss_priority=False),
    "d3_whittle": dict(selector=WhittlePolicy, miss_priority=False),
    "d5_fedcs_unit": dict(selector=lambda: FedCSDegradedPolicy(VALUE_UNIT), miss_priority=False),
    "d5_fedcs_devices": dict(selector=lambda: FedCSDegradedPolicy(VALUE_DEVICES),
                             miss_priority=False),
    "d4_fedex": dict(selector=FedExCarpPolicy, miss_priority=False),
}
#: Our arms that re-order when their own order does not fit (``reorder``).
REORDER_ARMS = ("ours", "ours_prio")
#: The whole-scheduler baselines that re-plan with their own admit_and_order.
BASELINE_ARMS = ("d1_max_aoi", "d2_oort", "d3_whittle", "d5_fedcs_unit", "d5_fedcs_devices")


def scheduler_for(case, arm) -> FLScheduler:
    spec = ARMS[arm]
    sch = FLScheduler(
        mission_budget_s=100.0,
        feasibility_model=case["model"],
        target_selector=None if spec["selector"] is None else spec["selector"](),
        miss_priority=spec["miss_priority"],
        replan_fallback=spec.get("replan_fallback", FALLBACK_REORDER),
    )
    sch.device_states.update(case["states"])
    sch.set_mission_round(case["mission_round"])
    return sch


def _check_invariants(res: ReplanResult, case, sch, pass_kind, label, *, two_opt_arm,
                      trim_arm=False):
    remainder, state = case["remainder"], case["state"]
    budget_end, protected = case["budget_end"], case["protected"]
    offset = case["snr_offset_db"]
    rule = sch.in_flight_rule(pass_kind)
    ids = [id(wp) for wp in remainder]
    route_ids = [id(wp) for wp in res.route]
    dropped_ids = [id(wp) for wp, _ in res.dropped]
    # A subset of the remainder, each stop once; the rest dropped with a
    # reason, listed in the remainder's (old) order.
    assert len(set(route_ids)) == len(route_ids), label
    assert set(route_ids) <= set(ids), label
    assert sorted(route_ids + dropped_ids) == sorted(ids), label
    assert dropped_ids == [i for i in ids if i in set(dropped_ids)], label
    assert all(reason in REASONS for _, reason in res.dropped), label
    assert res.order_used in ORDERS, label
    # The re-plan never admits a stop the predicate rejects.
    model = sch.feasibility_model or FeasibilityModel()

    def fold(route):
        return model.fold(route, state, rule=rule, budget_end=budget_end,
                          pass_kind=pass_kind, skip=False, protected=tuple(protected),
                          snr_offset_db=offset)

    assert fold(res.route).ok, (label, fold(res.route).rejected)
    # The whole-remainder check the mule runs first agrees with the re-plan.
    check = sch.fold_remainder(remainder, state=state, budget_end=budget_end,
                               pass_kind=pass_kind, protected=protected,
                               snr_offset_db=offset)
    assert check.ok == (res.order_used in (ORDER_CURRENT, ORDER_NONE)), label
    if res.order_used in (ORDER_CURRENT, ORDER_NONE):
        assert list(res.route) == remainder and not res.dropped, label
    if budget_end is None:
        assert res.order_used in (ORDER_CURRENT, ORDER_NONE), label
    # A protected stop is dropped only if the protected-only route fails.
    dropped_protected = [wp for wp, _ in res.dropped if any(wp is p for p in protected)]
    if dropped_protected:
        only = [wp for wp in remainder if any(wp is p for p in protected)]
        assert not fold(only).ok, label
    # Order preference (critic C3): the arm's own relative order whenever it
    # passes; 2-OPT only once that failed; the admission order only once both
    # failed (for the arms whose admission prices with the same predicate, so
    # the admitted set is exactly the route).
    kept = {id(wp) for wp in res.route}
    arm_order = [wp for wp in remainder if id(wp) in kept]
    if res.order_used == ORDER_ARM:
        assert route_ids == [id(wp) for wp in arm_order], label
    if res.order_used == ORDER_TWO_OPT:
        assert two_opt_arm or pass_kind is DELIVER, label
        assert not fold(arm_order).ok, label
    if res.order_used == ORDER_ADMISSION and (two_opt_arm or pass_kind is DELIVER):
        assert not fold(arm_order).ok, label
        if len(res.route) > 1:
            dock = None if getattr(model, "ferry", None) is None else model.ferry.dock
            assert not fold(order_contacts(res.route, state.pose, end=dock)).ok, label
    # ``trim`` never re-orders Pass 1: the arm's relative order, the admitted
    # protected stops in front (as the admission flies them), the rest after.
    if res.order_used == ORDER_ARM_TRIMMED:
        assert trim_arm and pass_kind is COLLECT, label
        guarded = [wp for wp in res.route if any(wp is p for p in protected)]
        others = [wp for wp in res.route if not any(wp is p for p in protected)]
        assert route_ids == [id(wp) for wp in guarded + others], label
        for part in (guarded, others):
            part_ids = {id(wp) for wp in part}
            assert [id(wp) for wp in part] == [i for i in ids if i in part_ids], label
    if trim_arm and pass_kind is COLLECT:
        assert res.order_used in (ORDER_CURRENT, ORDER_ARM, ORDER_ARM_TRIMMED), label


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_replan_invariants_on_random_instances(arm):
    orders = set()
    offsets = set()
    trim_arm = ARMS[arm].get("replan_fallback") == FALLBACK_TRIM
    for seed in range(N_CASES):
        case = make_case(seed)
        sch = scheduler_for(case, arm)
        for pass_kind in (COLLECT, DELIVER):
            label = f"{arm} seed={seed} pass={pass_kind.value}"
            kw = dict(state=case["state"], budget_end=case["budget_end"], pass_kind=pass_kind,
                      protected=case["protected"], snr_offset_db=case["snr_offset_db"])
            res = sch.replan_remainder(case["remainder"], **kw)
            _check_invariants(res, case, sch, pass_kind, label,
                              two_opt_arm=arm in REORDER_ARMS, trim_arm=trim_arm)
            orders.add((pass_kind, res.order_used))
            if res.changed:
                offsets.add(case["snr_offset_db"])
            # Deterministic: the same call gives the same answer.
            again = sch.replan_remainder(case["remainder"], **kw)
            assert [id(w) for w in again.route] == [id(w) for w in res.route], label
            assert again.order_used == res.order_used, label
            if pass_kind is COLLECT and arm == "d4_fedex":
                assert res.order_used == ORDER_NONE, label
            if pass_kind is COLLECT and arm in BASELINE_ARMS:
                assert res.order_used not in (ORDER_TWO_OPT, ORDER_ARM_TRIMMED), label
    # The instances reach the interesting branches, δ_obs != 0 included.
    if arm != "d4_fedex":
        assert (COLLECT, ORDER_CURRENT) in orders
        assert (COLLECT, ORDER_ARM) in orders
        assert (DELIVER, ORDER_ARM) in orders
        assert len(offsets) > 1
    if arm in REORDER_ARMS:
        assert (COLLECT, ORDER_TWO_OPT) in orders
        assert (COLLECT, ORDER_ADMISSION) in orders
    if trim_arm:
        assert (COLLECT, ORDER_ARM_TRIMMED) in orders
        # Pass 2 is every arm's nearest-first walk, re-ordered as before.
        assert (DELIVER, ORDER_TWO_OPT) in orders


def test_the_random_instances_drop_for_every_reason():
    """Guards the generator: overdue, budget and energy drops all occur."""
    seen = set()
    for seed in range(N_CASES):
        case = make_case(seed)
        res = scheduler_for(case, "ours").replan_remainder(
            case["remainder"], state=case["state"], budget_end=case["budget_end"],
            pass_kind=COLLECT, protected=case["protected"])
        seen.update(reason for _, reason in res.dropped)
    assert seen == set(REASONS)


def test_the_replan_never_admits_a_stop_the_predicate_rejects_at_its_departure():
    """The invariant stated stop by stop, as the mule meets it in flight: at
    every departure of the returned route, ``admit`` from the state the mule
    is then in accepts the next stop (design §3.4)."""
    checked = 0
    for seed in range(N_CASES):
        case = make_case(seed)
        for arm in ("ours", "ours_trim", "d1_max_aoi", "d2_oort", "d5_fedcs_unit"):
            sch = scheduler_for(case, arm)
            for pass_kind in (COLLECT, DELIVER):
                res = sch.replan_remainder(
                    case["remainder"], state=case["state"], budget_end=case["budget_end"],
                    pass_kind=pass_kind, protected=case["protected"],
                    snr_offset_db=case["snr_offset_db"])
                model = sch.feasibility_model or FeasibilityModel()
                st = case["state"]
                for wp in res.route:
                    v = model.admit(st, wp, rule=sch.in_flight_rule(pass_kind),
                                    budget_end=case["budget_end"], pass_kind=pass_kind,
                                    protected=any(wp is p for p in case["protected"]),
                                    snr_offset_db=case["snr_offset_db"])
                    assert v.ok, (seed, arm, pass_kind, v)
                    st = v.next_state
                    checked += 1
    assert checked > 10_000


# --------------------------------------------------------------------------- #
# Deterministic branches
# --------------------------------------------------------------------------- #

def _wp(x, y, did, deadline=1e12, bucket=Bucket.SCHEDULED_THIS_ROUND):
    return ContactWaypoint(position=(float(x), float(y), 0.0), devices=(DeviceID(did),),
                           bucket=bucket, deadline_ts=float(deadline))


def _no_dwell():
    return 0.0


def line_model(**kw):
    """1 m/s; no dwell wherever a member is (unless ``member_dwell_s`` is
    given); no upload; dock at 0.

    The dwell does not depend on member positions here, so the physics is
    given its own permissive map instead of the scheduler's device states.
    """
    kw.setdefault("upload_s", _no_dwell)
    kw.setdefault("device_states", defaultdict(lambda: DOCK))
    kw.setdefault("member_dwell_s", lambda d, p, o: 0.0)
    phys = FerryPhysics(dock=DOCK, p_move_w=1.0, p_hover_w=1.0, **kw)
    return FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0, ferry=phys)


def _sched(model=None, selector=None, **kw):
    return FLScheduler(feasibility_model=model or line_model(), target_selector=selector,
                       mission_budget_s=100.0, **kw)


def test_a_passing_order_is_kept():
    A, B = _wp(10, 0, "a"), _wp(-10, 0, "b")
    res = _sched().replan_remainder([B, A], state=FlightState(DOCK, 0.0), budget_end=1e9)
    assert res.order_used == ORDER_CURRENT and res.route == (B, A) and not res.dropped
    assert not res.changed


def test_the_arm_order_is_kept_over_a_shorter_2opt_route():
    """Critic C3: once the overdue stop is dropped, the arm's own order of the
    rest is flown although 2-OPT would fly it shorter."""
    A, B = _wp(-50, 0, "a"), _wp(10, 0, "b")
    X = _wp(0, 60, "x", deadline=5.0)          # 10 s away at best: overdue in any order
    pose = (0.0, 50.0, 0.0)
    res = _sched().replan_remainder([B, X, A], state=FlightState(pose, 0.0), budget_end=1e9)
    assert res.order_used == ORDER_ARM
    assert res.route == (B, A)
    assert res.dropped == ((X, REASON_OVERDUE),)
    # 2-OPT would have flown A first: 70.7 + 60 + 10 m against 51 + 60 + 50 m.
    m = line_model()
    arm_len = m.fold([B, A], FlightState(pose, 0.0), rule=RULE_BUDGET, budget_end=None,
                     skip=False).home
    two_opt_len = m.fold([A, B], FlightState(pose, 0.0), rule=RULE_BUDGET, budget_end=None,
                         skip=False).home
    assert two_opt_len < arm_len


def test_2opt_is_the_fallback_when_the_arm_order_does_not_fit():
    """The zigzag A, B, C costs 80 s home; the 2-OPT tour costs 60 s."""
    A, B, C = _wp(10, 0, "a"), _wp(-10, 0, "b"), _wp(20, 0, "c")
    res = _sched().replan_remainder([A, B, C], state=FlightState(DOCK, 0.0), budget_end=60.0)
    assert res.order_used == ORDER_TWO_OPT and not res.dropped
    assert set(map(id, res.route)) == {id(A), id(B), id(C)}
    assert line_model().fold(res.route, FlightState(DOCK, 0.0), rule=RULE_DEADLINE_BUDGET,
                             budget_end=60.0, skip=False).ok


def test_the_admission_order_is_the_last_resort():
    """P first (the flown order, and 2-OPT's canonical orientation) makes Q
    late; S3b's EDF order, Q first, fits."""
    P = _wp(-10, 0, "p")
    Q = _wp(100, 0, "q", deadline=100.0)
    res = _sched().replan_remainder([P, Q], state=FlightState(DOCK, 0.0), budget_end=1000.0)
    assert res.order_used == ORDER_ADMISSION
    assert res.route == (Q, P) and not res.dropped


def test_protected_stops_go_first_and_are_dropped_only_if_they_cannot_fit():
    """P1 is overdue but protected (Phase 4's cap): kept. B no longer fits."""
    P1 = _wp(10, 0, "p1", deadline=-1.0)
    A, B = _wp(-10, 0, "a"), _wp(0, 30, "b")
    s0 = FlightState(DOCK, 0.0)
    res = _sched().replan_remainder([P1, A, B], state=s0, budget_end=45.0, protected={P1})
    assert res.order_used == ORDER_ARM
    assert res.route == (P1, A) and res.dropped == ((B, REASON_BUDGET),)
    # Unprotected, the overdue stop is dropped instead.
    res = _sched().replan_remainder([P1, A, B], state=s0, budget_end=45.0)
    assert (P1, REASON_OVERDUE) in res.dropped and P1 not in res.route
    # A protected stop that cannot fit on its own (home at 20 s > 15 s) goes.
    res = _sched().replan_remainder([P1, A], state=s0, budget_end=15.0, protected=[P1])
    assert res.dropped[0] == (P1, REASON_BUDGET)


def test_no_budget_no_gate_even_in_ferry_mode():
    hopeless = [_wp(1e6, 0, "far", deadline=-1e9), _wp(-5, 0, "near", deadline=-1e9)]
    res = _sched().replan_remainder(hopeless, state=FlightState(DOCK, 0.0), budget_end=None)
    assert res.order_used == ORDER_CURRENT and list(res.route) == hopeless


def test_fedex_flies_on():
    far = _wp(1e6, 0, "far")
    res = _sched(selector=FedExCarpPolicy()).replan_remainder(
        [far], state=FlightState(DOCK, 0.0), budget_end=1.0)
    assert res.order_used == ORDER_NONE and res.route == (far,) and not res.dropped


def test_a_baseline_readmits_with_its_own_rule_and_keeps_its_order():
    """D1 has no deadline clause (Amendment 8) and no 2-OPT: the zigzag that
    2-OPT would repair is trimmed by MAX-AoI's own walk instead."""
    A, B, C = _wp(10, 0, "a"), _wp(-10, 0, "b"), _wp(20, 0, "c", deadline=-5.0)
    sch = _sched(selector=MaxAoIPolicy())
    for did, age in (("a", 50.0), ("b", 40.0), ("c", 30.0)):
        sch.device_states[DeviceID(did)] = DeviceSchedulerState(
            device_id=DeviceID(did), last_clean_ts=100.0 - age)
    res = sch.replan_remainder([A, B, C], state=FlightState(DOCK, 100.0), budget_end=160.0)
    assert res.order_used in (ORDER_ARM, ORDER_ADMISSION)
    assert res.order_used != ORDER_TWO_OPT
    # Oldest first: A (home 20 s), B (home 40 s); C would bring it to 80 s.
    assert res.route == (A, B) and res.dropped == ((C, REASON_BUDGET),)


def test_a_baseline_keeps_its_own_order_unless_2opt_is_asked_for():
    """The flown zigzag A, B, C no longer fits; MAX-AoI re-admits all three
    in its age order A, C, B, which fits. The baseline flies that order;
    2-OPT only on request. Our arm can be denied 2-OPT the same way."""
    A, B, C = _wp(10, 0, "a"), _wp(-10, 0, "b"), _wp(20, 0, "c")
    sch = _sched(selector=MaxAoIPolicy())
    for did, age in (("a", 50.0), ("c", 40.0), ("b", 30.0)):
        sch.device_states[DeviceID(did)] = DeviceSchedulerState(
            device_id=DeviceID(did), last_clean_ts=100.0 - age)
    s = FlightState(DOCK, 100.0)
    res = sch.replan_remainder([A, B, C], state=s, budget_end=160.0)
    assert res.order_used == ORDER_ADMISSION and res.route == (A, C, B)
    forced = sch.replan_remainder([A, B, C], state=s, budget_end=160.0, two_opt_fallback=True)
    assert forced.order_used == ORDER_TWO_OPT and not forced.dropped
    ours = _sched().replan_remainder([A, B, C], state=FlightState(DOCK, 0.0),
                                     budget_end=60.0, two_opt_fallback=False)
    assert ours.order_used == ORDER_ADMISSION and ours.route == (B, A, C)


def _aged(sch, **ages):
    """MAX-AoI ages at clock 100: last CLEAN ``age`` seconds before."""
    for did, age in ages.items():
        sch.device_states[DeviceID(did)] = DeviceSchedulerState(
            device_id=DeviceID(did), last_clean_ts=100.0 - age)
    return sch


def test_a_baseline_readmits_from_the_energy_already_spent():
    """Critic B10: the baseline's own walk sees the remaining capacity, so it
    admits what still fits instead of an older stop it can no longer afford.

    25 J at 1 W, dwell free: A (older) is 20 J there and back, B 12 J. With
    10 J already spent MAX-AoI's walk skips A (30 J) and flies B (22 J). A
    walk that forgot the 10 J would pick A, which the energy clause then
    refuses from the real state, and lose B although B fits on its own.
    """
    A, B = _wp(10, 0, "a"), _wp(-6, 0, "b")
    sch = _aged(_sched(model=line_model(energy_capacity_j=25.0), selector=MaxAoIPolicy()),
                a=50.0, b=40.0)
    fresh = sch.replan_remainder([A, B], state=FlightState(DOCK, 100.0), budget_end=1e9)
    assert fresh.route == (A,) and [wp for wp, _ in fresh.dropped] == [B]
    tired = sch.replan_remainder([A, B], state=FlightState(DOCK, 100.0, 10.0), budget_end=1e9)
    assert tired.order_used == ORDER_ARM
    assert tired.route == (B,) and tired.dropped == ((A, REASON_ENERGY),)


def test_pass_2_is_a_nearest_first_budget_walk_for_every_arm():
    near, far = _wp(10, 0, "near", deadline=-1.0), _wp(-100, 0, "far")
    for selector in (None, MaxAoIPolicy(), FedExCarpPolicy()):
        res = _sched(selector=selector).replan_remainder(
            [far, near], state=FlightState(DOCK, 0.0), budget_end=30.0, pass_kind=DELIVER)
        # Overdue does not matter in Pass 2; the far stop does not fit.
        assert res.route == (near,) and res.dropped == ((far, REASON_BUDGET),)


def test_pass_2_readmits_nearest_first_not_in_queue_order():
    """A at +20 m, B at -10 m, 45 s: A, B as queued no longer fits (home at
    60 s). Nearest first admits B (home at 20 s) and then A from B would be
    home at 60 s, so A goes; the queue's own order would have kept A instead."""
    A, B = _wp(20, 0, "a"), _wp(-10, 0, "b")
    for selector in (None, MaxAoIPolicy(), FedExCarpPolicy()):
        res = _sched(selector=selector).replan_remainder(
            [A, B], state=FlightState(DOCK, 0.0), budget_end=45.0, pass_kind=DELIVER)
        assert res.route == (B,) and res.dropped == ((A, REASON_BUDGET),)


def test_our_readmission_ranks_by_the_miss_priority_when_it_is_on():
    """Only one of E and M fits 25 s (each is home at 20 s, both at 40 s).
    S3b's EDF admits E, the earlier deadline; with the miss priority M, missed
    three times running, goes first."""
    E, M = _wp(10, 0, "e", deadline=500.0), _wp(-10, 0, "m", deadline=900.0)
    for prio, kept, lost in ((False, E, M), (True, M, E)):
        sch = _sched(miss_priority=prio)
        for did, streak in (("e", 0), ("m", 3)):
            sch.device_states[DeviceID(did)] = DeviceSchedulerState(
                device_id=DeviceID(did), miss_streak=streak)
        res = sch.replan_remainder([E, M], state=FlightState(DOCK, 0.0), budget_end=25.0)
        assert res.route == (kept,) and res.dropped == ((lost, REASON_BUDGET),), prio


def _db_model():
    """4 s of dwell per member at 0 dB; each dB of δ_obs saves 1 s."""
    return line_model(member_dwell_s=lambda d, p, o: max(0.0, 4.0 - o))


def test_our_readmission_prices_at_the_observed_rate():
    """δ_obs reaches S3b's re-admission. The flown order P, Q reaches Q after
    its deadline either way. At +3 dB (1 s dwells) EDF fits both in 32 s,
    Q first; at 0 dB (4 s dwells) P would be home at 38 s: dropped."""
    P, Q = _wp(10, 0, "p", deadline=100.0), _wp(-5, 0, "q", deadline=20.0)
    sch = _sched(model=_db_model())
    s0 = FlightState(DOCK, 0.0)
    at_3 = sch.replan_remainder([P, Q], state=s0, budget_end=32.0, snr_offset_db=3.0)
    assert {id(wp) for wp in at_3.route} == {id(P), id(Q)} and not at_3.dropped
    at_0 = sch.replan_remainder([P, Q], state=s0, budget_end=32.0)
    assert at_0.route == (Q,) and at_0.dropped == ((P, REASON_BUDGET),)


def test_the_guard_fold_holds_a_baseline_to_the_observed_rate():
    """A baseline's own walk prices at 0 dB; the final fold that skips holds
    its route to δ_obs. At 0 dB (4 s dwells) MAX-AoI admits A then C in 30 s;
    at -6 dB (10 s dwells) C from A would be home at 36 s, so the guard drops
    it. B never fits. The drops are listed in the remainder's order (C, B),
    although the guard found C last."""
    A, C, B = _wp(5, 0, "a"), _wp(8, 0, "c"), _wp(-20, 0, "b")
    sch = _aged(_sched(model=_db_model(), selector=MaxAoIPolicy()), a=50.0, c=40.0, b=30.0)
    s = FlightState(DOCK, 100.0)
    res = sch.replan_remainder([C, B, A], state=s, budget_end=130.0, snr_offset_db=-6.0)
    assert res.order_used == ORDER_ADMISSION
    assert res.route == (A,)
    assert res.dropped == ((C, REASON_BUDGET), (B, REASON_BUDGET))
    # At 0 dB the policy's own order of the same two stops fits as flown.
    res0 = sch.replan_remainder([C, B, A], state=s, budget_end=130.0)
    assert res0.order_used == ORDER_ARM and res0.route == (C, A)
    assert res0.dropped == ((B, REASON_BUDGET),)


def test_drops_are_listed_in_the_remainders_order():
    """Protected drops are found first (the protected-only fold), the rest
    after; the result still lists them in the order they were queued."""
    A, P = _wp(-20, 0, "a"), _wp(20, 0, "p")
    res = _sched().replan_remainder([A, P], state=FlightState(DOCK, 0.0), budget_end=30.0,
                                    protected={P})
    assert res.route == () and res.dropped == ((A, REASON_BUDGET), (P, REASON_BUDGET))


def test_a_baseline_readmits_in_the_mission_being_planned():
    """Whittle (D3) ages devices in missions from ``SelectorEnv.mission_round``;
    a re-plan hands it the scheduler's round. In mission 7, b (last CLEAN in
    mission 1, no merge recorded) is six missions old and a (mission 6) one,
    so b is kept; without the round Whittle would infer mission 1, tie them,
    and keep a."""
    A, B = _wp(-10, 0, "a"), _wp(10, 0, "b")
    for rnd, kept, lost, x_b in ((7, B, A, 6), (None, A, B, 1)):
        policy = WhittlePolicy()
        sch = _sched(selector=policy)
        sch.device_states[DeviceID("a")] = DeviceSchedulerState(
            device_id=DeviceID("a"), last_clean_round=6)
        sch.device_states[DeviceID("b")] = DeviceSchedulerState(
            device_id=DeviceID("b"), last_clean_round=1)
        sch.set_mission_round(rnd)
        res = sch.replan_remainder([A, B], state=FlightState(DOCK, 0.0), budget_end=25.0)
        assert res.route == (kept,) and res.dropped == ((lost, REASON_BUDGET),), rnd
        assert policy.last_device_inputs[DeviceID("b")][0] == x_b


def test_in_flight_rule_per_arm():
    assert _sched().in_flight_rule() == RULE_DEADLINE_BUDGET
    for policy in (MaxAoIPolicy(), WhittlePolicy(), FedCSDegradedPolicy()):
        assert _sched(selector=policy).in_flight_rule(COLLECT) == RULE_BUDGET
    assert _sched(selector=FedExCarpPolicy()).in_flight_rule(COLLECT) == RULE_NONE
    for selector in (None, MaxAoIPolicy(), FedExCarpPolicy()):
        assert _sched(selector=selector).in_flight_rule(DELIVER) == RULE_BUDGET
    assert _sched().in_flight_rule("deliver") == RULE_BUDGET


def test_fold_remainder_holds_the_whole_remainder_to_the_arms_rule():
    """The whole-remainder check at a departure (design §3.4): every stop, in
    the order it would be flown, under the arm's in-flight rule."""
    A = _wp(10, 0, "a")                         # finish 10 s, home at 20 s
    L = _wp(-10, 0, "late", deadline=5.0)       # finish 30 s (20 m on), home at 40 s
    s0 = FlightState(DOCK, 0.0)
    ours = _sched().fold_remainder([A, L], state=s0, budget_end=40.0)
    assert not ours.ok and ours.rejected == ((L, REASON_OVERDUE),)
    assert ours.route == (A, L) and ours.home == 40.0            # flown anyway
    # D1: the budget alone (Amendment 8), so the late stop fits.
    d1 = _sched(selector=MaxAoIPolicy()).fold_remainder([A, L], state=s0, budget_end=40.0)
    assert d1.ok and d1.home == 40.0
    assert not _sched(selector=MaxAoIPolicy()).fold_remainder(
        [A, L], state=s0, budget_end=39.0).ok
    # D4 never rejects but reports when it is home: its overrun.
    d4 = _sched(selector=FedExCarpPolicy()).fold_remainder([A, L], state=s0, budget_end=1.0)
    assert d4.ok and d4.home == 40.0
    # Pass 2 is the budget alone for every arm; no budget, no gate.
    assert _sched().fold_remainder([A, L], state=s0, budget_end=40.0, pass_kind=DELIVER).ok
    assert _sched().fold_remainder([A, L], state=s0, budget_end=None).ok
    # A protected stop skips the deadline clause here too; the energy spent
    # so far counts: 10 J spent, A needs 20 J there and back (30 J), then L
    # needs 30 J from A with 20 J spent (50 J > 45 J).
    assert _sched().fold_remainder([A, L], state=s0, budget_end=40.0, protected=[L]).ok
    tired = _sched(model=line_model(energy_capacity_j=45.0)).fold_remainder(
        [A, L], state=FlightState(DOCK, 0.0, 10.0), budget_end=1e9, protected=[L])
    assert tired.rejected == ((L, REASON_ENERGY),)


def test_the_replan_works_on_the_legacy_model_too():
    m = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    A, B, C = _wp(10, 0, "a"), _wp(-10, 0, "b"), _wp(20, 0, "c")
    res = FLScheduler(feasibility_model=m, mission_budget_s=1.0).replan_remainder(
        [A, B, C], state=FlightState(DOCK, 0.0), budget_end=40.0)
    # No return leg: A, B, C costs 10 + 20 + 30 = 60 s, the tour from the dock 40 s.
    assert res.order_used == ORDER_TWO_OPT and len(res.route) == 3


# --------------------------------------------------------------------------- #
# The ``trim`` fallback: the arm's order kept, what it cannot serve dropped
# --------------------------------------------------------------------------- #

def test_trim_keeps_the_arms_order_where_reorder_would_repair_it():
    """The zigzag A, B, C (home at 80 s) against 60 s: ``reorder`` flies the
    2-OPT path and keeps all three; ``trim`` keeps the zigzag and drops C,
    the stop it cannot serve in that order."""
    A, B, C = _wp(10, 0, "a"), _wp(-10, 0, "b"), _wp(20, 0, "c")
    s0 = FlightState(DOCK, 0.0)
    reorder = _sched().replan_remainder([A, B, C], state=s0, budget_end=60.0)
    assert reorder.order_used == ORDER_TWO_OPT and not reorder.dropped
    trim = _sched(replan_fallback=FALLBACK_TRIM).replan_remainder(
        [A, B, C], state=s0, budget_end=60.0)
    assert trim.order_used == ORDER_ARM_TRIMMED and trim.changed
    assert trim.route == (A, B) and trim.dropped == ((C, REASON_BUDGET),)
    # A passing arm order is still just kept; so is a passing remainder.
    assert _sched(replan_fallback=FALLBACK_TRIM).replan_remainder(
        [B, A, C], state=s0, budget_end=60.0).order_used == ORDER_CURRENT


def test_trim_keeps_admitted_protected_stops_in_front():
    """B, P, A with A due at 20 s: A arrives at 25 s in that order. ``trim``
    flies P (protected, admitted first) and then B, A in their own order from
    there, which drops A; a protected stop is never traded for another."""
    P = _wp(5, 0, "p", deadline=-1.0)
    A, B = _wp(15, 0, "a", deadline=20.0), _wp(-5, 0, "b")
    res = _sched(replan_fallback=FALLBACK_TRIM).replan_remainder(
        [B, P, A], state=FlightState(DOCK, 0.0), budget_end=100.0, protected={P})
    assert res.order_used == ORDER_ARM_TRIMMED
    assert res.route == (P, B) and res.dropped == ((A, REASON_OVERDUE),)


def test_trim_applies_to_our_arms_pass_1_only():
    """Pass 2 is every arm's nearest-first walk, and a baseline's admission
    order is its own policy's order, so neither changes under ``trim``."""
    A, B, C = _wp(10, 0, "a"), _wp(-10, 0, "b"), _wp(20, 0, "c")
    s0 = FlightState(DOCK, 0.0)
    for selector in (None, MaxAoIPolicy()):
        kw = dict(state=s0, budget_end=60.0, pass_kind=DELIVER)
        base = _sched(selector=selector).replan_remainder([A, B, C], **kw)
        trim = _sched(selector=selector, replan_fallback=FALLBACK_TRIM).replan_remainder(
            [A, B, C], **kw)
        assert (trim.route, trim.dropped, trim.order_used) == (
            base.route, base.dropped, base.order_used)
    sch = _aged(_sched(selector=MaxAoIPolicy()), a=50.0, c=40.0, b=30.0)
    trim_sch = _aged(_sched(selector=MaxAoIPolicy(), replan_fallback=FALLBACK_TRIM),
                     a=50.0, c=40.0, b=30.0)
    s = FlightState(DOCK, 100.0)
    assert (trim_sch.replan_remainder([A, B, C], state=s, budget_end=160.0).route
            == sch.replan_remainder([A, B, C], state=s, budget_end=160.0).route == (A, C, B))


def test_the_fallback_is_validated():
    assert FALLBACKS == (FALLBACK_REORDER, FALLBACK_TRIM)
    assert FLScheduler().replan_fallback == FALLBACK_REORDER
    with pytest.raises(FLSchedulerError, match="replan_fallback"):
        FLScheduler(replan_fallback="two_opt")
    with pytest.raises(ValueError, match="fallback"):
        replan_route([], state=FlightState(DOCK, 0.0), model=line_model(), rule=RULE_BUDGET,
                     budget_end=None, pass_kind=COLLECT, admission=lambda c, s: (c, ()),
                     fallback="skip")


# --------------------------------------------------------------------------- #
# The pre-flight order check in build_contact_queue (design §3.3)
# --------------------------------------------------------------------------- #

def _one_s_dwell(d, pass_kind, offset):
    return 1.0


def _ferry_5ms():
    return FeasibilityModel(ferry=FerryPhysics(
        dock=DOCK, member_dwell_s=_one_s_dwell, upload_s=_no_dwell,
        p_move_w=143.6, p_hover_w=168.5))


def _planner(*, validate=True, budget=100.0, model="ferry", **kw):
    """Two new devices: ``near`` 10 m out (Φ 60 s), ``far`` 115 m the other way
    (Φ 25 s). H1 flies the nearer one first and reaches ``far`` 4 s late; S3b's
    EDF order serves ``far`` first and fits."""
    sch = FLScheduler(
        now_fn=lambda: 1000.0, mission_budget_s=budget,
        feasibility_model=_ferry_5ms() if model == "ferry" else model,
        validate_flown_order=validate, **kw,
    )
    near, far = DeviceID("near"), DeviceID("far")
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=(near, far),
                                  issued_round=0, issued_at=0.0))
    sch.device_states[near].last_known_position = (10.0, 0.0, 0.0)
    sch.device_states[far].last_known_position = (-115.0, 0.0, 0.0)
    sch.device_states[far].deadline_fulfilment_s = 25.0
    sch.start_mission()
    return sch


def _devices(queue):
    return [str(d) for wp in queue for d in wp.devices]


def test_the_flown_order_is_validated_before_takeoff():
    sch = _planner()
    queue = sch.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)
    assert _devices(queue) == ["far", "near"]
    check = sch.last_order_check
    assert check is not None and check.order_used == ORDER_TWO_OPT and not check.dropped
    assert sch.last_feasibility.kept == queue and sch.last_feasibility.n_dropped == 0


def test_without_the_flag_the_arm_flies_its_own_order():
    sch = _planner(validate=False)
    queue = sch.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)
    assert _devices(queue) == ["near", "far"]
    assert sch.last_order_check is None


def test_no_budget_no_gate_and_no_check_even_in_ferry_mode():
    ferry = _planner(budget=None)
    legacy = _planner(budget=None, validate=False, model=None)
    q_ferry = ferry.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)
    q_legacy = legacy.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)
    assert _devices(q_ferry) == _devices(q_legacy) == ["near", "far"]
    assert ferry.last_feasibility is None and ferry.last_order_check is None


def test_a_legacy_model_is_never_validated():
    sch = _planner(model=FeasibilityModel())
    queue = sch.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)
    assert sch.last_order_check is None
    assert _devices(queue) == ["near", "far"]


def test_validation_drops_join_last_feasibility_by_reason(monkeypatch):
    """The mule widens every pre-flight drop, so the check's drops must land in
    ``last_feasibility`` under their reason, energy included (critic B10)."""
    sch = _planner()
    real = sch.replan_remainder

    def trimming(queue, **kw):
        res = real(queue, **kw)
        far, near = res.route
        return ReplanResult((far,), ((near, REASON_ENERGY),), ORDER_ADMISSION)

    monkeypatch.setattr(sch, "replan_remainder", trimming)
    queue = sch.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)
    assert _devices(queue) == ["far"]
    feas = sch.last_feasibility
    assert _devices(feas.dropped_energy) == ["near"]
    assert feas.n_dropped == 1 and _devices(feas.dropped) == ["near"]
    assert feas.kept == queue


def test_trim_before_takeoff_keeps_the_flown_order_and_widens_its_drops():
    """Under ``trim`` H1 keeps its own order: ``near`` first, and ``far``,
    which that order reaches late, is dropped and joins ``last_feasibility``
    as overdue, so the mule's pre-flight widening covers it."""
    sch = _planner(replan_fallback=FALLBACK_TRIM)
    queue = sch.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)
    assert _devices(queue) == ["near"]
    check = sch.last_order_check
    assert check.order_used == ORDER_ARM_TRIMMED
    assert [(_devices([wp]), why) for wp, why in check.dropped] == [(["far"], REASON_OVERDUE)]
    feas = sch.last_feasibility
    assert _devices(feas.dropped_overdue) == ["far"] and feas.n_dropped == 1
    assert feas.kept == queue


class _Farthest:
    """A non-delegating selector stand-in with its own order (farthest from
    the mule first), as a learned H2 would have: ``rank_contacts`` only."""

    def rank_contacts(self, members, states, env, pass_kind=COLLECT, admitted=None):
        return sorted(members, key=lambda wp: -math.dist(env.mule_pose, wp.position))


def _ferry_upload():
    return FeasibilityModel(ferry=FerryPhysics(
        dock=DOCK, member_dwell_s=_one_s_dwell, upload_s=lambda: 3.0,
        p_move_w=143.6, p_hover_w=168.5))


def _layout_plan(layout, phis, budget, *, selector=None, validate=True,
                 fallback=FALLBACK_REORDER, now=1e6):
    sch = FLScheduler(now_fn=lambda: now, mission_budget_s=budget,
                      feasibility_model=_ferry_upload(), validate_flown_order=validate,
                      target_selector=selector, replan_fallback=fallback)
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=tuple(layout),
                                  issued_round=1, issued_at=now))
    for did, pos in layout.items():
        sch.device_states[did].last_known_position = pos
        sch.device_states[did].deadline_fulfilment_s = phis[did]
    sch.start_mission()
    return sch, sch.build_contact_queue(rf_range_m=30.0, mule_pose=DOCK)


def test_before_takeoff_reorder_depends_only_on_the_admitted_set():
    """What critic C3's order preference cannot do before takeoff, pinned.

    S3b has just admitted the whole queue from the same state and budget, so
    the re-plan re-admits every stop and the arm's own order over them is the
    flown order that failed. Under ``reorder`` the check then never drops a
    stop (``last_feasibility`` keeps S3b's drops alone), and the route is
    2-OPT's or S3b's EDF order: two arms that fly different orders over the
    same set (H1's distance order and a farthest-first stand-in for H2) fly
    the same route whenever both checks fire. Under ``trim`` each arm keeps
    its order and drops what that order cannot serve, and the drops join
    ``last_feasibility``.
    """
    rng = random.Random(3)
    fired = both = trimmed_drops = 0
    for _ in range(150):
        n = rng.randint(3, 7)
        layout = {DeviceID(f"d{i}"): (rng.uniform(-120, 120), rng.uniform(-120, 120), 0.0)
                  for i in range(n)}
        phis = {d: rng.uniform(15, 120) for d in layout}
        budget = rng.uniform(40, 200)
        routes = {}
        for name, selector in (("h1", None), ("farthest", _Farthest())):
            plain, flown = _layout_plan(layout, phis, budget, selector=selector, validate=False)
            sch, queue = _layout_plan(layout, phis, budget, selector=selector)
            check = sch.last_order_check
            if check.order_used == ORDER_CURRENT:
                assert queue == flown and sch.last_feasibility.dropped == plain.last_feasibility.dropped
                continue
            fired += 1
            assert check.order_used in (ORDER_TWO_OPT, ORDER_ADMISSION)
            assert not check.dropped
            assert {(w.position, w.devices) for w in queue} == {
                (w.position, w.devices) for w in flown}
            assert sch.last_feasibility.dropped == plain.last_feasibility.dropped
            routes[name] = [(w.position, w.devices) for w in queue]
            # ``trim``: the flown order, what it cannot serve left out.
            tsch, tqueue = _layout_plan(layout, phis, budget, selector=selector,
                                        fallback=FALLBACK_TRIM)
            start = tsch.mission_start_ts
            walk = tsch.feasibility_model.fold(
                flown, FlightState(DOCK, start), rule=RULE_DEADLINE_BUDGET,
                budget_end=start + budget, skip=True)
            assert tqueue == list(walk.route)
            assert tsch.last_order_check.order_used == ORDER_ARM_TRIMMED
            assert tsch.last_order_check.dropped == walk.rejected
            assert tsch.last_feasibility.n_dropped == (
                plain.last_feasibility.n_dropped + len(walk.rejected))
            trimmed_drops += len(walk.rejected)
        if len(routes) == 2:
            both += 1
            assert routes["h1"] == routes["farthest"]
    assert fired > 40 and both > 15 and trimmed_drops > 0


def test_the_check_measures_the_budget_from_the_mission_start():
    """Planned 10 s after the mission started, with 28 s of budget: 18 s left.
    Farthest first (c, b, a) is home 19.8 s after planning, S3b's EDF order
    15.8 s. The check must see the 18 s S3b saw, not 28 s from now."""
    layout = {DeviceID("a"): (10.0, 0.0, 0.0), DeviceID("b"): (-12.0, 0.0, 0.0),
              DeviceID("c"): (20.0, 0.0, 0.0)}
    phis = dict.fromkeys(layout, 60.0)

    def plan(validate):
        clock = {"t": 1000.0}
        sch = FLScheduler(now_fn=lambda: clock["t"], mission_budget_s=28.0,
                          feasibility_model=_ferry_5ms(), validate_flown_order=validate,
                          target_selector=_Farthest())
        sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=tuple(layout),
                                      issued_round=1, issued_at=1000.0))
        for did, pos in layout.items():
            sch.device_states[did].last_known_position = pos
            sch.device_states[did].deadline_fulfilment_s = phis[did]
        sch.start_mission()
        clock["t"] = 1010.0
        return sch, sch.build_contact_queue(rf_range_m=5.0, mule_pose=DOCK)

    _, flown = plan(False)
    assert _devices(flown) == ["c", "b", "a"]
    sch, queue = plan(True)
    assert sch.mission_start_ts == 1000.0
    assert sch.last_order_check.order_used == ORDER_TWO_OPT
    assert sch.feasibility_model.fold(queue, FlightState(DOCK, 1010.0), rule=RULE_DEADLINE_BUDGET,
                                      budget_end=1028.0, skip=False).ok
    assert not sch.feasibility_model.fold(flown, FlightState(DOCK, 1010.0),
                                          rule=RULE_DEADLINE_BUDGET, budget_end=1028.0,
                                          skip=False).ok


def test_the_gate_still_precedes_the_selector_in_source():
    import inspect

    src = inspect.getsource(FLScheduler.build_contact_queue)
    assert src.index("filter_feasible(") < src.index("self._target_selector.rank_contacts(")
