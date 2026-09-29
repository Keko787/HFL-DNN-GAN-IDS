"""FedCS, degraded to a mule (arm D5) — Nishio & Yonetani, ICC 2019, Algorithm 3.

What is pinned here, and why each matters for the comparison:

* the extraction's worked example (B), recomputed, under both selection values;
* skip and stop return the same route for the paper's own key (value 1), as
  the checker proved, while skip matters for the device-weighted key;
* the devices key is ``len(devices) / total`` (not per transit second, not a
  session per device), a free contact goes first, and both keys match an
  independently written skip loop on random and tie-heavy instances;
* the budget is priced with ``<=`` and the shared cost model, like every arm;
* no budget admits everything, in Algorithm 3's greedy order;
* ``wp.deadline_ts`` is never read: one round deadline, not per-device ones,
  is the whole point of the arm;
* the route is deterministic and nothing passed in is mutated.
"""

from __future__ import annotations

import copy
import math
import random
from typing import List, Optional, Sequence

import pytest

from hermes.scheduler.policies.budget_walk import IN_FLIGHT_BUDGET
from hermes.scheduler.policies.fedcs_degraded import (
    VALUE_DEVICES,
    VALUE_UNIT,
    FedCSDegradedPolicy,
    fedcs_greedy_select,
)
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.selector.scope_guard import SelectorScopeViolation
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import (
    Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState, MissionPass,
)

NOW = 1_000.0
#: The Exp 4 runtime defaults, which worked example (B) uses.
DEFAULT_MODEL = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0)


def _wp(pos, *devs: str, deadline_ts: float = NOW + 1e6,
        bucket: Bucket = Bucket.SCHEDULED_THIS_ROUND) -> ContactWaypoint:
    x, y = pos[0], pos[1]
    z = pos[2] if len(pos) > 2 else 0.0
    return ContactWaypoint(
        position=(float(x), float(y), float(z)),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=bucket,
        deadline_ts=deadline_ts,
    )


def _env(pose=(0.0, 0.0, 0.0), now: float = NOW) -> SelectorEnv:
    return SelectorEnv(mule_pose=pose, mule_energy=1.0, rf_prior_snr_db=20.0,
                       beacon_window_s=30.0, now=now)


def _states(contacts: Sequence[ContactWaypoint]):
    """Populated states, so a mutation of any field would show."""
    out = {}
    for k, wp in enumerate(contacts):
        for d in wp.devices:
            st = DeviceSchedulerState(device_id=d)
            st.last_clean_ts = NOW - 10.0 * (k + 1)
            st.last_contact_ts = NOW - 5.0 * (k + 1)
            st.last_clean_round = k
            st.last_served_round = k + 1
            st.last_num_examples = 100 + k
            st.last_loss = 0.5
            st.miss_streak = k % 3
            st.last_known_position = tuple(wp.position)
            st.bucket = wp.bucket
            out[d] = st
    return out


def _admit(contacts, *, value=VALUE_UNIT, budget: Optional[float],
           pose=(0.0, 0.0, 0.0), now: float = NOW, model=DEFAULT_MODEL):
    return FedCSDegradedPolicy(value=value).admit_and_order(
        contacts, _states(contacts), _env(pose, now),
        mission_deadline_ts=None if budget is None else now + budget,
        feasibility_model=model,
    )


def _names(route) -> List[str]:
    return [wp.devices[0] for wp in route]


def _random_instance(rng: random.Random, *, single_device: bool = False):
    n = rng.randint(1, 12)
    contacts = []
    for i in range(n):
        k = 1 if single_device else rng.randint(1, 4)
        contacts.append(_wp(
            (rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(0, 10)),
            *[f"c{i}d{j}" for j in range(k)],
            deadline_ts=rng.uniform(0.0, 2 * NOW),
        ))
    pose = (rng.uniform(-50, 50), rng.uniform(-50, 50), 0.0)
    now = rng.uniform(0.0, 5_000.0)
    budget = rng.uniform(0.0, 120.0)
    model = FeasibilityModel(cruise_speed_m_s=rng.uniform(1.0, 20.0),
                             session_time_s=rng.uniform(0.0, 5.0))
    return contacts, pose, now, budget, model


def _stop_reference(contacts, *, pose, now, deadline, model):
    """Independent 'stop at the first miss' reading of Algorithm 3, value 1.

    The checker's reading of the paper's prose ("until elapsed time t reaches
    T_round"). Written separately from the policy on purpose.
    """
    remaining = list(contacts)
    route = []
    clock = now
    while remaining:
        priced = [(model.cost(pose, c.position)[1], tuple(c.position),
                   tuple(c.devices), i) for i, c in enumerate(remaining)]
        total, _, _, i = min(priced)
        if clock + total > deadline:
            break
        x = remaining.pop(i)
        clock += total
        pose = x.position
        route.append(x)
    return route


def _skip_reference(contacts, *, value, pose, now, deadline, model):
    """Independent 'skip' reading of Algorithm 3 for either selection value.

    Written from the spec, separately from the policy: score every remaining
    contact from the end of the route (the smallest total for 'unit'; the
    largest devices per total second for 'devices', a free contact scoring
    infinity), keep the best-scoring ones, break the tie on the smallest
    (position, devices), drop the pick, and admit it only if it fits.
    """
    remaining = list(contacts)
    route = []
    clock = now
    while remaining:
        totals = [model.cost(pose, c.position)[1] for c in remaining]
        if value == VALUE_UNIT:
            scores = [-t for t in totals]
        else:
            scores = [len(c.devices) / t if t > 0.0 else math.inf
                      for c, t in zip(remaining, totals)]
        best = max(scores)
        tied = [i for i, s in enumerate(scores) if s == best]
        i = min(tied, key=lambda j: (tuple(remaining[j].position),
                                     tuple(remaining[j].devices)))
        x, total = remaining.pop(i), totals[i]
        if deadline is not None and clock + total > deadline:
            continue
        clock += total
        pose = x.position
        route.append(x)
    return route


def _tie_heavy_instance(rng: random.Random):
    """Grid positions (equal distances, shared positions, contacts at the mule
    pose), zero session time now and then, and sometimes no deadline, so the
    tie-break and the free-contact branch are exercised."""
    n = rng.randint(1, 10)
    contacts = []
    for i in range(n):
        contacts.append(_wp(
            (10.0 * rng.randint(-3, 3), 10.0 * rng.randint(-3, 3), 0.0),
            *[f"c{i}d{j}" for j in range(rng.randint(1, 4))],
        ))
    pose = (10.0 * rng.randint(-1, 1), 10.0 * rng.randint(-1, 1), 0.0)
    now = float(rng.randint(0, 5_000))
    budget = None if rng.random() < 0.2 else float(rng.randint(0, 60))
    model = FeasibilityModel(
        cruise_speed_m_s=float(rng.choice([1, 2, 5, 10])),
        session_time_s=0.0 if rng.random() < 0.3 else float(rng.randint(1, 3)))
    return contacts, pose, now, budget, model


def _route_cost(route, *, pose, model) -> float:
    spent = 0.0
    for wp in route:
        spent += model.cost(pose, wp.position)[1]
        pose = wp.position
    return spent


# --------------------------------------------------------------------------- #
# 1. Identity
# --------------------------------------------------------------------------- #

def test_policy_identity_and_in_flight_rule():
    p = FedCSDegradedPolicy()
    assert p.name == "FEDCS"
    assert p.value == VALUE_UNIT
    # One round deadline and no per-device deadline, so flight re-checks the
    # budget only.
    assert p.in_flight_check == IN_FLIGHT_BUDGET
    assert FedCSDegradedPolicy(value=VALUE_DEVICES).value == VALUE_DEVICES


def test_unknown_value_is_refused():
    with pytest.raises(ValueError):
        FedCSDegradedPolicy(value="utility")
    with pytest.raises(ValueError):
        fedcs_greedy_select([_wp((1, 0), "a")], value="utility",
                            mule_pose=(0.0, 0.0, 0.0), now=NOW,
                            mission_deadline_ts=None)


def test_no_contacts_gives_an_empty_route():
    assert _admit([], budget=10.0) == []
    assert _admit([], budget=None) == []


# --------------------------------------------------------------------------- #
# 2. Worked example (B), recomputed
# --------------------------------------------------------------------------- #
#
# Cruise 5 m/s, session 1 s, pose (0,0,0), budget 12 s.
# P (10,0,0) 1 device, Q (0,30,0) 3 devices, R (20,0,0) 1 device.

P = _wp((10, 0, 0), "p")
Q = _wp((0, 30, 0), "q1", "q2", "q3")
R = _wp((20, 0, 0), "r")


def test_worked_example_leg_costs():
    """The numbers the two routes below rest on, recomputed from the model."""
    o = (0.0, 0.0, 0.0)
    m = DEFAULT_MODEL
    assert m.cost(o, P.position)[1] == pytest.approx(3.0)
    assert m.cost(o, R.position)[1] == pytest.approx(5.0)
    assert m.cost(o, Q.position)[1] == pytest.approx(7.0)
    assert m.cost(P.position, R.position)[1] == pytest.approx(3.0)
    assert m.cost(P.position, Q.position)[1] == pytest.approx(
        math.sqrt(1000.0) / 5.0 + 1.0)                     # 7.32
    assert m.cost(R.position, Q.position)[1] == pytest.approx(
        math.sqrt(1300.0) / 5.0 + 1.0)                     # 8.21
    # Unit key: P (clock 3), R (clock 6), then Q would end at 14.21 > 12.
    assert 6.0 + m.cost(R.position, Q.position)[1] == pytest.approx(14.2111, abs=1e-4)
    # Devices key: Q scores 3/7 = 0.429 > P 1/3 > R 1/5; from Q (clock 7),
    # P ends at 14.32 and R at 15.21, both over 12.
    assert 7.0 + m.cost(Q.position, P.position)[1] == pytest.approx(14.3246, abs=1e-4)
    assert 7.0 + m.cost(Q.position, R.position)[1] == pytest.approx(15.2111, abs=1e-4)


@pytest.mark.parametrize("model", [DEFAULT_MODEL, None],
                         ids=["explicit-model", "default-model"])
def test_worked_example_unit_key_routes_p_then_r(model):
    route = _admit([P, Q, R], value=VALUE_UNIT, budget=12.0, model=model)
    assert route == [P, R]
    assert sum(len(w.devices) for w in route) == 2


@pytest.mark.parametrize("model", [DEFAULT_MODEL, None],
                         ids=["explicit-model", "default-model"])
def test_worked_example_devices_key_routes_q_only(model):
    route = _admit([P, Q, R], value=VALUE_DEVICES, budget=12.0, model=model)
    assert route == [Q]
    assert sum(len(w.devices) for w in route) == 3


# --------------------------------------------------------------------------- #
# 3. Admission: <=, skip-not-stop, and the budget holds
# --------------------------------------------------------------------------- #

def test_admission_boundary_is_inclusive():
    """55 m at 5 m/s + 1 s = 12 s exactly. The paper's line 7 would reject
    (strict <); we admit, like greedy_budget_walk and S3b."""
    wp = _wp((55, 0, 0), "a")
    assert _admit([wp], budget=12.0) == [wp]
    assert _admit([wp], budget=12.0 - 1e-9) == []


def test_zero_budget_admits_nothing_that_costs_time():
    assert _admit([_wp((1, 0, 0), "a")], budget=0.0) == []


def test_skip_and_stop_agree_for_the_unit_key_on_random_instances():
    """Checker correction 4: with value 1 the pick is the cheapest remaining
    contact, so once it fails every other one fails too."""
    rng = random.Random(20190520)
    trimmed = admitted_some = 0
    for _ in range(400):
        contacts, pose, now, budget, model = _random_instance(rng)
        got = _admit(contacts, value=VALUE_UNIT, budget=budget, pose=pose,
                     now=now, model=model)
        ref = _stop_reference(contacts, pose=pose, now=now,
                              deadline=now + budget, model=model)
        assert [id(w) for w in got] == [id(w) for w in ref]
        trimmed += len(got) < len(contacts)
        admitted_some += len(got) > 0
    # The property is only meaningful if the budget actually binds sometimes
    # and admits something sometimes.
    assert trimmed > 50 and admitted_some > 50


def test_skip_matters_for_the_devices_key():
    """A dense far contact that does not fit must not veto a near one."""
    far_dense = _wp((60, 0, 0), *[f"f{i}" for i in range(20)])  # 13 s, 20/13
    near = _wp((10, 0, 0), "n")                                  # 3 s, 1/3
    assert _admit([far_dense, near], value=VALUE_DEVICES, budget=12.0) == [near]
    # Stopping at the first miss would have returned nothing.


@pytest.mark.parametrize("value", [VALUE_UNIT, VALUE_DEVICES])
def test_route_fits_the_budget_and_is_a_subset(value):
    rng = random.Random(7 if value == VALUE_UNIT else 8)
    for _ in range(300):
        contacts, pose, now, budget, model = _random_instance(rng)
        got = _admit(contacts, value=value, budget=budget, pose=pose, now=now,
                     model=model)
        given = {id(c) for c in contacts}
        assert all(id(w) in given for w in got)
        assert len({id(w) for w in got}) == len(got)
        assert _route_cost(got, pose=pose, model=model) <= budget + 1e-9


def test_devices_key_is_devices_per_total_seconds():
    """The devices key divides by total (transit + one session per contact),
    not by transit alone and not by one session per device.

    From the origin at 5 m/s with a 1 s session:
      A (0.5,0,0), 1 device: transit 0.1, total 1.1 -> 1/1.1 = 0.91
      B (5,0,0),   3 devices: transit 1.0, total 2.0 -> 3/2.0 = 1.50
    So B goes first. Dividing by transit (1/0.1 = 10 > 3/1 = 3) or charging a
    session per device (3/4 = 0.75 < 0.91) would pick A first instead. After
    B, A costs 4.5/5 + 1 = 1.9 s, so a 2.5 s budget keeps B alone; after A, B
    would cost 1.9 s and end at 3.0 s, so the wrong keys return [A].
    """
    a = _wp((0.5, 0, 0), "a")
    b = _wp((5, 0, 0), "b1", "b2", "b3")
    o = (0.0, 0.0, 0.0)
    assert DEFAULT_MODEL.cost(o, a.position) == pytest.approx((0.1, 1.1))
    assert DEFAULT_MODEL.cost(o, b.position) == pytest.approx((1.0, 2.0))
    assert DEFAULT_MODEL.cost(b.position, a.position)[1] == pytest.approx(1.9)
    assert _admit([a, b], value=VALUE_DEVICES, budget=2.5) == [b]
    assert _admit([a, b], value=VALUE_DEVICES, budget=None) == [b, a]
    # The unit key takes the cheaper A first, then B fits too (1.1 + 1.9).
    assert _admit([a, b], value=VALUE_UNIT, budget=None) == [a, b]
    assert _admit([a, b], value=VALUE_UNIT, budget=2.5) == [a]


@pytest.mark.parametrize("value", [VALUE_UNIT, VALUE_DEVICES])
def test_a_free_contact_is_taken_first(value):
    """With no session time, a contact at the mule pose costs 0 s. It is the
    argmin for 'unit' and scores infinity for 'devices', so it goes first even
    though a farther contact holds more devices."""
    model = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=0.0)
    here = _wp((0, 0, 0), "h")
    there = _wp((5, 0, 0), "t1", "t2", "t3")
    assert model.cost((0.0, 0.0, 0.0), here.position) == (0.0, 0.0)
    assert _admit([there, here], value=value, budget=None,
                  model=model) == [here, there]
    # A zero budget still admits the free contact (0 <= 0) and only it.
    assert _admit([there, here], value=value, budget=0.0,
                  model=model) == [here]


@pytest.mark.parametrize("value", [VALUE_UNIT, VALUE_DEVICES])
def test_policy_matches_an_independent_skip_reference(value):
    """Both keys against a separately written reading of the spec, on smooth
    random instances and on tie-heavy grid instances."""
    rng = random.Random(1804 if value == VALUE_UNIT else 8333)
    reordered = 0
    for k in range(600):
        if k % 2:
            contacts, pose, now, budget, model = _tie_heavy_instance(rng)
        else:
            contacts, pose, now, budget, model = _random_instance(rng)
        deadline = None if budget is None else now + budget
        got = _admit(contacts, value=value, budget=budget, pose=pose, now=now,
                     model=model)
        ref = _skip_reference(contacts, value=value, pose=pose, now=now,
                              deadline=deadline, model=model)
        assert [id(w) for w in got] == [id(w) for w in ref]
        reordered += [id(w) for w in got] != [id(c) for c in contacts][:len(got)]
    # The comparison is only meaningful if the route often differs from the
    # input order.
    assert reordered > 200


def test_both_keys_coincide_when_every_contact_holds_one_device():
    rng = random.Random(42)
    for _ in range(300):
        contacts, pose, now, budget, model = _random_instance(
            rng, single_device=True)
        unit = _admit(contacts, value=VALUE_UNIT, budget=budget, pose=pose,
                      now=now, model=model)
        dev = _admit(contacts, value=VALUE_DEVICES, budget=budget, pose=pose,
                     now=now, model=model)
        assert [id(w) for w in unit] == [id(w) for w in dev]


def test_the_argmax_is_recomputed_from_the_end_of_the_route():
    """Sorted once by distance from the origin the order would be a, d, b.
    Algorithm 3 re-prices from the end of the route: after a, b is 2 m away
    and d is 21 m away, so b comes next."""
    a = _wp((10, 0, 0), "a")      # 10 m from origin
    b = _wp((12, 0, 0), "b")      # 12 m from origin, 2 m from a
    d = _wp((-11, 0, 0), "d")     # 11 m from origin, 21 m from a
    assert _admit([b, d, a], budget=None) == [a, b, d]


# --------------------------------------------------------------------------- #
# 4. No deadline admits everything, in greedy order
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [VALUE_UNIT, VALUE_DEVICES])
def test_no_deadline_admits_all(value):
    rng = random.Random(99)
    for _ in range(100):
        contacts, pose, now, _budget, model = _random_instance(rng)
        got = _admit(contacts, value=value, budget=None, pose=pose, now=now,
                     model=model)
        assert sorted(id(w) for w in got) == sorted(id(c) for c in contacts)


def test_no_deadline_keeps_the_greedy_order():
    c30, c10, c20 = _wp((30, 0, 0), "c30"), _wp((10, 0, 0), "c10"), _wp((20, 0, 0), "c20")
    assert _admit([c30, c10, c20], budget=None) == [c10, c20, c30]
    # The devices key takes the dense contact first, even though it is farthest.
    dense = _wp((30, 0, 0), "x1", "x2", "x3", "x4", "x5", "x6", "x7")
    got = _admit([dense, c10, c20], value=VALUE_DEVICES, budget=None)
    assert got[0] is dense and len(got) == 3


def test_no_deadline_with_no_model_still_orders():
    got = FedCSDegradedPolicy().admit_and_order(
        [R, P, Q], {}, _env(), mission_deadline_ts=None, feasibility_model=None)
    assert got == [P, R, Q]


# --------------------------------------------------------------------------- #
# 5. Per-device deadlines are never read
# --------------------------------------------------------------------------- #

def test_contacts_past_their_deadline_are_still_admitted():
    """S3b would drop all three as overdue; FedCS has only the round deadline."""
    past = [_wp((10, 0, 0), "p", deadline_ts=0.0),
            _wp((20, 0, 0), "r", deadline_ts=NOW - 500.0),
            _wp((30, 0, 0), "s", deadline_ts=-1e9)]
    for value in (VALUE_UNIT, VALUE_DEVICES):
        assert _admit(past, value=value, budget=100.0) == past


class _NoDeadlineContact:
    """Duck-typed contact whose deadline and bucket may not be touched."""

    def __init__(self, position, devices):
        self.position = position
        self.devices = devices

    @property
    def deadline_ts(self):
        raise AssertionError("FedCS-degraded read wp.deadline_ts")

    @property
    def bucket(self):
        raise AssertionError("FedCS-degraded read wp.bucket")


@pytest.mark.parametrize("value", [VALUE_UNIT, VALUE_DEVICES])
def test_deadline_and_bucket_are_never_accessed(value):
    wps = [_NoDeadlineContact((10.0, 0.0, 0.0), (DeviceID("a"),)),
           _NoDeadlineContact((0.0, 30.0, 0.0), (DeviceID("b"), DeviceID("c"))),
           _NoDeadlineContact((20.0, 0.0, 0.0), (DeviceID("d"),))]
    pol = FedCSDegradedPolicy(value=value)
    pol.admit_and_order(wps, {}, _env(), mission_deadline_ts=NOW + 12.0,
                        feasibility_model=DEFAULT_MODEL)
    pol.admit_and_order(wps, {}, _env(), mission_deadline_ts=None,
                        feasibility_model=DEFAULT_MODEL)
    pol.rank_contacts(wps, {}, _env())


def test_rewriting_deadlines_does_not_change_the_route():
    rng = random.Random(3)
    for _ in range(100):
        contacts, pose, now, budget, model = _random_instance(rng)
        base = _admit(contacts, budget=budget, pose=pose, now=now, model=model)
        rewritten = [ContactWaypoint(position=c.position, devices=c.devices,
                                     bucket=Bucket.NEW,
                                     deadline_ts=rng.uniform(-1e6, 1e6))
                     for c in contacts]
        again = _admit(rewritten, budget=budget, pose=pose, now=now, model=model)
        assert [w.devices for w in base] == [w.devices for w in again]


# --------------------------------------------------------------------------- #
# 6. Determinism
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [VALUE_UNIT, VALUE_DEVICES])
def test_route_does_not_depend_on_input_order(value):
    rng = random.Random(11)
    for _ in range(60):
        contacts, pose, now, budget, model = _random_instance(rng)
        first = _admit(contacts, value=value, budget=budget, pose=pose,
                       now=now, model=model)
        for _ in range(5):
            shuffled = list(contacts)
            rng.shuffle(shuffled)
            got = _admit(shuffled, value=value, budget=budget, pose=pose,
                         now=now, model=model)
            assert [id(w) for w in got] == [id(w) for w in first]


def test_ties_break_on_position_then_devices():
    """Equal cost and equal score: the smaller (position, devices) goes first.

    The names run against the positions ('z' sits at the smaller position), so
    a tie-break on devices alone would put 'a' first.
    """
    left, right = _wp((-10, 0, 0), "z"), _wp((10, 0, 0), "a")
    for value in (VALUE_UNIT, VALUE_DEVICES):
        assert _names(_admit([right, left], value=value, budget=None)) == ["z", "a"]
        assert _names(_admit([left, right], value=value, budget=None)) == ["z", "a"]
    # Same position: devices decide.
    a, b = _wp((10, 0, 0), "b"), _wp((10, 0, 0), "a")
    assert _names(_admit([a, b], budget=None)) == ["a", "b"]


# --------------------------------------------------------------------------- #
# 7. No mutation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [VALUE_UNIT, VALUE_DEVICES])
def test_nothing_passed_in_is_mutated(value):
    contacts = [P, Q, R, _wp((5, 5, 0), "s", deadline_ts=0.0)]
    states = _states(contacts)
    states_before = copy.deepcopy(states)
    contacts_before = list(contacts)
    env = _env()
    pol = FedCSDegradedPolicy(value=value)
    pol.admit_and_order(contacts, states, env, mission_deadline_ts=NOW + 12.0,
                        feasibility_model=DEFAULT_MODEL)
    pol.admit_and_order(contacts, states, env, mission_deadline_ts=None,
                        feasibility_model=DEFAULT_MODEL)
    pol.rank_contacts(contacts, states, env)
    assert contacts == contacts_before
    assert states == states_before
    assert env.mule_pose == (0.0, 0.0, 0.0) and env.now == NOW


# --------------------------------------------------------------------------- #
# 8. Ordering-only surface and scheduler delegation
# --------------------------------------------------------------------------- #

def test_rank_contacts_is_pass_1_only():
    with pytest.raises(SelectorScopeViolation):
        FedCSDegradedPolicy().rank_contacts([P], {}, _env(),
                                            pass_kind=MissionPass.DELIVER)


def test_rank_contacts_refuses_unadmitted_devices():
    with pytest.raises(SelectorScopeViolation):
        FedCSDegradedPolicy().rank_contacts([P, R], {}, _env(),
                                            admitted=[DeviceID("p")])


def test_rank_contacts_returns_the_greedy_order_without_a_cut():
    assert FedCSDegradedPolicy().rank_contacts([Q, R, P], {}, _env()) == [P, R, Q]


def test_the_scheduler_delegates_to_fedcs():
    """Through FLScheduler: S1 and S3a run, then the FedCS route is returned.

    Devices at x = 0, 40, 80 at 1 m/s with no session time and a 50 s budget:
    the chain admits 0 (0 s) and 40 (40 s); 80 would end at 80 s.
    """
    from hermes.scheduler.fl_scheduler import FLScheduler
    from hermes.types import MissionSlice, MuleID

    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    sch = FLScheduler(now_fn=lambda: NOW, target_selector=FedCSDegradedPolicy(),
                      mission_budget_s=50.0, feasibility_model=model)
    devices = ("a", "b", "c")
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=tuple(DeviceID(d) for d in devices),
        issued_round=1, issued_at=NOW,
    ))
    for i, d in enumerate(devices):
        sch.device_states[DeviceID(d)].last_known_position = (i * 40.0, 0.0, 0.0)
    route = sch.build_contact_queue(rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0),
                                    mule_energy=1.0)
    assert [w.devices for w in route] == [(DeviceID("a"),), (DeviceID("b"),)]
