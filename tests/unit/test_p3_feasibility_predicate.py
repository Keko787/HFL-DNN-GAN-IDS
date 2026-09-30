"""FeRRy Phase 3 (unit U4): one FeasibilityModel, one predicate.

Pins, by hand arithmetic, what the ferry-mode predicate prices (design §3.1):
arrival, finish = arrival + dwell, home = finish + return + upload (Pass 1
only), the deadline clause on collection (or on delivery), the budget clause on
home, the energy clause only with a capacity, RULE_NONE admitting while still
reporting home, and the opt-in contract (no budget, no gate, ferry mode
included). Also: member dwell looked up from device states (critic B11),
members out of range or below the floor never charged (critic B12), the walks
built as folds of the one predicate (FedCS and FedEx included), and the legacy
predicate's exact boundary behaviour.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from hermes.scheduler import FLScheduler
from hermes.scheduler.policies.budget_walk import greedy_budget_walk
from hermes.scheduler.policies.fedcs_degraded import (
    VALUE_DEVICES,
    VALUE_UNIT,
    fedcs_greedy_select,
)
from hermes.scheduler.policies.fedex_carp import FedExCarpPolicy
from hermes.scheduler.routing.two_opt import order_contacts
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS_DELIVERY,
    REASON_BUDGET,
    REASON_ENERGY,
    REASON_OVERDUE,
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FeasibilityModel,
    FeasibilityResult,
    FerryPhysics,
    FlightState,
    filter_feasible,
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
down = lambda x: math.nextafter(x, -math.inf)   # noqa: E731


def _wp(x, y=0.0, *devs, deadline=1e12, bucket=Bucket.SCHEDULED_THIS_ROUND):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=bucket, deadline_ts=float(deadline))


def member_dwell(d, pass_kind, offset):
    """2 s + d/4 per member in Pass 1, 1 s + d/4 in Pass 2; the offset (dB)
    buys 1/8 s per dB; below the floor (None) beyond 50 m."""
    if d > 50.0:
        return None
    base = 2.0 if pass_kind is COLLECT else 1.0
    return base + d / 4.0 - offset / 8.0


def upload_3s():
    return 3.0


def physics(positions, **kw):
    kw.setdefault("upload_s", upload_3s)
    kw.setdefault("p_move_w", 100.0)
    kw.setdefault("p_hover_w", 200.0)
    return FerryPhysics(dock=DOCK, member_dwell_s=member_dwell,
                        device_states=positions, **kw)


# Stop A is 50 m from the dock (10 s at 5 m/s each way). Members: a1 at the
# stop (2 s), a2 4 m off (3 s), a3 60 m off (below the floor: not charged).
POS = {"a1": (30.0, 40.0, 0.0), "a2": (30.0, 44.0, 0.0), "a3": (30.0, 100.0, 0.0)}
A = _wp(30.0, 40.0, "a1", "a2", "a3", deadline=1015.0)
T0 = FlightState(DOCK, 1000.0)


def ferry_model(**kw):
    return FeasibilityModel(ferry=physics(POS, **kw))


# --------------------------------------------------------------------------- #
# Ferry arithmetic by hand
# --------------------------------------------------------------------------- #

def test_leg_prices_transit_dwell_return_and_upload():
    leg = ferry_model().leg(DOCK, A)
    assert (leg.transit_s, leg.dwell_s, leg.total_s, leg.return_s, leg.upload_s) == (
        10.0, 5.0, 15.0, 10.0, 3.0)
    # Pass 2 moves the Pass-2 payload and has no upload tail.
    leg2 = ferry_model().leg(DOCK, A, pass_kind=DELIVER)
    assert (leg2.dwell_s, leg2.upload_s) == (3.0, 0.0)
    # δ_obs: +8 dB buys 1 s per charged member.
    assert ferry_model().leg(DOCK, A, snr_offset_db=8.0).dwell_s == 3.0


def test_admit_times_and_next_state():
    v = ferry_model().admit(T0, A, rule=RULE_DEADLINE_BUDGET, budget_end=1028.0)
    assert v.ok and v.reason is None
    assert (v.arrival, v.finish, v.home) == (1010.0, 1015.0, 1028.0)
    # Energy: P_move * transit + P_hover * dwell = 100*10 + 200*5.
    assert v.next_state == FlightState(A.position, 1015.0, 2000.0)
    v2 = ferry_model().admit(T0, A, rule=RULE_BUDGET, budget_end=1e9, pass_kind=DELIVER)
    assert (v2.finish, v2.home) == (1013.0, 1023.0)


def test_deadline_bounds_collection_by_default():
    m = ferry_model()
    assert m.admit(T0, A, rule=RULE_DEADLINE_BUDGET, budget_end=1e9).ok        # finish == deadline
    late = _wp(30.0, 40.0, "a1", "a2", "a3", deadline=down(1015.0))
    v = m.admit(T0, late, rule=RULE_DEADLINE_BUDGET, budget_end=1e9)
    assert (v.ok, v.reason) == (False, REASON_OVERDUE)
    # Arrival alone (1010) is before the deadline: the dwell is what misses it.
    assert v.arrival < late.deadline_ts


def test_deadline_bounds_delivery_variant():
    m = ferry_model(deadline_bounds=DEADLINE_BOUNDS_DELIVERY)
    v = m.admit(T0, A, rule=RULE_DEADLINE_BUDGET, budget_end=1e9)      # home 1028 > 1015
    assert (v.ok, v.reason) == (False, REASON_OVERDUE)
    ok = _wp(30.0, 40.0, "a1", "a2", "a3", deadline=1028.0)
    assert m.admit(T0, ok, rule=RULE_DEADLINE_BUDGET, budget_end=1e9).ok


def test_delivery_bounds_each_stop_by_its_own_return_not_the_routes_landing():
    """``delivery`` is the plan's per-stop, single-contact predicate (final
    check, clock F1): each stop's own return and upload must meet ITS
    deadline. It does not bound when the update is actually delivered, the
    route's landing plus the upload, which is later for every stop but the
    last. Channel-free (one 1 s session per contact), no upload, 5 m/s: a
    at 10 m with D_a = t0 + 6 is flown first (EDF); its own return is home
    at t0 + 5 <= D_a, so it is admitted, and so is b at 40 m. The route
    lands at t0 + 18, so a's update reaches the cluster 12 s after D_a.
    Pinned as designed; a route-level bound (every flown stop's deadline on
    the landing) would be a spec change, and would refuse b here."""
    t0 = 1000.0
    a, b = _wp(10.0, 0.0, "a", deadline=t0 + 6.0), _wp(40.0, 0.0, "b")
    phys = FerryPhysics(dock=DOCK, member_dwell_s=None, upload_s=lambda: 0.0,
                        p_move_w=1.0, p_hover_w=1.0, deadline_bounds=DEADLINE_BOUNDS_DELIVERY)
    m = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=phys)
    res = filter_feasible([b, a], now=t0, mission_deadline_ts=t0 + 100.0, model=m)
    assert res.kept == [a, b] and res.n_dropped == 0
    walk = m.fold(res.kept, FlightState(DOCK, t0), rule=RULE_DEADLINE_BUDGET,
                  budget_end=t0 + 100.0, skip=True)
    va, vb = walk.verdicts
    assert (va.finish, va.home) == (t0 + 3.0, t0 + 5.0) and va.home <= a.deadline_ts
    assert (vb.finish, vb.home) == (t0 + 10.0, t0 + 18.0)
    # The route's landing (no upload here) is the last stop's own home.
    assert walk.home == vb.home == t0 + 18.0 > a.deadline_ts
    # The per-stop clause itself binds: a stop whose own return misses its
    # deadline is refused.
    tight = _wp(10.0, 0.0, "a", deadline=down(t0 + 5.0))
    assert m.admit(FlightState(DOCK, t0), tight, rule=RULE_DEADLINE_BUDGET,
                   budget_end=t0 + 100.0).reason == REASON_OVERDUE


def test_budget_clause_bounds_home():
    m = ferry_model()
    assert m.admit(T0, A, rule=RULE_BUDGET, budget_end=1028.0).ok
    v = m.admit(T0, A, rule=RULE_BUDGET, budget_end=down(1028.0))
    assert (v.ok, v.reason) == (False, REASON_BUDGET)
    # The legacy model has no return leg or upload: finish 1011 fits.
    assert FeasibilityModel().admit(T0, A, rule=RULE_BUDGET, budget_end=1011.0).ok


def test_budget_rule_has_no_deadline_clause():
    hopeless = _wp(30.0, 40.0, "a1", deadline=0.0)
    assert ferry_model().admit(T0, hopeless, rule=RULE_BUDGET, budget_end=1e9).ok
    assert not ferry_model().admit(T0, hopeless, rule=RULE_DEADLINE_BUDGET, budget_end=1e9).ok


def test_energy_clause_only_with_a_capacity():
    # need = 100 * (10 + 10) + 200 * 5 = 3000 J.
    assert ferry_model().admit(T0, A, rule=RULE_BUDGET, budget_end=1e9).ok     # no capacity
    assert ferry_model(energy_capacity_j=3000.0).admit(
        T0, A, rule=RULE_BUDGET, budget_end=1e9).ok
    v = ferry_model(energy_capacity_j=down(3000.0)).admit(
        T0, A, rule=RULE_BUDGET, budget_end=1e9)
    assert (v.ok, v.reason) == (False, REASON_ENERGY)
    # Energy already spent counts.
    spent = FlightState(DOCK, 1000.0, 1.0)
    v = ferry_model(energy_capacity_j=3000.0).admit(spent, A, rule=RULE_BUDGET, budget_end=1e9)
    assert v.reason == REASON_ENERGY


def test_clause_order_is_deadline_budget_energy():
    m = ferry_model(energy_capacity_j=1.0)
    late = _wp(30.0, 40.0, "a1", deadline=0.0)
    assert m.admit(T0, late, rule=RULE_DEADLINE_BUDGET, budget_end=0.0).reason == REASON_OVERDUE
    assert m.admit(T0, late, rule=RULE_BUDGET, budget_end=0.0).reason == REASON_BUDGET
    assert m.admit(T0, late, rule=RULE_BUDGET, budget_end=1e9).reason == REASON_ENERGY


def test_rule_none_always_admits_and_reports_home():
    m = ferry_model(energy_capacity_j=1.0)
    v = m.admit(T0, _wp(30.0, 40.0, "a1", deadline=0.0), rule=RULE_NONE, budget_end=0.0)
    assert v.ok and v.reason is None
    assert v.home == 1000.0 + 10.0 + 2.0 + 10.0 + 3.0


def test_protected_skips_only_the_deadline_clause():
    m = ferry_model()
    late = _wp(30.0, 40.0, "a1", deadline=0.0)
    assert m.admit(T0, late, rule=RULE_DEADLINE_BUDGET, budget_end=1e9, protected=True).ok
    v = m.admit(T0, late, rule=RULE_DEADLINE_BUDGET, budget_end=1000.0, protected=True)
    assert v.reason == REASON_BUDGET


@pytest.mark.parametrize("rule", [RULE_DEADLINE_BUDGET, RULE_BUDGET, RULE_NONE])
def test_no_budget_no_gate_even_in_ferry_mode(rule):
    """The opt-in contract (design §3.1): with no budget nothing is rejected."""
    m = ferry_model(energy_capacity_j=1.0)
    v = m.admit(T0, _wp(30.0, 40.0, "a1", deadline=0.0), rule=rule, budget_end=None)
    assert v.ok
    contacts = [_wp(30.0, 40.0, "a1", deadline=0.0), _wp(-30.0, 40.0, "a2", deadline=0.0)]
    res = filter_feasible(contacts, now=1000.0, mission_deadline_ts=None, model=m)
    assert res.kept == contacts and res.n_dropped == 0
    assert greedy_budget_walk(contacts, key=lambda c: (c.position,), mule_pose=DOCK,
                              now=1000.0, mission_deadline_ts=None, model=m) == sorted(
        contacts, key=lambda c: (c.position,))


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="rule must be one of"):
        ferry_model().admit(T0, A, rule="deadline", budget_end=1e9)


# --------------------------------------------------------------------------- #
# Dwell: member positions from device states (B11), no infinite charge (B12)
# --------------------------------------------------------------------------- #

def test_dwell_looks_member_positions_up_in_device_states():
    states = {DeviceID(k): DeviceSchedulerState(device_id=DeviceID(k), last_known_position=v)
              for k, v in POS.items()}
    phys = physics(states)
    assert phys.member_distances_m(A) == (0.0, 4.0, 60.0)
    assert phys.dwell_s(A) == 5.0
    # A mapping to plain positions works the same.
    assert physics(POS).dwell_s(A) == 5.0


def test_out_of_range_and_below_floor_members_are_never_charged():
    assert physics(POS).dwell_s(A) == 5.0                 # a3 (60 m) below the floor
    assert physics(POS, range_m=3.0).dwell_s(A) == 2.0     # a2 (4 m) out of range too
    inf_phys = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: math.inf,
                            upload_s=lambda: 0.0, p_move_w=1.0, p_hover_w=1.0,
                            device_states=POS)
    assert inf_phys.dwell_s(A) == 0.0
    nan_phys = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: math.nan,
                            upload_s=lambda: 0.0, p_move_w=1.0, p_hover_w=1.0)
    with pytest.raises(ValueError):
        nan_phys.dwell_s(A)


def test_members_are_priced_at_the_stop_without_a_map_and_refused_if_unknown():
    no_map = FerryPhysics(dock=DOCK, member_dwell_s=member_dwell, upload_s=lambda: 0.0,
                          p_move_w=1.0, p_hover_w=1.0)
    assert no_map.member_distances_m(A) == (0.0, 0.0, 0.0)
    assert no_map.dwell_s(A) == 6.0
    with pytest.raises(KeyError, match="B11"):
        physics({"a1": (0.0, 0.0, 0.0)}).dwell_s(A)


def test_the_clock_without_a_band_charges_one_session_per_contact():
    """Critic A1: with the mission clock on and no band, a contact costs the
    model's session time once, whatever its members; the tail still counts."""
    phys = FerryPhysics(dock=DOCK, member_dwell_s=None, upload_s=upload_3s,
                        p_move_w=100.0, p_hover_w=200.0)
    m = FeasibilityModel(session_time_s=1.0, ferry=phys)
    leg = m.leg(DOCK, A)
    assert (leg.transit_s, leg.dwell_s, leg.return_s, leg.upload_s) == (10.0, 1.0, 10.0, 3.0)
    v = m.admit(T0, A, rule=RULE_BUDGET, budget_end=1024.0)
    assert v.ok and (v.finish, v.home) == (1011.0, 1024.0)
    assert v.next_state.energy_j == 100.0 * 10.0 + 200.0 * 1.0
    with pytest.raises(TypeError, match="no member dwell"):
        phys.dwell_s(A)


@pytest.mark.parametrize("bad", [math.inf, -1.0, math.nan])
def test_a_predicted_upload_is_never_charged_as_infinite(bad):
    """Critic B12: a rate of 0 is never charged as inf. The mule caps a
    carrier predicted below the floor; anything else is refused, in the
    predicate and in the FedEx diagnostics alike."""
    m = ferry_model(upload_s=lambda: bad)
    with pytest.raises(ValueError, match="B12"):
        m.leg(DOCK, A)
    with pytest.raises(ValueError, match="B12"):
        m.admit(T0, A, rule=RULE_NONE, budget_end=None)
    with pytest.raises(ValueError, match="B12"):
        FedExCarpPolicy().admit_and_order([A], {}, SelectorEnv(mule_pose=DOCK, now=0.0),
                                          feasibility_model=m)
    # Pass 2 has no upload tail, so it never asks.
    assert m.leg(DOCK, A, pass_kind=DELIVER).upload_s == 0.0


def test_the_scheduler_binds_its_device_states_into_a_ferry_model():
    unbound = FeasibilityModel(ferry=FerryPhysics(
        dock=DOCK, member_dwell_s=member_dwell, upload_s=lambda: 0.0,
        p_move_w=1.0, p_hover_w=1.0))
    sch = FLScheduler(feasibility_model=unbound, mission_budget_s=100.0)
    assert sch.feasibility_model.ferry.device_states is sch.device_states
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=(DeviceID("a1"),),
                                  issued_round=0, issued_at=0.0))
    sch.device_states[DeviceID("a1")].last_known_position = (3.0, 4.0, 0.0)
    wp = _wp(0.0, 0.0, "a1")
    assert sch.feasibility_model.ferry.member_distances_m(wp) == (5.0,)   # live view
    # A legacy model, and a ferry model with its own map, are kept as given.
    legacy = FeasibilityModel(cruise_speed_m_s=2.0)
    assert FLScheduler(feasibility_model=legacy).feasibility_model is legacy
    own = ferry_model()
    assert FLScheduler(feasibility_model=own).feasibility_model is own


def test_ferry_physics_validation_and_equality():
    with pytest.raises(ValueError):
        physics(POS, deadline_bounds="arrival")
    with pytest.raises(ValueError):
        physics(POS, p_move_w=-1.0)
    with pytest.raises(ValueError):
        physics(POS, range_m=0.0)
    with pytest.raises(ValueError):
        FerryPhysics(dock=(0.0, 0.0), member_dwell_s=member_dwell, upload_s=lambda: 0.0,
                     p_move_w=1.0, p_hover_w=1.0)
    # The device-state map is not part of equality or hashing.
    a, b = physics(POS), physics({})
    assert a == b and hash(FeasibilityModel(ferry=a)) == hash(FeasibilityModel(ferry=b))


def test_with_energy_spent_hands_on_the_remaining_capacity():
    m = ferry_model(energy_capacity_j=5000.0)
    assert m.with_energy_spent(1200.0).ferry.energy_capacity_j == 3800.0
    assert ferry_model().with_energy_spent(1200.0) == ferry_model()        # no clause
    legacy = FeasibilityModel()
    assert legacy.with_energy_spent(10.0) is legacy


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

B = _wp(-30.0, 40.0, "b1")          # 60 m from A, 50 m from the dock
POS_AB = dict(POS, b1=(-30.0, 40.0, 0.0))


def test_fold_skip_and_no_skip():
    m = FeasibilityModel(ferry=physics(POS_AB))
    # A: finish 1015, home 1028. B from A: 12 s transit, 2 s dwell, finish
    # 1029, home 1029 + 10 + 3 = 1042.
    both = m.fold([A, B], T0, rule=RULE_BUDGET, budget_end=1042.0, skip=False)
    assert both.ok and both.route == (A, B) and both.home == 1042.0
    assert both.state.clock == 1029.0
    tight = m.fold([A, B], T0, rule=RULE_BUDGET, budget_end=1041.0, skip=False)
    assert not tight.ok and tight.rejected == ((B, REASON_BUDGET),)
    assert tight.route == (A, B)                       # flown anyway
    skipped = m.fold([A, B], T0, rule=RULE_BUDGET, budget_end=1041.0, skip=True)
    assert skipped.route == (A,) and skipped.rejected == ((B, REASON_BUDGET),)
    assert skipped.state.clock == 1015.0 and skipped.home == 1028.0
    empty = m.fold([], FlightState((30.0, 40.0, 0.0), 1015.0), rule=RULE_BUDGET,
                   budget_end=0.0, skip=True)
    assert empty.ok and empty.home == 1025.0           # the return leg, no upload


def test_ferry_walks_are_folds_of_the_predicate():
    m = FeasibilityModel(ferry=physics(POS_AB))
    contacts = [B, A]
    res = filter_feasible(contacts, now=1000.0, mission_deadline_ts=1041.0, model=m)
    ordered = sorted(contacts, key=lambda c: (c.deadline_ts, c.position, c.devices))
    walk = m.fold(ordered, T0, rule=RULE_DEADLINE_BUDGET, budget_end=1041.0, skip=True)
    assert res.kept == list(walk.route)
    assert res.dropped_budget == walk.rejected_by(REASON_BUDGET)
    key = lambda c: (c.position,)                                # noqa: E731
    route = greedy_budget_walk(contacts, key=key, mule_pose=DOCK, now=1000.0,
                               mission_deadline_ts=1041.0, model=m)
    assert route == list(m.fold(sorted(contacts, key=key), T0, rule=RULE_BUDGET,
                                budget_end=1041.0, skip=True).route)


def test_the_budget_walk_prices_the_pass_it_is_given():
    """The budgeted Pass-2 walk (design §3.2, §3.5) moves DELIVER bytes and
    has no upload tail. Here Pass 1 costs a 10 s dwell and a 5 s upload,
    Pass 2 a 1 s dwell; A is 10 m out at 1 m/s. Pass 2 needs 10 + 1 + 10 =
    21 s and Pass 1 35 s, so a 21 s budget keeps A only when the walk is told
    it is Pass 2."""
    A1 = _wp(10.0, 0.0, "a")
    phys = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: 10.0 if p is COLLECT else 1.0,
                        upload_s=lambda: 5.0, p_move_w=1.0, p_hover_w=1.0,
                        device_states={"a": A1.position})
    m = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0, ferry=phys)

    def walk(budget, **kw):
        return greedy_budget_walk([A1], key=lambda c: (0,), mule_pose=DOCK, now=0.0,
                                  mission_deadline_ts=budget, model=m, **kw)

    assert walk(21.0, pass_kind=DELIVER) == [A1]
    assert walk(down(21.0), pass_kind=DELIVER) == []
    assert walk(21.0, pass_kind="deliver") == [A1]
    assert walk(21.0) == [] and walk(34.9) == []             # Pass 1, the default
    assert walk(35.0) == [A1] and walk(35.0, pass_kind=COLLECT) == [A1]


def test_the_pass_may_be_given_as_its_string_value():
    """``MissionPass`` is a ``str`` enum, so ``"collect" == MissionPass.COLLECT``.
    The predicate normalises the pass: the string spelling prices exactly like
    the enum, the Pass-1 upload included (an identity test dropped it, and a
    27 s budget admitted A with ``"collect"`` although home is at 28 s)."""
    m = ferry_model()
    for text, enum in (("collect", COLLECT), ("deliver", DELIVER)):
        assert m.leg(DOCK, A, pass_kind=text) == m.leg(DOCK, A, pass_kind=enum)
        for rule in (RULE_DEADLINE_BUDGET, RULE_BUDGET, RULE_NONE):
            for budget in (None, 1023.0, down(1028.0), 1028.0):
                assert m.admit(T0, A, rule=rule, budget_end=budget, pass_kind=text) == m.admit(
                    T0, A, rule=rule, budget_end=budget, pass_kind=enum)
        ab = FeasibilityModel(ferry=physics(POS_AB))
        assert ab.fold([A, B], T0, rule=RULE_BUDGET, budget_end=1041.0, pass_kind=text,
                       skip=True) == ab.fold([A, B], T0, rule=RULE_BUDGET, budget_end=1041.0,
                                             pass_kind=enum, skip=True)
        legacy = FeasibilityModel()
        assert legacy.admit(T0, A, rule=RULE_BUDGET, budget_end=1011.0, pass_kind=text) == (
            legacy.admit(T0, A, rule=RULE_BUDGET, budget_end=1011.0, pass_kind=enum))
    assert m.leg(DOCK, A, pass_kind="collect").upload_s == 3.0
    v = m.admit(T0, A, rule=RULE_BUDGET, budget_end=down(1028.0), pass_kind="collect")
    assert (v.ok, v.reason, v.home) == (False, REASON_BUDGET, 1028.0)
    # The member dwell always receives the enum, whichever spelling came in.
    seen = []

    def recording(d, pass_kind, offset):
        seen.append(pass_kind)
        return member_dwell(d, pass_kind, offset)

    rec = FeasibilityModel(ferry=FerryPhysics(dock=DOCK, member_dwell_s=recording,
                                              upload_s=upload_3s, p_move_w=1.0, p_hover_w=1.0,
                                              device_states=POS))
    assert rec.leg(DOCK, A, pass_kind="deliver").dwell_s == 3.0
    assert rec.ferry.dwell_s(A, "collect") == 5.0
    assert seen and all(type(p) is MissionPass for p in seen)
    assert seen[0] is DELIVER and seen[-1] is COLLECT
    # An unknown pass is refused rather than priced as Pass 2.
    for call in (lambda: m.leg(DOCK, A, pass_kind="return"),
                 lambda: m.admit(T0, A, rule=RULE_BUDGET, budget_end=1e9, pass_kind="return"),
                 lambda: m.fold([A], T0, rule=RULE_BUDGET, budget_end=1e9, pass_kind="return",
                                skip=True),
                 lambda: m.ferry.dwell_s(A, "return")):
        with pytest.raises(ValueError):
            call()


def test_a_capacity_alone_does_not_gate():
    """The energy clause is part of the gate: without a budget a capacity is
    inert (the opt-in contract), in the predicate and in every walk."""
    m = ferry_model(energy_capacity_j=1.0)
    for rule in (RULE_DEADLINE_BUDGET, RULE_BUDGET):
        assert m.admit(T0, A, rule=rule, budget_end=None).ok
        assert not m.admit(T0, A, rule=rule, budget_end=1e9).ok
    assert filter_feasible([A], now=1000.0, mission_deadline_ts=None, model=m).kept == [A]
    assert fedcs_greedy_select([A], mule_pose=DOCK, now=1000.0, mission_deadline_ts=None,
                               model=m) == [A]


def test_filter_feasible_state_carries_energy_and_counts_energy_drops():
    m = FeasibilityModel(ferry=physics(POS, energy_capacity_j=3000.0))
    fresh = filter_feasible([A], now=1000.0, mission_deadline_ts=1e9, model=m)
    assert fresh.kept == [A] and fresh.n_dropped == 0
    tired = filter_feasible([A], now=1000.0, mission_deadline_ts=1e9, model=m,
                            state=FlightState(DOCK, 1000.0, 500.0))
    assert tired.kept == [] and tired.dropped_energy == [A]
    assert tired.n_dropped == 1 and tired.dropped == [A]
    # The same through the D-arm walk and FedCS.
    st = FlightState(DOCK, 1000.0, 500.0)
    assert greedy_budget_walk([A], key=lambda c: (0,), mule_pose=DOCK, now=1000.0,
                              mission_deadline_ts=1e9, model=m, state=st) == []
    assert fedcs_greedy_select([A], mule_pose=DOCK, now=1000.0,
                               mission_deadline_ts=1e9, model=m, state=st) == []


def test_feasibility_result_keeps_its_legacy_shape():
    r = FeasibilityResult([A], [B], [])
    assert r.dropped_energy == [] and r.n_dropped == 1 and r.dropped == [B]


# --------------------------------------------------------------------------- #
# The legacy predicate (ferry None): today's comparisons, today's order
# --------------------------------------------------------------------------- #

def test_legacy_leg_is_cost():
    m = FeasibilityModel(cruise_speed_m_s=4.0, session_time_s=7.0)
    leg = m.leg((1.5, -2.0, 0.0), _wp(41.5, -2.0, "x"))
    assert (leg.transit_s, leg.total_s) == m.cost((1.5, -2.0, 0.0), (41.5, -2.0, 0.0))
    assert (leg.dwell_s, leg.return_s, leg.upload_s) == (7.0, 0.0, 0.0)


def test_legacy_admit_boundaries():
    # 40 m at 4 m/s = 10 s of transit, then a 7 s session.
    m = FeasibilityModel(cruise_speed_m_s=4.0, session_time_s=7.0)
    s = FlightState(DOCK, 0.0)
    on_time = _wp(40.0, 0.0, "a", deadline=10.0)
    assert m.admit(s, on_time, rule=RULE_DEADLINE_BUDGET, budget_end=100.0).ok
    late = _wp(40.0, 0.0, "a", deadline=down(10.0))
    assert m.admit(s, late, rule=RULE_DEADLINE_BUDGET, budget_end=100.0).reason == REASON_OVERDUE
    fits = _wp(40.0, 0.0, "a")
    v = m.admit(s, fits, rule=RULE_BUDGET, budget_end=17.0)
    assert v.ok and (v.arrival, v.finish, v.home) == (10.0, 17.0, 17.0)
    assert m.admit(s, fits, rule=RULE_BUDGET, budget_end=down(17.0)).reason == REASON_BUDGET
    # Overdue is tested first, as S3b always did.
    assert m.admit(s, late, rule=RULE_DEADLINE_BUDGET, budget_end=0.0).reason == REASON_OVERDUE
    assert v.next_state == FlightState(fits.position, 17.0, 0.0)


# --------------------------------------------------------------------------- #
# FedCS (D5) and FedEx (D4) under the ferry predicate
# --------------------------------------------------------------------------- #

def _line_model(**kw):
    """1 m/s, members at their stops with no dwell, no upload."""
    phys = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: 0.0,
                        upload_s=lambda: 0.0, p_move_w=1.0, p_hover_w=1.0, **kw)
    return FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0, ferry=phys)


def test_fedcs_skip_is_not_stop_under_the_ferry_predicate():
    """The docstring's skip = stop holds only for the legacy model (critic B15).

    From x = 100: X at 110 is the cheapest leg (10 s) but 110 s from the dock
    (home 120 > 105); Y at 80 costs 20 s and is home at 100.
    """
    X, Y = _wp(110.0, 0.0, "x"), _wp(80.0, 0.0, "y")
    ferry = _line_model()
    for value in (VALUE_UNIT, VALUE_DEVICES):
        assert fedcs_greedy_select([X, Y], value=value, mule_pose=(100.0, 0.0, 0.0), now=0.0,
                                   mission_deadline_ts=105.0, model=ferry) == [Y]
    # The legacy model has no return leg: X fits (10 s), then Y (30 s more).
    legacy = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    assert fedcs_greedy_select([X, Y], mule_pose=(100.0, 0.0, 0.0), now=0.0,
                               mission_deadline_ts=105.0, model=legacy) == [X, Y]


def test_fedcs_selects_on_the_leg_total_under_ferry():
    """The pick ranks on transit + dwell: a near stop whose member sits 30 m
    away (a 30 s dwell here) loses to a farther one with no dwell."""
    near_slow, far_fast = _wp(10.0, 0.0, "slow"), _wp(0.0, 20.0, "fast")
    phys = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: d, upload_s=lambda: 0.0,
                        p_move_w=1.0, p_hover_w=1.0,
                        device_states={"slow": (10.0, 30.0, 0.0), "fast": (0.0, 20.0, 0.0)})
    m = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0, ferry=phys)
    assert fedcs_greedy_select([near_slow, far_fast], mule_pose=DOCK, now=0.0,
                               mission_deadline_ts=None, model=m) == [far_fast, near_slow]
    legacy = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    assert fedcs_greedy_select([near_slow, far_fast], mule_pose=DOCK, now=0.0,
                               mission_deadline_ts=None, model=legacy) == [near_slow, far_fast]


def test_fedex_diagnostics_under_the_ferry_model():
    """40 + 30 m out, 50 m back at 4 m/s; 7 s dwell per stop; 5 s upload."""
    NOW = 1000.0
    a, b = _wp(40.0, 0.0, "a"), _wp(40.0, 30.0, "b")
    phys = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: 7.0, upload_s=lambda: 5.0,
                        p_move_w=1.0, p_hover_w=1.0,
                        device_states={"a": a.position, "b": b.position})
    m = FeasibilityModel(cruise_speed_m_s=4.0, session_time_s=99.0, ferry=phys)
    for pol in (FedExCarpPolicy(), FedExCarpPolicy(depot=DOCK)):
        route = pol.admit_and_order([a, b], {}, SelectorEnv(mule_pose=DOCK, now=NOW),
                                    mission_deadline_ts=NOW + 49.0, feasibility_model=m)
        assert route == [a, b]
        # outbound (10 + 7) + (7.5 + 7) = 31.5; return 12.5; upload 5.
        assert pol.last_tour_cost_s == 49.0 and pol.last_upload_s == 5.0
        assert pol.last_return_leg_s == 12.5
        assert pol.last_tour_fits is True and pol.last_tour_overrun_s == 0.0
        assert pol.last_fits_without_return is True
        pol.admit_and_order([a, b], {}, SelectorEnv(mule_pose=DOCK, now=NOW),
                            mission_deadline_ts=NOW + 48.5, feasibility_model=m)
        assert pol.last_tour_fits is False and pol.last_tour_overrun_s == 0.5
        # Without the return leg (legacy meaning): now + outbound = 1031.5.
        pol.admit_and_order([a, b], {}, SelectorEnv(mule_pose=DOCK, now=NOW),
                            mission_deadline_ts=NOW + 31.5, feasibility_model=m)
        assert pol.last_fits_without_return is True
        assert pol.last_overrun_without_return_s == 0.0
    legacy = FedExCarpPolicy()
    legacy.admit_and_order([a, b], {}, SelectorEnv(mule_pose=DOCK, now=NOW),
                           mission_deadline_ts=NOW + 44.0,
                           feasibility_model=FeasibilityModel(cruise_speed_m_s=4.0,
                                                              session_time_s=7.0))
    assert legacy.last_upload_s is None and legacy.last_tour_cost_s == 44.0


def _ring():
    return [_wp(0.0, 50.0, "a"), _wp(-50.0, 0.0, "b"), _wp(50.0, -40.0, "c"),
            _wp(20.0, 20.0, "d")]


def _unit_ferry(**kw):
    """1 m/s, no dwell, no upload: the ferry predicate with the legacy tour's times."""
    phys = FerryPhysics(dock=DOCK, member_dwell_s=lambda d, p, o: 0.0, upload_s=lambda: 0.0,
                        p_move_w=1.0, p_hover_w=1.0, device_states=defaultdict(lambda: DOCK),
                        **kw)
    return FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0, ferry=phys)


def test_fedex_plans_the_closed_tour_from_the_dock_on_the_mission_clock():
    """Design §3.2: start = depot = dock in sim mode, so a closed tour. At the
    dock the ferry route is the very route both legacy modes plan."""
    ring = _ring()
    env = SelectorEnv(mule_pose=DOCK, now=0.0)
    legacy_model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)
    for depot in (None, DOCK):
        legacy = FedExCarpPolicy(depot=depot).admit_and_order(
            ring, {}, env, feasibility_model=legacy_model)
        pol = FedExCarpPolicy(depot=depot)
        ferry = pol.admit_and_order(ring, {}, env, mission_deadline_ts=1e9,
                                    feasibility_model=_unit_ferry())
        assert [id(w) for w in ferry] == [id(w) for w in legacy]
        # Without dwell or upload the diagnostics are the legacy tour's.
        legacy_pol = FedExCarpPolicy(depot=depot)
        legacy_pol.admit_and_order(ring, {}, env, mission_deadline_ts=1e9,
                                   feasibility_model=legacy_model)
        assert pol.last_tour_cost_s == pytest.approx(legacy_pol.last_tour_cost_s, rel=1e-12)
        assert pol.last_return_leg_s == pytest.approx(legacy_pol.last_return_leg_s, rel=1e-12)


def test_fedex_on_the_mission_clock_flies_home_to_the_ferry_dock():
    """A mule away from the dock plans the shortest path home through every
    contact, not the closed tour around where it happens to be; a depot other
    than the ferry dock would plan a tour the mule does not fly: refused."""
    ring = _ring()
    away = (100.0, 0.0, 0.0)
    env = SelectorEnv(mule_pose=away, now=0.0)
    pol = FedExCarpPolicy()
    route = pol.admit_and_order(ring, {}, env, mission_deadline_ts=1e9,
                                feasibility_model=_unit_ferry())
    assert [id(w) for w in route] == [id(w) for w in order_contacts(ring, away, end=DOCK)]
    last = route[-1].position
    assert pol.last_return_leg_s == math.dist(last, DOCK)
    with pytest.raises(ValueError, match="ferry dock"):
        FedExCarpPolicy(depot=(5.0, 0.0, 0.0)).admit_and_order(
            ring, {}, env, feasibility_model=_unit_ferry())
    # The legacy model keeps the legacy depot modes.
    FedExCarpPolicy(depot=(5.0, 0.0, 0.0)).admit_and_order(
        ring, {}, env, feasibility_model=FeasibilityModel())
