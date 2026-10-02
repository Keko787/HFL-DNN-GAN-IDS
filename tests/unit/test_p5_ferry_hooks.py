"""FeRRy Phase 5 (unit U4): the ferry runtime's pair and E3 readers (``hermes/mule/ferry.py``).

What is pinned (the Phase 5 spec: "Additive only", other choices 4 and 11;
critic A3, B7 and C6):

* ``stop_contexts`` prices each candidate next stop from the stop the mule is
  at as the departure check will price it: the leg on the flight model (the
  planner's transit), the dwell of b̄'s model at the mean SNR for the pass
  asked (the default band follows ``set_band``; an explicit band prices that
  class), and each class's median mean SNR over the members, in link order;
  home is the leg to the dock alone. The prices make U0's ``StopContext`` and
  ``PairView``.
* ``class_offsets_db`` differences each link before the median (critic C6):
  by hand, and ``I_b(t)`` plus the members' median shadowing on every class,
  so the classes' differences are the interference's; on layouts where the
  members' distances differ, ``median(realized) - median(mean)`` is another
  number, which the test would catch. A lone member's offset is what
  ``observe`` reads less its mean.
* ``e3_observation`` is Chen's observation from the pose (critic B7): the
  realized SNR within R_planar of the pose (inclusive) and the mean SNR beyond
  it; the reachable share by the gate's test from the pose (the floor
  inclusive); the remaining share; the offsets, the leg on the flight model's
  metric and the return's energy; the sortie's fields, the band the runtime
  flies, and ``energy_ref_j``, which is ``l1_state``'s reference. A band
  asked is priced throughout (its reach and its realized SNR), as a runtime
  flying it sees it, and ``t_s`` is the view's time, the runtime's clock
  aside.
* All four are pure: no clock charge, no band moved, the channel-free control
  refused.
* The lazy import: in a fresh interpreter no legacy call, nor the pair readers,
  loads ``next_stop`` or the plan package; ``e3_observation`` loads
  ``next_stop`` alone. ``next_stop`` is imported only inside a function or for
  the type checker.
* **Legacy identity (Freeze Rule 1).** On 27 cases (9 configurations x 3
  seeds) every output of the runtime, at its defaults and with its band moved
  to each class as plan mode moves it, is bit for bit 386c275's, loaded from
  git; and every statement and definition 386c275's ``ferry.py`` had is
  unchanged, the module's additions being the four readers, ``StopPrice`` and
  its import and export lines.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hermes.mule.ferry as ferry_module
from hermes.l1.channel_model import ContactChannel
from hermes.l1.contact_link import CLASSES, CLASSES_WITH_10MHZ
from hermes.l1.mission_clock import SIM_EPOCH_S, MissionClock
from hermes.mission.contact_plan import planar_distance_m
from hermes.mule.ferry import FerryRuntime, FerrySpec, StopPrice
from hermes.scheduler.plan.types import ArrivalView, PairView, StopContext
from hermes.scheduler.policies.next_stop import E3Stop, E3View
from hermes.scheduler.stages.s3b_feasibility import (
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    FlightState,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass

COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
RF = 60.0
SEED = 11
DOCK = (0.0, 0.0, 0.0)
T0 = SIM_EPOCH_S
REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "386c275"
GRID = (0.0, 10.0, 30.0, 59.0, 60.0, 70.0, 110.0, 119.0, 150.0, 200.0, 231.0, 300.0)
READERS = ("stop_contexts", "class_offsets_db", "energy_ref_j", "e3_observation")


def _wp(x, y, *devs, deadline=T0 + 1e4):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=deadline)


def _spec(**kw):
    kw.setdefault("rf_range_m", RF)
    kw.setdefault("seed", SEED)
    return FerrySpec.from_config(**kw)


def _runtime(band="wide", clock=None, **kw):
    rt = FerryRuntime(_spec(contact_band=band, **kw), clock, rf_range_m=RF)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    return rt


def _field(rng, n_stops, spread=250.0):
    """Random stops of 1-5 members each, members up to ``spread`` from their stop."""
    stops, pos, k = [], {}, 0
    for _ in range(n_stops):
        c = (rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0), 0.0)
        members = []
        for _ in range(rng.randint(1, 5)):
            r, a = rng.uniform(0.0, spread), rng.uniform(0.0, 2.0 * math.pi)
            did = DeviceID(f"d{k}")
            k += 1
            pos[did] = (c[0] + r * math.cos(a), c[1] + r * math.sin(a), 0.0)
            members.append(did)
        stops.append(ContactWaypoint(position=c, devices=tuple(members),
                                     bucket=Bucket.SCHEDULED_THIS_ROUND,
                                     deadline_ts=T0 + rng.uniform(10.0, 400.0)))
    return stops, pos


def _charged(rt):
    clock = rt.clock
    return (clock(), dict(clock.ledger()), rt.band, rt.range_planar_m, rt.band_index,
            rt.carrier)


# --------------------------------------------------------------------------- #
# stop_contexts: the pair slot's second half
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(6))
def test_each_candidate_is_priced_as_the_departure_check_will_price_it(seed):
    """The leg is the planner's transit and the clock's charge; the dwell is
    b̄'s model at the mean SNR, the one the departure check folds (the commit
    makes b̄'s model the scheduler's); each class's SNR is the median of the
    members' mean SNR, as ``annotate`` gives each member's."""
    rng = random.Random(seed)
    kw = dict(payload_bytes=rng.choice([None, 1_000_000]),
              contact_regime=rng.choice(["clean", "jittery"]),
              band_classes=rng.choice([CLASSES, CLASSES_WITH_10MHZ]))
    clock = MissionClock()
    rt = _runtime("wide", clock=clock, **kw)
    link, flight = rt.spec.link, rt.spec.flight
    stops, pos = _field(rng, 6)
    here, rest = stops[0], stops[1:]
    before = _charged(rt)
    for committed in link.names:
        rt.set_band(committed)
        model = rt.feasibility_model()
        bound = dataclasses.replace(model, ferry=model.ferry.bind(pos))
        for pass_kind in (COLLECT, DELIVER, "collect"):
            prices = rt.stop_contexts(here, rest, pos, pass_kind=pass_kind)
            assert len(prices) == len(rest) and all(isinstance(p, StopPrice) for p in prices)
            for wp, p in zip(rest, prices):
                leg = bound.leg(here.position, wp, pass_kind=pass_kind)
                assert p.travel_s == leg.transit_s == flight.leg_s(here.position, wp.position)
                assert p.pred_dwell_s == leg.dwell_s
                assert p.pred_snr_db == tuple(
                    float(statistics.median(
                        link.mean_snr_db(c, planar_distance_m(wp.position, pos[d]))
                        for d in wp.devices))
                    for c in link.names)
                assert p.pred_snr_db == tuple(
                    float(statistics.median(rt.annotate([wp], pos, band=c)[0].pred_snr_db))
                    for c in link.names)
                travel, dwell, snr = p                  # still the design's triple
                assert (travel, dwell, snr) == (p.travel_s, p.pred_dwell_s, p.pred_snr_db)
        # The departure check's own fold prices the same dwell on b̄.
        fold = bound.fold(rest, FlightState(here.position, T0), rule=RULE_DEADLINE_BUDGET,
                          budget_end=None, skip=False)
        for v, p in zip(fold.verdicts, rt.stop_contexts(here, rest, pos, pass_kind=COLLECT)):
            assert v.finish == v.arrival + p.pred_dwell_s
    rt.set_band("wide")
    assert _charged(rt) == before


def test_home_is_the_leg_to_the_dock_alone():
    rt = _runtime("medium", clock=MissionClock())
    here = _wp(40, -30, "a")
    pos = {DeviceID("a"): (45.0, -30.0, 0.0)}
    for pass_kind in (COLLECT, DELIVER):
        (home,) = rt.stop_contexts(here, [], pos, pass_kind=pass_kind)
        assert home == StopPrice(rt.spec.flight.leg_s(here.position, DOCK), 0.0, ())
        assert home.travel_s == 10.0                        # 50 m at 5 m/s
    ctx = StopContext.home(home.travel_s)
    assert ctx.is_home and ctx.travel_s == home.travel_s


def test_the_dwell_follows_the_band_and_the_pass_asked():
    """The default prices :attr:`band`, b̄ once set; an explicit band prices that
    class without moving the runtime's; the pass is required, and Pass 1
    prices push plus update."""
    rt = _runtime("wide", payload_bytes=1_000_000)
    stops, pos = _field(random.Random(7), 4, spread=200.0)
    here, rest = stops[0], stops[1:]
    with pytest.raises(TypeError, match="pass_kind"):
        rt.stop_contexts(here, rest, pos)
    for name in rt.spec.link.names:
        own = _runtime(name, payload_bytes=1_000_000)
        rt.set_band(name)
        assert rt.stop_contexts(here, rest, pos, pass_kind=COLLECT) == \
            own.stop_contexts(here, rest, pos, pass_kind=COLLECT)
        rt.set_band("wide")
        assert rt.stop_contexts(here, rest, pos, pass_kind=COLLECT, band=name) == \
            own.stop_contexts(here, rest, pos, pass_kind=COLLECT)
        assert rt.band == "wide"
    one = _wp(0, 0, "a")
    near = {DeviceID("a"): (0.0, 0.0, 0.0), DeviceID("b"): (5.0, 0.0, 0.0)}
    (p1,) = rt.stop_contexts(one, [_wp(5, 0, "b")], near, pass_kind=COLLECT)
    (p2,) = rt.stop_contexts(one, [_wp(5, 0, "b")], near, pass_kind=DELIVER)
    assert rt.session_bytes(COLLECT) == 2 * rt.session_bytes(DELIVER)
    assert p1.pred_dwell_s == pytest.approx(2.0 * p2.pred_dwell_s) and p1.pred_dwell_s > 0.0


def test_the_prices_make_the_pair_view():
    """U0's hand-off: per remainder stop the leg, the dwell on b̄ and each
    class's SNR in link order, which ``StopContext`` and ``PairView`` take as
    they are, with the observation and the offsets at the stop."""
    clock = MissionClock()
    clock.advance(42.0, "transit")
    rt = _runtime("wide", clock=clock, contact_regime="jittery", payload_bytes=1_000_000,
                  band_classes=CLASSES_WITH_10MHZ)
    rt.set_band("medium")
    stops, pos = _field(random.Random(3), 4, spread=120.0)
    here, rest = stops[0], stops[1:]
    t = clock()
    prices = rt.stop_contexts(here, rest, pos, pass_kind=COLLECT)
    ctxs = tuple(StopContext(stop=wp, index=i, travel_s=p.travel_s, pred_dwell_s=p.pred_dwell_s,
                             pred_snr_db=p.pred_snr_db, capped=False, exempt=False, age=1.0,
                             on_time=1.0, weight=1.0)
                 for i, (wp, p) in enumerate(zip(rest, prices)))
    view = PairView(arrival=rt.arrival_view(here, pos, t, pass_kind=COLLECT), pose=here.position,
                    observed_snr_db=rt.observe(here, pos, t).class_snr_db,
                    offsets_db=rt.class_offsets_db(here, pos, t), previous_offsets_db=None,
                    previous_age_s=None,
                    period_s=rt.spec.contact_channel.interference_period_s, clock_s=t,
                    budget_end=t + 300.0, budget_s=300.0, t_ref_s=200.0, energy_j=0.0,
                    energy_ref_j=rt.energy_ref_j(300.0), stops=ctxs,
                    demand=sum(len(wp.devices) for wp in stops), demand_weight=1.0, cap_s=None)
    assert view.remainder == tuple(rest) and len(view.offsets_db) == 4
    assert all(len(ctx.pred_snr_db) == 4 for ctx in view.stops)


# --------------------------------------------------------------------------- #
# class_offsets_db: the per-link offsets (critic C6)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("keying", ["time", "position"])
@pytest.mark.parametrize("seed", range(4))
def test_the_offsets_difference_each_link_before_the_median(seed, keying):
    rng = random.Random(seed)
    classes = rng.choice([CLASSES, CLASSES_WITH_10MHZ])
    clock = MissionClock()
    rt = _runtime("wide", clock=clock, contact_regime="jittery", shadow_keying=keying,
                  band_classes=classes)
    chan, link = rt.spec.contact_channel, rt.spec.link
    stops, pos = _field(rng, 8, spread=150.0)
    before = _charged(rt)
    other = 0
    for wp in stops:
        t = T0 + rng.uniform(0.0, 900.0)
        stop = wp.position
        dist = [(j, planar_distance_m(stop, pos[j])) for j in wp.devices]
        got = rt.class_offsets_db(wp, pos, t)
        assert got == tuple(
            float(statistics.median(chan.snr_db(t, c, d, link_key=j, stop_pos=stop)
                                    - link.mean_snr_db(c, d) for j, d in dist))
            for c in link.names)
        shadow = statistics.median(chan.shadow_db(t, j, stop_pos=stop) for j, _ in dist)
        for c, offset in zip(link.names, got):
            assert offset == pytest.approx(chan.interference_db(t, c) + shadow, abs=1e-9)
        medians = tuple(
            statistics.median(chan.snr_db(t, c, d, link_key=j, stop_pos=stop) for j, d in dist)
            - statistics.median(link.mean_snr_db(c, d) for _, d in dist)
            for c in link.names)
        other += any(abs(a - b) > 0.1 for a, b in zip(got, medians))
    assert other >= 1          # the medians' difference is another number: C6 is decisive
    assert _charged(rt) == before


def test_a_lone_members_offset_is_what_observe_reads_less_its_mean():
    clock = MissionClock()
    clock.advance(77.5, "transit")
    rt = _runtime("narrow", clock=clock, contact_regime="jittery")
    wp = _wp(10, 20, "a")
    pos = {DeviceID("a"): (40.0, 60.0, 0.0)}
    obs = rt.observe(wp, pos, clock())
    d = planar_distance_m(wp.position, pos[DeviceID("a")])
    assert rt.class_offsets_db(wp, pos, clock()) == tuple(
        s - rt.spec.link.mean_snr_db(c, d) for c, s in zip(rt.spec.link.names, obs.class_snr_db))


# --------------------------------------------------------------------------- #
# e3_observation: Chen's observation from the pose (critic B7)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("keying", ["time", "position"])
def test_e3_sees_each_stop_from_its_pose(keying):
    """Under position-keyed shadowing the pose keys the shadowing, not the stop."""
    clock = MissionClock()
    clock.advance(123.0, "transit")
    rt = _runtime("medium", clock=clock, contact_regime="jittery", payload_bytes=1_000_000,
                  energy_capacity_j=6e4, shadow_keying=keying)
    link, chan, flight = rt.spec.link, rt.spec.contact_channel, rt.spec.flight
    reach = link.range_planar_m("medium")
    stops, pos = _field(random.Random(5), 5, spread=200.0)
    pose, t = (20.0, -35.0, 0.0), clock()
    before = _charged(rt)
    view = rt.e3_observation(pose, stops, pos, t, demand=17, budget_end=t + 90.0, budget_s=150.0,
                             energy_j=4321.0, collected=stops[2].devices[:1])
    assert isinstance(view, E3View) and all(isinstance(s, E3Stop) for s in view.stops)
    assert (view.band, view.demand, view.clock_s, view.budget_end, view.budget_s,
            view.energy_j, view.energy_ref_j) == ("medium", 17, t, t + 90.0, 150.0, 4321.0, 6e4)
    assert len(view.stops) == len(stops)
    for i, (wp, seen) in enumerate(zip(stops, view.stops)):
        snr, reachable = [], 0
        for j in wp.devices:
            d = planar_distance_m(pose, pos[j])
            if d <= reach:
                s = chan.snr_db(t, "medium", d, link_key=j, stop_pos=pose)
                reachable += s >= link.snr_floor_db
            else:
                s = link.mean_snr_db("medium", d)
            snr.append(s)
        n = len(wp.devices)
        assert seen.members == n
        assert seen.remaining == ((n - 1) / n if i == 2 else 1.0)
        assert seen.snr_db == float(statistics.median(snr))
        assert seen.reachable == reachable / n
        assert (seen.dx_m, seen.dy_m) == (wp.position[0] - pose[0], wp.position[1] - pose[1])
        assert seen.distance_m / flight.cruise_speed_m_s == flight.leg_s(pose, wp.position)
        assert seen.return_energy_j == flight.energy.p_move_w * flight.leg_s(wp.position, DOCK)
    assert _charged(rt) == before


def test_beyond_reach_the_snr_is_the_mean_a_radio_map_gives():
    """Critic B7 iv: the realized SNR out of reach is the simulator's alone. A
    member at R_planar of the pose is within reach (the gate's inclusive test);
    one a nanometre beyond is priced at its mean. At some time the median over
    realized SNRs alone is another number, so the rule decides."""
    clock = MissionClock()
    rt = _runtime("wide", clock=clock, contact_regime="jittery")
    link, chan = rt.spec.link, rt.spec.contact_channel
    pose = (0.0, 0.0, 0.0)
    wp = _wp(100, 0, "near", "edge", "out", "far")
    pos = {DeviceID("near"): (10.0, 0.0, 0.0), DeviceID("edge"): (RF, 0.0, 0.0),
           DeviceID("out"): (RF + 1e-9, 0.0, 0.0), DeviceID("far"): (0.0, 200.0, 0.0)}
    differs = 0
    for k in range(40):
        t = T0 + 7.5 * k
        view = rt.e3_observation(pose, [wp], pos, t, demand=4, budget_end=None, budget_s=None,
                                 energy_j=0.0)
        realized = {j: chan.snr_db(t, "wide", planar_distance_m(pose, pos[j]), link_key=j,
                                   stop_pos=pose) for j in wp.devices}
        mean = {j: link.mean_snr_db("wide", planar_distance_m(pose, pos[j])) for j in wp.devices}
        expect = [realized["near"], realized["edge"], mean["out"], mean["far"]]
        (seen,) = view.stops
        assert seen.snr_db == float(statistics.median(expect))
        assert seen.reachable == sum(realized[j] >= link.snr_floor_db
                                     for j in ("near", "edge")) / 4
        differs += seen.snr_db != float(statistics.median(realized.values()))
    assert differs >= 10


def test_the_energy_reference_is_l1_states():
    """The capacity, else P_hover times the budget, else None, as ``l1_state``
    takes it (slot 7 = 1 - E/E_ref); a 0 reference (P_hover = 0) is returned as
    computed, and the views store it as None."""
    obs_wp, pos = _wp(0, 0, "a"), {DeviceID("a"): (5.0, 0.0, 0.0)}
    cases = [(dict(energy_capacity_j=5e4), 90.0, 5e4), (dict(energy_capacity_j=5e4), None, 5e4),
             (dict(), 90.0, 168.5 * 90.0), (dict(), None, None),
             (dict(p_hover_w=0.0), 90.0, 0.0)]
    for kw, budget, expect in cases:
        clock = MissionClock()
        rt = _runtime("wide", clock=clock, **kw)
        assert rt.energy_ref_j(budget) == expect
        state = rt.l1_state(rt.observe(obs_wp, pos, clock()), pose=DOCK, energy_j=1234.0,
                            budget_s=budget)
        assert state[7] == (np.float32(1.0) if not expect else np.float32(1.0 - 1234.0 / expect))
        view = rt.e3_observation(DOCK, [obs_wp], pos, clock(), demand=1, budget_end=None,
                                 budget_s=budget, energy_j=1234.0)
        assert view.energy_ref_j == (expect or None)


def test_e3_flies_its_band_and_needs_a_stop():
    clock = MissionClock()
    rt = _runtime("narrow", clock=clock)
    wp, pos = _wp(30, 0, "a"), {DeviceID("a"): (35.0, 0.0, 0.0)}
    assert rt.e3_observation(DOCK, [wp], pos, clock(), demand=1, budget_end=None, budget_s=None,
                             energy_j=0.0).band == "narrow"
    medium = rt.e3_observation(DOCK, [wp], pos, clock(), demand=1, budget_end=None,
                               budget_s=None, energy_j=0.0, band="medium")
    assert medium.band == "medium" and rt.band == "narrow"
    with pytest.raises(ValueError, match="at least one"):
        rt.e3_observation(DOCK, [], pos, clock(), demand=1, budget_end=None, budget_s=None,
                          energy_j=0.0)


def _between_reaches(link, pose):
    """One single-member stop per distance from ``pose``: within the shortest
    reach, between every two consecutive classes' reaches, and beyond the
    longest. Each stop's median is its member's SNR, and any two classes
    disagree on some member's reach."""
    reaches = sorted(link.range_planar_m(c) for c in link.names)
    dists = ([0.5 * reaches[0]] + [0.5 * (a + b) for a, b in zip(reaches, reaches[1:])]
             + [reaches[-1] + 40.0])
    stops, pos = [], {}
    for k, d in enumerate(dists):
        did, angle = DeviceID(f"m{k}"), 0.7 + 1.3 * k
        pos[did] = (pose[0] + d * math.cos(angle), pose[1] + d * math.sin(angle), 0.0)
        stops.append(_wp(pose[0] + 30.0 * k, pose[1] - 20.0, did))
    return stops, pos, reaches


@pytest.mark.parametrize("classes", [CLASSES, CLASSES_WITH_10MHZ], ids=["3-classes", "4-classes"])
def test_e3_on_a_band_asked_sees_what_that_bands_runtime_sees(classes):
    """An explicit band prices that class throughout, the reach from the pose
    and the realized SNR alike, and moves nothing: on a runtime flying any
    class, asking for class c gives the view a runtime flying c gives. The
    members sit between the classes' reaches, so a reach or an SNR taken from
    the runtime's own class would show."""
    clock = MissionClock()
    clock.advance(55.0, "transit")
    pose, t = (15.0, 25.0, 0.0), clock()
    kw = dict(demand=9, budget_end=t + 120.0, budget_s=200.0, energy_j=987.0)
    runtimes = {c: _runtime(c, clock=clock, contact_regime="jittery", band_classes=classes)
                for c in classes}
    stops, pos, reaches = _between_reaches(runtimes[classes[0]].spec.link, pose)
    assert len(set(reaches)) == len(classes)
    own = {c: rt.e3_observation(pose, stops, pos, t, **kw) for c, rt in runtimes.items()}
    for flown, rt in runtimes.items():
        before = _charged(rt)
        for asked in classes:
            view = rt.e3_observation(pose, stops, pos, t, band=asked, **kw)
            assert view == own[asked] and view.band == asked
            if asked != flown:
                assert view.stops != own[flown].stops
        assert _charged(rt) == before and rt.band == flown


def test_e3_observes_at_the_time_it_is_given():
    """``t_s`` is the observation's time: the view's clock carries it and the
    realized SNRs are read at it, whatever the runtime's clock reads (which
    the call does not advance)."""
    clock = MissionClock()
    rt = _runtime("medium", clock=clock, contact_regime="jittery")
    link, chan, reach = rt.spec.link, rt.spec.contact_channel, rt.range_planar_m
    pose = (0.0, 0.0, 0.0)
    stops, pos = _field(random.Random(9), 5, spread=150.0)

    def by_hand(wp, t):
        snr = []
        for j in wp.devices:
            d = planar_distance_m(pose, pos[j])
            snr.append(chan.snr_db(t, "medium", d, link_key=j, stop_pos=pose) if d <= reach
                       else link.mean_snr_db("medium", d))
        return float(statistics.median(snr))

    before = _charged(rt)
    t = clock() + 250.0
    view = rt.e3_observation(pose, stops, pos, t, demand=3, budget_end=None, budget_s=None,
                             energy_j=0.0)
    assert view.clock_s == t and clock() != t
    assert [s.snr_db for s in view.stops] == [by_hand(wp, t) for wp in stops]
    assert [s.snr_db for s in view.stops] != [by_hand(wp, clock()) for wp in stops]
    assert _charged(rt) == before


def test_e3_counts_a_member_at_the_floor_as_reachable(monkeypatch):
    """The reachable share is the contact gate's test from the pose, within
    reach and at or above the floor (``contact_plan``'s ``snr >= floor``): a
    member exactly at the floor counts, one a step under it does not."""
    clock = MissionClock()
    rt = _runtime("wide", clock=clock)
    floor = rt.spec.link.snr_floor_db
    fixed = {DeviceID("at"): floor, DeviceID("under"): math.nextafter(floor, -math.inf)}
    realized = ContactChannel.snr_db

    def snr_db(self, t, band, d_planar, link_key, *, stop_pos=None):
        if link_key in fixed:
            return fixed[link_key]
        return realized(self, t, band, d_planar, link_key, stop_pos=stop_pos)

    monkeypatch.setattr(ContactChannel, "snr_db", snr_db)
    wp = _wp(20, 0, "at", "under")
    pos = {DeviceID("at"): (10.0, 0.0, 0.0), DeviceID("under"): (20.0, 0.0, 0.0)}
    (seen,) = rt.e3_observation(DOCK, [wp], pos, clock(), demand=2, budget_end=None,
                                budget_s=None, energy_j=0.0).stops
    assert seen.reachable == 0.5
    assert seen.snr_db == float(statistics.median(fixed.values()))


@pytest.mark.parametrize("reader", READERS[:2] + READERS[3:])
def test_the_channel_free_control_is_refused(reader):
    clock = MissionClock()
    wp, pos = _wp(0, 0, "a"), {DeviceID("a"): (1.0, 0.0, 0.0)}
    for spec in (FerrySpec(), _spec()):
        rt = FerryRuntime(spec, clock, rf_range_m=RF)
        calls = {
            "stop_contexts": lambda: rt.stop_contexts(wp, [], pos, pass_kind=COLLECT),
            "class_offsets_db": lambda: rt.class_offsets_db(wp, pos, clock()),
            "e3_observation": lambda: rt.e3_observation(DOCK, [wp], pos, clock(), demand=1,
                                                        budget_end=None, budget_s=None,
                                                        energy_j=0.0),
        }
        with pytest.raises(ValueError, match="channel-free"):
            calls[reader]()


# --------------------------------------------------------------------------- #
# The lazy import (critic B7 iii)
# --------------------------------------------------------------------------- #

def _module_imports(path: Path):
    """(module, where) for every import in ``path``: 'top', 'checking' (under
    ``if TYPE_CHECKING``) or 'function' (inside a def)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    where = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "TYPE_CHECKING":
            for sub in ast.walk(node):
                where.setdefault(id(sub), "checking")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                where.setdefault(id(sub), "function")
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((alias.name, where.get(id(node), "top")) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.append(("." * node.level + (node.module or ""), where.get(id(node), "top")))
    return out


def test_only_e3_observation_loads_the_next_stop_protocol():
    imports = _module_imports(REPO / "hermes/mule/ferry.py")
    protocol = [w for m, w in imports if m.endswith("next_stop")]
    assert sorted(protocol) == ["checking", "function"]
    assert all(w in ("checking", "function") for m, w in imports
               if m.startswith("hermes.scheduler.plan"))
    code = "\n".join([
        "import sys",
        "import hermes.mule.ferry as f",
        "from hermes.l1.mission_clock import MissionClock",
        "from hermes.types import Bucket, ContactWaypoint, MissionPass",
        "def watch():",
        "    return sorted(m for m in sys.modules",
        "                  if m.startswith('hermes.scheduler.plan') or m.endswith('next_stop'))",
        "before = watch()",
        "assert not any(m.endswith('next_stop') for m in before), before",
        "def wp(x, d):",
        "    return ContactWaypoint(position=(x, 0.0, 0.0), devices=(d,),",
        "                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=1e9)",
        "a, b = wp(10.0, 'a'), wp(40.0, 'b')",
        "pos = {'a': (12.0, 0.0, 0.0), 'b': (45.0, 0.0, 0.0)}",
        "clock = MissionClock()",
        "spec = f.FerrySpec.from_config(rf_range_m=60.0, seed=1, contact_band='narrow')",
        "rt = f.FerryRuntime(spec, clock, rf_range_m=60.0)",
        "rt.feasibility_model(); rt.annotate([a], pos); rt.observe(a, pos, clock())",
        "rt.contact_plan(a, pos, pass_kind=MissionPass.COLLECT, mission_round=1)",
        "rt.l1_state(rt.observe(a, pos, clock()), pose=(0.0, 0.0, 0.0), energy_j=1.0,",
        "            budget_s=60.0)",
        "rt.stop_contexts(a, [b], pos, pass_kind=MissionPass.COLLECT)",
        "rt.stop_contexts(a, [], pos, pass_kind=MissionPass.COLLECT)",
        "rt.class_offsets_db(a, pos, clock()); rt.energy_ref_j(60.0)",
        "assert watch() == before, watch()",
        "view = rt.e3_observation((0.0, 0.0, 0.0), [a, b], pos, clock(), demand=2,",
        "                         budget_end=None, budget_s=None, energy_j=0.0)",
        "assert type(view).__module__ == 'hermes.scheduler.policies.next_stop'",
        "after = watch()",
        "assert after == sorted(before + ['hermes.scheduler.policies.next_stop']), after",
        "print('ok')",
    ])
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-3000:]


# --------------------------------------------------------------------------- #
# Legacy identity (Freeze Rule 1)
# --------------------------------------------------------------------------- #

def _ref_blob() -> bytes:
    try:
        return subprocess.run(["git", "show", f"{REF_COMMIT}:hermes/mule/ferry.py"], cwd=REPO,
                              capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:          # pragma: no cover - no git
        pytest.skip(f"git cannot show {REF_COMMIT}'s ferry.py: {e}")


@pytest.fixture(scope="module")
def ref_ferry(tmp_path_factory):
    """``hermes/mule/ferry.py`` as it was at 386c275, as a module of its own.

    It imports the live modules it depends on, so any difference below comes
    from ferry.py alone. Skipped where git or the commit is not available.
    """
    path = tmp_path_factory.mktemp("ref_ferry") / "_ref_ferry.py"
    path.write_bytes(_ref_blob())
    name = f"_p5_ref_{REF_COMMIT}_hermes_mule_ferry"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # dataclasses resolve annotations through it
    try:
        spec.loader.exec_module(mod)
        assert hasattr(mod.FerryRuntime, "set_band")                 # Phase 4 is in it
        assert not any(hasattr(mod.FerryRuntime, r) for r in READERS)  # Phase 5 is not
        yield mod
    finally:
        sys.modules.pop(name, None)


_AVAIL = {f"d{i}": round(0.15 + 0.85 * ((i * 37) % 100) / 100.0, 3) for i in range(40)}
_CONFIGS = {
    "control": None,                                          # FerrySpec(): no link at all
    "no-band": dict(),
    "wide": dict(contact_band="wide"),
    "medium-1MB": dict(contact_band="medium", payload_bytes=1_000_000),
    "narrow-seconds-energy": dict(contact_band="narrow", payload_bytes=1_000_000,
                                  backhaul_model="seconds", backhaul_period=400.0,
                                  energy_capacity_j=6e4),
    "wide-channel-jittery-delivery": dict(contact_band="wide", contact_reliability_source="channel",
                                          device_availability=_AVAIL, contact_regime="jittery",
                                          deadline_bounds="delivery"),
    "narrow-position-adaptive": dict(contact_band="narrow", shadow_keying="position",
                                     deadline_bounds="delivery_per_stop", backhaul_model="seconds",
                                     backhaul_period=250.0, backhaul_policy="adaptive",
                                     backhaul_regime="jittery"),
    "medium_wide-10MB": dict(contact_band="medium_wide", band_classes=CLASSES_WITH_10MHZ,
                             payload_bytes=10_000_000),
    "medium-floor-n3": dict(contact_band="medium", snr_floor_db=-10.0, n_pl=3.0),
}


def _plan_fields(plan):
    """A contact plan's decisions and prices, comparable across runtimes."""
    fields = (plan.arrival_ts, plan.targets, plan.unreachable, plan.band, plan.band_index,
              plan.stop, tuple(sorted(plan.snr_db.items())), plan.snr_floor_db,
              tuple(sorted(plan.drop_uplink)), plan.listen_s, plan.session_time_s,
              plan.payload_bytes)
    if plan.band is None:
        return fields
    return fields + (
        tuple(plan.dwell_s(n, s)
              for n in (1_000, 2_000_000) for s in (-10.0, -6.7, 0.0, 12.3, 25.0)),
        tuple(plan.snr_at(j, plan.arrival_ts + dt) for j in plan.members for dt in (0.0, 0.5, 3.7)),
    )


def _model_prices(model, stops, pos):
    """What a class model prices: its physics, the member dwell over a grid, the
    legs from the dock and folds of the stops under both gated rules."""
    ferry = model.ferry
    bound = dataclasses.replace(model, ferry=ferry.bind(pos))
    return (
        (ferry.dock, ferry.range_m, ferry.p_move_w, ferry.p_hover_w, ferry.energy_capacity_j,
         ferry.deadline_bounds, ferry.upload_time_s()),
        tuple(ferry.member_dwell_s(d, pk, off)
              for d in GRID for pk in (COLLECT, DELIVER) for off in (0.0, -2.0)),
        tuple(bound.leg(ferry.dock, wp, pass_kind=pk) for wp in stops for pk in (COLLECT, DELIVER)),
        tuple(bound.fold(stops, FlightState(ferry.dock, T0), rule=rule, budget_end=T0 + 200.0,
                         skip=skip)
              for rule in (RULE_DEADLINE_BUDGET, RULE_BUDGET) for skip in (True, False)),
    )


def _record(mod, cfg, seed):
    """Every output of a runtime built by ``mod``: at its defaults (the legacy
    faces), then with its band moved to each class as plan mode moves it."""
    rng = random.Random(seed)
    stops, pos = _field(rng, 4, spread=400.0)
    spec = mod.FerrySpec() if cfg is None else mod.FerrySpec.from_config(
        rf_range_m=RF, seed=seed, **cfg)
    clock = MissionClock()
    rt = mod.FerryRuntime(spec, clock, rf_range_m=RF, session_time_s=1.0)
    out = [("attrs", rt.band, rt.range_planar_m, rt.band_index, rt.banded),
           ("describe", spec.describe())]
    for theta, synth in ((18_756, 64), (1_000_000, 0)):
        rt.set_payload(theta_bytes=theta, synth_bytes=synth)
        phys = rt.physics()
        out.append(("pricing", rt.session_bytes(COLLECT), rt.session_bytes(DELIVER),
                    rt.predicted_upload_bytes(), rt.predicted_upload_s(), rt.held_carrier(),
                    None if spec.link is None else rt.upload_cap_s(theta)))
        out.append(("physics", phys.dock, phys.range_m, phys.p_move_w, phys.p_hover_w,
                    phys.energy_capacity_j, phys.deadline_bounds, phys.upload_time_s(),
                    phys.member_dwell_s is None, phys == rt.physics()))
        if rt.banded:
            grid = [(d, pk, off) for d in GRID + (400.0,) for pk in (COLLECT, DELIVER)
                    for off in (0.0, -3.0, 2.5)]
            out.append(("dwell", tuple(rt.member_dwell_s(*g) for g in grid),
                        tuple(phys.member_dwell_s(*g) for g in grid)))
        model = rt.feasibility_model()
        bound = dataclasses.replace(model, ferry=model.ferry.bind(pos))
        out.append(("legs", tuple(bound.leg(spec.flight.dock, wp, pass_kind=pk)
                                  for wp in stops for pk in (COLLECT, DELIVER))))
        for rule in (RULE_DEADLINE_BUDGET, RULE_BUDGET):
            for budget in (None, T0 + 60.0, T0 + 300.0):
                for skip in (True, False):
                    f = bound.fold(stops, FlightState(spec.flight.dock, T0), rule=rule,
                                   budget_end=budget, skip=skip)
                    out.append(("fold", rule, budget, skip, f.route, f.rejected, f.verdicts,
                                f.state, f.home))
    if spec.link is not None:
        planning = spec.feasibility_model(rf_range_m=RF, theta_bytes=18_756, synth_bytes=64)
        planning = dataclasses.replace(planning, ferry=planning.ferry.bind(pos))
        out.append(("spec-model", tuple(planning.leg(spec.flight.dock, wp) for wp in stops)))
    out.append(("annotate", tuple((a.band, a.range_m, a.pred_snr_db, a == wp)
                                  for a, wp in zip(rt.annotate(stops, pos), stops))))
    for k in range(3):
        clock.advance(rng.uniform(1.0, 120.0), "transit")
        for wp in stops:
            obs = rt.observe(wp, pos, clock())
            out.append(("observe", dataclasses.astuple(obs)))
            if rt.banded:
                state = rt.l1_state(obs, pose=(1.0, 2.0, 0.0), energy_j=500.0, budget_s=60.0)
                out.append(("l1", tuple(state.tolist())))
            for pk in (COLLECT, DELIVER):
                plan = rt.contact_plan(wp, pos, pass_kind=pk, mission_round=k + 1)
                out.append(("plan", _plan_fields(plan)))
                if rt.banded:
                    out.append(("rates", rt.rates_bps([plan.snr_db[j] for j in wp.devices])))
        out.append(("drops", tuple(sorted(rt.uplink_drops(sorted(pos), k + 1)))))
        up = rt.charge_upload(rng.choice([0, 18_756, 1_000_000]))
        out.append(("upload", None if up is None else dataclasses.astuple(up), clock(),
                    dict(clock.ledger()), rt.carrier, rt.energy_j()))
    if rt.banded:
        # The plan face (Phase 4): each class's model and outage, then the band
        # moved to each class, with every read on it and on explicit classes.
        names = spec.link.names
        for c in rt.plan_classes():
            out.append(("class", c.name, c.index, c.radius_m, _model_prices(c.model, stops, pos),
                        tuple(c.outage(d) for d in GRID)))
        t = clock()
        for name in names:
            rt.set_band(name)
            out.append(("set_band", rt.band, rt.range_planar_m, rt.band_index,
                        _model_prices(rt.feasibility_model(), stops, pos),
                        tuple(rt.outage_probability(d) for d in GRID)))
            out.append(("annotate-b", tuple((a.band, a.range_m, a.pred_snr_db)
                                            for a in rt.annotate(stops, pos))))
            for wp in stops:
                out.append(("observe-b", dataclasses.astuple(rt.observe(wp, pos, t))))
                for pk in (COLLECT, DELIVER):
                    view = rt.arrival_view(wp, pos, t, pass_kind=pk)
                    assert isinstance(view, ArrivalView)
                    out.append(("view", view))
                for other in names:
                    plan = rt.contact_plan(wp, pos, pass_kind=COLLECT, mission_round=9,
                                           band=other)
                    out.append(("plan-on", other, _plan_fields(plan),
                                rt.rates_bps([plan.snr_db[j] for j in wp.devices], band=other)))
            out.append(("after", clock(), dict(clock.ledger())))
    return out


@pytest.mark.parametrize("seed", [3, 17, 38])
@pytest.mark.parametrize("cfg", list(_CONFIGS), ids=list(_CONFIGS))
def test_the_runtime_at_its_defaults_is_386c275s(ref_ferry, cfg, seed):
    """Freeze Rule 1, 27 cases: every output of the runtime (the pricing, the
    physics and its models, legs and folds, annotations, observations, the L1
    state, contact plans, rates, the availability draw, the upload, and on
    the plan face each class's model and outage, ``set_band`` and the arrival
    views) is bit for bit the 386c275 module's, on every class, both payload
    modes, both backhaul models and both reliability sources."""
    live = _record(ferry_module, _CONFIGS[cfg], seed)
    ref = _record(ref_ferry, _CONFIGS[cfg], seed)
    assert len(live) == len(ref)
    for a, b in zip(live, ref):
        assert a == b, a[0]


def _shape(source: str):
    """(module-level statements, {qualified name: definition}) as AST dumps."""
    tree = ast.parse(source)
    top, defs = [], {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            body = []
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defs[f"{node.name}.{sub.name}"] = ast.dump(sub)
                else:
                    body.append(ast.dump(sub))
            top.append(("class", node.name, tuple(body),
                        tuple(ast.dump(d) for d in node.decorator_list),
                        tuple(ast.dump(b) for b in node.bases)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = ast.dump(node)
            top.append(("def", node.name))
        else:
            top.append(node)
    return top, defs


def test_every_statement_386c275_had_is_unchanged():
    """Additive only: every definition is the recorded one; the module's other
    statements are too, except the docstring, the typing import, the
    type-checking block and ``__all__``, which only gain; and what is new is
    the four readers and ``StopPrice``."""
    ref_top, ref_defs = _shape(_ref_blob().decode("utf-8"))
    live_top, live_defs = _shape((REPO / "hermes/mule/ferry.py").read_text(encoding="utf-8"))
    assert set(live_defs) - set(ref_defs) == {f"FerryRuntime.{r}" for r in READERS}
    assert sorted(n for n in ref_defs if live_defs.get(n) != ref_defs[n]) == []
    new_classes = [s for s in live_top if isinstance(s, tuple) and s[0] == "class"
                   and s not in ref_top]
    assert [s[1] for s in new_classes] == ["StopPrice"]
    live_top = [s for s in live_top if s not in new_classes]
    assert len(live_top) == len(ref_top)
    for old, new in zip(ref_top, live_top):
        if not isinstance(old, ast.AST):
            assert old == new
        elif ast.dump(old) == ast.dump(new):
            continue
        elif isinstance(old, ast.Expr):                          # the module docstring
            paragraphs = new.value.value.split("\n\n")
            assert [p for p in paragraphs if p in old.value.value.split("\n\n")] == \
                old.value.value.split("\n\n")
            assert [p for p in paragraphs if "FeRRy Phase 5" in p]
        elif isinstance(old, ast.ImportFrom):                    # typing
            assert old.module == new.module == "typing"
            gained = {a.name for a in new.names} - {a.name for a in old.names}
            assert {a.name for a in old.names} <= {a.name for a in new.names}
            assert gained == {"Collection", "NamedTuple"}
        elif isinstance(old, ast.If):                            # TYPE_CHECKING
            olds, news = [ast.dump(s) for s in old.body], [ast.dump(s) for s in new.body]
            assert [s for s in news if s in olds] == olds and len(news) == len(olds) + 1
        else:                                                    # __all__
            assert isinstance(old, ast.Assign) and old.targets[0].id == "__all__"
            olds = [e.value for e in old.value.elts]
            news = [e.value for e in new.value.elts]
            assert [n for n in news if n in olds] == olds and set(news) - set(olds) == {
                "StopPrice"}
