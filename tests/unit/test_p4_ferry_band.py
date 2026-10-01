"""FeRRy Phase 4 (unit U6): the ferry runtime's band (``hermes/mule/ferry.py``).

What is pinned:

* ``FerryRuntime.band`` starts as the spec's, and ``set_band`` moves it with
  the range and index a runtime built on that class has; it refuses the
  channel-free control and any class the link lacks, and leaves the runtime
  unchanged when it does.
* Critic B3: per-class physics is bound to its class when it is built. After
  every class's model is built, moving the runtime's band to each class x in
  turn leaves every class y pricing y exactly as before; and a class's model
  equals the model of a spec built on that class.
* ``plan_classes`` gives one ``PlanClass`` per link class, in link order, with
  the class's radius, a bound model and a bound outage; the outage is the
  spec's mean-SNR outage under the channel's own noise, a step on a noise-free
  channel (critic B4) and 1 beyond the class's range.
* ``arrival_view`` gates each class exactly as that class's contact plan does
  and prices each class's dwell at the arrival SNR for the pass it is given
  (which it requires); it charges nothing.
* After ``set_band``, every read without a band prices the committed class, and
  an explicit band prices that class on the one contact channel: what an arm
  flying it would see at that stop and time (the arms stay paired).
* Pass 2 flies the runtime's band (the spec, other choices 2): a Pass-2 contact
  plan on any other class is refused and charges nothing, whoever asks; Pass 1
  takes any class.
* Legacy identity (Freeze Rule 1): at its defaults (``set_band`` never called)
  the runtime's every output equals 6e6f92d's, compared against that commit's
  ``ferry.py`` loaded from git; and the legacy calls never load
  ``hermes.scheduler.plan``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

import hermes.mule.ferry as ferry_module
from hermes.l1.channel_model import CONTACT_REGIMES, ContactChannel
from hermes.l1.contact_link import CLASSES_WITH_10MHZ, ContactLink
from hermes.l1.mission_clock import SIM_EPOCH_S, MissionClock
from hermes.mule.ferry import FerryRuntime, FerrySpec
from hermes.scheduler.plan.types import ArrivalView, PlanClass, PlanOptions, PlanSetup
from hermes.scheduler.stages.s3b_feasibility import (
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    FeasibilityModel,
    FlightState,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass

COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
RF = 60.0
SEED = 11
DOCK = (0.0, 0.0, 0.0)
T0 = SIM_EPOCH_S
BANDS = ("wide", "medium", "narrow")
REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "6e6f92d"


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


# A stop at (40, 30) whose members sit at planar distances 10, 50 and 59 m (in
# every class's range), 70 and 110 m (beyond wide's 60 m, inside medium's
# 119.5 m), 200 m (narrow's only) and 300 m (beyond every class), on the axes
# so that each distance is exact.
STOP = _wp(40, 30, "a", "b", "c", "e", "f", "g", "h")
POS = {
    DeviceID("a"): (50.0, 30.0, 0.0), DeviceID("b"): (40.0, 80.0, 0.0),
    DeviceID("c"): (-19.0, 30.0, 0.0), DeviceID("e"): (110.0, 30.0, 0.0),
    DeviceID("f"): (40.0, 140.0, 0.0), DeviceID("g"): (240.0, 30.0, 0.0),
    DeviceID("h"): (40.0, 330.0, 0.0),
}
STOP_2 = _wp(-30, -20, "a", "c")
GRID = (0.0, 10.0, 30.0, 59.0, 60.0, 70.0, 110.0, 119.0, 150.0, 200.0, 231.0, 300.0)


def _prices(model):
    """Everything a class model prices: its range, the member dwell over a
    grid of distances, passes and SNR offsets, the leg to STOP and a fold."""
    ferry = model.ferry
    bound = dataclasses.replace(model, ferry=ferry.bind(POS))
    return (
        ferry.range_m,
        tuple(ferry.member_dwell_s(d, pk, off)
              for d in GRID for pk in (COLLECT, DELIVER) for off in (0.0, -2.0)),
        tuple(bound.leg(DOCK, STOP, pass_kind=pk) for pk in (COLLECT, DELIVER)),
        bound.fold([STOP, STOP_2], FlightState(DOCK, T0), rule=RULE_DEADLINE_BUDGET,
                   budget_end=T0 + 400.0, skip=False),
    )


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


# --------------------------------------------------------------------------- #
# The band and set_band
# --------------------------------------------------------------------------- #

def test_the_band_starts_as_the_specs_and_the_control_has_none():
    for band in BANDS:
        rt = FerryRuntime(_spec(contact_band=band), None, rf_range_m=RF)
        assert rt.band == band == rt.spec.band
    assert FerryRuntime(FerrySpec(), None, rf_range_m=RF).band is None
    assert FerryRuntime(_spec(), None, rf_range_m=RF).band is None     # a link but no band


def test_set_band_moves_the_band_with_the_range_and_index_of_that_class():
    rt = _runtime("wide")
    link = rt.spec.link
    assert (rt.band, rt.range_planar_m, rt.band_index) == ("wide", RF, 0)
    for name in ("narrow", "medium", "narrow", "wide"):
        rt.set_band(name)
        own = FerryRuntime(_spec(contact_band=name), None, rf_range_m=RF)
        assert (rt.band, rt.range_planar_m, rt.band_index) == (
            own.band, own.range_planar_m, own.band_index)
        assert (rt.range_planar_m, rt.band_index) == (link.range_planar_m(name), link.index(name))
    # Back on wide, the range is rf_range_m bit for bit (S3a's radius); the spec never moved.
    assert rt.range_planar_m == RF and rt.spec.band == "wide"
    four = _runtime("wide", band_classes=CLASSES_WITH_10MHZ)
    four.set_band("medium_wide")
    assert four.band_index == 3
    assert four.range_planar_m == four.spec.link.range_planar_m("medium_wide")


@pytest.mark.parametrize("band, error", [
    ("huge", ValueError), ("", ValueError), ("medium_wide", ValueError),   # not on this link
    ("Wide", ValueError), (0, TypeError), (None, TypeError), (True, TypeError),
])
def test_set_band_refuses_a_class_the_link_lacks_and_changes_nothing(band, error):
    rt = _runtime("medium")
    before = (rt.band, rt.range_planar_m, rt.band_index)
    with pytest.raises(error):
        rt.set_band(band)
    assert (rt.band, rt.range_planar_m, rt.band_index) == before


def test_the_channel_free_control_has_no_classes_to_choose_from():
    clock = MissionClock()
    wp = _wp(0, 0, "a")
    for spec in (FerrySpec(), _spec()):          # hand-built, and from_config without a band
        rt = FerryRuntime(spec, clock, rf_range_m=RF)
        for call in (
            lambda: rt.set_band("wide"),
            lambda: rt.physics("wide"),
            lambda: rt.feasibility_model(band="wide"),
            lambda: rt.plan_classes(),
            lambda: rt.arrival_view(wp, POS, clock(), pass_kind=COLLECT),
            lambda: rt.outage_probability(10.0),
            lambda: rt.annotate([wp], POS, band="wide"),
            lambda: rt.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=1, band="wide"),
        ):
            with pytest.raises(ValueError, match="channel-free|no contact band"):
                call()
        assert rt.band is None and rt.range_planar_m == RF and rt.band_index is None
        # The band-less reads stay the recorded channel-free ones.
        phys = rt.physics()
        assert phys.member_dwell_s is None and phys.range_m is None
        assert rt.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=1).band is None


# --------------------------------------------------------------------------- #
# Per-class physics (critic B3)
# --------------------------------------------------------------------------- #

def test_each_class_model_keeps_its_own_class_whatever_band_is_set():
    """Critic B3: build every class's model, then move the runtime's band to
    each class x in turn: every class y still prices y, exactly as before."""
    rt = _runtime("wide", payload_bytes=1_000_000)
    classes = rt.plan_classes()
    reference_model = rt.feasibility_model()            # the mule's own, built on wide
    by_name = {"narrow": rt.feasibility_model(band="narrow")}
    before = {c.name: _prices(c.model) for c in classes}
    ref_before = _prices(reference_model)
    named_before = _prices(by_name["narrow"])
    assert len(set(map(repr, before.values()))) == len(classes)   # the classes do differ
    for x in ("narrow", "medium", "wide", "narrow"):
        rt.set_band(x)
        for c in classes:
            assert _prices(c.model) == before[c.name], (x, c.name)
        assert _prices(reference_model) == ref_before
        assert _prices(by_name["narrow"]) == named_before
    # The runtime's own member dwell, called without a band, does follow the
    # band set: a model that read it at call time would have moved too.
    narrow = classes[2].model.ferry.member_dwell_s
    rt.set_band("narrow")
    assert rt.member_dwell_s(30.0, COLLECT, 0.0) == narrow(30.0, COLLECT, 0.0)
    rt.set_band("wide")
    assert rt.member_dwell_s(30.0, COLLECT, 0.0) != narrow(30.0, COLLECT, 0.0)


@pytest.mark.parametrize("payload", [None, 1_000_000])
@pytest.mark.parametrize("backhaul", ["mission", "seconds"])
def test_a_class_model_is_the_model_of_a_spec_built_on_that_class(payload, backhaul):
    kw = dict(payload_bytes=payload, backhaul_model=backhaul, energy_capacity_j=9e4,
              deadline_bounds="delivery")
    if backhaul == "seconds":
        kw["backhaul_period"] = 500.0
    rt = _runtime("medium", **kw)
    for c in rt.plan_classes():
        own = _runtime(c.name, **kw)                       # the same seed, flying class c
        p, q = c.model.ferry, own.physics()
        assert c.radius_m == p.range_m == q.range_m == own.range_planar_m
        assert c.index == own.band_index
        assert (p.dock, p.p_move_w, p.p_hover_w, p.energy_capacity_j, p.deadline_bounds) == (
            q.dock, q.p_move_w, q.p_hover_w, q.energy_capacity_j, q.deadline_bounds)
        assert p.upload_time_s() == q.upload_time_s()
        assert _prices(c.model) == _prices(own.feasibility_model())
        assert (c.model.cruise_speed_m_s, c.model.session_time_s) == (
            own.feasibility_model().cruise_speed_m_s, own.feasibility_model().session_time_s)


def test_a_class_model_follows_the_payload_the_mule_carries():
    """Only the band is bound: the θ of the pass is read when the model prices."""
    rt = _runtime("wide")
    narrow = rt.plan_classes()[2].model.ferry
    small = narrow.member_dwell_s(30.0, COLLECT, 0.0)
    rt.set_payload(theta_bytes=500_000, synth_bytes=0)
    assert narrow.member_dwell_s(30.0, COLLECT, 0.0) == rt.member_dwell_s(30.0, COLLECT, 0.0,
                                                                          band="narrow") > small


def test_physics_built_twice_for_one_class_compares_equal():
    """As the bound method it replaces did: equal and hashing alike per class."""
    rt = _runtime("wide")
    assert rt.physics() == rt.physics() and hash(rt.physics()) == hash(rt.physics())
    assert rt.physics("narrow") == rt.physics("narrow") != rt.physics("wide")
    assert rt.physics() == rt.physics("wide")


# --------------------------------------------------------------------------- #
# The planner's classes and their outage
# --------------------------------------------------------------------------- #

def test_plan_classes_hold_every_link_class_in_link_order():
    rt = _runtime("medium")
    link = rt.spec.link
    classes = rt.plan_classes()
    assert all(isinstance(c, PlanClass) for c in classes)
    assert [c.name for c in classes] == list(link.names) == list(BANDS)
    assert [c.index for c in classes] == [0, 1, 2]
    assert [c.radius_m for c in classes] == [link.range_planar_m(n) for n in BANDS]
    assert classes[0].radius_m == RF
    for c in classes:
        assert c.model.ferry.member_dwell_s is not None
        assert c.model.cruise_speed_m_s == rt.spec.flight.cruise_speed_m_s
    # A base carries the session time; a base that already has physics is refused.
    base = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=2.0)
    assert {c.model.session_time_s for c in rt.plan_classes(base)} == {2.0}
    with pytest.raises(ValueError, match="already carries"):
        rt.plan_classes(rt.feasibility_model())
    # They make the mule's setup: one hover power, the reference among them.
    setup = PlanSetup(options=PlanOptions(), classes=classes, reference="medium",
                      t_ref_s=200.0, turnaround_s=30.0)
    assert setup.searched == classes and setup.p_hover_w == rt.spec.flight.energy.p_hover_w
    four = _runtime("wide", band_classes=CLASSES_WITH_10MHZ).plan_classes()
    assert [c.name for c in four] == list(CLASSES_WITH_10MHZ) and four[3].index == 3


@pytest.mark.parametrize("regime", ["clean", "jittery"])
def test_the_outage_is_the_mean_snr_outage_under_the_channels_noise(regime):
    """Spec, other choices 7: Φ((floor − SNR_b(d)) / σ_eff), σ_eff² = σ_sh² + σ_I² + A²/2."""
    rt = _runtime("wide", contact_regime=regime)
    link = rt.spec.link
    amp, sigma_i = CONTACT_REGIMES[regime]
    sigma = math.sqrt(link.shadow_sigma_db ** 2 + sigma_i ** 2 + amp ** 2 / 2.0)
    for c in rt.plan_classes():
        edge = link.range_planar_m(c.name)
        previous = -1.0
        for d in (0.0, 10.0, 0.5 * edge, 0.9 * edge, edge):
            expect = NormalDist().cdf((link.snr_floor_db - link.mean_snr_db(c.name, d)) / sigma)
            assert rt.outage_probability(d, band=c.name) == c.outage(d) == expect
            assert previous < expect < 0.5       # rises with distance, below 1/2 in range
            previous = expect
        # At the edge the mean sits M_sh above the floor.
        assert c.outage(edge) == pytest.approx(NormalDist().cdf(-link.shadow_margin_db / sigma))
        # Beyond the class's range the gate never solicits the member.
        assert c.outage(edge + 1e-6) == c.outage(1e4) == 1.0
    # Without a band argument the outage is the runtime's band's.
    rt.set_band("narrow")
    assert rt.outage_probability(150.0) == rt.outage_probability(150.0, band="narrow") < 1.0
    assert rt.outage_probability(150.0, band="wide") == 1.0


def test_a_noise_free_channel_makes_the_outage_the_gates_step():
    """Critic B4's deterministic channel keeps the link's σ in the mean and
    draws no noise: the SNR is the mean, so the outage is 1 below the floor and
    0 at or above it. A 0.2 edge quantile puts the mean below the floor inside
    each class's range, so both sides of the step are in range."""
    link = ContactLink(anchor_planar_m=RF, margin_quantile=0.2)
    chan = ContactChannel(link.mean_snr_db, salt=1, bands=link.names, interference_amp_db=0.0,
                          interference_sigma_db=0.0, shadow_sigma_db=0.0)
    rt = FerryRuntime(FerrySpec(band="wide", link=link, contact_channel=chan), None, rf_range_m=RF)
    for name in link.names:
        edge = link.range_planar_m(name)
        for d in [edge * k / 50.0 for k in range(51)]:
            below = link.mean_snr_db(name, d) < link.snr_floor_db
            assert rt.outage_probability(d, band=name) == (1.0 if below else 0.0)
        assert rt.outage_probability(0.0, band=name) == 0.0
        assert rt.outage_probability(edge, band=name) == 1.0   # the mean is 3.4 dB under the floor


# --------------------------------------------------------------------------- #
# The arrival view (FX, decision 5)
# --------------------------------------------------------------------------- #

def _field(rng, n_stops, spread=250.0):
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


@pytest.mark.parametrize("seed", range(6))
def test_the_arrival_view_gates_each_class_as_its_contact_plan_would(seed):
    rng = random.Random(seed)
    clock = MissionClock()
    payload, regime = rng.choice([None, 1_000_000]), rng.choice(["clean", "jittery"])
    rt = _runtime("wide", clock=clock, payload_bytes=payload, contact_regime=regime)
    # Pass 2 flies b̄ only, so another class's Pass-2 plan is its pinned arm's.
    pinned = {name: _runtime(name, clock=clock, payload_bytes=payload, contact_regime=regime)
              for name in BANDS}
    link = rt.spec.link
    stops, pos = _field(rng, 5)
    for committed in ("wide", "narrow", "medium"):
        rt.set_band(committed)
        for wp in stops:
            clock.advance(rng.uniform(0.5, 60.0), "transit")
            t, ledger = clock(), dict(clock.ledger())
            for pass_kind in (COLLECT, DELIVER):
                view = rt.arrival_view(wp, pos, t, pass_kind=pass_kind)
                assert isinstance(view, ArrivalView)
                assert view.committed == committed and view.devices == wp.devices
                assert [(c.name, c.index) for c in view.classes] == list(zip(BANDS, range(3)))
                nbytes = rt.session_bytes(pass_kind)
                for entry in view.classes:
                    flier = (rt if pass_kind is COLLECT or entry.name == committed
                             else pinned[entry.name])
                    plan = flier.contact_plan(wp, pos, pass_kind=pass_kind, mission_round=1,
                                              band=entry.name)
                    assert entry.targets == plan.targets
                    assert entry.dwell_s == sum(
                        link.dwell_s(nbytes, entry.name, plan.snr_db[j]) for j in plan.targets)
            assert clock() == t and dict(clock.ledger()) == ledger      # nothing charged


def test_the_view_gates_the_range_inclusively_as_s3a_and_the_plan_do():
    """A member exactly at R_planar(wide) is in range (S3a's inclusive test);
    one a nanometre beyond is not, whatever its SNR."""
    clock = MissionClock()
    rt = _runtime("wide", clock=clock)
    chan, floor = rt.spec.contact_channel, rt.spec.link.snr_floor_db
    wp = _wp(0, 0, "edge", "out")
    pos = {DeviceID("edge"): (RF, 0.0, 0.0), DeviceID("out"): (RF + 1e-9, 0.0, 0.0)}

    def above(t):
        return all(chan.snr_db(t, "wide", d, link_key=j, stop_pos=wp.position) >= floor
                   for j, d in (("edge", RF), ("out", RF + 1e-9)))

    t = next(T0 + 0.5 * k for k in range(1, 20_000) if above(T0 + 0.5 * k))
    clock.advance(t - T0, "transit")
    view = rt.arrival_view(wp, pos, clock(), pass_kind=COLLECT)
    assert view.entry("wide").targets == (DeviceID("edge"),)
    assert view.entry("narrow").targets == (DeviceID("edge"), DeviceID("out"))
    plan = rt.contact_plan(wp, pos, pass_kind=COLLECT, mission_round=1)
    assert plan.targets == ("edge",) and plan.unreachable == ("out",)


# --------------------------------------------------------------------------- #
# Reads with and without a band
# --------------------------------------------------------------------------- #

def test_after_set_band_every_read_without_a_band_prices_the_committed_class():
    clock = MissionClock()
    clock.advance(123.4, "transit")
    rt = _runtime("wide", clock=clock, payload_bytes=500_000)
    rt.set_band("narrow")
    own = _runtime("narrow", clock=clock, payload_bytes=500_000)
    wp = _wp(40, 30, "a", "b", "e", "g")
    t = clock()
    assert [(w.band, w.range_m, w.pred_snr_db) for w in rt.annotate([wp, STOP_2], POS)] == [
        (w.band, w.range_m, w.pred_snr_db) for w in own.annotate([wp, STOP_2], POS)]
    assert rt.observe(wp, POS, t) == own.observe(wp, POS, t)
    for pass_kind in (COLLECT, DELIVER):
        assert _plan_fields(rt.contact_plan(wp, POS, pass_kind=pass_kind, mission_round=2)) == \
            _plan_fields(own.contact_plan(wp, POS, pass_kind=pass_kind, mission_round=2))
    assert rt.rates_bps([-7.0, 0.0, 9.5]) == own.rates_bps([-7.0, 0.0, 9.5])
    assert rt.member_dwell_s(70.0, COLLECT, 0.0) == own.member_dwell_s(70.0, COLLECT, 0.0)
    assert _prices(rt.feasibility_model()) == _prices(own.feasibility_model())
    assert rt.arrival_view(wp, POS, t, pass_kind=COLLECT).committed == "narrow"


def test_an_explicit_band_is_what_an_arm_flying_that_class_sees_there():
    """One ContactChannel serves every class: the plan FX builds on class c at a
    Pass-1 stop and time is the plan an arm pinned to c builds there (paired)."""
    clock = MissionClock()
    clock.advance(57.25, "transit")
    rt = _runtime("wide", clock=clock, contact_regime="jittery", payload_bytes=1_000_000)
    wp, t = STOP, clock()
    for name in BANDS:
        own = _runtime(name, clock=clock, contact_regime="jittery", payload_bytes=1_000_000)
        assert _plan_fields(rt.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=3,
                                            band=name)) == \
            _plan_fields(own.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=3))
        assert rt.observe(wp, POS, t, band=name) == own.observe(wp, POS, t)
        assert rt.rates_bps([0.0, 11.0], band=name) == own.rates_bps([0.0, 11.0])
        assert [(w.band, w.range_m, w.pred_snr_db) for w in rt.annotate([wp], POS, band=name)] == \
            [(w.band, w.range_m, w.pred_snr_db) for w in own.annotate([wp], POS)]
        assert all(rt.member_dwell_s(d, COLLECT, 0.0, band=name)
                   == own.member_dwell_s(d, COLLECT, 0.0) for d in GRID)
    # An explicit band never moves the runtime's own.
    assert (rt.band, rt.range_planar_m, rt.band_index) == ("wide", RF, 0)


def test_pass_2_flies_the_runtimes_band_only():
    """The spec, other choices 2: "FX switches band per stop in Pass 1 only;
    Pass 2 flies b̄". Whoever asks, a Pass-2 plan on a class other than the
    runtime's is refused, charges nothing and moves nothing; on b̄, named or
    not, it is the plan of an arm pinned to b̄. Pass 1 still takes any class.
    Legacy runs never name a band, and naming the spec's own is allowed."""
    clock = MissionClock()
    clock.advance(57.25, "transit")
    kw = dict(contact_regime="jittery", payload_bytes=1_000_000)
    rt = _runtime("wide", clock=clock, **kw)
    wp, t, ledger = STOP, clock(), dict(clock.ledger())
    for committed in ("narrow", "wide", "medium"):
        rt.set_band(committed)
        pinned = _plan_fields(_runtime(committed, clock=clock, **kw).contact_plan(
            wp, POS, pass_kind=DELIVER, mission_round=3))
        for band in (None, committed):
            for pass_kind in (DELIVER, "deliver"):
                assert _plan_fields(rt.contact_plan(wp, POS, pass_kind=pass_kind, mission_round=3,
                                                    band=band)) == pinned
        for other in (b for b in BANDS if b != committed):
            for pass_kind in (DELIVER, "deliver"):
                with pytest.raises(ValueError, match="Pass 2 flies the committed class"):
                    rt.contact_plan(wp, POS, pass_kind=pass_kind, mission_round=3, band=other)
            assert rt.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=3,
                                   band=other).band == other
        assert (rt.band, rt.range_planar_m, rt.band_index) == (
            committed, rt.spec.link.range_planar_m(committed), rt.spec.link.index(committed))
    assert clock() == t and dict(clock.ledger()) == ledger
    legacy = _runtime("medium", clock=clock)
    assert _plan_fields(legacy.contact_plan(wp, POS, pass_kind=DELIVER, mission_round=3,
                                            band="medium")) == \
        _plan_fields(legacy.contact_plan(wp, POS, pass_kind=DELIVER, mission_round=3))
    with pytest.raises(ValueError, match="Pass 2 flies the committed class 'medium'"):
        legacy.contact_plan(wp, POS, pass_kind=DELIVER, mission_round=3, band="wide")


def test_the_arrival_view_is_priced_for_the_pass_it_is_given():
    """The pass is required, as for the contact plan, so no view is priced for
    the other pass's bytes: Pass 1 prices push + update, Pass 2 the push."""
    clock = MissionClock()
    clock.advance(12.5, "transit")
    rt = _runtime("wide", clock=clock, payload_bytes=1_000_000)
    with pytest.raises(TypeError, match="pass_kind"):
        rt.arrival_view(STOP, POS, clock())
    collect = rt.arrival_view(STOP, POS, clock(), pass_kind=COLLECT)
    deliver = rt.arrival_view(STOP, POS, clock(), pass_kind="deliver")
    assert rt.session_bytes(COLLECT) == 2 * rt.session_bytes(DELIVER)
    for a, b in zip(collect.classes, deliver.classes):
        assert a.targets == b.targets                  # one gate for both passes
        assert a.dwell_s > b.dwell_s or not a.targets  # twice the bytes in Pass 1


@pytest.mark.parametrize("call", ["annotate", "observe", "contact_plan", "rates_bps",
                                  "member_dwell_s", "physics", "outage_probability"])
def test_an_explicit_band_the_link_lacks_is_refused(call):
    clock = MissionClock()
    rt = _runtime("wide", clock=clock)
    calls = {
        "annotate": lambda b: rt.annotate([STOP], POS, band=b),
        "observe": lambda b: rt.observe(STOP, POS, clock(), band=b),
        "contact_plan": lambda b: rt.contact_plan(STOP, POS, pass_kind=COLLECT, mission_round=1,
                                                  band=b),
        "rates_bps": lambda b: rt.rates_bps([0.0], band=b),
        "member_dwell_s": lambda b: rt.member_dwell_s(1.0, COLLECT, 0.0, band=b),
        "physics": lambda b: rt.physics(b),
        "outage_probability": lambda b: rt.outage_probability(1.0, band=b),
    }
    with pytest.raises(ValueError, match="unknown band class"):
        calls[call]("medium_wide")
    with pytest.raises(TypeError):
        calls[call](1)


# --------------------------------------------------------------------------- #
# Legacy identity (Freeze Rule 1)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ref_ferry(tmp_path_factory):
    """``hermes/mule/ferry.py`` as it was at 6e6f92d, as a module of its own.

    It imports the live modules it depends on, so any difference below comes
    from ferry.py alone. Skipped where git or the commit is not available.
    """
    try:
        blob = subprocess.run(["git", "show", f"{REF_COMMIT}:hermes/mule/ferry.py"], cwd=REPO,
                              capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:          # pragma: no cover - no git
        pytest.skip(f"git cannot show {REF_COMMIT}'s ferry.py: {e}")
    path = tmp_path_factory.mktemp("ref_ferry") / "_ref_ferry.py"
    path.write_bytes(blob)
    name = f"_p4_ref_{REF_COMMIT}_hermes_mule_ferry"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # dataclasses resolve annotations through it
    try:
        spec.loader.exec_module(mod)
        assert not hasattr(mod.FerryRuntime, "set_band")      # really the old module
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


def _record(mod, cfg, seed):
    """Every output of a runtime built by ``mod`` at its defaults, in order."""
    rng = random.Random(seed)
    stops, pos = _field(rng, 4, spread=400.0)
    spec = mod.FerrySpec() if cfg is None else mod.FerrySpec.from_config(
        rf_range_m=RF, seed=seed, **cfg)
    clock = MissionClock()
    rt = mod.FerryRuntime(spec, clock, rf_range_m=RF, session_time_s=1.0)
    out = [("attrs", rt.range_planar_m, rt.band_index, rt.banded)]
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
    return out


@pytest.mark.parametrize("seed", [3, 17, 38])
@pytest.mark.parametrize("cfg", list(_CONFIGS), ids=list(_CONFIGS))
def test_the_runtime_at_its_defaults_is_6e6f92ds(ref_ferry, cfg, seed):
    """Freeze Rule 1: with set_band never called, every output of the runtime
    (the pricing, the physics and its models, legs and folds, annotations,
    observations, the L1 state, contact plans, rates, the availability draw
    and the upload) is bit for bit the 6e6f92d module's, on every class, both
    payload modes, both backhaul models and both reliability sources."""
    live = _record(ferry_module, _CONFIGS[cfg], seed)
    ref = _record(ref_ferry, _CONFIGS[cfg], seed)
    assert len(live) == len(ref)
    for a, b in zip(live, ref):
        assert a == b, a[0]


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


def test_the_plan_package_is_imported_by_the_plan_mode_builders_only():
    """U0's hand-off: the legacy import path must not load hermes.scheduler.plan."""
    plan_imports = [(m, w) for m, w in _module_imports(REPO / "hermes/mule/ferry.py")
                    if m.startswith("hermes.scheduler.plan")]
    assert plan_imports and all(w in ("checking", "function") for _, w in plan_imports)
    code = "\n".join([
        "import sys",
        "import hermes.mule.ferry as f",
        "from hermes.l1.mission_clock import MissionClock",
        "from hermes.types import Bucket, ContactWaypoint, MissionPass",
        "before = {m for m in sys.modules if m.startswith('hermes.scheduler.plan')}",
        "wp = ContactWaypoint(position=(0.0, 0.0, 0.0), devices=('a',),",
        "                     bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=1e9)",
        "pos = {'a': (10.0, 0.0, 0.0)}",
        "for band in (None, 'narrow'):",
        "    clock = MissionClock()",
        "    spec = f.FerrySpec.from_config(rf_range_m=60.0, seed=1, contact_band=band)",
        "    rt = f.FerryRuntime(spec, clock, rf_range_m=60.0)",
        "    rt.feasibility_model(); rt.physics(); rt.annotate([wp], pos)",
        "    rt.observe(wp, pos, clock())",
        "    rt.contact_plan(wp, pos, pass_kind=MissionPass.COLLECT, mission_round=1)",
        "    if band:",
        "        rt.member_dwell_s(5.0, MissionPass.COLLECT, 0.0); rt.rates_bps([3.0])",
        "        rt.set_band('wide'); rt.outage_probability(5.0)",
        "after = {m for m in sys.modules if m.startswith('hermes.scheduler.plan')}",
        "assert after == before, sorted(after - before)",
        "rt.plan_classes(); rt.arrival_view(wp, pos, clock(), pass_kind=MissionPass.COLLECT)",
        "assert 'hermes.scheduler.plan.types' in sys.modules",
        "print('ok')",
    ])
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
