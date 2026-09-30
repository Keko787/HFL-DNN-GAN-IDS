"""FeRRy Phase 3, unit U6: the mule's ferry glue (``hermes/mule/ferry.py``).

What is pinned: the spec's validation and ``from_config`` defaults; the
payload model's byte rules, which must be the host's own (``ContactPlan``);
the physics the scheduler prices with (R_planar(b), the member dwell at the
mean SNR, the predicted upload on the held carrier, the energy powers); the
waypoint annotations; the stop observation and contact plan at arrival; the
keyed availability draw and that the ground truth never leaves it (critic
B16); the L1 state (design section 4.6); the upload charge, its carrier
policy and the below-floor cap (critic B12).
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics

import numpy as np
import pytest

from hermes.l1.channel_model import (
    SALT_AVAILABILITY,
    SALT_BACKHAUL,
    SALT_CONTACT,
    BackhaulChannel,
    ContactChannel,
    ferry_salt,
    keyed_uniform,
    loss_from_snr,
)
from hermes.l1.contact_link import ContactLink
from hermes.l1.mission_clock import EnergyModel, FlightModel, MissionClock, SIM_EPOCH_S
from hermes.mission.contact_plan import ContactPlan
from hermes.mule.ferry import (
    FerryRuntime,
    FerrySpec,
    PayloadModel,
    StopObservation,
    backhaul_record,
)
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass

COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
SEED = 11
RF = 60.0


def _wp(x, y, *devs, deadline=1e7):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=deadline)


def _spec(**kw):
    kw.setdefault("rf_range_m", RF)
    kw.setdefault("seed", SEED)
    return FerrySpec.from_config(**kw)


def _theta(n=4):
    return [np.zeros((n,), dtype=np.float32), np.ones((3, 3), dtype=np.float32)]


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #

def test_the_default_spec_is_the_channel_free_control():
    spec = FerrySpec()
    assert spec.band is None and spec.link is None and spec.backhaul is None
    assert not spec.banded
    assert spec.in_flight_response == "abort" and spec.replan_fallback == "reorder"
    assert spec.reliability_source == "origin" and spec.deadline_bounds == "collection"
    assert spec.flight == FlightModel() and spec.payload == PayloadModel()


@pytest.mark.parametrize("kw, match", [
    (dict(band="wide"), "needs a ContactLink"),
    (dict(band="wide", link=ContactLink(anchor_planar_m=RF)), "needs a ContactChannel"),
    (dict(band="huge", link=ContactLink(anchor_planar_m=RF)), "unknown band"),
    (dict(contact_channel=ContactChannel.from_link(ContactLink(anchor_planar_m=RF), salt=1)),
     "needs a contact band"),
    (dict(backhaul=BackhaulChannel(salt=1, period_s=100.0)), "needs the ContactLink"),
    (dict(reliability_source="channel"), "critic B16"),
    (dict(in_flight_response="hover"), "in_flight_response"),
    (dict(replan_fallback="shuffle"), "replan_fallback"),
    (dict(deadline_bounds="landing"), "deadline_bounds"),
    (dict(backhaul_policy="greedy"), "backhaul_policy"),
    (dict(availability={"d": 1.5}), r"\[0, 1\]"),
])
def test_the_spec_refuses_what_it_cannot_run(kw, match):
    with pytest.raises((ValueError, TypeError), match=match):
        FerrySpec(**kw)


def test_channel_reliability_needs_a_band_and_its_salt():
    link = ContactLink(anchor_planar_m=RF)
    chan = ContactChannel.from_link(link, salt=3)
    with pytest.raises(ValueError, match="availability_salt"):
        FerrySpec(band="wide", link=link, contact_channel=chan, reliability_source="channel")
    spec = FerrySpec(band="wide", link=link, contact_channel=chan,
                     reliability_source="channel", availability_salt=5,
                     availability={"a": 0.5})
    assert spec.availability == {"a": 0.5}


def test_from_config_builds_the_design_defaults():
    spec = _spec(contact_band="wide", backhaul_model="seconds", n_missions=4, t_nom_s=219.0,
                 contact_reliability_source="channel", device_availability={"a": 0.3})
    assert spec.link.anchor_planar_m == RF and spec.link.n_pl == 2.2
    assert spec.link.altitude_m == 25.0 and spec.link.shadow_sigma_db == 4.0
    assert spec.contact_channel.salt == ferry_salt(SEED, SALT_CONTACT)
    assert spec.contact_channel.bands == spec.link.names
    assert spec.backhaul.salt == ferry_salt(SEED, SALT_BACKHAUL)
    assert spec.backhaul.period_s == 4 * 219.0 and spec.backhaul.regime == "clean"
    assert spec.availability_salt == ferry_salt(SEED, SALT_AVAILABILITY)
    assert dict(spec.availability) == {"a": 0.3}
    assert spec.flight == FlightModel()
    assert spec.flight.energy == EnergyModel()


def test_from_config_keeps_the_ground_truth_only_for_the_channel_source():
    spec = _spec(contact_band="wide", device_availability={"a": 0.3})
    assert dict(spec.availability) == {} and spec.availability_salt is None


def test_from_config_needs_the_backhaul_period():
    with pytest.raises(ValueError, match="P_bh"):
        _spec(backhaul_model="seconds")
    assert _spec(backhaul_model="seconds", backhaul_period=300.0).backhaul.period_s == 300.0
    with pytest.raises(ValueError, match="backhaul_model"):
        _spec(backhaul_model="hourly")


def test_from_config_energy_follows_the_cruise_speed_and_overrides():
    spec = _spec(cruise_speed_m_s=10.0, energy_capacity_j=5e4)
    assert spec.flight.cruise_speed_m_s == 10.0
    assert spec.flight.energy == EnergyModel.at_speed(10.0, capacity_j=5e4)
    spec = _spec(p_move_w=100.0, p_hover_w=120.0)
    assert (spec.flight.energy.p_move_w, spec.flight.energy.p_hover_w) == (100.0, 120.0)


def test_describe_is_json_and_never_carries_the_ground_truth():
    """Critic B16: the availability map stays with the draw."""
    spec = _spec(contact_band="wide", backhaul_model="seconds", backhaul_period=100.0,
                 contact_reliability_source="channel",
                 device_availability={"dev-a": 0.123456, "dev-b": 0.654321})
    record = spec.describe()
    text = json.dumps(record)
    assert "0.123456" not in text and "0.654321" not in text
    assert record["device_availability_n"] == 2
    assert "availability" not in repr(spec).replace("device_availability_n", "")
    assert record["contact_band"] == "wide" and record["backhaul_model"] == "seconds"
    assert record["energy_params"]["status"] == "simulated"
    assert record["band_classes"]["classes"][0]["range_planar_m"] == RF
    assert FerrySpec().describe()["band_classes"] is None


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #

def test_the_payload_model_is_the_hosts_byte_rule():
    measured = PayloadModel()
    assert measured.session_bytes(COLLECT, push_bytes=100, update_bytes=40) == 140
    assert measured.session_bytes(DELIVER, push_bytes=100, update_bytes=40) == 100
    assert measured.upload_bytes(40) == 40
    declared = PayloadModel(1_000_000)
    assert declared.session_bytes(COLLECT, push_bytes=100, update_bytes=40) == 2_000_000
    assert declared.session_bytes("deliver", push_bytes=100) == 1_000_000
    assert declared.upload_bytes(40) == 1_000_000
    # An empty partial carries no model: nothing to price.
    assert declared.upload_bytes(0) == 0
    # The host prices a declared session the same way.
    plan = ContactPlan(arrival_ts=0.0, targets=("a",), clock=lambda: 0.0,
                       advance=lambda dt, k: 0.0, payload_bytes=1_000_000)
    assert plan.session_bytes(140, 2) == declared.session_bytes(COLLECT, push_bytes=100,
                                                                update_bytes=40)
    for bad in (-1, 1.5, True):
        with pytest.raises((TypeError, ValueError)):
            PayloadModel(bad)


def test_the_runtime_measures_what_the_pass_pushes():
    rt = FerryRuntime(_spec(contact_band="wide"), MissionClock(), rf_range_m=RF)
    theta, synth = _theta(), [np.zeros((2, 4), dtype=np.float32)]
    rt.observe_payload(theta, synth)
    t, s = 4 * 4 + 9 * 4, 2 * 4 * 4
    assert rt.session_bytes(COLLECT) == (t + s) + t
    assert rt.session_bytes(DELIVER) == t + s
    assert rt.predicted_upload_bytes() == t


# --------------------------------------------------------------------------- #
# The scheduler's physics
# --------------------------------------------------------------------------- #

def test_the_range_is_r_planar_of_the_band_and_rf_range_without_one():
    assert FerryRuntime(_spec(contact_band="wide"), None, rf_range_m=RF).range_planar_m == RF
    medium = FerryRuntime(_spec(contact_band="medium"), None, rf_range_m=RF)
    assert medium.range_planar_m == medium.spec.link.range_planar_m("medium") > RF
    assert FerryRuntime(FerrySpec(), None, rf_range_m=RF).range_planar_m == RF


def test_the_link_must_be_anchored_at_the_runs_rf_range():
    with pytest.raises(ValueError, match="R_planar"):
        FerryRuntime(_spec(contact_band="wide", rf_range_m=50.0), None, rf_range_m=RF)


def test_the_member_dwell_is_the_links_at_the_mean_snr():
    spec = _spec(contact_band="wide", payload_bytes=1_000_000)
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    link = spec.link
    for d in (0.0, 30.0, 59.0):
        for off in (0.0, -3.0):
            assert rt.member_dwell_s(d, COLLECT, off) == link.dwell_s(
                2_000_000, "wide", link.mean_snr_db("wide", d) + off)
            assert rt.member_dwell_s(d, DELIVER, off) == link.dwell_s(
                1_000_000, "wide", link.mean_snr_db("wide", d) + off)
    # Far beyond the edge the mean is below the floor: unreachable, never inf.
    assert rt.member_dwell_s(5_000.0, COLLECT, 0.0) is None


def test_the_physics_prices_with_the_runtime():
    spec = _spec(contact_band="wide", backhaul_model="seconds", backhaul_period=500.0,
                 energy_capacity_j=9e4, deadline_bounds="delivery")
    rt = FerryRuntime(spec, None, rf_range_m=RF, session_time_s=1.0)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    phys = rt.physics()
    assert phys.dock == spec.flight.dock and phys.range_m == RF
    assert phys.p_move_w == 143.6 and phys.p_hover_w == 168.5
    assert phys.energy_capacity_j == 9e4 and phys.deadline_bounds == "delivery"
    assert phys.member_dwell_s(12.0, COLLECT, 0.0) == rt.member_dwell_s(12.0, COLLECT, 0.0)
    carrier = spec.backhaul.fixed_band()
    rate = spec.link.rate_bps("wide", spec.backhaul.pred_snr_db(carrier))
    assert phys.upload_time_s() == pytest.approx(8 * 18_756 / rate, rel=1e-15)
    # The live view: a new payload re-prices the same physics object.
    rt.set_payload(theta_bytes=1_000, synth_bytes=0)
    assert phys.upload_time_s() == pytest.approx(8 * 1_000 / rate, rel=1e-15)


def test_the_upload_is_predicted_on_the_carrier_the_mule_holds():
    """Fixed: ``argmax g_c`` (here carrier 1, not 0); H3: the carrier its
    controller holds since the last upload."""
    spec = FerrySpec.from_config(rf_range_m=RF, seed=1, backhaul_model="seconds",
                                 backhaul_period=500.0)
    bh, link = spec.backhaul, spec.link
    # Carrier means 17, 21 and 12 dB: three different CQIs, so three rates.
    bh.gains_db = (5.0, 9.0, 0.0)
    assert bh.fixed_band() == 1

    def predicted(carrier, nbytes=10_000):
        return 8 * nbytes / link.rate_bps("wide", bh.pred_snr_db(carrier))

    rt = FerryRuntime(spec, MissionClock(), rf_range_m=RF)
    rt.set_payload(theta_bytes=10_000)
    assert rt.held_carrier() == 1 and rt.predicted_upload_s() == predicted(1)
    adaptive = FerryRuntime(dataclasses.replace(spec, backhaul_policy="adaptive"),
                            MissionClock(), rf_range_m=RF)
    adaptive.set_payload(theta_bytes=10_000)
    assert adaptive.predicted_upload_s() == predicted(1)       # nothing held yet
    adaptive.carrier = 2
    assert adaptive.held_carrier() == 2 and adaptive.predicted_upload_s() == predicted(2)
    assert predicted(2) != predicted(1)


def test_the_channel_free_physics_charges_one_session_per_contact():
    rt = FerryRuntime(FerrySpec(), None, rf_range_m=RF, session_time_s=1.0)
    phys = rt.physics()
    assert phys.member_dwell_s is None and phys.range_m is None
    assert phys.upload_time_s() == 0.0          # no seconds-axis backhaul
    model = rt.feasibility_model()
    leg = model.leg((0.0, 0.0, 0.0), _wp(30, 40, "a", "b"))
    assert (leg.transit_s, leg.dwell_s, leg.return_s, leg.upload_s) == (10.0, 1.0, 10.0, 0.0)


def test_a_carrier_whose_mean_is_below_the_floor_is_priced_at_the_cap():
    """Critic B12 in the planner: never an infinite upload."""
    spec = _spec(backhaul_model="seconds", backhaul_period=100.0)
    spec = dataclasses.replace(spec, backhaul=BackhaulChannel(
        salt=1, period_s=100.0, base_db=-60.0))
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    rt.set_payload(theta_bytes=10_000)
    assert rt.predicted_upload_s() == rt.upload_cap_s(10_000)
    assert math.isfinite(rt.physics().upload_time_s())


def test_the_feasibility_model_must_fly_the_flight_models_speed():
    rt = FerryRuntime(FerrySpec(), None, rf_range_m=RF)
    with pytest.raises(ValueError, match="m/s"):
        rt.feasibility_model(FeasibilityModel(cruise_speed_m_s=7.0))
    ferry_model = rt.feasibility_model()
    with pytest.raises(ValueError, match="already carries"):
        rt.feasibility_model(ferry_model)
    base = FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=2.0)
    assert rt.feasibility_model(base).session_time_s == 2.0


def test_spec_feasibility_model_prices_a_fixed_payload_for_callers_without_a_mule():
    spec = _spec(contact_band="wide", backhaul_model="seconds", backhaul_period=100.0)
    model = spec.feasibility_model(rf_range_m=RF, theta_bytes=18_756, synth_bytes=64)
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    assert model.ferry.upload_time_s() == rt.predicted_upload_s()
    assert model.ferry.member_dwell_s(20.0, COLLECT, 0.0) == rt.member_dwell_s(20.0, COLLECT, 0.0)


# --------------------------------------------------------------------------- #
# Stops
# --------------------------------------------------------------------------- #

POS = {DeviceID("a"): (10.0, 0.0, 0.0), DeviceID("b"): (0.0, 50.0, 0.0),
       DeviceID("c"): (0.0, 70.0, 0.0)}


def test_annotate_fills_the_three_fields_without_changing_identity():
    spec = _spec(contact_band="wide")
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    wp = _wp(0, 0, "a", "b")
    (ann,) = rt.annotate([wp], POS)
    assert ann == wp and hash(ann) == hash(wp) and ann is not wp
    assert ann.band == "wide" and ann.range_m == RF
    assert ann.pred_snr_db == (spec.link.mean_snr_db("wide", 10.0),
                               spec.link.mean_snr_db("wide", 50.0))
    # Distances are to the stop, wherever it is: a 3-4-5 triangle.
    (off,) = rt.annotate([_wp(13.0, 4.0, "a")], POS)
    assert off.pred_snr_db == (spec.link.mean_snr_db("wide", 5.0),)
    (free,) = FerryRuntime(FerrySpec(), None, rf_range_m=RF).annotate([wp], POS)
    assert free.band is None and free.range_m == RF and free.pred_snr_db is None


def test_the_observation_is_the_channel_at_arrival():
    spec = _spec(contact_band="wide")
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    wp = _wp(0, 0, "a", "b")
    t = SIM_EPOCH_S + 123.4
    obs = rt.observe(wp, POS, t)
    chan, link = spec.contact_channel, spec.link
    assert obs.distances_m == (10.0, 50.0)
    assert obs.snr_db == tuple(chan.snr_db(t, "wide", d, link_key=j, stop_pos=wp.position)
                               for j, d in zip(wp.devices, (10.0, 50.0)))
    assert obs.class_snr_db == tuple(
        statistics.median([chan.snr_db(t, name, d, link_key=j, stop_pos=wp.position)
                           for j, d in zip(wp.devices, (10.0, 50.0))])
        for name in link.names)
    assert obs.max_slant_m == link.slant_m(50.0)
    free = FerryRuntime(FerrySpec(), None, rf_range_m=RF).observe(wp, POS, t)
    assert free.snr_db is None and free.class_snr_db is None and free.max_slant_m is None


def test_the_class_snr_is_the_median_over_the_members_not_the_mean():
    """Design section 4.6: slots 0-2 are the MEDIAN realized SNR per class.
    With two members the median is the mean, so three members at distinct
    distances pin it."""
    spec = _spec(contact_band="wide")
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    wp = _wp(0, 0, "a", "b", "c")
    t = SIM_EPOCH_S + 57.0
    obs = rt.observe(wp, POS, t)
    chan, link = spec.contact_channel, spec.link
    per_class = [
        [chan.snr_db(t, name, d, link_key=j, stop_pos=wp.position)
         for j, d in zip(wp.devices, (10.0, 50.0, 70.0))]
        for name in link.names
    ]
    assert all(len(set(v)) == 3 for v in per_class)          # distinct: a real median
    assert obs.class_snr_db == tuple(statistics.median(v) for v in per_class)
    assert obs.class_snr_db != tuple(statistics.fmean(v) for v in per_class)
    assert obs.max_slant_m == link.slant_m(70.0)


def test_the_contact_plan_gates_by_r_planar_and_the_floor():
    clock = MissionClock()
    clock.advance(42.0, "transit")
    spec = _spec(contact_band="wide", payload_bytes=500_000)
    rt = FerryRuntime(spec, clock, rf_range_m=RF, session_time_s=1.0)
    wp = _wp(0, 0, "a", "b", "c")          # c is 70 m out: beyond R_planar(wide)
    plan = rt.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=1)
    assert plan.arrival_ts == clock() and plan.band == "wide" and plan.band_index == 0
    assert plan.stop == (0.0, 0.0, 0.0)
    assert "c" in plan.unreachable and "a" in plan.targets
    assert plan.listen_s == 1.0 and plan.payload_bytes == 500_000
    chan = spec.contact_channel
    assert plan.snr_db["b"] == chan.snr_db(clock(), "wide", 50.0, link_key="b",
                                           stop_pos=wp.position)
    # Each target is priced at its own session start (critic C2).
    assert plan.snr_at("b", clock() + 9.0) == chan.snr_db(clock() + 9.0, "wide", 50.0,
                                                          link_key="b", stop_pos=wp.position)
    assert plan.dwell_s(1000, 5.0) == spec.link.dwell_s(1000, "wide", 5.0)


def test_a_member_in_range_but_below_the_floor_at_arrival_is_unreachable():
    """Critic B12 on the contact side: the plan gates by the link's floor as
    well as by R_planar(b). Member e sits 59 m out, inside R_planar(wide) =
    60 m, where the realized SNR is below -6.7 dB on about one arrival in ten
    (the 90 % edge availability, D1). At such an arrival e is unreachable in
    both passes: never solicited and never priced (its rate is 0, so its
    dwell would be None), while a is a target."""
    spec = _spec(contact_band="wide", payload_bytes=500_000)
    link, chan = spec.link, spec.contact_channel
    floor = link.snr_floor_db
    stop = (0.0, 0.0, 0.0)
    positions = {DeviceID("a"): (10.0, 0.0, 0.0), DeviceID("e"): (0.0, 59.0, 0.0)}
    # The first half-second after the epoch at which e is below the floor
    # (the channel is a pure function of the seed and the time).
    t_below = next(
        SIM_EPOCH_S + 0.5 * k for k in range(1, 20_000)
        if chan.snr_db(SIM_EPOCH_S + 0.5 * k, "wide", 59.0, link_key="e", stop_pos=stop) < floor
    )
    clock = MissionClock()
    clock.advance(t_below - SIM_EPOCH_S, "transit")
    rt = FerryRuntime(spec, clock, rf_range_m=RF)
    assert 59.0 < rt.range_planar_m == RF               # in range: only the floor gates e
    wp = _wp(0, 0, "a", "e")
    for pass_kind in (COLLECT, DELIVER):
        plan = rt.contact_plan(wp, positions, pass_kind=pass_kind, mission_round=1)
        assert plan.arrival_ts == t_below == clock()
        assert plan.snr_floor_db == floor
        assert plan.snr_db["e"] < floor <= plan.snr_db["a"]
        assert plan.targets == ("a",) and plan.unreachable == ("e",)
        assert plan.dwell_s(500_000, plan.snr_db["e"]) is None
    # A moment later the same member at the same place is a target again.
    t_above = next(t_below + 0.5 * k for k in range(1, 20_000)
                   if chan.snr_db(t_below + 0.5 * k, "wide", 59.0, link_key="e",
                                  stop_pos=stop) >= floor)
    clock.advance(t_above - t_below, "dwell")
    plan = rt.contact_plan(wp, positions, pass_kind=COLLECT, mission_round=1)
    assert plan.targets == ("a", "e") and plan.unreachable == ()


def test_position_keyed_shadowing_reaches_the_plan():
    """Critic C5: shadowing keyed by (device, stop cell) needs the stop."""
    clock = MissionClock()
    spec = _spec(contact_band="wide", shadow_keying="position")
    rt = FerryRuntime(spec, clock, rf_range_m=RF)
    wp = _wp(40.0, 40.0, "a", "b")
    positions = {DeviceID("a"): (40.0, 50.0, 0.0), DeviceID("b"): (45.0, 40.0, 0.0)}
    plan = rt.contact_plan(wp, positions, pass_kind=COLLECT, mission_round=1)
    chan = spec.contact_channel
    assert chan.shadow_keying == "position"
    assert plan.snr_db["a"] == chan.snr_db(clock(), "wide", 10.0, link_key="a",
                                           stop_pos=wp.position)


def test_the_spec_prices_t_nom_with_the_scheduler_helper():
    """Spec Q1: the driver's T_nom comes from U4's helper fed with this spec's
    physics (``FerrySpec.feasibility_model``); by hand for one device."""
    from hermes.scheduler.fl_scheduler import (
        median_nominal_mission_period_s,
        nominal_mission_period_s,
    )

    spec = FerrySpec()                     # no band: 1 s per contact, no upload
    model = spec.feasibility_model(rf_range_m=RF, theta_bytes=18_756)
    layout = {DeviceID("d0"): (30.0, 40.0, 0.0)}
    t = nominal_mission_period_s(layout, rf_range_m=RF, feasibility_model=model,
                                 turnaround_s=spec.flight.turnaround_s)
    # Pass 1: 10 s out, 1 s, 10 s home; 30 s turnaround; Pass 2 the same.
    assert t == 21.0 + 30.0 + 21.0
    wide = _spec(contact_band="wide", backhaul_model="seconds", backhaul_period=1e3)
    wide_model = wide.feasibility_model(rf_range_m=RF, theta_bytes=18_756, synth_bytes=64)
    t_nom = median_nominal_mission_period_s(
        [layout, {DeviceID("d0"): (0.0, 10.0, 0.0)}], rf_range_m=RF,
        feasibility_model=wide_model, turnaround_s=30.0)
    assert 30.0 < t_nom < t


def test_t_nom_can_be_computed_before_the_backhaul_period_is_known():
    """A seconds-axis cell's P_bh = n_missions * T_nom, and T_nom comes from
    the spec: the planner reads only the carrier means (``base + g_c``), so a
    placeholder period gives the same T_nom, and the cell's spec is then
    built from it (``from_config`` docstring). The regime is not a
    placeholder: it sets ``base``."""
    from hermes.scheduler.fl_scheduler import median_nominal_mission_period_s

    layouts = [{DeviceID("d0"): (30.0, 40.0, 0.0), DeviceID("d1"): (-50.0, 10.0, 0.0)},
               {DeviceID("d0"): (0.0, 10.0, 0.0)}]

    def t_nom(spec):
        model = spec.feasibility_model(rf_range_m=RF, theta_bytes=18_756, synth_bytes=64)
        return median_nominal_mission_period_s(layouts, rf_range_m=RF,
                                               feasibility_model=model, turnaround_s=30.0)

    common = dict(contact_band="wide", backhaul_model="seconds", backhaul_regime="jittery")
    placeholder = t_nom(_spec(backhaul_period=1.0, **common))
    assert placeholder == t_nom(_spec(backhaul_period=1e6, **common))
    cell = _spec(n_missions=4, t_nom_s=placeholder, **common)
    assert cell.backhaul.period_s == 4 * placeholder and t_nom(cell) == placeholder
    assert t_nom(_spec(backhaul_period=1.0, **dict(common, backhaul_regime="clean"))) != placeholder


def test_the_channel_free_plan_solicits_every_member():
    clock = MissionClock()
    rt = FerryRuntime(FerrySpec(), clock, rf_range_m=RF, session_time_s=1.0)
    plan = rt.contact_plan(_wp(0, 0, "a", "c"), POS, pass_kind=COLLECT, mission_round=1)
    assert plan.band is None and plan.targets == ("a", "c") and plan.unreachable == ()
    assert plan.session_time_s == 1.0 and plan.drop_uplink == frozenset()


def test_a_planning_only_runtime_builds_no_contact_plan():
    rt = FerryRuntime(FerrySpec(), None, rf_range_m=RF)
    with pytest.raises(ValueError, match="no clock"):
        rt.contact_plan(_wp(0, 0, "a"), POS, pass_kind=COLLECT, mission_round=1)


# --------------------------------------------------------------------------- #
# The availability draw (spec Q8, critic B16)
# --------------------------------------------------------------------------- #

def _channel_spec(avail):
    return _spec(contact_band="wide", contact_reliability_source="channel",
                 device_availability=avail)


def test_the_uplink_drop_is_the_keyed_draw_against_rel():
    avail = {f"d{i}": r for i, r in enumerate((0.15, 0.4, 0.7, 0.95))}
    spec = _channel_spec(avail)
    rt = FerryRuntime(spec, MissionClock(), rf_range_m=RF)
    members = [DeviceID(d) for d in avail]
    salt = ferry_salt(SEED, SALT_AVAILABILITY)
    for rnd in range(1, 30):
        expect = {d for d in members if keyed_uniform(salt, d, rnd) >= avail[d]}
        assert rt.uplink_drops(members, rnd) == expect
        # A pure function of (seed, device, round): order and repeats do not matter.
        assert rt.uplink_drops(list(reversed(members)), rnd) == expect
    drops = sum(len(rt.uplink_drops(members, r)) for r in range(1, 2001))
    assert drops / 2000 == pytest.approx(sum(1 - r for r in avail.values()), abs=0.08)


def test_certain_and_missing_availability():
    rt = FerryRuntime(_channel_spec({"always": 1.0, "never": 0.0}), MissionClock(),
                      rf_range_m=RF)
    for rnd in range(1, 50):
        assert rt.uplink_drops(["always", "never", "unknown"], rnd) == {"never"}


def test_the_draw_applies_to_pass_1_with_the_channel_source_only():
    avail = {"a": 0.0, "b": 0.0}
    clock = MissionClock()
    rt = FerryRuntime(_channel_spec(avail), clock, rf_range_m=RF)
    wp = _wp(0, 0, "a", "b")
    assert rt.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=3).drop_uplink == {"a", "b"}
    assert rt.contact_plan(wp, POS, pass_kind=DELIVER, mission_round=3).drop_uplink == frozenset()
    origin = FerryRuntime(_spec(contact_band="wide"), clock, rf_range_m=RF)
    assert origin.contact_plan(wp, POS, pass_kind=COLLECT, mission_round=3).drop_uplink == frozenset()


def test_the_ground_truth_reaches_neither_the_physics_nor_the_plan():
    """Critic B16: decision code sees outcomes, never rel_i."""
    spec = _channel_spec({"a": 0.777777, "b": 0.333333})
    rt = FerryRuntime(spec, MissionClock(), rf_range_m=RF)
    phys = rt.physics()
    blob = repr(phys) + repr(dataclasses.asdict(
        dataclasses.replace(phys, member_dwell_s=None, upload_s=lambda: 0.0)))
    assert "0.777777" not in blob and "0.333333" not in blob
    plan = rt.contact_plan(_wp(0, 0, "a", "b"), POS, pass_kind=COLLECT, mission_round=1)
    values = [getattr(plan, f.name) for f in dataclasses.fields(plan)]
    assert not any(v in (0.777777, 0.333333) for v in values if isinstance(v, float))
    assert isinstance(plan.drop_uplink, frozenset)


# --------------------------------------------------------------------------- #
# The L1 state (design section 4.6)
# --------------------------------------------------------------------------- #

def test_the_l1_state_slots():
    spec = _spec(contact_band="wide", energy_capacity_j=10_000.0)
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    obs = StopObservation(t_s=1.0, devices=("a", "b"), distances_m=(3.0, 4.0),
                          snr_db=(9.0, 12.0), class_snr_db=(15.0, 21.0, 27.0, 99.0),
                          max_slant_m=250.0)
    s = rt.l1_state(obs, pose=(100.0, -200.0, 0.0), energy_j=2_500.0, budget_s=60.0)
    assert s.dtype == np.float32 and s.shape == (8,)
    np.testing.assert_allclose(s, [0.5, 0.7, 0.9, 2.5, 1.0, -2.0, 0.0, 0.75], rtol=1e-6)
    no_cap = FerryRuntime(_spec(contact_band="wide"), None, rf_range_m=RF)
    s = no_cap.l1_state(obs, pose=(0, 0, 0), energy_j=168.5 * 30.0, budget_s=60.0)
    assert s[7] == pytest.approx(0.5)                      # E_ref = P_hover * budget
    s = no_cap.l1_state(obs, pose=(0, 0, 0), energy_j=1e9, budget_s=None)
    assert s[7] == 1.0                                     # no reference: stays 1.0
    with pytest.raises(ValueError, match="band"):
        no_cap.l1_state(StopObservation(1.0, ("a",), (1.0,)), pose=(0, 0, 0),
                        energy_j=0.0, budget_s=None)


# --------------------------------------------------------------------------- #
# The upload (design section 2.3, critic B12)
# --------------------------------------------------------------------------- #

def _backhaul_runtime(policy="fixed", *, base_db=None, regime="jittery"):
    spec = _spec(backhaul_model="seconds", backhaul_period=400.0, backhaul_policy=policy,
                 backhaul_regime=regime)
    if base_db is not None:
        spec = dataclasses.replace(spec, backhaul=BackhaulChannel(
            salt=spec.backhaul.salt, period_s=400.0, base_db=base_db))
    clock = MissionClock()
    return FerryRuntime(spec, clock, rf_range_m=RF), clock


def test_the_upload_is_priced_at_the_carriers_snr_when_it_starts():
    rt, clock = _backhaul_runtime()
    clock.advance(77.0, "transit")
    t0 = clock()
    up = rt.charge_upload(18_756)
    bh, link = rt.spec.backhaul, rt.spec.link
    assert up.carrier == bh.fixed_band() and up.t_start_s == t0
    assert up.snr_db == bh.snr_db(t0, up.carrier)
    assert up.upload_s == 8 * 18_756 / link.rate_bps("wide", up.snr_db)
    assert up.p_loss == loss_from_snr(up.snr_db) and not up.below_floor
    assert clock() == t0 + up.upload_s and clock.ledger()["upload"] == up.upload_s
    record = backhaul_record(up)
    assert record["t_upload_s"] == clock() and record["carrier"] == up.carrier
    json.dumps(record)
    # The causal RF prior saw the upload (critic B4).
    assert rt.rf_prior.prior_snr_db(carrier=up.carrier) == up.snr_db


def test_h3_holds_its_controllers_carrier_across_uploads():
    rt, clock = _backhaul_runtime("adaptive")
    seen = []
    for _ in range(12):
        clock.advance(97.0, "transit")
        current = rt.carrier
        up = rt.charge_upload(1_000)
        assert up.carrier == rt.spec.backhaul.select_carrier(
            up.t_start_s, adaptive=True, current=current)
        assert rt.held_carrier() == up.carrier
        seen.append(up.carrier)
    assert len(set(seen)) > 1                  # jittery: the controller moves


def test_a_backhaul_below_the_floor_is_a_lost_upload_with_a_capped_charge():
    """Critic B12: rate 0 is never charged as inf."""
    rt, clock = _backhaul_runtime(base_db=-60.0)
    t0 = clock()
    up = rt.charge_upload(1_000_000)
    assert up.below_floor and up.p_loss == 1.0
    assert up.upload_s == rt.upload_cap_s(1_000_000)
    cap = rt.spec.link.dwell_s(1_000_000, "wide", rt.spec.link.snr_floor_db)
    assert up.upload_s == cap and math.isfinite(cap) and cap > 0
    assert clock() == t0 + cap
    assert rt.charge_upload(0).upload_s == 0.0          # an empty partial


def test_no_seconds_backhaul_charges_no_upload():
    clock = MissionClock()
    rt = FerryRuntime(FerrySpec(), clock, rf_range_m=RF)
    assert rt.charge_upload(10_000) is None and clock() == SIM_EPOCH_S
    assert backhaul_record(None) is None and rt.rf_prior is None


def test_energy_is_the_ledgers():
    clock = MissionClock()
    rt = FerryRuntime(FerrySpec(), clock, rf_range_m=RF)
    clock.advance(10.0, "transit")
    clock.advance(2.0, "dwell")
    clock.advance(30.0, "turnaround")
    assert rt.energy_j() == pytest.approx(143.6 * 10.0 + 168.5 * 2.0)
