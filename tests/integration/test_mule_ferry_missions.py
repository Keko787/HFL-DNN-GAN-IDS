"""FeRRy Phase 3, unit U6: whole missions on the mission clock, in process.

Real ``ClientMission`` devices, a real cluster and a real ``MuleSupervisor``
on a ``MissionClock`` (``tests/integration/_ferry_harness.py``: synchronous
threads, the wall clock parked at 1.7e9 s so a leaked wall stamp shows).

Pinned (design sections 2.3 and 3.3-3.7, unit U6's test list):

* the ledger sums to ``sim_end - sim_start``, and every stop's arithmetic
  (departure + transit = arrival, arrival + dwell + listen = end) holds;
* the pose resets to the dock at each takeoff and every pass lands there;
* the S3b budget is stamped at takeoff, Pass 2's at its own takeoff;
* the turnaround is charged once per mission, upload or not;
* a 5 km leg costs no wall time;
* the predicate holds at every actual departure, with its tail;
* ``abort`` gives up the tail, ``replan`` repairs the remainder; both widen
  what they give up at the simulated drop time; S3c counts the contacts flown;
  only ``replan`` checks a budgeted Pass 2 in flight, whose energy clause
  counts from its own takeoff;
* the beacon hook (inserts, every refusal, Pass 1 only); the L1 state; the
  keyed availability draw;
* no wall stamp reaches the scheduler, a delta or a ledger (critic B3), and a
  DOWN carrying wall-clock deadline overrides ends the mission loudly;
* a backhaul below the floor is a lost upload with a capped charge, and a
  member in range but below the floor at arrival is never solicited or
  charged (B12);
* the Lamport sync at the dock with two mules, also after an empty mission
  that docks (``dock_on_empty``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hermes.l1.channel_model import BackhaulChannel, loss_from_snr
from hermes.l1.mission_clock import LEDGER_KINDS, SIM_EPOCH_S
from hermes.mule import MuleSupervisorError
from hermes.mule.ferry import FerrySpec
from hermes.scheduler.stages.s3b_feasibility import FlightState
from hermes.scheduler.stages.s3c_mission_window import MissionWindowAdapter
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeliveryOutcome,
    DeviceID,
    DeviceSchedulerState,
    FLState,
    MissionOutcome,
    MissionPass,
    MuleID,
    sign_down_bundle,
)
from hermes.types.bundles import BackhaulUpload

from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

logging.getLogger("hermes.mission.client_mission").setLevel(logging.ERROR)

COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
DOCK = (0.0, 0.0, 0.0)
T0 = SIM_EPOCH_S

#: Five single-device contacts on a line from the dock (rf_range_m 5): the
#: planned flight is exact without a band (1 s per contact), so the numbers
#: below are hand-checked. e answers at 3 m, a at 10 m, b at 20 m, c at 30 m,
#: d at 40 m.
LINE = (
    ("dev-e", (3.0, 0.0, 0.0)),
    ("dev-a", (10.0, 0.0, 0.0)),
    ("dev-b", (20.0, 0.0, 0.0)),
    ("dev-c", (30.0, 0.0, 0.0)),
    ("dev-d", (40.0, 0.0, 0.0)),
)


def full_spec(**kw) -> FerrySpec:
    """Wide band, the seconds-axis backhaul, 1 MB each way (visible airtime)."""
    kw.setdefault("contact_band", "wide")
    kw.setdefault("backhaul_model", "seconds")
    kw.setdefault("backhaul_period", 800.0)
    kw.setdefault("payload_bytes", 1_000_000)
    return FerrySpec.from_config(rf_range_m=kw.pop("rf_range_m", 60.0),
                                 seed=kw.pop("seed", 7), **kw)


def fly(*, ferry=None, missions=1, layout=GH.LAYOUT, flaky=GH.FLAKY, silent=(),
        before=None, world_kw=None, **sup_kw):
    """Run ``missions`` missions on the clock; one record per mission."""
    w = H.World(layout=layout, flaky=flaky, **(world_kw or {}))
    mid = w.mule_ids[0]
    w.rfs[mid].silent.update(DeviceID(d) for d in silent)
    records = []
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=True, ferry=ferry, **sup_kw)
        stamps = []
        start = sup.scheduler.start_mission
        sup.scheduler.start_mission = lambda: stamps.append(start()) or stamps[-1]
        w.bootstrap()
        for m in range(missions):
            if before is not None:
                before(w, sup, m)
            pose_before = tuple(sup.mule_pose)
            wall = time.perf_counter()
            r = sup.run_one_mission()
            records.append(SimpleNamespace(
                result=r, pose_before=pose_before, pose_after=tuple(sup.mule_pose),
                wall_s=time.perf_counter() - wall, deltas=w.take_deltas(mid),
                budget_stamp=stamps[-1] if stamps else None,
            ))
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return w, sup, records


def _stop_wp(stop) -> ContactWaypoint:
    return ContactWaypoint(position=tuple(stop["position"]),
                           devices=tuple(DeviceID(d) for d in stop["devices"]),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=stop["deadline_ts"])


def _check_stop_arithmetic(r) -> None:
    for flown in (r.pass_1_flown, r.pass_2_flown):
        prev_end = None
        for s in flown:
            assert s["arrival_s"] == s["depart_s"] + s["transit_s"]
            assert s["end_s"] == pytest.approx(s["arrival_s"] + s["dwell_s"] + s["listen_s"],
                                               abs=1e-9)
            if prev_end is not None:
                assert s["depart_s"] == prev_end           # nothing charged in between
            prev_end = s["end_s"]


# --------------------------------------------------------------------------- #
# The ledger, the pose and the stamps
# --------------------------------------------------------------------------- #

def test_the_ledger_accounts_for_every_simulated_second():
    w, sup, recs = fly(ferry=full_spec(), missions=3)
    spec = sup.ferry
    for k, rec in enumerate(recs):
        r = rec.result
        assert list(r.sim_ledger) == list(LEDGER_KINDS)
        assert sum(r.sim_ledger.values()) == pytest.approx(r.sim_end_s - r.sim_start_s, abs=1e-6)
        if k:
            assert r.sim_start_s == recs[k - 1].result.sim_end_s     # takes off at the landing
        assert r.energy_j == spec.flight.energy.energy_j(r.sim_ledger)
        _check_stop_arithmetic(r)
        stops = r.pass_1_flown + r.pass_2_flown
        assert r.sim_ledger["transit"] == pytest.approx(sum(s["transit_s"] for s in stops))
        assert r.sim_ledger["dwell"] == pytest.approx(sum(s["dwell_s"] for s in stops))
        assert r.sim_ledger["listen"] == pytest.approx(sum(s["listen_s"] for s in stops))
        home = sum(spec.flight.leg_s(flown[-1]["position"], DOCK)
                   for flown in (r.pass_1_flown, r.pass_2_flown))
        assert r.sim_ledger["return"] == pytest.approx(home)
        assert r.sim_ledger["turnaround"] == 30.0
        assert r.sim_ledger["dwell"] > 1.0                      # 1 MB each way on wide
        # The upload: the fixed carrier, priced at its SNR, completed as the UP says.
        up = w.server.ups[k]
        assert r.backhaul["carrier"] == spec.backhaul.fixed_band()
        assert r.sim_ledger["upload"] == r.backhaul["upload_s"] > 0.0
        assert up.sim_upload_ts == r.backhaul["t_upload_s"]
        assert up.backhaul.carrier == r.backhaul["carrier"] and not up.backhaul.below_floor
        assert r.sim_pass_2_start_s == pytest.approx(up.sim_upload_ts + 30.0)
        assert r.band == "wide"
        where = dict(GH.LAYOUT)
        for s in r.pass_1_flown + r.pass_2_flown:
            assert s["band"] == "wide" and set(s["snr_db"]) == set(s["devices"])
            assert s["rate_bps"] == {d: spec.link.rate_bps("wide", v)
                                     for d, v in s["snr_db"].items()}
            # Each member's SNR is the channel's for its own distance to the
            # stop, at the arrival (positions from the layout itself).
            stop = tuple(s["position"])
            for d, v in s["snr_db"].items():
                dist = math.dist(stop[:2], where[d][:2])
                assert v == pytest.approx(spec.contact_channel.snr_db(
                    s["arrival_s"], "wide", dist, link_key=d, stop_pos=stop), abs=1e-9)
            assert set(s["targets"]) | set(s["unreachable"]) == set(s["devices"])


def test_the_rf_prior_follows_the_uploads_causally():
    """Critic B4: with a seconds-axis backhaul the planner's RF prior is the
    SNR last observed on the held carrier, 20 dB before the first upload."""
    priors = []

    def watch(w, sup, m):
        priors.append(sup.rf_prior_snr_db)

    _, sup, recs = fly(ferry=full_spec(), missions=3, before=watch)
    assert priors[0] == 20.0
    for k in (1, 2):
        assert priors[k] == recs[k - 1].result.backhaul["snr_db"]
    assert sup.rf_prior_snr_db == recs[-1].result.backhaul["snr_db"]


def test_energy_drops_before_takeoff_are_widened_and_planned():
    """Critic B10: S3b's energy clause (a 2 kJ battery, simulated) drops c and
    d from LINE; the mule widens them at takeoff and S3c counts them."""
    from hermes.l1.mission_clock import EnergyModel, FlightModel

    spec = FerrySpec(flight=FlightModel(energy=EnergyModel(capacity_j=2_000.0)))
    _, sup, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, ferry=spec,
                         mission_budget_s=100.0,
                         mission_window_adapter=MissionWindowAdapter(enabled=True, window=2))
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"], ["dev-b"]]
    widened = _widened(rec)
    assert set(widened) == {"dev-c", "dev-d"}
    assert all(d.contact_ts == T0 and not d.answered for d in widened.values())
    assert sup.scheduler._window_adapter._history[-1] == (3, 5)
    # ... and recorded with their reason (``pass_1_preflight_drops``).
    assert [(d["devices"], d["position"], d["reason"]) for d in r.pass_1_preflight_drops] == [
        (["dev-c"], [30.0, 0.0, 0.0], "energy"), (["dev-d"], [40.0, 0.0, 0.0], "energy")]
    # Pass 1's simulated energy (the legs, the hovering, the way home) fits
    # the battery; Pass 2 is not gated without pass_2_budget (principle 13).
    energy = spec.flight.energy
    move = sum(s["transit_s"] for s in r.pass_1_flown) + spec.flight.leg_s(
        r.pass_1_flown[-1]["position"], DOCK)
    hover = sum(s["dwell_s"] + s["listen_s"] for s in r.pass_1_flown)
    assert energy.p_move_w * move + energy.p_hover_w * hover <= 2_000.0
    assert r.energy_j == energy.energy_j(r.sim_ledger) > 2_000.0


def test_the_mule_takes_off_from_the_dock_and_lands_there():
    _, sup, recs = fly(ferry=full_spec(), missions=3)
    for rec in recs:
        r = rec.result
        assert rec.pose_before == DOCK and rec.pose_after == DOCK
        assert r.pass_1_flown[0]["depart_pose"] == list(DOCK)
        assert r.pass_2_flown[0]["depart_pose"] == list(DOCK)
        assert r.pass_2_flown[0]["depart_s"] == r.sim_pass_2_start_s
    # A takeoff away from the dock is a wiring bug, not a mission.
    sup._next_theta = sup._next_theta or [np.zeros((4,), dtype=np.float32)]
    sup.mule_pose = (1.0, 0.0, 0.0)
    with pytest.raises(Exception, match="takeoff away from the dock"):
        sup.run_one_mission()


def test_an_empty_dock_report_is_stamped_on_the_mission_clock():
    """Critic B3 in ``_dock_empty``: a report it has to build itself (the
    host kept no ledger) takes the clock's time, not the wall's."""
    w, sup, _ = fly(layout=LINE, flaky={}, rf_range_m=5.0, dock_on_empty=True)
    with H.Patched(w.clock):
        sup.mission.last_unmerged = None
        w.server.tasks.clear()
        assert sup._dock_empty(99, None, sim_upload_ts=sup._now(), backhaul=None)
    up = w.server.ups[-1]
    report = up.round_close_report
    assert report.started_at == report.finished_at == sup._now() < H.SIM_CEILING
    assert up.sim_upload_ts == sup._now()


def test_the_budget_is_stamped_at_takeoff_and_pass_2_at_its_own():
    """LINE with a 10 s budget: Pass 1 keeps e and a (b, c, d do not fit),
    Pass 2 is walked from its own takeoff and skips the same three."""
    _, sup, recs = fly(layout=LINE, flaky={}, rf_range_m=5.0, missions=2,
                       mission_budget_s=10.0, pass_2_budget=True)
    for rec in recs:
        r = rec.result
        t0 = r.sim_start_s
        assert rec.budget_stamp == t0
        assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"]]
        t2 = r.sim_pass_2_start_s
        # e ends at 1.6, a at 4, home at 6; no upload time without a seconds
        # backhaul; then the 30 s turnaround.
        assert t2 == pytest.approx(t0 + 36.0)
        assert r.delivery_report.started_at == t2
        lines = {str(l.device_id): l for l in r.delivery_report.lines}
        skipped = {d for d, l in lines.items() if l.outcome is DeliveryOutcome.SKIPPED}
        assert skipped == {"dev-b", "dev-c", "dev-d"}
        assert all(lines[d].contact_ts == t2 for d in skipped)
        assert [s["devices"] for s in r.pass_2_flown] == [["dev-e"], ["dev-a"]]
        assert r.pass_2_budget_overrun_s == 0.0
        assert r.sim_end_s <= t2 + 10.0
        assert r.budget_overrun_s == 0.0


def test_the_pass_2_origin_follows_the_upload_and_the_turnaround():
    _, _, recs = fly(layout=LINE, flaky={}, rf_range_m=5.0)
    r = recs[0].result
    # Pass 1 flies e, a, b, c, d (13 s) and returns (8 s); no upload time.
    assert r.pass_1_flown[-1]["end_s"] == pytest.approx(T0 + 13.0)
    assert r.sim_pass_2_start_s == pytest.approx(T0 + 21.0 + 30.0)


@pytest.mark.parametrize("dock_on_empty", [False, True])
def test_turnaround_is_charged_once_per_mission_even_without_an_upload(dock_on_empty):
    def everyone_away(w, sup, m):
        if m == 1:
            w.set_state([d for d, _ in LINE], FLState.UNAVAILABLE)

    w, _, recs = fly(layout=LINE, flaky={}, rf_range_m=5.0, missions=2,
                     before=everyone_away, dock_on_empty=dock_on_empty)
    assert not recs[0].result.empty and recs[1].result.empty
    for rec in recs:
        assert rec.result.sim_ledger["turnaround"] == 30.0
        assert sum(rec.result.sim_ledger.values()) == pytest.approx(
            rec.result.sim_end_s - rec.result.sim_start_s, abs=1e-9)
        assert rec.pose_after == DOCK
    empty = recs[1].result
    assert empty.docked_empty is dock_on_empty
    assert len(w.server.ups) == (2 if dock_on_empty else 1)
    if dock_on_empty:
        assert w.server.ups[1].sim_upload_ts == empty.sim_end_s - 30.0


def test_a_5_km_leg_costs_no_wall_time():
    """P-02's guarantee on the clock: flight is charged, never waited for."""
    far = (("dev-far", (5000.0, 0.0, 0.0)),)
    _, _, recs = fly(layout=far, flaky={}, missions=2)
    for rec in recs:
        assert rec.result.sim_ledger["transit"] == pytest.approx(2 * 1000.0)
        assert rec.result.sim_ledger["return"] == pytest.approx(2 * 1000.0)
        assert rec.wall_s < 1.0
        assert rec.result.pass_1_flown[0]["transit_s"] == 1000.0


# --------------------------------------------------------------------------- #
# The in-flight response
# --------------------------------------------------------------------------- #

def _line_scenario(response, *, adapter=True):
    """LINE with a 28 s budget, dev-b's window cut to 7.5 s, dev-a silent.

    Planned from the dock at t0 (e, a, b, c, d; 1 s per contact): b finishes
    at 7 <= 7.5 and the flight is home at 21. S3b's EDF walk (b first) admits
    all five within 28. In flight dev-a is silent, so its contact costs the
    1 s listen window as well: leaving a at t0 + 5, b would finish at 8 >
    7.5. ``abort`` gives up b, c and d there; ``replan`` drops b alone
    (overdue) and flies c and d (home at 21 <= 28).
    """
    def tighten(w, sup, m):
        sup.scheduler.device_states[DeviceID("dev-b")].deadline_fulfilment_s = 7.5

    kw = dict(mission_window_adapter=MissionWindowAdapter(enabled=True, window=2)) \
        if adapter else {}
    return fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-a",), before=tighten,
               ferry=FerrySpec(in_flight_response=response), mission_budget_s=28.0, **kw)


def _widened(rec):
    return {str(d.device_id): d for src, d in rec.deltas if src == "direct"}


def test_abort_on_the_mission_clock_gives_up_the_tail():
    _, sup, (rec,) = _line_scenario("abort")
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"]]
    (abort,) = r.aborts
    assert abort["t_s"] == pytest.approx(T0 + 5.0)
    assert (abort["pass"], abort["reason"]) == ("collect", "overdue")
    assert abort["abandoned"] == [["dev-b"], ["dev-c"], ["dev-d"]]
    assert r.replans == []
    widened = _widened(rec)
    assert set(widened) == {"dev-b", "dev-c", "dev-d"}
    for delta in widened.values():
        assert delta.contact_ts == abort["t_s"]             # the sim drop time
        assert delta.outcome is MissionOutcome.TIMEOUT
        assert not delta.answered and delta.synthetic
    history = sup.scheduler._window_adapter._history
    assert history[-1] == (2, 5)                            # served e and a
    assert r.pass_1_flown[-1]["end_s"] == pytest.approx(T0 + 5.0)
    outcomes = {str(l.device_id): l.outcome for l in r.report.lines}
    assert outcomes == {"dev-e": MissionOutcome.CLEAN, "dev-a": MissionOutcome.TIMEOUT}


def test_replan_on_the_mission_clock_repairs_the_remainder():
    _, sup, (rec,) = _line_scenario("replan")
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"], ["dev-c"], ["dev-d"]]
    assert r.aborts == []
    (event,) = r.replans
    assert event["t_s"] == pytest.approx(T0 + 5.0) and event["pass"] == "collect"
    assert event["order_used"] == "arm"
    assert event["rejected"][0] == {"devices": ["dev-b"], "reason": "overdue"}
    assert event["dropped"] == [{"devices": ["dev-b"], "reason": "overdue"}]
    assert event["route"] == [["dev-c"], ["dev-d"]] and event["delta_obs_db"] == 0.0
    widened = _widened(rec)
    assert set(widened) == {"dev-b"} and widened["dev-b"].contact_ts == event["t_s"]
    assert sup.scheduler._window_adapter._history[-1] == (4, 5)
    # c at 30 m: arrival 9, end 10; d: arrival 12, end 13; home 21.
    assert [s["end_s"] for s in r.pass_1_flown] == pytest.approx(
        [T0 + 1.6, T0 + 5.0, T0 + 10.0, T0 + 13.0])
    assert r.budget_overrun_s == 0.0
    # The flown-order check ran before takeoff and kept the plan.
    assert sup.scheduler.last_order_check.order_used == "current"


def test_whatever_order_the_replan_returns_is_flown_as_returned():
    """``order_used`` is the scheduler's business (``arm``, ``two_opt``,
    ``admission``, ``arm_trimmed``): the mule flies the route it gets and
    treats every drop as final."""
    from hermes.scheduler.routing.replan import ORDER_ARM_TRIMMED, ReplanResult

    def trimmed(w, sup, m):
        sup.scheduler.device_states[DeviceID("dev-b")].deadline_fulfilment_s = 7.5
        real, calls = sup.scheduler.replan_remainder, []

        def replan(remainder, **kw):
            calls.append(kw["state"])
            if len(calls) == 1:            # the pre-flight flown-order check (design 3.3)
                return real(remainder, **kw)
            by = {wp.devices[0]: wp for wp in remainder}
            return ReplanResult((by["dev-d"],), ((by["dev-b"], "overdue"), (by["dev-c"], "budget")),
                                ORDER_ARM_TRIMMED)

        sup.scheduler.replan_remainder = replan

    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-a",), before=trimmed,
                       ferry=FerrySpec(in_flight_response="replan", replan_fallback="trim"),
                       mission_budget_s=28.0)
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"], ["dev-d"]]
    assert r.replans[0]["order_used"] == "arm_trimmed"
    assert set(_widened(rec)) == {"dev-b", "dev-c"}


def test_abort_checks_only_the_next_stop():
    """Amendment 8's rule on the clock: the tail is given up only when the
    NEXT stop fails. dev-d's window is cut to 13.5 s (planned to finish at
    13, at 14 after dev-a's listen window): the mule flies on to b and c and
    gives d up at c, not at a, where the whole remainder already failed."""
    def tighten(w, sup, m):
        sup.scheduler.device_states[DeviceID("dev-d")].deadline_fulfilment_s = 13.5

    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-a",),
                       before=tighten, ferry=FerrySpec(in_flight_response="abort"),
                       mission_budget_s=32.0)
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"], ["dev-b"], ["dev-c"]]
    (abort,) = r.aborts
    assert abort["t_s"] == pytest.approx(T0 + 11.0) and abort["reason"] == "overdue"
    assert abort["abandoned"] == [["dev-d"]]


@pytest.mark.parametrize("response, key, at", [
    ("abort", "aborts", 12.0),      # only when d is next, at c
    ("replan", "replans", 5.0),     # the whole remainder, already at a
])
def test_the_energy_clause_binds_in_flight_on_the_energy_spent(response, key, at):
    """Critic B10 in flight: a 3.15 kJ battery (simulated) fits the planned
    LINE flight (3.14 kJ), but dev-a's and dev-b's listen windows hover 2 s
    more (337 J), so the energy spent leaves too little for d and home.
    ``abort`` finds out when d is next; ``replan`` folds the whole remainder
    and finds out after a's listen window already (b's still to come)."""
    from hermes.l1.mission_clock import EnergyModel, FlightModel

    spec = FerrySpec(flight=FlightModel(energy=EnergyModel(capacity_j=3_150.0)),
                     in_flight_response=response)
    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-a", "dev-b"),
                       ferry=spec, mission_budget_s=100.0)
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"], ["dev-b"], ["dev-c"]]
    (event,) = getattr(r, key)
    assert event["t_s"] == pytest.approx(T0 + at)
    if response == "abort":
        assert event["reason"] == "energy" and event["abandoned"] == [["dev-d"]]
    else:
        assert event["dropped"] == [{"devices": ["dev-d"], "reason": "energy"}]
    last = r.pass_1_flown[-1]
    energy = spec.flight.energy
    assert last["depart_energy_j"] == pytest.approx(
        energy.p_move_w * 4.0 + energy.p_hover_w * (3.0 + 2.0))


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_d4_flies_on_and_records_its_overrun(response):
    """D4 (FedEx) declares no in-flight check: on the clock it flies its
    whole tour in either response and the overrun is measured (design 3.2)."""
    from hermes.scheduler.policies import FedExCarpPolicy

    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-a",),
                       ferry=FerrySpec(in_flight_response=response),
                       target_selector=FedExCarpPolicy(depot=DOCK), mission_budget_s=10.0)
    r = rec.result
    assert sorted(d for s in r.pass_1_flown for d in s["devices"]) == sorted(d for d, _ in LINE)
    assert r.aborts == [] and r.replans == []
    landing = r.pass_1_flown[-1]["end_s"] + math.dist(r.pass_1_flown[-1]["position"], DOCK) / 5.0
    assert r.budget_overrun_s > 0.0
    assert r.budget_overrun_s == pytest.approx(landing - (r.sim_start_s + 10.0))


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_the_predicate_holds_at_every_departure(response):
    """At every actual Pass-1 departure the next stop is admitted from the
    state the mule was in, with its return-and-upload tail (design 3.4)."""
    for budget in (28.0, 40.0):
        _, sup, (rec,) = _line_scenario(response, adapter=False) if budget == 28.0 else \
            fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-a", "dev-c"),
                ferry=FerrySpec(in_flight_response=response), mission_budget_s=budget)
        _check_departures(sup, rec.result, budget)
    # The golden layout on wide at 1 MB each way, the deadline law restated
    # for simulated seconds (spec Q1): the recorded 60 s window would leave
    # most stops overdue before takeoff.
    for budget in (430.0, 520.0):
        _, sup, recs = fly(ferry=full_spec(in_flight_response=response), missions=3,
                           mission_budget_s=budget, deadline_time_scale=20.0)
        assert sum(len(rec.result.pass_1_flown) for rec in recs) >= 9
        for rec in recs:
            _check_departures(sup, rec.result, budget)


def _check_departures(sup, r, budget):
    model = sup.scheduler.feasibility_model
    rule = sup.scheduler.in_flight_rule(COLLECT)
    budget_end = r.sim_start_s + budget
    assert r.pass_1_flown
    for s in r.pass_1_flown:
        state = FlightState(tuple(s["depart_pose"]), s["depart_s"], s["depart_energy_j"])
        verdict = model.admit(state, _stop_wp(s), rule=rule, budget_end=budget_end,
                              pass_kind=COLLECT)
        assert verdict.ok, (s, verdict)
        assert verdict.home <= budget_end


# --------------------------------------------------------------------------- #
# What Deadline(j) bounds: the route-level delivery bound (clock F1)
# --------------------------------------------------------------------------- #

#: The final check's two-stop line: dev-a at 10 m, dev-b at 40 m.
F1_LINE = (("dev-a", (10.0, 0.0, 0.0)), ("dev-b", (40.0, 0.0, 0.0)))


def _cut(**windows):
    def before(w, sup, m):
        for did, phi in windows.items():
            sup.scheduler.device_states[DeviceID(did)].deadline_fulfilment_s = phi
    return before


def _fly_bounds(bounds, response, *, layout, windows, silent=()):
    """One noise-free mission (no band, no upload time) with S3c on."""
    w, sup, (rec,) = fly(layout=layout, flaky={}, rf_range_m=5.0, silent=silent,
                         before=_cut(**windows),
                         ferry=FerrySpec(deadline_bounds=bounds, in_flight_response=response),
                         mission_budget_s=100.0,
                         mission_window_adapter=MissionWindowAdapter(enabled=True, window=2))
    (up,) = w.server.ups
    return sup, rec, up


def _clean_delivery_margins(r, up):
    """For every CLEAN line: its Deadline(j) minus when its update reached the
    cluster (the UP's ``sim_upload_ts``); negative is late."""
    deadlines = r.pass_1_device_deadlines
    return {str(l.device_id): deadlines[l.device_id] - up.sim_upload_ts
            for l in r.report.lines if l.outcome is MissionOutcome.CLEAN}


def _check_departures_on_board(sup, r):
    """The predicate held at every actual Pass-1 departure from the state the
    mule was in, the updates it then carried included: ``deliver_by``
    rebuilt from the CLEAN members of the stops already flown."""
    model = sup.scheduler.feasibility_model
    outcomes = {str(l.device_id): l.outcome for l in r.report.lines}
    on_board = math.inf
    for s in r.pass_1_flown:
        state = FlightState(tuple(s["depart_pose"]), s["depart_s"], s["depart_energy_j"],
                            on_board)
        verdict = model.admit(state, _stop_wp(s), rule=sup.scheduler.in_flight_rule(COLLECT),
                              budget_end=r.sim_start_s + 100.0, pass_kind=COLLECT)
        assert verdict.ok, (s, verdict)
        for d in s["devices"]:
            if outcomes[d] is MissionOutcome.CLEAN:
                on_board = min(on_board, r.pass_1_device_deadlines[DeviceID(d)])
    return on_board


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_delivery_drops_a_stop_that_would_land_an_update_late_before_takeoff(response):
    """Clock F1 end to end. dev-a (10 m) is due 6 s after takeoff; channel-
    free, no upload time. ``delivery_per_stop`` admits a (its own return is
    home at 5 s) and b, and lands a's update at 18 s, 12 s late.
    ``delivery`` admits a, then refuses b before takeoff with the reason
    ``delivery`` (b's own home, 18 s, meets b's own deadline): b is widened
    at takeoff and counted in S3c's planned, and the mission lands a's
    update at 5 s."""
    sup, rec, up = _fly_bounds("delivery_per_stop", response, layout=F1_LINE,
                               windows={"dev-a": 6.0})
    r = rec.result
    t0, due_a = r.sim_start_s, r.pass_1_device_deadlines[DeviceID("dev-a")]
    assert due_a == t0 + 6.0
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-a"], ["dev-b"]]
    assert up.sim_upload_ts == t0 + 18.0 and _clean_delivery_margins(r, up)["dev-a"] == -12.0
    assert r.pass_1_preflight_drops == []
    assert r.delivery_overrun_s is None              # recorded under "delivery" only

    sup, rec, up = _fly_bounds("delivery", response, layout=F1_LINE, windows={"dev-a": 6.0})
    r = rec.result
    t0, deadlines = r.sim_start_s, r.pass_1_device_deadlines
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-a"]]
    (drop,) = r.pass_1_preflight_drops
    assert (drop["devices"], drop["position"], drop["reason"]) == (
        ["dev-b"], [40.0, 0.0, 0.0], "delivery")
    assert drop["deadline_ts"] == deadlines[DeviceID("dev-b")] >= t0 + 18.0   # not late itself
    assert r.aborts == [] and r.replans == []
    widened = _widened(rec)
    assert set(widened) == {"dev-b"} and widened["dev-b"].contact_ts == t0
    assert sup.scheduler._window_adapter._history[-1] == (1, 2)
    assert up.sim_upload_ts == t0 + 5.0
    margins = _clean_delivery_margins(r, up)
    assert margins == {"dev-a": 1.0}
    assert _check_departures_on_board(sup, r) == deadlines[DeviceID("dev-a")]
    assert r.delivery_overrun_s == 0.0


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_delivery_holds_the_rest_of_a_flight_to_the_updates_on_board(response):
    """In flight. LINE, dev-e due 21 s after takeoff, dev-a silent. The plan
    (e, a, b, c, d; 1 s per contact) lands at 21 s, so S3b and the flown
    order keep all five. a's listen window costs 1 s: from then on d would
    land e's update (on board, CLEAN) at 22 s. ``abort`` finds out when d
    is next, at c; ``replan`` folds the whole remainder and drops d at a.
    Either way d is refused as ``delivery`` and widened at the drop time,
    and the mission lands at 17 s: every CLEAN update is on time. a's own
    update never came, so its deadline never goes on board.
    ``delivery_per_stop`` flies d and lands e's update 1 s late."""
    layout_kw = dict(layout=LINE, windows={"dev-e": 21.0}, silent=("dev-a",))
    sup, rec, up = _fly_bounds("delivery_per_stop", response, **layout_kw)
    r = rec.result
    assert len(r.pass_1_flown) == 5 and r.aborts == [] and r.replans == []
    assert up.sim_upload_ts == r.sim_start_s + 22.0
    assert _clean_delivery_margins(r, up)["dev-e"] == -1.0
    assert r.delivery_overrun_s is None

    sup, rec, up = _fly_bounds("delivery", response, **layout_kw)
    r = rec.result
    t0, deadlines = r.sim_start_s, r.pass_1_device_deadlines
    assert deadlines[DeviceID("dev-e")] == t0 + 21.0
    assert all(deadlines[DeviceID(d)] >= t0 + 22.0 for d in ("dev-a", "dev-b", "dev-c", "dev-d"))
    assert r.pass_1_preflight_drops == []
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"], ["dev-b"], ["dev-c"]]
    if response == "abort":
        (abort,) = r.aborts
        assert abort["t_s"] == pytest.approx(t0 + 11.0)
        assert (abort["reason"], abort["abandoned"]) == ("delivery", [["dev-d"]])
        dropped_at = abort["t_s"]
    else:
        (event,) = r.replans
        assert event["t_s"] == pytest.approx(t0 + 5.0) and event["order_used"] == "arm"
        assert event["rejected"] == [{"devices": ["dev-d"], "reason": "delivery"}]
        assert event["dropped"] == [{"devices": ["dev-d"], "reason": "delivery"}]
        assert event["route"] == [["dev-b"], ["dev-c"]]
        assert sup.scheduler.last_order_check.order_used == "current"
        dropped_at = event["t_s"]
    widened = _widened(rec)
    assert set(widened) == {"dev-d"} and widened["dev-d"].contact_ts == dropped_at
    assert sup.scheduler._window_adapter._history[-1] == (4, 5)
    assert up.sim_upload_ts == pytest.approx(t0 + 17.0)
    margins = _clean_delivery_margins(r, up)
    assert set(margins) == {"dev-e", "dev-b", "dev-c"} and min(margins.values()) >= 0.0
    assert margins["dev-e"] == pytest.approx(4.0)
    assert _check_departures_on_board(sup, r) == t0 + 21.0
    assert r.delivery_overrun_s == 0.0


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_delivery_cannot_recheck_a_contact_that_overruns_where_the_flight_ends(response):
    """The limit of the route-level bound. LINE, dev-e due 21 s after
    takeoff, dev-d silent. The plan lands at 21 s and every departure check
    passes: d's listen window, the 1 s nobody priced, comes after the last
    one. With e's update on board and nothing left to decide, the mule
    flies home and lands it 1 s late. ``delivery_overrun_s`` records that,
    as ``budget_overrun_s`` records a budget overrun; it is 0 in the
    missions above."""
    sup, rec, up = _fly_bounds("delivery", response, layout=LINE, windows={"dev-e": 21.0},
                               silent=("dev-d",))
    r = rec.result
    t0 = r.sim_start_s
    assert [s["devices"] for s in r.pass_1_flown] == [[d] for d, _ in LINE]
    assert [s["listen_s"] for s in r.pass_1_flown] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert r.aborts == [] and r.replans == [] and r.pass_1_preflight_drops == []
    assert _check_departures_on_board(sup, r) == t0 + 21.0       # every departure passed
    assert up.sim_upload_ts == t0 + 22.0
    assert _clean_delivery_margins(r, up)["dev-e"] == -1.0
    assert r.delivery_overrun_s == 1.0


#: LINE with its first stop shared: dev-f and dev-e 1 m either side of the
#: line at 3 m, so S3a's stop (their centroid) is LINE's first and the flight
#: times are LINE's.
SHARED = (("dev-f", (3.0, 1.0, 0.0)), ("dev-e", (3.0, -1.0, 0.0))) + LINE[1:]


@pytest.mark.parametrize("response", ["abort", "replan"])
@pytest.mark.parametrize("layout, silent", [(LINE, "dev-e"), (SHARED, "dev-f")],
                         ids=["alone", "shared"])
def test_a_silent_members_deadline_never_goes_on_board(response, layout, silent):
    """Only an update actually collected is on board. The silent member is
    due 21 s after takeoff, the tightest deadline, at the first stop: alone
    (LINE's dev-e), or sharing it with dev-e, which answers and keeps the
    default window (the stop's ``deadline_ts`` is the pair's minimum). S3b
    folds the stop's deadline before takeoff, as it assumes every planned
    member answers, and the plan lands at 21 s. In flight the silent
    member's listen window costs 1 s, and its deadline, whose update never
    came, stays off board: every stop is flown and the mission lands at
    22 s, after that deadline, with every collected update on time. Held
    to the silent member's deadline, or to the stop's, d would be refused."""
    sup, rec, up = _fly_bounds("delivery", response, layout=layout, windows={silent: 21.0},
                               silent=(silent,))
    r = rec.result
    t0, deadlines = r.sim_start_s, r.pass_1_device_deadlines
    first = r.pass_1_flown[0]
    assert silent in first["devices"] and first["listen_s"] == 1.0
    assert first["deadline_ts"] == deadlines[DeviceID(silent)] == t0 + 21.0
    assert [s["devices"] for s in r.pass_1_flown[1:]] == [[d] for d, _ in LINE[1:]]
    assert r.aborts == [] and r.replans == [] and r.pass_1_preflight_drops == []
    outcomes = {str(l.device_id): l.outcome for l in r.report.lines}
    assert outcomes.pop(silent) is MissionOutcome.TIMEOUT
    assert set(outcomes.values()) == {MissionOutcome.CLEAN}
    assert up.sim_upload_ts == t0 + 22.0 > deadlines[DeviceID(silent)]
    margins = _clean_delivery_margins(r, up)
    assert set(margins) == set(outcomes) and min(margins.values()) >= 0.0
    assert r.delivery_overrun_s == 0.0
    assert _check_departures_on_board(sup, r) == min(deadlines[DeviceID(d)] for d in outcomes)


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_the_beacon_hook_holds_an_insert_to_the_updates_on_board(response):
    """dev-x (50 m) is offered while the mule serves dev-e, due 21 s after
    takeoff, so at the next departure e's update is on board. Under
    ``delivery_per_stop`` the offer fits and is inserted, and the mission
    lands e's update 5 s late. Under ``delivery`` the hook folds the edited
    remainder from the departure state, e's deadline included, and refuses
    the offer; the plan's five stops land e's update at 21 s. From a state
    that carried nothing every place would pass: only the update on board
    stood in the way."""
    def hook(w, sup, m):
        sup.scheduler.device_states[DeviceID("dev-e")].deadline_fulfilment_s = 21.0
        _offer_during(sup, "run_contact", ["dev-x"])

    def run(bounds):
        w, sup, (rec,) = _beacon_world([], budget=100.0, hook=hook, ferry=FerrySpec(
            deadline_bounds=bounds, in_flight_response=response))
        (up,) = w.server.ups
        return sup, rec.result, up

    _, r, up = run("delivery_per_stop")
    (insert,) = r.inserts
    assert insert["devices"] == ["dev-x"] and insert["t_s"] == r.pass_1_flown[1]["depart_s"]
    assert up.sim_upload_ts == r.sim_start_s + 26.0
    assert r.pass_1_device_deadlines[DeviceID("dev-e")] - up.sim_upload_ts == -5.0

    sup, r, up = run("delivery")
    t0, depart = r.sim_start_s, r.pass_1_flown[1]
    assert r.inserts == []
    assert r.offers_refused == [
        {"t_s": depart["depart_s"], "devices": ["dev-x"], "reason": "does not fit"}]
    assert [s["devices"] for s in r.pass_1_flown] == [[d] for d, _ in LINE]
    assert up.sim_upload_ts == t0 + 21.0 and r.delivery_overrun_s == 0.0
    x = ContactWaypoint(position=(50.0, 0.0, 0.0), devices=(DeviceID("dev-x"),),
                        bucket=Bucket.BEACON_ACTIVE, deadline_ts=math.inf)
    rest = [_stop_wp(s) for s in r.pass_1_flown[1:]]
    free = FlightState(tuple(depart["depart_pose"]), depart["depart_s"],
                       depart["depart_energy_j"])
    carrying = dataclasses.replace(free, deliver_by=t0 + 21.0)
    for i in range(len(rest) + 1):
        edited = rest[:i] + [x] + rest[i:]
        assert sup.scheduler.fold_remainder(edited, state=free, budget_end=t0 + 100.0).ok
        assert not sup.scheduler.fold_remainder(edited, state=carrying,
                                                budget_end=t0 + 100.0).ok


@pytest.mark.parametrize("response", ["abort", "replan"])
def test_an_inserted_members_own_deadline_goes_on_board_not_its_stops(response):
    """A beacon insert keeps each member's own Deadline(j) for the on-board
    bound. The offer (dev-x, dev-w) is taken at takeoff: a stop 5 m behind
    the dock, at dev-x, with dev-w 2 m off and due 24.5 s after takeoff
    (dev-x keeps the default window), so the stop's ``deadline_ts`` is
    dev-w's. It fits first, and the edited plan lands at 24 s. dev-w is
    silent: its listen window costs 1 s, and dev-x's update, on board, is
    held to dev-x's own deadline, not the stop's, so every stop is flown and
    the mission lands at 25 s, after dev-w's deadline, with nothing late.
    Held to the stop's deadline, d would be refused."""
    def hook(w, sup, m):
        sup.scheduler.device_states[DeviceID("dev-w")].deadline_fulfilment_s = 24.5

    w, sup, (rec,) = _beacon_world(
        [("dev-x", "dev-w")], budget=100.0, hook=hook, silent=("dev-w",),
        known=(("dev-x", (-5.0, 0.0, 0.0)), ("dev-w", (-5.0, 2.0, 0.0))),
        extra={"dev-w": (-5.0, 2.0, 0.0)},
        ferry=FerrySpec(deadline_bounds="delivery", in_flight_response=response))
    (up,) = w.server.ups
    r = rec.result
    t0 = r.sim_start_s
    (insert,) = r.inserts
    assert (insert["devices"], insert["index"], insert["t_s"]) == (["dev-x", "dev-w"], 0, t0)
    assert insert["home_s"] == pytest.approx(t0 + 24.0)
    first = r.pass_1_flown[0]
    assert first["deadline_ts"] == t0 + 24.5 and first["listen_s"] == 1.0
    assert [s["devices"] for s in r.pass_1_flown[1:]] == [[d] for d, _ in LINE]
    assert r.aborts == [] and r.replans == []
    outcomes = {str(l.device_id): l.outcome for l in r.report.lines}
    assert outcomes.pop("dev-w") is MissionOutcome.TIMEOUT
    assert set(outcomes.values()) == {MissionOutcome.CLEAN}
    assert up.sim_upload_ts == pytest.approx(t0 + 25.0)
    assert r.delivery_overrun_s == 0.0


def test_replan_repairs_pass_2_when_its_budget_is_on():
    """Pass 2 in flight: a silent device's listen window makes the tail miss
    the Pass-2 budget; replan drops what no longer fits as SKIPPED lines."""
    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-e",),
                       ferry=FerrySpec(in_flight_response="replan"),
                       mission_budget_s=21.0, pass_2_budget=True)
    r = rec.result
    replans = [e for e in r.replans if e["pass"] == "deliver"]
    assert replans and replans[0]["dropped"]
    lines = {str(l.device_id): l for l in r.delivery_report.lines}
    for drop in replans[0]["dropped"]:
        (did,) = drop["devices"]
        assert lines[did].outcome is DeliveryOutcome.SKIPPED
        assert lines[did].contact_ts == replans[0]["t_s"]
    assert r.pass_2_budget_overrun_s == 0.0


def test_abort_does_not_check_pass_2_in_flight():
    """``abort`` is the recorded rule on the clock, and the recorded rule
    walks a budgeted Pass 2 before takeoff only. The same scenario as above:
    dev-e's listen window makes the walked Pass 2 land 1 s past its budget,
    and the mule flies all of it and records the overrun."""
    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, silent=("dev-e",),
                       ferry=FerrySpec(in_flight_response="abort"),
                       mission_budget_s=21.0, pass_2_budget=True)
    r = rec.result
    assert [a["pass"] for a in r.aborts] == ["collect"]     # Pass 1 gave d up
    assert [s["devices"] for s in r.pass_2_flown] == [[d] for d, _ in LINE]
    assert r.pass_2_budget_overrun_s == pytest.approx(1.0)
    lines = {str(l.device_id): l.outcome for l in r.delivery_report.lines}
    assert DeliveryOutcome.SKIPPED not in lines.values()


def test_pass_2_is_a_sortie_of_its_own_for_the_energy_clause():
    """The Pass-2 walk starts from ``FlightState(DOCK, t2)`` with nothing
    spent (design section 3.5), and the in-flight check under ``replan``
    counts the same way: only the energy spent since t2. A 3.15 kJ battery
    (simulated) fits each LINE pass (3.14 kJ) but not both, and Pass 2 still
    flies every stop without a re-plan."""
    from hermes.l1.mission_clock import EnergyModel, FlightModel

    spec = FerrySpec(flight=FlightModel(energy=EnergyModel(capacity_j=3_150.0)),
                     in_flight_response="replan")
    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0, ferry=spec,
                       mission_budget_s=100.0, pass_2_budget=True)
    r = rec.result
    energy = spec.flight.energy
    assert [s["devices"] for s in r.pass_2_flown] == [[d] for d, _ in LINE]
    assert r.replans == [] and r.aborts == []
    spent = 0.0
    for s in r.pass_2_flown:
        assert s["depart_energy_j"] == pytest.approx(spent, abs=1e-9)
        spent += energy.p_move_w * s["transit_s"] + energy.p_hover_w * (s["dwell_s"] + s["listen_s"])
    home = spec.flight.leg_s(r.pass_2_flown[-1]["position"], DOCK)
    assert spent + energy.p_move_w * home <= 3_150.0
    assert r.energy_j > 3_150.0                                 # both passes together


# --------------------------------------------------------------------------- #
# The beacon hook (design section 3.6)
# --------------------------------------------------------------------------- #

def _offer(*devices):
    return ContactWaypoint(position=(0.0, 0.0, 0.0), devices=tuple(DeviceID(d) for d in devices),
                           bucket=Bucket.BEACON_ACTIVE, deadline_ts=0.0)


def _beacon_world(offers, *, budget=None, known=(("dev-x", (50.0, 0.0, 0.0)),),
                  state_kw=None, hook=None, missions=1, extra=None, **sup_kw):
    """LINE plus devices outside the slice; ``offers`` are queued before each
    mission's takeoff, ``hook(w, sup, m)`` runs after that. ``extra`` adds
    devices to the link besides dev-x and dev-y."""
    def setup(w, sup, m):
        for did, pos in known:
            sup.scheduler.device_states[DeviceID(did)] = DeviceSchedulerState(
                device_id=DeviceID(did), last_known_position=pos, **(state_kw or {}))
        for devices in offers:
            sup.offer_contact(_offer(*devices))
        if hook is not None:
            hook(w, sup, m)

    extra = {"dev-x": (50.0, 0.0, 0.0), "dev-y": (15.0, 30.0, 0.0), **(extra or {})}
    kw = {} if budget is None else {"mission_budget_s": budget}
    return fly(layout=LINE, flaky={}, rf_range_m=5.0, before=setup, missions=missions,
               world_kw={"extra_devices": extra},
               mission_window_adapter=MissionWindowAdapter(enabled=True, window=2),
               **kw, **sup_kw)


def _offer_during(sup, name, devices, *, call=1):
    """Offer ``devices`` while the host runs its ``call``-th ``name`` contact
    (``run_contact`` in Pass 1, ``deliver_contact`` in Pass 2): a beacon
    heard in flight."""
    real, calls = getattr(sup.mission, name), []

    def contact(*args, **kw):
        calls.append(None)
        if len(calls) == call:
            sup.offer_contact(_offer(*devices))
        return real(*args, **kw)

    setattr(sup.mission, name, contact)


def test_an_offer_that_fits_is_inserted_at_its_cheapest_place():
    _, sup, (rec,) = _beacon_world([("dev-x",)])
    r = rec.result
    (insert,) = r.inserts
    assert insert["devices"] == ["dev-x"] and insert["position"] == [50.0, 0.0, 0.0]
    # Brute force over every place in the takeoff remainder.
    plan = list(r.pass_1_queue)
    stop = sup._ferry_run.annotate([ContactWaypoint(
        position=(50.0, 0.0, 0.0), devices=(DeviceID("dev-x"),), bucket=Bucket.BEACON_ACTIVE,
        deadline_ts=T0 + 60.0)], {DeviceID("dev-x"): (50.0, 0.0, 0.0)})[0]
    homes = [sup.scheduler.fold_remainder(plan[:i] + [stop] + plan[i:],
                                          state=FlightState(DOCK, T0), budget_end=None).home
             for i in range(len(plan) + 1)]
    assert insert["index"] == homes.index(min(homes))
    assert insert["home_s"] == min(homes)
    flown = [s["devices"] for s in r.pass_1_flown]
    assert flown[insert["index"]] == ["dev-x"] and len(flown) == 6
    outcomes = {str(l.device_id): l.outcome for l in r.report.lines}
    assert outcomes["dev-x"] is MissionOutcome.CLEAN
    # Counted in planned and served; never put in the slice.
    assert sup.scheduler._window_adapter._history[-1] == (6, 6)
    assert sup.scheduler.device_states[DeviceID("dev-x")].is_in_slice is False
    assert r.offers_refused == []


def test_offers_that_cannot_be_served_are_refused():
    """With a 21 s budget LINE fits exactly, so nothing more can."""
    _, sup, (rec,) = _beacon_world(
        [("dev-x",), ("dev-zz",), ("dev-a",), ("dev-y",), ("dev-x", "dev-e")],
        budget=21.0, known=(("dev-x", (50.0, 0.0, 0.0)), ("dev-y", (0.0, 0.0, 0.0))))
    r = rec.result
    assert r.inserts == []
    reasons = {tuple(o["devices"]): o["reason"] for o in r.offers_refused}
    assert reasons == {
        ("dev-x",): "does not fit",
        ("dev-zz",): "unknown device",
        ("dev-a",): "already planned this mission",
        ("dev-y",): "position unknown",
        ("dev-x", "dev-e"): "already planned this mission",
    }
    assert [s["devices"] for s in r.pass_1_flown] == [[d] for d, _ in LINE]
    assert sup.scheduler._window_adapter._history[-1] == (5, 5)


def test_members_of_one_offer_must_share_a_stop():
    _, _, (rec,) = _beacon_world(
        [("dev-x", "dev-y")],
        known=(("dev-x", (50.0, 0.0, 0.0)), ("dev-y", (15.0, 30.0, 0.0))))
    assert rec.result.offers_refused[0]["reason"] == "members not within range of one stop"


def test_an_offer_naming_a_device_twice_is_refused_not_flown():
    """A stop solicits each member once (its contact plan refuses repeats).
    Accepted, such an offer would end the mission at that stop, after its
    leg was charged, and the next takeoff would fail away from the dock;
    it is refused at the departure instead."""
    _, _, (rec,) = _beacon_world([("dev-x", "dev-x")])
    r = rec.result
    assert r.inserts == []
    assert r.offers_refused == [
        {"t_s": T0, "devices": ["dev-x", "dev-x"], "reason": "members repeat"}]
    assert [s["devices"] for s in r.pass_1_flown] == [[d] for d, _ in LINE]
    assert rec.pose_after == DOCK


def test_a_device_inserted_once_is_not_inserted_again_that_mission():
    """dev-x is inserted at takeoff and offered again during the first
    contact: at the second departure it is already planned."""
    _, sup, (rec,) = _beacon_world(
        [("dev-x",)], hook=lambda w, sup, m: _offer_during(sup, "run_contact", ["dev-x"]))
    r = rec.result
    assert [i["devices"] for i in r.inserts] == [["dev-x"]] and r.inserts[0]["t_s"] == T0
    assert r.offers_refused == [{"t_s": r.pass_1_flown[1]["depart_s"], "devices": ["dev-x"],
                                 "reason": "already planned this mission"}]
    assert [s["devices"] for s in r.pass_1_flown].count(["dev-x"]) == 1
    assert sup.scheduler._window_adapter._history[-1] == (6, 6)


def test_a_device_dropped_before_takeoff_is_not_brought_back():
    """With a 10 s budget S3b keeps e and a and drops b, c and d before
    takeoff; they were widened then and count in planned, so the hook may
    not insert them that mission."""
    _, sup, (rec,) = _beacon_world([("dev-b",)], budget=10.0)
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["dev-e"], ["dev-a"]]
    assert r.inserts == []
    assert r.offers_refused == [
        {"t_s": T0, "devices": ["dev-b"], "reason": "already planned this mission"}]
    assert sup.scheduler._window_adapter._history[-1] == (2, 5)


def test_an_offer_overdue_everywhere_does_not_fit():
    """The inserted stop carries its members' tightest Deadline(j), computed
    at the departure: dev-x's 5 s window closes before the mule can be there
    (10 s away at best), so no place in the remainder passes, although the
    100 s budget alone would take it."""
    _, sup, (rec,) = _beacon_world([("dev-x",)], budget=100.0,
                                   state_kw={"deadline_fulfilment_s": 5.0})
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [[d] for d, _ in LINE]
    assert r.inserts == []
    assert r.offers_refused == [{"t_s": T0, "devices": ["dev-x"], "reason": "does not fit"}]
    # Only the deadline stood in the way.
    st = sup.scheduler.device_states[DeviceID("dev-x")]
    late = ContactWaypoint(position=(50.0, 0.0, 0.0), devices=(DeviceID("dev-x"),),
                           bucket=Bucket.BEACON_ACTIVE, deadline_ts=math.inf)
    plan = list(r.pass_1_queue)
    assert sup.scheduler.fold_remainder(plan + [late], state=FlightState(DOCK, T0),
                                        budget_end=T0 + 100.0).ok
    assert st.deadline_fulfilment_s == 5.0


def test_an_offer_heard_during_pass_2_waits_for_the_next_takeoff():
    """The hook serves Pass 1 (design section 3.6): an offer that comes in
    during Pass 2 stays queued and is taken at the next mission's first
    departure, where it fits."""
    def hook(w, sup, m):
        if m == 0:
            _offer_during(sup, "deliver_contact", ["dev-x"])

    _, sup, recs = _beacon_world([], hook=hook, missions=2)
    first, second = recs[0].result, recs[1].result
    assert first.inserts == [] and first.offers_refused == []
    assert [s["devices"] for s in first.pass_2_flown] == [[d] for d, _ in LINE]
    (insert,) = second.inserts
    assert insert["devices"] == ["dev-x"] and insert["t_s"] == second.sim_start_s
    assert [s["devices"] for s in second.pass_1_flown].count(["dev-x"]) == 1
    assert second.offers_refused == [] and sup._offers == []


def test_the_beacon_hook_is_inert_without_offers():
    _, _, (rec,) = fly(layout=LINE, flaky={}, rf_range_m=5.0)
    assert rec.result.inserts == [] and rec.result.offers_refused == []


# --------------------------------------------------------------------------- #
# The L1 state (design section 4.6)
# --------------------------------------------------------------------------- #

class RecordingActor:
    """A channel actor that records each state and answers 0, 1, 2, 0, ..."""

    def __init__(self):
        self.states = []

    def argmax(self, state):
        self.states.append(np.array(state, dtype=np.float32))
        return (len(self.states) - 1) % 3


def test_the_l1_choice_is_recorded_from_what_the_mule_observed_at_arrival():
    actor = RecordingActor()
    _, sup, (rec,) = fly(ferry=full_spec(), channel_actor=actor, mission_budget_s=600.0,
                         deadline_time_scale=20.0)
    r, fx = rec.result, sup._ferry_run
    stops = r.pass_1_flown + r.pass_2_flown
    assert len(r.pass_1_flown) == 5 and len(actor.states) == len(stops)
    assert [s["l1_choice"] for s in stops] == [k % 3 for k in range(len(stops))]
    assert r.pass_1_channel_choices == [s["l1_choice"] for s in r.pass_1_flown]
    positions = sup._ferry_positions([_stop_wp(s) for s in stops])
    energy = fx.spec.flight.energy
    takeoffs = {0, len(r.pass_1_flown)}                 # each pass's first stop
    for k, (s, state) in enumerate(zip(stops, actor.states)):
        obs = fx.observe(_stop_wp(s), positions, s["arrival_s"])
        np.testing.assert_allclose(state[:3], np.float32(np.array(obs.class_snr_db[:3]) / 30.0))
        assert state[3] == np.float32(obs.max_slant_m / 100.0)
        np.testing.assert_allclose(state[4:7], np.float32(np.array(s["position"]) / 100.0))
        # Slot 7 at EVERY stop (final check, clock F2): the sortie's energy
        # at arrival, the departure state's plus the leg; Pass 2 restarts at
        # 0 at its takeoff, as the energy clause does.
        spent = s["depart_energy_j"] + energy.p_move_w * s["transit_s"]
        assert state[7] == pytest.approx(1.0 - spent / (energy.p_hover_w * 600.0), abs=1e-6)
        assert (s["depart_energy_j"] == 0.0) if k in takeoffs else (s["depart_energy_j"] > 0.0)
        assert state[7] <= 1.0
        if k < len(r.pass_1_flown):
            # Pass 1 is held to the budget at every departure, so every
            # arrival is inside it, and flying costs less than hovering: the
            # energy is at most P_hover * budget. Pass 2 is not gated here.
            assert energy.p_move_w <= energy.p_hover_w and state[7] >= 0.0


def test_the_l1_energy_slot_counts_the_sortie_as_the_energy_clause_does():
    """Clock F2 of the final check: slot 7 counted the whole mission's
    energy, Pass 1's included, while the departure state and the energy
    clause count each sortie from its own takeoff (Pass 2 restarts at 0).
    With a 300 s budget on both passes and 1 MB each way, that reading fell
    below 0 in Pass 2. Under ``replan`` every departure of both passes is
    held to its pass's budget, so every arrival is inside it; flying costs
    less than hovering, so the sortie's energy is at most P_hover * budget
    = E_ref and 0 <= slot 7 <= 1 at every stop."""
    actor = RecordingActor()
    budget = 300.0
    _, sup, (rec,) = fly(ferry=full_spec(in_flight_response="replan"), channel_actor=actor,
                         mission_budget_s=budget, pass_2_budget=True, deadline_time_scale=20.0)
    r = rec.result
    energy = sup.ferry.flight.energy
    assert energy.capacity_j is None and energy.p_move_w <= energy.p_hover_w
    e_ref = energy.p_hover_w * budget
    n1, stops = len(r.pass_1_flown), r.pass_1_flown + r.pass_2_flown
    assert n1 and r.pass_2_flown and len(actor.states) == len(stops)
    last = r.pass_1_flown[-1]
    e_pass_1 = (last["depart_energy_j"]
                + energy.p_move_w * (last["transit_s"] + sup.ferry.flight.leg_s(last["position"], DOCK))
                + energy.p_hover_w * (last["dwell_s"] + last["listen_s"]))
    mission_reading = []
    for k, (s, state) in enumerate(zip(stops, actor.states)):
        sortie = s["depart_energy_j"] + energy.p_move_w * s["transit_s"]
        assert state[7] == pytest.approx(1.0 - sortie / e_ref, abs=1e-6)
        assert 0.0 <= state[7] <= 1.0
        mission_reading.append(1.0 - (sortie + (e_pass_1 if k >= n1 else 0.0)) / e_ref)
    assert min(mission_reading[n1:]) < 0.0, "no longer the case the fix is about"


def test_without_a_band_the_l1_choice_uses_the_recorded_state():
    actor = RecordingActor()
    _, sup, (rec,) = fly(channel_actor=actor, rf_prior_snr_db=12.0)
    r = rec.result
    for s, state in zip(r.pass_1_flown + r.pass_2_flown, actor.states):
        pose = np.array(s["depart_pose"])
        dist = float(np.sqrt(((pose - np.array(s["position"])) ** 2).sum()))
        np.testing.assert_allclose(
            state, np.float32([0.4, 0.4, 0.4, dist / 100.0, *(pose / 100.0), 1.0]))


# --------------------------------------------------------------------------- #
# The availability draw, the stamps and the backhaul floor
# --------------------------------------------------------------------------- #

def test_the_channel_reliability_source_drops_uplinks_by_the_keyed_draw():
    avail = {d: 1.0 for d, _ in GH.LAYOUT}
    avail["dev-01"] = 0.0
    spec = full_spec(contact_reliability_source="channel", device_availability=avail)
    w, _, (rec,) = fly(ferry=spec, flaky={})
    r = rec.result
    stop = next(s for s in r.pass_1_flown if "dev-01" in s["devices"])
    assert stop["uplink_dropped"] == ["dev-01"] and "dev-01" in stop["missing"]
    assert stop["listen_s"] == 1.0
    line = next(l for l in r.report.lines if str(l.device_id) == "dev-01")
    assert line.outcome is MissionOutcome.TIMEOUT and line.bytes_sent > 0
    assert line.bytes_received == 0
    delta = next(d for src, d in rec.deltas if src == "session" and str(d.device_id) == "dev-01")
    assert delta.answered is True
    # The device adopted the basis it was pushed.
    assert w.devices[DeviceID("dev-01")].last_push_round == r.mission_round
    others = {str(l.device_id): l.outcome for l in r.report.lines if str(l.device_id) != "dev-01"}
    assert set(others.values()) == {MissionOutcome.CLEAN}


def test_no_wall_stamp_reaches_the_scheduler_in_sim_mode():
    """Critic B3: devices stamp adverts, updates and acks with their wall
    clock (1.7e9 here); none of it may reach a delta, a line, a record or a
    device state."""
    def refuse(w, sup, m):
        state = FLState.UNAVAILABLE if m == 1 else FLState.FL_OPEN
        w.set_state(["dev-05"], state)

    _, sup, recs = fly(ferry=full_spec(), missions=3, before=refuse, mission_budget_s=520.0,
                       deadline_time_scale=20.0)
    assert all(len(rec.result.pass_1_flown) >= 3 for rec in recs)
    for rec in recs:
        r = rec.result
        assert all(d.contact_ts < H.SIM_CEILING for _, d in rec.deltas), rec.deltas
        assert all(s < H.SIM_CEILING for s in H.ledger_stamps(r))
        assert all(v < H.SIM_CEILING for v in H.state_stamps(sup.scheduler.device_states).values())
        assert sup.scheduler.mission_start_ts < H.SIM_CEILING
    served = [st for st in sup.scheduler.device_states.values() if st.last_clean_ts > 0]
    assert served and all(T0 <= st.last_clean_ts < H.SIM_CEILING for st in served)


def test_a_backhaul_below_the_floor_is_a_lost_upload_with_a_capped_charge():
    spec = full_spec()
    spec = dataclasses.replace(spec, backhaul=BackhaulChannel(
        salt=spec.backhaul.salt, period_s=800.0, base_db=-60.0))
    w, sup, (rec,) = fly(ferry=spec)
    r = rec.result
    link = spec.link
    cap = link.dwell_s(1_000_000, "wide", link.snr_floor_db)
    assert r.backhaul["below_floor"] and r.backhaul["p_loss"] == 1.0
    assert r.backhaul["upload_s"] == cap == r.sim_ledger["upload"]
    assert math.isfinite(r.sim_end_s)
    up = w.server.ups[0]
    assert up.backhaul.below_floor and up.sim_upload_ts == r.backhaul["t_upload_s"]
    # The planner prices the carrier the same way: finite.
    assert sup.scheduler.feasibility_model.ferry.upload_time_s() == cap


#: dev-b and dev-c 59 m either side of dev-a: inside R_planar(wide) = 60 m,
#: at the edge, where the realized SNR is below the floor on about one arrival
#: in ten (the 90 % edge availability, decision D1).
EDGE = (
    ("dev-a", (100.0, 0.0, 0.0)),
    ("dev-b", (100.0, 59.0, 0.0)),
    ("dev-c", (100.0, -59.0, 0.0)),
)


def test_an_in_range_member_below_the_floor_is_never_solicited_or_charged():
    """Critic B12 on the contact side, through the supervisor's plan (the
    gate by R_planar(b) AND the SNR floor at arrival). S3a anchors the stop
    on dev-a (the highest delivery priority), so b and c sit 59 m out; with
    this trial seed dev-c is below the floor when the mule arrives in both
    passes of the first mission. It is unreachable there: never solicited,
    no airtime and no listen window for it, TIMEOUT ``answered=False``
    (Pass 2: UNDELIVERED, nothing sent), stamped at the arrival. Solicited,
    its session could not be priced at all (the rate below the floor is 0)."""
    commits = []

    def setup(w, sup, m):
        sup.scheduler.device_states[DeviceID("dev-a")].delivery_priority = 1_000
        for name in ("run_contact", "deliver_contact"):
            real = getattr(sup.mission, name)

            def record(*args, _real=real, **kw):
                out = _real(*args, **kw)
                commits.append(sup.mission.last_contact)
                return out

            setattr(sup.mission, name, record)

    spec = full_spec(seed=27)
    floor = spec.link.snr_floor_db
    _, _, (rec,) = fly(ferry=spec, layout=EDGE, flaky={}, before=setup)
    r = rec.result
    stops = r.pass_1_flown + r.pass_2_flown
    assert len(stops) == len(commits) == 2
    below = []
    for s, commit in zip(stops, commits):
        assert s["position"] == [100.0, 0.0, 0.0]
        assert sorted(s["devices"]) == ["dev-a", "dev-b", "dev-c"]
        low = [d for d in s["devices"] if s["snr_db"][d] < floor]
        below.append(low)
        # Every member is within 59 m of the stop: only the floor gates.
        assert s["unreachable"] == low
        assert s["targets"] == [d for d in s["devices"] if d not in low]
        assert commit.arrival_ts == s["arrival_s"]
        assert [str(d) for d in commit.solicited] == s["targets"]
        assert [str(d) for d in commit.session_dwell_s] == s["targets"]   # airtime: targets only
        assert s["dwell_s"] == pytest.approx(sum(commit.session_dwell_s.values()))
        assert commit.missing == () and s["listen_s"] == 0.0
        for d in low:
            assert commit.contact_ts[DeviceID(d)] == s["arrival_s"]
    assert below == [["dev-c"], ["dev-c"]], "the seed no longer puts dev-c below the floor"
    arrival_1, arrival_2 = r.pass_1_flown[0]["arrival_s"], r.pass_2_flown[0]["arrival_s"]
    lines = {str(l.device_id): l for l in r.report.lines}
    assert lines["dev-c"].outcome is MissionOutcome.TIMEOUT
    assert (lines["dev-c"].bytes_sent, lines["dev-c"].bytes_received) == (0, 0)
    assert lines["dev-c"].contact_ts == arrival_1 and lines["dev-c"].snr_db < floor
    assert {lines[d].outcome for d in ("dev-a", "dev-b")} == {MissionOutcome.CLEAN}
    (delta,) = [d for src, d in rec.deltas if src == "session" and str(d.device_id) == "dev-c"]
    assert delta.outcome is MissionOutcome.TIMEOUT and delta.answered is False
    assert delta.contact_ts == arrival_1
    deliveries = {str(l.device_id): l for l in r.delivery_report.lines}
    assert deliveries["dev-c"].outcome is DeliveryOutcome.UNDELIVERED
    assert deliveries["dev-c"].bytes_sent == 0 and deliveries["dev-c"].contact_ts == arrival_2
    assert {deliveries[d].outcome for d in ("dev-a", "dev-b")} == {DeliveryOutcome.DELIVERED}


# --------------------------------------------------------------------------- #
# Two mules: the Lamport sync at the dock (design section 2.3)
# --------------------------------------------------------------------------- #

def test_the_lamport_sync_adopts_the_clusters_simulated_time():
    """Quorum 2: each mule's DOWN carries the latest upload the cluster has
    ingested, and the mule takes off for Pass 2 at max(its upload + the
    turnaround, that time)."""
    w = H.World(mule_ids=["mule-a", "mule-b"], assignment=GH.K2_ASSIGNMENT,
                min_participation=2, echo_sim_ts=True)
    ma, mb = MuleID("mule-a"), MuleID("mule-b")
    results = {}
    with H.Patched(w.clock):
        sa = w.supervisor(ma, sim=True, down_wait_s=30.0)
        sb = w.supervisor(mb, sim=True, down_wait_s=30.0)
        w.bootstrap()
        w.server.tasks[ma].append(lambda: results.setdefault("b", sb.run_one_mission()))
        results["a"] = sa.run_one_mission()
    downs = {str(d.mule_id): d for why, d in w.server.downs if why == "post-aggregation"}
    ups = {str(u.mule_id): u for u in w.server.ups}
    latest = max(u.sim_upload_ts for u in ups.values())
    for who, mule in (("a", "mule-a"), ("b", "mule-b")):
        r = results[who]
        assert downs[mule].cluster_sim_ts == latest
        t_up = ups[mule].sim_upload_ts
        assert r.sim_pass_2_start_s == max(t_up + 30.0, latest)
        assert r.sim_ledger["dock_wait"] == pytest.approx(max(0.0, latest - (t_up + 30.0)))
    assert any(results[k].sim_ledger["dock_wait"] > 0 for k in results)


def test_an_empty_mission_that_docks_prices_its_upload_and_syncs():
    """``dock_on_empty`` on the clock (critic B6's K = 2 case): quorum 2, the
    cluster echoing its simulated time, a seconds-axis backhaul. mule-b's
    devices all refuse, so its mission is empty and docks an empty partial
    inside mule-a's wait. That dock is priced like any upload: 0 bytes, so
    0 s, but a carrier, an SNR and a p_loss on the UP (what the cluster's
    keyed loss draw reads), and the held carrier and the RF prior advance.
    It ends with the Lamport sync: mule-b lands at max(its upload + the
    turnaround, the DOWN's cluster_sim_ts), here mule-a's later upload."""
    w = H.World(mule_ids=["mule-a", "mule-b"], assignment=GH.K2_ASSIGNMENT,
                min_participation=2, echo_sim_ts=True)
    ma, mb = MuleID("mule-a"), MuleID("mule-b")
    results = {}
    with H.Patched(w.clock):
        sa = w.supervisor(ma, sim=True, down_wait_s=30.0, dock_on_empty=True, ferry=full_spec())
        sb = w.supervisor(mb, sim=True, down_wait_s=30.0, dock_on_empty=True, ferry=full_spec())
        w.bootstrap()
        w.set_state([d for d, m in GH.K2_ASSIGNMENT.items() if m == "mule-b"], FLState.UNAVAILABLE)
        w.server.tasks[ma].append(lambda: results.setdefault("b", sb.run_one_mission()))
        results["a"] = sa.run_one_mission()
    rb = results["b"]
    assert rb.empty and rb.docked_empty and not rb.down_timeout
    ups = {str(u.mule_id): u for u in w.server.ups}
    up = ups["mule-b"]
    bh, spec = up.backhaul, sb.ferry
    # The upload, priced: the fixed carrier at its SNR when the upload starts.
    assert isinstance(bh, BackhaulUpload)
    assert bh.nbytes == 0 and bh.upload_s == 0.0 and not bh.below_floor
    assert bh.carrier == spec.backhaul.fixed_band()
    assert bh.snr_db == spec.backhaul.snr_db(bh.t_start_s, bh.carrier)
    assert bh.p_loss == loss_from_snr(bh.snr_db)
    assert up.sim_upload_ts == bh.t_start_s == rb.backhaul["t_upload_s"]
    assert rb.backhaul["carrier"] == bh.carrier and rb.backhaul["bytes"] == 0
    assert sb._ferry_run.carrier == bh.carrier and sb.rf_prior_snr_db == bh.snr_db
    assert rb.sim_ledger["upload"] == 0.0 and rb.sim_ledger["turnaround"] == 30.0
    # The sync: the DOWN carries the latest upload ingested, mule-a's.
    (down,) = [d for why, d in w.server.downs
               if why == "post-aggregation" and str(d.mule_id) == "mule-b"]
    t_up, latest = up.sim_upload_ts, ups["mule-a"].sim_upload_ts
    assert down.cluster_sim_ts == latest > t_up + 30.0
    assert rb.sim_ledger["dock_wait"] == pytest.approx(latest - (t_up + 30.0))
    assert rb.sim_end_s == max(t_up + 30.0, latest) == sb.mission_clock()
    assert sum(rb.sim_ledger.values()) == pytest.approx(rb.sim_end_s - rb.sim_start_s, abs=1e-6)
    assert rb.pass_2_flown == [] and rb.sim_pass_2_start_s is None


# --------------------------------------------------------------------------- #
# A DOWN the mission clock cannot take (critic A1/B3)
# --------------------------------------------------------------------------- #

def test_a_slice_the_clock_refuses_at_the_inter_pass_dock_ends_the_mission():
    """The inter-pass DOWN carries a cluster deadline override, a wall-clock
    stamp. The sim-mode scheduler refuses it and with it the slice;
    ``ClientCluster`` only logs that, so the supervisor raises rather than
    fly Pass 2 as if the dock had worked. Nothing wall-stamped got in."""
    w = H.World(layout=LINE, flaky={})
    mid = w.mule_ids[0]
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=True, rf_range_m=5.0)
        w.bootstrap()
        dispatch = w.cluster.dispatch_down_bundle

        def with_override(mule_id):
            bundle = dispatch(mule_id)
            bundle.cluster_amendments.deadline_overrides = {DeviceID("dev-a"): H.WALL_T0 + 60.0}
            sign_down_bundle(bundle)
            return bundle

        w.cluster.dispatch_down_bundle = with_override
        with pytest.raises(MuleSupervisorError, match="not ingested.*wall-clock stamps"):
            sup.run_one_mission()
    assert len(w.server.ups) == 1 and sup.mule_pose == DOCK
    assert not [c for c in w.rfs[mid].calls if c[0] == "solicit" and c[1] == "deliver"]
    assert all(st.deadline_override_ts is None for st in sup.scheduler.device_states.values())
    assert sup._ferry_slice_error is None
