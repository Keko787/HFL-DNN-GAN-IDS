"""Exp 5 addendum, Study 5.12: each device's local fit on the simulated clock.

Every recorded run charged a device's fit nothing on the simulated clock. With
``train_time_params`` (``--train-time-s`` and its shape flags) each device's
fit takes ``T_j`` simulated seconds and a Pass-1 contact before it ends finds
no update ready. Pinned:

* **The draw** (``experiments/exp4/compute.py``): settings validated and
  completed; ``T_j`` keyed by the seed and the device, the median without a
  spread, exactly ``round(share * N)`` stragglers at the factor.
* **The fit clock** (``hermes/mule/fit_clock.py``): the first takeoff starts
  every fit; a model reaching a device restarts its fit; a contact before the
  fit ends is not ready.
* **The contact**: ``ContactPlan.not_ready`` is validated; the host names the
  not-ready targets in the solicit, pushes them nothing, stamps them with no
  airtime and no listen as TIMEOUT, and lists them in ``ContactCommit.not_ready``
  (``pushed`` lists who got a model); the device answers its advert and waits
  for no push, keeping its basis.
* **The config, builder, driver and runner**: the two mule fields (set
  together, simulated clock only), the builder's draw over every device split
  per mule slice, the driver's refusal off the simulated clock, ``ferry_params``
  showing the settings, and the flags passed only when given.
* **A whole F trial**: each mission records ``train_fits`` (the first one
  every device's first fit, at the trial's start) and each Pass-1 stop
  ``not_ready``, exactly what a fit clock replayed over those records marks; a
  not-ready device gets no model there. The default trial gains no key.
"""

from __future__ import annotations

import json
from collections import deque

import pytest

from experiments.exp4 import runner_main
from experiments.exp4.compute import (
    check_train_time_params,
    device_train_times,
    stragglers,
)
from experiments.exp4.driver import Exp4Driver, train_time_ferry_params
from experiments.exp4.topology_builder import build_exp4_topology
from hermes.l1.mission_clock import MissionClock
from hermes.mule import MuleSupervisorError
from hermes.mule.fit_clock import FitClock
from hermes.processes.config import (
    ADDENDUM_MULE_FIELDS,
    TRAIN_TIME_MULE_FIELDS,
    MuleConfig,
    mule_config_errors,
    mule_config_to_json,
)
from hermes.transport import LoopbackRFLink
from hermes.types import DiscPush, FLOpenSolicit, MissionOutcome

from tests.unit import test_client_mission_ferry as CMF
from tests.unit import test_host_mission_ferry as HMF
from tests.unit.test_mule_clock_wiring import _sup

PARAMS = {"median_s": 40.0, "sigma": 0.5, "straggler_share": 0.2, "straggler_factor": 5.0}
IDS = [f"exp4-dev-{i:03d}" for i in range(12)]


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #

def test_the_settings_are_completed_and_refused_when_they_cannot_be_drawn():
    assert check_train_time_params(None) is None
    assert check_train_time_params({"median_s": 30}) == {
        "median_s": 30.0, "sigma": 0.0, "straggler_share": 0.0, "straggler_factor": 1.0}
    for bad, match in (({}, "median_s"), ({"median_s": -1.0}, ">= 0"),
                       ({"median_s": 1.0, "sigma": -0.1}, "sigma"),
                       ({"median_s": 1.0, "straggler_share": 1.5}, "share"),
                       ({"median_s": 1.0, "straggler_factor": 0.5}, "factor"),
                       ({"median_s": 1.0, "speed": 2.0}, "unknown"),
                       ({"median_s": float("inf")}, "finite")):
        with pytest.raises(ValueError, match=match):
            check_train_time_params(bad)
    with pytest.raises(TypeError):
        check_train_time_params({"median_s": True})


def test_each_device_draws_its_own_time_from_the_seed():
    times = device_train_times(IDS, seed=7, params=PARAMS)
    assert list(times) == IDS and all(t > 0 for t in times.values())
    assert times == device_train_times(IDS, seed=7, params=PARAMS)
    assert times != device_train_times(IDS, seed=8, params=PARAMS)
    # Without stragglers a device's time does not depend on the other devices.
    plain = dict(PARAMS, straggler_share=0.0)
    full = device_train_times(IDS, seed=7, params=plain)
    assert device_train_times(IDS[:5], seed=7, params=plain) == {d: full[d] for d in IDS[:5]}
    assert set(device_train_times(IDS, seed=7, params={"median_s": 30.0}).values()) == {30.0}


@pytest.mark.parametrize("n, k", [(6, 1), (12, 2), (24, 5), (5, 1), (2, 0)])
def test_exactly_round_share_n_devices_straggle_at_the_factor(n, k):
    ids = IDS[:n] if n <= len(IDS) else [f"d{i}" for i in range(n)]
    slow = stragglers(ids, seed=3, share=0.2)
    assert len(slow) == k
    with_slow = device_train_times(ids, seed=3, params=PARAMS)
    without = device_train_times(ids, seed=3, params=dict(PARAMS, straggler_share=0.0))
    for d in ids:
        assert with_slow[d] == pytest.approx(without[d] * (5.0 if d in slow else 1.0))


# --------------------------------------------------------------------------- #
# The fit clock
# --------------------------------------------------------------------------- #

def test_the_fit_clock_times_each_fit_from_the_last_model_received():
    fc = FitClock({"a": 10.0, "b": 50.0})
    with pytest.raises(RuntimeError, match="take_off"):
        fc.not_ready(["a"], 0.0)
    assert fc.take_off(100.0) == [("a", 100.0), ("b", 100.0)]
    assert fc.take_off(200.0) == [] and fc.origin == 100.0
    assert fc.not_ready(["a", "b", "c"], 105.0) == {"a", "b"}   # c has no time: ready
    assert fc.not_ready(["a", "b"], 110.0) == {"b"}             # a's fit ends at 110
    assert fc.received(["a", "c"], {"a": 120.0, "c": 121.0}) == [("a", 120.0)]
    assert fc.ready_at("a") == 130.0 and fc.ready_at("b") == 150.0
    assert fc.not_ready(["a", "b"], 129.9) == {"a", "b"}
    assert fc.not_ready(["a", "b"], 150.0) == frozenset()
    for bad in ({}, {"a": -1.0}, {"a": float("nan")}, {1: 3.0}):
        with pytest.raises((TypeError, ValueError)):
            FitClock(bad)


def test_the_supervisor_refuses_train_times_off_the_mission_clock():
    with pytest.raises(MuleSupervisorError, match="mission_clock"):
        _sup(train_time_s={"d": 3.0})
    assert _sup()._fits is None
    sup = _sup(mission_clock=MissionClock(), train_time_s={"d": 3.0})
    assert sup._fits.train_time_s("d") == 3.0


# --------------------------------------------------------------------------- #
# The contact
# --------------------------------------------------------------------------- #

def test_the_plan_refuses_a_not_ready_mark_it_cannot_honour():
    clock = HMF.RecClock()
    D = HMF.D
    with pytest.raises(ValueError, match="outside the contact"):
        HMF.dataclasses.replace(HMF.banded_plan(clock, D[:2]), not_ready=frozenset({D[5]}))
    with pytest.raises(ValueError, match="overlap"):
        HMF.dataclasses.replace(HMF.banded_plan(clock, D[:2], drop=[D[1]]),
                                not_ready=frozenset({D[1]}))
    assert HMF.banded_plan(clock, D[:2]).not_ready == frozenset()


class _SolicitRF(HMF.FerryRF):
    """``FerryRF`` that keeps each solicit's ``not_ready``."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.not_ready = []

    def solicit(self, msg, device_ids):
        self.not_ready.append(tuple(msg.not_ready))
        return super().solicit(msg, device_ids)


def test_a_not_ready_target_is_told_so_pushed_nothing_and_costs_nothing():
    D = HMF.D
    rf = _SolicitRF(D[:3])
    clock = HMF.RecClock()
    host = HMF.make_host(rf, clock, ttl=5.0)                  # a TTL wait would show
    host.open_round(HMF.theta(), theta_version=5)
    t_arr = clock()
    plan = HMF.dataclasses.replace(HMF.banded_plan(clock, D[:3]), not_ready=frozenset({D[1]}))
    out = host.run_contact(D[:3], HMF.SYNTH, plan=plan)
    assert rf.not_ready == [(D[1],)]
    assert [c[1] for c in rf.pushes()] == [D[0], D[2]]       # D1 got no push
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT,
                   D[2]: MissionOutcome.CLEAN}
    lines = HMF.lines_by_id(host._report)
    line = lines[D[1]]
    # Stamped where D0's session ended, with no airtime of its own.
    assert t_arr < line.contact_ts == lines[D[0]].contact_ts
    assert line.bytes_sent == line.bytes_received == 0
    commit = host.last_contact
    assert commit.not_ready == (D[1],) and commit.pushed == (D[0], D[2])
    # Each collected update's own uplink airtime: its bytes at its session's SNR.
    grad_bytes = HMF.make_grad(D[0], 1, in_reply_to=0).byte_count
    assert set(commit.uplink_dwell_s) == {D[0], D[2]}
    assert commit.uplink_dwell_s[D[0]] == pytest.approx(HMF.dwell_fn(grad_bytes, 10.0))
    assert commit.uplink_dwell_s[D[0]] < commit.session_dwell_s[D[0]]
    assert D[1] not in commit.missing and D[1] not in commit.session_dwell_s
    assert [k for _dt, k in clock.charges] == ["dwell"]       # no listen for it
    assert {r.device_id: r.in_session for r in host._contacts.records}[D[1]] is False


def test_a_plain_contact_names_nobody_and_a_delivery_refuses_the_mark():
    D = HMF.D
    rf = _SolicitRF(D[:2])
    clock = HMF.RecClock()
    host = HMF.make_host(rf, clock)
    host.open_round(HMF.theta(), theta_version=5)
    host.run_contact(D[:2], HMF.SYNTH, plan=HMF.banded_plan(clock, D[:2]))
    assert rf.not_ready == [()] and host.last_contact.not_ready == ()
    assert host.last_contact.pushed == (D[0], D[1])
    host.open_pass_2(HMF.theta(2.0), theta_version=6)
    with pytest.raises(ValueError, match="Pass-1 mark"):
        host.deliver_contact(D[:1], HMF.SYNTH, plan=HMF.dataclasses.replace(
            HMF.banded_plan(clock, D[:1]), not_ready=frozenset({D[0]})))


def test_the_device_answers_and_waits_for_no_push_keeping_its_basis():
    rf = LoopbackRFLink()
    rf.register_device(CMF.DEV)
    cm = CMF._device(rf)
    cm._set_theta_basis(CMF._theta(3.0), [], version=2)
    t, out = CMF._serve_in_background(cm)
    rf.solicit(FLOpenSolicit(mule_id=CMF.MULE, mission_round=4, issued_at=1e6, solicit_id=9,
                             not_ready=(CMF.DEV,)), [CMF.DEV])
    assert rf.recv_ready_adv(timeout=5.0).in_reply_to == 9
    t.join(5.0)
    assert not t.is_alive() and out["oc"] is None           # returned without a push
    assert cm._theta_basis_version == 2
    # The next solicit, which does not name it, is served as before.
    t, out = CMF._serve_in_background(cm)
    rf.solicit(FLOpenSolicit(mule_id=CMF.MULE, mission_round=5, issued_at=1e6 + 9,
                             solicit_id=10), [CMF.DEV])
    rf.recv_ready_adv(timeout=5.0)
    rf.push_disc(CMF.DEV, DiscPush(mule_id=CMF.MULE, mission_round=5,
                                   theta_disc=CMF._theta(), synth_batch=[], basis_version=4))
    assert rf.recv_gradient(CMF.DEV, timeout=5.0).in_reply_to == 10
    t.join(5.0)
    assert out["oc"] is MissionOutcome.CLEAN


# --------------------------------------------------------------------------- #
# Config, builder, driver, runner
# --------------------------------------------------------------------------- #

def test_the_mule_fields_are_set_together_on_the_simulated_clock():
    assert TRAIN_TIME_MULE_FIELDS == ("device_train_time_s", "train_time_params")
    assert set(TRAIN_TIME_MULE_FIELDS) <= set(ADDENDUM_MULE_FIELDS)
    raw = json.loads(mule_config_to_json(MuleConfig(mule_id="m")))
    assert raw["device_train_time_s"] is None and raw["train_time_params"] is None
    wall = MuleConfig(mule_id="m", device_train_time_s={"d": 3.0}, train_time_params=PARAMS)
    assert any("only on the simulated" in e for e in mule_config_errors(wall))
    sim = dict(mule_id="m", mission_clock="sim", rf_range_m=60.0, trial_seed=1)
    assert mule_config_errors(MuleConfig(**sim, device_train_time_s={"d": 3.0},
                                         train_time_params=PARAMS)) == []
    assert any("together" in e for e in mule_config_errors(
        MuleConfig(**sim, device_train_time_s={"d": 3.0})))
    assert any("finite times" in e for e in mule_config_errors(
        MuleConfig(**sim, device_train_time_s={"d": -1.0}, train_time_params=PARAMS)))


def test_the_builder_draws_every_devices_time_and_splits_it_per_mule():
    kw = dict(n_devices=6, seed=11, rf_range_m=60.0, n_missions=2, mission_clock="sim",
              ferry_settings={"contact_band": "wide"})
    topo = build_exp4_topology(**kw, train_time_params={"median_s": 25.0})
    (mule,) = topo.mules
    ids = [d.device_id for d in topo.devices]
    assert mule.device_train_time_s == device_train_times(ids, seed=11,
                                                          params={"median_s": 25.0})
    assert mule.train_time_params == check_train_time_params({"median_s": 25.0})
    assert build_exp4_topology(**kw).mules[0].device_train_time_s is None
    two = build_exp4_topology(**kw, train_time_params=PARAMS, n_mules=2, min_participation=2,
                              dock_on_empty=True)
    whole = device_train_times(ids, seed=11, params=PARAMS)
    for m in two.mules:
        assert m.device_train_time_s == {d: whole[d] for d in m.expected_devices}
    assert sorted(d for m in two.mules for d in m.device_train_time_s) == sorted(ids)
    with pytest.raises(ValueError, match="simulated mission clock"):
        build_exp4_topology(n_devices=6, seed=11, rf_range_m=60.0, n_missions=2,
                            train_time_params=PARAMS)


def test_the_driver_refuses_train_time_off_the_simulated_clock():
    with pytest.raises(ValueError, match="simulated mission clock"):
        Exp4Driver(train_time_params={"median_s": 30.0})
    with pytest.raises(ValueError, match="median_s"):
        Exp4Driver(mission_clock="sim", train_time_params={"sigma": 0.5})
    d = Exp4Driver(mission_clock="sim", train_time_params={"median_s": 30.0})
    assert d.train_time_params == check_train_time_params({"median_s": 30.0})
    assert train_time_ferry_params({}) == {}
    assert train_time_ferry_params({"train_time_params": PARAMS}) == {
        "train_time_params": PARAMS}


def _driver_kwargs(monkeypatch, argv):
    seen = {}

    class _Stop(Exception):
        pass

    def fake(**kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(runner_main, "Exp4Driver", fake)
    with pytest.raises(_Stop):
        runner_main.main(argv)
    return seen


def test_the_runner_passes_the_settings_only_when_given(tmp_path, monkeypatch):
    base = ["--csv", str(tmp_path / "t.csv"), "--arms", "F", "--mission-clock", "sim"]
    assert "train_time_params" not in _driver_kwargs(monkeypatch, base)
    got = _driver_kwargs(monkeypatch, base + ["--train-time-s", "40", "--straggler-share",
                                              "0.2", "--straggler-factor", "5"])
    assert got["train_time_params"] == {"median_s": 40.0, "straggler_share": 0.2,
                                        "straggler_factor": 5.0}
    with pytest.raises(SystemExit):
        runner_main.main(base + ["--train-time-sigma", "0.5"])


# --------------------------------------------------------------------------- #
# A whole F trial
# --------------------------------------------------------------------------- #

def _capture(settings, cell):
    from tests.golden import _build_p3_sim as UG4
    from tests.golden import _build_p4_plan as UG5
    from tests.golden.test_golden_p4_plan import plain

    driver = Exp4Driver(**settings)
    with UG4.in_process_orchestrator(), UG5.flight_slot_spy() as calls:
        UG4.InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        orch = UG4.InProcessOrchestrator.last
    return plain(UG5.case_of(settings, cell, row, orch, calls))


@pytest.fixture(scope="module")
def f_45s():
    from tests.golden import _build_p4_plan as UG5

    settings, cell = UG5.TRIALS["f_45s"]
    return (_capture(dict(settings), cell),
            _capture(dict(settings, train_time_params=PARAMS), cell))


def _missions(case):
    (missions,) = case["mission_completed"].values()
    return missions


def test_the_default_trial_gains_no_key(f_45s):
    plain, _ = f_45s
    for m in _missions(plain):
        assert "train_fits" not in m
        assert all("not_ready" not in s for s in m["pass_1_flown"])
    assert "train_time_params" not in json.loads(plain["row"]["ferry_params"])


def test_the_trial_records_its_fits_and_marks_exactly_what_its_fit_clock_does(f_45s):
    _, timed = f_45s
    (mule,) = [cfg for name, cfg in timed["configs"].items() if name.startswith("mule")]
    times = mule["device_train_time_s"]
    assert json.loads(timed["row"]["ferry_params"])["train_time_params"] == PARAMS
    missions = _missions(timed)
    first = missions[0]
    assert first["train_fits"][:len(times)] == [[d, first["sim_start_s"]] for d in sorted(times)]
    ((ready,),) = timed["mule_ready"].values()
    assert ready["train_time_params"] == PARAMS and ready["train_time_n"] == len(times)
    assert "device_train_time_s" not in ready
    ((plain_ready,),) = f_45s[0]["mule_ready"].values()
    assert not {"train_time_params", "train_time_n"} & set(plain_ready)
    starts = {d: deque() for d in times}
    for m in missions:
        for d, t in m["train_fits"]:
            starts[d].append(t)
    marked = 0
    for m in missions:
        for stop in m["pass_1_flown"]:
            a = stop["arrival_s"]

            def last_start(d):
                return max(t for t in starts[d] if t < a)

            expect = sorted(d for d in stop["targets"] if a < last_start(d) + times[d])
            assert sorted(stop["not_ready"]) == expect
            # A not-ready device got no model at that stop.
            for d in stop["not_ready"]:
                assert not [t for t in starts[d] if a <= t <= stop["end_s"]]
            marked += len(expect)
    assert marked > 0                     # the settings make some contact wait


def test_mission_completed_carries_the_fits_last_only_when_set():
    from hermes.processes.mule import SIM_MISSION_OPTIONAL_FIELDS, _sim_mission_fields
    from tests.unit.test_exp5_decision_cost import _result

    assert SIM_MISSION_OPTIONAL_FIELDS[-1] == "train_fits"
    plain = _sim_mission_fields(_result())
    assert "train_fits" not in plain
    fits = [["d0", 1.0e6], ["d1", 1.0e6 + 3.5]]
    timed = _sim_mission_fields(_result(train_fits=fits))
    assert list(timed)[-2:] == ["train_fits", "energy_status"] and timed["train_fits"] == fits
    # Kept when empty: the fit clock ran and no fit started that mission.
    assert _sim_mission_fields(_result(train_fits=[]))["train_fits"] == []


def test_the_not_ready_targets_are_targets_with_no_uplink_drop(f_45s):
    _, timed = f_45s
    for m in _missions(timed):
        for stop in m["pass_1_flown"]:
            assert set(stop["not_ready"]) <= set(stop["targets"])
            assert not set(stop["not_ready"]) & set(stop["uplink_dropped"])
            # Only an update that arrived has an uplink airtime.
            assert not set(stop["uplink_s"]) & set(stop["not_ready"])
            assert all(0.0 < sec <= stop["dwell_s"] for sec in stop["uplink_s"].values())


# --------------------------------------------------------------------------- #
# The scorer's compute columns
# --------------------------------------------------------------------------- #

def test_busy_seconds_cut_each_fit_at_its_restart_or_the_trials_end():
    from experiments.analysis.traces_scorer import device_busy_s

    fits = [("a", 0.0), ("b", 0.0), ("a", 4.0), ("a", 30.0), ("b", 50.0)]
    busy = device_busy_s(fits, {"a": 10.0, "b": 20.0}, end_s=35.0)
    # a: 4 (restarted) + 10 + 5 (the trial ended); b: 20 + 0 (started at the end).
    assert busy == {"a": 19.0, "b": 20.0}


def test_the_compute_columns_read_the_fit_records(f_45s):
    from experiments.analysis.traces_scorer import (
        COMPUTE_COLUMNS,
        ComputeReport,
        compute_report,
        device_busy_s,
    )
    from types import SimpleNamespace

    from experiments.exp4 import events_consumer as EC
    from experiments.exp4.events_consumer import FitStop, MissionRecord

    assert ComputeReport().to_row() == {c: "" for c in COMPUTE_COLUMNS}
    _, timed = f_45s
    (mule,) = [cfg for name, cfg in timed["configs"].items() if name.startswith("mule")]
    missions = []
    for m in _missions(timed):
        missions.append(MissionRecord(
            mission_round=m["mission_round"], pass_1_contacts=0, pass_2_contacts=0,
            pass_1_updates=None, pass_1_scheduled=None, pass_1_clean_devices=(),
            delivered=None, undelivered=None, duration_s=None, mule_id="exp4-mule",
            sim_end_s=m["sim_end_s"],
            pass_1_outcomes=tuple((o["device"], o["outcome"], o["contact_ts"])
                                  for o in m.get("pass_1_outcomes") or ()),
            train_fits=EC._train_fits(m["train_fits"]),
            fit_stops=EC._fit_stops(m["pass_1_flown"]),
        ))
        assert missions[-1].fit_stops == tuple(
            FitStop(tuple(s["targets"]), tuple(s["not_ready"]), dict(s["uplink_s"]))
            for s in m["pass_1_flown"])
        assert missions[-1].train_fits == tuple((d, t) for d, t in m["train_fits"])
    obs = SimpleNamespace(missions=missions)
    r = compute_report(obs, [mule], p_comp_w=2.0, p_tx_w=0.5)
    assert (r.train_time_median_s, r.train_time_sigma, r.straggler_share,
            r.straggler_factor) == (40.0, 0.5, 0.2, 5.0)
    stops = [s for m in _missions(timed) for s in m["pass_1_flown"]]
    assert r.pass_1_target_contacts == sum(len(s["targets"]) for s in stops)
    assert r.not_ready_contacts == sum(len(s["not_ready"]) for s in stops) > 0
    assert r.not_ready_share == pytest.approx(r.not_ready_contacts / r.pass_1_target_contacts)
    clean = sum(1 for m in missions for o in m.pass_1_outcomes if o[1] == "clean")
    assert r.pass_1_clean_share == pytest.approx(clean / r.pass_1_target_contacts)
    fits = [f for m in missions for f in m.train_fits]
    busy = device_busy_s(fits, mule["device_train_time_s"], missions[-1].sim_end_s)
    uplink = {}
    for s in stops:
        for d, sec in s["uplink_s"].items():
            uplink[d] = uplink.get(d, 0.0) + sec
    assert r.device_train_busy_s == pytest.approx(sum(busy.values()))
    assert r.device_uplink_s == pytest.approx(sum(uplink.values())) and r.device_uplink_s > 0
    energy = {d: 2.0 * busy.get(d, 0.0) + 0.5 * uplink.get(d, 0.0) for d in busy}
    assert r.device_energy_j_total == pytest.approx(sum(energy.values()))
    assert r.device_energy_j_max == pytest.approx(max(energy.values()))
    assert (r.device_p_comp_w, r.device_p_tx_w) == (2.0, 0.5)
    # The parsers: no record without the fit clock's keys, bad entries skipped.
    assert EC._fit_stops([{"targets": ["a"]}]) is None and EC._train_fits(None) is None
    assert EC._train_fits([["a", 1.0], ["b"], [3, 2.0], ["c", "x"]]) == (("a", 1.0),)
    # A trace without fit records: the settings only, the rest blank.
    bare = compute_report(SimpleNamespace(missions=[]), [mule])
    assert bare.train_time_median_s == 40.0 and bare.not_ready_contacts is None


def test_a_zero_fit_time_flies_as_recorded_and_records_its_fits():
    from tests.golden import _build_p4_plan as UG5

    settings, cell = UG5.TRIALS["f_45s"]
    plain = _capture(dict(settings), cell)
    zero = _capture(dict(settings, train_time_params={"median_s": 0.0}), cell)
    for a, b in zip(_missions(plain), _missions(zero)):
        assert b["train_fits"]                              # recorded
        for sa, sb in zip(a["pass_1_flown"], b["pass_1_flown"]):
            assert sb["not_ready"] == []
            assert {k: v for k, v in sb.items() if k not in ("not_ready", "uplink_s")} == sa
        assert a["pass_1_outcomes"] == b["pass_1_outcomes"]


def test_a_kept_trace_scores_the_settings_and_the_columns_last(tmp_path):
    """End to end through a kept trace (a FerrySim FX episode, in process): the
    scorer's ``ferry_params`` is the row's, and the compute columns come last,
    only when asked."""
    import logging

    from experiments.analysis.traces_scorer import COMPUTE_COLUMNS, score_trial
    from experiments.ferrysim import cells as C
    from experiments.ferrysim.episode import Policy, run_episode

    cell = C.cell_named("jit-n6-90")
    logging.disable(logging.WARNING)
    try:
        result = run_episode(cell, C.stream_seeds(C.VAL_STREAM, cell.name, 1)[0],
                             Policy.of_arm("FX"),
                             driver_overrides={"trace_root": str(tmp_path),
                                               "train_time_params": PARAMS})
    finally:
        logging.disable(logging.NOTSET)
    (trace,) = [d for d in tmp_path.rglob("*__FX__*") if d.is_dir()]
    plain = score_trial(trace).to_row()
    assert plain["ferry_params"] == result.row["ferry_params"]
    assert json.loads(plain["ferry_params"])["train_time_params"] == PARAMS
    assert not set(COMPUTE_COLUMNS) & set(plain)
    row = score_trial(trace, compute_columns=True).to_row()
    assert list(row) == list(plain) + list(COMPUTE_COLUMNS)
    assert row["train_time_median_s"] == 40.0 and row["not_ready_contacts"] > 0
    assert 0.0 < row["not_ready_share"] < 1.0 and 0.0 <= row["pass_1_clean_share"] < 1.0
    assert row["device_energy_j_total"] == pytest.approx(
        5.0 * row["device_train_busy_s"] + 1.0 * row["device_uplink_s"])
