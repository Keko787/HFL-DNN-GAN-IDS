"""FeRRy Phase 3, unit U7 — Exp 4 ferry cells: the builder, the driver and the CLI.

* The builder: with the defaults, the recorded topology; with
  ``mission_clock="sim"`` every role on the clock with the trial seed,
  devices answering only the newest solicit, the RF token on every role, and
  under the channel reliability source the devices' own draw replaced by the
  mule's per-slice ground truth (design section 4.8); positions and the
  recorded reliability formula unchanged.
* The driver's guards (H0 on the clock, critic A5; sim-only settings on the
  wall clock; the channel source without a band and the seconds model
  without the clock, critic B16; two backhaul loss models; several mules
  below a full quorum, critic B9, run since unit U9), the per-trial RF
  token, T_nom (deterministic, one value for every arm, cached; with several
  mules the slowest slice) and
  what derives from it (the deadline time unit, Φ₀ in missions, D5's period,
  the backhaul period), critic B4 (no non-causal RF prior on the clock; under
  the mission model the chosen carrier's SNR per mission, the trace the loss
  schedule comes from, for the mule to adopt one upload at a time), the
  seconds model replacing the flat realism loss, the D4 CARP split's
  per-client airtime, the re-costed wall budget and hard kill (critic B14),
  the input-width pin (design R8) through ``run_trial``, and the provenance
  columns (blank at their recorded values; Φ₀ and the input width recorded).
* The runner's flags reach the driver; H0 is refused or dropped on the clock;
  the soft cap follows the re-costed budget.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, fields

import pytest

from experiments.exp4 import driver as driver_module
from experiments.exp4.driver import (
    CANONICAL_INPUT_DIM,
    PROVENANCE_COLUMNS,
    Exp4Driver,
    _TrialClock,
)
from experiments.exp4.model_task import _u32, device_reliabilities
from experiments.exp4.topology_builder import build_exp4_topology, device_positions
from experiments.runner import Cell
from hermes.mule.ferry import FerrySpec
from hermes.processes.config import MuleConfig, mule_config_errors

from tests.golden import _build_topology as T

PHASE_3_COLUMNS = (
    "mission_clock", "contact_band", "in_flight_response", "backhaul_model",
    "contact_reliability_source", "deadline_time_scale", "initial_window_s",
    "t_nom_s", "session_ttl_s", "ferry_params", "l1_channel", "realism", "input_dim",
)


def _cell(arm="H1", seed=7, trial=0, **params) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=trial, seed=seed, params=p)


def _sim(**kw) -> Exp4Driver:
    kw.setdefault("mission_clock", "sim")
    kw.setdefault("realism", True)
    return Exp4Driver(**kw)


# --------------------------------------------------------------------------- #
# The topology builder
# --------------------------------------------------------------------------- #

def test_the_builders_defaults_leave_every_new_field_at_its_default():
    topo = build_exp4_topology(n_devices=6, rf_range_m=60.0, n_missions=4, seed=7,
                               device_reliability=True, field_radius_m=100.0)
    (mule,) = topo.mules
    for f in fields(MuleConfig):
        if f.name in ("mule_id", "rf_range_m", "session_ttl_s", "n_missions", "aggregation_params",
                      "deadline_params"):
            continue
        assert getattr(mule, f.name) == getattr(MuleConfig(mule_id="x"), f.name), f.name
    assert (topo.cluster.mission_clock, topo.cluster.backhaul_model) == ("wall", "mission")
    assert all(not d.newest_solicit_only and d.rf_link_token is None for d in topo.devices)


def test_a_sim_build_keeps_positions_and_the_recorded_reliabilities():
    common = dict(n_devices=6, rf_range_m=60.0, n_missions=4, seed=7, device_reliability=True,
                  field_radius_m=100.0, reliabilities=device_reliabilities(7, 6))
    wall = build_exp4_topology(**common)
    sim = build_exp4_topology(**common, mission_clock="sim", input_dim=21,
                              ferry_settings={"contact_band": "wide"}, rf_link_token="tok")
    assert [d.position for d in sim.devices] == [d.position for d in wall.devices]
    assert [d.contact_reliability for d in sim.devices] == [d.contact_reliability for d in wall.devices]
    assert all(d.newest_solicit_only and d.rf_link_token == "tok" for d in sim.devices)
    (mule,) = sim.mules
    assert (mule.mission_clock, mule.trial_seed, mule.contact_band) == ("sim", 7, "wide")
    assert (mule.input_dim, mule.rf_link_token, mule.device_availability) == (21, "tok", {})
    c = sim.cluster
    assert (c.mission_clock, c.trial_seed, c.backhaul_model) == ("sim", 7, "mission")


def test_device_positions_is_the_builders_own_draw():
    topo = build_exp4_topology(n_devices=5, rf_range_m=60.0, n_missions=1, seed=99)
    assert [d.position[:2] for d in topo.devices] == device_positions(5, 99, 24.0)


@pytest.mark.parametrize("k", [1, 2])
def test_the_channel_source_moves_the_reliability_draw_to_the_mule(k):
    kw = dict(n_devices=6, rf_range_m=60.0, n_missions=4, seed=7, device_reliability=True,
              field_radius_m=100.0, reliabilities=device_reliabilities(7, 6),
              mission_clock="sim",
              ferry_settings={"contact_band": "wide", "contact_reliability_source": "channel"})
    if k > 1:
        kw.update(n_mules=2, min_participation=2, dock_on_empty=True)
    topo = build_exp4_topology(**kw)
    assert all(d.contact_reliability is None for d in topo.devices)
    rel = dict(zip((d.device_id for d in topo.devices), device_reliabilities(7, 6)))
    merged = {}
    for mule in topo.mules:
        assert set(mule.device_availability) == set(topo.devices_of(mule.mule_id))
        merged.update(mule.device_availability)
    assert merged == rel


def test_the_band_classes_reach_every_mule_and_the_cluster():
    """The cluster reads a report line's band index with the mules' classes."""
    classes = ["wide", "medium", "narrow", "medium_wide"]
    topo = build_exp4_topology(n_devices=4, rf_range_m=60.0, n_missions=2, seed=3,
                               mission_clock="sim", n_mules=2, min_participation=2,
                               dock_on_empty=True,
                               ferry_settings={"contact_band": "wide",
                                               "contact_band_classes": classes})
    assert topo.cluster.contact_band_classes == classes
    assert all(m.contact_band_classes == classes for m in topo.mules)
    assert topo.mules[0].contact_band_classes is not topo.mules[1].contact_band_classes


def test_the_channel_source_draws_the_availability_without_realism_too():
    topo = build_exp4_topology(n_devices=3, rf_range_m=60.0, n_missions=2, seed=5,
                               mission_clock="sim",
                               ferry_settings={"contact_band": "wide",
                                               "contact_reliability_source": "channel"})
    assert list(topo.mules[0].device_availability.values()) == device_reliabilities(5, 3)


@pytest.mark.parametrize("kw,match", [
    (dict(mission_clock="sim", ferry_settings={"warp": 9}), "not MuleConfig ferry fields"),
    (dict(mission_clock="sim", ferry_settings={"device_availability": {}}), "derives"),
    (dict(ferry_settings={"contact_band": "wide"}), "mission_clock='sim'"),
    (dict(mission_clock="gps"), "mission_clock"),
    (dict(rf_prior_schedule_db=[9.0], backhaul_loss_schedule=[0.1]), "rf_prior_snr_db"),
])
def test_the_builder_refuses_misplaced_ferry_settings(kw, match):
    with pytest.raises(ValueError, match=match):
        build_exp4_topology(n_devices=2, rf_range_m=60.0, n_missions=1, seed=1, **kw)


# --------------------------------------------------------------------------- #
# The driver: guards
# --------------------------------------------------------------------------- #

def test_h0_is_refused_on_the_simulated_clock():
    with pytest.raises(ValueError, match="critic A5"):
        _sim(real_model=True).run_trial(_cell("H0"))


@pytest.mark.parametrize("kw,match", [
    (dict(contact_band="wide"), "mission_clock='sim'"),
    (dict(backhaul_model="seconds"), "mission_clock='sim'"),
    (dict(deadline_time_scale="t_nom"), "mission_clock='sim'"),
    (dict(ferry_physics={"n_pl": 3.0}), "mission_clock='sim'"),
    (dict(expected_input_dim=21), "mission_clock='sim'"),
    (dict(mission_clock="sim", l1_channel=True, backhaul_model="seconds"), "two backhaul"),
    (dict(mission_clock="sim", contact_reliability_source="channel"), "critic B16"),
    (dict(mission_clock="sim", agg_period_t_nom=True), "agg:cutoff"),
    (dict(mission_clock="sim", deadline_time_scale="fast"), "t_nom"),
    (dict(mission_clock="sim", ferry_physics={"warp": 9}), "ferry physics"),
    (dict(mission_clock="sim", contact_band="ultrawide"), "ultrawide"),
    (dict(mission_clock="sim", initial_window_s=60.0, initial_window_missions=6.0), "not both"),
    (dict(mission_clock="lunar"), "mission_clock"),
])
def test_the_driver_refuses_what_cannot_run_or_would_mis_measure(kw, match):
    with pytest.raises(ValueError, match=match):
        Exp4Driver(**kw)


def test_a_full_quorum_of_several_mules_runs_on_the_clock():
    row, topo = T.run_stub_trial(_sim(n_mules=2, min_participation=2), _cell("H1", N=8))
    assert len(topo.mules) == 2 and topo.cluster.min_participation == 2
    assert row["mission_clock"] == "sim"


# --------------------------------------------------------------------------- #
# The driver: token, T_nom and what derives from it
# --------------------------------------------------------------------------- #

def test_the_link_token_is_one_per_trial_and_reproducible():
    a = Exp4Driver.trial_link_token(_cell("H1"))
    assert a == Exp4Driver.trial_link_token(_cell("H1"))
    others = {Exp4Driver.trial_link_token(c) for c in (
        _cell("H2"), _cell("H1", trial=1), _cell("H1", seed=8), _cell("H1", N=7))}
    assert a not in others and len(others) == 4
    _row, topo = T.run_stub_trial(_sim(), _cell("H1"))
    assert {m.rf_link_token for m in topo.mules} == {d.rf_link_token for d in topo.devices} == {a}


def test_the_token_is_off_on_the_wall_clock_unless_asked_for():
    _row, topo = T.run_stub_trial(Exp4Driver(), _cell("H1"))
    assert topo.mules[0].rf_link_token is None
    _row, topo = T.run_stub_trial(Exp4Driver(rf_link_token=True), _cell("H1"))
    assert topo.mules[0].rf_link_token == Exp4Driver.trial_link_token(_cell("H1"))
    _row, topo = T.run_stub_trial(_sim(rf_link_token=False), _cell("H1"))
    assert topo.mules[0].rf_link_token is None


def _t_nom(driver, arm="H1", regime="jittery", **kw):
    settings = driver.ferry_settings(arm=arm, regime=regime)
    theta, synth = driver._payload_bytes(None)
    return driver.nominal_period_s(n_devices=kw.get("n", 6), rf_range_m=60.0, regime=regime,
                                   settings=settings, theta_bytes=theta, synth_bytes=synth)


def test_t_nom_is_one_deterministic_value_for_every_arm_and_is_cached(monkeypatch):
    drv = _sim(contact_band="wide", t_nom_layouts=5)
    t = _t_nom(drv)
    assert t == _t_nom(_sim(contact_band="wide", t_nom_layouts=5))
    assert _t_nom(drv, arm="H3") == _t_nom(drv, arm="D1") == t
    calls = []
    import hermes.scheduler.fl_scheduler as fls

    real = fls.nominal_mission_period_s
    monkeypatch.setattr(fls, "nominal_mission_period_s", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert _t_nom(drv) == t and calls == []
    # The median over the reference layouts, priced with the wide-band spec.
    from hermes.scheduler.fl_scheduler import nominal_mission_period_s
    from hermes.types import DeviceID

    periods = []
    for k in range(5):
        seed = _u32(6, "t_nom", k)
        spec = FerrySpec.from_config(rf_range_m=60.0, seed=seed, contact_band="wide",
                                     backhaul_regime="jittery")
        model = spec.feasibility_model(rf_range_m=60.0, theta_bytes=52, synth_bytes=64)
        xy = device_positions(6, seed, 100.0)
        periods.append(real(
            {DeviceID(f"exp4-dev-{i:03d}"): (x, y, 0.0) for i, (x, y) in enumerate(xy)},
            rf_range_m=60.0, feasibility_model=model, turnaround_s=30.0,
        ))
    assert t == statistics.median(periods)
    assert 60.0 < t < 400.0


def test_t_nom_sets_the_time_unit_the_window_the_d5_period_and_the_backhaul_period():
    drv = _sim(contact_band="wide", backhaul_model="seconds", deadline_time_scale="t_nom",
               initial_window_missions=6.0, aggregation="agg:cutoff", agg_period_t_nom=True,
               t_nom_layouts=5)
    row, topo = T.run_stub_trial(drv, _cell("H3"))
    t = _t_nom(drv, arm="H3")
    (mule,) = topo.mules
    assert mule.t_nom_s == t and row["t_nom_s"] == t
    assert mule.deadline_time_scale == pytest.approx(t / 10.0)
    assert mule.initial_window_s == pytest.approx(60.0)          # six missions at T_nom / 10
    assert mule.aggregation_params["period_s"] == t
    assert topo.cluster.aggregation_params["period_s"] == t
    assert json.loads(row["aggregation_params"])["period_s"] == t
    spec = FerrySpec.from_config(**mule.ferry_spec_kwargs())
    assert spec.backhaul.period_s == 4 * t                      # P_bh = n_missions * T_nom
    assert mule.backhaul_policy == "adaptive"                   # H3's controller
    params = json.loads(row["ferry_params"])
    assert params["backhaul_period_s"] == 4 * t and params["t_nom_computed"] is True


def test_t_nom_with_several_mules_is_the_median_of_each_layouts_slowest_slice():
    """A quorum of every mule waits for the slowest: each reference layout is
    split into the default angular slices and priced as its slowest slice."""
    from experiments.exp4.topology_builder import angular_slices
    from hermes.scheduler.fl_scheduler import nominal_mission_period_s
    from hermes.types import DeviceID

    drv = _sim(contact_band="wide", n_mules=2, min_participation=2, t_nom_layouts=5)
    t = _t_nom(drv, n=8)
    per_slice = []
    for k in range(5):
        seed = _u32(8, "t_nom", k)
        spec = FerrySpec.from_config(rf_range_m=60.0, seed=seed, contact_band="wide",
                                     backhaul_regime="jittery")
        model = spec.feasibility_model(rf_range_m=60.0, theta_bytes=52, synth_bytes=64)
        xy = device_positions(8, seed, 100.0)
        assign = angular_slices(xy, 2)
        per_slice.append([
            nominal_mission_period_s(
                {DeviceID(f"exp4-dev-{i:03d}"): (xy[i][0], xy[i][1], 0.0)
                 for i in range(8) if assign[i] == m},
                rf_range_m=60.0, feasibility_model=model, turnaround_s=30.0,
            )
            for m in range(2)
        ])
    assert t == statistics.median(max(p) for p in per_slice)
    # The slices differ, so the rule matters: the faster slices would give less.
    assert statistics.median(min(p) for p in per_slice) < t


def test_a_given_t_nom_is_used_and_not_computed(monkeypatch):
    monkeypatch.setattr(Exp4Driver, "nominal_period_s",
                        lambda self, **kw: pytest.fail("T_nom was given"))
    row, topo = T.run_stub_trial(_sim(backhaul_model="seconds", t_nom_s=150.0), _cell("H1"))
    assert topo.mules[0].t_nom_s == 150.0 and json.loads(row["ferry_params"])["t_nom_computed"] is False


def test_no_t_nom_is_computed_when_nothing_needs_it(monkeypatch):
    monkeypatch.setattr(Exp4Driver, "nominal_period_s",
                        lambda self, **kw: pytest.fail("T_nom was not needed"))
    row, topo = T.run_stub_trial(_sim(contact_band="wide"), _cell("H1"))
    assert topo.mules[0].t_nom_s is None and row["t_nom_s"] == ""


# --------------------------------------------------------------------------- #
# The driver: backhaul, RF prior, CARP, budgets, input width
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arm", ["H2", "H3"])
def test_the_clock_never_gets_the_non_causal_rf_prior(arm):
    """Critic B4: the mean over the realized trace uses the future. A ferry
    cell under the mission model keeps the recorded loss schedule, and its
    mule gets the chosen carrier's SNR at each mission instead, which it
    adopts one upload at a time (hermes/processes/mule.py)."""
    from hermes.l1.channel_model import ChannelModel, backhaul_plan, loss_from_snr

    cell = _cell(arm)
    _row, wall = T.run_stub_trial(Exp4Driver(realism=True, l1_channel=True), cell)
    _row, sim = T.run_stub_trial(_sim(l1_channel=True), cell)
    (w,), (s,) = wall.mules, sim.mules
    assert w.rf_prior_snr_db is not None and w.rf_prior_schedule_db is None
    assert s.rf_prior_snr_db is None and s.backhaul_model == "mission"
    assert sim.cluster.backhaul_loss_schedule == wall.cluster.backhaul_loss_schedule
    model = ChannelModel(n_bands=3, n_missions=4, seed=cell.seed, jittery=True)
    plan = backhaul_plan(model, adaptive=(arm == "H3"))
    schedule = s.rf_prior_schedule_db
    assert schedule == [model.snr(m, b) for m, b in enumerate(plan.chosen_bands)]
    # One trace: the losses are loss_from_snr of it entry for entry, and the
    # wall clock's non-causal prior is its mean over every mission.
    assert [loss_from_snr(v) for v in schedule] == sim.cluster.backhaul_loss_schedule
    assert sum(schedule) / len(schedule) == w.rf_prior_snr_db


def test_every_mule_of_a_ferry_cell_gets_its_own_copy_of_the_prior_schedule():
    _row, topo = T.run_stub_trial(_sim(l1_channel=True, n_mules=2, min_participation=2),
                                  _cell("H3", N=8))
    a, b = topo.mules
    assert a.rf_prior_schedule_db == b.rf_prior_schedule_db
    assert a.rf_prior_schedule_db is not b.rf_prior_schedule_db
    assert len(a.rf_prior_schedule_db) == len(topo.cluster.backhaul_loss_schedule) == 4


def test_no_prior_schedule_without_the_l1_channel():
    _row, topo = T.run_stub_trial(_sim(), _cell("H3"))
    assert topo.mules[0].rf_prior_schedule_db is None and topo.mules[0].rf_prior_snr_db is None


def test_the_seconds_model_replaces_the_flat_realism_loss():
    _row, topo = T.run_stub_trial(_sim(backhaul_model="seconds", t_nom_s=200.0), _cell("H1"))
    assert topo.cluster.backhaul_loss_pct == 0.0 and topo.cluster.backhaul_model == "seconds"
    _row, topo = T.run_stub_trial(_sim(), _cell("H1"))
    assert topo.cluster.backhaul_loss_pct == 2.0                 # the mission model keeps it


def test_d4_prices_its_carp_split_with_the_predicted_airtime(monkeypatch):
    seen = []
    real = driver_module.d4_slice_assignment

    def _spy(devices, n, seed, **kw):
        seen.append(kw)
        return real(devices, n, seed, **kw)

    monkeypatch.setattr(driver_module, "d4_slice_assignment", _spy)
    T.run_stub_trial(Exp4Driver(realism=True, n_mules=2, min_participation=2), _cell("D4", N=7))
    assert seen[-1] == {}                                        # recorded: no new keywords
    drv = _sim(contact_band="wide", payload_bytes=1_000_000, n_mules=2, min_participation=2,
               ferry_physics={"cruise_speed_m_s": 8.0})
    T.run_stub_trial(drv, _cell("D4", N=7))
    kw = seen[-1]
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=7, contact_band="wide",
                                 payload_bytes=1_000_000, cruise_speed_m_s=8.0)
    expected = spec.link.dwell_s(2_000_000, "wide", spec.link.mean_snr_db("wide", 30.0))
    assert kw["t_trans_s"] == pytest.approx(expected) and kw["cruise_speed_m_s"] == 8.0
    assert drv.carp_t_trans_s(drv.ferry_settings(arm="D4", regime="clean"), rf_range_m=60.0,
                              seed=7, theta_bytes=52, synth_bytes=64) == pytest.approx(expected)
    # Without a band the contact costs the session time, the recorded value.
    assert _sim().carp_t_trans_s(_sim().ferry_settings(arm="D4", regime="clean"),
                                 rf_range_m=60.0, seed=7, theta_bytes=52, synth_bytes=64) == 1.0


def test_the_wall_budget_is_re_costed_on_the_clock_only():
    wall = Exp4Driver(trial_budget_s=120.0)
    assert wall.trial_wall_budget_s(n_devices=6, n_missions=4) == 120.0
    sim = _sim(trial_budget_s=120.0)
    assert sim.ferry_wall_bound_s(n_devices=6, n_missions=4) == 90.0 + 4 * (2 * 6 * 9.0 + 10.0)
    assert sim.trial_wall_budget_s(n_devices=6, n_missions=4) == 562.0
    ttl30 = _sim(session_ttl_s=30.0)
    assert ttl30.trial_wall_budget_s(n_devices=6, n_missions=4) == 90.0 + 4 * (2 * 6 * 90.0 + 10.0)
    k2 = _sim(n_mules=2, min_participation=2)
    bound = 90.0 + 4 * 2 * (2 * 3 * 9.0 + 10.0)
    assert k2.ferry_wall_bound_s(n_devices=6, n_missions=4) == bound
    big = _sim(trial_budget_s=5000.0)
    assert big.trial_wall_budget_s(n_devices=6, n_missions=4) == 5000.0


def _status(root, cell) -> dict:
    path = root / driver_module.trace_dir_name(cell) / driver_module.TRIAL_STATUS_FILE
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("sim", [False, True])
def test_the_hard_kill_and_the_status_marker_use_the_trials_own_budget(monkeypatch, tmp_path, sim):
    """Critic B14: the orchestrator is given the re-costed wall budget on the
    clock (562 s for N = 6, 4 missions at the 3 s TTL) and exactly
    ``trial_budget_s`` on the wall clock; the kept trace's status marker
    records the budget that applied."""
    waited = []
    monkeypatch.setattr(Exp4Driver, "_await_mules",
                        lambda self, orch, budget_s: waited.append(budget_s) or True)
    drv = (_sim if sim else Exp4Driver)(realism=True, trace_root=tmp_path)
    cell = _cell("H1")
    T.run_stub_trial(drv, cell)
    budget = 562.0 if sim else 120.0
    assert drv.trial_budget_s == 120.0
    assert waited == [budget] == [drv.trial_wall_budget_s(n_devices=6, n_missions=4)]
    assert (_status(tmp_path, cell)["status"], _status(tmp_path, cell)["trial_budget_s"]) == (
        "ok", budget)
    # Whether the driver computed T_nom is recorded on simulated trials only
    # (the scorer reads it before inferring it); wall markers keep their key set.
    if sim:
        assert _status(tmp_path, cell)["t_nom_computed"] is False
    else:
        assert "t_nom_computed" not in _status(tmp_path, cell)


def test_a_trial_whose_t_nom_the_driver_computed_says_so_in_its_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(Exp4Driver, "_await_mules", lambda self, orch, budget_s: True)
    cell = _cell("H1")
    T.run_stub_trial(_sim(deadline_time_scale="t_nom", trace_root=tmp_path), cell)
    assert _status(tmp_path, cell)["t_nom_computed"] is True


def test_a_trial_past_its_re_costed_budget_is_killed_and_labelled(monkeypatch, tmp_path):
    monkeypatch.setattr(Exp4Driver, "_await_mules", lambda self, orch, budget_s: False)
    cell = _cell("H1")
    with pytest.raises(driver_module.Exp4TrialTimeout, match="exceeded 562s budget"):
        T.run_stub_trial(_sim(trace_root=tmp_path), cell)
    status = _status(tmp_path, cell)
    assert (status["status"], status["trial_budget_s"]) == ("error", 562.0)
    assert "562s" in status["error"]


def test_several_mules_wait_for_their_down_as_long_as_the_re_costed_budget(tmp_path):
    root = tmp_path / "traces"
    drv = _sim(n_mules=2, min_participation=2, trace_root=root)
    row, topo = T.run_stub_trial(drv, _cell("H1", N=8, n_missions=2))
    budget = drv.trial_wall_budget_s(n_devices=8, n_missions=2)
    assert budget > drv.trial_budget_s
    assert {m.down_wait_s for m in topo.mules} == {budget}
    assert json.loads(row["dock_params"])["down_wait_s"] == budget


def test_the_priced_payload_is_the_trials_own_seed_model(tmp_path):
    """The real model's θ comes from the trial's seed weights file; the stub
    run prices the cluster's 13-parameter model (52 B); both push the
    cluster's synthetic batch (2 x 8 float32 = 64 B)."""
    import numpy as np

    from experiments.exp4.model_task import save_weights

    drv = _sim()
    assert drv._payload_bytes(None) == (52, 64)
    path = tmp_path / "theta.npz"
    save_weights(path, [np.zeros((21, 64), dtype=np.float32), np.zeros(64, dtype=np.float32)])
    assert drv._payload_bytes(str(path)) == (21 * 64 * 4 + 64 * 4, 64)


def test_the_input_width_of_a_ferry_cell_is_pinned():
    drv = _sim(real_model=True)
    resolve = lambda dim: drv._resolve_clock(  # noqa: E731
        _cell("H1"), arm="H1", regime="clean", n_devices=6, rf_range_m=60.0, n_missions=4,
        init_theta_path=None, input_dim=dim)
    assert resolve(CANONICAL_INPUT_DIM).input_dim == 21
    with pytest.raises(ValueError, match="design R8"):
        resolve(46)
    assert _sim(real_model=True, expected_input_dim=46)._resolve_clock(
        _cell("H1"), arm="H1", regime="clean", n_devices=6, rf_range_m=60.0, n_missions=4,
        init_theta_path=None, input_dim=46).input_dim == 46
    assert _sim(real_model=True, data_source="synthetic").declared_input_dim() == 46


def _fake_real_model(monkeypatch, tmp_path, input_dim):
    """The driver's real-model flow without data or TensorFlow: ``_build_task``
    and ``prepare_trial`` stubbed to a prep of this input width, with a seed
    weights file the driver reads for the payload."""
    import numpy as np

    from experiments.exp4.model_task import save_weights
    from experiments.exp4.prep import TrialPrep

    theta = tmp_path / "theta_init.npz"
    save_weights(theta, [np.zeros((input_dim, 8), dtype=np.float32),
                         np.zeros(8, dtype=np.float32)])
    monkeypatch.setattr(Exp4Driver, "_build_task", lambda self, n, seed: None)
    monkeypatch.setattr(driver_module, "prepare_trial", lambda prep_dir, *, task, theta_seed: TrialPrep(
        input_dim=input_dim, shard_paths=[str(tmp_path / f"shard-{i}.npz") for i in range(6)],
        test_path=str(tmp_path / "test.npz"), init_theta_path=str(theta),
        is_synthetic=False, n_train=600,
    ))


@pytest.mark.parametrize("sim", [False, True])
def test_a_real_model_row_records_its_input_width(monkeypatch, tmp_path, sim):
    """Critic D3 / design R8: the width is a provenance column on either clock
    (and a ferry cell's mule carries it for mule_ready); blank on the stub."""
    _fake_real_model(monkeypatch, tmp_path, CANONICAL_INPUT_DIM)
    drv = _sim(real_model=True) if sim else Exp4Driver(real_model=True, realism=True)
    row, topo = T.run_stub_trial(drv, _cell("H1"))
    assert row["input_dim"] == 21 and topo.cluster.input_dim == 21
    assert topo.mules[0].input_dim == (21 if sim else None)
    stub_row, _topo = T.run_stub_trial(_sim() if sim else Exp4Driver(realism=True), _cell("H1"))
    assert stub_row["input_dim"] == ""


def test_a_ferry_cell_whose_model_has_another_width_is_refused(monkeypatch, tmp_path):
    """Design R8 through ``run_trial``: the silent 46-input fallback model is
    refused on the clock; the wall clock runs it and records the width."""
    _fake_real_model(monkeypatch, tmp_path, 46)
    with pytest.raises(ValueError, match="design R8"):
        T.run_stub_trial(_sim(real_model=True), _cell("H1"))
    row, _topo = T.run_stub_trial(Exp4Driver(real_model=True, realism=True), _cell("H1"))
    assert row["input_dim"] == 46


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def test_the_phase_3_columns_are_appended_and_blank_at_their_recorded_values():
    assert PROVENANCE_COLUMNS[-len(PHASE_3_COLUMNS):] == PHASE_3_COLUMNS
    row, _topo = T.run_stub_trial(Exp4Driver(), _cell("H1", regime="clean"))
    assert {c: row[c] for c in PHASE_3_COLUMNS} == {c: "" for c in PHASE_3_COLUMNS}
    row, _topo = T.run_stub_trial(Exp4Driver(realism=True, l1_channel=True,
                                             deadline_time_scale=2.0, session_ttl_s=5.0),
                                  _cell("H3"))
    assert (row["realism"], row["l1_channel"], row["deadline_time_scale"], row["session_ttl_s"]) \
        == (1, 1, 2.0, 5.0)
    assert row["mission_clock"] == "" and row["ferry_params"] == ""


def test_the_initial_window_column_records_the_phi0_the_mule_was_given():
    row, topo = T.run_stub_trial(Exp4Driver(initial_window_s=90.0), _cell("H1", regime="clean"))
    assert row["initial_window_s"] == 90.0 == topo.mules[0].initial_window_s
    assert row["deadline_time_scale"] == ""
    # Four nominal missions of 150 s at the unit T_nom / 10 s: 4 * 150 / 15 in
    # the law's recorded unit, which the scale stretches back to 600 s.
    drv = _sim(t_nom_s=150.0, deadline_time_scale="t_nom", initial_window_missions=4.0)
    row, topo = T.run_stub_trial(drv, _cell("H1"))
    assert row["initial_window_s"] == pytest.approx(40.0) == topo.mules[0].initial_window_s
    assert row["deadline_time_scale"] == pytest.approx(15.0)


def test_a_ferry_row_records_its_settings():
    row, topo = T.run_stub_trial(
        _sim(contact_band="wide", backhaul_model="seconds", t_nom_s=210.0,
             contact_reliability_source="channel", in_flight_response="replan",
             payload_bytes=1_000_000, ferry_physics={"n_pl": 3.0}),
        _cell("H1"))
    assert (row["mission_clock"], row["contact_band"], row["backhaul_model"]) == ("sim", "wide", "seconds")
    assert (row["contact_reliability_source"], row["in_flight_response"]) == ("channel", "replan")
    assert row["t_nom_s"] == 210.0 and row["deadline_time_scale"] == ""
    params = json.loads(row["ferry_params"])
    assert params["n_pl"] == 3.0 and params["payload_bytes"] == 1_000_000
    assert params["backhaul_period_s"] == 4 * 210.0
    assert "device_availability" not in params and "contact_band" not in params


# --------------------------------------------------------------------------- #
# The runner CLI
# --------------------------------------------------------------------------- #

def _runner(monkeypatch, argv):
    from experiments.exp4 import runner_main

    captured, runner_kwargs = {}, {}

    class _Driver(Exp4Driver):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    class _Runner:
        def __init__(self, *args, **kwargs):
            runner_kwargs.update(kwargs)

        def run(self, run_trial):
            return 0

    monkeypatch.setattr(runner_main, "Exp4Driver", _Driver)
    monkeypatch.setattr(runner_main, "TrialRunner", _Runner)
    assert runner_main.main(argv) == 0
    return captured, runner_kwargs


def test_the_runner_defaults_are_the_recorded_run(monkeypatch, tmp_path):
    kwargs, runner = _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv")])
    assert kwargs["mission_clock"] == "wall" and kwargs["deadline_time_scale"] == 1.0
    assert kwargs["ferry_physics"] == {} and kwargs["rf_link_token"] is None
    assert runner["timeout_s"] == 120.0


def test_the_runner_passes_the_phase_3_flags(monkeypatch, tmp_path):
    kwargs, runner = _runner(monkeypatch, [
        "--csv", str(tmp_path / "t.csv"), "--arms", "H1", "H3", "--N", "6", "--n-missions", "4",
        "--mission-clock", "sim", "--contact-band", "wide", "--backhaul-model", "seconds",
        "--in-flight-response", "replan", "--replan-fallback", "trim",
        "--contact-reliability-source", "channel", "--payload-bytes", "1000000",
        "--deadline-time-scale", "t_nom", "--initial-window-missions", "6",
        "--session-ttl-s", "30", "--n-pl", "3.0", "--shadow-keying", "position",
        "--turnaround-s", "20", "--no-rf-link-token", "--t-nom-layouts", "7",
    ])
    assert kwargs["mission_clock"] == "sim" and kwargs["contact_band"] == "wide"
    assert (kwargs["in_flight_response"], kwargs["replan_fallback"]) == ("replan", "trim")
    assert kwargs["payload_bytes"] == 1_000_000 and kwargs["deadline_time_scale"] == "t_nom"
    assert kwargs["ferry_physics"] == {"n_pl": 3.0, "shadow_keying": "position", "turnaround_s": 20.0}
    assert kwargs["rf_link_token"] is False and kwargs["t_nom_layouts"] == 7
    assert runner["timeout_s"] == 90.0 + 4 * (2 * 6 * 90.0 + 10.0)


def test_the_runner_refuses_an_explicit_h0_on_the_clock(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv"), "--mission-clock", "sim",
                              "--real-model", "--arms", "H0", "H1"])


def test_the_runner_drops_h0_from_the_default_arms_on_the_clock(monkeypatch, tmp_path):
    from experiments.exp4 import runner_main

    grids = []
    real = runner_main._build_grid
    monkeypatch.setattr(runner_main, "_build_grid", lambda **kw: grids.append(kw) or real(**kw))
    _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv"), "--mission-clock", "sim",
                          "--real-model"])
    assert "H0" not in grids[0]["arms"] and "H1" in grids[0]["arms"]


def test_a_wall_clock_run_keeps_h0_with_the_real_model(monkeypatch, tmp_path):
    from experiments.exp4 import runner_main

    grids = []
    real = runner_main._build_grid
    monkeypatch.setattr(runner_main, "_build_grid", lambda **kw: grids.append(kw) or real(**kw))
    _runner(monkeypatch, ["--csv", str(tmp_path / "t.csv"), "--real-model"])
    assert "H0" in grids[0]["arms"]
