"""FeRRy Phase 3, unit U7 — the per-role config switches and their guards.

* Old per-role JSON (afa9526's key sets) still loads, every new key at its
  default; a sim-clock config round-trips through JSON.
* ``MuleConfig.ferry_spec_kwargs`` is the one mapping to
  ``FerrySpec.from_config``: every keyword is fed, and the config's defaults
  are the spec's (so an unset field never changes the physics).
* The guards (critic B4, B16): sim-only settings on the wall clock, the
  seconds backhaul without the clock, the channel reliability source without
  a band, a device drawing its own reliability under the channel source
  (several mules on the clock below a full quorum or under FedBuff, critic
  B9, run since unit U9), mismatched
  clocks, backhaul models and RF link tokens, and the mission model's causal
  RF prior schedule (sim only, not with the seconds model, finite, from the
  trace the cluster's loss schedule comes from). A recorded topology
  validates.
* The one index rule every per-mission schedule is read with.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import pytest

from hermes.mule.ferry import FerrySpec
from hermes.processes.config import (
    FERRY_SPEC_FIELDS,
    SIM_ONLY_MULE_FIELDS,
    ClusterConfig,
    DeviceConfig,
    MuleConfig,
    TopologyConfig,
    TopologyValidationError,
    cluster_config_errors,
    cluster_config_from_json,
    cluster_config_to_json,
    device_config_from_json,
    mission_schedule_index,
    mule_config_errors,
    mule_config_from_json,
    mule_config_to_json,
)

#: The keys each role's JSON had at afa9526 (tests/golden/data/topology.json).
AFA9526_MULE_KEYS = (
    "mule_id", "rf_host", "rf_port", "dock_host", "dock_port", "expected_devices",
    "rf_range_m", "session_ttl_s", "n_missions", "use_rl_selector", "selector_weights_path",
    "rf_prior_snr_db", "mission_budget_s", "contact_policy", "mission_window_adaptation",
    "mission_window_history", "mission_window_target", "mission_window_gain",
    "mission_window_max_scale", "aggregation", "aggregation_params", "pass_2_budget",
    "deadline_law", "deadline_params", "miss_priority", "down_wait_s", "dock_on_empty",
    "whittle_variant", "whittle_weights", "fedcs_value",
)
AFA9526_CLUSTER_KEYS = (
    "cluster_id", "dock_host", "dock_port", "expected_mules", "seed_devices",
    "synth_batch_size", "min_participation", "tier3_url", "init_theta_path",
    "eval_test_path", "input_dim", "backhaul_loss_pct", "backhaul_rng_seed",
    "backhaul_loss_schedule", "aggregation", "aggregation_params",
)
AFA9526_DEVICE_KEYS = (
    "device_id", "mule_rf_host", "mule_rf_port", "position", "n_serves", "train_shard_path",
    "input_dim", "local_epochs", "local_batch_size", "contact_reliability", "fedprox_rho",
)


def _sim_mule(**kw) -> MuleConfig:
    kw.setdefault("mule_id", "m")
    kw.setdefault("mission_clock", "sim")
    kw.setdefault("trial_seed", 7)
    kw.setdefault("n_missions", 4)
    return MuleConfig(**kw)


# --------------------------------------------------------------------------- #
# Old JSON loads; new keys default; round-trips
# --------------------------------------------------------------------------- #

def test_old_mule_json_loads_with_every_new_key_at_its_default():
    full = json.loads(mule_config_to_json(MuleConfig(mule_id="m1", rf_range_m=60.0)))
    old = {k: full[k] for k in AFA9526_MULE_KEYS}
    cfg = mule_config_from_json(json.dumps(old))
    assert cfg == MuleConfig(mule_id="m1", rf_range_m=60.0)
    assert cfg.mission_clock == "wall" and cfg.contact_band is None
    assert cfg.backhaul_model == "mission" and cfg.device_availability == {}
    assert cfg.deadline_time_scale == 1.0 and cfg.initial_window_s is None
    assert cfg.rf_link_token is None and mule_config_errors(cfg) == []


def test_old_cluster_and_device_json_load_with_the_new_keys_at_defaults():
    full = json.loads(cluster_config_to_json(ClusterConfig(cluster_id="c")))
    cfg = cluster_config_from_json(json.dumps({k: full[k] for k in AFA9526_CLUSTER_KEYS}))
    assert cfg == ClusterConfig(cluster_id="c")
    assert (cfg.mission_clock, cfg.backhaul_model, cfg.trial_seed) == ("wall", "mission", None)
    dev = DeviceConfig(device_id="d", position=(1.0, 2.0, 0.0))
    raw = {k: json.loads(json.dumps(dataclasses.asdict(dev)))[k] for k in AFA9526_DEVICE_KEYS}
    loaded = device_config_from_json(json.dumps(raw))
    assert loaded == dev
    assert loaded.newest_solicit_only is False and loaded.rf_link_token is None


def test_a_sim_config_round_trips_through_json():
    cfg = _sim_mule(
        rf_range_m=60.0, contact_band="wide", contact_band_classes=["wide", "medium", "narrow"],
        backhaul_model="seconds", backhaul_policy="adaptive", backhaul_regime="jittery",
        t_nom_s=219.0, contact_reliability_source="channel",
        device_availability={"d0": 0.4, "d1": 0.9}, payload_bytes=1_000_000,
        in_flight_response="replan", replan_fallback="trim", n_pl=3.0, shadow_keying="position",
        deadline_time_scale=21.9, initial_window_s=60.0, input_dim=21, rf_link_token="tok",
    )
    assert mule_config_from_json(mule_config_to_json(cfg)) == cfg
    assert mule_config_errors(cfg) == []
    topo = TopologyConfig(
        cluster=ClusterConfig(cluster_id="c", mission_clock="sim", backhaul_model="seconds",
                              trial_seed=7, contact_band_classes=["wide", "medium", "narrow"]),
        mules=[cfg], devices=[DeviceConfig(device_id=d, rf_link_token="tok") for d in ("d0", "d1")],
    )
    assert TopologyConfig.from_json(topo.to_json()).to_json() == topo.to_json()


# --------------------------------------------------------------------------- #
# The one mapping to FerrySpec.from_config
# --------------------------------------------------------------------------- #

def test_every_from_config_keyword_is_fed_and_the_defaults_agree():
    params = inspect.signature(FerrySpec.from_config).parameters
    fed = set(FERRY_SPEC_FIELDS.values()) | {"rf_range_m", "seed", "n_missions"}
    assert fed == set(params) - {"cls"}
    defaults = MuleConfig(mule_id="m")
    for name, kwarg in FERRY_SPEC_FIELDS.items():
        mine = getattr(defaults, name)
        theirs = params[kwarg].default
        assert mine == theirs or (mine == {} and theirs is None), (name, mine, theirs)


def test_ferry_spec_kwargs_build_the_spec_the_mule_would():
    cfg = _sim_mule(rf_range_m=50.0, contact_band="medium", backhaul_model="seconds",
                    backhaul_period_s=800.0, payload_bytes=10_000, listen_s=2.0,
                    contact_reliability_source="channel", device_availability={"d": 0.5})
    spec = FerrySpec.from_config(**cfg.ferry_spec_kwargs())
    direct = FerrySpec.from_config(
        rf_range_m=50.0, seed=7, n_missions=4, contact_band="medium", backhaul_model="seconds",
        backhaul_period=800.0, payload_bytes=10_000, listen_s=2.0,
        contact_reliability_source="channel", device_availability={"d": 0.5},
    )
    assert spec.describe() == direct.describe()
    assert dict(spec.availability) == {"d": 0.5}
    # The kwargs are copies: mutating them leaves the config alone.
    kw = cfg.ferry_spec_kwargs()
    kw["device_availability"]["d"] = 0.0
    assert cfg.device_availability == {"d": 0.5}


# --------------------------------------------------------------------------- #
# Per-role guards
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,value", [
    ("contact_band", "wide"),
    ("backhaul_model", "seconds"),                  # critic B16
    ("contact_reliability_source", "channel"),
    ("in_flight_response", "replan"),
    ("replan_fallback", "trim"),
    ("payload_bytes", 1_000_000),
    ("device_availability", {"d": 0.5}),
    ("n_pl", 3.0),
    ("trial_seed", 7),
    ("input_dim", 21),
    ("rf_prior_schedule_db", [9.0, 11.0]),          # critic B4: the wall clock takes the mean
])
def test_sim_only_settings_are_refused_on_the_wall_clock(field, value):
    errors = mule_config_errors(MuleConfig(mule_id="m", **{field: value}))
    assert errors and field in errors[0] and "mission_clock='sim'" in errors[0]


def test_the_deadline_time_unit_and_the_token_are_allowed_on_the_wall_clock():
    cfg = MuleConfig(mule_id="m", deadline_time_scale=2.0, initial_window_s=90.0,
                     rf_link_token="tok")
    assert mule_config_errors(cfg) == []
    assert "deadline_time_scale" not in SIM_ONLY_MULE_FIELDS


@pytest.mark.parametrize("kw,match", [
    (dict(rf_range_m=None), "rf_range_m"),
    (dict(trial_seed=None), "trial_seed"),
    (dict(contact_reliability_source="channel"), "needs a contact_band"),     # critic B16
    (dict(contact_band="wide", device_availability={"d": 0.5}), "ground truth"),
    (dict(backhaul_model="seconds", n_missions=None, t_nom_s=219.0), "needs its period"),
    (dict(backhaul_model="seconds"), "needs its period"),
    (dict(backhaul_model="hourly"), "backhaul_model"),
    (dict(t_nom_s=-1.0), "t_nom_s"),
    # Critic B4: the schedule is the mission model's prior; the seconds model has its own.
    (dict(backhaul_model="seconds", backhaul_period_s=800.0, rf_prior_schedule_db=[9.0]),
     "critic B4"),
    (dict(rf_prior_schedule_db=[]), "non-empty list"),
    (dict(rf_prior_schedule_db=[9.0, float("nan")]), "finite SNRs"),
    (dict(rf_prior_schedule_db=[9.0, True]), "finite SNRs"),
    (dict(rf_prior_schedule_db="9.0"), "list"),
])
def test_sim_settings_that_cannot_run_are_refused(kw, match):
    kw.setdefault("rf_range_m", 60.0)
    errors = mule_config_errors(_sim_mule(**kw))
    assert any(match in e for e in errors), errors


def test_the_three_deadline_bounds_and_nothing_else():
    """``collection`` (default), ``delivery_per_stop`` and the route-level
    ``delivery``: each runs on the clock and builds its spec; anything else
    is refused before the mule starts; on the wall clock only the default."""
    from hermes.processes.config import DEADLINE_BOUNDS
    from hermes.scheduler.stages.s3b_feasibility import DEADLINE_BOUNDS as SCHEDULER_BOUNDS

    assert DEADLINE_BOUNDS == SCHEDULER_BOUNDS == ("collection", "delivery_per_stop", "delivery")
    assert MuleConfig(mule_id="m").deadline_bounds == "collection"
    for bounds in DEADLINE_BOUNDS:
        cfg = _sim_mule(rf_range_m=60.0, deadline_bounds=bounds)
        assert mule_config_errors(cfg) == []
        assert FerrySpec.from_config(**cfg.ferry_spec_kwargs()).deadline_bounds == bounds
        assert mule_config_from_json(mule_config_to_json(cfg)) == cfg
    for bad in ("landing", "delivery-per-stop", "per_stop", ""):
        errors = mule_config_errors(_sim_mule(rf_range_m=60.0, deadline_bounds=bad))
        assert any("deadline_bounds must be one of" in e for e in errors), (bad, errors)
    for bounds in ("delivery_per_stop", "delivery"):
        errors = mule_config_errors(MuleConfig(mule_id="m", deadline_bounds=bounds))
        assert errors and "deadline_bounds" in errors[0] and "mission_clock='sim'" in errors[0]
    assert mule_config_errors(MuleConfig(mule_id="m")) == []


def test_the_mission_models_rf_prior_schedule_is_allowed_on_the_clock():
    cfg = _sim_mule(rf_range_m=60.0, rf_prior_schedule_db=[9.0, 11.5, 3.25, 12.0])
    assert mule_config_errors(cfg) == []
    assert mule_config_from_json(mule_config_to_json(cfg)) == cfg


@pytest.mark.parametrize("mission_round,length,index", [
    (None, 3, 0), (0, 3, 0), (1, 3, 0), (2, 3, 1), (3, 3, 2), (4, 3, 2), (99, 3, 2),
    (-5, 3, 0), (1, 1, 0), (7, 1, 0),
])
def test_a_mission_reads_its_rounds_entry_clamped_to_the_schedule(mission_round, length, index):
    """One rule for the cluster's loss draw, its events' p_loss and the mule's
    RF prior: entry mission_round - 1, clamped (the EX-4.3 rule)."""
    assert mission_schedule_index(mission_round, length) == index


def test_a_bad_clock_name_is_refused():
    assert "mission_clock" in mule_config_errors(MuleConfig(mule_id="m", mission_clock="gps"))[0]
    assert "mission_clock" in cluster_config_errors(
        ClusterConfig(cluster_id="c", mission_clock="gps"))[0]


def test_the_cluster_refuses_the_seconds_model_off_the_clock_and_without_a_seed():
    assert "critic B16" in cluster_config_errors(
        ClusterConfig(cluster_id="c", backhaul_model="seconds"))[0]
    assert "trial_seed" in cluster_config_errors(
        ClusterConfig(cluster_id="c", mission_clock="sim", backhaul_model="seconds"))[0]
    assert cluster_config_errors(ClusterConfig(
        cluster_id="c", mission_clock="sim", backhaul_model="seconds", trial_seed=1)) == []
    assert cluster_config_errors(ClusterConfig(cluster_id="c")) == []


# --------------------------------------------------------------------------- #
# Topology guards
# --------------------------------------------------------------------------- #

def _topology(*, k=1, quorum=None, aggregation="agg:plain", clock="sim", mule_kw=None,
              cluster_kw=None, device_kw=None) -> TopologyConfig:
    mule_kw = dict(mule_kw or {})
    mules = [
        (_sim_mule if clock == "sim" else MuleConfig)(
            mule_id=f"m{i}", rf_range_m=60.0, expected_devices=[f"d{i}"], **mule_kw)
        for i in range(k)
    ]
    cluster = ClusterConfig(cluster_id="c", min_participation=quorum or k,
                            aggregation=aggregation, mission_clock=clock,
                            **(cluster_kw or {}))
    devices = [DeviceConfig(device_id=f"d{i}", **(device_kw or {})) for i in range(k)]
    return TopologyConfig(cluster=cluster, mules=mules, devices=devices)


def test_a_recorded_topology_validates():
    topo = _topology(k=2, clock="wall", quorum=1, aggregation="agg:cutoff")
    topo.validate()
    assert topo.device_to_mule == {"d0": "m0", "d1": "m1"}


def test_sim_with_several_mules_runs_at_any_quorum_since_u9():
    """Critic B9's refusal is lifted: below a full quorum, or under FedBuff,
    the cluster folds the uploads in simulated order (unit U9), with a DOWN
    wait on every mule (an upload may be held)."""
    wait = dict(down_wait_s=60.0)
    _topology(k=2, quorum=2).validate()
    _topology(k=3, quorum=1, aggregation="agg:cutoff", mule_kw=wait).validate()
    _topology(k=2, quorum=2, aggregation="agg:fedbuff", mule_kw=wait).validate()
    with pytest.raises(TopologyValidationError, match="down_wait_s"):
        _topology(k=3, quorum=1, aggregation="agg:cutoff").validate()
    # One mule at quorum 1 is the ordinary single-mule dock.
    _topology(k=1, quorum=1).validate()


def test_one_trial_runs_on_one_clock_and_one_backhaul_model():
    topo = _topology(k=1)
    topo.cluster.mission_clock = "wall"
    with pytest.raises(TopologyValidationError, match="one clock"):
        topo.validate()
    topo = _topology(k=1, mule_kw=dict(backhaul_model="seconds", t_nom_s=200.0))
    with pytest.raises(TopologyValidationError, match="backhaul_model"):
        topo.validate()
    topo = _topology(k=1, mule_kw=dict(backhaul_model="seconds", t_nom_s=200.0),
                     cluster_kw=dict(backhaul_model="seconds", trial_seed=8))
    with pytest.raises(TopologyValidationError, match="trial_seed"):
        topo.validate()
    _topology(k=1, mule_kw=dict(backhaul_model="seconds", t_nom_s=200.0),
              cluster_kw=dict(backhaul_model="seconds", trial_seed=7)).validate()


def test_the_band_classes_must_agree_between_mules_and_cluster():
    with pytest.raises(TopologyValidationError, match="contact_band_classes"):
        _topology(k=1, mule_kw=dict(contact_band_classes=["wide", "narrow"])).validate()


def test_the_channel_source_refuses_devices_that_still_draw_their_own_reliability():
    kw = dict(contact_band="wide", contact_reliability_source="channel",
              device_availability={"d0": 0.5})
    with pytest.raises(TopologyValidationError, match="drawn twice"):
        _topology(k=1, mule_kw=kw, device_kw=dict(contact_reliability=0.5)).validate()
    _topology(k=1, mule_kw=kw).validate()


def test_the_rf_prior_schedule_needs_the_clusters_loss_schedule_of_the_same_trace():
    """Critic B4: the mule's prior and the cluster's losses come from one L1 trace."""
    kw = dict(rf_prior_schedule_db=[9.0, 11.0, 7.0])
    with pytest.raises(TopologyValidationError, match="one L1 trace"):
        _topology(k=1, mule_kw=kw).validate()
    with pytest.raises(TopologyValidationError, match="2 entries"):
        _topology(k=1, mule_kw=kw,
                  cluster_kw=dict(backhaul_loss_schedule=[0.1, 0.2])).validate()
    _topology(k=1, mule_kw=kw, cluster_kw=dict(backhaul_loss_schedule=[0.1, 0.2, 0.3])).validate()
    # A mule without one needs nothing of the cluster.
    _topology(k=1, cluster_kw=dict(backhaul_loss_schedule=[0.1, 0.2, 0.3])).validate()


def test_a_device_must_carry_its_mules_link_token():
    with pytest.raises(TopologyValidationError, match="rf_link_token"):
        _topology(k=1, clock="wall", mule_kw=dict(rf_link_token="a")).validate()
    with pytest.raises(TopologyValidationError, match="rf_link_token"):
        _topology(k=1, clock="wall", mule_kw=dict(rf_link_token="a"),
                  device_kw=dict(rf_link_token="b")).validate()
    _topology(k=1, clock="wall", mule_kw=dict(rf_link_token="a"),
              device_kw=dict(rf_link_token="a")).validate()


def test_a_role_error_is_raised_by_validate():
    with pytest.raises(TopologyValidationError, match="mule 'm0'.*trial_seed"):
        _topology(k=1, mule_kw=dict(trial_seed=None)).validate()
    with pytest.raises(TopologyValidationError, match="cluster:.*critic B16"):
        _topology(k=1, clock="wall", cluster_kw=dict(backhaul_model="seconds")).validate()
