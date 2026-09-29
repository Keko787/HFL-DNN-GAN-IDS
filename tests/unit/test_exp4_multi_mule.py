"""FeRRy Phase 2 — the Exp 4 topology with K mules, and arms D3-D5.

* R7: ``n_mules=1`` builds exactly the recorded topology (pinned against a
  copy of the builder as it was before Phase 2); ``n_mules>1`` splits the
  same seeded devices into disjoint, spatially contiguous slices, wires every
  device to its mule, seeds the cluster with the same map, and honours an
  explicit ``slice_assignment``.
* R6: ``agg:plain`` with several mules is refused unless the quorum is every
  mule (the builder and the driver both refuse it), and so is any quorum
  strictly between 1 and every mule, which can strand the run's last partial.
* R8: arms D3-D5 set the right policy and options; D4 with several mules
  splits its devices by CARP once per trial; bad combinations are refused;
  the new provenance columns; a mule that exits non-zero fails the trial.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from typing import List, Optional

import pytest

from experiments.exp4 import driver as driver_module
from experiments.exp4.driver import (
    ARMS,
    PROVENANCE_COLUMNS,
    TRIAL_STATUS_FILE,
    Exp4Driver,
    Exp4MuleFailure,
    d4_slice_assignment,
    trace_dir_name,
)
from experiments.exp4.topology_builder import angular_slices, build_exp4_topology
from experiments.runner import Cell
from hermes.processes import ClusterConfig, DeviceConfig, MuleConfig, TopologyConfig
from hermes.processes.config import mule_config_to_json


# --------------------------------------------------------------------------- #
# R7 — one mule: the recorded topology, byte for byte
# --------------------------------------------------------------------------- #

def _legacy_topology(
    *, n_devices, rf_range_m, n_missions, seed, spread_m=None, session_ttl_s=3.0,
    synth_batch_size=2, min_participation=1, device_reliability=False,
    reliabilities=None, world_radius_m=100.0, field_radius_m=None,
    backhaul_loss_pct=0.0, backhaul_rng_seed=None, mission_budget_s=None,
    contact_policy=None, aggregation="agg:plain", aggregation_params=None,
    fedprox_rho=0.0, pass_2_budget=False,
) -> TopologyConfig:
    """The builder's stub path as it was before Phase 2 (commit 6458c78)."""
    rng = random.Random(seed)
    if spread_m is None:
        spread_m = field_radius_m if field_radius_m is not None else min(rf_range_m * 0.4, 25.0)
    devices: List[DeviceConfig] = []
    for i in range(n_devices):
        x = rng.uniform(-spread_m, spread_m)
        y = rng.uniform(-spread_m, spread_m)
        contact_reliability: Optional[float] = None
        if device_reliability:
            rel_i = float(reliabilities[i]) if reliabilities else 0.575
            d = (float(x) ** 2 + float(y) ** 2) ** 0.5
            d_eff = min(d, rf_range_m)
            rf_factor = max(0.4, 1.0 - d_eff / (3.0 * world_radius_m))
            contact_reliability = max(0.0, min(1.0, rel_i * rf_factor))
        devices.append(DeviceConfig(
            device_id=f"exp4-dev-{i:03d}", position=(float(x), float(y), 0.0),
            train_shard_path=None, input_dim=None, local_epochs=1, local_batch_size=64,
            contact_reliability=contact_reliability, fedprox_rho=float(fedprox_rho),
        ))
    cluster = ClusterConfig(
        cluster_id="exp4-cluster", dock_host="127.0.0.1", dock_port=0,
        synth_batch_size=synth_batch_size, min_participation=min_participation,
        init_theta_path=None, eval_test_path=None, input_dim=None,
        backhaul_loss_pct=backhaul_loss_pct, backhaul_rng_seed=backhaul_rng_seed,
        backhaul_loss_schedule=None, aggregation=str(aggregation),
        aggregation_params=dict(aggregation_params or {}),
    )
    mule = MuleConfig(
        mule_id="exp4-mule", rf_host="127.0.0.1", rf_port=0, rf_range_m=float(rf_range_m),
        session_ttl_s=session_ttl_s, n_missions=int(n_missions), use_rl_selector=False,
        selector_weights_path=None, rf_prior_snr_db=None, mission_budget_s=mission_budget_s,
        contact_policy=contact_policy, mission_window_adaptation=False,
        mission_window_history=5, mission_window_target=0.8, mission_window_gain=2.0,
        mission_window_max_scale=4.0, aggregation=str(aggregation),
        aggregation_params=dict(aggregation_params or {}), pass_2_budget=bool(pass_2_budget),
        deadline_law="additive", deadline_params={}, miss_priority=False,
    )
    topo = TopologyConfig(cluster=cluster, mules=[mule], devices=devices)
    topo.validate()
    return topo


@pytest.mark.parametrize("kwargs", [
    dict(n_devices=4, rf_range_m=60.0, n_missions=2, seed=12345),
    dict(n_devices=6, rf_range_m=60.0, n_missions=3, seed=7, device_reliability=True,
         reliabilities=[0.2, 0.9, 0.5, 0.33, 0.75, 1.0], field_radius_m=100.0,
         backhaul_loss_pct=2.0, backhaul_rng_seed=99, mission_budget_s=60.0,
         contact_policy="max_aoi", aggregation="agg:cutoff",
         aggregation_params={"a_max": 2}, pass_2_budget=True, fedprox_rho=0.01),
], ids=["stub", "realism-budget-cutoff"])
def test_one_mule_is_the_recorded_topology(kwargs):
    legacy = _legacy_topology(**kwargs)
    assert asdict(build_exp4_topology(**kwargs)) == asdict(legacy)
    assert asdict(build_exp4_topology(n_mules=1, **kwargs)) == asdict(legacy)
    topo = build_exp4_topology(**kwargs)
    assert [m.mule_id for m in topo.mules] == ["exp4-mule"]
    assert topo.mules[0].expected_devices == []           # assigned by validate()
    assert topo.cluster.expected_mules == [] and topo.cluster.seed_devices == []


def test_one_mule_positions_are_the_seeds_draws():
    topo = build_exp4_topology(n_devices=5, rf_range_m=60.0, n_missions=1, seed=31)
    rng = random.Random(31)
    expected = []
    for _ in range(5):
        x = rng.uniform(-24.0, 24.0)                     # spread = min(0.4·60, 25)
        y = rng.uniform(-24.0, 24.0)
        expected.append((x, y, 0.0))
    assert [d.position for d in topo.devices] == expected


# --------------------------------------------------------------------------- #
# R7 — K mules over disjoint spatial slices
# --------------------------------------------------------------------------- #

def _angle(dev: DeviceConfig) -> float:
    return math.atan2(dev.position[1], dev.position[0])


def _runs_around_the_circle(labels: List[int]) -> int:
    """Maximal runs of equal labels in circular order."""
    changes = sum(1 for i in range(len(labels)) if labels[i] != labels[i - 1])
    return max(1, changes)


@pytest.mark.parametrize("n_devices, n_mules, seed", [(6, 2, 3), (9, 3, 11), (7, 3, 2024)])
def test_k_mules_get_disjoint_contiguous_slices_of_the_same_devices(n_devices, n_mules, seed):
    one = build_exp4_topology(n_devices=n_devices, rf_range_m=60.0, n_missions=2, seed=seed)
    topo = build_exp4_topology(
        n_devices=n_devices, rf_range_m=60.0, n_missions=2, seed=seed,
        n_mules=n_mules, aggregation="agg:cutoff",
    )
    # Same seeded devices as the single-mule topology.
    assert [asdict(d) for d in topo.devices] == [asdict(d) for d in one.devices]
    ids = [m.mule_id for m in topo.mules]
    assert ids == [f"exp4-mule-{k}" for k in range(n_mules)]
    slices = [set(m.expected_devices) for m in topo.mules]
    assert set().union(*slices) == {d.device_id for d in topo.devices}
    assert sum(len(s) for s in slices) == n_devices                   # disjoint
    assert max(map(len, slices)) - min(map(len, slices)) <= 1          # balanced
    # Spatially separated: around the dock, each slice is one unbroken arc.
    owner = {d: k for k, s in enumerate(slices) for d in s}
    by_angle = sorted(topo.devices, key=_angle)
    assert _runs_around_the_circle([owner[d.device_id] for d in by_angle]) == n_mules
    # Every device is wired to its mule, and the cluster is seeded the same way.
    for d in topo.devices:
        assert topo.mule_for(d.device_id) == ids[owner[d.device_id]]
    assert topo.cluster.expected_mules == ids
    assert {s["device_id"]: s["assigned_mule"] for s in topo.cluster.seed_devices} == {
        d: ids[k] for d, k in owner.items()
    }
    # The mules differ from the single mule only in id and slice.
    for m in topo.mules:
        cfg = asdict(m)
        cfg.pop("mule_id"), cfg.pop("expected_devices")
        ref = asdict(one.mules[0])
        ref.pop("mule_id"), ref.pop("expected_devices")
        ref["aggregation"] = "agg:cutoff"
        assert cfg == ref


def test_the_widest_empty_arc_is_where_the_slices_wrap():
    # Six devices on the right half-plane, none on the left: the wrap point
    # must fall in the empty left half, so slice 0 and slice 1 are both arcs
    # of the right half and neither straddles the -x axis.
    pts = [(math.cos(a), math.sin(a)) for a in (-1.2, -0.7, -0.2, 0.2, 0.7, 1.2)]
    assert angular_slices(pts, 2) == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}


def test_a_slice_assignment_overrides_the_default_split():
    assignment = {0: 1, 1: 0, 2: 1, 3: 1}
    topo = build_exp4_topology(
        n_devices=4, rf_range_m=60.0, n_missions=1, seed=5, n_mules=2,
        aggregation="agg:fedex", slice_assignment=assignment,
    )
    assert topo.mules[0].expected_devices == ["exp4-dev-001"]
    assert topo.mules[1].expected_devices == ["exp4-dev-000", "exp4-dev-002", "exp4-dev-003"]


@pytest.mark.parametrize("kwargs, match", [
    (dict(n_mules=2), "agg:plain with 2 mules needs min_participation=2"),
    (dict(n_mules=2, min_participation=3, dock_on_empty=True), "min_participation must be in"),
    (dict(n_mules=2, aggregation="agg:cutoff", min_participation=2), "needs dock_on_empty"),
    (dict(n_mules=2, aggregation="agg:cutoff", slice_assignment={0: 0, 1: 1}),
     "every device index"),
    (dict(n_mules=2, aggregation="agg:cutoff", slice_assignment={0: 0, 1: 2, 2: 0, 3: 1}),
     "outside mules"),
    (dict(n_mules=5, aggregation="agg:cutoff"), "cannot fill 5 disjoint slices"),
    (dict(n_mules=3, aggregation="agg:cutoff", min_participation=2, dock_on_empty=True),
     r"use 1 \(asynchronous merges\) or 3"),
    (dict(slice_assignment={0: 0, 1: 1, 2: 0, 3: 0}), "n_mules=1"),
    (dict(n_mules=0), "n_mules must be >= 1"),
])
def test_the_builder_refuses_what_would_mis_measure_or_stall(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_exp4_topology(n_devices=4, rf_range_m=60.0, n_missions=1, seed=1, **kwargs)


def test_plain_is_allowed_when_every_mule_is_in_the_quorum():
    topo = build_exp4_topology(
        n_devices=4, rf_range_m=60.0, n_missions=1, seed=1, n_mules=2,
        min_participation=2, dock_on_empty=True,
    )
    assert topo.cluster.min_participation == 2
    assert all(m.dock_on_empty for m in topo.mules)


def test_fedbuff_needs_no_dock_on_empty_for_a_larger_quorum():
    # FedBuff ignores min_participation (K is its own quorum), so it cannot stall.
    build_exp4_topology(
        n_devices=4, rf_range_m=60.0, n_missions=1, seed=1, n_mules=2,
        min_participation=2, aggregation="agg:fedbuff",
    )
    build_exp4_topology(
        n_devices=4, rf_range_m=60.0, n_missions=1, seed=1, n_mules=3,
        min_participation=2, aggregation="agg:fedbuff",
    )


# --------------------------------------------------------------------------- #
# R6 / R8 — the driver
# --------------------------------------------------------------------------- #

CELL = Cell(
    cell_id="N=6|n_missions=2|regime=clean|rrf=60.0", arm="H1", trial_index=0, seed=77,
    params={"N": 6, "rrf": 60.0, "n_missions": 2, "regime": "clean"},
)


def _cell(arm: str) -> Cell:
    return Cell(cell_id=CELL.cell_id, arm=arm, trial_index=0, seed=77, params=CELL.params)


def test_the_new_arms_are_registered():
    assert {"D3", "D4", "D5"} <= set(ARMS)


@pytest.mark.parametrize("kwargs, match", [
    (dict(n_mules=2), "agg:plain with n_mules=2 needs min_participation=2"),
    (dict(n_mules=2, min_participation=3), "min_participation must be in"),
    (dict(n_mules=1, min_participation=2), "min_participation must be in"),
    (dict(n_mules=0), "n_mules must be >= 1"),
    (dict(n_mules=2, min_participation=2, aggregation="agg:cutoff", dock_on_empty=False),
     "needs dock_on_empty"),
    (dict(whittle_variant="optimal"), "whittle_variant"),
    (dict(whittle_weights="shapley"), "whittle_weights"),
    (dict(fedcs_value="bytes"), "fedcs_value"),
    (dict(down_wait_s=0.0), "down_wait_s must be > 0"),
    (dict(n_mules=3, min_participation=2, aggregation="agg:cutoff"),
     r"use 1 \(asynchronous merges\) or 3"),
])
def test_the_driver_refuses_bad_combinations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Exp4Driver(**kwargs)


def _capture_topologies(monkeypatch) -> list:
    seen = []

    def _fake_run_topology(self, topo, **kwargs):
        seen.append(topo)
        return {}

    monkeypatch.setattr(Exp4Driver, "_run_topology", _fake_run_topology)
    return seen


def test_one_mule_h1_gets_the_topology_phase_1_built(monkeypatch):
    """Phase 2 passes nothing new for one mule: the topology is the one the
    Phase-1 driver built (its merge and law settings, no multi-mule ones)."""
    from hermes.mission.aggregation_rules import AggregationSpec

    seen = _capture_topologies(monkeypatch)
    Exp4Driver().run_trial(CELL)
    (topo,) = seen
    ref = build_exp4_topology(
        n_devices=6, rf_range_m=60.0, n_missions=2, seed=77,
        use_rl_selector=False, selector_weights_path=None,
        aggregation="agg:plain", aggregation_params=AggregationSpec().to_params(),
        fedprox_rho=0.0, pass_2_budget=False, deadline_law="additive",
        deadline_params={}, miss_priority=False,
    )
    assert asdict(topo) == asdict(ref)
    (mule,) = topo.mules
    assert mule.mule_id == "exp4-mule" and mule.down_wait_s is None
    assert mule.dock_on_empty is False and topo.cluster.min_participation == 1


@pytest.mark.parametrize("arm, policy, options", [
    ("D3", "whittle", {"whittle_variant": "literal", "whittle_weights": "uniform"}),
    ("D4", "fedex", {}),
    ("D5", "fedcs", {"fedcs_value": "devices"}),
])
def test_each_new_arm_sets_its_policy_and_options(monkeypatch, arm, policy, options):
    seen = _capture_topologies(monkeypatch)
    Exp4Driver(whittle_variant="literal", fedcs_value="devices").run_trial(_cell(arm))
    (topo,) = seen
    (mule,) = topo.mules
    assert mule.contact_policy == policy
    for key, value in options.items():
        assert getattr(mule, key) == value


def test_the_arm_policy_map_is_what_the_driver_reads(monkeypatch):
    """One table names each whole-scheduler arm's policy; the driver used to
    repeat it as an if-chain, so the two could drift apart."""
    seen = _capture_topologies(monkeypatch)
    monkeypatch.setitem(driver_module._ARM_POLICY, "D1", "fedcs")
    Exp4Driver().run_trial(_cell("D1"))
    (topo,) = seen
    (mule,) = topo.mules
    assert mule.contact_policy == "fedcs"


def test_d2_still_needs_a_real_model(monkeypatch):
    seen = _capture_topologies(monkeypatch)
    with pytest.raises(ValueError, match=r"arm D2 \(Oort\) requires --real-model"):
        Exp4Driver().run_trial(_cell("D2"))
    assert seen == []


def test_d3_with_oort_weights_needs_a_real_model(monkeypatch):
    seen = _capture_topologies(monkeypatch)
    with pytest.raises(ValueError, match="requires --real-model"):
        Exp4Driver(whittle_weights="oort").run_trial(_cell("D3"))
    assert seen == []


def test_several_mules_dock_on_empty_and_wait_for_the_trial(monkeypatch):
    seen = _capture_topologies(monkeypatch)
    Exp4Driver(n_mules=2, aggregation="agg:cutoff", trial_budget_s=90.0).run_trial(CELL)
    (topo,) = seen
    assert [m.mule_id for m in topo.mules] == ["exp4-mule-0", "exp4-mule-1"]
    assert all(m.dock_on_empty and m.down_wait_s == 90.0 for m in topo.mules)
    assert topo.cluster.min_participation == 1


def test_d4_splits_its_devices_by_carp_once_per_trial(monkeypatch):
    seen = _capture_topologies(monkeypatch)
    driver = Exp4Driver(n_mules=2, aggregation="agg:fedex")
    driver.run_trial(_cell("D4"))
    driver.run_trial(_cell("D4"))
    first, second = seen
    assert asdict(first) == asdict(second)               # a pure function of the seed
    default = build_exp4_topology(n_devices=6, rf_range_m=60.0, n_missions=2, seed=77)
    expected = d4_slice_assignment(default.devices, 2, 77)
    for k, mule in enumerate(first.mules):
        assert mule.contact_policy == "fedex"
        assert mule.expected_devices == [
            d.device_id for i, d in enumerate(default.devices) if expected[i] == k
        ]


def test_the_carp_split_lowers_the_fedex_objective_below_round_robin():
    from hermes.scheduler.policies.fedex_carp import carp_cost
    from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
    from hermes.types import DeviceID

    devices = build_exp4_topology(
        n_devices=8, rf_range_m=60.0, n_missions=1, seed=4, field_radius_m=100.0,
    ).devices
    positions = {DeviceID(d.device_id): d.position for d in devices}
    model = FeasibilityModel()
    kw = dict(n_transporters=2, speeds=[model.cruise_speed_m_s] * 2,
              t_trans=model.session_time_s)
    split = d4_slice_assignment(devices, 2, 4)
    carp = carp_cost({DeviceID(devices[i].device_id): k for i, k in split.items()},
                     positions, **kw)
    round_robin = carp_cost({DeviceID(d.device_id): i % 2 for i, d in enumerate(devices)},
                            positions, **kw)
    assert carp.async_cost <= round_robin.async_cost


# --------------------------------------------------------------------------- #
# Provenance and mule failures, through _run_topology on a stand-in orchestrator
# --------------------------------------------------------------------------- #

class _Handle:
    def __init__(self, rc: int):
        self._rc = rc

    def returncode(self):
        return self._rc


def _fake_orchestrator(tmp_path, *, mule_rc: int = 0):
    class FakeOrchestrator:
        def __init__(self, topo, capture_output=True):
            self.tmpdir = tmp_path / f"run{len(list(tmp_path.glob('run*')))}"
            self.tmpdir.mkdir()
            self.mule_handles = {m.mule_id: _Handle(mule_rc) for m in topo.mules}
            for m in topo.mules:
                (self.tmpdir / f"mule-{m.mule_id}.json").write_text(mule_config_to_json(m))

        def start_all(self, timeout):
            return

        def shutdown_all(self, timeout, cleanup_tmpdir):
            return

        def cleanup(self):
            return

    return FakeOrchestrator


def _no_wait(monkeypatch):
    monkeypatch.setattr(Exp4Driver, "_await_mules", lambda self, orch, budget_s: True)


def test_single_mule_rows_add_only_the_count_and_blanks(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path))
    _no_wait(monkeypatch)
    row = Exp4Driver().run_trial(CELL)
    assert {c: row[c] for c in ("n_mules", "min_participation", "dock_params", "policy_params")} \
        == {"n_mules": 1, "min_participation": "", "dock_params": "", "policy_params": ""}
    assert set(PROVENANCE_COLUMNS) <= set(row)


def test_multi_mule_rows_record_the_fleet_the_dock_and_the_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path))
    _no_wait(monkeypatch)
    row = Exp4Driver(
        n_mules=3, min_participation=3, dock_on_empty=True, down_wait_s=45.0,
        whittle_variant="literal", default_n_devices=6,
    ).run_trial(_cell("D3"))
    assert row["n_mules"] == 3 and row["min_participation"] == 3
    assert json.loads(row["dock_params"]) == {"dock_on_empty": True, "down_wait_s": 45.0}
    assert json.loads(row["policy_params"]) == {"variant": "literal", "weights": "uniform"}


def test_a_mule_that_exits_non_zero_fails_the_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver_module, "MultiProcessOrchestrator", _fake_orchestrator(tmp_path, mule_rc=3),
    )
    _no_wait(monkeypatch)
    root = tmp_path / "traces"
    with pytest.raises(Exp4MuleFailure, match="exited non-zero"):
        Exp4Driver(trace_root=root).run_trial(CELL)
    marker = json.loads((root / trace_dir_name(CELL) / TRIAL_STATUS_FILE).read_text())
    assert marker["status"] == "error" and "Exp4MuleFailure" in marker["error"]


# --------------------------------------------------------------------------- #
# The runner CLI
# --------------------------------------------------------------------------- #

def _captured_driver_kwargs(monkeypatch, argv) -> dict:
    from experiments.exp4 import runner_main

    captured = {}

    class _Driver(Exp4Driver):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    class _Runner:
        def __init__(self, *args, **kwargs):
            return

        def run(self, run_trial):
            return 0

    monkeypatch.setattr(runner_main, "Exp4Driver", _Driver)
    monkeypatch.setattr(runner_main, "TrialRunner", _Runner)
    assert runner_main.main(argv) == 0
    return captured


def test_the_runner_passes_the_phase_2_flags_to_the_driver(monkeypatch, tmp_path):
    kwargs = _captured_driver_kwargs(monkeypatch, [
        "--csv", str(tmp_path / "t.csv"), "--arms", "D3", "D4", "D5",
        "--aggregation", "agg:fedex", "--agg-fedex-n", "12",
        "--n-mules", "3", "--min-participation", "1", "--no-dock-on-empty",
        "--down-wait-s", "40", "--whittle-variant", "literal",
        "--whittle-weights", "uniform", "--fedcs-value", "devices",
    ])
    assert kwargs["aggregation"] == "agg:fedex"
    assert kwargs["aggregation_params"] == {"fedex_n": 12}
    assert (kwargs["n_mules"], kwargs["min_participation"]) == (3, 1)
    assert kwargs["dock_on_empty"] is False and kwargs["down_wait_s"] == 40.0
    assert (kwargs["whittle_variant"], kwargs["whittle_weights"], kwargs["fedcs_value"]) \
        == ("literal", "uniform", "devices")


def test_the_runner_defaults_leave_one_mule_as_recorded(monkeypatch, tmp_path):
    kwargs = _captured_driver_kwargs(monkeypatch, ["--csv", str(tmp_path / "t.csv")])
    assert (kwargs["n_mules"], kwargs["min_participation"]) == (1, 1)
    assert kwargs["dock_on_empty"] is None and kwargs["down_wait_s"] is None
    assert kwargs["aggregation_params"] == {}


def test_the_runner_refuses_plain_with_a_partial_quorum(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _captured_driver_kwargs(monkeypatch, [
            "--csv", str(tmp_path / "t.csv"), "--n-mules", "2",
        ])
    assert not (tmp_path / "t.csv").exists()
