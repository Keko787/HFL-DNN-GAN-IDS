"""FeRRy Phase 3, unit U9 — several mules on the simulated clock, over real processes.

Real subprocesses over real TCP:

* **Three mules at quorum 1, arrival order against simulated order.** mule-0
  flies one device: short missions in simulated time, but its device never
  delivers (``contact_reliability=0``), so every contact waits a session TTL
  and its missions are slow in wall time. mule-1 flies five devices: long
  missions in simulated time, quick in wall time. mule-2 flies one mission
  and leaves. The cluster folds every upload in simulated order (so mule-1's
  first upload, the first to arrive, waits for mule-0's earlier ones), never
  late; each DOWN answers with the uploader's own time, so no mule's clock is
  dragged (``dock_wait`` 0, critic B9); every mule's departure is traced and
  nobody waits for the mule that left early; and nothing deadlocks.
* **Determinism.** The same trial twice folds the same uploads in the same
  order, at the same simulated times, and every mule flies the same
  simulated missions: the fold order no longer depends on wall timing.
* **Through the Exp 4 driver** (critic B9's refusal lifted): two mules at
  quorum 1 under ``agg:fedex`` with the seconds backhaul, and under FedBuff,
  run to the end on the simulated clock with their uploads in order.

Marked ``slow``: each trial spawns a process tree (a few seconds).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from experiments.exp4.driver import Exp4Driver, trace_dir_name
from experiments.exp4.topology_builder import build_exp4_topology
from experiments.runner import Cell
from hermes.processes import MultiProcessOrchestrator

TURNAROUND_S = 30.0
FOLDS = ("up_bundle_ingested", "backhaul_upload_lost")
MISSION_SIM_KEYS = ("sim_start_s", "sim_end_s", "sim_ledger", "sim_pass_2_start_s",
                    "pass_1_flown", "pass_2_flown", "pass_1_outcomes", "delivered", "undelivered")


def _rows(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _three_mule_trial():
    slices = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2}
    topo = build_exp4_topology(
        n_devices=7, rf_range_m=60.0, n_missions=4, seed=5, n_mules=3,
        min_participation=1, aggregation="agg:cutoff", slice_assignment=slices,
        field_radius_m=100.0, mission_clock="sim", down_wait_s=120.0, dock_on_empty=True,
        session_ttl_s=1.5,
    )
    topo.mules[2].n_missions = 1                       # finishes early
    wall_slow = set(topo.mules[0].expected_devices)
    for dev in topo.devices:
        if dev.device_id in wall_slow:
            dev.contact_reliability = 0.0
    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=30.0)
        for mule_id, handle in orch.mule_handles.items():
            try:
                handle.proc.wait(timeout=240.0)
            except Exception as e:  # pragma: no cover - reported below
                pytest.fail(f"mule {mule_id} did not finish (deadlock?): {e}\n"
                            f"{handle.stderr_tail(80)}")
        codes = {m: h.returncode() for m, h in orch.mule_handles.items()}
        stderr = {m: h.stderr_tail(120) for m, h in orch.mule_handles.items()}
    finally:
        orch.shutdown_all(timeout=10.0, cleanup_tmpdir=False)
    try:
        (cluster_log,) = sorted(orch.tmpdir.glob("cluster-*.jsonl"))
        cluster = _rows(cluster_log)
        mules = {m: _rows(orch.tmpdir / f"mule-{m}.jsonl") for m in codes}
    finally:
        orch.cleanup()
    assert codes == {m: 0 for m in codes}, stderr
    return cluster, mules


def _folds(cluster: List[dict]) -> List[tuple]:
    return [(e["mule_id"], e["mission_round"], e["sim_upload_ts"], e["sim_order_seq"])
            for e in cluster if e["event"] in FOLDS]


def _missions(mules: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    return {m: [e for e in rows if e["event"] == "mission_completed"] for m, rows in mules.items()}


@pytest.fixture(scope="module")
def three_mules():
    return _three_mule_trial()


@pytest.mark.slow
def test_three_mules_fold_in_simulated_order_and_nobody_is_dragged(three_mules):
    cluster, mules = three_mules
    (ready,) = [e for e in cluster if e["event"] == "cluster_ready"]
    assert ready["sim_order"] == "conservative"
    missions = _missions(mules)
    assert {m: len(v) for m, v in missions.items()} == {
        "exp4-mule-0": 4, "exp4-mule-1": 4, "exp4-mule-2": 1}
    for rows in mules.values():
        assert not [e for e in rows if e["event"] in ("mission_failed", "dock_down_timeout")]

    folds = _folds(cluster)
    assert len(folds) == 9
    assert [s for *_rest, s in folds] == list(range(1, 10))
    assert [t for _m, _r, t, _s in folds] == sorted(t for _m, _r, t, _s in folds)
    assert not [e for e in cluster if e["event"] in FOLDS and e["sim_order_late"]]
    # The gate had to reorder: mule-1's first upload arrived while mule-0 was
    # still behind it in simulated time, and waited for it.
    held = {(e["mule_id"], e["mission_round"]): e["held_wall_s"]
            for e in cluster if e["event"] in FOLDS}
    assert held[("exp4-mule-1", 1)] > 0.3, held
    first_1 = next(i for i, f in enumerate(folds) if f[0] == "exp4-mule-1")
    assert any(f[0] == "exp4-mule-0" for f in folds[first_1 + 1:])

    # Quorum 1 in simulated order: every DOWN carries the uploader's own time,
    # so no mule waits in simulated time for another's later upload.
    uploads = {}
    for mule, done in missions.items():
        for e in done:
            assert e["sim_ledger"]["dock_wait"] == 0.0, (mule, e["mission_round"])
            if e["sim_pass_2_start_s"] is not None:
                uploads[(mule, e["mission_round"])] = e["sim_pass_2_start_s"] - TURNAROUND_S
    for mule, rnd, t, _s in folds:
        if (mule, rnd) in uploads:
            assert t == pytest.approx(uploads[(mule, rnd)], abs=1e-6)
    closed = [e["sim_ts"] for e in cluster if e["event"] == "cluster_round_closed"]
    assert closed == sorted(closed)

    departed = [e["mule_id"] for e in cluster if e["event"] == "mule_departed"]
    assert sorted(departed) == sorted(mules) and departed[0] == "exp4-mule-2"


@pytest.mark.slow
def test_the_same_trial_twice_folds_the_same_uploads_in_the_same_order(three_mules):
    cluster_a, mules_a = three_mules
    cluster_b, mules_b = _three_mule_trial()
    assert _folds(cluster_a) == _folds(cluster_b)
    for name in mules_a:
        a = [{k: e.get(k) for k in MISSION_SIM_KEYS} for e in _missions(mules_a)[name]]
        b = [{k: e.get(k) for k in MISSION_SIM_KEYS} for e in _missions(mules_b)[name]]
        assert a == b, name


def _cell(k: int, seed: int) -> Cell:
    params = {"N": 4 * k, "rrf": 60.0, "n_missions": 3, "regime": "jittery"}
    return Cell(cell_id="|".join(f"{a}={b}" for a, b in sorted(params.items())), arm="H1",
                trial_index=0, seed=seed, params=params)


@pytest.mark.slow
@pytest.mark.parametrize("kw", [
    dict(aggregation="agg:fedex", backhaul_model="seconds", contact_band="wide"),
    dict(aggregation="agg:fedbuff", min_participation=2),
], ids=["fedex-quorum1-seconds", "fedbuff"])
def test_two_mules_below_a_full_quorum_run_through_the_driver(tmp_path, kw):
    cell = _cell(2, 4242)
    driver = Exp4Driver(mission_clock="sim", realism=True, n_mules=2, trial_budget_s=240.0,
                        trace_root=tmp_path, **kw)
    row = dict(driver.run_trial(cell))
    assert row["mission_failures"] == 0 and row["missions_completed"] == 6, row
    trace = tmp_path / trace_dir_name(cell)
    cluster = _rows(next(trace.glob("cluster-*.jsonl")))
    (ready,) = [e for e in cluster if e["event"] == "cluster_ready"]
    assert ready["sim_order"] == "conservative"
    folds = _folds(cluster)
    assert [t for _m, _r, t, _s in folds] == sorted(t for _m, _r, t, _s in folds)
    assert [s for *_rest, s in folds] == list(range(1, len(folds) + 1))
    assert not [e for e in cluster if e["event"] in FOLDS and e["sim_order_late"]]
    for f in sorted(trace.glob("mule-*.jsonl")):
        for e in _rows(f):
            if e["event"] == "mission_completed":
                assert e["sim_ledger"]["dock_wait"] == 0.0, (f.name, e["mission_round"])
