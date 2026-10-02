"""Exp 5 addendum, Study 5.11 (a): the decision cost.

The build plan's addendum of 2 Oct 2026 ("Decision cost and scaling"): the
planner's wall time per mission was recorded (``plan_wall_s``) but never
scored, and the flight clock's per-decision wall time was not recorded at all.
Pinned:

* **The mule records each flight decision's wall time** beside its record,
  never inside it: ``MissionRunResult.pass_1_pairs_wall`` holds one entry per
  record of ``pass_1_pairs`` and ``pass_1_e3_wall`` one per call of
  ``pass_1_e3``, each ``{"decide_s", "mask_s"}``. On a clock that counts its
  reads the values are exact: a pair decision reads the clock once before
  binding the predicate, twice around each of its ``pairs`` predicate calls
  and once at the answer, so ``mask_s = pairs`` and ``decide_s = 2·pairs + 1``;
  an E3 call times its mask once (1) and its policy once (1), so
  ``mask_s = 1`` and ``decide_s = 2``. On the real clock 0 <= mask <= decide.
  Each is None exactly when its record is (every arm without a learned
  filling, and a pair mission that decided nothing), and the records still
  hold no wall time.
* **mission_completed** carries them after ``pass_1_e3_unvisited`` and before
  ``energy_status``, left out when None or empty.
* **The consumer** reads them as :class:`DecisionWall` tuples, strictly: a
  time that is not a finite number >= 0 reads as None, and an entry that is
  not a record as two Nones, so the rest stay in step with their decisions.
* **The scorer's cost columns** (:data:`COST_COLUMNS`), hand-worked, only with
  ``cost_columns`` (``--cost-columns``), after the τ and pair columns; the
  default row is unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import time
from types import SimpleNamespace

import pytest

from experiments.analysis.traces_scorer import (
    COST_COLUMNS,
    PHASE_5_COLUMNS,
    CostReport,
    cost_report,
    main,
    score_trial,
    score_traces,
)
from experiments.exp4.events_consumer import DecisionWall, observation_from_rows
from experiments.ferrysim.inprocess import DECISION_WALL_FIELDS, WALL_TOKEN, mask_wall_times
from hermes.l1.mission_clock import SIM_EPOCH_S
from hermes.mule.mule_main import MissionRunResult
from hermes.processes.mule import SIM_MISSION_OPTIONAL_FIELDS, _sim_mission_fields

from tests.integration import test_p5_pair_missions as PM
from tests.unit import test_p5_e3_hook as EH

WALL = 1_700_000_000.0
E = SIM_EPOCH_S
TAUS = (0.82,)
#: The cost columns in the plan's order, restated so that a reordered constant fails.
SPEC_COLUMNS = (
    "plan_wall_s_mean", "plan_wall_s_p95", "plan_search_shares",
    "pair_wall_s_mean", "pair_wall_s_p95", "pair_mask_wall_s_mean",
    "e3_wall_s_mean", "e3_wall_s_p95", "e3_mask_wall_s_mean",
    "flight_decisions_per_mission",
    "trial_processes", "peak_rss_mib_total",
    "peak_rss_mib_cluster", "peak_rss_mib_mule", "peak_rss_mib_device",
)


class CountingClock:
    """``time.perf_counter`` that advances by one second at every read."""

    def __init__(self):
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return float(self.reads)


# --------------------------------------------------------------------------- #
# 1. The mule
# --------------------------------------------------------------------------- #

def _pair_trial(*, missions=2, budget=120.0):
    seed, layout = PM.ref_layout(6, n=12)
    _, _, recs = PM.fly(spec=PM.noisy_spec(seed), layout=layout, missions=missions,
                        **PM.plan_kw(budget=budget, s=3, scorer="greedy_1"))
    return [rec.result for rec in recs]


def test_each_pair_decision_is_timed_beside_its_record_exactly(monkeypatch):
    monkeypatch.setattr(time, "perf_counter", CountingClock())
    results = _pair_trial()
    decided = [r for r in results if r.pass_1_pairs]
    assert decided
    for r in decided:
        assert len(r.pass_1_pairs_wall) == len(r.pass_1_pairs)
        for record, wall in zip(r.pass_1_pairs, r.pass_1_pairs_wall):
            assert wall == {"decide_s": 2.0 * record["pairs"] + 1.0,
                            "mask_s": float(record["pairs"])}
            assert not any("wall" in key for key in record)
        assert r.pass_1_e3_wall is None


def test_on_the_real_clock_the_mask_is_part_of_the_decision():
    for r in _pair_trial():
        if r.pass_1_pairs is None:
            assert r.pass_1_pairs_wall is None
            continue
        for wall in r.pass_1_pairs_wall:
            assert set(wall) == {"decide_s", "mask_s"}
            assert all(isinstance(v, float) and math.isfinite(v) for v in wall.values())
            assert 0.0 <= wall["mask_s"] <= wall["decide_s"]
    assert json.loads(json.dumps([r.pass_1_pairs_wall for r in _pair_trial()]))


def test_a_pair_mission_that_decides_nothing_records_no_wall():
    (r,) = _pair_trial(missions=1, budget=1.0)
    assert r.pass_1_pairs is None and r.pass_1_pairs_wall is None


def test_each_e3_call_is_timed_beside_its_record_exactly(monkeypatch):
    monkeypatch.setattr(time, "perf_counter", CountingClock())
    policy = EH.StubE3()
    _, _, recs = EH.fly(policy, budget=200.0, missions=2)
    for rec in recs:
        r = rec.result
        assert len(r.pass_1_e3_wall) == len(r.pass_1_e3) == len(policy.calls) // 2
        assert r.pass_1_e3_wall == [{"decide_s": 2.0, "mask_s": 1.0}] * len(r.pass_1_e3)
        assert not any("wall" in key for entry in r.pass_1_e3 for key in entry)
        assert r.pass_1_pairs_wall is None


def test_e3_on_the_real_clock():
    _, _, (rec,) = EH.fly(EH.StubE3(), budget=200.0)
    for wall in rec.result.pass_1_e3_wall:
        assert 0.0 <= wall["mask_s"] <= wall["decide_s"] and math.isfinite(wall["decide_s"])


def test_an_arm_without_a_learned_filling_records_no_decision_wall():
    seed, layout = PM.ref_layout(6, n=12)
    for slot in ("committed", "cross_heuristic"):
        _, _, recs = PM.fly(spec=PM.noisy_spec(seed), layout=layout, missions=2,
                            **PM.plan_kw(budget=120.0, s=3, slot=slot))
        for rec in recs:
            assert rec.result.plan_wall_s is not None
            assert (rec.result.pass_1_pairs_wall, rec.result.pass_1_e3_wall) == (None, None)


# --------------------------------------------------------------------------- #
# 2. mission_completed, and FerrySim's masked case
# --------------------------------------------------------------------------- #

def _result(**kw) -> MissionRunResult:
    return MissionRunResult(
        mission_round=1, empty=True, sim_start_s=E, sim_end_s=E + 30.0,
        sim_ledger={"turnaround": 30.0}, pass_1_flown=[], pass_2_flown=[], replans=[],
        aborts=[], inserts=[], offers_refused=[], energy_j=100.0, band="wide",
        pass_1_preflight_drops=[], **kw)


def test_mission_completed_carries_the_walls_after_e3s_records_only_when_set():
    assert SIM_MISSION_OPTIONAL_FIELDS[-2:] == ("pass_1_pairs_wall", "pass_1_e3_wall")
    wall = [{"decide_s": 0.002, "mask_s": 0.001}]
    paired = _sim_mission_fields(_result(pass_1_pairs=[{"t_s": E}], pass_1_pairs_wall=wall))
    assert list(paired)[-3:] == ["pass_1_pairs", "pass_1_pairs_wall", "energy_status"]
    assert paired["pass_1_pairs_wall"] == wall
    e3 = _sim_mission_fields(_result(pass_1_e3=[{"t_s": E}], pass_1_e3_wall=wall))
    assert list(e3)[-3:] == ["pass_1_e3", "pass_1_e3_wall", "energy_status"]
    plain = _sim_mission_fields(_result())
    for empty in ([], None):
        assert _sim_mission_fields(_result(pass_1_pairs_wall=empty,
                                           pass_1_e3_wall=empty)) == plain
    assert not {"pass_1_pairs_wall", "pass_1_e3_wall"} & set(plain)


def test_ferrysims_case_masks_every_decision_wall_and_keeps_the_keys():
    assert DECISION_WALL_FIELDS == ("pass_1_pairs_wall", "pass_1_e3_wall")
    event = {"plan_wall_s": "f:0.25",
             "pass_1_pairs_wall": [{"decide_s": "f:0.002", "mask_s": "f:0.001"}],
             "pass_1_e3_wall": [{"decide_s": "f:0.5", "mask_s": "f:0.0"}]}
    (masked,) = mask_wall_times({"mission_completed": {"m": [dict(event)]}})[
        "mission_completed"]["m"]
    assert masked == {"plan_wall_s": WALL_TOKEN,
                      "pass_1_pairs_wall": [{"decide_s": WALL_TOKEN, "mask_s": WALL_TOKEN}],
                      "pass_1_e3_wall": [{"decide_s": WALL_TOKEN, "mask_s": WALL_TOKEN}]}


# --------------------------------------------------------------------------- #
# 3. The consumer
# --------------------------------------------------------------------------- #

def _mission(rnd, *, plan_search=None, plan_wall=None, pairs=None, e3=None, mule="m1"):
    row = {"ts": WALL + 10.0 * rnd + 6.0, "event": "mission_completed", "role": "mule",
           "id": mule, "mission_round": rnd, "pass_1_contacts": 0, "pass_2_contacts": 0,
           "pass_1_clean_devices": [], "pass_1_merged_devices": [],
           "pass_1_merged_updates": 0, "sim_start_s": E + 200.0 * rnd,
           "sim_end_s": E + 200.0 * rnd + 150.0, "band": "wide", "pass_1_flown": []}
    if plan_search is not None:
        row["plan"] = {"band": "wide", "search": plan_search, "demand": ["a"], "served": ["a"],
                       "score": {"v": -1.0, "mission_s": 100.0}}
    if plan_wall is not None:
        row["plan_wall_s"] = plan_wall
    if pairs is not None:
        row["pass_1_pairs"] = [{"band": "wide", "next_index": None} for _ in pairs]
        row["pass_1_pairs_wall"] = pairs
    if e3 is not None:
        row["pass_1_e3"] = [{"next_index": None} for _ in e3]
        row["pass_1_e3_wall"] = e3
    return [{"ts": WALL + 10.0 * rnd, "event": "mission_started", "role": "mule", "id": mule,
             "sim_start_s": E + 200.0 * rnd}, row]


def _obs(*missions, mule="m1"):
    rows = [{"ts": WALL - 5.0, "event": "mule_ready", "role": "mule", "id": mule,
             "mission_clock": "sim"}]
    for rnd, kw in enumerate(missions, start=1):
        rows += _mission(rnd, mule=mule, **kw)
    return observation_from_rows(cluster_rows=[], mule_rows=rows, device_rows=[],
                                 n_devices=1)


def _w(decide, mask):
    return {"decide_s": decide, "mask_s": mask}


def test_the_consumer_reads_each_decisions_wall_in_order():
    obs = _obs(dict(pairs=[_w(0.003, 0.001), _w(0.005, 0.004)]),
               dict(e3=[_w(0.2, 0.1)]), dict())
    m1, m2, m3 = obs.missions
    assert m1.pair_walls == (DecisionWall(0.003, 0.001), DecisionWall(0.005, 0.004))
    assert (m1.e3_walls, m2.pair_walls) == (None, None)
    assert m2.e3_walls == (DecisionWall(0.2, 0.1),)
    assert (m3.pair_walls, m3.e3_walls) == (None, None)


@pytest.mark.parametrize("value", [True, "0.1", -0.001, float("nan"), float("inf"), None, [0.1]])
def test_a_time_in_a_form_the_mule_never_writes_reads_as_none(value):
    (m,) = _obs(dict(pairs=[_w(value, 0.001), "not a record", _w(0.002, value)])).missions
    assert m.pair_walls == (DecisionWall(None, 0.001), DecisionWall(None, None),
                            DecisionWall(0.002, None))


def test_a_wall_field_that_is_not_a_list_reads_as_absent():
    (m,) = _obs(dict(pairs={"decide_s": 0.1})).missions
    assert m.pair_walls is None


# --------------------------------------------------------------------------- #
# 4. The scorer's cost columns
# --------------------------------------------------------------------------- #

FQ_MULE = {"mule_id": "m1", "rf_range_m": 60.0, "n_missions": 4, "mission_clock": "sim",
           "plan_mode": "ferry", "flight_slot": "pair_q"}
F_MULE = dict(FQ_MULE, flight_slot="committed")
E3_MULE = dict(FQ_MULE, plan_mode="legacy", flight_slot="committed", contact_policy="chen_dqn")
H1_MULE = dict(E3_MULE, contact_policy=None)

#: Four plan-mode missions of an FQ trial: planner walls 0.1, 0.2, 0.3 and 0.4 s
#: (mean 0.25, numpy's p95 0.385); committed modes exact, exact, local and
#: stop_subsets; the pair slot decided 2, 1, 0 and 1 times (4 decisions over 4
#: missions: 1 a mission), decide times 1, 2, 3, 4 ms (mean 2.5 ms, p95 3.85 ms)
#: and mask times 0.5, 1, 1.5 ms and one unreadable (mean 1 ms).
FQ_MISSIONS = (
    dict(plan_search="exact", plan_wall=0.1, pairs=[_w(0.001, 0.0005), _w(0.002, 0.001)]),
    dict(plan_search="exact", plan_wall=0.2, pairs=[_w(0.003, 0.0015)]),
    dict(plan_search="local", plan_wall=0.3),
    dict(plan_search="stop_subsets", plan_wall=0.4, pairs=[_w(0.004, "x")]),
)


def test_the_columns_are_the_plans():
    assert COST_COLUMNS == SPEC_COLUMNS
    assert not set(COST_COLUMNS) & set(PHASE_5_COLUMNS)


def test_an_fq_trials_cost_by_hand():
    got = cost_report(_obs(*FQ_MISSIONS), mule_cfg=FQ_MULE)
    assert got.plan_wall_s_mean == pytest.approx(0.25)
    assert got.plan_wall_s_p95 == pytest.approx(0.385)
    assert got.plan_search_shares == {"exact": 0.5, "stop_subsets": 0.25, "local": 0.25}
    assert got.pair_wall_s_mean == pytest.approx(0.0025)
    assert got.pair_wall_s_p95 == pytest.approx(0.00385)
    assert got.pair_mask_wall_s_mean == pytest.approx(0.001)
    assert (got.e3_wall_s_mean, got.e3_wall_s_p95, got.e3_mask_wall_s_mean) == (None,) * 3
    assert got.flight_decisions_per_mission == pytest.approx(1.0)


def test_e3s_cost_by_hand():
    obs = _obs(dict(e3=[_w(0.2, 0.1), _w(0.4, 0.1)]), dict(e3=[_w(0.6, 0.1)]), dict())
    got = cost_report(obs, mule_cfg=E3_MULE)
    assert (got.plan_wall_s_mean, got.plan_wall_s_p95, got.plan_search_shares) == (None,) * 3
    assert got.e3_wall_s_mean == pytest.approx(0.4)
    assert got.e3_wall_s_p95 == pytest.approx(0.58)
    assert got.e3_mask_wall_s_mean == pytest.approx(0.1)
    assert got.flight_decisions_per_mission == pytest.approx(1.0)       # 3 over 3 missions
    assert got.pair_wall_s_mean is None


def test_a_plan_arm_without_a_learned_filling_has_only_the_planners_columns():
    missions = tuple(dict(m, pairs=None) for m in FQ_MISSIONS)
    got = cost_report(_obs(*missions), mule_cfg=F_MULE)
    assert got.plan_wall_s_mean == pytest.approx(0.25)
    assert got.flight_decisions_per_mission is None
    assert got.pair_wall_s_mean is None


def test_a_learned_trial_that_decided_nothing_counts_zero_decisions():
    got = cost_report(_obs(dict(plan_search="exact", plan_wall=0.1)), mule_cfg=FQ_MULE)
    assert got.flight_decisions_per_mission == 0.0
    assert (got.pair_wall_s_mean, got.pair_wall_s_p95) == (None, None)


def test_an_arm_with_no_decision_cost_scores_blank_and_a_trial_with_no_mission_too():
    assert cost_report(_obs(dict(), dict()), mule_cfg=H1_MULE) == CostReport()
    assert cost_report(_obs(), mule_cfg=FQ_MULE) == CostReport()


def test_with_several_mules_every_mules_decisions_count():
    rows = []
    for mule, (rnd, m) in (("m1", (1, FQ_MISSIONS[0])), ("m2", (1, FQ_MISSIONS[1]))):
        rows.append({"ts": WALL - 5.0, "event": "mule_ready", "role": "mule", "id": mule,
                     "mission_clock": "sim"})
        rows += _mission(rnd, mule=mule, **m)
    obs = observation_from_rows(cluster_rows=[], mule_rows=rows, device_rows=[], n_devices=1)
    got = cost_report(obs, mule_cfg=FQ_MULE)
    assert got.plan_wall_s_mean == pytest.approx(0.15)
    assert got.pair_wall_s_mean == pytest.approx(0.002)                # 1, 2 and 3 ms
    assert got.flight_decisions_per_mission == pytest.approx(1.5)


def _write_trace(root, arm, *, mule_cfg, missions):
    d = root / f"N=1-regime=jittery-rrf=60.0__{arm}__t0__s42"
    d.mkdir(parents=True)
    rows = [{"ts": WALL - 5.0, "event": "mule_ready", "role": "mule", "id": "m1",
             "mission_clock": "sim"}]
    for rnd, kw in enumerate(missions, start=1):
        rows += _mission(rnd, **kw)
    files = {
        "cluster-c1.jsonl": [{"ts": WALL - 6.0, "event": "cluster_ready", "role": "cluster",
                              "id": "c1", "mission_clock": "sim"}],
        "mule-m1.jsonl": rows,
        "device-a.jsonl": [{"ts": WALL - 5.0, "event": "device_ready", "role": "device",
                            "id": "a"}],
    }
    for name, events in files.items():
        (d / name).write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (d / "mule-m1.json").write_text(json.dumps(mule_cfg), encoding="utf-8")
    (d / "cluster.json").write_text(json.dumps(
        {"seed_devices": [{"device_id": "a", "position": [10.0, 0.0, 0.0]}]}), encoding="utf-8")
    return d


def test_the_columns_appear_only_with_cost_columns_and_after_the_pair_columns(tmp_path):
    d = _write_trace(tmp_path, "FQ", mule_cfg=FQ_MULE, missions=FQ_MISSIONS)
    plain = score_trial(d, taus=TAUS).to_row()
    assert not set(COST_COLUMNS) & set(plain)
    costed = score_trial(d, taus=TAUS, cost_columns=True).to_row()
    assert list(costed) == list(plain) + list(COST_COLUMNS)
    assert {c: costed[c] for c in plain} == plain
    both = score_trial(d, taus=TAUS, pair_columns=True, cost_columns=True).to_row()
    assert list(both) == list(plain) + list(PHASE_5_COLUMNS) + list(COST_COLUMNS)
    assert json.loads(costed["plan_search_shares"]) == {
        "exact": 0.5, "stop_subsets": 0.25, "local": 0.25}
    (scored,) = score_traces(tmp_path, taus=TAUS, cost_columns=True)
    assert scored.to_row() == costed
    blank = score_trial(_write_trace(tmp_path, "H1", mule_cfg=H1_MULE, missions=(dict(),)),
                        taus=TAUS, cost_columns=True).to_row()
    assert {c: blank[c] for c in COST_COLUMNS} == dict.fromkeys(COST_COLUMNS, "")


def test_the_cli_writes_the_columns_only_when_asked(tmp_path):
    root = tmp_path / "traces"
    _write_trace(root, "FQ", mule_cfg=FQ_MULE, missions=FQ_MISSIONS)
    out = tmp_path / "scored.csv"
    assert main(["--traces", str(root), "--csv", str(out)]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    assert not set(COST_COLUMNS) & set(row)
    assert main(["--traces", str(root), "--csv", str(out), "--cost-columns"]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    assert list(row)[-len(COST_COLUMNS):] == list(COST_COLUMNS)
    assert float(row["plan_wall_s_mean"]) == pytest.approx(0.25)
    assert float(row["flight_decisions_per_mission"]) == pytest.approx(1.0)
    assert row["e3_wall_s_mean"] == ""
