"""FeRRy Phase 4 (unit U8): the plan arms through the real orchestrator.

Real subprocesses over real TCP, through ``Exp4Driver`` with the traces kept
(the Phase 4 spec's units table, row U8; unit_U3b.md section 7.2):

* **One small stub trial of every plan arm** on the pilots' flags (simulated
  clock, wide reference band, the T_nom deadline unit, re-plan with the trim
  fallback, agg:cutoff, the channel reliability source, 1 MB, a 60 s budget)
  with the age cap at 2. Each ends ok; the kept mule JSON carries the arm's
  plan fields; ``mule_ready`` states them as the scheduler runs them;
  ``mission_completed`` carries each mission's closed plan and its wall time,
  and FB+<class> flies only its class; the row's provenance matches.
* **Member subsets for the H and D arms** on the Phase 3 cliff
  (``device_positions(8, 777, 100.0)``, narrow, 1 MB, 60 s), where every
  mission flies empty at ``whole``: H1, D1 and D5 fly part of the field-wide
  stop every mission under ``--member-admission subset`` (D2 needs the real
  model and is refused on the stub).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.exp4.driver import PLAN_ARMS, Exp4Driver, trace_dir_name
from experiments.runner import Cell
from hermes.processes.config import PLAN_MULE_FIELDS

#: The pilots' flags (the Phase 4 spec, decision 7) at a declared 1 MB and the
#: plan's 60 s budget, the cap S = 2 and T_nom over 5 reference layouts.
PILOT = dict(
    mission_clock="sim", realism=True, contact_band="wide", deadline_time_scale="t_nom",
    in_flight_response="replan", replan_fallback="trim", aggregation="agg:cutoff",
    contact_reliability_source="channel", payload_bytes=1_000_000, mission_budget_s=60.0,
    age_cap_missions=2, t_nom_layouts=5, trial_budget_s=240.0,
)
#: Trial T2 of the Phase 3 final check: the one field-wide narrow stop needs
#: 99.1 s at 1 MB, so under 60 s the recorded gates admit nothing.
CLIFF = dict(
    mission_clock="sim", realism=True, contact_band="narrow", backhaul_model="seconds",
    payload_bytes=1_000_000, contact_reliability_source="origin", mission_budget_s=60.0,
    in_flight_response="abort", deadline_time_scale="t_nom", trial_budget_s=240.0,
)


def _cell(arm: str, seed: int, **params) -> Cell:
    p = {"N": 4, "rrf": 60.0, "n_missions": 2, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


def _events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _trial(root: Path, settings, cell):
    row = dict(Exp4Driver(**settings, trace_root=root).run_trial(cell))
    n = int(cell.params["n_missions"])
    assert (row["missions_completed"], row["mission_failures"]) == (n, 0), row
    trace = root / trace_dir_name(cell)
    (mule_json,) = [json.loads(p.read_text(encoding="utf-8")) for p in trace.glob("mule-*.json")]
    (log,) = list(trace.glob("mule-*.jsonl"))
    by_name = {}
    for e in _events(log):
        by_name.setdefault(e["event"], []).append(e)
    status = json.loads((trace / "trial_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "ok"
    return row, mule_json, by_name


@pytest.mark.slow
@pytest.mark.parametrize("arm", PLAN_ARMS)
def test_every_plan_arm_runs_as_a_real_trial(tmp_path, arm):
    row, mule, events = _trial(tmp_path, PILOT, _cell(arm, 4242))
    assert mule["plan_mode"] == "ferry" and mule["replan_fallback"] == "trim"
    assert mule["t_nom_s"] is not None and mule["t_nom_s"] > 0.0
    pinned = arm[len("FB+"):] if arm.startswith("FB+") else None
    assert mule["band_class_policy"] == ("search" if pinned is None else f"fixed:{pinned}")
    assert mule["contact_band"] == (pinned or "wide")
    assert mule["flight_slot"] == ("cross_heuristic" if arm == "FX" else "committed")
    assert mule["age_cap_missions"] == (None if arm == "F-cap" else 2)
    assert mule["miss_priority"] is (arm != "F-prio")
    assert mule["member_admission"] == "subset"
    (ready,) = events["mule_ready"]
    for name in ("plan_mode", "band_class_policy", "flight_slot", "member_admission",
                 "age_cap_missions", "age_cap_lookahead"):
        assert ready[name] == mule[name], name
    if arm == "F-cov":
        assert (ready["plan_score_params"]["c_cov_per_device"],
                ready["plan_score_params"]["c_link"]) == (0.0, 0.0)
    for e in events["mission_completed"]:
        plan = e["plan"]
        assert plan["band_class_policy"] == mule["band_class_policy"]
        assert plan["visited"] is not None and plan["band"] == e["band"]
        assert isinstance(e["plan_wall_s"], float) and "pass_1_policy_drops" not in e
        assert plan["cap"]["s"] == mule["age_cap_missions"]
        if pinned is not None:
            assert {s["band"] for s in e["pass_1_flown"] + e["pass_2_flown"]} <= {pinned}
            assert e["band"] == pinned
    # Every arm serves someone within two missions (F-cov once the cap binds).
    assert any(e["pass_1_flown"] for e in events["mission_completed"])
    assert row["contact_band"] == (pinned or "search")
    assert row["miss_priority"] == int(arm != "F-prio")
    params = json.loads(row["ferry_params"])
    assert {f: params[f] for f in PLAN_MULE_FIELDS} == {f: mule[f] for f in PLAN_MULE_FIELDS}


@pytest.mark.slow
@pytest.mark.parametrize("arm", ["H1", "D1", "D5"])
def test_member_subsets_fly_the_cliff_as_real_trials(tmp_path, arm):
    cell = _cell(arm, 777, N=8, regime="clean")
    row, mule, events = _trial(tmp_path, dict(CLIFF, member_admission="subset"), cell)
    assert mule["member_admission"] == "subset" and mule["plan_mode"] == "legacy"
    assert json.loads(row["ferry_params"])["member_admission"] == "subset"
    (ready,) = events["mule_ready"]
    assert ready["member_admission"] == "subset" and "plan_mode" not in ready
    for e in events["mission_completed"]:
        flown = [d for s in e["pass_1_flown"] for d in s["devices"]]
        assert 0 < len(flown) < 8, e["pass_1_flown"]
        if arm == "H1":
            assert "pass_1_policy_drops" not in e
            assert [d["reason"] for d in e["pass_1_preflight_drops"]] == ["budget"]
        else:
            drops = e["pass_1_policy_drops"]
            assert drops and all(d["widened"] is False for d in drops)
