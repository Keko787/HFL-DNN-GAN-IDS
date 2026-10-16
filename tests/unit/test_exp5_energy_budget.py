"""Exp 5's exploratory energy and budget analysis (scripts/exp5/energy_budget.py).

Energy to tau counts the missions that started before tau was reached (the one in
flight in full, every mule's), energy per update is total energy over merged
updates, the trace values attach to the trial they came from, and the scorer's
paired comparison on them claims for the arm that uses less."""

from __future__ import annotations

import json
from pathlib import Path

from tests.unit.test_exp5_scoring import EXP5, SC, _load, _plan, _write

EB = _load("exp5_energy_budget", EXP5 / "energy_budget.py")


def _mission(start, end, energy, updates, mule="exp4-mule"):
    return {"event": "mission_completed", "id": mule, "sim_start_s": start, "sim_end_s": end,
            "energy_j": energy, "pass_1_merged_updates": updates}


def _trace(d: Path, missions_by_mule):
    d.mkdir(parents=True, exist_ok=True)
    for mule, missions in missions_by_mule.items():
        with (d / f"mule-{mule}.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "mule_ready", "id": mule}) + "\n")
            for m in missions:
                f.write(json.dumps(m) + "\n")


def test_missions_are_read_from_every_mule_in_start_order(tmp_path):
    _trace(tmp_path, {"a": [_mission(1000200.0, 1000300.0, 20.0, 2)],
                      "b": [_mission(1000000.0, 1000150.0, 10.0, 1),
                            _mission(1000400.0, 1000500.0, 30.0, 3)]})
    ms = EB.missions_from_trace(tmp_path)
    assert [m["energy_j"] for m in ms] == [10.0, 20.0, 30.0]
    assert [m["updates"] for m in ms] == [1, 2, 3]


def test_energy_to_tau_counts_the_missions_started_before_tau():
    ms = [{"start": 1e6 + s, "end": 1e6 + s + 150, "energy_j": e, "updates": 1}
          for s, e in [(0.0, 10.0), (200.0, 20.0), (400.0, 30.0)]]
    assert EB.energy_to_tau(ms, 250.0) == 30.0        # the second mission is in flight: in full
    assert EB.energy_to_tau(ms, 200.0) == 10.0        # starts exactly at tau: not counted
    assert EB.energy_to_tau(ms, 1e4) == 60.0
    assert EB.energy_to_tau(ms, None) is None         # never reached tau
    assert EB.energy_to_tau([], 100.0) is None


def test_energy_per_update_is_total_energy_over_merged_updates():
    ms = [{"start": 0, "end": 1, "energy_j": 30.0, "updates": 2},
          {"start": 2, "end": 3, "energy_j": 10.0, "updates": 0}]
    assert EB.energy_per_update(ms) == 20.0
    assert EB.energy_per_update([{"start": 0, "end": 1, "energy_j": 5.0, "updates": 0}]) is None


class _Rows:
    def __init__(self, rows):
        self.scored_rows = rows


def test_trace_values_attach_to_their_own_trial(tmp_path):
    root = tmp_path / "results" / "x_traces"
    _trace(root / "cell__F__t0__s11", {"m": [_mission(0.0, 100.0, 40.0, 2),
                                             _mission(130.0, 230.0, 60.0, 3)]})
    _trace(root / "cell__F__t1__s22", {"m": [_mission(0.0, 100.0, 50.0, 1)]})
    rows = [{"trial_index": "0", "seed": "11", "trace_root": "results\\x_traces", EB.COL: "120.0"},
            {"trial_index": "1", "seed": "22", "trace_root": "results\\x_traces", EB.COL: ""},
            {"trial_index": "2", "seed": "33", "trace_root": "results\\x_traces", EB.COL: "90.0"}]
    missing = EB.attach_energy(_Rows(rows), SC, repo=tmp_path)
    assert missing == 1                                              # trial 2 has no trace
    assert float(rows[0]["energy_to_tau_j"]) == 40.0                 # only mission 1 started before 120 s
    assert float(rows[0]["energy_per_update_j"]) == 20.0             # 100 J over 5 updates
    assert rows[1]["energy_to_tau_j"] == ""                          # never reached tau
    assert float(rows[1]["energy_per_update_j"]) == 50.0
    assert rows[2]["energy_to_tau_j"] == "" and rows[2]["energy_per_update_j"] == ""


def _energy_entry(tmp: Path, variant: str, values):
    rows = [{"cell_id": "c", "arm": variant, "trial_index": i, "seed": f"s{i}", "status": "ok",
             "energy_to_tau_j": v} for i, v in enumerate(values)]
    c = tmp / f"n6k1_knee__{variant}.csv"
    s = tmp / f"n6k1_knee__{variant}_scored.csv"
    _write(c, rows)
    _write(s, rows)
    e = SC.Entry("sx", "n6k1_knee", variant, variant, c, s, len(values))
    SC.load(e)
    return e


def test_the_arm_that_uses_less_energy_is_claimed(tmp_path):
    plan = _plan()
    ferry = _energy_entry(tmp_path, "FX", [30000.0 + 100 * i for i in range(12)])
    base = _energy_entry(tmp_path, "H1", [45000.0 + 120 * i for i in range(12)])
    c = SC.compare_pair(ferry, base, "energy_to_tau_j", plan.specs["sx"], plan)
    c.study = "energy/n6k1_knee/energy_to_tau_j/FX"
    SC.holm([c], plan)
    assert c.n_pairs == 12 and c.mean_diff < 0
    assert c.claim and c.favours == "FX"
    assert EB.lower_count(base, ferry, "energy_to_tau_j") == (12, 12)
