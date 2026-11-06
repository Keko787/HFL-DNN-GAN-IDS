"""Exp 5's exploratory band-change energy analysis (scripts/exp5/band_energy.py).

A mission's energy splits into moving and hovering at the model's two powers, the pairing
stops at the first stop the two missions don't share, the arrival price counts every target
both ways, and Holm is monotone and capped at 1."""

from __future__ import annotations

import json

import pytest

from tests.unit.test_exp5_scoring import EXP5, _load

BE = _load("exp5_band_energy", EXP5 / "band_energy.py")


def _stop(band, pos=(1.0, 2.0, 0.0), devices=("a", "b"), dwell=10.0, rates=None):
    return {"position": list(pos), "devices": list(devices), "targets": list(devices), "band": band,
            "dwell_s": dwell, "rate_bps": rates or {d: 8e6 for d in devices}}


def _mission(band, stops, plan=None):
    return {"plan": plan or band, "stops": stops, "planned": [sorted(s["devices"]) for s in stops],
            "energy_j": 1.0, "move_j": 0.0, "hover_j": 1.0, "updates": 1, "round": 1}


def test_a_mission_is_read_with_its_energy_split_and_in_order(tmp_path):
    path = tmp_path / "mule-exp4-mule.jsonl"
    late = {"event": "mission_completed", "mission_round": 2, "band": "wide", "energy_j": 5.0,
            "sim_ledger": {"transit": 1.0, "return": 1.0, "dwell": 2.0, "listen": 1.0},
            "pass_1_merged_updates": 3, "pass_1_flown": [_stop("wide"), "not a stop"],
            "pass_1_plan": [{"devices": ["b", "a"]}]}
    early = dict(late, mission_round=1, band="narrow")
    lines = [{"event": "mule_ready"}, late, early]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    missions = BE.read_missions(path)
    assert [m["plan"] for m in missions] == ["narrow", "wide"]
    assert missions[0]["move_j"] == pytest.approx(2 * BE.P_MOVE_W)
    assert missions[0]["hover_j"] == pytest.approx(3 * BE.P_HOVER_W)
    assert missions[0]["updates"] == 3
    assert len(missions[0]["stops"]) == 1
    assert missions[0]["planned"] == [["a", "b"]]


def test_a_mission_that_flew_the_plans_band_did_not_leave_it():
    assert not BE.left_plan(_mission("medium", [_stop("medium"), _stop("medium")]))
    assert BE.left_plan(_mission("medium", [_stop("medium"), _stop("wide")]))


def test_first_change_finds_the_first_differing_band_on_a_shared_route():
    plan = _mission("medium", [_stop("medium"), _stop("medium", pos=(5, 5, 0))])
    arm = _mission("medium", [_stop("medium"), _stop("wide", pos=(5, 5, 0))])
    x, y = BE.first_change(plan, arm)
    assert (x["band"], y["band"]) == ("medium", "wide")


def test_first_change_stops_where_the_routes_part():
    plan = _mission("medium", [_stop("medium"), _stop("medium", pos=(5, 5, 0))])
    arm = _mission("medium", [_stop("medium"), _stop("wide", pos=(9, 9, 0))])
    assert BE.first_change(plan, arm) is None
    assert BE.first_change(plan, plan) is None


def test_different_members_are_a_different_stop():
    plan = _mission("medium", [_stop("medium")])
    arm = _mission("medium", [_stop("wide", devices=("a", "c"))])
    assert BE.first_change(plan, arm) is None


def test_the_arrival_price_counts_every_target_both_ways():
    stop = _stop("wide", rates={"a": 8e6, "b": 4e6})
    # 1 MB is 8 Mbit each way: 2 s at 8 Mbit/s, 4 s at 4 Mbit/s.
    assert BE.arrival_price_s(stop, payload_bytes=1_000_000) == pytest.approx(2.0 + 4.0)


def test_the_arrival_price_skips_a_target_with_no_rate():
    stop = _stop("wide", rates={"a": 8e6, "b": 0.0})
    assert BE.arrival_price_s(stop, payload_bytes=1_000_000) == pytest.approx(2.0)


def test_holm_scales_by_rank_stays_monotone_and_caps_at_one():
    assert BE.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert BE.holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_a_clear_saving_has_a_ci_below_zero_and_a_small_p():
    diffs = [-1.0, -2.0, -1.5, -1.2, -1.8, -1.1, -1.9, -1.4, -1.6, -1.3]
    lo, hi = BE.boot_ci(diffs, n=500)
    assert hi < 0.0
    assert BE.sign_flip_p(diffs, n=2000) < 0.01
