"""Phase 0 — scoring retained Exp 4 traces.

Three layers under test:

1. **The mule's new trace fields** (``pass_1_plan``, ``pass_1_outcomes`` on
   ``mission_completed``), which make the deadline-miss rate scorable.
2. **The consumer**, which now keeps timestamps and, for traces recorded
   before Freeze Amendment 5, places each ``backhaul_upload_lost`` in the
   mission whose time window contains it — so round closure stops counting
   dropped rounds as closed.
3. **The scorer**, end to end on a synthetic trial whose ages, closure and
   time to τ are worked out by hand below.

The same scorer, run on the recorded L1 cell (``results/exp4_matrix/C_traces``),
reproduces the reach counts in ``HERMES_Matrix_Results.md`` (τ = 0.82: 28/40
vs 19/40; τ = 0.75: 39/40 vs 30/40) and every unaffected CSV column.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from experiments.analysis.traces_scorer import (
    age_profile,
    deadline_misses,
    main,
    parse_trial_dir,
    score_trial,
    score_traces,
    tau_reach,
    write_scores_csv,
)
from experiments.exp4.events_consumer import observation_from_rows
from experiments.exp4.metrics import summarise_observation
from hermes.processes.mule import _pass_1_outcomes_payload, _pass_1_plan_payload
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MuleID,
)

DEVICES = ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# A four-mission trial, in the shape of a trace recorded before Amendment 5
# --------------------------------------------------------------------------- #
#
#   mission  window       Pass-1 CLEAN   upload             merged
#   1        1000–1010    a, b           ingested @1005     a, b   (eval r1 @1006, acc 0.70)
#   2        1010–1020    —              (empty, no dock)   —
#   3        1020–1030    a              LOST @1025         —
#   4        1030–1040    c              ingested @1035     c      (eval r2 @1036, acc 0.85)
#
# Ages after each mission (a, b, c): (0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 0)
# Network AoU, uniform weights: 1/3, 4/3, 7/3, 2  -> mean 1.5, final 2.0
# Round closure (>= 1 update and upload not lost): missions 1 and 4 -> 0.5

def _missions():
    return [
        dict(rnd=1, start=1000.0, end=1010.0, clean=["a", "b"]),
        dict(rnd=2, start=1010.0, end=1020.0, clean=[]),
        dict(rnd=3, start=1020.0, end=1030.0, clean=["a"]),
        dict(rnd=4, start=1030.0, end=1040.0, clean=["c"]),
    ]


def _mule_rows(missions, mule="m1"):
    rows = [{"ts": 999.0, "event": "dock_bootstrapped", "role": "mule", "id": mule}]
    for i, m in enumerate(missions):
        rows.append({"ts": m["start"], "event": "mission_started", "role": "mule",
                     "id": mule, "mission_index": i})
        if not m["clean"]:
            rows.append({"ts": m["end"], "event": "mission_empty", "role": "mule",
                         "id": mule, "mission_round": m["rnd"]})
        row = {
            "ts": m["end"], "event": "mission_completed", "role": "mule", "id": mule,
            "mission_round": m["rnd"], "pass_1_contacts": 1,
            "pass_2_contacts": 1 if m["clean"] else 0,
            "pass_1_updates": len(m["clean"]) if m["clean"] else None,
            "pass_1_scheduled": 3,
            "pass_1_clean_devices": m["clean"] or None,
        }
        for field in ("pass_1_plan", "pass_1_outcomes"):
            if field in m:
                row[field] = m[field]
        rows.append(row)
    return rows


def _cluster_rows():
    def ev(ts, event, **kw):
        return {"ts": ts, "event": event, "role": "cluster", "id": "c1", **kw}

    return [
        ev(998.0, "cluster_ready"),
        ev(999.0, "model_eval", cluster_round=0, accuracy=0.36, auc=0.3, loss=0.7, n_test=100),
        ev(1005.0, "up_bundle_ingested", mule_id="m1", mission_round=None),
        ev(1005.0, "cluster_round_closed", cluster_round=1),
        ev(1006.0, "model_eval", cluster_round=1, accuracy=0.70, auc=0.7, loss=0.6, n_test=100),
        ev(1025.0, "backhaul_upload_lost", mule_id="m1", mission_round=None),
        ev(1035.0, "up_bundle_ingested", mule_id="m1", mission_round=None),
        ev(1035.0, "cluster_round_closed", cluster_round=2),
        ev(1036.0, "model_eval", cluster_round=2, accuracy=0.85, auc=0.9, loss=0.4, n_test=100),
    ]


def _device_rows():
    return [{"ts": 998.0, "event": "device_ready", "role": "device", "id": d} for d in DEVICES]


def _obs(missions=None, cluster_rows=None):
    return observation_from_rows(
        cluster_rows=_cluster_rows() if cluster_rows is None else cluster_rows,
        mule_rows=_mule_rows(_missions() if missions is None else missions),
        device_rows=_device_rows(),
        n_devices=len(DEVICES),
    )


# --------------------------------------------------------------------------- #
# 1. The mule's trace fields
# --------------------------------------------------------------------------- #

def test_the_plan_payload_records_each_contacts_devices_and_deadline():
    queue = [
        ContactWaypoint(position=(1.0, 0.0, 0.0), devices=(DeviceID("a"), DeviceID("b")),
                        bucket=Bucket.NEW, deadline_ts=1005.0),
        ContactWaypoint(position=(9.0, 0.0, 0.0), devices=(DeviceID("c"),),
                        bucket=Bucket.NEW, deadline_ts=1008.0),
    ]
    assert _pass_1_plan_payload(queue) == [
        {"devices": ["a", "b"], "deadline_ts": 1005.0},
        {"devices": ["c"], "deadline_ts": 1008.0},
    ]


def test_the_outcomes_payload_records_every_session():
    report = MissionRoundCloseReport(
        mule_id=MuleID("m1"), mission_round=1, started_at=0.0, finished_at=1.0,
        lines=[
            MissionRoundCloseLine(device_id=DeviceID("a"), outcome=MissionOutcome.CLEAN,
                                  contact_ts=1004.0),
            MissionRoundCloseLine(device_id=DeviceID("b"), outcome=MissionOutcome.TIMEOUT,
                                  contact_ts=1006.0),
        ],
    )
    assert _pass_1_outcomes_payload(SimpleNamespace(report=report, empty=False)) == [
        {"device": "a", "outcome": "clean", "contact_ts": 1004.0},
        {"device": "b", "outcome": "timeout", "contact_ts": 1006.0},
    ]


def test_an_empty_mission_records_no_sessions_rather_than_no_field():
    assert _pass_1_outcomes_payload(SimpleNamespace(report=None, empty=True)) == []
    assert _pass_1_outcomes_payload(SimpleNamespace(report=None, empty=False)) is None


# --------------------------------------------------------------------------- #
# 2. The consumer
# --------------------------------------------------------------------------- #

def test_old_traces_get_their_lost_rounds_back_from_timestamps():
    obs = _obs()
    assert obs.backhaul_lost_rounds == {3}
    closure = summarise_observation(
        obs, n_devices=3, rf_range_m=60.0, n_missions_target=4,
    ).round_close_rate_kmin1
    assert closure == pytest.approx(0.5)      # the recorded CSVs said 0.75


def test_a_recorded_round_is_used_directly():
    cluster = [r for r in _cluster_rows() if r["event"] != "backhaul_upload_lost"]
    cluster.append({"ts": 1015.0, "event": "backhaul_upload_lost", "role": "cluster",
                    "id": "c1", "mule_id": "m1", "mission_round": 2})
    assert _obs(cluster_rows=cluster).backhaul_lost_rounds == {2}


def test_rows_without_timestamps_are_left_as_before():
    """Hand-built rows carry no ``ts``: nothing can be placed, nothing changes."""
    cluster = [{k: v for k, v in r.items() if k != "ts"} for r in _cluster_rows()]
    missions = [{k: v for k, v in r.items() if k != "ts"} for r in _mule_rows(_missions())]
    obs = observation_from_rows(
        cluster_rows=cluster, mule_rows=missions, device_rows=_device_rows(), n_devices=3,
    )
    assert obs.backhaul_lost_rounds == set()
    assert obs.backhaul_losses == 1


def test_missions_keep_their_time_window_and_mule():
    m1 = _obs().missions[0]
    assert (m1.mule_id, m1.started_ts, m1.completed_ts) == ("m1", 1000.0, 1010.0)
    assert m1.contains(1005.0) and not m1.contains(1011.0)
    assert m1.pass_1_deadlines is None and m1.pass_1_outcomes is None


# --------------------------------------------------------------------------- #
# 3. The scorer's parts
# --------------------------------------------------------------------------- #

def test_ages_and_network_aou_match_the_hand_worked_trial():
    ages = age_profile(_obs(), DEVICES)
    assert ages.n_missions == 4
    assert ages.network_aou_mean == pytest.approx(1.5)
    assert ages.network_aou_final == pytest.approx(2.0)
    assert ages.age_max == 3
    assert ages.age_p95 == pytest.approx(3.0)
    assert ages.merged_updates == {"a": 1, "b": 1, "c": 1}   # a's mission-3 update was lost
    assert ages.jain_merged == pytest.approx(1.0)


def test_weights_tilt_network_aou_toward_the_weighted_devices():
    # All weight on c: its ages are 1, 2, 3, 0.
    ages = age_profile(_obs(), DEVICES, weights={"a": 0.0, "b": 0.0, "c": 5.0})
    assert ages.network_aou_mean == pytest.approx(1.5)
    assert ages.network_aou_final == pytest.approx(0.0)
    with pytest.raises(ValueError, match="missing"):
        age_profile(_obs(), DEVICES, weights={"a": 1.0})


def test_time_to_tau_in_missions_rounds_and_seconds():
    obs = _obs()
    late = tau_reach(obs, 0.82)
    assert (late.reached, late.mission, late.cluster_round, late.wall_s) == (True, 4, 2, 36.0)
    early = tau_reach(obs, 0.65)
    assert (early.mission, early.cluster_round, early.wall_s) == (1, 1, 6.0)
    assert not tau_reach(obs, 0.9).reached


def test_the_seed_model_never_counts_as_reaching_tau():
    assert tau_reach(_obs(), 0.3).cluster_round == 1     # round 0 scored 0.36


def test_an_evaluation_just_after_its_mission_belongs_to_that_mission():
    cluster = _cluster_rows()
    cluster[-1] = dict(cluster[-1], ts=1040.5)            # after mission 4 closed
    assert tau_reach(_obs(cluster_rows=cluster), 0.82).mission == 4


def test_deadline_misses_count_late_failed_aborted_and_empty():
    missions = [
        dict(rnd=1, start=1000.0, end=1010.0, clean=["a", "b"],
             pass_1_plan=[{"devices": ["a"], "deadline_ts": 1005.0},
                          {"devices": ["b", "c", "d"], "deadline_ts": 1008.0}],
             pass_1_outcomes=[{"device": "a", "outcome": "clean", "contact_ts": 1004.0},
                              {"device": "b", "outcome": "clean", "contact_ts": 1009.0},
                              {"device": "c", "outcome": "timeout", "contact_ts": 1007.0}]),
        # d was aborted: planned, no session. The empty mission misses everyone.
        dict(rnd=2, start=1010.0, end=1020.0, clean=[],
             pass_1_plan=[{"devices": ["a"], "deadline_ts": 1015.0}],
             pass_1_outcomes=[]),
    ]
    misses = deadline_misses(_obs(missions=missions))
    assert (misses.missions_with_plan, misses.admitted, misses.missed) == (2, 5, 4)
    assert misses.rate == pytest.approx(0.8)


def test_traces_without_the_plan_have_no_deadline_rate():
    assert deadline_misses(_obs()).rate is None


# --------------------------------------------------------------------------- #
# 4. The scorer end to end, on a trial directory
# --------------------------------------------------------------------------- #

TRIAL = "N=3-regime=jittery-rrf=60.0__H1__t7__s42"


def _write_trial(root, name=TRIAL, *, mules=("m1",)):
    d = root / name
    d.mkdir(parents=True)
    rows = _cluster_rows()
    (d / "cluster-c1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for mule in mules:
        (d / f"mule-{mule}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in _mule_rows(_missions(), mule=mule)) + "\n"
        )
        (d / f"mule-{mule}.json").write_text(json.dumps({"rf_range_m": 60.0, "n_missions": 4}))
    for dev in DEVICES:
        (d / f"device-{dev}.jsonl").write_text(
            json.dumps({"ts": 998.0, "event": "device_ready", "role": "device", "id": dev}) + "\n"
        )
    (d / "cluster.json").write_text(json.dumps(
        {"seed_devices": [{"device_id": dev, "position": [0, 0, 0]} for dev in DEVICES]}
    ))
    return d


def test_trial_directory_names_parse():
    key = parse_trial_dir(TRIAL)
    assert (key.cell_id, key.arm, key.trial_index, key.seed) == (
        "N=3-regime=jittery-rrf=60.0", "H1", 7, 42,
    )
    with pytest.raises(ValueError, match="not a trace directory"):
        parse_trial_dir("notes")


def test_score_trial_end_to_end(tmp_path):
    score = score_trial(_write_trial(tmp_path), taus=(0.82, 0.9))
    assert score.key.arm == "H1" and score.n_devices == 3
    assert score.summary.round_close_rate_kmin1 == pytest.approx(0.5)
    assert (score.missions_empty, score.backhaul_lost_missions) == (1, 1)
    assert score.ages.network_aou_mean == pytest.approx(1.5)
    assert [r.reached for r in score.tau] == [True, False]
    assert score.deadlines.rate is None

    row = score.to_row()
    assert row["reached_tau0.82"] == 1 and row["missions_to_tau0.82"] == 4
    assert row["reached_tau0.9"] == 0 and row["missions_to_tau0.9"] == ""
    assert row["deadline_miss_rate"] == ""


def test_score_traces_skips_other_directories_and_filters_arms(tmp_path):
    _write_trial(tmp_path)
    _write_trial(tmp_path, "N=3-regime=jittery-rrf=60.0__D1__t7__s42")
    (tmp_path / "notes").mkdir()
    assert [s.key.arm for s in score_traces(tmp_path)] == ["D1", "H1"]
    assert [s.key.arm for s in score_traces(tmp_path, arms=["H1"])] == ["H1"]


def test_multi_mule_traces_are_refused(tmp_path):
    with pytest.raises(NotImplementedError, match="single-mule"):
        score_trial(_write_trial(tmp_path, mules=("m1", "m2")))


def test_csv_and_cli(tmp_path):
    root = tmp_path / "traces"
    _write_trial(root)
    out = tmp_path / "scored.csv"
    write_scores_csv(score_traces(root), out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["network_aou_mean"] == "1.5"

    cli_out = tmp_path / "cli.csv"
    assert main(["--traces", str(root), "--tau", "0.82", "--csv", str(cli_out)]) == 0
    assert cli_out.exists()
