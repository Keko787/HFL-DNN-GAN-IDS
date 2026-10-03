"""FeRRy Phase 4 (unit U9): scoring the plan clock, and the age cap's S*.

The Phase 4 spec's units table, row U9, with other choices 6, 12 and 13 and the
user's decision 1. Pinned:

* **The consumer.** ``MissionRecord`` carries the class flown, each Pass-1
  stop's class, the closed plan (``PlanCommit.describe()``, built here with
  the real type so a schema change fails these tests), its wall time and a
  baseline's drops; None where a mission does not record them.
* **The scorer's Phase 4 columns** on synthetic traces, each value worked by
  hand: cap violations counted from the ages at the trace's own S or at
  ``--age-cap-s`` (own-mule missions only with several mules), the mule's
  log by cause, the band shares per stop, the plan's served share (in
  devices, not weights) and V, the far devices' service, a baseline's drops
  (0 when it dropped nothing, blank where the trace cannot say); where they
  sit in the row and how the CLI takes ``--age-cap-s``. A trace whose plans
  ran two caps is refused, with ``--age-cap-s`` or without.
* **Blank on older traces.** UG4's Phase 3 fixture (the eight 6e6f92d
  trials, rebuilt as kept traces) and the recorded wall-clock cells score
  blank in every new column, and every other column exactly as the 6e6f92d
  scorer (loaded from git) scores them; a fresh wall-clock trial too.
* **Provenance parity** with the driver's row (unit_U3b.md section 5.5): on
  the driver's own trials of every plan arm and of the H and D arms under
  member subsets, with nothing spawned, and on real-process stub trials of
  every plan arm with their traces kept.
* **The S* tool** (``experiments/analysis/age_cap_s_star.py``): on S3a's
  fresh stops alone (``hover=False``) it reproduces the design probe's table
  (the Phase 4 design probe's S* table) line for line on the probe's
  layouts; on the stops the plan is offered (the hover rule, the user's
  decision of 2026-09-30, its default) its covers are schedules and none is
  shorter; its servable set is the plan's own (U1's ``servable_alone`` on
  those stops, and the labels of a plan-mode scheduler on its own layouts,
  in the mission loop too); its greedy bound is a valid cover no shorter
  than the exact one, and its member walk goes on past a stop that admits
  nobody; its own layouts (the tight cluster without realism), decision 1's
  S, an energy capacity (the devices it refuses are the planner's
  ``unplannable``), the seconds backhaul (priced as the mission model), a
  budget that must be finite and positive; the report's (ii) marked wherever
  S + 1 was not measured free of violations; every CLI flag reaching the
  driver and the report; and it agrees with unit U7's independent S* on
  critic B4's deterministic spec.
* **Imports.** Scoring loads no plan module (the legacy import path).
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import inspect
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.analysis import age_cap_s_star as S
from experiments.analysis.traces_scorer import (
    PHASE_4_COLUMNS,
    age_profile,
    band_shares,
    cap_violation_events,
    cap_violations,
    far_devices,
    far_served_share,
    main,
    parse_trial_dir,
    plan_means,
    policy_drops,
    score_trial,
    score_traces,
    trace_cap_s,
    trial_provenance,
)
from experiments.exp4.driver import (
    PLAN_ARMS,
    PROVENANCE_COLUMNS,
    TRIAL_STATUS_FILE,
    Exp4Driver,
    trace_dir_name,
)
from experiments.exp4.events_consumer import MissionRecord, observation_from_rows
from experiments.exp4.model_task import _u32
from experiments.exp4.topology_builder import device_positions, device_spread_m
from experiments.runner import Cell
from hermes.l1.mission_clock import SIM_CEILING_S, SIM_EPOCH_S
from hermes.scheduler.stages.s3b_feasibility import RULE_NONE, FlightState
from hermes.types import Bucket, ContactWaypoint, DeviceID
from hermes.types.scheduler import (
    CAP_REASONS,
    CAP_UNPLANNABLE,
    PLAN_SCORE_KEYS,
    CapViolation,
    PlanCommit,
)

REPO = Path(__file__).resolve().parents[2]
WALL = 1_700_000_000.0
E = SIM_EPOCH_S
BANDS = ("wide", "medium", "narrow")
DEVICES = ("a", "b", "c", "d")
#: a and b within 60 m of the dock, c and d beyond it (Study 5.4's far devices).
POS = {"a": (10.0, 0.0, 0.0), "b": (0.0, 20.0, 0.0), "c": (80.0, 0.0, 0.0),
       "d": (0.0, -100.0, 0.0)}


# --------------------------------------------------------------------------- #
# A plan-mode trial, by hand
# --------------------------------------------------------------------------- #

def _plan(rnd, *, demand, served, v, s=2, ages=None, unplannable=(), visited=None,
          merged=None, weights=None, band="wide"):
    """A closed commit's ``describe()``, as ``mission_completed.plan`` holds it.

    Built with the real ``PlanCommit``, whose checks hold these by hand: the
    capped devices are those aged at least S; each one left out gets a
    plan-time reason (``unplannable`` if listed, else ``crowded``), each one
    served but not visited ``dropped_in_flight``, each one visited but not
    merged ``not_merged``.
    """
    ages = dict(ages or {})
    served = tuple(served)
    visited = set(served if visited is None else visited)
    merged = set(visited if merged is None else merged)
    capped = frozenset(d for d, a in ages.items() if s is not None and a >= s)
    queue = tuple(ContactWaypoint(position=POS[d], devices=(DeviceID(d),),
                                  bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=E + 1e4)
                  for d in served)
    score = {k: 0.0 for k in PLAN_SCORE_KEYS}
    score.update(v=float(v), mission_s=100.0 + rnd)
    at_plan = [CapViolation(DeviceID(d), ages[d], "unplannable" if d in unplannable else "crowded")
               for d in sorted(capped - set(served))]
    commit = PlanCommit(
        mission_round=rnd, band=band, band_index=BANDS.index(band), band_class_policy="search",
        queue=queue, demand=tuple(DeviceID(d) for d in demand),
        weights={DeviceID(d): float((weights or {}).get(d, 1.0)) for d in demand},
        budget_end=E + 60.0, t_ref_s=200.0, score=score,
        constants=(1.0, float(len(demand)), float(len(demand)), 0.1), search_mode="exact",
        n_candidates=5, per_class=({"band": band, "served": len(served)},), cap_s=s,
        ages={DeviceID(d): a for d, a in ages.items()}, capped=capped,
        violations=tuple(at_plan),
    )
    close = [CapViolation(DeviceID(d), ages[d], "dropped_in_flight")
             for d in sorted((capped & set(served)) - visited)]
    close += [CapViolation(DeviceID(d), ages[d], "not_merged")
              for d in sorted((capped & visited) - merged)]
    return commit.close({DeviceID(d) for d in visited}, close).describe()


#: Five missions at S = 2 (ages before each mission are m − U, the last merge
#: U, never merged 0):
#:
#:   m  demand   served  visited  merged  bands flown         plan-time ages -> capped
#:   1  a b c d  a c     a c      a c     wide wide           1 1 1 1 -> none
#:   2  a b c d  a b     a b      a b     narrow              1 2 1 2 -> b d (d crowded)
#:   3  (none)   -       -        -       -                   -
#:   4  c d      c d     c        -       medium              c 3, d 4 (d dropped in
#:                                                            flight, c not merged)
#:   5  a b c d  a d     a d      a d     narrow wide         3 3 4 5 -> all (b crowded,
#:                                                            c unplannable)
#:
#: The scorer's ages after each mission (a b c d): 0 1 0 1 / 0 0 1 2 / 1 1 2 3 /
#: 2 2 3 4 / 0 3 4 0. At S = 2 that is 1 + 2 + 4 + 2 = 9 (device, mission)
#: pairs over all four devices; at S = 3, 1 + 2 + 2 = 5 over b, c, d; at S = 4,
#: 2 over c, d. The mule's log: crowded 2, unplannable 1, dropped_in_flight 1,
#: not_merged 1. Band shares over the six stops: wide 3, narrow 2, medium 1.
#: Served shares 2/4, 2/4, (empty demand, left out), 2/2, 2/4: mean 0.625. V
#: -1, -2, -3, -0.5, -1.5: mean -1.6. The far devices c and d merged once each
#: over 5 missions each: 0.2.
MISSIONS = [
    dict(rnd=1, merged=["a", "c"], visited=["a", "c"], bands=["wide", "wide"],
         plan=_plan(1, demand=DEVICES, served=["a", "c"], v=-1.0,
                    ages=dict(a=1, b=1, c=1, d=1), weights=dict(d=9.0))),
    dict(rnd=2, merged=["a", "b"], visited=["a", "b"], bands=["narrow"],
         plan=_plan(2, demand=DEVICES, served=["a", "b"], v=-2.0,
                    ages=dict(a=1, b=2, c=1, d=2), band="narrow")),
    dict(rnd=3, merged=[], visited=[], bands=[], plan=_plan(3, demand=(), served=(), v=-3.0)),
    dict(rnd=4, merged=[], visited=["c"], bands=["medium"],
         plan=_plan(4, demand=("c", "d"), served=["c", "d"], v=-0.5, ages=dict(c=3, d=4),
                    visited=["c"], merged=[], band="medium")),
    dict(rnd=5, merged=["a", "d"], visited=["a", "d"], bands=["narrow", "wide"],
         plan=_plan(5, demand=DEVICES, served=["a", "d"], v=-1.5,
                    ages=dict(a=3, b=3, c=4, d=5), unplannable=["c"])),
]

#: A Phase 4 build's mule config (it carries ``plan_mode``), plan mode, S = 2.
PLAN_MULE = {"mule_id": "m1", "rf_range_m": 60.0, "n_missions": 5, "session_ttl_s": 3.0,
             "mission_clock": "sim", "trial_seed": 42, "contact_band": "wide",
             "backhaul_model": "mission", "plan_mode": "ferry", "band_class_policy": "search",
             "member_admission": "subset", "flight_slot": "committed", "age_cap_missions": 2,
             "age_cap_lookahead": 0, "plan_score_params": {}, "plan_search_params": {},
             "t_nom_s": 200.0, "miss_priority": True}
#: The same build's H1 mule (plan fields at their defaults).
H1_MULE = dict(PLAN_MULE, plan_mode="legacy", band_class_policy="search",
               age_cap_missions=None, miss_priority=False, t_nom_s=None)
D1_MULE = dict(H1_MULE, contact_policy="max_aoi")


def _mule_rows(missions, *, mule="m1", clock="sim"):
    sim = clock == "sim"
    ready = {"ts": WALL - 5.0, "event": "mule_ready", "role": "mule", "id": mule}
    if sim:
        ready["mission_clock"] = "sim"
    rows = [ready]
    for i, m in enumerate(missions):
        start, end = WALL + 10.0 * i, WALL + 10.0 * i + 6.0
        started = {"ts": start, "event": "mission_started", "role": "mule", "id": mule}
        row = {"ts": end, "event": "mission_completed", "role": "mule", "id": mule,
               "mission_round": m["rnd"], "pass_1_contacts": len(m["bands"]),
               "pass_2_contacts": 0, "pass_1_clean_devices": list(m["visited"]),
               "pass_1_merged_devices": list(m["merged"]),
               "pass_1_merged_updates": len(m["merged"])}
        if sim:
            started["sim_start_s"] = E + 200.0 * i
            row.update(sim_start_s=E + 200.0 * i, sim_end_s=E + 200.0 * i + 150.0,
                       band=m["bands"][0] if m["bands"] else "wide",
                       pass_1_flown=[{"band": b, "devices": []} for b in m["bands"]])
            if m.get("plan") is not None:
                row.update(plan=m["plan"], plan_wall_s=0.01 * m["rnd"])
            if m.get("drops"):
                row["pass_1_policy_drops"] = m["drops"]
        rows += [started, row]
    return rows


def _write_trace(root, name, *, missions=MISSIONS, mule_cfg=PLAN_MULE, clock="sim",
                 positions=POS):
    d = root / name
    d.mkdir(parents=True)
    cluster = [{"ts": WALL - 6.0, "event": "cluster_ready", "role": "cluster", "id": "c1"}]
    if clock == "sim":
        cluster[0]["mission_clock"] = "sim"
    files = {
        "cluster-c1.jsonl": cluster,
        "mule-m1.jsonl": _mule_rows(missions, clock=clock),
        "device-a.jsonl": [{"ts": WALL - 5.0, "event": "device_ready", "role": "device",
                            "id": dev} for dev in DEVICES],
    }
    for fname, rows in files.items():
        (d / fname).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (d / "mule-m1.json").write_text(json.dumps(mule_cfg), encoding="utf-8")
    (d / "cluster.json").write_text(json.dumps({"seed_devices": [
        {"device_id": dev, "position": list(positions[dev])} for dev in DEVICES]}),
        encoding="utf-8")
    return d


TRIAL = "N=4-regime=jittery-rrf=60.0__F__t0__s42"


def _obs(missions=MISSIONS, *, clock="sim"):
    return observation_from_rows(cluster_rows=[], mule_rows=_mule_rows(missions, clock=clock),
                                 device_rows=[], n_devices=len(DEVICES))


# --------------------------------------------------------------------------- #
# 1. The consumer
# --------------------------------------------------------------------------- #

def test_a_plan_mode_mission_carries_its_plan_band_and_drops():
    m1, m2, m3, m4, m5 = _obs().missions
    assert (m1.band, m1.flown_bands) == ("wide", ("wide", "wide"))
    assert (m1.plan_band, m1.plan_search, m1.plan_demand, m1.plan_served) == (
        "wide", "exact", DEVICES, ("a", "c"))
    assert (m1.plan_v, m1.plan_mission_s, m1.plan_wall_s) == (-1.0, 101.0, 0.01)
    assert (m1.cap_s, m1.cap_capped, m1.cap_violations) == (2, (), ())
    assert m1.has_plan and m1.policy_drops is None
    # The commit's own order: plan-time reasons first, then the close's.
    assert m2.cap_violations == (("d", 2, "crowded"),)
    assert (m3.plan_demand, m3.plan_served, m3.has_plan) == ((), (), True)
    assert m4.cap_violations == (("d", 4, "dropped_in_flight"), ("c", 3, "not_merged"))
    assert m4.cap_capped == ("c", "d") and m4.flown_bands == ("medium",)
    assert m5.cap_violations == (("c", 4, "unplannable"), ("b", 3, "crowded"))


def test_a_baselines_drops_are_read_as_reported_and_an_absent_field_as_none():
    drops = [{"position": [1.0, 2.0, 0.0], "devices": ["b", "d"], "deadline_ts": E + 9.0,
              "reason": "budget", "widened": False},
             {"position": [3.0, 4.0, 0.0], "devices": ["a"], "deadline_ts": E + 9.0,
              "reason": "energy", "widened": False}]
    missions = [dict(m, plan=None) for m in MISSIONS]
    missions[1] = dict(missions[1], drops=drops)
    obs = _obs(missions)
    assert obs.missions[1].policy_drops == ((("b", "d"), "budget"), (("a",), "energy"))
    assert [m.policy_drops for m in obs.missions if m is not obs.missions[1]] == [None] * 4
    assert policy_drops(obs) == 3 and policy_drops(_obs()) == 0
    assert not any(m.has_plan for m in obs.missions)
    assert {(m.plan_demand, m.plan_v, m.cap_s, m.plan_wall_s) for m in obs.missions} == {
        (None, None, None, None)}


def test_a_wall_clock_mission_records_none_of_it():
    for m in _obs(clock="wall").missions:
        assert (m.band, m.flown_bands, m.plan_band, m.plan_demand, m.plan_v, m.cap_s,
                m.cap_violations, m.plan_wall_s, m.policy_drops) == (None,) * 9
        assert not m.has_plan


def test_every_new_record_field_defaults_to_none():
    record = MissionRecord(mission_round=1, pass_1_contacts=0, pass_2_contacts=0,
                           pass_1_updates=None, pass_1_scheduled=None,
                           pass_1_clean_devices=(), delivered=None, undelivered=None,
                           duration_s=None)
    new = ("band", "flown_bands", "plan_band", "plan_search", "plan_demand", "plan_served",
           "plan_v", "plan_mission_s", "cap_s", "cap_capped", "cap_violations", "plan_wall_s",
           "policy_drops")
    assert {name: getattr(record, name) for name in new} == dict.fromkeys(new)
    assert not record.has_plan


# --------------------------------------------------------------------------- #
# 2. The scorer's columns, on the hand-built trial
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s, pairs, devices", [(1, 14, 4), (2, 9, 4), (3, 5, 3), (4, 2, 2),
                                               (6, 0, 0)])
def test_cap_violations_count_the_pairs_aged_at_least_s(s, pairs, devices):
    got = cap_violations(_obs(), DEVICES, s)
    assert (got.s, got.pairs, got.devices) == (s, pairs, devices)
    # The same ages the scorer's age profile reports: its worst is d's 4.
    assert age_profile(_obs(), DEVICES).age_max == 4


def test_a_mission_that_merges_nothing_ages_every_device_and_a_lost_upload_merges_nothing():
    """Mission 2's upload lost on the backhaul: its CLEAN a and b never reach
    the model, so a counts from mission 1 and b was never merged. "Merged" is
    the scorer's (``merged_devices``), as for ``age_profile``."""
    rows = _mule_rows(MISSIONS)
    lost = [{"ts": WALL + 13.0, "event": "backhaul_upload_lost", "role": "cluster",
             "id": "c1", "mule_id": "m1", "mission_round": 2}]
    obs = observation_from_rows(cluster_rows=lost, mule_rows=rows, device_rows=[],
                                n_devices=len(DEVICES))
    # Ages after each mission (a b c d): 0 1 0 1 / 1 2 1 2 / 2 3 2 3 / 3 4 3 4 / 0 5 4 0.
    assert (cap_violations(obs, DEVICES, 2).pairs, cap_violations(obs, DEVICES, 2).devices) == (
        2 + 4 + 4 + 2, 4)
    assert cap_violations(obs, DEVICES, 5).pairs == 1


def test_cap_violations_refuse_a_cap_below_one():
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            cap_violations(_obs(), DEVICES, bad)


def _flight(mule, rnd, t, merged):
    return [{"ts": t, "event": "mission_started", "role": "mule", "id": mule},
            {"ts": t + 1.0, "event": "mission_completed", "role": "mule", "id": mule,
             "mission_round": rnd, "pass_1_contacts": 1, "pass_2_contacts": 0,
             "pass_1_clean_devices": merged, "pass_1_merged_devices": merged}]


def test_with_several_mules_a_device_counts_only_its_own_mules_missions():
    """Mule A flies a1 and a2, mule B flies b1, interleaved: A1 (a1 merged), B1,
    A2 (a2), B2, A3. Ages after each device's own missions: a1 0, 1, 2 at A1,
    A2, A3; a2 1, 0, 1; b1 1, 2 at B1, B2. At S = 2 that is a1 at A3 and b1
    at B2. Sampled after every fleet mission, b1 would also count at A3."""
    rows = [*_flight("A", 1, 0.0, ["a1"]), *_flight("B", 1, 2.0, []),
            *_flight("A", 2, 4.0, ["a2"]), *_flight("B", 2, 6.0, []),
            *_flight("A", 3, 8.0, [])]
    obs = observation_from_rows(cluster_rows=[], mule_rows=rows, device_rows=[], n_devices=3,
                                mule_slices={"A": ("a1", "a2"), "B": ("b1",)})
    got = cap_violations(obs, ("a1", "a2", "b1"), 2)
    assert (got.pairs, got.devices) == (2, 2)
    # a1 merged once over its 3 missions, b1 never over its 2.
    assert far_served_share(obs, ("a1", "a2", "b1"), {"a1", "b1"}) == pytest.approx(1 / 5)


def test_the_trace_s_is_its_plans_else_its_plan_mode_config():
    obs = _obs()
    assert trace_cap_s(obs, PLAN_MULE) == 2
    assert trace_cap_s(obs, {}) == 2                       # the plans say it
    no_plans = _obs([dict(m, plan=None) for m in MISSIONS])
    assert trace_cap_s(no_plans, PLAN_MULE) == 2           # a plan mule that planned nothing
    assert trace_cap_s(no_plans, dict(PLAN_MULE, age_cap_missions=None)) is None
    assert trace_cap_s(no_plans, dict(PLAN_MULE, plan_mode="legacy")) is None
    assert trace_cap_s(no_plans, H1_MULE) is None
    other = [dict(MISSIONS[0], plan=_plan(1, demand=DEVICES, served=["a", "c"], v=-1.0, s=3,
                                          ages=dict(a=1, b=1, c=1, d=1)))] + MISSIONS[1:]
    with pytest.raises(ValueError, match="different age caps"):
        trace_cap_s(_obs(other), PLAN_MULE)


@pytest.mark.parametrize("age_cap_s", [None, 2, 3])
def test_a_trace_whose_plans_ran_two_caps_is_refused_by_name(tmp_path, age_cap_s):
    """Refused with a study S or without: the pairs would be counted at one S,
    but ``cap_violation_events``, the mule's own log, would add up mission 1's
    at S = 3 and the others' at S = 2."""
    other = [dict(MISSIONS[0], plan=_plan(1, demand=DEVICES, served=["a", "c"], v=-1.0, s=3,
                                          ages=dict(a=1, b=1, c=1, d=1)))] + MISSIONS[1:]
    root = tmp_path / "traces"
    d = _write_trace(root, TRIAL, missions=other)
    with pytest.raises(ValueError, match=rf"^{TRIAL}: the trace's plans ran different age caps"):
        score_trial(d, age_cap_s=age_cap_s)
    given = [] if age_cap_s is None else ["--age-cap-s", str(age_cap_s)]
    with pytest.raises(ValueError, match=r"different age caps: \[2, 3\]"):
        main(["--traces", str(root), *given])


def test_the_mules_log_is_counted_by_cause_every_cause_listed():
    assert cap_violation_events(_obs()) == {
        "unplannable": 1, "crowded": 2, "dropped_in_flight": 1, "not_merged": 1}
    assert set(cap_violation_events(_obs())) == set(CAP_REASONS)
    # A cause no mission logged still reads 0: the cap ran, and failed no one so.
    assert cap_violation_events(_obs(MISSIONS[:2])) == {
        "unplannable": 0, "crowded": 1, "dropped_in_flight": 0, "not_merged": 0}
    assert cap_violation_events(_obs(MISSIONS[:1])) == dict.fromkeys(CAP_REASONS, 0)
    capless = [dict(m, plan=_plan(m["rnd"], demand=(), served=(), v=-1.0, s=None))
               for m in MISSIONS]
    assert cap_violation_events(_obs(capless)) is None       # F-cap logs nothing
    assert cap_violation_events(_obs([dict(m, plan=None) for m in MISSIONS])) is None


def test_band_shares_count_every_pass_1_stop_on_its_own_class():
    assert band_shares(_obs()) == pytest.approx({"medium": 1 / 6, "narrow": 1 / 3, "wide": 0.5})
    assert list(band_shares(_obs())) == ["medium", "narrow", "wide"]
    assert band_shares(_obs([dict(m, bands=[]) for m in MISSIONS])) is None
    assert band_shares(_obs(clock="wall")) is None


def test_the_plan_share_is_counted_in_devices_not_in_weights():
    """Mission 1 weighs d nine times the others: its weighted served share is
    2/12, its device share 2/4, which is what the column holds."""
    served, v = plan_means(_obs())
    assert served == pytest.approx(0.625) and v == pytest.approx(-1.6)
    assert plan_means(_obs([dict(m, plan=None) for m in MISSIONS])) == (None, None)


def test_far_devices_are_beyond_rf_range_of_the_dock_in_the_plane():
    assert far_devices(POS, 60.0) == ("c", "d")
    assert far_devices({**POS, "e": (60.0, 0.0, 50.0)}, 60.0) == ("c", "d")   # exactly at R
    assert far_devices(POS, 200.0) == ()
    assert far_served_share(_obs(), DEVICES, far_devices(POS, 60.0)) == pytest.approx(0.2)
    assert far_served_share(_obs(), DEVICES, ()) is None


def _row(root, name, **kw):
    return score_trial(_write_trace(root, name, **kw), taus=(0.82,)).to_row()


def test_the_columns_of_a_plan_mode_trial_and_where_they_sit(tmp_path):
    row = _row(tmp_path, TRIAL)
    cols = list(row)
    at = cols.index("deadline_basis") + 1
    assert tuple(cols[at:at + len(PHASE_4_COLUMNS)]) == PHASE_4_COLUMNS
    assert cols[at + len(PHASE_4_COLUMNS)] == "reached_tau0.82"
    assert {c: row[c] for c in PHASE_4_COLUMNS} == {
        "cap_s": 2, "cap_violations": 9, "cap_violation_devices": 4,
        "cap_violation_events": json.dumps(
            {"crowded": 2, "dropped_in_flight": 1, "not_merged": 1, "unplannable": 1},
            sort_keys=True),
        "band_shares": json.dumps({"medium": 1 / 6, "narrow": 1 / 3, "wide": 0.5},
                                  sort_keys=True),
        "plan_served_share_mean": pytest.approx(0.625),
        "plan_v_mean": pytest.approx(-1.6),
        "far_served_share": pytest.approx(0.2),
        "policy_drops": "",
    }
    # Plan mode: the provenance shows the search and the plan fields.
    assert row["contact_band"] == "search"
    assert json.loads(row["ferry_params"])["age_cap_missions"] == 2


def test_violations_are_counted_at_the_traces_s_unless_one_is_given(tmp_path):
    d = _write_trace(tmp_path, TRIAL)
    at_own = score_trial(d).to_row()
    at_3 = score_trial(d, age_cap_s=3).to_row()
    assert (at_own["cap_s"], at_own["cap_violations"], at_own["cap_violation_devices"]) == (2, 9, 4)
    assert (at_3["cap_s"], at_3["cap_violations"], at_3["cap_violation_devices"]) == (3, 5, 3)
    # The mule's own log stays the one it wrote, at the S it ran.
    assert at_3["cap_violation_events"] == at_own["cap_violation_events"]
    assert {c: at_3[c] for c in PHASE_4_COLUMNS[4:]} == {c: at_own[c] for c in PHASE_4_COLUMNS[4:]}
    for bad in (0, -2, True):
        with pytest.raises(ValueError):
            score_trial(d, age_cap_s=bad)


def test_an_arm_without_a_cap_is_scored_at_the_given_s_only(tmp_path):
    """F-cap, and every H and D arm, run no cap: their cap columns are blank
    unless --age-cap-s scores the study's arms at one S."""
    capless = [dict(m, plan=_plan(m["rnd"], demand=(), served=(), v=-1.0, s=None))
               for m in MISSIONS]
    d = _write_trace(tmp_path, TRIAL, missions=capless,
                     mule_cfg=dict(PLAN_MULE, age_cap_missions=None))
    blank = score_trial(d).to_row()
    assert [blank[c] for c in PHASE_4_COLUMNS[:4]] == ["", "", "", ""]
    given = score_trial(d, age_cap_s=2).to_row()
    assert [given[c] for c in PHASE_4_COLUMNS[:4]] == [2, 9, 4, ""]


def _no_plan(missions=MISSIONS, **extra):
    return [dict(m, plan=None, **extra) for m in missions]


def test_a_phase_4_h_arm_fills_what_it_records_and_leaves_the_plan_blank(tmp_path):
    row = _row(tmp_path, "N=4-regime=jittery-rrf=60.0__H1__t0__s42",
               missions=_no_plan(), mule_cfg=H1_MULE)
    assert {c: row[c] for c in PHASE_4_COLUMNS} == {
        "cap_s": "", "cap_violations": "", "cap_violation_devices": "",
        "cap_violation_events": "",
        "band_shares": json.dumps({"medium": 1 / 6, "narrow": 1 / 3, "wide": 0.5},
                                  sort_keys=True),
        "plan_served_share_mean": "", "plan_v_mean": "",
        "far_served_share": pytest.approx(0.2), "policy_drops": "",
    }
    assert row["contact_band"] == "wide" and row["ferry_params"] != ""


def test_a_baselines_drops_are_zero_when_it_reported_none_and_blank_before_phase_4(tmp_path):
    drops = [{"devices": ["b", "d"], "reason": "budget", "widened": False},
             {"devices": ["a"], "reason": "energy", "widened": False}]
    missions = _no_plan()
    missions[1] = dict(missions[1], drops=drops)
    name = "N=4-regime=jittery-rrf=60.0__D1__t{}__s42"
    assert _row(tmp_path, name.format(0), missions=missions, mule_cfg=D1_MULE)[
        "policy_drops"] == 3
    assert _row(tmp_path, name.format(1), missions=_no_plan(), mule_cfg=D1_MULE)[
        "policy_drops"] == 0
    # A Phase 3 D1 (its config has no plan fields) reported none of its drops.
    phase_3 = {k: v for k, v in D1_MULE.items() if k not in (
        "plan_mode", "band_class_policy", "member_admission", "flight_slot",
        "age_cap_missions", "age_cap_lookahead", "plan_score_params", "plan_search_params")}
    old = _row(tmp_path, name.format(2), missions=_no_plan(), mule_cfg=phase_3)
    assert {c: old[c] for c in PHASE_4_COLUMNS} == dict.fromkeys(PHASE_4_COLUMNS, "")
    # On the wall clock no baseline reports its drops.
    wall = _row(tmp_path, name.format(3), missions=_no_plan(), clock="wall",
                mule_cfg={k: v for k, v in D1_MULE.items() if k != "mission_clock"})
    assert {c: wall[c] for c in PHASE_4_COLUMNS} == dict.fromkeys(PHASE_4_COLUMNS, "")


def test_the_far_share_needs_positions_and_the_rf_range(tmp_path):
    d = _write_trace(tmp_path, TRIAL)
    (d / "cluster.json").write_text(json.dumps({"seed_devices": [
        {"device_id": dev} for dev in DEVICES]}), encoding="utf-8")
    assert score_trial(d).to_row()["far_served_share"] == ""
    for dev in DEVICES:
        (d / f"device-{dev}.json").write_text(json.dumps(
            {"device_id": dev, "position": list(POS[dev])}), encoding="utf-8")
    assert score_trial(d).to_row()["far_served_share"] == pytest.approx(0.2)
    # "Far" is beyond the mule's rf_range_m: without it the column is blank,
    # and the rest of the row is scored.
    (d / "mule-m1.json").write_text(json.dumps(
        {k: v for k, v in PLAN_MULE.items() if k != "rf_range_m"}), encoding="utf-8")
    row = score_trial(d).to_row()
    assert (row["far_served_share"], row["cap_s"], row["cap_violations"]) == ("", 2, 9)


def test_the_cli_takes_the_age_cap_and_says_when_a_trace_ran_another(tmp_path, capsys):
    root = tmp_path / "traces"
    _write_trace(root, TRIAL)
    out = tmp_path / "scored.csv"
    assert main(["--traces", str(root), "--age-cap-s", "3", "--csv", str(out)]) == 0
    (row,) = list(csv.DictReader(open(out, encoding="utf-8")))
    assert (row["cap_s"], row["cap_violations"], row["cap_violation_devices"]) == ("3", "5", "3")
    assert "ran the age cap S = 2" in capsys.readouterr().out
    assert main(["--traces", str(root), "--csv", str(out)]) == 0
    (row,) = list(csv.DictReader(open(out, encoding="utf-8")))
    assert row["cap_s"] == "2" and "ran the age cap" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--traces", str(root), "--age-cap-s", "0"])


# --------------------------------------------------------------------------- #
# 3. Older traces score blank, and as the 6e6f92d scorer scores them
# --------------------------------------------------------------------------- #

REF_COMMIT = "6e6f92d"


def _git_show(path: str):
    if shutil.which("git") is None:
        return None
    done = subprocess.run(["git", "show", f"{REF_COMMIT}:{path}"], cwd=REPO,
                          capture_output=True, timeout=120)
    return done.stdout if done.returncode == 0 else None


@pytest.fixture(scope="module")
def ref_scorer(tmp_path_factory):
    """``traces_scorer`` and its consumer as recorded at 6e6f92d, from git.

    Each blob runs as a module of its own under a private name; while the
    scorer runs, ``experiments.exp4.events_consumer`` names the recorded
    consumer, so the reference scores with the consumer it was written for.
    What else they import is live (the driver's provenance columns, the
    metrics), unchanged by this unit.
    """
    blobs = {name: _git_show(path) for name, path in (
        ("consumer", "experiments/exp4/events_consumer.py"),
        ("scorer", "experiments/analysis/traces_scorer.py"))}
    if any(b is None for b in blobs.values()):
        pytest.skip(f"git or commit {REF_COMMIT} is not available")
    where = tmp_path_factory.mktemp("ref6e6f92d")
    names = {"consumer": f"experiments.exp4._p4ref_{REF_COMMIT}_events_consumer",
             "scorer": f"experiments.analysis._p4ref_{REF_COMMIT}_traces_scorer"}
    live = sys.modules["experiments.exp4.events_consumer"]
    loaded = {}
    try:
        for key in ("consumer", "scorer"):
            path = where / f"{key}.py"
            path.write_bytes(blobs[key])
            spec = importlib.util.spec_from_file_location(names[key], path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[names[key]] = module
            if key == "scorer":
                sys.modules["experiments.exp4.events_consumer"] = loaded["consumer"]
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules["experiments.exp4.events_consumer"] = live
            loaded[key] = module
        # The live ferry fields, less the Exp 5 addendum's (Study 5.15), which
        # this commit's config did not have: the reference writes the
        # ferry_params it wrote then (the live scorer leaves them out at None).
        from hermes.processes.config import FERRY_PARAMS_OMITTED_AT_NONE, FERRY_SPEC_FIELDS

        loaded["scorer"].FERRY_SPEC_FIELDS = {
            k: v for k, v in FERRY_SPEC_FIELDS.items() if k not in FERRY_PARAMS_OMITTED_AT_NONE}
        yield loaded["scorer"]
    finally:
        for name in names.values():
            sys.modules.pop(name, None)


def _same_but_blank(live, ref):
    """The live row is the reference row with the Phase 4 columns added, blank,
    after ``deadline_basis``; every other column is the reference's, exactly."""
    assert [c for c in live if c not in PHASE_4_COLUMNS] == list(ref)
    at = list(live).index("deadline_basis") + 1
    assert tuple(list(live)[at:at + len(PHASE_4_COLUMNS)]) == PHASE_4_COLUMNS
    assert {c: live[c] for c in PHASE_4_COLUMNS} == dict.fromkeys(PHASE_4_COLUMNS, "")
    assert {c: live[c] for c in ref} == ref


def _plain(value):
    """A canonical fixture value (``tests/golden/_canon.py``) as the JSON it was."""
    if isinstance(value, str) and value.startswith("f:"):
        return float(value[2:])
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k != "__type__"}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _p3_trace(root, name, case):
    """UG4's 6e6f92d trial ``name`` as the kept trace it was: every role's JSONL
    in its recorded order, and the per-role JSON. One trace root per trial,
    since the two cliff trials share a cell, arm and seed."""
    cell = case["inputs"]["cell"]
    d = root / name / trace_dir_name(Cell(cell_id=cell["cell_id"], arm=cell["arm"],
                                          trial_index=cell["trial_index"], seed=cell["seed"],
                                          params={}))
    d.mkdir(parents=True)
    for fname, cfg in case["configs"].items():
        (d / fname).write_text(json.dumps(_plain(cfg)), encoding="utf-8")
    named = {e: {m: list(v) for m, v in case[e].items()}
             for e in ("mule_ready", "mission_started", "mission_completed")}
    other = {m: list(v) for m, v in case["mule_events"]["other"].items()}
    for mule, seq in case["mule_events"]["names"].items():
        rows = [named[e][mule].pop(0) if e in named else other[mule].pop(0) for e in seq]
        (d / f"mule-{mule}.jsonl").write_text(
            "\n".join(json.dumps(_plain(r)) for r in rows) + "\n", encoding="utf-8")
    cluster_id = _plain(case["configs"]["cluster.json"])["cluster_id"]
    (d / f"cluster-{cluster_id}.jsonl").write_text(
        "\n".join(json.dumps(_plain(r)) for r in case["cluster_events"]) + "\n",
        encoding="utf-8")
    for dev, rows in case["device_events"].items():
        (d / f"device-{dev}.jsonl").write_text(
            "\n".join(json.dumps(_plain(r)) for r in rows) + "\n", encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def p3_traces(tmp_path_factory):
    fixture = json.loads((REPO / "tests/golden/data/p3_sim.json").read_text(encoding="utf-8"))
    root = tmp_path_factory.mktemp("p3sim")
    return {name: _p3_trace(root, name, case) for name, case in fixture["cases"].items()}


def test_the_phase_3_fixture_rebuilds_as_the_traces_it_recorded(p3_traces):
    """Each rebuilt trace holds its trial's missions on the simulated clock,
    and nothing Phase 4 writes: no plan fields in the mule config, no plan and
    no drop report in any mission (D1 and D3 left stops out then unreported)."""
    assert len(p3_traces) == 8
    for name, d in p3_traces.items():
        row = score_trial(d).to_row()
        assert row["mission_clock"] == "sim" and row["missions_completed"] >= 3, name
        assert "plan_mode" not in json.loads(next(d.glob("mule-*.json")).read_text())
        done = [e for e in _events(next(d.glob("mule-*.jsonl")))
                if e["event"] == "mission_completed"]
        assert not any({"plan", "plan_wall_s", "pass_1_policy_drops"} & set(e) for e in done)
        assert all(isinstance(e["pass_1_flown"], list) for e in done)


@pytest.mark.parametrize("name", [
    "h1_replan_trim_wide", "d1_max_aoi", "d3_whittle", "d4_route_only", "h1_narrow_cliff_empty",
    "h1_narrow_cliff_flown", "h1_medium_replan", "h1_narrow_measured"])
def test_a_phase_3_trace_scores_blank_and_as_6e6f92d_scores_it(p3_traces, ref_scorer, name):
    d = p3_traces[name]
    live = score_trial(d, taus=(0.82, 0.5)).to_row()
    ref = ref_scorer.score_trial(d, taus=(0.82, 0.5)).to_row()
    _same_but_blank(live, ref)
    # --age-cap-s scores the ages of any arm; nothing else appears.
    at_3 = score_trial(d, taus=(0.82, 0.5), age_cap_s=3).to_row()
    assert at_3["cap_s"] == 3 and isinstance(at_3["cap_violations"], int)
    assert {c: at_3[c] for c in PHASE_4_COLUMNS[3:]} == dict.fromkeys(PHASE_4_COLUMNS[3:], "")


@pytest.mark.parametrize("cell", ["C_traces", "C2_traces"])
def test_the_recorded_wall_clock_cells_score_blank_and_as_6e6f92d_scores_them(ref_scorer, cell):
    root = REPO / "results" / "exp4_matrix" / cell
    if not root.is_dir():
        pytest.skip(f"recorded traces {root} not present")
    live = score_traces(root, taus=(0.82, 0.75), include_failed=True)
    ref = ref_scorer.score_traces(root, taus=(0.82, 0.75), include_failed=True)
    assert len(live) == len(ref) > 0
    for a, b in zip(live, ref):
        _same_but_blank(a.to_row(), b.to_row())


# --------------------------------------------------------------------------- #
# 4. Provenance parity with the driver
# --------------------------------------------------------------------------- #

def _cell(arm, seed=4242, **params):
    p = {"N": 4, "rrf": 60.0, "n_missions": 2, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


#: The Phase 4 pilots' flags (decision 7), 1 MB, 60 s, S = 2, T_nom over 5
#: layouts: as U8's trials of the plan arms.
PILOT = dict(
    mission_clock="sim", realism=True, contact_band="wide", deadline_time_scale="t_nom",
    in_flight_response="replan", replan_fallback="trim", aggregation="agg:cutoff",
    contact_reliability_source="channel", payload_bytes=1_000_000, mission_budget_s=60.0,
    age_cap_missions=2, t_nom_layouts=5, trial_budget_s=240.0,
)
#: Nothing but plan mode needs T_nom here (the recorded time unit, the
#: mission backhaul, no Φ₀): the scorer must infer that the driver computed it.
PLAN_ONLY_T_NOM = dict(mission_clock="sim", realism=True, contact_band="wide",
                       payload_bytes=1_000_000, mission_budget_s=60.0, t_nom_layouts=3)
#: The Phase 3 cliff (trial T2): the one field-wide narrow stop, 1 MB, 60 s.
CLIFF = dict(mission_clock="sim", realism=True, contact_band="narrow", backhaul_model="seconds",
             payload_bytes=1_000_000, contact_reliability_source="origin",
             mission_budget_s=60.0, in_flight_response="abort", deadline_time_scale="t_nom",
             trial_budget_s=240.0, t_nom_layouts=3)

PARITY = [
    *[(f"{arm} (pilots' flags, S = 2)", PILOT, arm) for arm in PLAN_ARMS],
    ("F, T_nom needed by plan mode alone", PLAN_ONLY_T_NOM, "F"),
    ("FB+narrow under whole stops", dict(PILOT, member_admission="whole"), "FB+narrow"),
    ("F with score and search settings",
     dict(PILOT, plan_score_params={"c_cov_per_device": 0.25, "c_energy": 0.0},
          plan_search_params={"exact_max_devices": 4}, age_cap_lookahead=1), "F"),
    ("H1 under member subsets", dict(CLIFF, member_admission="subset"), "H1"),
    ("D1 under member subsets", dict(CLIFF, member_admission="subset"), "D1"),
    ("D5 under member subsets", dict(CLIFF, member_admission="subset"), "D5"),
    ("D4 (always whole)", dict(CLIFF, member_admission="subset"), "D4"),
    ("H1 at the defaults of a Phase 4 sim cell", dict(CLIFF), "H1"),
]


def _kept_config_trace(root, driver, cell):
    """The driver's row for ``cell`` (nothing spawned) and the configs a kept
    trace of it holds, as the real orchestrator writes them."""
    from tests.golden import _build_topology as T

    row, topo = T.run_stub_trial(driver, cell)
    d = root / trace_dir_name(cell)
    d.mkdir(parents=True)
    roles = T.role_configs(topo)
    (d / "cluster.json").write_text(json.dumps(roles["cluster"]), encoding="utf-8")
    for mid, cfg in roles["mules"].items():
        (d / f"mule-{mid}.json").write_text(json.dumps(cfg), encoding="utf-8")
    for did, cfg in roles["devices"].items():
        (d / f"device-{did}.json").write_text(json.dumps(cfg), encoding="utf-8")
    return row, d


@pytest.mark.parametrize("name, settings, arm", PARITY, ids=[p[0] for p in PARITY])
def test_the_provenance_is_the_drivers_row(tmp_path, name, settings, arm):
    seed = 777 if settings is CLIFF or settings.get("contact_band") == "narrow" else 4242
    params = dict(N=8, regime="clean") if settings.get("contact_band") == "narrow" else {}
    row, d = _kept_config_trace(tmp_path, Exp4Driver(**settings), _cell(arm, seed, **params))
    assert trial_provenance(d) == {c: row[c] for c in PROVENANCE_COLUMNS}
    params = json.loads(row["ferry_params"])
    if arm in PLAN_ARMS:
        assert params["plan_mode"] == "ferry" and params["t_nom_computed"] is True
        assert row["contact_band"] == ("search" if not arm.startswith("FB+") else arm[3:])
    elif settings.get("member_admission") == "subset" and arm != "D4":
        assert params["member_admission"] == "subset" and "plan_mode" not in params
    else:
        assert not {"plan_mode", "member_admission"} & set(params)


def test_every_plan_arm_label_names_its_kept_trace():
    """The kept trace's directory holds the label whole, and the scorer reads
    it back (``+`` and ``-`` are neither Windows-illegal nor the separator)."""
    for arm in PLAN_ARMS:
        cell = _cell(arm, 99, N=6)
        key = parse_trial_dir(trace_dir_name(cell))
        assert (key.arm, key.trial_index, key.seed) == (arm, 0, 99)


def test_a_given_t_nom_is_read_from_the_marker(tmp_path):
    """Given --t-nom-s, a plan arm's T_nom was not computed. The configs do
    not say so; the driver's marker does, and the scorer reads it first."""
    row, d = _kept_config_trace(tmp_path, Exp4Driver(**PLAN_ONLY_T_NOM, t_nom_s=180.0),
                                _cell("F"))
    assert json.loads(row["ferry_params"])["t_nom_computed"] is False
    assert json.loads(trial_provenance(d)["ferry_params"])["t_nom_computed"] is True
    (d / TRIAL_STATUS_FILE).write_text(json.dumps({"status": "ok", "t_nom_computed": False}))
    assert trial_provenance(d) == {c: row[c] for c in PROVENANCE_COLUMNS}


def _events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


REAL = [*[(arm, PILOT, {}) for arm in PLAN_ARMS],
        ("D1", dict(CLIFF, member_admission="subset"), dict(N=8, regime="clean")),
        ("H1", {"trial_budget_s": 240.0}, {})]


@pytest.mark.slow
@pytest.mark.parametrize("arm, settings, params", REAL, ids=[
    f"{a}-{'wall' if 'mission_clock' not in s else 'sim'}" for a, s, _ in REAL])
def test_a_real_trial_scores_its_provenance_and_its_plan(tmp_path, arm, settings, params):
    """Real processes over TCP (the stub model), the trace kept. The kept
    trace's provenance is the driver's row, column for column; the Phase 4
    columns are what its events say."""
    seed = 777 if arm == "D1" else 4242
    cell = _cell(arm, seed, **params)
    row = dict(Exp4Driver(**settings, trace_root=tmp_path).run_trial(cell))
    trace = tmp_path / trace_dir_name(cell)
    assert json.loads((trace / TRIAL_STATUS_FILE).read_text())["status"] == "ok"
    score = score_trial(trace)
    got = score.to_row()
    assert (score.key.arm, score.key.seed) == (arm, seed)
    assert trial_provenance(trace) == {c: row[c] for c in PROVENANCE_COLUMNS}
    assert {c: got[c] for c in PROVENANCE_COLUMNS} == {c: row[c] for c in PROVENANCE_COLUMNS}
    (log,) = list(trace.glob("mule-*.jsonl"))
    done = [e for e in _events(log) if e["event"] == "mission_completed"]
    new = {c: got[c] for c in PHASE_4_COLUMNS}
    if "mission_clock" not in settings:                       # the wall clock
        assert new == dict.fromkeys(PHASE_4_COLUMNS, "")
        return
    stops = [s["band"] for e in done for s in e["pass_1_flown"]]
    shares = json.loads(new["band_shares"]) if new["band_shares"] else {}
    assert shares == pytest.approx({b: stops.count(b) / len(stops) for b in set(stops)})
    # The far devices' merged updates (the ages' own count) over their missions.
    seeds = json.loads((trace / "cluster.json").read_text())["seed_devices"]
    far = [s["device_id"] for s in seeds if math.hypot(*s["position"][:2]) > 60.0]
    merged = score.ages.merged_updates
    assert new["far_served_share"] == (
        pytest.approx(sum(merged[d] for d in far) / (len(far) * len(done))) if far else "")
    if arm not in PLAN_ARMS:
        assert [new[c] for c in PHASE_4_COLUMNS[:4]] == ["", "", "", ""]
        assert new["plan_served_share_mean"] == new["plan_v_mean"] == ""
        dropped = sum(len(p["devices"]) for e in done for p in e.get("pass_1_policy_drops", ()))
        assert new["policy_drops"] == dropped > 0
        return
    assert new["policy_drops"] == ""
    plans = [e["plan"] for e in done]
    shares = [len(p["served"]) / len(p["demand"]) for p in plans if p["demand"]]
    assert new["plan_served_share_mean"] == pytest.approx(sum(shares) / len(shares))
    assert new["plan_v_mean"] == pytest.approx(sum(p["score"]["v"] for p in plans) / len(plans))
    if arm == "F-cap":
        assert [new[c] for c in PHASE_4_COLUMNS[:4]] == ["", "", "", ""]
        return
    assert new["cap_s"] == 2 and isinstance(new["cap_violations"], int)
    logged = [v["reason"] for p in plans for v in p["cap"]["violations"]]
    assert json.loads(new["cap_violation_events"]) == {r: logged.count(r) for r in CAP_REASONS}
    if arm.startswith("FB+"):
        assert set(json.loads(new["band_shares"])) == {arm[3:]}


# --------------------------------------------------------------------------- #
# 5. The S* tool
# --------------------------------------------------------------------------- #

#: The Phase 4 design probe's table: the design probe's S* per band policy,
#: payload, admission and budget, on its own layouts (``_u32(6, "t_nom", 1000 +
#: k)``, 30 of them), with 18,756 B for the measured θ. The rows' q90 is the
#: probe's own index, ``int(0.9 * (n - 1))``, over the layouts where every
#: device is servable; "none" counts the others.
PROBE_OUT = """
=== payload measured; N=6; 30 layouts; median S3a stops: wide 4, medium 2, narrow 1
whole  B=   30 | wide: med 3.0 q90 3 max 3 (none 28/30) | medium: med 1.0 q90 2 max 3 (none 21/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
whole  B=   45 | wide: med 3.0 q90 4 max 4 (none  8/30) | medium: med 2.0 q90 2 max 3 (none  4/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
whole  B=   60 | wide: med 2.0 q90 3 max 3 (none  0/30) | medium: med 1.0 q90 2 max 3 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
whole  B=   90 | wide: med 1.0 q90 2 max 2 (none  0/30) | medium: med 1.0 q90 1 max 2 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
whole  B=  120 | wide: med 1.0 q90 1 max 1 (none  0/30) | medium: med 1.0 q90 1 max 1 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
subset B=   30 | wide: med 3.0 q90 3 max 3 (none 28/30) | medium: med 1.0 q90 2 max 3 (none 21/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
subset B=   45 | wide: med 3.0 q90 4 max 4 (none  8/30) | medium: med 2.0 q90 2 max 3 (none  4/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
subset B=   60 | wide: med 2.0 q90 3 max 3 (none  0/30) | medium: med 1.0 q90 2 max 3 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
subset B=   90 | wide: med 1.0 q90 2 max 2 (none  0/30) | medium: med 1.0 q90 1 max 2 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
subset B=  120 | wide: med 1.0 q90 1 max 1 (none  0/30) | medium: med 1.0 q90 1 max 1 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)

=== payload 1 MB; N=6; 30 layouts; median S3a stops: wide 4, medium 2, narrow 1
whole  B=   30 | wide: med 3.0 q90 3 max 3 (none 29/30) | medium:    -   (none 30/30) | narrow:    -   (none 30/30) | search: med 3.0 q90 3 max 3 (none 29/30)
whole  B=   45 | wide: med 3.0 q90 4 max 4 (none 13/30) | medium: med 2.0 q90 2 max 3 (none 13/30) | narrow:    -   (none 30/30) | search: med 2.0 q90 3 max 4 (none  9/30)
whole  B=   60 | wide: med 2.0 q90 3 max 3 (none  0/30) | medium: med 2.0 q90 2 max 4 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none 29/30) | search: med 2.0 q90 2 max 3 (none  0/30)
whole  B=   90 | wide: med 2.0 q90 2 max 2 (none  0/30) | medium: med 1.0 q90 1 max 2 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  1/30) | search: med 1.0 q90 1 max 1 (none  0/30)
whole  B=  120 | wide: med 1.0 q90 1 max 1 (none  0/30) | medium: med 1.0 q90 1 max 2 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
subset B=   30 | wide: med 3.0 q90 3 max 3 (none 29/30) | medium: med 2.0 q90 3 max 3 (none 25/30) | narrow: med 3.5 q90 4 max 5 (none 16/30) | search: med 3.0 q90 4 max 5 (none 12/30)
subset B=   45 | wide: med 3.0 q90 4 max 4 (none 13/30) | medium: med 2.0 q90 3 max 4 (none  5/30) | narrow: med 3.0 q90 3 max 3 (none  0/30) | search: med 2.0 q90 3 max 3 (none  0/30)
subset B=   60 | wide: med 2.0 q90 3 max 3 (none  0/30) | medium: med 2.0 q90 2 max 4 (none  0/30) | narrow: med 2.0 q90 2 max 2 (none  0/30) | search: med 2.0 q90 2 max 2 (none  0/30)
subset B=   90 | wide: med 2.0 q90 2 max 2 (none  0/30) | medium: med 1.0 q90 1 max 2 (none  0/30) | narrow: med 1.0 q90 1 max 2 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
subset B=  120 | wide: med 1.0 q90 1 max 1 (none  0/30) | medium: med 1.0 q90 1 max 2 (none  0/30) | narrow: med 1.0 q90 1 max 1 (none  0/30) | search: med 1.0 q90 1 max 1 (none  0/30)
""".splitlines()        # the probe's first line is blank, as each section's is

PROBE_THETA = 18756
PROBE_BUDGETS = (30.0, 45.0, 60.0, 90.0, 120.0)
SPREAD = device_spread_m(60.0, field_radius_m=100.0)


@pytest.fixture(scope="module")
def probe_worlds():
    """The design probe's 30 layouts, priced by the tool, per payload."""
    out = {}
    for payload in (None, 1_000_000):
        drv = Exp4Driver(mission_clock="sim", realism=True, contact_band="wide",
                         payload_bytes=payload)
        _, synth = drv._payload_bytes(None)
        out[payload] = [
            S.planning_world(drv, layout, rf_range_m=60.0, regime="jittery",
                             theta_bytes=PROBE_THETA, synth_bytes=synth)
            for layout in S.reference_layouts(6, count=30, spread_m=SPREAD, tag="t_nom",
                                              offset=1000)
        ]
    return out


def _probe_cell(values):
    ok = [v for v in values if v is not None]
    none = sum(1 for v in values if v is None)
    if not ok:
        return f"   -   (none {none:2d}/{len(values)})"
    q90 = sorted(ok)[int(0.9 * (len(ok) - 1))]
    return (f"med {statistics.median(ok):3.1f} q90 {q90} max {max(ok)} "
            f"(none {none:2d}/{len(values)})")


def test_the_tool_reproduces_the_design_probe(probe_worlds):
    """Every line of the design probe's table, from the tool's per-layout S*
    (a layout with an unservable device is the probe's "none"). The probe
    priced S3a's fresh stops alone, before the hover rule (the user's
    decision of 2026-09-30), so the tool reproduces it with ``hover=False``."""
    lines = []
    for payload, worlds in probe_worlds.items():
        stops = {b: statistics.median(len(w.classes[b].stops) for w in worlds) for b in BANDS}
        lines += ["", f"=== payload {'measured' if payload is None else '1 MB'}; N=6; 30 "
                      f"layouts; median S3a stops: "
                      + ", ".join(f"{b} {stops[b]:.0f}" for b in BANDS)]
        for adm in ("whole", "subset"):
            for budget in PROBE_BUDGETS:
                cells = []
                for pol in (*BANDS, "search"):
                    classes = BANDS if pol == "search" else (pol,)
                    stars = [S.layout_s_star(w, classes=classes, budget_s=budget, admission=adm,
                                             hover=False)
                             for w in worlds]
                    cells.append(f"{pol}: " + _probe_cell(
                        [None if x.unservable else x.s_star for x in stars]))
                lines.append(f"{adm:6} B={budget:5.0f} | " + " | ".join(cells))
    assert lines == PROBE_OUT


_HOMES = {}


def _independent_homes(world, band, admission, budget):
    """Every device set one mission of ``band`` serves, with its home, by an
    enumeration of its own (the design probe's) on the stops the tool offers
    at ``budget`` (the hover rule's): the touched stops reduced by hand, every
    order folded with no gate, the earliest home kept."""
    key = (world, band, admission, budget)
    if key in _HOMES:
        return _HOMES[key]
    cls = world.classes[band]
    offered = world.stops(band, budget)
    units = ([frozenset(str(d) for d in wp.devices) for wp in offered] if admission == "whole"
             else [frozenset([str(d)]) for wp in offered for d in wp.devices])
    homes = {}
    for r in range(1, len(units) + 1):
        for combo in itertools.combinations(units, r):
            chosen = frozenset().union(*combo)
            stops = [dataclasses.replace(wp, devices=tuple(d for d in wp.devices
                                                           if str(d) in chosen))
                     for wp in offered if any(str(d) in chosen for d in wp.devices)]
            homes[chosen] = min(cls.model.fold(list(p), FlightState(cls.dock, 0.0),
                                               rule=RULE_NONE, budget_end=None, skip=False).home
                                for p in itertools.permutations(stops))
    _HOMES[key] = homes
    return homes


def _feasible_sets(world, classes, budget, admission):
    return {s for band in classes
            for s, home in _independent_homes(world, band, admission, budget).items()
            if home <= budget}


@pytest.mark.parametrize("admission", ["subset", "whole"])
@pytest.mark.parametrize("family", ["F", "FB+wide", "FB+narrow"])
def test_each_cover_is_a_schedule_and_none_is_shorter(probe_worlds, admission, family):
    for world in probe_worlds[1_000_000][:10]:
        classes = S.family_classes(family, world.class_names)
        for budget in (30.0, 45.0, 60.0):
            star = S.layout_s_star(world, classes=classes, budget_s=budget, admission=admission)
            feasible = _feasible_sets(world, classes, budget, admission)
            servable = frozenset().union(*feasible) if feasible else frozenset()
            assert set(star.servable) == servable
            assert set(star.unservable) == set(world.layout.devices) - servable
            assert star.exact and len(star.cover) == star.s_star
            assert all(frozenset(m) in feasible for m in star.cover)
            assert frozenset().union(*map(frozenset, star.cover)) == servable
            if not star.s_star:
                assert not servable
                continue
            shorter = itertools.combinations(sorted(feasible, key=sorted), star.s_star - 1)
            assert not any(frozenset().union(*c) == servable for c in shorter)


def test_the_unservable_devices_are_the_planners_unplannable(probe_worlds):
    """Under member subsets a device no class serves alone is what the plan
    labels ``unplannable``: U1's ``servable_alone`` from the dock at takeoff,
    on the stops the plan is offered (the hover rule's). Five devices, one on
    each of five layouts, are unservable, all on FB+wide at 30 s: 127-139 m
    out, they are over 30 s alone even from the edge of wide's 60 m reach."""
    from hermes.scheduler.stages.s3d_age_cap import servable_alone

    seen = 0
    for world in probe_worlds[1_000_000]:
        for budget in (30.0, 45.0):
            for family in ("F", "FB+wide", "FB+medium"):
                classes = S.family_classes(family, world.class_names)
                star = S.layout_s_star(world, classes=classes, budget_s=budget)
                cls0 = world.classes[classes[0]]
                planner = servable_alone(
                    [DeviceID(d) for d in world.layout.devices],
                    [(world.classes[c].model, world.stops(c, budget)) for c in classes],
                    start=FlightState(cls0.dock, 0.0), budget_end=budget)
                assert set(star.servable) == {str(d) for d in planner}
                seen += len(star.unservable)
    assert seen == 5


def _capped_plan(world, family, budget, admission):
    """A plan-mode scheduler on ``world``'s layout, priced by the tool's own
    class models, with every device capped (S = 1 at mission 1), planned at
    the dock at 0 s under ``budget``: its commit."""
    from hermes.scheduler import FLScheduler
    from hermes.scheduler.plan import AgeCapSpec, PlanClass, PlanOptions, PlanSetup
    from hermes.types import MissionSlice, MuleID

    fixed = None if family == S.FAMILY_SEARCH else family[len(S.FAMILY_FIXED_PREFIX):]
    classes = tuple(PlanClass(name=c.name, index=c.index, radius_m=c.radius_m, model=c.model,
                              outage=lambda d: 0.0) for c in world.classes.values())
    options = PlanOptions(band_class_policy="search" if fixed is None else f"fixed:{fixed}",
                          member_admission=admission, cap=AgeCapSpec(s_missions=1))
    setup = PlanSetup(options=options, classes=classes, reference=fixed or "wide",
                      t_ref_s=200.0, turnaround_s=30.0)
    sch = FLScheduler(now_fn=lambda: 0.0, mission_budget_s=budget, replan_fallback="trim",
                      member_admission=admission, plan_mode="ferry", plan=setup)
    sch.ingest_slice(MissionSlice(mule_id=MuleID("e"), device_ids=tuple(
        DeviceID(d) for d in world.layout.devices), issued_round=0, issued_at=0.0))
    for d, pos in world.layout.positions:
        sch.device_states[DeviceID(d)].last_known_position = pos
    sch.start_mission()
    sch.set_mission_round(1)
    sch.build_ferry_plan()
    return sch.last_plan


@pytest.mark.parametrize("budget", [30.0, 45.0])
def test_the_tools_servable_set_is_the_plans_on_its_own_layouts(budget):
    """The final check's E2E2-01, on the user's decision of 2026-09-30: the
    tool and the plan price one stop family, so on the tool's own 30 layouts
    (the pilots' cell, 1 MB), in every family, the devices the tool can cover
    are exactly those a plan-mode scheduler with every device capped serves
    or labels ``crowded``, and its unservable devices are the plan's
    ``unplannable``: three at 30 s, on FB+wide, none at 45 s. Under ``whole``
    the plan's labels judge the device alone, so they are the same, and the
    tool's unservable devices hold them (it also leaves out a device whose
    whole stop no mission flies)."""
    drv = Exp4Driver(**MB_DRIVER)
    unservable = 0
    for layout in S.reference_layouts(6, count=30, spread_m=SPREAD):
        world = _world(drv, layout)
        for family in ("F", "FB+wide", "FB+medium", "FB+narrow"):
            classes = S.family_classes(family, world.class_names)
            labels = {}
            for admission in ("subset", "whole"):
                star = S.layout_s_star(world, classes=classes, budget_s=budget,
                                       admission=admission)
                commit = _capped_plan(world, family, budget, admission)
                assert commit.capped == frozenset(map(DeviceID, layout.devices))
                unplannable = {str(v.device) for v in commit.violations
                               if v.reason == CAP_UNPLANNABLE}
                labels[admission] = unplannable
                if admission == "subset":
                    assert set(star.unservable) == unplannable, (layout.index, family)
                    assert set(star.servable) == set(layout.devices) - unplannable
                    unservable += len(unplannable)
                else:
                    assert unplannable <= set(star.unservable), (layout.index, family)
            assert labels["whole"] == labels["subset"]
    assert unservable == (3 if budget == 30.0 else 0)


@pytest.mark.parametrize("admission", ["subset", "whole"])
def test_the_greedy_bound_is_a_cover_no_shorter_than_the_exact_one(probe_worlds, admission):
    looser = 0
    for world in probe_worlds[1_000_000]:
        for family in ("F", "FB+wide", "FB+medium", "FB+narrow"):
            classes = S.family_classes(family, world.class_names)
            for budget in (45.0, 60.0, 90.0):
                exact = S.layout_s_star(world, classes=classes, budget_s=budget,
                                        admission=admission)
                greedy = S.layout_s_star(world, classes=classes, budget_s=budget,
                                         admission=admission, exact_max_devices=0)
                assert not greedy.exact and greedy.s_star >= exact.s_star
                assert (greedy.servable, greedy.unservable) == (exact.servable, exact.unservable)
                assert frozenset().union(*map(frozenset, greedy.cover)) == set(exact.servable)
                for mission in greedy.cover:
                    assert any(world.homes(c, admission, budget).get(frozenset(mission), math.inf)
                               <= budget for c in classes)
                looser += greedy.s_star > exact.s_star
    # A bound, and a close one here: above 10 % of cases it would say little
    # (a Phase 4 build probe: 0 to 12 % per family at 30-90 s).
    assert looser <= 30 * 4 * 3 // 10


def test_a_larger_cell_runs_the_greedy_bound():
    report = S.s_star_report(Exp4Driver(mission_clock="sim", realism=True, contact_band="wide",
                                        payload_bytes=1_000_000),
                             n_devices=12, budgets=(90.0,), layouts=2, families=("F",))
    for star in report.stars["F"][90.0]:
        assert not star.exact and star.s_star == len(star.cover) >= 1
        assert frozenset().union(*map(frozenset, star.cover)) == set(star.servable)
        assert set(star.servable) | set(star.unservable) == {S.device_name(i) for i in range(12)}


#: The pilots' cell's driver (1 MB declared, so θ and the batch price nothing).
MB_DRIVER = dict(mission_clock="sim", realism=True, contact_band="wide", payload_bytes=1_000_000)


def _world(driver, layout):
    theta, synth = driver._payload_bytes(None)
    return S.planning_world(driver, layout, rf_range_m=60.0, regime="jittery",
                            theta_bytes=theta, synth_bytes=synth)


def test_the_greedy_walk_goes_on_past_a_stop_that_admits_nobody():
    """Three devices, each its own wide stop: x and z 70 m either side of the
    dock, y 150 m out between them, so the 2-OPT tour flies y in the middle.
    At a budget where {x, z} and y alone fit but y cannot join x or z, the
    walk drops y's stop where it reaches it and goes on to the last stop: 2
    missions, the exact S*. Ending the walk there (a candidate plan's
    ``require_all``) would serve one device per mission and need 3."""
    from hermes.scheduler.routing.two_opt import order_contacts

    x, y, z = (S.device_name(i) for i in range(3))
    world = _world(Exp4Driver(**MB_DRIVER), S.Layout(index=0, seed=12345, positions=(
        (x, (70.0, 0.0, 0.0)), (y, (0.0, 150.0, 0.0)), (z, (-70.0, 0.0, 0.0)))))
    wide = world.classes["wide"]
    assert sorted(tuple(map(str, wp.devices)) for wp in wide.stops) == [(x,), (y,), (z,)]
    tour = order_contacts(list(wide.stops), wide.dock, end=wide.dock)
    assert tuple(map(str, tour[1].devices)) == (y,)
    homes = world.homes("wide", "subset")
    fits = max(homes[frozenset([y])], homes[frozenset([x, z])])
    joins = min(homes[frozenset([x, y])], homes[frozenset([y, z])])
    assert fits < joins
    budget = (fits + joins) / 2
    exact = S.layout_s_star(world, classes=("wide",), budget_s=budget)
    greedy = S.layout_s_star(world, classes=("wide",), budget_s=budget, exact_max_devices=0)
    assert exact.cover == greedy.cover == ((x, z), (y,))
    assert not greedy.exact and greedy.s_star == exact.s_star == 2


def test_the_tools_own_layouts_do_not_depend_on_the_grid():
    layouts = S.reference_layouts(6, count=3, spread_m=SPREAD)
    for k, layout in enumerate(layouts):
        seed = _u32(6, "s_star", k)
        assert (layout.index, layout.seed) == (k, seed)
        assert [p[:2] for _, p in layout.positions] == device_positions(6, seed, SPREAD)
        assert layout.devices == tuple(f"exp4-dev-{i:03d}" for i in range(6))
        assert seed not in {_u32(6, "t_nom", j) for j in range(40)}


def test_without_realism_the_layouts_are_the_tight_cluster():
    """A trial's spread: the realism field when the driver has realism, else
    the tight EX-4.0 cluster, min(0.4 rf_range_m, 25 m). Each layout of the
    report is priced on the cluster's layout; the field's give other S*."""
    drv = Exp4Driver(**dict(MB_DRIVER, realism=False))
    report = S.s_star_report(drv, n_devices=6, budgets=(45.0,), layouts=3, families=("FB+wide",))

    def stars(spread):
        return tuple(S.layout_s_star(_world(drv, layout), classes=("wide",), budget_s=45.0)
                     for layout in S.reference_layouts(6, count=3, spread_m=spread))

    assert device_spread_m(60.0) == 24.0 != SPREAD
    assert report.stars["FB+wide"][45.0] == stars(device_spread_m(60.0))
    assert report.stars["FB+wide"][45.0] != stars(SPREAD)


def test_an_energy_capacity_leaves_the_devices_it_refuses_unservable(probe_worlds):
    """At 2,000 J the energy clause refuses some devices even alone at the
    stops the plan is offered, at any budget. (At 3,000 J it refuses none: a
    device its S3a stop cannot serve alone, for the energy as for the time,
    is offered its hover stop, nearer the dock, which flies less.) The tool
    reports exactly the planner's servable devices (U1's ``servable_alone``
    prices the energy clause too), on the exact and the greedy path, and
    covers only them; the same layouts without the capacity (the probe's
    worlds) leave none unservable."""
    from hermes.scheduler.stages.s3d_age_cap import servable_alone

    drv = Exp4Driver(**MB_DRIVER, ferry_physics={"energy_capacity_j": 2000.0})
    _, synth = drv._payload_bytes(None)
    refused = {}
    for k, layout in enumerate(S.reference_layouts(6, count=3, spread_m=SPREAD, tag="t_nom",
                                                   offset=1000)):
        world = S.planning_world(drv, layout, rf_range_m=60.0, regime="jittery",
                                 theta_bytes=PROBE_THETA, synth_bytes=synth)
        for budget in (90.0, 1e9):
            planner = {str(d) for d in servable_alone(
                [DeviceID(d) for d in layout.devices],
                [(world.classes[c].model, world.stops(c, budget)) for c in BANDS],
                start=FlightState(world.classes["wide"].dock, 0.0), budget_end=budget)}
            for exact_max in (S.EXACT_MAX_DEVICES, 0):
                star = S.layout_s_star(world, classes=BANDS, budget_s=budget,
                                       exact_max_devices=exact_max)
                assert set(star.servable) == planner, (k, budget, exact_max)
                assert set().union(*map(set, star.cover)) == planner
                assert star.s_star == len(star.cover)
            refused.setdefault(budget, []).append(len(star.unservable))
            free = S.layout_s_star(probe_worlds[1_000_000][k], classes=BANDS, budget_s=budget)
            assert free.unservable == ()
    # The capacity refuses some devices, not all, and the same at both
    # budgets: the energy clause, not the time.
    assert 0 < sum(refused[90.0]) < 18 and refused[90.0] == refused[1e9]


def test_a_seconds_backhaul_prices_every_mission_as_the_mission_model():
    """The seconds-axis spec needs a backhaul period, which planning never
    reads (the tool gives a placeholder, as T_nom's layouts do), and the
    fixed carrier's upload is priced at the mission model's SNR: every home
    is the mission model's."""
    worlds = {
        model: [_world(Exp4Driver(**MB_DRIVER, backhaul_model=model), layout)
                for layout in S.reference_layouts(6, count=2, spread_m=SPREAD, tag="t_nom",
                                                  offset=1000)]
        for model in ("mission", "seconds")
    }
    for mission, seconds in zip(worlds["mission"], worlds["seconds"]):
        for band, admission in itertools.product(BANDS, ("subset", "whole")):
            assert seconds.homes(band, admission) == mission.homes(band, admission)
            assert seconds.stops(band, 45.0) == mission.stops(band, 45.0)
            assert seconds.homes(band, admission, 45.0) == mission.homes(band, admission, 45.0)


@pytest.mark.parametrize("budget", [math.inf, 0.0, -5.0, math.nan])
def test_a_budget_is_a_finite_number_of_seconds_above_zero(probe_worlds, budget):
    """Infinite, a mission an energy capacity refuses (priced inf) would pass
    it; 0, below or NaN, nothing passes, and S would come from the floor."""
    with pytest.raises(ValueError, match=r"budget_s must be a finite number of seconds > 0"):
        S.layout_s_star(probe_worlds[1_000_000][0], classes=BANDS, budget_s=budget)


def test_decision_1_covers_90_percent_of_layouts_at_every_budget_and_at_least_two():
    assert S.cover_value([1] * 27 + [5] * 3) == 1          # the 27th of 30
    assert S.cover_value([1] * 26 + [5] * 4) == 5
    assert S.cover_value([3] * 17 + [4] * 3) == 4          # the 18th of 20
    assert S.cover_value(list(range(1, 30))) == 27         # 26.1 of 29 rounds up
    assert S.cover_value([2]) == 2
    assert S.decision_1_s({90.0: [1] * 30, 45.0: [1] * 30}) == 2          # never below 2
    assert S.decision_1_s({90.0: [1] * 30, 45.0: [3] * 27 + [9] * 3}) == 3
    assert S.decision_1_s({90.0: [4] * 30, 45.0: [3] * 30}) == 4
    with pytest.raises(ValueError):
        S.cover_value([])


def test_the_families_fly_every_class_or_the_pinned_one():
    assert S.family_classes("F", BANDS) == BANDS
    assert S.family_classes("FB+medium", BANDS) == ("medium",)
    for bad in ("FB+ultra", "FX", "H1"):
        with pytest.raises(ValueError):
            S.family_classes(bad, BANDS)


def test_the_report_and_the_cli(tmp_path, capsys):
    out = tmp_path / "s_star.json"
    assert S.main(["--budgets", "90", "45", "--payload-bytes", "1000000", "--layouts", "4",
                   "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["families"]) == {"F", "FB+wide", "FB+medium", "FB+narrow"}
    s = report["families"]["F"]["s"]
    assert s >= 2 and report["families"]["F"]["s_plus_1"] == s + 1
    assert f"S = {s} (i), S + 1 = {s + 1} (ii)" in printed
    assert len(report["families"]["F"]["budgets"]["45.0"]) == 4
    with pytest.raises(SystemExit):
        S.main(["--budgets", "60", "--contact-band", "ultra"])
    with pytest.raises(ValueError, match=r"^S\* needs band classes"):
        S.s_star_report(Exp4Driver(mission_clock="sim"), n_devices=6, budgets=(60.0,),
                        layouts=1)
    with pytest.raises(ValueError, match="mission_clock"):
        S.s_star_report(Exp4Driver(), n_devices=6, budgets=(60.0,), layouts=1)


def test_the_report_marks_each_s_plus_1_it_cannot_vouch_for():
    """(ii) prints bare only where the module docstring measured S + 1 free of
    violations: F under subset admission with every budget within 45 to 90 s
    (on the layouts S covers). A pinned class can crowd at the stress budget,
    every family crowds at 30 s (F 19 times in 360 missions at its S + 1 = 4
    with 90, 60, 45 and 30 s, in the hover fix's review probe),
    nothing beyond 90 s was measured, and under whole admission the cover
    leaves out devices the plan reports every mission: each family's line,
    and the cell's, names its caveats (the review of the hover fix, finding
    3: F's line and the cell's printed a bare (ii) with 30 s among the
    budgets)."""
    star = S.LayoutStar(layout=0, s_star=3, exact=True, servable=("a", "b", "c"),
                        unservable=(), cover=(("a",), ("b",), ("c",)))

    def printed(budgets, admission="subset"):
        families = ("F", "FB+wide")
        return S.format_report(S.SStarReport(
            budgets=budgets, families=families, admission=admission,
            stars={f: {b: (star,) for b in budgets} for f in families})).splitlines()

    def label(lines, family):
        (line,) = [x for x in lines if x.startswith(f"{family} ") and "(i)" in x]
        return line.split("S + 1 = 4 ", 1)[1]

    pilots = printed((90.0, 45.0))
    assert label(pilots, "F") == "(ii)"
    assert label(pilots, "FB+wide") == "(ii, no guarantee for a pinned class)"
    assert pilots[-1].endswith("S = 3 (i), S + 1 = 4 (ii). Admission: subset.")
    wide = printed((90.0, 60.0, 45.0, 30.0))
    assert label(wide, "F") == "(ii, no guarantee at 30 s)"
    assert label(wide, "FB+wide") == "(ii, no guarantee for a pinned class or at 30 s)"
    assert wide[-1].endswith("S + 1 = 4 (ii, no guarantee at 30 s). Admission: subset.")
    whole = printed((120.0, 45.0), admission="whole")
    assert label(whole, "F") == "(ii, no guarantee at 120 s or under whole admission)"
    assert whole[-1].endswith(
        "S + 1 = 4 (ii, no guarantee at 120 s or under whole admission). Admission: whole.")
    # The measured range is closed at both ends.
    assert S.S_PLUS_1_MEASURED_S == (45.0, 90.0)
    assert S.s_plus_1_caveats("F", (45.0, 60.0, 90.0), "subset") == ()
    assert S.s_plus_1_caveats("F", (44.9, 90.1), "subset") == ("at 44.9, 90.1 s",)


def _recorded_cli(monkeypatch):
    """Record what the S* CLI passes to its driver and its report.

    The driver is still the real one, built from the keywords the CLI gives,
    so its own checks run. The report's keywords are bound to the real
    ``s_star_report``'s signature (a keyword it lacks fails here), and it
    returns S* = 3 on one layout per family and budget.
    """
    made, asked = [], []
    real = S.s_star_report

    def driver(**kw):
        made.append(kw)
        return Exp4Driver(**kw)

    def report(drv, **kw):
        inspect.signature(real).bind(drv, **kw)
        asked.append((drv, kw))
        star = S.LayoutStar(layout=0, s_star=3, exact=True, servable=("a", "b", "c"),
                            unservable=(), cover=(("a",), ("b",), ("c",)))
        budgets = tuple(float(b) for b in kw["budgets"])
        families = tuple(kw["families"] or (S.FAMILY_SEARCH,))
        return S.SStarReport(budgets=budgets, families=families, admission=kw["admission"],
                             stars={f: {b: (star,) for b in budgets} for f in families})

    monkeypatch.setattr(S, "Exp4Driver", driver)
    monkeypatch.setattr(S, "s_star_report", report)
    return made, asked


def test_every_flag_of_the_s_star_cli_reaches_the_driver_or_the_report(
        tmp_path, monkeypatch, capsys):
    """The pilots set S with this CLI: each flag, at a value other than its
    default, must reach the driver (the cell's settings) or the report."""
    made, asked = _recorded_cli(monkeypatch)
    out = tmp_path / "s_star.json"
    assert S.main([
        "--budgets", "75", "35", "--N", "5", "--rrf", "70", "--regime", "clean",
        "--layouts", "3", "--contact-band", "medium", "--contact-band-classes", "wide", "medium",
        "--backhaul-model", "seconds", "--payload-bytes", "500000", "--theta-bytes", "18756",
        "--ferry-physics", '{"energy_capacity_j": 3000.0}', "--no-realism",
        "--member-admission", "whole", "--families", "F", "FB+medium",
        "--exact-max-devices", "4", "--layout-tag", "t_nom", "--layout-offset", "1000",
        "--json", str(out),
    ]) == 0
    assert made == [dict(mission_clock="sim", realism=False, contact_band="medium",
                         contact_band_classes=["wide", "medium"], backhaul_model="seconds",
                         payload_bytes=500000, ferry_physics={"energy_capacity_j": 3000.0})]
    ((drv, kw),) = asked
    assert isinstance(drv, Exp4Driver) and not drv.realism
    assert kw == dict(n_devices=5, budgets=[75.0, 35.0], rf_range_m=70.0, regime="clean",
                      families=["F", "FB+medium"], admission="whole", layouts=3,
                      theta_bytes=18756, exact_max_devices=4, layout_tag="t_nom",
                      layout_offset=1000)
    assert ("S = 3 (i), S + 1 = 4 (ii, no guarantee at 35 s or under whole admission)"
            in capsys.readouterr().out)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert (written["admission"], written["families"]["FB+medium"]["s_plus_1"]) == ("whole", 4)


def test_the_s_star_cli_defaults_and_its_usage_errors(monkeypatch):
    made, asked = _recorded_cli(monkeypatch)
    assert S.main(["--budgets", "60"]) == 0
    assert made == [dict(mission_clock="sim", realism=True, contact_band="wide",
                         contact_band_classes=None, backhaul_model="mission",
                         payload_bytes=None, ferry_physics={})]
    assert asked[0][1] == dict(n_devices=6, budgets=[60.0], rf_range_m=60.0, regime="jittery",
                               families=None, admission="subset", layouts=30, theta_bytes=None,
                               exact_max_devices=6, layout_tag="s_star", layout_offset=0)
    for bad in (["--ferry-physics", "{not json"], ["--member-admission", "some"],
                ["--backhaul-model", "hourly"], ["--regime", "stormy"]):
        with pytest.raises(SystemExit):
            S.main(["--budgets", "60", *bad])
    assert len(made) == 1


def test_the_tool_agrees_with_u7s_s_star_on_the_deterministic_spec(probe_worlds):
    """Unit U7's own S* (``test_p4_plan_missions.s_star``, critic B4's spec: the
    link's sigma kept, every noise term 0) on the same 30 layouts at the
    budgets its loopback flies: the S* tool gives the same number on each,
    for F and for each pinned class. U7 places the hover rule's stops by a
    brute force of its own (``segment_hover``), the tool by ``plan/hover.py``."""
    sys.path.insert(0, str(REPO))
    from tests.integration import test_p4_plan_missions as U7

    for k, world in enumerate(probe_worlds[1_000_000]):
        seed, layout = U7.ref_layout(k)
        assert seed == world.layout.seed
        spec = U7.det_spec(seed, layout)
        for budget in (45.0, 60.0):
            for bands in (BANDS, ("medium",)):
                theirs = U7.s_star(spec, layout, budget, bands=bands)
                ours = S.layout_s_star(world, classes=bands, budget_s=budget)
                assert (ours.s_star if not ours.unservable else None) == theirs, (k, budget)


# --------------------------------------------------------------------------- #
# 6. Imports
# --------------------------------------------------------------------------- #

def test_scoring_loads_no_plan_module(tmp_path, p3_traces):
    """The scorer and the consumer are on the recorded path: scoring a trace,
    Phase 3's or Phase 4's, loads neither the plan package nor the cap stage."""
    plan_trace = _write_trace(tmp_path, TRIAL)
    code = (
        "import sys\n"
        "from experiments.analysis.traces_scorer import score_trial\n"
        "for d in sys.argv[1:]:\n"
        "    score_trial(d, age_cap_s=2).to_row()\n"
        "bad = [m for m in sys.modules if m.startswith('hermes.scheduler.plan')\n"
        "       or m.endswith(('s3d_age_cap', 'cross_heuristic'))]\n"
        "print(bad)\n"
    )
    done = subprocess.run([sys.executable, "-c", code, str(p3_traces["d1_max_aoi"]),
                           str(plan_trace)], cwd=REPO, capture_output=True, text=True,
                          timeout=120, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_the_simulated_stamps_of_the_hand_built_trial_stay_below_the_ceiling():
    assert E + 200.0 * len(MISSIONS) < SIM_CEILING_S < WALL
