"""FeRRy Phase 3, unit U8 — analysis on simulated time.

* **The clock domain** is read from ``mule_ready.mission_clock`` (absent is
  the wall clock, as on every recorded trace), and a trace whose clocks
  disagree is refused with a clear error rather than scored: a simulated
  ``contact_ts`` held to a wall-clock deadline would read as 0 % misses, and
  the reverse as 100 %.
* **Simulated time where time matters**: the missions (and with them the ages
  and Network AoU) complete in the order of their simulated ends; time to τ
  gains simulated seconds beside the wall ones; deadline misses compare
  simulated contacts with simulated deadlines.
* **The mission-clock columns** beside the wall ones (critic B13): the
  simulated mission duration, the clock's ledger, the SIMULATED energy, the
  budget overrun and its rate, the re-plans, aborts and inserts; blank on the
  wall clock, where ``mission_duration_s_mean`` stays wall time.
* **Provenance**: the driver's Phase 3 columns, derived from a trace's own
  configs, equal the driver's row on the same trial across a matrix of driver
  settings (the per-role JSON is the real orchestrator's).

Legacy identity (every recorded trace scores as before) is pinned by
``test_traces_scorer_recorded.py`` and ``test_multi_mule_scoring.py`` on the
kept L1 cell; the unit's report re-scores all 600 kept trials.
"""

from __future__ import annotations

import csv
import json
import re
from types import SimpleNamespace

import pytest

from experiments.analysis.traces_scorer import (
    _ordered,
    age_profile,
    deadline_misses,
    score_trial,
    tau_reach,
    trial_provenance,
    write_scores_csv,
)
from experiments.exp4.driver import (
    PROVENANCE_COLUMNS,
    TRIAL_STATUS_FILE,
    Exp4Driver,
    trace_dir_name,
)
from experiments.exp4.events_consumer import (
    ClockDomainError,
    Exp4Observation,
    MissionRecord,
    completion_order,
    observation_from_rows,
    trace_clock_domain,
)
from experiments.exp4.metrics import (
    SIM_COLUMNS,
    SIM_LEDGER_COLUMNS,
    Exp4MetricSummary,
    summarise_flat_fl,
    summarise_observation,
)
from experiments.runner import Cell
from hermes.l1.mission_clock import LEDGER_KINDS, SIM_CEILING_S, SIM_EPOCH_S
from hermes.processes.mule import _sim_mission_fields

from tests.golden import _build_topology as T

WALL = 1_700_000_000.0          # a wall-clock envelope time
E = SIM_EPOCH_S                 # where every mission clock starts
DEVICES = ["a", "b", "c"]


# --------------------------------------------------------------------------- #
# A three-mission trial on the mission clock, one mule
# --------------------------------------------------------------------------- #
#
#   mission  wall (s)  sim (s after E)  CLEAN  upload (sim s)   evaluation
#   1        0–6       0–160            a, b   ingested @70     r1 acc 0.70 @ wall 3.5
#   2        6–12      160–330          c      ingested @230    r2 acc 0.85 @ wall 9.5
#   3        12–18     330–500          a      LOST @400
#
# Pass-1 deadlines and contacts (sim s after E): mission 1 holds a, b, c to
# 60; a is CLEAN at 48 (on time), b at 75 (late), c times out: 2 misses.
# Missions 2 and 3 are on time. Admitted 5, missed 2.
# Ages after each mission (a, b, c): (0, 0, 1), (1, 1, 0), (2, 2, 1)
# -> Network AoU 1/3, 2/3, 5/3; mean 8/9, final 5/3.
# In-flight responses and beacon inserts, per mission: re-plans 2, 0, 1;
# aborts 0, 2 (one per pass), 0; inserts 2, 0, 2. The trial's totals, 3, 2
# and 4, are not its counts of missions with any (2, 1 and 2).

def _ledger(transit, dwell, listen, ret, upload, turnaround, dock_wait):
    return dict(zip(LEDGER_KINDS, (transit, dwell, listen, ret, upload, turnaround, dock_wait)))


SIM_MISSIONS = [
    dict(rnd=1, wall=(0.0, 6.0), sim=(0.0, 160.0), clean=["a", "b"],
         ledger=_ledger(90.0, 0.5, 2.0, 36.0, 0.25, 30.0, 1.25), energy_j=19_000.0,
         budget_overrun_s=0.0,
         replans=[{"t_s": E + 20.0, "pass": "collect"}, {"t_s": E + 35.0, "pass": "collect"}],
         aborts=[], inserts=[{"t_s": E + 10.0}, {"t_s": E + 30.0}],
         plan=[(["a", "b"], {"a": 60.0, "b": 60.0}), (["c"], {"c": 60.0})],
         outcomes=[("a", "clean", 48.0), ("b", "clean", 75.0), ("c", "timeout", 80.0)]),
    dict(rnd=2, wall=(6.0, 12.0), sim=(160.0, 330.0), clean=["c"],
         ledger=_ledger(90.0, 0.5, 1.0, 36.0, 0.25, 30.0, 12.25), energy_j=19_500.0,
         budget_overrun_s=12.5, replans=[],
         aborts=[{"t_s": E + 200.0, "pass": "collect"}, {"t_s": E + 300.0, "pass": "deliver"}],
         inserts=[],
         plan=[(["c"], {"c": 300.0})], outcomes=[("c", "clean", 250.0)]),
    dict(rnd=3, wall=(12.0, 18.0), sim=(330.0, 500.0), clean=["a"],
         ledger=_ledger(95.0, 0.5, 2.0, 36.5, 0.0, 30.0, 6.0), energy_j=20_000.0,
         budget_overrun_s=0.0, replans=[{"t_s": E + 350.0, "pass": "collect"}], aborts=[],
         inserts=[{"t_s": E + 340.0}, {"t_s": E + 345.0}],
         plan=[(["a"], {"a": 400.0})], outcomes=[("a", "clean", 390.0)]),
]


def _mule_rows(missions=SIM_MISSIONS, *, mule="m1", clock="sim"):
    """A mule's stream as ``hermes.processes.mule`` writes it; the simulated
    record through the process's own ``_sim_mission_fields``. On the wall
    clock the same missions, with no simulated fields and every Pass-1 stamp
    in wall time."""
    sim = clock == "sim"
    base = E if sim else WALL
    ready = {"ts": WALL - 5.0, "event": "mule_ready", "role": "mule", "id": mule}
    if sim:
        ready.update(mission_clock="sim", clock_epoch_s=E)
    rows = [ready, {"ts": WALL - 1.0, "event": "dock_bootstrapped", "role": "mule", "id": mule}]
    for i, m in enumerate(missions):
        start, end = WALL + m["wall"][0], WALL + m["wall"][1]
        started = {"ts": start, "event": "mission_started", "role": "mule", "id": mule,
                   "mission_index": i}
        if sim:
            started["sim_start_s"] = E + m["sim"][0]
        rows.append(started)
        if not m["clean"]:
            rows.append({"ts": end, "event": "mission_empty", "role": "mule", "id": mule,
                         "mission_round": m["rnd"]})
        row = {
            "ts": end, "event": "mission_completed", "role": "mule", "id": mule,
            "mission_round": m["rnd"], "pass_1_contacts": 1, "pass_2_contacts": 1,
            "duration_s": end - start, "pass_1_updates": len(m["clean"]) or None,
            "pass_1_scheduled": 3, "pass_1_clean_devices": m["clean"] or None,
            "pass_1_plan": [
                {"devices": list(devs), "deadline_ts": base + min(own.values()),
                 "device_deadlines": {d: base + t for d, t in own.items()}}
                for devs, own in m.get("plan", ())
            ],
            "pass_1_outcomes": [
                {"device": d, "outcome": o, "contact_ts": base + t, "basis_version": None,
                 "age": None}
                for d, o, t in m.get("outcomes", ())
            ],
        }
        if sim:
            row.update(_sim_mission_fields(SimpleNamespace(
                sim_start_s=E + m["sim"][0], sim_end_s=E + m["sim"][1], sim_ledger=m["ledger"],
                sim_pass_2_start_s=None, pass_1_flown=[], pass_2_flown=[],
                replans=m["replans"], aborts=m["aborts"], inserts=m["inserts"],
                offers_refused=[], budget_overrun_s=m["budget_overrun_s"],
                pass_2_budget_overrun_s=None, energy_j=m["energy_j"], band="wide",
                backhaul=None,
            )))
        rows.append(row)
    return rows


def _cluster_rows(*, clock="sim", mule="m1"):
    """The cluster's stream; on the simulated clock its events carry the
    simulated fields ``hermes.processes.cluster`` adds."""
    sim = clock == "sim"

    def ev(t, event, sim_fields=None, **kw):
        row = {"ts": WALL + t, "event": event, "role": "cluster", "id": "c1", **kw}
        if sim and sim_fields is not None:
            row.update(sim_fields)
        return row

    def up(sim_ts):
        return {"sim_upload_ts": sim_ts, "carrier": 2, "snr_db": 9.0, "p_loss": 0.05}

    return [
        ev(-6.0, "cluster_ready", {"mission_clock": "sim", "backhaul_model": "seconds"}),
        ev(-5.5, "model_eval", {"sim_ts": None}, cluster_round=0, accuracy=0.36, auc=0.3,
           loss=0.7, n_test=100),
        ev(3.0, "up_bundle_ingested", up(E + 70.0), mule_id=mule, mission_round=1),
        ev(3.0, "cluster_round_closed", {"sim_ts": E + 70.0}, cluster_round=1),
        ev(3.5, "model_eval", {"sim_ts": E + 70.0}, cluster_round=1, accuracy=0.70, auc=0.7,
           loss=0.6, n_test=100),
        ev(9.0, "up_bundle_ingested", up(E + 230.0), mule_id=mule, mission_round=2),
        ev(9.0, "cluster_round_closed", {"sim_ts": E + 230.0}, cluster_round=2),
        ev(9.5, "model_eval", {"sim_ts": E + 230.0}, cluster_round=2, accuracy=0.85, auc=0.9,
           loss=0.4, n_test=100),
        ev(15.0, "backhaul_upload_lost", up(E + 400.0), mule_id=mule, mission_round=3),
    ]


def _device_rows():
    return [{"ts": WALL - 5.0, "event": "device_ready", "role": "device", "id": d} for d in DEVICES]


def _obs(clock="sim", *, mule_rows=None, cluster_rows=None):
    return observation_from_rows(
        cluster_rows=_cluster_rows(clock=clock) if cluster_rows is None else cluster_rows,
        mule_rows=_mule_rows(clock=clock) if mule_rows is None else mule_rows,
        device_rows=_device_rows(), n_devices=len(DEVICES),
    )


def _summary(obs):
    return summarise_observation(obs, n_devices=3, rf_range_m=60.0, n_missions_target=3)


# --------------------------------------------------------------------------- #
# 1. The clock domain
# --------------------------------------------------------------------------- #

def test_the_clock_is_read_from_mule_ready_and_absent_means_wall():
    assert _obs("sim").mission_clock == "sim"
    assert _obs("wall").mission_clock == "wall"
    assert trace_clock_domain([], []) == "wall"
    # Without a mule_ready (a mule that died before announcing itself) the
    # cluster's announcement stands.
    rows = [r for r in _mule_rows() if r["event"] != "mule_ready"]
    assert trace_clock_domain(_cluster_rows(), rows) == "sim"
    # Hand-built wall traces stamp everything in small numbers: one side.
    small = [dict(r, ts=r["ts"] - WALL) for r in _mule_rows(clock="wall")]
    for r in small:
        for c in r.get("pass_1_plan") or ():
            c["deadline_ts"] -= WALL
            c["device_deadlines"] = {d: t - WALL for d, t in c["device_deadlines"].items()}
        for s in r.get("pass_1_outcomes") or ():
            s["contact_ts"] -= WALL
    assert trace_clock_domain([], small) == "wall"


def test_a_sim_mission_carries_its_simulated_record():
    first = _obs("sim").missions[0]
    assert (first.sim_start_s, first.sim_end_s, first.sim_duration_s) == (E, E + 160.0, 160.0)
    assert first.ledger() == SIM_MISSIONS[0]["ledger"]
    assert list(first.ledger()) == list(LEDGER_KINDS)
    assert sum(first.ledger().values()) == pytest.approx(first.sim_duration_s)
    assert (first.energy_j, first.budget_overrun_s) == (19_000.0, 0.0)
    # Each is how many the mission recorded, not whether it recorded any.
    assert (first.replans, first.aborts, first.inserts) == (2, 0, 2)
    assert [(m.replans, m.aborts, m.inserts) for m in _obs("sim").missions] == [
        (2, 0, 2), (0, 2, 0), (1, 0, 2)]
    assert [e.sim_ts for e in _obs("sim").model_evals] == [None, E + 70.0, E + 230.0]

    wall = _obs("wall").missions[0]
    assert (wall.sim_start_s, wall.sim_end_s, wall.sim_ledger, wall.energy_j,
            wall.budget_overrun_s, wall.replans, wall.aborts, wall.inserts) == (None,) * 8
    assert wall.sim_duration_s is None and wall.ledger() is None
    assert [e.sim_ts for e in _obs("wall").model_evals] == [None, None, None]


def _replace_stamp(rows, mission_round, field, value):
    """``rows`` with one Pass-1 stamp of one mission replaced: its first
    session's ``contact_ts``, its first contact's ``deadline_ts``, or that
    contact's first member's own deadline (``device_deadlines``)."""
    out = json.loads(json.dumps(rows))
    for r in out:
        if r.get("event") == "mission_completed" and r["mission_round"] == mission_round:
            if field == "contact_ts":
                r["pass_1_outcomes"][0]["contact_ts"] = value
            elif field == "device_deadlines":
                own = r["pass_1_plan"][0]["device_deadlines"]
                own[next(iter(own))] = value
            else:
                r["pass_1_plan"][0]["deadline_ts"] = value
    return out


def test_a_simulated_contact_against_a_wall_deadline_is_refused_not_scored():
    """The spec's example: a trace whose sessions were stamped on the mission
    clock while its plan's deadlines are wall stamps. Scored, every contact
    would be 'on time' (a simulated stamp is a million seconds; a wall one
    over a billion)."""
    rows = _mule_rows(clock="wall")
    for r in rows:
        for s in r.get("pass_1_outcomes") or ():
            s["contact_ts"] = s["contact_ts"] - WALL + E          # simulated
    with pytest.raises(ClockDomainError, match="clock domains disagree") as info:
        _obs("wall", mule_rows=rows)
    assert "wall-clock stamp" in str(info.value) and "simulated one" in str(info.value)
    assert "0 % or 100 %" in str(info.value)

    # What the refusal prevents: mission 1's late member b reads as on time.
    mission = MissionRecord(
        mission_round=1, pass_1_contacts=1, pass_2_contacts=1, pass_1_updates=2,
        pass_1_scheduled=3, pass_1_clean_devices=("a", "b"), delivered=None,
        undelivered=None, duration_s=None,
        pass_1_deadlines=(("a", WALL + 60.0), ("b", WALL + 60.0)),
        pass_1_outcomes=(("a", "clean", E + 48.0), ("b", "clean", E + 75.0)),
    )
    fooled = Exp4Observation(n_devices=3, cluster_rounds_closed=0, up_bundles_ingested=0,
                             missions=[mission])
    assert deadline_misses(fooled).missed == 0


def test_a_wall_contact_against_a_simulated_deadline_is_refused_not_scored():
    """The reverse: a trace that names no clock (so wall-clock) whose sessions
    are wall stamps while its plan's deadlines, the contacts' and the members'
    own, are simulated. Scored, every contact would be late."""
    rows = _mule_rows(clock="wall")
    for r in rows:
        for c in r.get("pass_1_plan") or ():
            c["deadline_ts"] = c["deadline_ts"] - WALL + E                # simulated
            c["device_deadlines"] = {d: t - WALL + E for d, t in c["device_deadlines"].items()}
    with pytest.raises(ClockDomainError, match="clock domains disagree") as info:
        _obs("wall", mule_rows=rows)
    message = str(info.value)
    assert "round 1) pass_1_outcomes[a].contact_ts" in message
    assert "is a wall-clock stamp" in message
    assert "round 1) pass_1_plan[0].deadline_ts" in message and "a simulated one" in message

    # What the refusal prevents: mission 1's on-time member a reads as late.
    mission = MissionRecord(
        mission_round=1, pass_1_contacts=1, pass_2_contacts=1, pass_1_updates=2,
        pass_1_scheduled=3, pass_1_clean_devices=("a", "b"), delivered=None,
        undelivered=None, duration_s=None,
        pass_1_deadlines=(("a", E + 60.0), ("b", E + 60.0)),
        pass_1_outcomes=(("a", "clean", WALL + 48.0), ("b", "clean", WALL + 75.0)),
    )
    fooled = Exp4Observation(n_devices=3, cluster_rounds_closed=0, up_bundles_ingested=0,
                             missions=[mission])
    assert deadline_misses(fooled).missed == 2


@pytest.mark.parametrize("field, value, where", [
    ("contact_ts", WALL + 8.0, "pass_1_outcomes[c].contact_ts"),
    ("deadline_ts", WALL + 300.0, "pass_1_plan[0].deadline_ts"),
    # The member's own Deadline(j), the one its miss is scored against
    # (basis "device"), while its contact's deadline stays simulated.
    ("device_deadlines", WALL + 300.0, "pass_1_plan[0].device_deadlines[c]"),
    # The ceiling itself is wall time, as it is to the clock, which never
    # reaches it.
    ("contact_ts", SIM_CEILING_S, "pass_1_outcomes[c].contact_ts"),
], ids=["contact", "contact deadline", "member deadline", "at the ceiling"])
def test_a_wall_stamp_in_a_sim_trace_is_refused(field, value, where):
    rows = _replace_stamp(_mule_rows(), 2, field, value)
    with pytest.raises(ClockDomainError, match="not a simulated time") as info:
        _obs("sim", mule_rows=rows)
    assert where in str(info.value) and "round 2" in str(info.value)


def _drop_field(rows, event, field):
    return [{k: v for k, v in r.items() if not (r.get("event") == event and k == field)}
            for r in rows]


@pytest.mark.parametrize("case, match", [
    ("cluster sim_ts is wall", "cluster cluster_round_closed sim_ts"),
    # The upload's own simulated completion, while every sim_ts stays simulated.
    ("cluster sim_upload_ts is wall", "cluster up_bundle_ingested sim_upload_ts"),
    ("mission without sim_start_s", "has no sim_start_s"),
    ("mission without sim_end_s", "has no sim_end_s"),
    ("sim fields on the wall clock", "a field only the simulated clock writes"),
    # A wall trace whose only simulated field is a takeoff's.
    ("sim_start_s on a wall mission_started", "mission_started (mule m1) sim_start_s"),
    ("mules disagree", "the mules ran on different clocks"),
    ("cluster disagrees", "the mules ran on the 'sim' clock and the cluster on the 'wall' one"),
    ("unknown clock", "names mission clock 'lunar'"),
])
def test_traces_whose_clocks_disagree_are_refused(case, match):
    mule, cluster = _mule_rows(), _cluster_rows()
    if case == "cluster sim_ts is wall":
        cluster = [dict(r, sim_ts=WALL + 3.0) if r["event"] == "cluster_round_closed" else r
                   for r in cluster]
    elif case == "cluster sim_upload_ts is wall":
        cluster = [dict(r, sim_upload_ts=WALL + 3.0) if r["event"] == "up_bundle_ingested" else r
                   for r in cluster]
    elif case == "mission without sim_start_s":
        mule = _drop_field(mule, "mission_completed", "sim_start_s")
    elif case == "mission without sim_end_s":
        mule = _drop_field(mule, "mission_completed", "sim_end_s")
    elif case == "sim_start_s on a wall mission_started":
        mule, cluster = _mule_rows(clock="wall"), _cluster_rows(clock="wall")
        takeoff = next(r for r in mule if r["event"] == "mission_started")
        takeoff["sim_start_s"] = E + 60.0
    elif case == "sim fields on the wall clock":
        mule = _drop_field(mule, "mule_ready", "mission_clock")
        cluster = _drop_field(cluster, "cluster_ready", "mission_clock")
    elif case == "mules disagree":
        mule = mule + [dict(_mule_rows(clock="wall", mule="m2")[0])]
    elif case == "cluster disagrees":
        cluster = _drop_field(cluster, "cluster_ready", "mission_clock")
    elif case == "unknown clock":
        mule = [dict(r, mission_clock="lunar") if r["event"] == "mule_ready" else r
                for r in mule]
    with pytest.raises(ClockDomainError, match=re.escape(match)):
        _obs("sim", mule_rows=mule, cluster_rows=cluster)


# --------------------------------------------------------------------------- #
# 2. Simulated time where time matters
# --------------------------------------------------------------------------- #

def test_deadline_misses_compare_simulated_contacts_with_simulated_deadlines():
    for clock in ("sim", "wall"):       # the two switch domains together
        misses = deadline_misses(_obs(clock))
        assert (misses.missions_with_plan, misses.admitted, misses.missed) == (3, 5, 2)
        assert (misses.rate, misses.basis) == (pytest.approx(0.4), "device")


def test_time_to_tau_gains_simulated_seconds_beside_the_wall_ones():
    obs = _obs("sim")
    late = tau_reach(obs, 0.82)
    assert (late.reached, late.mission, late.cluster_round) == (True, 2, 2)
    assert (late.wall_s, late.sim_s) == (9.5, 230.0)
    early = tau_reach(obs, 0.65)
    assert (early.mission, early.wall_s, early.sim_s) == (1, 3.5, 70.0)
    assert tau_reach(obs, 0.9) == tau_reach(_obs("wall"), 0.9)          # never reached
    wall = tau_reach(_obs("wall"), 0.82)
    assert (wall.mission, wall.wall_s, wall.sim_s) == (2, 9.5, None)


def test_ages_and_network_aou_on_the_mission_clock():
    for clock in ("sim", "wall"):       # one mule: the same order on either clock
        ages = age_profile(_obs(clock), DEVICES)
        assert ages.network_aou_mean == pytest.approx(8 / 9)
        assert ages.network_aou_final == pytest.approx(5 / 3)
        assert ages.merged_updates == {"a": 1, "b": 1, "c": 1}   # a's mission-3 upload was lost


#   Two mules on the mission clock, a full quorum. Their processes finish in
#   one order (wall) and their missions end in another (sim); B took off
#   later, its bootstrap DOWN having synced it to a later cluster time:
#
#   mission  wall window  sim (s after E)  CLEAN
#   A1       0–10         0–200            a1
#   B1       1–11         30–150           b1, b2
#   A2       10–20        200–390          a1, a2
#   B2       11–21        150–400          — (empty)
#
#   Completion order: wall A1, B1, A2, B2; simulated B1, A1, A2, B2.
#   Network AoU (a1, a2, b1, b2; each device ages by its own mule's missions):
#   simulated order 0, 1/4, 0, 1/2 -> mean 3/16; wall order 1/4, 1/4, 0, 1/2 -> 1/4.

TWO_MULES = {
    "A": [dict(rnd=1, wall=(0.0, 10.0), sim=(0.0, 200.0), clean=["a1"]),
          dict(rnd=2, wall=(10.0, 20.0), sim=(200.0, 390.0), clean=["a1", "a2"])],
    "B": [dict(rnd=1, wall=(1.0, 11.0), sim=(30.0, 150.0), clean=["b1", "b2"]),
          dict(rnd=2, wall=(11.0, 21.0), sim=(150.0, 400.0), clean=[])],
}


def _two_mule_obs(clock, fleet=TWO_MULES):
    mule_rows = []
    for mule, missions in fleet.items():
        missions = [dict(m, ledger=_ledger(0, 0, 0, 0, 0, 0, 0), energy_j=0.0,
                         budget_overrun_s=None, replans=[], aborts=[], inserts=[])
                    for m in missions]
        mule_rows += _mule_rows(missions, mule=mule, clock=clock)

    def ev(t, event, sim_fields, **kw):
        row = {"ts": WALL + t, "event": event, "role": "cluster", "id": "c1", **kw}
        return dict(row, **sim_fields) if clock == "sim" else row

    cluster = [
        ev(-6.0, "cluster_ready", {"mission_clock": "sim"}),
        # Round 1 at quorum 2: A1 uploads at simulated 120, B1 at 90; B's
        # arrival closes it, and the round's simulated time is the later upload.
        ev(5.0, "up_bundle_ingested", {"sim_upload_ts": E + 120.0}, mule_id="A", mission_round=1),
        ev(7.0, "up_bundle_ingested", {"sim_upload_ts": E + 90.0}, mule_id="B", mission_round=1),
        ev(7.0, "cluster_round_closed", {"sim_ts": E + 120.0}, cluster_round=1),
        ev(7.5, "model_eval", {"sim_ts": E + 120.0}, cluster_round=1, accuracy=0.9, auc=0.9,
           loss=0.3, n_test=100),
    ]
    return observation_from_rows(
        cluster_rows=cluster, mule_rows=mule_rows, device_rows=[], n_devices=4,
        mule_slices={"A": ("a1", "a2"), "B": ("b1", "b2")},
    )


def test_missions_are_ordered_by_their_simulated_ends():
    sim, wall = _two_mule_obs("sim"), _two_mule_obs("wall")
    keys = lambda missions: [(m.mule_id, m.mission_round) for m in missions]  # noqa: E731
    assert keys(sim.ordered_missions()) == [("B", 1), ("A", 1), ("A", 2), ("B", 2)]
    assert keys(wall.ordered_missions()) == [("A", 1), ("B", 1), ("A", 2), ("B", 2)]
    # The recorded rule is kept for a caller that names no clock.
    assert keys(_ordered(sim.missions)) == keys(wall.ordered_missions())
    assert keys(_ordered(sim.missions, sim.mission_clock)) == keys(sim.ordered_missions())

    devices = ["a1", "a2", "b1", "b2"]
    assert age_profile(sim, devices).network_aou_mean == pytest.approx(3 / 16)
    assert age_profile(wall, devices).network_aou_mean == pytest.approx(1 / 4)
    assert age_profile(sim, devices).network_aou_final == pytest.approx(1 / 2)


def test_with_several_mules_time_to_tau_counts_from_the_fleets_first_takeoff():
    reach = tau_reach(_two_mule_obs("sim"), 0.82)
    # B's upload closed the round: its first mission, 120 simulated seconds
    # after the fleet took off (A, at the epoch; B left 30 s later).
    assert (reach.mule_id, reach.mission, reach.wall_s, reach.sim_s) == ("B", 1, 7.5, 120.0)
    assert tau_reach(_two_mule_obs("wall"), 0.82).sim_s is None


def test_the_fleets_first_takeoff_is_the_earliest_whichever_mule_is_read_first():
    """A's stream is read first (its file sorts first), but here its bootstrap
    synced it to a later simulated time than B's: the fleet took off with B,
    at 30 s, so the evaluation's 120 s is 90 s after it."""
    late_a = dict(TWO_MULES, A=[dict(TWO_MULES["A"][0], sim=(40.0, 200.0)), TWO_MULES["A"][1]])
    obs = _two_mule_obs("sim", late_a)
    assert (obs.missions[0].mule_id, obs.missions[0].sim_start_s) == ("A", E + 40.0)
    reach = tau_reach(obs, 0.82)
    assert (reach.mule_id, reach.mission, reach.wall_s, reach.sim_s) == ("B", 1, 7.5, 90.0)


def test_the_completion_order_falls_back_to_the_recorded_one_without_stamps():
    missions = list(_two_mule_obs("sim").missions)
    missions[0] = MissionRecord(**{**missions[0].__dict__, "sim_end_s": None})
    assert completion_order(missions, "sim") == missions
    assert completion_order([], "sim") == []


# --------------------------------------------------------------------------- #
# 3. The mission-clock columns
# --------------------------------------------------------------------------- #

def test_the_mission_clock_columns():
    s = _summary(_obs("sim"))
    assert s.mission_duration_s_mean == pytest.approx(6.0)          # still wall time
    assert s.sim_mission_duration_s_mean == pytest.approx(500.0 / 3)
    means = {col: getattr(s, col) for _kind, col in SIM_LEDGER_COLUMNS}
    assert means == pytest.approx({
        "sim_transit_s_mean": 275.0 / 3, "sim_dwell_s_mean": 0.5,
        "sim_listen_s_mean": 5.0 / 3, "sim_return_s_mean": 108.5 / 3,
        "sim_upload_s_mean": 0.5 / 3, "sim_turnaround_s_mean": 30.0,
        "sim_dock_wait_s_mean": 6.5,
    })
    assert sum(means.values()) == pytest.approx(s.sim_mission_duration_s_mean)
    assert (s.sim_energy_j_mean, s.energy_status) == (pytest.approx(19_500.0), "simulated")
    assert s.sim_budget_overrun_s_mean == pytest.approx(12.5 / 3)
    assert s.sim_budget_overrun_rate == pytest.approx(1 / 3)       # mission 2 of 3 overran
    # The trial's totals, not its counts of missions with any (2, 1, 2).
    assert (s.sim_replans, s.sim_aborts, s.sim_inserts) == (3, 2, 4)
    row = s.to_row()
    assert [row[c] for c in ("sim_replans", "sim_aborts", "sim_inserts", "energy_status")] == [
        3, 2, 4, "simulated"]


def test_the_mission_clock_columns_are_blank_on_the_wall_clock():
    s = _summary(_obs("wall"))
    assert s.mission_duration_s_mean == pytest.approx(6.0)
    assert {c: s.to_row()[c] for c in SIM_COLUMNS} == {c: "" for c in SIM_COLUMNS}
    flat = summarise_flat_fl(model_evals=[], round_logs=[], per_client_participation={},
                             n_devices=3, rf_range_m=60.0, n_missions_target=2)
    assert {c: flat.to_row()[c] for c in SIM_COLUMNS} == {c: "" for c in SIM_COLUMNS}


def test_budget_figures_are_blank_without_a_budget_and_counts_zero_without_events():
    missions = [dict(m, budget_overrun_s=None, replans=[], aborts=[], inserts=[])
                for m in SIM_MISSIONS]
    s = _summary(_obs("sim", mule_rows=_mule_rows(missions)))
    assert (s.sim_budget_overrun_s_mean, s.sim_budget_overrun_rate) == (None, None)
    assert (s.sim_replans, s.sim_aborts, s.sim_inserts) == (0, 0, 0)


def test_the_columns_follow_every_earlier_one_in_the_drivers_schema():
    columns = Exp4MetricSummary.csv_columns()
    assert columns[-len(SIM_COLUMNS):] == list(SIM_COLUMNS)
    assert columns[:-len(SIM_COLUMNS)][-3:] == ["rounds_evaluated", "t_at_tau_round", "tau"]
    assert list(_summary(_obs("sim")).to_row()) == columns
    assert [kind for kind, _col in SIM_LEDGER_COLUMNS] == list(LEDGER_KINDS)


# --------------------------------------------------------------------------- #
# 4. The scorer end to end
# --------------------------------------------------------------------------- #

def _write_trial(root, name, *, clock="sim", mule_cfg=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "cluster-c1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _cluster_rows(clock=clock)) + "\n", encoding="utf-8")
    (d / "mule-m1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _mule_rows(clock=clock)) + "\n", encoding="utf-8")
    (d / "device-a.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _device_rows()) + "\n", encoding="utf-8")
    cfg = {"mule_id": "m1", "rf_range_m": 60.0, "n_missions": 3, "session_ttl_s": 3.0}
    if clock == "sim":
        cfg.update(mission_clock="sim", trial_seed=42, contact_band="wide",
                   backhaul_model="mission")
    cfg.update(mule_cfg or {})
    (d / "mule-m1.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "cluster.json").write_text(json.dumps(
        {"seed_devices": [{"device_id": dev, "position": [0, 0, 0]} for dev in DEVICES]}
    ), encoding="utf-8")
    return d


SIM_TRIAL = "N=3-regime=jittery-rrf=60.0__H1__t0__s42"
WALL_TRIAL = "N=3-regime=jittery-rrf=60.0__H1__t1__s43"


def test_a_sim_trial_scores_in_simulated_time(tmp_path):
    score = score_trial(_write_trial(tmp_path, SIM_TRIAL), taus=(0.82, 0.65))
    row = score.to_row()
    assert (row["mission_clock"], row["contact_band"]) == ("sim", "wide")
    assert (row["wall_s_to_tau0.82"], row["sim_s_to_tau0.82"]) == (9.5, 230.0)
    assert (row["wall_s_to_tau0.65"], row["sim_s_to_tau0.65"]) == (3.5, 70.0)
    assert list(row).index("sim_s_to_tau0.82") == list(row).index("wall_s_to_tau0.82") + 1
    assert row["sim_mission_duration_s_mean"] == pytest.approx(500.0 / 3)
    assert (row["deadline_admitted"], row["deadline_missed"]) == (5, 2)
    assert row["network_aou_mean"] == pytest.approx(8 / 9)
    assert row["round_close_rate_kmin1"] == pytest.approx(2 / 3)    # mission 3 lost

    # Wall and simulated trials share one fixed schema.
    wall = score_trial(_write_trial(tmp_path, WALL_TRIAL, clock="wall"), taus=(0.82, 0.65))
    assert list(wall.to_row()) == list(row)
    assert (wall.to_row()["sim_s_to_tau0.82"], wall.to_row()["wall_s_to_tau0.82"]) == ("", 9.5)
    out = tmp_path / "scored.csv"
    write_scores_csv([wall, score], out)
    read = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [(r["mission_clock"], r["sim_s_to_tau0.82"], r["energy_status"]) for r in read] == [
        ("", "", ""), ("sim", "230.0", "simulated"),
    ]


def test_the_scorer_names_the_trial_it_refuses(tmp_path):
    d = _write_trial(tmp_path, SIM_TRIAL)
    rows = _replace_stamp(_mule_rows(), 2, "contact_ts", WALL + 8.0)
    (d / "mule-m1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                     encoding="utf-8")
    with pytest.raises(ClockDomainError, match=rf"^{SIM_TRIAL}: clock domains disagree"):
        score_trial(d)


def test_a_mule_config_on_the_other_clock_than_its_events_is_refused(tmp_path):
    d = _write_trial(tmp_path, SIM_TRIAL, mule_cfg={"mission_clock": "wall"})
    with pytest.raises(ClockDomainError, match="mule config sets mission_clock 'wall'"):
        score_trial(d)


def test_a_sim_trial_that_died_before_announcing_its_clock_scores_under_its_status(tmp_path):
    """A startup failure: no process announced itself and no mission flew, so
    no event names a clock for the config's to disagree with. The trial scores
    as an empty one under its marker's status (``--include-failed`` shows it),
    with its configured clock in the provenance; it is not refused."""
    d = _write_trial(tmp_path, SIM_TRIAL)
    for stream in ("cluster-c1.jsonl", "mule-m1.jsonl"):
        (d / stream).write_text("", encoding="utf-8")
    (d / TRIAL_STATUS_FILE).write_text(json.dumps({"status": "error", "error": "startup"}))
    score = score_trial(d)
    assert (score.status, score.summary.missions_completed) == ("error", 0)
    assert score.provenance["mission_clock"] == "sim"
    # The metric columns follow the events' clock, and no event names one.
    assert score.to_row()["energy_status"] == ""


# --------------------------------------------------------------------------- #
# 5. Provenance
# --------------------------------------------------------------------------- #

def test_a_trace_recorded_before_phase_3_derives_the_three_new_settings(tmp_path):
    """The rules the driver's docstring gives for a kept trace: realism from a
    device's own contact reliability, the L1 channel from the cluster's loss
    schedule, the model width from the cluster; the recorded 3 s TTL and every
    clock setting blank."""
    d = tmp_path / WALL_TRIAL
    d.mkdir()
    (d / "mule-m1.json").write_text(json.dumps(
        {"mule_id": "m1", "rf_range_m": 60.0, "n_missions": 4, "session_ttl_s": 3.0}))
    (d / "cluster.json").write_text(json.dumps(
        {"backhaul_loss_schedule": [0.1, 0.2, 0.1, 0.3], "input_dim": 21}))
    (d / "device-a.json").write_text(json.dumps(
        {"device_id": "a", "position": [3.0, 4.0, 0.0], "contact_reliability": 0.62}))
    got = {c: trial_provenance(d)[c] for c in PROVENANCE_COLUMNS[-13:]}
    assert got == {**{c: "" for c in PROVENANCE_COLUMNS[-13:]},
                   "l1_channel": 1, "realism": 1, "input_dim": 21}
    (d / "mule-m1.json").write_text(json.dumps({"session_ttl_s": 5.0}))
    (d / "cluster.json").write_text(json.dumps({"backhaul_loss_schedule": None}))
    (d / "device-a.json").write_text(json.dumps({"contact_reliability": None}))
    got = {c: trial_provenance(d)[c] for c in PROVENANCE_COLUMNS[-13:]}
    assert got == {**{c: "" for c in PROVENANCE_COLUMNS[-13:]}, "session_ttl_s": 5.0}


def _cell(arm="H1", seed=7, **params) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 4, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


def _sim(**kw) -> Exp4Driver:
    kw.setdefault("mission_clock", "sim")
    kw.setdefault("realism", True)
    return Exp4Driver(**kw)


def _kept_trace(root, driver, cell):
    """The driver's row for ``cell`` (nothing spawned), and the trace directory
    a kept trial would leave: the per-role JSON the real orchestrator writes
    for the driver's topology."""
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


PARITY = [
    ("wall defaults", lambda: Exp4Driver(), _cell("H1", regime="clean")),
    ("wall realism, L1, time unit and TTL",
     lambda: Exp4Driver(realism=True, l1_channel=True, deadline_time_scale=2.0, session_ttl_s=5.0),
     _cell("H3")),
    ("wall Φ₀", lambda: Exp4Driver(initial_window_s=90.0), _cell("H1", regime="clean")),
    ("sim, no band (the channel-free control)", lambda: _sim(), _cell("H1")),
    ("sim, wide, seconds backhaul (T_nom computed), channel source, replan, payload, physics",
     lambda: _sim(contact_band="wide", backhaul_model="seconds",
                  contact_reliability_source="channel", in_flight_response="replan",
                  payload_bytes=1_000_000, ferry_physics={"n_pl": 3.0}),
     _cell("H1")),
    ("sim, channel source without realism",
     lambda: _sim(realism=False, contact_band="wide", contact_reliability_source="channel"),
     _cell("H1")),
    ("sim, time unit T_nom / 10 s and Φ₀ in missions (T_nom computed)",
     lambda: _sim(deadline_time_scale="t_nom", initial_window_missions=6.0), _cell("H1")),
    # T_nom computed for one setting alone, for each setting the scorer's
    # inference reads (the seconds backhaul and D5's period alone are below).
    ("sim, time unit T_nom / 10 s alone (T_nom computed for it)",
     lambda: _sim(deadline_time_scale="t_nom"), _cell("H1")),
    ("sim, Φ₀ in missions at the recorded time unit (T_nom computed for Φ₀ alone)",
     lambda: _sim(initial_window_missions=6.0), _cell("H1")),
    ("sim, a numeric time unit and Φ₀ in seconds (no T_nom)",
     lambda: _sim(deadline_time_scale=2.0, initial_window_s=90.0), _cell("H1")),
    ("sim, T_nom given, nothing needs it", lambda: _sim(t_nom_s=210.0), _cell("H1")),
    ("sim, T_nom given, the seconds backhaul's period given too (nothing needs T_nom)",
     lambda: _sim(contact_band="wide", backhaul_model="seconds", backhaul_period_s=500.0,
                  t_nom_s=210.0),
     _cell("H1")),
    ("sim, two mules at a full quorum",
     lambda: _sim(n_mules=2, min_participation=2, contact_band="wide"), _cell("H1", N=8)),
    ("sim, D4 split by CARP between two mules",
     lambda: _sim(n_mules=2, min_participation=2, contact_band="wide"), _cell("D4", N=8)),
    ("sim, L1 channel under the mission model",
     lambda: _sim(l1_channel=True, contact_band="wide"), _cell("H3", seed=2191267877)),
    ("sim, H3's adaptive seconds backhaul",
     lambda: _sim(contact_band="wide", backhaul_model="seconds"), _cell("H3")),
    ("sim, D5's merge period at T_nom",
     lambda: _sim(aggregation="agg:cutoff", aggregation_params={"a_max": 3},
                  agg_period_t_nom=True, mission_budget_s=45.0),
     _cell("D5", regime="clean")),
]


@pytest.mark.parametrize("name, make, cell", PARITY, ids=[p[0] for p in PARITY])
def test_the_provenance_is_the_drivers_own_row(tmp_path, name, make, cell):
    """Every provenance column, formatted exactly as the driver's CSV row
    writes it, derived from the configs a kept trace holds."""
    row, d = _kept_trace(tmp_path, make(), cell)
    assert trial_provenance(d) == {c: row[c] for c in PROVENANCE_COLUMNS}


@pytest.mark.parametrize("make, cell", [
    (lambda: _sim(t_nom_s=150.0, deadline_time_scale="t_nom"), _cell("H1")),
    (lambda: _sim(contact_band="wide", backhaul_model="seconds", t_nom_s=210.0), _cell("H1")),
    (lambda: _sim(aggregation="agg:cutoff", aggregation_params={"a_max": 3},
                  agg_period_t_nom=True, mission_budget_s=45.0, t_nom_s=210.0),
     _cell("D5", regime="clean")),
    (lambda: _sim(t_nom_s=150.0, initial_window_missions=4.0), _cell("H1")),
    (lambda: _sim(t_nom_s=210.0, initial_window_s=90.0), _cell("H1")),
    (lambda: _sim(t_nom_s=150.0, deadline_time_scale="t_nom", initial_window_missions=4.0),
     _cell("H1")),
], ids=["time unit", "seconds backhaul", "D5's period", "Φ₀ in missions", "Φ₀ in seconds",
        "time unit and Φ₀ in missions"])
def test_a_given_t_nom_beside_a_setting_that_may_use_it_is_read_from_the_marker(
        tmp_path, make, cell):
    """The configs record T_nom but not whether the driver computed it.
    Without ``--t-nom-s`` the driver records one only when a setting needed
    it, so every such trial is inferred exactly (the PARITY cases). A T_nom
    given beside a setting that may use it (the seconds backhaul without a
    period, the time unit at T_nom / 10 s, D5's period at T_nom, or any Φ₀,
    which the configs record in seconds however it was given) is taken as
    computed unless the trial's marker says otherwise. The driver's marker
    does not say yet (a hand-off), so such a trial differs from its row in
    that one flag."""
    row, d = _kept_trace(tmp_path, make(), cell)
    got, want = trial_provenance(d), {c: row[c] for c in PROVENANCE_COLUMNS}
    assert {c: got[c] for c in want if c != "ferry_params"} == {
        c: want[c] for c in want if c != "ferry_params"}
    assert json.loads(want["ferry_params"])["t_nom_computed"] is False
    assert json.loads(got["ferry_params"]) == {**json.loads(want["ferry_params"]),
                                               "t_nom_computed": True}
    # The marker as the driver writes it today leaves the inference standing.
    marker = {"status": "ok", "error": "", "n_missions_target": 4, "run_s": 1.0,
              "trial_budget_s": 120.0}
    (d / TRIAL_STATUS_FILE).write_text(json.dumps(marker))
    assert trial_provenance(d) == got
    (d / TRIAL_STATUS_FILE).write_text(json.dumps({**marker, "t_nom_computed": False}))
    assert trial_provenance(d) == want


def test_the_realism_of_a_channel_source_trace_comes_from_its_layout(tmp_path):
    """Under the channel source the devices carry no reliability of their own
    and the mule's availability map is filled either way; only the layout
    (the realism field, or the tight cluster the seed draws without it) says."""
    on_row, on = _kept_trace(tmp_path / "on", _sim(
        contact_band="wide", contact_reliability_source="channel"), _cell("H1"))
    off_row, off = _kept_trace(tmp_path / "off", _sim(
        realism=False, contact_band="wide", contact_reliability_source="channel"), _cell("H1"))
    for d in (on, off):
        assert all(json.loads(p.read_text())["contact_reliability"] is None
                   for p in d.glob("device-*.json"))
        assert json.loads(next(d.glob("mule-*.json")).read_text())["device_availability"]
    assert (on_row["realism"], off_row["realism"]) == (1, "")
    assert (trial_provenance(on)["realism"], trial_provenance(off)["realism"]) == (1, "")


def test_the_simulated_stamps_stay_below_the_ceiling_the_check_uses():
    assert E < SIM_CEILING_S < WALL
