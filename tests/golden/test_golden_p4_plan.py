"""Phase 4's plan arms, pinned at 386c275 (FeRRy Phase 5, unit UG5).

Freeze Rule 1 in Phase 5: every mechanism sits behind a switch whose default is
the recorded pipeline, and "recorded" now has three faces: the wall clock
(afa9526), Phase 3's simulated clock with ``plan_mode=legacy`` (6e6f92d, UG4)
and Phase 4's plan arms (386c275, here). These tests run the eight stub trials
of ``_build_p4_plan`` in this process (UG4's in-process trial: the driver, the
per-role JSON the real orchestrator writes, the mule, cluster and device
services, and the mule process's own service loop) and compare each part of
each trial with ``data/p4_plan.json``, captured on the untouched tree at
386c275. A field added with a default passes; a removed or renamed field, a
changed value, a new event, or a device appearing in or leaving a device-keyed
map fails (critic A3's rule, see ``_build_p3_sim``). So does a changed call to
the flight slot, an argument added to one included, with a default or without
(the Phase 5 spec, other choices 1; ``_build_p4_plan.compare_part``). The
planner's wall time (``plan_wall_s``) is kept as a key and masked as a value.

Most of the rest read the fixture itself, so that it keeps exercising what
Phase 5 must leave alone: each arm's settings; the cap binding and its
violations; member subsets at the plan and in flight; hover stops; FX's band at
arrival and its next stop after a stop, the departure check before that pick,
and the committed slot's plan order. The last part checks ``make_baseline.py``'s
third baseline, recorded at 386c275, and that it gates the Phase 4 tests which
pin what no trial here reaches (the beacon hook, the exempt stops' protection,
K = 2).
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import pytest

from experiments.exp4.driver import Exp4Driver
from tests.golden import _build_p3_sim as UG4
from tests.golden import _build_p4_plan as B
from tests.golden import _canon
from tests.golden import make_baseline as MB

GOLDEN = B.load_golden()
CASES: Dict[str, Any] = GOLDEN["cases"]

ARM = {name: cell.arm for name, (_settings, cell) in B.TRIALS.items()}
#: The six arms on one layout at 45 s.
AT_45 = tuple(n for n in B.TRIAL_NAMES if n.endswith("_45s"))
FX_TRIALS = tuple(n for n in B.TRIAL_NAMES if ARM[n] == "FX")
COMMITTED = tuple(n for n in B.TRIAL_NAMES if ARM[n] != "FX")


def plain(value: Any) -> Any:
    """A canonical value back as plain JSON: records untyped, floats as floats."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if k != _canon.TYPE_KEY}
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, str) and value.startswith("f:"):
        return float(value[2:])
    return value


def missions(name: str) -> List[Dict[str, Any]]:
    (mule,) = CASES[name]["mission_completed"].values()
    return plain(mule)


def row(name: str) -> Dict[str, Any]:
    return plain(CASES[name]["row"])


def slot_calls(name: str, call: str) -> List[Dict[str, Any]]:
    return [c for c in plain(CASES[name]["flight_slot"]) if c["call"] == call]


def config(name: str, prefix: str) -> List[Dict[str, Any]]:
    return [plain(v) for k, v in CASES[name]["configs"].items() if k.startswith(prefix)]


def device_positions(name: str) -> Dict[str, List[float]]:
    return {cfg["device_id"]: cfg["position"] for cfg in config(name, "device-")}


def dock(name: str) -> List[float]:
    (ready,) = plain(CASES[name]["mule_ready"]).values()
    return ready[0]["dock"]


def mission_at(ms: Sequence[Dict[str, Any]], clock: float) -> Dict[str, Any]:
    (m,) = [m for m in ms if m["sim_start_s"] <= clock <= m["sim_end_s"]]
    return m


def plan_index(plan: Sequence[Dict[str, Any]], devices: Sequence[str], start: int) -> Optional[int]:
    """The first planned stop at or after ``start`` holding every one of ``devices``."""
    for i in range(start, len(plan)):
        if set(devices) <= set(plan[i]["devices"]):
            return i
    return None


def flies_the_plan_order(m: Dict[str, Any]) -> bool:
    """Every Pass-1 stop flown is a planned stop (or a trim of one), in the plan's order."""
    at = 0
    for s in m["pass_1_flown"]:
        i = plan_index(m["pass_1_plan"], s["devices"], at)
        if i is None:
            return False
        at = i + 1
    return True


def on_the_dock_side(p: Sequence[float], device: Sequence[float], dk: Sequence[float]) -> bool:
    """``p`` on the segment from the dock to ``device``, the device's own position excluded."""
    px, py = p[0] - dk[0], p[1] - dk[1]
    qx, qy = device[0] - dk[0], device[1] - dk[1]
    qq = qx * qx + qy * qy
    t = (px * qx + py * qy) / qq
    off = abs(px * qy - py * qx) / math.sqrt(qq)
    return off < 1e-6 and -1e-9 <= t < 1.0 - 1e-9


# --------------------------------------------------------------------------- #
# The fixture
# --------------------------------------------------------------------------- #

def test_the_fixture_is_the_386c275_capture_of_the_named_trials():
    meta = GOLDEN["_meta"]
    assert meta["base_commit"] == B.BASE_COMMIT == "386c27552e249da07550bc9042b4a907c7e8e684"
    assert meta["unit"] == "UG5"
    assert tuple(CASES) == B.TRIAL_NAMES
    assert B.PARTS == UG4.PARTS + ("flight_slot",)
    for name in B.TRIAL_NAMES:
        assert set(CASES[name]) == {"inputs", *B.PARTS}
        # The trial table is the one the fixture was captured with.
        assert CASES[name]["inputs"] == UG4.inputs_of(*B.TRIALS[name]), name


def test_the_45s_trials_share_one_layout_and_fx_at_60s_differs_only_by_its_budget():
    """Six arms on one cell, as in one CSV (seed 59), and FX on the same cell
    at the exit gate's 60 s; FX at N = 12 is its own cell (seed 26)."""
    first = CASES[AT_45[0]]["inputs"]
    for name in AT_45:
        inputs = CASES[name]["inputs"]
        assert inputs["settings"] == first["settings"], name
        assert {k: v for k, v in inputs["cell"].items() if k != "arm"} == (
            {k: v for k, v in first["cell"].items() if k != "arm"}), name
    assert sorted(ARM[n] for n in AT_45) == sorted(
        ["F", "FX", "FB+medium", "F-cov", "F-cap", "F-prio"])
    at_45, at_60 = CASES["fx_45s"]["inputs"], CASES["fx_60s"]["inputs"]
    assert at_60["cell"] == at_45["cell"]
    budget = "mission_budget_s"
    assert ({k: v for k, v in at_60["settings"].items() if k != budget}
            == {k: v for k, v in at_45["settings"].items() if k != budget})
    assert (plain(at_45["settings"][budget]), plain(at_60["settings"][budget])) == (45.0, 60.0)
    n12 = plain(CASES["fx_n12_120s"]["inputs"])
    assert (n12["cell"]["params"]["N"], n12["cell"]["params"]["n_missions"]) == (12, 4)
    assert (n12["settings"][budget], n12["settings"]["age_cap_missions"]) == (120.0, 3)
    assert {plain(CASES[n]["inputs"])["cell"]["seed"] for n in B.TRIAL_NAMES} == {
        B.SEED, B.N12_SEED}


# --------------------------------------------------------------------------- #
# The pins
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("part", B.PARTS)
@pytest.mark.parametrize("name", B.TRIAL_NAMES)
def test_the_plan_arms_at_their_defaults_are_the_386c275_ones(name, part):
    golden = CASES[name][part]
    current = B.capture(name)[part]
    problems = B.compare_part(part, golden, current)
    assert not problems, (
        f"{name}.{part}: Phase 4's plan-arm behaviour pinned at 386c275 changed "
        f"({len(problems)} mismatch(es) shown; a key added with a default would pass, "
        f"an argument added to a flight-slot call would not):\n  "
        + "\n  ".join(problems)
    )


# --------------------------------------------------------------------------- #
# What the fixture exercises
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", B.TRIAL_NAMES)
def test_each_trial_flies_its_plan_arm_on_the_cells_settings(name):
    settings, cell = B.TRIALS[name]
    arm = ARM[name]
    (mule,) = config(name, "mule-")
    assert (mule["plan_mode"], mule["member_admission"], mule["replan_fallback"]) == (
        "ferry", "subset", "trim")
    assert mule["band_class_policy"] == ("fixed:medium" if arm == "FB+medium" else "search")
    assert mule["contact_band"] == ("medium" if arm == "FB+medium" else "wide")
    assert mule["flight_slot"] == ("cross_heuristic" if arm == "FX" else "committed")
    assert mule["age_cap_missions"] == (None if arm == "F-cap" else settings["age_cap_missions"])
    assert mule["miss_priority"] is (arm != "F-prio")
    assert mule["plan_score_params"] == (
        {"c_cov_per_device": 0.0, "c_link": 0.0} if arm == "F-cov" else {})
    assert (mule["contact_regime"], mule["payload_bytes"], mule["mission_budget_s"]) == (
        "jittery", 1_000_000, settings["mission_budget_s"])
    assert mule["n_missions"] == cell.params["n_missions"] == len(missions(name))
    r = row(name)
    assert (r["contact_band"], r["miss_priority"]) == (
        "medium" if arm == "FB+medium" else "search", int(arm != "F-prio"))
    assert json.loads(r["ferry_params"])["flight_slot"] == mule["flight_slot"]
    assert {c["slot"] for c in plain(CASES[name]["flight_slot"])} == {mule["flight_slot"]}
    for m in missions(name):
        assert m["plan"]["band_class_policy"] == mule["band_class_policy"]
        # Pass 2 flies b̄, whatever the slot.
        assert all(s["band"] == m["band"] for s in m["pass_2_flown"]), m["mission_round"]
    if arm == "FB+medium":
        flown = [s for m in missions(name) for s in m["pass_1_flown"] + m["pass_2_flown"]]
        assert flown and {s["band"] for s in flown} == {"medium"}


def test_the_cap_binds_and_is_violated_for_three_causes():
    """Every capped arm has capped devices (aged S or more, S's lookahead 0)
    and violations only of capped devices; F-cap has no cap. Between them the
    trials violate the cap for three causes (none ``unplannable``: each capped
    device a plan left out could have been served alone, from its hover point
    if need be, so it is ``crowded``)."""
    causes = set()
    for name in B.TRIAL_NAMES:
        ms = missions(name)
        if ARM[name] == "F-cap":
            assert all(m["plan"]["cap"]["s"] is None and m["plan"]["cap"]["capped"] == []
                       for m in ms)
            continue
        assert any(m["plan"]["cap"]["capped"] for m in ms), name
        for m in ms:
            cap = m["plan"]["cap"]
            assert cap["s"] == B.TRIALS[name][0]["age_cap_missions"]
            assert cap["capped"] == sorted(d for d, a in cap["ages"].items() if a >= cap["s"])
            assert {v["device"] for v in cap["violations"]} <= set(cap["capped"])
            causes |= {v["reason"] for v in cap["violations"]}
    assert causes == {"crowded", "dropped_in_flight", "not_merged"}


@pytest.mark.parametrize("name", [n for n in B.TRIAL_NAMES if n != "fx_n12_120s"])
def test_member_subsets_reduce_an_s3a_stop_at_the_plan(name):
    """A reduced stop keeps S3a's position (``member_subset.reduce_stop``), and
    the members the plan left out are dropped there, under the reason
    ``plan``: so a pre-flight drop at the position of a stop flown in Pass 1
    is the complement of a stop the plan reduced."""
    found = []
    for m in missions(name):
        at = {tuple(s["position"]): s for s in m["pass_1_flown"]}
        for d in m["pass_1_preflight_drops"]:
            s = at.get(tuple(d["position"]))
            if s is not None:
                found.append((d, s))
    assert found
    for d, s in found:
        assert d["reason"] == "plan" and not set(d["devices"]) & set(s["devices"])


@pytest.mark.parametrize("name", ["f_45s", "fx_45s", "f_prio_45s", "fx_n12_120s"])
def test_the_departure_check_trims_a_stops_members_after_a_stop(name):
    """In flight the re-plan trims the committed plan (``arm_trimmed``): a stop
    keeps the members that still fit, and the rest are dropped for the budget."""
    trims = []
    for m in missions(name):
        if not m["pass_1_flown"]:
            continue
        first = m["pass_1_flown"][0]["depart_s"]
        for r in m["replans"]:
            if r["pass"] != "collect" or r["t_s"] <= first or r["order_used"] != "arm_trimmed":
                continue
            dropped = {d for x in r["dropped"] for d in x["devices"]}
            for kept in r["route"]:
                for before in r["before"]:
                    if set(kept) < set(before):
                        trims.append(kept)
                        assert set(before) - set(kept) <= dropped
                        assert {x["reason"] for x in r["dropped"]} == {"budget"}
    assert trims


@pytest.mark.parametrize("name", ["f_45s", "fx_45s", "fb_medium_45s", "f_cov_45s", "f_prio_45s"])
def test_hover_stops_serve_capped_devices_from_the_dock_side(name):
    """The hover rule (``plan/hover.py``): a capped device its S3a stop cannot
    serve alone gets a one-device stop at its best hover point, on the segment
    from the dock to the device. Such a stop is not marked in the trace
    (Experiment_4_Run_Guide.md section 2.7); it is found by its position."""
    positions, dk = device_positions(name), dock(name)
    found = [(m, s) for m in missions(name) for s in m["pass_1_flown"]
             if len(s["devices"]) == 1
             and on_the_dock_side(s["position"], positions[s["devices"][0]], dk)]
    assert found
    for m, s in found:
        assert s["devices"][0] in m["plan"]["cap"]["capped"]


@pytest.mark.parametrize("name", FX_TRIALS)
def test_fx_serves_each_stop_on_the_fastest_class_that_covers_the_committed_one(name):
    """Decision 5 with critic A7's rule, read from the slot's own calls: in
    Pass 1 the class flown is, among the classes whose targets include every
    target of b̄ at the arrival SNR, the one with the least dwell; FX switches
    class at some arrivals; in Pass 2 it reads no view and keeps b̄."""
    calls = slot_calls(name, "band_at_arrival")
    pass_1 = [c for c in calls if c["pass"] == "collect"]
    flown = [s for m in missions(name) for s in m["pass_1_flown"]]
    assert len(pass_1) == len(flown)
    switched = 0
    for c, s in zip(pass_1, flown):
        view = c["view"]
        classes = {k["name"]: k for k in view["classes"]}
        need = set(classes[view["committed"]]["targets"])
        covering = [k for k in view["classes"] if need <= set(k["targets"])]
        chosen = classes[c["band"] or view["committed"]]
        assert chosen in covering
        assert chosen["dwell_s"] == min(k["dwell_s"] for k in covering)
        assert view["devices"] == s["devices"] and s["band"] == chosen["name"]
        switched += c["band"] is not None
    assert switched
    pass_2 = [c for c in calls if c["pass"] == "deliver"]
    assert pass_2 and all(c["view"] is None and c["band"] is None for c in pass_2)


def _nearest_first(remainder: Sequence[Dict[str, Any]], pose: Sequence[float]) -> List[int]:
    """``cross_heuristic.nearest_first`` on recorded waypoints."""
    def dist(p):
        return sum((a - b) ** 2 for a, b in zip(pose, p)) ** 0.5
    return sorted(range(len(remainder)),
                  key=lambda i: (dist(remainder[i]["position"]), tuple(remainder[i]["position"]),
                                 tuple(remainder[i]["devices"]), i))


@pytest.mark.parametrize("name", FX_TRIALS)
def test_fx_picks_the_nearest_stop_whose_move_to_the_front_fits(name):
    """FX's next-stop rule, read from its calls: in Pass 1 after a stop, the
    candidates are tried nearest first, each moved to the front of the plan
    order, until ``fits`` (the departure check's fold) passes one; at takeoff,
    in Pass 2 and for a lone stop it takes the plan's next stop without a fold."""
    for c in slot_calls(name, "next_stop"):
        rem = c["remainder"]
        if not (c["pass"] == "collect" and c["after_stop"]) or len(rem) == 1:
            assert (c["index"], c["fits"]) == (0, [])
            continue
        order = _nearest_first(rem, c["state"]["pose"])
        tried = c["fits"]
        assert 1 <= len(tried) <= len(order)
        for t, i in zip(tried, order):
            assert t["order"] == [s["devices"] for s in [rem[i]] + rem[:i] + rem[i + 1:]]
        oks = [t["ok"] for t in tried]
        if any(oks):
            assert oks == [False] * (len(oks) - 1) + [True]
            assert c["index"] == order[len(tried) - 1]
        else:
            assert len(tried) == len(order) and c["index"] == 0


def test_fx_reorders_after_a_stop_and_the_departure_check_runs_before_its_pick():
    """At N = 12 FX re-orders in two sorties, its fold refuses candidate orders,
    and Phase 4's order holds at every Pass-1 departure: the departure check
    (and its trim) first, then the slot's pick from what the check kept. After
    the first re-order the check trims that sortie's remainder."""
    name = "fx_n12_120s"
    calls = [c for c in slot_calls(name, "next_stop") if c["pass"] == "collect"]
    reorders = [c for c in calls if c["index"] > 0]
    assert len(reorders) == 2 and all(c["after_stop"] for c in reorders)
    assert sum(1 for c in calls if any(not t["ok"] for t in c["fits"])) >= 2
    ms = missions(name)
    picked = {id(mission_at(ms, c["state"]["clock"])) for c in reorders}
    assert len(picked) == 2
    for m in ms:
        assert flies_the_plan_order(m) is (id(m) not in picked), m["mission_round"]
    checked = 0
    for m in ms:
        for r in m["replans"]:
            if r["pass"] != "collect" or not r["route"]:
                continue
            (c,) = [c for c in calls if c["state"]["clock"] == r["t_s"]]
            assert [s["devices"] for s in c["remainder"]] == r["route"]
            checked += 1
    assert checked >= 2
    first = reorders[0]
    m = mission_at(ms, first["state"]["clock"])
    later = [r for r in m["replans"]
             if r["pass"] == "collect" and r["t_s"] > first["state"]["clock"]]
    assert later and all(r["order_used"] == "arm_trimmed" for r in later)


@pytest.mark.parametrize("name", COMMITTED)
def test_the_committed_slot_flies_the_plan_in_its_order_on_its_class(name):
    """F's slot (and FB+medium's and the ablations'): the plan's next stop with
    no fold, no view and no switch, so Pass 1 flies the committed plan in its
    order (stops trimmed or dropped in flight aside) on b̄."""
    for c in slot_calls(name, "next_stop"):
        assert (c["index"], c["fits"]) == (0, [])
    for c in slot_calls(name, "band_at_arrival"):
        assert c["view"] is None and c["band"] is None
    for m in missions(name):
        assert flies_the_plan_order(m), m["mission_round"]
        assert all(s["band"] == m["band"] for s in m["pass_1_flown"])


def test_f_cov_serves_only_what_the_cap_forces():
    """F-cov, F without the coverage term (decision 3, "cap-only service"):
    before the cap binds it flies empty, and then it serves capped devices only."""
    ms = missions("f_cov_45s")
    assert ms[0]["plan"]["cap"]["capped"] == [] and ms[0]["pass_1_flown"] == []
    for m in ms:
        assert set(m["plan"]["served"]) <= set(m["plan"]["cap"]["capped"]), m["mission_round"]
    assert any(m["plan"]["served"] for m in ms)


def test_f_prio_weighs_by_age_alone_and_flies_as_f_on_this_layout():
    """F-prio's coverage weight is the device's age, F's the age times one plus
    its miss streak (decision 3). On this layout the two weights lead to the
    same flights, so F-prio differs from F in its plans' weights and its row's
    ``miss_priority`` only."""
    f, p = missions("f_45s"), missions("f_prio_45s")
    for m in p:
        assert m["plan"]["weights"] == {d: float(a) for d, a in m["plan"]["cap"]["ages"].items()}
    assert any(a["plan"]["weights"] != b["plan"]["weights"] for a, b in zip(f, p))

    def flights(ms):
        return [[(s["devices"], s["band"], s["position"]) for s in m["pass_1_flown"] + m["pass_2_flown"]]
                for m in ms]

    assert flights(f) == flights(p)
    rf, rp = row("f_45s"), row("f_prio_45s")
    assert {k for k in rf if rf[k] != rp[k]} == {"miss_priority"}


def test_the_planners_wall_time_is_kept_and_masked():
    for name in B.TRIAL_NAMES:
        assert all(m["plan_wall_s"] == B.WALL_TOKEN for m in missions(name)), name
    for wall in ("f:0.0123", "f:0.0", "f:2.5"):
        assert B.masked_wall(wall) == B.WALL_TOKEN
    for other in ("f:-1.0", "f:nan", "f:inf", None, 0.5, "x"):
        assert B.masked_wall(other) == other


def test_the_rows_device_serve_columns_are_harness_values():
    """As in UG4's trials: no device service loop runs in process, so no
    ``device_served`` event is written (the README says so)."""
    for name in B.TRIAL_NAMES:
        r = row(name)
        assert (r["coverage"], r["jains_fairness"], r["participation_entropy"]) == (
            0.0, 1.0, 0.0), name
        for evs in plain(CASES[name]["device_events"]).values():
            assert [e["event"] for e in evs] == ["device_ready"], name


# --------------------------------------------------------------------------- #
# The comparison rule and the harness
# --------------------------------------------------------------------------- #

def test_an_added_key_passes_and_everything_else_is_caught():
    """Critic A3's rule on FX's fourth mission at N = 12, and ``compare_part``'s
    stricter rule on a call to its slot, where an added argument fails too."""
    name = "fx_n12_120s"
    ids = frozenset(CASES[name]["device_events"])
    golden = CASES[name]["mission_completed"]
    raw = json.loads(json.dumps(golden))
    (mule,) = raw

    def diff_after(change) -> List[str]:
        current = json.loads(json.dumps(raw))
        change(current[mule][3])
        return _canon.diff(golden, current)

    assert diff_after(lambda m: None) == []
    # Added fields pass, at the top and inside a record.
    assert diff_after(lambda m: m.update(pass_1_pairs=[{"band": "wide"}])) == []
    assert diff_after(lambda m: m["plan"].update(pair="x")) == []
    # Another wall time is still the token; a removed one, a renamed field, a
    # changed value, a device in a device-keyed map: caught.
    assert diff_after(lambda m: m.pop("plan_wall_s"))
    assert diff_after(lambda m: m.update(plan_wall=m.pop("plan_wall_s")))
    assert diff_after(lambda m: m.update(energy_j=_canon.f(plain(m["energy_j"]) + 1e-9)))
    assert diff_after(lambda m: m["deadline_state"].pop(sorted(m["deadline_state"])[0]))
    assert diff_after(lambda m: m["pass_1_flown"].pop())
    current = json.loads(json.dumps(raw))
    current[mule][3]["plan_wall_s"] = B.masked_wall(_canon.f(0.123))
    assert _canon.diff(golden, current) == []
    # The slot's calls (the Phase 5 spec, other choices 1): a field added inside
    # an argument passes; an argument added or left out, another index, verdict
    # or order does not. Critic A3's rule alone would pass the added argument.
    calls = CASES[name]["flight_slot"]
    k = next(i for i, c in enumerate(calls) if c["call"] == "next_stop" and plain(c)["index"] > 0)

    def slot_diff(change) -> List[str]:
        current = json.loads(json.dumps(calls))
        change(current[k])
        return B.compare_part("flight_slot", calls, current)

    def add_view(c):
        c[B.ADDED_ARGUMENTS] = {"view": None}

    assert slot_diff(lambda c: None) == []
    assert slot_diff(lambda c: c["state"].update(phase=0.5)) == []
    assert slot_diff(lambda c: c["remainder"][0].update(pair="x")) == []
    assert slot_diff(add_view) == [
        f"$[{k}].{B.ADDED_ARGUMENTS}: next_stop of the cross_heuristic slot (collect) was "
        f"passed ['view'], which its 386c275 call does not pass"]
    with_view = json.loads(json.dumps(calls))
    add_view(with_view[k])
    assert _canon.diff(calls, with_view) == []
    assert slot_diff(lambda c: c.pop("after_stop")) == [
        f"$[{k}].after_stop: missing in current (golden true)"]
    assert slot_diff(lambda c: c.update(index=0))
    assert slot_diff(lambda c: c["fits"][0].update(ok=not c["fits"][0]["ok"]))
    assert slot_diff(lambda c: c["fits"].pop())
    assert slot_diff(lambda c: c["remainder"].reverse())
    assert slot_diff(lambda c: c["state"].pop("deliver_by"))


def test_the_flight_slot_spy_only_records():
    """The trial run without the spy is the trial run with it, part for part."""
    name = "fx_n12_120s"
    settings, cell = B.TRIALS[name]
    with UG4.in_process_orchestrator():
        UG4.InProcessOrchestrator.last = None
        row_ = dict(Exp4Driver(**settings).run_trial(cell))
        orch = UG4.InProcessOrchestrator.last
    bare = B.mask_wall_times(UG4.case_of(settings, cell, row_, orch))
    spied = B.capture(name)
    for part in UG4.PARTS:
        assert bare[part] == spied[part], part


def test_the_flight_slot_spy_restores_the_slots_even_on_an_error():
    from hermes.scheduler.policies import cross_heuristic as ch

    def methods():
        return {(cls.__name__, n): cls.__dict__[n] for cls in (ch.CommittedSlot, ch.CrossHeuristic)
                for n in ("next_stop", "band_at_arrival")}

    before = methods()
    with pytest.raises(RuntimeError, match="inside"):
        with B.flight_slot_spy() as calls:
            assert methods() != before and calls == []
            raise RuntimeError("inside")
    assert methods() == before


# --------------------------------------------------------------------------- #
# A slot call whose arguments change
# --------------------------------------------------------------------------- #

def _takes_view(orig):
    """The slot's side: ``next_stop`` declares one more keyword, with a default it ignores."""
    def next_stop(self, remainder, state, *, fits, pass_kind, after_stop, view=None):
        return orig(self, remainder, state, fits=fits, pass_kind=pass_kind, after_stop=after_stop)
    return next_stop


def _passes_view(method):
    """The mule's side: every ``next_stop`` call passes that keyword."""
    def next_stop(self, *args, **kwargs):
        return method(self, *args, view=None, **kwargs)
    return next_stop


def _defaults_after_stop(orig):
    """The slot's side: ``after_stop`` defaults to its takeoff value."""
    def next_stop(self, remainder, state, *, fits, pass_kind, after_stop=False):
        return orig(self, remainder, state, fits=fits, pass_kind=pass_kind, after_stop=after_stop)
    return next_stop


def _omits_after_stop(method):
    """The mule's side: no ``next_stop`` call passes ``after_stop``."""
    def next_stop(self, *args, **kwargs):
        kwargs.pop("after_stop")
        return method(self, *args, **kwargs)
    return next_stop


def _band_takes_stop(orig):
    """The slot's side: ``band_at_arrival`` declares a keyword it ignores."""
    def band_at_arrival(self, view, *, pass_kind, stop=None):
        return orig(self, view, pass_kind=pass_kind)
    return band_at_arrival


def _band_passes_stop(method):
    """The mule's side: every ``band_at_arrival`` call passes it."""
    def band_at_arrival(self, *args, **kwargs):
        return method(self, *args, stop=0, **kwargs)
    return band_at_arrival


#: case -> (trial, method, the slot's side, the mule's side, the argument, added).
#: The removal runs on F only: FX reads ``after_stop``, so its flight would change.
CALL_CHANGES = {
    "f_next_stop_view_added": ("f_45s", "next_stop", _takes_view, _passes_view, "view", True),
    "fx_n12_next_stop_view_added": (
        "fx_n12_120s", "next_stop", _takes_view, _passes_view, "view", True),
    "fx_band_at_arrival_stop_added": (
        "fx_45s", "band_at_arrival", _band_takes_stop, _band_passes_stop, "stop", True),
    "f_next_stop_after_stop_left_out": (
        "f_45s", "next_stop", _defaults_after_stop, _omits_after_stop, "after_stop", False),
}


def _trial_with(name, method, slot_side, mule_side, monkeypatch) -> Dict[str, Any]:
    """Trial ``name`` with the mule's calls of ``method`` changed, no file edited.

    Both fillings' ``method`` becomes ``slot_side`` of it before the spy wraps
    it, and ``mule_side`` of the spy's wrapper sits between the mule and the
    spy: the slot accepts the changed call, as both halves of a Phase 5 edit
    (the slot's signature and its call site) would make it.
    """
    from hermes.scheduler.policies import cross_heuristic as ch

    slots = (ch.CommittedSlot, ch.CrossHeuristic)
    for cls in slots:
        monkeypatch.setattr(cls, method, slot_side(cls.__dict__[method]))
    settings, cell = B.TRIALS[name]
    with UG4.in_process_orchestrator(), B.flight_slot_spy() as calls:
        spied = {cls: cls.__dict__[method] for cls in slots}
        try:
            for cls, wrapper in spied.items():
                setattr(cls, method, mule_side(wrapper))
            UG4.InProcessOrchestrator.last = None
            row_ = dict(Exp4Driver(**settings).run_trial(cell))
            orch = UG4.InProcessOrchestrator.last
        finally:
            for cls, wrapper in spied.items():
                setattr(cls, method, wrapper)
    return B.case_of(settings, cell, row_, orch, calls)


@pytest.mark.parametrize("case", sorted(CALL_CHANGES))
def test_a_slot_call_whose_arguments_change_flies_on_and_fails_by_name(case, monkeypatch):
    """The Phase 5 spec, other choices 1: the committed and FX slots keep their
    arguments. Here the slot accepts the changed call (a keyword it declares
    with a default, or a 386c275 argument it now defaults), so the trial flies
    as at 386c275, part for part, and critic A3's rule alone would pass an
    added argument; ``compare_part`` fails each call that changed, by the
    argument's name, instead of the mule failing on a ``TypeError``."""
    name, method, slot_side, mule_side, argument, added = CALL_CHANGES[case]
    current = _trial_with(name, method, slot_side, mule_side, monkeypatch)
    for part in UG4.PARTS:
        assert _canon.diff(CASES[name][part], current[part]) == [], part
    golden, slot = CASES[name]["flight_slot"], current["flight_slot"]
    changed = [i for i, c in enumerate(golden) if c["call"] == method]
    assert len(slot) == len(golden) and changed
    problems = B.compare_part("flight_slot", golden, slot)
    assert problems
    if added:
        assert _canon.diff(golden, slot) == []
        lines = B.added_slot_arguments(slot, limit=len(slot))
        assert problems == lines[:25]
        assert [int(re.match(r"\$\[(\d+)\]", ln).group(1)) for ln in lines] == changed
        assert all(f"was passed ['{argument}']" in ln for ln in lines)
    else:
        assert problems == [f"$[{i}].{argument}: missing in current "
                            f"(golden {json.dumps(golden[i][argument])})" for i in changed]


def test_a_call_the_slot_refuses_is_refused_by_the_slot_itself():
    """The spy takes what the method takes: a keyword the slot does not declare
    reaches the slot, which refuses it with its own error, as it would without
    the spy (a trial's mule fails on it); an error in the slot's body passes
    through as it is. Neither call is recorded."""
    from hermes.scheduler.policies import cross_heuristic as ch

    def errors(cls):
        out = []
        for kw in ({"no_such_argument": None}, {}):
            try:
                cls().next_stop([], None, fits=None, pass_kind="collect", after_stop=False, **kw)
            except (TypeError, ValueError) as exc:
                out.append((type(exc), str(exc)))
        return out

    for cls in (ch.CommittedSlot, ch.CrossHeuristic):
        bare = errors(cls)
        assert [t for t, _ in bare] == [TypeError, ValueError]
        assert "unexpected keyword argument 'no_such_argument'" in bare[0][1]
        with B.flight_slot_spy() as calls:
            assert errors(cls) == bare
        assert calls == []


# --------------------------------------------------------------------------- #
# make_baseline.py: the third baseline, recorded at 386c275
# --------------------------------------------------------------------------- #

def test_the_386c275_base_is_the_third_recorded_baseline():
    assert list(MB.BASES)[:3] == ["afa9526", "6e6f92d", "386c275"]
    base = MB.BASES["386c275"]
    assert base.commit == B.BASE_COMMIT
    assert base.path == MB.HERE / "pytest_baseline_386c275.txt"
    assert (base.unit, base.phase) == ("FeRRy Phase 5 unit UG5", "Phase 5")
    assert MB.base_of("386c275") is base and MB.base_of(B.BASE_COMMIT) is base


def _junit(path, cases):
    """A JUnit XML of ``(name, kind, message, text)`` cases of this module."""
    suite = ET.Element("testsuite", time="2.5")
    for name, kind, message, text in cases:
        tc = ET.SubElement(suite, "testcase", classname="tests.golden.test_golden_p4_plan",
                           name=name)
        if kind == "failure":
            ET.SubElement(tc, kind, message=message).text = text
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def test_write_records_the_386c275_base_only_at_its_commit(tmp_path, monkeypatch, capsys):
    xml = _junit(tmp_path / "run.xml", [
        ("test_a", "passed", "", ""),
        ("test_b", "failure", "AssertionError: x\nassert 1 == 2",
         "tests\\golden\\test_golden_p4_plan.py:7: AssertionError"),
    ])
    head = {"sha": MB.BASES["6e6f92d"].commit}
    monkeypatch.setattr(MB, "_git", lambda *a: head["sha"] if a[0] == "rev-parse" else "")
    out = tmp_path / "b.txt"
    assert MB.main(["write", str(xml), "--base", "386c275", "--baseline", str(out)]) == 2
    assert "refusing to write: the baseline is 386c275's" in capsys.readouterr().err
    head["sha"] = B.BASE_COMMIT
    assert MB.main(["write", str(xml), "--base", "386c275", "--baseline", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# FeRRy Phase 5 unit UG5 - pytest baseline at 386c275 (main), "
                           "before any Phase 5 change.\n")
    assert MB.baseline_label(out) == "386c275"
    assert MB.read_baseline(out) == MB.results_from_junit(xml)[0]


def test_compare_checks_the_386c275_baseline_by_default(tmp_path, monkeypatch, capsys):
    ok = _junit(tmp_path / "ok.xml", [("test_a", "passed", "", "")])
    bad = _junit(tmp_path / "bad.xml", [("test_a", "failure", "AssertionError: x\nassert 0",
                                         "tests\\golden\\test_golden_p4_plan.py:9: AssertionError")])
    results, _ = MB.results_from_junit(ok)
    bases = {}
    for key, base in MB.BASES.items():
        path = tmp_path / base.path.name
        path.write_text(MB.render_baseline(results, suite_time=1.0, head=base.commit),
                        encoding="utf-8")
        bases[key] = base._replace(path=path)
    monkeypatch.setattr(MB, "BASES", bases)
    assert MB.main(["compare", str(ok)]) == 0
    text = capsys.readouterr().out
    assert "== the 386c275 baseline (pytest_baseline_386c275.txt)" in text
    assert text.count("same as the") == len(bases) and "same as the 386c275 baseline" in text
    assert MB.main(["compare", str(bad), "--base", "386c275"]) == 1
    text = capsys.readouterr().out
    assert "DIFFERS from the 386c275 baseline" in text and "6e6f92d" not in text
    bases["386c275"] = bases["386c275"]._replace(path=tmp_path / "missing.txt")
    assert MB.main(["compare", str(ok)]) == MB.MISSING_STATUS
    assert "MISSING: " in capsys.readouterr().out


def test_the_386c275_baseline_is_recorded_and_gates_phase_4():
    """The third baseline is on disk (a checkout without it fails here, and
    ``compare`` exits 2), says it was recorded at 386c275, agrees with its own
    header, keeps every 6e6f92d test, holds every Phase 4 test file and these
    goldens, and fails exactly the 6e6f92d baseline's deterministic failures,
    with their signatures (the flaky smoke test passes or fails its known way)."""
    path = MB.BASES["386c275"].path
    assert path.is_file(), f"{path} is missing: it is recorded at 386c275 and must be committed"
    assert MB.baseline_label(path) == "386c275"
    b386, e6f = MB.read_baseline(path), MB.read_baseline(MB.BASES["6e6f92d"].path)
    totals = re.search(r"^# Suite time: .* Totals: (.*), total=(\d+)$",
                       path.read_text(encoding="utf-8"), re.M)
    assert totals and int(totals.group(2)) == len(b386)
    assert {k: int(v) for k, v in (kv.split("=") for kv in totals.group(1).split(", "))} == (
        Counter(r.outcome for r in b386.values()))
    assert set(e6f) <= set(b386)
    phase4 = {p.relative_to(MB.REPO).as_posix() for p in (MB.REPO / "tests").rglob("test_p4_*.py")}
    files = {nid.split("::")[0] for nid in b386}
    assert phase4 and phase4 <= files
    assert "tests/golden/test_golden_p4_plan.py" in files

    def deterministic_failures(results):
        return {nid: r.signature for nid, r in results.items()
                if r.outcome in ("failed", "error") and nid not in MB.FLAKY}

    assert deterministic_failures(b386) == deterministic_failures(e6f)
    assert len(deterministic_failures(b386)) == 5
    for nid, known in MB.FLAKY.items():
        assert b386[nid].outcome == "passed" or (
            b386[nid].outcome == "failed" and b386[nid].signature in known), nid


#: Phase 4's tests that pin what no trial here shows: the beacon hook's place
#: between the departure check and the slot's pick, and its offers; the plan's
#: exempt stops held protected in flight (``FLScheduler.plan_protected``, which
#: U4's ``fits_after_service`` reuses); and K = 2 (Phase 5 critic A4).
PINNED_BY_PHASE_4 = tuple(
    f"tests/{path}::{test}" for path, test in (
        ("integration/test_p4_plan_missions.py",
         "test_the_departure_check_then_the_beacon_hook_then_the_slot"),
        ("integration/test_p4_plan_missions.py",
         "test_the_beacon_hook_groups_an_offer_within_the_committed_classs_range"),
        ("integration/test_p4_plan_missions.py",
         "test_in_flight_the_exempt_stops_are_protected_and_the_re_plan_dates_each_member"),
        ("integration/test_p4_plan_missions.py",
         "test_fx_fits_is_the_departure_checks_fold_of_the_reordered_rest"),
        ("integration/test_p4_plan_missions.py", "test_capped_members_do_not_lower_deliver_by"),
        ("unit/test_p4_fl_scheduler_plan.py",
         "test_plan_protected_is_the_exempt_stops_of_what_remains"),
        ("unit/test_p4_fl_scheduler_plan.py", "test_the_plan_mode_re_plan_is_the_member_trim"),
        ("integration/test_p4_plan_missions.py",
         "test_two_mules_each_plan_their_own_slice_from_their_own_ages"),
    )
)


def test_what_no_trial_reaches_is_pinned_by_phase_4s_tests_under_the_386c275_baseline():
    """No trial has a beacon offer: at 386c275 nothing outside the tests calls
    ``MuleSupervisor.offer_contact``, so the hook runs at every Pass-1
    departure with an empty queue and its place in the loop never shows here.
    The exempt stops are protected in seven of the trials but never decisively
    (the same trials, part for part, with ``plan_protected`` returning
    nothing), and there is no K = 2 trial. The third baseline records each of
    Phase 4's tests that pin them as passing, so its gate holds them."""
    for name in B.TRIAL_NAMES:
        for m in missions(name):
            assert (m["inserts"], m["offers_refused"]) == ([], []), (name, m["mission_round"])
    b386 = MB.read_baseline(MB.BASES["386c275"].path)
    for nid in PINNED_BY_PHASE_4:
        assert b386[nid].outcome == "passed", nid
