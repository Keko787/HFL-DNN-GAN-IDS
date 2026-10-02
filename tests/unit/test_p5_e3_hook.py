"""FeRRy Phase 5 (unit U5): arm E3's per-departure hook in the mule supervisor.

Arm E3 (after Chen et al.; the user's decision 7 (a)) is a legacy-mode
whole-scheduler policy that names each Pass-1 stop in flight
(``policies/next_stop.py``'s protocol; the Phase 5 spec, other choices 11). Its
policy (unit U6) is not built yet, so a stub that declares
``chooses_next_stop`` stands in for it here, flying real ``ClientMission``
devices, a real cluster and a real ``MuleSupervisor`` on the mission clock
(``tests/integration/_ferry_harness.py``). Pinned (the Phase 5 spec's units
table, row U5):

* **When it is asked.** At takeoff and at every Pass-1 departure, after the
  departure check and the beacon hook, in that order (E3's own check is
  ``none``, so the check keeps the remainder as it is), never in Pass 2,
  never on the wall clock; the mule flies the stop it names, a beacon insert
  included, which the policy is offered from the departure that inserted it.
* **What it sees.** Chen's observation from the departure
  (``FerryRuntime.e3_observation``) and ``admissible(i)``, S3b's single-contact
  predicate under the budget rule with the return and the upload priced. The
  observation's N is the mule's slice with every device planned or inserted
  outside it: a slice device the route leaves out still counts, and an insert
  counts from its insertion.
* **None ends the pass.** The stops left are reported
  (``pass_1_e3_unvisited``) and never widened; an answer off the protocol is
  refused; each call is recorded (``pass_1_e3``).
* **The D arms are unaffected**, and only E3's path loads the protocol module.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes.mule.ferry import FerrySpec
from hermes.scheduler.policies import FedExCarpPolicy, MaxAoIPolicy
from hermes.scheduler.policies.next_stop import E3View, pass_1_only
from hermes.scheduler.stages.s3b_feasibility import (
    REASON_OVERDUE,
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState, MissionPass

from tests.golden import _canon
from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

logging.getLogger("hermes.mission.client_mission").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parents[2]
COLLECT = MissionPass.COLLECT
DOCK = (0.0, 0.0, 0.0)
#: The deadline law's unit: deadlines 60 000 s out, so no deadline enters (E3
#: has no deadline gate, and the D arms' in-flight check is the budget alone).
FAR_UNIT = 1000.0
#: GH.LAYOUT's seven devices make five wide stops: {dev-00, dev-01} at the
#: dock, {dev-02, dev-03}, dev-04, dev-05 and dev-06.
N_STOPS = 5
#: Two devices outside the mule's slice, on its link: beacon offers.
EXTRA = {"x": (30.0, 20.0, 0.0), "y": (-20.0, 35.0, 0.0)}


class StubE3:
    """A whole-scheduler policy that names each Pass-1 stop in flight (E3's protocol).

    Every S3a contact is admitted, nearest first, as U6's policy will admit
    them (the Phase 5 spec, other choices 11); the in-flight check is ``none``.
    ``next_stop`` takes the last admissible stop, so its flight differs from
    the queue's order, and records each call. ``answer`` overrides the answer
    to test the supervisor's refusals.
    """

    name = "stub_e3"
    in_flight_check = "none"
    admits_member_subsets = False
    chooses_next_stop = True

    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    def admit_and_order(self, contacts, device_states, env, *, mission_deadline_ts=None,
                        feasibility_model=None, **kw):
        pose = tuple(env.mule_pose)
        return sorted(contacts, key=lambda wp: (math.dist(pose, wp.position), tuple(wp.position)))

    def next_stop(self, remainder, state, *, view, admissible, pass_kind, after_stop):
        pass_1_only(pass_kind)
        mask = [admissible(i) for i in range(len(remainder))]
        self.calls.append(SimpleNamespace(remainder=list(remainder), state=state, view=view,
                                          mask=mask, pass_kind=pass_kind, after_stop=after_stop))
        if self.answer is not None:
            return self.answer(remainder, mask)
        fit = [i for i, ok in enumerate(mask) if ok]
        return fit[-1] if fit else None


def spec(**kw):
    """The pilots' wide contact band, 1 MB, ``replan`` with ``trim``."""
    kw.setdefault("payload_bytes", 1_000_000)
    return FerrySpec.from_config(rf_range_m=60.0, seed=7, contact_band="wide",
                                 in_flight_response="replan", replan_fallback="trim", **kw)


def fly(policy, *, budget, missions=1, sim=True, unit=FAR_UNIT, before=None, world_kw=None):
    """Run ``missions`` missions of a legacy mule flying ``policy``; one record
    each. ``before(w, sup, m)`` runs before each mission's takeoff."""
    w = H.World(layout=GH.LAYOUT, flaky={}, **(world_kw or {}))
    mid = w.mule_ids[0]
    out = []
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=sim, ferry=spec() if sim else None, mission_budget_s=budget,
                           deadline_time_scale=unit, target_selector=policy)
        w.bootstrap()
        for m in range(missions):
            if before is not None:
                before(w, sup, m)
            out.append(SimpleNamespace(result=sup.run_one_mission(), deltas=w.take_deltas(mid)))
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return w, sup, out


def widened(rec):
    """The devices given a synthetic TIMEOUT this mission (widening)."""
    return sorted({str(d.device_id) for src, d in rec.deltas if src == "direct" and d.synthetic})


def devices(stops):
    return [[str(d) for d in wp.devices] for wp in stops]


def calls_of(policy, k, n_missions):
    """The policy's calls split by mission: a call at takeoff opens a mission."""
    out, cur = [], None
    for c in policy.calls:
        if not c.after_stop:
            cur = []
            out.append(cur)
        cur.append(c)
    assert len(out) == n_missions
    return out[k]


# --------------------------------------------------------------------------- #
# When the hook asks, and what the mule flies
# --------------------------------------------------------------------------- #

def test_e3_names_each_pass_1_stop_at_takeoff_and_every_departure_only():
    """Other choices 11 (critic B7 i): the policy is asked at takeoff
    (``after_stop`` False) and at every Pass-1 departure, never in Pass 2,
    which flies every slice stop in the queue's order; the mule flies the stop
    it names. With a 400 s budget every stop stays admissible, the pass ends
    when none is left, and nothing is reported unvisited."""
    policy = StubE3()
    _, sup, recs = fly(policy, budget=400.0, missions=2)
    assert len(policy.calls) == 2 * N_STOPS
    for k, rec in enumerate(recs):
        r = rec.result
        calls = calls_of(policy, k, 2)
        assert [c.after_stop for c in calls] == [False] + [True] * (N_STOPS - 1)
        assert all(c.pass_kind is COLLECT for c in calls)
        assert len(r.pass_1_flown) == len(calls) == N_STOPS
        for c, flown in zip(calls, r.pass_1_flown):
            pick = max(i for i, ok in enumerate(c.mask) if ok)
            assert flown["devices"] == [str(d) for d in c.remainder[pick].devices]
            assert c.state.clock == flown["depart_s"]
        assert [s["devices"] for s in r.pass_1_flown] != devices(r.pass_1_queue)
        assert [s["devices"] for s in r.pass_2_flown] == devices(r.pass_1_queue)
        assert r.pass_1_e3_unvisited is None and widened(rec) == []
        assert r.pass_1_pairs is None and r.pass_1_policy_drops is None


def test_each_call_is_recorded_json_ready_and_without_a_wall_time():
    """``pass_1_e3``: one record per call, in order, of what the policy was
    offered and what it named: the stops left, the predicate's verdict on each,
    and the index flown (None and ``home`` when the pass ended)."""
    policy = StubE3()
    _, _, (rec,) = fly(policy, budget=200.0)
    r = rec.result
    assert len(r.pass_1_e3) == len(policy.calls) == len(r.pass_1_flown) + 1
    for c, entry in zip(policy.calls, r.pass_1_e3):
        fit = [i for i, ok in enumerate(c.mask) if ok]
        index = fit[-1] if fit else None
        assert entry == {
            "t_s": c.state.clock, "after_stop": c.after_stop, "stops": devices(c.remainder),
            "admissible": c.mask, "next_index": index,
            "next": "home" if index is None else devices([c.remainder[index]])[0],
        }
    assert json.loads(json.dumps(r.pass_1_e3)) == r.pass_1_e3
    assert not any("wall" in key for entry in r.pass_1_e3 for key in entry)


@pytest.mark.parametrize("budget, n_calls, ends_by_none", [
    (400.0, N_STOPS, False),
    (200.0, 4, True),
], ids=["every_stop_flown", "none_ends_the_pass"])
def test_e3_picks_after_the_departure_check_and_the_beacon_hook(budget, n_calls, ends_by_none):
    """Other choices 11, in the order Phase 4 pinned for the flight slot: at
    takeoff and at every Pass-1 departure the departure check runs first (the
    arm's own; E3's is ``none``, which keeps the remainder as it is), then the
    beacon hook, then the policy names the stop. The last departure runs the
    beacon hook alone, unless the policy's None has ended the pass there.
    Pass 2 runs neither the hook nor the policy (and, with no Pass-2 budget,
    no check)."""
    order = []

    class Ordered(StubE3):
        def next_stop(self, remainder, state, **kw):
            index = super().next_stop(remainder, state, **kw)
            order.append(("e3", kw["after_stop"]))
            return index

    def spies(w, sup, m):
        real_check, real_offers = sup._ferry_departure, sup._take_offers

        def check(*a, **kw):
            order.append(("check", kw["pass_kind"]))
            return real_check(*a, **kw)

        def offers(*a, **kw):
            order.append(("beacon",))
            return real_offers(*a, **kw)

        sup._ferry_departure, sup._take_offers = check, offers

    _, _, (rec,) = fly(Ordered(), budget=budget, before=spies)
    r = rec.result
    expected = []
    for k in range(n_calls):
        expected += [("check", COLLECT), ("beacon",), ("e3", k > 0)]
    if not ends_by_none:
        expected.append(("beacon",))
    assert order == expected
    assert len(r.pass_1_e3) == n_calls
    assert (r.pass_1_e3[-1]["next_index"] is None) == ends_by_none
    assert len(r.pass_2_flown) == N_STOPS


def _offer(device):
    return ContactWaypoint(position=DOCK, devices=(DeviceID(device),),
                           bucket=Bucket.BEACON_ACTIVE, deadline_ts=0.0)


def _offer_x_and_y(w, sup, m):
    """x is offered before takeoff, y during the mission's first contact (a
    beacon heard in flight); both are tracked, outside the mule's slice."""
    for did, pos in EXTRA.items():
        sup.scheduler.device_states[DeviceID(did)] = DeviceSchedulerState(
            device_id=DeviceID(did), last_known_position=pos)
    sup.offer_contact(_offer("x"))
    real, calls = sup.mission.run_contact, []

    def contact(*a, **kw):
        calls.append(None)
        if len(calls) == 1:
            sup.offer_contact(_offer("y"))
        return real(*a, **kw)

    sup.mission.run_contact = contact


def test_e3_is_offered_the_beacon_hooks_inserts_and_flies_them():
    """The beacon hook settles the stops left before the policy picks: x,
    offered before takeoff, is inserted at takeoff and is among the stops the
    takeoff call is offered; y, heard during the first contact, is inserted at
    the next departure and offered from that call on. Every call's record
    holds the stops it was offered, inserts included, and the mule flies both
    inserts (E3's check is ``none``, so every insert fits)."""
    policy = StubE3()
    _, _, (rec,) = fly(policy, budget=400.0, before=_offer_x_and_y,
                       world_kw={"extra_devices": EXTRA})
    r = rec.result
    assert [(i["devices"], i["t_s"]) for i in r.inserts] == [
        (["x"], r.sim_start_s), (["y"], r.pass_1_flown[1]["depart_s"])]
    assert r.offers_refused == []
    offered = [devices(c.remainder) for c in policy.calls]
    assert ["x"] in offered[0] and ["y"] not in offered[0]
    assert policy.calls[1].after_stop and ["y"] in offered[1]
    assert [entry["stops"] for entry in r.pass_1_e3] == offered
    flown = [stop["devices"] for stop in r.pass_1_flown]
    assert len(flown) == N_STOPS + 2 and ["x"] in flown and ["y"] in flown
    assert len(policy.calls) == len(flown)


# --------------------------------------------------------------------------- #
# What the policy sees
# --------------------------------------------------------------------------- #

def test_the_view_is_chens_observation_from_the_departure():
    """The observation is ``FerryRuntime.e3_observation`` of the stops left, from
    the departure's pose and time on the contact band, with the sortie's fields
    from the mule: N the slice (seven devices), the budget's end and length,
    and the energy spent this sortie. It is pure, so recomputed afterwards from
    the same arguments it is equal, which pins the supervisor's arguments; the
    updates collected so far leave every candidate's share at 1 (critic B7 ii)."""
    budget = 400.0
    policy = StubE3()
    _, sup, (rec,) = fly(policy, budget=budget)
    r, fx = rec.result, sup._ferry_run
    end = r.sim_start_s + budget
    positions = sup._ferry_positions(r.pass_1_queue)
    clean = {line.device_id: line.contact_ts for line in r.report.lines
             if line.outcome.is_on_time()}
    for c in policy.calls:
        view, state = c.view, c.state
        assert isinstance(view, E3View)
        collected = frozenset(d for d, t in clean.items() if t <= state.clock)
        again = fx.e3_observation(state.pose, c.remainder, positions, state.clock, demand=7,
                                  budget_end=end, budget_s=budget, energy_j=state.energy_j,
                                  collected=collected)
        assert view == again
        assert (view.band, view.demand, view.clock_s) == ("wide", 7, state.clock)
        assert (view.budget_end, view.budget_s, view.energy_j) == (end, budget, state.energy_j)
        for stop, wp in zip(view.stops, c.remainder):
            assert stop.members == len(wp.devices) and stop.remaining == 1.0
            assert (stop.dx_m, stop.dy_m) == (wp.position[0] - state.pose[0],
                                              wp.position[1] - state.pose[1])
    assert any(c.view.energy_j > 0.0 for c in policy.calls)


class _LeavesDev06Out(StubE3):
    """Admits every contact but dev-06's, the farthest: a slice device the route
    leaves out, which the trace reports as a policy drop and never widens."""

    def admit_and_order(self, contacts, device_states, env, **kw):
        route = super().admit_and_order(contacts, device_states, env, **kw)
        return [wp for wp in route if DeviceID("dev-06") not in wp.devices]


def test_n_is_the_slice_with_every_device_planned_or_inserted_outside_it():
    """E3's reward is |C_k| / N (the design's bytes reward). Legacy mode has no
    plan demand, so the supervisor reads N as every device the sortie answers
    for (U5's reading, for UD to record): the mule's slice (seven devices),
    with every device planned or inserted outside it. A slice device the route
    leaves out still counts: dev-06's stop is left out, six devices are
    planned and N stays seven. An insert counts from its insertion: x,
    inserted at takeoff, from the takeoff call, and y, inserted at the
    departure after the first stop, from that call."""
    policy = _LeavesDev06Out()
    _, _, (rec,) = fly(policy, budget=400.0)
    r = rec.result
    assert sum(len(wp.devices) for wp in r.pass_1_queue) == 6
    assert [drop["devices"] for drop in r.pass_1_policy_drops] == [["dev-06"]]
    assert widened(rec) == []
    assert [c.view.demand for c in policy.calls] == [7] * (N_STOPS - 1)

    policy = StubE3()
    _, _, (rec,) = fly(policy, budget=400.0, before=_offer_x_and_y,
                       world_kw={"extra_devices": EXTRA})
    assert [i["devices"] for i in rec.result.inserts] == [["x"], ["y"]]
    assert [c.view.demand for c in policy.calls] == [8] + [9] * (len(policy.calls) - 1)


@pytest.mark.parametrize("budget", (200.0, 120.0, 80.0))
def test_admissible_is_s3bs_single_contact_predicate_landing_included(budget):
    """Chen's safety controller is S3b's single-contact predicate under the
    budget rule (``FeasibilityModel.admit``, ``RULE_BUDGET``) from the
    departure's state: no deadline enters, and the stop's return leg and the
    upload are priced, so a stop whose service ends within the budget but
    whose landing does not is refused."""
    policy = StubE3()
    _, sup, (rec,) = fly(policy, budget=budget)
    model = sup.scheduler.feasibility_model
    end = rec.result.sim_start_s + budget
    by_landing = 0
    for c in policy.calls:
        verdicts = [model.admit(c.state, wp, rule=RULE_BUDGET, budget_end=end, pass_kind=COLLECT)
                    for wp in c.remainder]
        assert c.mask == [v.ok for v in verdicts]
        by_landing += sum(1 for v in verdicts if not v.ok and v.finish <= end < v.home)
    assert by_landing > 0


def test_no_deadline_gates_e3():
    """E3 flies none of FeRRy's deadline machinery (build plan L1026): with every
    deadline 60 s after takeoff (the recorded unit), stops S3b's deadline rule
    refuses as overdue are admissible to E3, and are flown."""
    budget = 400.0
    policy = StubE3()
    _, sup, (rec,) = fly(policy, budget=budget, unit=1.0)
    model = sup.scheduler.feasibility_model
    end = rec.result.sim_start_s + budget
    overdue = 0
    for c in policy.calls:
        for wp, ok in zip(c.remainder, c.mask):
            strict = model.admit(c.state, wp, rule=RULE_DEADLINE_BUDGET, budget_end=end,
                                 pass_kind=COLLECT)
            if strict.reason == REASON_OVERDUE:
                assert ok
                overdue += 1
    assert overdue > 0
    assert len(rec.result.pass_1_flown) == N_STOPS


# --------------------------------------------------------------------------- #
# None ends the pass; the rest is reported, never widened
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("budget, flown, left", [
    (200.0, 3, [["dev-04"], ["dev-05"]]),
    (120.0, 2, [["dev-04"], ["dev-05"], ["dev-02", "dev-03"]]),
    (80.0, 2, [["dev-04"], ["dev-05"], ["dev-06"]]),
])
def test_none_ends_the_pass_and_the_stops_left_are_reported_never_widened(budget, flown, left):
    """Other choices 11: when no stop is admissible the policy answers None, the
    pass ends and the mule flies home. The stops left go to
    ``pass_1_e3_unvisited`` (as a baseline's drops are reported, the user's
    decision 6) and no device gets a synthetic TIMEOUT for them; Pass 2 still
    delivers to every slice stop."""
    policy = StubE3()
    _, sup, (rec,) = fly(policy, budget=budget)
    r = rec.result
    last = policy.calls[-1]
    assert len(r.pass_1_flown) == flown and not any(last.mask)
    assert r.pass_1_e3[-1]["next_index"] is None and r.pass_1_e3[-1]["next"] == "home"
    assert devices(last.remainder) == left
    assert r.pass_1_e3_unvisited == [
        {"position": [float(c) for c in wp.position], "devices": [str(d) for d in wp.devices],
         "deadline_ts": float(wp.deadline_ts), "widened": False} for wp in last.remainder]
    assert widened(rec) == []
    assert all(sup.scheduler.device_states[DeviceID(d)].miss_streak == 0
               for stop in left for d in stop)
    assert len(r.pass_2_flown) == N_STOPS
    assert r.pass_1_preflight_drops == [] and r.replans == [] and r.aborts == []


@pytest.mark.parametrize("answer, error, match", [
    (lambda rest, mask: next(i for i, ok in enumerate(mask) if not ok), ValueError,
     "not admissible"),
    (lambda rest, mask: None, ValueError, "only when no stop is admissible"),
    (lambda rest, mask: len(rest), ValueError, "outside the remainder"),
    (lambda rest, mask: True, TypeError, "remainder index or None"),
], ids=["inadmissible_stop", "none_while_a_stop_fits", "past_the_remainder", "bool"])
def test_an_answer_off_the_protocol_is_refused(answer, error, match):
    """``next_stop.checked_choice`` holds the answer: a policy can neither fly a
    stop the safety controller refuses nor end the pass by choice. At 80 s the
    farthest stop is inadmissible at takeoff, so each bad answer is reachable
    at the first call."""
    with pytest.raises(error, match=match):
        fly(StubE3(answer=answer), budget=80.0)


def test_admissible_refuses_an_index_off_the_remainder():
    """``admissible(i)`` answers for the stops left only, so a policy cannot read
    a negative index as one from the end."""
    class Probing(StubE3):
        def next_stop(self, remainder, state, *, admissible, **kw):
            admissible(-1)

    with pytest.raises(ValueError, match="stop.s. left"):
        fly(Probing(), budget=400.0)


# --------------------------------------------------------------------------- #
# The D arms, the wall clock, and what loads
# --------------------------------------------------------------------------- #

class _DeclaresFalse(MaxAoIPolicy):
    chooses_next_stop = False


class _DeclaresTruthy(MaxAoIPolicy):
    chooses_next_stop = 1


def _canonical(results):
    return [_canon.canon(r) for r in results]


def test_the_d_arms_fly_their_recorded_path():
    """Freeze Rule 1: no D policy declares ``chooses_next_stop``, so D1 and D4
    fly as recorded with no E3 field; a policy that declares it False, or a
    truthy value that is not True, flies exactly as D1 does, so a stand-in
    whose attributes are all truthy is not taken for E3."""
    d1 = [rec.result for rec in fly(MaxAoIPolicy(), budget=70.0, missions=2)[2]]
    for r in d1 + [rec.result for rec in fly(FedExCarpPolicy(depot=DOCK), budget=70.0)[2]]:
        assert (r.pass_1_e3, r.pass_1_e3_unvisited, r.pass_1_pairs) == (None, None, None)
    assert any(r.pass_1_policy_drops for r in d1)
    for policy in (_DeclaresFalse(), _DeclaresTruthy()):
        again = [rec.result for rec in fly(policy, budget=70.0, missions=2)[2]]
        assert _canonical(again) == _canonical(d1)


def test_on_the_wall_clock_the_policy_is_never_asked():
    """The hook is the mission clock's (E3 runs on the simulated clock only):
    on the wall clock the recorded two-pass path flies the policy's route as
    planned, and nothing is recorded."""
    policy = StubE3()
    _, _, (rec,) = fly(policy, budget=400.0, sim=False)
    assert policy.calls == []
    assert rec.result.pass_1_e3 is None and rec.result.pass_1_e3_unvisited is None


_LOADS = r"""
import sys
from hermes.mule.ferry import FerrySpec
from hermes.scheduler.policies import FedExCarpPolicy, MaxAoIPolicy
from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H


class StubE3:
    name, in_flight_check, admits_member_subsets, chooses_next_stop = (
        "stub_e3", "none", False, True)

    def admit_and_order(self, contacts, device_states, env, **kw):
        return list(contacts)

    def next_stop(self, remainder, state, *, view, admissible, pass_kind, after_stop):
        fit = [i for i in range(len(remainder)) if admissible(i)]
        return fit[-1] if fit else None


def fly(policy, budget):
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=7, contact_band="wide",
                                 payload_bytes=1_000_000, in_flight_response="replan",
                                 replan_fallback="trim")
    w = H.World(layout=GH.LAYOUT, flaky={})
    with H.Patched(w.clock):
        sup = w.supervisor(w.mule_ids[0], sim=True, ferry=spec, mission_budget_s=budget,
                           deadline_time_scale=1000.0, target_selector=policy)
        w.bootstrap()
        return [sup.run_one_mission() for _ in range(2)]


rs = [r for policy in (None, MaxAoIPolicy(), FedExCarpPolicy(depot=(0.0, 0.0, 0.0)))
      for r in fly(policy, 70.0)]
assert any(r.pass_1_policy_drops for r in rs) and all(r.pass_1_e3 is None for r in rs)
assert "hermes.scheduler.policies.next_stop" not in sys.modules
assert any(r.pass_1_e3_unvisited for r in fly(StubE3(), 200.0))
assert "hermes.scheduler.policies.next_stop" in sys.modules
loaded = sorted(m for m in sys.modules if m.startswith("hermes.scheduler.plan")
                or m.endswith(("pair_slot", "pair_q", "pair_replay", "pair_features",
                               "chen_dqn", "cross_heuristic")))
assert not loaded, loaded
print("ok")
"""


def test_only_e3s_path_loads_the_next_stop_protocol():
    """Critic B7 iii: H1, D1 and D4 on the clock load neither the protocol module
    nor any plan or pair module, in a fresh interpreter; E3's path loads the
    protocol and still no plan module (E3 flies none of the plan's machinery)."""
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", _LOADS], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")
