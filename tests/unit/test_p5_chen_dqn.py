"""FeRRy Phase 5 (unit U6): arm E3, the numpy port of Chen et al.'s DQN recipe.

E3 (``hermes/scheduler/policies/chen_dqn.py``; the user's decision 7 (a); the
Phase 5 spec, other choices 11) is a legacy-mode whole-scheduler policy that
names each Pass-1 stop in flight through the per-departure protocol
(``policies/next_stop.py``) and the supervisor's hook (unit U5). Its missions
fly here through the integration harness's real ``ClientMission`` devices,
cluster and ``MuleSupervisor`` on the mission clock
(``tests/integration/_ferry_harness.py``): E3 is not a driver arm until unit
U7, so FerrySim cannot fly it yet. What is pinned (the spec's units table,
row U6, and critic B7):

* **Every contact admitted.** Before takeoff E3 admits every S3a contact,
  nearest the takeoff pose first, whatever the budget or the deadlines, and
  reads neither the device states nor the feasibility model: the scheduler
  reports no policy drop and S3b's gate does not run. Whole stops only: the
  scheduler refuses member subsets for E3.
* **Check ``none``.** The scheduler's in-flight rule for E3 is ``RULE_NONE``:
  the departure check never re-plans or aborts, under ``replan`` or under
  ``abort`` (the Exp 4 driver's default), and each call is offered the
  previous call's stops less the one flown.
* **The mask is S3b's ``admit`` under ``RULE_BUDGET``, landing included**
  (and the energy clause with a battery), recomputed independently from each
  call's state; E3 flies only an admissible stop, the masked argmax of its
  network's Q (ties to the lowest row), and ``admissible`` must answer bools.
* **The pass ends when nothing fits**: None, nothing scored or drawn, the
  stops left reported and never widened, Pass 2 unchanged.
* **No deadline gate**: stops S3b's deadline rule refuses as overdue are
  admissible to E3 and flown.
* **The band fixed**: every stop of both passes on the cell's contact band;
  a view on another band, and a checkpoint of another band, are refused.
* **The declared constant columns**: on a FerrySim-like sample (random N =
  6 and 12 layouts on the realism field, the jittery contact channel, 1 MB,
  FerrySim's budgets) ``remaining`` is 1 on every row and ``reachable`` 0 on
  most, as declared (critic B7 ii), and no other column is constant.
* **Checkpoints**: the E3 kind, schema and band round trip into the policy;
  any other sha, kind, schema or band is refused; a bootstrap loads; a
  trained checkpoint keeps its purpose and provenance, so the runner takes
  it; the manifest kept is the policy's own JSON copy.
* **Training**: ε-greedy over the admissible stops on the trainer's live
  network's online Q as it stands at each decision (not the target copy, not
  earlier weights), two draws per decision, every decision to the sink, a
  mission's steps exactly its Pass-1 stops flown and the pair learner's
  transitions as they are, the same seeds the same steps; no reference phase.
* **Pass 2 refused**; **layering**: no plan module, no recorded arm loads E3.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hermes.mule.ferry import FerrySpec
from hermes.processes import config as CFG
from hermes.scheduler.policies import chen_dqn as C
from hermes.scheduler.policies.budget_walk import IN_FLIGHT_NONE
from hermes.scheduler.policies.next_stop import E3Stop, E3View, NextStopPolicy
from hermes.scheduler.routing.replan import ORDER_NONE
from hermes.scheduler.selector import pair_q as Q
from hermes.scheduler.selector.pair_q import CheckpointError, PairQConfig, PairQNet
from hermes.scheduler.selector.pair_replay import PairBatch, PairTransition
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import (
    REASON_OVERDUE,
    RULE_BUDGET,
    RULE_DEADLINE_BUDGET,
    RULE_NONE,
    FlightState,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass

from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

logging.getLogger("hermes.mission.client_mission").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parents[2]
COLLECT = MissionPass.COLLECT
DELIVER = MissionPass.DELIVER
#: The deadline law's unit: deadlines 60 000 s out, so no deadline enters.
FAR_UNIT = 1000.0
#: GH.LAYOUT's seven devices make five wide stops (U5's E3 hook tests).
N_STOPS = 5
#: Each column's index in a row.
COL = {name: j for j, name in enumerate(C.E3_COLUMNS)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class Recorded(C.ChenDQNPolicy):
    """The policy, recording each call's arguments and answer (it decides nothing)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def next_stop(self, remainder, state, *, view, admissible, pass_kind, after_stop):
        answer = super().next_stop(remainder, state, view=view, admissible=admissible,
                                   pass_kind=pass_kind, after_stop=after_stop)
        self.calls.append(SimpleNamespace(
            remainder=list(remainder), state=state, view=view, pass_kind=pass_kind,
            mask=[admissible(i) for i in range(len(remainder))], after_stop=after_stop,
            answer=answer))
        return answer


def policy(seed=0, band="wide", record=True):
    """E3 around a fresh network (a checkpoint's weights in a real arm)."""
    make = Recorded if record else C.ChenDQNPolicy
    return make(C.new_e3_network(seed=seed), band=band)


def spec(*, band="wide", regime="clean", seed=7, capacity=None, response="replan"):
    """The pilots' configuration: 1 MB, ``replan`` with ``trim``, one contact band;
    ``capacity`` sets a simulated battery (the energy clause); ``response``
    ``abort`` is the Exp 4 driver's default departure response."""
    return FerrySpec.from_config(rf_range_m=60.0, seed=seed, contact_band=band,
                                 payload_bytes=1_000_000, contact_regime=regime,
                                 in_flight_response=response, replan_fallback="trim",
                                 energy_capacity_j=capacity)


def fly(e3, *, budget, missions=1, unit=FAR_UNIT, band="wide", layout=GH.LAYOUT,
        regime="clean", seed=7, capacity=None, response="replan", after=None):
    """``missions`` missions of a legacy mule flying ``e3``; one record each.
    ``after(m, result)`` runs after each mission."""
    w = H.World(layout=layout, flaky={})
    mid = w.mule_ids[0]
    out = []
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=True, ferry=spec(band=band, regime=regime, seed=seed,
                                                     capacity=capacity, response=response),
                           mission_budget_s=budget, deadline_time_scale=unit,
                           target_selector=e3)
        w.bootstrap()
        for m in range(missions):
            result = sup.run_one_mission()
            out.append(SimpleNamespace(result=result, deltas=w.take_deltas(mid)))
            if after is not None:
                after(m, result)
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return w, sup, out


def mission_end(recs, clock, budget):
    """The budget's end of the mission a call at ``clock`` belongs to."""
    return max(r.result.sim_start_s for r in recs if r.result.sim_start_s <= clock) + budget


def widened(rec):
    """The devices given a synthetic TIMEOUT this mission (widening)."""
    return sorted({str(d.device_id) for src, d in rec.deltas if src == "direct" and d.synthetic})


def devices(stops):
    return [[str(d) for d in wp.devices] for wp in stops]


def wp_at(x, y, members=1, deadline=0.0, tag="", z=0.0):
    """A contact at (x, y, z) with ``members`` devices named after it."""
    ids = tuple(DeviceID(f"d{tag}{x:g}_{y:g}_{j}") for j in range(members))
    return ContactWaypoint(position=(float(x), float(y), float(z)), devices=ids,
                           bucket=Bucket.NEW, deadline_ts=float(deadline))


def stop(**changes):
    base = dict(members=1, remaining=1.0, snr_db=-5.0, reachable=0.0, dx_m=10.0, dy_m=0.0,
                distance_m=10.0, return_energy_j=100.0)
    base.update(changes)
    return E3Stop(**base)


def view(stops, **changes):
    base = dict(band="wide", demand=6, clock_s=100.0, budget_end=190.0, budget_s=90.0,
                energy_j=500.0, energy_ref_j=9000.0)
    base.update(changes)
    return E3View(stops=tuple(stops), **base)


def random_case(rng, k):
    """A view of ``k`` stops with random observations, and the stops it observes."""
    stops, wps = [], []
    for i in range(k):
        members = rng.randint(1, 3)
        dx, dy = rng.uniform(-150, 150), rng.uniform(-150, 150)
        stops.append(stop(members=members, snr_db=rng.uniform(-13, 15),
                          reachable=rng.choice([0.0, 0.0, 0.5, 1.0]), dx_m=dx, dy_m=dy,
                          distance_m=math.hypot(dx, dy), return_energy_j=rng.uniform(0, 4000)))
        wps.append(wp_at(dx, dy, members, tag=f"{i}_"))
    v = view(stops, demand=12, clock_s=rng.uniform(0, 90), energy_j=rng.uniform(0, 6000))
    return v, wps


def linear_net(weights, bias=0.0):
    """A linear E3 network: Q = weights . row + bias (``hidden=()``)."""
    net = PairQNet(C.E3_DIM, PairQConfig(hidden=()), seed=0)
    w = np.zeros((C.E3_DIM, 1))
    for name, value in weights.items():
        w[COL[name], 0] = value
    net.set_weights({"layer0_W": w, "layer0_b": np.array([float(bias)])})
    return net


def ask(e3, v, wps, mask, **changes):
    args = dict(view=v, admissible=lambda i: mask[i], pass_kind=COLLECT, after_stop=True)
    args.update(changes)
    return e3.next_stop(wps, None, **args)


class CountingRandom(random.Random):
    """A seeded stream that counts its draws."""

    draws = 0

    def random(self):
        self.draws += 1
        return super().random()


class Untouchable:
    """Raises on any use: what E3's admission must not read."""

    def __getattr__(self, name):
        raise AssertionError(f"E3's admission read {name!r}")

    def __getitem__(self, key):
        raise AssertionError(f"E3's admission read [{key!r}]")

    def __iter__(self):
        raise AssertionError("E3's admission iterated over it")

    def __len__(self):
        raise AssertionError("E3's admission took its length")


def provenance(**changes):
    """A bootstrap checkpoint's provenance, as the trainer (unit U8b) writes it."""
    out = dict(reward={"kind": "bytes"}, training={"episodes": 0}, seeds={"init": 1},
               cell_family="jittery", cell_family_sha256="ab" * 32, trainer_commit=None,
               dirty=True, episodes_trained=0, validation=[], held_out=None)
    out.update(changes)
    return out


def trained_provenance(**changes):
    """A trained checkpoint's provenance once the evaluator has scored it (units
    U8b and U8a): a clean tree, episodes trained, a validation curve and a
    held-out score, so no campaign refusal applies (critic B9)."""
    out = provenance(training={"episodes": 2000}, seeds={"init": 1, "train": 7},
                     trainer_commit="a" * 40, dirty=False, episodes_trained=2000,
                     validation=[{"episode": 1000, "return_mean": 0.31}],
                     held_out={"episodes": 1000, "return_mean": 0.33})
    out.update(changes)
    return out


def stand_in_batch(seed, size=64):
    """Transitions over random rows of E3's width, a quarter of them done and the
    rest with one to five admitted next rows: a stand-in for E3's steps that
    moves a live network's online weights off its target copy in a few
    updates, far enough to change decisions."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(size):
        x, reward = rng.normal(size=C.E3_DIM), float(rng.uniform())
        if k % 4 == 3:
            out.append(PairTransition(x, reward, True))
        else:
            width = int(rng.integers(1, 6))
            out.append(PairTransition(x, reward, False, rng.normal(size=(width, C.E3_DIM)),
                                      np.ones(width, dtype=bool)))
    return PairBatch.of(out)


# --------------------------------------------------------------------------- #
# The protocol it declares
# --------------------------------------------------------------------------- #

def test_e3_declares_the_whole_scheduler_protocol_and_its_switch_value():
    """The config's switch value and the checkpoint kind restate the policy's name
    (``processes/config.py``, ``selector/pair_q.py``); the in-flight check is
    ``none``; whole stops only; the per-departure protocol, a bool True, so the
    supervisor's ``is True`` test takes it. It names no band: the slot's band
    hooks are the plan's, and E3 flies the cell's one band."""
    e3 = policy()
    assert C.ChenDQNPolicy.name == CFG.CONTACT_POLICY_CHEN_DQN == Q.KIND_CHEN_DQN == "chen_dqn"
    assert e3.in_flight_check == IN_FLIGHT_NONE == "none"
    assert e3.admits_member_subsets is False and e3.chooses_next_stop is True
    assert isinstance(e3, NextStopPolicy) and callable(e3.admit_and_order)
    for hook in ("band_at_arrival", "decides_at_arrival", "pair_at_arrival", "rank_contacts"):
        assert not hasattr(e3, hook)
    assert (e3.band, e3.manifest, e3.training) == ("wide", None, False)


def test_e3_visits_whole_stops_so_member_subsets_are_refused():
    """E3 visits a stop for all its members (``admits_member_subsets`` False; the
    config guard requires ``member_admission="whole"``), so the scheduler refuses
    to plan member subsets for it, as for any whole-scheduler policy that does
    not take them."""
    w = H.World(layout=GH.LAYOUT, flaky={})
    with H.Patched(w.clock):
        with pytest.raises(Exception, match="does not take member subsets"):
            w.supervisor(w.mule_ids[0], sim=True, ferry=spec(), mission_budget_s=120.0,
                         target_selector=policy(), member_admission="subset")


# --------------------------------------------------------------------------- #
# Every contact admitted
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(6))
def test_every_contact_is_admitted_nearest_the_takeoff_pose_first(seed):
    """Chen's safety controller acts at each step, so before takeoff E3 drops
    nothing: every contact, overdue or not, whatever the budget, as the same
    objects, nearest the pose first (the flight model's metric), then the
    position and the members, whatever order they came in. The device states,
    the feasibility model and the budget are not read."""
    rng = random.Random(seed)
    contacts = [wp_at(rng.uniform(-150, 150), rng.uniform(-150, 150), rng.randint(1, 3),
                      deadline=rng.choice([-1e9, 0.0, 1e9]), tag=f"{i}_",
                      z=rng.choice([0.0, 40.0]))
                for i in range(rng.randint(1, 12))]
    x, y, z = contacts[0].position
    contacts.append(wp_at(x, y, tag="twin_", z=z))
    pose = (rng.uniform(-50, 50), rng.uniform(-50, 50), rng.choice([0.0, 25.0]))
    env = SelectorEnv(mule_pose=pose, now=1e6)
    route = policy().admit_and_order(contacts, Untouchable(), env, mission_deadline_ts=1e6 + 1.0,
                                     feasibility_model=Untouchable())
    assert sorted(map(id, route)) == sorted(map(id, contacts))
    keys = [(math.dist(pose, wp.position), wp.position, [str(d) for d in wp.devices])
            for wp in route]
    assert keys == sorted(keys)
    shuffled = list(contacts)
    rng.shuffle(shuffled)
    again = policy().admit_and_order(shuffled, {}, env)
    assert [id(wp) for wp in again] == [id(wp) for wp in route]
    assert policy().admit_and_order([], {}, env) == []
    with pytest.raises(TypeError, match="ContactWaypoint"):
        policy().admit_and_order([pose], {}, env)


@pytest.mark.parametrize("env, error", [
    (SelectorEnv(mule_pose=(math.nan, 0.0, 0.0)), ValueError),
    (SimpleNamespace(mule_pose=()), ValueError),
    (SimpleNamespace(), TypeError),
], ids=["nan", "empty", "missing"])
def test_admission_orders_from_the_mules_pose_or_refuses(env, error):
    """The order is nearest the mule's pose: without a finite pose there is none."""
    with pytest.raises(error, match="pose"):
        policy().admit_and_order([wp_at(10.0, 0.0)], {}, env)


def test_a_distance_tie_goes_to_the_position_then_the_members():
    """Contacts equally far from the pose come in position order, and two at one
    position in the order of their members, as documented; the members'
    names, which run the other way here, decide only the second tie. The
    order of the rows, and so every tie between equal Q, never depends on the
    order the contacts came in."""
    east, north = wp_at(10.0, 0.0, tag="a"), wp_at(0.0, 10.0, tag="m")
    west, twin = wp_at(-10.0, 0.0, tag="z"), wp_at(-10.0, 0.0, tag="b")
    env = SelectorEnv(mule_pose=(0.0, 0.0, 0.0), now=0.0)
    for contacts in itertools.permutations((east, north, west, twin)):
        route = policy(record=False).admit_and_order(list(contacts), {}, env)
        assert [id(wp) for wp in route] == [id(wp) for wp in (twin, west, north, east)]


@pytest.mark.parametrize("budget, unit", [(400.0, FAR_UNIT), (30.0, FAR_UNIT), (30.0, 1.0)])
def test_every_contact_is_admitted_in_flight_whatever_the_budget_or_deadlines(budget, unit):
    """In a mission: the Pass-1 queue holds every S3a contact (all seven slice
    devices) whatever the budget or the deadline unit, nearest the dock first;
    the scheduler reports no policy drop and runs no S3b gate."""
    e3 = policy()
    _, sup, recs = fly(e3, budget=budget, unit=unit, missions=2)
    for rec in recs:
        r = rec.result
        assert len(r.pass_1_queue) == N_STOPS
        assert sorted(d for stop_ in devices(r.pass_1_queue) for d in stop_) == sorted(
            did for did, _ in GH.LAYOUT)
        dists = [math.dist((0.0, 0.0, 0.0), wp.position) for wp in r.pass_1_queue]
        assert dists == sorted(dists)
        assert r.pass_1_policy_drops is None and r.pass_1_preflight_drops == []
    assert sup.scheduler.last_policy_drops == [] and sup.scheduler.last_feasibility is None


# --------------------------------------------------------------------------- #
# Check "none"
# --------------------------------------------------------------------------- #

def test_the_in_flight_check_is_none():
    """Freeze Amendment 8: the scheduler holds E3's remainder to ``RULE_NONE`` in
    Pass 1 (Pass 2 is every arm's budget walk), so a fold never rejects and a
    re-plan keeps the remainder as it is, even past the budget's end."""
    _, sup, (rec,) = fly(policy(), budget=400.0)
    sched = sup.scheduler
    assert sched.in_flight_rule(COLLECT) == RULE_NONE
    assert sched.in_flight_rule(DELIVER) == RULE_BUDGET
    stops = list(rec.result.pass_1_queue)
    state = FlightState((0.0, 0.0, 0.0), 1e6)
    assert sched.fold_remainder(stops, state=state, budget_end=1.0).ok
    again = sched.replan_remainder(stops, state=state, budget_end=1.0)
    assert (again.route, again.dropped, again.order_used) == (tuple(stops), (), ORDER_NONE)


@pytest.mark.parametrize("budget", (200.0, 80.0, 30.0))
def test_in_flight_nothing_is_re_planned_or_trimmed(budget):
    """In a mission under a tight budget: no re-plan and no abort, and each call
    after a stop is offered the previous call's stops less the one flown (no
    beacon offer here), so only E3's own None ends the pass. The departure
    check's response does not enter: under ``abort``, the Exp 4 driver's
    default, E3 makes the same calls and flies the same Pass 1 as under
    ``replan``, with the same records."""
    seen = {}
    for response in ("replan", "abort"):
        e3 = policy(seed=1)
        _, _, recs = fly(e3, budget=budget, missions=2, response=response)
        for rec in recs:
            assert rec.result.replans == [] and rec.result.aborts == []
        for before, call in zip(e3.calls, e3.calls[1:]):
            if not call.after_stop:
                continue
            flown = before.remainder[before.answer]
            assert [id(wp) for wp in call.remainder] == [
                id(wp) for wp in before.remainder if wp is not flown]
        seen[response] = ([(call.answer, call.mask) for call in e3.calls],
                          [(rec.result.pass_1_flown, rec.result.pass_1_e3,
                            rec.result.pass_1_e3_unvisited) for rec in recs])
    assert seen["abort"] == seen["replan"]


# --------------------------------------------------------------------------- #
# The mask: S3b's admit under the budget rule, landing included
# --------------------------------------------------------------------------- #

def test_the_mask_is_s3bs_admit_under_the_budget_rule_landing_included():
    """Chen's safety controller is S3b's single-contact predicate under
    ``RULE_BUDGET`` (``FeasibilityModel.admit``) from each call's state,
    recomputed here independently: the transit, the dwell, the stop's return
    leg and the upload are priced, so a stop whose service ends within the
    budget but whose landing does not is refused. E3 names an admissible stop
    exactly when there is one."""
    by_landing = 0
    for budget in (200.0, 120.0, 80.0, 45.0):
        for seed in (0, 1):
            e3 = policy(seed=seed)
            _, sup, recs = fly(e3, budget=budget, missions=2)
            model = sup.scheduler.feasibility_model
            for call in e3.calls:
                end = mission_end(recs, call.state.clock, budget)
                verdicts = [model.admit(call.state, wp, rule=RULE_BUDGET, budget_end=end,
                                        pass_kind=COLLECT) for wp in call.remainder]
                assert call.mask == [v.ok for v in verdicts]
                if any(call.mask):
                    assert call.mask[call.answer]
                else:
                    assert call.answer is None
                by_landing += sum(1 for v in verdicts if not v.ok and v.finish <= end < v.home)
    assert by_landing > 0


@pytest.mark.parametrize("capacity", (1e4, 2e4))
def test_the_mask_holds_the_energy_clause_when_a_battery_is_set(capacity):
    """With a simulated battery and a budget that does not bind (400 s), Chen's
    safety controller is the energy clause of the same predicate: stops whose
    flight and return the battery cannot pay are refused, and the view's energy
    reference is the battery (``energy_left``)."""
    e3 = policy()
    _, sup, recs = fly(e3, budget=400.0, missions=2, capacity=capacity)
    model = sup.scheduler.feasibility_model
    reasons = []
    for call in e3.calls:
        end = mission_end(recs, call.state.clock, 400.0)
        verdicts = [model.admit(call.state, wp, rule=RULE_BUDGET, budget_end=end,
                                pass_kind=COLLECT) for wp in call.remainder]
        assert call.mask == [v.ok for v in verdicts]
        reasons += [v.reason for v in verdicts]
        assert call.view.energy_ref_j == capacity
    assert "energy" in reasons and "budget" not in reasons
    assert all(len(r.result.pass_1_flown) < N_STOPS for r in recs)


@pytest.mark.parametrize("seed", range(8))
def test_the_answer_is_the_masked_argmax_of_the_networks_q(seed):
    """On random views and masks: ``admissible`` is asked about every stop, in
    order, and the answer is the admissible row with the highest Q of
    :func:`e3_rows` (``pair_q.masked_argmax``), or None when no stop is
    admissible."""
    rng = random.Random(seed)
    e3 = C.ChenDQNPolicy(C.new_e3_network(seed=seed), band="wide")
    for _ in range(25):
        v, wps = random_case(rng, rng.randint(1, 8))
        mask = [rng.random() < 0.6 for _ in wps]
        asked = []

        def admissible(i, mask=mask, asked=asked):
            asked.append(i)
            return mask[i]

        answer = e3.next_stop(wps, None, view=v, admissible=admissible, pass_kind=COLLECT,
                              after_stop=rng.random() < 0.5)
        assert asked == list(range(len(wps)))
        if any(mask):
            assert answer == Q.masked_argmax(e3.net.q(C.e3_rows(v)), np.array(mask))
            assert mask[answer] and isinstance(answer, int)
        else:
            assert answer is None


def test_ties_go_to_the_lowest_admissible_row():
    """The pair learner's tie rule (``pair_q.masked_argmax``): with every Q equal,
    the lowest admissible row, so the remainder's order (nearest the takeoff
    pose first) breaks the tie."""
    flat = linear_net({})
    v, wps = random_case(random.Random(3), 5)
    e3 = C.ChenDQNPolicy(flat, band="wide")
    assert ask(e3, v, wps, [False, False, True, True, True]) == 2
    assert ask(e3, v, wps, [True] * 5) == 0
    assert ask(e3, v, wps, [False, False, False, False, True]) == 4


@pytest.mark.parametrize("seed", range(5))
def test_a_negative_distance_score_flies_the_nearest_admissible_stop(seed):
    """The rows are wired to the stops: a network that scores -distance names the
    nearest admissible stop, as FX's next-stop rule would among those."""
    rng = random.Random(seed)
    e3 = C.ChenDQNPolicy(linear_net({"distance": -1.0}), band="wide")
    for _ in range(20):
        v, wps = random_case(rng, rng.randint(1, 8))
        mask = [rng.random() < 0.7 for _ in wps]
        fit = [i for i, ok in enumerate(mask) if ok]
        want = min(fit, key=lambda i: (v.stops[i].distance_m, i)) if fit else None
        assert ask(e3, v, wps, mask) == want


@pytest.mark.parametrize("answer", [1, np.bool_(True), "yes", None, SimpleNamespace(ok=True)])
def test_admissible_must_answer_a_bool(answer):
    """A verdict passed whole (truthy) or a 0/1 would admit stops the predicate
    refused: only a bool is read."""
    v, wps = random_case(random.Random(0), 3)
    with pytest.raises(TypeError, match="must answer a bool"):
        policy().next_stop(wps, None, view=v, admissible=lambda i: answer, pass_kind=COLLECT,
                           after_stop=False)


@pytest.mark.parametrize("changes, error, match", [
    ({"view": "a view"}, TypeError, "E3View"),
    ({"remainder": []}, ValueError, "none is left"),
    ({"remainder": ["a stop"]}, TypeError, "ContactWaypoints"),
    ({"drop_one": True}, ValueError, "one row per stop left"),
    ({"members": True}, ValueError, "observes other stops"),
    ({"after_stop": 1}, TypeError, "after_stop"),
    ({"admissible": None}, TypeError, "safety controller"),
], ids=["no_view", "empty", "not_a_stop", "count", "members", "after_stop", "no_admissible"])
def test_a_call_off_the_protocol_is_refused(changes, error, match):
    """The view observes the remainder, one stop each with matching members, and
    the call carries a bool ``after_stop`` and E3's safety controller."""
    v, wps = random_case(random.Random(1), 3)
    args = dict(view=v, admissible=lambda i: True, pass_kind=COLLECT, after_stop=False)
    changes = dict(changes)
    remainder = list(wps)
    if changes.pop("drop_one", False):
        remainder = remainder[:2]
    if changes.pop("members", False):
        remainder[1] = wp_at(*remainder[1].position[:2], members=v.stops[1].members + 1)
    remainder = changes.pop("remainder", remainder)
    args.update(changes)
    with pytest.raises(error, match=match):
        policy().next_stop(remainder, None, **args)


# --------------------------------------------------------------------------- #
# The pass ends when nothing fits
# --------------------------------------------------------------------------- #

def test_nothing_admissible_ends_the_pass_with_nothing_scored_or_drawn():
    """No admissible stop: None, without a forward pass, a draw or a step, so a
    training run's stream and steps are those of the decisions alone."""
    v, wps = random_case(random.Random(2), 4)
    e3 = policy()
    net = C.new_e3_network(seed=5)
    rng, steps = CountingRandom(0), []
    e3.attach_trainer(net=net, epsilon=0.5, rng=rng, sink=steps.append)

    def no_q(rows):
        raise AssertionError("nothing is scored when nothing fits")

    net.q = no_q
    assert ask(e3, v, wps, [False] * 4) is None
    assert rng.draws == 0 and steps == []


@pytest.mark.parametrize("budget", (120.0, 80.0, 30.0, 5.0))
def test_the_pass_ends_when_nothing_fits_and_the_rest_is_reported_never_widened(budget):
    """When no stop is admissible E3 answers None: the pass ends and the mule
    flies home, the stops left go to ``pass_1_e3_unvisited`` (never widened: no
    synthetic TIMEOUT, no miss streak), and Pass 2 still delivers to every
    slice stop. At 5 s nothing fits at takeoff, so nothing is flown and the
    round is the recorded empty round, with no Pass 2 (nothing to deliver)."""
    e3 = policy(seed=2)
    _, sup, (rec,) = fly(e3, budget=budget)
    r = rec.result
    last = e3.calls[-1]
    assert not any(last.mask) and last.answer is None
    assert len(r.pass_1_flown) == len(e3.calls) - 1 < N_STOPS
    assert (len(r.pass_1_flown) == 0) == (budget == 5.0)
    assert all(c.answer is not None for c in e3.calls[:-1])
    assert r.pass_1_e3[-1]["next_index"] is None and r.pass_1_e3[-1]["next"] == "home"
    assert r.pass_1_e3_unvisited == [
        {"position": [float(c) for c in wp.position], "devices": [str(d) for d in wp.devices],
         "deadline_ts": float(wp.deadline_ts), "widened": False} for wp in last.remainder]
    assert widened(rec) == []
    assert all(sup.scheduler.device_states[DeviceID(d)].miss_streak == 0
               for stop_ in devices(last.remainder) for d in stop_)
    if r.pass_1_flown:
        assert len(r.pass_2_flown) == N_STOPS and not r.empty
    else:
        assert r.empty and r.pass_2_flown == [] and len(r.pass_1_e3) == 1
        assert len(r.pass_1_e3_unvisited) == N_STOPS


# --------------------------------------------------------------------------- #
# No deadline gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("budget", (400.0, 200.0))
def test_no_deadline_gates_e3(budget):
    """E3 flies none of FeRRy's deadline machinery (build plan L1026): under the
    recorded deadline unit (1 s), stops S3b's deadline rule refuses as overdue
    while the budget admits them are admissible to E3, and E3 flies some of
    them."""
    e3 = policy(seed=4)
    _, sup, recs = fly(e3, budget=budget, unit=1.0, missions=2)
    model = sup.scheduler.feasibility_model
    overdue = flown = 0
    for call in e3.calls:
        end = mission_end(recs, call.state.clock, budget)
        for i, (wp, ok) in enumerate(zip(call.remainder, call.mask)):
            strict = model.admit(call.state, wp, rule=RULE_DEADLINE_BUDGET, budget_end=end,
                                 pass_kind=COLLECT)
            loose = model.admit(call.state, wp, rule=RULE_BUDGET, budget_end=end,
                                pass_kind=COLLECT)
            assert ok == loose.ok
            if strict.reason == REASON_OVERDUE and loose.ok:
                overdue += 1
                flown += call.answer == i
    assert overdue > 0 and flown > 0


# --------------------------------------------------------------------------- #
# The band fixed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("band", ("wide", "medium", "narrow"))
def test_e3_flies_its_one_band_in_both_passes(band):
    """FL-blind and band-fixed (build plan L1026): every stop of Pass 1 and Pass 2
    is flown on the cell's contact band, and every call observes that band;
    E3 is asked in Pass 1 only (critic B7 i)."""
    e3 = policy(band=band)
    _, _, recs = fly(e3, budget=400.0, band=band, missions=2)
    for rec in recs:
        r = rec.result
        assert r.pass_1_flown and r.pass_2_flown
        assert {s["band"] for s in r.pass_1_flown + r.pass_2_flown} == {band}
    assert {c.view.band for c in e3.calls} == {band}
    assert {c.pass_kind for c in e3.calls} == {COLLECT}


def test_a_view_on_another_band_is_refused():
    """A policy for one band never scores another band's observation: in a
    mission on medium, E3 for wide refuses at its first call."""
    v, wps = random_case(random.Random(4), 3)
    with pytest.raises(ValueError, match="the band is fixed"):
        ask(policy(band="medium"), v, wps, [True] * 3)
    with pytest.raises(ValueError, match="the band is fixed"):
        fly(policy(band="wide"), budget=400.0, band="medium")


# --------------------------------------------------------------------------- #
# The rows, and the declared constant columns
# --------------------------------------------------------------------------- #

def test_the_rows_are_chens_observation_in_the_schemas_units():
    """Each column as the module documents it, row i for stop i, the sortie's
    columns alike on every row; without an energy reference the return costs 0
    and the energy left is 1, without a budget the time left is 1; with a
    reference, both energy columns are plain ratios to it, whatever its size."""
    a = stop(members=2, remaining=1.0, snr_db=-7.5, reachable=0.5, dx_m=-30.0, dy_m=40.0,
             distance_m=50.0, return_energy_j=1800.0)
    b = stop(members=1, snr_db=12.0, reachable=0.0, dx_m=120.0, dy_m=-5.0, distance_m=121.0,
             return_energy_j=900.0)
    rows = C.e3_rows(view([a, b], demand=8, clock_s=130.0, budget_end=190.0, budget_s=90.0,
                          energy_j=2250.0, energy_ref_j=9000.0))
    assert rows.dtype == np.float64 and rows.shape == (2, C.E3_DIM) == (2, 10)
    assert rows[0].tolist() == [1.0, -0.75, 0.5, -0.3, 0.4, 0.5, 0.25, 0.2, 0.75, 60.0 / 90.0]
    assert rows[1].tolist() == [1.0, 1.2, 0.0, 1.2, -0.05, 1.21, 0.125, 0.1, 0.75, 60.0 / 90.0]
    bare = C.e3_rows(view([a], energy_ref_j=None, budget_end=None, budget_s=None))
    assert bare[0, COL["return_energy"]] == 0.0
    assert bare[0, COL["energy_left"]] == 1.0 and bare[0, COL["time_left"]] == 1.0
    assert C.e3_rows(view([a], energy_ref_j=0.0))[0, COL["energy_left"]] == 1.0
    small = C.e3_rows(view([a], energy_j=0.25, energy_ref_j=0.5))   # ratios, never floored
    assert small[0, COL["return_energy"]] == 3600.0 and small[0, COL["energy_left"]] == 0.5
    over = C.e3_rows(view([a], clock_s=200.0, energy_j=12000.0))
    assert over[0, COL["time_left"]] < 0.0 and over[0, COL["energy_left"]] < 0.0
    with pytest.raises(TypeError, match="E3View"):
        C.e3_rows([a])


def test_the_schema_names_the_rows_their_units_and_the_declared_columns():
    """The schema a checkpoint records (``pair_q`` binds it into the sha): the
    version, the width, the columns in order, the declared columns and the two
    units; a fresh dict each time."""
    schema = C.e3_schema()
    assert schema == {
        "version": "e3_v1", "dim": 10,
        "columns": ["remaining", "snr", "reachable", "dx", "dy", "distance", "members",
                    "return_energy", "energy_left", "time_left"],
        "declared_constant": ["remaining", "reachable"],
        "length_scale_m": 100.0, "snr_scale_db": 10.0,
    }
    schema["columns"].append("x")
    assert C.e3_schema()["columns"][-1] == "time_left"
    assert set(C.E3_DECLARED_CONSTANT) < set(C.E3_COLUMNS) and C.E3_DIM == len(C.E3_COLUMNS)


def _sample_rows():
    """E3's rows over a FerrySim-like sample: random layouts on the realism field
    (+-100 m) at N = 6 (45 and 90 s) and N = 12 (120 and 180 s), the jittery
    contact channel, 1 MB, four missions each, flown ε-greedy (0.5) so the
    sample covers more than one network's choices."""
    rows, per_step = [], []
    for n, budgets in ((6, (45.0, 90.0)), (12, (120.0, 180.0))):
        for budget in budgets:
            for seed in range(3):
                rng = random.Random(seed)
                layout = tuple((f"dev-{i:02d}", (rng.uniform(-100, 100), rng.uniform(-100, 100),
                                                  0.0)) for i in range(n))
                steps = []
                e3 = policy(seed=seed, record=False)
                e3.attach_trainer(net=C.new_e3_network(seed=seed), epsilon=0.5,
                                  rng=random.Random(seed), sink=steps.append)
                fly(e3, budget=budget, missions=4, layout=layout, regime="jittery",
                    seed=1000 * n + seed)
                for step in steps:
                    rows.extend(step.rows)
                    per_step.append(step.matrix)
    return np.array(rows), per_step


def test_no_column_is_constant_on_a_ferrysim_like_sample_except_as_declared():
    """Critic B7 ii: ``remaining`` is 1 on every row and ``reachable`` is 0 on most
    (near zero), as declared; every other column varies. The sortie's two
    columns are the same on every row of a call."""
    rows, per_step = _sample_rows()
    assert rows.shape[0] > 300 and np.isfinite(rows).all()
    assert (rows[:, COL["remaining"]] == 1.0).all()
    reach = rows[:, COL["reachable"]]
    assert np.mean(reach == 0.0) >= 0.6 and reach.mean() <= 0.25
    for name in C.E3_COLUMNS:
        if name in C.E3_DECLARED_CONSTANT:
            continue
        col = rows[:, COL[name]]
        assert col.std() > 0.05 and len(np.unique(col)) >= 5, name
    for matrix in per_step:
        for name in ("energy_left", "time_left"):
            assert (matrix[:, COL[name]] == matrix[0, COL[name]]).all()


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #

def test_an_e3_checkpoint_round_trips_into_the_policy(tmp_path):
    """A bootstrap checkpoint written by :func:`save_e3_checkpoint` loads into the
    policy the mule builds (:meth:`ChenDQNPolicy.from_checkpoint`), with the same
    weights, its manifest kept for the provenance (a copy each time): the E3
    kind, the schema, the band as the one class. No training state is checked."""
    net = C.new_e3_network(seed=11, config=PairQConfig(hidden=(16,), gamma=0.99))
    path = tmp_path / "e3" / "g0.99_s1.npz"
    sha = C.save_e3_checkpoint(net, path, band="medium", purpose="bootstrap",
                               provenance=provenance())
    e3 = C.ChenDQNPolicy.from_checkpoint(str(path), expect_sha256=sha, band="medium")
    rows = np.random.default_rng(0).normal(size=(6, C.E3_DIM))
    assert np.array_equal(e3.net.q(rows), net.q(rows)) and e3.band == "medium"
    manifest = e3.manifest
    assert (manifest["kind"], manifest["purpose"], manifest["sha256"], manifest["classes"],
            manifest["schema"], manifest["gamma"]) == (
        "chen_dqn", "bootstrap", sha, ["medium"], C.e3_schema(), 0.99)
    manifest["classes"].append("wide")
    assert e3.manifest["classes"] == ["medium"]
    loaded, again = C.load_e3_network(path, expect_sha256=sha, band="medium")
    assert again == e3.manifest and loaded.config.hidden == (16,)
    assert Q.campaign_refusals(e3.manifest)          # a bootstrap: the runner refuses it
    announced = Q.manifest_provenance(e3.manifest)   # mule_ready.policy_checkpoint, less the tag
    assert (announced["sha256"], announced["kind"], announced["classes"],
            announced["schema"]) == (sha, "chen_dqn", ["medium"], "e3_v1")


def test_a_trained_checkpoint_keeps_its_purpose_and_provenance(tmp_path):
    """A trained checkpoint (unit U8b's result) is written as trained, with the
    trainer's provenance as given, so the policy the mule builds from it
    announces it as such and the runner's refusals (critic B9;
    ``pair_q.campaign_refusals``) pass it. The purpose is in the header the sha
    binds (resolution R2): the same weights and provenance saved as a bootstrap
    have another sha, which a config naming the trained one cannot load."""
    net = C.new_e3_network(seed=13)
    given = trained_provenance()
    path = tmp_path / "e3" / "trained.npz"
    sha = C.save_e3_checkpoint(net, path, band="narrow", purpose="trained", provenance=given)
    e3 = C.ChenDQNPolicy.from_checkpoint(path, expect_sha256=sha, band="narrow")
    manifest = e3.manifest
    assert (manifest["purpose"], manifest["kind"], manifest["sha256"], manifest["classes"]) == (
        "trained", "chen_dqn", sha, ["narrow"])
    assert {key: manifest[key] for key in Q.PROVENANCE_KEYS} == given
    assert Q.campaign_refusals(manifest) == []
    announced = Q.manifest_provenance(manifest)
    assert (announced["purpose"], announced["episodes_trained"], announced["seeds"]) == (
        "trained", 2000, {"init": 1, "train": 7})
    rows = np.random.default_rng(1).normal(size=(4, C.E3_DIM))
    assert np.array_equal(e3.net.q(rows), net.q(rows))
    bootstrap = tmp_path / "e3" / "bootstrap.npz"
    other = C.save_e3_checkpoint(net, bootstrap, band="narrow", purpose="bootstrap",
                                 provenance=given)
    assert other != sha
    assert C.load_e3_network(bootstrap, expect_sha256=other, band="narrow")[1]["purpose"] == (
        "bootstrap")
    with pytest.raises(CheckpointError, match="not the expected"):
        C.ChenDQNPolicy.from_checkpoint(bootstrap, expect_sha256=sha, band="narrow")


def _other_kind(net, path):
    return net.save(path, kind="pair_q", purpose="bootstrap", schema=C.e3_schema(),
                    classes=["wide"], provenance=provenance())


def _other_schema(net, path):
    return net.save(path, kind="chen_dqn", purpose="bootstrap",
                    schema=dict(C.e3_schema(), snr_scale_db=20.0), classes=["wide"],
                    provenance=provenance())


@pytest.mark.parametrize("write, load, match", [
    (None, {"expect_sha256": "0" * 64}, "not the expected"),
    (None, {"band": "medium"}, "trained on the classes"),
    (_other_kind, {}, "'pair_q' checkpoint"),
    (_other_schema, {}, "feature schema"),
], ids=["sha", "band", "kind", "schema"])
def test_the_loader_refuses_another_sha_band_kind_or_schema(tmp_path, write, load, match):
    """The mule flies only the checkpoint its config names, written for E3's rows
    on its band: anything else is a ``CheckpointError`` (a ``ValueError``)."""
    net = C.new_e3_network(seed=1)
    path = tmp_path / "e3.npz"
    sha = (write or (lambda n, p: C.save_e3_checkpoint(
        n, p, band="wide", purpose="bootstrap", provenance=provenance())))(net, path)
    args = dict(expect_sha256=sha, band="wide")
    args.update(load)
    with pytest.raises(CheckpointError, match=match):
        C.ChenDQNPolicy.from_checkpoint(path, **args)


@pytest.mark.parametrize("make, error, match", [
    (lambda: C.ChenDQNPolicy(object(), band="wide"), TypeError, "PairQNet"),
    (lambda: C.ChenDQNPolicy(PairQNet(5, seed=0), band="wide"), ValueError, "rows are 10"),
    (lambda: C.ChenDQNPolicy(C.new_e3_network(seed=0), band=""), TypeError, "contact band"),
    (lambda: C.ChenDQNPolicy(C.new_e3_network(seed=0), band=None), TypeError, "contact band"),
    (lambda: C.save_e3_checkpoint(PairQNet(5, seed=0), "x.npz", band="wide",
                                  purpose="bootstrap", provenance=provenance()),
     ValueError, "rows are 10"),
], ids=["not_a_net", "width", "empty_band", "no_band", "save_width"])
def test_the_policy_refuses_a_network_that_is_not_e3s(make, error, match):
    with pytest.raises(error, match=match):
        make()


@pytest.mark.parametrize("changes, match", [
    ({"kind": "pair_q"}, "'pair_q' checkpoint"),
    ({"classes": ["medium"]}, "trained on"),
    ({"schema": {"version": "e3_v0", "dim": 10}}, "other rows"),
], ids=["kind", "band", "schema"])
def test_a_manifest_must_be_an_e3_one_of_the_band(tmp_path, changes, match):
    net = C.new_e3_network(seed=0)
    sha = C.save_e3_checkpoint(net, tmp_path / "e3.npz", band="wide", purpose="bootstrap",
                               provenance=provenance())
    manifest = C.load_e3_network(tmp_path / "e3.npz", expect_sha256=sha, band="wide")[1]
    assert C.ChenDQNPolicy(net, band="wide", manifest=manifest).manifest == manifest
    with pytest.raises(ValueError, match=match):
        C.ChenDQNPolicy(net, band="wide", manifest=dict(manifest, **changes))
    with pytest.raises(TypeError, match="mapping"):
        C.ChenDQNPolicy(net, band="wide", manifest=[manifest])


def test_the_policy_keeps_its_own_json_copy_of_the_manifest(tmp_path):
    """The manifest the mule announces (``mule_ready.policy_checkpoint``) is the
    policy's own JSON copy, taken when it is built: a later change to the
    caller's mapping, nested values included, does not reach it, and a value
    JSON cannot carry (a NaN) is refused rather than kept."""
    net = C.new_e3_network(seed=0)
    sha = C.save_e3_checkpoint(net, tmp_path / "e3.npz", band="wide", purpose="bootstrap",
                               provenance=provenance())
    manifest = C.load_e3_network(tmp_path / "e3.npz", expect_sha256=sha, band="wide")[1]
    theirs = json.loads(json.dumps(manifest))
    e3 = C.ChenDQNPolicy(net, band="wide", manifest=theirs)
    theirs["classes"].append("medium")
    theirs["schema"]["columns"].append("x")
    theirs["seeds"]["init"] = 2
    assert e3.manifest == manifest
    with pytest.raises(ValueError, match="not JSON"):
        C.ChenDQNPolicy(net, band="wide", manifest=dict(manifest, gamma=math.nan))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def test_a_trainer_at_epsilon_zero_flies_as_none_and_sinks_every_decision():
    """ε = 0 flies as no trainer does: the masked argmax of the live network's
    online Q, with its weights as they stand at the call. The network is as the
    trainer's loop leaves it between two target syncs: updated before the
    episode and again after every decision (in the sink, as unit U8b's loop may
    update it), so at every call its online Q differs from its target copy's,
    which only the learner's targets read, and it moves between calls; a
    policy acting on the target copy, or on the weights of an earlier call,
    flies otherwise. Two numbers are drawn per decision, and each decision
    reaches the sink: its view, rows, mask, the online Q of each row as the
    decision used it, the row flown and its members."""
    rng = random.Random(6)
    live, batch = C.new_e3_network(seed=8), stand_in_batch(8)
    for _ in range(20):
        live.update(batch)
    trained = policy(seed=99, record=False)
    stream, steps = CountingRandom(1), []

    def sink(step):
        steps.append(step)
        live.update(batch)

    trained.attach_trainer(net=live, epsilon=0.0, rng=stream, sink=sink)
    assert trained.net is live and trained.training
    first = None
    decided = by_target = by_first = 0
    for _ in range(30):
        v, wps = random_case(rng, rng.randint(1, 6))
        mask = [rng.random() < 0.7 for _ in wps]
        after = rng.random() < 0.5
        rows, flags = C.e3_rows(v), np.array(mask)
        online, target = live.q(rows), live.q_target(rows)
        now = C.new_e3_network(seed=0)
        now.set_weights(live.weights())          # the online weights of this call
        first = now if first is None else first
        answer = ask(trained, v, wps, mask, after_stop=after)
        assert answer == ask(C.ChenDQNPolicy(now, band="wide"), v, wps, mask)
        if answer is None:
            continue
        decided += 1
        assert answer == Q.masked_argmax(online, flags) and not np.array_equal(online, target)
        by_target += answer != Q.masked_argmax(target, flags)
        by_first += answer != Q.masked_argmax(first.q(rows), flags)
        step = steps[-1]
        assert step.view is v and step.mask == tuple(mask) and step.row == answer
        assert np.array_equal(step.matrix, rows) and step.matrix.shape == (len(wps), 10)
        assert step.scores == tuple(online.tolist())
        assert np.array_equal(step.x, rows[answer]) and step.effective_mask == step.mask
        assert step.after_stop is after and step.devices == tuple(
            str(d) for d in wps[answer].devices)
    assert len(steps) == decided > 10 and stream.draws == 2 * decided
    assert (live.updates, live.syncs) == (20 + decided, 0)
    assert by_target >= 5 and by_first >= 2


def test_exploration_is_uniform_over_the_admissible_stops_only():
    """ε = 1: every admissible stop is flown sometimes, an inadmissible one never
    (``pair_q.behaviour_row``)."""
    v, wps = random_case(random.Random(7), 6)
    mask = [True, False, True, True, False, True]
    e3 = policy(record=False)
    e3.attach_trainer(net=C.new_e3_network(seed=0), epsilon=1.0, rng=random.Random(3),
                      sink=lambda step: None)
    picks = [ask(e3, v, wps, mask) for _ in range(400)]
    assert set(picks) == {0, 2, 3, 5}
    assert min(picks.count(i) for i in (0, 2, 3, 5)) > 60


@pytest.mark.parametrize("call, error, match", [
    (dict(net=None), TypeError, "PairQNet"),
    (dict(net=PairQNet(5, seed=0)), ValueError, "rows are 10"),
    (dict(epsilon=1.5), ValueError, r"\[0, 1\]"),
    (dict(epsilon=True), TypeError, "epsilon"),
    (dict(rng=object()), TypeError, "seeded stream"),
    (dict(sink=None), TypeError, "sink"),
    (dict(around_reference=True), ValueError, "no reference"),
    (dict(around_reference=1), TypeError, "around_reference"),
], ids=["no_net", "width", "epsilon", "bool_epsilon", "rng", "sink", "reference",
        "truthy_reference"])
def test_a_trainer_is_refused_unless_it_is_e3s(call, error, match):
    """The live network reads E3's rows; ε is a probability; the stream and the
    sink are callable; E3 has no reference phase (Chen's recipe has none)."""
    args = dict(net=C.new_e3_network(seed=0), epsilon=0.1, rng=random.Random(0),
                sink=lambda step: None)
    args.update(call)
    with pytest.raises(error, match=match):
        policy().attach_trainer(**args)


def test_a_trainer_attaches_before_the_first_call_only():
    """Every decision of an episode is trained alike: the mule builds a fresh
    policy per trial, and a trainer attached after the first call is refused,
    even when that call ended the pass."""
    v, wps = random_case(random.Random(8), 2)
    e3 = policy()
    assert ask(e3, v, wps, [False, False]) is None
    with pytest.raises(RuntimeError, match="before the policy's first call"):
        e3.attach_trainer(net=C.new_e3_network(seed=0), epsilon=0.0, rng=random.Random(0),
                          sink=lambda step: None)
    behaviour = Q.BehaviourSchedule(reference_episodes=0, epsilon_start=1.0,
                                    epsilon_end=0.1, decay_fraction=0.25).at(0, 10)
    fresh = policy()
    fresh.attach_trainer(net=C.new_e3_network(seed=0), rng=random.Random(0),
                         sink=lambda step: None, **dataclasses.asdict(behaviour))
    assert fresh.training


def test_a_missions_steps_are_its_pass_1_stops_flown_in_order():
    """For the trainer (unit U8b): the steps of a mission are its Pass-1 stops
    flown, one each, in order, with that stop's members, the departure's clock
    as the view's, the first at takeoff, each flying an admissible stop; a
    mission with no stop flown has none. So a trainer pairs step k with stop
    k's reward and forms the transitions as :class:`E3Step` says."""
    steps, marks = [], []
    e3 = policy(seed=3, record=False)
    e3.attach_trainer(net=C.new_e3_network(seed=3), epsilon=0.3, rng=random.Random(9),
                      sink=steps.append)
    _, _, recs = fly(e3, budget=120.0, missions=4, after=lambda m, r: marks.append(len(steps)))
    start = 0
    for rec, end in zip(recs, marks):
        flown = rec.result.pass_1_flown
        mine = steps[start:end]
        start = end
        assert len(mine) == len(flown) >= 1
        assert [list(s.devices) for s in mine] == [f["devices"] for f in flown]
        assert [s.view.clock_s for s in mine] == [f["depart_s"] for f in flown]
        assert [s.after_stop for s in mine] == [False] + [True] * (len(mine) - 1)
        assert all(s.mask[s.row] for s in mine)
    assert start == len(steps)
    none = []
    idle = policy(record=False)
    idle.attach_trainer(net=C.new_e3_network(seed=3), epsilon=0.3, rng=random.Random(9),
                        sink=none.append)
    _, _, (empty,) = fly(idle, budget=5.0)
    assert empty.result.pass_1_flown == [] and none == []


def test_a_missions_steps_form_the_pair_learners_transitions():
    """The steps feed the pair learner as they are (unit U8b's loop): step k's
    ``x``, then the next step's ``matrix`` and ``mask`` as the next decision, the
    mission's last step done; a batch of them updates the live network. The
    reward here is a stand-in (the stop's members over N); FerrySim's bytes
    reward is the real one."""
    steps, marks = [], []
    live = C.new_e3_network(seed=4, config=PairQConfig(gamma=0.9))
    e3 = policy(seed=4, record=False)
    e3.attach_trainer(net=live, epsilon=0.5, rng=random.Random(4), sink=steps.append)
    fly(e3, budget=200.0, missions=3, after=lambda m, r: marks.append(len(steps)))
    transitions, start = [], 0
    for end in marks:
        mission = steps[start:end]
        start = end
        for k, step in enumerate(mission):
            reward = len(step.devices) / step.view.demand
            if k + 1 < len(mission):
                nxt = mission[k + 1]
                transitions.append(PairTransition(step.x, reward, False, nxt.matrix,
                                                  np.array(nxt.mask)))
            else:
                transitions.append(PairTransition(step.x, reward, True))
    assert len(transitions) == len(steps) > 5 and sum(t.done for t in transitions) == len(marks)
    before = live.weights()
    loss = live.update(PairBatch.of(transitions))
    assert math.isfinite(loss) and live.updates == 1
    assert any(not np.array_equal(before[k], live.weights()[k]) for k in before)
    assert e3.net is live


def test_the_same_seeds_fly_the_same_steps_and_another_stream_differs():
    """Determinism (the spec, conventions): the same network and stream give the
    same steps and the same records, wall stamps aside; another stream explores
    otherwise."""
    def run(stream_seed):
        steps = []
        e3 = policy(seed=0, record=False)
        e3.attach_trainer(net=C.new_e3_network(seed=12), epsilon=0.5,
                          rng=random.Random(stream_seed), sink=steps.append)
        _, _, recs = fly(e3, budget=200.0, missions=3)
        return steps, [rec.result.pass_1_e3 for rec in recs]

    first, again, other = run(1), run(1), run(2)
    assert first == again and len(first[0]) > 5
    assert [s.row for s in first[0]] != [s.row for s in other[0]]


# --------------------------------------------------------------------------- #
# Pass 2, and what loads
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("pass_kind, match", [
    (DELIVER, "Pass-1 stops only"),
    ("deliver", "Pass-1 stops only"),
    ("ferry", "must name a mission pass"),
])
def test_pass_2_is_refused(pass_kind, match):
    """Critic B7 i: Pass 2 delivers to every slice stop nearest first, with no
    selector; E3 refuses a call there itself (``next_stop.pass_1_only``)."""
    v, wps = random_case(random.Random(9), 2)
    with pytest.raises(ValueError, match=match):
        ask(policy(), v, wps, [True, True], pass_kind=pass_kind)
    assert ask(policy(), v, wps, [True, True], pass_kind="collect") in (0, 1)


def _imports(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


def test_e3_imports_neither_the_plan_nor_any_runtime_package():
    """The spec's conventions: numpy, the standard library, ``hermes.types``, the
    protocol, the budget walk's check names and the pair learner, nothing else."""
    names = _imports(REPO / "hermes/scheduler/policies/chen_dqn.py")
    ours = {n for n in names if n.split(".")[0] not in sys.stdlib_module_names}
    assert ours == {"numpy", "hermes.types.scheduler", "hermes.scheduler.policies.budget_walk",
                    "hermes.scheduler.policies.next_stop", "hermes.scheduler.selector.pair_q"}
    assert ".chen_dqn" not in _imports(REPO / "hermes/scheduler/policies/__init__.py")


_LOADS = r"""
import sys
import hermes.scheduler.policies, hermes.scheduler.selector
base = set(sys.modules)
import hermes.scheduler.policies.chen_dqn
far = ("hermes.scheduler.plan", "hermes.l1", "hermes.mule", "hermes.mission", "experiments",
       "tests")
pairs = ("pair_slot", "pair_features", "pair_replay", "cross_heuristic")
print([sorted(m for m in base if m.endswith((".chen_dqn", ".next_stop", ".pair_q"))),
       sorted(m for m in sys.modules if m.startswith(far)),
       sorted(m for m in sys.modules if m.endswith(pairs))])

from hermes.mule.ferry import FerrySpec
from hermes.scheduler.policies.chen_dqn import ChenDQNPolicy, new_e3_network
from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

w = H.World(layout=GH.LAYOUT, flaky={})
spec = FerrySpec.from_config(rf_range_m=60.0, seed=7, contact_band="wide",
                             payload_bytes=1_000_000, in_flight_response="replan",
                             replan_fallback="trim")
with H.Patched(w.clock):
    sup = w.supervisor(w.mule_ids[0], sim=True, ferry=spec, mission_budget_s=120.0,
                       deadline_time_scale=1000.0,
                       target_selector=ChenDQNPolicy(new_e3_network(seed=0), band="wide"))
    w.bootstrap()
    results = [sup.run_one_mission() for _ in range(2)]
assert all(r.pass_1_e3 and r.pass_1_e3_unvisited for r in results)
print(sorted(m for m in sys.modules if m.startswith("hermes.scheduler.plan")
             or m.endswith(pairs)))
"""


def test_no_recorded_path_loads_e3_and_e3_loads_no_plan():
    """In a fresh interpreter: the policies and selector packages (every D arm
    and H2 load them) load neither E3 nor its protocol nor the pair learner;
    E3 loads nothing of the plan, l1, mule, mission, experiments or tests, nor
    any pair module; and E3's missions on the clock load no plan module."""
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", _LOADS], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().splitlines()[-2:] == ["[[], [], []]", "[]"]


def test_the_module_exports_what_it_documents():
    assert set(C.__all__) == {
        "E3_COLUMNS", "E3_DECLARED_CONSTANT", "E3_DIM", "E3_SCHEMA_VERSION", "LENGTH_SCALE_M",
        "SNR_SCALE_DB", "ChenDQNPolicy", "E3Sink", "E3Step", "e3_rows", "e3_schema",
        "load_e3_network", "new_e3_network", "save_e3_checkpoint"}
    assert all(hasattr(C, name) for name in C.__all__)
    assert json.loads(json.dumps(C.e3_schema())) == C.e3_schema()
