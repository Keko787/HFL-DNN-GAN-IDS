"""FeRRy Phase 4 (unit U7): plan-mode missions of the mule supervisor, in process.

Real ``ClientMission`` devices, a real cluster and a real ``MuleSupervisor`` on
a ``MissionClock`` (``tests/integration/_ferry_harness.py``: synchronous
threads, the wall clock parked at 1.7e9 s), flying the plan clock
(``plan_mode="ferry"``; the Phase 4 spec's units table, row U7). Pinned:

* **The wiring.** Plan mode and member subsets need the mission clock; plan
  mode refuses what it cannot fly (no band, no T_nom, a budgeted Pass 2,
  critic B8; ``abort`` with a cap, critic A10) and the scheduler refuses
  ``reorder`` (critic B11); a recorded mule builds its scheduler with exactly
  the recorded arguments (unit_U3b.md section 5.3).
* **The commit** (other choices 1 and 2): each mission flies its committed
  class b̄ in both passes, the annotations and Pass 2's radius included; FB+c
  flies only class c.
* **Critic B4's deterministic loopback** (A2, A4): the link keeps its sigma,
  the channel is built with its constructor and every noise term 0,
  availability is 1 and the deadline clause is out of the way; S* is computed
  on that same spec by an exact set cover that shares nothing with the
  planner (spec item 13), on the stops the plan is offered (the hover rule,
  placed by a brute force of this file's own). On all 30 of the critic's
  reference layouts, at 45 s and 60 s, the plan causes no violation at S*+1;
  at S* the myopic plan crowds capped devices on a few layouts (layout 25
  among them); at S*-1 the violations carry their reasons, each checked
  against the plan's own stops and the device's best hover point.
* **The hover rule** (the user's decision of 2026-09-30; final check PLAN-1,
  E2E2-01): FB+medium at 45 s serves the far devices that starved before
  (critic layouts 0, 4 and 26) from their hover points at S*+1, with no
  violation; at 30 s on layout 13 no mission flies empty while a capped
  device fits at its hover point; and on the S* tool's own layouts the plan
  labels ``crowded`` exactly what the tool can serve, the crowding that
  partition drift leaves at S*+1 pinned.
* **The close** (other choices 6): the visited set is the members of the
  stops flown, after an in-flight trim and with a beacon insert;
  ``dropped_in_flight`` and ``not_merged`` follow from it and from the merge,
  which under the age cutoff (``agg:cutoff``, the pilots') is not the round
  report's CLEAN lines; the empty path closes with nothing merged.
* **Drops** (other choices 8; decision 6): what the plan left out by choice
  is widened, recorded after the four clause reasons and kept out of S3c's
  planned count; an H arm's member subset complement is widened like a
  budget drop; a D arm's drops are reported (``pass_1_policy_drops``) entry
  for entry as the scheduler labelled them, and never widened.
* **In flight** (other choices 9): the departure check, the re-plan, the
  beacon hook and FX's ``fits`` hold the plan's exempt stops protected (an
  empty set with the cap off); the re-plan dates each member from the mule's
  record; capped members, a mixed stop's included, do not lower
  ``deliver_by``; the beacon hook groups an offer within the committed
  class's range; the order is departure check, beacon hook, slot.
* **FX** (decision 5; critic A7): after a stop, the nearest stop whose move to
  the front still fits (from the departure's state, the updates on board
  included), else the plan's next; on arrival, the fastest class reaching
  every committed target; neither at takeoff nor in Pass 2; at the last stop
  it lands no later than F.
* **K = 2** (critic B10): each mule plans its own slice, from its own ages.
* **Determinism and the defaults** (critic B12, D2): a repeated trial gives
  identical results bar ``plan_wall_s``; at the defaults no plan field appears
  and the plan package is never loaded; ``pass_1_policy_drops`` appears only
  when a D arm left something out.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.exp4.driver import Exp4Driver
from experiments.exp4.model_task import _u32
from experiments.exp4.topology_builder import device_positions, device_spread_m
from hermes.l1.channel_model import SALT_CONTACT, ContactChannel, ferry_salt
from hermes.mission.aggregation_rules import AggregationSpec
from hermes.mule import MuleSupervisorError
from hermes.mule import mule_main
from hermes.mule.ferry import FerryRuntime, FerrySpec
from hermes.scheduler import FLScheduler, FLSchedulerError
from hermes.scheduler.plan import AgeCapSpec, PlanOptions, PlanSetup
from hermes.scheduler.plan import types as PT
from hermes.scheduler.policies import FedExCarpPolicy, MaxAoIPolicy
from hermes.scheduler.policies.cross_heuristic import (
    CommittedSlot,
    CrossHeuristic,
    fastest_covering_class,
    moved_to_front,
)
from hermes.scheduler.stages.s3a_cluster import cluster_by_rf_range
from hermes.scheduler.stages.s3b_feasibility import RULE_NONE, FeasibilityModel, FlightState
from hermes.types import Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState, MissionPass
from hermes.types import MissionSlice, MuleID
from hermes.types.scheduler import (
    CAP_CROWDED,
    CAP_DROPPED_IN_FLIGHT,
    CAP_NOT_MERGED,
    CAP_PLAN_REASONS,
    CAP_UNPLANNABLE,
)

from tests.golden import _canon
from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

logging.getLogger("hermes.mission.client_mission").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parents[2]
COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
DOCK = (0.0, 0.0, 0.0)
BANDS = ("wide", "medium", "narrow")
N = 6
#: The critic's field (probe A): the realism field, radius 100 m (EX-4.2).
FIELD_M = device_spread_m(60.0, field_radius_m=100.0)
#: The deadline law's unit here: Deadline(j) = t0 + 60 000 s, so the deadline
#: clause never binds, as the critic's probes left deadlines out.
FAR_UNIT = 1000.0
#: T in the plan score (decision 2 (b)), near this cell's T_nom (the median
#: nominal period of the driver's 30 reference layouts on this spec, 202 s).
#: At kappa = 1 coverage dominates and T only breaks ties: the layout-25 trace
#: pinned below is the same at 150, 202 and 300 s.
T_NOM = 200.0


# --------------------------------------------------------------------------- #
# The world
# --------------------------------------------------------------------------- #

def ref_layout(k):
    """The critic's reference layout ``k`` (probe A): N = 6 in the 100 m field,
    seed ``_u32(6, "t_nom", 1000 + k)``, apart from T_nom's own layouts."""
    seed = _u32(N, "t_nom", 1000 + k)
    xy = device_positions(N, seed, FIELD_M)
    return seed, tuple((f"d{i}", (x, y, 0.0)) for i, (x, y) in enumerate(xy))


def det_spec(seed, layout, *, band="wide", payload_bytes=1_000_000, **kw):
    """Critic B4's deterministic physics.

    The link is the pilots' and keeps its sigma: ``from_config`` with sigma 0
    would cut every mean SNR's margin by 5.13 dB, and ``from_link`` refuses
    another sigma. The contact channel is built with its constructor and every
    noise term 0 (shadowing, the interference amplitude and noise), so each SNR
    is the link's mean and every contact takes what the planner priced.
    Availability is 1 (the pilots' ``channel`` source, every draw a success),
    so a device the plan serves is merged (critic A2). The backhaul is the
    recorded ``mission`` model in the jittery regime: the critic's probe A2
    built its layouts with the driver's jittery cell, and this spec reproduces
    its singleton homes on layout 25 to 0.1 s.
    """
    kw.setdefault("in_flight_response", "replan")
    kw.setdefault("replan_fallback", "trim")
    kw.setdefault("backhaul_regime", "jittery")
    spec = FerrySpec.from_config(
        rf_range_m=60.0, seed=seed, contact_band=band, payload_bytes=payload_bytes,
        contact_reliability_source="channel",
        device_availability={did: 1.0 for did, _ in layout}, **kw)
    link = spec.link
    quiet = ContactChannel(link.mean_snr_db, salt=ferry_salt(seed, SALT_CONTACT), bands=link.names,
                           shadow_sigma_db=0.0, interference_amp_db=0.0,
                           interference_sigma_db=0.0)
    return dataclasses.replace(spec, contact_channel=quiet)


def noisy_spec(seed, **kw):
    """The pilots' channel (jittery), 1 MB, ``replan`` with ``trim``."""
    kw.setdefault("payload_bytes", 1_000_000)
    return FerrySpec.from_config(rf_range_m=60.0, seed=seed, contact_band="wide",
                                 contact_regime="jittery", in_flight_response="replan",
                                 replan_fallback="trim", **kw)


def options(*, s=None, lookahead=0, policy="search", slot="committed", admission="subset"):
    return PlanOptions(band_class_policy=policy, member_admission=admission, flight_slot=slot,
                       cap=AgeCapSpec(s_missions=s, lookahead=lookahead))


def plan_kw(*, budget, s=None, unit=FAR_UNIT, admission="subset", **opt):
    """The supervisor keywords of a plan arm: F by default (search, member
    subsets, the committed slot, miss priority on)."""
    return dict(mission_budget_s=budget, deadline_time_scale=unit, miss_priority=True,
                member_admission=admission, plan_mode="ferry",
                plan_options=options(s=s, admission=admission, **opt), t_nom_s=T_NOM)


def fly(*, spec, layout, missions=1, before=None, silent=(), world_kw=None, **sup_kw):
    """Run ``missions`` missions on the clock; one record per mission."""
    w = H.World(layout=layout, flaky={}, **(world_kw or {}))
    mid = w.mule_ids[0]
    w.rfs[mid].silent.update(DeviceID(d) for d in silent)
    records = []
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=True, ferry=spec, **sup_kw)
        w.bootstrap()
        for m in range(missions):
            if before is not None:
                before(w, sup, m)
            r = sup.run_one_mission()
            records.append(SimpleNamespace(result=r, deltas=w.take_deltas(mid)))
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return w, sup, records


def violations(r):
    return [(v["device"], v["age"], v["reason"]) for v in r.plan["cap"]["violations"]]


def widened(rec):
    """The devices the mule gave a synthetic TIMEOUT this mission (widening)."""
    return sorted({str(d.device_id) for src, d in rec.deltas if src == "direct" and d.synthetic})


def _planner(layout):
    """A fresh legacy scheduler holding ``layout``: every device new, in the slice."""
    pos = {DeviceID(d): p for d, p in layout}
    planner = FLScheduler(now_fn=lambda: 0.0)
    planner.ingest_slice(MissionSlice(mule_id=MuleID("probe"), device_ids=tuple(pos),
                                      issued_round=0, issued_at=0.0))
    for d, p in pos.items():
        planner.device_states[d].last_known_position = p
    return planner


def class_model(spec, planner, band):
    """``spec``'s physics on ``band`` (the declared payload), bound to ``planner``."""
    rt = FerryRuntime(spec, None, rf_range_m=60.0)
    rt.set_payload(theta_bytes=52)
    model = rt.feasibility_model(FeasibilityModel(cruise_speed_m_s=spec.flight.cruise_speed_m_s),
                                 band=band)
    return dataclasses.replace(model, ferry=model.ferry.bind(planner.device_states))


#: Points of the dock -> device segment this file's own hover rule tries (a
#: brute force: the plan's hover point is plan/hover.py's).
SEGMENT_POINTS = 2000


def alone_home(model, position, did):
    """``did`` alone at ``position``: the home from the dock at 0 s, no gate, the
    upload included; None when the stop would not reach it (beyond the class's
    range, or below the floor), as the model would then charge it nothing."""
    wp = ContactWaypoint(position=tuple(position), devices=(DeviceID(did),),
                         bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=math.inf)
    (dist,) = model.ferry.member_distances_m(wp)
    if dist > model.ferry.range_m or model.ferry.member_dwell_s(dist, COLLECT, 0.0) is None:
        return None
    return model.fold([wp], FlightState(DOCK, 0.0), rule=RULE_NONE, budget_end=None,
                      skip=False).home


def fits_alone(model, position, did, budget):
    """True when ``did`` alone at ``position`` is home within ``budget``."""
    home = alone_home(model, position, did)
    return home is not None and home <= budget


def segment_hover(model, did, where):
    """``did``'s best hover point by brute force: of ``SEGMENT_POINTS + 1``
    points evenly spread on the segment from ``did`` (at ``where``) towards the
    dock, as far as the class's range, the one with the earliest alone home;
    ``(home, point)``."""
    length = math.dist(where, DOCK)
    best = None
    for i in range(SEGMENT_POINTS + 1):
        r = min(length, model.ferry.range_m) * i / SEGMENT_POINTS
        t = r / length if length else 0.0
        point = tuple(a + t * (b - a) for a, b in zip(where, DOCK))
        home = alone_home(model, point, did)
        if home is not None and (best is None or home < best[0]):
            best = (home, point)
    return best


def hover_family(stops, model, budget, where):
    """The user's hover rule (2026-09-30) on ``stops``, every device as if capped:
    a device its S3a stop cannot serve alone within ``budget`` leaves that stop
    for a stop of its own at its :func:`segment_hover` point."""
    out = []
    for wp in stops:
        moved = [d for d in wp.devices if not fits_alone(model, wp.position, d, budget)]
        rest = tuple(d for d in wp.devices if d not in moved)
        if rest:
            out.append(dataclasses.replace(wp, devices=rest))
        out += [dataclasses.replace(wp, devices=(d,),
                                    position=segment_hover(model, d, where[d])[1])
                for d in moved]
    return out


def s_star(spec, layout, budget, bands=None):
    """Spec item 13's S*, computed here with nothing shared with the planner.

    Planning level, on ``spec``'s own physics, under the budget rule with the
    deadlines left out: a set of devices is servable in one mission when, on
    some band class (of ``bands``, every class by default), the stops a fresh
    plan is offered (S3a's, every device new, as the S* tool clusters them,
    with the hover rule, :func:`hover_family`) reduced to the set and flown in
    their best order from the dock are home with the upload done within
    ``budget``. S* is the fewest such sets that cover the slice, an exact set
    cover (N = 6); None when none covers it. Priced by a plain
    ``FeasibilityModel.fold`` over every order: none of the search (U4), the
    cap (U1), the score (U2) or the hover module.
    """
    planner = _planner(layout)
    devs = sorted(planner.device_states)
    where = {DeviceID(d): p for d, p in layout}
    feasible = set()
    for name in bands or spec.link.names:
        model = class_model(spec, planner, name)
        stops = planner.build_contact_queue(rf_range_m=spec.link.range_planar_m(name),
                                            mule_pose=DOCK)
        stops = hover_family(stops, model, budget, where)
        owner = {d: wp for wp in stops for d in wp.devices}
        for r in range(1, len(devs) + 1):
            for subset in itertools.combinations(devs, r):
                touched = {}
                for d in subset:
                    touched.setdefault(owner[d], []).append(d)
                reduced = [dataclasses.replace(wp, devices=tuple(d for d in wp.devices if d in ms))
                           for wp, ms in touched.items()]
                home = min(model.fold(list(order), FlightState(DOCK, 0.0), rule=RULE_NONE,
                                      budget_end=None, skip=False).home
                           for order in itertools.permutations(reduced))
                if home <= budget:
                    feasible.add(frozenset(subset))
    if not feasible or frozenset().union(*feasible) != frozenset(devs):
        return None
    maximal = [s for s in feasible if not any(s < t for t in feasible)]
    for k in range(1, len(devs) + 1):
        if any(frozenset().union(*combo) == frozenset(devs)
               for combo in itertools.combinations(maximal, k)):
            return k
    return None


def priced_home(spec, layout, order, *, band="wide"):
    """The planner's Pass-1 home of flying ``order`` (device ids, each a stop of
    its own at its position, as S3a makes LINE3's and the beacon hook an
    insert's) on ``band`` from the dock at 0 s, the upload included."""
    planner = _planner(layout)
    model = class_model(spec, planner, band)
    where = dict(layout)
    route = [ContactWaypoint(position=where[d], devices=(DeviceID(d),),
                             bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=math.inf)
             for d in order]
    return model.fold(route, FlightState(DOCK, 0.0), rule=RULE_NONE, budget_end=None,
                      skip=False).home


# --------------------------------------------------------------------------- #
# The wiring
# --------------------------------------------------------------------------- #

def test_the_plan_modes_are_the_plan_packages():
    """mule_main restates them, so a legacy mule never loads the plan package."""
    assert mule_main._PLAN_MODES == PT.PLAN_MODES
    assert mule_main._PLAN_MODE_LEGACY == PT.PLAN_MODE_LEGACY
    assert mule_main._PLAN_MODE_FERRY == PT.PLAN_MODE_FERRY


def _build(*, sim=True, spec=None, **kw):
    """A supervisor on layout 0, built and not flown (the harness ignores
    ``ferry`` on the wall clock)."""
    seed, layout = ref_layout(0)
    w = H.World(layout=layout, flaky={})
    ferry = det_spec(seed, layout) if spec is None else spec
    with H.Patched(w.clock):
        return w.supervisor(w.mule_ids[0], sim=sim, ferry=ferry, **kw)


@pytest.mark.parametrize("kw, match", [
    (dict(member_admission="subset"), "runs on the mission clock"),
    (dict(member_admission="bogus"), "runs on the mission clock"),
    (dict(plan_mode="ferry", plan_options=options(), t_nom_s=T_NOM), "plans on the mission clock"),
    (dict(plan_mode="nope"), "plan_mode must be one of"),
    (dict(plan_options=options()), "configure plan mode"),
    (dict(t_nom_s=T_NOM), "configure plan mode"),
])
def test_without_the_mission_clock_only_the_recorded_values_pass(kw, match):
    """unit_U3b.md section 1.4 and the Phase 4 spec, other choices 10: member
    subsets price each member with the ferry physics and the plan clock plans
    with it, which only the mission clock carries."""
    with pytest.raises(MuleSupervisorError, match=match):
        _build(sim=False, **kw)


def _plan_refusals():
    seed, layout = ref_layout(0)
    good = plan_kw(budget=45.0)
    return [
        (dict(good, spec=FerrySpec()), "needs a contact band"),
        (dict(good, t_nom_s=None), "set t_nom_s"),
        (dict(good, plan_options=None), "needs plan_options"),
        (dict(good, plan_options=options().describe()), "needs plan_options"),
        (dict(good, pass_2_budget=True), "critic B8"),
        (dict(plan_kw(budget=45.0, s=2), spec=det_spec(seed, layout, in_flight_response="abort")),
         "critic A10"),
        (dict(good, t_nom_s=0.0), "t_ref_s must be > 0"),
        (dict(good, plan_options=options(policy="fixed:medium")),
         "must pin the run's contact_band"),
        (dict(plan_mode="nope"), "plan_mode must be one of"),
        (dict(plan_options=options()), "configure plan mode"),
    ]


@pytest.mark.parametrize("kw, match", _plan_refusals())
def test_plan_mode_refuses_what_it_cannot_fly(kw, match):
    with pytest.raises(MuleSupervisorError, match=match):
        _build(**kw)


def test_the_scheduler_refuses_what_plan_mode_cannot_honour():
    """The scheduler owns these refusals, and the supervisor passes it what it
    needs to make them: ``reorder`` (re-ordering belongs to the flight slot,
    critic B11), options whose ``member_admission`` is not the switch's (one
    source, R5), a target selector, and an unknown ``member_admission``."""
    seed, layout = ref_layout(0)
    reorder = det_spec(seed, layout, replan_fallback="reorder")
    with pytest.raises(FLSchedulerError, match="'reorder' is refused"):
        _build(spec=reorder, **plan_kw(budget=45.0))
    with pytest.raises(FLSchedulerError, match="one source"):
        _build(**dict(plan_kw(budget=45.0), member_admission="whole"))
    with pytest.raises(FLSchedulerError, match="no target_selector"):
        _build(target_selector=MaxAoIPolicy(), **plan_kw(budget=45.0))
    with pytest.raises(FLSchedulerError, match="member_admission must be one of"):
        _build(member_admission="bogus")


def test_abort_is_flown_in_plan_mode_without_a_cap():
    """Critic A10 refuses ``abort`` only with a cap: abort gives up the whole
    tail, which only a cap forbids."""
    seed, layout = ref_layout(0)
    spec = det_spec(seed, layout, in_flight_response="abort")
    _, sup, (rec,) = fly(spec=spec, layout=layout, **plan_kw(budget=45.0))
    assert rec.result.plan["served"] and violations(rec.result) == []
    assert isinstance(sup._flight_slot, CommittedSlot)


class _Recording(FLScheduler):
    calls = []

    def __init__(self, **kw):
        type(self).calls.append(dict(kw))
        super().__init__(**kw)


def test_a_recorded_mule_builds_its_scheduler_with_the_recorded_arguments(monkeypatch):
    """unit_U3b.md section 5.3: the Phase 4 switches reach the scheduler only
    when they are not the recorded values, so a recorded mule, on either
    clock, builds its scheduler with exactly the arguments it always did.
    Plan mode passes the setup the mule built: one class per class of the
    link, the run's band as the reference, T = ``t_nom_s`` and the turnaround."""
    monkeypatch.setattr(mule_main, "FLScheduler", _Recording)
    new = {"member_admission", "plan_mode", "plan"}
    _Recording.calls.clear()
    _build(sim=False)
    _build()
    _build(member_admission="whole")
    assert [new & set(kw) for kw in _Recording.calls] == [set(), set(), set()]
    _Recording.calls.clear()
    _build(member_admission="subset")
    (kw,) = _Recording.calls
    assert {k: kw[k] for k in new & set(kw)} == {"member_admission": "subset"}
    _Recording.calls.clear()
    sup = _build(**plan_kw(budget=45.0, slot="cross_heuristic"))
    (kw,) = _Recording.calls
    assert kw["plan_mode"] == "ferry" and kw["member_admission"] == "subset"
    setup = kw["plan"]
    assert isinstance(setup, PlanSetup)
    assert [(c.name, c.index, c.radius_m) for c in setup.classes] == [
        (name, i, sup.ferry.link.range_planar_m(name)) for i, name in enumerate(BANDS)]
    assert (setup.reference, setup.t_ref_s, setup.turnaround_s) == ("wide", T_NOM, 30.0)
    assert setup.options == options(slot="cross_heuristic")
    assert isinstance(sup._flight_slot, CrossHeuristic)
    assert sup.scheduler.plan_mode == "ferry" and sup.scheduler.member_admission == "subset"


# --------------------------------------------------------------------------- #
# The commit: b̄ in both passes
# --------------------------------------------------------------------------- #

def test_each_mission_flies_its_committed_class_in_both_passes():
    """Other choices 2: the plan commits one class b̄ per mission and the
    runtime flies it in both passes. The annotations, every stop of both passes
    and the result's ``band`` are b̄; Pass 2 clusters at R_planar(b̄) (one
    field-wide stop on narrow, where wide's 60 m needs several). b̄ moves
    between missions, as the plan chooses it again at every dock."""
    seed, layout = ref_layout(25)
    spec = det_spec(seed, layout)
    where = {d: p for d, p in layout}
    _, sup, recs = fly(spec=spec, layout=layout, missions=3, **plan_kw(budget=45.0, s=3))
    committed = []
    for rec in recs:
        r = rec.result
        band = r.plan["band"]
        committed.append(band)
        radius = spec.link.range_planar_m(band)
        assert r.band == band
        assert [(wp.band, wp.range_m) for wp in r.pass_1_queue] == [(band, radius)] * len(
            r.pass_1_queue)
        assert {s["band"] for s in r.pass_1_flown + r.pass_2_flown} == {band}
        pass_2 = [s for s in r.pass_2_flown]
        assert sorted(d for s in pass_2 for d in s["devices"]) == sorted(where)
        for s in pass_2:
            assert all(math.dist(s["position"][:2], where[d][:2]) <= radius + 1e-9
                       for d in s["devices"])
        if band == "narrow":
            assert len(pass_2) == 1
        assert r.plan["visited"] == sorted(d for s in r.pass_1_flown for d in s["devices"])
        assert r.plan["visited"] == r.plan["served"]
        assert 0.0 <= r.plan_wall_s < 1.0             # the pilots' "at most 1 s per mission"
        assert "wall" not in json.dumps(r.plan)       # the only wall time is plan_wall_s
        # Design D-M: in this world the mission flies exactly what the commit
        # priced, both passes on b̄ (R4: the predicted whole mission).
        assert r.plan["score"]["mission_s"] == pytest.approx(r.sim_end_s - r.sim_start_s,
                                                            abs=1e-6)
    assert committed == ["narrow", "medium", "medium"]
    assert sup._ferry_run.band == committed[-1]


@pytest.mark.parametrize("budget", (45.0, 60.0))
def test_f_never_ranks_below_the_best_fixed_class_under_the_plans_key(budget):
    """Critic A3 (U4, U7): "F never ranks below the best FB+" holds for the
    plan's key: F searches every class and FB+c only c, so F's commit is the
    best FB+c's under (cap key, -served weight share, -V, class index), the
    default ``lexicographic`` rank's key (R11). Mission 1 of the 30 layouts
    with every device capped (S = 1), so the cap key decides first; each arm
    flown by the supervisor from the same world, the keys read from the
    commits it records."""
    for k in range(30):
        seed, layout = ref_layout(k)
        keys = {}
        for policy in ("search",) + tuple(f"fixed:{b}" for b in BANDS):
            spec = det_spec(seed, layout, band=policy.split(":")[-1] if ":" in policy else "wide")
            _, _, (rec,) = fly(spec=spec, layout=layout,
                               **plan_kw(budget=budget, s=1, policy=policy))
            p = rec.result.plan
            cap = p["cap"]
            unserved = sorted((cap["ages"][d] for d in cap["capped"] if d not in p["served"]),
                              reverse=True)
            share = p["score"]["served_weight"] / p["score"]["demand_weight"]
            keys[policy] = (tuple(unserved), -round(share, 9), -round(p["score"]["v"], 9),
                            p["band_index"])
        best_fixed = min(key for policy, key in keys.items() if policy != "search")
        assert keys["search"] == best_fixed, (k, keys)


@pytest.mark.parametrize("band", BANDS)
def test_fb_plus_c_flies_only_class_c(band):
    """Decision 7's "FB+c flies only class c": with ``fixed:c`` the plan
    searches c alone and every stop of both passes flies c."""
    seed, layout = ref_layout(25)
    spec = det_spec(seed, layout, band=band)
    _, _, recs = fly(spec=spec, layout=layout, missions=3,
                     **plan_kw(budget=45.0, s=3, policy=f"fixed:{band}"))
    for rec in recs:
        r = rec.result
        assert r.plan["band"] == r.band == band
        assert r.plan["band_class_policy"] == f"fixed:{band}"
        assert [c["band"] for c in r.plan["per_class"]] == [band]
        assert r.pass_1_flown and {s["band"] for s in r.pass_1_flown + r.pass_2_flown} == {band}


# --------------------------------------------------------------------------- #
# Critic B4's deterministic loopback (A2, A4)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("budget, spread", [(45.0, {1: 4, 2: 23, 3: 3}), (60.0, {1: 8, 2: 22})])
def test_at_s_star_plus_one_the_plan_causes_no_violation(budget, spread):
    """Critic A4, the design's third test: at S = S*+1 over 3S missions no
    capped device is ever left out, on each of the critic's 30 layouts at both
    budgets. The world is deterministic, so nothing but the plan could cause a
    violation, and nothing does: no violation of any cause, no re-plan, no
    overrun. S* here is the tool's (spec item 13), computed independently on
    this very spec (critic B4), on the stops the plan is offered (the hover
    rule, :func:`hover_family`). Its spread over the layouts is the design's
    and the critic's (probe A: 45 s {1: 4, 2: 22, 3: 4}, 60 s {1: 8, 2: 22})
    but for layout 12 at 45 s, whose S* falls from 3 to 2 when its far device
    is served from its hover point."""
    counts = {}
    for k in range(30):
        seed, layout = ref_layout(k)
        spec = det_spec(seed, layout)
        ss = s_star(spec, layout, budget)
        counts[ss] = counts.get(ss, 0) + 1
        s = ss + 1
        _, _, recs = fly(spec=spec, layout=layout, missions=3 * s, **plan_kw(budget=budget, s=s))
        for rec in recs:
            r = rec.result
            assert violations(r) == [], (k, r.mission_round)
            assert r.replans == [] and r.budget_overrun_s == 0.0, (k, r.mission_round)
            assert r.plan["cap"]["s"] == s
    assert counts == spread


def test_at_s_star_the_myopic_plan_can_crowd_a_capped_device():
    """Critic A4 pinned layout 25 at S = S* as a pinwheel: two capped devices
    cannot share a mission, so the planner, which looks one mission ahead,
    leaves one out although a covering schedule exists.

    Under the plan score the user chose (decision 2 (b): the whole mission,
    Pass 2 included, against T_nom) the pinwheel is stronger than the critic
    found under the design's score (Pass 1 against the budget): one
    ``crowded`` there, three here over the 3S = 6 missions. Mission 1 already
    differs: this score takes narrow {d0, d1, d2}, whose Pass 2 is the
    shortest, where the critic's took medium {d0, d1, d4}. Pinned as flown;
    across the 30 layouts at S* only layouts 1, 18 and 25 crowd anyone, and
    only ``crowded`` occurs. Layout 18 crowds twice under the default
    ``lexicographic`` rank (R11) and three times under ``weighted``: at
    mission 4 both leave the capped d4 out, and the served weight share then
    takes wide {d0, d1, d2, d3} (0.71) where V alone took the shorter medium
    {d1, d2, d3} (0.65), so d0 is not capped at mission 5."""
    seed, layout = ref_layout(25)
    spec = det_spec(seed, layout)
    assert s_star(spec, layout, 45.0) == 2
    _, _, recs = fly(spec=spec, layout=layout, missions=6, **plan_kw(budget=45.0, s=2))
    first = recs[0].result.plan
    assert (first["band"], first["served"]) == ("narrow", ["d0", "d1", "d2"])
    found = [(rec.result.mission_round,) + v for rec in recs for v in violations(rec.result)]
    assert found == [(2, "d4", 2, CAP_CROWDED), (5, "d4", 2, CAP_CROWDED),
                     (6, "d5", 2, CAP_CROWDED)]
    crowding = {}
    for k in range(30):
        seed, layout = ref_layout(k)
        spec = det_spec(seed, layout)
        s = s_star(spec, layout, 45.0)
        _, _, recs = fly(spec=spec, layout=layout, missions=3 * s, **plan_kw(budget=45.0, s=s))
        reasons = [v[2] for rec in recs for v in violations(rec.result)]
        assert set(reasons) <= {CAP_CROWDED}
        if reasons:
            crowding[k] = len(reasons)
    assert crowding == {1: 1, 18: 2, 25: 3}


def _plan_time_servability(sup):
    """Wrap ``build_ferry_plan`` to record, per mission, which capped device
    left out a class could serve alone: at the stop the plan clustered it in,
    or at its best hover point (the user's decision of 2026-09-30).

    Re-derived from the plan's own inputs right after it is made: each
    searched class's S3a stops over the plan's demand (the scheduler's states,
    its deadlines), each device alone at its stop, and alone at this file's
    brute-force hover point (:func:`segment_hover`), priced by the class's
    model from the dock at takeoff, under the budget only (a capped device is
    exempt from its own deadline). Neither U1's ``servable_alone``, nor U5's
    labels, nor the hover module is used."""
    seen = {}
    real = sup.scheduler.build_ferry_plan

    def build(**kw):
        route = real(**kw)
        sch = sup.scheduler
        commit = sch.last_plan
        budget = commit.budget_end - sch.mission_start_ts
        alone = set()
        for cls in sch.plan_setup.searched:
            stops = cluster_by_rf_range(eligible_device_ids=list(commit.demand),
                                        device_states=sch.device_states,
                                        deadlines=sch.last_plan_deadlines, rf_range_m=cls.radius_m)
            for wp in stops:
                for d in wp.devices:
                    where = tuple(sch.device_states[d].last_known_position)
                    if (fits_alone(cls.model, wp.position, d, budget)
                            or segment_hover(cls.model, d, where)[0] <= budget):
                        alone.add(str(d))
        seen[commit.mission_round] = alone
        return route

    sup.scheduler.build_ferry_plan = build
    return seen


def test_at_s_star_minus_one_the_violations_carry_their_reasons():
    """Below S* the cap asks for more than any schedule gives, so violations
    occur, on every layout with S* >= 2 (45 s), each with its planning age and
    the plan-time reason: ``crowded`` when some class could have served the
    device alone, at the stop the plan clustered it in or at its best hover
    point, ``unplannable`` otherwise (other choices 6; the user's decision of
    2026-09-30), checked against the plan's own stops and this file's own
    hover points. At 45 s every device of these layouts fits alone at its
    hover point on some class, so every violation is ``crowded``: layout 13's
    d3, at mission 2, was ``unplannable`` before the hover rule (S3a had
    re-clustered it into a narrow stop that cannot serve it alone within
    45 s). FB+wide at 30 s cannot serve layout 26's d1 even from its hover
    point, at the edge of wide's 60 m reach (34.0 s alone): it is
    ``unplannable`` at every mission, and its stop there is its hover stop."""
    total = {}
    for k in range(30):
        seed, layout = ref_layout(k)
        spec = det_spec(seed, layout)
        s = s_star(spec, layout, 45.0) - 1
        if s < 1:
            continue
        seen = {}

        def before(w, sup, m, seen=seen):
            if m == 0:
                seen["alone"] = _plan_time_servability(sup)

        _, _, recs = fly(spec=spec, layout=layout, missions=3 * s, before=before,
                         **plan_kw(budget=45.0, s=s))
        found = []
        for rec in recs:
            r = rec.result
            ages = r.plan["cap"]["ages"]
            for device, age, reason in violations(r):
                assert age == ages[device] >= s and device not in r.plan["served"]
                assert reason == (CAP_CROWDED if device in seen["alone"][r.mission_round]
                                  else CAP_UNPLANNABLE), (k, r.mission_round, device)
                found.append((r.mission_round, device, reason))
        assert found, k
        total[k] = found
        if k == 13:
            assert found == [(2, "d3", CAP_CROWDED), (2, "d4", CAP_CROWDED),
                             (5, "d3", CAP_CROWDED), (6, "d1", CAP_CROWDED)]
    assert len(total) == 26
    assert {reason for found in total.values() for _, _, reason in found} == {CAP_CROWDED}
    seed, layout = ref_layout(26)
    spec = det_spec(seed, layout)
    seen = {}

    def before(w, sup, m):
        if m == 0:
            seen["alone"] = _plan_time_servability(sup)

    _, sup, recs = fly(spec=spec, layout=layout, missions=2, before=before,
                       **plan_kw(budget=30.0, s=1, policy="fixed:wide"))
    model = sup.scheduler.plan_setup.class_named("wide").model
    home, point = segment_hover(model, "d1", dict(layout)["d1"])
    assert home == pytest.approx(34.01, abs=0.01)
    for rec in recs:
        r = rec.result
        got = {d: reason for d, _, reason in violations(r)}
        assert got["d1"] == CAP_UNPLANNABLE and "d1" not in seen["alone"][r.mission_round]
        assert {d: reason for d, reason in got.items() if d != "d1"} == {
            d: CAP_CROWDED for d in got if d != "d1"}
        (drop,) = [d for d in r.pass_1_preflight_drops if "d1" in d["devices"]]
        assert drop["devices"] == ["d1"] and drop["reason"] == "budget"
        assert math.dist(drop["position"], point) < 0.1


# --------------------------------------------------------------------------- #
# The hover rule (the user's decision of 2026-09-30; final check PLAN-1, E2E2-01)
# --------------------------------------------------------------------------- #

#: The final check's starving devices: FB+medium at 45 s, 1 MB, on the critic's
#: layouts. Each one's alone home at its own position (the far singleton S3a
#: made of it from mission 3 or 4 on), the first mission it was ``unplannable``
#: in before the fix, and its alone home at its best hover point: the dock for
#: d5 and d3 (within medium's 119.5 m reach of it), the edge of that reach for
#: d1 (136.2 m out).
STARVED = {0: ("d5", 47.40, 5, 13.25), 4: ("d3", 46.63, 5, 13.25), 26: ("d1", 56.65, 4, 19.93)}


@pytest.mark.parametrize("k", sorted(STARVED))
def test_fb_medium_serves_the_far_device_from_its_hover_point_at_s_star_plus_one(k):
    """The verifier's FB+c S*+1 test (PLAN-1, E2E2-01): FB+medium at the 45 s
    stress budget, 1 MB, on critic B4's loopback, at S = the class's S*+1 (2 +
    1 on each layout) over 3S missions. S3a re-clusters every mission on S3's
    deadlines; a device the plan leaves out is widened, anchors last, and
    becomes a stop of its own at its own position, over the budget alone, so
    before the fix it was ``unplannable`` (read as physics) from mission 5, 5
    and 4 to the last, and never served again. Now it leaves that stop, once
    capped, for its best hover point, where it fits: no violation of any
    cause, no capped device left out, every planning age at most S, and the
    device is served every S missions at the latest, from its hover stop
    whenever it is capped."""
    device, own_home, first_starved, hover_home = STARVED[k]
    seed, layout = ref_layout(k)
    where = dict(layout)
    spec = det_spec(seed, layout, band="medium")
    s = s_star(spec, layout, 45.0, bands=("medium",)) + 1
    assert s == 3
    planner = _planner(layout)
    model = class_model(spec, planner, "medium")
    assert alone_home(model, where[device], device) == pytest.approx(own_home, abs=0.01)
    assert segment_hover(model, device, where[device])[0] == pytest.approx(hover_home, abs=0.01)
    _, _, recs = fly(spec=spec, layout=layout, missions=3 * s,
                     **plan_kw(budget=45.0, s=s, policy="fixed:medium"))
    assert first_starved <= 3 * s
    served = []
    for rec in recs:
        r = rec.result
        p = r.plan
        assert violations(r) == [], (k, r.mission_round)
        assert set(p["cap"]["capped"]) <= set(p["served"])
        assert max(p["cap"]["ages"].values()) <= s
        if device in p["served"]:
            served.append(r.mission_round)
        if device in p["cap"]["capped"]:
            (stop,) = [x for x in r.pass_1_flown if device in x["devices"]]
            assert stop["devices"] == [device]
            reach = spec.link.range_planar_m("medium")
            assert math.dist(stop["position"], where[device]) <= reach + 1e-9
            assert alone_home(model, stop["position"], device) == pytest.approx(hover_home,
                                                                                abs=0.01)
    gaps = [b - a for a, b in zip([0] + served, served + [3 * s + 1])]
    assert served and max(gaps) <= s, served


def _capped_fit_at_hover(sup, notes):
    """Wrap ``build_ferry_plan`` to record, per mission, the capped devices that
    some class serves alone at their best hover point (:func:`segment_hover`,
    this file's brute force) from the dock at takeoff, within the budget."""
    real = sup.scheduler.build_ferry_plan

    def build(**kw):
        route = real(**kw)
        sch = sup.scheduler
        commit = sch.last_plan
        budget = commit.budget_end - sch.mission_start_ts
        notes[commit.mission_round] = sorted(
            str(d) for d in commit.capped
            if any(segment_hover(c.model, d, tuple(sch.device_states[d].last_known_position))[0]
                   <= budget for c in sch.plan_setup.searched))
        return route

    sup.scheduler.build_ferry_plan = build


def test_fb_medium_at_30_s_never_flies_empty_while_a_capped_device_fits_at_its_hover_point():
    """The final check's empty missions (PLAN-1, its verifier on critic layout
    13): FB+medium at 30 s, 1 MB, S = 3, on critic B4's loopback, ten missions.
    Before the fix the plan flew empty from mission 3 on (seven of eight empty
    missions with a capped device that fits alone at its hover point, all
    labelled ``unplannable``): every S3a stop of medium was over 30 s for each
    member alone. Now no mission flies empty while a capped device fits
    there, and none is ``unplannable``. Mission 2 still flies empty, with
    nothing capped: the hover rule moves capped devices only, and uncapped
    ones keep S3a's stops (the decision)."""
    seed, layout = ref_layout(13)
    notes = {}

    def before(w, sup, m):
        if m == 0:
            _capped_fit_at_hover(sup, notes)

    _, _, recs = fly(spec=det_spec(seed, layout, band="medium"), layout=layout, missions=10,
                     before=before, **plan_kw(budget=30.0, s=3, policy="fixed:medium"))
    empty = []
    for rec in recs:
        r = rec.result
        assert CAP_UNPLANNABLE not in {reason for _, _, reason in violations(r)}
        if not r.plan["served"]:
            empty.append(r.mission_round)
            assert notes[r.mission_round] == [], r.mission_round
    assert empty == [2] and recs[1].result.plan["cap"]["capped"] == []
    assert all(notes[m] for m in range(3, 11))


#: The final check's E2E2-01 cases, on the S* tool's own layouts (1 MB, 45 s):
#: per (family, layout), the ``crowded`` violations (mission, device) the plan
#: still has at the tool's S*+1 over 3S missions. They come from partition
#: drift: from mission 3 S3a re-clusters layout 0's medium stops around
#: exp4-dev-005, where each capped device fits alone but not beside another,
#: so one is left out each mission, in turn, and served a mission later.
DRIFT = {
    ("FB+medium", 0): [(4, "exp4-dev-003"), (5, "exp4-dev-004"), (6, "exp4-dev-000"),
                       (7, "exp4-dev-002"), (8, "exp4-dev-003"), (9, "exp4-dev-004")],
    ("FB+medium", 27): [],
    ("FB+wide", 29): [(4, "exp4-dev-002")],
}


@pytest.mark.parametrize("family, k", sorted(DRIFT))
def test_at_the_tools_s_star_plus_one_the_plan_labels_as_the_tool_serves(family, k):
    """The final check's E2E2-01 (the user's decision of 2026-09-30): the S*
    tool and the plan price one stop family, so in the mission loop a capped
    device the plan leaves out is ``crowded`` exactly when the tool can serve
    it, whatever S3a's partition has drifted to. On the tool's own layouts 0
    and 27 (FB+medium) and 29 (FB+wide) at 45 s, critic B4's loopback, at the
    tool's S*+1 (2 + 1, as this file's own S* finds) over 3S missions: no
    ``unplannable``, where before the fix layout 27's exp4-dev-000, which the
    tool serves, was ``unplannable`` from mission 5 on and never served again.
    What S*+1 does not remove is the crowding of partition drift
    (:data:`DRIFT`), each device at most one mission past S."""
    from experiments.analysis import age_cap_s_star as S

    driver = Exp4Driver(mission_clock="sim", realism=True, contact_band="wide",
                        payload_bytes=1_000_000)
    theta, synth = driver._payload_bytes(None)
    tool_layout = S.reference_layouts(N, count=k + 1, spread_m=FIELD_M)[k]
    world = S.planning_world(driver, tool_layout, rf_range_m=60.0, regime="jittery",
                             theta_bytes=theta, synth_bytes=synth)
    band = family[len("FB+"):]
    star = S.layout_s_star(world, classes=(band,), budget_s=45.0)
    layout = tool_layout.positions
    spec = det_spec(tool_layout.seed, layout, band=band)
    assert s_star(spec, layout, 45.0, bands=(band,)) == star.s_star == 2
    assert star.unservable == ()
    s = star.s_star + 1
    _, _, recs = fly(spec=spec, layout=layout, missions=3 * s,
                     **plan_kw(budget=45.0, s=s, policy=f"fixed:{band}"))
    found = []
    for rec in recs:
        r = rec.result
        for device, age, reason in violations(r):
            assert reason == CAP_CROWDED and device in star.servable, (r.mission_round, device)
            found.append((r.mission_round, device))
        assert max(r.plan["cap"]["ages"].values()) <= s + 1
    assert found == DRIFT[(family, k)]


# --------------------------------------------------------------------------- #
# The close and the drops
# --------------------------------------------------------------------------- #

#: Three single-device wide stops on a line from the dock.
LINE3 = (("a", (70.0, 0.0, 0.0)), ("b", (140.0, 0.0, 0.0)), ("c", (210.0, 0.0, 0.0)))


def _tight_line(**kw):
    """LINE3 on FB+wide with every device capped (S = 1) and a budget 0.05 s
    above the plan's own predicted Pass-1 home: a silent member's listen
    window (1 s, against 0.8 s of priced dwell) leaves the rest unaffordable."""
    spec = det_spec(7, LINE3, **kw)
    return spec, priced_home(spec, LINE3, "abc") + 0.05


def test_the_close_records_the_stops_flown_and_the_merge():
    """Other choices 6, U1's close rules as the mule feeds them: b stays
    silent, so at its departure the last stop no longer fits and the re-plan
    trims it. The visited set is the members of the stops actually flown (a,
    b), so c, planned and capped, is ``dropped_in_flight``; b was visited and
    never merged, so it is ``not_merged``."""
    spec, budget = _tight_line()
    _, sup, (rec,) = fly(spec=spec, layout=LINE3, silent=("b",),
                         **plan_kw(budget=budget, s=1, policy="fixed:wide"))
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["a"], ["b"]]
    (replan,) = r.replans
    assert replan["before"] == [["c"]] and replan["dropped"] == [{"devices": ["c"],
                                                                 "reason": "budget"}]
    assert r.plan["served"] == ["a", "b", "c"] and r.plan["visited"] == ["a", "b"]
    assert violations(r) == [("c", 1, CAP_DROPPED_IN_FLIGHT), ("b", 1, CAP_NOT_MERGED)]
    assert [str(line.device_id) for line in r.report.lines if line.outcome.is_on_time()] == ["a"]
    assert sup.scheduler.last_plan.closed and r.plan == sup.scheduler.last_plan.describe()


def test_an_empty_mission_closes_its_plan_with_nothing_merged():
    """The empty path (Pass 1 collected no update) closes the plan with
    ``merged = ()``: every capped device visited is ``not_merged``, and the one
    the re-plan trimmed is ``dropped_in_flight``."""
    spec, budget = _tight_line()
    _, sup, (rec,) = fly(spec=spec, layout=LINE3, silent=("a", "b", "c"),
                         **plan_kw(budget=budget, s=1, policy="fixed:wide"))
    r = rec.result
    assert r.empty and r.report is None
    assert r.plan["visited"] == ["a", "b"]
    assert violations(r) == [("c", 1, CAP_DROPPED_IN_FLIGHT), ("a", 1, CAP_NOT_MERGED),
                             ("b", 1, CAP_NOT_MERGED)]
    assert sup.scheduler.last_plan.closed


def test_under_the_age_cutoff_the_close_reads_the_merge_not_the_report():
    """Other choices 6 under the pilots' aggregation (decision 7's
    ``agg:cutoff``): an update can be CLEAN in the round report and still be
    left out of the merge, past its age cutoff (the merge's
    ``excluded_devices``), and ``not_merged`` means the merge did not use it.
    b misses mission 1's delivery (silent in its Pass 2), then trains offline
    on the basis it holds (principle 14), so in mission 2 it answers with an
    update one version stale, which the cutoff (a_max = 0) excludes. The close
    is given exactly what record_merged is given, so b, visited and capped
    (S = 1), is ``not_merged``, and its merge anchor stays at mission 1."""
    b = DeviceID("b")
    layout = (("a", (40.0, 0.0, 0.0)), ("b", (90.0, 30.0, 0.0)), ("c", (140.0, 0.0, 0.0)))
    given = []

    def before(w, sup, m):
        rf, sch = w.rfs[w.mule_ids[0]], sup.scheduler
        if m == 0:
            real_open, real_record, real_close = (sup.mission.open_pass_2, sch.record_merged,
                                                  sch.close_plan)

            def open_pass_2(*a, **kw):
                rf.silent.add(b)
                return real_open(*a, **kw)

            def record(merged, mission_round):
                given.append(("record", sorted(map(str, merged))))
                return real_record(merged, mission_round)

            def close(flown, merged):
                given.append(("close", sorted(map(str, merged))))
                return real_close(flown, merged)

            sup.mission.open_pass_2 = open_pass_2
            sch.record_merged, sch.close_plan = record, close
        else:
            rf.silent.discard(b)
            w.devices[b].train_offline()

    _, sup, recs = fly(spec=det_spec(7, layout), layout=layout, missions=2, before=before,
                       world_kw={"aggregation": AggregationSpec(rule="agg:cutoff", a_max=0)},
                       **plan_kw(budget=1000.0, s=1, policy="fixed:wide"))
    first, second = (rec.result for rec in recs)
    assert violations(first) == []
    assert sorted(str(line.device_id) for line in second.report.lines
                  if line.outcome.is_on_time()) == ["a", "b", "c"]
    assert [str(d) for d in second.aggregate.excluded_devices] == ["b"]
    assert given == [("record", ["a", "b", "c"]), ("close", ["a", "b", "c"]),
                     ("record", ["a", "c"]), ("close", ["a", "c"])]
    assert second.plan["visited"] == ["a", "b", "c"] and second.plan["cap"]["ages"]["b"] == 1
    assert violations(second) == [("b", 1, CAP_NOT_MERGED)]
    assert {d: sup.scheduler.device_states[DeviceID(d)].last_merged_round for d in "abc"} == {
        "a": 2, "b": 1, "c": 2}


def test_the_plans_own_choices_are_widened_recorded_and_left_out_of_s3c():
    """Other choices 8 (critic C6): on layout 25 mission 1 the plan serves
    three devices and leaves d3, d4 and d5 out by choice (``plan``: each fits
    alone). They are widened like any drop, recorded as the plan labelled them,
    barred from the beacon hook, and left out of S3c's planned count."""
    seed, layout = ref_layout(25)
    spec = det_spec(seed, layout)
    seen = {}

    def before(w, sup, m):
        sch = sup.scheduler
        real_plan, real_outcome = sch.build_ferry_plan, sch.record_mission_outcome

        def build(**kw):
            route = real_plan(**kw)
            seen["feas"] = sup.scheduler.last_feasibility
            return route

        def outcome(**kw):
            seen["s3c"] = kw
            return real_outcome(**kw)

        sch.build_ferry_plan, sch.record_mission_outcome = build, outcome
        sup.offer_contact(ContactWaypoint(position=DOCK, devices=(DeviceID("d3"),),
                                          bucket=Bucket.BEACON_ACTIVE, deadline_ts=0.0))

    _, _, (rec,) = fly(spec=spec, layout=layout, before=before, **plan_kw(budget=45.0, s=3))
    r = rec.result
    feas = seen["feas"]
    assert [sorted(map(str, wp.devices)) for wp in feas.dropped_plan] == [["d3", "d4", "d5"]]
    assert not (feas.dropped_overdue or feas.dropped_budget or feas.dropped_energy)
    assert r.pass_1_preflight_drops == [
        {"position": [float(c) for c in wp.position], "devices": [str(d) for d in wp.devices],
         "deadline_ts": float(wp.deadline_ts), "reason": "plan"} for wp in feas.dropped_plan]
    assert widened(rec) == ["d3", "d4", "d5"]
    assert [o["reason"] for o in r.offers_refused] == ["already planned this mission"]
    assert seen["s3c"] == {"served": 3, "planned": 3}


def test_the_plans_own_choices_follow_the_clause_drops_in_the_record():
    """Other choices 8, the record's order: ``pass_1_preflight_drops`` lists the
    four clause reasons first, in the order a legacy mule records them, and
    the plan's own choices (``plan``) after them. FB+wide, every device capped
    (S = 1), a budget between one device's home and two's: u and v each fit
    alone, not together, and z, 400 m out, fits nowhere. The plan serves v; z
    is left out for the budget (``unplannable``) and u by choice
    (``crowded``). Both are widened."""
    layout = (("u", (60.0, 0.0, 0.0)), ("v", (-60.0, 0.0, 0.0)), ("z", (400.0, 0.0, 0.0)))
    spec = det_spec(7, layout)
    one, two = priced_home(spec, layout, "u"), priced_home(spec, layout, "uv")
    _, _, (rec,) = fly(spec=spec, layout=layout,
                       **plan_kw(budget=(one + two) / 2, s=1, policy="fixed:wide"))
    r = rec.result
    assert r.plan["served"] == ["v"] and [s["devices"] for s in r.pass_1_flown] == [["v"]]
    assert [(d["devices"], d["reason"]) for d in r.pass_1_preflight_drops] == [
        (["z"], "budget"), (["u"], "plan")]
    assert widened(rec) == ["u", "z"]
    assert violations(r) == [("z", 1, CAP_UNPLANNABLE), ("u", 1, CAP_CROWDED)]


def test_an_h_arms_member_subset_complement_is_widened_like_a_budget_drop():
    """unit_U3b.md section 7.2 (U7): the Phase 3 cliff (trial T2, narrow, 1 MB,
    60 s) under ``subset`` flies H1's reduced stop {0, 1, 2, 4, 6}; the rest,
    {3, 5, 7}, is recorded as a budget drop and widened: each misses once.
    Under ``whole`` the recorded cliff stands (nothing flies)."""
    for admission, flown, drops in (
            ("subset", [["dev-0", "dev-1", "dev-2", "dev-4", "dev-6"]],
             [(["dev-3", "dev-5", "dev-7"], "budget")]),
            ("whole", [], [([f"dev-{i}" for i in range(8)], "budget")])):
        _, sup, (rec,) = _cliff(member_admission=admission)
        r = rec.result
        assert [s["devices"] for s in r.pass_1_flown] == flown
        assert [(d["devices"], d["reason"]) for d in r.pass_1_preflight_drops] == drops
        assert widened(rec) == drops[0][0]
        assert all(sup.scheduler.device_states[DeviceID(d)].miss_streak == 1 for d in drops[0][0])
        assert r.pass_1_policy_drops is None and r.plan is None


T2_LAYOUT = tuple((f"dev-{i}", (x, y, 0.0))
                  for i, (x, y) in enumerate(device_positions(8, 777, 100.0)))


def _cliff(*, budget=60.0, energy_capacity_j=None, **kw):
    """Trial T2 of the Phase 3 final check (tests/unit/test_p3_final_fixes_mule.py):
    narrow, 1 MB, the seconds backhaul, 60 s, T2's deadline unit; the budget
    and a simulated energy capacity can be set."""
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=777, contact_band="narrow",
                                 backhaul_model="seconds", backhaul_period=750.0,
                                 backhaul_regime="clean", payload_bytes=1_000_000,
                                 energy_capacity_j=energy_capacity_j)
    return fly(spec=spec, layout=T2_LAYOUT, mission_budget_s=budget, deadline_time_scale=25.0,
               **kw)


def test_under_whole_a_plan_arm_flies_whole_stops():
    """R5: ``member_admission="whole"`` flies whole stops in every mode, so a
    ``whole`` plan arm keeps the narrow-band cliff for comparison (design D-D).
    On trial T2 at 60 s, FB+narrow under ``whole`` flies nothing: its one
    field-wide stop, judged whole, is left out as ``budget`` and widened, and
    the plan closes on the empty path. Under ``subset`` it flies the members
    that fit (U3's F order) and leaves the rest out by choice (``plan``)."""
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=777, contact_band="narrow",
                                 backhaul_model="seconds", backhaul_period=750.0,
                                 backhaul_regime="clean", payload_bytes=1_000_000,
                                 replan_fallback="trim")
    every = [f"dev-{i}" for i in range(8)]
    for admission, flown, drops in (
            ("whole", [], [(every, "budget")]),
            ("subset", [["dev-0", "dev-1", "dev-2", "dev-4", "dev-6"]],
             [(["dev-3", "dev-5", "dev-7"], "plan")])):
        _, sup, (rec,) = fly(spec=spec, layout=T2_LAYOUT, **plan_kw(
            budget=60.0, unit=25.0, admission=admission, policy="fixed:narrow"))
        r = rec.result
        assert sup.scheduler.member_admission == admission
        assert [s["devices"] for s in r.pass_1_flown] == flown and r.empty is (not flown)
        assert [(d["devices"], d["reason"]) for d in r.pass_1_preflight_drops] == drops
        assert widened(rec) == drops[0][0]
        assert r.plan["visited"] == sorted(d for s in flown for d in s)
        assert r.band == r.plan["band"] == "narrow"


def _policy_report(seen):
    """A ``before`` hook recording the scheduler's own drop report
    (``last_policy_drops``: (contact, reason) pairs) right after each plan."""
    def before(w, sup, m):
        sch = sup.scheduler
        real = sch.build_contact_queue

        def build(**kw):
            queue = real(**kw)
            seen.append(list(sch.last_policy_drops))
            return queue

        sch.build_contact_queue = build

    return before


def _reported(wp, reason):
    """The ``pass_1_policy_drops`` entry of the scheduler's ``(wp, reason)``."""
    return {"position": [float(c) for c in wp.position], "devices": [str(d) for d in wp.devices],
            "deadline_ts": float(wp.deadline_ts), "reason": reason, "widened": False}


def test_a_d_arm_reports_what_it_left_out_and_never_widens_it():
    """The user's decision 6: on the cliff D1 (MaxAoI) leaves the field-wide
    contact out whole under ``whole`` and its complement {5, 6, 7} under
    ``subset``. Each is reported once, entry for entry as the scheduler
    reported it (its position, deadline and label, and ``"widened":
    false``), and none of those devices is widened; the pre-flight record
    (our arms' drops) stays empty. The label is the scheduler's: at 200 s the
    field-wide contact fits the budget, and an energy capacity between its
    Pass-2 and Pass-1 needs (12 kJ, as in U5's labelling test) leaves it out
    as ``energy``; without that capacity it flies and nothing is reported. D4
    flies everything, so it reports nothing and the field stays absent
    (critic D2)."""
    every = [f"dev-{i}" for i in range(8)]
    for kw, flown, left, label in (
            (dict(member_admission="whole"), [], every, "budget"),
            (dict(member_admission="subset"), [every[:5]], every[5:], "budget"),
            (dict(budget=200.0, energy_capacity_j=12000.0), [], every, "energy")):
        seen = []
        _, sup, (rec,) = _cliff(target_selector=MaxAoIPolicy(), before=_policy_report(seen), **kw)
        r = rec.result
        assert [s["devices"] for s in r.pass_1_flown] == flown
        ((wp, reason),) = seen[0]
        assert sorted(map(str, wp.devices)) == left and reason == label
        assert r.pass_1_policy_drops == [_reported(wp, reason)]
        assert r.pass_1_preflight_drops == [] and widened(rec) == []
        assert all(sup.scheduler.device_states[DeviceID(d)].miss_streak == 0 for d in left)
    _, _, (rec,) = _cliff(target_selector=MaxAoIPolicy(), budget=200.0)
    assert rec.result.pass_1_flown[0]["devices"] == every
    assert rec.result.pass_1_policy_drops is None
    _, _, (rec,) = _cliff(target_selector=FedExCarpPolicy(depot=DOCK))
    assert len(rec.result.pass_1_flown[0]["devices"]) == 8
    assert rec.result.pass_1_policy_drops is None


# --------------------------------------------------------------------------- #
# In flight
# --------------------------------------------------------------------------- #

def _calls(sup, log):
    """Record every fold and re-plan the supervisor asks the scheduler for."""
    sch = sup.scheduler
    real_fold, real_replan = sch.fold_remainder, sch.replan_remainder

    def fold(remainder, **kw):
        log.append(("fold", list(remainder), dict(kw)))
        return real_fold(remainder, **kw)

    def replan(remainder, **kw):
        log.append(("replan", list(remainder), dict(kw)))
        return real_replan(remainder, **kw)

    sch.fold_remainder, sch.replan_remainder = fold, replan


def test_in_flight_the_exempt_stops_are_protected_and_the_re_plan_dates_each_member():
    """Other choices 9: in plan mode the departure check, the beacon hook's
    fold and the re-plan hold the stops whose every member the plan capped
    exempt from their own deadline (here, with S = 1, every stop), each set
    taken on the stops folded; Pass 2 holds none. The re-plan gets each
    member's own deadline from the mule's record: the plan's, and the beacon
    insert's own for x, which the plan never dated. x (between a and b, off
    the plan's demand) is inserted at takeoff; b is silent, so the re-plan
    trims c at b's departure."""
    spec, budget = _tight_line()
    budget += priced_home(spec, LINE3 + (("x", (105.0, 0.0, 0.0)),), "axbc") - priced_home(
        spec, LINE3, "abc")
    log = []

    def before(w, sup, m):
        sup.scheduler.device_states[DeviceID("x")] = DeviceSchedulerState(
            device_id=DeviceID("x"), last_known_position=(105.0, 0.0, 0.0))
        sup.offer_contact(ContactWaypoint(position=DOCK, devices=(DeviceID("x"),),
                                          bucket=Bucket.BEACON_ACTIVE, deadline_ts=0.0))
        _calls(sup, log)

    _, sup, (rec,) = fly(spec=spec, layout=LINE3, silent=("b",), before=before,
                         world_kw={"extra_devices": {"x": (105.0, 0.0, 0.0)}},
                         **plan_kw(budget=budget, s=1, policy="fixed:wide"))
    r = rec.result
    (insert,) = r.inserts
    assert insert["devices"] == ["x"] and insert["index"] == 1
    assert [s["devices"] for s in r.pass_1_flown] == [["a"], ["x"], ["b"]]
    assert r.plan["visited"] == ["a", "b", "x"]
    assert violations(r) == [("c", 1, CAP_DROPPED_IN_FLIGHT), ("b", 1, CAP_NOT_MERGED)]
    capped = set(r.plan["cap"]["capped"])
    pass_1 = [(what, stops, kw) for what, stops, kw in log if kw["pass_kind"] is COLLECT]
    assert len(pass_1) > 4
    for what, stops, kw in pass_1:
        exempt = frozenset(wp for wp in stops if {str(d) for d in wp.devices} <= capped)
        assert kw["protected"] == exempt
    assert any(len(kw["protected"]) >= 3 for _, _, kw in pass_1)          # the beacon's folds
    (replan,) = [kw for what, _, kw in pass_1 if what == "replan"]
    deadlines = {str(d): t for d, t in replan["deadlines"].items()}
    assert set(deadlines) == {"a", "b", "c", "x"}
    assert {d: deadlines[d] for d in "abc"} == {str(d): t for d, t in
                                                r.pass_1_device_deadlines.items()}
    assert math.isfinite(deadlines["x"])


def test_a_legacy_mule_passes_no_protected_set_and_no_deadlines():
    """Phase 3 gave none of these calls a protected set; a legacy mule on the
    clock still passes none, and no deadlines to the re-plan, through the same
    silent-member trim."""
    spec, budget = _tight_line()
    log = []
    _, _, (rec,) = fly(spec=spec, layout=LINE3, silent=("b",),
                       before=lambda w, sup, m: _calls(sup, log),
                       mission_budget_s=budget, deadline_time_scale=FAR_UNIT)
    assert rec.result.replans and any(what == "replan" for what, _, _ in log)
    assert all("protected" not in kw and "deadlines" not in kw for _, _, kw in log)


def test_without_a_cap_plan_mode_still_passes_the_exempt_set_and_the_deadlines():
    """The same trim in plan mode with the cap off (arm F-cap): every Pass-1
    fold and the re-plan get the plan's exempt set, empty here, and the
    re-plan still gets the mule's deadlines. They date what the plan never
    dated (the beacon hook's inserts, U5's hand-off), which an empty exempt
    set says nothing about."""
    spec, budget = _tight_line()
    log = []
    _, _, (rec,) = fly(spec=spec, layout=LINE3, silent=("b",),
                       before=lambda w, sup, m: _calls(sup, log),
                       **plan_kw(budget=budget, policy="fixed:wide"))
    r = rec.result
    assert r.plan["cap"]["s"] is None and r.plan["served"] == ["a", "b", "c"]
    assert [s["devices"] for s in r.pass_1_flown] == [["a"], ["b"]]
    pass_1 = [(what, kw) for what, _, kw in log if kw["pass_kind"] is COLLECT]
    assert [what for what, _ in pass_1] == ["fold", "fold", "fold", "replan"]
    assert all(kw["protected"] == frozenset() for _, kw in pass_1)
    replan = pass_1[-1][1]
    assert {str(d): t for d, t in replan["deadlines"].items()} == {
        str(d): t for d, t in r.pass_1_device_deadlines.items()}


def test_the_departure_check_then_the_beacon_hook_then_the_slot():
    """R6 and U6's hand-off: in Pass 1 each departure runs the check, then the
    beacon hook, then the flight slot's pick, so the slot reads the remainder
    both have settled; the first pick of each pass is at takeoff
    (``after_stop`` False), and Pass 2, unchecked in plan mode, has only the
    slot."""
    seed, layout = ref_layout(9)
    spec = det_spec(seed, layout)
    order = []

    def before(w, sup, m):
        real_dep, real_offers, slot = sup._ferry_departure, sup._take_offers, sup._flight_slot
        real_next = slot.next_stop

        def dep(*a, **kw):
            order.append(("check", kw["pass_kind"]))
            return real_dep(*a, **kw)

        def offers(*a, **kw):
            order.append(("beacon", COLLECT))
            return real_offers(*a, **kw)

        def pick(remainder, state, **kw):
            order.append(("slot", kw["pass_kind"], kw["after_stop"]))
            return real_next(remainder, state, **kw)

        sup._ferry_departure, sup._take_offers, slot.next_stop = dep, offers, pick

    _, _, (rec,) = fly(spec=spec, layout=layout, before=before,
                       **plan_kw(budget=90.0, s=2, slot="cross_heuristic"))
    r = rec.result
    n1, n2 = len(r.pass_1_flown), len(r.pass_2_flown)
    assert n1 >= 2 and n2 >= 2
    expected = []
    for i in range(n1):
        expected += [("check", COLLECT), ("beacon", COLLECT), ("slot", COLLECT, i > 0)]
    expected += [("beacon", COLLECT)]                      # the last departure: home
    expected += [("slot", DELIVER, i > 0) for i in range(n2)]
    assert order == expected


def test_the_beacon_hook_groups_an_offer_within_the_committed_classs_range():
    """Design section 2.6 and U6's hand-off: after ``set_band`` the beacon hook
    groups an offer's members into one stop within R_planar(b̄), the class the
    stop will be flown on, not the run's reference class. On layout 25 at
    120 s F commits narrow on a wide-reference spec. x and y lie 80 m apart,
    beyond wide's 60 m and within narrow's 232 m: their offer is inserted as
    one stop, flown on narrow and reaching both. z and w, 268 m apart, are
    beyond narrow's reach too and are refused."""
    seed, layout = ref_layout(25)
    spec = det_spec(seed, layout)
    extra = {"x": (-40.0, 0.0, 0.0), "y": (-40.0, 80.0, 0.0),
             "z": (-60.0, -60.0, 0.0), "w": (180.0, 60.0, 0.0)}

    def before(w, sup, m):
        for d, pos in extra.items():
            sup.scheduler.device_states[DeviceID(d)] = DeviceSchedulerState(
                device_id=DeviceID(d), last_known_position=pos)
        for pair in ("xy", "zw"):
            sup.offer_contact(ContactWaypoint(position=DOCK, devices=tuple(map(DeviceID, pair)),
                                              bucket=Bucket.BEACON_ACTIVE, deadline_ts=0.0))

    _, _, (rec,) = fly(spec=spec, layout=layout, before=before,
                       world_kw={"extra_devices": extra}, **plan_kw(budget=120.0))
    r = rec.result
    wide, narrow = spec.link.range_planar_m("wide"), spec.link.range_planar_m("narrow")
    assert wide < math.dist(extra["x"], extra["y"]) <= narrow < math.dist(extra["z"], extra["w"])
    assert spec.band == "wide" and r.plan["band"] == r.band == "narrow"
    (insert,) = r.inserts
    assert insert["devices"] == ["x", "y"] and insert["position"] == list(extra["x"])
    assert [(o["devices"], o["reason"]) for o in r.offers_refused] == [
        (["z", "w"], "members not within range of one stop")]
    (stop,) = [s for s in r.pass_1_flown if s["devices"] == ["x", "y"]]
    assert (stop["band"], stop["targets"], stop["unreachable"]) == ("narrow", ["x", "y"], [])
    assert {"x", "y"} <= {str(line.device_id) for line in r.report.lines
                          if line.outcome.is_on_time()}
    assert r.replans == [] and r.budget_overrun_s == 0.0


def test_capped_members_do_not_lower_deliver_by():
    """Other choices 9 under ``deadline_bounds="delivery"``: an update the cap
    made the plan fetch does not hold the rest of the route to its deadline,
    as the predicate never lowers ``deliver_by`` for a stop exempt from its own
    (critic A11; s3b's ``admit``). With every member capped it stays inf at
    every departure, and no re-plan refuses the rest as ``delivery`` even
    though each deadline (unit 1.0: t0 + 60 s) falls before the landing
    (87 s). With the cap off (arm F-cap, far deadlines) each collected
    member's deadline lowers it, as in Phase 3."""
    held = {}
    for s, unit in ((1, 1.0), (None, FAR_UNIT)):
        log = []
        spec = det_spec(7, LINE3, deadline_bounds="delivery")
        _, _, (rec,) = fly(spec=spec, layout=LINE3, before=lambda w, sup, m: _calls(sup, log),
                           **plan_kw(budget=1000.0, s=s, unit=unit, policy="fixed:wide"))
        r = rec.result
        assert [stop["devices"] for stop in r.pass_1_flown] == [["a"], ["b"], ["c"]]
        assert r.replans == [] and r.delivery_overrun_s == 0.0 and violations(r) == []
        # One departure check per stop: at takeoff, after a, after b.
        held[s] = [kw["state"].deliver_by for what, _, kw in log
                   if what == "fold" and kw["pass_kind"] is COLLECT]
        deadlines = r.pass_1_device_deadlines
    assert held[1] == [math.inf] * 3
    a, b = deadlines[DeviceID("a")], deadlines[DeviceID("b")]
    assert held[None] == [math.inf, a, min(a, b)] and a < math.inf


def test_a_capped_member_of_a_mixed_stop_does_not_lower_deliver_by():
    """Other choices 9, member by member: the rule is per capped member, not
    per exempt stop. At mission 2 (S = 2) p is capped (silent in mission 1, so
    never merged) and shares a stop with q, which is not; p's deadline falls
    1 s after takeoff, q's and r's far away. The predicate dates the mixed
    stop by q alone (B2), and the mule holds the rest of the route to q's
    deadline, not p's, although both updates are on board: r, flown well
    after p's deadline has passed, is not refused as ``delivery``. (The
    deadline is set through the state's override, the formula's first input;
    the mission clock refuses only the cluster's wall-clock overrides.)"""
    layout = (("p", (40.0, 0.0, 0.0)), ("q", (48.0, 6.0, 0.0)), ("r", (150.0, 0.0, 0.0)))
    p = DeviceID("p")
    log = []

    def before(w, sup, m):
        rf = w.rfs[w.mule_ids[0]]
        if m == 0:
            rf.silent.add(p)
            return
        rf.silent.discard(p)
        sup.scheduler.device_states[p].deadline_override_ts = sup._now() + 1.0
        _calls(sup, log)

    spec = det_spec(7, layout, deadline_bounds="delivery")
    _, _, recs = fly(spec=spec, layout=layout, missions=2, before=before,
                     **plan_kw(budget=1000.0, s=2, policy="fixed:wide"))
    r = recs[1].result
    assert r.plan["cap"]["capped"] == ["p"] and r.plan["cap"]["ages"] == {"p": 2, "q": 1, "r": 1}
    deadlines = {str(d): t for d, t in r.pass_1_device_deadlines.items()}
    assert deadlines["p"] == pytest.approx(r.sim_start_s + 1.0) and deadlines["p"] < deadlines["q"]
    assert [([str(d) for d in wp.devices], wp.deadline_ts) for wp in r.pass_1_queue] == [
        (["p", "q"], deadlines["q"]), (["r"], deadlines["r"])]
    # One departure check per stop, at takeoff and after {p, q}.
    held = [kw["state"].deliver_by for what, _, kw in log
            if what == "fold" and kw["pass_kind"] is COLLECT]
    assert held == [math.inf, deadlines["q"]]
    assert [s["devices"] for s in r.pass_1_flown] == [["p", "q"], ["r"]]
    assert r.pass_1_flown[0]["end_s"] > deadlines["p"]
    assert sorted(str(line.device_id) for line in r.report.lines
                  if line.outcome.is_on_time()) == ["p", "q", "r"]
    assert r.replans == [] and r.delivery_overrun_s == 0.0 and violations(r) == []


# --------------------------------------------------------------------------- #
# FX
# --------------------------------------------------------------------------- #

#: Three single-device stops: the plan's tour is c, b, a (tied with its
#: reverse; the key breaks the tie on the first stop's position); after c the
#: nearest is a. 10 MB each way, so that wide with a stop per device beats one
#: narrow stop.
TRI = (("c", (40.0, 60.0, 0.0)), ("a", (100.0, 0.0, 0.0)), ("b", (150.0, 80.0, 0.0)))


def _tri(budget, slot, *, before=None):
    spec = det_spec(7, TRI, payload_bytes=10_000_000)
    return fly(spec=spec, layout=TRI, before=before,
               **plan_kw(budget=budget, s=1, unit=1.0, slot=slot))


def test_fx_reorders_after_a_stop_to_the_nearest_stop_that_still_fits():
    """Decision 5's next-stop half. F flies the plan, c, b, a. After c, FX flies
    a, the nearest, when the whole reordered rest still fits the budget (150
    s), and keeps b when it does not (108 s: a-first lands at 112.2 s, the plan
    at 103.6 s). Every device is capped (S = 1) and every deadline falls at t0 +
    60 s, so a-first fits only with the exempt stops protected, in ``fits`` as
    in the departure check (neither re-plans). The first stop is the plan's:
    nothing is chosen at takeoff."""
    flown = {}
    for budget, slot in ((150.0, "committed"), (150.0, "cross_heuristic"),
                         (108.0, "cross_heuristic")):
        _, _, (rec,) = _tri(budget, slot)
        r = rec.result
        assert r.plan["served"] == ["a", "b", "c"] and violations(r) == []
        assert r.replans == [] and r.budget_overrun_s == 0.0
        flown[budget, slot] = [s["devices"][0] for s in r.pass_1_flown]
    assert flown == {(150.0, "committed"): ["c", "b", "a"],
                     (150.0, "cross_heuristic"): ["c", "a", "b"],
                     (108.0, "cross_heuristic"): ["c", "b", "a"]}


def test_fx_fits_is_the_departure_checks_fold_of_the_reordered_rest():
    """The wiring of ``fits``: the slot gets the departure's state, and ``fits``
    of any reordered remainder is the scheduler's fold of it from that state,
    to this pass's budget end, with its exempt stops protected."""
    calls = []

    def before(w, sup, m):
        slot = sup._flight_slot
        real = slot.next_stop

        def pick(remainder, state, *, fits, pass_kind, after_stop):
            if pass_kind is COLLECT and after_stop and len(remainder) > 1:
                capped = set(sup.scheduler.last_plan.capped)
                end = sup.scheduler.mission_start_ts + sup.scheduler.mission_budget_s
                for i in range(len(remainder)):
                    order = moved_to_front(remainder, i)
                    exempt = frozenset(wp for wp in order if set(wp.devices) <= capped)
                    fold = sup.scheduler.fold_remainder(order, state=state, budget_end=end,
                                                        pass_kind=COLLECT, protected=exempt)
                    calls.append((fits(order), fold.ok))
            return real(remainder, state, fits=fits, pass_kind=pass_kind, after_stop=after_stop)

        slot.next_stop = pick

    _tri(108.0, "cross_heuristic", before=before)
    assert calls == [(True, True), (False, False)]      # after c: [b, a] fits, [a, b] no longer
    calls.clear()
    _tri(150.0, "cross_heuristic", before=before)
    assert calls == [(True, True), (True, True)]


def test_fx_fits_holds_the_rest_to_the_updates_on_board():
    """``fits`` folds from the departure's own state, ``deliver_by`` included
    (``deadline_bounds="delivery"``), so FX never picks an order the next
    departure check would refuse. TRI with the cap off at unit 2.0 (every
    window 120 s) and c's own window 108 s: once c's update is on board, the
    rest must be home by c's deadline. After c, a-first would be home at
    112.2 s, within a's and b's deadlines and the 150 s budget but past c's,
    so ``fits`` refuses it, as the departure check's fold does, and FX flies
    the plan's b, a. From the same state with nothing on board the fold would
    take a-first."""
    calls = []

    def before(w, sup, m):
        sup.scheduler.device_states[DeviceID("c")].deadline_fulfilment_s = 108.0
        slot = sup._flight_slot
        real = slot.next_stop

        def pick(remainder, state, *, fits, pass_kind, after_stop):
            if pass_kind is COLLECT and after_stop and len(remainder) > 1:
                end = sup.scheduler.mission_start_ts + sup.scheduler.mission_budget_s
                for i in range(len(remainder)):
                    order = moved_to_front(remainder, i)
                    kw = dict(budget_end=end, pass_kind=COLLECT, protected=frozenset())
                    fold = sup.scheduler.fold_remainder(order, state=state, **kw)
                    unloaded = sup.scheduler.fold_remainder(
                        order, state=dataclasses.replace(state, deliver_by=math.inf), **kw)
                    calls.append((str(order[0].devices[0]), fits(order), fold.ok, unloaded.ok))
            return real(remainder, state, fits=fits, pass_kind=pass_kind, after_stop=after_stop)

        slot.next_stop = pick

    spec = det_spec(7, TRI, payload_bytes=10_000_000, deadline_bounds="delivery")
    _, _, (rec,) = fly(spec=spec, layout=TRI, before=before,
                       **plan_kw(budget=150.0, unit=2.0, slot="cross_heuristic"))
    r = rec.result
    assert r.plan["cap"]["s"] is None and r.plan["served"] == ["a", "b", "c"]
    assert {str(d): t - r.sim_start_s for d, t in r.pass_1_device_deadlines.items()} == (
        pytest.approx({"c": 108.0, "a": 120.0, "b": 120.0}))
    assert calls == [("b", True, True, True), ("a", False, False, True)]
    assert [s["devices"] for s in r.pass_1_flown] == [["c"], ["b"], ["a"]]
    assert r.replans == [] and r.delivery_overrun_s == 0.0 and r.budget_overrun_s == 0.0


def test_fx_flies_the_fastest_class_covering_the_committed_targets_on_arrival():
    """Decision 5's band half on the pilots' noisy channel: at every Pass-1
    arrival FX flies the fastest class whose targets there include every
    target of b̄, from the arrival view at that instant (critic A7), so at the
    arrival SNR it never dwells longer than b̄ and never reaches fewer devices;
    Pass 2 flies b̄ in the queue's order; F flies b̄ at every stop."""
    switched = stops = 0
    for k in range(6):
        seed, layout = ref_layout(k)
        where = {DeviceID(d): p for d, p in layout}
        for slot in ("cross_heuristic", "committed"):
            spec = noisy_spec(seed)
            _, sup, recs = fly(spec=spec, layout=layout, missions=2,
                               **plan_kw(budget=60.0, s=3, slot=slot))
            rt = sup._ferry_run
            for rec in recs:
                r = rec.result
                committed = r.plan["band"]
                rt.set_band(committed)
                assert {s["band"] for s in r.pass_2_flown} == {committed}
                if r.pass_1_flown:
                    assert r.pass_1_flown[0]["devices"] == [str(d) for d in
                                                            r.pass_1_queue[0].devices]
                for s in r.pass_1_flown:
                    # The record's rates are the flown class's at its own SNRs.
                    assert s["rate_bps"] == {d: spec.link.rate_bps(s["band"], v)
                                             for d, v in s["snr_db"].items()}
                    wp = ContactWaypoint(position=tuple(s["position"]),
                                         devices=tuple(DeviceID(d) for d in s["devices"]),
                                         bucket=Bucket.SCHEDULED_THIS_ROUND,
                                         deadline_ts=s["deadline_ts"])
                    view = rt.arrival_view(wp, where, s["arrival_s"], pass_kind=COLLECT)
                    if slot == "committed":
                        assert s["band"] == committed
                        continue
                    pick, base = fastest_covering_class(view), view.committed_entry
                    assert s["band"] == pick.name
                    assert pick.dwell_s <= base.dwell_s
                    assert set(base.targets) <= set(pick.targets) == set(s["targets"])
                    stops += 1
                    switched += pick.name != committed
    assert stops >= 10 and switched >= 3, (stops, switched)


def test_fx_at_the_last_stop_never_lands_later_than_f():
    """Critic A7: nothing re-checks after the last Pass-1 stop, so the class FX
    picks there must not add dwell. In the deterministic world a contact takes
    what the arrival view priced, and F and FX fly the same plan from the same
    state in mission 1, here in the same order (a plan of two stops at most):
    stop by stop FX never dwells longer than F and reaches every member F
    reaches, so it lands no later, within the budget, and on each layout
    whose last stop FX flies on another class, strictly earlier."""
    earlier = 0
    for k in range(10):
        seed, layout = ref_layout(k)
        spec = det_spec(seed, layout)
        runs = {}
        for slot in ("committed", "cross_heuristic"):
            _, _, (rec,) = fly(spec=spec, layout=layout, **plan_kw(budget=90.0, slot=slot))
            r = rec.result
            assert r.budget_overrun_s == 0.0 and r.replans == []
            runs[slot] = r
        f, fx = runs["committed"], runs["cross_heuristic"]
        assert f.plan == fx.plan
        assert [s["devices"] for s in f.pass_1_flown] == [s["devices"] for s in fx.pass_1_flown]
        for sf, sx in zip(f.pass_1_flown, fx.pass_1_flown):
            assert sx["dwell_s"] <= sf["dwell_s"] + 1e-9
            assert set(sf["targets"]) <= set(sx["targets"])
        last_f, last_x = f.pass_1_flown[-1], fx.pass_1_flown[-1]
        assert last_x["end_s"] <= last_f["end_s"] + 1e-9
        if last_x["band"] != last_f["band"]:
            assert last_x["end_s"] < last_f["end_s"]
            earlier += 1
    assert earlier >= 3


# --------------------------------------------------------------------------- #
# K = 2 (critic B10)
# --------------------------------------------------------------------------- #

def test_two_mules_each_plan_their_own_slice_from_their_own_ages():
    """At K > 1 plan mode runs per slice (the spec's kept choices; critic B10):
    two plan-mode mules share a quorum-2 cluster (mule B flies inside mule A's
    DOWN wait), and each plans only its own slice, ages its devices by its own
    missions and its own merges, and syncs to the cluster's simulated time at
    the dock. The mule credits a merge when its own merge uses the update;
    the cluster's round closes later, when the quorum is met (UD records how
    the scorer's violations then differ)."""
    seed = 11
    spec = det_spec(seed, GH.LAYOUT)
    w = H.World(mule_ids=["mule-a", "mule-b"], assignment=GH.K2_ASSIGNMENT,
                min_participation=2, echo_sim_ts=True, flaky={})
    ma, mb = MuleID("mule-a"), MuleID("mule-b")
    results = {"mule-a": [], "mule-b": []}
    with H.Patched(w.clock):
        sa = w.supervisor(ma, sim=True, ferry=spec, down_wait_s=30.0, **plan_kw(budget=150.0, s=2))
        sb = w.supervisor(mb, sim=True, ferry=spec, down_wait_s=30.0, **plan_kw(budget=150.0, s=2))
        w.bootstrap()
        for m in range(4):
            w.server.tasks[ma].append(lambda: results["mule-b"].append(sb.run_one_mission()))
            results["mule-a"].append(sa.run_one_mission())
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    for mule, rs in results.items():
        slice_ = sorted(d for d, m in GH.K2_ASSIGNMENT.items() if m == mule)
        merged_at = {d: 0 for d in slice_}
        assert [r.mission_round for r in rs] == [1, 2, 3, 4]
        for r in rs:
            p = r.plan
            assert p["mission_round"] == r.mission_round
            assert sorted(p["demand"]) == slice_ and set(p["served"]) <= set(slice_)
            assert p["cap"]["ages"] == {d: r.mission_round - merged_at[d] for d in p["demand"]}
            assert violations(r) == []
            assert not r.empty and p["served"]
            for line in r.report.lines:
                if line.outcome.is_on_time():
                    merged_at[str(line.device_id)] = r.mission_round
    ups = {}
    for up in w.server.ups:
        ups.setdefault(str(up.mule_id), []).append(up.sim_upload_ts)
    for mule, rs in results.items():
        for r, t_up in zip(rs, ups[mule]):
            assert r.sim_pass_2_start_s >= t_up + 30.0


# --------------------------------------------------------------------------- #
# Determinism and the defaults
# --------------------------------------------------------------------------- #

def test_a_repeated_trial_gives_identical_results_bar_the_plan_wall_time():
    """Decision 7's "a repeated trial gives identical traces, bar wall stamps"
    (critic B12): FX on the pilots' noisy channel with the cap binding, four
    missions, twice. Every result field is the same bit for bit once the
    plan's wall time, the only wall time a result holds, is set aside."""
    def trial():
        seed, layout = ref_layout(25)
        _, _, recs = fly(spec=noisy_spec(seed), layout=layout, missions=4,
                         **plan_kw(budget=45.0, s=2, slot="cross_heuristic"))
        results = [rec.result for rec in recs]
        assert all(isinstance(r.plan_wall_s, float) and r.plan_wall_s >= 0.0 for r in results)
        return [_canon.canon(dataclasses.replace(r, plan_wall_s=None)) for r in results]

    one, two = trial(), trial()
    assert one == two
    assert any(v for r in one for v in r["plan"]["cap"]["violations"])


def _legacy_missions(monkeypatch):
    """H1 and D4 on the clock at the defaults, and H1 on the wall clock."""
    def refuse(self, band):
        raise AssertionError("a legacy mule never moves the runtime's band")

    monkeypatch.setattr(FerryRuntime, "set_band", refuse)
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=7, contact_band="medium",
                                 backhaul_model="seconds", backhaul_period=800.0,
                                 payload_bytes=1_000_000, in_flight_response="replan",
                                 replan_fallback="trim")
    out = []
    for kw in ({}, {"target_selector": FedExCarpPolicy(depot=DOCK)}):
        _, sup, recs = fly(spec=spec, layout=GH.LAYOUT, missions=2, mission_budget_s=70.0, **kw)
        out += [(sup, rec.result, spec.band) for rec in recs]
    w = H.World(flaky={})
    with H.Patched(w.clock):
        sup = w.supervisor(w.mule_ids[0], sim=False, mission_budget_s=70.0)
        w.bootstrap()
        out.append((sup, sup.run_one_mission(), None))
    return out


def test_at_the_defaults_no_plan_field_appears(monkeypatch):
    """Freeze Rule 1 and UG4's hand-off: the goldens let an added key pass, so
    the absence is pinned here. A legacy mule builds no flight slot, never
    moves the runtime's band, flies ``contact_band`` at every stop and leaves
    ``plan``, ``plan_wall_s`` and ``pass_1_policy_drops`` None (D4 left
    nothing out), on the clock and off it."""
    missions = _legacy_missions(monkeypatch)
    assert len(missions) == 5
    for sup, r, band in missions:
        assert (r.plan, r.plan_wall_s, r.pass_1_policy_drops) == (None, None, None)
        assert sup._flight_slot is None and sup.scheduler.plan_mode == "legacy"
        assert sup.scheduler.member_admission == "whole"
        if band is not None:
            assert r.band == band and r.pass_1_flown
            assert {s["band"] for s in r.pass_1_flown + r.pass_2_flown} == {band}


_LEGACY_PATH = r"""
import sys
from types import SimpleNamespace
from experiments.exp4.topology_builder import device_positions
from hermes.mule.ferry import FerrySpec
from hermes.scheduler.policies import FedExCarpPolicy, MaxAoIPolicy
from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

T2 = tuple((f"dev-{i}", (x, y, 0.0)) for i, (x, y) in enumerate(device_positions(8, 777, 100.0)))

def fly(layout, budget, silent=(), sim=True, **kw):
    spec = FerrySpec.from_config(rf_range_m=60.0, seed=777, contact_band="narrow",
                                 backhaul_model="seconds", backhaul_period=750.0,
                                 payload_bytes=1_000_000, in_flight_response="replan",
                                 replan_fallback="trim")
    w = H.World(layout=layout, flaky={})
    w.rfs[w.mule_ids[0]].silent.update(silent)
    with H.Patched(w.clock):
        sup = w.supervisor(w.mule_ids[0], sim=sim, ferry=spec if sim else None,
                           mission_budget_s=budget, deadline_time_scale=25.0, **kw)
        w.bootstrap()
        return [sup.run_one_mission() for _ in range(2)]

rs = (fly(T2, 99.5) + fly(T2, 60.0, target_selector=MaxAoIPolicy())
      + fly(GH.LAYOUT, 70.0, target_selector=FedExCarpPolicy(depot=(0.0, 0.0, 0.0)))
      + fly(GH.LAYOUT, 70.0, silent=("dev-01",)) + fly(GH.LAYOUT, 70.0, sim=False))
assert any(r.pass_1_policy_drops for r in rs) and any(r.pass_1_preflight_drops for r in rs)
loaded = sorted(m for m in sys.modules if m.startswith("hermes.scheduler.plan")
                or m.endswith(("s3d_age_cap", "cross_heuristic")))
assert not loaded, loaded
print("ok")
"""


def test_a_legacy_mule_never_loads_the_plan_package():
    """Freeze Rule 1: at the defaults, missions on both clocks (H1 with its
    pre-flight drops, in-flight re-plans and a silent member, D1 reporting its
    drops, D4) load neither the plan package, nor the age-cap stage, nor the
    flight slot, in a fresh interpreter."""
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", _LEGACY_PATH], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")
