"""FeRRy Phase 5 (unit U5): pair-slot missions of the mule supervisor, in process.

Real ``ClientMission`` devices, a real cluster and a real ``MuleSupervisor`` on
a ``MissionClock`` (``tests/integration/_ferry_harness.py``), flying plan mode
with the flight slot ``pair_q`` (the Phase 5 spec, other choices 1): at each
Pass-1 arrival the slot decides the class the stop is served on and the stop
flown next; the supervisor moves that stop to the front after the service, so
the departure check folds the order the pair set; each decision is recorded
and closed when the mission ends. The scorers are U3's scripted references
(the learned one, U1's, is not built yet). Pinned (the units table, row U5):

* **The wiring.** ``pair_q`` flies the ``pair_slot`` it is given and nothing
  else; the keyword is refused off ``pair_q``; ``install_flight_slot``,
  FerrySim's seam, works before the first mission only and flies exactly as
  the config path.
* **The decision at the arrival**, from a view taken at that instant from the
  runtime and the commit, with the previous Pass-1 observation of the trial;
  every record's mask is the scheduler's predicate re-asked from a state
  rebuilt from the trace; Pass 2 flies b̄ in the queue's order.
* **The reorder**, then the departure check folding it; **the trim** that check
  makes, recorded as ``trimmed_next``.
* **The records** close on all three exits (the empty round, no DOWN, the
  normal path), terminal exactly at the sortie's last decision, a ``home``
  that a beacon insert follows included, with the merge's own weights and
  ``late`` read against each member's own deadline, never its stop's. The
  view's N counts a beacon insert from the decision after it.
* **F, FX and the legacy arms** fly their recorded trials and carry no Phase 5
  field, and load no Phase 5 module; a K = 2 loopback; determinism.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.exp4.model_task import _u32
from experiments.exp4.topology_builder import device_positions, device_spread_m
from hermes.l1.channel_model import SALT_CONTACT, ContactChannel, ferry_salt
from hermes.mission.aggregation_rules import AggregationSpec
from hermes.mule import MuleSupervisorError
from hermes.mule.ferry import FerrySpec
from hermes.processes import mule as process
from hermes.scheduler.plan import AgeCapSpec, PlanOptions
from hermes.scheduler.plan.types import PAIR_FALLBACK_MASK_EMPTY
from hermes.scheduler.policies import FedExCarpPolicy, MaxAoIPolicy
from hermes.scheduler.policies.cross_heuristic import CrossHeuristic, moved_to_front
from hermes.scheduler.policies.pair_slot import (
    CLOSE_KEYS,
    DECISION_KEYS,
    SCRIPTED_SCORERS,
    PairQSlot,
    scripted_scorer,
)
from hermes.scheduler.stages.s3b_feasibility import FlightState
from hermes.types import Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState, MissionPass
from hermes.types import MuleID

from tests.golden import _build_p4_plan as UG5
from tests.golden import _canon
from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

logging.getLogger("hermes.mission.client_mission").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parents[2]
COLLECT = MissionPass.COLLECT
DOCK = (0.0, 0.0, 0.0)
#: The realism field of Phase 4's tests (critic probe A), radius 100 m.
FIELD_M = device_spread_m(60.0, field_radius_m=100.0)
#: The deadline law's unit: deadlines 60 000 s out, so no deadline binds.
FAR_UNIT = 1000.0
#: T in the plan score, near the N = 6 cell's T_nom (Phase 4's tests).
T_NOM = 200.0
#: The Phase 5 trace fields, none of which a recorded arm may carry.
PHASE_5_MISSION_FIELDS = ("pass_1_pairs", "pass_1_e3", "pass_1_e3_unvisited")
PHASE_5_READY_FIELDS = ("pair", "policy_checkpoint")


# --------------------------------------------------------------------------- #
# The world
# --------------------------------------------------------------------------- #

def ref_layout(k, n=6):
    """Phase 4's reference layout ``k`` with ``n`` devices in the 100 m field."""
    seed = _u32(n, "t_nom", 1000 + k)
    xy = device_positions(n, seed, FIELD_M)
    return seed, tuple((f"d{i}", (x, y, 0.0)) for i, (x, y) in enumerate(xy))


def det_spec(seed, layout, *, band="wide", payload_bytes=1_000_000, **kw):
    """Critic B4's deterministic physics (Phase 4's tests): the link keeps its
    sigma, the contact channel has every noise term 0 and availability is 1,
    so every contact takes what the planner priced."""
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
    """The Phase 5 cells' contact channel: jittery, 1 MB, ``replan`` with ``trim``."""
    kw.setdefault("payload_bytes", 1_000_000)
    return FerrySpec.from_config(rf_range_m=60.0, seed=seed, contact_band="wide",
                                 contact_regime="jittery", in_flight_response="replan",
                                 replan_fallback="trim", **kw)


def options(*, slot="pair_q", s=None, policy="search"):
    return PlanOptions(band_class_policy=policy, member_admission="subset", flight_slot=slot,
                       cap=AgeCapSpec(s_missions=s))


def plan_kw(*, budget, slot="pair_q", scorer="fx_pair", s=None, unit=FAR_UNIT, policy="search",
            pair_slot=None):
    """A plan arm's supervisor keywords: FQ with a scripted scorer by default."""
    kw = dict(mission_budget_s=budget, deadline_time_scale=unit, miss_priority=True,
              member_admission="subset", plan_mode="ferry",
              plan_options=options(slot=slot, s=s, policy=policy), t_nom_s=T_NOM)
    if slot == "pair_q":
        kw["pair_slot"] = PairQSlot(scripted_scorer(scorer)) if pair_slot is None else pair_slot
    return kw


def fly(*, spec, layout, missions=1, before=None, after=None, silent=(), world_kw=None,
        **sup_kw):
    """Run ``missions`` missions on the clock; one record per mission. ``after``
    runs right after each mission, while the scheduler still holds its commit."""
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
            if after is not None:
                after(w, sup, m, r)
            records.append(SimpleNamespace(result=r, deltas=w.take_deltas(mid)))
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return w, sup, records


def widened(rec):
    return sorted({str(d.device_id) for src, d in rec.deltas if src == "direct" and d.synthetic})


def spy_decisions(sup, log):
    """Record each decision's view and choice, and the flight states its mask
    was asked from (``FLScheduler.fits_after_service``), changing nothing."""
    slot, sch = sup._flight_slot, sup.scheduler
    real, real_fits = slot.pair_at_arrival, sch.fits_after_service
    current = []

    def fits(rest, **kw):
        if current:
            current[0].states.append(kw["state"])
        return real_fits(rest, **kw)

    def pick(view, **kw):
        entry = SimpleNamespace(view=view, choice=None, states=[])
        log.append(entry)
        current.append(entry)
        try:
            entry.choice = real(view, **kw)
        finally:
            current.clear()
        return entry.choice

    slot.pair_at_arrival, sch.fits_after_service = pick, fits


def spy_checks(sup, log):
    """Record the remainder each departure check is given, by pass."""
    real = sup._ferry_departure

    def dep(fx, remainder, state, **kw):
        log.append((kw["pass_kind"], [[str(d) for d in wp.devices] for wp in remainder]))
        return real(fx, remainder, state, **kw)

    sup._ferry_departure = dep


def landing(spec, r):
    """The Pass-1 landing from the trace: the last stop's end and the return leg."""
    last = r.pass_1_flown[-1]
    return last["end_s"] + spec.flight.leg_s(tuple(last["position"]), DOCK)


def clean_lines(r):
    report = r.report or r.unmerged_report
    return {} if report is None else {
        str(line.device_id): line for line in report.lines if line.outcome.is_on_time()}


def _repeat(results):
    """The results with their wall times dropped: the planner's and, since the
    Exp 5 addendum (Study 5.11 (a)), each flight decision's."""
    return [_canon.canon(dataclasses.replace(r, plan_wall_s=None, pass_1_pairs_wall=None,
                                             pass_1_e3_wall=None))
            for r in results]


#: Decision-rich cells: N = 12 in the 100 m field, 120 s, S = 3, the jittery
#: channel (Study 5.5's cell at the stand-in budget). On these layouts the
#: scorers make up to five decisions per sortie, switch class, re-order, meet
#: empty masks and see the departure check trim their order.
N12_LAYOUTS = (0, 4, 6)


# --------------------------------------------------------------------------- #
# The wiring
# --------------------------------------------------------------------------- #

def _build(**kw):
    seed, layout = ref_layout(0)
    w = H.World(layout=layout, flaky={})
    sim = kw.pop("sim", True)
    with H.Patched(w.clock):
        return w.supervisor(w.mule_ids[0], sim=sim, ferry=det_spec(seed, layout), **kw)


def test_pair_q_flies_the_pair_slot_it_is_given():
    """Other choices 1: ``_init_plan`` returns the ``pair_slot`` for
    ``flight_slot="pair_q"``, so the slot is never None in plan mode."""
    slot = PairQSlot(scripted_scorer("greedy_1"))
    sup = _build(**plan_kw(budget=60.0, pair_slot=slot))
    assert sup._flight_slot is slot and sup.scheduler.plan_mode == "ferry"


@pytest.mark.parametrize("kw, match", [
    (dict(plan_kw(budget=60.0), pair_slot=None), "flies a pair slot"),
    (dict(plan_kw(budget=60.0, slot="committed"), pair_slot=PairQSlot(scripted_scorer("hyb"))),
     "pair_q' only"),
    (dict(plan_kw(budget=60.0, slot="cross_heuristic"),
          pair_slot=PairQSlot(scripted_scorer("hyb"))), "pair_q' only"),
    (dict(plan_kw(budget=60.0), pair_slot=CrossHeuristic()), "PairQSlot"),
    (dict(mission_budget_s=60.0, pair_slot=PairQSlot(scripted_scorer("hyb"))),
     "legacy mule has no flight slot"),
    (dict(mission_budget_s=60.0, sim=False, pair_slot=PairQSlot(scripted_scorer("hyb"))),
     "legacy mule has no flight slot"),
], ids=["pair_q_without_a_slot", "slot_beside_committed", "slot_beside_fx", "not_a_pair_slot",
        "legacy_on_the_clock", "legacy_on_the_wall_clock"])
def test_the_pair_slot_keyword_is_refused_off_pair_q(kw, match):
    """A mule flies a pair only when its options name ``pair_q``, and then only a
    ``PairQSlot``, whose guards hold its scorer to admitted pairs."""
    with pytest.raises(MuleSupervisorError, match=match):
        _build(**kw)


def test_install_flight_slot_is_ferrysims_seam_before_the_first_mission():
    """Other choices 8: FerrySim installs the episode's ``PairQSlot`` on the
    arm's FX configuration (or F's) before the first mission. Refused on a
    legacy mule, after a mission has started, for anything but a pair slot,
    and under a pinned band, which flies the committed slot only."""
    for slot_name in ("cross_heuristic", "committed"):
        sup = _build(**plan_kw(budget=60.0, slot=slot_name))
        slot = PairQSlot(scripted_scorer("fx_pair"))
        sup.install_flight_slot(slot)
        assert sup._flight_slot is slot
    with pytest.raises(MuleSupervisorError, match="legacy mule has none"):
        _build(mission_budget_s=60.0).install_flight_slot(PairQSlot(scripted_scorer("hyb")))
    with pytest.raises(TypeError, match="PairQSlot"):
        _build(**plan_kw(budget=60.0, slot="cross_heuristic")).install_flight_slot(
            CrossHeuristic())
    with pytest.raises(MuleSupervisorError, match="pins class 'wide'"):
        _build(**plan_kw(budget=60.0, slot="committed", policy="fixed:wide")).install_flight_slot(
            PairQSlot(scripted_scorer("hyb")))
    seed, layout = ref_layout(0)
    _, sup, _ = fly(spec=det_spec(seed, layout), layout=layout,
                    **plan_kw(budget=90.0, slot="cross_heuristic"))
    with pytest.raises(MuleSupervisorError, match="before the first mission"):
        sup.install_flight_slot(PairQSlot(scripted_scorer("hyb")))
    assert isinstance(sup._flight_slot, CrossHeuristic)


def test_the_install_path_flies_exactly_as_the_config_path():
    """The two paths to the slot: FX's configuration with the slot installed
    flies every mission as the ``pair_q`` configuration does, decisions and
    records included, since the scheduler never reads the flight slot."""
    seed, layout = ref_layout(0, n=12)
    spec = noisy_spec(seed)
    _, _, config = fly(spec=spec, layout=layout, missions=3,
                       **plan_kw(budget=120.0, s=3, scorer="greedy_1"))

    def install(w, sup, m):
        if m == 0:
            sup.install_flight_slot(PairQSlot(scripted_scorer("greedy_1")))

    _, _, installed = fly(spec=spec, layout=layout, missions=3, before=install,
                          **plan_kw(budget=120.0, s=3, slot="cross_heuristic"))
    one = [rec.result for rec in config]
    assert _repeat(one) == _repeat([rec.result for rec in installed])
    assert sum(len(r.pass_1_pairs or ()) for r in one) >= 6


# --------------------------------------------------------------------------- #
# The decisions and their records
# --------------------------------------------------------------------------- #

def _check_records(spec, r, examples):
    """Every record of mission ``r`` against its trace: one per Pass-1 stop, in
    order, of that stop and class; the pair flown admitted, or FX's with the
    mask empty; the close as the mission settled it."""
    pairs = r.pass_1_pairs or []
    assert len(pairs) == len(r.pass_1_flown)
    clean = clean_lines(r)
    merged = set() if r.aggregate is None else {str(d) for d in r.aggregate.contributing_devices}
    for k, (rec, stop) in enumerate(zip(pairs, r.pass_1_flown)):
        assert list(rec) == list(DECISION_KEYS) + list(CLOSE_KEYS)
        assert rec["devices"] == stop["devices"] and rec["band"] == stop["band"]
        assert rec["t_s"] == stop["arrival_s"] and rec["committed"] == r.band
        if rec["fallback"] is None:
            assert [rec["band"], rec["next_index"]] in rec["admitted_pairs"]
        else:
            assert rec["fallback"] == PAIR_FALLBACK_MASK_EMPTY and rec["admitted_pairs"] == []
            assert rec["band"] == rec["fx_band"] and rec["next_index"] in (0, None)
        assert (rec["next_index"] is None) == (rec["next"] == "home")
        if rec["next_index"] is not None and not rec["trimmed_next"] and k + 1 < len(pairs):
            assert pairs[k + 1]["devices"] == rec["next"]
        assert rec["collected"] == [d for d in stop["devices"] if d in clean]
        assert rec["w"] == [float(examples[d]) if d in merged else 0.0 for d in rec["collected"]]
        assert rec["late"] == [d for d in rec["collected"]
                               if clean[d].contact_ts > r.pass_1_device_deadlines[DeviceID(d)]]
        assert rec["terminal"] == (k == len(pairs) - 1)
        if not rec["terminal"]:
            assert rec["t_next_s"] == pairs[k + 1]["t_s"]
        elif r.empty:
            assert rec["t_next_s"] == pytest.approx(landing(spec, r), abs=1e-6)
        else:
            assert rec["t_next_s"] == pytest.approx(
                landing(spec, r) + r.sim_ledger.get("upload", 0.0), abs=1e-6)
    assert json.loads(json.dumps(pairs)) == pairs
    assert not any("wall" in key for rec in pairs for key in rec)


@pytest.mark.parametrize("scorer", SCRIPTED_SCORERS)
def test_each_scorer_makes_one_decision_at_each_pass_1_arrival(scorer):
    """Other choices 1 on the decision-rich cell: one decision per Pass-1 stop,
    its class the one the stop was flown on, the stop flown next the one it
    named unless the departure check trimmed the order, closed with the
    members collected there and the merge's weight for each (agg:plain: the
    update's example count, 4 + the device's index in the harness), the late
    ones, and the next arrival or, at the sortie's last decision, the end of
    the upload. Pass 2 flies b̄ in the queue's order, and E3's fields stay
    None. Not vacuous: several decisions per sortie, class switches, empty
    masks and trims occur, and re-orders for the nearest-first scorers."""
    seen = {"multi": 0, "switch": 0, "empty": 0, "trimmed": 0, "reorder": 0}
    for k in N12_LAYOUTS:
        seed, layout = ref_layout(k, n=12)
        spec = noisy_spec(seed)
        examples = {d: 4 + i for i, (d, _) in enumerate(layout)}
        _, _, recs = fly(spec=spec, layout=layout, missions=3,
                         **plan_kw(budget=120.0, s=3, scorer=scorer))
        for rec in recs:
            r = rec.result
            _check_records(spec, r, examples)
            pairs = r.pass_1_pairs or []
            assert all(p["scorer"] == scorer for p in pairs)
            assert [s["band"] for s in r.pass_2_flown] == [r.band] * len(r.pass_2_flown)
            assert [s["devices"] for s in r.pass_2_flown] == [
                [str(d) for d in wp.devices] for wp in r.pass_2_queue]
            assert (r.pass_1_e3, r.pass_1_e3_unvisited) == (None, None)
            seen["multi"] += len(pairs) >= 2
            seen["switch"] += sum(p["band"] != p["committed"] for p in pairs)
            seen["empty"] += sum(p["fallback"] is not None for p in pairs)
            seen["trimmed"] += sum(p["trimmed_next"] for p in pairs)
            seen["reorder"] += sum(p["next_index"] not in (0, None) for p in pairs)
    assert seen["multi"] >= 5 and seen["switch"] and seen["empty"] and seen["trimmed"], seen
    if scorer in ("fx_pair", "greedy_1"):
        assert seen["reorder"], seen


#: Three single-device stops (Phase 4's FX test): the plan's tour is c, b, a;
#: after c the nearest stop is a. 10 MB, every device capped (S = 1) and every
#: deadline at t0 + 60 s, so a-first fits only with the exempt stops protected.
TRI = (("c", (40.0, 60.0, 0.0)), ("a", (100.0, 0.0, 0.0)), ("b", (150.0, 80.0, 0.0)))


def test_the_reorder_then_the_departure_check_folds_it():
    """Other choices 1: after the stop the pair's next stop is moved to the
    front, and the next departure check folds that order. At 150 s ``fx_pair``
    picks a after c (the nearest stop whose pair fits), so the check after c
    folds [a, b] and the mule flies c, a, b; ``committed_pair`` keeps the
    plan's b although a-first fits too. At 108 s a-first lands past the budget
    as priced, so the mask refuses it and the plan's order is flown. Every
    deadline falls 60 s after takeoff, so the last stop's update is collected
    late, and its record says so."""
    flights = {}
    spec = det_spec(7, TRI, payload_bytes=10_000_000)
    for budget, scorer in ((150.0, "fx_pair"), (150.0, "committed_pair"), (108.0, "fx_pair")):
        log = []
        _, _, (rec,) = fly(spec=spec, layout=TRI, before=lambda w, sup, m: spy_checks(sup, log),
                           **plan_kw(budget=budget, s=1, unit=1.0, scorer=scorer))
        r = rec.result
        _check_records(spec, r, {d: 4 + i for i, (d, _) in enumerate(TRI)})
        assert [p["late"] for p in r.pass_1_pairs] == [[], [], r.pass_1_pairs[-1]["devices"]]
        first = r.pass_1_pairs[0]
        checks = [order for kind, order in log if kind is COLLECT]
        flights[budget, scorer] = ([s["devices"][0] for s in r.pass_1_flown],
                                   first["next_index"], checks)
        assert (["wide", 1] in first["admitted_pairs"]) == (budget == 150.0)
        assert r.replans == [] and r.budget_overrun_s == 0.0
        assert not any(p["trimmed_next"] for p in r.pass_1_pairs)
    plan_order = [[["c"], ["b"], ["a"]], [["b"], ["a"]], [["a"]]]
    assert flights == {
        (150.0, "fx_pair"): (["c", "a", "b"], 1, [[["c"], ["b"], ["a"]], [["a"], ["b"]], [["b"]]]),
        (150.0, "committed_pair"): (["c", "b", "a"], 0, plan_order),
        (108.0, "fx_pair"): (["c", "b", "a"], 0, plan_order),
    }


def test_the_pair_then_the_departure_check_the_beacon_hook_and_the_slot():
    """R6 and other choices 1, the order Phase 4 pinned for FX: the pair is
    decided at each Pass-1 arrival; at the next departure the check folds the
    order it set, then the beacon hook runs, then the slot picks, index 0; the
    first pick is at takeoff, the last departure runs the beacon hook alone,
    and Pass 2, unchecked in plan mode, has only the slot's picks and no pair."""
    seed, layout = ref_layout(9)
    order = []

    def before(w, sup, m):
        real_dep, real_offers, slot = sup._ferry_departure, sup._take_offers, sup._flight_slot
        real_next, real_pair = slot.next_stop, slot.pair_at_arrival

        def dep(*a, **kw):
            order.append(("check", kw["pass_kind"]))
            return real_dep(*a, **kw)

        def offers(*a, **kw):
            order.append(("beacon",))
            return real_offers(*a, **kw)

        def pick(remainder, state, **kw):
            index = real_next(remainder, state, **kw)
            order.append(("slot", kw["pass_kind"], kw["after_stop"], index))
            return index

        def pair(view, **kw):
            order.append(("pair",))
            return real_pair(view, **kw)

        sup._ferry_departure, sup._take_offers = dep, offers
        slot.next_stop, slot.pair_at_arrival = pick, pair

    _, _, (rec,) = fly(spec=det_spec(seed, layout), layout=layout, before=before,
                       **plan_kw(budget=90.0, s=2))
    r = rec.result
    n1, n2 = len(r.pass_1_flown), len(r.pass_2_flown)
    assert n1 >= 2 and n2 >= 2 and r.replans == []
    expected = []
    for i in range(n1):
        expected += [("check", COLLECT), ("beacon",), ("slot", COLLECT, i > 0, 0), ("pair",)]
    expected += [("beacon",)]
    expected += [("slot", MissionPass.DELIVER, i > 0, 0) for i in range(n2)]
    assert order == expected


#: Three single-device stops on a line from the dock (Phase 4's LINE3).
LINE3 = (("a", (70.0, 0.0, 0.0)), ("b", (140.0, 0.0, 0.0)), ("c", (210.0, 0.0, 0.0)))


def _line3_budget(spec, margin):
    """The budget at which a, b and c fly with ``margin`` seconds to spare."""
    _, _, (rec,) = fly(spec=spec, layout=LINE3, **plan_kw(budget=1000.0))
    r = rec.result
    assert [s["devices"] for s in r.pass_1_flown] == [["a"], ["b"], ["c"]]
    return landing(spec, r) - r.sim_start_s + r.sim_ledger["upload"] + margin


@pytest.mark.parametrize("response, s", [("replan", 1), ("abort", None)])
def test_a_departure_check_that_trims_the_pairs_order_is_recorded(response, s):
    """``trimmed_next``, on a one-class (wide) link so that a, b and c are a
    stop each: b is silent, so its service runs 0.2 s past what the mask priced
    (a 1 s listen window against 0.8 s of dwell), and with 0.05 s to spare the
    order its pair set (c next) no longer fits at b's departure. The check
    re-plans it (``replan``: c is trimmed and widened) or gives up the pass
    (``abort``: c is abandoned and widened); b's decision records it and is the
    sortie's last. With 0.3 s to spare nothing is trimmed."""
    spec = det_spec(7, LINE3, band_classes=("wide",), in_flight_response=response)
    for margin, trimmed in ((0.05, True), (0.3, False)):
        log = []
        _, _, (rec,) = fly(spec=spec, layout=LINE3, silent=("b",),
                           before=lambda w, sup, m: spy_checks(sup, log),
                           **plan_kw(budget=_line3_budget(spec, margin), s=s))
        r = rec.result
        pairs = r.pass_1_pairs
        flown = [stop["devices"] for stop in r.pass_1_flown]
        assert [p["devices"] for p in pairs] == flown
        assert [p["next"] for p in pairs[:2]] == [["b"], ["c"]]
        assert [order for _, order in log][:3] == [[["a"], ["b"], ["c"]], [["b"], ["c"]], [["c"]]]
        if trimmed:
            assert flown == [["a"], ["b"]]
            assert [p["trimmed_next"] for p in pairs] == [False, True]
            assert [p["terminal"] for p in pairs] == [False, True]
            assert pairs[1]["collected"] == []
            if response == "replan":
                assert [(x["before"], x["route"]) for x in r.replans] == [([["c"]], [])]
            else:
                assert [x["abandoned"] for x in r.aborts] == [[["c"]]]
            assert widened(rec) == ["c"]
        else:
            assert flown == [["a"], ["b"], ["c"]]
            assert [p["trimmed_next"] for p in pairs] == [False] * 3
            assert r.replans == [] and r.aborts == []


def test_a_home_that_a_beacon_insert_follows_is_not_terminal():
    """Critic B12: the beacon hook runs at the last departure, so an insert can
    follow a ``home`` decision. That decision is then not terminal: its sortie
    goes on, and it ends at the inserted stop's arrival, where the next
    decision is made, the terminal one. x is offered at the arrival where the
    pair names home. The slot may serve x (an insert is admitted), and the mask
    dates x by the mule's record, its deadline from the insertion. The view's
    N is the plan's demand at the first decision and counts x at the second,
    so a member's share of N never passes 1."""
    offered, asked, demands = [], [], []

    def before(w, sup, m):
        sup.scheduler.device_states[DeviceID("x")] = DeviceSchedulerState(
            device_id=DeviceID("x"), last_known_position=(105.0, 30.0, 0.0))
        real_fits = sup.scheduler.fits_after_service

        def fits(rest, **kw):
            asked.append(kw)
            return real_fits(rest, **kw)

        sup.scheduler.fits_after_service = fits
        slot = sup._flight_slot
        real = slot.pair_at_arrival

        def pick(view, **kw):
            demands.append((view.demand, frozenset(sup.scheduler.last_plan.demand)))
            choice = real(view, **kw)
            if choice.next_index is None and not offered:
                offered.append(view.clock_s)
                sup.offer_contact(ContactWaypoint(position=DOCK, devices=(DeviceID("x"),),
                                                  bucket=Bucket.BEACON_ACTIVE, deadline_ts=0.0))
            return choice

        slot.pair_at_arrival = pick

    spec = det_spec(7, LINE3)
    _, _, (rec,) = fly(spec=spec, layout=LINE3, before=before,
                       world_kw={"extra_devices": {"x": (105.0, 30.0, 0.0)}},
                       **plan_kw(budget=1000.0, s=1))
    r = rec.result
    first, second = r.pass_1_pairs
    (insert,) = r.inserts
    assert insert["devices"] == ["x"] and r.pass_1_flown[-1]["devices"] == ["x"]
    assert first["next"] == "home" and offered == [first["t_s"]]
    assert first["terminal"] is False and first["t_next_s"] == second["t_s"]
    assert second["devices"] == ["x"] and second["collected"] == ["x"]
    assert second["terminal"] is True
    assert second["t_next_s"] == pytest.approx(landing(spec, r) + r.sim_ledger["upload"])
    at_x = [kw for kw in asked if tuple(kw["served_at"].devices) == (DeviceID("x"),)]
    assert len(at_x) == second["pairs"] and all(kw["state"].clock == second["t_s"] for kw in at_x)
    assert all(kw["deadlines"][DeviceID("x")] < float("inf") for kw in at_x)
    assert DeviceID("x") not in r.pass_1_device_deadlines
    ((n_first, demand), (n_second, same)) = demands
    assert same == demand and DeviceID("x") not in demand
    assert (n_first, n_second) == (len(demand), len(demand) + 1)


def test_late_holds_each_member_to_its_own_deadline_not_its_stops():
    """``late`` lists the members collected after their own Deadline(j) (the
    mule's record: the plan's, or a beacon insert's), as the deadline scorer
    reads a CLEAN session, never against their stop's ``deadline_ts``, the
    tightest member's. On layout 0 at N = 12 with a 3 s deadline unit, the
    second mission serves a stop of three members after the tightest one's
    deadline: that member is late, and the two whose own deadlines fall later
    are not, though the stop's deadline has passed for them too. At its own
    deadline a member is on time, as the scorer has it (a CLEAN session no
    later than Deadline(j)): the second mission's decisions, closed again with
    each member's own deadline moved to its contact stamp, have no late
    member, and with it a microsecond before the stamp every collected member
    is late."""
    closes = []

    def before(w, sup, m):
        if m == 0:
            real = sup._ferry_close_pairs

            def close(slot, log, **kw):
                closes.append((sup, slot, log, kw))
                return real(slot, log, **kw)

            sup._ferry_close_pairs = close

    seed, layout = ref_layout(0, n=12)
    spec = noisy_spec(seed)
    examples = {d: 4 + i for i, (d, _) in enumerate(layout)}
    _, _, recs = fly(spec=spec, layout=layout, missions=2, before=before,
                     **plan_kw(budget=120.0, s=3, unit=3.0, scorer="greedy_1"))
    late, on_time = [], []      # members collected after their stop's deadline
    for rec in recs:
        r = rec.result
        _check_records(spec, r, examples)
        clean = clean_lines(r)
        for p, stop in zip(r.pass_1_pairs or [], r.pass_1_flown):
            for d in p["collected"]:
                stamp = clean[d].contact_ts
                if stamp > stop["deadline_ts"]:
                    past_its_own = stamp > r.pass_1_device_deadlines[DeviceID(d)]
                    assert (d in p["late"]) == past_its_own
                    (late if past_its_own else on_time).append(d)
    assert late and on_time, (late, on_time)
    sup, slot, log, kw = closes[1]
    closed = recs[1].result.pass_1_pairs
    stamps = {did: t for d in log.pairs for did, t in d.stamps.items()}
    for shift, all_late in ((0.0, False), (-1e-6, True)):
        log.deadlines = {**log.deadlines, **{did: t + shift for did, t in stamps.items()}}
        again = type(sup)._ferry_close_pairs(sup, slot, log, **kw)
        assert [p["collected"] for p in again] == [p["collected"] for p in closed]
        assert [p["late"] for p in again] == [p["collected"] if all_late else [] for p in again]


def _training_slot(sinks):
    """A slot with a trainer at ε = 0, which flies as no trainer does, so its
    sink receives each mission's closed records (U3's hand-off)."""
    slot = PairQSlot(scripted_scorer("fx_pair"))
    slot.attach_trainer(epsilon=0.0, rng=random.Random(5),
                        sink=lambda steps, records: sinks.append((steps, records)))
    return slot


def _no_down():
    """A mule whose DOWN never comes: a quorum-2 cluster whose other mule never
    flies, and a 0.05 s wait (FeRRy Phase 2's ``down_wait_s``)."""
    sinks = []
    spec = det_spec(11, GH.LAYOUT)
    w = H.World(mule_ids=["mule-a", "mule-b"], assignment=GH.K2_ASSIGNMENT, min_participation=2,
                flaky={})
    ma = MuleID("mule-a")
    with H.Patched(w.clock):
        sup = w.supervisor(ma, sim=True, ferry=spec, down_wait_s=0.05,
                           **plan_kw(budget=150.0, s=2, pair_slot=_training_slot(sinks)))
        w.server.bootstrap(ma)
        assert sup.wait_for_initial_dock(timeout=2.0)
        r = sup.run_one_mission()
    return spec, r, sinks


@pytest.mark.parametrize("exit_", ["normal", "empty", "no_down"])
def test_the_records_close_on_every_exit(exit_):
    """Other choices 1 (critic B2): every exit returns through ``_ferry_result``,
    so the records close on each, and ``close_mission`` gets them once per
    mission. The empty round merged nothing: its decisions collected nothing,
    and its last one ends at the landing, even when the mule docks with an
    empty partial. With no DOWN the upload stands, so the merge's weights and
    the end of the upload hold as on the normal path."""
    if exit_ == "no_down":
        spec, r, sinks = _no_down()
        layout = GH.LAYOUT
        assert r.down_timeout and not r.empty and r.pass_2_flown == []
    else:
        sinks = []
        spec, layout = det_spec(7, LINE3), LINE3
        silent = ("a", "b", "c") if exit_ == "empty" else ()
        _, _, (rec,) = fly(spec=spec, layout=LINE3, silent=silent, dock_on_empty=True,
                           **plan_kw(budget=1000.0, s=1, pair_slot=_training_slot(sinks)))
        r = rec.result
        assert r.empty == (exit_ == "empty") and not r.down_timeout
    _check_records(spec, r, {d: 4 + i for i, (d, _) in enumerate(layout)})
    ((steps, records),) = sinks
    assert list(records) == r.pass_1_pairs and len(steps) == len(records) > 0
    last = r.pass_1_pairs[-1]
    if exit_ == "empty":
        assert r.docked_empty and all(p["collected"] == [] for p in r.pass_1_pairs)
        assert last["t_next_s"] == landing(spec, r)
    else:
        assert all(p["collected"] for p in r.pass_1_pairs)
        assert last["t_next_s"] == pytest.approx(landing(spec, r) + r.sim_ledger["upload"])
        assert last["t_next_s"] > landing(spec, r)


def test_a_mission_without_a_decision_closes_none():
    """A ``pair_q`` mission that flies no Pass-1 stop makes no decision: its
    record list is None, and ``close_mission`` is still called, with none."""
    sinks = []
    layout = (("far", (900.0, 0.0, 0.0)),)
    _, _, (rec,) = fly(spec=det_spec(7, layout), layout=layout,
                       **plan_kw(budget=60.0, pair_slot=_training_slot(sinks)))
    r = rec.result
    assert r.pass_1_flown == [] and r.pass_1_pairs is None
    assert sinks == [((), ())]


def test_under_the_age_cutoff_w_is_the_weight_the_merge_gave():
    """The reward reads what the merge pays (the user's decision 4 (a)): under
    ``agg:cutoff`` an update collected CLEAN past its cutoff is in
    ``collected`` with weight 0, the others with w_i (n_i at age 0). b misses
    mission 1's delivery and answers mission 2 with an update one version
    stale, which the cutoff (a_max = 0) excludes (Phase 4's close test)."""
    b = DeviceID("b")
    layout = (("a", (40.0, 0.0, 0.0)), ("b", (90.0, 30.0, 0.0)), ("c", (140.0, 0.0, 0.0)))

    def before(w, sup, m):
        rf = w.rfs[w.mule_ids[0]]
        if m == 0:
            real_open = sup.mission.open_pass_2

            def open_pass_2(*a, **kw):
                rf.silent.add(b)
                return real_open(*a, **kw)

            sup.mission.open_pass_2 = open_pass_2
        else:
            rf.silent.discard(b)
            w.devices[b].train_offline()

    _, _, recs = fly(spec=det_spec(7, layout), layout=layout, missions=2, before=before,
                     world_kw={"aggregation": AggregationSpec(rule="agg:cutoff", a_max=0)},
                     **plan_kw(budget=1000.0, s=1))
    first, second = (rec.result for rec in recs)
    assert [p["collected"] for p in first.pass_1_pairs] == [["a", "b", "c"]]
    assert [p["w"] for p in first.pass_1_pairs] == [[4.0, 5.0, 6.0]]
    assert second.aggregate.excluded_devices == (b,)
    assert [p["collected"] for p in second.pass_1_pairs] == [["a", "b", "c"]]
    assert [p["w"] for p in second.pass_1_pairs] == [[4.0, 0.0, 6.0]]


# --------------------------------------------------------------------------- #
# The view and the mask at the decision's time
# --------------------------------------------------------------------------- #

def _rebuilt(spec, stop, deliver_by=float("inf")):
    """The arrival at a flown stop as the trace records it: its waypoint, and
    the flight state there (the departure's energy plus the leg at flight
    power, and ``deliver_by``, the trace's own, from the caller)."""
    wp = ContactWaypoint(position=tuple(stop["position"]),
                         devices=tuple(DeviceID(d) for d in stop["devices"]),
                         bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=stop["deadline_ts"])
    energy = stop["depart_energy_j"] + spec.flight.energy.p_move_w * stop["transit_s"]
    return wp, FlightState(wp.position, stop["arrival_s"], energy, deliver_by)


def test_every_records_mask_was_the_predicate_at_its_time():
    """Freeze principle 12, re-checked from the trace: right after each
    mission, while the scheduler still holds its commit, every decision's mask
    is asked again of the scheduler's predicate (``fits_after_service``) from a
    state rebuilt from the trace and the arrival view recomputed at that time.
    The state is the stop at its arrival time, the energy spent by then, and,
    under route-level ``delivery``, the earliest own deadline of the uncapped
    members collected CLEAN at the stops before. It is the record's mask
    exactly, so the pair flown fitted when it was chosen, or no pair did and
    FX's was flown. Over the decision-rich cells, one with an energy capacity
    and one under ``delivery`` with near deadlines, so that the budget, the
    energy and the updates on board each refuse pairs, and TRI at 108 s, where
    the mask refuses the reorder."""
    reasons = {}
    seed0, layout0 = ref_layout(0, n=12)
    cases = [(noisy_spec(ref_layout(k, n=12)[0]), ref_layout(k, n=12)[1], 120.0, 3, FAR_UNIT, 3)
             for k in N12_LAYOUTS]
    cases += [(noisy_spec(seed0, energy_capacity_j=15000.0), layout0, 120.0, 3, FAR_UNIT, 3),
              (noisy_spec(seed0, deadline_bounds="delivery"), layout0, 120.0, 3, 3.0, 3),
              (det_spec(7, TRI, payload_bytes=10_000_000), TRI, 108.0, 1, 1.0, 1)]
    for spec, layout, budget, s, unit, missions in cases:
        decisions = []
        delivery = spec.deadline_bounds == "delivery"

        def after(w, sup, m, r, decisions=decisions, spec=spec, budget=budget,
                  delivery=delivery):
            mission = list(decisions)
            decisions.clear()
            assert len(mission) == len(r.pass_1_flown)
            end = r.sim_start_s + budget
            positions = sup._ferry_positions(r.pass_1_queue)
            clean, capped = clean_lines(r), set(r.plan["cap"]["capped"])
            deliver_by = float("inf")
            for d, stop, rec in zip(mission, r.pass_1_flown, r.pass_1_pairs):
                wp, state = _rebuilt(spec, stop, deliver_by)
                arrival = sup._ferry_run.arrival_view(wp, positions, stop["arrival_s"],
                                                      pass_kind=COLLECT)
                assert d.view.arrival == arrival
                assert d.view.clock_s == state.clock and d.view.pose == wp.position
                assert d.view.energy_j == pytest.approx(state.energy_j)
                # The mask was asked once per pair, from the arrival's own state.
                assert len(d.states) == len(d.view.pairs)
                for asked in d.states:
                    assert (asked.pose, asked.clock, asked.deliver_by) == (
                        state.pose, state.clock, state.deliver_by)
                    assert asked.energy_j == pytest.approx(state.energy_j)
                reasons["on_board"] = reasons.get("on_board", 0) + (deliver_by < float("inf"))
                mask = []
                for band, index in d.view.pairs:
                    entry = arrival.entry(band)
                    rest = [] if index is None else moved_to_front(list(d.view.remainder), index)
                    res = sup.scheduler.fits_after_service(
                        rest, served_at=wp, state=state, dwell_s=entry.dwell_s,
                        collected=entry.targets, budget_end=end,
                        deadlines=r.pass_1_device_deadlines)
                    mask.append(res.ok)
                    for _, why in res.rejected[:1]:
                        reasons[why] = reasons.get(why, 0) + 1
                assert rec["admitted_pairs"] == [[b, i] for (b, i), ok in zip(d.view.pairs, mask)
                                                 if ok]
                reasons["asked"] = reasons.get("asked", 0) + len(mask)
                if delivery:
                    deliver_by = min([deliver_by] + [
                        r.pass_1_device_deadlines[DeviceID(x)] for x in stop["devices"]
                        if x in clean and x not in capped])

        fly(spec=spec, layout=layout, missions=missions, after=after,
            before=lambda w, sup, m, decisions=decisions: (
                spy_decisions(sup, decisions) if m == 0 else None),
            **plan_kw(budget=budget, s=s, unit=unit, scorer="greedy_1"))
    assert reasons["asked"] > 150, reasons
    assert all(reasons.get(why, 0) > 0 for why in ("budget", "energy", "delivery", "on_board")), (
        reasons)


def test_the_view_is_taken_at_the_arrival_from_the_runtime_and_the_commit():
    """Other choices 1 and 4, as U4 and U0 hand them over: the view is the
    runtime's at the arrival (the observed SNR and offsets per class, every
    candidate priced from the stop), the commit's (T_nom, N and its weight, S,
    each candidate's cap flags, mean age and coverage weight) and the sortie's
    (the budget, P_c). The previous offsets are the trial's last Pass-1
    observation, carried across stops and missions, with their age, and reset
    with the mule. At S = 2 and 120 s the candidates include stops the cap
    leaves alone, binds for every member (exempt) and binds for some (mixed);
    at 90 s the plan leaves part of its demand out, so N (the demand) is not
    the devices served."""
    seed, layout = ref_layout(4, n=12)
    spec = noisy_spec(seed)
    flags, short = set(), 0
    for budget in (120.0, 90.0):
        decisions, checked = [], []

        def after(w, sup, m, r, decisions=decisions, checked=checked, budget=budget):
            nonlocal short
            fx, commit = sup._ferry_run, sup.scheduler.last_plan
            positions = sup._ferry_positions(r.pass_1_queue)
            mission = decisions[len(checked):]
            assert len(mission) == len(r.pass_1_flown)
            short += bool(mission) and len(commit.served) < len(commit.demand)
            for d, stop in zip(mission, r.pass_1_flown):
                view, wp = d.view, _rebuilt(spec, stop)[0]
                t = view.clock_s
                assert view.observed_snr_db == fx.observe(wp, positions, t).class_snr_db
                assert view.offsets_db == fx.class_offsets_db(wp, positions, t)
                prices = fx.stop_contexts(wp, list(view.remainder), positions,
                                          pass_kind=COLLECT)
                assert [(c.travel_s, c.pred_dwell_s, c.pred_snr_db) for c in view.stops] == [
                    tuple(p) for p in prices]
                assert (view.t_ref_s, view.cap_s, view.period_s) == (T_NOM, 2, 60.0)
                assert (view.budget_s, view.budget_end) == (budget, r.sim_start_s + budget)
                assert view.energy_ref_j == fx.energy_ref_j(budget) > 0.0
                assert view.demand == len(commit.demand)
                assert view.demand_weight == pytest.approx(
                    sum(commit.weights[d_] for d_ in commit.demand))
                exempt = sup.scheduler.plan_protected(list(view.remainder))
                for ctx in view.stops:
                    if ctx.is_home:
                        continue
                    members = list(ctx.stop.devices)
                    assert ctx.capped == any(d_ in commit.capped for d_ in members)
                    assert ctx.exempt == (ctx.stop in exempt)
                    assert ctx.age == pytest.approx(
                        sum(commit.ages.get(d_, 0) for d_ in members) / len(members))
                    assert ctx.weight == pytest.approx(
                        sum(commit.weights.get(d_, 0.0) for d_ in members))
                    assert 0.0 <= ctx.on_time <= 1.0
                    flags.add((ctx.capped, ctx.exempt))
                checked.append(d)

        fly(spec=spec, layout=layout, missions=3, after=after,
            before=lambda w, sup, m, decisions=decisions: (
                spy_decisions(sup, decisions) if m == 0 else None),
            **plan_kw(budget=budget, s=2))
        assert len(checked) == len(decisions) >= 3
        first = checked[0].view
        assert first.previous_offsets_db is None and first.previous_age_s is None
        for prev, d in zip(checked, checked[1:]):
            assert d.view.previous_offsets_db == prev.view.offsets_db
            assert d.view.previous_age_s == d.view.clock_s - prev.view.clock_s
        assert any(len(c.view.covering) > 1 for c in checked)
    assert flags == {(False, False), (True, False), (True, True)}
    assert short >= 2


# --------------------------------------------------------------------------- #
# F, FX, the legacy arms; K = 2; determinism; what loads
# --------------------------------------------------------------------------- #

def test_the_phase_5_fields_appear_only_in_pair_q_missions():
    """Freeze Rule 1 (the goldens let an added key pass, so the absence is
    pinned here): F, FX and the legacy arms on both clocks leave every Phase 5
    field of the result None, and their ``mission_completed`` carries none of
    them; a ``pair_q`` mission records its decisions and no E3 field."""
    seed, layout = ref_layout(0, n=12)
    spec = noisy_spec(seed)
    results = []
    for slot in ("committed", "cross_heuristic", "pair_q"):
        _, _, recs = fly(spec=spec, layout=layout, missions=2,
                         **plan_kw(budget=120.0, s=3, slot=slot))
        results += [(slot, rec.result) for rec in recs]
    for policy in (None, MaxAoIPolicy(), FedExCarpPolicy(depot=DOCK)):
        _, _, recs = fly(spec=spec, layout=layout, mission_budget_s=120.0,
                         deadline_time_scale=FAR_UNIT, target_selector=policy)
        results += [("legacy", rec.result) for rec in recs]
    w = H.World(flaky={})
    with H.Patched(w.clock):
        sup = w.supervisor(w.mule_ids[0], sim=False, mission_budget_s=70.0)
        w.bootstrap()
        results.append(("wall", sup.run_one_mission()))
    for slot, r in results:
        if slot == "pair_q":
            assert r.pass_1_pairs and (r.pass_1_e3, r.pass_1_e3_unvisited) == (None, None)
            continue
        assert all(getattr(r, name) is None for name in PHASE_5_MISSION_FIELDS), slot
        if slot != "wall":
            assert not set(PHASE_5_MISSION_FIELDS) & set(process._sim_mission_fields(r)), slot


@pytest.mark.parametrize("trial", ["f_45s", "fx_45s", "fx_n12_120s"])
def test_f_and_fx_fly_their_386c275_trials_with_no_phase_5_field(trial):
    """Freeze Rule 1's third face (UG5's oracles): F and FX, run end to end
    through the driver, fly their 386c275 trials part for part, the flight
    slot's calls and their arguments included; and since an added key passes
    there, no ``mule_ready`` or ``mission_completed`` carries a Phase 5 field."""
    golden = UG5.load_golden()["cases"][trial]
    case = UG5.capture(trial)
    for part in UG5.PARTS:
        assert UG5.compare_part(part, golden[part], case[part]) == [], part
    for events in case["mule_ready"].values():
        assert events and all(not set(PHASE_5_READY_FIELDS) & set(e) for e in events)
    for events in case["mission_completed"].values():
        assert events and all(not set(PHASE_5_MISSION_FIELDS) & set(e) for e in events)
    assert case["flight_slot"] and all(UG5.ADDED_ARGUMENTS not in c for c in case["flight_slot"])


def test_two_pair_q_mules_each_decide_on_their_own_slice():
    """K = 2 (critic B10; the spec's kept choices): two plan-mode mules, each
    with its own pair slot and scorer, share a quorum-2 cluster (mule B flies
    inside mule A's DOWN wait). Each decides only about its own slice's stops
    and closes its own records; neither slot sees the other's decisions."""
    spec = det_spec(11, GH.LAYOUT)
    w = H.World(mule_ids=["mule-a", "mule-b"], assignment=GH.K2_ASSIGNMENT, min_participation=2,
                echo_sim_ts=True, flaky={})
    ma, mb = MuleID("mule-a"), MuleID("mule-b")
    sinks = {"mule-a": [], "mule-b": []}
    results = {"mule-a": [], "mule-b": []}
    scorers = {"mule-a": "fx_pair", "mule-b": "greedy_1"}
    with H.Patched(w.clock):
        sups = {}
        for mid in (ma, mb):
            slot = PairQSlot(scripted_scorer(scorers[str(mid)]))
            slot.attach_trainer(epsilon=0.0, rng=random.Random(1),
                                sink=lambda steps, rs, m=str(mid): sinks[m].append(rs))
            sups[mid] = w.supervisor(mid, sim=True, ferry=spec, down_wait_s=30.0,
                                     **plan_kw(budget=150.0, s=2, pair_slot=slot))
        w.bootstrap()
        for _ in range(3):
            w.server.tasks[ma].append(
                lambda: results["mule-b"].append(sups[mb].run_one_mission()))
            results["mule-a"].append(sups[ma].run_one_mission())
            w.clock.advance(GH.BETWEEN_MISSIONS_DT)
    for mule, rs in results.items():
        mine = {d for d, m in GH.K2_ASSIGNMENT.items() if m == mule}
        assert len(rs) == 3 and len(sinks[mule]) == 3
        for r, closed in zip(rs, sinks[mule]):
            assert r.pass_1_pairs and list(closed) == r.pass_1_pairs
            for p in r.pass_1_pairs:
                assert set(p["devices"]) <= mine and p["scorer"] == scorers[mule]
                assert p["next"] == "home" or set(p["next"]) <= mine
    assert sum(len(r.pass_1_pairs) for r in results["mule-a"]) >= 6


def test_a_repeated_pair_q_trial_gives_identical_results_bar_the_plan_wall_time():
    """Critic B12 for the decisions: the records hold no wall time and the slot
    draws nothing without a trainer, so a repeated trial on the jittery channel
    gives identical results, records included, bar the planner's wall time."""
    seed, layout = ref_layout(6, n=12)

    def trial():
        _, _, recs = fly(spec=noisy_spec(seed), layout=layout, missions=3,
                         **plan_kw(budget=120.0, s=3, scorer="greedy_1"))
        return [rec.result for rec in recs]

    one, two = trial(), trial()
    assert _repeat(one) == _repeat(two)
    assert sum(len(r.pass_1_pairs or ()) for r in one) >= 6


_LOADS = r"""
import sys
from experiments.exp4.model_task import _u32
from experiments.exp4.topology_builder import device_positions, device_spread_m
from hermes.mule.ferry import FerrySpec
from hermes.scheduler.plan import AgeCapSpec, PlanOptions
from hermes.scheduler.policies import FedExCarpPolicy, MaxAoIPolicy
from tests.integration import _ferry_harness as H

PHASE_5 = ("pair_slot", "pair_features", "pair_q", "pair_replay", "chen_dqn", "next_stop")


def loaded():
    return sorted(m for m in sys.modules if m.startswith("experiments.ferrysim")
                  or m.rsplit(".", 1)[-1] in PHASE_5)


seed = _u32(12, "t_nom", 1000)
layout = tuple((f"d{i}", (x, y, 0.0)) for i, (x, y) in enumerate(
    device_positions(12, seed, device_spread_m(60.0, field_radius_m=100.0))))
spec = FerrySpec.from_config(rf_range_m=60.0, seed=seed, contact_band="wide",
                             contact_regime="jittery", in_flight_response="replan",
                             replan_fallback="trim", payload_bytes=1_000_000)


def fly(**kw):
    w = H.World(layout=layout, flaky={})
    with H.Patched(w.clock):
        sup = w.supervisor(w.mule_ids[0], sim=True, ferry=spec, mission_budget_s=120.0,
                           deadline_time_scale=1000.0, **kw)
        w.bootstrap()
        return [sup.run_one_mission() for _ in range(2)]


assert not loaded(), loaded()
for slot in ("committed", "cross_heuristic"):
    rs = fly(plan_mode="ferry", member_admission="subset", t_nom_s=200.0, miss_priority=True,
             plan_options=PlanOptions(flight_slot=slot, cap=AgeCapSpec(s_missions=3)))
    assert any(r.pass_1_flown for r in rs)
for policy in (None, MaxAoIPolicy(), FedExCarpPolicy(depot=(0.0, 0.0, 0.0))):
    fly(target_selector=policy)
assert not loaded(), loaded()
print("ok")
"""


def test_f_fx_and_the_legacy_arms_load_no_phase_5_module():
    """The spec's kept choices (other choices 13): F, FX, H1, D1 and D4 load none
    of ``pair_slot``, ``pair_features``, ``pair_q``, ``pair_replay``,
    ``chen_dqn``, ``next_stop`` or ``experiments.ferrysim``, in a fresh
    interpreter: the pair slot's module is imported only on the ``pair_q``
    path, and E3's protocol only on E3's."""
    env = dict(os.environ, PYTHONPATH=str(REPO), PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", _LOADS], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")
