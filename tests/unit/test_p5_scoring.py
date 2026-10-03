"""FeRRy Phase 5 (unit U9): scoring the learned fillings, and Study 5.5's statistics.

The Phase 5 spec's units table, row U9, with other choices 6, 9 and 12 and the
user's decision 5 (a). Pinned:

* **The consumer.** ``MissionRecord`` carries the pair slot's closed decisions
  (``pass_1_pairs``, built here with the slot's own ``closed_record``, so a
  schema change fails these tests), arm E3's next-stop calls (``pass_1_e3``)
  and the stops its pass left (``pass_1_e3_unvisited``); None where a mission
  does not record them. Each field is read only in the JSON form the mule
  writes it in, so a field in another form (a bool, float or string index, a
  bool or string number, ...) reads as None, a list as empty, and a record
  whose choice is in another form is no decision: an index that cannot be
  read never reads as home.
* **The scorer's Phase 5 columns** on synthetic traces, each value worked by
  hand, only with ``pair_columns`` (``--pair-columns``), after the τ columns
  and in the spec's order (other choices 9, restated here); blank on a trial
  that flew no learned filling; the counts 0 and the means blank on a
  pair-slot trial that made no decision; FerrySim's installed slot (FX's
  configuration) counted by its records; every mule's decisions pooled. The
  shares count the slot's choices, a choice the departure check then
  re-planned included.
* **The default row unchanged** on Phase 3 and Phase 4 traces: UG4's eight
  6e6f92d trials and UG5's eight 386c275 plan-arm trials, rebuilt as the kept
  traces they were, score exactly as the 386c275 scorer (loaded from git)
  scores them, column for column; with ``pair_columns`` the seven columns
  follow, blank.
* **Provenance parity** for the Phase 5 arms: E3's ``policy_params`` (its
  checkpoint by tag and sha, which the scorer wrote blank before this unit)
  and the FQ arms' ``ferry_params`` equal the driver's row with nothing
  spawned, and on stub trials of FQ and E3 run through the real services in
  this process with their traces kept, whose columns equal a recount of the
  kept records; so do a FerrySim episode's.
* **TOST** against scipy's one-sided t tests, closed forms at 1 and 2 degrees
  of freedom, its (1 − 2α) interval at the default α and at others, the
  zero-variance case and its refusals.
* **The trend test** (Page's L) against scipy's ``page_trend_test`` (exact and
  asymptotic, both directions), scipy's exhaustive within-seed
  ``permutation_test`` where seeds tie, and a null enumerated by hand; the
  normal approximation's permutation variance; ρ as the mean per-seed
  Spearman ρ; the seeded bootstrap over the seeds, with its draws, confidence
  and seed; ``auto``'s bounds, inclusive; refusals.
* **Imports.** Scoring a Phase 5 trace loads no plan module and no Phase 5
  module.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from experiments.analysis.stats import (
    EXACT_MAX_LEVELS,
    EXACT_MAX_SEEDS,
    TREND_ALTERNATIVES,
    bootstrap_ci,
    tost_paired,
    trend_test,
)
from experiments.analysis.traces_scorer import (
    PHASE_5_COLUMNS,
    PairReport,
    main,
    pair_report,
    score_trial,
    score_traces,
    trial_provenance,
)
from experiments.exp4.driver import PROVENANCE_COLUMNS, Exp4Driver, trace_dir_name
from experiments.exp4.events_consumer import (
    PAIR_MASK_EMPTY,
    E3Call,
    MissionRecord,
    PairDecision,
    consume_run_dir,
    observation_from_rows,
)
from experiments.runner import Cell
from hermes.l1.mission_clock import SIM_EPOCH_S
from hermes.scheduler.plan.types import PAIR_FALLBACK_MASK_EMPTY
from hermes.scheduler.policies.chen_dqn import new_e3_network, save_e3_checkpoint
from hermes.scheduler.policies.pair_slot import CLOSE_KEYS, DECISION_KEYS, HOME, closed_record
from hermes.scheduler.selector.pair_features import PairFeatureSchema
from hermes.scheduler.selector.pair_q import KIND_PAIR_Q, PairQConfig, PairQNet

from tests.golden import _build_p3_sim as UG4
from tests.golden import _build_p4_plan as UG5
from tests.golden import _build_topology as T

REPO = Path(__file__).resolve().parents[2]
WALL = 1_700_000_000.0
E = SIM_EPOCH_S
SHA = "0123456789abcdef" * 4
TAUS = (0.82, 0.5)
#: The modules no scoring path may load (the Phase 5 spec, other choices 13).
PHASE_5_MODULES = ("pair_slot", "pair_features", "pair_q", "pair_replay", "chen_dqn",
                   "next_stop")
#: The Phase 5 columns in the spec's order (other choices 9), restated rather
#: than read from the scorer, so a reordered constant fails.
SPEC_COLUMNS = ("pair_decisions", "pair_feasible_mean", "pair_mask_empty",
                "pair_fx_agree_share", "pair_band_off_bbar_share", "pair_reorder_share",
                "e3_unvisited_mean")


# --------------------------------------------------------------------------- #
# Hand-built traces
# --------------------------------------------------------------------------- #

def _decision(devices, *, committed, band, next_index, nxt=None, pairs, feasible, admitted,
              fallback=None, fx_band, fx_next, agrees_fx, t_s, t_next_s, terminal=False,
              collected=None, late=(), trimmed_next=False):
    """A closed ``pass_1_pairs`` record, closed by the slot's own ``closed_record``
    (so it holds exactly the slot's keys, ``DECISION_KEYS`` then ``CLOSE_KEYS``)."""
    collected = list(devices) if collected is None else list(collected)
    record = {
        "t_s": E + t_s, "devices": list(devices), "committed": committed, "band": band,
        "next_index": next_index, "next": HOME if next_index is None else list(nxt),
        "pairs": pairs, "feasible": feasible, "admitted_pairs": [list(p) for p in admitted],
        "fallback": fallback, "fx_band": fx_band, "fx_next": fx_next, "agrees_fx": agrees_fx,
        "scorer": "pair_v1", "q": 0.125, "q_fx": -0.5,
    }
    out = closed_record(record, collected=collected, weights={d: 10.0 for d in collected},
                        late=late, t_next_s=E + t_next_s, terminal=terminal,
                        trimmed_next=trimmed_next)
    assert list(out) == list(DECISION_KEYS) + list(CLOSE_KEYS)
    return out


#: Three missions of an FQ trial, worked by hand. Mission 2 is an empty round:
#: no stop flown, so no decision and no ``pass_1_pairs``.
#:
#:   decision  stop  b̄       band    next   admitted  fallback    FX's pair
#:   1         a     wide     wide    0      3         -           chosen
#:   2         b     wide     medium  home   1         -           not chosen
#:   3         c     narrow   narrow  2      2         -           not chosen
#:   4         d     narrow   wide    0      0         mask_empty  chosen
#:   5         e     narrow   narrow  home   1         -           chosen
#:
#: Five decisions; (3 + 1 + 2 + 0 + 1) / 5 = 1.4 pairs admitted on average; one
#: empty mask; FX's pair chosen at 3 of 5 (0.6); served off b̄ at 2 of 5 (0.4:
#: decisions 2 and 4); a stop other than the remainder's head chosen at 1 of 5
#: (0.2: decision 3, whose order the departure check then re-planned,
#: ``trimmed_next``: the shares count choices; home is no re-order).
PAIR_MISSIONS = [
    [_decision(["a"], committed="wide", band="wide", next_index=0, nxt=["b"], pairs=4,
               feasible=3, admitted=[("wide", 0), ("wide", 1), ("medium", 0)], fx_band="wide",
               fx_next=0, agrees_fx=True, t_s=10.0, t_next_s=20.0),
     _decision(["b"], committed="wide", band="medium", next_index=None, pairs=2, feasible=1,
               admitted=[("medium", None)], fx_band="wide", fx_next=None, agrees_fx=False,
               t_s=20.0, t_next_s=60.0, terminal=True, collected=[])],
    None,
    [_decision(["c"], committed="narrow", band="narrow", next_index=2, nxt=["e"], pairs=6,
               feasible=2, admitted=[("narrow", 2), ("wide", 1)], fx_band="wide", fx_next=1,
               agrees_fx=False, t_s=410.0, t_next_s=420.0, trimmed_next=True),
     _decision(["d"], committed="narrow", band="wide", next_index=0, nxt=["e"], pairs=3,
               feasible=0, admitted=[], fallback=PAIR_MASK_EMPTY, fx_band="wide", fx_next=0,
               agrees_fx=True, t_s=420.0, t_next_s=430.0, late=["d"]),
     _decision(["e"], committed="narrow", band="narrow", next_index=None, pairs=1,
               feasible=1, admitted=[("narrow", None)], fx_band="narrow", fx_next=None,
               agrees_fx=True, t_s=430.0, t_next_s=470.0, terminal=True)],
]
PAIR_COLUMNS_BY_HAND = {
    "pair_decisions": 5, "pair_feasible_mean": pytest.approx(1.4), "pair_mask_empty": 1,
    "pair_fx_agree_share": pytest.approx(0.6), "pair_band_off_bbar_share": pytest.approx(0.4),
    "pair_reorder_share": pytest.approx(0.2), "e3_unvisited_mean": "",
}


def _e3_call(stops, admissible, next_index, *, after_stop, t_s):
    return {"t_s": E + t_s, "after_stop": after_stop, "stops": [list(s) for s in stops],
            "admissible": list(admissible), "next_index": next_index,
            "next": "home" if next_index is None else list(stops[next_index])}


def _left(*stops):
    return [{"position": [10.0, 0.0, 0.0], "devices": list(s), "deadline_ts": E + 1e4,
             "widened": False} for s in stops]


#: Four missions of an E3 trial. Its pass left 2, 0, 1 and 0 stops unvisited
#: (the mule writes the field only when it left some): 3 / 4 = 0.75 a mission.
#: The last mission flew no stop and made no call.
E3_MISSIONS = [
    dict(e3=[_e3_call([["a"], ["b"], ["c"]], [True, True, False], 0, after_stop=False, t_s=0.0),
             _e3_call([["b"], ["c"]], [False, False], None, after_stop=True, t_s=9.0)],
         left=_left(["b"], ["c"])),
    dict(e3=[_e3_call([["a"]], [True], 0, after_stop=False, t_s=200.0)], left=None),
    dict(e3=[_e3_call([["d"], ["e"]], [True, False], 0, after_stop=False, t_s=400.0),
             _e3_call([["e"]], [False], None, after_stop=True, t_s=409.0)],
         left=_left(["e"])),
    dict(e3=None, left=None),
]

#: A Phase 5 build's FQ mule config: the plan fields and the six checkpoint keys.
FQ_MULE = {
    "mule_id": "m1", "rf_range_m": 60.0, "n_missions": 3, "session_ttl_s": 3.0,
    "mission_clock": "sim", "trial_seed": 42, "contact_band": "wide",
    "backhaul_model": "mission", "in_flight_response": "replan", "plan_mode": "ferry",
    "band_class_policy": "search", "member_admission": "subset", "flight_slot": "pair_q",
    "age_cap_missions": None, "age_cap_lookahead": 0, "plan_score_params": {},
    "plan_search_params": {}, "t_nom_s": 200.0, "miss_priority": True,
    "pair_checkpoint": "results/exp5/checkpoints/s55/main/g0.9_s0.npz",
    "pair_checkpoint_sha256": SHA, "pair_checkpoint_tag": "main",
    "policy_checkpoint": None, "policy_checkpoint_sha256": None, "policy_checkpoint_tag": None,
}
#: FX's: the configuration FerrySim installs its pair slots on.
FX_MULE = dict(FQ_MULE, flight_slot="cross_heuristic", pair_checkpoint=None,
               pair_checkpoint_sha256=None, pair_checkpoint_tag=None)
#: E3's: legacy mode, Chen et al.'s DQN as the whole scheduler.
E3_MULE = dict(FX_MULE, plan_mode="legacy", flight_slot="committed", member_admission="whole",
               t_nom_s=None, miss_priority=False, contact_policy="chen_dqn",
               policy_checkpoint="results/exp5/checkpoints/s53/e3/e3_s0.npz",
               policy_checkpoint_sha256=SHA, policy_checkpoint_tag="e3")
H1_MULE = dict(E3_MULE, contact_policy=None, member_admission="subset",
               policy_checkpoint=None, policy_checkpoint_sha256=None, policy_checkpoint_tag=None)
DEVICES = ("a", "b", "c", "d", "e")


def _mule_rows(pairs=(), e3=(), *, mule="m1"):
    """A mule's events: one mission per entry of ``pairs`` (its decisions, or
    None) or of ``e3`` (its calls and the stops it left)."""
    rows = [{"ts": WALL - 5.0, "event": "mule_ready", "role": "mule", "id": mule,
             "mission_clock": "sim"}]
    missions = [dict(pairs=p) for p in pairs] or [dict(e3=m["e3"], left=m["left"]) for m in e3]
    for i, m in enumerate(missions):
        start = WALL + 10.0 * i
        flown = [r["devices"] for r in m.get("pairs") or ()
                 if isinstance(r, dict) and isinstance(r["devices"], list)] or [
            c["next"] for c in m.get("e3") or () if c["next"] != "home"]
        row = {"ts": start + 6.0, "event": "mission_completed", "role": "mule", "id": mule,
               "mission_round": i + 1, "pass_1_contacts": len(flown), "pass_2_contacts": 0,
               "pass_1_clean_devices": [d for s in flown for d in s],
               "pass_1_merged_devices": [d for s in flown for d in s],
               "pass_1_merged_updates": sum(len(s) for s in flown),
               "sim_start_s": E + 200.0 * i, "sim_end_s": E + 200.0 * i + 150.0, "band": "wide",
               "pass_1_flown": [{"band": "wide", "devices": s} for s in flown]}
        if m.get("pairs"):
            row["pass_1_pairs"] = m["pairs"]
        if m.get("e3"):
            row["pass_1_e3"] = m["e3"]
        if m.get("left"):
            row["pass_1_e3_unvisited"] = m["left"]
        rows += [{"ts": start, "event": "mission_started", "role": "mule", "id": mule,
                  "sim_start_s": E + 200.0 * i}, row]
    return rows


def _write_trace(root, arm, *, mule_cfg, pairs=(), e3=()):
    d = root / f"N=5-regime=jittery-rrf=60.0__{arm}__t0__s42"
    d.mkdir(parents=True)
    files = {
        "cluster-c1.jsonl": [{"ts": WALL - 6.0, "event": "cluster_ready", "role": "cluster",
                              "id": "c1", "mission_clock": "sim"}],
        "mule-m1.jsonl": _mule_rows(pairs, e3),
        "device-a.jsonl": [{"ts": WALL - 5.0, "event": "device_ready", "role": "device",
                            "id": dev} for dev in DEVICES],
    }
    for name, rows in files.items():
        (d / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (d / "mule-m1.json").write_text(json.dumps(mule_cfg), encoding="utf-8")
    (d / "cluster.json").write_text(json.dumps({"seed_devices": [
        {"device_id": dev, "position": [10.0 * i, 0.0, 0.0]} for i, dev in enumerate(DEVICES)]}),
        encoding="utf-8")
    return d


def _obs(pairs=(), e3=()):
    return observation_from_rows(cluster_rows=[], mule_rows=_mule_rows(pairs, e3),
                                 device_rows=[], n_devices=len(DEVICES))


def _row(root, arm, *, pair_columns=True, **kw):
    return score_trial(_write_trace(root, arm, **kw), taus=TAUS,
                       pair_columns=pair_columns).to_row()


def _columns(row):
    return {c: row[c] for c in PHASE_5_COLUMNS}


BLANK = dict.fromkeys(PHASE_5_COLUMNS, "")


# --------------------------------------------------------------------------- #
# 1. The consumer
# --------------------------------------------------------------------------- #

def test_a_pair_q_mission_carries_its_closed_decisions():
    m1, m2, m3 = _obs(PAIR_MISSIONS).missions
    assert m2.pair_decisions is None                      # no decision, no field
    assert [len(m.pair_decisions) for m in (m1, m3)] == [2, 3]
    first, home = m1.pair_decisions
    assert first == PairDecision(
        t_s=E + 10.0, devices=("a",), committed="wide", band="wide", next_index=0,
        next_devices=("b",), pairs=4, feasible=3,
        admitted_pairs=(("wide", 0), ("wide", 1), ("medium", 0)), fallback=None,
        fx_band="wide", fx_next=0, agrees_fx=True, scorer="pair_v1", q=0.125, q_fx=-0.5,
        collected=("a",), w=(10.0,), late=(), t_next_s=E + 20.0, terminal=False,
        trimmed_next=False)
    assert (home.next_index, home.next_devices, home.admitted_pairs) == (
        None, None, (("medium", None),))
    assert (home.collected, home.w, home.terminal) == ((), (), True)
    c, d, e = m3.pair_decisions
    assert [x.reorders for x in (first, home, c, d, e)] == [False, False, True, False, False]
    assert [x.mask_empty for x in (first, home, c, d, e)] == [False, False, False, True, False]
    assert (c.trimmed_next, d.late, d.fallback) == (True, ("d",), "mask_empty")
    # The consumer loads no plan module, so it restates the slot's fallback tag.
    assert PAIR_MASK_EMPTY == PAIR_FALLBACK_MASK_EMPTY


def test_e3_missions_carry_their_calls_and_unvisited_stops():
    m1, m2, m3, m4 = _obs(e3=E3_MISSIONS).missions
    assert m1.e3_calls == (
        E3Call(t_s=E + 0.0, after_stop=False, stops=(("a",), ("b",), ("c",)),
               admissible=(True, True, False), next_index=0),
        E3Call(t_s=E + 9.0, after_stop=True, stops=(("b",), ("c",)),
               admissible=(False, False), next_index=None))
    assert [m.e3_unvisited for m in (m1, m2, m3, m4)] == [
        (("b",), ("c",)), None, (("e",),), None]
    assert m4.e3_calls is None
    assert all(m.pair_decisions is None for m in (m1, m2, m3, m4))


def test_every_phase_5_record_field_defaults_to_none():
    record = MissionRecord(mission_round=1, pass_1_contacts=0, pass_2_contacts=0,
                           pass_1_updates=None, pass_1_scheduled=None,
                           pass_1_clean_devices=(), delivered=None, undelivered=None,
                           duration_s=None)
    assert (record.pair_decisions, record.e3_calls, record.e3_unvisited) == (None, None, None)
    for m in _obs([None, None]).missions:                 # missions that record none
        assert (m.pair_decisions, m.e3_calls, m.e3_unvisited) == (None, None, None)


#: A key left out of a record, a form the mule never writes (it writes every key).
MISSING = object()

#: One field of a closed record in a form the mule never writes, and what the
#: decision holds in its place; every other field reads as written. The slot
#: writes a number as a finite int or float, a count or an index as an int
#: >= 0 (None for home), a flag as a bool, a name as a string and ids as a list
#: of strings (``pair_slot.closed_record``).
ODD_FIELDS = {
    "t_s-bool": ("t_s", True, {"t_s": None}),
    "t_s-numeric-string": ("t_s", "12", {"t_s": None}),
    "t_s-nan": ("t_s", math.nan, {"t_s": None}),
    "t_next_s-inf": ("t_next_s", math.inf, {"t_next_s": None}),
    "q-nan-string": ("q", "nan", {"q": None}),
    "q_fx-bool": ("q_fx", False, {"q_fx": None}),
    "pairs-bool": ("pairs", True, {"pairs": None}),
    "pairs-numeric-string": ("pairs", "4", {"pairs": None}),
    "feasible-float": ("feasible", 2.9, {"feasible": None}),
    "feasible-whole-float": ("feasible", 2.0, {"feasible": None}),
    "feasible-negative": ("feasible", -1, {"feasible": None}),
    "feasible-missing": ("feasible", MISSING, {"feasible": None}),
    "agrees_fx-int": ("agrees_fx", 1, {"agrees_fx": None}),
    "terminal-string": ("terminal", "yes", {"terminal": None}),
    "trimmed_next-int": ("trimmed_next", 0, {"trimmed_next": None}),
    "committed-int": ("committed", 7, {"committed": None}),
    "fallback-list": ("fallback", [PAIR_MASK_EMPTY], {"fallback": None}),
    "scorer-int": ("scorer", 1, {"scorer": None}),
    "devices-string": ("devices", "a", {"devices": ()}),
    "devices-int-id": ("devices", ["a", 7], {"devices": ()}),
    "collected-int-id": ("collected", ["a", 7], {"collected": ()}),
    "late-int-id": ("late", [7], {"late": ()}),
    "next-int-id": ("next", [7], {"next_devices": None}),
    "w-string": ("w", [5.0, "heavy"], {"w": ()}),
    "w-bool": ("w", [True], {"w": ()}),
    "w-nan": ("w", [math.nan], {"w": ()}),
    # An admitted pair that cannot be read is left out; the others are read.
    "admitted-bool-index": ("admitted_pairs", [["wide", True], ["wide", 1]],
                            {"admitted_pairs": (("wide", 1),)}),
    "admitted-float-index": ("admitted_pairs", [["wide", 1.5], ["medium", 0]],
                             {"admitted_pairs": (("medium", 0),)}),
    "admitted-string-index": ("admitted_pairs", [["wide", "1"], ["medium", None]],
                              {"admitted_pairs": (("medium", None),)}),
    "admitted-negative-index": ("admitted_pairs", [["wide", -1]], {"admitted_pairs": ()}),
    "admitted-int-band": ("admitted_pairs", [[7, 1]], {"admitted_pairs": ()}),
    "admitted-short": ("admitted_pairs", [["wide"], ["wide", 1]],
                       {"admitted_pairs": (("wide", 1),)}),
    # FX's pair is read whole: a half alone would pair FX's band with home.
    "fx_next-bool": ("fx_next", True, {"fx_band": None, "fx_next": None}),
    "fx_next-whole-float": ("fx_next", 0.0, {"fx_band": None, "fx_next": None}),
    "fx_next-numeric-string": ("fx_next", "2", {"fx_band": None, "fx_next": None}),
    "fx_next-missing": ("fx_next", MISSING, {"fx_band": None, "fx_next": None}),
    "fx_band-int": ("fx_band", 7, {"fx_band": None, "fx_next": None}),
}
#: A record whose choice, ``band`` and ``next_index``, is in a form the mule
#: never writes.
ODD_CHOICES = {
    "next_index-true": ("next_index", True),
    "next_index-false": ("next_index", False),
    "next_index-float": ("next_index", 1.7),
    "next_index-whole-float": ("next_index", 1.0),
    "next_index-numeric-string": ("next_index", "2"),
    "next_index-negative": ("next_index", -1),
    "next_index-missing": ("next_index", MISSING),
    "band-none": ("band", None),
    "band-int": ("band", 7),
    "band-missing": ("band", MISSING),
}


def _changed(record, key, value):
    """``record`` with ``key`` set to ``value``, or left out for :data:`MISSING`."""
    out = {k: v for k, v in record.items() if k != key}
    if value is not MISSING:
        out[key] = value
    return out


def _decisions(*records):
    """The decisions of one mission whose ``pass_1_pairs`` holds ``records``."""
    (m,) = observation_from_rows(cluster_rows=[], mule_rows=_mule_rows([list(records)]),
                                 device_rows=[], n_devices=5).missions
    return m.pair_decisions


@pytest.mark.parametrize("key, value, reads", ODD_FIELDS.values(), ids=list(ODD_FIELDS))
def test_a_field_in_a_form_the_mule_never_writes_reads_as_none(key, value, reads):
    """Never as a value of another kind: a bool is no index 1 or weight 1.0, a
    float index is not truncated, a numeric string is not parsed, and a number
    that is not finite is none the mule writes. The decision is still read."""
    good = PAIR_MISSIONS[0][0]
    (want,) = _decisions(good)
    assert _decisions(_changed(good, key, value)) == (dataclasses.replace(want, **reads),)


@pytest.mark.parametrize("key, value", ODD_CHOICES.values(), ids=list(ODD_CHOICES))
def test_a_record_whose_choice_cannot_be_read_holds_no_decision(key, value):
    """``next_index`` None is home, so an index that cannot be read must not read
    as None: the record is skipped, as an entry that is not a record is, and
    the decisions around it are read."""
    first, home = PAIR_MISSIONS[0]
    assert _decisions(first, _changed(first, key, value), "not a record", home) == (
        _decisions(first, home))


def test_the_strict_reading_refuses_no_form_a_record_may_hold():
    """An int is a number (JSON does not tell 3 from 3.0 once a trace is
    re-written), and None is home in each index, FX's next stop and an
    admitted pair's included. The stub trials below read every record a real
    mule wrote as written."""
    good = PAIR_MISSIONS[0][0]
    (want,) = _decisions(good)
    (got,) = _decisions(dict(good, t_s=7, w=[3], fx_next=None,
                             admitted_pairs=[["medium", None], ["wide", 2]]))
    assert got == dataclasses.replace(want, t_s=7.0, w=(3.0,), fx_next=None,
                                      admitted_pairs=(("medium", None), ("wide", 2)))
    assert (got.fx_band, got.fx_next) == ("wide", None)


def test_e3s_records_in_a_form_the_mule_never_writes():
    """As the pair slot's: a field in another form reads as None or empty, and a
    call whose choice cannot be read is skipped (its None would read as home)."""
    call = _e3_call([["a"], ["b"]], [True, False], 0, after_stop=True, t_s=5.0)
    odd = dict(call, t_s="5", after_stop=0, stops=[["a"], [1]], admissible=[True, 1])
    unread = [_changed(call, "next_index", v) for v in (False, True, 0.0, 0.9, "0", -1, MISSING)]
    rows = _mule_rows([None])
    rows[-1].update(pass_1_e3=[odd, "not a record", *unread, call], pass_1_e3_unvisited=[
        {"devices": ["x"]}, {"position": [1, 2]}, "x", {"devices": [1]}, {"devices": "x"}])
    (m,) = observation_from_rows(cluster_rows=[], mule_rows=rows, device_rows=[],
                                 n_devices=5).missions
    assert m.e3_calls == (
        E3Call(t_s=None, after_stop=None, stops=(), admissible=(), next_index=0),
        E3Call(t_s=E + 5.0, after_stop=True, stops=(("a",), ("b",)), admissible=(True, False),
               next_index=0))
    assert m.e3_unvisited == (("x",),)


def test_a_phase_5_field_that_is_not_a_list_reads_as_absent():
    for raw in ("pairs", {"a": 1}, 3):
        rows = _mule_rows([None])
        rows[-1].update(pass_1_pairs=raw, pass_1_e3=raw, pass_1_e3_unvisited=raw)
        (m,) = observation_from_rows(cluster_rows=[], mule_rows=rows, device_rows=[],
                                     n_devices=5).missions
        assert (m.pair_decisions, m.e3_calls, m.e3_unvisited) == (None, None, None)


# --------------------------------------------------------------------------- #
# 2. The scorer's columns, on the hand-built trials
# --------------------------------------------------------------------------- #

def test_the_columns_of_a_pair_q_trial_and_where_they_sit(tmp_path):
    assert PHASE_5_COLUMNS == SPEC_COLUMNS
    row = _row(tmp_path, "FQ", mule_cfg=FQ_MULE, pairs=PAIR_MISSIONS)
    cols = list(row)
    assert cols[-len(SPEC_COLUMNS):] == list(SPEC_COLUMNS)
    assert cols[-len(SPEC_COLUMNS) - 1] == f"sim_s_to_tau{TAUS[-1]:g}"   # after the τ ones
    assert _columns(row) == PAIR_COLUMNS_BY_HAND
    # Stops served per mission (Study 5.6) stays the summary's own column.
    assert row["pass1_contacts_mean"] == pytest.approx(5 / 3)
    # The provenance names the checkpoint by tag and sha, never by path.
    params = json.loads(row["ferry_params"])
    assert (params["pair_tag"], params["pair_sha256"]) == ("main", SHA)
    assert not [v for v in row.values() if isinstance(v, str) and "checkpoints" in v]
    assert row["policy_params"] == ""


def test_the_columns_appear_only_with_pair_columns(tmp_path):
    d = _write_trace(tmp_path, "FQ", mule_cfg=FQ_MULE, pairs=PAIR_MISSIONS)
    plain = score_trial(d, taus=TAUS)
    paired = score_trial(d, taus=TAUS, pair_columns=True).to_row()
    assert plain.pairs is None and not set(PHASE_5_COLUMNS) & set(plain.to_row())
    assert list(paired) == list(plain.to_row()) + list(PHASE_5_COLUMNS)
    assert {c: paired[c] for c in plain.to_row()} == plain.to_row()
    (scored,) = score_traces(tmp_path, taus=TAUS, pair_columns=True)
    assert scored.to_row() == paired
    (scored,) = score_traces(tmp_path, taus=TAUS)
    assert scored.to_row() == plain.to_row()


def _csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_cli_writes_the_columns_only_when_asked(tmp_path):
    root = tmp_path / "traces"
    _write_trace(root, "FQ", mule_cfg=FQ_MULE, pairs=PAIR_MISSIONS)
    out = tmp_path / "scored.csv"
    assert main(["--traces", str(root), "--csv", str(out)]) == 0
    (row,) = _csv(out)
    assert not set(PHASE_5_COLUMNS) & set(row)
    assert main(["--traces", str(root), "--csv", str(out), "--pair-columns"]) == 0
    (row,) = _csv(out)
    assert list(row)[-len(SPEC_COLUMNS):] == list(SPEC_COLUMNS)
    assert {c: row[c] for c in SPEC_COLUMNS} == {
        "pair_decisions": "5", "pair_feasible_mean": "1.4", "pair_mask_empty": "1",
        "pair_fx_agree_share": "0.6", "pair_band_off_bbar_share": "0.4",
        "pair_reorder_share": "0.2", "e3_unvisited_mean": ""}


def test_e3s_column_is_its_unvisited_stops_per_mission(tmp_path):
    row = _row(tmp_path, "E3", mule_cfg=E3_MULE, e3=E3_MISSIONS)
    assert _columns(row) == dict(BLANK, e3_unvisited_mean=pytest.approx(0.75))
    assert row["policy_params"] == json.dumps({"policy_sha256": SHA, "policy_tag": "e3"},
                                              sort_keys=True)
    visited = [dict(m, left=None) for m in E3_MISSIONS]
    assert _row(tmp_path / "all", "E3", mule_cfg=E3_MULE, e3=visited)["e3_unvisited_mean"] == 0.0
    # E3's configuration says so even where no mission recorded a call.
    silent = [dict(e3=None, left=None)] * 2
    assert _row(tmp_path / "none", "E3", mule_cfg=E3_MULE, e3=silent)["e3_unvisited_mean"] == 0.0


def test_a_pair_q_trial_without_a_decision_counts_none_and_averages_nothing(tmp_path):
    row = _row(tmp_path, "FQ", mule_cfg=FQ_MULE, pairs=[None, None])
    assert _columns(row) == dict(BLANK, pair_decisions=0, pair_mask_empty=0)


def test_ferrysims_installed_slot_is_counted_by_its_records(tmp_path):
    """FerrySim flies its pair slots on FX's configuration
    (``install_flight_slot``), so the trace names FX's slot; its records say
    what flew."""
    row = _row(tmp_path, "FX", mule_cfg=FX_MULE, pairs=PAIR_MISSIONS)
    assert _columns(row) == PAIR_COLUMNS_BY_HAND
    assert json.loads(row["ferry_params"])["flight_slot"] == "cross_heuristic"


@pytest.mark.parametrize("arm, cfg", [("FX", FX_MULE), ("H1", H1_MULE)])
def test_an_arm_without_a_learned_filling_scores_blank(tmp_path, arm, cfg):
    assert _columns(_row(tmp_path, arm, mule_cfg=cfg, pairs=[None, None, None])) == BLANK


def test_a_learned_trial_with_no_mission_has_nothing_to_average():
    for cfg in (FQ_MULE, E3_MULE):
        assert pair_report(_obs(), mule_cfg=cfg) == (
            PairReport(pair_decisions=0, pair_mask_empty=0) if cfg is FQ_MULE else PairReport())
    # E3's records under another configuration count by the records.
    assert pair_report(_obs(e3=E3_MISSIONS), mule_cfg=H1_MULE).e3_unvisited_mean == 0.75


def test_a_share_leaves_out_a_decision_that_does_not_record_its_field():
    """And the empty mask is read from ``fallback`` (unit U3's record), not
    from ``feasible``, which this decision does not record."""
    first = PAIR_MISSIONS[0][0]
    odd = dict(PAIR_MISSIONS[0][1], agrees_fx=None, committed=None, feasible="many",
               fallback=PAIR_MASK_EMPTY)
    report = pair_report(_obs([[first, odd]]), mule_cfg=FQ_MULE)
    assert report == PairReport(pair_decisions=2, pair_feasible_mean=3.0, pair_mask_empty=1,
                                pair_fx_agree_share=1.0, pair_band_off_bbar_share=0.0,
                                pair_reorder_share=0.0)


def test_with_several_mules_every_mules_decisions_count():
    rows = _mule_rows(PAIR_MISSIONS[:1]) + _mule_rows(PAIR_MISSIONS[2:], mule="m2")
    obs = observation_from_rows(cluster_rows=[], mule_rows=rows, device_rows=[], n_devices=5)
    assert obs.n_mules == 2
    assert pair_report(obs, mule_cfg=FQ_MULE) == pair_report(_obs(PAIR_MISSIONS),
                                                             mule_cfg=FQ_MULE)


# --------------------------------------------------------------------------- #
# 3. Phase 3 and Phase 4 traces score their 386c275 rows
# --------------------------------------------------------------------------- #

REF_COMMIT = "386c275"


def _git_show(path: str):
    if shutil.which("git") is None:
        return None
    done = subprocess.run(["git", "show", f"{REF_COMMIT}:{path}"], cwd=REPO,
                          capture_output=True, timeout=120)
    return done.stdout if done.returncode == 0 else None


@pytest.fixture(scope="module")
def ref_scorer(tmp_path_factory):
    """``traces_scorer`` and its consumer as they were at 386c275 (Phase 4), from git.

    Each blob runs as a module of its own under a private name; while the
    scorer runs, ``experiments.exp4.events_consumer`` names the recorded
    consumer, so the reference scores with the consumer it was written for.
    What else they import is live (the driver's provenance columns, the
    metrics), and unchanged by this unit.
    """
    blobs = {name: _git_show(path) for name, path in (
        ("consumer", "experiments/exp4/events_consumer.py"),
        ("scorer", "experiments/analysis/traces_scorer.py"))}
    if any(b is None for b in blobs.values()):
        pytest.skip(f"git or commit {REF_COMMIT} is not available")
    where = tmp_path_factory.mktemp("ref386c275")
    names = {"consumer": f"experiments.exp4._p5ref_{REF_COMMIT}_events_consumer",
             "scorer": f"experiments.analysis._p5ref_{REF_COMMIT}_traces_scorer"}
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


def _plain(value):
    """A canonical fixture value (``tests/golden/_canon.py``) as the JSON it was."""
    if isinstance(value, str) and value.startswith("f:"):
        return float(value[2:])
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k != "__type__"}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _kept_trace(root: Path, case) -> Path:
    """A golden trial as the kept trace it was: every role's JSONL in its
    recorded order, and the per-role JSON."""
    cell = case["inputs"]["cell"]
    d = root / trace_dir_name(Cell(cell_id=cell["cell_id"], arm=cell["arm"],
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


#: UG4's Phase 3 trials (6e6f92d) and UG5's Phase 4 plan-arm trials (386c275).
RECORDED = [("p3", n) for n in UG4.TRIAL_NAMES] + [("p4", n) for n in UG5.TRIAL_NAMES]


@pytest.fixture(scope="module")
def recorded_traces(tmp_path_factory):
    root = tmp_path_factory.mktemp("u9_recorded")
    out = {}
    for face, module in (("p3", UG4), ("p4", UG5)):
        for name, case in module.load_golden()["cases"].items():
            out[face, name] = _kept_trace(root / face / name, case)
    return out


def _same(live, ref):
    """Two rows equal column for column and in order (nan equal to nan)."""
    assert list(live) == list(ref)
    for col in ref:
        a, b = live[col], ref[col]
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
            assert math.isnan(b), col
        else:
            assert (col, a) == (col, b)


@pytest.mark.parametrize("face, name", RECORDED, ids=[f"{f}-{n}" for f, n in RECORDED])
def test_a_phase_3_or_4_trace_scores_its_386c275_row(recorded_traces, ref_scorer, face, name):
    d = recorded_traces[face, name]
    ref = ref_scorer.score_trial(d, taus=TAUS).to_row()
    _same(score_trial(d, taus=TAUS).to_row(), ref)
    paired = score_trial(d, taus=TAUS, pair_columns=True).to_row()
    assert list(paired) == list(ref) + list(PHASE_5_COLUMNS)
    _same({c: paired[c] for c in ref}, ref)
    assert _columns(paired) == BLANK
    # A plan arm's trace is a Phase 4 one: no mission records a Phase 5 field.
    done = [json.loads(line) for p in d.glob("mule-*.jsonl")
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert not {"pass_1_pairs", "pass_1_e3", "pass_1_e3_unvisited"} & {
        k for e in done for k in e}


# --------------------------------------------------------------------------- #
# 4. Provenance parity, and stub trials of the learned arms with their traces kept
# --------------------------------------------------------------------------- #

def _trained(**changes):
    """A trained, scored checkpoint's provenance (unit U8b's, critic B9)."""
    out = dict(reward={"kind": "derived", "c_t": 0.1, "c_cov": 1.0},
               training={"episodes": 2000}, seeds={"init": 1, "train": 7},
               cell_family="jittery", cell_family_sha256="ab" * 32, trainer_commit="a" * 40,
               dirty=False, episodes_trained=2000,
               validation=[{"episode": 1000, "return_mean": 0.31}],
               held_out={"episodes": 1000, "return_mean": 0.33})
    out.update(changes)
    return out


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    """One pair checkpoint for tags main and g50 each, and E3's, made under tmp."""
    root = tmp_path_factory.mktemp("u9_ckpt")
    schema = PairFeatureSchema(("wide", "medium", "narrow"))
    pair, shas = {}, {}
    for seed, tag in enumerate(("main", "g50")):
        path = root / tag / f"g0.5_s{seed}.npz"
        shas[tag] = PairQNet(schema.dim, PairQConfig(gamma=0.5), seed=seed).save(
            path, kind=KIND_PAIR_Q, purpose="trained", schema=schema.to_json(),
            classes=list(schema.classes), provenance=_trained())
        pair[tag] = str(path)
    path = root / "e3" / "e3_s0.npz"
    shas["e3"] = save_e3_checkpoint(new_e3_network(seed=3), path, band="wide",
                                    purpose="trained",
                                    provenance=_trained(reward={"kind": "bytes"}))
    return {"pair": pair, "policy": {"e3": str(path)}, "shas": shas}


@pytest.fixture(scope="module")
def parity_driver(ckpts):
    """UG5's flags (the Phase 4 pilots', 1 MB, the jittery contact channel, 45 s,
    S = 2) on the seconds backhaul (where H1+L1 flies), with the checkpoints."""
    return Exp4Driver(**dict(UG5.PLAN_PILOT, t_nom_layouts=3, backhaul_model="seconds",
                             pair_checkpoints=ckpts["pair"],
                             policy_checkpoints=ckpts["policy"]))


PARITY_ARMS = ("FQ", "FQ-g50", "E3", "F", "FX", "H1", "D3", "D5", "H1+L1")


@pytest.mark.parametrize("arm", PARITY_ARMS)
def test_the_scored_provenance_is_the_drivers_row(tmp_path, parity_driver, ckpts, arm):
    """The configs of a kept trace, as the real orchestrator writes them (nothing
    spawned): every provenance column the scorer derives is the driver's,
    E3's checkpoint in ``policy_params`` and an FQ arm's in ``ferry_params``."""
    cell = UG5._cell(arm, UG5.SEED)
    row, topo = T.run_stub_trial(parity_driver, cell)
    d = tmp_path / trace_dir_name(cell)
    d.mkdir()
    roles = T.role_configs(topo)
    (d / "cluster.json").write_text(json.dumps(roles["cluster"]), encoding="utf-8")
    for mid, cfg in roles["mules"].items():
        (d / f"mule-{mid}.json").write_text(json.dumps(cfg), encoding="utf-8")
    for did, cfg in roles["devices"].items():
        (d / f"device-{did}.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert trial_provenance(d) == {c: row[c] for c in PROVENANCE_COLUMNS}
    if arm == "E3":
        assert json.loads(row["policy_params"]) == {"policy_sha256": ckpts["shas"]["e3"],
                                                    "policy_tag": "e3"}
    if arm.startswith("FQ"):
        tag = "main" if arm == "FQ" else "g50"
        params = json.loads(row["ferry_params"])
        assert (params["pair_tag"], params["pair_sha256"]) == (tag, ckpts["shas"][tag])


def _in_process(settings, cell, root):
    """(row, kept trace) of one trial run through the real services in this
    process (UG4's harness), its traces kept under ``root``."""
    driver = Exp4Driver(**settings, trace_root=str(root))
    with UG4.in_process_orchestrator():
        UG4.InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
    return row, root / trace_dir_name(cell)


def _completed(trace):
    (log,) = sorted(trace.glob("mule-*.jsonl"))
    return [e for e in (json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
                        if line.strip()) if e["event"] == "mission_completed"]


def _recount(records):
    """The pair columns, recounted from raw ``pass_1_pairs`` records by a reading
    of their own."""
    return {
        "pair_decisions": len(records),
        "pair_feasible_mean": pytest.approx(statistics.fmean(r["feasible"] for r in records)),
        "pair_mask_empty": sum(r["fallback"] == "mask_empty" for r in records),
        "pair_fx_agree_share": pytest.approx(statistics.fmean(r["agrees_fx"] for r in records)),
        "pair_band_off_bbar_share": pytest.approx(
            statistics.fmean(r["band"] != r["committed"] for r in records)),
        "pair_reorder_share": pytest.approx(
            statistics.fmean(r["next_index"] not in (0, None) for r in records)),
        "e3_unvisited_mean": "",
    }


def _as_written(r):
    """A raw ``pass_1_pairs`` record as the decision it holds, field for field
    as the mule wrote it."""
    return PairDecision(
        t_s=r["t_s"], devices=tuple(r["devices"]), committed=r["committed"], band=r["band"],
        next_index=r["next_index"], next_devices=None if r["next"] == HOME else tuple(r["next"]),
        pairs=r["pairs"], feasible=r["feasible"],
        admitted_pairs=tuple((b, i) for b, i in r["admitted_pairs"]), fallback=r["fallback"],
        fx_band=r["fx_band"], fx_next=r["fx_next"], agrees_fx=r["agrees_fx"],
        scorer=r["scorer"], q=r["q"], q_fx=r["q_fx"], collected=tuple(r["collected"]),
        w=tuple(r["w"]), late=tuple(r["late"]), t_next_s=r["t_next_s"],
        terminal=r["terminal"], trimmed_next=r["trimmed_next"])


def _read_as_written(trace, done, n_devices):
    """Whether the consumer reads every kept record of ``trace`` as the mule wrote
    it: the strict reading refuses no form the mule writes."""
    missions = consume_run_dir(trace, n_devices=n_devices).missions
    assert len(missions) == len(done)
    for e, m in zip(done, missions):
        pairs = e.get("pass_1_pairs")
        assert m.pair_decisions == (None if pairs is None else tuple(map(_as_written, pairs)))
        calls = e.get("pass_1_e3")
        assert m.e3_calls == (None if calls is None else tuple(
            E3Call(t_s=c["t_s"], after_stop=c["after_stop"],
                   stops=tuple(tuple(s) for s in c["stops"]),
                   admissible=tuple(c["admissible"]), next_index=c["next_index"])
            for c in calls))
        left = e.get("pass_1_e3_unvisited")
        assert m.e3_unvisited == (None if left is None else tuple(
            tuple(s["devices"]) for s in left))
    return True


def test_a_stub_fq_trial_scores_its_kept_decisions(tmp_path, ckpts):
    """FQ at N = 12, 120 s, S = 3 (UG5's decision-rich trial): one closed
    record per Pass-1 stop flown, each read as written, and the columns are
    their recount."""
    settings = dict(UG5.PLAN_PILOT, t_nom_layouts=5, mission_budget_s=120.0,
                    age_cap_missions=3, pair_checkpoints={"main": ckpts["pair"]["main"]})
    row, trace = _in_process(settings, UG5._cell("FQ", UG5.N12_SEED, N=12, n_missions=4),
                             tmp_path)
    assert (row["missions_completed"], row["mission_failures"]) == (4, 0)
    done = _completed(trace)
    records = [r for e in done for r in e.get("pass_1_pairs", ())]
    assert len(records) == sum(len(e["pass_1_flown"]) for e in done) > 4
    assert _read_as_written(trace, done, n_devices=12)
    # An empty mask falls back on FX's pair, so the agreement share counts it as
    # agreeing.
    empty = [r for r in records if r["fallback"] == "mask_empty"]
    assert empty and all(r["agrees_fx"] and r["feasible"] == 0 for r in empty)
    got = score_trial(trace, taus=TAUS, pair_columns=True).to_row()
    assert _columns(got) == _recount(records)
    assert {c: got[c] for c in PROVENANCE_COLUMNS} == {c: row[c] for c in PROVENANCE_COLUMNS}


def test_a_stub_e3_trial_scores_its_provenance_and_unvisited_stops(tmp_path, ckpts):
    """E3 on the Phase 3 pilots' flags (wide, 60 s, legacy mode): its row's
    ``policy_params`` names the checkpoint, and the scorer's is the same; every
    call and unvisited stop it kept is read as written; its pass leaves stops
    unvisited, which the column averages over the missions."""
    settings = dict(UG4.PILOT, t_nom_layouts=5, policy_checkpoints=ckpts["policy"])
    row, trace = _in_process(settings, UG4._cell("E3", UG4.SEED), tmp_path)
    assert (row["missions_completed"], row["mission_failures"]) == (4, 0)
    got = score_trial(trace, taus=TAUS, pair_columns=True).to_row()
    assert {c: got[c] for c in PROVENANCE_COLUMNS} == {c: row[c] for c in PROVENANCE_COLUMNS}
    assert json.loads(got["policy_params"]) == {"policy_sha256": ckpts["shas"]["e3"],
                                                "policy_tag": "e3"}
    done = _completed(trace)
    left = [len(e.get("pass_1_e3_unvisited", ())) for e in done]
    assert all("pass_1_e3" in e for e in done) and sum(left) > 0
    assert _read_as_written(trace, done, n_devices=6)
    assert _columns(got) == dict(BLANK, e3_unvisited_mean=pytest.approx(statistics.fmean(left)))


def test_a_ferrysim_episodes_kept_trace_scores_its_slots_decisions(tmp_path):
    """A FerrySim episode of a scripted slot, its trace kept for the scorer
    (unit U8a): the trace names FX's configuration, every kept record is read
    as written, and the columns are the recount of the mule's closed records
    the episode read in process."""
    from experiments.ferrysim import cells as C
    from experiments.ferrysim import episode as EP

    cell = C.cell_named("jit-n12-120")
    seed = C.stream_seeds(C.VAL_STREAM, cell.name, 1)[0]
    ep = EP.run_episode(cell, seed, EP.Policy.scripted("greedy_1"),
                        driver_overrides={"trace_root": tmp_path, "t_nom_layouts": 3})
    root = tmp_path / cell.name / "greedy_1"
    (trace,) = root.iterdir()
    assert json.loads(next(trace.glob("mule-*.json")).read_text())["flight_slot"] == (
        "cross_heuristic")
    records = [r for mission in ep.pair_records if mission for r in mission]
    assert len(records) == ep.decisions > 0
    assert _read_as_written(trace, _completed(trace), n_devices=12)
    (scored,) = score_traces(root, taus=TAUS, pair_columns=True)
    assert _columns(scored.to_row()) == _recount(records)
    (plain,) = score_traces(root, taus=TAUS)
    assert not set(PHASE_5_COLUMNS) & set(plain.to_row())


# --------------------------------------------------------------------------- #
# 5. TOST
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("alpha", [None, 0.025, 0.1, 0.2])
def test_tost_is_scipys_two_one_sided_paired_t_tests(alpha):
    """Each one-sided test is scipy's ``ttest_1samp`` of the differences against
    the bound, and equivalence is the (1 − 2α) t interval inside (−ε, ε): at
    the default α, 0.05 (None here), and at others."""
    level = 1.0 - 2.0 * (0.05 if alpha is None else alpha)
    kw = {} if alpha is None else {"alpha": alpha}
    rng = np.random.default_rng(11)
    for _ in range(300):
        n = int(rng.integers(2, 25))
        a = rng.normal(0.0, 1.0, n)
        b = a + rng.normal(rng.uniform(-0.5, 0.5), rng.uniform(0.05, 1.0), n)
        eps = float(rng.uniform(0.05, 1.5))
        res = tost_paired(a, b, margin=eps, **kw)
        d = a - b
        lower = sps.ttest_1samp(d, -eps, alternative="greater")
        upper = sps.ttest_1samp(d, eps, alternative="less")
        assert (res.n_pairs, res.df, res.margin) == (n, n - 1, eps)
        assert res.alpha == (0.05 if alpha is None else alpha)
        assert res.mean_diff == pytest.approx(d.mean(), rel=1e-12, abs=1e-15)
        assert res.t_lower == pytest.approx(lower.statistic, rel=1e-9)
        assert res.t_upper == pytest.approx(upper.statistic, rel=1e-9)
        assert res.p_lower == pytest.approx(lower.pvalue, rel=1e-9, abs=1e-300)
        assert res.p_upper == pytest.approx(upper.pvalue, rel=1e-9, abs=1e-300)
        assert res.p_value == max(res.p_lower, res.p_upper)
        low, high = sps.t.interval(level, n - 1, loc=d.mean(), scale=sps.sem(d))
        assert (res.ci_low, res.ci_high) == (pytest.approx(low, rel=1e-9, abs=1e-12),
                                             pytest.approx(high, rel=1e-9, abs=1e-12))
        assert res.equivalent == (res.ci_low > -eps and res.ci_high < eps)


def test_tost_on_hand_worked_cases():
    """With one degree of freedom t is Cauchy, F(t) = 1/2 + atan(t) / π, so its
    1 − α point is tan((1/2 − α) π); with two, F(t) = 1/2 + t / (2 √(2 + t²)),
    so its 1 − α point is c √(2 / (1 − c²)) with c = 1 − 2α (0.9 √(2 / 0.19)
    at α = 0.05). The (1 − 2α) interval is the mean ± that point × se."""
    # n = 2: d = [0, 2], mean 1, sd √2, se 1; ε = 2 gives t = 3 and t = −1.
    res = tost_paired([2.0, 4.0], [2.0, 2.0], margin=2.0)
    assert (res.mean_diff, res.df) == (1.0, 1)
    assert (res.se, res.t_lower, res.t_upper) == (pytest.approx(1.0), pytest.approx(3.0),
                                                  pytest.approx(-1.0))
    assert res.p_lower == pytest.approx(0.5 - math.atan(3.0) / math.pi, rel=1e-12)
    assert res.p_upper == pytest.approx(0.25, rel=1e-12)
    assert res.p_value == res.p_upper and not res.equivalent
    t95 = math.tan(0.45 * math.pi)
    assert (res.ci_low, res.ci_high) == (pytest.approx(1.0 - t95), pytest.approx(1.0 + t95))
    for alpha in (0.025, 0.1):
        res = tost_paired([2.0, 4.0], [2.0, 2.0], margin=2.0, alpha=alpha)
        t = math.tan((0.5 - alpha) * math.pi)
        assert (res.ci_low, res.ci_high) == (pytest.approx(1.0 - t), pytest.approx(1.0 + t))
    # n = 3: d = [−1, 0, 1], mean 0, sd 1, se 1/√3; ε = 3 gives t = ±3√3.
    def f2(t):
        return 0.5 + t / (2.0 * math.sqrt(2.0 + t * t))
    res = tost_paired([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], margin=3.0)
    t = 3.0 * math.sqrt(3.0)
    assert res.p_lower == pytest.approx(1.0 - f2(t), rel=1e-9)
    assert res.p_upper == pytest.approx(f2(-t), rel=1e-9)
    assert res.equivalent                                 # p = 0.0175
    half = 0.9 * math.sqrt(2.0 / 0.19) / math.sqrt(3.0)
    assert (res.ci_low, res.ci_high) == (pytest.approx(-half), pytest.approx(half))
    # So at α = 0.025 too (its 95 % interval, ±2.48, lies inside ±3), but not at
    # α = 0.01, whose 98 % interval, ±4.02, reaches past ±3.
    for alpha, equivalent in ((0.025, True), (0.01, False)):
        c = 1.0 - 2.0 * alpha
        half = c * math.sqrt(2.0 / (1.0 - c * c)) / math.sqrt(3.0)
        res = tost_paired([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], margin=3.0, alpha=alpha)
        assert (res.ci_low, res.ci_high) == (pytest.approx(-half), pytest.approx(half))
        assert res.equivalent is equivalent
    # At ε = 1 the same differences are not shown equivalent: p = 1/2 − √3 / (2 √5).
    res = tost_paired([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], margin=1.0)
    assert res.p_value == pytest.approx(0.5 - math.sqrt(3.0) / (2.0 * math.sqrt(5.0)), rel=1e-9)
    assert not res.equivalent


def test_tost_is_symmetric_in_its_arms_and_strict_at_alpha():
    a, b = [0.30, 0.10, 0.25, 0.40], [0.20, 0.15, 0.20, 0.30]
    ab, ba = tost_paired(a, b, margin=0.2), tost_paired(b, a, margin=0.2)
    assert ba.mean_diff == pytest.approx(-ab.mean_diff)
    assert (ba.p_lower, ba.p_upper) == (pytest.approx(ab.p_upper), pytest.approx(ab.p_lower))
    assert ba.p_value == pytest.approx(ab.p_value) and ab.equivalent and ba.equivalent
    assert not tost_paired(a, b, margin=0.2, alpha=ab.p_value).equivalent


def test_tost_with_zero_variance_knows_the_difference_exactly():
    inside = tost_paired([1.0, 2.0, 3.0], [0.5, 1.5, 2.5], margin=1.0)   # d = 0.5 throughout
    assert (inside.se, inside.p_lower, inside.p_upper) == (0.0, 0.0, 0.0)
    assert (inside.t_lower, inside.t_upper) == (math.inf, -math.inf)
    assert inside.equivalent and (inside.ci_low, inside.ci_high) == (0.5, 0.5)
    on = tost_paired([1.0, 1.0], [0.0, 0.0], margin=1.0)                 # on the bound
    assert (on.p_lower, on.p_upper, on.equivalent) == (0.0, 1.0, False)
    assert math.isnan(on.t_upper)
    beyond = tost_paired([0.0, 0.0], [3.0, 3.0], margin=1.0)
    assert (beyond.p_lower, beyond.p_upper, beyond.t_lower) == (1.0, 0.0, -math.inf)
    assert not beyond.equivalent


@pytest.mark.parametrize("kw, match", [
    (dict(margin=0.0), "margin"), (dict(margin=-1.0), "margin"),
    (dict(margin=math.nan), "margin"), (dict(margin=math.inf), "margin"),
    (dict(margin=True), "margin"), (dict(margin="0.1"), "margin"),
    (dict(margin=0.1, alpha=0.5), "alpha"), (dict(margin=0.1, alpha=0.0), "alpha"),
])
def test_tost_refuses_a_margin_or_alpha_it_cannot_read(kw, match):
    with pytest.raises(ValueError, match=match):
        tost_paired([1.0, 2.0], [1.5, 2.5], **kw)


@pytest.mark.parametrize("a, b, match", [
    ([1.0, 2.0], [1.0], "same length"), ([1.0], [2.0], "at least 2"),
    ([1.0, math.nan], [1.0, 2.0], "finite"),
])
def test_tost_refuses_unpaired_or_incomplete_seeds(a, b, match):
    with pytest.raises(ValueError, match=match):
        tost_paired(a, b, margin=0.5)


# --------------------------------------------------------------------------- #
# 6. The trend test: Page's L
# --------------------------------------------------------------------------- #

GAMMAS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)


def _by_level(data, levels=None):
    """``{level: per-seed values}`` from an (n seeds, k levels) array."""
    levels = np.linspace(0.0, 0.99, data.shape[1]) if levels is None else levels
    return {float(g): data[:, j] for j, g in enumerate(levels)}


def _page_l(*samples, axis=-1):
    """Page's L of samples given level by level, for scipy's permutation test."""
    data = np.stack(samples, axis=-1)
    ranks = sps.rankdata(data, axis=-1)
    return (ranks * np.arange(1, data.shape[-1] + 1)).sum(axis=(-1, -2))


def test_page_counts_a_hand_worked_null():
    """Two seeds, three levels. One seed's L = r1 + 2 r2 + 3 r3 is 14 (ranks
    1 2 3), 13 (1 3 2 or 2 1 3), 11 (2 3 1 or 3 1 2) or 10 (3 2 1), so over two
    seeds the 36 equally likely L are 20: 1, 21: 4, 22: 4, 23: 4, 24: 10, 25: 4,
    26: 4, 27: 4, 28: 1. Seeds ranking 1 2 3 and 1 3 2 give L = 27: P(L ≥ 27) =
    5/36, P(L ≤ 27) = 35/36, two-sided 10/36; ρ is the mean of their Spearman ρ,
    (1 + 1/2) / 2."""
    by = {0.0: [1.0, 1.0], 0.5: [2.0, 3.0], 0.9: [3.0, 2.0]}
    res = trend_test(by, method="exact")
    assert (res.statistic, res.rho, res.n_seeds, res.method) == (27.0, pytest.approx(0.75), 2,
                                                                 "exact")
    assert res.p_value == pytest.approx(5 / 36, rel=1e-12) and not res.significant
    assert trend_test(by, alternative="decreasing").p_value == pytest.approx(35 / 36, rel=1e-12)
    assert trend_test(by, alternative="two-sided").p_value == pytest.approx(10 / 36, rel=1e-12)
    table = {20: 1, 21: 4, 22: 4, 23: 4, 24: 10, 25: 4, 26: 4, 27: 4, 28: 1}
    for one, two in itertools.product(itertools.permutations((1.0, 2.0, 3.0)), repeat=2):
        got = trend_test({0.0: [one[0], two[0]], 0.5: [one[1], two[1]], 0.9: [one[2], two[2]]},
                         n_bootstraps=10)
        l_obs = sum(j * r for j, r in enumerate(one, 1)) + sum(j * r for j, r in enumerate(two, 1))
        assert got.statistic == l_obs
        assert got.p_value == pytest.approx(
            sum(c for value, c in table.items() if value >= l_obs) / 36, rel=1e-12)


def test_page_is_scipys_exact_and_asymptotic_test_without_ties():
    rng = np.random.default_rng(5)
    for n, k in [*itertools.product((2, 5, 10), (3, 4, 6, 7)), (3, 8)]:
        data = rng.normal(0.0, 1.0, (n, k)) + rng.uniform(-0.4, 0.4) * np.arange(k)
        by = _by_level(data)
        for method in ("exact", "asymptotic"):
            up = trend_test(by, method=method, n_bootstraps=10)
            down = trend_test(by, method=method, alternative="decreasing", n_bootstraps=10)
            two = trend_test(by, method=method, alternative="two-sided", n_bootstraps=10)
            s_up = sps.page_trend_test(data, method=method)
            s_down = sps.page_trend_test(data, method=method,
                                         predicted_ranks=list(range(k, 0, -1)))
            assert (up.statistic, up.method) == (s_up.statistic, method)
            assert up.p_value == pytest.approx(s_up.pvalue, rel=1e-9)
            assert down.p_value == pytest.approx(s_down.pvalue, rel=1e-9)
            assert two.p_value == pytest.approx(min(1.0, 2.0 * min(s_up.pvalue, s_down.pvalue)),
                                                rel=1e-9)


def test_page_with_ties_is_scipys_exhaustive_within_seed_permutation_test():
    """Where seeds tie two levels the exact null keeps the ties: every ordering
    of each seed's average ranks, as scipy's permutation test enumerates them
    (``permutation_type="samples"`` permutes the values of one seed across the
    levels)."""
    rng = np.random.default_rng(9)
    for n, k in ((2, 3), (3, 3), (2, 4), (3, 4)):
        data = rng.integers(0, 3, (n, k)).astype(float)
        assert any(len(set(row)) < k for row in data)
        by = _by_level(data)
        tails = {}
        for alternative, scipy_alternative in (("increasing", "greater"),
                                               ("decreasing", "less")):
            tails[alternative] = sps.permutation_test(
                tuple(data.T), _page_l, permutation_type="samples",
                alternative=scipy_alternative, n_resamples=np.inf, vectorized=True).pvalue
            got = trend_test(by, method="exact", alternative=alternative, n_bootstraps=10)
            assert got.p_value == pytest.approx(tails[alternative], rel=1e-9)
        two = trend_test(by, method="exact", alternative="two-sided", n_bootstraps=10)
        assert two.p_value == pytest.approx(min(1.0, 2.0 * min(tails.values())), rel=1e-9)


def test_the_normal_approximation_takes_the_permutation_variance_with_ties():
    """L's null mean and variance, each seed's from all its orderings (ties
    kept), give the asymptotic test's z exactly."""
    data = np.array([[0.0, 1.0, 1.0, 2.0], [3.0, 3.0, 1.0, 0.0], [5.0, 5.0, 5.0, 6.0]])
    ranks = sps.rankdata(data, axis=1)
    mean = var = 0.0
    for row in ranks:
        values = [sum(j * r for j, r in enumerate(p, 1)) for p in itertools.permutations(row)]
        mean += float(np.mean(values))
        var += float(np.var(values))
    assert mean == pytest.approx(3 * 4 * 25 / 4)          # n k (k + 1)² / 4, ties or not
    res = trend_test(_by_level(data), method="asymptotic", n_bootstraps=10)
    z = (res.statistic - mean) / math.sqrt(var)
    assert res.p_value == pytest.approx(sps.norm.sf(z), rel=1e-12)
    down = trend_test(_by_level(data), method="asymptotic", alternative="decreasing",
                      n_bootstraps=10)
    assert down.p_value == pytest.approx(sps.norm.cdf(z), rel=1e-12)


def test_rho_is_the_mean_spearman_rho_over_the_seeds_with_a_bootstrap_over_them():
    """Study 5.5's shape, six γ and ten seeds, rising with γ."""
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, (10, 6)) + 0.4 * np.arange(6)
    res = trend_test(_by_level(data, GAMMAS))
    per_seed = [sps.spearmanr(np.arange(6), row).statistic for row in data]
    assert res.rho == pytest.approx(np.mean(per_seed), rel=1e-12)
    _, low, high = bootstrap_ci(per_seed, np.mean, n_bootstraps=2000, seed=42)
    assert (res.ci_low, res.ci_high) == (pytest.approx(low, rel=1e-12),
                                         pytest.approx(high, rel=1e-12))
    assert 0.0 < res.ci_low <= res.rho <= res.ci_high
    assert res.method == "exact" and res.significant
    assert res == trend_test(_by_level(data, GAMMAS))     # seeded: the same twice
    assert trend_test(_by_level(data, GAMMAS), seed=7).ci_low != res.ci_low
    rising = np.tile(np.arange(6.0), (4, 1))
    assert trend_test(_by_level(rising)).rho == pytest.approx(1.0)
    assert trend_test(_by_level(-rising)).rho == pytest.approx(-1.0)


def test_the_bootstrap_takes_its_draws_confidence_and_seed():
    """Each of ``n_bootstraps``, ``confidence`` and ``seed`` reaches the bootstrap
    over the seeds: the interval is ``bootstrap_ci``'s at the same three, and
    differs from the one with any of them at its default."""
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, (10, 6)) + 0.4 * np.arange(6)
    per_seed = [sps.spearmanr(np.arange(6), row).statistic for row in data]
    settings = dict(n_bootstraps=50, confidence=0.9, seed=7)
    res = trend_test(_by_level(data, GAMMAS), **settings)
    want = bootstrap_ci(per_seed, np.mean, **settings)
    assert (res.rho, res.ci_low, res.ci_high) == tuple(pytest.approx(v, rel=1e-12) for v in want)
    for default in (dict(n_bootstraps=2000), dict(confidence=0.95), dict(seed=42)):
        other = bootstrap_ci(per_seed, np.mean, **dict(settings, **default))
        assert other[1:] != pytest.approx(want[1:], rel=1e-9)
    # The p-value does not depend on them.
    assert res.p_value == trend_test(_by_level(data, GAMMAS)).p_value


def test_a_seeds_own_level_is_no_trend():
    """Ranking within each seed sets aside how well a seed does at every level."""
    rng = np.random.default_rng(4)
    data = rng.normal(0.0, 1.0, (8, 5))
    shifted = data + 100.0 * np.arange(8)[:, None]
    a, b = trend_test(_by_level(data)), trend_test(_by_level(shifted))
    assert (a.statistic, a.p_value, a.rho, a.ci_low, a.ci_high) == (
        b.statistic, b.p_value, b.rho, b.ci_low, b.ci_high)


def test_the_levels_are_read_in_ascending_order():
    rng = np.random.default_rng(6)
    data = rng.normal(0.0, 1.0, (6, 4)) + 0.5 * np.arange(4)
    ordered = {g: data[:, j] for j, g in enumerate((0.0, 0.25, 0.5, 0.9))}
    shuffled = {g: ordered[g] for g in (0.9, 0.0, 0.5, 0.25)}
    res = trend_test(ordered)
    assert trend_test(shuffled) == res
    assert res.levels == (0.0, 0.25, 0.5, 0.9)
    assert res.means == pytest.approx(tuple(data.mean(axis=0)))
    assert trend_test({1: data[:, 0], 2: data[:, 1], 3: data[:, 2]}).levels == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("alternative", TREND_ALTERNATIVES)
@pytest.mark.parametrize("method", ["exact", "asymptotic"])
def test_seeds_that_tie_every_level_show_no_trend(method, alternative):
    res = trend_test({0.0: [1.0, 2.0], 0.5: [1.0, 2.0], 0.9: [1.0, 2.0]}, method=method,
                     alternative=alternative, n_bootstraps=10)
    assert (res.statistic, res.rho, res.p_value) == (24.0, 0.0, 1.0)


def test_auto_is_exact_at_study_5_5s_size_and_asymptotic_beyond():
    """``auto`` is exact up to 8 levels and 200 seeds, both bounds inclusive, and
    asymptotic with one level or one seed more."""
    assert (EXACT_MAX_LEVELS, EXACT_MAX_SEEDS) == (8, 200)
    rng = np.random.default_rng(8)
    assert trend_test(_by_level(rng.normal(size=(10, 6)), GAMMAS)).method == "exact"
    for (n, k), method in [((3, 8), "exact"), ((4, 9), "asymptotic"),
                           ((200, 3), "exact"), ((201, 3), "asymptotic")]:
        by = _by_level(rng.normal(size=(n, k)))
        auto = trend_test(by, n_bootstraps=10)
        assert auto.method == method
        assert auto == trend_test(by, method=method, n_bootstraps=10)
    with pytest.raises(ValueError, match="at most 8 levels"):
        trend_test(_by_level(rng.normal(size=(4, 9))), method="exact")


THREE = {0.0: [1.0, 2.0], 0.5: [2.0, 3.0], 0.9: [3.0, 4.0]}


@pytest.mark.parametrize("by, kw, match", [
    ({0.0: [1.0, 2.0], 0.5: [2.0, 3.0]}, {}, "at least 3 levels"),
    ({True: [1.0, 2.0], 0.5: [2.0, 3.0], 0.9: [3.0, 4.0]}, {}, "finite numbers"),
    ({math.nan: [1.0, 2.0], 0.5: [2.0, 3.0], 0.9: [3.0, 4.0]}, {}, "finite numbers"),
    ({"g0": [1.0, 2.0], 0.5: [2.0, 3.0], 0.9: [3.0, 4.0]}, {}, "finite numbers"),
    ({0.0: [1.0, 2.0], 0.5: [2.0, 3.0], 0.9: [3.0]}, {}, "same length"),
    ({0.0: [1.0, 2.0], 0.5: [2.0, math.nan], 0.9: [3.0, 4.0]}, {}, "finite"),
    ({0.0: [1.0], 0.5: [2.0], 0.9: [3.0]}, {}, "at least 2"),
    (THREE, {"alternative": "greater"}, "alternative"),
    (THREE, {"method": "permutation"}, "method"),
])
def test_the_trend_test_refuses_what_it_cannot_read(by, kw, match):
    with pytest.raises(ValueError, match=match):
        trend_test(by, **kw)


# --------------------------------------------------------------------------- #
# 7. Imports
# --------------------------------------------------------------------------- #

def test_scoring_a_phase_5_trace_loads_no_plan_or_phase_5_module(tmp_path):
    """The scorer and the consumer stay on the recorded path: scoring an FQ and
    an E3 trace with the pair columns loads no plan module, no Phase 5 module
    and nothing of FerrySim."""
    fq = _write_trace(tmp_path, "FQ", mule_cfg=FQ_MULE, pairs=PAIR_MISSIONS)
    e3 = _write_trace(tmp_path, "E3", mule_cfg=E3_MULE, e3=E3_MISSIONS)
    code = (
        "import sys\n"
        "from experiments.analysis.traces_scorer import score_trial\n"
        "for d in sys.argv[1:]:\n"
        "    score_trial(d, pair_columns=True).to_row()\n"
        "bad = sorted(m for m in sys.modules\n"
        "             if m.startswith(('hermes.scheduler.plan', 'experiments.ferrysim'))\n"
        f"             or m.rsplit('.', 1)[-1] in {PHASE_5_MODULES!r})\n"
        "print(bad)\n"
    )
    done = subprocess.run([sys.executable, "-c", code, str(fq), str(e3)], cwd=REPO,
                          capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
