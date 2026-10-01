"""FeRRy Phase 4 (unit U0): the plan types.

Pins what the types every later Phase 4 unit shares accept, refuse and derive:
the switch values (each listing its recorded value first), the band-class
policy, the age cap's promotion rule (age >= S - L), the score settings and
their resolved constants (c2 = kappa * N_demand, c3 = c2 unless set), the
search bounds, the member fold, the per-class physics guard (critic B3), the
plan key (cap key, then V to 9 decimals, then class, then stops; critic C5),
the search result's per-class summaries (checked as the commit checks them),
the arrival view (critic B14), the options (a pinned band never pairs with the
FX slot) and the mule's setup, and the commit: its invariants (it flies the
class its policy pins, names a known search, and needs the mission round under
a cap, critic B9), its close, and a deterministic, JSON-ready describe() with
no wall time (critic B12). Also the layering: the new modules are
numpy-free and import nothing from hermes.l1, hermes.mule or experiments, and
the commit's home, hermes.types.scheduler, imports nothing from the scheduler.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes.scheduler.plan as plan_pkg
from hermes.scheduler.plan import types as plan_types
from hermes.scheduler.plan import (
    BAND_POLICY_SEARCH,
    CAP_CLOSE_REASONS,
    CAP_CROWDED,
    CAP_DROPPED_IN_FLIGHT,
    CAP_NOT_MERGED,
    CAP_PLAN_REASONS,
    CAP_REASONS,
    CAP_UNPLANNABLE,
    COVERAGE_WEIGHTS,
    FLIGHT_SLOTS,
    MEMBER_ADMISSIONS,
    PLAN_MODES,
    PLAN_SCORE_KEYS,
    REASON_PLAN,
    SEARCH_MODES,
    AgeCapSpec,
    ArrivalClass,
    ArrivalView,
    Candidate,
    CapState,
    CapViolation,
    MemberFold,
    PlanClass,
    PlanCommit,
    PlanOptions,
    PlanScoreParams,
    PlanSearchParams,
    PlanSetup,
    ScoreTerms,
    SearchResult,
    fixed_band_policy,
    parse_band_policy,
)
from hermes.scheduler.stages.s3b_feasibility import (
    REASONS,
    FeasibilityModel,
    FerryPhysics,
    FlightState,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID
import hermes.types.scheduler as sched_types

A, B, C, D, E = (DeviceID(x) for x in ("a", "b", "c", "d", "e"))
DOCK = (0.0, 0.0, 0.0)
REPO = Path(__file__).resolve().parents[2]


def _wp(devices, pos=(10.0, 0.0, 0.0), deadline=100.0):
    return ContactWaypoint(position=pos, devices=tuple(devices),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=deadline)


def _model(range_m=60.0, p_hover_w=168.5, dwell=True):
    physics = FerryPhysics(
        dock=DOCK, member_dwell_s=(lambda d, p, o: 1.0) if dwell else None,
        upload_s=lambda: 0.0, p_move_w=143.6, p_hover_w=p_hover_w, range_m=range_m,
    )
    return FeasibilityModel(cruise_speed_m_s=5.0, session_time_s=1.0, ferry=physics)


def _cls(name="wide", index=0, radius=60.0, p_hover_w=168.5):
    return PlanClass(name=name, index=index, radius_m=radius,
                     model=_model(range_m=radius, p_hover_w=p_hover_w), outage=lambda d: 0.0)


def _terms(v=-1.0, **overrides):
    values = dict(v=v, delta_s=100.0, time=0.2, coverage=5.0 / 13.0, link=0.1,
                  energy_j=1000.0, energy=0.05, served_weight=8.0, demand_weight=13.0)
    values.update(overrides)
    return ScoreTerms(**values)


def _fold(route=(), dropped=(), feasible=True):
    return MemberFold(route=tuple(route), dropped=tuple(dropped), state=FlightState(DOCK, 0.0),
                      home=50.0, feasible=feasible, energy_j=10.0)


def _cand(route=(), v=-1.0, cls=None, cap_key=()):
    return Candidate(cls=cls or _cls(), fold=_fold(route), terms=_terms(v=v), cap_key=cap_key)


STOP_AB = _wp((A, B), pos=(10.0, 0.0, 0.0))
STOP_C = _wp((C,), pos=(0.0, 30.0, 0.0), deadline=90.0)
AGES = {A: 3, B: 1, C: 4, D: 5}          # S = 3: a, c, d capped; d left out


def _commit(**overrides):
    values = dict(
        mission_round=5, band="wide", band_index=0, band_class_policy=BAND_POLICY_SEARCH,
        queue=(STOP_AB, STOP_C), demand=(A, B, C, D),
        weights={A: 3.0, B: 1.0, C: 4.0, D: 5.0}, budget_end=60.0, t_ref_s=220.0,
        score=_terms().as_dict(), constants=PlanScoreParams().constants(4),
        search_mode="exact", n_candidates=17,
        per_class=({"band": "wide", "stops": 2, "candidates": 17, "v": -1.0, "served": 3},),
        cap_s=3, cap_lookahead=0, ages=dict(AGES), capped=frozenset({A, C, D}),
        violations=(CapViolation(D, 5, CAP_CROWDED),),
    )
    values.update(overrides)
    return PlanCommit(**values)


def _closed():
    return _commit().close(
        visited={A, B},
        violations=(CapViolation(A, 3, CAP_NOT_MERGED), CapViolation(C, 4, CAP_DROPPED_IN_FLIGHT)),
    )


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #

_STDLIB = {"__future__", "collections", "dataclasses", "math", "numbers", "types", "typing"}
_FORBIDDEN = ("numpy", "hermes.l1", "hermes.mule", "hermes.mission", "experiments")


def _imports(path: Path):
    """(module, inside ``if TYPE_CHECKING``) for every import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    checking = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            checking.update(id(n) for n in ast.walk(node))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((alias.name, id(node) in checking) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            out.append((name, id(node) in checking))
    return out


def test_plan_types_import_only_the_standard_library_and_hermes_types():
    """Numpy-free, nothing from hermes.l1, the mule or experiments; S3b's types
    for the type checker only, so no import cycle can form through the stages."""
    for name, checking in _imports(REPO / "hermes/scheduler/plan/types.py"):
        assert not name.startswith(_FORBIDDEN), name
        if checking:
            assert name == "hermes.scheduler.stages.s3b_feasibility", name
        else:
            assert name.split(".")[0] in _STDLIB or name.startswith("hermes.types."), name


def test_plan_package_init_imports_only_its_types():
    for name, checking in _imports(REPO / "hermes/scheduler/plan/__init__.py"):
        assert name in ("__future__", ".types") and not checking, name


def test_the_commit_home_imports_nothing_from_the_scheduler():
    """hermes.types is the base layer every process imports: the commit type it
    now holds must not pull the scheduler, L1 or experiments in."""
    for name, _ in _imports(REPO / "hermes/types/scheduler.py"):
        assert not name.startswith(("hermes.scheduler",) + _FORBIDDEN), name


def test_the_package_re_exports_every_type_and_the_commit_names_are_one():
    assert plan_pkg.__all__ == plan_types.__all__
    for name in plan_pkg.__all__:
        assert getattr(plan_pkg, name) is getattr(plan_types, name)
    # What the commit records and checks is defined once, beside it in hermes.types.
    for name in ("CAP_REASONS", "CapViolation", "PlanCommit", "PLAN_SCORE_KEYS",
                 "BAND_POLICY_SEARCH", "BAND_POLICY_FIXED_PREFIX", "parse_band_policy",
                 "fixed_band_policy", "SEARCH_EXACT", "SEARCH_STOP_SUBSETS", "SEARCH_LOCAL",
                 "SEARCH_MODES"):
        assert getattr(plan_pkg, name) is getattr(sched_types, name)


# --------------------------------------------------------------------------- #
# Switch values and the band-class policy
# --------------------------------------------------------------------------- #

def test_switch_values_list_the_recorded_value_first():
    assert PLAN_MODES == ("legacy", "ferry")
    assert MEMBER_ADMISSIONS == ("whole", "subset")            # decision 4 (b)
    assert FLIGHT_SLOTS == ("committed", "cross_heuristic")    # decision 5
    assert COVERAGE_WEIGHTS == ("age", "uniform")              # decision 3
    assert SEARCH_MODES == ("exact", "stop_subsets", "local")
    assert CAP_REASONS == ("unplannable", "crowded", "dropped_in_flight", "not_merged")
    assert CAP_PLAN_REASONS + CAP_CLOSE_REASONS == CAP_REASONS


def test_the_plan_drop_reason_stays_out_of_the_predicates_reasons():
    assert REASON_PLAN == "plan"
    assert REASON_PLAN not in REASONS
    assert REASONS == ("overdue", "budget", "energy", "delivery")   # pinned in Phase 3


def test_band_class_policy_parses_search_and_fixed():
    assert parse_band_policy("search") is None
    assert parse_band_policy("fixed:narrow") == "narrow"
    assert parse_band_policy("fixed:medium_wide") == "medium_wide"
    assert fixed_band_policy("wide") == "fixed:wide"


@pytest.mark.parametrize("bad", [
    "", "Search", "wide", "fixed:", "fixed: wide", "fixed:wide ", "fixed:a:b", None, 3, b"search",
])
def test_band_class_policy_refuses_anything_else(bad):
    with pytest.raises(ValueError):
        parse_band_policy(bad)


def test_fixed_band_policy_refuses_a_bad_name():
    with pytest.raises(ValueError):
        fixed_band_policy("")


# --------------------------------------------------------------------------- #
# The age cap
# --------------------------------------------------------------------------- #

def test_the_cap_is_off_by_default():
    spec = AgeCapSpec()
    assert not spec.enabled and spec.threshold is None
    assert not spec.caps(0) and not spec.caps(10_000)


def test_a_device_is_capped_once_its_age_reaches_s_less_the_lookahead():
    spec = AgeCapSpec(s_missions=3)
    assert spec.threshold == 3
    assert [spec.caps(a) for a in (1, 2, 3, 4)] == [False, False, True, True]
    ahead = AgeCapSpec(s_missions=3, lookahead=1)
    assert ahead.threshold == 2
    assert [ahead.caps(a) for a in (1, 2, 3)] == [False, True, True]


def test_s_one_caps_every_device_that_has_an_age():
    """Critic A1: ages start at 1, so S = 1 caps everyone; legal for the D4 check (S - 1)."""
    assert AgeCapSpec(s_missions=1).caps(1)


@pytest.mark.parametrize("kwargs", [
    {"s_missions": 0}, {"s_missions": -1}, {"s_missions": True}, {"s_missions": 2.0},
    {"s_missions": "3"}, {"lookahead": -1}, {"lookahead": True}, {"lookahead": 1.5},
])
def test_the_cap_refuses_bad_settings(kwargs):
    with pytest.raises((TypeError, ValueError)):
        AgeCapSpec(**kwargs)


def test_caps_refuses_a_negative_or_boolean_age():
    with pytest.raises(ValueError):
        AgeCapSpec(s_missions=2).caps(-1)
    with pytest.raises(TypeError):
        AgeCapSpec(s_missions=2).caps(True)


def test_cap_state_derives_the_capped_set_from_the_ages():
    state = CapState(spec=AgeCapSpec(s_missions=3), ages=dict(AGES))
    assert state.capped == frozenset({A, C, D})
    off = CapState(spec=AgeCapSpec(), ages=dict(AGES))
    assert off.capped == frozenset() and dict(off.ages) == AGES       # ages kept for the weights
    assert list(state.ages) == [A, B, C, D]                           # input order kept
    with pytest.raises(TypeError):
        state.ages[A] = 9                                             # read only
    with pytest.raises(TypeError):
        CapState(spec=AgeCapSpec(s_missions=3), ages=dict(AGES), capped=frozenset())


@pytest.mark.parametrize("ages", [{A: -1}, {A: True}, {A: 1.0}, {"": 1}, {3: 1}])
def test_cap_state_refuses_bad_ages(ages):
    with pytest.raises((TypeError, ValueError)):
        CapState(spec=AgeCapSpec(s_missions=2), ages=ages)


def test_cap_state_needs_a_spec():
    with pytest.raises(TypeError):
        CapState(spec=3, ages={})


def test_a_cap_violation_carries_one_of_the_reasons():
    v = CapViolation(D, 5, CAP_CROWDED)
    assert v.describe() == {"device": "d", "age": 5, "reason": "crowded"}
    assert sorted([CapViolation(B, 1, CAP_NOT_MERGED), v, CapViolation(A, 2, CAP_CROWDED)]) == [
        CapViolation(A, 2, CAP_CROWDED), CapViolation(B, 1, CAP_NOT_MERGED), v,
    ]
    for bad in ((D, 5, "late"), (D, -1, CAP_CROWDED), (D, True, CAP_CROWDED), ("", 1, CAP_CROWDED)):
        with pytest.raises((TypeError, ValueError)):
            CapViolation(*bad)


# --------------------------------------------------------------------------- #
# The score settings and terms
# --------------------------------------------------------------------------- #

def test_score_defaults_are_the_spec_constants():
    """c1 = 1, kappa = 1, c3 = c2, c4 = 0.1 (spec, other choices 7; decision 2)."""
    p = PlanScoreParams()
    assert p.constants(6) == (1.0, 6.0, 6.0, 0.1)
    assert p.constants(0) == (1.0, 0.0, 0.0, 0.1)
    assert p.coverage_weights == "age" and p.dwell_in_delta is True
    assert PlanScoreParams(c_cov_per_device=0.25, c_link=2.0).constants(4) == (1.0, 1.0, 2.0, 0.1)


def test_f_cov_switches_the_coverage_and_link_terms_off():
    assert PlanScoreParams(c_cov_per_device=0, c_link=0).constants(6) == (1.0, 0.0, 0.0, 0.1)


def test_score_settings_come_from_the_config_mapping():
    assert PlanScoreParams.from_mapping(None) == PlanScoreParams()
    assert PlanScoreParams.from_mapping({}) == PlanScoreParams()
    swept = PlanScoreParams.from_mapping({"c_cov_per_device": 0.15, "c_energy": 0})
    assert swept.c_cov_per_device == 0.15 and swept.c_energy == 0.0
    assert isinstance(swept.c_energy, float)
    assert PlanScoreParams.from_mapping(swept.as_dict()) == swept
    with pytest.raises(ValueError, match="kappa"):
        PlanScoreParams.from_mapping({"kappa": 1.0})      # a misspelt key never falls back
    with pytest.raises(TypeError):
        PlanScoreParams.from_mapping([("c_time", 1.0)])


@pytest.mark.parametrize("kwargs", [
    {"c_time": -1.0}, {"c_cov_per_device": math.nan}, {"c_energy": math.inf},
    {"c_link": -0.5}, {"c_time": True}, {"c_time": "1"}, {"coverage_weights": "flat"},
    {"dwell_in_delta": 0}, {"dwell_in_delta": "false"},
])
def test_score_settings_refuse_bad_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PlanScoreParams(**kwargs)


def test_constants_refuse_a_bad_demand_size():
    with pytest.raises((TypeError, ValueError)):
        PlanScoreParams().constants(-1)
    with pytest.raises(TypeError):
        PlanScoreParams().constants(2.0)


def test_score_terms_follow_the_commit_keys():
    assert tuple(f.name for f in dataclasses.fields(ScoreTerms)) == PLAN_SCORE_KEYS
    terms = _terms(v=-3)
    assert list(terms.as_dict()) == list(PLAN_SCORE_KEYS)
    assert terms.v == -3.0 and isinstance(terms.v, float)


@pytest.mark.parametrize("overrides", [
    {"v": math.nan}, {"delta_s": -1.0}, {"time": -0.1}, {"energy_j": -1.0}, {"energy": math.inf},
    {"served_weight": -1.0}, {"demand_weight": -2.0}, {"coverage": True},
])
def test_score_terms_refuse_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _terms(**overrides)


# --------------------------------------------------------------------------- #
# The search settings, the fold, the classes and candidates
# --------------------------------------------------------------------------- #

def test_search_defaults_and_config_mapping():
    p = PlanSearchParams()
    assert (p.exact_max_devices, p.exhaustive_max_stops) == (6, 6)
    assert (p.heuristic_max_passes, p.heuristic_max_evaluations) == (50, 2000)
    assert PlanSearchParams.from_mapping({}) == p
    zero = PlanSearchParams.from_mapping({"exact_max_devices": 0, "exhaustive_max_stops": 0})
    assert PlanSearchParams.from_mapping(zero.as_dict()) == zero
    with pytest.raises(ValueError):
        PlanSearchParams.from_mapping({"max_passes": 5})


@pytest.mark.parametrize("kwargs", [
    {"exact_max_devices": -1}, {"exhaustive_max_stops": 6.0}, {"heuristic_max_passes": 0},
    {"heuristic_max_evaluations": 0}, {"heuristic_max_passes": True},
])
def test_search_settings_refuse_bad_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PlanSearchParams(**kwargs)


def test_a_member_fold_serves_the_members_of_its_route():
    fold = _fold(route=[STOP_AB], dropped=[(_wp((C,)), REASON_PLAN)])
    assert fold.served == frozenset({A, B})
    assert isinstance(fold.route, tuple) and fold.dropped == ((_wp((C,)), "plan"),)
    assert _fold(route=[], feasible=False).served == frozenset()


@pytest.mark.parametrize("kwargs", [
    {"route": [STOP_AB], "dropped": [(_wp((A,)), "budget")]},          # a in both
    {"route": [STOP_AB, _wp((B, C))]},                                  # b twice
    {"route": ["not a waypoint"]},
    {"dropped": [(_wp((C,)),)]},
    {"dropped": [(_wp((C,)), "")]},
    {"feasible": 1},
    {"home": math.nan},
    {"energy_j": -1.0},
    {"state": (0.0, 0.0)},
])
def test_a_member_fold_refuses_bad_values(kwargs):
    values = dict(route=(), dropped=(), state=FlightState(DOCK, 0.0), home=50.0,
                  feasible=True, energy_j=10.0)
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        MemberFold(**values)


def test_a_plan_class_prices_its_own_band():
    wide = _cls()
    assert wide.radius_m == 60.0 and wide.model.ferry.range_m == 60.0
    # A physics without a range is not checked against the radius.
    free = FeasibilityModel(ferry=dataclasses.replace(_model().ferry, range_m=None))
    assert PlanClass(name="wide", index=0, radius_m=60.0, model=free, outage=lambda d: 0.0)


@pytest.mark.parametrize("kwargs", [
    {"model": FeasibilityModel()},                      # legacy model: no ferry physics
    {"model": _model(dwell=False)},                     # the channel-free clock: no member dwell
    {"model": _model(range_m=120.0)},                   # another class's range (critic B3)
    {"radius_m": 0.0}, {"radius_m": math.inf}, {"index": -1}, {"index": True},
    {"name": ""}, {"outage": 0.5},
])
def test_a_plan_class_refuses_bad_values(kwargs):
    values = dict(name="wide", index=0, radius_m=60.0, model=_model(), outage=lambda d: 0.0)
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        PlanClass(**values)


def test_the_plan_key_puts_the_cap_first():
    """A plan that keeps every capped device beats any that leaves one out, whatever V."""
    keeps = _cand(v=-50.0)
    leaves = _cand(v=-0.1, cap_key=(3,))
    assert keeps.key < leaves.key


def test_the_cap_key_minimises_the_oldest_unserved_age():
    """Critic C5: (4, 4, 4) beats (5, 3); at the same oldest age, fewer wins."""
    assert _cand(cap_key=(4, 4, 4)).key < _cand(cap_key=(5, 3)).key
    assert _cand(cap_key=(4,)).key < _cand(cap_key=(4, 4)).key


def test_the_plan_key_then_takes_the_higher_v_rounded_to_nine_decimals():
    assert _cand(v=-1.0).key < _cand(v=-2.0).key
    medium = _cls(name="medium", index=1, radius=80.0)
    # V within 1e-10 is a tie: the lower class index wins.
    assert _cand(v=-1.0 - 1e-10, cls=_cls()).key < _cand(v=-1.0, cls=medium).key
    # Same V and class: the stops decide, deterministically.
    near, far = _wp((A,), pos=(1.0, 0.0, 0.0)), _wp((A,), pos=(2.0, 0.0, 0.0))
    assert _cand(route=[near]).key < _cand(route=[far]).key
    assert _cand(route=[]).key < _cand(route=[near]).key


def test_a_candidate_is_an_admitted_plan():
    with pytest.raises(ValueError):
        Candidate(cls=_cls(), fold=_fold(feasible=False), terms=_terms())
    with pytest.raises(ValueError):
        _cand(cap_key=(3, 5))                          # not largest first
    with pytest.raises((TypeError, ValueError)):
        _cand(cap_key=(-1,))
    with pytest.raises(TypeError):
        Candidate(cls="wide", fold=_fold(), terms=_terms())
    cand = _cand(route=[STOP_AB], cap_key=[4, 2])
    assert cand.cap_key == (4, 2) and cand.band == "wide" and cand.served == {A, B}


def test_a_search_result_names_its_mode():
    best = _cand()
    result = SearchResult(best=best, mode="exact", n_candidates=12,
                          per_class=[{"band": "wide", "candidates": 12}])
    assert result.per_class[0]["band"] == "wide"
    for kwargs in ({"mode": "greedy"}, {"n_candidates": 0}, {"per_class": [{"stops": 2}]},
                   {"best": "wide"}):
        values = dict(best=best, mode="exact", n_candidates=12)
        values.update(kwargs)
        with pytest.raises((TypeError, ValueError)):
            SearchResult(**values)


_BAD_SUMMARIES = [
    pytest.param([{"stops": 2}], "naming its class", id="no-band"),
    pytest.param([{"band": ""}], "naming its class", id="empty-band"),
    pytest.param([{"band": "wide", "v": math.nan}], "finite", id="not-finite"),
    pytest.param([{"band": "wide", "detail": object()}], "JSON-ready", id="not-json"),
    pytest.param([{"band": "wide", "wall_s": 0.02}], "wall", id="wall-key"),
    pytest.param([{"band": "wide", "timing": {"search_wall_s": 0.01}}], "wall",
                 id="nested-wall-key"),
    pytest.param([{"band": "wide"}, {"band": "wide", "v": -2.0}], "one summary per class",
                 id="class-twice"),
    pytest.param([{"band": "narrow"}], "no summary of the chosen class", id="chosen-missing"),
]


@pytest.mark.parametrize("per_class, reason", _BAD_SUMMARIES)
def test_the_search_result_checks_its_summaries_as_the_commit_does(per_class, reason):
    """The commit writes the search's summaries to the trace, so a summary it
    would refuse must fail U4's own result, not U5's commit mid-mission."""
    with pytest.raises((TypeError, ValueError), match=reason):
        SearchResult(best=_cand(), mode="exact", n_candidates=1, per_class=per_class)
    with pytest.raises((TypeError, ValueError), match=reason):
        _commit(per_class=per_class)


def test_the_search_results_summaries_pass_into_the_commit_unchanged():
    """U5 commits what U4 returns: checked, JSON-ready, read-only copies."""
    raw = [{"band": "wide", "stops": (1, 2), "v": -1}, {"band": "narrow", "served": 0}]
    result = SearchResult(best=_cand(), mode="stop_subsets", n_candidates=9, per_class=raw)
    described = [{"band": "wide", "stops": [1, 2], "v": -1}, {"band": "narrow", "served": 0}]
    assert [dict(s) for s in result.per_class] == described
    with pytest.raises(TypeError):
        result.per_class[0]["band"] = "narrow"
    raw[0]["band"] = "medium"                                         # a copy, not the caller's
    assert result.per_class[0]["band"] == "wide"
    commit = _commit(per_class=result.per_class, search_mode=result.mode,
                     n_candidates=result.n_candidates)
    assert commit.describe()["per_class"] == described
    assert SearchResult(best=_cand(), mode="exact", n_candidates=1).per_class == ()


# --------------------------------------------------------------------------- #
# The arrival view (FX)
# --------------------------------------------------------------------------- #

def _view(**overrides):
    values = dict(
        devices=(A, B, C), committed="medium",
        classes=(ArrivalClass("wide", 0, (A, B), 12.0),
                 ArrivalClass("medium", 1, (A, B), 9.0),
                 ArrivalClass("narrow", 2, (A, B, C), 20.0)),
    )
    values.update(overrides)
    return ArrivalView(**values)


def test_the_arrival_view_holds_every_class_at_the_stop():
    view = _view()
    assert view.committed_entry == ArrivalClass("medium", 1, (A, B), 9.0)
    assert view.entry("narrow").targets == (A, B, C)
    with pytest.raises(KeyError):
        view.entry("medium_wide")


@pytest.mark.parametrize("overrides", [
    {"committed": "medium_wide"},
    {"devices": ()},
    {"devices": (A, A, B)},
    {"devices": "abc"},
    {"classes": (ArrivalClass("wide", 0, (A, E), 1.0), ArrivalClass("medium", 1, (), 0.0))},
    {"classes": (ArrivalClass("medium", 0, (A,), 1.0), ArrivalClass("medium", 1, (), 0.0))},
    {"classes": (ArrivalClass("wide", 1, (A,), 1.0), ArrivalClass("medium", 1, (), 0.0))},
    {"classes": ("medium",)},
])
def test_the_arrival_view_refuses_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _view(**overrides)


def test_an_arrival_class_refuses_bad_values():
    for bad in (("", 0, (), 0.0), ("wide", -1, (), 0.0), ("wide", 0, (A, A), 0.0),
                ("wide", 0, (), -1.0), ("wide", 0, (), math.nan)):
        with pytest.raises((TypeError, ValueError)):
            ArrivalClass(*bad)


# --------------------------------------------------------------------------- #
# Options and the setup
# --------------------------------------------------------------------------- #

def test_the_options_default_to_arm_f():
    o = PlanOptions()
    assert (o.band_class_policy, o.member_admission, o.flight_slot) == ("search", "subset", "committed")
    assert o.fixed_band is None and not o.cap.enabled
    assert o.score == PlanScoreParams() and o.search == PlanSearchParams()


def test_the_options_come_from_and_describe_the_config_fields():
    o = PlanOptions.from_config(
        band_class_policy="fixed:narrow", member_admission="whole", flight_slot="committed",
        age_cap_missions=3, age_cap_lookahead=1, plan_score_params={"c_cov_per_device": 0.25},
        plan_search_params={"heuristic_max_passes": 10},
    )
    assert o.fixed_band == "narrow" and o.cap == AgeCapSpec(3, 1)
    described = o.describe()
    assert list(described) == [
        "band_class_policy", "flight_slot", "member_admission", "age_cap_missions",
        "age_cap_lookahead", "plan_score_params", "plan_search_params",
    ]
    assert described["plan_score_params"]["c_cov_per_device"] == 0.25
    assert PlanOptions.from_config(**described) == o
    assert json.loads(json.dumps(described)) == described
    fx = PlanOptions.from_config(flight_slot="cross_heuristic", age_cap_missions=2)   # arm FX
    assert fx.flight_slot == "cross_heuristic" and PlanOptions.from_config(**fx.describe()) == fx


def test_a_pinned_band_flies_the_committed_slot_only():
    """FB+c flies only class c (decision 7), but the FX slot switches band on
    arrival (decision 5), and FX without that switch is not FX (decision 5 (c)).
    No arm pairs them (spec, other choices 11), so the pair is refused rather
    than given a meaning of its own."""
    assert PlanOptions(band_class_policy="fixed:narrow").flight_slot == "committed"
    assert PlanOptions(flight_slot="cross_heuristic").fixed_band is None          # FX searches
    with pytest.raises(ValueError, match="cross_heuristic"):
        PlanOptions(band_class_policy="fixed:narrow", flight_slot="cross_heuristic")
    with pytest.raises(ValueError, match="cross_heuristic"):
        PlanOptions.from_config(band_class_policy="fixed:wide", flight_slot="cross_heuristic")


@pytest.mark.parametrize("kwargs", [
    {"band_class_policy": "fixed:"}, {"member_admission": "partial"},
    {"flight_slot": "pair_q"}, {"age_cap_missions": 0}, {"plan_score_params": {"c_x": 1}},
])
def test_the_options_refuse_bad_config(kwargs):
    with pytest.raises((TypeError, ValueError)):
        PlanOptions.from_config(**kwargs)


def test_the_options_refuse_wrong_types():
    with pytest.raises(TypeError):
        PlanOptions(cap=3)
    with pytest.raises(TypeError):
        PlanOptions(score={})
    with pytest.raises(TypeError):
        PlanOptions(search=None)


CLASSES = (_cls("wide", 0, 60.0), _cls("medium", 1, 80.0), _cls("narrow", 2, 120.0))


def _setup(**overrides):
    values = dict(options=PlanOptions(), classes=CLASSES, reference="wide",
                  t_ref_s=220.0, turnaround_s=30.0)
    values.update(overrides)
    return PlanSetup(**values)


def test_the_setup_searches_every_class_or_the_pinned_one():
    assert _setup().searched == CLASSES
    pinned = _setup(options=PlanOptions(band_class_policy="fixed:medium"), reference="medium")
    assert pinned.searched == (CLASSES[1],)
    assert pinned.class_named("narrow") is CLASSES[2]
    assert _setup().p_hover_w == 168.5
    with pytest.raises(KeyError):
        _setup().class_named("medium_wide")


@pytest.mark.parametrize("overrides", [
    {"options": PlanOptions(band_class_policy="fixed:medium")},     # must pin contact_band
    {"options": PlanOptions(band_class_policy="fixed:ultra"), "reference": "ultra"},
    {"reference": "medium_wide"},
    {"classes": ()},
    {"classes": (CLASSES[0], _cls("wide", 1, 80.0))},
    {"classes": (CLASSES[0], _cls("medium", 0, 80.0))},
    {"classes": (CLASSES[0], "medium")},
    {"classes": (CLASSES[0], _cls("medium", 1, 80.0, p_hover_w=150.0))},
    {"t_ref_s": 0.0}, {"t_ref_s": math.inf}, {"turnaround_s": -1.0},
    {"options": {}},
])
def test_the_setup_refuses_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _setup(**overrides)


def test_the_energy_term_needs_a_hover_power():
    still = (_cls("wide", 0, 60.0, p_hover_w=0.0),)
    with pytest.raises(ValueError):
        _setup(classes=still)
    no_energy = PlanOptions(score=PlanScoreParams(c_energy=0.0))
    assert _setup(classes=still, options=no_energy).p_hover_w == 0.0


# --------------------------------------------------------------------------- #
# The commit
# --------------------------------------------------------------------------- #

def test_a_commit_serves_the_members_of_its_stops():
    commit = _commit()
    assert commit.served == frozenset({A, B, C})
    assert not commit.closed and commit.visited is None
    assert commit.constants == (1.0, 4.0, 4.0, 0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        commit.band = "narrow"
    with pytest.raises(TypeError):
        commit.weights[A] = 0.0
    with pytest.raises(TypeError):
        commit.score["v"] = 0.0


def test_the_commit_describes_itself_deterministically():
    commit = _commit(weights={D: 5.0, C: 4.0, B: 1.0, A: 3.0})       # stored in demand order
    assert commit.describe() == {
        "mission_round": 5, "band": "wide", "band_index": 0, "band_class_policy": "search",
        "search": "exact", "candidates": 17,
        "per_class": [{"band": "wide", "stops": 2, "candidates": 17, "v": -1.0, "served": 3}],
        "budget_end": 60.0,
        "score": {
            "v": -1.0, "delta_s": 100.0, "time": 0.2, "coverage": 5.0 / 13.0, "link": 0.1,
            "energy_j": 1000.0, "energy": 0.05, "served_weight": 8.0, "demand_weight": 13.0,
            "c": [1.0, 4.0, 4.0, 0.1], "t_ref_s": 220.0,
        },
        "demand": ["a", "b", "c", "d"],
        "weights": {"a": 3.0, "b": 1.0, "c": 4.0, "d": 5.0},
        "served": ["a", "b", "c"],
        "cap": {
            "s": 3, "lookahead": 0, "ages": {"a": 3, "b": 1, "c": 4, "d": 5},
            "capped": ["a", "c", "d"],
            "violations": [{"device": "d", "age": 5, "reason": "crowded"}],
        },
        "visited": None,
    }
    assert list(commit.describe()["weights"]) == ["a", "b", "c", "d"]


def test_describe_is_json_and_holds_no_wall_time():
    commit = _closed()
    described = commit.describe()
    text = json.dumps(described, allow_nan=False)
    assert json.loads(text) == described

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert not [k for k in keys(described) if "wall" in k]
    described["per_class"][0]["band"] = "narrow"                    # a copy, not the commit
    described["cap"]["violations"].clear()
    assert commit.describe()["per_class"][0]["band"] == "wide"
    assert len(commit.describe()["cap"]["violations"]) == 3


def test_describe_does_not_depend_on_the_hash_seed():
    """Set order varies with PYTHONHASHSEED; the record must not (critic B12)."""
    code = (
        "import json, sys; sys.path.insert(0, 'tests/unit'); "
        "import test_p4_types as t; print(json.dumps(t._closed().describe()))"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
        done = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stderr
        outs.add(done.stdout)
    assert len(outs) == 1


def test_closing_records_the_visited_set_and_the_close_time_violations():
    commit = _commit()
    closed = _closed()
    assert closed.closed and closed.visited == frozenset({A, B})
    assert [(v.device, v.reason) for v in closed.violations] == [
        (D, CAP_CROWDED), (C, CAP_DROPPED_IN_FLIGHT), (A, CAP_NOT_MERGED),
    ]
    assert closed.describe()["visited"] == ["a", "b"]
    assert commit.visited is None                                    # the original is untouched
    with pytest.raises(ValueError):
        closed.close(visited={A})                                    # once only
    # Every served capped device merged: nothing to add.
    assert _commit().close(visited={A, B, C}).violations == (CapViolation(D, 5, CAP_CROWDED),)


@pytest.mark.parametrize("visited, added", [
    ({A, B}, ()),                                                    # c dropped, unreported
    ({A, B, C}, (CapViolation(C, 4, CAP_DROPPED_IN_FLIGHT),)),      # c was visited
    ({B, C}, (CapViolation(A, 3, CAP_NOT_MERGED),)),                 # a not visited: owes dropped
    ({A, B, C}, (CapViolation(B, 1, CAP_NOT_MERGED),)),              # b is not capped
    ({A, B, C}, (CapViolation(A, 2, CAP_NOT_MERGED),)),              # a's age is 3
    ({A, B, C}, (CapViolation(A, 3, CAP_CROWDED),)),                 # plan-time reason
    ({A, B, C}, (CapViolation(A, 3, CAP_NOT_MERGED), CapViolation(A, 3, CAP_NOT_MERGED))),
    ({A, B, C}, ("not_merged",)),
    ("abc", ()),                                                     # one string, not ids
])
def test_close_refuses_inconsistent_violations(visited, added):
    with pytest.raises((TypeError, ValueError)):
        _commit().close(visited=visited, violations=added)


def test_violations_are_kept_in_reason_order_then_by_device():
    """Whatever order the planner reports them in, the record is the same."""
    only_ab = _commit(queue=(STOP_AB,), violations=(
        CapViolation(D, 5, CAP_CROWDED), CapViolation(C, 4, CAP_UNPLANNABLE),
    ))
    assert [(v.device, v.reason) for v in only_ab.violations] == [
        (C, CAP_UNPLANNABLE), (D, CAP_CROWDED),
    ]
    both_crowded = _commit(queue=(STOP_AB,), violations=(
        CapViolation(D, 5, CAP_CROWDED), CapViolation(C, 4, CAP_CROWDED),
    ))
    assert [v.device for v in both_crowded.violations] == [C, D]


def test_a_commit_without_a_cap_has_no_capped_devices():
    commit = _commit(cap_s=None, capped=frozenset(), violations=())
    assert commit.describe()["cap"]["s"] is None
    assert _commit(cap_s=None, capped=frozenset(), violations=(), ages={}).ages == {}
    with pytest.raises(ValueError):
        _commit(cap_s=None)                                          # capped devices, no cap


def test_a_capped_commit_needs_its_mission_round():
    """Critic B9: an age counts the mule's missions, so a cap needs the round.
    The scheduler refuses a cap without one first (spec, other choices 1);
    the commit agrees."""
    with pytest.raises(ValueError, match="B9"):
        _commit(mission_round=None)
    uncapped = _commit(mission_round=None, cap_s=None, capped=frozenset(), violations=())
    assert uncapped.describe()["mission_round"] is None


def test_the_commit_obeys_its_band_policy_and_names_its_search():
    """FB+c flies only class c (decision 7), so a commit under fixed:<c> flies
    c; its search mode is one of the search's (spec, other choices 3)."""
    assert _commit(band_class_policy="fixed:wide").describe()["band_class_policy"] == "fixed:wide"
    for mode in SEARCH_MODES:
        assert _commit(search_mode=mode).describe()["search"] == mode
    with pytest.raises(ValueError, match="pins 'narrow'"):
        _commit(band_class_policy="fixed:narrow")
    for bad in ("bogus", "fixed:", "Search", None):
        with pytest.raises(ValueError, match="band_class_policy"):
            _commit(band_class_policy=bad)
    for bad in ("greedy", "", None):
        with pytest.raises(ValueError, match="search_mode"):
            _commit(search_mode=bad)


@pytest.mark.parametrize("overrides", [
    {"queue": (STOP_AB, _wp((C, E)))},                               # e is not demanded
    {"queue": (STOP_AB, _wp((B, C)))},                               # b served twice
    {"queue": (STOP_AB, "stop")},
    {"demand": (A, B, C, D, D)},
    {"demand": (A, B, C, "")},
    {"demand": "abcd"},                                              # one string, not ids
    {"weights": {A: 3.0, B: 1.0, C: 4.0}},
    {"weights": {A: 3.0, B: 1.0, C: 4.0, D: -5.0}},
    {"weights": {A: 3.0, B: 1.0, C: 4.0, D: math.nan}},
    {"score": {k: v for k, v in _terms().as_dict().items() if k != "link"}},
    {"score": dict(_terms().as_dict(), v=math.inf)},
    {"score": dict(_terms().as_dict(), pass_2_s=math.nan)},
    {"score": {**_terms().as_dict(), 7: 1.0}},
    {"score": dict(_terms().as_dict(), c=1.0)},                      # describe() writes c
    {"score": dict(_terms().as_dict(), t_ref_s=1.0)},
    {"constants": (1.0, 4.0, 4.0)},
    {"constants": (1.0, -4.0, 4.0, 0.1)},
    {"per_class": ({"stops": 2},)},
    {"per_class": ({"band": "narrow"},)},                            # no summary of the band
    {"per_class": ({"band": "wide", "v": math.nan},)},
    {"per_class": ({"band": "wide", "detail": object()},)},
    {"t_ref_s": 0.0},
    {"budget_end": math.inf},
    {"band": ""}, {"band_index": -1}, {"search_mode": None}, {"n_candidates": 1.5},
    {"mission_round": -1},
    {"cap_s": 0},
    {"cap_lookahead": -1},
    {"ages": {A: 3, B: 1, C: 4}},                                    # d has no age
    {"ages": {}},                                                    # a cap needs every age
    {"capped": frozenset({A, D})},                                   # c is aged 4 >= 3
    {"capped": frozenset({A, B, C, D})},                             # b is aged 1 < 3
    {"violations": ()},                                              # d left out, unreported
    {"violations": (CapViolation(D, 4, CAP_CROWDED),)},              # d's age is 5
    {"violations": (CapViolation(D, 5, CAP_CROWDED), CapViolation(C, 4, CAP_CROWDED))},
    {"violations": (CapViolation(D, 5, CAP_CROWDED), CapViolation(B, 1, CAP_CROWDED))},
    {"violations": (CapViolation(D, 5, CAP_CROWDED), CapViolation(D, 5, CAP_UNPLANNABLE))},
    {"violations": (CapViolation(D, 5, CAP_CROWDED), CapViolation(A, 3, CAP_NOT_MERGED))},
    {"violations": (CapViolation(D, 5, CAP_UNPLANNABLE), "late")},
    {"visited": {A, ""}},
])
def test_the_commit_refuses_what_breaks_its_invariants(overrides):
    with pytest.raises((TypeError, ValueError)):
        _commit(**overrides)


def test_the_commit_takes_extra_finite_score_terms():
    commit = _commit(score=dict(_terms().as_dict(), pass_2_s=40.0))
    assert list(commit.score) == list(PLAN_SCORE_KEYS) + ["pass_2_s"]
    assert commit.describe()["score"]["pass_2_s"] == 40.0


def test_the_capped_set_follows_s_less_the_lookahead():
    # S = 3, L = 2: the threshold is 1, so b (aged 1) is capped too.
    ahead = _commit(cap_lookahead=2, capped=frozenset({A, B, C, D}))
    assert ahead.capped == {A, B, C, D}
    with pytest.raises(ValueError):
        _commit(cap_lookahead=2)                                     # b left out of capped


@pytest.mark.parametrize("overrides", [
    {"score": dict(_terms().as_dict(), plan_wall_s=0.02)},
    {"per_class": ({"band": "wide", "wall_s": 0.02},)},
    {"per_class": ({"band": "wide", "timing": {"search_wall_s": 0.01}},)},
])
def test_no_wall_time_enters_the_commit(overrides):
    """Critic B12: the planner's wall time is ``plan_wall_s``, a trace field of
    its own, so a repeated trial describes the same commit."""
    with pytest.raises(ValueError, match="wall"):
        _commit(**overrides)
