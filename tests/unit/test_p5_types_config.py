"""FeRRy Phase 5 (unit U0): the pair types, E3's protocol and the checkpoint config.

What is pinned (the Phase 5 spec: the switches table, other choices 1, 2, 6 and
11, critic A10, B7 and B11):

* **Switch values.** ``flight_slot`` gains ``pair_q`` after the recorded
  values, in the plan types and restated in the config; a pinned band refuses
  it as it refuses FX; ``contact_policy`` names E3 ``chen_dqn``.
* **The pair types** (``hermes.scheduler.plan.types``). A candidate next stop
  (:class:`StopContext`) is a remainder stop with its index and prices, or
  home with its leg alone; what the slot sees at an arrival (:class:`PairView`)
  offers the classes that reach every committed target, FX's candidates, times
  the remainder's stops, or home alone when it is empty, class-major in link
  order; its per-class tuples follow the link; a device appears once among the
  stop and the remainder. A scorer (:class:`PairScorer`) gives one finite
  number per pair. A choice (:class:`PairChoice`) falls back exactly when no
  pair fits, and then to FX's band with no reorder; its record is JSON-ready,
  holds no wall time and is read-only all the way down.
* **E3's protocol** (``hermes.scheduler.policies.next_stop``): one observation
  per remainder stop; an answer is an admissible index, or None exactly when
  no stop is admissible; Pass 1 only; no existing policy declares
  ``chooses_next_stop``, so the mule's hook runs for none of them. Numpy-free,
  no plan import, and nothing loads it on the way. Both views take a 0 energy
  reference (P_hover = 0) as none, as ``FerryRuntime.l1_state`` does.
* **Config.** The six checkpoint fields sit right before the plan fields
  (which a Phase 4 test pins last), default None, simulated-clock only, and
  neither ferry-spec nor plan fields. Every guard fires for its own reason and
  no other: each triple all or none and only beside its switch, ``pair_q`` in
  plan mode, ``chen_dqn`` on the simulated clock in legacy mode with a band and
  whole stops, the sha and the tag well formed; the arms' configs pass.
* **Old JSON and Rule 1 at the defaults.** The per-role mule JSON recorded on
  each legacy face (the afa9526 topology golden, UG4's 6e6f92d trials, UG5's
  386c275 plan arms) loads with the checkpoints unset and validates; re-run,
  the trials of the two simulated faces gain exactly the six keys at None in
  that JSON and nothing anywhere else.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import hermes.scheduler.plan as plan_pkg
import hermes.scheduler.policies as policies_pkg
from hermes.l1.mission_clock import EnergyModel
from hermes.processes import config as C
from hermes.processes.config import (
    CHECKPOINT_MULE_FIELDS,
    FERRY_SPEC_FIELDS,
    PAIR_CHECKPOINT_FIELDS,
    PLAN_MULE_FIELDS,
    POLICY_CHECKPOINT_FIELDS,
    SIM_ONLY_MULE_FIELDS,
    ClusterConfig,
    MuleConfig,
    TopologyConfig,
    TopologyValidationError,
    mule_config_errors,
    mule_config_from_json,
    mule_config_to_json,
)
from hermes.scheduler.plan import types as PT
from hermes.scheduler.plan.types import (
    PAIR_FALLBACK_MASK_EMPTY,
    ArrivalClass,
    ArrivalView,
    PairChoice,
    PairScorer,
    PairView,
    PlanOptions,
    StopContext,
    check_pair_scores,
)
from hermes.scheduler.policies.cross_heuristic import (
    CommittedSlot,
    CrossHeuristic,
    fastest_covering_class,
)
from hermes.scheduler.policies.next_stop import (
    E3Stop,
    E3View,
    NextStopPolicy,
    checked_choice,
    pass_1_only,
)
from hermes.scheduler.selector.target_selector_rl import TargetSelectorRL
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass

from tests.golden import _build_p3_sim as UG4
from tests.golden import _build_p4_plan as UG5
from tests.golden import _canon

REPO = Path(__file__).resolve().parents[2]
COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
SHA = "0123456789abcdef" * 4
PAIR = dict(pair_checkpoint="results/exp5/checkpoints/s55/main/g0.9_s1.npz",
            pair_checkpoint_sha256=SHA, pair_checkpoint_tag="main")
POLICY = dict(policy_checkpoint="results/exp5/checkpoints/s53/e3/g0.9_s1.npz",
              policy_checkpoint_sha256="f" * 64, policy_checkpoint_tag="e3")
SIX = ("pair_checkpoint", "pair_checkpoint_sha256", "pair_checkpoint_tag",
       "policy_checkpoint", "policy_checkpoint_sha256", "policy_checkpoint_tag")


def _wp(x, y, *devs, deadline=1.0e6):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=deadline)


#: At the arrival's stop (members a, b) wide reaches a only, medium and narrow
#: both, so the covering classes are medium (b-bar) and narrow.
ARRIVAL = ArrivalView(devices=("a", "b"), committed="medium", classes=(
    ArrivalClass("wide", 0, ("a",), 2.0),
    ArrivalClass("medium", 1, ("a", "b"), 5.0),
    ArrivalClass("narrow", 2, ("a", "b"), 9.0),
))
STOP_C, STOP_DE = _wp(30, 0, "c"), _wp(0, 40, "d", "e")


def _ctx(index, stop, **overrides):
    values = dict(stop=stop, index=index, travel_s=6.0 + index, pred_dwell_s=4.0,
                  pred_snr_db=(18.0, 14.0, 9.5), capped=False, exempt=False, age=1.5,
                  on_time=0.5, weight=2.0)
    values.update(overrides)
    return StopContext(**values)


STOPS = (_ctx(0, STOP_C), _ctx(1, STOP_DE, capped=True, exempt=True))


def _view(**overrides):
    values = dict(arrival=ARRIVAL, pose=(10.0, 0.0, 0.0), observed_snr_db=(12.0, 9.0, 4.0),
                  offsets_db=(0.5, -1.0, 2.0), previous_offsets_db=None, previous_age_s=None,
                  period_s=60.0, clock_s=1.0e6 + 30.0, budget_end=1.0e6 + 90.0, budget_s=90.0,
                  t_ref_s=200.0, energy_j=4000.0, energy_ref_j=None, stops=STOPS, demand=6,
                  demand_weight=9.0, cap_s=2)
    values.update(overrides)
    return PairView(**values)


def _choice(**overrides):
    values = dict(band="narrow", next_index=1, total=4, feasible=3, fallback=None, q=0.125,
                  fx_band="medium", fx_next=0,
                  record={"t_s": 1.0e6 + 30.0, "band": "narrow", "pairs": 4, "devices": ("a", "b")})
    values.update(overrides)
    return PairChoice(**values)


def _sim_mule(**kw) -> MuleConfig:
    kw.setdefault("mule_id", "m")
    kw.setdefault("mission_clock", "sim")
    kw.setdefault("trial_seed", 7)
    kw.setdefault("n_missions", 4)
    kw.setdefault("rf_range_m", 60.0)
    return MuleConfig(**kw)


def _plan_mule(**kw) -> MuleConfig:
    """Arm F's mule config: plan mode on the pilots' settings."""
    base = dict(plan_mode="ferry", contact_band="wide", t_nom_s=200.0,
                in_flight_response="replan", replan_fallback="trim",
                member_admission="subset", age_cap_missions=2)
    base.update(kw)
    return _sim_mule(**base)


def _fq(**kw) -> MuleConfig:
    """An FQ arm's mule config: F's with the pair slot and its checkpoint."""
    return _plan_mule(**dict(dict(flight_slot="pair_q", **PAIR), **kw))


def _e3(**kw) -> MuleConfig:
    """Arm E3's mule config: legacy mode on the simulated clock, a band, its checkpoint."""
    return _sim_mule(**dict(dict(contact_band="wide", contact_policy="chen_dqn", **POLICY), **kw))


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #

def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    return sorted(set(names))


def test_the_next_stop_protocol_imports_no_numpy_and_no_plan():
    """Critic B7 iii and other choices 11: the runtime imports it lazily on E3's
    path only, and E3 flies none of the plan's machinery."""
    assert _imports(REPO / "hermes/scheduler/policies/next_stop.py") == [
        "__future__", "dataclasses", "hermes.types.scheduler", "math", "numbers", "typing"]


def _fresh(code: str) -> str:
    """The last line ``code`` prints in a fresh interpreter at the repo root."""
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.strip().splitlines()[-1]


def test_nothing_on_a_recorded_path_loads_the_protocol_or_the_plan():
    """In fresh interpreters: the policies package, which every D arm loads,
    loads neither the protocol nor the plan; the protocol loads no plan
    module; the plan types, which F and FX load, load no protocol."""
    assert ".next_stop" not in _imports(REPO / "hermes/scheduler/policies/__init__.py")
    plan_or_protocol = ("sorted(m for m in sys.modules if m.startswith('hermes.scheduler.plan') "
                        "or m.endswith('next_stop'))")
    assert _fresh(f"import sys, hermes.scheduler.policies; print({plan_or_protocol})") == "[]"
    assert _fresh(f"import sys, hermes.scheduler.policies.next_stop; print({plan_or_protocol})"
                  ) == "['hermes.scheduler.policies.next_stop']"
    assert _fresh(f"import sys, hermes.scheduler.plan.types; print({plan_or_protocol})"
                  ) == "['hermes.scheduler.plan', 'hermes.scheduler.plan.types']"


# --------------------------------------------------------------------------- #
# Switch values
# --------------------------------------------------------------------------- #

def test_the_flight_slot_gains_pair_q_after_the_recorded_values():
    assert PT.FLIGHT_SLOTS == C.FLIGHT_SLOTS == ("committed", "cross_heuristic", "pair_q")
    assert PT.FLIGHT_SLOT_PAIR_Q == C.FLIGHT_SLOT_PAIR_Q == "pair_q"
    assert MuleConfig(mule_id="m").flight_slot == PT.FLIGHT_SLOTS[0] == "committed"
    assert C.CONTACT_POLICY_CHEN_DQN == "chen_dqn"
    assert plan_pkg.__all__ == PT.__all__          # the package's re-export rule holds
    assert PT.PAIR_FALLBACKS == (PAIR_FALLBACK_MASK_EMPTY,) == ("mask_empty",)


def test_the_options_take_pair_q_with_a_searched_band_and_refuse_a_pinned_one():
    """The switches table: a pinned band refuses it, as PlanOptions already
    refuses every slot that chooses the band on arrival."""
    o = PlanOptions.from_config(flight_slot="pair_q", age_cap_missions=2)
    assert o.flight_slot == "pair_q" and o.fixed_band is None
    assert PlanOptions.from_config(**o.describe()) == o
    with pytest.raises(ValueError, match="'pair_q' flight slot"):
        PlanOptions(band_class_policy="fixed:narrow", flight_slot="pair_q")
    with pytest.raises(ValueError, match="'pair_q' flight slot"):
        PlanOptions.from_config(band_class_policy="fixed:wide", flight_slot="pair_q")


# --------------------------------------------------------------------------- #
# The pair types
# --------------------------------------------------------------------------- #

def test_a_stop_context_holds_a_remainder_stop():
    ctx = _ctx(1, STOP_DE, travel_s=7, pred_dwell_s=3, pred_snr_db=[18, 14, 9.5], age=2,
               on_time=1, weight=0)
    assert ctx.stop is STOP_DE and ctx.index == 1 and not ctx.is_home
    assert (ctx.travel_s, ctx.pred_dwell_s, ctx.age, ctx.on_time, ctx.weight) == (7.0, 3.0, 2.0,
                                                                                  1.0, 0.0)
    assert ctx.pred_snr_db == (18.0, 14.0, 9.5)
    assert all(type(v) is float for v in ctx.pred_snr_db + (ctx.travel_s, ctx.age))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.index = 0


def test_home_carries_its_leg_only():
    home = StopContext.home(12.5)
    assert home.is_home and home.stop is None and home.index is None
    assert (home.travel_s, home.pred_dwell_s, home.pred_snr_db) == (12.5, 0.0, ())
    assert (home.capped, home.exempt, home.age, home.on_time, home.weight) == (
        False, False, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("overrides", [
    {"travel_s": -1.0}, {"travel_s": math.nan}, {"travel_s": True}, {"travel_s": "6"},
    {"pred_dwell_s": -0.5}, {"pred_dwell_s": math.inf},
    {"pred_snr_db": ()},                                     # a stop holds one per class
    {"pred_snr_db": "18"}, {"pred_snr_db": (18.0, math.nan, 9.0)}, {"pred_snr_db": None},
    {"capped": 1}, {"exempt": "no"},
    {"exempt": True, "capped": False},                       # exempt implies capped
    {"age": -1.0}, {"on_time": 1.5}, {"on_time": -0.1}, {"weight": -2.0},
    {"index": None},                                         # a stop has its place
    {"index": -1}, {"index": 1.0}, {"index": True},
    {"stop": "c"}, {"stop": (30.0, 0.0, 0.0)},
])
def test_a_stop_context_refuses_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _ctx(0, STOP_C, **overrides)


@pytest.mark.parametrize("overrides", [
    {"index": 0}, {"pred_dwell_s": 1.0}, {"pred_snr_db": (1.0, 2.0, 3.0)}, {"capped": True},
    {"age": 1.0}, {"on_time": 0.5}, {"weight": 1.0}, {"travel_s": -3.0},
])
def test_home_refuses_a_stops_values(overrides):
    values = dict(stop=None, index=None, travel_s=5.0, pred_dwell_s=0.0, pred_snr_db=(),
                  capped=False, exempt=False, age=0.0, on_time=0.0, weight=0.0)
    values.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        StopContext(**values)


def test_a_view_offers_the_covering_classes_times_the_remainder_class_major():
    view = _view()
    assert [c.name for c in view.covering] == ["medium", "narrow"]      # wide misses b
    assert view.pairs == (("medium", 0), ("medium", 1), ("narrow", 0), ("narrow", 1))
    assert [view.row(*pair) for pair in view.pairs] == [0, 1, 2, 3]
    assert view.remainder == (STOP_C, STOP_DE) and not view.homebound
    with pytest.raises(ValueError, match="no pair"):
        view.row("wide", 0)                                  # not covering
    with pytest.raises(ValueError, match="no pair"):
        view.row("medium", None)                             # home is not offered yet
    assert view.observed_snr_db == (12.0, 9.0, 4.0) and view.pose == (10.0, 0.0, 0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.clock_s = 0.0


def test_a_view_at_the_last_stop_offers_home_alone():
    view = _view(stops=(StopContext.home(25.0),))
    assert view.homebound and view.remainder == ()
    assert view.pairs == (("medium", None), ("narrow", None))
    assert view.row("narrow", None) == 1


def test_the_covering_classes_are_fx_s_candidates():
    """Other choices 2: a pair's band reaches every device the committed class
    reaches there, which is FX's candidate set; FX's own class is the fastest
    of them, so it is always offered (and is the empty mask's fallback)."""
    rng = random.Random(5)
    devices = tuple(f"d{i}" for i in range(6))
    names = ("wide", "medium", "narrow", "medium_wide")
    for _ in range(300):
        n_classes = rng.randint(1, 4)
        classes = tuple(
            ArrivalClass(names[i], i, tuple(d for d in devices if rng.random() < 0.6),
                         rng.choice([0.0, 1.0, 5.0, 5.0, 9.5]))
            for i in range(n_classes))
        arrival = ArrivalView(devices=devices, committed=rng.choice(classes).name,
                              classes=classes)
        view = _view(arrival=arrival, observed_snr_db=(1.0,) * n_classes,
                     offsets_db=(0.0,) * n_classes,
                     stops=(StopContext.home(10.0),))
        need = set(arrival.committed_entry.targets)
        assert arrival.committed_entry in view.covering
        assert set(view.covering) == {c for c in classes if need.issubset(c.targets)}
        fx = fastest_covering_class(arrival)
        assert fx in view.covering
        assert fx == min(view.covering, key=lambda c: (c.dwell_s, -len(c.targets),
                                                       c.name != arrival.committed, c.index))


def test_the_previous_offsets_and_their_age_come_together():
    """Critic A3: the second reading of the phase; both None at the trial's first."""
    assert _view().previous_offsets_db is None
    view = _view(previous_offsets_db=[1, 0, -1], previous_age_s=0)
    assert view.previous_offsets_db == (1.0, 0.0, -1.0) and view.previous_age_s == 0.0
    for overrides in ({"previous_offsets_db": (1.0, 0.0, -1.0)}, {"previous_age_s": 30.0}):
        with pytest.raises(ValueError, match="together"):
            _view(**overrides)


def _arrival_out_of_order():
    return ArrivalView(devices=("a", "b"), committed="medium", classes=(
        ArrivalClass("medium", 1, ("a", "b"), 5.0), ArrivalClass("wide", 0, ("a",), 2.0)))


@pytest.mark.parametrize("overrides", [
    {"arrival": "medium"},
    {"arrival": _arrival_out_of_order(), "observed_snr_db": (1.0, 2.0),
     "offsets_db": (0.0, 0.0), "stops": (StopContext.home(1.0),)},       # not in link order
    {"pose": (10.0, 0.0)}, {"pose": (10.0, math.nan, 0.0)}, {"pose": "abc"},
    {"observed_snr_db": (12.0, 9.0)}, {"observed_snr_db": (12.0, 9.0, math.inf)},
    {"offsets_db": (0.5, -1.0, 2.0, 0.0)}, {"offsets_db": None},
    {"previous_offsets_db": (1.0, 2.0), "previous_age_s": 3.0},
    {"previous_offsets_db": (1.0, 2.0, 3.0), "previous_age_s": -3.0},
    {"period_s": 0.0}, {"period_s": -60.0},
    {"clock_s": math.inf}, {"budget_end": math.nan}, {"budget_s": 0.0},
    {"t_ref_s": 0.0}, {"energy_j": -1.0},
    {"energy_ref_j": -1.0}, {"energy_ref_j": math.nan}, {"energy_ref_j": math.inf},
    {"energy_ref_j": True}, {"energy_ref_j": "5000"},
    {"stops": ()},
    {"stops": (StopContext.home(5.0), _ctx(0, STOP_C))},                 # home only alone
    {"stops": (_ctx(1, STOP_C), _ctx(0, STOP_DE))},                      # remainder order
    {"stops": (_ctx(0, STOP_C), _ctx(2, STOP_DE))},
    {"stops": (_ctx(0, STOP_C, pred_snr_db=(1.0, 2.0)),)},               # one per class
    {"stops": (_ctx(0, _wp(30, 0, "a")),)},                              # a is at the stop
    {"stops": (_ctx(0, STOP_C), _ctx(1, _wp(5, 5, "c", "f")))},          # c twice
    {"stops": ("c",)}, {"stops": None},
    {"demand": 0}, {"demand": 2.0}, {"demand_weight": -1.0},
    {"cap_s": 0}, {"cap_s": True},
])
def test_a_view_refuses_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _view(**overrides)


class _Nearest:
    """A scripted scorer: the committed class first, then the nearest stop."""

    name = "nearest"

    def score(self, view, *, mask):
        out = []
        for band, index in view.pairs:
            rank = 0 if band == view.arrival.committed else 1
            travel = 0.0 if index is None else view.stops[index].travel_s
            out.append(-(rank * 1000.0 + travel))
        return out


def test_scorers_fill_one_protocol_with_one_number_per_pair():
    view = _view()
    scorer = _Nearest()
    assert isinstance(scorer, PairScorer)
    mask = (True, False, True, True)
    scores = check_pair_scores(view, scorer.score(view, mask=mask))
    assert scores == (-6.0, -7.0, -1006.0, -1007.0)
    assert check_pair_scores(view, np.array([1, 2, 3, 4], dtype=np.float32)) == (1.0, 2.0,
                                                                                3.0, 4.0)

    class _NoName:
        def score(self, view, *, mask):
            return [0.0] * len(view.pairs)

    class _NoScore:
        name = "x"

    assert not isinstance(_NoName(), PairScorer) and not isinstance(_NoScore(), PairScorer)


@pytest.mark.parametrize("scores", [
    [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, math.nan, 3.0, 4.0],
    [1.0, 2.0, math.inf, 4.0], [True, 2.0, 3.0, 4.0], "1234", None, [1.0, "2", 3.0, 4.0],
])
def test_a_scorer_gives_one_finite_number_per_pair(scores):
    with pytest.raises((TypeError, ValueError)):
        check_pair_scores(_view(), scores)


def test_a_choice_records_its_pair_and_fx_s():
    choice = _choice()
    assert choice.pair == ("narrow", 1) and not choice.agrees_fx
    assert _choice(band="medium", next_index=0).agrees_fx
    assert choice.describe() == {"t_s": 1.0e6 + 30.0, "band": "narrow", "pairs": 4,
                                 "devices": ["a", "b"]}
    assert choice.record["devices"] == ("a", "b")
    with pytest.raises(TypeError):
        choice.record["band"] = "wide"                       # read-only
    fresh = choice.describe()
    fresh["band"] = "wide"
    fresh["devices"].append("z")
    assert choice.describe()["band"] == "narrow" and choice.describe()["devices"] == ["a", "b"]
    assert json.loads(json.dumps(choice.describe(), allow_nan=False)) == choice.describe()
    assert _choice(q=None).q is None and type(_choice(q=1).q) is float


def test_a_choice_s_record_is_read_only_all_the_way_down():
    """The record is checked once, when the choice is made, so no nested list
    or map may change after: else a wall time or a non-JSON value slips past
    the checks, and ``describe`` fails at the mission's close. ``describe``
    still gives plain lists and dicts, each call a fresh copy."""
    record = {"pairs": [{"band": "wide", "next": [0, None]}, {"band": "medium", "next": [1]}],
              "inner": {"a": 1, "deep": {"b": [2, 3]}}, "devices": ("a", "b")}
    choice = _choice(record=record)
    attempts = [
        lambda: choice.record["pairs"].append(3),
        lambda: choice.record["pairs"][0].__setitem__("wall_s", 0.1),
        lambda: choice.record["pairs"][0]["next"].append(2),
        lambda: choice.record["inner"].__setitem__("a", 2),
        lambda: choice.record["inner"].__setitem__("plan_wall_s", 0.1),
        lambda: choice.record["inner"]["deep"].__setitem__("b", object()),
        lambda: choice.record["inner"]["deep"]["b"].append(math.nan),
        lambda: choice.record["inner"].update(a=2),
        lambda: choice.record["inner"].pop("a"),
    ]
    for attempt in attempts:
        with pytest.raises((TypeError, AttributeError)):
            attempt()
    expected = {"pairs": [{"band": "wide", "next": [0, None]}, {"band": "medium", "next": [1]}],
                "inner": {"a": 1, "deep": {"b": [2, 3]}}, "devices": ["a", "b"]}
    assert choice.describe() == expected
    fresh = choice.describe()
    assert type(fresh) is dict and type(fresh["pairs"]) is list
    assert type(fresh["pairs"][0]) is dict and type(fresh["inner"]["deep"]["b"]) is list
    fresh["pairs"][0]["next"].append(7)
    fresh["inner"]["deep"]["b"].clear()
    assert choice.describe() == expected
    assert record["pairs"][0]["next"] == [0, None]           # the caller's record is copied
    record["inner"]["a"] = 99
    assert choice.describe() == expected
    assert choice == _choice(record=expected)                # lists and tuples alike


@pytest.mark.parametrize("next_index", [0, None])
def test_the_empty_mask_flies_fx_s_band_with_no_reorder(next_index):
    """Other choices 1: no pair fits, so FX's pair, recorded as mask_empty."""
    choice = _choice(band="medium", next_index=next_index, feasible=0,
                     fallback=PAIR_FALLBACK_MASK_EMPTY, fx_next=next_index)
    assert choice.fallback == "mask_empty" and choice.agrees_fx


@pytest.mark.parametrize("overrides", [
    {"feasible": 0},                                         # nothing fits: a fallback
    {"fallback": "mask_empty"},                              # pairs fit: no fallback
    {"fallback": "mask_empty", "feasible": 0},               # the fallback is FX's band
    {"fallback": "mask_empty", "feasible": 0, "band": "medium", "next_index": 1},  # no reorder
    {"fallback": "fx", "feasible": 0, "band": "medium", "next_index": 0},
    {"feasible": 5}, {"total": 0, "feasible": 0, "fallback": "mask_empty"},
    {"feasible": -1}, {"total": 4.0},
    {"band": ""}, {"band": None}, {"fx_band": 3},
    {"next_index": -1}, {"next_index": True}, {"fx_next": 1.5},
    {"q": math.nan}, {"q": "0.1"},
    {"record": None}, {"record": [("t_s", 1.0)]},
])
def test_a_choice_refuses_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _choice(**overrides)


@pytest.mark.parametrize("record", [
    {"plan_wall_s": 0.01},
    {"pairs": [{"band": "wide", "wall_s": 0.01}]},
    {"timing": {"choice_wall_s": 0.001}},
    {1: "one"}, {"q": math.nan}, {"q": math.inf}, {"x": object()}, {"x": {"a", "b"}},
])
def test_a_record_holds_json_and_no_wall_time(record):
    """Critic B12: a repeated trial records the same decisions."""
    with pytest.raises((TypeError, ValueError)):
        _choice(record=record)


# --------------------------------------------------------------------------- #
# E3's protocol
# --------------------------------------------------------------------------- #

def _e3_stop(**overrides):
    values = dict(members=2, remaining=1.0, snr_db=-3.5, reachable=0.0, dx_m=30.0, dy_m=-40.0,
                  distance_m=50.0, return_energy_j=1435.0)
    values.update(overrides)
    return E3Stop(**values)


def _e3_view(**overrides):
    values = dict(stops=(_e3_stop(), _e3_stop(members=1, dx_m=-10.0, dy_m=0.0, distance_m=10.0)),
                  band="wide", demand=6, clock_s=1.0e6 + 20.0, budget_end=1.0e6 + 60.0,
                  budget_s=60.0, energy_j=2000.0, energy_ref_j=None)
    values.update(overrides)
    return E3View(**values)


def test_e3_sees_one_observation_per_remainder_stop():
    view = _e3_view()
    assert len(view.stops) == 2 and view.stops[1].members == 1
    assert view.time_left_s == 40.0 and _e3_view(budget_end=None).time_left_s is None
    assert type(_e3_stop(snr_db=-3).snr_db) is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.band = "narrow"


@pytest.mark.parametrize("overrides", [
    {"members": 0}, {"members": True}, {"members": 1.0}, {"remaining": 1.5},
    {"remaining": -0.1}, {"reachable": 2.0}, {"snr_db": math.nan}, {"dx_m": math.inf},
    {"distance_m": -1.0}, {"return_energy_j": -5.0},
])
def test_an_e3_stop_refuses_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _e3_stop(**overrides)


@pytest.mark.parametrize("overrides", [
    {"stops": ()}, {"stops": ("c",)}, {"band": ""}, {"band": None}, {"demand": 0},
    {"clock_s": math.nan}, {"budget_end": math.inf}, {"budget_s": 0.0}, {"energy_j": -1.0},
    {"energy_ref_j": -1.0}, {"energy_ref_j": math.nan}, {"energy_ref_j": math.inf},
    {"energy_ref_j": True}, {"energy_ref_j": "5000"},
])
def test_an_e3_view_refuses_bad_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        _e3_view(**overrides)


@pytest.mark.parametrize("make", [_view, _e3_view], ids=["pair_view", "e3_view"])
def test_a_zero_energy_reference_is_none_as_l1_state_reads_it(make):
    """``FerryRuntime.l1_state``'s reference is the capacity if one is set,
    else P_hover times the budget, and ``EnergyModel`` accepts P_hover = 0
    (a capacity must be > 0), so the product can be 0; ``l1_state`` reads a 0
    as no reference (``if not e_ref``). Both views store it as None, so a
    reader that tests ``is None`` never divides by 0, where refusing it would
    stop a valid arm at its first decision."""
    energy = EnergyModel(p_hover_w=0.0)
    assert energy.capacity_j is None
    with pytest.raises(ValueError):
        EnergyModel(capacity_j=0.0)
    for ref in (energy.p_hover_w * 90.0, 0, 0.0, -0.0):
        assert make(energy_ref_j=ref).energy_ref_j is None, ref
    assert make(energy_ref_j=None).energy_ref_j is None
    kept = make(energy_ref_j=5000)
    assert kept.energy_ref_j == 5000.0 and type(kept.energy_ref_j) is float
    assert make(energy_ref_j=1.0e-9).energy_ref_j == 1.0e-9


class _StubE3:
    chooses_next_stop = True

    def next_stop(self, remainder, state, *, view, admissible, pass_kind, after_stop):
        pass_1_only(pass_kind)
        return next((i for i in range(len(remainder)) if admissible(i)), None)


def test_a_next_stop_policy_declares_the_flag_and_no_existing_policy_does():
    """The hook runs for a policy that declares ``chooses_next_stop``, read
    with ``getattr`` and default False: no D arm, slot or selector has it, so
    every recorded path flies as it did (the switches table)."""
    assert isinstance(_StubE3(), NextStopPolicy)
    assert not isinstance(CrossHeuristic(), NextStopPolicy)
    existing = [obj for obj in (getattr(policies_pkg, n) for n in policies_pkg.__all__)
                if inspect.isclass(obj) and not issubclass(obj, BaseException)]
    existing += [CommittedSlot, CrossHeuristic, TargetSelectorRL]
    assert len(existing) == 10
    assert [cls.__name__ for cls in existing
            if getattr(cls, "chooses_next_stop", False) is not False] == []
    remainder = [STOP_C, STOP_DE]
    pick = _StubE3().next_stop(remainder, None, view=_e3_view(), admissible=lambda i: i == 1,
                               pass_kind="collect", after_stop=False)
    assert checked_choice(pick, remainder, admissible=lambda i: i == 1) == 1


def test_an_answer_is_an_admissible_index_or_none_when_none_is():
    remainder = [STOP_C, STOP_DE]

    def only(*ok):
        return lambda i: i in ok

    assert checked_choice(1, remainder, admissible=only(1)) == 1
    assert checked_choice(np.int64(0), remainder) == 0 and checked_choice(None, remainder) is None
    assert checked_choice(None, remainder, admissible=only()) is None
    for bad, admissible in ((2, None), (-1, None), (True, None), (1.0, None), ("0", None),
                            (0, only(1)), (None, only(1))):
        with pytest.raises((TypeError, ValueError)):
            checked_choice(bad, remainder, admissible=admissible)


def test_the_protocol_acts_in_pass_1_only():
    """Critic B7 i: Pass 2 delivers to every slice stop in the queue's order."""
    assert pass_1_only(COLLECT) is COLLECT and pass_1_only("collect") is COLLECT
    for bad in (DELIVER, "deliver", "Collect", None, 1):
        with pytest.raises(ValueError):
            pass_1_only(bad)
    with pytest.raises(ValueError, match="Pass-1"):
        _StubE3().next_stop([STOP_C], None, view=_e3_view(stops=(_e3_stop(),)),
                            admissible=lambda i: True, pass_kind=DELIVER, after_stop=True)


# --------------------------------------------------------------------------- #
# Config: the fields
# --------------------------------------------------------------------------- #

def test_the_checkpoint_fields_come_right_before_the_plan_fields():
    """Declared before ``plan_mode``, because a Phase 4 test pins the plan
    fields as the last ones (test_p4_config_driver.py:186-187)."""
    assert CHECKPOINT_MULE_FIELDS == PAIR_CHECKPOINT_FIELDS + POLICY_CHECKPOINT_FIELDS == SIX
    names = [f.name for f in dataclasses.fields(MuleConfig)]
    n_plan = len(PLAN_MULE_FIELDS)
    assert names[-n_plan:] == list(PLAN_MULE_FIELDS)
    assert names[-n_plan - len(SIX):-n_plan] == list(SIX)
    assert names[-n_plan - len(SIX) - 1] == "rf_prior_schedule_db"


def test_the_checkpoint_fields_are_sim_only_and_neither_ferry_spec_nor_plan_fields():
    """Other choices 6: the Phase 3 and Phase 4 ``ferry_params`` keep their strings."""
    defaults = MuleConfig(mule_id="m")
    assert {f: getattr(defaults, f) for f in SIX} == dict.fromkeys(SIX)
    assert set(SIX) <= set(SIM_ONLY_MULE_FIELDS)
    assert not set(SIX) & (set(FERRY_SPEC_FIELDS) | set(PLAN_MULE_FIELDS))
    assert _fq().ferry_spec_kwargs() == _plan_mule(flight_slot="pair_q").ferry_spec_kwargs()
    assert _e3().ferry_spec_kwargs() == _sim_mule(contact_band="wide").ferry_spec_kwargs()


def _decanon(value):
    """A golden's canonical JSON (``tests/golden/_canon``) back as plain JSON."""
    if isinstance(value, dict):
        return {k: _decanon(v) for k, v in value.items() if k != _canon.TYPE_KEY}
    if isinstance(value, list):
        return [_decanon(v) for v in value]
    if isinstance(value, str) and value.startswith("f:"):
        return float(value[2:])
    return value


def _recorded_mule_jsons(face):
    """Every mule config a legacy face recorded, as plain JSON objects."""
    if face == "afa9526":
        cases = json.loads((REPO / "tests/golden/data/topology.json").read_text("utf-8"))["cases"]
        return [_decanon(m) for case in cases.values() for m in case.get("mules", {}).values()]
    module = UG4 if face == "6e6f92d" else UG5
    cases = module.load_golden()["cases"]
    return [_decanon(case["configs"]["mule-exp4-mule.json"]) for case in cases.values()]


@pytest.mark.parametrize("face", ["afa9526", "6e6f92d", "386c275"])
def test_recorded_per_role_json_loads_with_the_checkpoints_unset(face):
    """An old JSON loads at the defaults: each legacy face's recorded mule
    configs (wall clock; Phase 3's simulated clock; Phase 4's plan arms) load
    with the six fields None and pass every guard, the new ones included."""
    recorded = _recorded_mule_jsons(face)
    assert len(recorded) >= 8
    for raw in recorded:
        assert not set(SIX) & set(raw)
        cfg = mule_config_from_json(json.dumps(raw))
        assert {f: getattr(cfg, f) for f in SIX} == dict.fromkeys(SIX)
        assert mule_config_errors(cfg) == []
        now = json.loads(mule_config_to_json(cfg))
        assert set(raw) <= set(now) and set(SIX) <= set(now) - set(raw)
        if face == "386c275":
            assert set(now) - set(raw) == set(SIX)
        assert {k: now[k] for k in raw} == raw


@pytest.mark.parametrize("cfg", [_fq(), _e3()], ids=["FQ", "E3"])
def test_a_checkpointed_config_round_trips_through_json(cfg):
    assert mule_config_errors(cfg) == []
    back = mule_config_from_json(mule_config_to_json(cfg))
    assert back == cfg


# --------------------------------------------------------------------------- #
# Config: the guards, each for its own reason
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field", SIX)
def test_each_checkpoint_field_is_refused_on_the_wall_clock(field):
    errors = mule_config_errors(MuleConfig(mule_id="m", **{field: dict(PAIR, **POLICY)[field]}))
    assert len(errors) == 1 and field in errors[0] and "mission_clock='sim'" in errors[0]


def test_e3_is_refused_on_the_wall_clock():
    errors = mule_config_errors(MuleConfig(mule_id="m", contact_policy="chen_dqn"))
    assert errors == ["contact_policy='chen_dqn' (arm E3) chooses each next stop in flight on "
                      "the simulated mission clock: set mission_clock='sim'"]
    errors = mule_config_errors(MuleConfig(mule_id="m", flight_slot="pair_q", **PAIR))
    assert len(errors) == 1 and "mission_clock='sim'" in errors[0]
    assert all(name in errors[0] for name in PAIR_CHECKPOINT_FIELDS + ("flight_slot",))


@pytest.mark.parametrize("cfg", [
    _fq(),
    _fq(age_cap_missions=None),                                             # FQ on F-cap's plan
    _fq(plan_score_params={"c_cov_per_device": 0.0, "c_link": 0.0}),       # FQ-cov
    _fq(plan_score_params={"dwell_in_delta": False}),                      # FQ-dwell
    _fq(member_admission="whole", age_cap_lookahead=1, mission_budget_s=120.0),
    _fq(pair_checkpoint="C:/elsewhere/g0_s3.npz", pair_checkpoint_tag="g99"),
    _e3(), _e3(member_admission="whole", mission_budget_s=60.0),
    _e3(contact_band="narrow", policy_checkpoint_tag="E3-bootstrap_1"),
], ids=["FQ", "FQ-cap", "FQ-cov", "FQ-dwell", "FQ-whole", "FQ-g99", "E3", "E3-budget",
        "E3-narrow"])
def test_the_configs_the_learned_arms_fly_pass(cfg):
    assert mule_config_errors(cfg) == []


@pytest.mark.parametrize("cfg,match", [
    # pair_q: all three fields, well formed; a searched band; plan mode.
    (_fq(pair_checkpoint=None), "(missing: pair_checkpoint)"),
    (_fq(pair_checkpoint_sha256=None), "(missing: pair_checkpoint_sha256)"),
    (_fq(pair_checkpoint_tag=None), "(missing: pair_checkpoint_tag)"),
    (_fq(pair_checkpoint=None, pair_checkpoint_sha256=None, pair_checkpoint_tag=None),
     "flight_slot='pair_q' flies a verified checkpoint, never a random one"),
    (_fq(pair_checkpoint=""), "pair_checkpoint must be the checkpoint's path"),
    (_fq(pair_checkpoint="  "), "pair_checkpoint must be the checkpoint's path"),
    (_fq(pair_checkpoint=3), "pair_checkpoint must be the checkpoint's path"),
    (_fq(pair_checkpoint_sha256=SHA.upper()), "pair_checkpoint_sha256 must be the sha256"),
    (_fq(pair_checkpoint_tag="g0/s1"), "pair_checkpoint_tag must be a tag"),
    (_fq(band_class_policy="fixed:wide"),
     "plan options: band_class_policy 'fixed:wide' pins one class, but the 'pair_q'"),
    (_sim_mule(contact_band="wide", flight_slot="pair_q", **PAIR),
     "flight_slot: plan mode only"),
    (_sim_mule(contact_band="wide", flight_slot="pair_q"),                 # no checkpoint either:
     "flight_slot: plan mode only"),                                      # still one reason
    (_fq(**POLICY), "policy_checkpoint, policy_checkpoint_sha256, policy_checkpoint_tag: "
                    "only with contact_policy='chen_dqn'"),
    # chen_dqn: legacy mode on the simulated clock, a band, whole stops, all three fields.
    (_e3(policy_checkpoint=None), "(missing: policy_checkpoint)"),
    (_e3(policy_checkpoint_sha256="sha"), "policy_checkpoint_sha256 must be the sha256"),
    (_e3(policy_checkpoint_tag=""), "policy_checkpoint_tag must be a tag"),
    (_e3(contact_band=None), "flies the cell's one contact band: set contact_band"),
    (_e3(member_admission="subset"), "visits a stop for all its members, so it runs 'whole'"),
    (_e3(plan_mode="ferry", t_nom_s=200.0, in_flight_response="replan",
         replan_fallback="trim"), "it takes no contact_policy (got 'chen_dqn')"),
    (_e3(plan_mode="ferry", t_nom_s=200.0, in_flight_response="replan", replan_fallback="trim",
         member_admission="subset", policy_checkpoint=None, policy_checkpoint_sha256=None,
         policy_checkpoint_tag=None),                                     # nor E3's settings:
     "it takes no contact_policy (got 'chen_dqn')"),                      # still one reason
    (_e3(**PAIR), "pair_checkpoint, pair_checkpoint_sha256, pair_checkpoint_tag: only with "
                  "flight_slot='pair_q'"),
    # A field beside no switch, or another one.
    (_plan_mule(**PAIR), "only with flight_slot='pair_q'"),                       # F
    (_plan_mule(flight_slot="cross_heuristic", pair_checkpoint_tag="main"),       # FX
     "pair_checkpoint_tag: only with flight_slot='pair_q'"),
    (_sim_mule(contact_band="wide", **POLICY), "only with contact_policy='chen_dqn'"),  # H1
    (_sim_mule(contact_band="wide", contact_policy="max_aoi", policy_checkpoint="x.npz"),
     "policy_checkpoint: only with contact_policy='chen_dqn'"),                   # D1
])
def test_the_learned_fillings_refuse_what_they_cannot_fly(cfg, match):
    """The switches table and other choices 6: each mistake reads as one reason."""
    errors = mule_config_errors(cfg)
    assert len(errors) == 1 and match in errors[0], errors


@pytest.mark.parametrize("tag", ["main", "hand", "dwell", "cov", "g0", "g25", "g99", "e3",
                                 "FQ-g90", "a_b", "x" * 28, "0"])
def test_a_tag_names_a_directory_and_a_flag_token(tag):
    assert mule_config_errors(_fq(pair_checkpoint_tag=tag)) == []


@pytest.mark.parametrize("tag", ["x" * 29, ".", "..", "g0.9", "a/b", "a\\b", "a b", "a=b", "-a",
                                 "_a", "\u00e9", " main", True, 3])
def test_a_tag_refuses_what_a_path_or_a_flag_cannot_carry(tag):
    errors = mule_config_errors(_fq(pair_checkpoint_tag=tag))
    assert len(errors) == 1 and "pair_checkpoint_tag must be a tag" in errors[0]


@pytest.mark.parametrize("sha,ok", [
    (SHA, True), ("0" * 64, True), ("f" * 64, True),
    (SHA.upper(), False), ("g" * 64, False), ("a" * 63, False), ("a" * 65, False),
    (" " + SHA[1:], False), (SHA.encode("ascii"), False), (int("1" * 20), False),
])
def test_a_sha_is_hashlibs_hex_digest(sha, ok):
    errors = mule_config_errors(_e3(policy_checkpoint_sha256=sha))
    assert errors == [] if ok else (len(errors) == 1 and "must be the sha256" in errors[0])


def test_a_topology_whose_mule_misplaces_a_checkpoint_is_refused():
    topo = TopologyConfig(cluster=ClusterConfig(cluster_id="c"),
                          mules=[MuleConfig(mule_id="m", rf_range_m=60.0, **PAIR)], devices=[])
    with pytest.raises(TopologyValidationError, match="pair_checkpoint, pair_checkpoint_sha256, "
                                                      "pair_checkpoint_tag: only on the simulated"):
        topo.validate()


# --------------------------------------------------------------------------- #
# Rule 1 at the defaults: the two simulated faces, re-run
# --------------------------------------------------------------------------- #

def _per_role_additions(golden_case, case):
    """Keys the re-run trial added, per part, and the mule JSON's new values."""
    added = {part: UG4.added_keys(golden_case[part], case[part]) for part in golden_case
             if part != "inputs"}
    mule = case["configs"]["mule-exp4-mule.json"]
    return added, {f: mule.get(f) for f in SIX}


@pytest.mark.parametrize("name", UG5.TRIAL_NAMES)
def test_at_the_defaults_the_plan_arms_gain_only_the_six_keys_at_none(name):
    """UG5's oracles let an added key pass by design, so absence is pinned here:
    on Phase 4's plan arms the per-role mule JSON gains the six checkpoint keys,
    all None, and no row, event or slot call gains anything."""
    added, values = _per_role_additions(UG5.load_golden()["cases"][name], UG5.capture(name))
    assert sorted(added.pop("configs")) == sorted(f"$.mule-exp4-mule.json.{f}" for f in SIX)
    assert added == {part: [] for part in added} and set(added) == set(UG5.PARTS) - {"configs"}
    assert values == dict.fromkeys(SIX)


@pytest.mark.parametrize("name", UG4.TRIAL_NAMES)
def test_at_the_defaults_the_simulated_clock_gains_only_the_six_keys_at_none(name):
    """UG4's oracles at 6e6f92d: beyond what Phase 4 added there (its plan
    fields in the mule JSON and a D arm's ``pass_1_policy_drops``, pinned in
    test_p4_config_driver.py), Phase 5 adds the six keys at None, and nothing
    else."""
    added, values = _per_role_additions(UG4.load_golden()["cases"][name], UG4.capture(name))
    phase_4 = {f"$.mule-exp4-mule.json.{f}" for f in PLAN_MULE_FIELDS}
    assert set(added.pop("configs")) - phase_4 == {f"$.mule-exp4-mule.json.{f}" for f in SIX}
    assert [k for keys in added.values() for k in keys
            if not k.endswith(".pass_1_policy_drops")] == []
    assert values == dict.fromkeys(SIX)
