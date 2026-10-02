"""FeRRy Phase 5 (unit U3): the pair slot and its scope guard
(``hermes/scheduler/policies/pair_slot.py``; ``selector/scope_guard.assert_pairs_admitted``).

What is pinned (the Phase 5 spec, unit U3, other choices 1, 2 and 8; the user's
decision 1 (a)):

* **the pass guard**: the slot decides at Pass-1 arrivals only. In Pass 2 it
  is the committed slot (index 0, b̄, no view), and Pass 1's band comes only
  with the pair, never through ``band_at_arrival``;
* **takeoff**: the plan's first stop with no fold and no score, and index 0
  after every stop, since the order is set at the arrival;
* **the scope guard** on a foreign stop or device: a class that would drop a
  committed target, a stop outside the remainder, home with stops left, a
  non-admitted device at the stop or in the remainder. The slot runs it before
  the mask and the scorer;
* **the mask**: every pair asked once, in row order, and only a bool answer
  read. The chosen pair is in the mask, or the mask is empty and the pair is
  FX's with no reorder, flagged ``mask_empty`` (a property test over random
  views, and every scorer on the real runtime and predicate). The binding
  (``bind_fits_pair``) answers for the view's pairs only;
* **the scorer's numbers**: one finite number per pair, checked before the
  pick, wherever a bad one sits (a refused row, the fallback's row);
* **tie-breaks**: class-major in link order, home alone (and last), the lowest
  row; the slot's pick is the learner's masked argmax (``pair_q``);
* **critic A7 at the last stop**, on the real runtime and U4's predicate: the
  "most devices" class whose landing overruns is masked, so ``greedy_1`` flies
  the class that lands in time; with every landing over the budget the
  fallback flies FX's pair and logs it;
* **T2**: with one class there is one row per stop; the pick is the argmax
  over the stop rows of a ``rank_contacts``-shaped sort (−score, then a
  deterministic key: here the row, the remainder's order, which the pair
  view's contract fixes); a −travel scorer reproduces FX's nearest feasible
  stop on random instances (exact distance ties aside, which ``fx_pair``
  breaks as FX does);
* **the scripted references**, each its rule inside the slot's rules:
  ``fx_pair`` flies FX's rule (its band at the arrival, its next stop by the
  departure check's fold from the service as priced) wherever a pair on FX's
  band is admitted or none is, and the record holds that pair;
  ``committed_pair`` ranks F's pair first and ``hyb`` FX's band in the plan's
  order; ``greedy_1`` against FX on constructed cases, its last tie-break
  FX's. Past the departure after the stop, on the real runtime under both
  in-flight responses, each flies the fixed arm of its name (FX, F, a fixed
  HYB) where that arm's own pair (its band, 0) is admitted, and parts from
  it only where the slot's module docstring says;
* **records**: the decision's fields, JSON-ready, with no wall time, the same
  on a repeat; ``closed_record``'s fields and refusals;
* **training**: ε-greedy over the admitted pairs only, around FX's pair in the
  reference phase; two draws per decision with an admitted pair, none on an
  empty mask or a refused score; the sink gets each mission's steps with
  their closed records, as fresh plain dicts;
* **Freeze Rule 1**: ``scope_guard.py`` differs from 386c275's by the new
  function (with its import names and a docstring paragraph);
  ``assert_candidates_admitted`` behaves as recorded; importing the selector or
  the policies package loads no plan or pair module, and the slot imports
  nothing from ``hermes.l1``, ``hermes.mule``, ``hermes.mission`` or
  ``experiments``.

The loopback missions (the reorder after the stop, the departure check
folding it, the records closed on every exit) are the supervisor's, unit U5's
``tests/integration/test_p5_pair_missions.py``.
"""

from __future__ import annotations

import ast
import collections
import dataclasses
import importlib.util
import json
import math
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hermes.l1.contact_link import CLASSES, CLASSES_WITH_10MHZ
from hermes.l1.mission_clock import SIM_EPOCH_S, MissionClock
from hermes.mule.ferry import RESPONSE_ABORT, RESPONSE_REPLAN, FerryRuntime, FerrySpec
from hermes.scheduler import FLScheduler
from hermes.scheduler.plan import PlanOptions, PlanScoreParams, PlanSetup
from hermes.scheduler.plan.types import (
    FLIGHT_SLOT_PAIR_Q,
    PAIR_FALLBACK_MASK_EMPTY,
    ArrivalClass,
    ArrivalView,
    PairChoice,
    PairScorer,
    PairView,
    StopContext,
)
from hermes.scheduler.policies.cross_heuristic import (
    CommittedSlot,
    CrossHeuristic,
    fastest_covering_class,
    moved_to_front,
    nearest_first,
)
from hermes.scheduler.policies.pair_slot import (
    CLOSE_KEYS,
    COMMITTED_PAIR,
    DECISION_KEYS,
    FX_PAIR,
    GREEDY_1,
    HOME,
    HYB,
    SCRIPTED_SCORERS,
    CommittedPairScorer,
    FXPairScorer,
    Greedy1Scorer,
    HybScorer,
    PairQSlot,
    PairStep,
    bind_fits_pair,
    closed_record,
    scripted_scorer,
)
from hermes.scheduler.selector import pair_q
from hermes.scheduler.selector.scope_guard import (
    SelectorScopeViolation,
    assert_candidates_admitted,
    assert_pairs_admitted,
)
from hermes.scheduler.stages.s3b_feasibility import (
    DEADLINE_BOUNDS,
    DEADLINE_BOUNDS_DELIVERY,
    REASON_BUDGET,
    FlightState,
)
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionPass
from hermes.types.scheduler import PlanCommit

REPO = Path(__file__).resolve().parents[2]
REF_COMMIT = "386c275"
SCOPE_GUARD = "hermes/scheduler/selector/scope_guard.py"
COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER
T0 = SIM_EPOCH_S
RF = 60.0
SPEED = 5.0                                   # the flight model's default cruise speed, m/s
DOCK = (0.0, 0.0, 0.0)
_SCORE = dict(v=-1.0, delta_s=10.0, time=0.01, coverage=0.0, link=0.0, energy_j=0.0,
              energy=0.0, served_weight=1.0, demand_weight=1.0)


def _wp(x, y, *devs, deadline=T0 + 1e4):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=float(deadline))


def _leg(a, b):
    """The leg's seconds on the flight model's metric (``FlightModel.leg_s``)."""
    return math.dist(a, b) / SPEED


def _arrival(committed="medium", entries=None, devices=("k1", "k2", "k3")):
    """``entries``: (name, targets, dwell) per class, in link order."""
    entries = BASE_CLASSES if entries is None else entries
    return ArrivalView(
        devices=tuple(DeviceID(d) for d in devices), committed=committed,
        classes=tuple(ArrivalClass(name, i, tuple(DeviceID(t) for t in targets), dwell)
                      for i, (name, targets, dwell) in enumerate(entries)))


def _view(arrival, remainder=None, *, pose=(0.0, 0.0, 0.0), clock=T0 + 100.0):
    """A pair view at ``pose``: one context per stop of ``remainder``, its leg on the
    flight model's metric, or home alone when the remainder is empty."""
    remainder = REMAINDER if remainder is None else tuple(remainder)
    n = len(arrival.classes)
    if remainder:
        stops = tuple(StopContext(stop=wp, index=i, travel_s=_leg(pose, wp.position),
                                  pred_dwell_s=4.0, pred_snr_db=(10.0,) * n, capped=False,
                                  exempt=False, age=1.0, on_time=0.5,
                                  weight=float(len(wp.devices)))
                      for i, wp in enumerate(remainder))
    else:
        stops = (StopContext.home(_leg(pose, DOCK)),)
    demand = len(arrival.devices) + sum(len(wp.devices) for wp in remainder)
    return PairView(arrival=arrival, pose=tuple(float(c) for c in pose),
                    observed_snr_db=(10.0,) * n, offsets_db=(0.0,) * n,
                    previous_offsets_db=None, previous_age_s=None, period_s=60.0,
                    clock_s=clock, budget_end=T0 + 1000.0, budget_s=1000.0, t_ref_s=200.0,
                    energy_j=0.0, energy_ref_j=None, stops=stops, demand=demand,
                    demand_weight=float(demand), cap_s=None)


def _admitted(view):
    return set(view.arrival.devices) | {d for wp in view.remainder for d in wp.devices}


def _rows(view):
    return {pair: row for row, pair in enumerate(view.pairs)}


class _Fixed:
    """A scorer with given numbers (a sequence, or a function of the view); records its calls."""

    def __init__(self, scores, *, name="fixed", q_values=False):
        self.scores, self.name, self.q_values = scores, name, q_values
        self.calls = []

    def score(self, view, *, mask):
        self.calls.append(mask)
        return self.scores(view) if callable(self.scores) else self.scores


class _Ask:
    """A ``fits_pair`` answering from ``rule(band, index)``; records each question."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = []

    def __call__(self, band, index):
        self.calls.append((band, index))
        return self.rule(band, index)


def _admit(*pairs):
    """A ``fits_pair`` that admits exactly ``pairs``."""
    allowed = set(pairs)
    return _Ask(lambda band, index: (band, index) in allowed)


def _all(view):
    return _Ask(lambda band, index: True)


def _none(view):
    return _Ask(lambda band, index: False)


def _decide(slot, view, fits, *, admitted=None):
    return slot.pair_at_arrival(view, fits_pair=fits, pass_kind=COLLECT,
                                admitted=_admitted(view) if admitted is None else admitted)


def _pick(scorer, view, fits):
    return _decide(PairQSlot(scorer), view, fits).pair


# At the stop, the pose (0, 0): b is 10 m away, c 20 m, a 30 m. Plan order a, b, c,
# so FX's nearest-first order is (1, 2, 0).
A, B, C = _wp(30, 0, "a"), _wp(0, 10, "b"), _wp(-20, 0, "c")
REMAINDER = (A, B, C)
# Every class reaches k1 and k2, the committed medium's targets, so all three
# cover: wide fastest, narrow reaching k3 too at the most dwell.
BASE_CLASSES = (("wide", ("k1", "k2"), 3.0), ("medium", ("k1", "k2"), 6.0),
                ("narrow", ("k1", "k2", "k3"), 9.0))
VIEW = _view(_arrival())
HOME_VIEW = _view(_arrival(), (), pose=(0.0, 40.0, 0.0))
STATE = FlightState((0.0, 0.0, 0.0), T0)


# --------------------------------------------------------------------------- #
# The slot's place: the pass guard and takeoff
# --------------------------------------------------------------------------- #

def test_the_slot_is_the_pair_q_filling_with_an_injected_scorer():
    scorer = FXPairScorer()
    slot = PairQSlot(scorer)
    assert slot.name == FLIGHT_SLOT_PAIR_Q == "pair_q"
    assert slot.scorer is scorer and not slot.training
    assert repr(slot) == "PairQSlot(scorer='fx_pair')"

    class NoScore:
        name = "x"

    for bad in (None, NoScore(), object(), lambda view, mask: ()):
        with pytest.raises(TypeError, match="PairScorer"):
            PairQSlot(bad)
    with pytest.raises(TypeError, match="name"):
        PairQSlot(_Fixed((), name=""))
    with pytest.raises(TypeError, match="q_values"):
        PairQSlot(_Fixed((), q_values=1))


def test_the_slot_decides_at_pass_1_arrivals_only():
    scorer, fits = _Fixed(lambda v: [0.0] * len(v.pairs)), _all(VIEW)
    slot = PairQSlot(scorer)
    assert slot.decides_at_arrival(COLLECT) and slot.decides_at_arrival("collect")
    assert slot.reads_arrival_view(COLLECT)
    assert not slot.decides_at_arrival(DELIVER) and not slot.reads_arrival_view(DELIVER)
    # Pass 2 flies b̄: no band, and the view is not read.
    assert slot.band_at_arrival(None, pass_kind=DELIVER) is None
    assert slot.band_at_arrival(object(), pass_kind="deliver") is None
    # Pass 1's band comes with the next stop, as one decision.
    with pytest.raises(RuntimeError, match="pair_at_arrival"):
        slot.band_at_arrival(VIEW.arrival, pass_kind=COLLECT)
    with pytest.raises(ValueError, match="Pass-1 arrivals only"):
        slot.pair_at_arrival(VIEW, fits_pair=fits, pass_kind=DELIVER, admitted=_admitted(VIEW))
    assert fits.calls == [] and scorer.calls == []
    for bad in (None, "both", 3):
        for call in (lambda p: slot.decides_at_arrival(p), lambda p: slot.reads_arrival_view(p),
                     lambda p: slot.band_at_arrival(None, pass_kind=p),
                     lambda p: slot.pair_at_arrival(VIEW, fits_pair=fits, pass_kind=p,
                                                    admitted=_admitted(VIEW)),
                     lambda p: slot.next_stop(REMAINDER, STATE, fits=None, pass_kind=p,
                                              after_stop=False)):
            with pytest.raises(ValueError, match="pass_kind"):
                call(bad)


def test_takeoff_flies_the_plans_first_stop_and_every_pick_is_index_0():
    """At takeoff the plan's first stop (R6), after a Pass-1 stop the stop the
    pair chose at the arrival (moved to the front by the supervisor), in Pass 2
    the queue's order: index 0 in every call, with no fold and no score, as the
    committed slot answers."""
    orders = []
    scorer = _Fixed(lambda v: [0.0] * len(v.pairs))
    slot = PairQSlot(scorer)
    for pass_kind in (COLLECT, DELIVER):
        for after in (False, True):
            for remainder in (REMAINDER, REMAINDER[:1]):
                got = slot.next_stop(remainder, STATE, fits=orders.append, pass_kind=pass_kind,
                                     after_stop=after)
                assert got == 0 == CommittedSlot().next_stop(
                    remainder, STATE, fits=None, pass_kind=pass_kind, after_stop=after)
    assert orders == [] and scorer.calls == []
    with pytest.raises(ValueError, match="none is left"):
        slot.next_stop([], STATE, fits=None, pass_kind=COLLECT, after_stop=False)
    with pytest.raises(TypeError, match="ContactWaypoints"):
        slot.next_stop([A, "b"], STATE, fits=None, pass_kind=COLLECT, after_stop=True)
    for bad in (None, 1, "yes"):
        with pytest.raises(TypeError, match="after_stop"):
            slot.next_stop(REMAINDER, STATE, fits=None, pass_kind=COLLECT, after_stop=bad)


def test_pair_at_arrival_refuses_what_is_no_view_or_no_predicate():
    slot = PairQSlot(FXPairScorer())
    with pytest.raises(TypeError, match="PairView"):
        slot.pair_at_arrival(VIEW.arrival, fits_pair=_all(VIEW), pass_kind=COLLECT,
                             admitted=_admitted(VIEW))
    with pytest.raises(TypeError, match="fits_pair"):
        slot.pair_at_arrival(VIEW, fits_pair=None, pass_kind=COLLECT, admitted=_admitted(VIEW))


# --------------------------------------------------------------------------- #
# The scope guard
# --------------------------------------------------------------------------- #

def _guard(pairs, *, remainder=REMAINDER, classes=("wide", "medium", "narrow"),
           admitted=("k1", "k2", "k3", "a", "b", "c"), serving=("k1", "k2", "k3")):
    return assert_pairs_admitted(pairs, remainder=remainder, classes=classes,
                                 admitted=[DeviceID(d) for d in admitted],
                                 serving=[DeviceID(d) for d in serving])


def test_the_scope_guard_admits_the_pairs_within_the_plan():
    assert _guard(VIEW.pairs) is None
    assert _guard([("wide", 2), ("narrow", 0)]) is None
    assert _guard([]) is None
    assert _guard([("wide", None), ("medium", None)], remainder=()) is None
    # Without a served stop, only the remainder's devices are read.
    assert assert_pairs_admitted([("wide", 0)], remainder=[A], classes=["wide"],
                                 admitted=["a"]) is None


_FOREIGN = {
    "stop_past_the_remainder": (dict(pairs=[("wide", 3)]), "cannot fly to stop 3"),
    "negative_stop": (dict(pairs=[("wide", -1)]), "cannot fly to stop -1"),
    "bool_for_a_stop": (dict(pairs=[("wide", True)]), "cannot fly to stop True"),
    "float_for_a_stop": (dict(pairs=[("wide", 1.0)]), "cannot fly to stop 1.0"),
    "home_with_stops_left": (dict(pairs=[("wide", 0), ("wide", None)]),
                             r"cannot fly home with 3 stop\(s\) of the plan left"),
    "stop_when_none_is_left": (dict(pairs=[("wide", 0)], remainder=()), "cannot fly to stop 0"),
    "class_dropping_a_committed_target": (dict(pairs=[("wide", 0), ("tiny", 1)]),
                                          "cannot serve the stop on 'tiny'"),
    "foreign_device_in_the_remainder": (
        dict(pairs=[("wide", 0)], remainder=(A, B, _wp(5, 5, "ghost"))),
        r"non-admitted devices: \['ghost'\]"),
    "foreign_device_at_the_stop": (dict(pairs=[("wide", 0)], serving=("k1", "k9")),
                                   r"non-admitted devices: \['k9'\]"),
    "device_the_plan_dropped": (dict(pairs=[("wide", 0)], admitted=("k1", "k2", "k3", "a", "c")),
                                r"non-admitted devices: \['b'\]"),
}


@pytest.mark.parametrize("case", list(_FOREIGN))
def test_the_scope_guard_refuses_a_foreign_stop_class_or_device(case):
    kwargs, match = _FOREIGN[case]
    with pytest.raises(SelectorScopeViolation, match=match):
        _guard(**kwargs)


def test_the_slot_runs_the_scope_guard_before_the_mask_and_the_scorer():
    for admitted in (_admitted(VIEW) - {"b"}, _admitted(VIEW) - {"k3"}, set()):
        scorer, fits = _Fixed(lambda v: [0.0] * len(v.pairs)), _all(VIEW)
        with pytest.raises(SelectorScopeViolation, match="non-admitted"):
            _decide(PairQSlot(scorer), VIEW, fits, admitted=admitted)
        assert fits.calls == [] and scorer.calls == []
    # A device inserted by the beacon hook is admitted when the supervisor says so.
    insert = _wp(50, 50, "ins")
    view = _view(_arrival(), REMAINDER + (insert,))
    assert _decide(PairQSlot(FXPairScorer()), view, _all(view)).pair == ("wide", 1)
    with pytest.raises(SelectorScopeViolation, match=r"\['ins'\]"):
        _decide(PairQSlot(FXPairScorer()), view, _all(view), admitted=_admitted(VIEW))


@dataclasses.dataclass(frozen=True)
class _LeakyView(PairView):
    """A view whose pair list also offers ``leak``: what the guard is there to stop."""

    leak: tuple = ()

    @property
    def pairs(self):
        return super().pairs + self.leak


@pytest.mark.parametrize("leak, match", [
    (("wide", 0), "cannot serve the stop on 'wide'"),
    (("narrow", None), "cannot fly home"),
    (("narrow", 3), "cannot fly to stop 3"),
], ids=["class_missing_a_committed_target", "home_with_stops_left", "stop_past_the_remainder"])
def test_the_slot_holds_the_pairs_to_the_covering_classes_and_the_remainder(leak, match):
    """The guard checks the pairs against the classes that reach every committed
    target and against the remainder, not against what the view lists: a view
    that offered more is refused before anything is asked or scored."""
    base = _view(_arrival("narrow"))                     # only narrow reaches k3
    assert [c.name for c in base.covering] == ["narrow"]
    view = _LeakyView(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                      leak=(leak,))
    scorer, fits = _Fixed(lambda v: [0.0] * len(v.pairs)), _all(view)
    with pytest.raises(SelectorScopeViolation, match=match):
        _decide(PairQSlot(scorer), view, fits)
    assert fits.calls == [] and scorer.calls == []


# --------------------------------------------------------------------------- #
# The mask and the fallback
# --------------------------------------------------------------------------- #

def test_the_mask_asks_every_pair_once_in_row_order_and_the_scorer_sees_it():
    fits = _admit(("wide", 2), ("narrow", 0))
    scorer = _Fixed(lambda v: [0.0] * len(v.pairs))
    choice = _decide(PairQSlot(scorer), VIEW, fits)
    assert fits.calls == list(VIEW.pairs)
    assert scorer.calls == [tuple(p in {("wide", 2), ("narrow", 0)} for p in VIEW.pairs)]
    assert (choice.pair, choice.total, choice.feasible) == (("wide", 2), 9, 2)


@pytest.mark.parametrize("answer", [1, 0, None, "yes", object()],
                         ids=["one", "zero", "none", "string", "object"])
def test_the_mask_reads_only_a_bool(answer):
    """A fold result passed whole would be truthy and admit every pair."""
    with pytest.raises(TypeError, match="must answer a bool"):
        _decide(PairQSlot(FXPairScorer()), VIEW, lambda band, index: answer)


def test_an_empty_mask_flies_fxs_pair_with_no_reorder_and_logs_it():
    """The scorer prefers narrow and the farthest stop; nothing fits; the mule
    flies FX's pair, its fastest covering class, with the plan's next stop."""
    prefer = _Fixed(lambda v: [float(r) for r in range(len(v.pairs))], q_values=True)
    choice = _decide(PairQSlot(prefer), VIEW, _none(VIEW))
    assert choice.pair == ("wide", 0) == (fastest_covering_class(VIEW.arrival).name, 0)
    assert (choice.fallback, choice.feasible, choice.total) == (PAIR_FALLBACK_MASK_EMPTY, 0, 9)
    assert (choice.fx_band, choice.fx_next, choice.agrees_fx) == ("wide", 0, True)
    assert choice.q == 0.0                               # the score of the pair flown
    rec = choice.describe()
    assert rec["fallback"] == "mask_empty" and rec["admitted_pairs"] == [] and rec["agrees_fx"]
    assert prefer.calls == [(False,) * 9]                # the scorer was still asked
    # At the last stop: FX's class, home.
    choice = _decide(PairQSlot(prefer), HOME_VIEW, _none(HOME_VIEW))
    assert choice.pair == ("wide", None) and choice.fallback == PAIR_FALLBACK_MASK_EMPTY
    assert choice.describe()["next"] == HOME


#: (the scorer's numbers on VIEW's nine pairs, the pairs admitted, the error, its message).
_BAD_SCORES = {
    "nan_in_a_refused_row": ([0.0] * 5 + [math.nan] + [0.0] * 3, [("wide", 1)], ValueError,
                             "finite"),
    "inf_in_an_admitted_row": ([0.0, math.inf] + [0.0] * 7, [("wide", 1)], ValueError, "finite"),
    "nan_at_the_fallbacks_row": ([math.nan] + [0.0] * 8, [], ValueError, "finite"),
    "too_few": ([0.0] * 8, [("wide", 1)], ValueError, "one number per pair"),
    "too_many": ([0.0] * 10, [("wide", 1)], ValueError, "one number per pair"),
    "too_few_on_an_empty_mask": ([0.0] * 8, [], ValueError, "one number per pair"),
    "a_bool": ([True] + [0.0] * 8, [("wide", 1)], TypeError, "number"),
    "a_string": ("0" * 9, [("wide", 1)], TypeError, "sequence of numbers"),
    "none": (None, [], TypeError, "sequence of numbers"),
}


@pytest.mark.parametrize("case", list(_BAD_SCORES))
def test_the_slot_refuses_numbers_that_are_not_one_finite_number_per_pair(case):
    """``check_pair_scores`` at the slot, for the learned adapter's numbers:
    wherever a bad number sits, a refused row or the fallback's row, which
    neither the pick nor the fallback reads, and whether or not the numbers are
    Q values, the decision is refused before anything is drawn or kept, so a
    NaN never reaches a record or a trainer's step."""
    scores, admitted, error, match = _BAD_SCORES[case]
    for q_values in (False, True):
        rng, ref = random.Random(4), random.Random(4)
        sink = _Sink()
        slot = PairQSlot(_Fixed(scores, q_values=q_values))
        slot.attach_trainer(epsilon=0.5, rng=rng, sink=sink)
        with pytest.raises(error, match=match):
            _decide(slot, VIEW, _admit(*admitted))
        assert rng.getstate() == ref.getstate()                 # nothing drawn
        slot.close_mission([])                                  # and no step kept
        assert sink.calls == [((), ())]


def _random_view(rng, *, n_classes=None, n_stops=None):
    """A random arrival: 1 to 4 classes with random targets (the committed one
    at random), 0 to 5 stops left (home when none), the stop anywhere."""
    n_classes = rng.randint(1, 4) if n_classes is None else n_classes
    devices = [f"k{i}" for i in range(rng.randint(1, 4))]
    entries = []
    for i in range(n_classes):
        targets = tuple(d for d in devices if rng.random() < 0.6)
        dwell = 0.0 if not targets else rng.choice([rng.uniform(0.5, 40.0), 5.0, 5.0])
        entries.append((f"c{i}", targets, dwell))
    arrival = _arrival(rng.choice(entries)[0], tuple(entries), tuple(devices))
    n_stops = rng.randint(0, 5) if n_stops is None else n_stops
    remainder = tuple(_wp(rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0),
                          *[f"s{j}m{m}" for m in range(rng.randint(1, 3))])
                      for j in range(n_stops))
    pose = (rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0), 0.0)
    return _view(arrival, remainder, pose=pose, clock=T0 + rng.uniform(0.0, 500.0))


def test_the_chosen_pair_is_in_the_mask_or_flagged_on_random_views():
    """Random views, masks and scores (with ties): an admitted pair, the one
    ranked highest with ties to the lowest row, or with nothing admitted FX's
    pair with no reorder, flagged; the record lists exactly the admitted pairs."""
    rng = random.Random("u3-mask")
    seen = collections.Counter()
    for _ in range(2000):
        view = _random_view(rng)
        pairs, rows = view.pairs, _rows(view)
        p = rng.choice([0.0, 0.2, 0.5, 1.0])
        mask = tuple(rng.random() < p for _ in pairs)
        scores = [rng.choice([0.0, 1.0, 2.0, rng.uniform(-5.0, 5.0)]) for _ in pairs]
        choice = _decide(PairQSlot(_Fixed(scores)), view,
                         _Ask(lambda band, index: mask[rows[(band, index)]]))
        rec = choice.describe()
        assert rec["admitted_pairs"] == [list(pair) for pair, ok in zip(pairs, mask) if ok]
        assert (choice.total, choice.feasible) == (len(pairs), sum(mask))
        assert (rec["band"], rec["next_index"], rec["fx_band"], rec["fx_next"]) == (
            choice.band, choice.next_index, choice.fx_band, choice.fx_next)
        assert rec["agrees_fx"] == choice.agrees_fx == (choice.pair == (
            choice.fx_band, choice.fx_next))
        seen["agrees with FX" if choice.agrees_fx else "departs from FX"] += 1
        seen["same band, another stop"] += (
            choice.band == choice.fx_band and choice.next_index != choice.fx_next)
        fx = fastest_covering_class(view.arrival).name
        if any(mask):
            row = rows[choice.pair]
            assert mask[row] and choice.fallback is None
            best = max(scores[r] for r in range(len(pairs)) if mask[r])
            assert row == min(r for r in range(len(pairs)) if mask[r] and scores[r] == best)
            seen["admitted"] += 1
            seen["tie decided"] += sum(mask[r] and scores[r] == best for r in rows.values()) > 1
        else:
            assert choice.fallback == PAIR_FALLBACK_MASK_EMPTY
            assert choice.pair == (fx, None if view.homebound else 0)
            seen["empty"] += 1
        seen["home" if view.homebound else "stops"] += 1
    assert min(seen[k] for k in ("admitted", "tie decided", "empty", "home", "stops",
                                 "agrees with FX", "departs from FX")) > 100, seen
    assert seen["same band, another stop"] > 50, seen


# --------------------------------------------------------------------------- #
# Tie-breaks
# --------------------------------------------------------------------------- #

def test_ties_go_to_the_lowest_row_class_major_in_link_order():
    """Rows run class by class in link order, then through the remainder in its
    order: under a constant score a later stop on an earlier class beats the
    first stop on a later class."""
    flat = _Fixed(lambda v: [0.0] * len(v.pairs))
    assert VIEW.pairs == tuple((c, i) for c in ("wide", "medium", "narrow") for i in range(3))
    assert _pick(flat, VIEW, _all(VIEW)) == ("wide", 0)
    assert _pick(flat, VIEW, _admit(("wide", 2), ("medium", 0), ("narrow", 1))) == ("wide", 2)
    assert _pick(flat, VIEW, _admit(("medium", 1), ("narrow", 0))) == ("medium", 1)
    # Ties only among the best: the lowest of their rows.
    best = _Fixed([0.0, 5.0, 1.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0])
    assert _pick(best, VIEW, _all(VIEW)) == ("wide", 1)
    assert _pick(best, VIEW, _admit(("medium", 2), ("medium", 1), ("narrow", 0))) == (
        "medium", 1)
    # The rows follow the link's order, not the committed class (medium, above);
    # a class that misses a committed target offers no row at all.
    narrow = _view(_arrival("narrow"))
    assert narrow.pairs == tuple(("narrow", i) for i in range(3))
    assert _pick(flat, narrow, _all(narrow)) == ("narrow", 0)


def test_home_stands_alone_at_the_last_stop():
    """Home is offered only once no stop is left, alone, so it is the last pair
    of every class; ties go to the first covering class."""
    flat = _Fixed(lambda v: [0.0] * len(v.pairs))
    assert HOME_VIEW.homebound and HOME_VIEW.remainder == ()
    assert HOME_VIEW.pairs == (("wide", None), ("medium", None), ("narrow", None))
    assert _pick(flat, HOME_VIEW, _all(HOME_VIEW)) == ("wide", None)
    assert _pick(flat, HOME_VIEW, _admit(("narrow", None))) == ("narrow", None)
    with pytest.raises(ValueError, match="home is offered only when the remainder is empty"):
        dataclasses.replace(VIEW, stops=VIEW.stops + HOME_VIEW.stops)


def test_the_slots_pick_is_the_learners_masked_argmax():
    """One rule for the flight and the learner's double-DQN target
    (``pair_q.masked_argmax``): the highest score, ties to the lowest row."""
    rng = random.Random("u3-argmax")
    for _ in range(500):
        view = _random_view(rng, n_stops=rng.randint(1, 5))
        mask = tuple(rng.random() < 0.6 for _ in view.pairs)
        if not any(mask):
            continue
        scores = [rng.choice([0.0, 1.0, -1.0]) for _ in view.pairs]
        rows = _rows(view)
        choice = _decide(PairQSlot(_Fixed(scores)), view,
                         _Ask(lambda band, index: mask[rows[(band, index)]]))
        assert rows[choice.pair] == pair_q.masked_argmax(scores, mask)


# --------------------------------------------------------------------------- #
# T2: one band
# --------------------------------------------------------------------------- #

def _one_class_view(rng, n_stops):
    arrival = _arrival("wide", (("wide", ("k1",), rng.uniform(0.5, 20.0)),), ("k1",))
    remainder = tuple(_wp(rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0), f"s{j}")
                      for j in range(n_stops))
    return _view(arrival, remainder, pose=(rng.uniform(-150.0, 150.0),
                                           rng.uniform(-150.0, 150.0), 0.0))


def test_t2_with_one_class_there_is_one_row_per_stop():
    rng = random.Random("u3-t2-rows")
    for n_stops in range(0, 7):
        view = _one_class_view(rng, n_stops)
        expect = [("wide", i) for i in range(n_stops)] or [("wide", None)]
        assert list(view.pairs) == expect
        choice = _decide(PairQSlot(FXPairScorer()), view, _all(view))
        assert choice.band == "wide" and choice.total == len(expect)


def test_t2_the_pick_is_the_argmax_over_the_stop_rows_with_rank_contacts_shaped_ties():
    """``rank_contacts`` sorts its contacts by (−Q, then a deterministic key) and
    the head is flown (``target_selector_rl.py``). With one class the slot does
    the same over the stop rows: its key is the row, the remainder's order (the
    pair view's tie rule), where ``rank_contacts`` keys on (position, devices)."""
    rng = random.Random("u3-t2-ties")
    ties = 0
    for _ in range(600):
        n = rng.randint(1, 6)
        view = _one_class_view(rng, n)
        q = [rng.choice([0.0, 1.0, 2.0]) for _ in range(n)]
        mask = [rng.random() < 0.7 for _ in range(n)]
        ranking = sorted(range(n), key=lambda i: (-q[i], i))
        head = next((i for i in ranking if mask[i]), 0)      # nothing admitted: FX's, no reorder
        choice = _decide(PairQSlot(_Fixed(q)), view, _Ask(lambda band, index: mask[index]))
        assert choice.pair == ("wide", head)
        ties += sum(mask[i] and q[i] == q[head] for i in range(n)) > 1
    assert ties > 100


def test_t2_a_minus_travel_scorer_reproduces_fxs_nearest_feasible_stop():
    """A scorer of −travel over the stop rows flies FX's next stop: the nearest
    stop whose move to the front keeps the rest feasible, else the plan's next
    (``CrossHeuristic.next_stop`` at the departure, ``fits`` answering as the
    mask does for that stop)."""
    rng = random.Random("u3-t2-travel")
    seen = collections.Counter()
    minus_travel = _Fixed(lambda v: [-ctx.travel_s for ctx in v.stops])
    for _ in range(1000):
        view = _one_class_view(rng, rng.randint(1, 6))
        remainder = list(view.remainder)
        fits_at = {wp: rng.random() < 0.5 for wp in remainder}
        choice = _decide(PairQSlot(minus_travel), view,
                         _Ask(lambda band, index: fits_at[remainder[index]]))
        fx = CrossHeuristic().next_stop(remainder, FlightState(view.pose, view.clock_s),
                                        fits=lambda order: fits_at[order[0]],
                                        pass_kind=COLLECT, after_stop=True)
        assert choice.pair == ("wide", fx) == (choice.fx_band, choice.fx_next)
        seen["passes nearer stops" if fx != nearest_first(remainder, view.pose)[0] else
             "nearest"] += 1
        seen["none fits" if not any(fits_at.values()) else "some fit"] += 1
    assert min(seen.values()) > 50, seen


def test_exact_distance_ties_fall_as_fx_breaks_them_under_fx_pair_and_greedy_1():
    """Two stops 10 m away, the plan listing (10, 0) before (0, 10): FX tries
    (0, 10) first (ties to the position); ``fx_pair`` follows it, and so does
    ``greedy_1``, whose last tie-break is FX's nearest-first order (its own
    keys tie: the same class, so the same targets and dwell, and the same
    travel), while a plain −travel score ties and the slot takes the lower
    row, the plan's order."""
    x, y = _wp(10, 0, "x"), _wp(0, 10, "y")
    view = _view(_arrival(), (x, y))
    assert nearest_first([x, y], view.pose) == [1, 0]
    assert view.stops[0].travel_s == view.stops[1].travel_s
    assert _pick(FXPairScorer(), view, _all(view)) == ("wide", 1)
    assert _pick(Greedy1Scorer(), view, _all(view)) == ("narrow", 1)     # narrow reaches k3
    assert _pick(Greedy1Scorer(), view, _admit(("medium", 0), ("medium", 1))) == ("medium", 1)
    minus_travel = _Fixed(lambda v: [-ctx.travel_s for ctx in v.stops] * len(v.covering))
    assert _pick(minus_travel, view, _all(view)) == ("wide", 0)


# --------------------------------------------------------------------------- #
# The scripted references
# --------------------------------------------------------------------------- #

def test_the_scripted_scorers_are_pair_scorers_named_by_their_reference():
    assert SCRIPTED_SCORERS == (FX_PAIR, COMMITTED_PAIR, HYB, GREEDY_1) == (
        "fx_pair", "committed_pair", "hyb", "greedy_1")
    kinds = (FXPairScorer, CommittedPairScorer, HybScorer, Greedy1Scorer)
    rng = random.Random("u3-scorers")
    for name, kind in zip(SCRIPTED_SCORERS, kinds):
        scorer = scripted_scorer(name)
        assert isinstance(scorer, kind) and isinstance(scorer, PairScorer)
        assert scorer.name == name and scorer.q_values is False
        for _ in range(200):
            view = _random_view(rng)
            got = scorer.score(view, mask=(True,) * len(view.pairs))
            # One finite number per pair, every pair ranked apart, the mask unread.
            assert len(got) == len(view.pairs) and all(math.isfinite(s) for s in got)
            assert sorted(got) == [float(-r) for r in reversed(range(len(got)))]
            assert scorer.score(view, mask=(False,) * len(view.pairs)) == got
        with pytest.raises(TypeError, match="PairView"):
            scorer.score(VIEW.arrival, mask=())
    for bad in ("FX", "", None, ["fx_pair"]):
        with pytest.raises(ValueError, match="scripted scorer"):
            scripted_scorer(bad)


def test_fx_pairs_order_starts_with_fxs_band_and_its_nearest_stop():
    rng = random.Random("u3-fx-order")
    for _ in range(500):
        view = _random_view(rng)
        scores = FXPairScorer().score(view, mask=())
        top = view.pairs[max(range(len(scores)), key=scores.__getitem__)]
        nxt = None if view.homebound else nearest_first(view.remainder, view.pose)[0]
        assert top == (fastest_covering_class(view.arrival).name, nxt)


def test_where_the_mask_refuses_fxs_band_fx_pair_flies_the_next_band_and_records_fxs():
    """The one place ``fx_pair`` departs from the FX arm: every stop refused on
    FX's band while another band's pair fits. The slot chooses within the mask,
    so it flies the next band in FX's order, to that band's nearest stop that
    fits; the record keeps FX's own pair, the plan's next stop on its band (FX's
    next-stop rule when nothing fits), and the disagreement."""
    choice = _decide(PairQSlot(FXPairScorer()), VIEW,
                     _admit(("medium", 0), ("medium", 2), ("narrow", 1)))
    assert choice.pair == ("medium", 2) and choice.fallback is None
    assert (choice.fx_band, choice.fx_next, choice.agrees_fx) == ("wide", 0, False)


def test_committed_pair_and_hyb_rank_fs_and_hybs_pairs_first_within_the_slots_rules():
    """``committed_pair`` ranks F's pair (b̄, 0) first and ``hyb`` HYB's (FX's
    band, 0), and each flies it where it is admitted. Elsewhere the slot's
    rules part them from the fixed F and HYB, which fly that pair whatever the
    mask says: where it is refused they take the nearest thing to it that
    fits, a later stop (a reorder) or another band, and on an empty mask the
    slot flies FX's pair, which is not F's when FX's band is not b̄. Past the
    departure: ``test_each_reference_flies_its_fixed_arm_where_that_arms_pair_fits``."""
    flown = {name: _pick(scripted_scorer(name), VIEW, _all(VIEW)) for name in SCRIPTED_SCORERS}
    assert flown == {"fx_pair": ("wide", 1), "committed_pair": ("medium", 0),
                     "hyb": ("wide", 0), "greedy_1": ("narrow", 1)}
    # F's pair refused: b̄ on the plan's earliest stop that fits; b̄ refused
    # everywhere: the fastest other covering class, in the plan's order.
    assert _pick(CommittedPairScorer(), VIEW,
                 _admit(("medium", 1), ("medium", 2), ("wide", 0))) == ("medium", 1)
    assert _pick(CommittedPairScorer(), VIEW,
                 _admit(("narrow", 0), ("wide", 2), ("wide", 1))) == ("wide", 1)
    assert _pick(HybScorer(), VIEW, _admit(("wide", 2), ("wide", 1), ("medium", 0))) == (
        "wide", 1)
    assert _pick(HybScorer(), VIEW, _admit(("narrow", 0), ("medium", 2))) == ("medium", 2)
    # An empty mask flies FX's pair whatever the scorer: here wide, while F's b̄ is medium.
    for name in SCRIPTED_SCORERS:
        choice = _decide(PairQSlot(scripted_scorer(name)), VIEW, _none(VIEW))
        assert choice.pair == ("wide", 0) != (VIEW.arrival.committed, 0)
        assert choice.fallback == PAIR_FALLBACK_MASK_EMPTY
    # At the last stop every reference flies home on its band.
    flown = {name: _pick(scripted_scorer(name), HOME_VIEW, _all(HOME_VIEW))
             for name in SCRIPTED_SCORERS}
    assert flown == {"fx_pair": ("wide", None), "committed_pair": ("medium", None),
                     "hyb": ("wide", None), "greedy_1": ("narrow", None)}


#: (classes, committed class, the pairs admitted (None: all), greedy_1's pair, FX's pair).
_GREEDY_CASES = {
    # Critic A2: a covering class reaching one more device fits; greedy_1 takes
    # it (one device outweighs any time saved), FX keeps the fastest class.
    "reach_for_dwell_that_fits": (
        (("wide", ("k1",), 2.0), ("medium", ("k1", "k2"), 8.0)), "wide", None,
        ("medium", 1), ("wide", 1)),
    # Critic A7: the class reaching more overruns, so the mask refuses it, and
    # greedy_1 flies FX's pair.
    "reach_that_overruns_is_masked": (
        (("wide", ("k1",), 2.0), ("medium", ("k1", "k2"), 8.0)), "wide",
        [("wide", 0), ("wide", 1), ("wide", 2)], ("wide", 1), ("wide", 1)),
    # The same members on every class: the least dwell plus travel is FX's pair.
    "same_members_everywhere": (
        (("wide", ("k1", "k2"), 3.0), ("medium", ("k1", "k2"), 6.0)), "medium", None,
        ("wide", 1), ("wide", 1)),
    # FX's nearest stop refused on its band: FX flies wide to c (3 + 4 s), while
    # medium to b (4 + 2 s) gets to the next arrival sooner.
    "mask_couples_band_and_stop": (
        (("wide", ("k1", "k2"), 3.0), ("medium", ("k1", "k2"), 4.0)), "medium",
        [("wide", 0), ("wide", 2), ("medium", 0), ("medium", 1), ("medium", 2)],
        ("medium", 1), ("wide", 2)),
    # wide to c and medium to b tie at 7 s: FX's band order breaks the tie.
    "tie_goes_to_fxs_band_order": (
        (("wide", ("k1", "k2"), 3.0), ("medium", ("k1", "k2"), 5.0)), "medium",
        [("wide", 0), ("wide", 2), ("medium", 0), ("medium", 1), ("medium", 2)],
        ("wide", 2), ("wide", 2)),
    # Two classes reach both members: the one sooner at the next arrival.
    "most_devices_then_sooner": (
        (("wide", ("k1",), 1.0), ("medium", ("k1", "k2"), 9.0),
         ("narrow", ("k1", "k2"), 5.0)), "wide", None, ("narrow", 1), ("wide", 1)),
}


@pytest.mark.parametrize("case", list(_GREEDY_CASES))
def test_greedy_1_against_fx_on_constructed_cases(case):
    entries, committed, fits, greedy, fx = _GREEDY_CASES[case]
    view = _view(_arrival(committed, entries, ("k1", "k2")))

    def ask():
        return _all(view) if fits is None else _admit(*fits)

    assert _pick(Greedy1Scorer(), view, ask()) == greedy
    assert _pick(FXPairScorer(), view, ask()) == fx
    choice = _decide(PairQSlot(Greedy1Scorer()), view, ask())
    assert (choice.fx_band, choice.fx_next) == fx and choice.agrees_fx == (greedy == fx)


# --------------------------------------------------------------------------- #
# On the real runtime and U4's predicate
# --------------------------------------------------------------------------- #

def _plan_scheduler(rt, positions, queue, bbar, budget_end, now, deadlines):
    """A plan-mode scheduler after a hand-built commit of ``queue`` on ``bbar``."""
    demand = tuple(d for wp in queue for d in wp.devices)
    setup = PlanSetup(options=PlanOptions(), classes=rt.plan_classes(), reference="wide",
                      t_ref_s=200.0, turnaround_s=rt.spec.flight.turnaround_s)
    commit = PlanCommit(mission_round=5, band=bbar, band_index=rt.band_index,
                        band_class_policy="search", queue=tuple(queue), demand=demand,
                        weights={d: 1.0 for d in demand}, budget_end=budget_end,
                        t_ref_s=200.0, score=_SCORE,
                        constants=PlanScoreParams().constants(len(demand)),
                        search_mode="exact", n_candidates=1)
    model = rt.feasibility_model()
    model = dataclasses.replace(model, ferry=model.ferry.bind(positions))
    sch = FLScheduler(now_fn=lambda: now, feasibility_model=model, replan_fallback="trim",
                      member_admission="subset", plan_mode="ferry", plan=setup)
    sch.last_plan = commit
    sch.last_plan_deadlines = dict(deadlines)
    return sch


def _runtime_view(rt, k, rest, positions, t, *, budget_end, energy_j):
    """The pair view as the supervisor builds it from the runtime (U4's readers)."""
    prices = rt.stop_contexts(k, rest, positions, pass_kind=COLLECT)
    if rest:
        stops = tuple(StopContext(stop=wp, index=i, travel_s=p.travel_s,
                                  pred_dwell_s=p.pred_dwell_s, pred_snr_db=p.pred_snr_db,
                                  capped=False, exempt=False, age=0.0, on_time=0.5,
                                  weight=float(len(wp.devices)))
                      for i, (wp, p) in enumerate(zip(rest, prices)))
    else:
        stops = (StopContext.home(prices[0].travel_s),)
    demand = len(k.devices) + sum(len(wp.devices) for wp in rest)
    budget_s = budget_end - T0
    return PairView(arrival=rt.arrival_view(k, positions, t, pass_kind=COLLECT), pose=k.position,
                    observed_snr_db=rt.observe(k, positions, t).class_snr_db,
                    offsets_db=rt.class_offsets_db(k, positions, t), previous_offsets_db=None,
                    previous_age_s=None,
                    period_s=rt.spec.contact_channel.interference_period_s, clock_s=t,
                    budget_end=budget_end, budget_s=budget_s, t_ref_s=200.0,
                    energy_j=energy_j, energy_ref_j=rt.energy_ref_j(budget_s), stops=stops,
                    demand=demand, demand_weight=float(demand), cap_s=None)


def _a7_last_stop(margin_s):
    """Phase 4's A7 layout (``test_p4_cross_heuristic.py``): the last stop at
    (0, 40), a 10 m out and e 59 m out, at an arrival where e is below the floor
    on wide and above it on medium and narrow; the budget ends ``margin_s``
    after wide's landing."""
    clock = MissionClock()
    spec = FerrySpec.from_config(rf_range_m=RF, seed=11, contact_band="wide",
                                 payload_bytes=1_000_000)
    rt = FerryRuntime(spec, clock, rf_range_m=RF)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    chan, floor = rt.spec.contact_channel, rt.spec.link.snr_floor_db
    last = _wp(0, 40, "a", "e")
    pos = {DeviceID("a"): (10.0, 40.0, 0.0), DeviceID("e"): (0.0, 99.0, 0.0)}

    def e_on(t, band):
        return chan.snr_db(t, band, 59.0, link_key="e", stop_pos=last.position)

    t = next(T0 + 0.5 * k for k in range(1, 20_000)
             if e_on(T0 + 0.5 * k, "wide") < floor <= min(e_on(T0 + 0.5 * k, "medium"),
                                                         e_on(T0 + 0.5 * k, "narrow")))
    clock.advance(t - T0, "transit")
    arrival = rt.arrival_view(last, pos, clock(), pass_kind=COLLECT)
    home = rt.spec.flight.leg_s(last.position, rt.spec.flight.dock) + rt.predicted_upload_s()
    land = {c.name: clock() + c.dwell_s + home for c in arrival.classes}
    budget_end = land["wide"] + margin_s
    sch = _plan_scheduler(rt, pos, [last], "wide", budget_end, clock(),
                          {d: T0 + 1e4 for d in last.devices})
    state = FlightState(last.position, clock(), rt.energy_j())
    view = _runtime_view(rt, last, [], pos, clock(), budget_end=budget_end,
                         energy_j=rt.energy_j())
    fits = bind_fits_pair(sch, view, served_at=last, state=state, budget_end=budget_end)
    return sch, view, last, state, budget_end, land, fits


def test_a7_at_the_last_stop_an_overrunning_home_is_masked():
    """Critic A7 at the last Pass-1 stop, where no departure check follows: the
    "most devices" rule (greedy_1) ranks medium first, for e, but its landing
    passes the budget the committed wide meets with a second to spare, so the
    mask refuses it (``budget``) and the slot flies wide home."""
    sch, view, last, state, budget_end, land, fits = _a7_last_stop(1.0)
    assert view.homebound and view.arrival.committed == "wide"
    assert [(c.name, c.targets) for c in view.covering] == [
        ("wide", ("a",)), ("medium", ("a", "e")), ("narrow", ("a", "e"))]
    scores = Greedy1Scorer().score(view, mask=())
    assert view.pairs[max(range(3), key=scores.__getitem__)] == ("medium", None)
    for entry in view.covering:
        res = sch.fits_after_service([], served_at=last, state=state, dwell_s=entry.dwell_s,
                                     collected=entry.targets, budget_end=budget_end)
        assert res.ok == (land[entry.name] <= budget_end) == (entry.name == "wide")
        assert res.rejected == () if res.ok else res.rejected[0][1] == REASON_BUDGET
    choice = _decide(PairQSlot(Greedy1Scorer()), view, fits)
    assert choice.pair == ("wide", None) and choice.fallback is None
    rec = choice.describe()
    assert rec["admitted_pairs"] == [["wide", None]] and rec["feasible"] == 1
    assert rec["next"] == HOME and rec["agrees_fx"]


def test_a7_when_every_landing_overruns_the_fallback_flies_fxs_pair_and_logs_it():
    """The Phase 5 design's finding 3: FX's own pair already overruns at the
    observed rate at about a quarter of N = 6's last arrivals. The mask is then
    empty, and the mule flies FX's pair, home on wide, recorded ``mask_empty``."""
    sch, view, last, state, budget_end, land, fits = _a7_last_stop(-1.0)
    choice = _decide(PairQSlot(Greedy1Scorer()), view, fits)
    assert choice.pair == ("wide", None) and choice.fallback == PAIR_FALLBACK_MASK_EMPTY
    assert choice.feasible == 0 and choice.describe()["admitted_pairs"] == []


def _field(rng, *, n_stops=(1, 6)):
    """A random Pass-1 arrival on the real runtime, with a plan-mode scheduler
    after a hand-built commit: ``n_stops`` (at least, at most) stops, the one
    served included, 3 or 4 classes, b̄ and the deadline bounds at random."""
    classes = rng.choice([CLASSES, CLASSES_WITH_10MHZ])
    spec = FerrySpec.from_config(rf_range_m=RF, seed=rng.randrange(10 ** 6), contact_band="wide",
                                 band_classes=classes, payload_bytes=1_000_000,
                                 contact_regime=rng.choice(["clean", "jittery"]),
                                 deadline_bounds=rng.choice(DEADLINE_BOUNDS))
    rt = FerryRuntime(spec, None, rf_range_m=RF)
    rt.set_payload(theta_bytes=18_756, synth_bytes=64)
    rt.set_band(rng.choice(classes))
    t = T0 + rng.uniform(0.0, 120.0)
    positions, deadlines, stops = {}, {}, []
    for _ in range(rng.randint(*n_stops)):
        cx, cy = rng.uniform(-150.0, 150.0), rng.uniform(-150.0, 150.0)
        members = []
        for _ in range(rng.randint(1, 3)):
            r, a = rt.range_planar_m * math.sqrt(rng.random()), rng.uniform(0.0, 2.0 * math.pi)
            did = DeviceID(f"d{len(positions)}")
            positions[did] = (cx + r * math.cos(a), cy + r * math.sin(a), 0.0)
            deadlines[did] = t + rng.uniform(20.0, 600.0)
            members.append(did)
        stops.append(ContactWaypoint(position=(cx, cy, 0.0), devices=tuple(members),
                                     bucket=Bucket.SCHEDULED_THIS_ROUND,
                                     deadline_ts=min(deadlines[d] for d in members)))
    k, rest = stops[0], stops[1:]
    budget_end = t + rng.uniform(60.0, 600.0)
    sch = _plan_scheduler(rt, positions, stops, rt.band, budget_end, t, deadlines)
    state = FlightState(k.position, t, rng.uniform(0.0, 3e4))
    view = _runtime_view(rt, k, rest, positions, t, budget_end=budget_end,
                         energy_j=state.energy_j)
    return rt, sch, k, rest, state, budget_end, view


class _Spy:
    """A scheduler that answers ``fits_after_service`` from a rule and records each call."""

    def __init__(self, rule):
        self.rule, self.calls = rule, []

    def fits_after_service(self, rest, **kwargs):
        self.calls.append((list(rest), kwargs))
        return types.SimpleNamespace(ok=self.rule(list(rest), kwargs))


def test_bind_fits_pair_asks_the_scheduler_once_per_pair_with_the_pairs_service_and_order():
    """The spec's mask (other choices 2) as the supervisor binds it (U4's
    hand-off): the stop served on the pair's class, priced at that class's dwell
    at the arrival SNR with its targets collected, then the remainder with the
    pair's stop first and the rest in plan order (home: none), from the
    arrival's state, under Pass 1's budget end and the mule's deadline record."""
    k = _wp(0, 0, "k1", "k2", "k3")
    record = {DeviceID("ins"): T0 + 50.0}
    spy = _Spy(lambda rest, kwargs: rest[0] != B)
    fits = bind_fits_pair(spy, VIEW, served_at=k, state=STATE, budget_end=T0 + 777.0,
                          deadlines=record)
    assert [fits(band, index) for band, index in VIEW.pairs] == [True, False, True] * 3
    assert len(spy.calls) == len(VIEW.pairs)
    for (rest, kwargs), (band, index) in zip(spy.calls, VIEW.pairs):
        entry = VIEW.arrival.entry(band)
        assert rest == [REMAINDER[index]] + [wp for i, wp in enumerate(REMAINDER) if i != index]
        assert kwargs == dict(served_at=k, state=STATE, dwell_s=entry.dwell_s,
                              collected=entry.targets, budget_end=T0 + 777.0,
                              pass_kind=COLLECT, deadlines=record)
    home = _Spy(lambda rest, kwargs: True)
    assert bind_fits_pair(home, HOME_VIEW, served_at=k, state=STATE, budget_end=None)(
        "narrow", None) is True
    ((rest, kwargs),) = home.calls
    assert rest == [] and kwargs["dwell_s"] == 9.0 and kwargs["collected"] == tuple(
        DeviceID(d) for d in ("k1", "k2", "k3")) and kwargs["deadlines"] is None


def _order(rest, index):
    return [] if index is None else [rest[index]] + rest[:index] + rest[index + 1:]


def test_bind_fits_pair_asks_the_predicate_about_the_pairs_order():
    """On the real runtime, under every ``deadline_bounds``, the binding answers
    for every pair what the scheduler's predicate answers for the pair's order."""
    rng = random.Random("u3-bind")
    asked = 0
    for _ in range(40):
        rt, sch, k, rest, state, budget_end, view = _field(rng)
        fits = bind_fits_pair(sch, view, served_at=k, state=state, budget_end=budget_end)
        for band, index in view.pairs:
            entry = view.arrival.entry(band)
            res = sch.fits_after_service(_order(rest, index), served_at=k, state=state,
                                         dwell_s=entry.dwell_s, collected=entry.targets,
                                         budget_end=budget_end)
            assert fits(band, index) is res.ok
            asked += 1
        with pytest.raises(ValueError, match="exactly when no stop is left"):
            fits(view.covering[0].name, 0 if not rest else None)
    assert asked > 100
    rt, sch, k, rest, state, budget_end, view = _field(random.Random("u3-bind-other"))
    with pytest.raises(ValueError, match="the view is of the stop served"):
        bind_fits_pair(sch, view, served_at=_wp(1, 1, "z"), state=state, budget_end=budget_end)
    with pytest.raises(TypeError, match="served_at"):
        bind_fits_pair(sch, view, served_at=k.position, state=state, budget_end=budget_end)


@pytest.mark.parametrize("band, index", [
    ("narrow", -1), ("narrow", 3), ("narrow", True), ("narrow", 1.0), ("narrow", "1"),
    ("wide", 0), ("tiny", 0), (None, 0), (["narrow"], 0),
], ids=["negative_index", "index_past_the_remainder", "bool_index", "float_index",
        "string_index", "class_missing_a_committed_target", "class_off_the_link", "no_class",
        "unhashable_class"])
def test_bind_fits_pair_answers_only_for_the_views_pairs(band, index):
    """The binding is public (the supervisor and FerrySim ask it), so it holds
    its callers to ``view.pairs`` as the scope guard holds the slot, before
    anything is folded: ``moved_to_front`` would fold a route holding stops
    twice for -1, read a bool as a stop, or fail on an index past the
    remainder, and a class that misses a committed target is no pair at all.
    With no budget every fold passes, so an answer would be a silent True."""
    view = _view(_arrival("narrow"))                     # only narrow reaches k3
    assert view.pairs == (("narrow", 0), ("narrow", 1), ("narrow", 2))
    spy = _Spy(lambda rest, kwargs: True)
    fits = bind_fits_pair(spy, view, served_at=_wp(0, 0, "k1", "k2", "k3"), state=STATE,
                          budget_end=None)
    with pytest.raises(ValueError, match="the view's pairs only"):
        fits(band, index)
    assert spy.calls == []
    assert fits("narrow", 2) is True and len(spy.calls) == 1


def _after_service(rt, sch, k, state, band, view):
    """The flight state the service of ``k`` on ``band`` leaves when it goes as
    priced: the class's dwell at hover power, and every target answered, so
    under route-level ``delivery`` their deadlines lower ``deliver_by``
    (``_ferry_fly_pass``)."""
    entry = view.arrival.entry(band)
    deliver_by = state.deliver_by
    if rt.spec.deadline_bounds == DEADLINE_BOUNDS_DELIVERY:
        deliver_by = min([deliver_by] + [sch.last_plan_deadlines[d] for d in entry.targets])
    return FlightState(k.position, state.clock + entry.dwell_s,
                       state.energy_j + rt.spec.flight.energy.p_hover_w * entry.dwell_s,
                       deliver_by)


def _departure_fits(sch, departure, budget_end):
    """The supervisor's ``fits`` (``_ferry_next_stop``): the departure check's
    fold of an order from ``departure``, the plan's exempt stops protected."""
    def fits(order):
        return sch.fold_remainder(order, state=departure, budget_end=budget_end,
                                  pass_kind=COLLECT, protected=sch.plan_protected(list(order))).ok

    return fits


def _fx_rule(rt, sch, k, rest, state, budget_end, view):
    """FX's pair by its own rules at this arrival, on the remainder as it
    stands: CrossHeuristic's band, then its next stop from the state the
    service left as priced, with the supervisor's ``fits``, before any
    departure check (which the arm runs first: ``_flight``)."""
    fx = CrossHeuristic()
    band = fx.band_at_arrival(view.arrival, pass_kind=COLLECT) or view.arrival.committed
    departure = _after_service(rt, sch, k, state, band, view)
    nxt = None if not rest else fx.next_stop(
        rest, departure, fits=_departure_fits(sch, departure, budget_end), pass_kind=COLLECT,
        after_stop=True)
    return band, nxt


def _flight(rt, sch, k, state, budget_end, view, band, remainder, response, slot):
    """The supervisor's next step after serving ``k`` on ``band`` as priced
    (``_ferry_fly_pass``, ``_ferry_departure``): the departure check on
    ``remainder`` as it stands (``abort``: the next stop alone, else the pass
    is given up; ``replan``: the whole remainder, re-planned by the scheduler
    when it fails), then ``slot``'s pick, which is flown. Returns the band, the
    stop flown next (None when none is), the stops left after it, and what the
    check did."""
    departure = _after_service(rt, sch, k, state, band, view)
    fits = _departure_fits(sch, departure, budget_end)
    route, check = list(remainder), "kept"
    if response == RESPONSE_ABORT:
        if not fits(route[:1]):
            return band, None, (), "gave up"
    elif not fits(route):
        route = list(sch.replan_remainder(
            route, state=departure, budget_end=budget_end, pass_kind=COLLECT,
            protected=sch.plan_protected(route), deadlines=sch.last_plan_deadlines).route)
        check = "re-planned"
    if not route:
        return band, None, (), check
    index = slot.next_stop(route, departure, fits=fits, pass_kind=COLLECT, after_stop=True)
    return band, route[index], tuple(route[:index] + route[index + 1:]), check


def test_on_the_real_runtime_every_scorer_flies_an_admitted_pair_or_fxs_flagged():
    """Every scorer's pair, re-asked of the scheduler, fits, or nothing fits and
    the pair is FX's with no reorder; ``fx_pair`` flies FX's rule's pair (its
    band, then its next stop from the service as priced, before any departure
    check) wherever a pair on FX's band is admitted or none is, and every
    record holds that pair."""
    rng = random.Random("u3-real")
    seen = collections.Counter()
    for _ in range(150):
        rt, sch, k, rest, state, budget_end, view = _field(rng)
        fx_pair = _fx_rule(rt, sch, k, rest, state, budget_end, view)
        noise = random.Random(rng.random())
        scorers = [scripted_scorer(name) for name in SCRIPTED_SCORERS] + [
            _Fixed(lambda v: [noise.uniform(-1.0, 1.0) for _ in v.pairs])]
        for scorer in scorers:
            fits = bind_fits_pair(sch, view, served_at=k, state=state, budget_end=budget_end)
            choice = _decide(PairQSlot(scorer), view, fits)
            band, index = choice.pair
            entry = view.arrival.entry(band)
            ok = sch.fits_after_service(_order(rest, index), served_at=k, state=state,
                                        dwell_s=entry.dwell_s, collected=entry.targets,
                                        budget_end=budget_end).ok
            assert ok == (choice.fallback is None)
            assert (choice.fx_band, choice.fx_next) == fx_pair
            if choice.fallback is not None:
                assert choice.pair == (fx_pair[0], None if not rest else 0) == fx_pair
            if scorer.name == FX_PAIR:
                on_fx_band = [p for p in choice.describe()["admitted_pairs"]
                              if p[0] == fx_pair[0]]
                if on_fx_band or choice.fallback is not None:
                    assert choice.pair == fx_pair
                    seen["fx_pair is FX's rule"] += 1
                seen["fx reorders"] += bool(fx_pair[1])
                seen["fx switches band"] += fx_pair[0] != view.arrival.committed
            seen["admitted" if ok else "empty"] += 1
    assert seen["admitted"] > 100 and seen["empty"] > 100, seen
    assert seen["fx_pair is FX's rule"] > 100 and seen["fx reorders"] > 10, seen
    assert seen["fx switches band"] > 10, seen


@pytest.mark.parametrize("response", [RESPONSE_ABORT, RESPONSE_REPLAN])
def test_each_reference_flies_its_fixed_arm_where_that_arms_pair_fits(response):
    """The slot's references against the fixed arms of their names, past the
    departure after this stop, as priced (``_flight``): ``fx_pair`` against FX,
    ``committed_pair`` against F (the committed slot) and ``hyb`` against a
    fixed HYB (FX's band, then the committed slot's order). Each flies its
    arm's flight wherever the arm's own pair, its band with the plan's next
    stop, is admitted. Elsewhere:

    * ``fx_pair`` under ``abort`` parts from FX only where the mask refuses
      FX's band while it admits another band's pair: the arm's check folds the
      next stop alone, which passes whenever a stop on FX's band is admitted.
      Under ``replan`` the arm's check re-plans exactly where ``[fx_band, 0]``
      is missing from the record's ``admitted_pairs``, and FX then picks from
      what is left, so the two part there: after the slot's reorder (which
      drops nothing) and after the fallback (the slot flies the re-planned
      route's head, FX its nearest feasible stop on it). The record's
      ``agrees_fx`` is FX's rule's agreement, and stays True there;
    * ``committed_pair`` and ``hyb`` reorder (or change band) where their
      arm's pair is refused, and on an empty mask fly FX's pair, which is F's
      only when FX's band is b̄, and always HYB's.
    """
    rng = random.Random(f"u3-arms-{response}")
    arms = {FX_PAIR: CrossHeuristic(), COMMITTED_PAIR: CommittedSlot(), HYB: CommittedSlot()}
    reorder = PairQSlot(FXPairScorer())               # the slot's pick after the stop: index 0
    seen = collections.Counter()
    for _ in range(300):
        rt, sch, k, rest, state, budget_end, view = _field(rng, n_stops=(2, 6))
        fits = bind_fits_pair(sch, view, served_at=k, state=state, budget_end=budget_end)
        for name, arm in arms.items():
            choice = _decide(PairQSlot(scripted_scorer(name)), view, fits)
            rec = choice.describe()
            fx_band, committed = rec["fx_band"], rec["committed"]
            band = committed if name == COMMITTED_PAIR else fx_band
            admitted = {tuple(pair) for pair in rec["admitted_pairs"]}
            own = (band, 0) in admitted
            on_band = any(b == band for b, _ in admitted)
            slot = _flight(rt, sch, k, state, budget_end, view, choice.band,
                           moved_to_front(rest, choice.next_index), response, reorder)
            fixed = _flight(rt, sch, k, state, budget_end, view, band, rest, response, arm)
            same = slot == fixed
            if own:
                assert same, (name, response, rec)
            if name == FX_PAIR and response == RESPONSE_ABORT:
                assert same or (bool(admitted) and not on_band), rec
                seen["fx_pair as FX, its next stop refused"] += same and on_band and not own
                seen["fx_pair as FX, empty mask"] += same and not admitted
            elif name == FX_PAIR:
                # The arm re-plans exactly where the record lacks [fx_band, 0], and
                # only there do the two part; on FX's band the record still agrees.
                assert (fixed[3] == "re-planned") == (not own), (fixed, rec)
                if not same and choice.band == fx_band:
                    assert rec["agrees_fx"], rec
                    seen["fx_pair parts after a reorder"] += on_band
                    seen["fx_pair parts after the fallback"] += not admitted
            elif name == COMMITTED_PAIR:
                if not admitted:
                    assert same == (fx_band == committed), rec
                seen["committed_pair parts at an empty mask"] += not same and not admitted
                seen["committed_pair parts after a reorder"] += not same and on_band and not own
            else:
                assert same or (bool(admitted) and not own), rec
                seen["hyb parts after a reorder"] += not same and on_band and not own
    if response == RESPONSE_ABORT:
        assert seen["fx_pair as FX, its next stop refused"] >= 5, seen
        assert seen["fx_pair as FX, empty mask"] >= 20, seen
    else:
        assert seen["fx_pair parts after a reorder"] >= 5, seen
        assert seen["fx_pair parts after the fallback"] >= 10, seen
    assert seen["committed_pair parts at an empty mask"] >= 10, seen
    assert seen["committed_pair parts after a reorder"] >= 5, seen
    assert seen["hyb parts after a reorder"] >= 5, seen


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

def test_a_decision_record_holds_the_pair_its_mask_and_fxs_pair_and_no_wall_time():
    scores = [0.1, 0.25, 0.2, 0.0, 0.0, 0.0, 0.3, 0.123456789, 0.0]
    admitted = (("wide", 1), ("wide", 2), ("narrow", 1), ("narrow", 2))
    choice = _decide(PairQSlot(_Fixed(scores, name="pair_v1", q_values=True)), VIEW,
                     _admit(*admitted))
    assert isinstance(choice, PairChoice)
    assert (choice.band, choice.next_index, choice.q) == ("wide", 1, 0.25)
    rec = choice.describe()
    assert list(rec) == list(DECISION_KEYS)
    assert rec == {
        "t_s": T0 + 100.0, "devices": ["k1", "k2", "k3"], "committed": "medium",
        "band": "wide", "next_index": 1, "next": ["b"], "pairs": 9, "feasible": 4,
        "admitted_pairs": [["wide", 1], ["wide", 2], ["narrow", 1], ["narrow", 2]],
        "fallback": None, "fx_band": "wide", "fx_next": 1, "agrees_fx": True,
        "scorer": "pair_v1", "q": 0.25, "q_fx": 0.25,
    }
    # A learned pick away from FX's: the values to 6 places.
    other = _decide(PairQSlot(_Fixed([0.0] * 7 + [0.123456789, 0.0], name="pair_v1",
                                     q_values=True)), VIEW, _admit(*admitted)).describe()
    assert (other["band"], other["next_index"], other["next"], other["agrees_fx"]) == (
        "narrow", 1, ["b"], False)
    assert (other["q"], other["q_fx"]) == (0.123457, 0.0)
    for record in (rec, other):
        assert json.loads(json.dumps(record, allow_nan=False)) == record
        assert not any("wall" in key for key in record)
    # The same arrival decided again records the same, and the record is read-only.
    assert _decide(PairQSlot(_Fixed(scores, name="pair_v1", q_values=True)), VIEW,
                   _admit(*admitted)).describe() == rec
    with pytest.raises(TypeError):
        choice.record["band"] = "medium"
    # A scripted scorer's numbers are ranks, not values: no q. HYB keeps FX's
    # band but not its stop, so it does not agree with FX's pair.
    plain = _decide(PairQSlot(HybScorer()), VIEW, _all(VIEW)).describe()
    assert (plain["q"], plain["q_fx"], plain["scorer"], plain["next"]) == (
        None, None, "hyb", ["a"])
    assert (plain["band"], plain["next_index"], plain["fx_band"], plain["fx_next"]) == (
        "wide", 0, "wide", 1)
    assert plain["agrees_fx"] is False


def test_closed_record_adds_what_the_mission_settled():
    choice = _decide(PairQSlot(FXPairScorer()), VIEW, _all(VIEW))
    closed = closed_record(choice.record, collected=["k2", "k1"], weights={"k1": 7.0, "k2": 3},
                           late=["k1", "k2"], t_next_s=T0 + 131.5, terminal=False,
                           trimmed_next=True)
    assert list(closed) == list(DECISION_KEYS) + list(CLOSE_KEYS)
    assert {key: closed[key] for key in CLOSE_KEYS} == {
        "collected": ["k2", "k1"], "w": [3.0, 7.0], "late": ["k2", "k1"],
        "t_next_s": T0 + 131.5, "terminal": False, "trimmed_next": True}
    assert closed_record(choice.record, collected=["k2", "k1"], weights={"k1": 7.0, "k2": 3},
                         late={"k1"}, t_next_s=T0 + 131.5, terminal=False,
                         trimmed_next=True)["late"] == ["k1"]
    assert {key: closed[key] for key in DECISION_KEYS} == choice.describe()
    assert json.loads(json.dumps(closed, allow_nan=False)) == closed
    # From a describe() copy alike; nothing collected (a failed contact) closes too.
    empty = closed_record(choice.describe(), collected=[], weights={}, late=[],
                          t_next_s=T0 + 100.0, terminal=True, trimmed_next=False)
    assert (empty["collected"], empty["w"], empty["late"]) == ([], [], [])


_UNSETTLED = {
    "collected_outside_the_stop": (dict(collected=["k1", "a"], weights={"k1": 1.0, "a": 1.0}),
                                   ValueError, "outside the stop"),
    "collected_twice": (dict(collected=["k1", "k1"]), ValueError, "each device once"),
    "collected_a_string": (dict(collected="k1"), TypeError, "lists device ids"),
    "a_weight_missing": (dict(weights={"k1": 1.0}), ValueError, "exactly the collected devices"),
    "a_weight_extra": (dict(weights={"k1": 1.0, "k2": 1.0, "k3": 1.0}), ValueError,
                       "exactly the collected"),
    "weights_a_list": (dict(weights=[1.0, 2.0]), TypeError, "weights map"),
    "weight_negative": (dict(weights={"k1": -1.0, "k2": 1.0}), ValueError, ">= 0"),
    "weight_nan": (dict(weights={"k1": math.nan, "k2": 1.0}), ValueError, "finite"),
    "weight_a_bool": (dict(weights={"k1": True, "k2": 1.0}), TypeError, "number"),
    "late_not_collected": (dict(late=["k3"]), ValueError, "not collected"),
    "next_before_the_arrival": (dict(t_next_s=T0 + 99.0), ValueError, "after the arrival"),
    "next_infinite": (dict(t_next_s=math.inf), ValueError, "finite"),
    "terminal_an_int": (dict(terminal=1), TypeError, "terminal"),
    "trimmed_next_none": (dict(trimmed_next=None), TypeError, "trimmed_next"),
}


@pytest.mark.parametrize("case", list(_UNSETTLED))
def test_closed_record_refuses_what_the_mission_cannot_have_settled(case):
    change, error, match = _UNSETTLED[case]
    choice = _decide(PairQSlot(FXPairScorer()), VIEW, _all(VIEW))
    kwargs = dict(collected=["k1", "k2"], weights={"k1": 1.0, "k2": 2.0}, late=[],
                  t_next_s=T0 + 120.0, terminal=False, trimmed_next=False)
    kwargs.update(change)
    with pytest.raises(error, match=match):
        closed_record(choice.record, **kwargs)


def test_a_record_is_closed_once():
    choice = _decide(PairQSlot(FXPairScorer()), VIEW, _all(VIEW))
    closed = closed_record(choice.record, collected=[], weights={}, late=[],
                           t_next_s=T0 + 120.0, terminal=True, trimmed_next=False)
    with pytest.raises(ValueError, match="closed already"):
        closed_record(closed, collected=[], weights={}, late=[], t_next_s=T0 + 120.0,
                      terminal=True, trimmed_next=False)
    for bad in ({**choice.describe(), "extra": 1}, {"t_s": 1.0}, ["t_s"]):
        with pytest.raises((ValueError, TypeError), match="record"):
            closed_record(bad, collected=[], weights={}, late=[], t_next_s=T0 + 120.0,
                          terminal=True, trimmed_next=False)


# --------------------------------------------------------------------------- #
# Training (FerrySim)
# --------------------------------------------------------------------------- #

class _Sink:
    def __init__(self):
        self.calls = []

    def __call__(self, steps, records):
        self.calls.append((steps, records))


def _close(choice, t_next_s, *, terminal=False):
    return closed_record(choice.record, collected=[], weights={}, late=[], t_next_s=t_next_s,
                         terminal=terminal, trimmed_next=False)


def test_a_trainer_explores_the_admitted_pairs_only():
    admitted = [("wide", 2), ("medium", 0), ("medium", 1), ("narrow", 2)]
    slot = PairQSlot(FXPairScorer())
    slot.attach_trainer(epsilon=1.0, rng=random.Random(3), sink=_Sink())
    assert slot.training
    flown = collections.Counter(_decide(slot, VIEW, _admit(*admitted)).pair for _ in range(400))
    assert set(flown) == set(admitted) and min(flown.values()) > 60


def test_epsilon_0_flies_the_scorers_argmax_or_fxs_pair_in_the_reference_phase():
    prefer_narrow = _Fixed(lambda v: [1.0 if band == "narrow" else 0.0 for band, _ in v.pairs])
    cases = [  # (admitted, around FX's pair, flown)
        (None, False, ("narrow", 0)),
        (None, True, ("wide", 1)),                                    # FX's pair
        ([("wide", 2), ("wide", 0), ("narrow", 0)], True, ("wide", 2)),  # FX's nearest that fits
        ([("medium", 1), ("narrow", 2)], True, ("narrow", 2)),        # FX's band refused
    ]
    for admitted, around, flown in cases:
        slot = PairQSlot(prefer_narrow)
        slot.attach_trainer(epsilon=0.0, rng=random.Random(0), sink=_Sink(),
                            around_reference=around)
        fits = _all(VIEW) if admitted is None else _admit(*admitted)
        assert _decide(slot, VIEW, fits).pair == flown
    # At ε = 0 off the reference phase a trainer flies exactly as no trainer does,
    # so an evaluator may attach one only to receive each mission's steps.
    rng = random.Random("u3-epsilon-0")
    trained = PairQSlot(_Fixed(lambda v: [float((len(band) * 7 + (index or 0) * 3) % 5)
                                          for band, index in v.pairs]))
    trained.attach_trainer(epsilon=0.0, rng=random.Random(5), sink=_Sink())
    for _ in range(300):
        view = _random_view(rng)
        rows = _rows(view)
        mask = tuple(rng.random() < 0.5 for _ in view.pairs)

        def ask():
            return _Ask(lambda band, index: mask[rows[(band, index)]])

        untrained = PairQSlot(trained.scorer)
        assert _decide(trained, view, ask()).describe() == _decide(untrained, view,
                                                                   ask()).describe()


def test_a_trainer_draws_two_numbers_per_decision_and_none_on_an_empty_mask():
    """So runs that differ in ε or in their weights draw alike (``behaviour_row``)."""
    rng, ref = random.Random(11), random.Random(11)
    slot = PairQSlot(FXPairScorer())
    slot.attach_trainer(epsilon=0.25, rng=rng, sink=_Sink())
    _decide(slot, VIEW, _all(VIEW))
    ref.random(), ref.random()
    assert rng.getstate() == ref.getstate()
    choice = _decide(slot, VIEW, _none(VIEW))
    assert choice.fallback == PAIR_FALLBACK_MASK_EMPTY and rng.getstate() == ref.getstate()
    _decide(slot, HOME_VIEW, _admit(("medium", None)))
    ref.random(), ref.random()
    assert rng.getstate() == ref.getstate()


def test_the_sink_gets_each_missions_steps_with_their_closed_records():
    sink = _Sink()
    scorer = _Fixed(lambda v: [float(r) for r in range(len(v.pairs))])
    slot = PairQSlot(scorer)
    slot.attach_trainer(epsilon=0.0, rng=random.Random(1), sink=sink)
    first = _decide(slot, VIEW, _admit(("wide", 0), ("medium", 2)))
    second = _decide(slot, HOME_VIEW, _none(HOME_VIEW))
    assert sink.calls == []
    records = [_close(first, T0 + 140.0), _close(second, T0 + 200.0, terminal=True)]
    slot.close_mission([types.MappingProxyType(record) for record in records])
    ((steps, got),) = sink.calls
    # The sink gets the checked copies: fresh plain dicts, whatever mapping closed them.
    assert got == tuple(records) and all(type(record) is dict for record in got)
    assert not any(mine is theirs for mine, theirs in zip(got, records))
    assert [step.choice for step in steps] == [first, second]
    one, two = steps
    assert isinstance(one, PairStep) and one.view is VIEW and two.view is HOME_VIEW
    assert one.row == _rows(VIEW)[("medium", 2)] == 5
    assert one.mask == tuple(p in {("wide", 0), ("medium", 2)} for p in VIEW.pairs)
    assert one.effective_mask == one.mask and one.scores == tuple(float(r) for r in range(9))
    # An empty mask's step was taken among FX's pair alone, which was flown.
    assert two.mask == (False,) * 3 and two.row == 0
    assert two.effective_mask == (True, False, False)
    # The next mission starts empty; a mission with no decision closes empty.
    slot.close_mission([])
    assert sink.calls[1] == ((), ())


def test_close_mission_refuses_records_that_do_not_close_its_decisions():
    def trained(*views):
        slot = PairQSlot(FXPairScorer())
        slot.attach_trainer(epsilon=0.0, rng=random.Random(1), sink=_Sink())
        return slot, [_decide(slot, view, _all(view)) for view in views]

    slot, (a,) = trained(VIEW)
    with pytest.raises(ValueError, match=r"made 1 pair decision\(s\), and 0"):
        slot.close_mission([])
    slot, (a,) = trained(VIEW)
    with pytest.raises(ValueError, match="missing"):
        slot.close_mission([a.describe()])                          # not closed
    slot, (a,) = trained(VIEW)
    other = _decide(PairQSlot(HybScorer()), VIEW, _all(VIEW))
    with pytest.raises(ValueError, match="not the record of the decision"):
        slot.close_mission([_close(other, T0 + 150.0)])
    slot, (a, b) = trained(VIEW, HOME_VIEW)
    with pytest.raises(ValueError, match="not the record of the decision"):
        slot.close_mission([_close(b, T0 + 150.0), _close(a, T0 + 160.0)])     # out of order
    slot, (a,) = trained(VIEW)
    by_hand = {**closed_record(a.record, collected=["k1", "k2"], weights={"k1": 1, "k2": 1},
                               late=["k1", "k2"], t_next_s=T0 + 150.0, terminal=True,
                               trimmed_next=False), "late": ["k2", "k1"]}
    with pytest.raises(ValueError, match="not closed by closed_record"):
        slot.close_mission([by_hand])
    slot, (a,) = trained(VIEW)
    with pytest.raises(ValueError, match="unknown"):
        slot.close_mission([{**_close(a, T0 + 150.0), "extra": 0}])
    slot, (a,) = trained(VIEW)
    with pytest.raises(TypeError, match="closed pair record"):
        slot.close_mission(["record"])


def test_without_a_trainer_close_mission_is_a_no_op():
    slot = PairQSlot(FXPairScorer())
    _decide(slot, VIEW, _all(VIEW))
    assert slot.close_mission(["anything", 1]) is None
    assert slot.close_mission([]) is None


def test_attach_trainer_refuses_a_bad_setting_or_a_slot_that_has_decided():
    for kwargs, error, match in [
        (dict(epsilon=True), TypeError, "epsilon"), (dict(epsilon="0.1"), TypeError, "epsilon"),
        (dict(epsilon=-0.1), ValueError, "epsilon"), (dict(epsilon=1.5), ValueError, "epsilon"),
        (dict(epsilon=math.nan), ValueError, "epsilon"), (dict(rng=None), TypeError, "rng"),
        (dict(rng=3), TypeError, "rng"), (dict(sink=None), TypeError, "sink"),
        (dict(around_reference="yes"), TypeError, "around_reference"),
    ]:
        settings = dict(epsilon=0.1, rng=random.Random(0), sink=_Sink())
        settings.update(kwargs)
        slot = PairQSlot(FXPairScorer())
        with pytest.raises(error, match=match):
            slot.attach_trainer(**settings)
        assert not slot.training
    # The training schedule's behaviour is the two settings, as a dataclass.
    slot = PairQSlot(FXPairScorer())
    behaviour = pair_q.BehaviourSchedule().at(0, 10)
    slot.attach_trainer(**dataclasses.asdict(behaviour), rng=random.Random(0), sink=_Sink())
    assert slot.training
    _decide(slot, VIEW, _all(VIEW))
    with pytest.raises(RuntimeError, match="before the slot's first decision"):
        slot.attach_trainer(epsilon=0.0, rng=random.Random(0), sink=_Sink())


# --------------------------------------------------------------------------- #
# Freeze Rule 1 and layering
# --------------------------------------------------------------------------- #

def _ref_source(path):
    try:
        return subprocess.run(["git", "show", f"{REF_COMMIT}:{path}"], cwd=REPO,
                              capture_output=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:          # pragma: no cover - no git
        pytest.skip(f"git cannot show {REF_COMMIT}'s {path}: {e}")


@pytest.fixture(scope="module")
def ref_scope_guard(tmp_path_factory):
    """``selector/scope_guard.py`` as it was at 386c275, as a module of its own."""
    path = tmp_path_factory.mktemp("ref_scope_guard") / "_ref_scope_guard.py"
    path.write_bytes(_ref_source(SCOPE_GUARD))
    name = f"_p5_ref_{REF_COMMIT}_scope_guard"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
        assert not hasattr(mod, "assert_pairs_admitted")              # Phase 5 is not in it
        yield mod
    finally:
        sys.modules.pop(name, None)


def test_scope_guard_differs_from_386c275s_by_the_new_function_only():
    old = ast.parse(_ref_source(SCOPE_GUARD).decode("utf-8"))
    new = ast.parse((REPO / SCOPE_GUARD).read_text(encoding="utf-8"))

    def defs(tree):
        return {node.name: ast.dump(node) for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}

    def imports(tree):
        return {(node.module, alias.name) for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) for alias in node.names}

    def rest(tree):
        return [ast.dump(node) for node in tree.body[1:]
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.ImportFrom))]

    assert set(defs(new)) - set(defs(old)) == {"assert_pairs_admitted"}
    assert {name: defs(new)[name] for name in defs(old)} == defs(old)
    added = imports(new) - imports(old)
    assert imports(old) <= imports(new) and {module for module, _ in added} == {"typing"}
    assert ast.get_docstring(new).startswith(ast.get_docstring(old))
    assert rest(new) == rest(old) == []


def test_assert_candidates_admitted_is_386c275s(ref_scope_guard):
    rng = random.Random("u3-candidates")
    pool = [DeviceID(f"d{i}") for i in range(8)]
    raised = 0
    for _ in range(300):
        candidates = rng.sample(pool, rng.randint(0, 6))
        admitted = rng.sample(pool, rng.randint(0, 8))
        outcomes = []
        for guard, error in ((ref_scope_guard.assert_candidates_admitted,
                              ref_scope_guard.SelectorScopeViolation),
                             (assert_candidates_admitted, SelectorScopeViolation)):
            try:
                outcomes.append(("returned", guard(candidates, admitted)))
            except error as exc:
                outcomes.append(("raised", str(exc)))
        assert outcomes[0] == outcomes[1]
        raised += outcomes[1][0] == "raised"
    assert 50 < raised < 250


def _loaded(statement):
    """The ``hermes`` and ``experiments`` modules a fresh interpreter holds after ``statement``."""
    code = ("import json, sys\n" + statement + "\nprint(json.dumps(sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('hermes', 'experiments'))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True, capture_output=True,
                         text=True, encoding="utf-8", timeout=120).stdout
    return set(json.loads(out.strip().splitlines()[-1]))


_PLAN_OR_PAIR = {
    "hermes.scheduler.plan", "hermes.scheduler.plan.types",
    "hermes.scheduler.policies.cross_heuristic", "hermes.scheduler.policies.next_stop",
    "hermes.scheduler.policies.pair_slot", "hermes.scheduler.selector.pair_q",
    "hermes.scheduler.selector.pair_replay", "hermes.scheduler.selector.pair_features",
}


def test_no_recorded_path_loads_the_slot_and_the_slot_loads_no_runtime():
    """Every D arm loads the selector package, and so the scope guard: neither
    loads a plan or pair module. The policies package does not load the slot.
    The slot loads the plan types, the FX filling, the learner's rule and the
    guard, and nothing of the runtime, the mission or ``experiments``."""
    selector = _loaded("import hermes.scheduler.selector.scope_guard")
    policies = _loaded("import hermes.scheduler.policies")
    assert "hermes.scheduler.selector.scope_guard" in selector
    assert not (selector | policies) & _PLAN_OR_PAIR
    slot = _loaded("import hermes.scheduler.policies.pair_slot")
    assert {"hermes.scheduler.policies.pair_slot", "hermes.scheduler.selector.pair_q",
            "hermes.scheduler.plan.types", "hermes.scheduler.policies.cross_heuristic"} <= slot
    assert not {m for m in slot if m.startswith(("hermes.l1", "hermes.mule", "hermes.mission",
                                                "experiments"))}
    assert not slot & {"hermes.scheduler.policies.next_stop",
                       "hermes.scheduler.selector.pair_replay",
                       "hermes.scheduler.selector.pair_features"}


def test_the_slot_module_imports_only_the_plan_types_the_fx_filling_the_learner_and_the_guard():
    tree = ast.parse((REPO / "hermes/scheduler/policies/pair_slot.py").read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    assert modules == {
        "__future__", "collections.abc", "math", "numbers", "dataclasses", "typing",
        "hermes.scheduler.plan.types", "hermes.scheduler.policies.cross_heuristic",
        "hermes.scheduler.selector.pair_q", "hermes.scheduler.selector.scope_guard",
        "hermes.types.ids", "hermes.types.scheduler",
    }
