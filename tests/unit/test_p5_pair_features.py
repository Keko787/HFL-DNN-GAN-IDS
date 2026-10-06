"""FeRRy Phase 5 (unit U1): the pair features ``pair_v1``, their schema, the
learned score's adapter and the slot built from a checkpoint
(``hermes/scheduler/selector/pair_features.py``).

What is pinned (the Phase 5 spec, unit U1 and other choices 4; the user's
decision 6 (a); orchestrator resolution R8):

* **shapes and the schema**: one row per pair of ``view.pairs``, in that
  order, ``schema.dim`` wide; ``pair_v1``'s column list over the link's
  classes, with and without the phase block; the schema's JSON is what a
  checkpoint's header records, and a schema of another version, layout or
  scale is refused;
* **the covering classes** are FX's candidate set and the view's;
* **each column's formula**, worked by hand (the log-scaled slack and its
  exemption, the clipped shares, the capped age, the phases), and what each
  column varies with across a view's rows: the band, the next stop, both, or
  neither; the phase block at the view's own P_c, not the channel's default
  60 s (Study 5.6 sets P_c per cell, decision 6 (a));
* **bounds and flags**: every column finite and within its declared bounds on
  random and extreme views; flags are 0 or 1 and the view's;
* **the home row**, and **defaults for NEW devices** (the on-time prior, the
  plan's age, no weight for a beacon insert, no previous reading, no budget or
  energy reference);
* **T2**: with one class there is one row per stop and the band block is
  constant;
* **causal phase features**: in loopback ``pair_q`` missions (U5's harness)
  whose contact channel raises if read beyond the mule's clock while the mule
  decides, every decision reads the channel at its own instant, the phase
  columns are those readings at the channel's period (the default 60 s and
  30 s), and the previous reading is the last decision's, carried across
  stops and sorties and reset each trial; a scorer that reads ahead is
  caught;
* **a FerrySim sample**: no column constant but the declared sparse ones,
  every column in bounds, NEW devices in mission 1, and the view's N is the
  reward's; a cell's interference period (30 s) reaches the views and the
  phase columns;
* **the adapter**: the slot's learned ``PairScorer`` (Q values, the schema's
  name, the live online network's numbers in the pairs' order, not the
  target network's, the mask not read), the masked argmax in the slot, and a
  deterministic FerrySim episode;
* **build_pair_slot** from a verified checkpoint, flying as the network it
  saved; the loader's refusals of what the link would misread and of a path
  with no checkpoint file, every one a ``CheckpointError``, and the caller's
  argument errors raised before any file is read, never one;
* **layering**: the module loads nothing from the runtime, the slot's module
  only when a slot is built, and the selector package does not load it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.exp4.model_task import _u32
from experiments.exp4.topology_builder import device_positions, device_spread_m
from experiments.ferrysim import cells as FC
from experiments.ferrysim.episode import Policy, Trainer, run_episode
from hermes.l1.channel_model import ContactChannel
from hermes.l1.mission_clock import SIM_EPOCH_S, MissionClock
from hermes.mission.contact_plan import planar_distance_m
from hermes.mule.ferry import FerrySpec
from hermes.scheduler.plan import AgeCapSpec, PlanOptions
from hermes.scheduler.plan.types import (
    ArrivalClass,
    ArrivalView,
    PairScorer,
    PairView,
    StopContext,
)
from hermes.scheduler.policies.cross_heuristic import fastest_covering_class
from hermes.scheduler.policies.pair_slot import PairQSlot
from hermes.scheduler.selector.features import _on_time_rate
from hermes.scheduler.selector.pair_features import (
    DEPENDS,
    DEPENDS_BAND,
    DEPENDS_NEXT,
    DEPENDS_STATE,
    OFFSET_SCALE_DB,
    PAIR_FEATURE_SCHEMA,
    PREVIOUS_AGE_CAP_PERIODS,
    SCHEMA_CONSTANTS,
    SPARSE_COLUMNS,
    LearnedPairScorer,
    PairFeatureSchema,
    build_pair_slot,
    covering_classes,
    load_pair_scorer,
    pair_rows,
)
from hermes.scheduler.selector.pair_q import (
    CheckpointError,
    PairQConfig,
    PairQNet,
    manifest_path,
    masked_argmax,
    verify_checkpoint,
)
from hermes.scheduler.selector.pair_replay import PairBatch, PairTransition
from hermes.types import Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState, MissionPass

from tests.golden import _mule_harness as GH
from tests.integration import _ferry_harness as H

REPO = Path(__file__).resolve().parents[2]
COLLECT = MissionPass.COLLECT
T0 = SIM_EPOCH_S
CLASSES = ("wide", "medium", "narrow")
#: Link class tuples by size (the D1 classes, and the four-class option's).
CLASS_SETS = {1: ("wide",), 2: ("wide", "narrow"), 3: CLASSES,
              4: ("wide", "medium", "narrow", "ten")}
BAND_BLOCK = ("snr_here", "dwell_here", "gain_here")
TINY = PairQConfig(hidden=(8,))


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #

def _wp(x, y, *devices, deadline):
    return ContactWaypoint(position=(float(x), float(y), 0.0),
                           devices=tuple(DeviceID(d) for d in devices),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=float(deadline))


def _ctx(wp, index, *, travel, dwell, snr, capped=False, exempt=False, age=0.0, on_time=0.5,
         weight=0.0):
    return StopContext(stop=wp, index=index, travel_s=float(travel), pred_dwell_s=float(dwell),
                       pred_snr_db=tuple(float(s) for s in snr), capped=capped, exempt=exempt,
                       age=float(age), on_time=float(on_time), weight=float(weight))


#: The hand-worked view. At stop k (members k1-k3) the committed class is medium;
#: wide and medium reach k1 and k2, narrow all three, so every class covers. The
#: remainder: A, 400 s of slack on wide; B overdue and capped; C exempt, its
#: (all-capped) deadline the earliest of all; D undated.
CLOCK = T0 + 100.0
HAND_ARRIVAL = ArrivalView(
    devices=("k1", "k2", "k3"), committed="medium",
    classes=(ArrivalClass("wide", 0, ("k1", "k2"), 6.0),
             ArrivalClass("medium", 1, ("k1", "k2"), 12.0),
             ArrivalClass("narrow", 2, ("k1", "k2", "k3"), 30.0)))
STOP_A = _wp(30, 0, "a1", "a2", deadline=CLOCK + 6.0 + 20.0 + 14.0 + 400.0)
STOP_B = _wp(0, 10, "b1", deadline=T0)
STOP_C = _wp(-20, 0, "c1", deadline=T0 - 500.0)
STOP_D = _wp(0, -40, "d1", deadline=math.inf)
HAND_STOPS = (
    _ctx(STOP_A, 0, travel=20, dwell=14, snr=(9, 15, 21), age=3, on_time=0.25, weight=6),
    _ctx(STOP_B, 1, travel=10, dwell=8, snr=(3, 9, 15), capped=True, age=2, weight=4),
    _ctx(STOP_C, 2, travel=40, dwell=5, snr=(0, 6, 12), capped=True, exempt=True, age=4,
         on_time=1.0, weight=2),
    _ctx(STOP_D, 3, travel=30, dwell=3, snr=(-3, 3, 9)),
)
HAND = PairView(
    arrival=HAND_ARRIVAL, pose=(0.0, 0.0, 0.0), observed_snr_db=(12.0, 18.0, 24.0),
    offsets_db=(5.0, -2.5, 1.0), previous_offsets_db=(-4.0, 3.0, 0.5), previous_age_s=75.0,
    period_s=60.0, clock_s=CLOCK, budget_end=T0 + 300.0, budget_s=250.0, t_ref_s=200.0,
    energy_j=5000.0, energy_ref_j=20000.0, stops=HAND_STOPS, demand=10, demand_weight=20.0,
    cap_s=2)
SCHEMA = PairFeatureSchema(CLASSES)


def _random_view(rng, *, n_classes=None, n_stops=None, previous=None, extreme=False):
    """A valid pair view with random fields; ``extreme`` stretches budgets,
    energy, deadlines and ages far past anything a flight sees."""
    names = CLASS_SETS[rng.randint(1, 4) if n_classes is None else n_classes]
    members = [f"k{i}" for i in range(rng.randint(1, 4))]
    committed = rng.choice(names)
    scale = 50.0 if extreme else 1.0
    classes = []
    for i, name in enumerate(names):
        targets = tuple(d for d in members if rng.random() < 0.7)
        classes.append(ArrivalClass(name, i, targets,
                                    rng.uniform(0.5, 120.0) * scale if targets else 0.0))
    arrival = ArrivalView(devices=tuple(members), committed=committed, classes=tuple(classes))
    clock = T0 + rng.uniform(0.0, 500.0)
    count = rng.randint(0, 5) if n_stops is None else n_stops
    stops = []
    for i in range(count):
        wp = _wp(rng.uniform(-100, 100), rng.uniform(-100, 100),
                 *(f"s{i}_{j}" for j in range(rng.randint(1, 3))),
                 deadline=(math.inf if rng.random() < 0.1
                           else clock + rng.uniform(-300.0, 2000.0) * scale))
        capped = rng.random() < 0.4
        stops.append(_ctx(wp, i, travel=rng.uniform(0.0, 60.0), dwell=rng.uniform(0.0, 80.0),
                          snr=[rng.uniform(-10.0, 40.0) for _ in names], capped=capped,
                          exempt=capped and rng.random() < 0.5, age=rng.uniform(0, 6) * scale,
                          on_time=rng.random(), weight=rng.choice([0.0, rng.uniform(0, 10)])))
    if not stops:
        stops = [StopContext.home(rng.uniform(0.0, 60.0))]
    devices = len(members) + sum(len(c.stop.devices) for c in stops if c.stop is not None)
    weights = sum(c.weight for c in stops)
    has_previous = rng.random() < 0.7 if previous is None else previous
    budget_end = rng.choice([None, clock + rng.uniform(-100.0, 300.0) * scale])
    energy_ref = rng.choice([None, 0.0, rng.uniform(1000.0, 30000.0)])
    return PairView(
        arrival=arrival, pose=(rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0),
        observed_snr_db=tuple(rng.uniform(-10.0, 40.0) for _ in names),
        offsets_db=tuple(rng.gauss(0.0, 6.0) for _ in names),
        previous_offsets_db=(tuple(rng.gauss(0.0, 6.0) for _ in names) if has_previous
                             else None),
        previous_age_s=rng.uniform(0.0, 600.0) * scale if has_previous else None,
        period_s=rng.choice([15.0, 30.0, 60.0, 120.0]), clock_s=clock, budget_end=budget_end,
        budget_s=rng.choice([None, rng.uniform(30.0, 300.0)]), t_ref_s=rng.uniform(50.0, 400.0),
        energy_j=rng.uniform(0.0, 30000.0) * scale, energy_ref_j=energy_ref, stops=tuple(stops),
        demand=devices + rng.randint(0, 5),
        demand_weight=rng.choice([weights, weights + rng.uniform(0.0, 20.0)]),
        cap_s=rng.choice([None, 1, 2, 3]))


def _views(seed, count, **kw):
    rng = random.Random(seed)
    return [_random_view(rng, **kw) for _ in range(count)]


def _col(rows, schema, name):
    return rows[:, schema.index(name)]


def _row(view, schema, band, index):
    rows, pairs = pair_rows(view, schema)
    return dict(zip(schema.columns, rows[pairs.index((band, index))]))


# --------------------------------------------------------------------------- #
# The schema
# --------------------------------------------------------------------------- #

def test_the_schema_is_pair_v1_over_the_links_classes():
    """The row layout is the contract between the features, the trainer and every
    checkpoint: pinned column by column for the D1 link, with the phase block."""
    assert PAIR_FEATURE_SCHEMA == "pair_v1" and SCHEMA.version == "pair_v1"
    assert SCHEMA.classes == CLASSES and SCHEMA.phase is True
    per_class = lambda name: [f"{name}[{c}]" for c in CLASSES]  # noqa: E731
    assert list(SCHEMA.columns) == (
        per_class("band") + ["snr_here", "dwell_here", "gain_here", "reach_here", "travel"]
        + per_class("snr_next")
        + ["dwell_next", "slack_next", "exempt_next", "age_next", "on_time_next",
           "members_next", "capped_next", "home", "clock_left", "energy_left",
           "remainder_share", "least_slack", "weight_share"]
        + per_class("offset") + per_class("prev_offset")
        + ["has_prev", "prev_age", "prev_sin", "prev_cos", "arrival_sin", "arrival_cos"])
    assert SCHEMA.dim == len(SCHEMA.columns) == 36
    depends = {column.name: column.depends for column in SCHEMA.column_specs}
    assert {name for name, dep in depends.items() if dep == "pair"} == {
        "slack_next", "clock_left", "arrival_sin", "arrival_cos"}
    assert {name for name, dep in depends.items() if dep == "band"} == set(
        per_class("band") + ["snr_here", "dwell_here", "gain_here"])
    assert {name for name, dep in depends.items() if dep == "next"} == set(
        ["travel"] + per_class("snr_next") + ["dwell_next", "exempt_next", "age_next",
                                               "on_time_next", "members_next", "capped_next",
                                               "home"])
    flags = {column.name for column in SCHEMA.column_specs if column.flag}
    assert flags == set(per_class("band") + ["exempt_next", "capped_next", "home", "has_prev"])
    unit, sign, positive, free = (0.0, 1.0), (-1.0, 1.0), (0.0, None), (None, None)
    assert {column.name: (column.low, column.high) for column in SCHEMA.column_specs} == {
        **{name: unit for name in flags},
        **{name: free for name in per_class("snr_next") + per_class("offset")
           + per_class("prev_offset") + ["snr_here", "slack_next", "least_slack"]},
        **{name: positive for name in ("dwell_here", "travel", "dwell_next", "age_next")},
        **{name: unit for name in ("gain_here", "reach_here", "on_time_next", "members_next",
                                   "remainder_share", "weight_share")},
        **{name: sign for name in ("clock_left", "energy_left", "prev_sin", "prev_cos",
                                   "arrival_sin", "arrival_cos")},
        "prev_age": (0.0, 4.0),
    }
    no_phase = PairFeatureSchema(CLASSES, phase=False)
    assert no_phase.columns == SCHEMA.columns[:SCHEMA.index("offset[wide]")]
    assert no_phase.dim == 24
    for n, names in CLASS_SETS.items():
        assert PairFeatureSchema(names).dim == 4 * n + 24
        assert PairFeatureSchema(names, phase=False).dim == 2 * n + 18
    for column in SCHEMA.column_specs:
        assert column.depends in DEPENDS
        if column.flag:
            assert (column.low, column.high) == (0.0, 1.0)
    assert len(set(SCHEMA.columns)) == SCHEMA.dim
    assert SCHEMA.index("home") == SCHEMA.columns.index("home")
    with pytest.raises(ValueError, match="no column 'value_next'"):
        SCHEMA.index("value_next")
    # Feature 11, the value of s, is gone (critic C5).
    assert not any("value" in name for name in SCHEMA.columns)
    assert PairFeatureSchema.for_view(HAND) == SCHEMA
    assert PairFeatureSchema.for_view(HAND, phase=False) == no_phase


def test_the_schema_json_round_trips_and_names_its_classes_phase_columns_and_scales():
    data = SCHEMA.to_json()
    assert json.loads(json.dumps(data)) == data
    assert data == {"version": "pair_v1", "dim": 36, "classes": list(CLASSES), "phase": True,
                    "columns": list(SCHEMA.columns), "constants": dict(SCHEMA_CONSTANTS)}
    assert dict(SCHEMA_CONSTANTS) == {"snr_scale_db": 30.0, "offset_scale_db": 10.0,
                                      "previous_age_cap_periods": 4.0, "share_clip": 1.0}
    for schema in (SCHEMA, PairFeatureSchema(CLASSES, phase=False),
                   PairFeatureSchema(CLASS_SETS[4]), PairFeatureSchema(("wide",))):
        assert PairFeatureSchema.from_json(json.loads(json.dumps(schema.to_json()))) == schema


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(version="pair_v0"), "not 'pair_v1'"),
    (lambda d: d.pop("constants"), "missing"),
    (lambda d: d.update(extra=1), "unknown"),
    (lambda d: d.update(dim=35), "dim"),
    (lambda d: d["columns"].reverse(), "columns"),
    (lambda d: d["columns"].__setitem__(5, "value_next"), "columns"),
    (lambda d: d["constants"].update(snr_scale_db=20.0), "constants"),
    (lambda d: d["constants"].update(previous_age_cap_periods=3.0), "constants"),
    (lambda d: d.update(phase=False), "dim"),
    (lambda d: d.update(classes=["wide", "medium"]), "dim"),
    (lambda d: d.update(classes="wide"), "list"),
    (lambda d: d.update(classes=["wide", "wide", "narrow"]), "once"),
    (lambda d: d.update(phase=1), "bool"),
])
def test_a_schema_of_another_version_layout_or_scale_is_refused(edit, match):
    data = SCHEMA.to_json()
    edit(data)
    with pytest.raises((ValueError, TypeError), match=match):
        PairFeatureSchema.from_json(data)


def test_the_schema_refuses_what_is_no_link_class_tuple():
    for bad in ((), ("wide", ""), ("wide", "wide"), (1, 2)):
        with pytest.raises(ValueError):
            PairFeatureSchema(bad)
    for bad in ("wide", None, 3):
        with pytest.raises(TypeError):
            PairFeatureSchema(bad)
    with pytest.raises(TypeError, match="phase"):
        PairFeatureSchema(CLASSES, phase="yes")
    with pytest.raises(ValueError, match="pair_v2"):
        PairFeatureSchema(CLASSES, version="pair_v2")
    with pytest.raises(TypeError):
        PairFeatureSchema.from_json(["pair_v1"])
    assert PairFeatureSchema(["wide", "narrow"]).classes == ("wide", "narrow")


# --------------------------------------------------------------------------- #
# The covering classes
# --------------------------------------------------------------------------- #

def test_the_covering_classes_are_fx_candidates_in_link_order():
    """The mask's classes (other choices 2) are FX's candidate set: they hold b̄
    and FX's band, and they are the view's own ``covering``."""
    rng = random.Random(11)
    for view in _views(1, 300):
        got = covering_classes(view.arrival)
        assert got == view.covering
        need = set(view.arrival.committed_entry.targets)
        assert {c.name for c in got} == {c.name for c in view.arrival.classes
                                          if need <= set(c.targets)}
        assert view.arrival.committed_entry in got
        assert fastest_covering_class(view.arrival) in got
        assert [c.index for c in got] == sorted(c.index for c in got)
        # Out of link order in the view, still link order out.
        shuffled = list(view.arrival.classes)
        rng.shuffle(shuffled)
        again = ArrivalView(devices=view.arrival.devices, committed=view.arrival.committed,
                            classes=tuple(shuffled))
        assert covering_classes(again) == got
    with pytest.raises(TypeError):
        covering_classes(HAND)


# --------------------------------------------------------------------------- #
# Shapes, and what each column varies with
# --------------------------------------------------------------------------- #

def test_the_rows_are_one_per_pair_in_the_views_order():
    for view in _views(2, 200):
        schema = PairFeatureSchema.for_view(view)
        rows, pairs = pair_rows(view, schema)
        assert pairs == view.pairs
        assert rows.dtype == np.float64 and rows.shape == (len(pairs), schema.dim)
        assert rows.flags.c_contiguous and np.isfinite(rows).all()
        for r, (band, index) in enumerate(pairs):
            assert rows[r, schema.index(f"band[{band}]")] == 1.0
            assert rows[r, schema.index("home")] == (1.0 if index is None else 0.0)
            members = 0 if index is None else len(view.remainder[index].devices)
            assert rows[r, schema.index("members_next")] == members / view.demand
        again, _ = pair_rows(view, schema)
        assert again.tobytes() == rows.tobytes()
        assert not (rows == 0.0)[np.signbit(rows)].any()


def test_the_rows_refuse_another_links_view():
    with pytest.raises(ValueError, match="classes"):
        pair_rows(HAND, PairFeatureSchema(CLASS_SETS[4]))
    with pytest.raises(ValueError, match="classes"):
        pair_rows(HAND, PairFeatureSchema(("narrow", "medium", "wide")))
    with pytest.raises(TypeError, match="PairView"):
        pair_rows(HAND.arrival, SCHEMA)
    with pytest.raises(TypeError, match="PairFeatureSchema"):
        pair_rows(HAND, SCHEMA.to_json())


def test_each_column_varies_only_with_what_it_depends_on():
    """``band`` columns are one value per band, ``next`` columns one per
    candidate, ``state`` columns one per view; ``pair`` columns may vary with both."""
    seen_pair_variation = set()
    for view in _views(3, 300, n_stops=3) + _views(4, 50, n_stops=0):
        schema = PairFeatureSchema.for_view(view)
        rows, pairs = pair_rows(view, schema)
        for j, column in enumerate(schema.column_specs):
            groups = {}
            for (band, index), value in zip(pairs, rows[:, j]):
                key = {DEPENDS_BAND: band, DEPENDS_NEXT: index, DEPENDS_STATE: None}.get(
                    column.depends, (band, index))
                groups.setdefault(key, set()).add(value)
            assert all(len(values) == 1 for values in groups.values()), column
            if column.depends == "pair" and len(set(rows[:, j])) > 1:
                seen_pair_variation.add(column.name)
    assert seen_pair_variation == {"slack_next", "clock_left", "arrival_sin", "arrival_cos"}


def test_with_one_class_there_is_one_row_per_stop_and_the_band_block_is_constant():
    """T2, restated as a structural reduction (the spec, units table): on a
    one-class link, and on the rows of b̄ alone, the band block is one value."""
    for view in _views(5, 100, n_classes=1):
        schema = PairFeatureSchema.for_view(view)
        rows, pairs = pair_rows(view, schema)
        assert len(rows) == len(view.stops)
        assert [index for _, index in pairs] == [ctx.index for ctx in view.stops]
        assert (_col(rows, schema, "band[wide]") == 1.0).all()
        assert (_col(rows, schema, "gain_here") == 0.0).all()
        for name in BAND_BLOCK:
            assert len(set(_col(rows, schema, name))) == 1
    for view in _views(6, 100, n_classes=3, n_stops=4):
        schema = PairFeatureSchema.for_view(view)
        rows, pairs = pair_rows(view, schema)
        mine = [r for r, (band, _) in enumerate(pairs) if band == view.arrival.committed]
        assert len(mine) == 4
        block = [schema.index(f"band[{c}]") for c in CLASSES] + [schema.index(n)
                                                                  for n in BAND_BLOCK]
        assert len({tuple(rows[r, block]) for r in mine}) == 1
        assert rows[mine[0], schema.index("gain_here")] == 0.0


# --------------------------------------------------------------------------- #
# Each column's formula
# --------------------------------------------------------------------------- #

def test_each_column_is_its_formula_on_the_hand_worked_view():
    """Spec other choices 4: T = T_nom = 200 s, N = 10, S = 2, P_c = 60 s, b̄ =
    medium (dwell 12 s). Shared by every row: the reach and the pooled
    context, the offsets and the previous reading (75 s old)."""
    rows, pairs = pair_rows(HAND, SCHEMA)
    assert pairs == tuple((b, i) for b in CLASSES for i in range(4))
    shared = {
        "reach_here": 0.2, "energy_left": 0.75, "remainder_share": 0.5, "weight_share": 0.6,
        # A: 394 s after medium's service; B: -130 s; C exempt and D undated are out.
        "least_slack": -math.log1p(130 / 200),
        "offset[wide]": 0.5, "offset[medium]": -0.25, "offset[narrow]": 0.1,
        "prev_offset[wide]": -0.4, "prev_offset[medium]": 0.3, "prev_offset[narrow]": 0.05,
        "has_prev": 1.0, "prev_age": 1.25, "prev_sin": 1.0, "prev_cos": 0.0,
    }
    for r in range(len(pairs)):
        got = dict(zip(SCHEMA.columns, rows[r]))
        for name, want in shared.items():
            assert got[name] == pytest.approx(want, abs=1e-12), name
    wide_a = _row(HAND, SCHEMA, "wide", 0)
    turn = 2 * math.pi * (6 + 20) / 60
    assert wide_a == pytest.approx({
        **shared,
        "band[wide]": 1.0, "band[medium]": 0.0, "band[narrow]": 0.0,
        "snr_here": 12 / 30, "dwell_here": 6 / 200, "gain_here": 0.0, "travel": 20 / 200,
        "snr_next[wide]": 0.3, "snr_next[medium]": 0.5, "snr_next[narrow]": 0.7,
        "dwell_next": 14 / 200, "slack_next": math.log(3.0), "exempt_next": 0.0,
        "age_next": 1.5, "on_time_next": 0.25, "members_next": 0.2, "capped_next": 0.0,
        "home": 0.0, "clock_left": (300 - 100 - 6 - 20) / 250,
        "arrival_sin": math.sin(turn), "arrival_cos": math.cos(turn),
    }, abs=1e-12)
    narrow_b = _row(HAND, SCHEMA, "narrow", 1)
    assert narrow_b["gain_here"] == pytest.approx(0.1)
    assert narrow_b["snr_here"] == pytest.approx(0.8) and narrow_b["dwell_here"] == 0.15
    # Overdue: the slack keeps its sign, log-scaled and unclipped (critic A10 (iv)).
    assert narrow_b["slack_next"] == pytest.approx(-math.log1p((100 + 30 + 10 + 8) / 200))
    assert narrow_b["capped_next"] == 1.0 and narrow_b["exempt_next"] == 0.0
    assert narrow_b["clock_left"] == pytest.approx((300 - 100 - 30 - 10) / 250)
    assert _row(HAND, SCHEMA, "medium", 0)["slack_next"] == pytest.approx(math.log1p(394 / 200))
    # Exempt: no slack, and the flag; undated: likewise, without the cap flag.
    exempt = _row(HAND, SCHEMA, "medium", 2)
    assert (exempt["slack_next"], exempt["exempt_next"], exempt["capped_next"]) == (0.0, 1.0, 1.0)
    assert (exempt["age_next"], exempt["on_time_next"], exempt["travel"]) == (2.0, 1.0, 0.2)
    undated = _row(HAND, SCHEMA, "wide", 3)
    assert (undated["slack_next"], undated["exempt_next"], undated["capped_next"]) == (0.0, 1.0,
                                                                                       0.0)
    # The least slack is the least slack_next of b̄'s rows over the dated stops.
    medium = [_row(HAND, SCHEMA, "medium", i)["slack_next"] for i in (0, 1)]
    assert rows[0, SCHEMA.index("least_slack")] == min(medium)


def test_the_defaults_and_the_clips():
    """What a missing reading, reference or budget gives, and where the shares clip."""
    first = dataclasses.replace(HAND, previous_offsets_db=None, previous_age_s=None)
    row = _row(first, SCHEMA, "wide", 0)
    assert [row[f"prev_offset[{c}]"] for c in CLASSES] == [0.0, 0.0, 0.0]
    assert (row["has_prev"], row["prev_age"], row["prev_sin"], row["prev_cos"]) == (0, 0, 0, 0)
    # The age is capped at 4 P_c; its phase is the true one.
    old = _row(dataclasses.replace(HAND, previous_age_s=1000.0), SCHEMA, "wide", 0)
    assert old["prev_age"] == PREVIOUS_AGE_CAP_PERIODS == 4.0
    assert old["prev_sin"] == pytest.approx(math.sin(2 * math.pi * 1000 / 60))
    assert old["prev_cos"] == pytest.approx(math.cos(2 * math.pi * 1000 / 60))
    assert _row(dataclasses.replace(HAND, previous_age_s=240.0), SCHEMA, "wide", 0)[
        "prev_age"] == 4.0
    # No energy reference: a full battery (U0's hand-off); past twice it: clipped.
    assert _row(dataclasses.replace(HAND, energy_ref_j=None), SCHEMA, "wide", 0)[
        "energy_left"] == 1.0
    assert _row(dataclasses.replace(HAND, energy_ref_j=0.0), SCHEMA, "wide", 0)[
        "energy_left"] == 1.0
    assert _row(dataclasses.replace(HAND, energy_j=50000.0), SCHEMA, "wide", 0)[
        "energy_left"] == -1.0
    # No budget: the whole of it left; a budget end without a length reads in T.
    assert _row(dataclasses.replace(HAND, budget_end=None), SCHEMA, "narrow", 1)[
        "clock_left"] == 1.0
    assert _row(dataclasses.replace(HAND, budget_s=None), SCHEMA, "wide", 0)[
        "clock_left"] == pytest.approx((300 - 100 - 6 - 20) / 200)
    assert _row(dataclasses.replace(HAND, budget_end=T0 - 5000.0), SCHEMA, "wide", 0)[
        "clock_left"] == -1.0
    assert _row(dataclasses.replace(HAND, budget_end=T0 + 1e6), SCHEMA, "wide", 0)[
        "clock_left"] == 1.0
    # No cap: the age in missions; no committed weight: no share.
    assert _row(dataclasses.replace(HAND, cap_s=None), SCHEMA, "wide", 0)["age_next"] == 3.0
    weightless = dataclasses.replace(
        HAND, demand_weight=0.0,
        stops=tuple(dataclasses.replace(c, weight=0.0) for c in HAND_STOPS))
    assert _row(weightless, SCHEMA, "wide", 0)["weight_share"] == 0.0
    # Only dated, non-exempt stops have a slack to be least.
    free = dataclasses.replace(HAND, stops=(dataclasses.replace(HAND_STOPS[2], index=0),
                                            dataclasses.replace(HAND_STOPS[3], index=1)))
    assert _row(free, SCHEMA, "wide", 0)["least_slack"] == 0.0
    # The slack is not clipped: 500 T of slack, or 500 T overdue, reads as such.
    far = dataclasses.replace(HAND, stops=(
        dataclasses.replace(HAND_STOPS[0], stop=_wp(30, 0, "a1", "a2", deadline=CLOCK + 1e5)),
        dataclasses.replace(HAND_STOPS[1], stop=_wp(0, 10, "b1", deadline=CLOCK - 1e5)),
    ) + HAND_STOPS[2:])
    assert _row(far, SCHEMA, "wide", 0)["slack_next"] == pytest.approx(
        math.log1p((1e5 - 6 - 20 - 14) / 200))
    assert _row(far, SCHEMA, "wide", 1)["slack_next"] == pytest.approx(
        -math.log1p((1e5 + 6 + 10 + 8) / 200))
    assert _row(far, SCHEMA, "wide", 1)["least_slack"] < -6.0
    # A negative zero reads as zero, so equal rows are equal bytes.
    signed = pair_rows(dataclasses.replace(HAND, offsets_db=(-0.0, -0.0, -0.0)), SCHEMA)[0]
    assert not np.signbit(signed[:, SCHEMA.index("offset[wide]")]).any()


PHASE_TIMING = ("prev_age", "prev_sin", "prev_cos", "arrival_sin", "arrival_cos")


def test_the_phase_block_reads_the_views_period():
    """P_c is the view's own, the channel's configuration, which Study 5.6 sets
    per cell (decision 6 (a)), not the default 60 s: on the hand-worked view
    at 45 s the previous reading, 75 s old, is 5/3 periods old, at the phase
    4 pi / 3; each pair's arrival is (dwell + leg) / 45 s of a turn; the age's
    cap is 4 x 45 s; and no column outside the phase timing moves with P_c."""
    view = dataclasses.replace(HAND, period_s=45.0)
    rows, pairs = pair_rows(view, SCHEMA)
    at_60, _ = pair_rows(HAND, SCHEMA)
    timing = [SCHEMA.index(name) for name in PHASE_TIMING]
    others = [j for j in range(SCHEMA.dim) if j not in timing]
    assert np.array_equal(rows[:, others], at_60[:, others])
    for r, (band, index) in enumerate(pairs):
        row = dict(zip(SCHEMA.columns, rows[r]))
        assert row["prev_age"] == pytest.approx(5.0 / 3.0, abs=1e-12)
        assert row["prev_sin"] == pytest.approx(-math.sqrt(3.0) / 2.0, abs=1e-12)
        assert row["prev_cos"] == pytest.approx(-0.5, abs=1e-12)
        turn = 2 * math.pi * (HAND_ARRIVAL.entry(band).dwell_s + HAND_STOPS[index].travel_s) / 45
        assert row["arrival_sin"] == pytest.approx(math.sin(turn), abs=1e-12)
        assert row["arrival_cos"] == pytest.approx(math.cos(turn), abs=1e-12)
    # wide to A: 6 s of dwell and a 20 s leg, 26/45 of a turn.
    assert rows[0, SCHEMA.index("arrival_cos")] == pytest.approx(math.cos(2 * math.pi * 26 / 45))
    # 200 s is past 4 x 45 s but within 4 x 60 s; the phase stays the true age's.
    old = _row(dataclasses.replace(view, previous_age_s=200.0), SCHEMA, "wide", 0)
    assert old["prev_age"] == 4.0
    assert old["prev_sin"] == pytest.approx(math.sin(2 * math.pi * 200 / 45), abs=1e-12)
    assert _row(dataclasses.replace(HAND, previous_age_s=200.0), SCHEMA, "wide", 0)[
        "prev_age"] == pytest.approx(200 / 60)


def test_the_home_row():
    """Home, offered alone when the remainder is empty: its leg to the dock, no
    stop's columns, no remainder, and the landing's clock and phase."""
    view = dataclasses.replace(HAND, stops=(StopContext.home(25.0),))
    rows, pairs = pair_rows(view, SCHEMA)
    assert pairs == (("wide", None), ("medium", None), ("narrow", None))
    for (band, _), values in zip(pairs, rows):
        row = dict(zip(SCHEMA.columns, values))
        dwell = HAND_ARRIVAL.entry(band).dwell_s
        assert row["home"] == 1.0 and row["travel"] == 25 / 200
        for name in ("dwell_next", "slack_next", "exempt_next", "age_next", "on_time_next",
                     "members_next", "capped_next", "remainder_share", "least_slack",
                     "weight_share") + tuple(f"snr_next[{c}]" for c in CLASSES):
            assert row[name] == 0.0, name
        assert row["clock_left"] == pytest.approx((300 - 100 - dwell - 25) / 250)
        assert row["arrival_sin"] == pytest.approx(math.sin(2 * math.pi * (dwell + 25) / 60))
        assert row["arrival_cos"] == pytest.approx(math.cos(2 * math.pi * (dwell + 25) / 60))
        assert row["reach_here"] == 0.2 and row["has_prev"] == 1.0


# --------------------------------------------------------------------------- #
# Bounds and flags
# --------------------------------------------------------------------------- #

def _check_bounds(rows, schema):
    assert np.isfinite(rows).all()
    for j, column in enumerate(schema.column_specs):
        values = rows[:, j]
        if column.low is not None:
            assert values.min() >= column.low, (column.name, values.min())
        if column.high is not None:
            assert values.max() <= column.high, (column.name, values.max())
        if column.flag:
            assert set(values) <= {0.0, 1.0}, column.name


def test_every_column_is_finite_and_within_its_bounds():
    for extreme in (False, True):
        for view in _views(7 + extreme, 300, extreme=extreme):
            full = PairFeatureSchema.for_view(view)
            rows, _ = pair_rows(view, full)
            _check_bounds(rows, full)
            # The phase flag only appends the phase block.
            bare = PairFeatureSchema.for_view(view, phase=False)
            plain, _ = pair_rows(view, bare)
            _check_bounds(plain, bare)
            assert plain.tobytes() == np.ascontiguousarray(rows[:, :bare.dim]).tobytes()


def test_the_flags_are_the_views():
    for view in _views(9, 300):
        schema = PairFeatureSchema.for_view(view)
        rows, pairs = pair_rows(view, schema)
        assert (rows[:, [schema.index(f"band[{c}]") for c in schema.classes]].sum(axis=1)
                == 1.0).all()
        for r, (band, index) in enumerate(pairs):
            row = dict(zip(schema.columns, rows[r]))
            ctx = view.stops[0] if index is None else view.stops[index]
            assert row["home"] == float(ctx.is_home)
            assert row["capped_next"] == float(ctx.capped)
            undated = not ctx.is_home and not math.isfinite(ctx.stop.deadline_ts)
            assert row["exempt_next"] == float(ctx.exempt or undated)
            assert row["has_prev"] == float(view.previous_offsets_db is not None)


# --------------------------------------------------------------------------- #
# NEW devices
# --------------------------------------------------------------------------- #

def test_new_devices_read_the_mules_defaults():
    """A device the mule never heard from has the on-time prior 0.5
    (``features._on_time_rate``); in mission 1 the plan's age is 1 (``m -
    last_merged_round``); a beacon insert has no plan age and no weight. The
    columns pass them through, finite, and the trial's first arrival has no
    previous reading."""
    fresh = DeviceSchedulerState(device_id=DeviceID("n1"))
    assert _on_time_rate(fresh) == 0.5
    new_stop = _ctx(_wp(10, 10, "n1", "n2", deadline=CLOCK + 900), 0, travel=5, dwell=4,
                    snr=(10, 16, 22), age=1.0, on_time=_on_time_rate(fresh), weight=2.0)
    insert = _ctx(_wp(-10, 10, "x1", deadline=CLOCK + 900), 1, travel=5, dwell=4,
                  snr=(10, 16, 22), age=0.0, on_time=_on_time_rate(fresh), weight=0.0)
    view = dataclasses.replace(HAND, stops=(new_stop, insert), previous_offsets_db=None,
                               previous_age_s=None, demand=6, demand_weight=5.0)
    rows, _ = pair_rows(view, SCHEMA)
    _check_bounds(rows, SCHEMA)
    new, ins = _row(view, SCHEMA, "medium", 0), _row(view, SCHEMA, "medium", 1)
    assert (new["on_time_next"], new["age_next"]) == (0.5, 0.5)
    assert (ins["on_time_next"], ins["age_next"]) == (0.5, 0.0)
    assert new["weight_share"] == ins["weight_share"] == pytest.approx(2.0 / 5.0)
    assert new["has_prev"] == 0.0 and new["prev_age"] == 0.0


# --------------------------------------------------------------------------- #
# Causal phase features (loopback pair_q missions, U5's harness)
# --------------------------------------------------------------------------- #

#: The Phase 4 tests' 100 m field, and their layouts (U5's ``ref_layout``).
FIELD_M = device_spread_m(60.0, field_radius_m=100.0)


def _layout(k, n=12):
    seed = _u32(n, "t_nom", 1000 + k)
    xy = device_positions(n, seed, FIELD_M)
    return seed, tuple((f"d{i}", (x, y, 0.0)) for i, (x, y) in enumerate(xy))


def _jittery_spec(seed, period_s=None):
    """The Phase 5 cells' contact channel: jittery, 1 MB, ``replan`` with ``trim``;
    ``period_s`` sets the interference period P_c (None: the regime's own)."""
    period = {} if period_s is None else {"interference_period_s": period_s}
    return FerrySpec.from_config(rf_range_m=60.0, seed=seed, contact_band="wide",
                                 contact_regime="jittery", in_flight_response="replan",
                                 replan_fallback="trim", payload_bytes=1_000_000, **period)


class _NowOnly(ContactChannel):
    """The mule's contact channel, refusing a read beyond the mule's clock while
    the mule decides at an arrival (the spec's channel stub).

    Every time-dependent term goes through ``shadow_db`` and ``interference_db``
    (``snr_db`` reads both), so those two are the gate. Outside a decision the
    contact's own pricing reads each session's start ahead of the clock (the
    simulator prices the sessions before charging them), which is the
    simulator's realization and no decision's input, so it is let through.
    """

    def _at(self, t):
        if self.deciding:
            ahead = float(t) - self.clock()
            self.reads.append(ahead)
            if ahead > 0.0:
                self.violations.append(ahead)
                raise AssertionError(f"a decision read the channel {ahead} s beyond now")
        return t

    def shadow_db(self, t, link_key, *, stop_pos=None):
        return super().shadow_db(self._at(t), link_key, stop_pos=stop_pos)

    def interference_db(self, t, band):
        return super().interference_db(self._at(t), band)


def _now_only(spec, clock):
    stub = object.__new__(_NowOnly)
    stub.__dict__.update(spec.contact_channel.__dict__)
    stub.clock, stub.deciding, stub.reads, stub.violations = clock, False, [], []
    return dataclasses.replace(spec, contact_channel=stub), stub


class _Seen(LearnedPairScorer):
    """The learned score, keeping each view it scores."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.views = []

    def score(self, view, *, mask):
        self.views.append(view)
        return super().score(view, mask=mask)


class _ReadsAhead(_Seen):
    """A look-ahead feature: the realized interference at the next stop's arrival."""

    channel = None

    def score(self, view, *, mask):
        ahead = view.arrival.committed_entry.dwell_s + view.stops[0].travel_s + 1.0
        self.channel.interference_db(view.clock_s + ahead, view.arrival.committed)
        return super().score(view, mask=mask)


def _fly(scorer, k, *, missions=3, reads_ahead=False, period_s=None):
    """Fly layout ``k`` (N = 12, 120 s, S = 3) with ``scorer`` in the pair slot and
    the now-only channel gated by the mule's decisions; a fresh mule, so a trial."""
    seed, layout = _layout(k)
    clock = MissionClock()
    spec, stub = _now_only(_jittery_spec(seed, period_s), clock)
    if reads_ahead:
        scorer.channel = stub
    world = H.World(layout=layout, flaky={})
    options = PlanOptions(band_class_policy="search", member_admission="subset",
                          flight_slot="pair_q", cap=AgeCapSpec(s_missions=3))
    results = []
    with H.Patched(world.clock):
        sup = world.supervisor(
            world.mule_ids[0], sim=True, ferry=spec, mission_clock=clock,
            mission_budget_s=120.0, deadline_time_scale=1000.0, miss_priority=True,
            member_admission="subset", plan_mode="ferry", plan_options=options, t_nom_s=200.0,
            pair_slot=PairQSlot(scorer))
        decide = sup._ferry_pair_at_arrival

        def deciding(*args, **kwargs):
            stub.deciding = True
            try:
                return decide(*args, **kwargs)
            finally:
                stub.deciding = False

        sup._ferry_pair_at_arrival = deciding
        world.bootstrap()
        for _ in range(missions):
            results.append(sup.run_one_mission())
            world.clock.advance(GH.BETWEEN_MISSIONS_DT)
    return spec, stub, dict(layout), results


def _offsets_now(chan, view, positions):
    """``FerryRuntime.class_offsets_db`` recomputed: per class, the median over the
    stop's members of realized less mean SNR, each link differenced first."""
    stop = tuple(view.pose)
    out = []
    for name in CLASSES:
        out.append(float(statistics.median(
            chan.snr_db(view.clock_s, name, planar_distance_m(stop, positions[j]), link_key=j,
                        stop_pos=stop)
            - chan.mean_snr_db(name, planar_distance_m(stop, positions[j]))
            for j in view.arrival.devices)))
    return tuple(out)


#: The causal test's loopback trials, (layout, P_c): the channel's default
#: period, and 30 s, which the test needs only as a period other than the
#: default, so that every phase column is seen to read the configured period.
#: At N = 12, 30 s is about one arrival-to-arrival lag (lag / P_c near 1), the
#: ratio-1 aliasing point of critic A3, not a Study 5.6 period: those are 4 x and
#: 2 x the lag (``cells.STUDY_5_6_CELLS``; the orchestrator's resolution R22).
CAUSAL_TRIALS = ((0, None), (4, 30.0))


def test_the_phase_features_are_causal_and_carried_across_stops_and_sorties():
    """Decision 6 (a) and critic A3: the two readings are the offsets the mule
    observes at this arrival and at the trial's previous Pass-1 arrival, with
    that arrival's age; nothing a decision reads comes from beyond its own
    instant, and the next arrival's phase is the configured period's turn over
    the dwell and the leg, at the default period and at another."""
    schema = PairFeatureSchema(CLASSES)
    trials = []
    for k, period_s in CAUSAL_TRIALS:
        scorer = _Seen(PairQNet(schema.dim, TINY, seed=k), schema)
        spec, stub, positions, results = _fly(scorer, k, period_s=period_s)
        assert stub.violations == []
        assert stub.reads and set(stub.reads) == {0.0}
        per_mission = [len(r.pass_1_pairs or ()) for r in results]
        assert sum(per_mission) == len(scorer.views)
        assert sum(1 for n in per_mission if n) >= 2 and max(per_mission) >= 2
        trials.append((spec, scorer.views, positions, per_mission))
    assert [spec.contact_channel.interference_period_s for spec, *_ in trials] == [60.0, 30.0]
    for spec, views, positions, per_mission in trials:
        period = spec.contact_channel.interference_period_s
        for i, view in enumerate(views):
            assert view.period_s == period
            assert view.offsets_db == _offsets_now(spec.contact_channel, view, positions)
            rows, pairs = pair_rows(view, schema)
            row = dict(zip(schema.columns, rows[0]))
            for c, offset in zip(CLASSES, view.offsets_db):
                assert row[f"offset[{c}]"] == offset / OFFSET_SCALE_DB
            if i == 0:
                # The trial's first arrival: no previous reading (reset each trial).
                assert view.previous_offsets_db is None and view.previous_age_s is None
                assert (row["has_prev"], row["prev_age"]) == (0.0, 0.0)
            else:
                before = views[i - 1]
                assert view.previous_offsets_db == before.offsets_db
                age = view.clock_s - before.clock_s
                assert view.previous_age_s == age > 0.0
                assert row["has_prev"] == 1.0
                assert row["prev_age"] == min(age, 4 * period) / period
                assert row["prev_sin"] == math.sin(2 * math.pi * age / period)
                assert row["prev_cos"] == math.cos(2 * math.pi * age / period)
                for c, offset in zip(CLASSES, before.offsets_db):
                    assert row[f"prev_offset[{c}]"] == offset / OFFSET_SCALE_DB
            for r, (band, index) in enumerate(pairs):
                ctx = view.stops[0] if index is None else view.stops[index]
                turn = 2 * math.pi * (view.arrival.entry(band).dwell_s + ctx.travel_s) / period
                assert rows[r, schema.index("arrival_sin")] == math.sin(turn)
                assert rows[r, schema.index("arrival_cos")] == math.cos(turn)
        # Carried across sorties: each mission's first decision reads the last
        # decision of the mission before it.
        starts = np.cumsum([0] + per_mission[:-1])
        crossed = [s for s, n in zip(starts, per_mission) if n and s > 0]
        assert crossed
        for s in crossed:
            assert views[s].previous_offsets_db == views[s - 1].offsets_db


def test_a_scorer_that_reads_the_channel_ahead_is_caught():
    """The stub's teeth: a feature reading the realized channel at the next
    stop's arrival, an oracle the policy may not have, raises in the decision."""
    schema = PairFeatureSchema(CLASSES)
    scorer = _ReadsAhead(PairQNet(schema.dim, TINY, seed=0), schema)
    with pytest.raises(AssertionError, match="beyond now"):
        _fly(scorer, 0, missions=1, reads_ahead=True)
    assert scorer.channel.violations and min(scorer.channel.violations) > 0.0


# --------------------------------------------------------------------------- #
# A FerrySim sample
# --------------------------------------------------------------------------- #

#: The sample: the jittery family's cells, validation-stream episodes (as the
#: headroom report flew them), flown by fx_pair with a trainer at ε = 0, which
#: flies exactly as no trainer and keeps each decision's view.
SAMPLE = (("jit-n12-90", 4), ("jit-n12-180", 2), ("jit-n6-75", 3))
#: Rows the pooled sample needs; the N = 12 cells fly one more episode each
#: until it has them (at most SAMPLE_ROUNDS more), so the sample holds at any
#: budget the cells are re-pinned to.
SAMPLE_ROWS = 250
SAMPLE_ROUNDS = 10


@pytest.fixture(scope="module")
def ferrysim_sample():
    def fly(name, i):
        seed = FC.stream_seeds(FC.VAL_STREAM, name, 1, start=i)[0]
        return name, run_episode(FC.cell_named(name), seed, Policy.scripted("fx_pair"),
                                 trial_index=i, trainer=Trainer())

    out = [fly(name, i) for name, count in SAMPLE for i in range(count)]
    flown = dict(SAMPLE)
    for _ in range(SAMPLE_ROUNDS):
        if len(_sample_rows(out)) > SAMPLE_ROWS:
            break
        for name in (c.name for c in FC.STUDY_5_5_CELLS):
            out.append(fly(name, flown[name]))
            flown[name] += 1
    return out


def _sample_rows(sample, cell=None):
    return np.vstack([pair_rows(step.view, SCHEMA)[0] for name, ep in sample
                      if cell in (None, name) for steps in ep.steps for step in steps])


def test_no_column_is_constant_on_a_ferrysim_sample_but_the_declared_sparse_ones(
        ferrysim_sample):
    """Other choices 4: no column is constant across a FerrySim sample except as
    declared, which was the plan's complaint about today's features (L917).
    Pooled, every column varies; a decision-rich cell's few episodes may miss
    only a declared rare event. The N = 6 control is left out of the per-cell
    check: at its measured budgets (the re-pin of 5 Oct 2026: 75 and 150 s) one
    stop serves all six devices in most missions, so the in-flight slot mostly
    weighs the dock alone (at the knee, every pair it saw over 24 episodes was
    the dock), which is the control's role (where looking ahead cannot matter)."""
    assert set(SPARSE_COLUMNS) <= set(SCHEMA.columns)
    rows = _sample_rows(ferrysim_sample)
    assert len(rows) > SAMPLE_ROWS
    constant = [name for j, name in enumerate(SCHEMA.columns) if len(set(rows[:, j])) == 1]
    assert constant == []
    for name in (c.name for c in FC.STUDY_5_5_CELLS):
        sub = _sample_rows(ferrysim_sample, name)
        constant = {c for j, c in enumerate(SCHEMA.columns) if len(set(sub[:, j])) == 1}
        assert constant <= set(SPARSE_COLUMNS), (name, constant)


def test_ferrysim_rows_are_in_bounds_and_mission_1_reads_new_devices(ferrysim_sample):
    _check_bounds(_sample_rows(ferrysim_sample), SCHEMA)
    for name, ep in ferrysim_sample:
        cap = FC.cell_named(name).cap_s
        first = True
        for m, steps in enumerate(ep.steps):
            for step in steps:
                rows, pairs = pair_rows(step.view, SCHEMA)
                assert _col(rows, SCHEMA, "has_prev")[0] == (0.0 if first else 1.0)
                first = False
                if m == 0:
                    # Mission 1: every device is NEW, its age 1 and its rate the prior.
                    stops = _col(rows, SCHEMA, "home") == 0.0
                    assert set(_col(rows, SCHEMA, "on_time_next")[stops]) <= {0.5}
                    assert set(_col(rows, SCHEMA, "age_next")[stops]) <= {1.0 / cap}


def test_the_views_n_is_the_rewards(ferrysim_sample):
    """members_next and the shares are per device of N, the reward's N (decision
    4: one device is worth 1/N), so a row's units are the reward's."""
    for _, ep in ferrysim_sample:
        for sortie, steps in zip(ep.sorties, ep.steps):
            for step in steps:
                assert step.view.demand == sortie.n_demand


def test_a_cells_interference_period_reaches_the_phase_columns():
    """Study 5.6's knob (decision 6 (a); U8a's ``FerryCell.interference_period_s``):
    a FerrySim cell at P_c = 30 s flies views of that period, and every phase
    column is its formula at 30 s, the previous reading being the last
    decision's across the trial's missions, its age capped at 4 x 30 s."""
    def flown(base, index):                      # a decision-rich episode, at any budget
        cell = dataclasses.replace(base, name=f"{base.name}-p30", interference_period_s=30.0)
        seed = FC.stream_seeds(FC.VAL_STREAM, cell.name, 1, start=index)[0]
        ep = run_episode(cell, seed, Policy.scripted("fx_pair"), trainer=Trainer())
        return ep, [step.view for steps in ep.steps for step in steps]

    for ep, views in (flown(base, i) for i in range(12) for base in FC.STUDY_5_5_CELLS):
        if len(views) >= 8 and sum(1 for steps in ep.steps if steps) >= 3:
            break
    assert len(views) >= 8 and sum(1 for steps in ep.steps if steps) >= 3
    assert {view.period_s for view in views} == {30.0}
    capped = 0
    for i, view in enumerate(views):
        rows, pairs = pair_rows(view, SCHEMA)
        row = dict(zip(SCHEMA.columns, rows[0]))
        if i == 0:
            assert view.previous_age_s is None and row["has_prev"] == 0.0
        else:
            age = view.clock_s - views[i - 1].clock_s
            assert view.previous_age_s == age > 0.0
            assert view.previous_offsets_db == views[i - 1].offsets_db
            assert row["prev_age"] == min(age, 120.0) / 30.0
            assert row["prev_sin"] == math.sin(2 * math.pi * age / 30.0)
            assert row["prev_cos"] == math.cos(2 * math.pi * age / 30.0)
            capped += age > 120.0
        for r, (band, index) in enumerate(pairs):
            ctx = view.stops[0] if index is None else view.stops[index]
            turn = 2 * math.pi * (view.arrival.entry(band).dwell_s + ctx.travel_s) / 30.0
            assert rows[r, SCHEMA.index("arrival_sin")] == math.sin(turn)
            assert rows[r, SCHEMA.index("arrival_cos")] == math.cos(turn)
    assert capped >= 1


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #

def test_the_adapter_is_the_slots_learned_pair_scorer():
    net = PairQNet(SCHEMA.dim, TINY, seed=3)
    scorer = LearnedPairScorer(net, SCHEMA)
    assert isinstance(scorer, PairScorer)
    assert scorer.q_values is True and scorer.name == "pair_v1"
    assert scorer.net is net and scorer.schema == SCHEMA and scorer.manifest is None
    rows, pairs = pair_rows(HAND, SCHEMA)
    got = scorer.score(HAND, mask=(True,) * len(pairs))
    assert type(got) is tuple and all(type(v) is float for v in got)
    assert got == tuple(float(v) for v in net.q(rows))
    # The mask is the slot's to apply, not the scorer's.
    assert scorer.score(HAND, mask=(False,) * len(pairs)) == got
    # The live online network: one learner update moves the online weights
    # only (the target syncs every 500 updates), and the behaviour policy
    # (critic C4) is scored by them at once, not by the lagging target.
    net.update(PairBatch.of([PairTransition(x, float(r % 3), True) for r, x in enumerate(rows)]))
    online = tuple(float(v) for v in net.q(rows))
    assert net.updates == 1 and net.syncs == 0
    assert scorer.score(HAND, mask=()) == online != got
    assert online != tuple(float(v) for v in net.q_target(rows))
    # Weights set by any route are scored at once too.
    net.set_weights({name: value * 2.0 for name, value in net.weights().items()})
    assert scorer.score(HAND, mask=()) == tuple(float(v) for v in net.q(rows)) != online
    assert "pair_v1" in repr(scorer)


def test_the_adapter_refuses_what_is_not_one_schemas():
    with pytest.raises(ValueError, match="wide"):
        LearnedPairScorer(PairQNet(SCHEMA.dim + 1, TINY, seed=0), SCHEMA)
    with pytest.raises(ValueError, match="wide"):
        LearnedPairScorer(PairQNet(SCHEMA.dim, TINY, seed=0), PairFeatureSchema(CLASSES, False))
    with pytest.raises(TypeError, match="PairQNet"):
        LearnedPairScorer(object(), SCHEMA)
    with pytest.raises(TypeError, match="PairFeatureSchema"):
        LearnedPairScorer(PairQNet(SCHEMA.dim, TINY, seed=0), SCHEMA.to_json())
    with pytest.raises(TypeError, match="manifest"):
        LearnedPairScorer(PairQNet(SCHEMA.dim, TINY, seed=0), SCHEMA, manifest=["x"])
    other = PairFeatureSchema(CLASS_SETS[4])
    scorer = LearnedPairScorer(PairQNet(other.dim, TINY, seed=0), other)
    with pytest.raises(ValueError, match="classes"):
        scorer.score(HAND, mask=())
    kept = LearnedPairScorer(PairQNet(SCHEMA.dim, TINY, seed=0), SCHEMA,
                             manifest={"seeds": {"run": [1]}})
    copy = kept.manifest
    copy["seeds"]["run"].append(2)
    assert kept.manifest == {"seeds": {"run": [1]}}


def test_the_slot_flies_the_masked_argmax_of_the_learned_q():
    """The adapter in U3's slot: the pick is the masked argmax of the network's
    Q (ties to the lowest row), recorded as Q values under the schema's name."""
    rng = random.Random(21)
    flown = 0
    for i, view in enumerate(_views(12, 120, n_classes=3)):
        net = PairQNet(SCHEMA.dim, TINY, seed=i)
        slot = PairQSlot(LearnedPairScorer(net, SCHEMA))
        mask = tuple(rng.random() < 0.6 for _ in view.pairs)
        allowed = {pair for pair, ok in zip(view.pairs, mask) if ok}
        admitted = set(view.arrival.devices) | {d for wp in view.remainder for d in wp.devices}
        choice = slot.pair_at_arrival(view, fits_pair=lambda b, k: (b, k) in allowed,
                                      pass_kind=COLLECT, admitted=admitted)
        q = net.q(pair_rows(view, SCHEMA)[0])
        assert choice.record["scorer"] == "pair_v1"
        if any(mask):
            row = masked_argmax(q, np.array(mask))
            assert choice.pair == view.pairs[row] and choice.fallback is None
            assert choice.q == float(q[row]) and choice.record["q"] == round(float(q[row]), 6)
            flown += 1
        else:
            assert choice.fallback == "mask_empty"
    assert flown > 90


def _learned_factory():
    return LearnedPairScorer(PairQNet(SCHEMA.dim, TINY, seed=7), SCHEMA)


def test_the_learned_score_flies_a_ferrysim_episode_deterministically():
    """U8a's hand-off: a learned policy enters FerrySim as ``Policy(label,
    scorer=factory)``; the same episode twice gives the same records."""
    cell = FC.cell_named("jit-n12-90")
    seed = FC.stream_seeds(FC.VAL_STREAM, cell.name, 1)[0]
    policy = Policy("pair_v1", scorer=_learned_factory)
    first = run_episode(cell, seed, policy)
    again = run_episode(cell, seed, policy)
    records = [r for mission in first.pair_records if mission for r in mission]
    assert first.decisions == len(records) > 3
    assert {r["scorer"] for r in records} == {"pair_v1"}
    assert all(r["q"] is not None and r["q_fx"] is not None for r in records)
    assert first.summary() == again.summary()


# --------------------------------------------------------------------------- #
# The slot from a checkpoint (resolution R8)
# --------------------------------------------------------------------------- #

PROVENANCE = {"reward": {}, "training": {}, "seeds": {}, "cell_family": None,
              "cell_family_sha256": None, "trainer_commit": None, "dirty": False,
              "episodes_trained": 0, "validation": [], "held_out": None}


def _save(directory, *, schema=SCHEMA, schema_json=None, classes=None, kind="pair_q",
          name="ck", seed=5):
    data = schema.to_json() if schema_json is None else schema_json
    net = PairQNet(data["dim"], TINY, seed=seed)
    path = Path(directory) / f"{name}.npz"
    sha = net.save(path, kind=kind, purpose="bootstrap", schema=data,
                   classes=list(schema.classes if classes is None else classes),
                   provenance=PROVENANCE)
    return path, sha, net


def test_build_pair_slot_flies_a_verified_checkpoint(tmp_path):
    path, sha, net = _save(tmp_path)
    slot = build_pair_slot(str(path), expect_sha256=sha, classes=list(CLASSES))
    assert isinstance(slot, PairQSlot) and slot.name == "pair_q"
    scorer = slot.scorer
    assert isinstance(scorer, LearnedPairScorer) and scorer.schema == SCHEMA
    assert all(np.array_equal(scorer.net.weights()[k], v) for k, v in net.weights().items())
    manifest = scorer.manifest
    assert manifest["sha256"] == sha and manifest["kind"] == "pair_q"
    assert manifest["schema"] == SCHEMA.to_json() and manifest["purpose"] == "bootstrap"
    memory = LearnedPairScorer(net, SCHEMA)
    for view in _views(13, 50, n_classes=3):
        assert scorer.score(view, mask=()) == memory.score(view, mask=())
    loaded = load_pair_scorer(path, expect_sha256=sha, classes=CLASSES, phase=True)
    assert loaded.schema == SCHEMA


def test_a_phase_free_checkpoint_reads_phase_free_rows(tmp_path):
    bare = PairFeatureSchema(CLASSES, phase=False)
    path, sha, _ = _save(tmp_path, schema=bare)
    scorer = load_pair_scorer(path, expect_sha256=sha, classes=CLASSES)
    assert scorer.schema == bare and scorer.net.feature_dim == 24
    assert len(scorer.score(HAND, mask=())) == len(HAND.pairs)
    with pytest.raises(CheckpointError, match="phase block is off"):
        load_pair_scorer(path, expect_sha256=sha, classes=CLASSES, phase=True)


def _tampered_columns():
    data = SCHEMA.to_json()
    data["columns"][3], data["columns"][4] = data["columns"][4], data["columns"][3]
    return data


@pytest.mark.parametrize("case, match", [
    ("wrong_sha", "not the expected"),
    ("chen_dqn", "a 'chen_dqn' checkpoint, not 'pair_q'"),
    ("four_classes", "and the link's are"),
    ("reordered", "and the link's are"),
    ("phase_asked_off", "phase block is on, and off was asked"),
    ("other_version", "'pair_v0' schema is not 'pair_v1'"),
    ("tampered_columns", r"\['columns'\] are not this module's"),
    ("other_scale", r"\['constants'\] are not this module's"),
    ("no_manifest", "no manifest"),
])
def test_the_loader_refuses_a_checkpoint_the_link_would_misread(tmp_path, case, match):
    """Other choices 4 and 6: the schema records the class tuple and the phase
    flag, and the loader refuses a mismatch; so are a wrong sha, another kind
    and a schema of another layout or scale. Each is a ``CheckpointError``
    (a ValueError), the one error a mule refuses to run on."""
    classes, phase, sha = CLASSES, None, None
    if case == "chen_dqn":
        path, sha, _ = _save(tmp_path, schema_json={"version": "e3_v1", "dim": 9},
                             classes=["wide"], kind="chen_dqn")
    elif case == "other_version":
        data = SCHEMA.to_json()
        data["version"] = "pair_v0"
        path, sha, _ = _save(tmp_path, schema_json=data)
    elif case == "tampered_columns":
        path, sha, _ = _save(tmp_path, schema_json=_tampered_columns())
    elif case == "other_scale":
        data = SCHEMA.to_json()
        data["constants"]["offset_scale_db"] = 5.0
        path, sha, _ = _save(tmp_path, schema_json=data)
    else:
        path, sha, _ = _save(tmp_path)
    if case == "wrong_sha":
        sha = "0" * 64
    elif case == "four_classes":
        classes = CLASS_SETS[4]
    elif case == "reordered":
        classes = ("narrow", "medium", "wide")
    elif case == "phase_asked_off":
        phase = False
    elif case == "no_manifest":
        manifest_path(path).unlink()
    with pytest.raises(CheckpointError, match=match):
        build_pair_slot(path, expect_sha256=sha, classes=classes, phase=phase)


@pytest.mark.parametrize("case, match", [
    ("not_npz", r"\.npz file"),
    ("nothing_there", "not found"),
    ("arrays_gone", "not found"),
    ("a_directory", "not found"),
])
def test_the_loader_refuses_a_path_with_no_checkpoint_file(tmp_path, case, match):
    """What a config path may name (``mule_config_errors`` asks only for a
    non-empty string) but holds no checkpoint is refused like the rest: a
    ``CheckpointError``, so a mule that catches it refuses to run cleanly,
    rather than meeting the reader's FileNotFoundError, an OSError."""
    path, sha, _ = _save(tmp_path)
    if case == "not_npz":
        path = path.with_suffix(".json")  # its manifest: a file, but no checkpoint
    elif case == "nothing_there":
        path = tmp_path / "nowhere.npz"
    elif case == "arrays_gone":
        path.unlink()  # the manifest stays
    elif case == "a_directory":
        path = tmp_path / "adir.npz"
        path.mkdir()
    for load in (load_pair_scorer, build_pair_slot):
        with pytest.raises(CheckpointError, match=match):
            load(path, expect_sha256=sha, classes=CLASSES)


def test_a_checkpoint_gone_between_the_loaders_two_reads_is_refused(tmp_path, monkeypatch):
    """The checkpoint is read twice (verified whole, then loaded against the
    schema it names): a file removed in between is refused like a missing one."""
    path, sha, _ = _save(tmp_path)

    def verify_then_remove(where):
        manifest = verify_checkpoint(where)
        Path(where).unlink()
        return manifest

    monkeypatch.setattr("hermes.scheduler.selector.pair_features.verify_checkpoint",
                        verify_then_remove)
    with pytest.raises(CheckpointError, match="not found"):
        load_pair_scorer(path, expect_sha256=sha, classes=CLASSES)


def test_the_loader_refuses_bad_arguments_before_reading_any_file(tmp_path):
    """Arguments that no valid config or link holds are the caller's bugs: they
    raise before the path is read, whatever is there, and never as a
    ``CheckpointError``, so a mule tells them from a refused checkpoint."""
    path, sha, _ = _save(tmp_path)
    for where in (path, tmp_path / "nowhere.npz", tmp_path / "ck.bin"):
        with pytest.raises(TypeError, match="phase"):
            load_pair_scorer(where, expect_sha256=sha, classes=CLASSES, phase="on")
        with pytest.raises(TypeError, match="class names"):
            load_pair_scorer(where, expect_sha256=sha, classes="wide")
        for classes in ((), ("wide", "wide")):
            with pytest.raises(ValueError) as caught:
                load_pair_scorer(where, expect_sha256=sha, classes=classes)
            assert not isinstance(caught.value, CheckpointError)
        for bad in ("ABC", None, "A" * 64, sha + "0"):
            with pytest.raises(ValueError, match="64 lowercase hex") as caught:
                load_pair_scorer(where, expect_sha256=bad, classes=CLASSES)
            assert not isinstance(caught.value, CheckpointError)
    for bad in (None, 3, b"ck.npz"):
        with pytest.raises(TypeError):
            load_pair_scorer(bad, expect_sha256=sha, classes=CLASSES)


def test_the_slot_built_from_a_checkpoint_flies_as_the_network_it_saved(tmp_path):
    """A loopback trial flown by ``build_pair_slot``'s slot equals the same
    trial flown by the network that was saved, record for record."""
    schema = PairFeatureSchema(CLASSES)
    path, sha, net = _save(tmp_path, seed=11)
    flights = []
    for scorer in (_Seen(net, schema),
                   _Seen(load_pair_scorer(path, expect_sha256=sha, classes=CLASSES).net,
                         schema)):
        _, _, _, results = _fly(scorer, 6, missions=2)
        flights.append([r.pass_1_pairs for r in results])
    assert flights[0] == flights[1]
    assert sum(len(m or ()) for m in flights[0]) >= 3


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #

_LOADS = r"""
import json, sys
import hermes.scheduler.selector
before = "hermes.scheduler.selector.pair_features" in sys.modules
import hermes.scheduler.selector.pair_features
print(json.dumps({
    "selector_loads_it": before,
    "runtime": sorted(n for n in sys.modules if n.split(".")[:2] in (
        ["hermes", "l1"], ["hermes", "mule"], ["hermes", "mission"]) or n.split(".")[0] in (
        "experiments", "tests")),
    "slot": "hermes.scheduler.policies.pair_slot" in sys.modules,
}))
"""


def test_the_module_loads_no_runtime_and_the_slot_only_when_built():
    """Plan mode only, scheduler-side (the spec's layering): the selector
    package does not load it, it loads nothing from the runtime or
    ``experiments``, and the slot's module only inside ``build_pair_slot``."""
    out = subprocess.run([sys.executable, "-c", _LOADS], cwd=str(REPO), capture_output=True,
                         text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {
        "selector_loads_it": False, "runtime": [], "slot": False}
