"""FeRRy Phase 3, unit U5: the ferry contact plan (design section 4.2).

``ContactPlan`` holds what the host needs for one stop: who is solicited (the
gate: within R_planar(b) and at or above the SNR floor), each member's SNR, the
band's airtime and the mission clock. It imports nothing from ``hermes.l1``;
these tests drive it with plain callables, plus one check against U3's real
``ContactLink`` so the gate and S3a's range agree at the wide class.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from hermes.mission import contact_plan as cp
from hermes.mission.contact_plan import ContactPlan, planar_distance_m
from hermes.types import DeviceID

A, B, C, D = (DeviceID(x) for x in ("a", "b", "c", "d"))
T0 = 1_000_000.0
FLOOR = -6.7


class Clock:
    """A settable stand-in for the mission clock (``__call__`` + ``advance``)."""

    def __init__(self, t: float = T0) -> None:
        self.t = t
        self.calls = []

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float, kind: str) -> float:
        self.calls.append((dt, kind))
        self.t += dt
        return self.t


def _dwell(nbytes: int, snr: float):
    """Toy airtime: 1 Mb/s per dB above the floor (+1), None below it."""
    if snr < FLOOR:
        return None
    return 8.0 * nbytes / (1e6 * (snr - FLOOR + 1.0))


def _banded(clock, **kw):
    args = dict(
        clock=clock, advance=clock.advance, band="wide", band_index=0,
        stop=(0.0, 0.0, 0.0),
        positions={A: (10.0, 0.0, 0.0), B: (0.0, 59.0, 0.0), C: (61.0, 0.0, 0.0),
                   D: (3.0, 4.0, 0.0)},
        range_planar_m=60.0,
        snr_fn=lambda j, d, t: {A: 5.0, B: 0.0, C: 12.0, D: -9.0}[j] - (t - T0),
        snr_floor_db=FLOOR, dwell_fn=_dwell,
    )
    args.update(kw)
    return ContactPlan.at_arrival([A, B, C, D], **args)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #

def test_the_gate_keeps_members_in_range_and_above_the_floor():
    clock = Clock()
    plan = _banded(clock)
    assert plan.arrival_ts == T0
    # C is 61 m away (out of range although its SNR is fine); D is at 5 m but
    # below the floor. Both are unreachable, in member order.
    assert plan.targets == (A, B)
    assert plan.unreachable == (C, D)
    assert dict(plan.snr_db) == {A: 5.0, B: 0.0, C: 12.0, D: -9.0}
    assert clock.calls == []            # building a plan never charges the clock


def test_the_range_and_floor_tests_are_inclusive():
    clock = Clock()
    plan = ContactPlan.at_arrival(
        [A, B], clock=clock, advance=clock.advance, band="wide", band_index=0,
        stop=(0.0, 0.0, 0.0), positions={A: (60.0, 0.0, 0.0), B: (1.0, 0.0, 0.0)},
        range_planar_m=60.0, snr_fn=lambda j, d, t: FLOOR if j == B else 3.0,
        snr_floor_db=FLOOR, dwell_fn=_dwell,
    )
    assert plan.targets == (A, B)       # exactly at the range, exactly at the floor


def test_no_range_means_only_the_snr_gate():
    clock = Clock()
    plan = _banded(clock, range_planar_m=None)
    assert plan.targets == (A, B, C)
    assert plan.unreachable == (D,)


def test_the_snr_function_gets_each_members_planar_distance_and_the_arrival_time():
    clock = Clock(T0 + 12.5)
    seen = []

    def snr_fn(j, d, t):
        seen.append((j, d, t))
        return 1.0

    ContactPlan.at_arrival(
        [A, D], clock=clock, advance=clock.advance, band="wide", band_index=0,
        stop=(1.0, 1.0, 0.0), positions={A: (4.0, 5.0, 0.0), D: (1.0, 1.0, 0.0)},
        range_planar_m=60.0, snr_fn=snr_fn, snr_floor_db=FLOOR, dwell_fn=_dwell,
    )
    assert seen == [(A, 5.0, T0 + 12.5), (D, 0.0, T0 + 12.5)]


def test_the_range_metric_is_s3a_s_so_the_wide_gate_never_drops_an_s3a_member():
    """At wide, R_planar(wide) == rf_range_m exactly (U3), and the gate uses S3a's
    own float operations: a member S3a admitted is in range bit for bit."""
    from hermes.scheduler.stages import s3a_cluster
    import random

    rng = random.Random(5)
    for _ in range(2000):
        a = (rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0)
        b = (rng.uniform(-100, 100), rng.uniform(-100, 100), 0.0)
        assert planar_distance_m(a, b) == s3a_cluster._distance(a, b)


def test_the_gate_agrees_with_the_real_contact_link_at_wide():
    link_mod = pytest.importorskip("hermes.l1.contact_link")
    link = link_mod.ContactLink(anchor_planar_m=60.0)
    clock = Clock()
    rng = link.range_planar_m("wide")
    assert rng == 60.0
    positions = {A: (60.0, 0.0, 0.0), B: (36.0, 48.0, 0.0), C: (60.0000001, 0.0, 0.0)}
    plan = ContactPlan.at_arrival(
        [A, B, C], clock=clock, advance=clock.advance, band="wide",
        band_index=link.index("wide"), stop=(0.0, 0.0, 0.0), positions=positions,
        range_planar_m=rng, snr_fn=lambda j, d, t: link.mean_snr_db("wide", d),
        snr_floor_db=link.snr_floor_db,
        dwell_fn=lambda n, s: link.dwell_s(n, "wide", s),
    )
    # A and B sit exactly on R_planar(wide) (60 m); C is just past it.
    assert plan.targets == (A, B)
    assert plan.unreachable == (C,)
    for did in (A, B):
        assert link.in_range("wide", planar_distance_m((0.0, 0.0, 0.0), positions[did]))


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #

def test_snr_at_reads_each_members_link_at_the_asked_time():
    clock = Clock()
    plan = _banded(clock)
    assert plan.snr_at(A, T0) == 5.0
    assert plan.snr_at(A, T0 + 2.0) == 3.0          # C2: the SNR moves with time
    assert plan.snr_at(B, T0 + 0.5) == -0.5


def test_snr_at_reads_every_member_at_its_own_distance():
    """Critic C2: the bound time function keeps each member's own planar
    distance (A 10 m, B 59 m, C 61 m, D 5 m), whoever is asked about first."""
    clock = Clock()
    seen = []

    def snr_fn(j, d, t):
        seen.append((j, d, t))
        return 20.0 - 0.1 * d - (t - T0)

    plan = _banded(clock, snr_fn=snr_fn)
    del seen[:]
    t = T0 + 2.5
    for j, d in ((D, 5.0), (C, 61.0), (A, 10.0), (B, 59.0)):
        assert plan.snr_at(j, t) == snr_fn(j, d, t)
    assert seen[::2] == [(D, 5.0, t), (C, 61.0, t), (A, 10.0, t), (B, 59.0, t)]


def test_snr_at_without_a_time_function_is_the_arrival_reading():
    clock = Clock()
    plan = ContactPlan(
        arrival_ts=T0, targets=(A,), clock=clock, advance=clock.advance,
        band="wide", band_index=0, snr_db={A: 4.0}, snr_floor_db=FLOOR,
        dwell_fn=_dwell,
    )
    assert plan.snr_at(A, T0 + 100.0) == 4.0


def test_session_bytes_measured_or_declared_per_direction():
    clock = Clock()
    measured = _banded(clock)
    assert measured.session_bytes(37_576, 2) == 37_576
    declared = _banded(clock, payload_bytes=1_000_000)
    assert declared.session_bytes(37_576, 2) == 2_000_000
    assert declared.session_bytes(18_820, 1) == 1_000_000


def test_session_dwell_below_the_floor_is_priced_at_the_floor_never_infinite():
    clock = Clock()
    plan = _banded(clock)
    assert plan.session_dwell_s(1000, 3.3) == _dwell(1000, 3.3)
    at_floor = _dwell(1000, FLOOR)
    assert plan.session_dwell_s(1000, FLOOR - 20.0) == at_floor
    assert math.isfinite(at_floor)


def test_dwell_s_is_the_bands_airtime_as_given_none_below_the_floor():
    clock = Clock()
    plan = _banded(clock)
    assert plan.dwell_s(1000, 3.3) == _dwell(1000, 3.3)
    assert plan.dwell_s(1000, FLOOR - 1.0) is None
    bandless = ContactPlan.at_arrival([A], clock=clock, advance=clock.advance)
    with pytest.raises(ValueError, match="needs a band"):
        bandless.dwell_s(1000, 3.3)
    with pytest.raises(ValueError, match="needs a band"):
        bandless.session_dwell_s(1000, 3.3)


def test_session_dwell_refuses_a_bad_dwell_function():
    clock = Clock()
    for bad in (lambda n, s: None, lambda n, s: -1.0, lambda n, s: math.inf,
                lambda n, s: float("nan")):
        plan = _banded(clock, dwell_fn=bad)
        with pytest.raises((ValueError, TypeError)):
            plan.session_dwell_s(1000, 3.0)


# --------------------------------------------------------------------------- #
# The clock without a band (critic A1)
# --------------------------------------------------------------------------- #

def test_without_a_band_every_member_is_a_target_and_there_is_no_link():
    clock = Clock(T0 + 3.0)
    plan = ContactPlan.at_arrival([A, B], clock=clock, advance=clock.advance,
                                  listen_s=1.0, session_time_s=1.0)
    assert plan.arrival_ts == T0 + 3.0
    assert plan.targets == (A, B) and plan.unreachable == ()
    assert plan.band is None and plan.band_index is None
    assert dict(plan.snr_db) == {}
    assert plan.snr_at(A, T0) is None
    assert plan.session_time_s == 1.0


@pytest.mark.parametrize("extra", [
    {"band_index": 0}, {"range_planar_m": 60.0}, {"snr_floor_db": FLOOR},
    {"dwell_fn": _dwell}, {"snr_fn": lambda j, d, t: 0.0},
    {"positions": {A: (0.0, 0.0, 0.0)}},
])
def test_without_a_band_link_arguments_are_refused(extra):
    clock = Clock()
    with pytest.raises(ValueError, match="without a band"):
        ContactPlan.at_arrival([A], clock=clock, advance=clock.advance, **extra)


def test_a_bandless_plan_refuses_an_unreachable_member_or_an_snr():
    clock = Clock()
    with pytest.raises(ValueError, match="no gate"):
        ContactPlan(arrival_ts=T0, targets=(A,), unreachable=(B,), clock=clock,
                    advance=clock.advance)
    with pytest.raises(ValueError, match="no link fields"):
        ContactPlan(arrival_ts=T0, targets=(A,), clock=clock, advance=clock.advance,
                    snr_db={A: 1.0})


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validation_refuses_malformed_plans():
    clock = Clock()
    base = dict(clock=clock, advance=clock.advance)
    with pytest.raises(ValueError, match="at least one member"):
        ContactPlan(arrival_ts=T0, targets=(), **base)
    with pytest.raises(ValueError, match="disjoint"):
        ContactPlan(arrival_ts=T0, targets=(A, A), **base)
    with pytest.raises(ValueError, match="finite"):
        ContactPlan(arrival_ts=math.nan, targets=(A,), **base)
    with pytest.raises(TypeError):
        ContactPlan(arrival_ts=T0, targets=(A,), clock=None, advance=clock.advance)
    with pytest.raises(ValueError, match="outside the contact"):
        ContactPlan(arrival_ts=T0, targets=(A,), drop_uplink={B}, **base)
    for name in ("listen_s", "session_time_s"):
        with pytest.raises(ValueError):
            ContactPlan(arrival_ts=T0, targets=(A,), **{name: -1.0}, **base)
    with pytest.raises(TypeError):
        ContactPlan(arrival_ts=T0, targets=(A,), payload_bytes=True, **base)
    with pytest.raises(ValueError):
        ContactPlan(arrival_ts=T0, targets=(A,), payload_bytes=-1, **base)


def test_a_banded_plan_needs_its_link_numbers():
    clock = Clock()
    good = dict(arrival_ts=T0, targets=(A,), clock=clock, advance=clock.advance,
                band="wide", band_index=0, snr_db={A: 1.0}, snr_floor_db=FLOOR,
                dwell_fn=_dwell)
    ContactPlan(**good)
    for key, value, err in (
        ("band_index", None, ValueError), ("band_index", -1, ValueError),
        ("band_index", True, ValueError), ("snr_floor_db", None, TypeError),
        ("dwell_fn", None, TypeError), ("snr_db", {}, ValueError),
        ("snr_db", {A: math.nan}, ValueError), ("band", 3, TypeError),
    ):
        with pytest.raises(err):
            ContactPlan(**{**good, key: value})


def test_at_arrival_refuses_missing_geometry_or_a_bad_snr():
    clock = Clock()
    with pytest.raises(ValueError, match="positions"):
        _banded(clock, positions=None)
    with pytest.raises(ValueError, match="no position"):
        _banded(clock, positions={A: (0.0, 0.0, 0.0)})
    with pytest.raises(ValueError, match="snr_fn gave"):
        _banded(clock, snr_fn=lambda j, d, t: math.nan)
    with pytest.raises(TypeError):
        _banded(clock, snr_fn=None)
    with pytest.raises(ValueError, match="repeat"):
        ContactPlan.at_arrival([A, A], clock=clock, advance=clock.advance)


def test_drop_uplink_is_kept_as_given_and_only_members_are_allowed():
    clock = Clock()
    plan = _banded(clock, drop_uplink=[A, D])
    assert plan.drop_uplink == frozenset({A, D})
    with pytest.raises(ValueError):
        _banded(clock, drop_uplink=[DeviceID("zz")])


def test_the_plan_is_immutable_and_compared_by_identity():
    clock = Clock()
    plan = _banded(clock)
    with pytest.raises(Exception):
        plan.targets = ()                     # frozen
    with pytest.raises(TypeError):
        plan.snr_db[A] = 0.0                  # read-only mapping
    assert plan != _banded(clock)             # holds callables: identity only


def test_the_module_imports_nothing_from_l1_the_scheduler_or_experiments():
    """The plan carries physics as numbers and callables (design section 4.2)."""
    tree = ast.parse(Path(cp.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for name in imported:
        assert not name.startswith(("hermes.l1", "hermes.scheduler", "experiments")), name


def test_ledger_kinds_match_the_mission_clock():
    clock_mod = pytest.importorskip("hermes.l1.mission_clock")
    assert cp.KIND_DWELL in clock_mod.LEDGER_KINDS
    assert cp.KIND_LISTEN in clock_mod.LEDGER_KINDS
