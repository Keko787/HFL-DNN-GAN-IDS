"""FeRRy Phase 3, unit U1: the simulated mission clock.

``hermes/l1/mission_clock.py`` takes over mission time from the wall clock.
These tests pin its arithmetic (charges, the Lamport max, refused charges, the
per-mission ledger), the two bounds of its epoch, its use as ``FLScheduler``'s
``now_fn``, the flight constants, and the published numbers of the
Zeng-Xu-Zhang energy model it carries.

The scheduler is imported inside the tests that drive it: this file tests the
clock, and a scheduler import error should fail those tests only.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import random
import sys
import threading
import time
from pathlib import Path

import pytest

import hermes.l1 as l1_pkg
from hermes.l1 import mission_clock
from hermes.l1.mission_clock import (
    DOCK_POSE,
    GROUND_KINDS,
    HOVER_KINDS,
    LEDGER_KINDS,
    MOVE_KINDS,
    SIM_CEILING_S,
    SIM_EPOCH_S,
    EnergyModel,
    FlightModel,
    MissionClock,
    zeng_power_w,
)

ZERO_LEDGER = dict.fromkeys(LEDGER_KINDS, 0.0)


# --------------------------------------------------------------------------- #
# The epoch
# --------------------------------------------------------------------------- #

def test_a_new_clock_reads_the_epoch():
    clock = MissionClock()
    assert clock() == SIM_EPOCH_S == 1.0e6
    assert type(clock()) is float
    assert clock.ledger() == ZERO_LEDGER
    assert clock.ledger_start_s == SIM_EPOCH_S


def test_the_epoch_sits_between_the_never_sentinel_and_wall_time():
    assert 0.0 < SIM_EPOCH_S < SIM_CEILING_S == 1.0e9
    # Every wall stamp lies above the ceiling, so a stamp's value names its clock.
    assert time.time() > SIM_CEILING_S


def test_a_clean_at_the_epoch_still_counts_as_served():
    """0.0 means "never" to S3's idle term. A clock starting at 0.0 would make a
    device served at its first instant look never served; the epoch does not."""
    from hermes.scheduler.stages.s3_deadline import compute_idle_time
    from hermes.types import DeviceID, DeviceSchedulerState

    clock = MissionClock()
    served = DeviceSchedulerState(device_id=DeviceID("a"), idle_time_ref_ts=clock())
    clock.advance(25.0, "transit")
    assert compute_idle_time(served, clock()) == 25.0

    at_zero = DeviceSchedulerState(device_id=DeviceID("a"), idle_time_ref_ts=0.0)
    assert compute_idle_time(at_zero, 25.0) == 0.0


# --------------------------------------------------------------------------- #
# advance
# --------------------------------------------------------------------------- #

def test_advance_moves_the_clock_and_books_the_kind():
    clock = MissionClock()
    assert clock.advance(12.5, "transit") == SIM_EPOCH_S + 12.5
    assert clock.advance(0.25, "dwell") == SIM_EPOCH_S + 12.75
    assert clock.advance(0.0, "listen") == SIM_EPOCH_S + 12.75   # a legal charge
    assert clock.advance(7.25, "transit") == SIM_EPOCH_S + 20.0
    assert clock() == SIM_EPOCH_S + 20.0
    assert clock.ledger() == {**ZERO_LEDGER, "transit": 19.75, "dwell": 0.25}


def test_advance_takes_ints_and_keeps_floats():
    clock = MissionClock()
    t = clock.advance(30, "turnaround")
    assert t == SIM_EPOCH_S + 30.0 and type(t) is float
    assert type(clock.ledger()["turnaround"]) is float


@pytest.mark.parametrize(
    "dt",
    [-1.0, -1e-12, float("inf"), float("-inf"), float("nan"), 10 ** 400],
    ids=["negative", "tiny-negative", "inf", "-inf", "nan", "int-beyond-float"],
)
def test_advance_refuses_negative_and_non_finite_charges(dt):
    clock = MissionClock()
    clock.advance(3.0, "transit")
    with pytest.raises(ValueError):
        clock.advance(dt, "dwell")
    assert clock() == SIM_EPOCH_S + 3.0
    assert clock.ledger() == {**ZERO_LEDGER, "transit": 3.0}


@pytest.mark.parametrize(
    "dt", [None, "1.0", True, [1.0]], ids=["none", "str", "bool", "list"],
)
def test_advance_refuses_non_numbers(dt):
    clock = MissionClock()
    with pytest.raises(TypeError):
        clock.advance(dt, "transit")
    assert clock() == SIM_EPOCH_S
    assert clock.ledger() == ZERO_LEDGER


def test_an_unknown_kind_is_refused_by_both_charges():
    clock = MissionClock()
    with pytest.raises(ValueError, match="kind"):
        clock.advance(1.0, "transt")
    with pytest.raises(ValueError, match="kind"):
        clock.advance_to(SIM_EPOCH_S + 5.0, "lamport")
    assert clock() == SIM_EPOCH_S
    assert clock.ledger() == ZERO_LEDGER


def test_an_unknown_kind_is_refused_even_when_nothing_would_be_charged():
    """A zero charge and a sync to a time already passed move nothing, but a
    misspelt kind is still the caller's bug. The dock sync takes that path most
    often (K = 1, and the bootstrap DOWN's 0.0 for "nothing ingested"), so it
    must fail there exactly as a charging call does."""
    clock = MissionClock()
    clock.advance(10.0, "upload")
    before = (clock(), clock.ledger(), clock.ledger_start_s)
    with pytest.raises(ValueError, match="kind"):
        clock.advance(0.0, "transt")
    for target in (0.0, SIM_EPOCH_S - 1.0, SIM_EPOCH_S, clock()):
        with pytest.raises(ValueError, match="kind"):
            clock.advance_to(target, "lamport")
    assert (clock(), clock.ledger(), clock.ledger_start_s) == before


def test_the_clock_is_never_carried_into_the_wall_domain():
    clock = MissionClock()
    with pytest.raises(ValueError, match="wall"):
        clock.advance(SIM_CEILING_S - SIM_EPOCH_S, "transit")   # lands on it
    assert clock() == SIM_EPOCH_S
    assert clock.ledger() == ZERO_LEDGER
    assert clock.advance(SIM_CEILING_S - SIM_EPOCH_S - 1.0, "transit") < SIM_CEILING_S


# --------------------------------------------------------------------------- #
# advance_to: the Lamport sync
# --------------------------------------------------------------------------- #

def test_advance_to_a_later_time_moves_there_and_books_the_gap():
    clock = MissionClock()
    clock.advance(10.0, "upload")
    assert clock.advance_to(SIM_EPOCH_S + 42.0, "dock_wait") == SIM_EPOCH_S + 42.0
    assert clock() == SIM_EPOCH_S + 42.0
    assert clock.ledger() == {**ZERO_LEDGER, "upload": 10.0, "dock_wait": 32.0}


@pytest.mark.parametrize(
    "target",
    [SIM_EPOCH_S + 10.0, SIM_EPOCH_S + 3.0, SIM_EPOCH_S, 0.0, -5.0],
    ids=["now", "earlier", "epoch", "nothing-ingested", "negative"],
)
def test_advance_to_never_goes_back(target):
    clock = MissionClock()
    clock.advance(10.0, "upload")
    assert clock.advance_to(target, "dock_wait") == SIM_EPOCH_S + 10.0
    assert clock() == SIM_EPOCH_S + 10.0
    assert clock.ledger() == {**ZERO_LEDGER, "upload": 10.0}


@pytest.mark.parametrize(
    "target",
    [float("inf"), float("nan"), SIM_CEILING_S, 1.7e9],
    ids=["inf", "nan", "ceiling", "wall-stamp"],
)
def test_advance_to_refuses_non_finite_and_wall_stamps(target):
    clock = MissionClock()
    with pytest.raises(ValueError):
        clock.advance_to(target, "dock_wait")
    assert clock() == SIM_EPOCH_S
    assert clock.ledger() == ZERO_LEDGER


# --------------------------------------------------------------------------- #
# The per-mission ledger
# --------------------------------------------------------------------------- #

def test_a_hand_worked_mission_books_every_second_once():
    flight = FlightModel()
    clock = MissionClock()
    clock.reset_ledger()                                         # takeoff
    takeoff = clock()
    stop = (30.0, 40.0, 0.0)                                     # 50 m: 10 s
    clock.advance(flight.leg_s(flight.dock, stop), "transit")    # Pass 1
    clock.advance(0.5, "dwell")
    clock.advance(flight.listen_s, "listen")                     # a reply missing
    clock.advance(flight.leg_s(stop, flight.dock), "return")
    clock.advance(0.25, "upload")                                # inter-pass dock
    clock.advance(flight.turnaround_s, "turnaround")
    clock.advance_to(clock() + 8.25, "dock_wait")                # cluster ahead
    clock.advance(flight.leg_s(flight.dock, stop), "transit")    # Pass 2
    clock.advance(0.25, "dwell")
    clock.advance(flight.leg_s(stop, flight.dock), "return")
    assert clock.ledger() == {
        "transit": 20.0, "dwell": 0.75, "listen": 1.0, "return": 20.0,
        "upload": 0.25, "turnaround": 30.0, "dock_wait": 8.25,
    }
    assert sum(clock.ledger().values()) == clock() - takeoff == 80.25


def test_the_ledger_by_kind_sums_to_the_elapsed_time():
    rng = random.Random(20260929)
    clock = MissionClock()
    clock.advance_to(SIM_EPOCH_S + 7.0, "dock_wait")   # bootstrap DOWN, pre-takeoff
    clock.reset_ledger()
    start = clock()
    charged = {kind: [] for kind in LEDGER_KINDS}
    for _ in range(1000):
        kind = rng.choice(LEDGER_KINDS)
        before = clock()
        if rng.random() < 0.25:
            after = clock.advance_to(before + rng.uniform(-60.0, 60.0), kind)
            charged[kind].append(after - before)    # 0 when the target had passed
        else:
            dt = rng.uniform(0.0, 120.0)
            after = clock.advance(dt, kind)
            charged[kind].append(dt)
        assert after >= before
    ledger = clock.ledger()
    assert list(ledger) == list(LEDGER_KINDS)
    for kind in LEDGER_KINDS:
        assert ledger[kind] == pytest.approx(math.fsum(charged[kind]), abs=1e-6)
    assert math.fsum(ledger.values()) == pytest.approx(clock() - start, abs=1e-6)
    assert clock.ledger_start_s == start


def test_reset_ledger_does_not_reset_the_clock():
    clock = MissionClock()
    clock.advance(100.0, "transit")
    clock.advance(30.0, "turnaround")
    t = clock()
    clock.reset_ledger()
    assert clock() == t == SIM_EPOCH_S + 130.0
    assert clock.ledger() == ZERO_LEDGER
    assert clock.ledger_start_s == t
    assert clock.advance(7.0, "return") == t + 7.0
    assert clock.ledger() == {**ZERO_LEDGER, "return": 7.0}


def test_the_ledger_is_a_copy():
    clock = MissionClock()
    clock.advance(3.0, "transit")
    snapshot = clock.ledger()
    snapshot["transit"] = 99.0
    snapshot["bogus"] = 1.0
    assert clock.ledger() == {**ZERO_LEDGER, "transit": 3.0}


def test_the_kinds_split_into_flight_hover_and_ground():
    assert LEDGER_KINDS == (
        "transit", "dwell", "listen", "return", "upload", "turnaround", "dock_wait",
    )
    split = MOVE_KINDS + HOVER_KINDS + GROUND_KINDS
    assert sorted(split) == sorted(LEDGER_KINDS)
    assert len(set(split)) == len(split)


# --------------------------------------------------------------------------- #
# Threads and the wall clock
# --------------------------------------------------------------------------- #

class _ReportingLock:
    """Stands in for the clock's lock and reports when a second thread has to
    wait for it, so the race test below needs no sleeps. A clock that charged
    without its lock would never report, and would lose the cut-in charge."""

    def __init__(self, waiting: threading.Event) -> None:
        self._lock = threading.Lock()
        self._waiting = waiting

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        if not blocking:
            return False
        self._waiting.set()
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_ReportingLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


#: The call stopped part-way, and the call that cuts in while it is stopped.
_STOPPED = {
    "advance": lambda clock: clock.advance(0.5, "transit"),
    "advance_to": lambda clock: clock.advance_to(SIM_EPOCH_S + 11.0, "dock_wait"),
    "reset_ledger": lambda clock: clock.reset_ledger(),
}
_CUT_IN = {
    "advance": lambda clock: clock.advance(2.0, "dwell"),
    "advance_to": lambda clock: clock.advance_to(SIM_EPOCH_S + 13.0, "dock_wait"),
    "reset_ledger": lambda clock: clock.reset_ledger(),
}
_WRITERS = frozenset(
    getattr(MissionClock, name).__code__
    for name in ("advance", "advance_to", "reset_ledger")
)


def _race_clock():
    clock = MissionClock()
    clock.advance(10.0, "upload")
    return clock


def _state(clock):
    return clock(), clock.ledger_start_s, tuple(clock.ledger().items())


def _line_tracer(on_line):
    """A ``sys.settrace`` hook calling ``on_line()`` before each line that the
    clock's writing methods execute."""
    def local(frame, event, arg):
        if event == "line":
            on_line()
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code in _WRITERS else None

    return tracer


def _lines_run_by(call):
    clock, lines = _race_clock(), []
    old = sys.gettrace()
    sys.settrace(_line_tracer(lambda: lines.append(None)))
    try:
        call(clock)
    finally:
        sys.settrace(old)
    return len(lines)


def _cut_in_before_line(stopped_call, cut_in, line):
    """Run ``stopped_call`` on one thread and stop it before its ``line``-th
    line; run ``cut_in`` on a second thread until it finishes or has to wait
    for the lock; then let both finish. Returns the clock's final state."""
    clock = _race_clock()
    stopped, went = threading.Event(), threading.Event()
    clock._lock = _ReportingLock(went)
    seen, errors = [], []

    def on_line():
        seen.append(None)
        if len(seen) == line:
            stopped.set()
            if not went.wait(timeout=10.0):
                errors.append("the cut-in call neither finished nor waited")

    def run_stopped():
        old = sys.gettrace()
        sys.settrace(_line_tracer(on_line))
        try:
            stopped_call(clock)
        except BaseException as exc:   # reported on the test's thread below
            errors.append(exc)
        finally:
            sys.settrace(old)

    def run_cut_in():
        try:
            cut_in(clock)
        except BaseException as exc:
            errors.append(exc)
        finally:
            went.set()

    first = threading.Thread(target=run_stopped, daemon=True)
    first.start()
    deadline = time.monotonic() + 10.0
    while not stopped.wait(timeout=0.01):
        assert first.is_alive() and time.monotonic() < deadline, (
            f"the call never reached line {line}: {errors}"
        )
    second = threading.Thread(target=run_cut_in, daemon=True)
    second.start()
    for thread in (first, second):
        thread.join(timeout=10.0)
        assert not thread.is_alive()
    assert not errors, errors
    return _state(clock)


@pytest.mark.parametrize("cut_in", sorted(_CUT_IN))
@pytest.mark.parametrize("stopped", sorted(_STOPPED))
def test_every_charge_lands_whole(stopped, cut_in):
    """Only the supervisor charges the clock, but a charge, sync or reset from
    another thread still lands whole. Another call cuts in before each line of
    a call in progress in turn, and every result must be one of the two serial
    orders. Every line is tried, so the test does not depend on thread
    scheduling: a stress test cannot see a missing lock under the GIL."""
    first, second = _STOPPED[stopped], _CUT_IN[cut_in]
    in_order, reversed_order = _race_clock(), _race_clock()
    first(in_order)
    second(in_order)
    second(reversed_order)
    first(reversed_order)
    serial = {_state(in_order), _state(reversed_order)}
    lines = _lines_run_by(first)
    assert lines >= 2
    for line in range(1, lines + 1):
        state = _cut_in_before_line(first, second, line)
        assert state in serial, f"{cut_in} cut in before line {line}: {state}"


def test_the_clocks_own_lock_excludes_a_second_charge():
    """The other half of the race test, which swaps the lock: the clock's own
    lock is a real one, so a charge waits while another thread holds it."""
    clock = MissionClock()
    done = threading.Event()

    def charge():
        clock.advance(1.0, "transit")
        done.set()

    worker = threading.Thread(target=charge, daemon=True)
    with clock._lock:
        worker.start()
        assert not done.wait(timeout=0.05)
    worker.join(timeout=10.0)
    assert done.is_set()
    assert clock() == SIM_EPOCH_S + 1.0


def test_charging_never_waits_on_the_wall_clock(monkeypatch):
    """A charge is an addition: a 5 km leg costs no wall time (finding P-02)."""
    def no_sleep(_seconds):
        raise AssertionError("the mission clock slept")

    monkeypatch.setattr(time, "sleep", no_sleep)
    clock = MissionClock()
    leg = FlightModel().leg_s(DOCK_POSE, (5000.0, 0.0, 0.0))
    t0 = time.perf_counter()
    assert clock.advance(leg, "transit") == SIM_EPOCH_S + 1000.0
    assert time.perf_counter() - t0 < 1.0


def test_stand_in_style_assignment_raises():
    """Stand-in clocks in older tests are moved with ``clock.t = ...``; on the
    real clock that raises instead of silently doing nothing."""
    clock = MissionClock()
    with pytest.raises(AttributeError):
        clock.t = SIM_EPOCH_S + 5.0
    assert clock() == SIM_EPOCH_S


# --------------------------------------------------------------------------- #
# A drop-in now_fn for the scheduler
# --------------------------------------------------------------------------- #

def _scheduler(clock, *, budget=None):
    """A scheduler on ``clock`` whose one device sits 30 m from the dock."""
    from hermes.scheduler.fl_scheduler import FLScheduler
    from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
    from hermes.types import DeviceID, MissionSlice, MuleID

    sch = FLScheduler(
        now_fn=clock,
        mission_budget_s=budget,
        feasibility_model=FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0),
    )
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=(DeviceID("a"),),
        issued_round=1,
        issued_at=clock(),
    ))
    sch.device_states[DeviceID("a")].last_known_position = (30.0, 0.0, 0.0)
    return sch


def _plan(sch):
    return sch.build_contact_queue(
        rf_range_m=10.0, mule_pose=DOCK_POSE, mule_energy=1.0,
    )


def test_the_scheduler_takes_the_clock_as_now_fn():
    """The S3b budget stamp and the planning ``now`` both come from the clock
    (the pattern of ``test_mission_budget_clock.py``, on simulated seconds)."""
    from hermes.types import DeviceID

    a = DeviceID("a")
    clock = MissionClock()
    sch = _scheduler(clock, budget=60.0)
    assert sch.mission_start_ts == SIM_EPOCH_S            # ingest_slice's stamp
    clock.advance(40.0, "turnaround")
    assert _plan(sch) == []                               # stale: 40 + 30 > 60
    assert sch.start_mission() == SIM_EPOCH_S + 40.0
    assert [wp.devices for wp in _plan(sch)] == [(a,)]    # a fresh 60 s: 30 fits
    # A device never served: Deadline = now + its 60 s window, simulated.
    assert sch.last_plan_deadlines == {a: SIM_EPOCH_S + 40.0 + 60.0}


def test_a_clean_stamped_by_the_clock_anchors_the_deadline():
    """A CLEAN stamped in simulated time and a simulated ``now`` compose: the
    idle term counts, so the deadline stays where the CLEAN put it instead of
    sliding forward with ``now``."""
    from hermes.types import DeviceID, MissionOutcome, MuleID, RoundCloseDelta

    a = DeviceID("a")
    clock = MissionClock()
    sch = _scheduler(clock)
    t_clean = clock.advance(12.0, "dwell")
    sch.ingest_round_close_delta(RoundCloseDelta(
        device_id=a,
        mule_id=MuleID("m1"),
        mission_round=1,
        outcome=MissionOutcome.CLEAN,
        utility=1.0,
        contact_ts=t_clean,
        answered=True,
    ))
    window = sch.device_states[a].deadline_fulfilment_s
    clock.advance(8.0, "return")
    _plan(sch)
    first = sch.last_plan_deadlines[a]
    clock.advance(30.0, "turnaround")
    clock.advance(25.0, "transit")
    _plan(sch)
    assert sch.last_plan_deadlines[a] == first == t_clean + window
    assert SIM_EPOCH_S < first < SIM_CEILING_S


# --------------------------------------------------------------------------- #
# The dock and the flight model
# --------------------------------------------------------------------------- #

def test_the_dock_is_the_origin_the_runtime_already_uses():
    from hermes.processes.mule import DOCK_POSE as runtime_dock

    assert DOCK_POSE == (0.0, 0.0, 0.0) == runtime_dock
    assert all(type(c) is float for c in DOCK_POSE)


def test_flight_model_defaults():
    flight = FlightModel()
    assert (
        flight.cruise_speed_m_s, flight.dock, flight.turnaround_s, flight.listen_s,
    ) == (5.0, DOCK_POSE, 30.0, 1.0)
    assert flight.energy == EnergyModel()
    assert hash(flight) == hash(FlightModel())   # frozen, so fit for provenance


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"cruise_speed_m_s": 0.0}, ValueError),
        ({"cruise_speed_m_s": -5.0}, ValueError),
        ({"cruise_speed_m_s": float("inf")}, ValueError),
        ({"cruise_speed_m_s": "5"}, TypeError),
        ({"turnaround_s": -1.0}, ValueError),
        ({"listen_s": float("nan")}, ValueError),
        ({"dock": (0.0, 0.0)}, ValueError),
        ({"dock": (0.0, float("nan"), 0.0)}, ValueError),
        ({"energy": "simulated"}, TypeError),
        ({"cruise_speed_m_s": 10.0, "energy": EnergyModel()}, ValueError),
    ],
)
def test_flight_model_refuses_impossible_constants(kwargs, error):
    with pytest.raises(error):
        FlightModel(**kwargs)


@pytest.mark.parametrize("speed", [0.0, -5.0])
def test_an_impossible_cruise_speed_is_named_as_such(speed):
    """Refused up front, rather than by the energy model derived from it."""
    with pytest.raises(ValueError, match="cruise_speed_m_s must be > 0"):
        FlightModel(cruise_speed_m_s=speed)


def test_flight_model_stores_floats():
    """Ints become floats, so provenance JSON reads 30.0, not 30."""
    flight = FlightModel(cruise_speed_m_s=5, dock=[1, 2, 0], turnaround_s=30, listen_s=1)
    assert flight.dock == (1.0, 2.0, 0.0)
    assert type(flight.dock) is tuple
    stored = (flight.cruise_speed_m_s, *flight.dock, flight.turnaround_s, flight.listen_s)
    assert all(type(v) is float for v in stored), stored
    assert flight.energy == EnergyModel()     # an int 5 still pairs with 5 m/s


def test_zero_turnaround_and_listen_are_legal():
    """0 s is a sweep or ablation value for both; only the speed must be > 0."""
    flight = FlightModel(turnaround_s=0.0, listen_s=0.0)
    assert (flight.turnaround_s, flight.listen_s) == (0.0, 0.0)


def test_flight_and_energy_models_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        FlightModel().turnaround_s = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        EnergyModel().capacity_j = 1.0


def test_leg_s_charges_exactly_the_leg_the_planner_predicts():
    from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel

    rng = random.Random(3)
    for _ in range(2000):
        # 1e-9 and 5e-7 m/s: legal but degenerate, below the planner's 1e-6 floor.
        speed = rng.choice((1.0, 2.5, 5.0, 12.0, 1e-9, 5e-7, rng.uniform(0.1, 30.0)))
        a = tuple(rng.uniform(-300.0, 300.0) for _ in range(3))
        b = tuple(rng.uniform(-300.0, 300.0) for _ in range(3))
        predicted, _total = FeasibilityModel(cruise_speed_m_s=speed).cost(a, b)
        assert FlightModel(cruise_speed_m_s=speed).leg_s(a, b) == predicted
    flight = FlightModel()
    assert flight.leg_s((3.0, 4.0, 0.0), (3.0, 4.0, 0.0)) == 0.0
    assert flight.leg_s(DOCK_POSE, (5000.0, 0.0, 0.0)) == 1000.0   # 5 km at 5 m/s


# --------------------------------------------------------------------------- #
# Energy: Zeng, Xu and Zhang 2019, simulated
# --------------------------------------------------------------------------- #

def _zeng_p0_pi(*, W=20.0, rho=1.225, A=0.503, u_tip=120.0, s=0.05, k=0.1,
                delta=0.012):
    """P0 and Pi from their definitions (Omega^3 R^3 = U_tip^3)."""
    p0 = delta / 8.0 * rho * s * A * u_tip ** 3
    p_i = (1.0 + k) * W ** 1.5 / math.sqrt(2.0 * rho * A)
    return p0, p_i


def _zeng_power_w(v, *, W=20.0, rho=1.225, A=0.503, u_tip=120.0, d0=0.6, s=0.05):
    """Eq. (6) term by term as the paper writes it (the difference form of the
    induced term), with Table I's set: an independent check of the module's
    constants and of its rearranged induced term, at flight speeds."""
    p0, p_i = _zeng_p0_pi(W=W, rho=rho, A=A, u_tip=u_tip, s=s)
    v0 = math.sqrt(W / (2.0 * rho * A))
    return (
        p0 * (1.0 + 3.0 * v * v / (u_tip * u_tip))
        + p_i * math.sqrt(
            math.sqrt(1.0 + v ** 4 / (4.0 * v0 ** 4)) - v * v / (2.0 * v0 * v0)
        )
        + 0.5 * d0 * rho * s * A * v ** 3
    )


def test_energy_defaults_are_the_published_zeng_numbers():
    energy = EnergyModel()
    assert (
        energy.p_move_w, energy.p_hover_w, energy.capacity_j, energy.status,
        energy.speed_m_s,
    ) == (143.6, 168.5, None, "simulated", 5.0)
    assert round(_zeng_power_w(5.0), 1) == energy.p_move_w    # P(5 m/s)
    assert round(_zeng_power_w(0.0), 1) == energy.p_hover_w   # hover: P0 + Pi


def test_the_quoted_p0_and_pi_belong_to_the_tables_disc_area():
    """The table's A = 0.503 m^2 gives P0 = 79.86 W and Pi = 88.63 W; the exact
    pi R^2 gives 79.80 W and 88.66 W. The defaults are the same either way."""
    exact_area = math.pi * 0.4 ** 2
    assert tuple(round(p, 2) for p in _zeng_p0_pi(A=0.503)) == (79.86, 88.63)
    assert tuple(round(p, 2) for p in _zeng_p0_pi(A=exact_area)) == (79.80, 88.66)
    for area in (0.503, exact_area):
        assert round(_zeng_power_w(5.0, A=area), 1) == 143.6
        assert round(_zeng_power_w(0.0, A=area), 1) == 168.5
    assert zeng_power_w(0.0) == pytest.approx(sum(_zeng_p0_pi()), rel=1e-12)


def test_zeng_power_is_the_published_formula():
    for v in (0.0, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 12.0, 15.0, 18.3, 25.0, 30.0):
        assert zeng_power_w(v) == pytest.approx(_zeng_power_w(v), rel=1e-12), v
    assert type(zeng_power_w(5)) is float


def test_zeng_power_stays_finite_far_beyond_flight_speeds():
    """The paper's difference form of the induced term cancels as the speed
    grows; the module's rearranged form keeps it finite and positive, and by
    1e5 m/s it has vanished beside the blade and parasite terms."""
    v = 1e5
    p0, _p_i = _zeng_p0_pi()
    blade_and_parasite = (
        p0 * (1.0 + 3.0 * v * v / 120.0 ** 2) + 0.5 * 0.6 * 1.225 * 0.05 * 0.503 * v ** 3
    )
    assert zeng_power_w(v) == pytest.approx(blade_and_parasite, rel=1e-9)
    assert zeng_power_w(v) > blade_and_parasite


@pytest.mark.parametrize(
    "v", [-1.0, float("inf"), float("nan"), 1e200, None],
    ids=["negative", "inf", "nan", "overflows", "none"],
)
def test_zeng_power_refuses_impossible_speeds(v):
    with pytest.raises((ValueError, TypeError)):
        zeng_power_w(v)


def test_the_default_energy_is_the_zeng_model_at_5_m_s():
    assert EnergyModel.at_speed(5.0) == EnergyModel()
    assert EnergyModel.at_speed(5) == EnergyModel()
    assert FlightModel().energy == EnergyModel.at_speed(FlightModel().cruise_speed_m_s)


def test_at_speed_gives_the_power_at_that_speed():
    energy = EnergyModel.at_speed(10.0, capacity_j=50_000)
    assert (energy.p_move_w, energy.p_hover_w, energy.speed_m_s, energy.capacity_j) == (
        126.0, 168.5, 10.0, 50_000.0,
    )
    assert energy.p_move_w == round(_zeng_power_w(10.0), 1)
    assert energy.status == "simulated"
    with pytest.raises(ValueError):
        EnergyModel.at_speed(0.0)


def test_move_j_per_m_follows_the_cruise_speed():
    """A speed sweep books the power at each speed: 126.0 W at 10 m/s is
    12.6 J/m, where the 5 m/s power would have booked 14.36 J/m."""
    fast = FlightModel(cruise_speed_m_s=10.0)
    assert fast.energy == EnergyModel.at_speed(10.0)
    assert fast.move_j_per_m == pytest.approx(12.6)
    leg = fast.leg_s(DOCK_POSE, (1000.0, 0.0, 0.0))                  # 100 s
    assert fast.energy.energy_j({"transit": leg}) == pytest.approx(12_600.0)
    declared = FlightModel(
        cruise_speed_m_s=4.0, energy=EnergyModel(p_move_w=100.0, speed_m_s=4.0),
    )
    assert declared.move_j_per_m == pytest.approx(25.0)


def test_a_flight_model_refuses_an_energy_model_for_another_speed():
    """The capacity clause is switched on with ``EnergyModel(capacity_j=...)``,
    which carries the 5 m/s power; at 10 m/s that would overstate flight
    energy by 14 %, so it is refused rather than used."""
    with pytest.raises(ValueError, match="speed"):
        FlightModel(cruise_speed_m_s=10.0, energy=EnergyModel(capacity_j=50_000.0))
    with pytest.raises(ValueError, match="speed"):
        dataclasses.replace(FlightModel(), cruise_speed_m_s=10.0)
    swept = dataclasses.replace(FlightModel(), cruise_speed_m_s=10.0, energy=None)
    assert swept.energy == EnergyModel.at_speed(10.0)
    capped = FlightModel(
        cruise_speed_m_s=10.0, energy=EnergyModel.at_speed(10.0, capacity_j=50_000.0),
    )
    assert capped.energy.capacity_j == 50_000.0


def test_flying_at_5_m_s_costs_28_7_joules_per_metre():
    flight = FlightModel()
    assert flight.move_j_per_m == pytest.approx(143.6 / 5.0)
    assert round(flight.move_j_per_m, 1) == 28.7
    ledger = {"transit": flight.leg_s(DOCK_POSE, (100.0, 0.0, 0.0))}   # 20 s
    assert flight.energy.energy_j(ledger) == pytest.approx(2872.0)
    assert flight.energy.energy_j(ledger) == pytest.approx(100.0 * flight.move_j_per_m)


def test_5_m_s_is_below_the_minimum_power_speed():
    """The declared caveat: 5 m/s is slower than the model's most economical
    speed (about 10.2 m/s), and at 5 m/s the mule draws 85 % of hover power."""
    speeds = [i / 100 for i in range(1, 4001)]
    v_min_power = min(speeds, key=zeng_power_w)
    assert v_min_power == pytest.approx(10.2, abs=0.05)
    assert round(zeng_power_w(v_min_power)) == 126                     # W
    assert round(zeng_power_w(v_min_power) / v_min_power, 1) == 12.3   # J/m
    assert FlightModel().cruise_speed_m_s < v_min_power
    energy = EnergyModel()
    assert energy.p_move_w / energy.p_hover_w == pytest.approx(0.852, abs=1e-3)


def test_energy_charges_flight_and_hover_but_not_ground_time():
    energy = EnergyModel()
    ledger = {
        "transit": 100.0, "return": 20.0, "dwell": 3.0, "listen": 1.0,
        "upload": 50.0, "turnaround": 30.0, "dock_wait": 7.0,
    }
    assert energy.energy_j(ledger) == pytest.approx(143.6 * 120.0 + 168.5 * 4.0)
    assert energy.energy_j({"upload": 50.0, "turnaround": 30.0, "dock_wait": 7.0}) == 0.0
    assert energy.energy_j({}) == 0.0


def test_mission_energy_comes_from_the_ledger_and_resets_with_it():
    clock, energy = MissionClock(), EnergyModel()
    clock.reset_ledger()                                          # takeoff
    for dt, kind in [(40.0, "transit"), (2.0, "dwell"), (1.0, "listen"),
                     (40.0, "return"), (30.0, "turnaround")]:
        clock.advance(dt, kind)
    assert energy.energy_j(clock.ledger()) == pytest.approx(143.6 * 80.0 + 168.5 * 3.0)
    clock.reset_ledger()                                          # next takeoff
    assert energy.energy_j(clock.ledger()) == 0.0


@pytest.mark.parametrize(
    "ledger",
    [{"transt": 1.0}, {"transit": -1.0}, {"dwell": float("nan")}, {"listen": float("inf")}],
    ids=["unknown-kind", "negative", "nan", "inf"],
)
def test_energy_refuses_a_malformed_ledger(ledger):
    with pytest.raises(ValueError):
        EnergyModel().energy_j(ledger)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"p_move_w": -1.0}, ValueError),
        ({"p_hover_w": float("inf")}, ValueError),
        ({"p_move_w": None}, TypeError),
        ({"capacity_j": 0.0}, ValueError),
        ({"capacity_j": float("nan")}, ValueError),
        ({"status": ""}, ValueError),
        ({"speed_m_s": 0.0}, ValueError),
        ({"speed_m_s": -5.0}, ValueError),
        ({"speed_m_s": "5"}, TypeError),
    ],
)
def test_energy_model_refuses_impossible_constants(kwargs, error):
    with pytest.raises(error):
        EnergyModel(**kwargs)


def test_energy_model_stores_floats():
    """Ints become floats, so ``energy_params`` in provenance JSON reads
    50000.0, not 50000."""
    energy = EnergyModel(p_move_w=100, p_hover_w=150, capacity_j=50_000, speed_m_s=4)
    stored = (energy.p_move_w, energy.p_hover_w, energy.capacity_j, energy.speed_m_s)
    assert stored == (100.0, 150.0, 50_000.0, 4.0)
    assert all(type(v) is float for v in stored), stored


def test_zero_power_is_legal():
    """An energy-off ablation: the model charges nothing."""
    energy = EnergyModel(p_move_w=0.0, p_hover_w=0.0)
    assert energy.energy_j({"transit": 10.0, "dwell": 5.0, "listen": 1.0}) == 0.0


# --------------------------------------------------------------------------- #
# Module hygiene
# --------------------------------------------------------------------------- #

def test_the_module_imports_only_the_standard_library():
    """numpy-free, no wall clock, and nothing from the scheduler or from
    ``experiments/`` (finding A-01)."""
    tree = ast.parse(Path(mission_clock.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    allowed = {"__future__", "dataclasses", "math", "numbers", "threading", "typing"}
    assert imported <= allowed, imported - allowed


def test_the_l1_package_exports_are_unchanged():
    assert sorted(l1_pkg.__all__) == [
        "CHANNEL_FREQS_GHZ", "ChannelDDQN", "RFPrior", "RFPriorStore",
    ]
    assert not hasattr(l1_pkg, "MissionClock")
