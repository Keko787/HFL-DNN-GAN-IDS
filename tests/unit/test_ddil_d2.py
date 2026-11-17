"""DDIL build D2 (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md, step 2).

Step 2's planner switches, each off (None) in every recorded run:

* the risk-priced dwell and its reserve (``FerrySpec.dwell_pricing``,
  ``dwell_reserve_k``; ``FerryRuntime.member_dwell_s``);
* the leave-by time per Pass-1 stop (``FerrySpec.leave_by``;
  ``ContactPlan.leave_by_ts``, the host's ferry commit, ``hermes.mule.leave_by``);
* the config, driver and runner plumbing.

What is pinned: each switch is inert while off; the price is the dwell's mean
over the SNR's spread, plus k standard deviations; no session, retry or re-poll
starts at or after the leave-by time, and what it cuts is committed as a
timeout; the leave-by time is the planned end plus the stop's share of the slack.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest

from hermes.l1.mission_clock import MissionClock
from hermes.mission.contact_plan import ContactPlan
from hermes.mule.ferry import _DWELL_Z, DWELL_PRICE_NODES, FerryRuntime, FerrySpec
from hermes.mule.leave_by import stop_leave_by_ts
from hermes.processes import config as C
from hermes.types import DeliveryOutcome, MissionOutcome, MissionPass
from tests.unit.test_ddil_d1 import _host, _spec, d1_plan
from tests.unit.test_host_mission_ferry import SYNTH, D, lines_by_id, theta

COLLECT = MissionPass.COLLECT
CLEAN, TIMEOUT = MissionOutcome.CLEAN, MissionOutcome.TIMEOUT
EPS = 1e-3


# --------------------------------------------------------------------------- #
# The contact plan's leave-by time
# --------------------------------------------------------------------------- #

def test_the_leave_by_time_is_inclusive_finite_and_banded_only():
    rf, clock, host = _host(D[:1])
    plan = d1_plan(clock, D[:1], lambda j, d, t: 9.0)
    assert plan.leave_by_ts is None and not plan.leaves_at(1e12)
    cut = dataclasses.replace(plan, leave_by_ts=clock() + 5.0)
    assert not cut.leaves_at(clock() + 4.999) and cut.leaves_at(clock() + 5.0)
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            dataclasses.replace(plan, leave_by_ts=bad)
    with pytest.raises(ValueError, match="leave_by_ts"):
        ContactPlan(arrival_ts=0.0, targets=("a",), clock=lambda: 0.0,
                    advance=lambda dt, k: 0.0, leave_by_ts=1.0)


def _run(members, snr_fn, *, leave_after=None, model=True, repoll=False, positions=None):
    """One Pass-1 contact; ``leave_after`` puts the leave-by time that far past the arrival."""
    rf, clock, host = _host(members)
    t_arr = clock()
    plan = d1_plan(clock, members, snr_fn, model=model, repoll=repoll, positions=positions)
    if leave_after is not None:
        plan = dataclasses.replace(plan, leave_by_ts=t_arr + leave_after)
    out = host.run_contact(members, SYNTH, plan=plan)
    return out, host, clock, t_arr, plan


@pytest.mark.parametrize("model", [True, False])
def test_far_off_the_leave_by_time_changes_nothing(model):
    out0, host0, _c0, t0, _p0 = _run(D[:3], lambda j, d, t: 9.0, model=model)
    out1, host1, _c1, t1, _p1 = _run(D[:3], lambda j, d, t: 9.0, model=model, leave_after=1e6)
    a, b = host0.last_contact, host1.last_contact
    assert out0 == out1 and b.left == () and a.session_dwell_s == b.session_dwell_s
    assert {k: v - t0 for k, v in a.contact_ts.items()} == pytest.approx(
        {k: v - t1 for k, v in b.contact_ts.items()})


@pytest.mark.parametrize("model", [True, False])
def test_no_session_starts_at_or_after_the_leave_by_time(model):
    out, host, clock, t_arr, plan = _run(D[:3], lambda j, d, t: 9.0, model=model,
                                         leave_after=EPS)
    # D0 starts at the arrival, before the leave-by time, and is served in full;
    # D1 and D2 would start after it and are cut.
    assert out == {D[0]: CLEAN, D[1]: TIMEOUT, D[2]: TIMEOUT}
    commit = host.last_contact
    assert commit.left == (D[1], D[2]) and commit.session_failed == ()
    assert D[1] not in commit.pushed and D[2] not in commit.pushed and commit.missing == ()
    lines = lines_by_id(host._report)
    for did in (D[1], D[2]):
        assert lines[did].bytes_sent == 0 and lines[did].bytes_received == 0
        assert lines[did].contact_ts == clock() == t_arr + commit.session_dwell_s[D[0]]
    assert [g.device_id for g in host._accepted] == [D[0]]
    assert set(commit.session_dwell_s) == {D[0]}


def test_a_retry_after_the_leave_by_time_is_not_started():
    rf, clock, host = _host(D[:3])
    t_arr = clock()

    def snr_fn(j, d, t):              # D1 passes the gate, then stays below the floor
        if j == D[1]:
            return 3.0 if t == t_arr else -30.0
        return 9.0

    base = d1_plan(clock, D[:3], snr_fn)
    w0 = base.session_walk(D[0], t_arr, 2_000_000)
    w1 = base.session_walk(D[1], t_arr + w0.dur_s, 2_000_000)
    assert not w1.done
    t_d2 = t_arr + w0.dur_s + w1.dur_s         # D2 starts here; D1's retry after D2
    plan = dataclasses.replace(base, leave_by_ts=t_d2 + EPS)
    out = host.run_contact(D[:3], SYNTH, plan=plan)
    assert out == {D[0]: CLEAN, D[1]: TIMEOUT, D[2]: CLEAN}
    commit = host.last_contact
    assert commit.session_retried == () and commit.session_failed == (D[1],)
    assert commit.left == (D[1],)
    line = lines_by_id(host._report)[D[1]]
    assert line.bytes_received == 0 and line.bytes_sent > 0     # its first try went out
    assert commit.session_dwell_s[D[1]] == pytest.approx(w1.dur_s)


def test_no_repoll_is_solicited_after_the_leave_by_time():
    positions = {D[0]: (1.0, 0.0, 0.0), D[1]: (2.0, 0.0, 0.0), D[2]: (90.0, 0.0, 0.0)}
    t0 = {}

    def snr_fn(j, d, t):
        t0.setdefault("t", t)
        if j == D[1]:
            return -30.0 if t == t0["t"] else 5.0
        return 9.0

    out, host, *_ = _run(D[:3], snr_fn, model=False, repoll=True, positions=positions)
    assert host.last_contact.repolled == (D[1],)
    t0.clear()
    out, host, *_ = _run(D[:3], snr_fn, model=False, repoll=True, positions=positions,
                         leave_after=EPS)
    assert host.last_contact.repolled == () and out[D[1]] == TIMEOUT and out[D[0]] == CLEAN


def test_a_delivery_cut_by_the_leave_by_time_is_undelivered():
    rf, clock, host = _host(D[:2])
    host.run_contact(D[:2], SYNTH, plan=d1_plan(clock, D[:2], lambda j, d, t: 9.0))
    host.close_round()
    host.open_pass_2(theta(2.0), theta_version=6)
    plan = dataclasses.replace(d1_plan(clock, D[:2], lambda j, d, t: 9.0),
                               leave_by_ts=clock() + EPS)
    out = host.deliver_contact(D[:2], SYNTH, plan=plan)
    assert out == {D[0]: DeliveryOutcome.DELIVERED, D[1]: DeliveryOutcome.UNDELIVERED}
    assert host.last_contact.left == (D[1],)


# --------------------------------------------------------------------------- #
# The leave-by time's rule
# --------------------------------------------------------------------------- #

class _Model:
    """A stand-in feasibility model: each stop's dwell, a fixed return and upload."""

    def __init__(self, dwell, back=10.0, upload=5.0, leg=4.0):
        self.dwell, self.back, self.upload, self.leg = dwell, back, upload, leg
        self.ferry = SimpleNamespace(member_dwell_s=object(),
                                     dwell_s=lambda wp, kind: self.dwell[wp.name])

    def admit(self, state, wp, **kw):
        finish = state.clock + self.dwell[wp.name]
        nxt = SimpleNamespace(clock=finish, pose=wp.position)
        return SimpleNamespace(finish=finish, home=finish + self.back + self.upload,
                               next_state=nxt)

    def fold(self, route, state, **kw):
        t = state.clock
        for wp in route:
            t += self.leg + self.dwell[wp.name]
        return SimpleNamespace(home=t + self.back + self.upload)


def _wp(name):
    return SimpleNamespace(name=name, position=(0.0, 0.0, 25.0))


def test_the_leave_by_time_is_the_planned_end_plus_the_stops_share_of_the_slack():
    model = _Model({"a": 20.0, "b": 30.0, "c": 50.0})
    a, b, c = _wp("a"), _wp("b"), _wp("c")
    # Home as planned: 100 + 20 + (4 + 30) + (4 + 50) + 10 + 5 = 223; budget 263: slack 40.
    lb = stop_leave_by_ts(model, a, [b, c], arrival_ts=100.0, budget_end=263.0)
    assert lb == pytest.approx(100.0 + 20.0 + 40.0 * 20.0 / 100.0)
    # Behind the plan by 40 s, the stop gives up its share of the deficit.
    lb = stop_leave_by_ts(model, a, [b, c], arrival_ts=100.0, budget_end=183.0)
    assert lb == pytest.approx(100.0 + 20.0 - 40.0 * 20.0 / 100.0)
    # The last stop takes the whole slack; far behind, it leaves at the arrival.
    assert stop_leave_by_ts(model, c, [], arrival_ts=100.0, budget_end=175.0) == \
        pytest.approx(100.0 + 50.0 + 10.0)
    assert stop_leave_by_ts(model, c, [], arrival_ts=100.0, budget_end=0.0) == 100.0
    assert stop_leave_by_ts(model, a, [b], arrival_ts=100.0, budget_end=None) is None


# --------------------------------------------------------------------------- #
# The risk-priced dwell
# --------------------------------------------------------------------------- #

def _rt(**kw):
    return FerryRuntime(_spec(**kw), MissionClock(), rf_range_m=60.0)


def test_the_d2_fields_are_off_by_default_and_out_of_the_record():
    spec = _spec()
    assert spec.dwell_pricing is None and spec.dwell_reserve_k is None and spec.leave_by is False
    assert not {"dwell_pricing", "dwell_reserve_k", "leave_by"} & set(spec.describe())
    rt, link = _rt(), spec.link
    for d in (5.0, 40.0, 120.0):
        snr = link.mean_snr_db(rt.band, d)
        assert rt.member_dwell_s(d, COLLECT, 0.0) == link.dwell_s(
            rt.session_bytes(COLLECT), rt.band, snr)


def _manual(rt, d, k=0.0):
    link, band = rt.spec.link, rt.band
    nbytes = rt.session_bytes(COLLECT)
    mean = link.mean_snr_db(band, d)
    offset, sigma = rt.snr_spread_db()
    floor = link.dwell_s(nbytes, band, link.snr_floor_db)
    times = []
    for z in _DWELL_Z:
        t = link.dwell_s(nbytes, band, mean + offset + sigma * z)
        times.append(floor if t is None else t)
    m = sum(times) / len(times)
    return m + k * math.sqrt(sum((t - m) ** 2 for t in times) / len(times))


@pytest.mark.parametrize("amp", [None, 17.0])
def test_the_expected_price_is_the_mean_over_the_spread_plus_k_deviations(amp):
    kw = {} if amp is None else {"interference_amp_db": amp}
    plain = _rt(dwell_pricing="expected", **kw)
    reserve = _rt(dwell_pricing="expected", dwell_reserve_k=1.0, **kw)
    assert len(_DWELL_Z) == DWELL_PRICE_NODES
    priced = 0
    for d in (5.0, 40.0, 120.0, 200.0):
        at_mean = _rt(**kw).member_dwell_s(d, COLLECT, 0.0)
        got = plain.member_dwell_s(d, COLLECT, 0.0)
        if at_mean is None:
            assert got is None                   # still unpriced below the floor
            continue
        priced += 1
        assert got == pytest.approx(_manual(plain, d))
        assert reserve.member_dwell_s(d, COLLECT, 0.0) == pytest.approx(_manual(reserve, d, 1.0))
        assert reserve.member_dwell_s(d, COLLECT, 0.0) >= got
    assert priced >= 2
    record = reserve.spec.describe()
    assert record["dwell_pricing"] == "expected" and record["dwell_reserve_k"] == 1.0


def test_the_price_reads_the_causal_estimate_when_it_is_on():
    rt = _rt(dwell_pricing="expected", channel_estimate="causal", interference_amp_db=17.0)
    assert rt.snr_spread_db() == rt.channel_estimate_db()
    assert _rt(dwell_pricing="expected").snr_spread_db()[0] == 0.0


@pytest.mark.parametrize("kw, match", [
    (dict(dwell_pricing="median"), "dwell_pricing"),
    (dict(dwell_reserve_k=1.0), "needs dwell_pricing"),
    (dict(dwell_pricing="expected", dwell_reserve_k=-1.0), ">= 0"),
])
def test_the_spec_refuses_what_it_cannot_price(kw, match):
    with pytest.raises(ValueError, match=match):
        _spec(**kw)


def test_the_switches_need_a_band():
    with pytest.raises(ValueError, match="leave_by"):
        FerrySpec(leave_by=True)
    with pytest.raises(ValueError, match="dwell_pricing"):
        FerrySpec(dwell_pricing="expected")
    assert _spec(leave_by=True).describe()["leave_by"] is True


# --------------------------------------------------------------------------- #
# Config, driver and runner
# --------------------------------------------------------------------------- #

def test_the_d2_fields_are_ferry_fields_sim_only_and_left_out_of_ferry_params_at_none():
    from experiments.exp4 import runner_main as RM
    from experiments.exp4.driver import FERRY_PHYSICS_FIELDS

    defaults = {f.name: f.default for f in dataclasses.fields(C.MuleConfig)}
    for f in C.DDIL_D2_MULE_FIELDS:
        assert defaults[f] is None
        assert C.FERRY_SPEC_FIELDS[f] == f and f in C.SIM_ONLY_MULE_FIELDS
        assert f in C.FERRY_PARAMS_OMITTED_AT_NONE and f in C.ADDENDUM_MULE_FIELDS
        assert f in FERRY_PHYSICS_FIELDS
    assert set(C.DDIL_D2_MULE_FIELDS) <= {n for n, _k, _t in RM._PHYSICS_FLAGS}
