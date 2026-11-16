"""DDIL build D1 (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md, step 1).

The environment switches and the planner's channel estimate, each off (None)
in every recorded run:

* the contact channel's shared interference and burst process
  (``hermes/l1/channel_model.py``);
* the session model, its retry and resumable transfers, and the re-poll
  (``ContactPlan``, the host's ferry commit);
* the spec's fields, the contact plan the runtime builds, and the causal
  channel estimate (``hermes/mule/ferry.py``);
* the config, driver and runner plumbing.

What is pinned: each switch is inert while off; each does what the plan says
when on; and the session walk is the step 0 replay's ``walk_follow``, so that
replay's pricing (s) carries over to the flights.
"""

from __future__ import annotations

import math
import statistics
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hermes.l1.channel_model import CONTACT_REGIMES, ContactChannel
from hermes.l1.contact_link import ContactLink
from hermes.l1.mission_clock import MissionClock
from hermes.mission.contact_plan import SESSION_WALK_CAP_S, ContactPlan
from hermes.mule.ferry import CAUSAL_PRIOR_READINGS, FerryRuntime, FerrySpec
from hermes.processes import config as C
from hermes.types import Bucket, ContactWaypoint, DeviceID, MissionOutcome, MissionPass
from tests.unit.test_host_mission_ferry import (
    FLOOR,
    SYNTH,
    D,
    FerryRF,
    RecClock,
    lines_by_id,
    make_host,
    theta,
)

REPO = Path(__file__).resolve().parents[2]
COLLECT, DELIVER = MissionPass.COLLECT, MissionPass.DELIVER


def _chan(**kw):
    return ContactChannel(lambda b, d: 0.0, salt=4242, regime="jittery", **kw)


# --------------------------------------------------------------------------- #
# The channel
# --------------------------------------------------------------------------- #

def test_the_channel_is_the_recorded_one_while_every_switch_is_off():
    base, off = _chan(), _chan(interference_shared=None, burst_duty=None,
                               burst_mean_s=None, burst_depth_db=None)
    assert base.describe() == off.describe()
    assert not {"interference_shared", "burst_duty", "burst_mean_s",
                "burst_depth_db"} & set(base.describe())
    for k in range(0, 400, 7):
        t = base.epoch_s + k + 0.3
        for b in base.bands:
            assert base.snr_db(t, b, 20.0, "d1") == off.snr_db(t, b, 20.0, "d1")
            assert base.burst_db(t) == 0.0 and not base.burst_on(t)


def test_shared_interference_gives_every_class_the_named_classs_term():
    per, shared = _chan(), _chan(interference_shared="narrow")
    assert shared.describe()["interference_shared"] == "narrow"
    for k in range(0, 300, 11):
        t = per.epoch_s + k + 0.5
        want = per.interference_db(t, "narrow")
        assert [shared.interference_db(t, b) for b in shared.bands] == [want] * 3
    with pytest.raises(ValueError, match="interference_shared"):
        _chan(interference_shared="huge")


@pytest.mark.parametrize("duty, mean", [(0.1, 6.7), (0.4, 6.7), (0.4, 20.0)])
def test_the_burst_process_keeps_its_share_and_mean_length(duty, mean):
    c = _chan(burst_duty=duty, burst_mean_s=mean, burst_depth_db=17.0)
    states = [c.burst_on(c.epoch_s + k + 0.5) for k in range(60_000)]
    runs, n = [], 0
    for on in states:
        if on:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    assert abs(sum(states) / len(states) - duty) < 0.03
    assert abs(statistics.mean(runs) - mean) / mean < 0.08


def test_the_burst_state_is_a_pure_function_of_time():
    kw = dict(burst_duty=0.4, burst_mean_s=6.7, burst_depth_db=17.0)
    a, b = _chan(**kw), _chan(**kw)
    ts = [a.epoch_s + x + 0.25 for x in range(-200, 2000, 13)]
    forward = [a.burst_on(t) for t in ts]
    backward = [b.burst_on(t) for t in reversed(ts)][::-1]
    assert forward == backward and any(forward) and not all(forward)
    assert _chan(**dict(kw, burst_duty=0.1)).describe()["burst_duty"] == 0.1


def test_a_burst_lowers_every_class_by_its_depth_and_nothing_else():
    base = _chan()
    burst = _chan(burst_duty=0.4, burst_mean_s=6.7, burst_depth_db=17.0)
    seen = set()
    for k in range(500):
        t = base.epoch_s + k + 0.5
        on = burst.burst_on(t)
        seen.add(on)
        for b in base.bands:
            want = base.snr_db(t, b, 30.0, "d2") - (17.0 if on else 0.0)
            assert burst.snr_db(t, b, 30.0, "d2") == pytest.approx(want, abs=1e-12)
    assert seen == {True, False}


@pytest.mark.parametrize("kw, match", [
    (dict(burst_duty=0.1), "set together"),
    (dict(burst_duty=1.0, burst_mean_s=5.0, burst_depth_db=17.0), r"\(0, 1\)"),
    (dict(burst_duty=0.1, burst_mean_s=0.5, burst_depth_db=17.0), "one bin"),
])
def test_the_burst_process_refuses_what_it_cannot_run(kw, match):
    with pytest.raises(ValueError, match=match):
        _chan(**kw)


# --------------------------------------------------------------------------- #
# The session walk is the step 0 replay's
# --------------------------------------------------------------------------- #

def _replay():
    sys.path.insert(0, str(REPO / "scripts" / "exp5"))
    try:
        import ddil_replay
    finally:
        sys.path.pop(0)
    return ddil_replay


def _walk_plan(snr_of_t, *, t_out=10.0, resumable=False, t0=1.0e6):
    clock = SimpleNamespace(now=t0)
    link = ContactLink(anchor_planar_m=60.0)
    return ContactPlan.at_arrival(
        ["a"], clock=lambda: clock.now, advance=lambda dt, k: clock.now,
        band="narrow", band_index=2, stop=(0.0, 0.0, 0.0), positions={"a": (10.0, 0.0, 0.0)},
        range_planar_m=200.0, snr_fn=lambda j, d, t: snr_of_t(t),
        snr_floor_db=link.snr_floor_db, dwell_fn=lambda n, s: link.dwell_s(n, "narrow", s),
        rate_fn=lambda s: link.rate_bps("narrow", s), session_t_out_s=t_out,
        session_resumable=resumable), link


@pytest.mark.parametrize("t_out", [2.0, 10.0])
def test_the_session_walk_is_the_replays_walk_follow(t_out):
    R = _replay()

    def snr_of_t(t):                     # a wave that dips below the floor for ~8 s a period
        return -2.0 + 8.0 * math.sin(2.0 * math.pi * (t - 1.0e6) / 37.0)

    for nbytes in (18_800, 400_000, 2_000_000):
        for start in (0.0, 3.5, 11.0, 20.25):
            plan, link = _walk_plan(snr_of_t, t_out=t_out)
            t0 = 1.0e6 + start
            got = plan.session_walk("a", t0, nbytes)
            tr = SimpleNamespace(link=link, snr=lambda mode, did, c, d, pos, band, t: snr_of_t(t))
            st = SimpleNamespace(position=(0.0, 0.0, 0.0), band="narrow")
            s = SimpleNamespace(did="a", nbytes=nbytes, d=10.0)
            want = R.walk_follow(tr, "per_class", st, s, "narrow", t0, t_out)
            assert (got.dur_s, got.done, got.bits, got.capped) == (
                want.dur, want.done, want.bits, want.capped)


def test_a_session_fails_t_out_after_the_snr_falls_below_the_floor():
    t0 = 1.0e6
    plan, _ = _walk_plan(lambda t: 10.0 if t < t0 + 3.0 else -30.0, t_out=10.0)
    w = plan.session_walk("a", t0, 50_000_000)        # far too big to finish in 3 s
    assert not w.done and w.dur_s == pytest.approx(13.0) and 0.0 < w.bits < 8 * 50_000_000
    plan, _ = _walk_plan(lambda t: -30.0, t_out=1e9)
    w = plan.session_walk("a", t0, 1_000)
    assert w.capped and not w.done and w.dur_s >= SESSION_WALK_CAP_S


def test_the_plan_refuses_a_half_wired_session_model():
    link = ContactLink(anchor_planar_m=60.0)
    common = dict(clock=lambda: 0.0, advance=lambda dt, k: 0.0, band="wide", band_index=0,
                  stop=(0.0, 0.0, 0.0), positions={"a": (1.0, 0.0, 0.0)},
                  snr_fn=lambda j, d, t: 5.0, snr_floor_db=FLOOR,
                  dwell_fn=lambda n, s: link.dwell_s(n, "wide", s))
    with pytest.raises(TypeError, match="rate_fn"):
        ContactPlan.at_arrival(["a"], session_t_out_s=10.0, **common)
    with pytest.raises(ValueError, match="session_resumable"):
        ContactPlan(arrival_ts=0.0, targets=("a",), clock=lambda: 0.0,
                    advance=lambda dt, k: 0.0, session_resumable=True)
    plan = ContactPlan.at_arrival(["a"], **common)
    assert not plan.session_model and plan.repoll_candidates == () and plan.rate_fn is None


# --------------------------------------------------------------------------- #
# The host's commit under the session model and the re-poll
# --------------------------------------------------------------------------- #

def rate_fn(snr):
    """The toy link of test_host_mission_ferry, as a rate: 1 Mb/s per dB above the floor (+1)."""
    return 0.0 if snr < FLOOR else 1e6 * (snr - FLOOR + 1.0)


def toy_dwell(nbytes, snr):
    return None if snr < FLOOR else 8.0 * nbytes / rate_fn(snr)


def d1_plan(clock, members, snr_fn, *, t_out=10.0, resumable=False, repoll=False,
            positions=None, payload=1_000_000, model=True):
    positions = positions or {d: (1.0, 0.0, 0.0) for d in members}
    extra = dict(rate_fn=rate_fn, session_t_out_s=t_out, session_resumable=resumable) \
        if model else {}
    return ContactPlan.at_arrival(
        members, clock=clock, advance=clock.advance, band="wide", band_index=0,
        stop=(0.0, 0.0, 0.0), positions=positions, range_planar_m=60.0, snr_fn=snr_fn,
        snr_floor_db=FLOOR, dwell_fn=toy_dwell, payload_bytes=payload, repoll=repoll,
        **extra)


def _host(members, script=None):
    rf = FerryRF(members, script)
    clock = RecClock()
    host = make_host(rf, clock)
    host.open_round(theta(), theta_version=5)
    return rf, clock, host


def test_the_rate_follows_the_channel_and_a_failed_session_is_retried_at_the_end():
    rf, clock, host = _host(D[:3])
    t_arr = clock()

    def snr_fn(j, d, t):        # D1 passes the gate, then sits below the floor until 13 s
        if j == D[1]:
            return 3.0 if (t == t_arr or t >= t_arr + 13.0) else -30.0
        return 9.0

    plan = d1_plan(clock, D[:3], snr_fn)
    out = host.run_contact(D[:3], SYNTH, plan=plan)
    assert out == {d: MissionOutcome.CLEAN for d in D[:3]}
    commit = host.last_contact
    assert commit.session_retried == (D[1],) and commit.session_failed == ()
    # Walk by hand: D0 at 9 dB, D1 fails 10 s after it started, D2, then D1's retry.
    t = t_arr
    w0 = plan.session_walk(D[0], t, 2_000_000)
    t += w0.dur_s
    w1 = plan.session_walk(D[1], t, 2_000_000)
    assert not w1.done and w1.dur_s == pytest.approx(10.0)
    t += w1.dur_s
    w2 = plan.session_walk(D[2], t, 2_000_000)
    t += w2.dur_s
    w1b = plan.session_walk(D[1], t, 2_000_000)
    assert w1b.done
    t += w1b.dur_s
    lines = lines_by_id(host._report)
    assert lines[D[0]].contact_ts == t_arr + w0.dur_s
    assert lines[D[1]].contact_ts == t == clock()
    assert commit.session_dwell_s[D[1]] == pytest.approx(w1.dur_s + w1b.dur_s)
    assert clock.charges == [(t - t_arr, "dwell")]                 # no listen
    assert [g.device_id for g in host._accepted] == D[:3]


def test_an_update_whose_retry_also_fails_is_lost():
    rf, clock, host = _host(D[:2])
    t_arr = clock()
    plan = d1_plan(clock, D[:2], lambda j, d, t: 9.0 if (j == D[0] or t == t_arr) else -30.0)
    assert plan.targets == (D[0], D[1])
    out = host.run_contact(D[:2], SYNTH, plan=plan)
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.TIMEOUT}
    commit = host.last_contact
    assert commit.session_retried == commit.session_failed == (D[1],)
    assert commit.missing == () and commit.listen_s == 0.0
    assert [g.device_id for g in host._accepted] == [D[0]]
    line = lines_by_id(host._report)[D[1]]
    assert line.bytes_received == 0 and line.bytes_sent > 0
    assert commit.session_dwell_s[D[1]] == pytest.approx(20.0)     # two tries of t_out


def test_a_failed_delivery_is_undelivered():
    from hermes.types import DeliveryOutcome

    rf, clock, host = _host(D[:2])
    host.run_contact(D[:2], SYNTH, plan=d1_plan(clock, D[:2], lambda j, d, t: 9.0))
    host.close_round()
    host.open_pass_2(theta(2.0), theta_version=6)
    t_arr = clock()
    plan = d1_plan(clock, D[:2], lambda j, d, t: 9.0 if (j == D[0] or t == t_arr) else -30.0)
    out = host.deliver_contact(D[:2], SYNTH, plan=plan)
    assert out == {D[0]: DeliveryOutcome.DELIVERED, D[1]: DeliveryOutcome.UNDELIVERED}
    assert host.last_contact.session_failed == (D[1],)


def test_a_resumable_retry_sends_only_what_is_left():
    def run(resumable):
        rf, clock, host = _host(D[:1])
        t_arr = clock()

        def snr_fn(j, d, t):           # 5 s up, 12 s down (the try fails), then up for good
            return 9.0 if (t < t_arr + 5.0 or t >= t_arr + 17.0) else -30.0

        host.run_contact(D[:1], SYNTH, plan=d1_plan(clock, D[:1], snr_fn,
                                                     resumable=resumable, payload=10_000_000))
        return host.last_contact.session_dwell_s[D[0]], host.last_contact.session_retried

    fresh, retried = run(False)
    resumed, _ = run(True)
    assert retried == (D[0],)
    assert resumed == pytest.approx(fresh - 5.0)      # the 5 s already sent are not resent


def test_the_repoll_solicits_a_gated_member_once_the_stops_sessions_end():
    rf, clock, host = _host(D[:3])
    t_arr = clock()
    positions = {D[0]: (1.0, 0.0, 0.0), D[1]: (2.0, 0.0, 0.0), D[2]: (90.0, 0.0, 0.0)}

    def snr_fn(j, d, t):             # D1 below the floor at arrival only; D2 out of range
        if j == D[1]:
            return -30.0 if t == t_arr else 5.0
        return 9.0

    plan = d1_plan(clock, D[:3], snr_fn, repoll=True, positions=positions, model=False)
    assert plan.targets == (D[0],) and plan.repoll_candidates == (D[1],)
    out = host.run_contact(D[:3], SYNTH, plan=plan)
    assert out == {D[0]: MissionOutcome.CLEAN, D[1]: MissionOutcome.CLEAN,
                   D[2]: MissionOutcome.TIMEOUT}
    solicits = rf.solicits()
    w0 = toy_dwell(2_000_000, 9.0)
    assert [s[2] for s in solicits] == [(D[0],), (D[1],)]
    assert solicits[1][3] == t_arr + w0                 # issued when D0's session ended
    assert solicits[1][1] != solicits[0][1]             # a solicit number of its own
    commit = host.last_contact
    assert commit.repolled == (D[1],) and commit.unreachable == (D[1], D[2])
    lines = lines_by_id(host._report)
    assert [l.device_id for l in host._report.lines] == D[:3]      # one line each
    assert lines[D[1]].contact_ts == t_arr + w0 + toy_dwell(2_000_000, 5.0) == clock()
    assert [g.device_id for g in host._accepted] == [D[0], D[1]]


def test_the_repoll_leaves_a_member_still_below_the_floor_alone():
    rf, clock, host = _host(D[:2])
    plan = d1_plan(clock, D[:2], lambda j, d, t: 9.0 if j == D[0] else -30.0, repoll=True,
                   model=False)
    assert plan.repoll_candidates == (D[1],)
    out = host.run_contact(D[:2], SYNTH, plan=plan)
    assert out[D[1]] is MissionOutcome.TIMEOUT
    assert len(rf.solicits()) == 1 and host.last_contact.repolled == ()


def test_without_the_repoll_switch_no_member_is_a_candidate():
    clock = RecClock()
    plan = d1_plan(clock, D[:2], lambda j, d, t: 9.0 if j == D[0] else -30.0, model=False)
    assert plan.repoll_candidates == ()


# --------------------------------------------------------------------------- #
# The spec, the runtime's contact plan, and the causal estimate
# --------------------------------------------------------------------------- #

def _spec(**kw):
    kw.setdefault("rf_range_m", 60.0)
    kw.setdefault("seed", 11)
    kw.setdefault("contact_band", "narrow")
    kw.setdefault("contact_regime", "jittery")
    return FerrySpec.from_config(**kw)


D1_ON = dict(session_t_out_s=10.0, session_resumable=False, contact_repoll=True,
             interference_shared="narrow", burst_duty=0.1, burst_mean_s=6.7,
             burst_depth_db=17.0, channel_estimate="causal")


def test_the_spec_and_its_record_are_the_recorded_ones_while_off():
    off = _spec()
    explicit = _spec(**{k: None for k in D1_ON})
    assert off.describe() == explicit.describe()
    assert not {"session_t_out_s", "session_resumable", "contact_repoll",
                "channel_estimate"} & set(off.describe())
    on = _spec(**D1_ON).describe()
    assert on["session_t_out_s"] == 10.0 and on["contact_repoll"] is True
    assert on["channel_estimate"] == "causal"
    assert on["channel_params"]["contact"]["interference_shared"] == "narrow"
    assert on["channel_params"]["contact"]["burst_duty"] == 0.1


@pytest.mark.parametrize("kw, match", [
    (dict(session_resumable=True), "session_resumable"),
    (dict(session_t_out_s=0.0), "session_t_out_s"),
    (dict(channel_estimate="oracle"), "channel_estimate"),
    (dict(contact_band=None, contact_repoll=True), "contact band"),
    (dict(contact_band=None, burst_duty=0.1, burst_mean_s=6.7, burst_depth_db=17.0),
     "contact band"),
])
def test_the_spec_refuses_what_d1_cannot_run(kw, match):
    with pytest.raises(ValueError, match=match):
        _spec(**kw)


def _wp(*devs):
    return ContactWaypoint(position=(0.0, 0.0, 0.0), devices=tuple(DeviceID(d) for d in devs),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=1e7)


def test_the_runtimes_contact_plan_carries_the_switches():
    pos = {"a": (5.0, 0.0, 0.0), "b": (150.0, 0.0, 0.0)}
    rt = FerryRuntime(_spec(**D1_ON), MissionClock(), rf_range_m=60.0)
    p1 = rt.contact_plan(_wp("a", "b"), pos, pass_kind=COLLECT, mission_round=1)
    assert p1.session_t_out_s == 10.0 and p1.session_model and p1.rate_fn is not None
    assert p1.session_step_s == rt.spec.contact_channel.noise_bin_s
    assert set(p1.repoll_candidates) <= set(p1.unreachable)
    p2 = rt.contact_plan(_wp("a", "b"), pos, pass_kind=DELIVER, mission_round=1)
    assert p2.repoll_candidates == () and p2.session_model
    off = FerryRuntime(_spec(), MissionClock(), rf_range_m=60.0)
    p0 = off.contact_plan(_wp("a", "b"), pos, pass_kind=COLLECT, mission_round=1)
    assert not p0.session_model and p0.rate_fn is None and p0.repoll_candidates == ()


def test_the_causal_estimate_starts_at_the_regimes_prior_and_learns_from_arrivals():
    spec = _spec(channel_estimate="causal", interference_amp_db=17.0)
    rt = FerryRuntime(spec, MissionClock(), rf_range_m=60.0)
    amp0, sig0 = CONTACT_REGIMES["jittery"]
    mean, sigma = rt.channel_estimate_db()
    assert mean == 0.0
    assert sigma == pytest.approx(math.sqrt(spec.link.shadow_sigma_db ** 2 + sig0 ** 2
                                            + amp0 ** 2 / 2.0))
    oracle = FerryRuntime(_spec(interference_amp_db=17.0), MissionClock(), rf_range_m=60.0)
    d = 150.0
    assert rt.outage_probability(d) != oracle.outage_probability(d)
    # Readings on a 17 dB channel spread wider than the prior: the estimate follows.
    pos = {f"m{i}": (10.0 + 20.0 * i, 0.0, 0.0) for i in range(6)}
    for k in range(40):
        rt.clock.advance(7.3, "transit")
        rt.contact_plan(_wp(*pos), pos, pass_kind=COLLECT, mission_round=1)
    assert rt._est_n == 240
    _, learned = rt.channel_estimate_db()
    assert learned > sigma + 2.0


def test_the_prior_weight_is_declared():
    assert CAUSAL_PRIOR_READINGS == 12.0


# --------------------------------------------------------------------------- #
# Config, driver and runner
# --------------------------------------------------------------------------- #

def test_the_d1_fields_are_ferry_fields_sim_only_and_left_out_of_ferry_params_at_none():
    from experiments.exp4.driver import FERRY_PHYSICS_FIELDS

    import dataclasses

    defaults = {f.name: f.default for f in dataclasses.fields(C.MuleConfig)}
    for f in C.DDIL_MULE_FIELDS:
        assert defaults[f] is None
        assert C.FERRY_SPEC_FIELDS[f] == f and f in C.SIM_ONLY_MULE_FIELDS
        assert f in C.FERRY_PARAMS_OMITTED_AT_NONE and f in C.ADDENDUM_MULE_FIELDS
        assert f in FERRY_PHYSICS_FIELDS
    assert C.ADDENDUM_MULE_FIELDS[-len(C.DDIL_MULE_FIELDS):] == C.DDIL_MULE_FIELDS


def test_the_runner_takes_each_switch_as_a_flag():
    from experiments.exp4 import runner_main as RM

    names = {n for n, _k, _t in RM._PHYSICS_FLAGS}
    assert set(C.DDIL_MULE_FIELDS) <= names
    assert RM._bool_flag("true") is True and RM._bool_flag("Off") is False
    with pytest.raises(Exception):
        RM._bool_flag("maybe")
