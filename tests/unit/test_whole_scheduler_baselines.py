"""Whole-scheduler baselines (arms D1/D2) — Freeze Amendment 4.

The earlier ordering-only arms were vacuous: S3b decides *who* is served before
any ranking policy runs, so a baseline confined to the selector slot could only
permute a list our gate had already decided, and every arm produced identical
results. These arms fix that by giving the baseline **admission authority** — it
replaces S3, S3b and S3.5 and returns the route itself.

What must hold for the comparison to be fair, and is pinned here:

* the baseline admits a **different set** than our gate would (or we are vacuous
  again);
* every arm faces the **same budget** and the **same travel cost model**, so the
  experiment measures the decision rule rather than whose physics is cheaper;
* delegation is **inert** for every arm that does not implement the method.
"""

from __future__ import annotations

import pytest

from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.policies import MaxAoIPolicy, OortPolicy
from hermes.scheduler.policies.budget_walk import greedy_budget_walk
from hermes.scheduler.selector.features import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import (
    Bucket, ContactWaypoint, DeviceID, DeviceSchedulerState,
    MissionSlice, MuleID,
)

NOW = 1_000.0
#: 1 m/s and no session time, so travel seconds == metres. Keeps the budget
#: arithmetic in these tests readable.
MODEL = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=0.0)


def _wp(x: float, *devs: str) -> ContactWaypoint:
    return ContactWaypoint(
        position=(x, 0.0, 0.0),
        devices=tuple(DeviceID(d) for d in devs),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=0.0,
    )


def _env(pose=(0.0, 0.0, 0.0), now=NOW) -> SelectorEnv:
    return SelectorEnv(mule_pose=pose, mule_energy=1.0, rf_prior_snr_db=20.0,
                       beacon_window_s=30.0, now=now)


def _states(**last_seen):
    return {DeviceID(d): DeviceSchedulerState(device_id=DeviceID(d),
                                              last_contact_ts=ts)
            for d, ts in last_seen.items()}


# --------------------------------------------------------------------------- #
# 1. The shared budget walk
# --------------------------------------------------------------------------- #

def test_the_walk_admits_only_what_fits():
    """10 + 20 + 30 m at 1 m/s = 60 s; a 35 s budget affords the first two."""
    wps = [_wp(10.0, "a"), _wp(30.0, "b"), _wp(60.0, "c")]
    got = greedy_budget_walk(wps, key=lambda w: w.position[0],
                             mule_pose=(0.0, 0.0, 0.0), now=NOW,
                             mission_deadline_ts=NOW + 35.0, model=MODEL)
    assert [w.devices[0] for w in got] == ["a", "b"]


def test_an_unaffordable_contact_is_skipped_not_fatal():
    """A distant high-rank contact must not veto a near one that still fits."""
    wps = [_wp(1_000.0, "far"), _wp(5.0, "near")]
    got = greedy_budget_walk(wps, key=lambda w: 0 if w.devices[0] == "far" else 1,
                             mule_pose=(0.0, 0.0, 0.0), now=NOW,
                             mission_deadline_ts=NOW + 50.0, model=MODEL)
    assert [w.devices[0] for w in got] == ["near"]


def test_cost_accumulates_along_the_route():
    """Each leg is priced from the PREVIOUS stop, not from the origin."""
    wps = [_wp(10.0, "a"), _wp(20.0, "b"), _wp(30.0, "c")]
    got = greedy_budget_walk(wps, key=lambda w: w.position[0],
                             mule_pose=(0.0, 0.0, 0.0), now=NOW,
                             mission_deadline_ts=NOW + 25.0, model=MODEL)
    # legs are 10 then 10 then 10; 25 s affords two.
    assert [w.devices[0] for w in got] == ["a", "b"]


def test_no_budget_admits_everything_in_rank_order():
    wps = [_wp(3.0, "c"), _wp(1.0, "a"), _wp(2.0, "b")]
    got = greedy_budget_walk(wps, key=lambda w: w.position[0],
                             mule_pose=(0.0, 0.0, 0.0), now=NOW,
                             mission_deadline_ts=None, model=MODEL)
    assert [w.devices[0] for w in got] == ["a", "b", "c"]


def test_a_zero_budget_admits_nothing():
    got = greedy_budget_walk([_wp(1.0, "a")], key=lambda w: 0.0,
                             mule_pose=(0.0, 0.0, 0.0), now=NOW,
                             mission_deadline_ts=NOW, model=MODEL)
    assert got == []


# --------------------------------------------------------------------------- #
# 2. D1 — MAX-AoI admits by age
# --------------------------------------------------------------------------- #

def test_d1_admits_the_stalest_first_under_budget():
    fresh, stale = _wp(10.0, "fresh"), _wp(20.0, "stale")
    st = _states(fresh=NOW - 5, stale=NOW - 900)
    got = MaxAoIPolicy().admit_and_order(
        [fresh, stale], st, _env(),
        mission_deadline_ts=NOW + 25.0, feasibility_model=MODEL,
    )
    # 20 m to 'stale' fits in 25 s; 'fresh' is then 10 m further and does not.
    assert [w.devices[0] for w in got] == ["stale"]


def test_d1_drops_what_it_cannot_reach():
    wps = [_wp(10.0, "a"), _wp(500.0, "b")]
    st = _states(a=NOW - 100, b=NOW - 900)      # b is staler but far
    got = MaxAoIPolicy().admit_and_order(
        wps, st, _env(), mission_deadline_ts=NOW + 60.0, feasibility_model=MODEL)
    assert [w.devices[0] for w in got] == ["a"]


def test_d1_admits_everything_when_the_budget_is_ample():
    wps = [_wp(1.0, "a"), _wp(2.0, "b")]
    st = _states(a=NOW - 10, b=NOW - 20)
    got = MaxAoIPolicy().admit_and_order(
        wps, st, _env(), mission_deadline_ts=NOW + 1e6, feasibility_model=MODEL)
    assert len(got) == 2


# --------------------------------------------------------------------------- #
# 3. D2 — Oort admits by utility
# --------------------------------------------------------------------------- #

def _oort_state(did, loss, n, served_round=1):
    s = DeviceSchedulerState(device_id=DeviceID(did), last_contact_ts=NOW - 10)
    s.last_loss, s.last_num_examples, s.last_served_round = loss, n, served_round
    s.on_time_count = 1
    return s


def test_d2_admits_the_highest_utility_first():
    lo, hi = _wp(10.0, "lo"), _wp(20.0, "hi")
    st = {DeviceID("lo"): _oort_state("lo", 0.1, 10),
          DeviceID("hi"): _oort_state("hi", 0.9, 100)}
    got = OortPolicy().admit_and_order(
        [lo, hi], st, _env(), mission_deadline_ts=NOW + 25.0,
        feasibility_model=MODEL)
    assert [w.devices[0] for w in got] == ["hi"]


def test_d2_respects_the_budget():
    wps = [_wp(10.0, "a"), _wp(500.0, "b")]
    st = {DeviceID("a"): _oort_state("a", 0.1, 10),
          DeviceID("b"): _oort_state("b", 9.9, 999)}   # far but high utility
    got = OortPolicy().admit_and_order(
        wps, st, _env(), mission_deadline_ts=NOW + 60.0, feasibility_model=MODEL)
    assert [w.devices[0] for w in got] == ["a"]


# --------------------------------------------------------------------------- #
# 4. The delegation hook — the whole point
# --------------------------------------------------------------------------- #

def _scheduler(policy, *, budget, devices, now=NOW):
    sch = FLScheduler(now_fn=lambda: now, target_selector=policy,
                      mission_budget_s=budget, feasibility_model=MODEL)
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=tuple(DeviceID(d) for d in devices),
        issued_round=1, issued_at=now,
    ))
    for i, d in enumerate(devices):
        sch.device_states[DeviceID(d)].last_known_position = (i * 40.0, 0.0, 0.0)
    return sch


def test_a_policy_without_admit_and_order_is_unaffected():
    """H0-H3 must run exactly as before — delegation is opt-in by method."""
    class _OrderingOnly:
        name = "ORDER"

        def rank_contacts(self, c, s, env=None, **kw):
            return list(c)

    sch = _scheduler(_OrderingOnly(), budget=1e6, devices=("a", "b", "c"))
    assert sch.build_contact_queue(rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0),
                                   mule_energy=1.0)


def test_the_baseline_really_replaces_our_gate():
    """The route must come from the policy, not from S3b.

    Pinned with a policy that admits nothing: if S3b were still deciding, the
    queue would be non-empty.
    """
    class _AdmitsNothing:
        name = "NONE"

        def rank_contacts(self, c, s, env=None, **kw):
            return list(c)

        def admit_and_order(self, c, s, env=None, **kw):
            return []

    sch = _scheduler(_AdmitsNothing(), budget=1e6, devices=("a", "b", "c"))
    assert sch.build_contact_queue(rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0),
                                   mule_energy=1.0) == []


def test_the_budget_reaches_the_policy():
    seen = {}

    class _Recorder:
        name = "REC"

        def rank_contacts(self, c, s, env=None, **kw):
            return list(c)

        def admit_and_order(self, c, s, env=None, *, mission_deadline_ts=None, **kw):
            seen["deadline"] = mission_deadline_ts
            return list(c)

    sch = _scheduler(_Recorder(), budget=120.0, devices=("a", "b"))
    sch.build_contact_queue(rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0),
                            mule_energy=1.0)
    assert seen["deadline"] == pytest.approx(NOW + 120.0)


def test_our_feasibility_model_is_handed_to_the_policy():
    """Both arms must price travel identically, or we measure the cost model."""
    seen = {}

    class _Recorder:
        name = "REC"

        def rank_contacts(self, c, s, env=None, **kw):
            return list(c)

        def admit_and_order(self, c, s, env=None, *, feasibility_model=None, **kw):
            seen["model"] = feasibility_model
            return list(c)

    sch = _scheduler(_Recorder(), budget=120.0, devices=("a", "b"))
    sch.build_contact_queue(rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0),
                            mule_energy=1.0)
    assert seen["model"] is MODEL


def test_no_budget_still_delegates():
    """Without enforcement the policy still owns ordering, with no admission cut."""
    sch = _scheduler(MaxAoIPolicy(), budget=None, devices=("a", "b", "c"))
    got = sch.build_contact_queue(rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0),
                                  mule_energy=1.0)
    assert len(got) == 3
