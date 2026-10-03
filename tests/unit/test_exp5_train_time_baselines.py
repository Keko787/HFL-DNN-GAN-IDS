"""Exp 5 addendum, Study 5.12: the baselines that read the devices' update times.

With the training time on the simulated clock (``hermes/mule/fit_clock.py``)
two whole-scheduler baselines read it (the user's decisions of 2026-10-03):

* **D5, FedCS (option (a), a readiness test, no waiting).** At each step of
  Algorithm 3 the candidates are the contacts with a member whose update is
  ready at the predicted arrival; a contact that is not ready stays for later
  steps, the walk ends when none is, and under member subsets a candidate is
  reduced to its ready members. What it leaves out that way is labelled
  ``not_ready`` in the trace.
* **D2, Oort's restored system-speed term.** Each explored member's utility is
  multiplied by ``(T / t_i) ** 2`` when ``t_i > T``, with ``t_i`` its fit time
  plus its predicted dwell and ``T`` the cell's T_nom.

Unbound (no train times), both are the recorded policies, call for call. The
mule process binds the fit clock; the driver computes T_nom for D2, and the
config refuses D2 with train times but without it.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from experiments.exp4.driver import Exp4Driver
from hermes.mule.fit_clock import FitClock
from hermes.processes.config import MuleConfig, mule_config_errors
from hermes.scheduler.policies import OortPolicy
from hermes.scheduler.policies.fedcs_degraded import FedCSDegradedPolicy, fedcs_greedy_select
from hermes.scheduler.policies.oort import DEFAULT_SPEED_ALPHA
from hermes.scheduler.stages.s3b_feasibility import MemberSubsets

from tests.unit import test_fedcs_degraded as FC

NOW = FC.NOW
MODEL = FC.DEFAULT_MODEL          # 5 m/s, 1 s per session (the legacy pricing)


def _fits(times, *, origin=NOW):
    fc = FitClock(times)
    fc.take_off(origin)
    return fc


# --------------------------------------------------------------------------- #
# D5: FedCS reads the update time as a readiness test
# --------------------------------------------------------------------------- #

def test_unbound_the_walk_is_the_recorded_one():
    contacts = [FC._wp((10, 0), "a"), FC._wp((50, 0), "b")]
    assert fedcs_greedy_select(contacts, mule_pose=(0, 0, 0), now=NOW,
                               mission_deadline_ts=None, model=MODEL) == contacts
    assert fedcs_greedy_select(contacts, mule_pose=(0, 0, 0), now=NOW,
                               mission_deadline_ts=None, model=MODEL,
                               ready_at=lambda d: None) == contacts


def test_a_contact_not_ready_at_its_arrival_waits_for_a_later_step():
    a, b = FC._wp((10, 0), "a"), FC._wp((50, 0), "b")
    # a (2 s away) is ready only at NOW + 15; b (10 s away) now. From the
    # dock only b is a candidate; after b (clock NOW + 11) a's arrival is
    # NOW + 11 + 8 = NOW + 19 >= NOW + 15, so it is picked next.
    ends = {"a": NOW + 15.0, "b": NOW}
    left = set()
    route = fedcs_greedy_select([a, b], mule_pose=(0, 0, 0), now=NOW, mission_deadline_ts=None,
                                model=MODEL, ready_at=ends.get, not_ready_out=left)
    assert route == [b, a] and left == set()


def test_the_walk_ends_when_no_remaining_contact_will_be_ready():
    a, b = FC._wp((10, 0), "a"), FC._wp((50, 0), "b")
    ends = {"a": NOW + 1e4, "b": NOW}
    left = set()
    route = fedcs_greedy_select([a, b], mule_pose=(0, 0, 0), now=NOW, mission_deadline_ts=None,
                                model=MODEL, ready_at=ends.get, not_ready_out=left)
    assert route == [b] and left == {"a"}
    nobody = set()
    assert fedcs_greedy_select([a], mule_pose=(0, 0, 0), now=NOW, mission_deadline_ts=None,
                               model=MODEL, ready_at=lambda d: NOW + 1e4,
                               not_ready_out=nobody) == [] and nobody == {"a"}


def test_whole_stops_fly_with_a_ready_member_and_subsets_keep_only_the_ready():
    stop = FC._wp((10, 0), "a", "b")
    ends = {"a": NOW, "b": NOW + 1e4}
    whole = fedcs_greedy_select([stop], mule_pose=(0, 0, 0), now=NOW, mission_deadline_ts=None,
                                model=MODEL, ready_at=ends.get)
    assert whole == [stop]                       # priced whole, as S3a formed it
    states = FC._states([stop])
    subsets = MemberSubsets({"a": NOW + 1e6, "b": NOW + 1e6}, states)
    left = set()
    (reduced,) = fedcs_greedy_select([stop], mule_pose=(0, 0, 0), now=NOW,
                                     mission_deadline_ts=NOW + 1e3, model=MODEL,
                                     member_subsets=subsets, ready_at=ends.get,
                                     not_ready_out=left)
    assert reduced.devices == ("a",) and reduced.position == stop.position
    assert left == {"b"}


def test_the_policy_reads_the_fit_clock_only_once_bound():
    a, b = FC._wp((10, 0), "a"), FC._wp((50, 0), "b")
    env = FC._env()
    policy = FedCSDegradedPolicy()
    states = FC._states([a, b])
    assert policy.admit_and_order([a, b], states, env, feasibility_model=MODEL) == [a, b]
    assert policy.last_not_ready == frozenset()
    policy.bind_fit_clock(_fits({"a": 1e4, "b": 0.0}), t_nom_s=None)
    assert policy.admit_and_order([a, b], states, env, feasibility_model=MODEL) == [b]
    assert policy.last_not_ready == {"a"}
    # The ordering-only surface stays a permutation (no admission cut).
    assert sorted(policy.rank_contacts([a, b], states, env), key=id) == sorted([a, b], key=id)


# --------------------------------------------------------------------------- #
# D2: Oort's system-speed term
# --------------------------------------------------------------------------- #

def test_the_speed_penalty_is_oorts():
    p = OortPolicy()
    assert DEFAULT_SPEED_ALPHA == 2.0 and p.speed_alpha == 2.0
    with pytest.raises(ValueError, match="T_nom"):
        p.bind_fit_clock(_fits({"a": 1.0}), t_nom_s=None)
    p.bind_fit_clock(_fits({"a": 1.0}), t_nom_s=100.0)
    assert p.speed_penalty(50.0) == p.speed_penalty(100.0) == 1.0
    assert p.speed_penalty(200.0) == pytest.approx(0.25)
    assert p.speed_penalty(math.inf) == 0.0


def test_the_round_time_is_the_fit_plus_the_members_dwell():
    p = OortPolicy()
    p.bind_fit_clock(_fits({"a": 40.0}), t_nom_s=100.0)
    wp = FC._wp((10, 0), "a", "z")
    assert p.round_time_s(wp, "a", None) == 40.0
    assert p.round_time_s(wp, "a", MODEL) == 41.0              # one session, no band
    assert p.round_time_s(wp, "z", MODEL) == 1.0               # no train time: 0


def _oort_states(contacts):
    states = FC._states(contacts)
    for st in states.values():
        st.last_loss, st.last_num_examples = 0.5, 100
    return states


def test_a_slow_device_ranks_below_an_equal_fast_one_only_with_the_term():
    slow, fast = FC._wp((10, 0), "s"), FC._wp((10, 1), "f")
    states = _oort_states([slow, fast])
    env = dataclasses.replace(FC._env(), mission_round=3)
    plain = OortPolicy().admit_and_order([slow, fast], states, env, feasibility_model=MODEL)
    timed = OortPolicy()
    timed.bind_fit_clock(_fits({"s": 400.0, "f": 10.0}), t_nom_s=100.0)
    ranked = timed.admit_and_order([slow, fast], states, env, feasibility_model=MODEL)
    assert ranked[0] is fast and set(map(id, ranked)) == set(map(id, plain))
    key = timed._rank_key(states, 3, model=MODEL)
    base = OortPolicy()._rank_key(states, 3, model=MODEL)
    # s: t = 401 s > T = 100 s, so its utility is scaled by (100 / 401)^2.
    assert key(slow)[0] == pytest.approx(base(slow)[0] * (100.0 / 401.0) ** 2)
    assert key(fast)[0] == pytest.approx(base(fast)[0])


# --------------------------------------------------------------------------- #
# Config, driver, and whole trials
# --------------------------------------------------------------------------- #

def test_d2_with_train_times_needs_t_nom():
    sim = dict(mule_id="m", mission_clock="sim", rf_range_m=60.0, trial_seed=1,
               device_train_time_s={"d": 3.0}, train_time_params={"median_s": 3.0})
    assert any("T_nom" in e for e in mule_config_errors(MuleConfig(**sim, contact_policy="oort")))
    assert not any("T_nom" in e for e in mule_config_errors(
        MuleConfig(**sim, contact_policy="oort", t_nom_s=200.0)))
    assert not any("T_nom" in e for e in mule_config_errors(
        MuleConfig(**sim, contact_policy="fedcs")))
    d = Exp4Driver(mission_clock="sim", train_time_params={"median_s": 30.0})
    assert d._needs_t_nom("D2") and not d._needs_t_nom("D5")
    assert not Exp4Driver(mission_clock="sim")._needs_t_nom("D2")


def _capture(settings, cell):
    from tests.golden import _build_p3_sim as UG4
    from tests.golden.test_golden_p4_plan import plain

    driver = Exp4Driver(**settings)
    with UG4.in_process_orchestrator():
        UG4.InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        orch = UG4.InProcessOrchestrator.last
    return plain(UG4.case_of(settings, cell, row, orch))


TT = {"median_s": 60.0, "straggler_share": 0.34, "straggler_factor": 4.0}


@pytest.fixture(scope="module")
def d5():
    from tests.golden import _build_p3_sim as UG4

    settings, cell = UG4.TRIALS["d3_whittle"]
    cell = dataclasses.replace(cell, arm="D5")
    return (_capture(dict(settings), cell),
            _capture(dict(settings, train_time_params=TT), cell))


def _missions(case):
    (missions,) = case["mission_completed"].values()
    return missions


def test_a_d5_trial_flies_ready_members_and_labels_what_it_left(d5):
    plain, timed = d5
    ((ready,),) = timed["mule_ready"].values()
    assert ready["train_time_policy"] == "FEDCS"
    ((plain_ready,),) = plain["mule_ready"].values()
    assert "train_time_policy" not in plain_ready
    assert all(d["reason"] != "not_ready"
               for m in _missions(plain) for d in m.get("pass_1_policy_drops") or ())
    reasons = [d["reason"] for m in _missions(timed) for d in m.get("pass_1_policy_drops") or ()]
    assert "not_ready" in reasons
    (mule,) = [cfg for name, cfg in timed["configs"].items() if name.startswith("mule")]
    times = mule["device_train_time_s"]
    from types import SimpleNamespace

    from experiments.analysis.traces_scorer import compute_report
    from experiments.exp4 import events_consumer as EC

    records = [EC.MissionRecord(
        mission_round=m["mission_round"], pass_1_contacts=0, pass_2_contacts=0,
        pass_1_updates=None, pass_1_scheduled=None, pass_1_clean_devices=(), delivered=None,
        undelivered=None, duration_s=None, mule_id="exp4-mule", sim_end_s=m["sim_end_s"],
        train_fits=EC._train_fits(m["train_fits"]), fit_stops=EC._fit_stops(m["pass_1_flown"]),
        policy_drops=EC._policy_drops(m.get("pass_1_policy_drops"))) for m in _missions(timed)]
    got = compute_report(SimpleNamespace(missions=records), [mule])
    assert got.policy_not_ready_drops == sum(
        len(d["devices"]) for m in _missions(timed) for d in m.get("pass_1_policy_drops") or ()
        if d["reason"] == "not_ready") > 0
    for m in _missions(timed):
        for stop in m["pass_1_flown"]:
            # FedCS flew it only with a member ready at the arrival.
            assert set(stop["not_ready"]) < set(stop["devices"]) or not stop["devices"]
        for drop in m.get("pass_1_policy_drops") or ():
            if drop["reason"] == "not_ready":
                assert all(d in times for d in drop["devices"])


def test_a_d2_trial_binds_oort_with_t_nom():
    from tests.golden import _build_p3_sim as UG4

    settings, cell = UG4.TRIALS["d3_whittle"]
    settings = {k: v for k, v in settings.items() if k != "deadline_time_scale"}
    cell = dataclasses.replace(cell, arm="D2")
    timed = _capture(dict(settings, real_model=True, data_source="synthetic",
                          train_time_params=TT), cell)
    ((ready,),) = timed["mule_ready"].values()
    assert ready["train_time_policy"] == "OORT" and ready["t_nom_s"] > 0
    assert timed["row"]["t_nom_s"] == ready["t_nom_s"]
