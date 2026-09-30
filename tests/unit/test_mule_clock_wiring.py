"""FeRRy Phase 3, unit U6: wiring the mission clock into the mule.

* ``MuleSupervisor`` on the clock: the clock is ``_now`` (never ``_clock``,
  critic B7), the scheduler's and the host's ``now_fn``; the scheduler prices
  with the ferry model, refuses wall-clock overrides and checks the flown
  order under ``replan``; the refusals of design section 2.2; the legacy
  construction is untouched.
* The sim-time plumbing (critic B8): ``UpBundle.sim_upload_ts`` and
  ``backhaul``, ``DownBundle.cluster_sim_ts``, ``ClientCluster`` carrying them,
  and the bootstrap DOWN's Lamport sync (``advance_to``), also for a mule
  restarted with a fresh clock.
* Oort (D2) at a re-plan ranks with the round its plan used.
"""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hermes.l1.mission_clock import MissionClock, SIM_EPOCH_S
from hermes.mule import ClientCluster, MuleSupervisor, MuleSupervisorError
from hermes.mule.ferry import FerrySpec
from hermes.mule.mule_main import mission_planned_devices
from hermes.scheduler.policies import OortPolicy
from hermes.scheduler.policies.budget_walk import greedy_budget_walk
from hermes.scheduler.policies.oort import statistical_utility
from hermes.scheduler.selector import SelectorEnv
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel, FeasibilityResult
from hermes.transport import LoopbackDockLink, LoopbackRFLink
from hermes.types import (
    Bucket,
    ClusterAmendment,
    ContactHistory,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    DownBundle,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    UpBundle,
    sign_down_bundle,
    sign_up_bundle,
    verify_up_bundle,
)
from hermes.types.bundles import BackhaulUpload

MULE = MuleID("m1")


def _sup(**kw):
    kw.setdefault("rf_range_m", 60.0)
    return MuleSupervisor(mule_id=MULE, rf=LoopbackRFLink(), dock=LoopbackDockLink(), **kw)


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

def test_the_clock_is_now_for_the_supervisor_the_scheduler_and_the_host():
    clock = MissionClock()
    sup = _sup(mission_clock=clock)
    assert sup._now is clock and sup.mission_clock is clock
    assert sup.scheduler._now is clock and sup.mission.now_fn is clock
    # Critic B7: tests bind supervisor methods onto stand-ins carrying _clock.
    assert not hasattr(sup, "_clock")
    assert isinstance(sup.ferry, FerrySpec) and sup.ferry.band is None
    sch = sup.scheduler
    assert sch.refuses_deadline_overrides and not sch.validates_flown_order
    assert sch.feasibility_model.ferry is not None
    assert sch.feasibility_model.ferry.device_states is sch.device_states   # B11
    assert sch.feasibility_model.cruise_speed_m_s == 5.0
    assert sch.feasibility_model.session_time_s == 1.0


def test_replan_turns_on_the_flown_order_check_and_its_fallback():
    spec = FerrySpec(in_flight_response="replan", replan_fallback="trim")
    sch = _sup(mission_clock=MissionClock(), ferry=spec).scheduler
    assert sch.validates_flown_order and sch.replan_fallback == "trim"


def test_the_time_scale_reaches_the_scheduler():
    sch = _sup(mission_clock=MissionClock(), deadline_time_scale=20.0,
               initial_window_s=90.0).scheduler
    assert sch.deadline_time_scale == 20.0 and sch.effective_initial_window_s == 1800.0


def test_the_wall_clock_mule_is_built_as_before():
    sup = _sup()
    assert sup._now is time.time and sup.mission_clock is None
    assert sup.ferry is None and sup._ferry_run is None
    assert sup.scheduler.feasibility_model is None
    assert sup.mission.now_fn is None
    assert not sup.scheduler.refuses_deadline_overrides
    assert not sup.scheduler.validates_flown_order
    assert sup.scheduler.deadline_law is None
    model = FeasibilityModel(cruise_speed_m_s=2.0)
    assert _sup(feasibility_model=model).scheduler.feasibility_model is model


@pytest.mark.parametrize("kw, match", [
    (dict(mission_clock=MissionClock(), now_fn=lambda: 0.0), "exclusive"),
    (dict(mission_clock=MissionClock(), rf_range_m=None), "single-pass"),
    (dict(ferry=FerrySpec()), "mission clock"),
    (dict(ferry=FerrySpec(in_flight_response="replan")), "mission clock"),
    (dict(mission_clock=MissionClock(), mule_pose=(5.0, 0.0, 0.0)), "starts at the dock"),
    (dict(mission_clock=MissionClock(),
          feasibility_model=FeasibilityModel(cruise_speed_m_s=3.0)), "m/s"),
    (dict(mission_clock=lambda: 1.0), "MissionClock"),
    (dict(mission_clock=MissionClock(), ferry="wide"), "FerrySpec"),
    (dict(mission_clock=MissionClock(),
          ferry=FerrySpec.from_config(rf_range_m=50.0, seed=1, contact_band="wide")),
     "R_planar"),
])
def test_what_the_clock_refuses(kw, match):
    with pytest.raises(MuleSupervisorError, match=match):
        _sup(**kw)


def test_a_ferry_physics_model_needs_the_clock():
    sup = _sup(mission_clock=MissionClock())
    with pytest.raises(MuleSupervisorError, match="mission clock"):
        _sup(feasibility_model=sup.scheduler.feasibility_model)


def test_the_scheduler_refuses_wall_clock_overrides_in_sim_mode():
    sup = _sup(mission_clock=MissionClock())
    amendment = ClusterAmendment(cluster_round=1, deadline_overrides={DeviceID("d"): 1.7e9})
    sl = MissionSlice(mule_id=MULE, device_ids=(DeviceID("d"),), issued_round=1, issued_at=0.0)
    with pytest.raises(Exception, match="wall-clock stamps"):
        sup.scheduler.ingest_slice(sl, amendment=amendment)


def test_the_beacon_hook_needs_the_clock():
    wp = ContactWaypoint(position=(0.0, 0.0, 0.0), devices=(DeviceID("d"),),
                         bucket=Bucket.NEW, deadline_ts=0.0)
    with pytest.raises(MuleSupervisorError, match="mission clock"):
        _sup().offer_contact(wp)
    sup = _sup(mission_clock=MissionClock())
    sup.offer_contact(wp)
    assert sup._offers == [wp]
    with pytest.raises(TypeError):
        sup.offer_contact("d")


def test_stand_ins_without_the_new_attributes_take_the_recorded_path():
    """Critic B7: ``run_one_mission`` reads the ferry runtime with getattr."""
    calls = []
    stand_in = SimpleNamespace(
        _next_theta=object(),
        scheduler=SimpleNamespace(start_mission=lambda: calls.append("stamp")),
        rf_range_m=60.0,
        _run_two_pass_mission=lambda: calls.append("two_pass") or "legacy",
    )
    assert MuleSupervisor.run_one_mission(stand_in) == "legacy"
    assert calls == ["stamp", "two_pass"]


def test_planned_devices_count_the_energy_drops():
    """Critic B10: S3c's planned count includes the energy clause's drops."""
    wp = lambda *d: ContactWaypoint(position=(0.0, 0.0, 0.0),  # noqa: E731
                                    devices=tuple(DeviceID(x) for x in d),
                                    bucket=Bucket.NEW, deadline_ts=0.0)
    feas = FeasibilityResult([wp("a")], [wp("b")], [wp("c", "d")], [wp("e", "f", "g")])
    assert mission_planned_devices([wp("a")], feas) == 1 + 1 + 2 + 3
    legacy = SimpleNamespace(dropped_overdue=[wp("b")], dropped_budget=[])
    assert mission_planned_devices([wp("a")], legacy) == 2


# --------------------------------------------------------------------------- #
# Bundles and the ClientCluster (critic B8)
# --------------------------------------------------------------------------- #

def _agg(round_=1):
    return PartialAggregate(mule_id=MULE, mission_round=round_,
                            weights=[np.array([0.1, 0.2], dtype=np.float32)],
                            num_examples=4, contributing_devices=(DeviceID("d1"),))


def _report(round_=1):
    r = MissionRoundCloseReport(mule_id=MULE, mission_round=round_,
                                started_at=1e6, finished_at=1e6 + 1)
    r.append(MissionRoundCloseLine(device_id=DeviceID("d1"), outcome=MissionOutcome.CLEAN,
                                   contact_ts=1e6 + 0.5))
    return r


def _down(round_=1, cluster_sim_ts=None, overrides=None):
    b = DownBundle(
        mule_id=MULE,
        mission_slice=MissionSlice(mule_id=MULE, device_ids=(DeviceID("d1"),),
                                   issued_round=round_, issued_at=0.0),
        theta_disc=[np.zeros((2,), dtype=np.float32)],
        synth_batch=[np.ones((3,), dtype=np.float32)],
        cluster_amendments=ClusterAmendment(cluster_round=round_,
                                            deadline_overrides=dict(overrides or {})),
        cluster_sim_ts=cluster_sim_ts,
    )
    sign_down_bundle(b)
    return b


UP_BH = BackhaulUpload(carrier=2, snr_db=7.5, p_loss=0.1, t_start_s=1e6 + 10,
                       upload_s=0.25, nbytes=18_756)


def test_the_new_bundle_fields_default_to_none_and_are_not_signed():
    up = UpBundle(mule_id=MULE, partial_aggregate=_agg(), round_close_report=_report(),
                  contact_history=ContactHistory(mule_id=MULE, mission_round=1))
    assert up.sim_upload_ts is None and up.backhaul is None
    legacy_sig = sign_up_bundle(up)
    sim = UpBundle(mule_id=MULE, partial_aggregate=_agg(), round_close_report=_report(),
                   contact_history=ContactHistory(mule_id=MULE, mission_round=1),
                   sim_upload_ts=1e6 + 10.25, backhaul=UP_BH)
    assert sign_up_bundle(sim) == legacy_sig and verify_up_bundle(sim)
    assert _down().cluster_sim_ts is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, "soon", True])
def test_sim_stamps_must_be_finite_numbers(bad):
    with pytest.raises((TypeError, ValueError)):
        UpBundle(mule_id=MULE, partial_aggregate=_agg(), round_close_report=_report(),
                 contact_history=ContactHistory(mule_id=MULE, mission_round=1),
                 sim_upload_ts=bad)
    with pytest.raises((TypeError, ValueError)):
        _down(cluster_sim_ts=bad)
    with pytest.raises(TypeError):
        UpBundle(mule_id=MULE, partial_aggregate=_agg(), round_close_report=_report(),
                 contact_history=ContactHistory(mule_id=MULE, mission_round=1),
                 backhaul={"carrier": 1})


def test_the_client_cluster_carries_sim_time_up_and_reads_it_down():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=MULE, dock=dock)
    assert cc.last_cluster_sim_ts() is None
    cc.collect(partial_aggregate=_agg(), report=_report(),
               contacts=ContactHistory(mule_id=MULE, mission_round=1),
               sim_upload_ts=1e6 + 10.25, backhaul=UP_BH)
    dock.send_down(_down(cluster_sim_ts=1e6 + 42.0))
    down = cc.run_dock_cycle()
    up = dock.recv_up(timeout=1.0)
    assert up.sim_upload_ts == 1e6 + 10.25 and up.backhaul == UP_BH
    assert down.cluster_sim_ts == 1e6 + 42.0 and cc.last_cluster_sim_ts() == 1e6 + 42.0
    # The stage is cleared with the rest: a legacy collect uploads None.
    cc.collect(partial_aggregate=_agg(2), report=_report(2),
               contacts=ContactHistory(mule_id=MULE, mission_round=2))
    dock.send_down(_down(2))
    cc.run_dock_cycle()
    up2 = dock.recv_up(timeout=1.0)
    assert up2.sim_upload_ts is None and up2.backhaul is None
    assert cc.last_cluster_sim_ts() is None


class _FailingDock(LoopbackDockLink):
    def __init__(self):
        super().__init__()
        self.fail = True

    def client_send_up(self, bundle):
        from hermes.transport import DockLinkError
        if self.fail:
            raise DockLinkError("down")
        return super().client_send_up(bundle)


def test_a_retried_upload_keeps_its_sim_time():
    dock = _FailingDock()
    cc = ClientCluster(mule_id=MULE, dock=dock)
    cc.collect(partial_aggregate=_agg(), report=_report(),
               contacts=ContactHistory(mule_id=MULE, mission_round=1),
               sim_upload_ts=1e6 + 10.25, backhaul=UP_BH)
    assert cc.run_dock_cycle() is None and cc.retry_queue_depth() == 1
    dock.fail = False
    dock.send_down(_down())
    cc.run_dock_cycle()
    up = dock.recv_up(timeout=1.0)
    assert up.sim_upload_ts == 1e6 + 10.25 and up.backhaul == UP_BH


# --------------------------------------------------------------------------- #
# The bootstrap DOWN and a restarted mule (critic B8)
# --------------------------------------------------------------------------- #

def _bootstrapped(clock, cluster_sim_ts):
    dock = LoopbackDockLink()
    sup = MuleSupervisor(mule_id=MULE, rf=LoopbackRFLink(), dock=dock, rf_range_m=60.0,
                         mission_clock=clock)
    dock.send_down(_down(cluster_sim_ts=cluster_sim_ts))
    assert sup.wait_for_initial_dock(timeout=1.0)
    return sup


def test_the_bootstrap_down_syncs_a_fresh_clock_to_the_cluster():
    clock = MissionClock()
    _bootstrapped(clock, SIM_EPOCH_S + 5_000.0)
    assert clock() == SIM_EPOCH_S + 5_000.0 and clock.ledger()["dock_wait"] == 5_000.0


def test_a_clock_ahead_of_the_cluster_does_not_move_and_none_means_no_sync():
    clock = MissionClock()
    clock.advance(900.0, "transit")
    _bootstrapped(clock, SIM_EPOCH_S + 100.0)
    assert clock() == SIM_EPOCH_S + 900.0
    fresh = MissionClock()
    _bootstrapped(fresh, None)
    assert fresh() == SIM_EPOCH_S


def test_a_wall_stamp_from_the_cluster_is_refused():
    dock = LoopbackDockLink()
    sup = MuleSupervisor(mule_id=MULE, rf=LoopbackRFLink(), dock=dock, rf_range_m=60.0,
                         mission_clock=MissionClock())
    dock.send_down(_down(cluster_sim_ts=1.7e9))
    with pytest.raises(MuleSupervisorError, match="cluster_sim_ts"):
        sup.wait_for_initial_dock(timeout=1.0)


def test_a_wall_clock_mule_ignores_cluster_sim_ts():
    dock = LoopbackDockLink()
    sup = MuleSupervisor(mule_id=MULE, rf=LoopbackRFLink(), dock=dock, rf_range_m=60.0)
    dock.send_down(_down(cluster_sim_ts=SIM_EPOCH_S + 5.0))
    assert sup.wait_for_initial_dock(timeout=1.0)


def test_a_slice_the_clock_refuses_fails_the_bootstrap_loudly():
    """Critic A1/B3: the sim-mode scheduler refuses a DOWN carrying the
    cluster's deadline overrides (wall-clock stamps), and with it the slice.
    ``ClientCluster`` only logs a failing sink and stages θ anyway, so the
    supervisor raises instead of reporting a bootstrap with no slice."""
    dock = LoopbackDockLink()
    sup = MuleSupervisor(mule_id=MULE, rf=LoopbackRFLink(), dock=dock, rf_range_m=60.0,
                         mission_clock=MissionClock())
    dock.send_down(_down(overrides={DeviceID("d1"): 1.7e9}))
    with pytest.raises(MuleSupervisorError, match="not ingested.*wall-clock stamps"):
        sup.wait_for_initial_dock(timeout=1.0)
    assert dict(sup.scheduler.device_states) == {}
    assert sup._ferry_slice_error is None                 # raised once, not again
    # A DOWN without overrides is taken as before.
    dock.send_down(_down(round_=2))
    assert sup.wait_for_initial_dock(timeout=1.0)
    assert list(sup.scheduler.device_states) == [DeviceID("d1")]
    # The recorded mule folds the same override, exactly as it always did.
    legacy_dock = LoopbackDockLink()
    legacy = MuleSupervisor(mule_id=MULE, rf=LoopbackRFLink(), dock=legacy_dock,
                            rf_range_m=60.0)
    legacy_dock.send_down(_down(overrides={DeviceID("d1"): 1.7e9}))
    assert legacy.wait_for_initial_dock(timeout=1.0)
    assert legacy.scheduler.device_states[DeviceID("d1")].deadline_override_ts == 1.7e9


# --------------------------------------------------------------------------- #
# Oort (D2) at a re-plan
# --------------------------------------------------------------------------- #

def _state(did, *, served, clean, loss=1.0, n=10):
    return DeviceSchedulerState(device_id=DeviceID(did), last_served_round=served,
                                last_clean_round=clean, last_loss=loss,
                                last_num_examples=n)


def _contact(did, x):
    return ContactWaypoint(position=(float(x), 0.0, 0.0), devices=(DeviceID(did),),
                           bucket=Bucket.SCHEDULED_THIS_ROUND, deadline_ts=0.0)


def _reference_admit_and_order(contacts, states, env, budget_end, model):
    """Today's (afa9526) D2 admission: R inferred from the candidates."""
    policy = OortPolicy()
    return greedy_budget_walk(
        contacts,
        key=policy._rank_key(states, policy._current_round(contacts, states)),
        mule_pose=env.mule_pose, now=env.now, mission_deadline_ts=budget_end, model=model,
    )


def _random_case(rng):
    n = rng.integers(2, 7)
    states, contacts = {}, []
    for i in range(n):
        did = f"d{i}"
        served = int(rng.integers(0, 6))
        states[DeviceID(did)] = _state(did, served=served, clean=int(rng.integers(0, served + 1)),
                                       loss=float(rng.uniform(0.1, 2.0)),
                                       n=int(rng.integers(1, 100)))
        contacts.append(_contact(did, float(rng.uniform(1, 60))))
    return contacts, states


def test_the_plan_is_ranked_exactly_as_before():
    """Legacy identity: the plan of a mission infers R from the candidates
    even though the scheduler hands the mule's round (Phase 2), including
    when that round is ahead of the inference."""
    rng = np.random.default_rng(3)
    model = FeasibilityModel(cruise_speed_m_s=1.0, session_time_s=1.0)
    for _ in range(400):
        contacts, states = _random_case(rng)
        mission_round = int(rng.integers(1, 12))
        env = SelectorEnv(mule_pose=(0.0, 0.0, 0.0), now=0.0, mission_round=mission_round)
        budget_end = float(rng.choice([30.0, 80.0, 200.0]))
        got = OortPolicy().admit_and_order(contacts, states, env,
                                           mission_deadline_ts=budget_end,
                                           feasibility_model=model)
        assert got == _reference_admit_and_order(contacts, states, env, budget_end, model)


def test_a_replan_ranks_with_the_round_its_plan_used():
    """The remainder's members alone would infer a lower round (none of them
    had an outcome last mission), and here that flips the ranking: the
    re-plan reuses the plan's round, so it ranks as the plan did.

    ``big`` is worth 20 with a small staleness term (L = 2), ``stale`` 10
    with a larger one (L = 1). With w = 25, R = 7 puts ``stale`` first
    (25 log 7 (1 - 1/sqrt 2) > 10) and R = 3 puts ``big`` first.
    """
    states = {
        DeviceID("served"): _state("served", served=6, clean=6, loss=1.0, n=10),
        DeviceID("big"): _state("big", served=2, clean=2, loss=1.0, n=20),
        DeviceID("stale"): _state("stale", served=2, clean=1, loss=1.0, n=10),
    }
    plan = [_contact("served", 5), _contact("big", 10), _contact("stale", 15)]
    policy = OortPolicy(staleness_weight=25.0)
    env = SelectorEnv(mule_pose=(0.0, 0.0, 0.0), now=0.0, mission_round=7)
    assert [wp.devices[0] for wp in policy.admit_and_order(plan, states, env)] == \
        ["stale", "big", "served"]
    assert policy._planned_round == (7, 7)
    remainder = plan[1:]
    assert policy._current_round(remainder, states) == 3        # the inference
    by_inference = sorted(remainder, key=policy._rank_key(states, 3))
    assert [wp.devices[0] for wp in by_inference] == ["big", "stale"]
    replan = policy.admit_and_order(remainder, states, env)
    assert [wp.devices[0] for wp in replan] == ["stale", "big"]
    # A new mission plans afresh, with its own inference.
    env8 = SelectorEnv(mule_pose=(0.0, 0.0, 0.0), now=0.0, mission_round=8)
    assert [wp.devices[0] for wp in policy.admit_and_order(remainder, states, env8)] == \
        ["big", "stale"]
    assert policy._planned_round == (8, 3)


def test_without_a_mission_round_every_call_infers():
    states = {DeviceID("a"): _state("a", served=4, clean=4)}
    policy = OortPolicy()
    env = SelectorEnv(mule_pose=(0.0, 0.0, 0.0), now=0.0)
    policy.admit_and_order([_contact("a", 1)], states, env)
    assert policy._planned_round is None
    assert statistical_utility(states[DeviceID("a")]) == 10.0
