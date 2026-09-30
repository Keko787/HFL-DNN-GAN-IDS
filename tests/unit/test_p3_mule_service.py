"""FeRRy Phase 3, unit U7 — the mule process on the simulated mission clock.

* On the wall clock the process builds the recorded supervisor and emits the
  recorded ``mule_ready``, ``mission_started`` and ``mission_completed``
  (afa9526's key sets; the values come from unchanged code).
* On the simulated clock it builds one ``MissionClock`` and the ``FerrySpec``
  its config describes, hands both to the supervisor, and its events gain the
  simulated fields of design section 2.5, JSON-ready even when a result holds
  numpy scalars. The ground-truth availability is never emitted (critic B16).
* A bootstrap DOWN refused on the clock ends the process with
  ``EXIT_BOOTSTRAP_FAILED``; on the wall clock the error propagates as before.
* The RF link token reaches the RF server (Amendment 10); a config the
  guards refuse never binds a port.
* The deadline time unit shows in ``mule_ready`` on either clock, and on the
  wall clock only when it is not the recorded one (spec Q1, audit #15).
* Critic B4 under the recorded ``mission`` backhaul model: the planner's RF
  prior is fed from the missions already uploaded, one schedule entry per
  mission round, and every sim ``mission_started`` records the prior its
  Pass-1 plan is handed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hermes.l1.mission_clock import SIM_EPOCH_S, MissionClock
from hermes.mule import MuleSupervisorError
from hermes.mule.ferry import FerrySpec
from hermes.mule.mule_main import MissionRunResult
from hermes.processes import mule as mule_process
from hermes.processes.config import MuleConfig
from hermes.transport import TCPDockLinkServer

AFA9526_READY = {
    "rf_host", "rf_port", "dock_host", "dock_port", "expected_devices", "rf_range_m",
    "session_ttl_s", "n_missions", "aggregation", "aggregation_params", "deadline_law",
    "deadline_params", "miss_priority", "pass_2_budget", "mission_budget_s", "down_wait_s",
    "dock_on_empty",
}
AFA9526_COMPLETED = {
    "mission_round", "queue_size", "pass_1_contacts", "pass_2_contacts", "duration_s",
    "pass_1_updates", "pass_1_scheduled", "pass_1_clean_devices", "delivered", "undelivered",
    "pass_1_plan", "pass_1_outcomes", "pass_1_merged_devices", "pass_1_merged_updates",
    "pass_1_merge", "pass_2_skipped", "deadline_state",
}


class _Events:
    def __init__(self):
        self.lines = []

    def emit(self, event, **fields):
        # What the JSONL emitter does: serialise the whole record, raising on
        # anything JSON cannot hold (numpy scalars included).
        json.dumps(fields)
        self.lines.append((event, fields))

    def close(self):
        return

    def named(self, event):
        return [f for e, f in self.lines if e == event]


@pytest.fixture
def service():
    server = TCPDockLinkServer(host="127.0.0.1", port=0)
    server.start()
    made = []

    def _make(**cfg_kwargs):
        events = _Events()
        cfg_kwargs.setdefault("rf_range_m", 60.0)
        cfg = MuleConfig(mule_id="m-p3", dock_port=server.port, **cfg_kwargs)
        svc = mule_process.MuleService(cfg, events=events)
        svc.supervisor.wait_for_initial_dock = lambda timeout=None: True
        made.append(svc)
        return svc, events

    yield _make
    for svc in made:
        svc.shutdown()
    server.close()


def _sim_kwargs(**kw):
    base = dict(mission_clock="sim", trial_seed=11, n_missions=2, contact_band="wide",
                backhaul_model="seconds", t_nom_s=200.0, rf_link_token="tok-1")
    base.update(kw)
    return base


def _sim_result(clock) -> MissionRunResult:
    """A sim-clock result with numpy scalars in its nested records."""
    start = clock()
    return MissionRunResult(
        mission_round=1, empty=True,
        sim_start_s=start, sim_end_s=start + np.float64(42.5),
        sim_ledger={"transit": np.float64(10.0), "dwell": 0.5, "listen": 1.0, "return": 1.0,
                    "upload": 0.0, "turnaround": 30.0, "dock_wait": 0.0},
        sim_pass_2_start_s=None,
        pass_1_flown=[{"devices": ["d0"], "arrival_s": np.float32(1.0), "snr_db": {"d0": 3.5},
                       "l1_choice": np.int64(1), "targets": ("d0",)}],
        pass_2_flown=[], replans=[], aborts=[], inserts=[], offers_refused=[],
        budget_overrun_s=None, pass_2_budget_overrun_s=None, energy_j=np.float64(1500.0),
        band="wide",
        backhaul={"carrier": 2, "snr_db": 9.0, "p_loss": 0.05, "t_upload_s": start + 12.0},
    )


# --------------------------------------------------------------------------- #
# The wall clock: the recorded process
# --------------------------------------------------------------------------- #

def test_the_wall_clock_process_is_the_recorded_one(service):
    svc, events = service(n_missions=1)
    assert svc.supervisor.mission_clock is None and svc.supervisor.ferry is None
    assert svc.rf._link_token is None
    (ready,) = events.named("mule_ready")
    assert set(ready) == AFA9526_READY
    svc.supervisor.run_one_mission = lambda: MissionRunResult(mission_round=1)
    svc.run()
    assert svc.exit_code == 0
    assert events.named("mission_started") == [{"mission_index": 0}]
    (done,) = events.named("mission_completed")
    assert set(done) == AFA9526_COMPLETED


def test_the_wall_clock_bootstrap_error_propagates_as_before(service):
    svc, _events = service(n_missions=1)

    def _refuse(timeout=None):
        raise MuleSupervisorError("boom")

    svc.supervisor.wait_for_initial_dock = _refuse
    with pytest.raises(MuleSupervisorError):
        svc.run()


def test_the_deadline_time_unit_reaches_the_scheduler_on_either_clock(service):
    svc, _ = service(deadline_time_scale=2.0, initial_window_s=90.0)
    sched = svc.supervisor.scheduler
    assert sched.deadline_time_scale == 2.0 and sched.effective_initial_window_s == 180.0
    svc, _ = service(**_sim_kwargs(deadline_time_scale=20.0))
    assert svc.supervisor.scheduler.deadline_time_scale == 20.0


TIME_UNIT_KEYS = {"deadline_time_scale", "initial_window_s", "effective_initial_window_s"}


@pytest.mark.parametrize("kw,shown", [
    (dict(), None),                                        # the recorded unit: no new key
    (dict(initial_window_s=60.0), None),                   # ... stated explicitly
    (dict(deadline_time_scale=2.0), (2.0, 60.0, 120.0)),
    (dict(initial_window_s=90.0), (1.0, 90.0, 90.0)),
    (dict(deadline_time_scale=2.0, initial_window_s=90.0), (2.0, 90.0, 180.0)),
    (dict(deadline_time_scale=2.0, initial_window_s=30.0), (2.0, 30.0, 60.0)),
])
def test_a_wall_clock_mule_ready_shows_a_time_unit_other_than_the_recorded_one(
    service, kw, shown,
):
    """The time unit is valid on either clock; ``deadline_params`` carries the
    scale but never Φ₀, so a wall-clock mule states the three fields itself,
    read back from its scheduler, whenever they are not the recorded ones."""
    svc, events = service(**kw)
    (ready,) = events.named("mule_ready")
    if shown is None:
        assert set(ready) == AFA9526_READY
        return
    assert set(ready) == AFA9526_READY | TIME_UNIT_KEYS
    assert (ready["deadline_time_scale"], ready["initial_window_s"],
            ready["effective_initial_window_s"]) == shown
    assert ready["effective_initial_window_s"] == svc.supervisor.scheduler.effective_initial_window_s


# --------------------------------------------------------------------------- #
# The simulated clock
# --------------------------------------------------------------------------- #

def test_the_sim_process_hands_the_supervisor_a_clock_and_the_configs_spec(service):
    svc, _ = service(**_sim_kwargs(contact_reliability_source="channel",
                                   device_availability={"d0": 0.5}))
    sup = svc.supervisor
    assert isinstance(sup.mission_clock, MissionClock) and sup.mission_clock() == SIM_EPOCH_S
    expected = FerrySpec.from_config(**svc.cfg.ferry_spec_kwargs())
    assert sup.ferry.describe() == expected.describe()
    assert dict(sup.ferry.availability) == {"d0": 0.5}
    assert sup.scheduler.refuses_deadline_overrides
    assert svc.rf._link_token == "tok-1"


def test_sim_mule_ready_carries_the_clock_and_every_ferry_setting(service):
    svc, events = service(**_sim_kwargs(contact_reliability_source="channel",
                                        device_availability={"d0": 0.25}, input_dim=21,
                                        payload_bytes=1_000_000))
    (ready,) = events.named("mule_ready")
    assert AFA9526_READY < set(ready)
    assert ready["mission_clock"] == "sim" and ready["clock_epoch_s"] == SIM_EPOCH_S
    spec = svc.supervisor.ferry
    for key, value in spec.describe().items():
        assert ready[key] == json.loads(json.dumps(value)), key
    assert ready["band_classes"]["classes"][0]["range_planar_m"] == 60.0
    assert ready["energy_params"]["status"] == "simulated"
    assert (ready["t_nom_s"], ready["trial_seed"], ready["input_dim"]) == (200.0, 11, 21)
    assert ready["deadline_time_scale"] == 1.0 and ready["effective_initial_window_s"] == 60.0
    assert ready["payload_bytes"] == 1_000_000
    assert ready["payload"] == {"payload_bytes": 1_000_000, "mode": "declared"}
    assert ready["device_availability_n"] == 1
    # The seconds model feeds the RF prior from its own channel (critic B4).
    assert ready["rf_prior_source"] == mule_process.RF_PRIOR_SECONDS_BACKHAUL
    # The ground truth itself is never emitted (critic B16).
    assert "0.25" not in json.dumps(ready) and "device_availability" not in ready


def test_sim_mission_events_carry_the_simulated_record(service):
    svc, events = service(**_sim_kwargs(n_missions=1))
    clock = svc.supervisor.mission_clock
    clock.advance(5.0, "transit")
    svc.supervisor.run_one_mission = lambda: _sim_result(clock)
    svc.run()
    assert svc.exit_code == 0
    assert events.named("mission_started") == [
        {"mission_index": 0, "sim_start_s": SIM_EPOCH_S + 5.0, "rf_prior_snr_db": 20.0}]
    (done,) = events.named("mission_completed")
    assert set(done) == AFA9526_COMPLETED | set(mule_process.SIM_MISSION_FIELDS) | {"energy_status"}
    assert done["sim_end_s"] == SIM_EPOCH_S + 47.5 and done["energy_status"] == "simulated"
    assert done["sim_ledger"]["transit"] == 10.0 and done["band"] == "wide"
    assert done["pass_1_flown"][0]["l1_choice"] == 1 and done["pass_1_flown"][0]["targets"] == ["d0"]
    assert type(done["energy_j"]) is float


def test_a_refused_bootstrap_is_fatal_on_the_clock(service):
    svc, events = service(**_sim_kwargs())

    def _refuse(timeout=None):
        raise MuleSupervisorError("the DOWN's slice was not ingested on the mission clock")

    svc.supervisor.wait_for_initial_dock = _refuse
    svc.supervisor.run_one_mission = lambda: pytest.fail("flew without a slice")
    svc.run()
    assert svc.exit_code == mule_process.EXIT_BOOTSTRAP_FAILED
    (failed,) = events.named("dock_bootstrap_failed")
    assert "not ingested" in failed["reason"]


@pytest.mark.parametrize("kw,match", [
    (dict(contact_band="wide"), "mission_clock='sim'"),                      # wall + band
    (dict(backhaul_model="seconds"), "mission_clock='sim'"),                 # critic B16
    (dict(mission_clock="sim", trial_seed=1, contact_reliability_source="channel"),
     "contact_band"),                                                        # critic B16
    (dict(mission_clock="sim", trial_seed=1, contact_band="ultrawide"), "ultrawide"),
    (dict(rf_prior_schedule_db=[9.0]), "mission_clock='sim'"),               # wall + prior
    (dict(mission_clock="sim", trial_seed=1, backhaul_model="seconds",
          backhaul_period_s=800.0, rf_prior_schedule_db=[9.0]), "critic B4"),
])
def test_a_refused_config_never_starts_a_mule(kw, match):
    with pytest.raises(ValueError, match=match):
        mule_process.MuleService(MuleConfig(mule_id="m", rf_range_m=60.0, **kw))


def test_the_token_admits_only_this_trials_devices(service):
    """The RF server started with the trial's token refuses another one."""
    from hermes.transport.tcp_rf_link import TCPRFLinkClient
    from hermes.types import DeviceID

    svc, _ = service(**_sim_kwargs())
    good = TCPRFLinkClient(device_id=DeviceID("d-good"), host="127.0.0.1",
                           port=svc.actual_rf_port, link_token="tok-1")
    bad = TCPRFLinkClient(device_id=DeviceID("d-bad"), host="127.0.0.1",
                          port=svc.actual_rf_port, link_token="tok-2")
    try:
        assert svc.rf.wait_for_devices([DeviceID("d-good")], timeout=5.0)
        assert not svc.rf.wait_for_devices([DeviceID("d-bad")], timeout=0.5)
    finally:
        good.close()
        bad.close()


# --------------------------------------------------------------------------- #
# Critic B4: the RF prior under the recorded ``mission`` backhaul model
# --------------------------------------------------------------------------- #

SCHEDULE = [11.0, 7.5, 13.25, 9.0]


@pytest.mark.parametrize("docks_empty", [False, True])
def test_the_mission_models_rf_prior_comes_from_past_uploads_only(service, docks_empty):
    """Round r's entry becomes the prior only once round r has uploaded, so
    each Pass-1 plan sees only uploads already made; a mission that did not
    dock observed nothing and leaves the prior alone (the seconds model's
    rule, hermes/l1/rf_prior.py)."""
    svc, events = service(**_sim_kwargs(backhaul_model="mission", n_missions=4,
                                        rf_prior_schedule_db=list(SCHEDULE)))
    sup = svc.supervisor
    planned = []

    def _mission():
        r = len(planned) + 1
        planned.append(sup.rf_prior_snr_db)       # what this mission's Pass-1 plan is handed
        empty = r == 2
        return MissionRunResult(mission_round=r, empty=empty,
                                docked_empty=empty and docks_empty)

    sup.run_one_mission = _mission
    svc.run()
    assert svc.exit_code == 0
    assert planned == [20.0, 11.0, 7.5 if docks_empty else 11.0, 13.25]
    assert sup.rf_prior_snr_db == 9.0
    assert [e["rf_prior_snr_db"] for e in events.named("mission_started")] == planned
    (ready,) = events.named("mule_ready")
    assert ready["rf_prior_source"] == mule_process.RF_PRIOR_MISSION_SCHEDULE
    # Only the source is stated up front, not the later missions' SNRs.
    assert "13.25" not in json.dumps(ready)


def test_a_round_past_the_schedule_reads_its_last_entry(service):
    svc, _events = service(**_sim_kwargs(backhaul_model="mission", n_missions=3,
                                         rf_prior_schedule_db=[4.0, 6.0]))
    rounds = iter((1, 2, 3))
    svc.supervisor.run_one_mission = lambda: MissionRunResult(mission_round=next(rounds))
    svc.run()
    assert svc.supervisor.rf_prior_snr_db == 6.0


def test_without_a_schedule_or_a_seconds_backhaul_the_prior_stays_put(service):
    svc, events = service(**_sim_kwargs(backhaul_model="mission", n_missions=2))
    rounds = iter((1, 2))
    svc.supervisor.run_one_mission = lambda: MissionRunResult(mission_round=next(rounds))
    svc.run()
    assert [e["rf_prior_snr_db"] for e in events.named("mission_started")] == [20.0, 20.0]
    (ready,) = events.named("mule_ready")
    assert ready["rf_prior_source"] == mule_process.RF_PRIOR_CONSTANT


def test_the_wall_clock_never_feeds_the_prior(service):
    """A recorded mule keeps its configured prior (the driver's) for every mission."""
    svc, events = service(n_missions=2, rf_prior_snr_db=12.5)
    rounds = iter((1, 2))
    svc.supervisor.run_one_mission = lambda: MissionRunResult(mission_round=next(rounds))
    svc.run()
    assert svc.supervisor.rf_prior_snr_db == 12.5
    assert events.named("mission_started") == [{"mission_index": 0}, {"mission_index": 1}]
