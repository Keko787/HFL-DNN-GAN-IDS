"""FeRRy Phase 3 final check, group "mule": finding E2E1-01.

Narrow band, a declared payload and a mission budget. S3a clusters at
R_planar(narrow) = 232.2 m, so in the realism field one contact holds every
device, and S3b prices and admits a contact as a whole (design sections 3.1
and 4.4). Under any budget below that contact's predicted home time the plan
is empty and every mission flies nothing; at or above it all N devices are
admitted, provided the deadline clause does not bind first. Under a budget the
band trade is therefore a cliff between 0 and N devices, set by contact
granularity, not the reach-against-dwell trade of design section 1 D1. Phase 3
changes no behaviour for it:

* the cliff is pinned at the scheduler level on trial T2's layout,
  ``device_positions(8, 777, 100.0)``, at T2's deadline unit, where the one
  contact's predicted home is 99.115 s after takeoff: 0 devices are admitted
  at 60 s and at 99.0 s, all 8 at 99.5 s. **Phase 4 must update this test
  deliberately:** admitting a member subset of a contact that does not fit
  (or capping S3a's contacts by predicted dwell) changes the 60 s and 99.0 s
  answers, which is the point of that change;
* the deadline unit decides which clause our arms hit first. At the runner's
  default unit 1.0, Deadline(j) is 60 s after takeoff (Phi_0) and the
  contact's predicted finish is 94.9 s, so S3b (H1-H3) drops it as
  ``overdue`` whatever the budget, until missed missions widen the window
  past that finish; the budget-only walks (D1-D3, D5; Freeze Amendment 8)
  keep the budget cliff. Pinned on the first mission for S3b and for D1;
* the diagnostic added for it: an empty mission's simulated
  ``mission_completed`` lists the drop that emptied it
  (``pass_1_preflight_drops``, reason ``budget`` at T2's unit), JSON-ready.
  It explains an empty plan (``pass_1_contacts`` 0), not a ``mission_empty``
  whose contacts were flown but answered nothing. The field is
  None on the wall clock, whose ``mission_completed`` never carries it
  (its key set is pinned in tests/unit/test_p3_mule_service.py).
"""

from __future__ import annotations

import json

import pytest

from experiments.exp4.topology_builder import device_positions
from hermes.l1.mission_clock import MissionClock
from hermes.mule.ferry import FerrySpec
from hermes.processes import mule as mule_process
from hermes.processes.config import MuleConfig
from hermes.scheduler import FLScheduler
from hermes.scheduler.policies.max_aoi import MaxAoIPolicy
from hermes.transport import TCPDockLinkServer
from hermes.types import DeviceID, MuleID
from hermes.types.registry import DeviceRecord, MissionSlice, SpectrumSig

from tests.integration import _ferry_harness as H

DOCK = (0.0, 0.0, 0.0)
#: Trial T2 of the final check (H1, N = 8, seed 777, the 100 m realism field).
T2_LAYOUT = device_positions(8, 777, 100.0)
IDS = tuple(f"dev-{i}" for i in range(len(T2_LAYOUT)))
#: T2's deadline unit ('t_nom', 24.999): Deadline(j) = t0 + 1500 s, so the
#: deadline clause never binds first and the drop is the budget's.
TIME_SCALE = 25.0
#: The runner's default deadline unit (--deadline-time-scale 1.0): Deadline(j)
#: = t0 + 60 s, before the field-wide contact's predicted finish.
DEFAULT_TIME_SCALE = 1.0


def _narrow_spec() -> FerrySpec:
    """T2's ferry physics: narrow band, 1 MB each way, the seconds backhaul
    (clean). The planner prices the upload on the carrier means, which do not
    depend on the backhaul period."""
    return FerrySpec.from_config(rf_range_m=60.0, seed=777, contact_band="narrow",
                                 backhaul_model="seconds", backhaul_period=750.0,
                                 backhaul_regime="clean", payload_bytes=1_000_000)


def _plan(budget_s: float, *, time_scale: float = TIME_SCALE, selector=None):
    """H1's Pass-1 plan at takeoff from the dock: S1, S3, S3a at
    R_planar(narrow), then S3b under the budget (no selector). With a
    whole-scheduler ``selector`` (a D arm) its own walk admits instead."""
    spec = _narrow_spec()
    clock = MissionClock()
    sch = FLScheduler(now_fn=clock, mission_budget_s=budget_s,
                      feasibility_model=spec.feasibility_model(rf_range_m=60.0, theta_bytes=52),
                      deadline_time_scale=time_scale, refuse_deadline_overrides=True,
                      target_selector=selector)
    records = [DeviceRecord(device_id=DeviceID(d), last_known_position=(x, y, 0.0),
                            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)))
               for d, (x, y) in zip(IDS, T2_LAYOUT)]
    sch.ingest_slice(MissionSlice(mule_id=MuleID("m"), device_ids=tuple(DeviceID(d) for d in IDS),
                                  issued_round=0, issued_at=clock()),
                     registry_records=records)
    sch.start_mission()
    queue = sch.build_contact_queue(rf_range_m=spec.link.range_planar_m("narrow"),
                                    mule_pose=DOCK)
    return queue, sch


def test_under_a_budget_narrow_admits_no_device_or_all_of_them():
    (contact,), sch = _plan(1e9)                        # one field-wide contact
    assert sorted(contact.devices) == sorted(IDS)
    leg = sch.feasibility_model.leg(DOCK, contact)
    # transit 4.04 s, the summed dwell of 8 members 90.85 s, return 4.04 s,
    # upload 0.18 s: the cliff.
    assert leg.transit_s + leg.dwell_s + leg.return_s + leg.upload_s == pytest.approx(
        99.1154, abs=1e-3)
    for budget in (60.0, 99.0):
        queue, sch = _plan(budget)
        feas = sch.last_feasibility
        assert queue == [] and feas.kept == []
        assert feas.dropped_budget == [contact]
        assert feas.dropped_overdue == [] and feas.dropped_energy == []
    queue, _ = _plan(99.5)
    assert queue == [contact]


def test_at_the_default_deadline_unit_s3b_drops_the_contact_as_overdue_at_any_budget():
    """At the runner's default unit the deadline clause binds before the
    budget: Deadline(j) = t0 + 60 s, and the contact finishes (transit plus
    summed dwell, the ``collection`` bound) 94.9 s after takeoff. S3b reads
    the deadline first, so our arms fly nothing above the knee too, and the
    drop reads ``overdue``, not ``budget``. D1's walk is budget-only and keeps
    the cliff. (Each missed mission widens the window by 10 s, so a later
    mission can serve the contact; that is the deadline law's, not pinned
    here.)"""
    (contact,), sch = _plan(1e9)
    leg = sch.feasibility_model.leg(DOCK, contact)
    assert leg.transit_s + leg.dwell_s == pytest.approx(94.893, abs=1e-3)
    t0 = MissionClock()()
    for budget in (60.0, 99.0, 99.5, 200.0, 1000.0):
        queue, sch = _plan(budget, time_scale=DEFAULT_TIME_SCALE)
        feas = sch.last_feasibility
        assert queue == [] and feas.kept == []
        assert [(sorted(c.devices), c.deadline_ts - t0) for c in feas.dropped_overdue] == [
            (sorted(contact.devices), 60.0)]
        assert feas.dropped_budget == [] and feas.dropped_energy == []
    for budget, admitted in ((99.0, 0), (99.5, len(IDS))):
        queue, sch = _plan(budget, time_scale=DEFAULT_TIME_SCALE, selector=MaxAoIPolicy())
        assert sum(len(c.devices) for c in queue) == admitted
        assert sch.last_feasibility is None      # a D arm's walk reports no drops


def _empty_narrow_mission():
    """T2's layout flown on the clock under a 60 s budget: S3b drops the one
    field-wide contact before takeoff, so the mission flies nothing."""
    layout = tuple((d, (x, y, 0.0)) for d, (x, y) in zip(IDS, T2_LAYOUT))
    w = H.World(layout=layout, flaky={})
    mid = w.mule_ids[0]
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=True, ferry=_narrow_spec(), mission_budget_s=60.0,
                           deadline_time_scale=TIME_SCALE)
        w.bootstrap()
        return sup.run_one_mission()


def test_an_empty_mission_records_the_budget_drop_that_emptied_it():
    r = _empty_narrow_mission()
    assert r.empty and r.pass_1_queue == [] and r.pass_1_flown == []
    assert r.sim_ledger["turnaround"] == 30.0 == sum(r.sim_ledger.values())
    (contact,), _ = _plan(1e9)
    assert r.pass_1_preflight_drops == [{
        "position": list(contact.position),
        "devices": [str(d) for d in contact.devices],
        "deadline_ts": min(r.pass_1_device_deadlines.values()),
        "reason": "budget",
    }]


class _Events:
    """What the JSONL emitter does with each record: serialise it whole,
    raising on anything JSON cannot hold."""

    def __init__(self):
        self.lines = []

    def emit(self, event, **fields):
        self.lines.append((event, json.loads(json.dumps(fields))))

    def close(self):
        return

    def named(self, event):
        return [f for e, f in self.lines if e == event]


def test_the_sim_mission_completed_of_an_empty_mission_lists_the_drop():
    """The mule process on the simulated clock emits ``mission_empty`` and
    then ``mission_completed``; the latter carries the drop that emptied the
    mission, as the other sim fields are carried (``SIM_MISSION_FIELDS``)."""
    r = _empty_narrow_mission()
    server = TCPDockLinkServer(host="127.0.0.1", port=0)
    server.start()
    try:
        events = _Events()
        cfg = MuleConfig(mule_id="m-p3", dock_port=server.port, rf_range_m=60.0,
                         mission_clock="sim", trial_seed=11, n_missions=1,
                         contact_band="narrow", backhaul_model="seconds", t_nom_s=200.0,
                         rf_link_token="tok-1")
        svc = mule_process.MuleService(cfg, events=events)
        try:
            svc.supervisor.wait_for_initial_dock = lambda timeout=None: True
            svc.supervisor.run_one_mission = lambda: r
            svc.run()
        finally:
            svc.shutdown()
    finally:
        server.close()
    assert svc.exit_code == 0
    assert events.named("mission_empty") == [{"mission_round": r.mission_round}]
    (done,) = events.named("mission_completed")
    assert done["pass_1_contacts"] == 0 and done["pass_1_preflight_drops"] == r.pass_1_preflight_drops
    assert [(d["reason"], len(d["devices"])) for d in done["pass_1_preflight_drops"]] == [
        ("budget", len(IDS))]


def test_the_record_is_sim_only():
    """A wall-clock mission never records it, pre-flight drops or not."""
    w = H.World()
    mid = w.mule_ids[0]
    with H.Patched(w.clock):
        sup = w.supervisor(mid, sim=False, mission_budget_s=60.0)
        w.bootstrap()
        r = sup.run_one_mission()
    assert sup.scheduler.last_feasibility.n_dropped > 0
    assert r.pass_1_preflight_drops is None
