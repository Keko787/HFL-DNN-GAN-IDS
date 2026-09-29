"""Phase 0/1 audit fixes, pinned through real two-pass missions.

* **#0 — train-ahead.** A device collected in Pass 1 and then skipped by a
  budgeted Pass 2 had nothing prepared, so its next Pass-1 contact trained in
  session on the NEW θ and reported age 0: ages never spread for the common
  case. With a budgeted Pass 2 the mule now asks Pass-1 devices to train
  ahead on the basis they adopt, so that device's next update is one round old.
* **#2 — the cutoff reads the planned window.** The merge cutoff was computed
  at close time, after every CLEAN had already tightened Φ, so an on-time
  update was cut with a window it was never planned under.
* **#7 — reachability, not the outcome tag, picks the law's factor.** A device
  that answered but whose upload was lost is reachable, and relaxes at
  β_partial, not β_timeout.

Reuses the loopback harness of ``test_phase1_two_pass_ages``.
"""

from __future__ import annotations

import logging
import threading

from hermes.mission import ClientMission
from hermes.mission.aggregation_rules import AGG_CUTOFF, AggregationSpec
from hermes.scheduler.stages.s3_deadline import DeadlineLaw
from hermes.types import DeliveryOutcome, FLState, MissionOutcome

from hermes.processes.mule import (
    _pass_1_collected,
    _pass_1_merged_devices,
    _pass_1_outcomes_payload,
)

from tests.integration.test_phase1_two_pass_ages import (
    DEVICE_IDS,
    _TIGHT,
    _run_mission,
    _setup,
    _train_factory,
)

FALLBACK = "falling back to in-session training"

#: Two contacts 200 m apart: dev-00/01 and dev-02/03.
_TWO_GROUPS = [(0.0, 0.0, 0.0), (10.0, 5.0, 0.0), (200.0, 0.0, 0.0), (210.0, 15.0, 0.0)]


def _skip_far_half(sup):
    """Make Pass 2 skip dev-02 and dev-03, whatever the geometry.

    The budget walk itself is pinned elsewhere; here only its outcome matters:
    those two devices were collected in Pass 1 and are not delivered to.
    """
    skip_ids = {DEVICE_IDS[2], DEVICE_IDS[3]}

    def _budget(queue):
        fly = [wp for wp in queue if not set(wp.devices) & skip_ids]
        skip = [wp for wp in queue if set(wp.devices) & skip_ids]
        return fly, skip

    sup._budget_pass_2 = _budget


def _wait_train_ahead(devices):
    for cm in devices:
        t = cm._train_thread
        if t is not None:
            t.join(timeout=5.0)


def test_a_device_skipped_in_pass_2_ships_an_update_one_round_old(caplog):
    spec = AggregationSpec(rule=AGG_CUTOFF)
    sup, cluster, devices = _setup(
        _TWO_GROUPS, spec, mission_budget_s=10_000.0, pass_2_budget=True,
    )
    _skip_far_half(sup)
    assert sup.mission.train_ahead is True

    r1 = _run_mission(sup, cluster, devices)
    assert {l.device_id for l in r1.report.lines if l.outcome.is_on_time()} == set(DEVICE_IDS)
    skipped = {l.device_id for l in r1.delivery_report.lines
               if l.outcome is DeliveryOutcome.SKIPPED}
    assert skipped == {DEVICE_IDS[2], DEVICE_IDS[3]}
    _wait_train_ahead(devices)
    by_id = {cm.device_id: cm for cm in devices}
    # Skipped devices trained ahead on the Pass-1 basis (version 0); delivered
    # devices hold an update on the Pass-2 θ (version 1).
    assert all(by_id[d]._prepared_basis_version == 0 for d in skipped)
    assert all(by_id[d]._prepared_basis_version == 1
               for d in set(DEVICE_IDS) - skipped)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        r2 = _run_mission(sup, cluster, devices)
    assert not [r for r in caplog.records if FALLBACK in r.getMessage()]
    agg = r2.aggregate
    assert agg.base_version == 1
    ages = dict(zip(agg.contributing_devices, agg.device_ages))
    assert {ages[d] for d in skipped} == {1}
    assert {ages[d] for d in set(DEVICE_IDS) - skipped} == {0}
    # The age-1 updates carry less weight than the current ones (hinge s(1)=½).
    weights = dict(zip(agg.contributing_devices, agg.device_weights))
    assert max(weights[d] for d in skipped) < min(
        weights[d] for d in set(DEVICE_IDS) - skipped
    )


def test_without_a_pass_2_budget_nothing_trains_ahead():
    sup, cluster, devices = _setup(_TIGHT, AggregationSpec(rule=AGG_CUTOFF))
    assert sup.mission.train_ahead is False
    _run_mission(sup, cluster, devices)
    assert all(cm._train_thread is None for cm in devices)


def test_the_cutoff_uses_the_window_each_device_was_planned_under():
    # Φ starts at 60 s; T = 10 s gives a planned cap of 6. Every session is
    # CLEAN, which the multiplicative law folds to Φ = 48 (cap 4) before close.
    spec = AggregationSpec(rule=AGG_CUTOFF, period_s=10.0)
    sup, cluster, devices = _setup(
        _TIGHT, spec, deadline_law=DeadlineLaw(form="multiplicative"),
    )
    seen = []
    real_close = sup.mission.close_round

    def _spy(**kwargs):
        seen.append(dict(kwargs.get("age_caps") or {}))
        return real_close(**kwargs)

    sup.mission.close_round = _spy
    _run_mission(sup, cluster, devices)
    assert seen and set(seen[0].values()) == {6}
    assert {st.deadline_fulfilment_s for st in sup.scheduler.device_states.values()} == {48.0}


def test_a_device_that_answered_but_lost_its_upload_relaxes_at_beta_partial():
    law = DeadlineLaw(form="multiplicative")
    sup, cluster, devices = _setup(_TIGHT, AggregationSpec(rule=AGG_CUTOFF),
                                   deadline_law=law)
    # Replace dev-00 with a device whose Pass-1 uplink always drops: it answers
    # the solicit and receives θ, but its update never reaches the mule.
    lossy = ClientMission(
        device_id=DEVICE_IDS[0], rf=devices[0].rf,
        local_train=_train_factory(300), solicit_timeout_s=1.5,
        disc_push_timeout_s=1.5, contact_reliability=0.0,
    )
    lossy.set_state(FLState.FL_OPEN)
    devices[0] = lossy

    r = _run_mission(sup, cluster, devices)
    outcomes = {l.device_id: l.outcome for l in r.report.lines}
    assert outcomes[DEVICE_IDS[0]] is MissionOutcome.TIMEOUT
    phi = {d: st.deadline_fulfilment_s for d, st in sup.scheduler.device_states.items()}
    assert phi[DEVICE_IDS[0]] == 60.0 * law.beta_partial     # 75, not 90
    assert all(phi[d] == 60.0 * law.beta_on for d in DEVICE_IDS[1:])
    # Phase 2 state: the lossy device was reached (it answered) but its update
    # was not merged, so its Age-of-Update anchor stays unset.
    states = sup.scheduler.device_states
    assert (states[DEVICE_IDS[0]].reach_attempts, states[DEVICE_IDS[0]].reach_answered) == (1, 1)
    assert states[DEVICE_IDS[0]].last_merged_round is None
    assert {states[d].last_merged_round for d in DEVICE_IDS[1:]} == {r.mission_round}


def test_a_device_that_never_answered_relaxes_at_beta_timeout():
    """The other half of the split: no advert, so β_timeout (60 → 90)."""
    law = DeadlineLaw(form="multiplicative")
    sup, cluster, devices = _setup(_TIGHT, AggregationSpec(rule=AGG_CUTOFF),
                                   deadline_law=law)
    # The answering devices must outlast the mule's wait for dev-00's advert.
    answering = []
    for i, cm in enumerate(devices[1:], start=1):
        slow = ClientMission(
            device_id=DEVICE_IDS[i], rf=cm.rf, local_train=_train_factory(300 + i),
            solicit_timeout_s=6.0, disc_push_timeout_s=6.0,
        )
        slow.set_state(FLState.FL_OPEN)
        answering.append(slow)

    r = _run_mission(sup, cluster, answering)             # dev-00 stays silent
    outcomes = {l.device_id: l.outcome for l in r.report.lines}
    assert outcomes[DEVICE_IDS[0]] is MissionOutcome.TIMEOUT
    phi = {d: st.deadline_fulfilment_s for d, st in sup.scheduler.device_states.items()}
    assert phi[DEVICE_IDS[0]] == 60.0 * law.beta_timeout     # 90, not 75
    st = sup.scheduler.device_states[DEVICE_IDS[0]]
    assert (st.reach_attempts, st.reach_answered) == (1, 0)   # attempted, unanswered
    assert all(phi[d] == 60.0 * law.beta_on for d in DEVICE_IDS[1:])


def test_a_mission_whose_every_update_was_cut_off_keeps_its_sessions():
    """Audit #3: every Pass-2 skipped, so mission 2's updates are all age 1,
    past a_max = 0. The round is empty, but its on-time sessions stay in the
    trace instead of being scored as deadline misses."""
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=0)
    sup, cluster, devices = _setup(
        _TIGHT, spec, mission_budget_s=10_000.0, pass_2_budget=True,
    )
    sup._budget_pass_2 = lambda queue: ([], list(queue))
    _run_mission(sup, cluster, devices)
    _wait_train_ahead(devices)

    # Mission 2: every device ships its prepared age-1 update, all are cut, so
    # Pass 1 is empty and there is no dock and no Pass 2 (one contact each).
    workers = [threading.Thread(target=cm.serve_once, daemon=True) for cm in devices]
    for t in workers:
        t.start()
    r2 = sup.run_one_mission()
    for t in workers:
        t.join(timeout=5.0)

    assert r2.empty and r2.report is None
    assert r2.unmerged_report is not None
    lines = r2.unmerged_report.lines
    assert {l.device_id for l in lines} == set(DEVICE_IDS)
    assert all(l.outcome is MissionOutcome.CLEAN and l.age == 1 for l in lines)
    rows = _pass_1_outcomes_payload(r2)
    assert sorted(r["device"] for r in rows) == sorted(DEVICE_IDS)
    assert all(r["outcome"] == "clean" for r in rows)
    assert _pass_1_merged_devices(r2) == []
    # The sessions completed, so they count as collections (completion
    # metrics), while the merged fields say nothing reached the model.
    updates, clean = _pass_1_collected(r2)
    assert updates == 4 and sorted(clean) == sorted(DEVICE_IDS)
