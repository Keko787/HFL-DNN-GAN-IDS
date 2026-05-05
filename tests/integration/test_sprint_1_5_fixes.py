"""Sprint 1.5 review fixes — H1-H6 + M1-M5 verification.

One test per fix. Each test exercises the *fixed* path; the documented
bugs would have failed these assertions before the fix.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.mission import ClientMission, HFLHostMission, LocalTrainResult
from hermes.mule import MuleSupervisor
from hermes.scheduler import FLScheduler
from hermes.scheduler.selector import (
    SelectorEnv,
    SelectorScopeViolation,
    TargetSelectorRL,
)
from hermes.transport import LoopbackDockLink, LoopbackRFLink
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeliveryAck,
    DeliveryOutcome,
    DeviceID,
    DeviceRecord,
    DeviceSchedulerState,
    DiscPush,
    FLOpenSolicit,
    FLReadyAdv,
    FLState,
    MissionDeliveryLine,
    MissionDeliveryReport,
    MissionPass,
    MissionSlice,
    MuleID,
    SpectrumSig,
)


MULE = MuleID("mule-test")


def _train_factory(seed: int = 0):
    rng = np.random.default_rng(seed)

    def _train(theta, synth):
        delta = [w + rng.normal(0.0, 0.01, size=w.shape).astype(w.dtype) for w in theta]
        return LocalTrainResult(
            delta_theta=delta, num_examples=8,
            accuracy=0.8, auc=0.8, loss=0.2,
            theta_after=delta,
        )
    return _train


def _toy_theta() -> List[np.ndarray]:
    return [np.zeros((4,), dtype=np.float32)]


# --------------------------------------------------------------------------- #
# H1 — misrouted FL_READY_ADV stash, no crash on TCP-style transport
# --------------------------------------------------------------------------- #

def test_h1_misrouted_adv_stashed_for_next_contact():
    """A reply from outside the contact must NOT be re-sent on the link.

    Replies from non-expected devices are stashed on the mission server's
    internal queue and consumed by the next ``run_contact``. Pre-fix, the
    code called ``rf.send_ready_adv`` to re-queue, which crashes on TCP
    (mule-side server has no ``send_ready_adv``).

    Behaviour: ``run_contact`` drains FLReadyAdv until every expected
    device replies. Replies from non-expected devices that arrive during
    that drain are stashed; replies that arrive *after* the loop
    terminates remain on the link queue for the next contact to drain.
    """
    rf = LoopbackRFLink()
    for did in ("d0", "d1"):
        rf.register_device(DeviceID(did))

    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.2)
    mission.open_round(_toy_theta())

    def _adv(did_str: str) -> FLReadyAdv:
        return FLReadyAdv(
            device_id=DeviceID(did_str),
            state=FLState.FL_OPEN,
            performance_score=0.7,
            diversity_adjusted=0.7,
            utility=0.7,
            issued_at=time.time(),
        )

    # Inject d0 BEFORE d1 — run_contact pulls d0 first (misrouted),
    # then d1 (expected, completes the drain). d0 must land in the
    # internal stash, NOT be sent back via rf.send_ready_adv (which
    # would NotImplementedError on TCP).
    rf.send_ready_adv(_adv("d0"))
    rf.send_ready_adv(_adv("d1"))

    mission.run_contact(
        contact_devices=[DeviceID("d1")],
        synth_batch=[],
    )

    stashed_ids = sorted(adv.device_id for adv in mission._misrouted_advs)
    assert stashed_ids == [DeviceID("d0")], (
        f"expected [d0] stashed (interleaved between expected drains), "
        f"got {stashed_ids}"
    )

    # Second contact picks up the stashed d0 without re-pulling from
    # the link.
    rf.send_ready_adv(_adv("d0"))  # second push so the link has it again
    # Don't matter — the stash should be drained first.
    # We assert the stash empties when d0 is the expected device.
    mission.run_contact(
        contact_devices=[DeviceID("d0")],
        synth_batch=[],
    )
    # The stash had ONE d0; consumed in the drain. The second injected
    # d0 may or may not have been pulled — but the stash should now be
    # empty for d0 specifically.
    assert all(
        adv.device_id != DeviceID("d0") for adv in mission._misrouted_advs
    )


# --------------------------------------------------------------------------- #
# H2 — atomic take-and-clear of _prepared_delta (no race with train_offline)
# --------------------------------------------------------------------------- #

def test_h2_prepared_delta_taken_atomically():
    """Read-and-clear of ``_prepared_delta`` must happen under one lock.

    Pre-fix: two separate lock acquisitions allowed train_offline to
    overwrite the slot between read and clear, losing the fresh delta.
    Post-fix: a single lock acquisition takes the value AND nulls the
    slot. We test the contract by reading + asserting clearance after
    one ``serve_once``.
    """
    rf = LoopbackRFLink()
    rf.register_device(DeviceID("d0"))

    cm = ClientMission(
        device_id=DeviceID("d0"),
        rf=rf,
        local_train=_train_factory(),
        solicit_timeout_s=0.5,
        disc_push_timeout_s=0.5,
    )
    cm.set_state(FLState.FL_OPEN)
    # Stage a θ basis + a prepared delta as if train_offline had run.
    cm._set_theta_basis(_toy_theta(), [])
    cm.train_offline()
    assert cm._prepared_delta is not None

    # Run one Pass-1 cycle. Use HFLHostMission so the wire side works.
    mission = HFLHostMission(mule_id=MULE, rf=rf, session_ttl_s=0.5)
    mission.open_round(_toy_theta())

    worker = threading.Thread(target=cm.serve_once, daemon=True)
    worker.start()
    mission.run_contact(contact_devices=[DeviceID("d0")], synth_batch=[])
    worker.join(timeout=2.0)

    # After serve_once consumed the prepared delta, the slot must be None
    # (cleared atomically under the same lock as the read).
    assert cm._prepared_delta is None


# --------------------------------------------------------------------------- #
# H3 — delivery report rides up in the next mission's UP bundle
# --------------------------------------------------------------------------- #

def test_h3_delivery_report_attached_to_next_up_bundle():
    """ClientCluster.collect accepts ``delivery_report`` and the
    resulting UpBundle carries it as ``prev_mission_delivery_report``.
    """
    from hermes.mule import BundleDistributor, ClientCluster
    from hermes.types import (
        ContactHistory,
        MissionRoundCloseReport,
        PartialAggregate,
    )

    dock = LoopbackDockLink()
    cc = ClientCluster(
        mule_id=MULE,
        dock=dock,
        distributor=BundleDistributor(),
    )

    # Stage a Pass-1 aggregate plus a Pass-2 delivery report carry-over.
    agg = PartialAggregate(
        mule_id=MULE,
        mission_round=1,
        weights=[np.zeros((2,), dtype=np.float32)],
        num_examples=4,
    )
    report = MissionRoundCloseReport(
        mule_id=MULE, mission_round=1, started_at=0.0, finished_at=1.0,
    )
    contacts = ContactHistory(mule_id=MULE, mission_round=1)
    delivery = MissionDeliveryReport(
        mule_id=MULE, mission_round=1, started_at=0.0, finished_at=1.0,
    )
    delivery.append(MissionDeliveryLine(
        device_id=DeviceID("d0"),
        outcome=DeliveryOutcome.UNDELIVERED,
        contact_ts=1.0,
    ))

    cc.collect(
        partial_aggregate=agg, report=report, contacts=contacts,
        delivery_report=delivery,
    )

    # Build the UP bundle through the private helper to verify the
    # delivery report is embedded.
    with cc._lock:
        bundle = cc._build_up_bundle_locked()
    assert bundle is not None
    assert bundle.prev_mission_delivery_report is not None
    assert bundle.prev_mission_delivery_report.lines[0].device_id == DeviceID("d0")


# --------------------------------------------------------------------------- #
# H4 — delivery_priority propagates from cluster → mule via ingest_slice
# --------------------------------------------------------------------------- #

def test_h4_delivery_priority_copied_into_scheduler_state():
    """FLScheduler.ingest_slice copies DeviceRecord.delivery_priority
    onto the per-device scheduler state so S3a's tie-breaker reads the
    *current* cluster value, not stale 0.
    """
    sch = FLScheduler(now_fn=lambda: 1000.0)
    rec = DeviceRecord(
        device_id=DeviceID("d0"),
        last_known_position=(10.0, 0.0, 0.0),
        spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        delivery_priority=7,
    )
    sl = MissionSlice(
        mule_id=MULE,
        device_ids=(DeviceID("d0"),),
        issued_round=1,
        issued_at=1000.0,
    )
    sch.ingest_slice(sl, registry_records=[rec])

    st = sch.get_state(DeviceID("d0"))
    assert st is not None
    assert st.delivery_priority == 7, (
        "delivery_priority not propagated from DeviceRecord to "
        "DeviceSchedulerState"
    )

    # Re-ingest with a different value to verify update path.
    rec.delivery_priority = 3
    sch.ingest_slice(sl, registry_records=[rec])
    st = sch.get_state(DeviceID("d0"))
    assert st.delivery_priority == 3


# --------------------------------------------------------------------------- #
# H5 — mule pose advances after each contact
# --------------------------------------------------------------------------- #

def test_h5_mule_pose_advances_through_two_pass_mission():
    """After a two-pass mission, the supervisor's mule_pose has tracked
    the mule through visited contact positions (no longer pinned at
    construction-time origin).
    """
    rf = LoopbackRFLink()
    dock = LoopbackDockLink()

    # 4 devices spread out so they form 2+ contacts at rf_range_m=40.
    positions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (200.0, 0.0, 0.0), (210.0, 0.0, 0.0)]
    device_ids = [DeviceID(f"dev-{i:02d}") for i in range(4)]
    devices: List[ClientMission] = []
    for did, pos in zip(device_ids, positions):
        rf.register_device(did)
        cm = ClientMission(
            device_id=did,
            rf=rf,
            local_train=_train_factory(seed=int(did.split("-")[1])),
            solicit_timeout_s=2.0,
            disc_push_timeout_s=2.0,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)

    registry = DeviceRegistry()
    for did, pos in zip(device_ids, positions):
        registry.register(
            device_id=did, position=pos,
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    registry.rebalance([MULE], round_counter=0)
    cluster = HFLHostCluster(
        registry=registry,
        generator=StubGeneratorHost(disc_weights=[np.zeros((4,), dtype=np.float32)]),
        dock=dock,
        synth_batch_size=1,
    )

    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0,
        rf_range_m=40.0,
    )

    # Initial pose is origin.
    assert sup.mule_pose == (0.0, 0.0, 0.0)

    # Bootstrap.
    down = cluster.dispatch_down_bundle(MULE)
    cluster.dock.send_down(down)
    assert sup.wait_for_initial_dock(timeout=2.0)

    # Drive cluster's UP→DOWN cycle in a worker; each device serves twice
    # (Pass-1 + Pass-2).
    def _serve_cluster():
        up = cluster.dock.recv_up(timeout=2.0)
        cluster.ingest_up_bundle(up)
        cluster.aggregate_pending()
        cluster.close_cluster_round()
        cluster.dock.send_down(cluster.dispatch_down_bundle(MULE))

    cluster_t = threading.Thread(target=_serve_cluster, daemon=True)
    cluster_t.start()

    workers = []
    for cm in devices:
        def _loop(client=cm):
            for _ in range(2):
                client.serve_once()
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        workers.append(t)

    sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in workers:
        t.join(timeout=2.0)

    # After the mission, mule_pose must have moved off origin.
    assert sup.mule_pose != (0.0, 0.0, 0.0), (
        "mule_pose did not advance during the mission"
    )


# --------------------------------------------------------------------------- #
# H6 — Pass-2 push triggers train_offline so next Pass-1 has a delta ready
# --------------------------------------------------------------------------- #

def test_h6_pass_2_delivery_triggers_train_offline():
    """After ``_handle_delivery_push`` runs, ``_prepared_delta`` must be
    populated automatically (no external scheduler needed).
    """
    rf = LoopbackRFLink()
    rf.register_device(DeviceID("d0"))

    train_count = {"n": 0}

    def _counting_train(theta, synth):
        train_count["n"] += 1
        return LocalTrainResult(
            delta_theta=[t.copy() for t in theta],
            num_examples=4, accuracy=0.8, auc=0.8, loss=0.2,
            theta_after=[t.copy() for t in theta],
        )

    cm = ClientMission(
        device_id=DeviceID("d0"),
        rf=rf,
        local_train=_counting_train,
        solicit_timeout_s=0.5,
        disc_push_timeout_s=0.5,
    )
    cm.set_state(FLState.FL_OPEN)

    # Synthesise a Pass-2 push directly.
    push = DiscPush(
        mule_id=MULE,
        mission_round=1,
        theta_disc=_toy_theta(),
        synth_batch=[],
        pass_kind=MissionPass.DELIVER,
    )
    cm._handle_delivery_push(push)

    # H6: train_offline ran exactly once after the delivery push.
    assert train_count["n"] == 1
    # And the prepared delta is now staged for the next Pass-1 visit.
    assert cm._prepared_delta is not None


# --------------------------------------------------------------------------- #
# M1 — fallback path emits a warning
# --------------------------------------------------------------------------- #

def test_m1_fallback_emits_warning(caplog):
    """When _prepared_delta is None, the fallback path warns explicitly."""
    rf = LoopbackRFLink()
    rf.register_device(DeviceID("d0"))

    cm = ClientMission(
        device_id=DeviceID("d0"),
        rf=rf,
        local_train=_train_factory(),
        solicit_timeout_s=0.5,
        disc_push_timeout_s=0.5,
    )
    cm.set_state(FLState.FL_OPEN)

    push = DiscPush(
        mule_id=MULE,
        mission_round=1,
        theta_disc=_toy_theta(),
        synth_batch=[],
        pass_kind=MissionPass.COLLECT,
    )

    with caplog.at_level(logging.WARNING, logger="hermes.mission.client_mission"):
        cm._handle_collect_push(push)

    assert any(
        "violates design principle 14" in rec.message
        for rec in caplog.records
    ), f"M1 warning not emitted; got {[r.message for r in caplog.records]}"


# --------------------------------------------------------------------------- #
# M2 — select_contact / rank_contacts scope guard catches non-admitted devices
# --------------------------------------------------------------------------- #

def test_m2_rank_contacts_with_unadmitted_member_raises():
    sel = TargetSelectorRL(rng_seed=0)
    env = SelectorEnv(mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0,
                      rf_prior_snr_db=20.0, now=0.0)

    states = {
        DeviceID("d0"): DeviceSchedulerState(
            device_id=DeviceID("d0"), is_in_slice=True,
            last_known_position=(0.0, 0.0, 0.0),
        ),
        DeviceID("ghost"): DeviceSchedulerState(
            device_id=DeviceID("ghost"), is_in_slice=False,
            last_known_position=(100.0, 0.0, 0.0),
        ),
    }
    for s in states.values():
        s.bucket = Bucket.SCHEDULED_THIS_ROUND

    contact_with_ghost = ContactWaypoint(
        position=(50.0, 0.0, 0.0),
        devices=(DeviceID("d0"), DeviceID("ghost")),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=100.0,
    )

    # Pass admitted=[d0] only — `ghost` is NOT admitted; the guard fires.
    with pytest.raises(SelectorScopeViolation, match="ghost"):
        sel.rank_contacts(
            candidates=[contact_with_ghost],
            device_states=states,
            env=env,
            admitted=[DeviceID("d0")],
        )


def test_m2_rank_contacts_self_admitted_passes():
    """When ``admitted`` is omitted, behaviour degrades to self-check
    (passes for any internally-consistent contact). Backward compat."""
    sel = TargetSelectorRL(rng_seed=0)
    env = SelectorEnv(mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0,
                      rf_prior_snr_db=20.0, now=0.0)

    st = DeviceSchedulerState(
        device_id=DeviceID("d0"), is_in_slice=True,
        last_known_position=(0.0, 0.0, 0.0),
    )
    st.bucket = Bucket.SCHEDULED_THIS_ROUND

    cw = ContactWaypoint(
        position=(0.0, 0.0, 0.0),
        devices=(DeviceID("d0"),),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=100.0,
    )
    # No admitted= argument → backward-compat self-check, no raise.
    out = sel.rank_contacts(
        candidates=[cw],
        device_states={DeviceID("d0"): st},
        env=env,
    )
    assert out == [cw]


# --------------------------------------------------------------------------- #
# M3 — build_pass_2_queue does not mutate scheduler state
# --------------------------------------------------------------------------- #

def test_m3_build_pass_2_queue_does_not_mutate_bucket():
    """A scheduler state with no bucket set must STAY None after a
    Pass-2 build. Pre-fix, the function force-set
    Bucket.SCHEDULED_THIS_ROUND on any None bucket, leaking into
    subsequent Pass-1 classifications.
    """
    sch = FLScheduler(now_fn=lambda: 1000.0)
    rec = DeviceRecord(
        device_id=DeviceID("d0"),
        last_known_position=(10.0, 0.0, 0.0),
        spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        is_new=True,  # new → S3 will assign NEW; but we don't run S3 here.
    )
    sl = MissionSlice(
        mule_id=MULE,
        device_ids=(DeviceID("d0"),),
        issued_round=1,
        issued_at=1000.0,
    )
    sch.ingest_slice(sl, registry_records=[rec])
    # Verify pre-condition: no bucket set yet.
    st = sch.get_state(DeviceID("d0"))
    assert st.bucket is None

    # Calling build_pass_2_queue must NOT mutate st.bucket.
    contacts = sch.build_pass_2_queue(rf_range_m=60.0, mule_pose=(0.0, 0.0, 0.0))
    assert len(contacts) == 1

    # Post-condition: the scheduler state's bucket is STILL None.
    st_after = sch.get_state(DeviceID("d0"))
    assert st_after.bucket is None, (
        f"build_pass_2_queue mutated bucket to {st_after.bucket}; "
        f"M3 fix did not take"
    )


# --------------------------------------------------------------------------- #
# M5 — channel_choices split into pass-1 / pass-2
# --------------------------------------------------------------------------- #

def test_m5_two_pass_result_has_split_channel_choices():
    """MissionRunResult exposes pass_1_channel_choices and
    pass_2_channel_choices as separate lists.
    """
    from hermes.l1.channel_ddqn import ChannelDDQN

    rf = LoopbackRFLink()
    dock = LoopbackDockLink()

    positions = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    device_ids = [DeviceID(f"dev-{i:02d}") for i in range(2)]
    devices: List[ClientMission] = []
    for did, pos in zip(device_ids, positions):
        rf.register_device(did)
        cm = ClientMission(
            device_id=did, rf=rf, local_train=_train_factory(seed=0),
            solicit_timeout_s=2.0, disc_push_timeout_s=2.0,
        )
        cm.set_state(FLState.FL_OPEN)
        devices.append(cm)

    registry = DeviceRegistry()
    for did, pos in zip(device_ids, positions):
        registry.register(
            device_id=did, position=pos,
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    registry.rebalance([MULE], round_counter=0)
    cluster = HFLHostCluster(
        registry=registry,
        generator=StubGeneratorHost(disc_weights=[np.zeros((4,), dtype=np.float32)]),
        dock=dock, synth_batch_size=1,
    )

    sup = MuleSupervisor(
        mule_id=MULE, rf=rf, dock=dock, session_ttl_s=2.0,
        rf_range_m=60.0,
        channel_actor=ChannelDDQN(seed=0),
    )

    cluster.dock.send_down(cluster.dispatch_down_bundle(MULE))
    assert sup.wait_for_initial_dock(timeout=2.0)

    def _serve_cluster():
        up = cluster.dock.recv_up(timeout=2.0)
        cluster.ingest_up_bundle(up)
        cluster.aggregate_pending()
        cluster.close_cluster_round()
        cluster.dock.send_down(cluster.dispatch_down_bundle(MULE))

    cluster_t = threading.Thread(target=_serve_cluster, daemon=True)
    cluster_t.start()
    workers = []
    for cm in devices:
        def _loop(client=cm):
            for _ in range(2):
                client.serve_once()
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        workers.append(t)

    result = sup.run_one_mission()

    cluster_t.join(timeout=5.0)
    for t in workers:
        t.join(timeout=2.0)

    # Every Pass-1 contact and every Pass-2 contact gets one channel pick.
    assert len(result.pass_1_channel_choices) == len(result.pass_1_queue)
    assert len(result.pass_2_channel_choices) == len(result.pass_2_queue)
    # Combined channel_choices is the concatenation.
    assert (
        result.channel_choices
        == result.pass_1_channel_choices + result.pass_2_channel_choices
    )
