"""Phase 4 FLScheduler — orchestrator and ingest-path tests."""

from __future__ import annotations

import time

import pytest

from hermes.scheduler import FLScheduler, FLSchedulerError
from hermes.types import (
    BeaconObservation,
    Bucket,
    ClusterAmendment,
    DeviceID,
    DeviceRecord,
    FLReadyAdv,
    FLState,
    MissionOutcome,
    MissionSlice,
    MuleID,
    RoundCloseDelta,
    SpectrumSig,
)


def _fixed_clock(t: float):
    return lambda: t


def _sig() -> SpectrumSig:
    return SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,))


def _slice(*ids: str, issued_at: float = 1000.0) -> MissionSlice:
    return MissionSlice(
        mule_id=MuleID("m1"),
        device_ids=tuple(DeviceID(i) for i in ids),
        issued_round=1,
        issued_at=issued_at,
    )


# --------------------------------------------------------------------------- #
# ingest_slice
# --------------------------------------------------------------------------- #

def test_ingest_slice_creates_states_for_new_members():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("a", "b"))
    assert set(sch.device_states.keys()) == {DeviceID("a"), DeviceID("b")}
    assert sch.device_states[DeviceID("a")].is_in_slice is True


def test_ingest_slice_flips_membership_on_prior_member():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("a", "b"))
    # Second slice drops "b", adds "c"
    sch.ingest_slice(_slice("a", "c"))
    assert sch.device_states[DeviceID("a")].is_in_slice is True
    assert sch.device_states[DeviceID("b")].is_in_slice is False
    assert sch.device_states[DeviceID("c")].is_in_slice is True


def test_ingest_slice_folds_amendment_overrides():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    amend = ClusterAmendment(
        cluster_round=1,
        deadline_overrides={DeviceID("a"): 9999.0},
    )
    sch.ingest_slice(_slice("a", "b"), amendment=amend)
    assert sch.device_states[DeviceID("a")].deadline_override_ts == 9999.0
    assert sch.device_states[DeviceID("b")].deadline_override_ts is None


def test_ingest_slice_seeds_position_from_registry():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    record = DeviceRecord(
        device_id=DeviceID("a"),
        last_known_position=(1.0, 2.0, 3.0),
        spectrum_sig=_sig(),
        is_new=False,
    )
    sch.ingest_slice(_slice("a"), registry_records=[record])
    st = sch.device_states[DeviceID("a")]
    assert st.last_known_position == (1.0, 2.0, 3.0)
    assert st.is_new is False


# --------------------------------------------------------------------------- #
# ingest_round_close_delta / ingest_beacon
# --------------------------------------------------------------------------- #

def test_ingest_delta_applies_fast_phase_fold():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("a"))
    delta = RoundCloseDelta(
        device_id=DeviceID("a"),
        mule_id=MuleID("m1"),
        mission_round=1,
        outcome=MissionOutcome.CLEAN,
        utility=0.9,
        contact_ts=1000.0,
    )
    sch.ingest_round_close_delta(delta)
    st = sch.device_states[DeviceID("a")]
    assert st.is_new is False
    assert st.last_outcome is MissionOutcome.CLEAN


def test_ingest_delta_for_untracked_device_raises():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    delta = RoundCloseDelta(
        device_id=DeviceID("ghost"),
        mule_id=MuleID("m1"),
        mission_round=1,
        outcome=MissionOutcome.CLEAN,
        utility=0.9,
        contact_ts=1000.0,
    )
    with pytest.raises(FLSchedulerError):
        sch.ingest_round_close_delta(delta)


def test_ingest_beacon_auto_creates_state():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_beacon(BeaconObservation(device_id=DeviceID("b"), observed_at=990.0))
    st = sch.device_states[DeviceID("b")]
    assert st.last_beacon_ts == 990.0
    assert st.is_new is True


# --------------------------------------------------------------------------- #
# ingest_ready_adv
# --------------------------------------------------------------------------- #

def test_ready_adv_passes_both_gates():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    adv = FLReadyAdv(
        device_id=DeviceID("a"),
        state=FLState.FL_OPEN,
        performance_score=0.9,
        diversity_adjusted=0.85,
        utility=0.88,
    )
    assert sch.ingest_ready_adv(adv) is True


def test_ready_adv_below_threshold_rejected():
    sch = FLScheduler(fl_threshold=0.9, now_fn=_fixed_clock(1000.0))
    adv = FLReadyAdv(
        device_id=DeviceID("a"),
        state=FLState.FL_OPEN,
        performance_score=0.5,
        diversity_adjusted=0.5,
        utility=0.5,
    )
    assert sch.ingest_ready_adv(adv) is False


def test_ready_adv_wrong_state_rejected():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    adv = FLReadyAdv(
        device_id=DeviceID("a"),
        state=FLState.BUSY,
        performance_score=0.99,
        diversity_adjusted=0.99,
        utility=0.99,
    )
    assert sch.ingest_ready_adv(adv) is False


# --------------------------------------------------------------------------- #
# build_target_queue — the end-to-end pipeline
# --------------------------------------------------------------------------- #

def test_build_queue_respects_bucket_priority():
    """Priority: new -> scheduled -> beacon-active."""
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))

    # Slice = 2 members: one new, one scheduled (not new).
    sch.ingest_slice(_slice("new_dev", "sched_dev"))
    # Mark sched_dev as already seen in a prior round.
    sch.device_states[DeviceID("sched_dev")].is_new = False
    # Third device: not in slice, but has a fresh beacon.
    sch.ingest_beacon(
        BeaconObservation(device_id=DeviceID("bcn"), observed_at=995.0)
    )
    sch.device_states[DeviceID("bcn")].is_new = False  # otherwise it'd bucket NEW

    queue = sch.build_target_queue(now=1000.0, mule_pose=(0.0, 0.0, 0.0))
    ordered_buckets = [wp.bucket for wp in queue]
    ordered_ids = [wp.device_id for wp in queue]

    assert ordered_buckets == [
        Bucket.NEW,
        Bucket.SCHEDULED_THIS_ROUND,
        Bucket.BEACON_ACTIVE,
    ]
    assert ordered_ids == [
        DeviceID("new_dev"),
        DeviceID("sched_dev"),
        DeviceID("bcn"),
    ]


def test_build_queue_intra_bucket_sorts_by_distance():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("far", "near", "mid"))
    sch.device_states[DeviceID("far")].last_known_position = (100.0, 0.0, 0.0)
    sch.device_states[DeviceID("near")].last_known_position = (1.0, 0.0, 0.0)
    sch.device_states[DeviceID("mid")].last_known_position = (10.0, 0.0, 0.0)

    queue = sch.build_target_queue(now=1000.0, mule_pose=(0.0, 0.0, 0.0))
    assert [wp.device_id for wp in queue] == [
        DeviceID("near"),
        DeviceID("mid"),
        DeviceID("far"),
    ]
    # All in NEW bucket since default is_new=True
    assert all(wp.bucket is Bucket.NEW for wp in queue)


def test_build_queue_filters_out_ineligible_devices():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("a"))
    # Orphan device never in slice, never beaconed.
    sch.device_states[DeviceID("orphan")] = sch.device_states[DeviceID("a")].__class__(
        device_id=DeviceID("orphan")
    )
    queue = sch.build_target_queue(now=1000.0)
    assert [wp.device_id for wp in queue] == [DeviceID("a")]


def test_queue_carries_deadline_and_position():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(
        _slice("a"),
        amendment=ClusterAmendment(
            cluster_round=1, deadline_overrides={DeviceID("a"): 5555.0}
        ),
    )
    sch.device_states[DeviceID("a")].last_known_position = (7.0, 8.0, 9.0)
    queue = sch.build_target_queue(now=1000.0)
    assert len(queue) == 1
    wp = queue[0]
    assert wp.deadline_ts == 5555.0
    assert wp.position == (7.0, 8.0, 9.0)


# --------------------------------------------------------------------------- #
# Sprint 1.5 — two-pass + clustering (unit-level wiring tests)
#
# The per-stage S3a clustering math has dedicated coverage in
# test_s3a_cluster.py; the two-pass mission integration is exercised in
# tests/integration/test_two_pass_contact.py. These tests pin the
# scheduler-level orchestration: build_contact_queue actually returns
# ContactWaypoints and build_pass_2_queue walks every slice member.
# --------------------------------------------------------------------------- #

def _seed_two_devices_in_one_cluster(sch: FLScheduler) -> None:
    """Two devices a metre apart so any sane rf_range groups them."""
    sch.ingest_slice(_slice("a", "b"))
    sch.device_states[DeviceID("a")].last_known_position = (0.0, 0.0, 0.0)
    sch.device_states[DeviceID("b")].last_known_position = (1.0, 0.0, 0.0)


def test_build_contact_queue_returns_contact_waypoints():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    _seed_two_devices_in_one_cluster(sch)
    contacts = sch.build_contact_queue(rf_range_m=10.0, now=1000.0)
    assert len(contacts) == 1
    contact = contacts[0]
    assert set(contact.devices) == {DeviceID("a"), DeviceID("b")}


def test_build_contact_queue_splits_when_devices_out_of_rf_range():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("near", "far"))
    sch.device_states[DeviceID("near")].last_known_position = (0.0, 0.0, 0.0)
    sch.device_states[DeviceID("far")].last_known_position = (200.0, 0.0, 0.0)
    contacts = sch.build_contact_queue(rf_range_m=10.0, now=1000.0)
    assert len(contacts) == 2
    members = {tuple(sorted(c.devices)) for c in contacts}
    assert members == {(DeviceID("far"),), (DeviceID("near"),)}


def test_build_contact_queue_rejects_zero_rf_range():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    _seed_two_devices_in_one_cluster(sch)
    with pytest.raises(FLSchedulerError):
        sch.build_contact_queue(rf_range_m=0.0)


def test_build_pass_2_queue_visits_every_slice_member():
    """Pass 2 is delivery-only — no S1 filtering, no bucket priority."""
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("a", "b", "c"))
    sch.device_states[DeviceID("a")].last_known_position = (0.0, 0.0, 0.0)
    sch.device_states[DeviceID("b")].last_known_position = (50.0, 0.0, 0.0)
    sch.device_states[DeviceID("c")].last_known_position = (100.0, 0.0, 0.0)
    contacts = sch.build_pass_2_queue(rf_range_m=20.0, mule_pose=(0.0, 0.0, 0.0))
    visited = {did for c in contacts for did in c.devices}
    assert visited == {DeviceID("a"), DeviceID("b"), DeviceID("c")}


def test_build_pass_2_queue_orders_nearest_first_from_mule_pose():
    sch = FLScheduler(now_fn=_fixed_clock(1000.0))
    sch.ingest_slice(_slice("near", "mid", "far"))
    sch.device_states[DeviceID("near")].last_known_position = (5.0, 0.0, 0.0)
    sch.device_states[DeviceID("mid")].last_known_position = (50.0, 0.0, 0.0)
    sch.device_states[DeviceID("far")].last_known_position = (500.0, 0.0, 0.0)
    contacts = sch.build_pass_2_queue(rf_range_m=10.0, mule_pose=(0.0, 0.0, 0.0))
    # Each device falls into its own contact at rf_range=10 m; visit order
    # should be near → mid → far when greedy-walking from origin.
    visit_order = [next(iter(c.devices)) for c in contacts]
    assert visit_order == [DeviceID("near"), DeviceID("mid"), DeviceID("far")]


def test_build_contact_queue_skips_selector_when_bucket_has_one_candidate():
    """Design §2.7: selector is only consulted when a bucket has ≥2 candidates.

    With one candidate there is nothing to choose between, so we skip the
    DDQN forward pass entirely. Pinned with a counting stub that fails
    the test if rank_contacts is invoked on a singleton bucket.
    """
    class _CountingSelector:
        def __init__(self):
            self.rank_calls = 0

        def rank_contacts(self, candidates, device_states, *, env, pass_kind, admitted):
            self.rank_calls += 1
            return list(candidates)

    spy = _CountingSelector()
    sch = FLScheduler(
        now_fn=_fixed_clock(1000.0),
        target_selector=spy,
    )
    sch.ingest_slice(_slice("solo"))
    sch.device_states[DeviceID("solo")].last_known_position = (0.0, 0.0, 0.0)
    sch.build_contact_queue(rf_range_m=10.0, now=1000.0)
    assert spy.rank_calls == 0, "selector must not run on singleton bucket"


def test_build_contact_queue_consults_selector_when_bucket_has_two_or_more():
    """Inverse of the singleton test — confirm the ≥2 path still fires."""
    class _CountingSelector:
        def __init__(self):
            self.rank_calls = 0

        def rank_contacts(self, candidates, device_states, *, env, pass_kind, admitted):
            self.rank_calls += 1
            return list(candidates)

    spy = _CountingSelector()
    sch = FLScheduler(
        now_fn=_fixed_clock(1000.0),
        target_selector=spy,
    )
    # Two devices placed far enough apart to land in separate contacts at
    # rf_range=10 m, putting both into the same NEW bucket.
    sch.ingest_slice(_slice("a", "b"))
    sch.device_states[DeviceID("a")].last_known_position = (0.0, 0.0, 0.0)
    sch.device_states[DeviceID("b")].last_known_position = (100.0, 0.0, 0.0)
    sch.build_contact_queue(rf_range_m=10.0, now=1000.0)
    assert spy.rank_calls == 1, "selector must run on multi-candidate bucket"
