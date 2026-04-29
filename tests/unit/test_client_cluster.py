"""Phase 3 ClientCluster tests — state, verify, retry, distribution."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pytest

from hermes.mule import (
    BundleDistributor,
    ClientCluster,
    ClientClusterError,
    ClientClusterState,
)
from hermes.transport import DockLinkError, LoopbackDockLink
from hermes.types import (
    ClusterAmendment,
    ContactHistory,
    DeviceID,
    DownBundle,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MissionSlice,
    MuleID,
    PartialAggregate,
    sign_down_bundle,
)


def _mule() -> MuleID:
    return MuleID("mA")


def _agg(round_: int = 1, mule: Optional[MuleID] = None) -> PartialAggregate:
    return PartialAggregate(
        mule_id=mule or _mule(),
        mission_round=round_,
        weights=[np.array([0.1, 0.2], dtype=np.float32)],
        num_examples=4,
        contributing_devices=(DeviceID("d1"),),
    )


def _report(round_: int = 1, mule: Optional[MuleID] = None) -> MissionRoundCloseReport:
    r = MissionRoundCloseReport(
        mule_id=mule or _mule(),
        mission_round=round_,
        started_at=0.0,
        finished_at=1.0,
    )
    r.append(
        MissionRoundCloseLine(
            device_id=DeviceID("d1"),
            outcome=MissionOutcome.CLEAN,
            contact_ts=0.5,
        )
    )
    return r


def _contacts(round_: int = 1, mule: Optional[MuleID] = None) -> ContactHistory:
    return ContactHistory(mule_id=mule or _mule(), mission_round=round_)


def _queue_signed_down(
    dock: LoopbackDockLink,
    mule: MuleID,
    round_: int,
    amendment: Optional[ClusterAmendment] = None,
) -> DownBundle:
    slice_ = MissionSlice(
        mule_id=mule,
        device_ids=(DeviceID("d1"), DeviceID("d2")),
        issued_round=round_,
        issued_at=time.time(),
    )
    bundle = DownBundle(
        mule_id=mule,
        mission_slice=slice_,
        theta_disc=[np.zeros((2,), dtype=np.float32)],
        synth_batch=[np.ones((3,), dtype=np.float32)],
        cluster_amendments=amendment or ClusterAmendment(cluster_round=round_),
    )
    sign_down_bundle(bundle)
    dock.send_down(bundle)
    return bundle


def _stage(cc: ClientCluster, round_: int = 1) -> None:
    cc.collect(
        partial_aggregate=_agg(round_=round_),
        report=_report(round_=round_),
        contacts=_contacts(round_=round_),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_state_is_await_dock():
    cc = ClientCluster(mule_id=_mule(), dock=LoopbackDockLink())
    assert cc.state is ClientClusterState.AWAIT_DOCK


def test_collect_rejects_mismatched_mule_id():
    cc = ClientCluster(mule_id=_mule(), dock=LoopbackDockLink())
    with pytest.raises(ClientClusterError):
        cc.collect(
            partial_aggregate=_agg(mule=MuleID("other")),
            report=_report(),
            contacts=_contacts(),
        )


def test_run_dock_cycle_without_staged_or_retry_raises():
    cc = ClientCluster(mule_id=_mule(), dock=LoopbackDockLink())
    with pytest.raises(ClientClusterError, match="nothing staged"):
        cc.run_dock_cycle()


def test_happy_path_up_down_verify_distribute():
    dock = LoopbackDockLink()
    seen_slices = []
    seen_models = []
    cc = ClientCluster(
        mule_id=_mule(),
        dock=dock,
        distributor=BundleDistributor(
            on_slice_and_amendment=lambda s, a: seen_slices.append((s, a)),
            on_next_round_model=lambda w, b: seen_models.append((w, b)),
        ),
    )

    _stage(cc)
    _queue_signed_down(dock, _mule(), round_=1)

    got = cc.run_dock_cycle()
    assert got is not None
    assert got.mule_id == _mule()
    assert len(seen_slices) == 1
    assert len(seen_models) == 1
    assert cc.state is ClientClusterState.AWAIT_DOCK
    assert cc.retry_queue_depth() == 0

    # Cluster-side really did receive the UP (FIFO queue)
    up = dock.recv_up(timeout=0.1)
    assert up.mule_id == _mule()
    assert up.bundle_sig != ""


def test_up_bundle_is_auto_signed_before_send():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=_mule(), dock=dock)
    _stage(cc)
    _queue_signed_down(dock, _mule(), round_=1)
    cc.run_dock_cycle()

    up = dock.recv_up(timeout=0.1)
    # Re-import to use verifier without pulling into every test
    from hermes.types import verify_up_bundle

    assert verify_up_bundle(up) is True


def test_verify_fails_on_tampered_down_bundle():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=_mule(), dock=dock)
    _stage(cc)
    tampered = _queue_signed_down(dock, _mule(), round_=1)
    tampered.theta_disc[0][0] = 99.0  # post-sign tamper

    with pytest.raises(ClientClusterError, match="signature"):
        cc.run_dock_cycle()


def test_verify_fails_on_mule_id_routing_mismatch():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=_mule(), dock=dock)
    _stage(cc)

    wrong = DownBundle(
        mule_id=MuleID("mOTHER"),
        mission_slice=MissionSlice(
            mule_id=MuleID("mOTHER"),
            device_ids=(DeviceID("d1"),),
            issued_round=1,
            issued_at=0.0,
        ),
        theta_disc=[np.zeros((2,), dtype=np.float32)],
        synth_batch=[],
        cluster_amendments=ClusterAmendment(cluster_round=1),
    )
    sign_down_bundle(wrong)
    # force-route onto OUR mule queue (simulates a broken router)
    q = dock._ensure_down_queue(_mule())  # type: ignore[attr-defined]
    q.put(wrong)

    with pytest.raises(ClientClusterError, match="wrong mule"):
        cc.run_dock_cycle()


def test_up_retry_persists_across_cycles():
    """Simulate UP drop on first dock, success on second."""
    class FlakyDock(LoopbackDockLink):
        def __init__(self, drop_n: int):
            super().__init__()
            self._drop_n = drop_n

        def client_send_up(self, bundle):
            if self._drop_n > 0:
                self._drop_n -= 1
                raise DockLinkError("simulated UP drop")
            return super().client_send_up(bundle)

    dock = FlakyDock(drop_n=1)
    cc = ClientCluster(mule_id=_mule(), dock=dock, max_retry_attempts=5)
    _stage(cc)

    # Cycle 1: UP drops. _send_bundles returns False -> run_dock_cycle returns None.
    got = cc.run_dock_cycle()
    assert got is None
    assert cc.retry_queue_depth() == 1

    # Cycle 2: retry queue carries the bundle. This time it lands.
    _queue_signed_down(dock, _mule(), round_=1)
    got = cc.run_dock_cycle()
    assert got is not None
    assert cc.retry_queue_depth() == 0


def test_up_retry_raises_at_max_attempts():
    class DeadDock(LoopbackDockLink):
        def client_send_up(self, bundle):
            raise DockLinkError("permadead")

    dock = DeadDock()
    cc = ClientCluster(mule_id=_mule(), dock=dock, max_retry_attempts=2)
    _stage(cc)

    # Cycle 1: attempts=1 (<max), returns None
    got = cc.run_dock_cycle()
    assert got is None
    assert cc.retry_queue_depth() == 1

    # Cycle 2: attempts=2 (==max) -> raises
    with pytest.raises(ClientClusterError, match="max_retry_attempts"):
        cc.run_dock_cycle()


def test_wait_for_dock_timeout_returns_false():
    class ClosedDock(LoopbackDockLink):
        def is_available(self) -> bool:
            return False

    cc = ClientCluster(
        mule_id=_mule(),
        dock=ClosedDock(),
        dock_poll_interval_s=0.01,
    )
    assert cc.wait_for_dock(timeout=0.05) is False


def test_wait_for_dock_returns_true_when_available():
    cc = ClientCluster(
        mule_id=_mule(),
        dock=LoopbackDockLink(),
        dock_poll_interval_s=0.01,
    )
    assert cc.wait_for_dock(timeout=0.5) is True


def test_state_progresses_through_cycle_and_returns_to_await():
    dock = LoopbackDockLink()
    cc = ClientCluster(mule_id=_mule(), dock=dock)
    _stage(cc)
    _queue_signed_down(dock, _mule(), round_=1)
    assert cc.state is ClientClusterState.COLLECT
    cc.run_dock_cycle()
    assert cc.state is ClientClusterState.AWAIT_DOCK


def test_distributor_sinks_are_exception_safe():
    dock = LoopbackDockLink()

    def boom_slice(_s, _a):
        raise RuntimeError("scheduler sink down")

    def boom_model(_w, _b):
        raise RuntimeError("mission sink down")

    cc = ClientCluster(
        mule_id=_mule(),
        dock=dock,
        distributor=BundleDistributor(
            on_slice_and_amendment=boom_slice,
            on_next_round_model=boom_model,
        ),
    )
    _stage(cc)
    _queue_signed_down(dock, _mule(), round_=1)
    # Neither sink exception should surface
    got = cc.run_dock_cycle()
    assert got is not None
    assert cc.state is ClientClusterState.AWAIT_DOCK
