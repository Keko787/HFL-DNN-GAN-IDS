"""Phase 3 bundle-signature tests."""

from __future__ import annotations

import numpy as np
import pytest

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
    UpBundle,
    sign_down_bundle,
    sign_up_bundle,
    verify_down_bundle,
    verify_up_bundle,
)


def _mule() -> MuleID:
    return MuleID("mA")


def _make_up() -> UpBundle:
    pa = PartialAggregate(
        mule_id=_mule(),
        mission_round=3,
        weights=[np.array([1.0, 2.0], dtype=np.float32)],
        num_examples=5,
        contributing_devices=(DeviceID("d1"),),
    )
    report = MissionRoundCloseReport(
        mule_id=_mule(),
        mission_round=3,
        started_at=0.0,
        finished_at=1.0,
        lines=[
            MissionRoundCloseLine(
                device_id=DeviceID("d1"),
                outcome=MissionOutcome.CLEAN,
                contact_ts=0.5,
            )
        ],
    )
    contacts = ContactHistory(mule_id=_mule(), mission_round=3)
    return UpBundle(
        mule_id=_mule(),
        partial_aggregate=pa,
        round_close_report=report,
        contact_history=contacts,
    )


def _make_down() -> DownBundle:
    slice_ = MissionSlice(
        mule_id=_mule(),
        device_ids=(DeviceID("d1"), DeviceID("d2")),
        issued_round=4,
        issued_at=0.0,
    )
    return DownBundle(
        mule_id=_mule(),
        mission_slice=slice_,
        theta_disc=[np.zeros((2,), dtype=np.float32)],
        synth_batch=[np.ones((3,), dtype=np.float32)],
        cluster_amendments=ClusterAmendment(cluster_round=4),
    )


def test_unsigned_bundle_does_not_verify():
    assert verify_up_bundle(_make_up()) is False
    assert verify_down_bundle(_make_down()) is False


def test_signed_up_bundle_verifies():
    b = _make_up()
    sign_up_bundle(b)
    assert b.bundle_sig != ""
    assert verify_up_bundle(b) is True


def test_signed_down_bundle_verifies():
    b = _make_down()
    sign_down_bundle(b)
    assert b.bundle_sig != ""
    assert verify_down_bundle(b) is True


def test_tampering_with_weights_breaks_signature():
    b = _make_up()
    sign_up_bundle(b)
    b.partial_aggregate.weights[0][0] = 99.0
    assert verify_up_bundle(b) is False


def test_tampering_with_slice_breaks_signature():
    b = _make_down()
    sign_down_bundle(b)
    # replace the frozen slice with a different one
    b.mission_slice = MissionSlice(
        mule_id=_mule(),
        device_ids=(DeviceID("dX"),),
        issued_round=4,
        issued_at=0.0,
    )
    assert verify_down_bundle(b) is False


def test_tampering_with_mule_id_breaks_signature():
    b = _make_up()
    sign_up_bundle(b)
    b.mule_id = MuleID("mB")
    b.partial_aggregate.mule_id = MuleID("mB")
    b.round_close_report.mule_id = MuleID("mB")
    assert verify_up_bundle(b) is False


def test_signatures_are_deterministic():
    b1 = _make_up()
    b2 = _make_up()
    assert sign_up_bundle(b1) == sign_up_bundle(b2)


def test_empty_sig_string_rejects():
    b = _make_down()
    # bundle built without signing -> bundle_sig is ""
    assert b.bundle_sig == ""
    assert verify_down_bundle(b) is False
