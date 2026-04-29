"""Phase 4 Stage 2A — on-contact readiness tests."""

from __future__ import annotations

from hermes.scheduler.stages import is_on_contact_ready
from hermes.types import DeviceID, FLReadyAdv, FLState


def _adv(**kw) -> FLReadyAdv:
    defaults = dict(
        device_id=DeviceID("d1"),
        state=FLState.FL_OPEN,
        performance_score=0.9,
        diversity_adjusted=0.8,
        utility=0.85,
        issued_at=0.0,
    )
    defaults.update(kw)
    return FLReadyAdv(**defaults)


def test_fl_open_advert_with_unset_timestamp_passes():
    assert is_on_contact_ready(_adv(), now=1000.0) is True


def test_busy_device_rejected():
    assert is_on_contact_ready(_adv(state=FLState.BUSY), now=1000.0) is False


def test_unavailable_device_rejected():
    assert is_on_contact_ready(_adv(state=FLState.UNAVAILABLE), now=1000.0) is False


def test_fresh_timestamp_passes():
    assert is_on_contact_ready(
        _adv(issued_at=999.0), now=1000.0, freshness_window_s=5.0
    ) is True


def test_stale_timestamp_rejected():
    assert is_on_contact_ready(
        _adv(issued_at=900.0), now=1000.0, freshness_window_s=5.0
    ) is False


def test_zero_window_skips_freshness_check():
    assert is_on_contact_ready(
        _adv(issued_at=1.0), now=1_000_000.0, freshness_window_s=0.0
    ) is True
