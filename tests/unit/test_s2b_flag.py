"""Phase 4 Stage 2B — FL_Threshold gate tests."""

from __future__ import annotations

from hermes.scheduler.stages import passes_fl_threshold
from hermes.scheduler.stages.s2b_flag import DEFAULT_FL_THRESHOLD
from hermes.types import DeviceID, FLReadyAdv, FLState


def _adv(utility: float) -> FLReadyAdv:
    return FLReadyAdv(
        device_id=DeviceID("d1"),
        state=FLState.FL_OPEN,
        performance_score=utility,
        diversity_adjusted=utility,
        utility=utility,
    )


def test_above_threshold_passes():
    assert passes_fl_threshold(_adv(utility=DEFAULT_FL_THRESHOLD + 0.01)) is True


def test_exactly_at_threshold_rejected():
    # Design §6.8: strict '>'.
    assert passes_fl_threshold(_adv(utility=DEFAULT_FL_THRESHOLD)) is False


def test_below_threshold_rejected():
    assert passes_fl_threshold(_adv(utility=DEFAULT_FL_THRESHOLD - 0.01)) is False


def test_custom_threshold_respected():
    adv = _adv(utility=0.4)
    assert passes_fl_threshold(adv, fl_threshold=0.3) is True
    assert passes_fl_threshold(adv, fl_threshold=0.5) is False
