"""FeRRy Phase 1 — the mule's per-device merge cutoffs (decision D5).

``MuleSupervisor._age_caps`` turns each slice device's deadline window into an
``agg:cutoff`` cap in cluster rounds, a_max_j = min(a_max, ⌊Φ_j·s / T⌋). The Φ
must be the one the deadline uses: clamped by the scheduler's law and scaled
by S3c. A cap read from the raw stored window, or without the S3c scale, lets
the merge and the deadline disagree about how late a device may be (audit #19).

The method reads only ``self.aggregation`` and ``self.scheduler``, so a
namespace carrying those two stands in for the supervisor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes.mission.aggregation_rules import (
    AGG_ASYNCHFL,
    AGG_CUTOFF,
    AGG_FEDBUFF,
    AGG_PLAIN,
    AggregationSpec,
)
from hermes.mule.mule_main import MuleSupervisor
from hermes.scheduler import FLScheduler
from hermes.scheduler.stages.s3_deadline import LAW_ADDITIVE, LAW_MULTIPLICATIVE, DeadlineLaw
from hermes.scheduler.stages.s3c_mission_window import MissionWindowAdapter
from hermes.types import DeviceID, MissionSlice, MuleID

#: Φ_j per device: under a 20 s period, 3, 6 and 50 rounds unclamped.
WINDOWS = {DeviceID("short"): 60.0, DeviceID("mid"): 130.0, DeviceID("long"): 1000.0}


def _caps(spec, *, law=None, adapter=None):
    sched = FLScheduler(
        now_fn=lambda: 1000.0, deadline_law=law, mission_window_adapter=adapter,
    )
    sched.ingest_slice(MissionSlice(
        mule_id=MuleID("m"), device_ids=tuple(WINDOWS), issued_round=0, issued_at=0.0,
    ))
    for did, phi in WINDOWS.items():
        sched.device_states[did].deadline_fulfilment_s = phi
    caps = MuleSupervisor._age_caps(SimpleNamespace(aggregation=spec, scheduler=sched))
    if caps is None:
        return None
    assert set(caps) == set(WINDOWS)
    return tuple(caps[did] for did in WINDOWS)


def test_the_fixed_cap_and_the_window_cap_take_the_smaller():
    spec = AggregationSpec(rule=AGG_CUTOFF, period_s=20.0, a_max=5)
    assert _caps(spec) == (3, 5, 5)


def test_a_fixed_cap_alone_applies_to_every_device():
    assert _caps(AggregationSpec(rule=AGG_CUTOFF, a_max=1)) == (1, 1, 1)


@pytest.mark.parametrize("law, expected", [
    (None, (3, 6, 50)),                                 # the recorded law: no ceiling
    (DeadlineLaw(form=LAW_ADDITIVE), (3, 6, 50)),
    (DeadlineLaw(form=LAW_MULTIPLICATIVE), (3, 6, 15)),  # Φ = 1000 clamps to 300
])
def test_the_window_cap_reads_the_laws_clamped_window(law, expected):
    assert _caps(AggregationSpec(rule=AGG_CUTOFF, period_s=20.0), law=law) == expected


def test_the_window_cap_is_scaled_by_s3c():
    adapter = MissionWindowAdapter(enabled=True, target_success=0.8, gain=2.0)
    adapter.record(3, 10)                   # 0.5 short of target: 1 + 2.0 × 0.5
    assert adapter.scale == pytest.approx(2.0)
    spec = AggregationSpec(rule=AGG_CUTOFF, period_s=20.0)
    assert _caps(spec, adapter=adapter) == (6, 13, 100)
    # Clamp, then scale, as compute_deadline does: 300 × 2, not 300.
    mult = DeadlineLaw(form=LAW_MULTIPLICATIVE)
    assert _caps(spec, law=mult, adapter=adapter) == (6, 13, 30)
    # The same adapter disabled is inert.
    assert _caps(spec, adapter=MissionWindowAdapter(enabled=False)) == (3, 6, 50)


@pytest.mark.parametrize("spec", [
    AggregationSpec(rule=AGG_PLAIN),
    AggregationSpec(rule=AGG_PLAIN, a_max=1, period_s=20.0),
    AggregationSpec(rule=AGG_CUTOFF),
    AggregationSpec(rule=AGG_ASYNCHFL, a_max=1, period_s=20.0),
    AggregationSpec(rule=AGG_FEDBUFF, a_max=1, period_s=20.0),
], ids=["plain", "plain-with-params", "cutoff-no-cap", "asynchfl", "fedbuff"])
def test_no_cutoff_rule_means_no_caps(spec):
    assert _caps(spec) is None
