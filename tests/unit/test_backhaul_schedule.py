"""Phase 0 — the backhaul loss schedule is indexed by the UP bundle's mission.

The cluster loop read ``getattr(up, "mission_round", None)``, but ``UpBundle``
carries its round on ``partial_aggregate``. The lookup returned None, so every
mission drew against the mission-1 loss probability of the L1 schedule, and the
``backhaul_upload_lost`` event carried no round — which left
``backhaul_lost_rounds`` empty and let dropped rounds count as closed in the
Exp 4 round-close metrics.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hermes.processes.cluster import ClusterService, _up_mission_round
from hermes.processes.config import ClusterConfig
from hermes.types import (
    ContactHistory,
    DeviceID,
    MissionOutcome,
    MissionRoundCloseLine,
    MissionRoundCloseReport,
    MuleID,
    PartialAggregate,
    UpBundle,
)


def _up(mission_round: int) -> UpBundle:
    mule = MuleID("m1")
    return UpBundle(
        mule_id=mule,
        partial_aggregate=PartialAggregate(
            mule_id=mule,
            mission_round=mission_round,
            weights=[np.zeros(2, dtype=np.float32)],
            num_examples=1,
            contributing_devices=(DeviceID("d1"),),
        ),
        round_close_report=MissionRoundCloseReport(
            mule_id=mule,
            mission_round=mission_round,
            started_at=0.0,
            finished_at=1.0,
            lines=[
                MissionRoundCloseLine(
                    device_id=DeviceID("d1"),
                    outcome=MissionOutcome.CLEAN,
                    contact_ts=0.5,
                )
            ],
        ),
        contact_history=ContactHistory(mule_id=mule, mission_round=mission_round),
    )


@pytest.fixture
def service_with_schedule():
    made = []

    def _make(schedule):
        cfg = ClusterConfig(
            cluster_id="cluster-backhaul-test",
            dock_host="127.0.0.1",
            dock_port=0,
            synth_batch_size=2,
            min_participation=1,
            backhaul_loss_schedule=list(schedule),
            backhaul_rng_seed=7,
        )
        svc = ClusterService(cfg)
        made.append(svc)
        return svc

    yield _make
    for svc in made:
        svc.shutdown()


def test_the_round_comes_from_the_partial_aggregate():
    up = _up(3)
    assert not hasattr(up, "mission_round")   # the attribute the old code read
    assert _up_mission_round(up) == 3


def test_no_bundle_means_no_round():
    assert _up_mission_round(None) is None


def test_each_mission_draws_against_its_own_loss_probability(service_with_schedule):
    # p = 1 always drops and p = 0 never does, so the draw is deterministic.
    svc = service_with_schedule([0.0, 1.0, 0.0])
    dropped = [svc._backhaul_dropped(_up_mission_round(_up(m))) for m in (1, 2, 3)]
    assert dropped == [False, True, False]


def test_rounds_past_the_schedule_use_its_last_entry(service_with_schedule):
    svc = service_with_schedule([0.0, 1.0])
    assert svc._backhaul_dropped(_up_mission_round(_up(5))) is True


def test_the_service_loop_takes_the_round_from_the_partial():
    """The loop is not unit-drivable without a docked mule, so pin the lookup
    it uses: it must not read the attribute ``UpBundle`` does not have."""
    src = inspect.getsource(ClusterService.run)
    assert 'getattr(up, "mission_round"' not in src
    assert "_up_mission_round(up)" in src
