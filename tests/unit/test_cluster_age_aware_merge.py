"""FeRRy Phase 1 — the cluster folds partials under the configured rule.

``agg:plain`` must keep producing exactly ``cross_mule_fedavg``'s overwrite;
the age-aware rules add a weighted update to θ at the server rate; FedBuff
holds updates until K have arrived and says so, so the service can still send
the waiting mule a DOWN.
"""

from __future__ import annotations

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.cross_mule_fedavg import cross_mule_fedavg
from hermes.cluster.host_cluster import StubGeneratorHost
from hermes.mission.aggregation_rules import (
    AGG_CUTOFF,
    AGG_FEDBUFF,
    AggregationConfigError,
    AggregationSpec,
    merge_on_mule,
)
from hermes.transport import LoopbackDockLink
from hermes.types import (
    ContactHistory,
    DeviceID,
    GradientSubmission,
    MissionRoundCloseReport,
    MuleID,
    SpectrumSig,
    UpBundle,
)
from hermes.types.fl_messages import UPDATE_FORM_DELTA, UPDATE_FORM_WEIGHTS

M1, M2 = MuleID("m1"), MuleID("m2")


def _theta0():
    return [np.zeros((4,), dtype=np.float32), np.full((3, 3), 0.5, dtype=np.float32)]


def _cluster(spec: AggregationSpec, n_devices: int = 4) -> HFLHostCluster:
    reg = DeviceRegistry()
    for i in range(n_devices):
        reg.register(
            device_id=DeviceID(f"d{i}"), position=(float(i), 0.0, 0.0),
            spectrum_sig=SpectrumSig(bands=(0,), last_good_snr_per_band=(20.0,)),
        )
    reg.rebalance([M1, M2], round_counter=0)
    return HFLHostCluster(
        registry=reg,
        generator=StubGeneratorHost(disc_weights=_theta0()),
        dock=LoopbackDockLink(),
        aggregation=spec,
    )


def _up(mule, spec, *, shift, base_version, n_updates=1, mission_round=1):
    theta = _theta0()
    subs = []
    for k in range(n_updates):
        local = [w + shift + 0.01 * k for w in theta]
        if spec.is_plain:
            payload, form = local, UPDATE_FORM_WEIGHTS
        else:
            payload, form = [a - b for a, b in zip(local, theta)], UPDATE_FORM_DELTA
        subs.append(GradientSubmission(
            device_id=DeviceID(f"{mule}-dev{k}"), mule_id=mule,
            mission_round=mission_round, delta_theta=payload, num_examples=10,
            submitted_at=0.0, basis_version=base_version, update_form=form,
        ))
    partial = merge_on_mule(
        spec, mule_id=mule, mission_round=mission_round,
        submissions=subs, base_version=base_version,
    )
    return UpBundle(
        mule_id=mule,
        partial_aggregate=partial,
        round_close_report=MissionRoundCloseReport(
            mule_id=mule, mission_round=mission_round, started_at=0.0, finished_at=0.0,
        ),
        contact_history=ContactHistory(mule_id=mule, mission_round=mission_round),
    )


def test_plain_cluster_still_overwrites_with_cross_mule_fedavg():
    spec = AggregationSpec()
    c = _cluster(spec)
    up = _up(M1, spec, shift=1.0, base_version=0)
    c.ingest_up_bundle(up)
    merged = c.aggregate_pending()
    ref = cross_mule_fedavg([up.partial_aggregate])
    assert all(np.array_equal(a, b) for a, b in zip(merged, ref))
    assert c.last_merge is None and not c.defers_merges


def test_cutoff_cluster_adds_the_update_to_theta():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    c = _cluster(spec)
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=0))
    merged = c.aggregate_pending()
    for got, t in zip(merged, _theta0()):
        np.testing.assert_allclose(got, t + 1.0, rtol=1e-6)
    assert c.last_merge["rule"] == AGG_CUTOFF and c.last_merge["partial_ages"] == [0]
    # the held global θ was updated too
    for got, held in zip(merged, c.generator.get_global_disc_weights()):
        assert np.array_equal(got, held)


def test_server_rate_scales_the_step():
    spec = AggregationSpec(rule=AGG_CUTOFF, server_lr=0.25)
    c = _cluster(spec)
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=0))
    merged = c.aggregate_pending()
    for got, t in zip(merged, _theta0()):
        np.testing.assert_allclose(got, t + 0.25, rtol=1e-6)


def test_cluster_refuses_a_partial_built_by_another_rule():
    c = _cluster(AggregationSpec(rule=AGG_CUTOFF))
    c.ingest_up_bundle(_up(M1, AggregationSpec(), shift=1.0, base_version=0))
    with pytest.raises(AggregationConfigError, match="Set the same rule"):
        c.aggregate_pending()


def test_fedbuff_defers_until_k_updates_then_applies_their_mean():
    spec = AggregationSpec(rule=AGG_FEDBUFF, buffer_k=3)
    c = _cluster(spec)
    assert c.defers_merges
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=0, n_updates=2))
    assert c.aggregate_pending() is None
    assert c.last_merge == {
        "rule": AGG_FEDBUFF, "applied": False, "buffered": 2, "k": 3, "partial_ages": [0],
    }
    assert c.cluster_round == 0 and c.pending_partials() == 0
    # The same mule docks again: its UP must not be dropped as a duplicate.
    c.ingest_up_bundle(_up(M1, spec, shift=4.0, base_version=0, n_updates=1, mission_round=2))
    merged = c.aggregate_pending()
    assert merged is not None and c.last_merge["applied"] and c.last_merge["buffered"] == 3
    # mean of the three updates: (1.00 + 1.01 + 4.00) / 3 on every element
    for got, t in zip(merged, _theta0()):
        np.testing.assert_allclose(got, t + (1.0 + 1.01 + 4.0) / 3.0, rtol=1e-5)


def test_fedbuff_k_defaults_to_the_registered_devices():
    c = _cluster(AggregationSpec(rule=AGG_FEDBUFF), n_devices=5)
    assert c._fedbuff_k() == 5
