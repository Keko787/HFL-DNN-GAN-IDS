"""FeRRy Phase 1 — the cluster folds partials under the configured rule.

``agg:plain`` must keep producing exactly ``cross_mule_fedavg``'s overwrite;
the age-aware rules add a weighted update to θ at the server rate; FedBuff
holds updates until K have arrived and says so, so the service can still send
the waiting mule a DOWN.

Staleness must shrink the step (the cluster divides by the live partials'
staleness-free mass), a fold whose every partial expired must take no step and
keep the round open, and ``last_outcome`` must tell the service truthfully
which uploads need a DOWN now.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hermes.cluster import DeviceRegistry, HFLHostCluster
from hermes.cluster.cross_mule_fedavg import cross_mule_fedavg
from hermes.cluster.host_cluster import (
    OUTCOME_DEFERRED,
    OUTCOME_EXPIRED,
    OUTCOME_MERGED,
    OUTCOME_QUORUM,
    StubGeneratorHost,
)
from hermes.mission.aggregation_rules import (
    AGG_ASYNCHFL,
    AGG_CUTOFF,
    AGG_FEDBUFF,
    AGG_FEDEX,
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


def _cluster(
    spec: AggregationSpec, n_devices: int = 4, *, min_participation: int = 1,
) -> HFLHostCluster:
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
        min_participation=min_participation,
    )


def _up(mule, spec, *, shift, base_version, n_updates=1, mission_round=1, losses=None):
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
            local_loss=None if losses is None else losses[k],
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
        "partials": [["m1", 1]],
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
    # With no opening mule named, K falls back to every registered device.
    c = _cluster(AggregationSpec(rule=AGG_FEDBUFF), n_devices=5)
    assert c._fedbuff_k() == 5


# --------------------------------------------------------------------------- #
# Staleness shrinks the step at the cluster
# --------------------------------------------------------------------------- #

def _advance(c: HFLHostCluster, rounds: int) -> None:
    for _ in range(rounds):
        c.close_cluster_round()
    assert c.cluster_round == rounds


def _theta_plus(step):
    return [t.astype(np.float64) + s for t, s in zip(_theta0(), step)]


def test_two_partials_at_different_ages_step_by_their_staleness():
    spec = AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form="exponential", decay=0.5)
    c = _cluster(spec, min_participation=2)
    _advance(c, 5)
    up1 = _up(M1, spec, shift=1.0, base_version=3)                # age 2, M = 10
    up2 = _up(M2, spec, shift=2.0, base_version=5, n_updates=2)   # age 0, M = 20
    c.ingest_up_bundle(up1)
    c.ingest_up_bundle(up2)
    merged = c.aggregate_pending()
    d1, d2 = up1.partial_aggregate.weights, up2.partial_aggregate.weights
    m1, m2 = 10.0, 20.0
    step = [(m1 * math.exp(-1.0) * a.astype(np.float64) + m2 * b.astype(np.float64)) / (m1 + m2)
            for a, b in zip(d1, d2)]
    for got, exp in zip(merged, _theta_plus(step)):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)
    assert c.last_outcome == OUTCOME_MERGED
    assert c.last_merge["partial_ages"] == [2, 0]
    assert c.last_merge["partial_weights"] == pytest.approx([m1 * math.exp(-1.0), m2])
    assert c.last_merge["partials"] == [["m1", 1], ["m2", 1]]


def test_a_partial_past_a_max_adds_nothing_and_does_not_dilute():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=1)
    c = _cluster(spec, min_participation=2)
    _advance(c, 5)
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=3))              # age 2 > 1
    up2 = _up(M2, spec, shift=2.0, base_version=5, n_updates=2)
    c.ingest_up_bundle(up2)
    merged = c.aggregate_pending()
    # the denominator is M2 alone, so θ moves by the whole of Δ2
    for got, exp in zip(merged, _theta_plus(up2.partial_aggregate.weights)):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)
    assert c.last_outcome == OUTCOME_MERGED
    assert c.last_merge["partial_ages"] == [2, 0]
    assert c.last_merge["partial_weights"] == [0.0, 20.0]
    # Only the live partial is reported as merged; the cut one is listed apart
    # so the trace scorer does not credit its update.
    assert c.last_merge["partials"] == [["m2", 1]]
    assert c.last_merge["expired_partials"] == [["m1", 1]]
    assert c.last_merge["n_updates"] == 2


def test_a_lone_expired_partial_takes_no_step_and_keeps_the_round_open():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=1)
    c = _cluster(spec)
    _advance(c, 2)
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=0))   # age 2 > a_max
    assert c.aggregate_pending() is None                            # no exception
    assert c.last_outcome == OUTCOME_EXPIRED
    assert c.last_merge == {
        "rule": AGG_CUTOFF, "applied": False, "outcome": OUTCOME_EXPIRED,
        "partial_ages": [2], "partial_weights": [0.0], "partials": [["m1", 1]],
    }
    assert c.cluster_round == 2 and c.pending_partials() == 0
    for got, t in zip(c.generator.get_global_disc_weights(), _theta0()):
        assert np.array_equal(got, t)
    # the same mule's fresh UP is accepted (not a duplicate) and merges
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=2, mission_round=2))
    assert c.aggregate_pending() is not None and c.last_outcome == OUTCOME_MERGED


def test_a_single_stale_partial_mixes_in_at_its_staleness():
    """One partial: θ + η·e^{-λ·age}·Δ, the FedAsync / Async-HFL mixing form."""
    spec = AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form="exponential", decay=0.5,
                           server_lr=0.5)
    c = _cluster(spec)
    _advance(c, 2)
    up = _up(M1, spec, shift=1.0, base_version=0)
    c.ingest_up_bundle(up)
    merged = c.aggregate_pending()
    step = [0.5 * math.exp(-1.0) * d.astype(np.float64) for d in up.partial_aggregate.weights]
    for got, exp in zip(merged, _theta_plus(step)):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)


def test_every_partial_current_keeps_the_flat_weighted_mean():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    c = _cluster(spec, min_participation=2)
    up1 = _up(M1, spec, shift=1.0, base_version=0)
    up2 = _up(M2, spec, shift=3.0, base_version=0, n_updates=3)
    c.ingest_up_bundle(up1)
    c.ingest_up_bundle(up2)
    merged = c.aggregate_pending()
    step = [(10 * a.astype(np.float64) + 30 * b.astype(np.float64)) / 40.0
            for a, b in zip(up1.partial_aggregate.weights, up2.partial_aggregate.weights)]
    for got, exp in zip(merged, _theta_plus(step)):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)


def test_loss_value_weighs_the_mules_by_their_devices_losses():
    spec = AggregationSpec(rule=AGG_CUTOFF, value="loss")
    c = _cluster(spec, min_participation=2)
    up1 = _up(M1, spec, shift=1.0, base_version=0, n_updates=2, losses=(0.2, 0.2))
    up2 = _up(M2, spec, shift=2.0, base_version=0, n_updates=2, losses=(0.6, 0.6))
    c.ingest_up_bundle(up1)
    c.ingest_up_bundle(up2)
    merged = c.aggregate_pending()
    w1, w2 = c.last_merge["partial_weights"]
    assert (w1, w2) == pytest.approx((4.0, 12.0)) and w2 / w1 == pytest.approx(3.0)
    step = [(a.astype(np.float64) + 3 * b.astype(np.float64)) / 4.0
            for a, b in zip(up1.partial_aggregate.weights, up2.partial_aggregate.weights)]
    for got, exp in zip(merged, _theta_plus(step)):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# Truthful outcomes, FedBuff quorum and K
# --------------------------------------------------------------------------- #

def test_fedbuff_ignores_min_participation_and_buffers_the_first_up():
    spec = AggregationSpec(rule=AGG_FEDBUFF, buffer_k=2)
    c = _cluster(spec, min_participation=2)
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=0))
    assert c.aggregate_pending() is None
    assert c.last_outcome == OUTCOME_DEFERRED
    assert c.last_merge["buffered"] == 1 and c.last_merge["partials"] == [["m1", 1]]
    assert c.pending_partials() == 0 and c._pending.seen_mules == []
    # K is the quorum: the same mule's next UP is accepted and completes it
    c.ingest_up_bundle(_up(M1, spec, shift=3.0, base_version=0, mission_round=2))
    assert c.pending_partials() == 1
    merged = c.aggregate_pending()
    assert merged is not None and c.last_outcome == OUTCOME_MERGED
    assert c.last_merge["applied"] and c.last_merge["partials"] == [["m1", 1], ["m1", 2]]
    for got, t in zip(merged, _theta0()):
        np.testing.assert_allclose(got, t + 2.0, rtol=1e-6)


def test_last_merge_and_last_outcome_reset_on_every_call():
    spec = AggregationSpec(rule=AGG_CUTOFF)
    c = _cluster(spec, min_participation=2)
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=0))
    c.ingest_up_bundle(_up(M2, spec, shift=1.0, base_version=0))
    assert c.aggregate_pending() is not None
    assert c.last_merge is not None and c.last_outcome == OUTCOME_MERGED
    c.close_cluster_round()
    assert c.aggregate_pending() is None                  # nothing pending
    assert c.last_merge is None and c.last_outcome is None
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=1, mission_round=2))
    assert c.aggregate_pending() is None                  # 1 < min_participation
    assert c.last_merge is None and c.last_outcome == OUTCOME_QUORUM


def test_fedbuff_k_defaults_to_the_opening_mules_slice():
    # 5 devices round-robin over m1, m2: slices of 3 and 2
    c = _cluster(AggregationSpec(rule=AGG_FEDBUFF), n_devices=5)
    assert c._fedbuff_k(M1) == 3 and c._fedbuff_k(M2) == 2
    assert c._fedbuff_k(MuleID("ghost")) == 5            # empty slice: all devices
    spec = c.aggregation
    c.ingest_up_bundle(_up(M2, spec, shift=1.0, base_version=0))
    assert c.aggregate_pending() is None and c.last_merge["k"] == 2
    c.ingest_up_bundle(_up(M1, spec, shift=1.0, base_version=0))
    assert c.aggregate_pending() is not None             # K = 2 reached, not 5
    assert c.last_merge["k"] == 2


# --------------------------------------------------------------------------- #
# The cluster service: who gets a DOWN, and what the merge events carry
# --------------------------------------------------------------------------- #

class _ScriptedDock:
    """Stands in for ``TCPDockLinkServer`` so ``ClusterService.run`` can be
    driven synchronously: ``recv_up`` hands out scripted UPs, then stops the
    service. Each DOWN is recorded with the number of UPs handed out so far.
    ``lost`` maps a mule to the UP count from which a DOWN to it fails, as a
    dropped socket would."""

    def __init__(self, svc, ups, mules, lost=None):
        self._svc, self._ups, self._mules = svc, list(ups), list(mules)
        self._lost = dict(lost or {})
        self.n_up = 0
        self.downs = []

    def registered_mules(self):
        return list(self._mules)

    def wait_for_mules(self, mules, timeout=None):
        return True

    def recv_up(self, timeout=None):
        if not self._ups:
            self._svc.request_stop()
            raise TimeoutError("script exhausted")
        self.n_up += 1
        return self._ups.pop(0)

    def send_down(self, bundle):
        mule = str(bundle.mule_id)
        if mule in self._lost and self.n_up >= self._lost[mule]:
            raise ConnectionError(f"{mule} dropped off the dock")
        self.downs.append((self.n_up, mule))

    def downs_after(self, n_up):
        return sorted(m for n, m in self.downs if n == n_up)

    def close(self):
        return


class _Events:
    def __init__(self):
        self.lines = []

    def emit(self, event, **fields):
        self.lines.append((event, fields))

    def close(self):
        return

    def named(self, event):
        return [f for e, f in self.lines if e == event]


def _run_service(
    spec: AggregationSpec, ups, *, min_participation: int, lost=None,
    ingest_failures: int = 0,
):
    from hermes.processes.cluster import ClusterService
    from hermes.processes.config import ClusterConfig

    cfg = ClusterConfig(
        cluster_id="cluster-merge-outcomes",
        dock_host="127.0.0.1",
        dock_port=0,
        expected_mules=["m1", "m2"],
        seed_devices=[{"device_id": f"d{i}", "position": [float(i), 0.0, 0.0]} for i in range(4)],
        synth_batch_size=2,
        min_participation=min_participation,
        aggregation=spec.rule,
        aggregation_params=spec.to_params(),
    )
    events = _Events()
    svc = ClusterService(cfg, events=events)
    svc.dock.close()   # the real TCP server is not needed
    dock = _ScriptedDock(svc, ups, ["m1", "m2"], lost=lost)
    svc.dock = dock
    try:
        svc.run()
    finally:
        svc.shutdown()
    assert svc.metrics.counter_value("ingest_failures") == ingest_failures
    return svc, dock, events


def test_service_sends_down_on_expired_but_not_on_quorum():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=0)
    ups = [
        _up(M1, spec, shift=1.0, base_version=0, mission_round=1),   # quorum
        _up(M2, spec, shift=1.0, base_version=0, mission_round=1),   # merged, V -> 1
        _up(M1, spec, shift=1.0, base_version=0, mission_round=2),   # quorum
        _up(M2, spec, shift=1.0, base_version=0, mission_round=2),   # both age 1: expired
    ]
    svc, dock, events = _run_service(spec, ups, min_participation=2)
    assert dock.downs_after(0) == ["m1", "m2"]          # bootstrap
    assert dock.downs_after(1) == [] and dock.downs_after(3) == []   # quorum: no DOWN
    assert dock.downs_after(2) == ["m1", "m2"]          # a merge sends every mule one
    # expired: m1's partial was dropped too, and m1 has waited for a DOWN since
    # UP #3, so it is released with the uploader rather than left to time out
    assert dock.downs_after(4) == ["m1", "m2"]
    assert svc.metrics.counter_value("down_bundles_dispatched") == 4
    (merge,) = events.named("cluster_merge")
    assert merge["mission_round"] == 1 and merge["partials"] == [["m1", 1], ["m2", 1]]
    (expired,) = events.named("cluster_merge_expired")
    assert expired["mule_id"] == "m2" and expired["mission_round"] == 2
    assert expired["outcome"] == OUTCOME_EXPIRED and expired["partial_ages"] == [1, 1]
    assert events.named("cluster_merge_deferred") == []
    assert len(events.named("cluster_round_closed")) == 1
    assert svc.cluster.cluster_round == 1               # the expired fold closed nothing


def _expiring_ups(spec):
    """m1 then m2 merge at V=0; their next partials, both age 1, expire."""
    return [
        _up(M1, spec, shift=1.0, base_version=0, mission_round=1),
        _up(M2, spec, shift=1.0, base_version=0, mission_round=1),
        _up(M1, spec, shift=1.0, base_version=0, mission_round=2),
        _up(M2, spec, shift=1.0, base_version=0, mission_round=2),
    ]


def test_expired_release_skips_a_member_that_dropped_off_the_dock():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=0)
    svc, dock, events = _run_service(
        spec, _expiring_ups(spec), min_participation=2, lost={"m1": 3},
    )
    assert dock.downs_after(4) == ["m2"]                # the uploader is still released
    assert svc.metrics.counter_value("dispatch_down_failures") == 1
    assert len(events.named("cluster_merge_expired")) == 1


def test_expired_release_survives_a_lost_uploader():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=0)
    svc, dock, events = _run_service(
        spec, _expiring_ups(spec), min_participation=2, lost={"m2": 4},
        ingest_failures=1,                              # the uploader's send, as before
    )
    assert dock.downs_after(4) == ["m1"]                # m1 is not stranded with it
    assert len(events.named("cluster_merge_expired")) == 1


def test_service_sends_down_on_deferred_and_tags_merges_with_the_mission():
    spec = AggregationSpec(rule=AGG_FEDBUFF, buffer_k=2)
    ups = [
        _up(M1, spec, shift=1.0, base_version=0, mission_round=1),   # deferred
        _up(M1, spec, shift=3.0, base_version=0, mission_round=2),   # K reached
    ]
    svc, dock, events = _run_service(spec, ups, min_participation=2)
    assert dock.downs_after(1) == ["m1"]
    # The flush answers the mule waiting at the dock, m1, and nobody else: m2
    # has not uploaded, so a DOWN now would sit in its queue and be read as
    # the answer to its next upload (FeRRy Phase 2; this used to go to every
    # docked mule).
    assert dock.downs_after(2) == ["m1"]
    (deferred,) = events.named("cluster_merge_deferred")
    assert deferred["mule_id"] == "m1" and deferred["mission_round"] == 1
    assert deferred["partials"] == [["m1", 1]] and deferred["buffered"] == 1
    (merge,) = events.named("cluster_merge")
    assert merge["mission_round"] == 2 and merge["partials"] == [["m1", 1], ["m1", 2]]
    assert svc.cluster.cluster_round == 1


# --------------------------------------------------------------------------- #
# agg:fedex — θ ← θ + (1/N)·Σ Δθ on each return (arm D4)
# --------------------------------------------------------------------------- #

def test_fedex_applies_each_return_at_one_over_the_client_count():
    spec = AggregationSpec(rule=AGG_FEDEX)
    c = _cluster(spec, n_devices=4)
    _advance(c, 3)
    up = _up(M1, spec, shift=2.0, base_version=0, n_updates=2)   # age 3: no discount
    c.ingest_up_bundle(up)
    merged = c.aggregate_pending()
    # The partial is the SUM of two updates (≈ 2·2.0); N = 4 registered devices.
    for got, exp in zip(merged, _theta_plus([w / 4.0 for w in up.partial_aggregate.weights])):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)
    assert c.last_outcome == OUTCOME_MERGED
    assert c.last_merge["n_clients"] == 4 and c.last_merge["n_updates"] == 2
    assert c.last_merge["partial_ages"] == [3]


def test_fedex_n_can_be_set():
    spec = AggregationSpec(rule=AGG_FEDEX, fedex_n=10)
    c = _cluster(spec, n_devices=4)
    up = _up(M1, spec, shift=1.0, base_version=0)
    c.ingest_up_bundle(up)
    merged = c.aggregate_pending()
    for got, exp in zip(merged, _theta_plus([w / 10.0 for w in up.partial_aggregate.weights])):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)


def test_fedex_lists_an_empty_partial_apart_from_what_it_merged():
    """Under a quorum of 2, M2's upload was lost and the cluster holds an empty
    partial in its place. The fold steps by M1's sum alone and names only M1
    as merged, like the weighted rules, so the trace does not credit M2."""
    from hermes.processes.cluster import _lost_upload_stand_in

    spec = AggregationSpec(rule=AGG_FEDEX)
    c = _cluster(spec, n_devices=4, min_participation=2)
    up = _up(M1, spec, shift=2.0, base_version=0)
    c.ingest_up_bundle(up)
    c.ingest_up_bundle(_lost_upload_stand_in(_up(M2, spec, shift=9.0, base_version=0), 1, spec))
    merged = c.aggregate_pending()
    for got, exp in zip(merged, _theta_plus([w / 4.0 for w in up.partial_aggregate.weights])):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-6)
    assert c.last_merge["partials"] == [["m1", 1]]
    assert c.last_merge["expired_partials"] == [["m2", 1]]
    assert c.last_merge["n_updates"] == 1
