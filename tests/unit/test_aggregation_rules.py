"""FeRRy Phase 1 — the L3 merge rules (``hermes/mission/aggregation_rules.py``).

Pins the three properties the build plan names for the weight formula: it
reduces to today's mean when every basis is current (the regression test),
it is exactly zero past the cutoff, and each rule's staleness function is the
one its source defines. Pure numpy; no links, no TensorFlow.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hermes.cluster.cross_mule_fedavg import (
    FedAvgError,
    apply_weighted_deltas,
    cross_mule_fedavg,
)
from hermes.mission.aggregation_rules import (
    AGG_ASYNCHFL,
    AGG_CUTOFF,
    AGG_FEDBUFF,
    AGG_FEDEX,
    AGG_PLAIN,
    AGG_SEQ,
    AggregationConfigError,
    AggregationSpec,
    FedBuffBuffer,
    age_cap,
    check_partial_form,
    merge_on_mule,
    partial_age,
    partial_staleness,
    staleness,
    update_age,
    update_weights,
)
from hermes.mission.partial_fedavg import (
    PartialFedAvgError,
    partial_fedavg,
    partial_fedavg_delta,
)
from hermes.types import DeviceID, GradientSubmission, MuleID
from hermes.types.fl_messages import UPDATE_FORM_DELTA, UPDATE_FORM_WEIGHTS

MULE = MuleID("mule-agg")
ROUND = 3


def _theta(seed: int = 0):
    rng = np.random.default_rng(seed)
    return [
        rng.normal(size=(4,)).astype(np.float32),
        rng.normal(size=(3, 3)).astype(np.float32),
    ]


def _local(theta, seed):
    rng = np.random.default_rng(1000 + seed)
    return [(w + rng.normal(0.0, 0.1, size=w.shape)).astype(w.dtype) for w in theta]


def _sub(did, weights, n, *, basis=None, form=UPDATE_FORM_WEIGHTS, loss=None):
    return GradientSubmission(
        device_id=DeviceID(did),
        mule_id=MULE,
        mission_round=ROUND,
        delta_theta=weights,
        num_examples=n,
        submitted_at=0.0,
        local_loss=loss,
        basis_version=basis,
        update_form=form,
    )


def _delta_sub(did, local, basis_theta, n, *, basis=None, loss=None):
    delta = [a - b for a, b in zip(local, basis_theta)]
    return _sub(did, delta, n, basis=basis, form=UPDATE_FORM_DELTA, loss=loss)


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #

def test_default_spec_is_plain_and_asks_for_full_weights():
    spec = AggregationSpec()
    assert spec.rule == AGG_PLAIN and spec.is_plain
    assert spec.update_form == UPDATE_FORM_WEIGHTS
    assert AggregationSpec(rule=AGG_CUTOFF).update_form == UPDATE_FORM_DELTA


@pytest.mark.parametrize("rule", [AGG_SEQ])
def test_planned_rules_are_refused_with_the_reason(rule):
    with pytest.raises(AggregationConfigError, match="not implemented yet"):
        AggregationSpec(rule=rule)


def test_unknown_rule_and_bad_parameters_are_refused():
    with pytest.raises(AggregationConfigError, match="unknown aggregation rule"):
        AggregationSpec(rule="agg:nope")
    for bad in (
        dict(server_lr=0.0), dict(a_max=-1), dict(period_s=0.0),
        dict(hinge_a=-1.0), dict(decay=-0.1), dict(value="gradient"),
        dict(poly_q=-0.5), dict(asynchfl_form="hinge"),
        dict(buffer_k=0),
    ):
        with pytest.raises(AggregationConfigError):
            AggregationSpec(rule=AGG_CUTOFF, **bad)


def test_from_config_round_trips_and_rejects_unknown_keys():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=2, period_s=12.5, server_lr=0.5)
    again = AggregationSpec.from_config(spec.rule, spec.to_params())
    assert again == spec
    assert AggregationSpec.from_config(None, {}) == AggregationSpec()
    with pytest.raises(AggregationConfigError, match="unknown aggregation parameter"):
        AggregationSpec.from_config(AGG_CUTOFF, {"eta": 0.5})


# --------------------------------------------------------------------------- #
# Staleness, age and cutoff
# --------------------------------------------------------------------------- #

def test_hinge_matches_fedasync():
    spec = AggregationSpec(rule=AGG_CUTOFF, hinge_a=2.0, hinge_b=1.0)
    assert staleness(spec, 0) == 1.0
    assert staleness(spec, 1) == 1.0
    assert staleness(spec, 2) == pytest.approx(1.0 / (2.0 * 1 + 1.0))
    assert staleness(spec, 4) == pytest.approx(1.0 / (2.0 * 3 + 1.0))
    assert staleness(spec, -3) == 1.0  # negative ages count as 0


def test_asynchfl_is_async_hfls_polynomial_by_default():
    """Async-HFL's s(a) = (a + 1)^-q, adopted from FedAsync (q = 0.5 by default);
    the exponential is available by name, and every recorded parameter dict
    (which holds ``decay`` whatever the rule) still builds."""
    spec = AggregationSpec(rule=AGG_ASYNCHFL)
    assert (spec.asynchfl_form, spec.poly_q) == ("polynomial", 0.5)
    assert staleness(spec, 0) == 1.0
    assert staleness(spec, 3) == pytest.approx(4.0 ** -0.5)
    assert staleness(AggregationSpec(rule=AGG_ASYNCHFL, poly_q=1.0), 4) == pytest.approx(0.2)
    assert staleness(AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form="exponential"),
                     2) == pytest.approx(math.exp(-1.0))
    old = {"server_lr": 1.0, "a_max": None, "period_s": None, "hinge_a": 1.0,
           "hinge_b": 0.0, "decay": 0.5, "value": "uniform", "buffer_k": None,
           "fedex_n": None}
    assert AggregationSpec.from_config(AGG_CUTOFF, old) == AggregationSpec(rule=AGG_CUTOFF)
    # A recorded agg:asynchfl dict names decay and no form: it flew the exponential.
    assert AggregationSpec.from_config(AGG_ASYNCHFL, old).asynchfl_form == "exponential"
    assert AggregationSpec.from_config(AGG_ASYNCHFL, {}).asynchfl_form == "polynomial"
    assert AggregationSpec.from_config(
        AGG_ASYNCHFL, {"decay": 0.3, "asynchfl_form": "polynomial"}).asynchfl_form == "polynomial"


def test_only_asynchfl_writes_its_form_and_exponent():
    """Every other rule's aggregation_params is the recorded dict (no new key), so
    recorded rows and traces read as before; agg:asynchfl's round-trips."""
    old_keys = {"server_lr", "a_max", "period_s", "hinge_a", "hinge_b", "decay", "value",
                "buffer_k", "fedex_n"}
    for rule in (AGG_CUTOFF, AGG_FEDBUFF):
        assert set(AggregationSpec(rule=rule).to_params()) == old_keys
    for form in ("polynomial", "exponential"):
        spec = AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form=form, poly_q=0.7)
        params = spec.to_params()
        assert set(params) == old_keys | {"asynchfl_form", "poly_q"}
        assert AggregationSpec.from_config(AGG_ASYNCHFL, params) == spec


def test_other_rules_staleness():
    exponential = AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form="exponential", decay=0.5)
    assert staleness(exponential, 2) == pytest.approx(math.exp(-1.0))
    assert staleness(AggregationSpec(rule=AGG_FEDBUFF), 3) == pytest.approx(0.5)
    assert staleness(AggregationSpec(), 7) == 1.0


def test_age_cap_converts_the_deadline_window_with_the_mission_period():
    spec = AggregationSpec(rule=AGG_CUTOFF, period_s=20.0)
    assert age_cap(spec, 60.0) == 3
    assert age_cap(spec, 59.9) == 2          # floor, not round
    assert age_cap(spec, 5.0) == 0
    both = AggregationSpec(rule=AGG_CUTOFF, period_s=20.0, a_max=1)
    assert age_cap(both, 60.0) == 1          # the smaller cap wins
    assert age_cap(AggregationSpec(rule=AGG_CUTOFF, a_max=2), None) == 2
    assert age_cap(AggregationSpec(rule=AGG_CUTOFF), 60.0) is None
    assert age_cap(AggregationSpec(rule=AGG_ASYNCHFL, a_max=1), 60.0) is None


def test_update_age():
    assert update_age(5, 3) == 2
    assert update_age(5, 5) == 0
    assert update_age(5, 7) == 0              # newer basis counts as current
    assert update_age(None, 3) is None
    assert update_age(5, None) is None


def test_weight_is_exactly_zero_past_the_cutoff():
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=1)
    th = _theta()
    subs = [
        _sub("a", th, 10, basis=5, form=UPDATE_FORM_DELTA),   # age 0
        _sub("b", th, 10, basis=4, form=UPDATE_FORM_DELTA),   # age 1 = cap
        _sub("c", th, 10, basis=3, form=UPDATE_FORM_DELTA),   # age 2 > cap
    ]
    caps = {DeviceID(d): 1 for d in "abc"}
    ws = update_weights(spec, subs, base_version=5, age_caps=caps)
    assert [w.age for w in ws] == [0, 1, 2]
    assert ws[0].weight == 10.0
    assert ws[1].weight == pytest.approx(10.0 * staleness(spec, 1))
    assert ws[2].weight == 0.0 and not ws[2].admitted


def test_fedbuff_weights_ignore_example_counts():
    spec = AggregationSpec(rule=AGG_FEDBUFF)
    th = _theta()
    ws = update_weights(
        spec,
        [_sub("a", th, 5, basis=2), _sub("b", th, 500, basis=0)],
        base_version=2,
    )
    assert ws[0].weight == 1.0
    assert ws[1].weight == pytest.approx(1.0 / math.sqrt(3.0))


def test_loss_value_proxy_is_the_raw_loss():
    spec = AggregationSpec(rule=AGG_CUTOFF, value="loss")
    th = _theta()
    subs = [_sub("a", th, 10, basis=1, loss=0.2), _sub("b", th, 10, basis=1, loss=0.6)]
    ws = update_weights(spec, subs, base_version=1)
    # v_i is the raw loss, not loss over the mean: mass = n_i * loss_i
    assert [w.mass for w in ws] == pytest.approx([2.0, 6.0])
    assert [w.weight for w in ws] == pytest.approx([2.0, 6.0])


# --------------------------------------------------------------------------- #
# Mule-side merge
# --------------------------------------------------------------------------- #

def test_plain_merge_is_partial_fedavg_unchanged_plus_ages():
    th = _theta()
    subs = [
        _sub("a", _local(th, 1), 10, basis=4),
        _sub("b", _local(th, 2), 30, basis=3),
    ]
    ref = partial_fedavg(MULE, ROUND, subs)
    got = merge_on_mule(
        AggregationSpec(), mule_id=MULE, mission_round=ROUND,
        submissions=subs, base_version=4,
    )
    for a, b in zip(got.weights, ref.weights):
        assert np.array_equal(a, b)
    assert got.num_examples == ref.num_examples == 40
    assert got.update_form == UPDATE_FORM_WEIGHTS and got.rule == AGG_PLAIN
    assert got.base_version == 4
    assert got.device_basis_versions == (4, 3)
    assert got.device_ages == (0, 1)


@pytest.mark.parametrize("rule", [AGG_CUTOFF, AGG_ASYNCHFL])
def test_every_basis_current_reduces_to_todays_mean(rule):
    """The regression test: θ_v + Σ (n_i/Σn)(θ_i − θ_v) = Σ (n_i/Σn)·θ_i."""
    theta_v = _theta(7)
    locals_ = [_local(theta_v, s) for s in range(3)]
    ns = [12, 30, 5]
    plain = merge_on_mule(
        AggregationSpec(), mule_id=MULE, mission_round=ROUND,
        submissions=[_sub(f"d{i}", w, n, basis=9) for i, (w, n) in enumerate(zip(locals_, ns))],
        base_version=9,
    )
    today = cross_mule_fedavg([plain])

    spec = AggregationSpec(rule=rule)
    partial = merge_on_mule(
        spec, mule_id=MULE, mission_round=ROUND,
        submissions=[
            _delta_sub(f"d{i}", w, theta_v, n, basis=9)
            for i, (w, n) in enumerate(zip(locals_, ns))
        ],
        base_version=9,
    )
    assert partial.update_form == UPDATE_FORM_DELTA and partial.rule == rule
    assert partial.device_ages == (0, 0, 0)
    assert partial.weight_mass == pytest.approx(sum(ns))
    merged = apply_weighted_deltas(theta_v, [partial], [partial.weight_mass], server_lr=1.0)
    for a, b in zip(merged, today):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_stale_update_is_down_weighted_and_expired_one_excluded():
    theta_v = _theta(3)
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=2, hinge_a=1.0, hinge_b=0.0)
    subs = [
        _delta_sub("fresh", _local(theta_v, 1), theta_v, 10, basis=6),
        _delta_sub("stale", _local(theta_v, 2), theta_v, 10, basis=5),
        _delta_sub("expired", _local(theta_v, 3), theta_v, 10, basis=2),
    ]
    caps = {DeviceID(d): 2 for d in ("fresh", "stale", "expired")}
    p = merge_on_mule(
        spec, mule_id=MULE, mission_round=ROUND, submissions=subs,
        base_version=6, age_caps=caps,
    )
    assert p.contributing_devices == (DeviceID("fresh"), DeviceID("stale"))
    assert p.excluded_devices == (DeviceID("expired"),)
    assert p.device_ages == (0, 1)
    # raw weights 10 and 10*1/2 over the staleness-free mass 10 + 10 (the
    # expired update adds nothing) -> 1/2 and 1/4, summing to the mean staleness
    assert p.device_weights == pytest.approx((1 / 2, 1 / 4))
    assert p.weight_mass == pytest.approx(20.0)
    expected = [
        (1 / 2) * a.astype(np.float64) + (1 / 4) * b.astype(np.float64)
        for a, b in zip(subs[0].delta_theta, subs[1].delta_theta)
    ]
    for got, exp in zip(p.weights, expected):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-7)


def test_all_updates_past_the_cutoff_raise_instead_of_dividing_by_zero():
    theta_v = _theta()
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=0)
    subs = [_delta_sub("a", _local(theta_v, 1), theta_v, 10, basis=1)]
    with pytest.raises(PartialFedAvgError, match="past their age cutoff"):
        merge_on_mule(
            spec, mule_id=MULE, mission_round=ROUND, submissions=subs,
            base_version=2, age_caps={DeviceID("a"): 0},
        )


def test_delta_merge_refuses_full_weights_and_zero_mass():
    th = _theta()
    with pytest.raises(PartialFedAvgError, match="delta merge needs"):
        partial_fedavg_delta(MULE, ROUND, [_sub("a", th, 3)], weights=[1.0], normalizer=1.0)
    sub = _sub("a", th, 3, form=UPDATE_FORM_DELTA)
    with pytest.raises(PartialFedAvgError, match="sum to"):
        partial_fedavg_delta(MULE, ROUND, [sub], weights=[0.0], normalizer=0.0)


def test_fedbuff_partial_averages_over_the_update_count():
    theta_v = _theta()
    spec = AggregationSpec(rule=AGG_FEDBUFF)
    subs = [
        _delta_sub("a", _local(theta_v, 1), theta_v, 10, basis=4),   # s = 1
        _delta_sub("b", _local(theta_v, 2), theta_v, 90, basis=1),   # age 3, s = 1/2
    ]
    p = merge_on_mule(spec, mule_id=MULE, mission_round=ROUND, submissions=subs, base_version=4)
    assert p.weight_mass == 2.0 and p.n_updates == 2
    expected = [
        (a.astype(np.float64) + 0.5 * b.astype(np.float64)) / 2.0
        for a, b in zip(subs[0].delta_theta, subs[1].delta_theta)
    ]
    for got, exp in zip(p.weights, expected):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-7)


def test_update_mass_is_staleness_free():
    """mass = n_i·v_i whatever the age; weight = mass·s(a_i), 0 past the cap."""
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=1, hinge_a=1.0, hinge_b=0.0)
    th = _theta()
    subs = [
        _sub("a", th, 10, basis=5, form=UPDATE_FORM_DELTA),   # age 0
        _sub("b", th, 30, basis=4, form=UPDATE_FORM_DELTA),   # age 1, s = 1/2
        _sub("c", th, 20, basis=3, form=UPDATE_FORM_DELTA),   # age 2 > cap
    ]
    ws = update_weights(spec, subs, base_version=5, age_caps={DeviceID(d): 1 for d in "abc"})
    assert [w.mass for w in ws] == [10.0, 30.0, 20.0]
    assert [w.weight for w in ws] == pytest.approx([10.0, 15.0, 0.0])
    fb = update_weights(AggregationSpec(rule=AGG_FEDBUFF), subs, base_version=5)
    assert [w.mass for w in fb] == [1.0, 1.0, 1.0]


def test_a_uniformly_stale_mission_shrinks_the_step():
    """Both updates of age 3 under asynchfl: e^{-1.5} times their mean, not
    the full mean (dividing by Σ w_i would cancel the common factor)."""
    theta_v = _theta(5)
    spec = AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form="exponential", decay=0.5)
    subs = [
        _delta_sub("a", _local(theta_v, 1), theta_v, 10, basis=2),
        _delta_sub("b", _local(theta_v, 2), theta_v, 30, basis=2),
    ]
    p = merge_on_mule(spec, mule_id=MULE, mission_round=ROUND, submissions=subs, base_version=5)
    s = math.exp(-1.5)
    assert p.device_ages == (3, 3)
    assert p.weight_mass == pytest.approx(40.0)
    assert sum(p.device_weights) == pytest.approx(s)
    for got, a, b in zip(p.weights, subs[0].delta_theta, subs[1].delta_theta):
        mean = (10 * a.astype(np.float64) + 30 * b.astype(np.float64)) / 40.0
        np.testing.assert_allclose(got, s * mean, rtol=1e-6, atol=1e-7)


def test_a_mixed_age_mission_divides_by_the_staleness_free_mass():
    theta_v = _theta(6)
    spec = AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form="exponential", decay=0.5)
    subs = [
        _delta_sub("fresh", _local(theta_v, 1), theta_v, 10, basis=4),   # age 0
        _delta_sub("old", _local(theta_v, 2), theta_v, 30, basis=2),     # age 2
    ]
    p = merge_on_mule(spec, mule_id=MULE, mission_round=ROUND, submissions=subs, base_version=4)
    e1 = math.exp(-1.0)
    assert p.weight_mass == pytest.approx(40.0)
    assert p.device_weights == pytest.approx((10 / 40, 30 * e1 / 40))
    for got, a, b in zip(p.weights, subs[0].delta_theta, subs[1].delta_theta):
        exp = (10 * a.astype(np.float64) + 30 * e1 * b.astype(np.float64)) / 40.0
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-7)


def test_an_excluded_update_does_not_move_the_admitted_weights():
    """v_i is formed after the cutoff: an expired high-loss update must not
    shift the fallback for a missing loss, nor anyone's weight or mass."""
    theta_v = _theta(8)
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=1, value="loss")
    a = _delta_sub("a", _local(theta_v, 1), theta_v, 10, basis=4, loss=0.2)
    b = _delta_sub("b", _local(theta_v, 2), theta_v, 10, basis=4, loss=None)
    expired = _delta_sub("x", _local(theta_v, 3), theta_v, 10, basis=1, loss=5.0)
    caps = {DeviceID(d): 1 for d in ("a", "b", "x")}
    with_x = update_weights(spec, [a, b, expired], base_version=4, age_caps=caps)
    without = update_weights(spec, [a, b], base_version=4, age_caps=caps)
    assert not with_x[2].admitted
    # b's missing loss takes the admitted mean, 0.2, so both masses are 10 * 0.2
    assert [w.mass for w in with_x[:2]] == pytest.approx([2.0, 2.0])
    assert [(w.weight, w.mass) for w in with_x[:2]] == [(w.weight, w.mass) for w in without]
    p_x = merge_on_mule(spec, mule_id=MULE, mission_round=ROUND,
                        submissions=[a, b, expired], base_version=4, age_caps=caps)
    p = merge_on_mule(spec, mule_id=MULE, mission_round=ROUND,
                      submissions=[a, b], base_version=4, age_caps=caps)
    assert p_x.excluded_devices == (DeviceID("x"),)
    assert p_x.weight_mass == p.weight_mass == pytest.approx(4.0)
    assert p_x.device_weights == p.device_weights
    for got, ref in zip(p_x.weights, p.weights):
        assert np.array_equal(got, ref)


def test_loss_partials_from_two_mules_combine_as_one_flat_merge():
    """M_m = Σ n_i·loss_i, so folding two mules' partials equals merging all
    their updates at once; a per-mule loss mean would weigh them 1:1."""
    theta_v = _theta(9)
    spec = AggregationSpec(rule=AGG_CUTOFF, value="loss")
    subs1 = [_delta_sub(f"a{k}", _local(theta_v, k), theta_v, 10, basis=2, loss=0.2)
             for k in (1, 2)]
    subs2 = [_delta_sub(f"b{k}", _local(theta_v, k + 2), theta_v, 10, basis=2, loss=0.6)
             for k in (1, 2)]
    p1 = merge_on_mule(spec, mule_id=MuleID("m1"), mission_round=ROUND,
                       submissions=subs1, base_version=2)
    p2 = merge_on_mule(spec, mule_id=MuleID("m2"), mission_round=ROUND,
                       submissions=subs2, base_version=2)
    assert (p1.weight_mass, p2.weight_mass) == pytest.approx((4.0, 12.0))   # 1:3
    two_level = apply_weighted_deltas(
        theta_v, [p1, p2], [p1.weight_mass, p2.weight_mass],
        normalizer=p1.weight_mass + p2.weight_mass,
    )
    flat = merge_on_mule(spec, mule_id=MULE, mission_round=ROUND,
                         submissions=subs1 + subs2, base_version=2)
    one_level = apply_weighted_deltas(
        theta_v, [flat], [flat.weight_mass], normalizer=flat.weight_mass,
    )
    for a, b in zip(two_level, one_level):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# Cluster-side fold
# --------------------------------------------------------------------------- #

def _delta_partial(theta_v, spec, *, base, seed, n=10):
    return merge_on_mule(
        spec, mule_id=MuleID(f"m{seed}"), mission_round=ROUND,
        submissions=[_delta_sub(f"x{seed}", _local(theta_v, seed), theta_v, n, basis=base)],
        base_version=base,
    )


def test_server_rate_mixes_the_update_in():
    theta = _theta(11)
    spec = AggregationSpec(rule=AGG_CUTOFF)
    p = _delta_partial(theta, spec, base=0, seed=1)
    half = apply_weighted_deltas(theta, [p], [1.0], server_lr=0.5)
    for t, d, h in zip(theta, p.weights, half):
        np.testing.assert_allclose(h, t + 0.5 * d, rtol=1e-6, atol=1e-7)


def test_zero_weight_partials_are_ignored_and_all_zero_raises():
    theta = _theta(12)
    spec = AggregationSpec(rule=AGG_CUTOFF)
    p1 = _delta_partial(theta, spec, base=0, seed=1)
    p2 = _delta_partial(theta, spec, base=0, seed=2)
    only_p1 = apply_weighted_deltas(theta, [p1, p2], [3.0, 0.0])
    ref = apply_weighted_deltas(theta, [p1], [3.0])
    for a, b in zip(only_p1, ref):
        assert np.array_equal(a, b)
    with pytest.raises(FedAvgError, match="no partial carries weight"):
        apply_weighted_deltas(theta, [p1, p2], [0.0, 0.0])


def test_an_explicit_normalizer_keeps_the_staleness_in_the_step():
    """W = M·s over N = M gives θ + η·s·Δ; None keeps dividing by Σ W."""
    theta = _theta(13)
    spec = AggregationSpec(rule=AGG_CUTOFF)
    p = _delta_partial(theta, spec, base=0, seed=1)
    m, s = p.weight_mass, 0.25
    shrunk = apply_weighted_deltas(theta, [p], [m * s], server_lr=0.5, normalizer=m)
    for t, d, got in zip(theta, p.weights, shrunk):
        np.testing.assert_allclose(got, t + 0.5 * s * d, rtol=1e-6, atol=1e-7)
    default = apply_weighted_deltas(theta, [p], [m * s], server_lr=0.5)
    explicit = apply_weighted_deltas(theta, [p], [m * s], server_lr=0.5, normalizer=m * s)
    for a, b in zip(default, explicit):
        assert np.array_equal(a, b)
    for bad in (0.0, -1.0):
        with pytest.raises(FedAvgError, match="normalizer must be > 0"):
            apply_weighted_deltas(theta, [p], [m], normalizer=bad)


def test_partial_staleness_and_cluster_cutoff():
    theta = _theta()
    spec = AggregationSpec(rule=AGG_CUTOFF, a_max=1)
    p = _delta_partial(theta, spec, base=3, seed=1)
    assert partial_age(p, 3) == 0
    assert partial_age(p, 5) == 2
    assert partial_staleness(spec, p, 4) == pytest.approx(0.5)
    assert partial_staleness(spec, p, 5) == 0.0          # past a_max at the cluster
    decay = AggregationSpec(rule=AGG_ASYNCHFL, asynchfl_form="exponential", decay=1.0)
    assert partial_staleness(decay, p, 5) == pytest.approx(math.exp(-2.0))


def test_check_partial_form_refuses_a_rule_mismatch():
    theta = _theta()
    p = _delta_partial(theta, AggregationSpec(rule=AGG_CUTOFF), base=0, seed=1)
    check_partial_form(AggregationSpec(rule=AGG_CUTOFF), p)
    with pytest.raises(AggregationConfigError, match="Set the same rule"):
        check_partial_form(AggregationSpec(), p)
    with pytest.raises(AggregationConfigError, match="Set the same rule"):
        check_partial_form(AggregationSpec(rule=AGG_ASYNCHFL), p)


def test_fedbuff_buffer_applies_the_mean_once_k_updates_arrive():
    theta = _theta(21)
    spec = AggregationSpec(rule=AGG_FEDBUFF, server_lr=1.0)
    buf = FedBuffBuffer(spec=spec, k=3)
    p1 = _delta_partial(theta, spec, base=0, seed=1)
    p2 = _delta_partial(theta, spec, base=0, seed=2)
    p3 = _delta_partial(theta, spec, base=0, seed=3)
    buf.add(p1, cluster_version=0)
    buf.add(p2, cluster_version=0)
    assert buf.count == 2 and not buf.ready
    buf.add(p3, cluster_version=0)
    assert buf.ready
    out = buf.apply(theta)
    for i, t in enumerate(theta):
        mean = (p1.weights[i].astype(np.float64) + p2.weights[i] + p3.weights[i]) / 3.0
        np.testing.assert_allclose(out[i], t + mean, rtol=1e-6, atol=1e-7)
    assert buf.count == 0 and not buf.ready


def test_fedbuff_buffer_remembers_its_members_until_a_flush():
    theta = _theta(22)
    spec = AggregationSpec(rule=AGG_FEDBUFF)
    buf = FedBuffBuffer(spec=spec, k=2)
    buf.add(_delta_partial(theta, spec, base=0, seed=1), cluster_version=0)
    assert buf.members == [(MuleID("m1"), ROUND)]
    buf.add(_delta_partial(theta, spec, base=0, seed=2), cluster_version=0)
    assert buf.members == [(MuleID("m1"), ROUND), (MuleID("m2"), ROUND)]
    buf.apply(theta)
    assert buf.members == []


# --------------------------------------------------------------------------- #
# agg:fedex — FedEx-Async's 1/N server step (arm D4, FeRRy Phase 2)
# --------------------------------------------------------------------------- #

def test_fedex_mule_sends_the_sum_of_its_updates_unweighted_by_n_or_age():
    theta = _theta(40)
    spec = AggregationSpec(rule=AGG_FEDEX)
    subs = [
        _delta_sub("a", _local(theta, 1), theta, 5, basis=3),     # age 2
        _delta_sub("b", _local(theta, 2), theta, 50, basis=5),    # age 0
    ]
    p = merge_on_mule(spec, mule_id=MULE, mission_round=ROUND, submissions=subs,
                      base_version=5, age_caps={DeviceID("a"): 0})
    expected = [a.astype(np.float64) + b for a, b in zip(subs[0].delta_theta, subs[1].delta_theta)]
    for got, exp in zip(p.weights, expected):
        np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-7)
    assert p.weight_mass == 2.0 and p.excluded_devices == ()   # no cutoff under FedEx
    assert p.device_weights == (0.5, 0.5)


def test_fedex_spec_validates_n():
    assert AggregationSpec(rule=AGG_FEDEX, fedex_n=8).fedex_n == 8
    with pytest.raises(AggregationConfigError):
        AggregationSpec(rule=AGG_FEDEX, fedex_n=0)
    assert staleness(AggregationSpec(rule=AGG_FEDEX), 7) == 1.0
