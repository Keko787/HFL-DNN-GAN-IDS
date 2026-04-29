"""Phase 5 — TargetSelectorRL scope guard + basic selection tests."""

from __future__ import annotations

import numpy as np
import pytest

from hermes.scheduler.selector import (
    DDQN,
    SelectorEnv,
    SelectorScopeViolation,
    TargetSelectorRL,
    assert_candidates_admitted,
)
from hermes.types import Bucket, DeviceID, DeviceSchedulerState, ServerID


def _states(*ids: str):
    return {
        DeviceID(i): DeviceSchedulerState(
            device_id=DeviceID(i),
            is_in_slice=True,
            last_known_position=(float(idx), 0.0, 0.0),
        )
        for idx, i in enumerate(ids)
    }


def _env() -> SelectorEnv:
    return SelectorEnv(
        mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0, rf_prior_snr_db=20.0, now=0.0
    )


# --------------------------------------------------------------------------- #
# Scope guard
# --------------------------------------------------------------------------- #

def test_scope_guard_accepts_admitted_subset():
    admitted = [DeviceID("a"), DeviceID("b"), DeviceID("c")]
    assert_candidates_admitted([DeviceID("a")], admitted)  # must not raise


def test_scope_guard_rejects_foreign_device():
    with pytest.raises(SelectorScopeViolation):
        assert_candidates_admitted([DeviceID("ghost")], [DeviceID("a")])


def test_selector_select_target_scope_guard_fires():
    sel = TargetSelectorRL()
    states = _states("a", "b")
    with pytest.raises(SelectorScopeViolation):
        sel.select_target(
            [DeviceID("a"), DeviceID("ghost")],
            states,
            bucket=Bucket.NEW,
            env=_env(),
            admitted=[DeviceID("a"), DeviceID("b")],
        )


# --------------------------------------------------------------------------- #
# Basic inference
# --------------------------------------------------------------------------- #

def test_select_target_empty_returns_none():
    sel = TargetSelectorRL()
    assert sel.select_target(
        [], {}, bucket=Bucket.NEW, env=_env()
    ) is None


def test_select_target_returns_one_of_candidates():
    sel = TargetSelectorRL()
    states = _states("a", "b", "c")
    got = sel.select_target(
        list(states.keys()), states, bucket=Bucket.NEW, env=_env()
    )
    assert got in states


def test_rank_produces_permutation_of_input():
    sel = TargetSelectorRL()
    states = _states("a", "b", "c", "d")
    ordered = sel.rank(
        list(states.keys()), states, bucket=Bucket.NEW, env=_env()
    )
    assert sorted(ordered) == sorted(states.keys())


def test_last_chosen_features_populated_after_select():
    sel = TargetSelectorRL()
    states = _states("a", "b")
    assert sel.last_chosen_features is None
    sel.select_target(list(states.keys()), states, bucket=Bucket.NEW, env=_env())
    feats = sel.last_chosen_features
    assert feats is not None
    assert feats.shape == (11,)


def test_feature_dim_mismatch_rejects_ddqn():
    wrong = DDQN(feature_dim=3)
    with pytest.raises(ValueError):
        TargetSelectorRL(ddqn=wrong)


# --------------------------------------------------------------------------- #
# Server selection
# --------------------------------------------------------------------------- #

def test_select_server_returns_one_of_options():
    sel = TargetSelectorRL()
    servers = [
        (ServerID("s1"), (10.0, 0.0, 0.0)),
        (ServerID("s2"), (100.0, 0.0, 0.0)),
    ]
    got = sel.select_server(
        servers, mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0
    )
    assert got in {ServerID("s1"), ServerID("s2")}


def test_select_server_empty_returns_none():
    sel = TargetSelectorRL()
    assert sel.select_server(
        [], mule_pose=(0.0, 0.0, 0.0), mule_energy=1.0
    ) is None


# --------------------------------------------------------------------------- #
# Epsilon validation
# --------------------------------------------------------------------------- #

def test_set_epsilon_validates_range():
    sel = TargetSelectorRL()
    sel.set_epsilon(0.5)
    with pytest.raises(ValueError):
        sel.set_epsilon(-0.1)
    with pytest.raises(ValueError):
        sel.set_epsilon(1.5)
