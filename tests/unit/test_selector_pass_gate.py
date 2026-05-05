"""Sprint 1.5 — selector ``pass_kind`` gate + per-contact API.

Pins down design §7 principle 13: the selector is Pass-1-only. Calling
any of ``select_target / rank / select_contact / rank_contacts`` with
``pass_kind=DELIVER`` raises :class:`SelectorScopeViolation`.

Also covers the per-contact extractor + ``select_contact`` happy paths.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from hermes.scheduler.selector import (
    FEATURE_DIM,
    SelectorEnv,
    SelectorScopeViolation,
    TargetSelectorRL,
    extract_features_for_contact,
    extract_features_contact_batch,
)
from hermes.types import (
    Bucket,
    ContactWaypoint,
    DeviceID,
    DeviceSchedulerState,
    MissionPass,
)


def _state(
    did: str,
    pos=(0.0, 0.0, 0.0),
    *,
    bucket: Bucket = Bucket.SCHEDULED_THIS_ROUND,
    on_time_count: int = 5,
    missed_count: int = 5,
) -> DeviceSchedulerState:
    st = DeviceSchedulerState(
        device_id=DeviceID(did),
        last_known_position=pos,
        is_in_slice=True,
        is_new=False,
        on_time_count=on_time_count,
        missed_count=missed_count,
    )
    st.bucket = bucket
    return st


def _state_map(states: List[DeviceSchedulerState]) -> Dict[DeviceID, DeviceSchedulerState]:
    return {s.device_id: s for s in states}


def _env() -> SelectorEnv:
    return SelectorEnv(
        mule_pose=(0.0, 0.0, 0.0),
        mule_energy=1.0,
        rf_prior_snr_db=20.0,
        now=0.0,
    )


# --------------------------------------------------------------------------- #
# pass_kind=DELIVER raises on every selector entry point
# --------------------------------------------------------------------------- #

def test_select_target_in_pass_2_raises():
    sel = TargetSelectorRL(rng_seed=0)
    with pytest.raises(SelectorScopeViolation, match="select_target.*pass_kind='deliver'"):
        sel.select_target(
            candidates=[DeviceID("d0")],
            device_states={DeviceID("d0"): _state("d0")},
            bucket=Bucket.SCHEDULED_THIS_ROUND,
            env=_env(),
            pass_kind=MissionPass.DELIVER,
        )


def test_rank_in_pass_2_raises():
    sel = TargetSelectorRL(rng_seed=0)
    with pytest.raises(SelectorScopeViolation, match="rank.*pass_kind='deliver'"):
        sel.rank(
            candidates=[DeviceID("d0")],
            device_states={DeviceID("d0"): _state("d0")},
            bucket=Bucket.SCHEDULED_THIS_ROUND,
            env=_env(),
            pass_kind=MissionPass.DELIVER,
        )


def test_select_contact_in_pass_2_raises():
    sel = TargetSelectorRL(rng_seed=0)
    states = [_state("d0", pos=(10.0, 0.0, 0.0))]
    cw = ContactWaypoint(
        position=(10.0, 0.0, 0.0),
        devices=(DeviceID("d0"),),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=100.0,
    )
    with pytest.raises(SelectorScopeViolation, match="select_contact.*pass_kind='deliver'"):
        sel.select_contact(
            candidates=[cw],
            device_states=_state_map(states),
            env=_env(),
            pass_kind=MissionPass.DELIVER,
        )


def test_rank_contacts_in_pass_2_raises():
    sel = TargetSelectorRL(rng_seed=0)
    states = [_state("d0")]
    cw = ContactWaypoint(
        position=(0.0, 0.0, 0.0),
        devices=(DeviceID("d0"),),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=100.0,
    )
    with pytest.raises(SelectorScopeViolation, match="rank_contacts.*pass_kind='deliver'"):
        sel.rank_contacts(
            candidates=[cw],
            device_states=_state_map(states),
            env=_env(),
            pass_kind=MissionPass.DELIVER,
        )


# --------------------------------------------------------------------------- #
# pass_kind=COLLECT (default) works
# --------------------------------------------------------------------------- #

def test_select_target_default_pass_is_collect():
    """No explicit pass_kind defaults to COLLECT and selects normally."""
    sel = TargetSelectorRL(rng_seed=0)
    states = [_state("d0"), _state("d1", pos=(50.0, 0.0, 0.0))]
    chosen = sel.select_target(
        candidates=[s.device_id for s in states],
        device_states=_state_map(states),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        env=_env(),
    )
    assert chosen in {DeviceID("d0"), DeviceID("d1")}


# --------------------------------------------------------------------------- #
# Per-contact feature extractor
# --------------------------------------------------------------------------- #

def test_extract_features_for_contact_shape():
    states = [
        _state("d0", pos=(10.0, 0.0, 0.0), on_time_count=8, missed_count=2),
        _state("d1", pos=(20.0, 0.0, 0.0), on_time_count=2, missed_count=8),
    ]
    cw = ContactWaypoint(
        position=(15.0, 0.0, 0.0),
        devices=(DeviceID("d0"), DeviceID("d1")),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=100.0,
    )
    feats = extract_features_for_contact(cw, _state_map(states), _env())
    assert feats.shape == (FEATURE_DIM,)
    # mean on_time_rate = mean(0.8, 0.2) = 0.5; slot 4
    assert feats[4] == pytest.approx(0.5, abs=1e-6)
    # member count = 2; slot 5 = 2/5 = 0.4
    assert feats[5] == pytest.approx(0.4, abs=1e-6)
    # bucket one-hot for SCHEDULED_THIS_ROUND in slot 7
    assert feats[7] == pytest.approx(1.0)


def test_extract_features_contact_batch_empty_returns_zero_rows():
    feats = extract_features_contact_batch([], {}, _env())
    assert feats.shape == (0, FEATURE_DIM)


def test_extract_features_for_contact_n_equals_one():
    """Single-device contact still produces a valid 11-dim vector."""
    states = [_state("d0", pos=(20.0, 30.0, 0.0))]
    cw = ContactWaypoint(
        position=(20.0, 30.0, 0.0),
        devices=(DeviceID("d0"),),
        bucket=Bucket.NEW,
        deadline_ts=100.0,
    )
    feats = extract_features_for_contact(cw, _state_map(states), _env())
    assert feats.shape == (FEATURE_DIM,)
    # member count = 1 / 5 = 0.2
    assert feats[5] == pytest.approx(0.2, abs=1e-6)
    # bucket = NEW → slot 6
    assert feats[6] == pytest.approx(1.0)
    assert feats[7] == pytest.approx(0.0)


def test_extract_features_for_contact_caps_member_count():
    """member_count > 5 caps to 1.0 (slot 5 is a normalised feature)."""
    states = [_state(f"d{i}", pos=(0.0, 0.0, 0.0)) for i in range(8)]
    cw = ContactWaypoint(
        position=(0.0, 0.0, 0.0),
        devices=tuple(s.device_id for s in states),
        bucket=Bucket.SCHEDULED_THIS_ROUND,
        deadline_ts=100.0,
    )
    feats = extract_features_for_contact(cw, _state_map(states), _env())
    assert feats[5] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# select_contact / rank_contacts happy paths
# --------------------------------------------------------------------------- #

def test_select_contact_empty_returns_none():
    sel = TargetSelectorRL(rng_seed=0)
    assert sel.select_contact(candidates=[], device_states={}, env=_env()) is None


def test_select_contact_picks_one():
    sel = TargetSelectorRL(rng_seed=0)
    states = [
        _state("d0", pos=(10.0, 0.0, 0.0)),
        _state("d1", pos=(50.0, 0.0, 0.0)),
    ]
    contacts = [
        ContactWaypoint(
            position=(10.0, 0.0, 0.0),
            devices=(DeviceID("d0"),),
            bucket=Bucket.SCHEDULED_THIS_ROUND,
            deadline_ts=100.0,
        ),
        ContactWaypoint(
            position=(50.0, 0.0, 0.0),
            devices=(DeviceID("d1"),),
            bucket=Bucket.SCHEDULED_THIS_ROUND,
            deadline_ts=100.0,
        ),
    ]
    chosen = sel.select_contact(
        candidates=contacts, device_states=_state_map(states), env=_env(),
    )
    assert chosen in contacts


def test_rank_contacts_returns_full_ordering():
    sel = TargetSelectorRL(rng_seed=0)
    states = [
        _state("d0", pos=(10.0, 0.0, 0.0)),
        _state("d1", pos=(50.0, 0.0, 0.0)),
        _state("d2", pos=(100.0, 0.0, 0.0)),
    ]
    contacts = [
        ContactWaypoint(
            position=s.last_known_position,
            devices=(s.device_id,),
            bucket=Bucket.SCHEDULED_THIS_ROUND,
            deadline_ts=100.0,
        )
        for s in states
    ]
    ordered = sel.rank_contacts(
        candidates=contacts, device_states=_state_map(states), env=_env(),
    )
    assert len(ordered) == 3
    assert set(ordered) == set(contacts)
