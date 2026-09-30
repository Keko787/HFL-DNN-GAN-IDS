"""FeRRy Phase 3 (unit U0): targeted solicits and the newest-solicit option.

In-process checks on ``LoopbackRFLink`` and the ``RFLink`` ABC, plus the
``DeviceConfig.newest_solicit_only`` switch. The TCP transport's versions of
the same behaviour are in ``tests/integration/test_rf_link_amendment10.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from hermes.processes.config import (
    DeviceConfig,
    device_config_from_json,
    device_config_to_json,
)
from hermes.transport import LoopbackRFLink, RFLink, RFLinkError
from hermes.types import DeviceID, FLOpenSolicit, MuleID

MULE = MuleID("m1")


def _solicit(mission_round: int) -> FLOpenSolicit:
    return FLOpenSolicit(mule_id=MULE, mission_round=mission_round, issued_at=0.0)


def _link(*devices: str, **kwargs) -> LoopbackRFLink:
    rf = LoopbackRFLink(**kwargs)
    for d in devices:
        rf.register_device(DeviceID(d))
    return rf


# --------------------------------------------------------------------------- #
# RFLink.solicit on the ABC
# --------------------------------------------------------------------------- #

class _BroadcastOnlyLink(RFLink):
    """An RFLink written before targeted solicits existed (a test double)."""

    def broadcast_open_solicit(self, msg): ...
    def recv_ready_adv(self, timeout=None): ...
    def push_disc(self, device_id, msg): ...
    def recv_gradient(self, device_id, timeout=None): ...
    def recv_delivery_ack(self, device_id, timeout=None): ...
    def recv_open_solicit(self, device_id, timeout=None): ...
    def send_ready_adv(self, msg): ...
    def recv_disc_push(self, device_id, timeout=None): ...
    def send_gradient(self, msg): ...
    def send_delivery_ack(self, msg): ...
    def close(self): ...


def test_solicit_is_not_abstract_and_raises_by_default():
    link = _BroadcastOnlyLink()  # still constructible: solicit is not abstract
    with pytest.raises(NotImplementedError, match="_BroadcastOnlyLink"):
        link.solicit(_solicit(1), [DeviceID("d1")])


# --------------------------------------------------------------------------- #
# LoopbackRFLink.solicit
# --------------------------------------------------------------------------- #

def test_solicit_reaches_only_the_listed_devices():
    rf = _link("d1", "d2", "d3")
    sent = rf.solicit(_solicit(4), [DeviceID("d3"), DeviceID("d1")])

    assert sent == [DeviceID("d3"), DeviceID("d1")]
    assert rf.recv_open_solicit(DeviceID("d1"), timeout=0.1).mission_round == 4
    assert rf.recv_open_solicit(DeviceID("d3"), timeout=0.1).mission_round == 4
    with pytest.raises(RFLinkError):
        rf.recv_open_solicit(DeviceID("d2"), timeout=0.05)


def test_solicit_skips_unknown_ids_without_creating_queues():
    rf = _link("d1")
    sent = rf.solicit(_solicit(1), [DeviceID("ghost"), DeviceID("d1")])

    assert sent == [DeviceID("d1")]
    assert rf.known_devices() == [DeviceID("d1")]
    # A device registering later must not find this stale solicit waiting.
    rf.register_device(DeviceID("ghost"))
    with pytest.raises(RFLinkError):
        rf.recv_open_solicit(DeviceID("ghost"), timeout=0.05)


def test_solicit_sends_once_per_device_in_the_given_order():
    rf = _link("d1", "d2")
    sent = rf.solicit(_solicit(2), [DeviceID("d2"), DeviceID("d1"), DeviceID("d2")])

    assert sent == [DeviceID("d2"), DeviceID("d1")]
    rf.recv_open_solicit(DeviceID("d2"), timeout=0.1)
    with pytest.raises(RFLinkError):  # no second copy for the repeated id
        rf.recv_open_solicit(DeviceID("d2"), timeout=0.05)


def test_solicit_with_no_ids_sends_nothing():
    rf = _link("d1")
    assert rf.solicit(_solicit(1), []) == []
    with pytest.raises(RFLinkError):
        rf.recv_open_solicit(DeviceID("d1"), timeout=0.05)


def test_solicit_refuses_a_bare_device_id():
    rf = _link("d1")
    with pytest.raises(TypeError, match="collection of device ids"):
        rf.solicit(_solicit(1), DeviceID("d1"))


def test_solicit_on_a_closed_link_raises():
    rf = _link("d1")
    rf.close()
    with pytest.raises(RFLinkError):
        rf.solicit(_solicit(1), [DeviceID("d1")])


def test_broadcast_still_reaches_every_device_after_a_targeted_solicit():
    rf = _link("d1", "d2")
    rf.solicit(_solicit(1), [DeviceID("d1")])
    rf.broadcast_open_solicit(_solicit(2))

    assert rf.recv_open_solicit(DeviceID("d1"), timeout=0.1).mission_round == 1
    assert rf.recv_open_solicit(DeviceID("d1"), timeout=0.1).mission_round == 2
    assert rf.recv_open_solicit(DeviceID("d2"), timeout=0.1).mission_round == 2


# --------------------------------------------------------------------------- #
# Newest-solicit option (critic B1)
# --------------------------------------------------------------------------- #

def test_default_answers_queued_solicits_in_arrival_order():
    rf = _link("d1")
    for r in (1, 2, 3):
        rf.broadcast_open_solicit(_solicit(r))

    got = [rf.recv_open_solicit(DeviceID("d1"), timeout=0.1).mission_round for _ in range(3)]
    assert got == [1, 2, 3]


def test_newest_solicit_only_answers_the_newest_and_drops_the_rest():
    rf = _link("d1", "d2", newest_solicit_only=True)
    for r in (1, 2, 3):
        rf.broadcast_open_solicit(_solicit(r))

    assert rf.recv_open_solicit(DeviceID("d1"), timeout=0.1).mission_round == 3
    with pytest.raises(RFLinkError):  # the older two are gone, not deferred
        rf.recv_open_solicit(DeviceID("d1"), timeout=0.05)
    # Per device: d2's queue is drained only when d2 reads it.
    assert rf.recv_open_solicit(DeviceID("d2"), timeout=0.1).mission_round == 3


def test_newest_solicit_only_still_blocks_until_one_arrives():
    rf = _link("d1", newest_solicit_only=True)
    with pytest.raises(RFLinkError, match="timed out"):
        rf.recv_open_solicit(DeviceID("d1"), timeout=0.05)
    rf.solicit(_solicit(7), [DeviceID("d1")])
    assert rf.recv_open_solicit(DeviceID("d1"), timeout=0.1).mission_round == 7


# --------------------------------------------------------------------------- #
# DeviceConfig.newest_solicit_only
# --------------------------------------------------------------------------- #

def test_device_config_newest_solicit_only_defaults_off():
    assert DeviceConfig(device_id="d1").newest_solicit_only is False


def test_device_config_newest_solicit_only_round_trips_through_json():
    cfg = DeviceConfig(device_id="d1", newest_solicit_only=True)
    back = device_config_from_json(device_config_to_json(cfg))
    assert back.newest_solicit_only is True
    assert back == cfg


def test_device_config_json_without_the_key_still_loads():
    raw = asdict(DeviceConfig(device_id="d1", position=(1.0, 2.0, 0.0)))
    # A per-role JSON written before Phase 3 has neither key.
    del raw["newest_solicit_only"]
    del raw["rf_link_token"]
    cfg = device_config_from_json(json.dumps(raw))
    assert cfg.newest_solicit_only is False
    assert cfg.rf_link_token is None
    assert cfg.position == (1.0, 2.0, 0.0)


# --------------------------------------------------------------------------- #
# DeviceConfig.rf_link_token (Amendment 10)
# --------------------------------------------------------------------------- #

def test_device_config_rf_link_token_defaults_to_none():
    assert DeviceConfig(device_id="d1").rf_link_token is None


def test_device_config_rf_link_token_round_trips_through_json():
    cfg = DeviceConfig(device_id="d1", rf_link_token="trial-3f9a")
    back = device_config_from_json(device_config_to_json(cfg))
    assert back.rf_link_token == "trial-3f9a"
    assert back == cfg
