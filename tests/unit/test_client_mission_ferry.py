"""FeRRy Phase 3, unit U5: the device side of the ferry contact.

* The device echoes the solicit's number (``FLOpenSolicit.solicit_id``) in its
  advert, in the update it sends for the push that follows, and in its
  delivery ack, so the mule can tell a reply to this contact from a late reply
  to an earlier one (critics B1, B2). An unnumbered (legacy) solicit gives 0
  everywhere, the field's default.
* A push marked ``uplink_drop`` (the mule's availability draw failed the
  uplink) is handled like EX-4.2's failed uplink: the basis is adopted, a
  train-ahead starts if asked, the prepared update is kept, nothing is sent,
  and the device's own reliability stream is not drawn.
"""

from __future__ import annotations

import threading

import numpy as np

from hermes.mission import ClientMission, LocalTrainResult
from hermes.transport import LoopbackRFLink
from hermes.types import (
    DeviceID,
    DiscPush,
    FLOpenSolicit,
    FLState,
    MissionOutcome,
    MissionPass,
    MuleID,
)

MULE = MuleID("m")
DEV = DeviceID("d0")


def _train(theta, synth):
    after = [np.asarray(w, dtype=np.float32) + np.float32(0.25) for w in theta]
    return LocalTrainResult(delta_theta=after, num_examples=6, accuracy=0.8, auc=0.8,
                            loss=0.2, theta_after=after)


def _theta(v=1.0):
    return [np.full((3,), v, dtype=np.float32)]


def _device(rf, **kw) -> ClientMission:
    cm = ClientMission(device_id=DEV, rf=rf, local_train=_train, solicit_timeout_s=2.0,
                       disc_push_timeout_s=2.0, **kw)
    cm.set_state(FLState.FL_OPEN)
    return cm


def _serve_in_background(cm):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("oc", cm.serve_once()), daemon=True)
    t.start()
    return t, out


def test_serve_once_echoes_the_solicit_number_in_advert_and_update():
    rf = LoopbackRFLink()
    rf.register_device(DEV)
    cm = _device(rf)
    t, out = _serve_in_background(cm)
    rf.solicit(FLOpenSolicit(mule_id=MULE, mission_round=4, issued_at=1e6,
                             solicit_id=17), [DEV])
    adv = rf.recv_ready_adv(timeout=5.0)
    assert adv.in_reply_to == 17
    rf.push_disc(DEV, DiscPush(mule_id=MULE, mission_round=4, theta_disc=_theta(),
                               synth_batch=[], basis_version=3))
    grad = rf.recv_gradient(DEV, timeout=5.0)
    t.join(5.0)
    assert out["oc"] is MissionOutcome.CLEAN
    assert grad.in_reply_to == 17 and grad.mission_round == 4


def test_serve_once_echoes_the_solicit_number_in_the_delivery_ack():
    rf = LoopbackRFLink()
    rf.register_device(DEV)
    cm = _device(rf)
    t, out = _serve_in_background(cm)
    rf.solicit(FLOpenSolicit(mule_id=MULE, mission_round=4, issued_at=1e6,
                             pass_kind=MissionPass.DELIVER, solicit_id=23), [DEV])
    assert rf.recv_ready_adv(timeout=5.0).in_reply_to == 23
    push = DiscPush(mule_id=MULE, mission_round=4, theta_disc=_theta(2.0), synth_batch=[],
                    pass_kind=MissionPass.DELIVER, basis_version=4)
    rf.push_disc(DEV, push)
    ack = rf.recv_delivery_ack(DEV, timeout=5.0)
    t.join(5.0)
    assert out["oc"] is MissionOutcome.CLEAN
    assert (ack.in_reply_to, ack.weights_sig, ack.mission_round) == (23, push.weights_sig, 4)


def test_serve_delivery_echoes_the_solicit_number():
    rf = LoopbackRFLink()
    rf.register_device(DEV)
    cm = _device(rf)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("oc", cm.serve_delivery()),
                         daemon=True)
    t.start()
    rf.solicit(FLOpenSolicit(mule_id=MULE, mission_round=2, issued_at=1e6,
                             pass_kind=MissionPass.DELIVER, solicit_id=5), [DEV])
    assert rf.recv_ready_adv(timeout=5.0).in_reply_to == 5
    rf.push_disc(DEV, DiscPush(mule_id=MULE, mission_round=2, theta_disc=_theta(),
                               synth_batch=[], pass_kind=MissionPass.DELIVER))
    assert rf.recv_delivery_ack(DEV, timeout=5.0).in_reply_to == 5
    t.join(5.0)


def test_an_unnumbered_solicit_gives_the_default_zero_everywhere():
    rf = LoopbackRFLink()
    rf.register_device(DEV)
    cm = _device(rf)
    t, _out = _serve_in_background(cm)
    rf.broadcast_open_solicit(FLOpenSolicit(mule_id=MULE, mission_round=1, issued_at=1.0))
    assert rf.recv_ready_adv(timeout=5.0).in_reply_to == 0
    rf.push_disc(DEV, DiscPush(mule_id=MULE, mission_round=1, theta_disc=_theta(),
                               synth_batch=[]))
    assert rf.recv_gradient(DEV, timeout=5.0).in_reply_to == 0
    t.join(5.0)
    assert cm.build_ready_adv().in_reply_to == 0          # beacons are unnumbered


def test_with_newest_solicit_only_a_stale_solicit_is_not_answered():
    """Critic B1: after missing a gather, the device answers the solicit the mule
    is gathering for now, not the one queued before it (U0's option)."""
    rf = LoopbackRFLink(newest_solicit_only=True)
    rf.register_device(DEV)
    cm = _device(rf)
    rf.solicit(FLOpenSolicit(mule_id=MULE, mission_round=1, issued_at=1e6,
                             solicit_id=1), [DEV])        # never served
    rf.solicit(FLOpenSolicit(mule_id=MULE, mission_round=2, issued_at=1e6 + 50,
                             solicit_id=2), [DEV])
    t, _out = _serve_in_background(cm)
    assert rf.recv_ready_adv(timeout=5.0).in_reply_to == 2
    rf.push_disc(DEV, DiscPush(mule_id=MULE, mission_round=2, theta_disc=_theta(),
                               synth_batch=[]))
    assert rf.recv_gradient(DEV, timeout=5.0).in_reply_to == 2
    t.join(5.0)


# --------------------------------------------------------------------------- #
# The mule-decided uplink drop
# --------------------------------------------------------------------------- #

class _SinkRF:
    def __init__(self):
        self.grads = []

    def register_device(self, _d):
        pass

    def send_gradient(self, g):
        self.grads.append(g)


def _push(**kw):
    return DiscPush(mule_id=MULE, mission_round=3, theta_disc=_theta(5.0), synth_batch=[],
                    basis_version=9, **kw)


def test_an_uplink_drop_adopts_the_basis_and_sends_nothing():
    rf = _SinkRF()
    cm = ClientMission(device_id=DEV, rf=rf, local_train=_train)
    assert cm._handle_collect_push(_push(uplink_drop=True), in_reply_to=4) is \
        MissionOutcome.TIMEOUT
    assert rf.grads == []
    np.testing.assert_array_equal(cm._theta_basis[0], _theta(5.0)[0])
    assert cm._theta_basis_version == 9
    assert cm._last_outcome is MissionOutcome.TIMEOUT


def test_an_uplink_drop_keeps_the_prepared_update_for_the_next_contact():
    rf = _SinkRF()
    cm = ClientMission(device_id=DEV, rf=rf, local_train=_train)
    cm._set_theta_basis(_theta(1.0), [], version=8)
    prepared = cm.train_offline()
    assert prepared is not None
    cm._handle_collect_push(_push(uplink_drop=True))
    assert cm._prepared_delta is prepared                  # not consumed
    assert cm._handle_collect_push(_push(), in_reply_to=6) is MissionOutcome.CLEAN
    assert len(rf.grads) == 1 and rf.grads[0].in_reply_to == 6
    assert rf.grads[0].basis_version == 8                  # the prepared update's basis


def test_an_uplink_drop_trains_ahead_when_asked():
    rf = _SinkRF()
    cm = ClientMission(device_id=DEV, rf=rf, local_train=_train)
    cm._handle_collect_push(_push(uplink_drop=True, train_ahead=True))
    cm._train_thread.join(5.0)
    assert cm._prepared_delta is not None and cm._prepared_basis_version == 9


def test_an_uplink_drop_does_not_draw_the_devices_reliability_stream():
    """The mule already decided; a device-side draw would shift its stream."""
    rf = _SinkRF()
    a = ClientMission(device_id=DEV, rf=rf, local_train=_train,
                      contact_reliability=0.5, contact_rng_seed=11)
    b = ClientMission(device_id=DEV, rf=rf, local_train=_train,
                      contact_reliability=0.5, contact_rng_seed=11)
    a._handle_collect_push(_push(uplink_drop=True))
    assert a._contact_rng.random() == b._contact_rng.random()


def test_the_handlers_still_take_a_push_alone():
    """The golden harnesses call the handlers with the push only (legacy)."""
    rf = _SinkRF()
    cm = ClientMission(device_id=DEV, rf=rf, local_train=_train)
    assert cm._handle_collect_push(_push()) is MissionOutcome.CLEAN
    assert rf.grads[0].in_reply_to == 0
