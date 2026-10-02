"""FeRRy Phase 5 (unit U2): the pair learner and its replay.

What is pinned (the Phase 5 spec: other choices 3 and 6, the units table's
row U2; critic B15 and C4):

* **The target** is double DQN over the next decision's admitted rows: the
  online network chooses (ties to the lowest row) and the target network
  values; masked rows, poisoned with NaN, inf or 1e9, never enter a target
  or any forward pass; γ = 0 gives r and reads no next row; done stops the
  bootstrap; variable-length candidate sets in one batch give each
  transition its own target.
* **The network**: the documented initial draws; one Q per row, and one Q,
  bit for bit, for equal rows, so the slot's tie rule (the lowest row)
  decides between equal pairs and rounding does not; ``set_weights`` copies
  into the target and from the caller.
* **The update**: the Huber loss (checked by hand) and its gradient
  (checked by finite differences through two tanh layers), the global-norm
  clip and Adam (both checked by hand, at the spec's settings and at
  settings that differ from every default, so that the one revision allowed
  before the sweep trains as its manifest says), and a hard target sync
  every ``target_sync`` updates. One update per decision after the warm-up.
* **The replay**: copied, read-only transitions; a done transition has no
  next decision and every other one an admitted next row; a FIFO ring;
  sampling deterministic under a seed, ``random.Random(seed).sample``'s.
* **Behaviour** (critic C4): ε-greedy around FX's pair for the first 500
  episodes at 0.3, then on Q with ε falling to 0.05 over the first half of
  training, and every setting of a configured schedule read (E3's has no FX
  phase); exploration uniform over the admitted rows only; two draws per
  decision; a wrong reference refused before any draw.
* **Checkpoints, format 2**: the round trip; the sha covers the weights
  and what they read, and the header holds the purpose and the learner's
  revision, so the sha binds those too (the orchestrator's resolution R2):
  a manifest relabelled from bootstrap to trained, or to another revision,
  is refused wherever a checkpoint is verified, a header rewritten to match
  makes a new sha, and a header written before the binding is refused; the
  provenance stays outside the sha; the legacy ``DDQN.load`` refuses
  format 2 and the pair loader refuses format 1; the loader refuses a
  missing or inconsistent manifest and every unexpected sha, kind, class
  tuple or schema; the runner's refusals of an untrained, unscored or dirty
  checkpoint (critic B9), where a held-out score holds its episodes and
  mean return.
* **Learning** (critic B15): on a known-answer pair of chains, only γ > 0
  learns the better first action, and where the myopic action is optimal
  every γ of Study 5.5's grid agrees, each learned Q at its analytic value.
* **Layering**: numpy and the standard library only; no recorded path loads
  either module; the legacy selector files are 386c275's.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hermes.scheduler.selector import pair_q as Q
from hermes.scheduler.selector import pair_replay as R
from hermes.scheduler.selector.ddqn import DDQN
from hermes.scheduler.selector.pair_q import (
    BehaviourSchedule,
    CheckpointError,
    LearnerSettings,
    PairQConfig,
    PairQLearner,
    PairQNet,
)
from hermes.scheduler.selector.pair_replay import PairBatch, PairReplay, PairTransition

REPO = Path(__file__).resolve().parents[2]

#: Study 5.5's γ grid (the user's decision 5 (a)).
GAMMAS = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)


def _done_batch(x, rewards) -> PairBatch:
    """Terminal transitions only: their targets are their rewards."""
    return PairBatch.of([PairTransition(np.asarray(row, dtype=float), float(r), True)
                         for row, r in zip(x, rewards)])


def _linear(weights, bias, *, gamma=0.5, **cfg) -> PairQNet:
    """A linear score q = x . w + b, both networks set to it."""
    width = len(weights)
    net = PairQNet(width, PairQConfig(hidden=(), gamma=gamma, **cfg), seed=0)
    net.set_weights({"layer0_W": np.asarray(weights, dtype=float).reshape(width, 1),
                     "layer0_b": np.asarray([bias], dtype=float)})
    return net


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #

def test_a_fresh_network_is_seeded_float64_and_its_target_is_a_copy():
    a, b, c = (PairQNet(7, seed=s) for s in (3, 3, 4))
    assert [w.shape for w in a.weights().values()] == [(7, 64), (64,), (64, 64), (64,), (64, 1),
                                                       (1,)]
    assert all(w.dtype == np.float64 for w in a.weights().values())
    assert all(np.array_equal(a.weights()[k], b.weights()[k]) for k in a.weights())
    assert not np.array_equal(a.weights()["layer0_W"], c.weights()["layer0_W"])
    assert all(np.array_equal(a.weights()[k], a.target_weights()[k]) for k in a.weights())
    assert (a.updates, a.syncs, a.last_update) == (0, 0, None)


def test_the_initial_weights_are_the_documented_draws():
    """W ~ N(0, 1/fan_in) from default_rng(seed), layer by layer, and b = 0: a
    manifest's init seed rebuilds the very network a run started from."""
    net = PairQNet(7, PairQConfig(hidden=(16, 8)), seed=3)
    rng = np.random.default_rng(3)
    weights = net.weights()
    for i, (fan_in, fan_out) in enumerate(((7, 16), (16, 8), (8, 1))):
        drawn = rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_in, fan_out))
        assert np.array_equal(weights[f"layer{i}_W"], drawn), i
        assert np.array_equal(weights[f"layer{i}_b"], np.zeros(fan_out)), i
    wide = PairQNet(400, PairQConfig(hidden=(200,)), seed=0).weights()["layer0_W"]
    assert np.std(wide) == pytest.approx(1.0 / np.sqrt(400), rel=0.02)


def test_each_row_gets_one_q_from_shared_weights():
    """A pointer score: a row's Q does not depend on the other rows or their order,
    so the candidate set can grow and shrink with the remainder."""
    net = PairQNet(5, seed=1)
    rows = np.random.default_rng(0).normal(size=(9, 5))
    q = net.q(rows)
    assert q.shape == (9,)
    np.testing.assert_allclose(q, [net.q(row[None, :])[0] for row in rows], rtol=0, atol=1e-12)
    order = np.random.default_rng(1).permutation(9)
    np.testing.assert_allclose(net.q(rows[order]), q[order], rtol=0, atol=1e-12)
    for bad in (rows[:, :4], rows[0], np.full((2, 5), np.nan)):
        with pytest.raises(ValueError):
            net.q(bad)


def test_equal_rows_get_one_q_so_ties_go_to_the_lowest_copy(monkeypatch):
    """A symmetric layout offers pairs whose rows are equal. Scored in one pass,
    equal rows could round apart in the last bit and a later copy win (on
    random rows, about a quarter of the time); each distinct row is scored
    once, so copies tie exactly and the lowest admitted copy wins, the slot's
    rule (plan.types.PairScorer)."""
    for width in (5, 28):
        for seed in range(6):
            net = PairQNet(width, PairQConfig(hidden=(16, 16)), seed=seed)
            net.set_weights(net.weights(), target={k: w * 0.5 for k, w in net.weights().items()})
            rng = np.random.default_rng(100 + seed)
            for k in range(2, 33):
                rows = rng.normal(size=(k, width))
                best = int(np.argmax(net.q(rows)))
                rows = np.vstack([rows, rows[best]])
                q, q_target = net.q(rows), net.q_target(rows)
                assert q[k] == q[best] and q_target[k] == q_target[best], (width, seed, k)
                mask = np.ones(k + 1, dtype=bool)
                assert net.masked_argmax(rows, mask) == best, (width, seed, k)
                mask[best] = False
                assert net.masked_argmax(rows, mask) == k, (width, seed, k)
    passes = []
    real = Q._forward
    monkeypatch.setattr(Q, "_forward", lambda params, x: passes.append(len(x)) or real(params, x))
    signed = np.array([[0.0, 1.0, 2.0], [0.5, 0.5, 0.5], [-0.0, 1.0, 2.0]])
    q = PairQNet(3, seed=0).q(signed)
    assert passes == [2] and q[0] == q[2]      # one pass over the distinct rows: -0.0 is 0.0


def test_set_weights_copies_into_the_target_and_from_the_caller():
    """U8b keeps its best weights with weights() and restores them with
    set_weights(best). The target network then holds its own copy (were it the
    online arrays, each update would drag it along: no target network), and the
    network keeps none of the caller's arrays (an update would rewrite best)."""
    net = PairQNet(3, PairQConfig(hidden=(4,), gamma=0.9), seed=0)
    best = net.weights()
    kept = copy.deepcopy(best)
    batch = PairBatch.of([
        PairTransition(np.ones(3), 1.0, False, np.eye(3), np.array([True, False, True])),
        PairTransition(np.array([0.0, 1.0, -1.0]), -1.0, True)])
    net.update(batch)
    for target in (None, best):
        net.set_weights(best, target=target)
        net.update(batch)
        online, after = net.weights(), net.target_weights()
        assert all(np.array_equal(after[k], kept[k]) for k in kept)
        assert any(not np.array_equal(online[k], kept[k]) for k in kept)
        assert all(np.array_equal(best[k], kept[k]) for k in kept)
    returned = net.weights()
    returned["layer0_b"][:] = 99.0
    assert not (net.weights()["layer0_b"] == 99.0).any()


def test_the_masked_argmax_ties_to_the_lowest_admitted_row():
    scores = [1.0, 3.0, 3.0, 2.0, 3.0]
    assert Q.masked_argmax(scores, np.ones(5, dtype=bool)) == 1
    assert Q.masked_argmax(scores, np.array([True, False, True, True, True])) == 2
    assert Q.masked_argmax(scores, np.array([True, False, False, True, False])) == 3
    with pytest.raises(ValueError, match="no row is admitted"):
        Q.masked_argmax(scores, np.zeros(5, dtype=bool))
    with pytest.raises(TypeError, match="bools"):
        Q.masked_argmax(scores, [1, 0, 1, 1, 1])
    with pytest.raises(FloatingPointError):
        Q.masked_argmax([1.0, np.nan], np.array([True, True]))
    assert Q.masked_argmax([np.nan, 1.0], np.array([False, True])) == 1


# --------------------------------------------------------------------------- #
# The target
# --------------------------------------------------------------------------- #

def test_poisoned_masked_rows_never_enter_a_target(monkeypatch):
    """Masked next rows hold NaN, inf and 1e9: no forward pass sees them, the
    targets are those of the admitted rows alone, and an update raises no
    floating-point warning."""
    online = _linear([1.0, 0.0, 0.0], 0.0, gamma=0.5)
    online.set_weights(online.weights(), target={"layer0_W": np.array([[2.0], [0.0], [0.0]]),
                                                 "layer0_b": np.array([0.0])})
    admitted_a = np.array([[0.1, 0.0, 0.0], [0.5, 1.0, 1.0]])
    admitted_b = np.array([[0.3, 0.0, 0.0]])
    poison = np.array([[np.nan, 0.0, 0.0], [np.inf, 1.0, 1.0], [1e9, 0.0, 0.0],
                       [-np.inf, 0.0, 0.0]])
    batch = PairBatch.of([
        PairTransition(np.ones(3), 1.0, False, np.vstack([poison[:2], admitted_a, poison[2:]]),
                       np.array([False, False, True, True, False, False])),
        PairTransition(np.ones(3), -1.0, False, np.vstack([admitted_b, poison]),
                       np.array([True, False, False, False, False])),
        PairTransition(np.ones(3), 2.0, True),
    ])
    seen = []
    real = Q._forward

    def spy(params, x):
        seen.append(np.array(x, copy=True))
        return real(params, x)

    monkeypatch.setattr(Q, "_forward", spy)
    y = online.targets(batch)
    # online prefers 0.5 over 0.1; the target values it at 2 * 0.5
    assert y.tolist() == [1.0 + 0.5 * 1.0, -1.0 + 0.5 * 0.6, 2.0]
    assert seen and all(np.isfinite(x).all() and np.abs(x).max() < 10 for x in seen)
    deep = PairQNet(3, PairQConfig(gamma=0.9), seed=5)
    with np.errstate(invalid="raise", over="raise", divide="raise"):
        loss = deep.update(batch)
    assert np.isfinite(loss)
    assert all(np.isfinite(x).all() and np.abs(x).max() < 10 for x in seen)


def test_gamma_zero_gives_r_and_reads_no_next_row(monkeypatch):
    rng = np.random.default_rng(2)
    transitions = [PairTransition(rng.normal(size=4), float(r), done,
                                  None if done else rng.normal(size=(3, 4)) * 1e3,
                                  None if done else np.array([True, False, True]))
                   for r, done in ((0.25, False), (-3.5, True), (7.0, False))]
    batch = PairBatch.of(transitions)
    net = PairQNet(4, PairQConfig(gamma=0.0), seed=0)

    def no_forward(params, x):
        raise AssertionError("with γ = 0 the targets read no next row")

    monkeypatch.setattr(Q, "_forward", no_forward)
    assert np.array_equal(net.targets(batch), np.array([0.25, -3.5, 7.0]))


def test_done_stops_the_bootstrap():
    net = _linear([1.0, 0.0], 0.0, gamma=0.9)
    rows = np.array([[4.0, 0.0], [2.0, 0.0]])
    batch = PairBatch.of([
        PairTransition(np.zeros(2), 1.0, True),
        PairTransition(np.zeros(2), 1.0, False, rows, np.array([True, True])),
        PairTransition(np.zeros(2), -2.0, True),
    ])
    assert net.targets(batch).tolist() == [1.0, 1.0 + 0.9 * 4.0, -2.0]
    with pytest.raises(ValueError, match="done transition"):
        PairTransition(np.zeros(2), 1.0, True, rows, np.array([True, True]))
    with pytest.raises(ValueError, match="done transition owns no next rows"):
        PairBatch(x=np.zeros((1, 2)), reward=np.zeros(1), done=np.array([True]),
                  next_rows=rows, next_mask=np.array([True, True]), segment=np.array([0, 0]))


@pytest.mark.parametrize("order, mask, value", [
    ((0, 1), (True, True), 0.5),      # online picks u (2 > 1); the target values u: 0.5
    ((1, 0), (True, True), 0.5),      # the same pair of rows in the other order
    ((0, 1), (False, True), 3.0),     # u masked: v, whatever the target thinks of u
])
def test_the_online_network_chooses_and_the_target_network_values(order, mask, value):
    """y = r + γ · Q_target(s', argmax_a Q_online(s', a)), not the target's own max
    (vanilla DQN: 3.0) nor the online value (2.0)."""
    net = _linear([2.0, 1.0], 0.0, gamma=0.5)
    net.set_weights(net.weights(), target={"layer0_W": np.array([[0.5], [3.0]]),
                                           "layer0_b": np.array([0.0])})
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    rows = np.stack([(u, v)[i] for i in order])
    batch = PairBatch.of([PairTransition(np.zeros(2), 1.0, False, rows, np.array(mask))])
    assert net.targets(batch).tolist() == [1.0 + 0.5 * value]


@pytest.mark.parametrize("order, value", [((0, 1), 0.5), ((1, 0), 3.0)])
def test_online_ties_go_to_the_lowest_next_row(order, value):
    net = _linear([1.0, 1.0], 0.0, gamma=0.5)
    net.set_weights(net.weights(), target={"layer0_W": np.array([[0.5], [3.0]]),
                                           "layer0_b": np.array([0.0])})
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    rows = np.stack([(u, v)[i] for i in order])
    batch = PairBatch.of([PairTransition(np.zeros(2), 1.0, False, rows, np.array([True, True]))])
    assert net.targets(batch).tolist() == [1.0 + 0.5 * value]


def test_variable_length_batches_give_each_transition_its_own_target():
    """1 to 15 candidates per next decision (the probes' range at N = 12), some
    masked, some transitions done: the batch's targets equal a transition-by-
    transition reference."""
    rng = np.random.default_rng(3)
    net = PairQNet(6, PairQConfig(gamma=0.75, hidden=(8, 8)), seed=1)
    net.set_weights(net.weights(), target={k: w + rng.normal(scale=0.3, size=w.shape)
                                           for k, w in net.weights().items()})
    transitions = []
    for i in range(40):
        k = int(rng.integers(1, 16))
        mask = rng.random(k) < 0.6
        mask[int(rng.integers(k))] = True
        done = i % 4 == 3
        transitions.append(PairTransition(rng.normal(size=6), float(rng.normal()), done,
                                          None if done else rng.normal(size=(k, 6)),
                                          None if done else mask))
    batch = PairBatch.of(transitions)
    assert batch.next_rows.shape[0] == sum(t.next_rows.shape[0] for t in transitions
                                           if not t.done)
    reference = []
    for t in transitions:
        if t.done:
            reference.append(t.reward)
            continue
        rows = t.next_rows[t.next_mask]
        best = int(np.argmax(net.q(rows)))
        reference.append(t.reward + 0.75 * net.q_target(rows)[best])
    np.testing.assert_allclose(net.targets(batch), reference, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# The update
# --------------------------------------------------------------------------- #

def test_the_gradient_matches_finite_differences_through_two_tanh_layers():
    """The Huber loss's gradient, errors on both sides of δ and of both signs,
    against central differences of the loss itself (targets held fixed)."""
    rng = np.random.default_rng(11)
    net = PairQNet(4, PairQConfig(hidden=(5, 3)), seed=2)
    x = rng.normal(size=(6, 4))
    batch = _done_batch(x, np.zeros(6))
    targets = net.q(x) - np.array([0.3, -0.4, 2.5, -3.0, 0.7, 1.8])
    loss, grads = net.loss_and_gradients(batch, targets)
    base = net.weights()

    def loss_at(weights):
        probe = copy.deepcopy(net)
        probe.set_weights(weights)
        return probe.loss_and_gradients(batch, targets)[0]

    step = 1e-6
    for name, array in base.items():
        numeric = np.zeros_like(array)
        for idx in np.ndindex(array.shape):
            up = {k: w.copy() for k, w in base.items()}
            down = {k: w.copy() for k, w in base.items()}
            up[name][idx] += step
            down[name][idx] -= step
            numeric[idx] = (loss_at(up) - loss_at(down)) / (2 * step)
        np.testing.assert_allclose(grads[name], numeric, rtol=1e-6, atol=1e-8, err_msg=name)
    assert loss == pytest.approx(loss_at(base), rel=0, abs=0)


#: A linear score's hand-worked batch: q = [0.6, 2.35, 0.6], r = [0.2, 0, 3.6],
#: so the errors are 0.4 (inside δ = 1), 2.35 and -3.0 (outside, both signs).
HAND_W, HAND_B = np.array([0.5, -0.25, 1.0]), 0.1
HAND_X = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 2.0], [3.0, 0.0, -1.0]])
HAND_R = np.array([0.2, 0.0, 3.6])


def _hand_gradient(x, r, w, b, delta=1.0):
    err = x @ w + b - r
    loss = np.mean(np.where(np.abs(err) <= delta, 0.5 * err ** 2,
                            delta * (np.abs(err) - 0.5 * delta)))
    dq = np.clip(err, -delta, delta) / len(err)
    return loss, x.T @ dq, dq.sum()


def test_the_loss_is_huber_by_hand():
    net = _linear(HAND_W, HAND_B)
    loss, grads = net.loss_and_gradients(_done_batch(HAND_X, HAND_R))
    hand_loss, hand_w, hand_b = _hand_gradient(HAND_X, HAND_R, HAND_W, HAND_B)
    assert hand_loss == pytest.approx((0.08 + 1.85 + 2.5) / 3, rel=1e-12)
    assert loss == pytest.approx(hand_loss, rel=1e-12)
    np.testing.assert_allclose(grads["layer0_W"][:, 0], hand_w, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(grads["layer0_b"], [hand_b], rtol=1e-12, atol=1e-15)


def test_adam_steps_equal_a_hand_computation():
    """Two steps, so the bias corrections at t = 1 and t = 2 are both checked."""
    net = _linear(HAND_W, HAND_B, lr=1e-3)
    batch = _done_batch(HAND_X, HAND_R)
    w, b = HAND_W.copy(), HAND_B
    m_w, v_w, m_b, v_b = np.zeros(3), np.zeros(3), 0.0, 0.0
    for t in (1, 2):
        loss, g_w, g_b = _hand_gradient(HAND_X, HAND_R, w, b)
        assert np.sqrt(np.sum(g_w ** 2) + g_b ** 2) < 10.0   # the clip does not bind here
        m_w, m_b = 0.9 * m_w + 0.1 * g_w, 0.9 * m_b + 0.1 * g_b
        v_w, v_b = 0.999 * v_w + 0.001 * g_w ** 2, 0.999 * v_b + 0.001 * g_b ** 2
        w = w - 1e-3 * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)
        b = b - 1e-3 * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)
        assert net.update(batch) == pytest.approx(loss, rel=1e-12)
        np.testing.assert_allclose(net.weights()["layer0_W"][:, 0], w, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(net.weights()["layer0_b"], [b], rtol=1e-12, atol=1e-15)
        assert net.last_update.clip_scale == 1.0
    assert net.adam_moments()["step"] == 2


#: Settings that differ from every default. The one learner revision allowed
#: before the sweep (other choices 12) changes exactly these, and the header,
#: the sha and the manifest record them, so an update that read a default in
#: place of its configured value would train unlike its manifest.
REVISED = dict(lr=0.01, adam_beta1=0.5, adam_beta2=0.9, adam_eps=1e-3, huber_delta=0.5,
               grad_clip=0.3)


def test_the_update_reads_every_configured_setting():
    """Three steps under REVISED, by hand: the Huber kink at δ = 0.5 (the errors
    0.4 inside it, 2.35 and -3.0 outside), the clip binding at a norm of 0.3, and
    Adam's moments, bias corrections, ε and step at β1 = 0.5, β2 = 0.9, ε = 1e-3
    and lr = 0.01 (at t = 1 the corrections cancel the betas, so the moments are
    checked too)."""
    net = _linear(HAND_W, HAND_B, **REVISED)
    assert {k: getattr(net.config, k) for k in REVISED} == REVISED
    batch = _done_batch(HAND_X, HAND_R)
    w, b = HAND_W.copy(), HAND_B
    m_w, v_w, m_b, v_b = np.zeros(3), np.zeros(3), 0.0, 0.0
    for t in (1, 2, 3):
        loss, g_w, g_b = _hand_gradient(HAND_X, HAND_R, w, b, delta=0.5)
        norm = float(np.sqrt(np.sum(g_w ** 2) + g_b ** 2))
        scale = 0.3 / norm
        assert scale < 1.0                                   # the clip binds at every step
        g_w, g_b = g_w * scale, g_b * scale
        m_w, m_b = 0.5 * m_w + 0.5 * g_w, 0.5 * m_b + 0.5 * g_b
        v_w, v_b = 0.9 * v_w + 0.1 * g_w ** 2, 0.9 * v_b + 0.1 * g_b ** 2
        w = w - 0.01 * (m_w / (1 - 0.5 ** t)) / (np.sqrt(v_w / (1 - 0.9 ** t)) + 1e-3)
        b = b - 0.01 * (m_b / (1 - 0.5 ** t)) / (np.sqrt(v_b / (1 - 0.9 ** t)) + 1e-3)
        assert net.update(batch) == pytest.approx(loss, rel=1e-12)
        assert net.last_update.grad_norm == pytest.approx(norm, rel=1e-12)
        assert net.last_update.clip_scale == pytest.approx(scale, rel=1e-12)
        moments = net.adam_moments()
        assert moments["step"] == t
        np.testing.assert_allclose(moments["m"]["layer0_W"][:, 0], m_w, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(moments["m"]["layer0_b"], [m_b], rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(moments["v"]["layer0_W"][:, 0], v_w, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(moments["v"]["layer0_b"], [v_b], rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(net.weights()["layer0_W"][:, 0], w, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(net.weights()["layer0_b"], [b], rtol=1e-12, atol=1e-15)


def test_the_global_norm_clip_scales_the_gradient_adam_sees():
    x = HAND_X * 100.0
    net = _linear(HAND_W, HAND_B, grad_clip=10.0)
    _, g_w, g_b = _hand_gradient(x, HAND_R, HAND_W, HAND_B)
    norm = float(np.sqrt(np.sum(g_w ** 2) + g_b ** 2))
    assert norm > 10.0
    net.update(_done_batch(x, HAND_R))
    assert net.last_update.grad_norm == pytest.approx(norm, rel=1e-12)
    assert net.last_update.clip_scale == pytest.approx(10.0 / norm, rel=1e-12)
    moments = net.adam_moments()
    np.testing.assert_allclose(moments["m"]["layer0_W"][:, 0], 0.1 * g_w * 10.0 / norm,
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(moments["v"]["layer0_b"], [0.001 * (g_b * 10.0 / norm) ** 2],
                               rtol=1e-12, atol=1e-18)
    clipped = np.sqrt(np.sum(moments["m"]["layer0_W"] ** 2) + moments["m"]["layer0_b"][0] ** 2)
    assert clipped == pytest.approx(0.1 * 10.0, rel=1e-12)


def test_the_target_network_syncs_every_target_sync_updates():
    net = PairQNet(3, PairQConfig(hidden=(4,), target_sync=3), seed=0)
    batch = _done_batch(np.eye(3), [1.0, -1.0, 0.5])
    initial = net.target_weights()
    history = []
    for _ in range(7):
        before = net.weights()
        net.update(batch)
        online, target = net.weights(), net.target_weights()
        assert any(not np.array_equal(before[k], online[k]) for k in online)
        history.append((all(np.array_equal(online[k], target[k]) for k in online),
                        all(np.array_equal(initial[k], target[k]) for k in online)))
        if net.updates % 3 == 0:
            initial = target
    assert [synced for synced, _ in history] == [False, False, True, False, False, True, False]
    assert all(kept for synced, kept in history if not synced)
    assert (net.updates, net.syncs) == (7, 2)


def test_a_diverged_update_raises_instead_of_training_on():
    net = _linear([1e300, 1e300], 0.0)
    with pytest.raises(FloatingPointError):
        with np.errstate(over="ignore", invalid="ignore"):
            net.update(_done_batch(np.array([[1e10, 1e10]]), [0.0]))


# --------------------------------------------------------------------------- #
# Settings and behaviour
# --------------------------------------------------------------------------- #

def test_the_defaults_are_the_specs_settings():
    """Other choices 3: 2 x 64 tanh, Adam at 1e-3, Huber δ = 1, clip 10, a sync
    every 500 updates, n-step 1; batches of 64, a replay of 50,000, a warm-up of
    1,000; 500 episodes around FX at ε = 0.3, then 0.3 -> 0.05 over the first half."""
    cfg = PairQConfig()
    assert (cfg.hidden, cfg.activation, cfg.lr, cfg.huber_delta, cfg.grad_clip,
            cfg.target_sync, cfg.n_step) == ((64, 64), "tanh", 1e-3, 1.0, 10.0, 500, 1)
    assert (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps) == (0.9, 0.999, 1e-8)
    settings = LearnerSettings()
    assert (settings.batch, settings.replay_capacity, settings.warmup_transitions) == (
        64, 50_000, 1_000)
    assert settings.replay_capacity == R.DEFAULT_CAPACITY == PairReplay(seed=0).capacity
    assert settings.behaviour == BehaviourSchedule(500, 0.3, 0.05, 0.5)
    assert PairQConfig.from_json(json.loads(json.dumps(cfg.to_json()))) == cfg
    assert json.loads(json.dumps(settings.to_json()))["behaviour"]["epsilon_start"] == 0.3
    with pytest.raises(ValueError, match=r"unknown \['momentum'\]"):
        PairQConfig.from_json({**cfg.to_json(), "momentum": 0.9})
    with pytest.raises(ValueError, match=r"missing \['lr'\]"):
        PairQConfig.from_json({k: v for k, v in cfg.to_json().items() if k != "lr"})


@pytest.mark.parametrize("changes", [
    dict(gamma=-0.1), dict(gamma=1.01), dict(gamma=float("nan")), dict(gamma=True),
    dict(n_step=2), dict(n_step=0), dict(activation="relu"), dict(hidden=(64, 0)),
    dict(hidden="64"), dict(lr=0.0), dict(target_sync=0), dict(adam_beta1=1.0),
    dict(adam_eps=0.0), dict(huber_delta=-1.0), dict(grad_clip=0.0),
])
def test_the_config_refuses_what_the_learner_cannot_run(changes):
    with pytest.raises((TypeError, ValueError)):
        PairQConfig(**changes)


def test_gamma_runs_the_whole_closed_interval():
    """Study 5.5 sweeps from γ = 0 (build plan L1151), which the legacy DDQN
    refuses (ddqn.py:103-104)."""
    assert [PairQConfig(gamma=g).gamma for g in (0, 1)] == [0.0, 1.0]
    with pytest.raises(ValueError):
        DDQN(feature_dim=2, gamma=0.0)


def test_the_behaviour_flies_around_fx_then_decays_over_the_first_half():
    schedule = BehaviourSchedule()
    at = {e: schedule.at(e, 10_000) for e in (0, 499, 500, 2750, 4999, 5000, 9999)}
    assert [(b.epsilon, b.around_reference) for b in (at[0], at[499])] == [(0.3, True)] * 2
    assert (at[500].epsilon, at[500].around_reference) == (0.3, False)
    assert at[2750].epsilon == pytest.approx(0.175, rel=1e-12)
    assert at[4999].epsilon == pytest.approx(0.3 - 0.25 * 4499 / 4500, rel=1e-12)
    assert [at[e].epsilon for e in (5000, 9999)] == [0.05, 0.05]
    assert all(not at[e].around_reference for e in (500, 2750, 5000, 9999))
    short = BehaviourSchedule().at(500, 600)      # the first half ends inside the FX phase
    assert (short.epsilon, short.around_reference) == (0.05, False)
    with pytest.raises(ValueError):
        schedule.at(10_000, 10_000)
    for bad in (dict(epsilon_start=0.05, epsilon_end=0.3), dict(decay_fraction=0.0),
                dict(epsilon_start=1.5), dict(reference_episodes=-1)):
        with pytest.raises(ValueError):
            BehaviourSchedule(**bad)


@pytest.mark.parametrize("schedule, expected", [
    # E3 with no FX phase (unit U6): from 1.0 to 0.1 over the first quarter of 1,000
    (BehaviourSchedule(reference_episodes=0, epsilon_start=1.0, epsilon_end=0.1,
                       decay_fraction=0.25),
     [(0, 1.0, False), (50, 0.82, False), (125, 0.55, False), (249, 1.0 - 0.9 * 249 / 250, False),
      (250, 0.1, False), (999, 0.1, False)]),
    # 100 episodes around the reference at 0.5, then to 0.0 by episode 750
    (BehaviourSchedule(reference_episodes=100, epsilon_start=0.5, epsilon_end=0.0,
                       decay_fraction=0.75),
     [(0, 0.5, True), (99, 0.5, True), (100, 0.5, False), (425, 0.25, False),
      (749, 0.5 / 650, False), (750, 0.0, False), (999, 0.0, False)]),
], ids=["e3_no_reference", "short_reference"])
def test_a_configured_schedule_reads_every_one_of_its_settings(schedule, expected):
    """Hand values at the phase boundaries of schedules that differ from the
    defaults in all four settings, over a run of 1,000 episodes."""
    got = [(e, schedule.at(e, 1000)) for e, _, _ in expected]
    for (episode, epsilon, around), (_, behaviour) in zip(expected, got):
        assert behaviour.epsilon == pytest.approx(epsilon, rel=1e-12, abs=1e-15), episode
        assert behaviour.around_reference is around, episode
    assert BehaviourSchedule(**schedule.to_json()) == schedule


class _Draws:
    """A stream that hands out given numbers and counts them."""

    def __init__(self, *values):
        self.values, self.calls = list(values), 0

    def random(self):
        self.calls += 1
        return self.values.pop(0)


def test_exploration_is_uniform_over_the_admitted_rows_only():
    mask = np.array([True, False, True, True, False, True])
    rng = random.Random(7)
    picks = [Q.behaviour_row(np.arange(6.0), mask, epsilon=1.0, rng=rng) for _ in range(4000)]
    counts = np.bincount(picks, minlength=6)
    assert counts[~mask].tolist() == [0, 0]
    assert all(850 < c < 1150 for c in counts[mask])
    assert Q.behaviour_row(np.zeros(6), mask, epsilon=0.5, rng=_Draws(0.4, 0.0)) == 0
    assert Q.behaviour_row(np.zeros(6), mask, epsilon=0.5, rng=_Draws(0.4, 0.999)) == 5


def test_greedy_is_fxs_pair_when_admitted_and_else_the_q_argmax():
    scores = np.array([0.0, 5.0, 1.0, 4.0])
    mask = np.array([True, False, True, True])
    for reference, row in ((2, 2), (1, 3), (None, 3)):   # row 1 is masked: Q decides
        draws = _Draws(0.99, 0.5)
        assert Q.behaviour_row(scores, mask, epsilon=0.3, rng=draws, reference=reference) == row
        assert draws.calls == 2
    draws = _Draws(0.1, 0.5)                             # explores: two draws as well
    Q.behaviour_row(scores, mask, epsilon=0.3, rng=draws, reference=2)
    assert draws.calls == 2
    with pytest.raises(ValueError):
        Q.behaviour_row(scores, np.zeros(4, dtype=bool), epsilon=0.3, rng=_Draws(0.5, 0.5))
    with pytest.raises(ValueError):
        Q.behaviour_row(scores, mask, epsilon=1.3, rng=_Draws(0.5, 0.5))


@pytest.mark.parametrize("draw", [0.1, 0.99], ids=["explores", "greedy"])
@pytest.mark.parametrize("reference, error", [
    (4, ValueError), (99, ValueError), (-1, ValueError), (True, TypeError), (2.0, TypeError),
    ("1", TypeError)])
def test_a_wrong_reference_is_refused_before_any_draw(reference, error, draw):
    """A wiring error (FX's row taken from another pair list) fails at every
    decision, not only at the greedy ones: the reference is checked first."""
    draws = _Draws(draw, 0.5)
    with pytest.raises(error):
        Q.behaviour_row(np.zeros(4), np.ones(4, dtype=bool), epsilon=0.3, rng=draws,
                        reference=reference)
    assert draws.calls == 0


def test_the_learner_updates_once_per_decision_after_the_warm_up():
    settings = LearnerSettings(batch=4, replay_capacity=50, warmup_transitions=10)
    net = PairQNet(2, PairQConfig(hidden=(3,)), seed=0)
    learner = PairQLearner(net, PairReplay(50, seed=1), settings)
    losses = [learner.observe(PairTransition(np.array([float(i), 1.0]), 1.0, True))
              for i in range(12)]
    assert losses[:9] == [None] * 9 and all(isinstance(x, float) for x in losses[9:])
    assert net.updates == 3 and learner.warm
    with pytest.raises(ValueError, match="manifest records the settings"):
        PairQLearner(net, PairReplay(49, seed=1), settings)
    with pytest.raises(ValueError):
        LearnerSettings(batch=64, warmup_transitions=10)


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #

def test_a_transition_is_copied_and_read_only():
    x, rows, mask = np.ones(3), np.ones((2, 3)), np.array([True, False])
    t = PairTransition(x, 1, False, rows, mask)
    x[0], rows[0, 0], mask[1] = 9.0, 9.0, True
    assert (t.x[0], t.next_rows[0, 0], bool(t.next_mask[1]), t.reward) == (1.0, 1.0, False, 1.0)
    for array in (t.x, t.next_rows, t.next_mask):
        with pytest.raises(ValueError):
            array[0] = 0


@pytest.mark.parametrize("args, error", [
    ((np.ones(2), 1.0, False), ValueError),                                # no next decision
    ((np.ones(2), 1.0, False, np.ones((2, 2)), np.array([False, False])), ValueError),
    ((np.ones(2), 1.0, False, np.ones((2, 2)), np.array([1, 0])), TypeError),
    ((np.ones(2), 1.0, False, np.ones((2, 3)), np.array([True, True])), ValueError),
    ((np.ones(2), 1.0, False, np.ones((0, 2)), np.array([], dtype=bool)), ValueError),
    ((np.ones(2), 1.0, False, np.array([[np.nan, 0.0]]), np.array([True])), ValueError),
    ((np.array([np.inf, 0.0]), 1.0, True), ValueError),
    ((np.ones(2), float("nan"), True), ValueError),
    ((np.ones(2), True, True), TypeError),
    ((np.ones(2), 1.0, 1), TypeError),
    ((np.ones(0), 1.0, True), ValueError),
])
def test_a_transition_refuses_what_a_target_cannot_use(args, error):
    with pytest.raises(error):
        PairTransition(*args)


def test_a_batch_concatenates_the_candidate_sets_in_transition_order():
    a = PairTransition(np.zeros(2), 1.0, False, np.array([[1.0, 0], [2.0, 0]]),
                       np.array([True, False]))
    b = PairTransition(np.zeros(2), 2.0, True)
    c = PairTransition(np.ones(2), 3.0, False, np.array([[3.0, 0], [4.0, 0], [5.0, 0]]),
                       np.array([False, True, True]))
    batch = PairBatch.of([a, b, c])
    assert (batch.size, batch.feature_dim) == (3, 2)
    assert batch.next_rows[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert batch.segment.tolist() == [0, 0, 2, 2, 2]
    assert batch.next_mask.tolist() == [True, False, False, True, True]
    assert batch.done.tolist() == [False, True, False]
    assert PairBatch.of([b]).next_rows.shape == (0, 2)
    with pytest.raises(ValueError, match="one width"):
        PairBatch.of([a, PairTransition(np.zeros(3), 1.0, True)])


@pytest.mark.parametrize("changes, match", [
    (dict(segment=np.array([1, 0])), "non-decreasing"),
    (dict(segment=np.array([0, 2])), "non-decreasing"),
    (dict(next_mask=np.array([False, False])), "admitted row"),
    (dict(done=np.array([True, True])), "done transition owns no next rows"),
    (dict(reward=np.array([1.0])), "one finite reward"),
])
def test_a_batch_refuses_inconsistent_arrays(changes, match):
    arrays = dict(x=np.zeros((2, 2)), reward=np.zeros(2), done=np.array([False, False]),
                  next_rows=np.ones((2, 2)), next_mask=np.array([True, True]),
                  segment=np.array([0, 1]))
    arrays.update(changes)
    with pytest.raises(ValueError, match=match):
        PairBatch(**arrays)


def _filled(seed, n=100):
    replay = PairReplay(1000, seed=seed)
    for i in range(n):
        replay.push(PairTransition(np.array([float(i), 0.0]), float(i), True))
    return replay


def test_sampling_is_deterministic_under_a_seed():
    a, b, other = _filled(5), _filled(5), _filled(6)
    draws_a = [a.sample(16).reward.tolist() for _ in range(5)]
    assert draws_a == [b.sample(16).reward.tolist() for _ in range(5)]
    assert draws_a != [other.sample(16).reward.tolist() for _ in range(5)]
    assert all(len(set(draw)) == 16 for draw in draws_a)          # without replacement
    stream = random.Random(5)                                     # the legacy buffer's draw
    assert draws_a == [[float(i) for i in stream.sample(range(100), 16)] for _ in range(5)]
    assert sorted(_filled(5, 10).sample(10).reward.tolist()) == [float(i) for i in range(10)]
    with pytest.raises(ValueError):
        _filled(5, 10).sample(11)


def test_the_ring_overwrites_the_oldest_first():
    replay = PairReplay(3, seed=0)
    for i in range(5):
        replay.push(PairTransition(np.array([1.0]), float(i), True))
    assert (len(replay), replay.pushed) == (3, 5)
    assert sorted(replay.sample(3).reward.tolist()) == [2.0, 3.0, 4.0]
    with pytest.raises(ValueError, match="1 wide"):
        replay.push(PairTransition(np.array([1.0, 2.0]), 0.0, True))
    with pytest.raises(TypeError):
        replay.push((np.ones(1), 0.0, True))
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        PairReplay(0, seed=0)


# --------------------------------------------------------------------------- #
# Checkpoints (format 2)
# --------------------------------------------------------------------------- #

CLASSES = ["wide", "medium"]
SCHEMA = {"version": "pair_test", "dim": 5, "classes": CLASSES, "phase": True}


def _provenance(**changes):
    out = dict(reward={"kind": "derived", "c_t": 0.1, "c_e": 0.0, "c_cov": 1.0},
               training={"episodes": 10, "learner": LearnerSettings().to_json()},
               seeds={"init": 1, "replay": 2, "behaviour": 3}, cell_family="n12_jittery",
               cell_family_sha256="ab" * 32, trainer_commit="386c275", dirty=False,
               episodes_trained=10, validation=[{"episode": 10, "return": 0.5}], held_out=None)
    out.update(changes)
    return out


def _saved(tmp_path, name="g0.9_s1.npz", net=None, **changes):
    net = PairQNet(5, PairQConfig(hidden=(8, 4), gamma=0.9), seed=1) if net is None else net
    args = dict(kind="pair_q", purpose="trained", schema=SCHEMA, classes=CLASSES,
                provenance=_provenance())
    args.update(changes)
    path = tmp_path / name
    return net, path, net.save(path, **args)


def _load(path, sha, **changes):
    args = dict(expect_sha256=sha, expect_kind="pair_q", expect_schema=SCHEMA,
                expect_classes=CLASSES)
    args.update(changes)
    return PairQNet.load(path, **args)


def _edit_manifest(path, **changes):
    where = Q.manifest_path(path)
    manifest = json.loads(where.read_text(encoding="utf-8"))
    manifest.update(changes)
    where.write_text(json.dumps(manifest), encoding="utf-8")


def _rewrite_arrays(path, edit):
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    edit(arrays)
    np.savez(path, **arrays)


def _arrays_sha(path):
    with np.load(path) as data:
        return Q.arrays_sha256({name: data[name] for name in data.files})


def _header_of(path):
    """A checkpoint's header array, decoded."""
    with np.load(path) as data:
        return json.loads(data["header"].tobytes().decode("ascii"))


def _set_header(arrays, **changes):
    """Rewrite the arrays' header with ``changes``, in save's canonical JSON; a
    None value drops the key."""
    header = json.loads(arrays["header"].tobytes().decode("ascii"))
    header.update(changes)
    text = json.dumps({k: v for k, v in header.items() if v is not None}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)
    arrays["header"] = np.frombuffer(text.encode("ascii"), dtype=np.uint8).copy()


def test_a_format_2_checkpoint_round_trips(tmp_path):
    net, path, sha = _saved(tmp_path)
    for _ in range(3):
        net.update(_done_batch(np.eye(5), np.arange(5.0)))
    sha = net.save(path, kind="pair_q", purpose="trained", schema=SCHEMA, classes=CLASSES,
                   provenance=_provenance())
    loaded, manifest = _load(path, sha)
    assert all(np.array_equal(net.weights()[k], loaded.weights()[k]) for k in net.weights())
    assert all(np.array_equal(loaded.weights()[k], loaded.target_weights()[k])
               for k in net.weights())
    rows = np.random.default_rng(0).normal(size=(7, 5))
    assert np.array_equal(net.q(rows), loaded.q(rows))
    assert loaded.config == net.config and loaded.updates == 0
    assert set(manifest) == set(Q.MANIFEST_KEYS)
    assert (manifest["format"], manifest["kind"], manifest["purpose"], manifest["sha256"],
            manifest["gamma"], manifest["learner_revision"]) == (
        2, "pair_q", "trained", sha, 0.9, Q.LEARNER_REVISION)
    assert manifest["network"] == {"feature_dim": 5, **net.config.to_json()}
    assert (manifest["schema"], manifest["classes"]) == (SCHEMA, CLASSES)
    assert _header_of(path) == {key: manifest[key] for key in Q.HEADER_KEYS}
    assert {k: manifest[k] for k in Q.PROVENANCE_KEYS} == json.loads(json.dumps(_provenance()))
    assert manifest["numpy"] == np.__version__ and "OPENBLAS_NUM_THREADS=" in manifest["blas"]
    assert Q.read_manifest(path) == manifest == Q.verify_checkpoint(path)
    text = Q.manifest_path(path).read_bytes()
    assert Q.manifest_path(path).name == "g0.9_s1.json"
    assert text.endswith(b"}\n") and b"\r" not in text and text.isascii()
    assert list(json.loads(text)) == sorted(json.loads(text))
    again = tmp_path / "again" / "g0.9_s1.npz"
    assert net.save(again, kind="pair_q", purpose="trained", schema=SCHEMA, classes=CLASSES,
                    provenance=_provenance()) == sha
    assert Q.manifest_path(again).read_bytes() == text       # no wall time in a manifest
    assert sorted(p.name for p in tmp_path.iterdir()) == ["again", "g0.9_s1.json",
                                                          "g0.9_s1.npz"]


def test_e3_checkpoints_are_their_own_kind(tmp_path):
    net = PairQNet(5, PairQConfig(hidden=(4,), gamma=0.99), seed=2)
    _, path, sha = _saved(tmp_path, net=net, kind="chen_dqn", purpose="bootstrap",
                          schema={"version": "chen_test", "dim": 5}, classes=["medium"],
                          provenance=_provenance(episodes_trained=0, validation=[]))
    loaded, manifest = _load(path, sha, expect_kind="chen_dqn",
                             expect_schema={"version": "chen_test", "dim": 5},
                             expect_classes=["medium"])
    assert manifest["purpose"] == "bootstrap" and loaded.config.hidden == (4,)
    with pytest.raises(CheckpointError, match="'chen_dqn' checkpoint"):
        _load(path, sha, expect_schema={"version": "chen_test", "dim": 5},
              expect_classes=["medium"])


def test_the_sha_covers_the_weights_and_what_they_read(tmp_path, monkeypatch):
    """Each header field changes the sha, the purpose included (resolution R2;
    the revision is the next test's). No provenance field changes it, nor the
    BLAS report, so the held-out score is filled in under the sha a config
    already names."""
    net = PairQNet(5, PairQConfig(hidden=(8, 4), gamma=0.9), seed=1)
    shas = {_saved(tmp_path, f"a{i}.npz", net=net, **changes)[2]
            for i, changes in enumerate([
                {}, dict(kind="chen_dqn"), dict(classes=["medium", "wide"],
                                                schema={**SCHEMA, "classes": ["medium", "wide"]}),
                dict(schema={**SCHEMA, "phase": False}), dict(purpose="bootstrap")])}
    other = PairQNet(5, PairQConfig(hidden=(8, 4), gamma=0.5), seed=1)
    shas.add(_saved(tmp_path, "g.npz", net=other)[2])
    assert len(shas) == 6
    base = _saved(tmp_path, "c.npz", net=net)[2]
    monkeypatch.setattr(Q, "_blas", lambda: "another BLAS; OPENBLAS_NUM_THREADS=8")
    scored = {"episodes": 1000, "return_mean": 0.25}
    same = {_saved(tmp_path, f"b{i}.npz", net=net, provenance=_provenance(**changes))[2]
            for i, changes in enumerate([
                dict(held_out=scored), dict(episodes_trained=0, validation=[]),
                dict(dirty=True, trainer_commit=None), dict(seeds={"init": 9}),
                dict(cell_family=None, cell_family_sha256=None)])}
    assert Q.read_manifest(tmp_path / "b0.npz")["blas"].startswith("another BLAS")
    assert same == {base} == {_arrays_sha(tmp_path / "c.npz")}  # the provenance is not hashed


def test_the_header_binds_the_purpose_and_the_learner_revision(tmp_path, monkeypatch):
    """Resolution R2: the purpose and the learner's revision are header fields,
    so the sha covers them. One network saved as bootstrap, as trained, and as
    trained by the next revision of the learner makes three checkpoints with
    three shas, and the sha a config names loads its own checkpoint only."""
    assert Q.HEADER_KEYS == ("format", "kind", "purpose", "learner_revision", "network",
                             "schema", "classes")
    assert Q.WRITER_KEYS == ("sha256", "gamma", "numpy", "blas")
    assert len(set(Q.MANIFEST_KEYS)) == len(Q.MANIFEST_KEYS) == 21
    net = PairQNet(5, PairQConfig(hidden=(8, 4), gamma=0.9), seed=1)
    saved = {}
    for purpose, revision in (("trained", 0), ("bootstrap", 0), ("trained", 1)):
        monkeypatch.setattr(Q, "LEARNER_REVISION", revision)
        _, path, sha = _saved(tmp_path, f"{purpose}_r{revision}.npz", net=net, purpose=purpose)
        loaded, manifest = _load(path, sha)
        header = _header_of(path)
        assert (header["purpose"], header["learner_revision"]) == (purpose, revision)
        assert header == {key: manifest[key] for key in Q.HEADER_KEYS}
        assert all(np.array_equal(loaded.weights()[k], w) for k, w in net.weights().items())
        saved[path] = sha
    assert len(set(saved.values())) == 3
    for path, own in saved.items():
        for sha in set(saved.values()) - {own}:
            with pytest.raises(CheckpointError, match="not the expected"):
                _load(path, sha)


@pytest.mark.parametrize("purpose, revision, relabel", [
    ("bootstrap", 0, dict(purpose="trained")),
    ("trained", 0, dict(purpose="bootstrap")),
    ("trained", 0, dict(learner_revision=1)),
    ("trained", 1, dict(learner_revision=0)),
], ids=["bootstrap_as_trained", "trained_as_bootstrap", "revision_0_as_1", "revision_1_as_0"])
def test_a_relabelled_purpose_or_revision_is_refused(tmp_path, monkeypatch, purpose, revision,
                                                     relabel):
    """Resolution R2. A manifest whose purpose or learner revision was edited is
    no longer its arrays' header's, so every verified read refuses it: the
    loader, verify_checkpoint (whose manifest the runner judges) and
    record_held_out, which then writes nothing. Read unverified (read_manifest
    checks the form only), a bootstrap relabelled as trained and scored by hand
    would pass campaign_refusals. Rewriting the header as well makes new
    arrays, so the sha a config names still loads only the original."""
    monkeypatch.setattr(Q, "LEARNER_REVISION", revision)
    _, path, sha = _saved(tmp_path, purpose=purpose)
    monkeypatch.undo()
    [(key, value)] = relabel.items()
    _edit_manifest(path, **relabel, held_out={"episodes": 1000, "return_mean": 0.5})
    edited = Q.manifest_path(path).read_bytes()
    match = f"the manifest's {key} is not the arrays' header's"
    with pytest.raises(CheckpointError, match=match):
        Q.verify_checkpoint(path)
    with pytest.raises(CheckpointError, match=match):
        _load(path, sha)
    with pytest.raises(CheckpointError, match=match):
        Q.record_held_out(path, {"episodes": 1000, "return_mean": 0.9})
    assert Q.manifest_path(path).read_bytes() == edited
    unverified = Q.read_manifest(path)
    assert unverified[key] == value
    if relabel == {"purpose": "trained"}:
        assert Q.campaign_refusals(unverified) == []
    _rewrite_arrays(path, lambda arrays: _set_header(arrays, **relabel))
    with pytest.raises(CheckpointError, match="not its manifest's"):
        Q.verify_checkpoint(path)
    resha = _arrays_sha(path)
    _edit_manifest(path, sha256=resha)
    assert resha != sha and Q.verify_checkpoint(path)[key] == value
    with pytest.raises(CheckpointError, match="not the expected"):
        _load(path, sha)


def test_a_header_written_before_the_binding_is_refused(tmp_path):
    """A format-2 checkpoint as written before resolution R2: a header without
    the purpose and the revision, and a manifest whose sha is its arrays'. It is
    refused as a broken header that names the keys it has, so no checkpoint
    flies with a purpose the sha does not bind."""
    _, path, _ = _saved(tmp_path)
    _rewrite_arrays(path, lambda arrays: _set_header(arrays, purpose=None,
                                                     learner_revision=None))
    sha = _arrays_sha(path)
    _edit_manifest(path, sha256=sha)
    for read in (Q.verify_checkpoint, lambda where: _load(where, sha)):
        with pytest.raises(CheckpointError, match=r"broken header: header keys are \[.*\], got "
                                                  r"\['classes', 'format', 'kind', 'network', "
                                                  r"'schema'\]"):
            read(path)


def test_the_legacy_ddqn_loader_refuses_format_2(tmp_path):
    _, path, _ = _saved(tmp_path)
    with pytest.raises(ValueError, match="format_version=2, expected 1"):
        DDQN.load(path)


def test_the_pair_loader_refuses_format_1(tmp_path):
    _, path, sha = _saved(tmp_path)
    legacy = tmp_path / "legacy.npz"
    DDQN(feature_dim=5, hidden=4, seed=0).save(legacy)
    with pytest.raises(CheckpointError, match="format 1 .the legacy DDQN"):
        _load(legacy, sha)
    shutil.copy(Q.manifest_path(path), Q.manifest_path(legacy))   # a format-2 manifest too
    with pytest.raises(CheckpointError, match="format 1 .the legacy DDQN"):
        _load(legacy, sha)


@pytest.mark.parametrize("changes, match", [
    (dict(expect_sha256="0" * 64), "not the expected"),
    (dict(expect_kind="chen_dqn"), "not 'chen_dqn'"),
    (dict(expect_classes=["medium", "wide"]), "trained on the classes"),
    (dict(expect_classes=["wide", "medium", "narrow"]), "trained on the classes"),
    (dict(expect_schema={**SCHEMA, "version": "pair_v1"}), r"keys that differ: \['version'\]"),
    (dict(expect_schema={**SCHEMA, "phase": False}), r"keys that differ: \['phase'\]"),
    (dict(expect_schema={k: v for k, v in SCHEMA.items() if k != "phase"}), "phase"),
])
def test_the_loader_refuses_what_the_caller_did_not_expect(tmp_path, changes, match):
    _, path, sha = _saved(tmp_path)
    with pytest.raises(CheckpointError, match=match):
        _load(path, sha, **changes)


def test_the_loader_refuses_a_missing_or_inconsistent_manifest(tmp_path):
    _, path, sha = _saved(tmp_path)
    original = Q.manifest_path(path).read_text(encoding="utf-8")
    network = json.loads(original)["network"]
    cases = [
        (dict(format=3), "format 3: format 2 only"),
        (dict(format=1), "format 1: format 2 only"),
        (dict(gamma=0.5), "the network's 0.9"),
        (dict(sha256="0" * 64), "not its manifest's"),
        (dict(kind="chen_dqn"), "manifest's kind is not the arrays' header's"),
        (dict(network={**network, "lr": 0.01}), "manifest's network is not the arrays' header's"),
        (dict(network={**network, "momentum": 0.9}), r"unknown \['momentum'\]"),
        (dict(classes=["medium", "wide"]), "manifest's classes is not the arrays' header's"),
        (dict(schema={**SCHEMA, "phase": False}), "manifest's schema is not"),
        (dict(purpose="final"), "purpose is one of"),
        (dict(dirty="no"), "dirty is a bool"),
        (dict(learner_revision="0"), "learner_revision is an int"),
        (dict(learner_revision=-1), "learner_revision is an int"),
        (dict(numpy=1), "numpy is a string"),
        (dict(blas=None), "blas is a string"),
        (dict(wall_s=1.0), "unknown"),
    ]
    for changes, match in cases:
        Q.manifest_path(path).write_text(original, encoding="utf-8")
        _edit_manifest(path, **changes)
        with pytest.raises(CheckpointError, match=match):
            _load(path, sha)
    Q.manifest_path(path).write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="not a JSON manifest"):
        _load(path, sha)
    Q.manifest_path(path).unlink()
    with pytest.raises(CheckpointError, match="no manifest beside it"):
        _load(path, sha)
    with pytest.raises(FileNotFoundError):
        _load(tmp_path / "absent.npz", sha)
    with pytest.raises(ValueError, match=r"\.npz file"):
        _load(tmp_path / "g0.9_s1.json", sha)


@pytest.mark.parametrize("edit, match", [
    (lambda a: a.update(layer0_W=a["layer0_W"] + 1e-12), "not its manifest's"),
    (lambda a: a.update(format_version=np.array(3)), "format 3;"),
    (lambda a: a.pop("format_version"), "no format_version"),
    (lambda a: a.update(extra=np.zeros(1)), "not the header's network's"),
    (lambda a: a.pop("layer1_b"), "not the header's network's"),
    (lambda a: a.update(layer0_W=a["layer0_W"].astype(np.float32)), "float64"),
    (lambda a: a.update(layer0_W=np.ascontiguousarray(a["layer0_W"][:, :3])),
     r"of shape \(5, 8\)"),
    (lambda a: a.update(layer0_W=np.full_like(a["layer0_W"], np.nan)), "finite"),
    (lambda a: a.update(header=np.frombuffer(b"{]", dtype=np.uint8)), "broken header"),
])
def test_the_loader_refuses_arrays_that_are_not_a_format_2_checkpoints(tmp_path, edit, match):
    _, path, sha = _saved(tmp_path)
    _rewrite_arrays(path, edit)
    with pytest.raises(CheckpointError, match=match):
        _load(path, sha)


def test_the_loader_refuses_files_that_are_no_archive(tmp_path):
    _, path, sha = _saved(tmp_path)
    path.write_bytes(b"not a zip at all")
    with pytest.raises(CheckpointError, match="not a format-2 checkpoint"):
        _load(path, sha)
    with open(path, "wb") as handle:
        np.save(handle, np.zeros(3))
    with pytest.raises(CheckpointError, match="one array, not an archive"):
        _load(path, sha)


@pytest.mark.parametrize("changes, match", [
    (dict(provenance=_provenance(wall_s=12.0)), "unknown"),
    (dict(provenance={k: v for k, v in _provenance().items() if k != "seeds"}), "missing"),
    (dict(provenance=_provenance(training={"started_wall": 1.0})), "wall time"),
    (dict(provenance=_provenance(episodes_trained=-1)), "episodes_trained"),
    (dict(provenance=_provenance(cell_family_sha256="abc")), "cell_family_sha256"),
    (dict(provenance=_provenance(reward=[1, 2])), "reward is a JSON object"),
    (dict(provenance=_provenance(validation=[float("nan")])), "finite numbers"),
    (dict(schema={**SCHEMA, "dim": 6}), "6 wide"),
    (dict(schema={"dim": 5}), "names its version"),
    (dict(schema={**SCHEMA, "classes": ["medium", "wide"]}), "schema's classes"),
    (dict(classes=["wide", "wide"]), "distinct"),
    (dict(kind="pair_x"), "kind must be one of"),
    (dict(purpose="final"), "purpose must be one of"),
    (dict(name="g0.9_s1.pt"), r"\.npz file"),
])
def test_save_refuses_a_checkpoint_it_could_not_describe(tmp_path, changes, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _saved(tmp_path, **changes)
    assert list(tmp_path.iterdir()) == []


def test_a_held_out_score_is_written_only_beside_its_own_arrays(tmp_path):
    _, path, sha = _saved(tmp_path)
    manifest = Q.record_held_out(path, {"episodes": 1000, "return_mean": 0.42})
    assert manifest["held_out"] == {"episodes": 1000, "return_mean": 0.42}
    assert Q.read_manifest(path) == manifest and _load(path, sha)[1] == manifest
    assert manifest["sha256"] == sha
    with pytest.raises(ValueError, match="wall time"):
        Q.record_held_out(path, {"scored_wall": 3.0})
    _rewrite_arrays(path, lambda a: a.update(layer0_b=a["layer0_b"] + 1.0))
    with pytest.raises(CheckpointError, match="not its manifest's"):
        Q.record_held_out(path, {"episodes": 1000, "return_mean": 1.0})
    assert Q.read_manifest(path)["held_out"] == {"episodes": 1000, "return_mean": 0.42}


def test_the_runner_refuses_untrained_unscored_or_dirty_checkpoints(tmp_path):
    """Critic B9: the refusals are the runner's; the loader applies none."""
    _, path, _ = _saved(tmp_path)
    manifest = Q.record_held_out(path, {"episodes": 1000, "return_mean": -0.4,
                                        "return_sd": 0.1, "salt": "ferrysim-heldout"})
    assert Q.campaign_refusals(manifest) == []
    cases = {
        "purpose": dict(purpose="bootstrap"),
        "trained on 0 episodes": dict(episodes_trained=0),
        "no held-out score": dict(held_out=None),
        "dirty tree": dict(dirty=True),
    }
    for match, changes in cases.items():
        reasons = Q.campaign_refusals({**manifest, **changes})
        assert len(reasons) == 1 and match in reasons[0]
    assert Q.campaign_refusals({**manifest, "dirty": True}, allow_dirty=True) == []
    with pytest.raises(CheckpointError):
        Q.campaign_refusals({**manifest, "format": 1})
    bootstrap = _saved(tmp_path, "boot.npz", purpose="bootstrap",
                       provenance=_provenance(episodes_trained=0, validation=[]))
    assert _load(bootstrap[1], bootstrap[2])[1]["purpose"] == "bootstrap"   # it loads


@pytest.mark.parametrize("held_out", [
    {}, {"return_mean": None}, {"episodes": 1000}, {"return_mean": 0.4},
    {"episodes": 0, "return_mean": 0.4}, {"episodes": True, "return_mean": 0.4},
    {"episodes": 1000.0, "return_mean": 0.4}, {"episodes": 1000, "return_mean": True},
    {"episodes": 1000, "return_mean": "0.4"}, [0.4],
])
def test_a_held_out_score_holds_its_episodes_and_mean_return(tmp_path, held_out):
    """Critic B9's "no held-out score" is decided on the score: an entry without
    the episodes scored and their mean undiscounted return (HELD_OUT_KEYS) is
    refused when it is recorded, saved, read or judged, so the runner never
    flies a checkpoint that was not scored."""
    _, path, sha = _saved(tmp_path)
    with pytest.raises((TypeError, ValueError), match="held_out"):
        Q.record_held_out(path, held_out)
    manifest = Q.read_manifest(path)
    assert manifest["held_out"] is None
    with pytest.raises(ValueError, match="held_out"):
        _saved(tmp_path, "scored.npz", provenance=_provenance(held_out=held_out))
    with pytest.raises(CheckpointError, match="held_out"):
        Q.campaign_refusals({**manifest, "held_out": held_out})
    _edit_manifest(path, held_out=held_out)
    with pytest.raises(CheckpointError, match="held_out"):
        _load(path, sha)


def test_the_mule_announces_the_manifests_provenance(tmp_path):
    _, path, sha = _saved(tmp_path)
    announced = Q.manifest_provenance(Q.read_manifest(path))
    assert announced == {
        "sha256": sha, "kind": "pair_q", "purpose": "trained", "classes": CLASSES,
        "gamma": 0.9, "reward": _provenance()["reward"], "seeds": _provenance()["seeds"],
        "episodes_trained": 10, "cell_family": "n12_jittery", "cell_family_sha256": "ab" * 32,
        "learner_revision": Q.LEARNER_REVISION, "schema": "pair_test"}
    assert "wall" not in json.dumps(announced)


# --------------------------------------------------------------------------- #
# Learning: the known-answer pair (critic B15)
# --------------------------------------------------------------------------- #

#: Five one-hot rows: at s0 the myopic action M pays 1.0 and the patient P
#: 0.8; one leads to a good state (G_A pays 4.0, G_B 1.0, so its value is the
#: max, 4.0), the other to a bad one (B_A pays 0, beside a masked row that
#: looks exactly like G_A: a learner that let it into the max would value the
#: bad state at 4.0 too).
_E = np.eye(6)
M, P, G_A, G_B, B_A = _E[0], _E[1], _E[2], _E[3], _E[4]


def _chain(myopic_is_better: bool):
    good = (np.stack([G_A, G_B]), np.array([True, True]))
    bad = (np.stack([B_A, G_A]), np.array([True, False]))
    return [PairTransition(M, 1.0, False, *(good if myopic_is_better else bad)),
            PairTransition(P, 0.8, False, *(bad if myopic_is_better else good)),
            PairTransition(G_A, 4.0, True), PairTransition(G_B, 1.0, True),
            PairTransition(B_A, 0.0, True)]


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("myopic_is_better", [False, True])
def test_only_a_far_sighted_learner_takes_the_patient_first_action(myopic_is_better, gamma):
    """The spec's own learner settings (2 x 64 tanh, Adam 1e-3, Huber, clip 10,
    a sync every 500 updates, batches of 64), 1,000 updates. Where the patient
    action is better, γ = 0 takes the myopic one and every γ > 0 of Study 5.5's
    grid the patient one; where the myopic action is optimal every γ takes it.
    Each learned Q is its analytic value: r + γ · (the max over the admitted
    next rows), so the max and the mask are both at work."""
    net = PairQNet(6, PairQConfig(gamma=gamma), seed=0)
    replay = PairReplay(500, seed=0)
    for _ in range(100):
        for t in _chain(myopic_is_better):
            replay.push(t)
    for _ in range(1000):
        net.update(replay.sample(64))
    q = net.q(np.stack([M, P]))
    best_after_myopic, best_after_patient = (4.0, 0.0) if myopic_is_better else (0.0, 4.0)
    np.testing.assert_allclose(q, [1.0 + gamma * best_after_myopic,
                                   0.8 + gamma * best_after_patient], atol=0.05)
    picks_patient = not myopic_is_better and gamma > 0
    assert net.masked_argmax(np.stack([M, P]), np.array([True, True])) == int(picks_patient)


# --------------------------------------------------------------------------- #
# Layering and the legacy selector
# --------------------------------------------------------------------------- #

def _imports(path: Path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return names


@pytest.mark.parametrize("module", ["pair_q", "pair_replay"])
def test_the_learner_and_replay_import_only_numpy_and_the_standard_library(module):
    names = _imports(REPO / f"hermes/scheduler/selector/{module}.py")
    assert "numpy" in names
    assert all(n == "numpy" or n.split(".")[0] in sys.stdlib_module_names for n in names), names


def test_no_recorded_path_loads_the_learner_and_it_loads_no_plan():
    """The selector package (every D arm and H2 load it) loads neither module,
    and the two load nothing beyond themselves: no plan package, so E3 can use
    them, and nothing of l1, mule, mission, experiments or tests."""
    package = _imports(REPO / "hermes/scheduler/selector/__init__.py")
    assert not {".pair_q", ".pair_replay"} & package
    code = (
        "import sys; import hermes.scheduler.selector; base = set(sys.modules); "
        "import hermes.scheduler.selector.pair_q, hermes.scheduler.selector.pair_replay; "
        "ours = ('hermes', 'experiments', 'tests'); "
        "far = ('hermes.scheduler.plan', 'hermes.l1', 'hermes.mule', 'hermes.mission', "
        "'experiments', 'tests'); "
        "print([sorted(m for m in base if m.endswith(('.pair_q', '.pair_replay'))), "
        "sorted(m for m in set(sys.modules) - base if m.startswith(ours)), "
        "sorted(m for m in sys.modules if m.startswith(far))])"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == (
        "[[], ['hermes.scheduler.selector.pair_q', 'hermes.scheduler.selector.pair_replay'], []]")


#: The legacy selector files the H2 golden and the recorded A/B failure pin
#: (the Phase 5 spec, "Untouched"); the pair learner is new code beside them.
LEGACY_SELECTOR = tuple(f"hermes/scheduler/selector/{name}.py" for name in (
    "ddqn", "replay", "features", "target_selector_rl", "selector_train", "sim_env", "__init__"))


def test_the_legacy_selector_files_are_386c275s():
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    if subprocess.run(["git", "cat-file", "-e", "386c275^{commit}"], cwd=REPO,
                      capture_output=True, timeout=60).returncode != 0:
        pytest.skip("commit 386c275 is not in this checkout")
    out = subprocess.run(["git", "diff", "--quiet", "386c275", "--", *LEGACY_SELECTOR], cwd=REPO,
                         capture_output=True, timeout=60)
    assert out.returncode == 0, out.stderr.decode(errors="replace")
