"""FeRRy Phase 5 (unit U2): the pair learner, a masked pointer double DQN in numpy.

**What it scores.** At each Pass-1 arrival the flight slot ranks the
admitted (band, next stop) pairs (the Phase 5 spec, other choices 1-3). The
pair features (unit U1) give one row per pair, pooled context included, and
this network gives one scalar Q per row from shared weights: a pointer
score, as the legacy DDQN's (``ddqn.py:1-15``), so the candidate set can
grow and shrink with the remainder. E3 (unit U6) uses the same network over
one row per candidate stop, under its own checkpoint kind.

**Why not the legacy DDQN** (design finding 5). It refuses γ = 0 and γ = 1
(``ddqn.py:103-104``) while Study 5.5 sweeps from 0 (build plan L1151); it
trains with plain SGD on a squared loss (``ddqn.py:188-211``); its target
scores the one next row stored with the transition, so it never takes a max
(``ddqn.py:171-186``); and the H2 golden pins its seeded init and forward
pass (``tests/golden/_mule_harness.py:600-602``). So this is new code, and
the legacy file is untouched (the spec, "Untouched").

**The update** (the spec, other choices 3; design D-C (a)):

* the target is y = r + γ · Q_target(s', a*), where a* is the argmax of
  Q_online over the next decision's admitted rows, ties to the lowest row:
  double DQN (van Hasselt, Guez and Silver, AAAI 2016), the online network
  choosing and the target network valuing. The next rows and their mask
  come with each transition (``pair_replay``), and only the admitted rows
  are ever forwarded, so a masked row cannot enter a target. γ ∈ [0, 1];
  γ = 0 and a done transition (the sortie's last decision; the flight Q's
  horizon is the sortie, memo L264) give y = r. n-step 1;
* the loss is the Huber loss (δ = 1) of Q_online(s, a) − y, averaged over
  the batch, with y held fixed (a semi-gradient);
* the gradient is clipped to a global norm of 10, then Adam takes a step
  (lr 1e-3; β1, β2 and ε at Kingma and Ba's defaults, which the spec does
  not change);
* the target network is a hard copy of the online one, refreshed every 500
  updates.

**Behaviour** (critic C4). Plain double DQN on FX-only transitions would
maximise over pairs it never trained on, and ε from 1.0 would fly noise. So
for the first 500 episodes the mule flies ε-greedy around FX's pair at ε =
0.3, and after that ε-greedy on Q, with ε falling linearly from 0.3 to 0.05
over the first half of training (:class:`BehaviourSchedule`). Exploration is
uniform over the admitted pairs only (:func:`behaviour_row`). Batches of 64,
one update per decision once 1,000 transitions are stored, in a replay of
50,000 (:class:`LearnerSettings`, :class:`PairQLearner`).

**Checkpoints: format 2** (the spec, other choices 6; design D-I). A
checkpoint is an ``.npz`` of the online weights (float64), its format
(``format_version`` = 2) and a header (canonical JSON as bytes: the kind,
the purpose (``bootstrap`` | ``trained``), the learner's revision, the
network and its settings, the feature schema and the class tuple), with a
JSON manifest beside it (same name, ``.json``). The sha256 is taken over
every array in the file, the header included (:func:`arrays_sha256`), so
the sha a config names pins the weights, what they were trained to read,
why they were written and which revision of the learner wrote them. The
purpose and the revision are in the header so that the sha covers them
(the orchestrator's resolution R2 of 2026-10-01): the runner refuses
anything but a trained checkpoint, and the report checks that one revision
trained a whole sweep, so neither may be relabelled. A manifest edited from
``bootstrap`` to ``trained``, or to another revision, no longer matches its
arrays' header and fails verification; writing a new header as well gives
new arrays and so a new sha. The manifest repeats the header, and γ (checked
against the header's network), and adds what the sha does not bind: the
reward and training specs, the seeds, the FerrySim cell family and its
hash, the trainer's commit and dirty flag, the numpy and BLAS versions,
``episodes_trained``, the validation curve and the held-out score (filled
in later by the evaluator, :func:`record_held_out`; it holds the episodes
scored and their mean return, :data:`HELD_OUT_KEYS`). The loader refuses a
missing manifest, a manifest that is not its arrays' (sha or header), a
sha other than the expected one, a kind, schema or class mismatch, and any
format but 2; the legacy ``DDQN.load`` refuses this format by its own
check (``ddqn.py:289-295``). Archives are read with numpy's object loading
off, so a checkpoint file can hold arrays and nothing that executes. The
campaign's refusals of an untrained, unscored or dirty checkpoint (critic
B9) are the runner's to apply, to a verified manifest
(:func:`verify_checkpoint`, then :func:`campaign_refusals`); the loader
checks no training state, so FerrySim's bootstrap checkpoints load. No
manifest holds a wall time.

**Determinism.** Every draw comes from a seeded stream: the initial weights
from ``numpy.random.default_rng(seed)``, exploration from the caller's
``random.Random``, sampling from the replay's. With ``OPENBLAS_NUM_THREADS=1``
a training run is reproducible to the byte (the spec, conventions).

**Layering.** Numpy and the standard library only (the spec, conventions):
E3 loads this module and never the plan package, and ``selector/__init__``
does not import it, so no recorded arm loads it. Neither this module nor
``pair_replay`` imports the other: an update reads any batch with
``pair_replay.PairBatch``'s arrays.

Not thread-safe under concurrent updates: one trainer owns a network.
Scoring is read-only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import numbers
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CHECKPOINT_KINDS",
    "CHECKPOINT_PURPOSES",
    "HELD_OUT_KEYS",
    "KIND_CHEN_DQN",
    "KIND_PAIR_Q",
    "LEARNER_REVISION",
    "MANIFEST_KEYS",
    "PAIR_Q_FORMAT_VERSION",
    "PROVENANCE_KEYS",
    "PURPOSE_BOOTSTRAP",
    "PURPOSE_TRAINED",
    "Behaviour",
    "BehaviourSchedule",
    "CheckpointError",
    "LearnerSettings",
    "PairQConfig",
    "PairQLearner",
    "PairQNet",
    "UpdateStats",
    "arrays_sha256",
    "behaviour_row",
    "campaign_refusals",
    "manifest_path",
    "manifest_provenance",
    "masked_argmax",
    "read_manifest",
    "record_held_out",
    "verify_checkpoint",
]

PathLike = Union[str, "os.PathLike[str]"]

#: The checkpoint format this module writes and the only one it reads. The
#: legacy DDQN wrote format 1 (``ddqn.py:43``) and refuses any other.
PAIR_Q_FORMAT_VERSION = 2

#: The learner's revision, written into every checkpoint's header (so the sha
#: binds it, resolution R2) and repeated in its manifest. The spec allows one
#: revision of the learner's settings, decided on the control cells before
#: the γ sweep (other choices 12); a revision bumps this, so the report can
#: check that every checkpoint of a sweep was trained by one learner.
LEARNER_REVISION = 0

#: What a checkpoint scores: the pair slot's (band, next stop) rows
#: (``flight_slot = "pair_q"``, the FQ arms) or E3's candidate stops
#: (``contact_policy = "chen_dqn"``); the config's switch values.
KIND_PAIR_Q = "pair_q"
KIND_CHEN_DQN = "chen_dqn"
CHECKPOINT_KINDS: Tuple[str, ...] = (KIND_PAIR_Q, KIND_CHEN_DQN)

#: Why a checkpoint was written: ``bootstrap``, the network a training run
#: starts from (FerrySim flies E3 from one), or ``trained``, a run's result.
#: The header holds it, so the sha binds it (resolution R2). Only the runner
#: tells the two apart (critic B9; :func:`campaign_refusals`).
PURPOSE_BOOTSTRAP = "bootstrap"
PURPOSE_TRAINED = "trained"
CHECKPOINT_PURPOSES: Tuple[str, ...] = (PURPOSE_BOOTSTRAP, PURPOSE_TRAINED)

#: The hidden layers' activation. tanh only (the spec, other choices 3); the
#: field exists so that a checkpoint states it.
ACTIVATION_TANH = "tanh"
ACTIVATIONS: Tuple[str, ...] = (ACTIVATION_TANH,)

#: The arrays of a format-2 ``.npz`` besides the weights.
ARRAY_FORMAT = "format_version"
ARRAY_HEADER = "header"

#: The manifest's keys, in three groups. The header's, which the sha binds;
#: the purpose and the learner's revision are among them so that neither can
#: be relabelled (resolution R2):
HEADER_KEYS: Tuple[str, ...] = (
    "format", "kind", "purpose", "learner_revision", "network", "schema", "classes",
)
#: What :meth:`PairQNet.save` adds itself, outside the header: the sha, γ
#: (checked against the header's network) and the numpy and BLAS versions
#: (informational):
WRITER_KEYS: Tuple[str, ...] = ("sha256", "gamma", "numpy", "blas")
#: The provenance the caller gives (the trainer, unit U8b), every key
#: required and none other (other choices 6): ``reward``, ``training`` and
#: ``seeds`` are JSON objects (the reward spec, the training spec, the seed
#: streams); ``cell_family`` and ``cell_family_sha256`` name FerrySim's cell
#: family and its hash (or None); ``trainer_commit`` (or None) and ``dirty``
#: describe the trainer's tree; ``episodes_trained`` counts training
#: episodes; ``validation`` is the validation curve (a list); ``held_out`` is
#: the held-out score, None until the evaluator fills it (:data:`HELD_OUT_KEYS`).
#: None of them is in the header, so the sha binds none: the held-out score is
#: written after the save, and the rest record how the weights were made
#: (resolution R2 keeps them informational).
PROVENANCE_KEYS: Tuple[str, ...] = (
    "reward", "training", "seeds", "cell_family", "cell_family_sha256", "trainer_commit",
    "dirty", "episodes_trained", "validation", "held_out",
)
MANIFEST_KEYS: Tuple[str, ...] = HEADER_KEYS + WRITER_KEYS + PROVENANCE_KEYS

#: What a held-out score holds at least, so that "no held-out score" (critic
#: B9) is decided on the score and not on a mapping's presence: ``episodes``,
#: the held-out episodes it averages (an int >= 1), and ``return_mean``, their
#: mean undiscounted return (the spec's evaluation; Study 5.5's score), a
#: finite number. The evaluator (unit U8a) may add entries of its own.
HELD_OUT_KEYS: Tuple[str, ...] = ("episodes", "return_mean")

#: A sha256 as ``hashlib`` writes it (``processes/config.py``'s rule).
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class CheckpointError(ValueError):
    """A checkpoint file the pair learner refuses to read or to fly."""


# --------------------------------------------------------------------------- #
# Value checks (this module imports nothing from hermes, so it has its own)
# --------------------------------------------------------------------------- #

def _int(value: Any, name: str, minimum: int = 0) -> int:
    """``value`` as an int >= ``minimum``; a bool is refused (``True`` is no count)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


def _real(value: Any, name: str) -> float:
    """``value`` as a finite float; a bool is refused."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _positive(value: Any, name: str) -> float:
    out = _real(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return out


def _unit(value: Any, name: str) -> float:
    """A float in [0, 1]."""
    out = _real(value, name)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return out


def _mask(value: Any, length: int, name: str = "mask") -> np.ndarray:
    """``value`` as a bool array of ``length`` flags; 0/1 integers are refused."""
    out = np.asarray(value)
    if out.dtype != np.bool_:
        raise TypeError(f"{name} must hold bools, got dtype {out.dtype}")
    if out.shape != (length,):
        raise ValueError(f"{name} holds one flag per row ({length}), got shape {out.shape}")
    return out


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{name} must be 64 lowercase hex digits as hashlib writes them, "
                         f"got {value!r}")
    return value


def _json_ready(value: Any, name: str) -> Any:
    """A fresh JSON-ready copy of ``value``: tuples as lists, numbers checked.

    Refuses what JSON cannot carry exactly (non-finite numbers, keys that are
    not strings, other objects) and any key that names a wall time: no
    record holds one (the spec's determinism convention), so two runs of
    one training write the same manifest.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"{name}: JSON carries finite numbers only, got {value!r}")
        return out
    if isinstance(value, Mapping):
        copy: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name}: keys must be strings, got {key!r}")
            if "wall" in key:
                raise ValueError(f"{name}: {key!r} names a wall time, which no record holds")
            copy[key] = _json_ready(item, f"{name}[{key!r}]")
        return copy
    if isinstance(value, (list, tuple)):
        return [_json_ready(item, f"{name}[]") for item in value]
    raise TypeError(f"{name} must be JSON-ready (numbers, strings, lists, maps), got {value!r}")


def _json_object(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object (a mapping), got {value!r}")
    return _json_ready(value, name)


def _canonical(value: Any) -> str:
    """The one JSON text of a JSON-ready value: sorted keys, no spaces, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _schema(value: Any, name: str = "schema") -> Dict[str, Any]:
    """A feature schema: a JSON object with a ``version`` name and the row width ``dim``.

    Beyond those two keys the schema is its owner's (``pair_v1``'s columns,
    classes and phase flag, unit U1; E3's rows, unit U6): this module only
    stores it, binds it to the weights and compares it whole.
    """
    out = _json_object(value, name)
    version = out.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{name} names its version (a non-empty string), got {version!r}")
    dim = out.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError(f"{name} states its row width 'dim' (an int >= 1), got {dim!r}")
    return out


def _classes(value: Any, name: str = "classes") -> List[str]:
    """The class tuple, in link order: distinct non-empty names."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence of class names, got {value!r}")
    out = list(value)
    if not out or any(not isinstance(c, str) or not c for c in out) or len(set(out)) != len(out):
        raise ValueError(f"{name} holds distinct non-empty class names, at least one: {value!r}")
    return out


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PairQConfig:
    """The network and its update rule (the spec, other choices 3; design D-C (a)).

    ``hidden`` = (64, 64) with tanh; ``()`` is a linear score (design D-C
    (c) as a configuration of (a)). ``gamma`` in [0, 1] (Study 5.5 sweeps
    {0, 0.25, 0.5, 0.75, 0.9, 0.99}; the default 0.9 is the design's, and a
    trainer always states its own). Adam at ``lr`` 1e-3 with Kingma and
    Ba's β1 = 0.9, β2 = 0.999 and ε = 1e-8; the Huber loss's ``huber_delta``
    1; the global-norm clip ``grad_clip`` 10; a hard target sync every
    ``target_sync`` = 500 updates; ``n_step`` 1, the only value taken.
    """

    hidden: Tuple[int, ...] = (64, 64)
    activation: str = ACTIVATION_TANH
    gamma: float = 0.9
    lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    huber_delta: float = 1.0
    grad_clip: float = 10.0
    target_sync: int = 500
    n_step: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.hidden, (str, bytes)) or not isinstance(self.hidden, (list, tuple)):
            raise TypeError(f"hidden must be a sequence of layer widths, got {self.hidden!r}")
        hidden = tuple(_int(width, "hidden width", 1) for width in self.hidden)
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        for beta_name in ("adam_beta1", "adam_beta2"):
            beta = _real(getattr(self, beta_name), beta_name)
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{beta_name} must be in [0, 1), got {beta!r}")
            object.__setattr__(self, beta_name, beta)
        if _int(self.n_step, "n_step", 1) != 1:
            raise ValueError(
                f"n_step must be 1 (the spec, other choices 3; the sortie-length return is "
                f"Monte Carlo, not this learner), got {self.n_step!r}"
            )
        object.__setattr__(self, "hidden", hidden)
        object.__setattr__(self, "gamma", _unit(self.gamma, "gamma"))
        object.__setattr__(self, "lr", _positive(self.lr, "lr"))
        object.__setattr__(self, "adam_eps", _positive(self.adam_eps, "adam_eps"))
        object.__setattr__(self, "huber_delta", _positive(self.huber_delta, "huber_delta"))
        object.__setattr__(self, "grad_clip", _positive(self.grad_clip, "grad_clip"))
        object.__setattr__(self, "target_sync", _int(self.target_sync, "target_sync", 1))
        object.__setattr__(self, "n_step", 1)

    def to_json(self) -> Dict[str, Any]:
        return {
            "hidden": list(self.hidden), "activation": self.activation, "gamma": self.gamma,
            "lr": self.lr, "adam_beta1": self.adam_beta1, "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps, "huber_delta": self.huber_delta,
            "grad_clip": self.grad_clip, "target_sync": self.target_sync, "n_step": self.n_step,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PairQConfig":
        """The config :meth:`to_json` wrote: exactly its keys."""
        if not isinstance(data, Mapping):
            raise TypeError(f"a network config is a mapping, got {data!r}")
        names = set(cls().to_json())
        if set(data) != names:
            raise ValueError(
                f"a network config has exactly the keys {sorted(names)}: missing "
                f"{sorted(names - set(data))}, unknown {sorted(set(data) - names)}"
            )
        return cls(**{key: (tuple(data[key]) if key == "hidden" and isinstance(
            data[key], list) else data[key]) for key in names})


@dataclass(frozen=True)
class Behaviour:
    """How one training episode flies: its ε, and whether greedy means FX's pair."""

    epsilon: float
    around_reference: bool


@dataclass(frozen=True)
class BehaviourSchedule:
    """The behaviour policy over a training run (the spec, other choices 3; critic C4).

    Episodes ``[0, reference_episodes)`` fly ε-greedy around the reference
    pair (FX's) at ``epsilon_start``. After them the mule flies ε-greedy on
    Q, ε falling linearly from ``epsilon_start`` at episode
    ``reference_episodes`` to ``epsilon_end`` at episode ``decay_fraction``
    × the run's episodes (the first half by default), and ``epsilon_end``
    from there on; when that episode is not after the reference phase, ε is
    ``epsilon_end`` straight after it. Defaults: 500 episodes, 0.3, 0.05,
    0.5. ε never starts at 1.0: a Q over pairs never trained is noise.
    """

    reference_episodes: int = 500
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    decay_fraction: float = 0.5

    def __post_init__(self) -> None:
        _int(self.reference_episodes, "reference_episodes")
        start = _unit(self.epsilon_start, "epsilon_start")
        end = _unit(self.epsilon_end, "epsilon_end")
        if end > start:
            raise ValueError(f"epsilon falls: epsilon_end ({end}) <= epsilon_start ({start})")
        fraction = _real(self.decay_fraction, "decay_fraction")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"decay_fraction must be in (0, 1], got {self.decay_fraction!r}")
        object.__setattr__(self, "epsilon_start", start)
        object.__setattr__(self, "epsilon_end", end)
        object.__setattr__(self, "decay_fraction", fraction)

    def at(self, episode: int, episodes: int) -> Behaviour:
        """The behaviour of training episode ``episode`` (0-based) of ``episodes``."""
        episodes = _int(episodes, "episodes", 1)
        episode = _int(episode, "episode")
        if episode >= episodes:
            raise ValueError(f"episode {episode} is not among the run's {episodes}")
        if episode < self.reference_episodes:
            return Behaviour(epsilon=self.epsilon_start, around_reference=True)
        decay_end = int(self.decay_fraction * episodes)
        span = decay_end - self.reference_episodes
        if span <= 0 or episode >= decay_end:
            return Behaviour(epsilon=self.epsilon_end, around_reference=False)
        share = (episode - self.reference_episodes) / span
        epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * share
        return Behaviour(epsilon=epsilon, around_reference=False)

    def to_json(self) -> Dict[str, Any]:
        return {"reference_episodes": self.reference_episodes,
                "epsilon_start": self.epsilon_start, "epsilon_end": self.epsilon_end,
                "decay_fraction": self.decay_fraction}


@dataclass(frozen=True)
class LearnerSettings:
    """The training loop's settings (the spec, other choices 3).

    Batches of ``batch`` = 64 transitions; a replay of ``replay_capacity`` =
    50,000 (``pair_replay.DEFAULT_CAPACITY``); no update until
    ``warmup_transitions`` = 1,000 are stored, then one update per decision
    (:class:`PairQLearner`); the behaviour policy's schedule. The trainer
    records :meth:`to_json` in its training spec.
    """

    batch: int = 64
    replay_capacity: int = 50_000
    warmup_transitions: int = 1_000
    behaviour: BehaviourSchedule = field(default_factory=BehaviourSchedule)

    def __post_init__(self) -> None:
        batch = _int(self.batch, "batch", 1)
        warmup = _int(self.warmup_transitions, "warmup_transitions", 1)
        capacity = _int(self.replay_capacity, "replay_capacity", 1)
        if not batch <= warmup <= capacity:
            raise ValueError(
                f"batch ({batch}) <= warmup_transitions ({warmup}) <= replay_capacity "
                f"({capacity}): the first update samples a full batch from a warm replay"
            )
        if not isinstance(self.behaviour, BehaviourSchedule):
            raise TypeError(f"behaviour must be a BehaviourSchedule, got {self.behaviour!r}")

    def to_json(self) -> Dict[str, Any]:
        return {"batch": self.batch, "replay_capacity": self.replay_capacity,
                "warmup_transitions": self.warmup_transitions,
                "behaviour": self.behaviour.to_json()}


# --------------------------------------------------------------------------- #
# Choosing a row
# --------------------------------------------------------------------------- #

def masked_argmax(scores: Any, mask: Any) -> int:
    """The admitted row with the highest score, ties to the lowest row.

    The slot's rule (``plan.types.PairScorer``): the pairs come class-major
    and then in the remainder's order, so the lowest row is the first class
    and the first stop. Refuses an empty mask (the slot flies FX's pair
    before asking) and a non-finite admitted score.
    """
    q = np.asarray(scores, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"scores are one per row (1-D), got shape {q.shape}")
    rows = np.flatnonzero(_mask(mask, q.shape[0]))
    if rows.size == 0:
        raise ValueError("no row is admitted: an empty mask flies FX's pair, the slot's fallback")
    admitted = q[rows]
    if not np.isfinite(admitted).all():
        raise FloatingPointError(f"an admitted row has a non-finite score: {admitted.tolist()}")
    return int(rows[int(np.argmax(admitted))])


def behaviour_row(scores: Any, mask: Any, *, epsilon: float, rng: Any,
                  reference: Optional[int] = None) -> int:
    """A training decision: ε-greedy over the admitted rows (critic C4).

    With probability ``epsilon`` a row drawn uniformly from the admitted
    ones; otherwise the greedy row, which is ``reference`` (FX's pair, in the
    schedule's reference phase) when it is given and admitted, and else
    :func:`masked_argmax` of ``scores``. ``rng`` is the episode's seeded
    stream (``random.Random`` or anything with ``random()``), and every call
    draws exactly two numbers, exploring or not, so that runs that differ in
    ε or in their weights draw the same numbers at the same decisions. The
    arguments, ``reference`` included, are checked before the draws, so a
    wrong one is refused at every decision and not only at the greedy ones
    (a refused call draws nothing).
    """
    epsilon = _unit(epsilon, "epsilon")
    q = np.asarray(scores, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"scores are one per row (1-D), got shape {q.shape}")
    flags = _mask(mask, q.shape[0])
    rows = np.flatnonzero(flags)
    if rows.size == 0:
        raise ValueError("no row is admitted: an empty mask flies FX's pair, the slot's fallback")
    if reference is not None:
        reference = _int(reference, "reference")
        if reference >= q.shape[0]:
            raise ValueError(f"reference row {reference} is not among the {q.shape[0]} rows")
    explore, pick = float(rng.random()), float(rng.random())
    if explore < epsilon:
        return int(rows[min(int(pick * rows.size), rows.size - 1)])
    if reference is not None and flags[reference]:
        return reference
    return masked_argmax(q, flags)


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #

def _param_names(layers: int) -> List[str]:
    """``layer0_W``, ``layer0_b``, ...: one weight and one bias per layer."""
    return [f"layer{i}_{part}" for i in range(layers) for part in ("W", "b")]


def _forward(params: Sequence[np.ndarray], x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """x (B, F) -> q (B,), with each layer's input kept for the backward pass."""
    inputs = [x]
    h = x
    layers = len(params) // 2
    for i in range(layers - 1):
        h = np.tanh(h @ params[2 * i] + params[2 * i + 1])
        inputs.append(h)
    q = (h @ params[-2] + params[-1])[:, 0]
    return q, inputs


def _scores(params: Sequence[np.ndarray], x: np.ndarray) -> np.ndarray:
    """x (K, F) -> q (K,), each distinct row forwarded once and equal rows given one Q.

    One pass over the candidate set can round two equal rows apart in the
    last bit, because the BLAS kernel treats a row by its position: on random
    rows, a later copy of the best row won the argmax about a quarter of the
    time. The slot breaks ties to the lowest row, the pairs
    come class-major and then in the remainder's order, and a symmetric
    layout offers equal rows, so the tie rule must decide between them, not
    rounding. Rows are compared after adding 0.0, which makes -0.0 equal to
    0.0. With no equal rows the pass is the plain one, number for number.
    """
    seen: Dict[bytes, int] = {}
    keep: List[int] = []
    index = np.empty(x.shape[0], dtype=np.intp)
    for i, row in enumerate(x + 0.0):
        key = row.tobytes()
        if key not in seen:
            seen[key] = len(keep)
            keep.append(i)
        index[i] = seen[key]
    if len(keep) == x.shape[0]:
        return _forward(params, x)[0]
    return _forward(params, x[keep])[0][index]


def _backward(params: Sequence[np.ndarray], inputs: Sequence[np.ndarray],
              dq: np.ndarray) -> List[np.ndarray]:
    """The gradient of every parameter, given dL/dq (B,)."""
    grads: List[np.ndarray] = [np.empty(0)] * len(params)
    d = dq[:, None]
    for i in reversed(range(len(params) // 2)):
        grads[2 * i] = inputs[i].T @ d
        grads[2 * i + 1] = d.sum(axis=0)
        if i > 0:
            d = (d @ params[2 * i].T) * (1.0 - inputs[i] * inputs[i])
    return grads


def _segment_argmax(values: np.ndarray, segment: np.ndarray, count: int) -> np.ndarray:
    """Per segment, the index into ``values`` of its largest value, ties to the
    lowest index; -1 for a segment with no value."""
    out = np.full(count, -1, dtype=np.int64)
    if values.size == 0:
        return out
    order = np.lexsort((np.arange(values.size), -values, segment))
    ordered = segment[order]
    first = np.ones(order.size, dtype=np.bool_)
    first[1:] = ordered[1:] != ordered[:-1]
    out[ordered[first]] = order[first]
    return out


def _batch_arrays(batch: Any, width: int) -> Tuple[np.ndarray, ...]:
    """A batch's arrays (``pair_replay.PairBatch``'s), checked against the network."""
    try:
        arrays = (batch.x, batch.reward, batch.done, batch.next_rows, batch.next_mask,
                  batch.segment)
    except AttributeError:
        raise TypeError(
            f"an update takes a batch with PairBatch's arrays (selector.pair_replay), got "
            f"{batch!r}") from None
    x = np.asarray(arrays[0], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != width:
        raise ValueError(f"the batch's rows are (B >= 1, {width}), got shape {x.shape}")
    size = x.shape[0]
    reward = np.asarray(arrays[1], dtype=np.float64)
    done = _mask(arrays[2], size, "done")
    rows = np.asarray(arrays[3], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != width:
        raise ValueError(f"the batch's next rows are (M, {width}), got shape {rows.shape}")
    mask = _mask(arrays[4], rows.shape[0], "next_mask")
    segment = np.asarray(arrays[5], dtype=np.int64)
    if reward.shape != (size,) or segment.shape != (rows.shape[0],):
        raise ValueError("the batch holds one reward per transition and one segment per row")
    if segment.size and (segment.min() < 0 or segment.max() >= size):
        raise ValueError(f"the batch's segments index its {size} transitions")
    return x, reward, done, rows, mask, segment


@dataclass(frozen=True)
class UpdateStats:
    """One update's numbers: the loss, the gradient's global norm before the
    clip, and the factor the clip applied (1.0 when it did not bind)."""

    loss: float
    grad_norm: float
    clip_scale: float


class PairQNet:
    """The pair learner: online and target copies of one MLP, and Adam's moments.

    ``PairQNet(feature_dim, cfg, seed=...)`` draws the initial weights from
    ``numpy.random.default_rng(seed)``: W ~ N(0, 1/fan_in), b = 0, the
    legacy DDQN's scale (``ddqn.py:62-71``), float64 throughout. The target
    network starts as a copy of the online one.
    """

    def __init__(self, feature_dim: int, cfg: Optional[PairQConfig] = None, *, seed: int):
        width = _int(feature_dim, "feature_dim", 1)
        cfg = PairQConfig() if cfg is None else cfg
        if not isinstance(cfg, PairQConfig):
            raise TypeError(f"cfg must be a PairQConfig, got {cfg!r}")
        seed = _int(seed, "seed")
        rng = np.random.default_rng(seed)
        params: List[np.ndarray] = []
        fan_in = width
        for fan_out in cfg.hidden + (1,):
            params.append(rng.normal(0.0, 1.0 / math.sqrt(fan_in), size=(fan_in, fan_out)))
            params.append(np.zeros(fan_out, dtype=np.float64))
            fan_in = fan_out
        self._setup(width, cfg, params)

    def _setup(self, width: int, cfg: PairQConfig, params: Sequence[np.ndarray]) -> None:
        self._width = width
        self._cfg = cfg
        self._names = _param_names(len(cfg.hidden) + 1)
        self._online = [np.array(p, dtype=np.float64) for p in params]
        self._target = [p.copy() for p in self._online]
        self._m = [np.zeros_like(p) for p in self._online]
        self._v = [np.zeros_like(p) for p in self._online]
        self._updates = 0
        self._syncs = 0
        self._last: Optional[UpdateStats] = None

    def __repr__(self) -> str:
        return (f"PairQNet(feature_dim={self._width}, hidden={self._cfg.hidden}, "
                f"gamma={self._cfg.gamma}, updates={self._updates})")

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def feature_dim(self) -> int:
        return self._width

    @property
    def config(self) -> PairQConfig:
        return self._cfg

    @property
    def gamma(self) -> float:
        return self._cfg.gamma

    @property
    def updates(self) -> int:
        """Updates taken; Adam's step count."""
        return self._updates

    @property
    def syncs(self) -> int:
        """Hard target syncs so far: one every ``target_sync`` updates."""
        return self._syncs

    @property
    def last_update(self) -> Optional[UpdateStats]:
        return self._last

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _rows(self, rows: Any) -> np.ndarray:
        x = np.asarray(rows, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self._width:
            raise ValueError(f"rows are (K, {self._width}), got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("rows hold finite features")
        return x

    def q(self, rows: Any) -> np.ndarray:
        """The online network's Q of each row: (K, F) -> (K,).

        Equal rows get one Q, bit for bit (:func:`_scores`), so the slot's tie
        rule, the lowest row, decides between them. Score a decision's whole
        candidate set in one call.
        """
        return _scores(self._online, self._rows(rows))

    def q_target(self, rows: Any) -> np.ndarray:
        """The target network's Q of each row: (K, F) -> (K,), equal rows alike."""
        return _scores(self._target, self._rows(rows))

    def masked_argmax(self, rows: Any, mask: Any) -> int:
        """The admitted row with the highest online Q, ties (equal rows included)
        to the lowest row."""
        return masked_argmax(self.q(rows), mask)

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def targets(self, batch: Any) -> np.ndarray:
        """The double-DQN targets of ``batch``: r + γ · Q_target(s', online argmax).

        Only the admitted next rows are forwarded, through either network. A
        done transition, and every transition when γ = 0, gets r exactly. The
        batch's rows go through in one pass for speed, so two equal next rows
        may differ in the last bit and either copy may win; their target
        values agree to rounding, and so does y.
        """
        _, reward, done, rows, mask, segment = _batch_arrays(batch, self._width)
        y = reward.copy()
        live = ~done
        if self._cfg.gamma == 0.0 or not live.any():
            return y
        admitted = rows[mask]
        if not np.isfinite(admitted).all():
            raise ValueError("the admitted next rows hold finite features")
        owner = segment[mask]
        q_online = _forward(self._online, admitted)[0]
        q_target = _forward(self._target, admitted)[0]
        if not (np.isfinite(q_online).all() and np.isfinite(q_target).all()):
            raise FloatingPointError("a network gives a non-finite Q on an admitted next row")
        best = _segment_argmax(q_online, owner, y.shape[0])
        if (best[live] < 0).any():
            raise ValueError("a transition that is not done needs an admitted next row")
        y[live] = y[live] + self._cfg.gamma * q_target[best[live]]
        return y

    def _loss_and_grads(self, batch: Any, targets: Any) -> Tuple[float, List[np.ndarray]]:
        x = _batch_arrays(batch, self._width)[0]
        if not np.isfinite(x).all():
            raise ValueError("the batch's rows hold finite features")
        y = np.asarray(targets, dtype=np.float64)
        if y.shape != (x.shape[0],) or not np.isfinite(y).all():
            raise ValueError(f"targets are one finite number per transition ({x.shape[0]})")
        q, inputs = _forward(self._online, x)
        err = q - y
        delta = self._cfg.huber_delta
        size = np.abs(err)
        quad = np.minimum(size, delta)
        loss = float(np.mean(0.5 * quad * quad + delta * (size - quad)))
        dq = np.clip(err, -delta, delta) / err.shape[0]
        return loss, _backward(self._online, inputs, dq)

    def loss_and_gradients(self, batch: Any, targets: Any = None
                           ) -> Tuple[float, Dict[str, np.ndarray]]:
        """The Huber loss of ``batch`` and its gradient by parameter name.

        ``targets`` defaults to :meth:`targets`; they are held fixed (the
        semi-gradient every DQN takes). Nothing is updated.
        """
        if targets is None:
            targets = self.targets(batch)
        loss, grads = self._loss_and_grads(batch, targets)
        return loss, dict(zip(self._names, grads))

    def update(self, batch: Any) -> float:
        """One step on ``batch``: targets, Huber gradient, global-norm clip, Adam,
        and a hard target sync every ``target_sync`` updates. Returns the loss."""
        loss, grads = self._loss_and_grads(batch, self.targets(batch))
        norm = math.sqrt(sum(float(np.vdot(g, g)) for g in grads))
        if not (math.isfinite(loss) and math.isfinite(norm)):
            raise FloatingPointError(f"the update diverged: loss {loss}, gradient norm {norm}")
        cfg = self._cfg
        scale = 1.0 if norm <= cfg.grad_clip else cfg.grad_clip / norm
        step = self._updates + 1
        beta1, beta2 = cfg.adam_beta1, cfg.adam_beta2
        unbias1, unbias2 = 1.0 - beta1 ** step, 1.0 - beta2 ** step
        for p, g, m, v in zip(self._online, grads, self._m, self._v):
            g = g * scale
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)
            p -= cfg.lr * (m / unbias1) / (np.sqrt(v / unbias2) + cfg.adam_eps)
        self._updates = step
        if step % cfg.target_sync == 0:
            self.sync_target()
        self._last = UpdateStats(loss=loss, grad_norm=norm, clip_scale=scale)
        return loss

    def sync_target(self) -> None:
        """Copy the online weights into the target network (a hard sync)."""
        self._target = [p.copy() for p in self._online]
        self._syncs += 1

    # ------------------------------------------------------------------ #
    # Weights
    # ------------------------------------------------------------------ #

    def weights(self) -> Dict[str, np.ndarray]:
        """Copies of the online weights, by name (``layer0_W``, ``layer0_b``, ...)."""
        return {name: p.copy() for name, p in zip(self._names, self._online)}

    def target_weights(self) -> Dict[str, np.ndarray]:
        return {name: p.copy() for name, p in zip(self._names, self._target)}

    def _checked(self, weights: Mapping[str, Any], which: str) -> List[np.ndarray]:
        if not isinstance(weights, Mapping) or set(weights) != set(self._names):
            raise ValueError(f"{which} weights are exactly {self._names}")
        out = []
        for name, current in zip(self._names, self._online):
            array = np.array(weights[name], dtype=np.float64)
            if array.shape != current.shape or not np.isfinite(array).all():
                raise ValueError(f"{which} {name} is a finite array of shape {current.shape}")
            out.append(array)
        return out

    def set_weights(self, online: Mapping[str, Any],
                    target: Optional[Mapping[str, Any]] = None) -> None:
        """Replace the online weights, and the target's (a copy of ``online`` by
        default). Adam's moments and the update count are kept."""
        new_online = self._checked(online, "online")
        new_target = ([p.copy() for p in new_online] if target is None
                      else self._checked(target, "target"))
        self._online, self._target = new_online, new_target

    def adam_moments(self) -> Dict[str, Any]:
        """Copies of Adam's state: ``step`` and the moments ``m`` and ``v`` by name."""
        return {"step": self._updates,
                "m": {n: a.copy() for n, a in zip(self._names, self._m)},
                "v": {n: a.copy() for n, a in zip(self._names, self._v)}}

    # ------------------------------------------------------------------ #
    # Checkpoints (format 2)
    # ------------------------------------------------------------------ #

    def _header(self, kind: str, purpose: str, schema: Mapping[str, Any],
                classes: Sequence[str]) -> Dict[str, Any]:
        """The header a checkpoint's sha binds: :data:`HEADER_KEYS`, the learner's
        revision being this module's :data:`LEARNER_REVISION`."""
        if kind not in CHECKPOINT_KINDS:
            raise ValueError(f"kind must be one of {CHECKPOINT_KINDS}, got {kind!r}")
        if purpose not in CHECKPOINT_PURPOSES:
            raise ValueError(f"purpose must be one of {CHECKPOINT_PURPOSES}, got {purpose!r}")
        schema = _schema(schema)
        classes = _classes(classes)
        if schema["dim"] != self._width:
            raise ValueError(
                f"the schema's rows are {schema['dim']} wide and the network's "
                f"{self._width}: they are one schema's")
        if "classes" in schema and schema["classes"] != classes:
            raise ValueError(f"the schema's classes {schema['classes']} are not {classes}")
        return {"format": PAIR_Q_FORMAT_VERSION, "kind": kind, "purpose": purpose,
                "learner_revision": LEARNER_REVISION,
                "network": {"feature_dim": self._width, **self._cfg.to_json()},
                "schema": schema, "classes": classes}

    def save(self, path: PathLike, *, kind: str, purpose: str, schema: Mapping[str, Any],
             classes: Sequence[str], provenance: Mapping[str, Any]) -> str:
        """Write the online weights as a format-2 checkpoint; return their sha256.

        ``path`` ends in ``.npz``; the manifest goes beside it
        (:func:`manifest_path`). ``kind`` is :data:`CHECKPOINT_KINDS`'s,
        ``purpose`` :data:`CHECKPOINT_PURPOSES`'s, ``schema`` the feature
        schema (a JSON object with ``version`` and ``dim``, the network's row
        width), ``classes`` the class tuple in link order (E3: its one
        contact band), and ``provenance`` exactly :data:`PROVENANCE_KEYS`.
        The header holds the kind, the purpose, :data:`LEARNER_REVISION`, the
        network and its settings, the schema and the classes, so each of them
        changes the sha (resolution R2 for the purpose and the revision): a
        bootstrap and a trained checkpoint of the same weights have two shas.
        The provenance is outside the header, so the held-out score can be
        filled in later under the sha a config already names.
        Each file is written whole or not at all (a temporary file, then a
        rename), the arrays first: a crash between the two leaves arrays
        whose manifest is missing or names other arrays, and the loader
        refuses both.
        """
        npz = _npz_path(path)
        header = self._header(kind, purpose, schema, classes)
        errors = _provenance_errors(provenance)
        if errors:
            raise ValueError("the provenance is not a manifest's: " + "; ".join(errors))
        if not all(np.isfinite(p).all() for p in self._online):
            raise FloatingPointError("a network with non-finite weights is not saved")
        arrays: Dict[str, np.ndarray] = {
            ARRAY_FORMAT: np.array(PAIR_Q_FORMAT_VERSION, dtype=np.int64),
            ARRAY_HEADER: np.frombuffer(_canonical(header).encode("ascii"), dtype=np.uint8).copy(),
        }
        arrays.update(self.weights())
        sha = arrays_sha256(arrays)
        manifest = dict(header)
        manifest.update(sha256=sha, gamma=self._cfg.gamma, numpy=np.__version__, blas=_blas())
        manifest.update(_json_object(provenance, "provenance"))
        errors = _manifest_errors(manifest)
        if errors:
            raise ValueError("the manifest is malformed: " + "; ".join(errors))
        ordered = {name: arrays[name] for name in sorted(arrays)}
        _write_atomic(npz, lambda handle: np.savez(handle, **ordered))
        _write_atomic(manifest_path(npz), lambda handle: handle.write(_manifest_bytes(manifest)))
        return sha

    @classmethod
    def load(cls, path: PathLike, *, expect_sha256: str, expect_kind: str,
             expect_schema: Mapping[str, Any], expect_classes: Sequence[str]
             ) -> Tuple["PairQNet", Dict[str, Any]]:
        """Read a format-2 checkpoint that is what the caller expects; return it
        and its manifest.

        Refuses (:class:`CheckpointError`) any format but 2 (the legacy
        DDQN's format 1 included), a missing or malformed manifest, arrays
        that are not the manifest's (sha) or not its header's (a relabelled
        purpose or learner revision among them), a sha other than
        ``expect_sha256`` (the config's), and a kind, class tuple or schema
        other than expected. Checks no training state: a bootstrap
        checkpoint loads (the runner refuses it for campaigns,
        :func:`campaign_refusals`), and so does one of another learner
        revision. The manifest returned is the verified one: its purpose and
        revision are the header's, which ``expect_sha256`` pins.
        """
        expect_sha256 = _sha(expect_sha256, "expect_sha256")
        if expect_kind not in CHECKPOINT_KINDS:
            raise ValueError(f"expect_kind must be one of {CHECKPOINT_KINDS}, got {expect_kind!r}")
        schema = _schema(expect_schema, "expect_schema")
        classes = _classes(expect_classes, "expect_classes")
        manifest, header, params = _read_checkpoint(path)
        npz = _npz_path(path)
        if manifest["sha256"] != expect_sha256:
            raise CheckpointError(
                f"{npz}: its arrays' sha256 is {manifest['sha256']}, not the expected "
                f"{expect_sha256}")
        if header["kind"] != expect_kind:
            raise CheckpointError(f"{npz}: a {header['kind']!r} checkpoint, not {expect_kind!r}")
        if header["classes"] != classes:
            raise CheckpointError(
                f"{npz}: trained on the classes {header['classes']}, not {classes}")
        if _canonical(header["schema"]) != _canonical(schema):
            stored = header["schema"]
            differ = sorted(k for k in set(stored) | set(schema)
                            if _canonical(stored.get(k)) != _canonical(schema.get(k)))
            raise CheckpointError(
                f"{npz}: its feature schema is not the expected one (keys that differ: {differ}; "
                f"stored {stored.get('version')!r}, expected {schema.get('version')!r})")
        net = cls.__new__(cls)
        config = dict(header["network"])
        width = config.pop("feature_dim")
        net._setup(width, PairQConfig.from_json(config), params)
        return net, manifest


# --------------------------------------------------------------------------- #
# The learning loop's one rule
# --------------------------------------------------------------------------- #

class PairQLearner:
    """A network and its replay, updated once per decision after the warm-up.

    The spec's rule (other choices 3): each decision's transition is pushed,
    and once the replay holds ``warmup_transitions`` every push is followed
    by one update on a sampled batch of ``batch``. ``replay`` is a
    ``pair_replay.PairReplay`` of ``settings.replay_capacity``.
    """

    def __init__(self, net: PairQNet, replay: Any, settings: Optional[LearnerSettings] = None):
        settings = LearnerSettings() if settings is None else settings
        if not isinstance(net, PairQNet):
            raise TypeError(f"net must be a PairQNet, got {net!r}")
        if not isinstance(settings, LearnerSettings):
            raise TypeError(f"settings must be LearnerSettings, got {settings!r}")
        if not all(hasattr(replay, name) for name in ("push", "sample", "capacity", "__len__")):
            raise TypeError(f"replay must be a PairReplay (selector.pair_replay), got {replay!r}")
        if replay.capacity != settings.replay_capacity:
            raise ValueError(
                f"the replay holds {replay.capacity} transitions and the settings say "
                f"{settings.replay_capacity}: the manifest records the settings")
        self._net = net
        self._replay = replay
        self._settings = settings

    @property
    def net(self) -> PairQNet:
        return self._net

    @property
    def replay(self) -> Any:
        return self._replay

    @property
    def settings(self) -> LearnerSettings:
        return self._settings

    @property
    def warm(self) -> bool:
        """True once the replay holds the warm-up's transitions."""
        return len(self._replay) >= self._settings.warmup_transitions

    def observe(self, transition: Any) -> Optional[float]:
        """Push one decision's transition; after the warm-up, take one update.

        Returns the update's loss, or None while the replay warms up.
        """
        self._replay.push(transition)
        if not self.warm:
            return None
        return self._net.update(self._replay.sample(self._settings.batch))


# --------------------------------------------------------------------------- #
# Checkpoint files
# --------------------------------------------------------------------------- #

def _npz_path(path: PathLike) -> Path:
    out = Path(path)
    if out.suffix != ".npz":
        raise ValueError(f"a format-2 checkpoint is an .npz file, got {str(out)!r}")
    return out


def manifest_path(path: PathLike) -> Path:
    """The manifest beside the checkpoint ``path``: the same name, ``.json``."""
    return _npz_path(path).with_suffix(".json")


def arrays_sha256(arrays: Mapping[str, Any]) -> str:
    """The sha256 of a checkpoint's arrays, the one a config names.

    Over the arrays in name order, each as its name, its little-endian dtype,
    its shape and its C-order bytes, NUL-separated: the same arrays give the
    same sha in any file and any process, and a change of value, dtype or
    shape changes it. The file's own bytes are not hashed (a zip's layout is
    numpy's business).
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        array = np.ascontiguousarray(array)
        for part in (name.encode("utf-8"), array.dtype.str.encode("ascii"),
                     repr(tuple(int(n) for n in array.shape)).encode("ascii")):
            digest.update(part)
            digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _blas() -> str:
    """numpy's BLAS as its build reports it, and the BLAS thread setting.

    Bit-level reproducibility holds per BLAS kernel and at one thread (the
    spec, conventions), so the manifest says which ran. Introspection only:
    a numpy without the report gives "unknown".
    """
    try:
        info = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        text = f"{info.get('name', 'unknown')} {info.get('version', 'unknown')}"
        detail = info.get("openblas configuration")
        if detail and detail != "unknown":
            text += f" ({detail})"
    except Exception:  # noqa: BLE001 - any failure of the report is "unknown"
        text = "unknown"
    threads = os.environ.get("OPENBLAS_NUM_THREADS", "unset")
    return f"{text}; OPENBLAS_NUM_THREADS={threads}"


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    return (text + "\n").encode("ascii")


def _write_atomic(path: Path, write: Callable[[Any], Any]) -> None:
    """Write ``path`` whole or not at all: a temporary file beside it, then a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                         dir=str(path.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            write(stream)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def _provenance_errors(provenance: Any) -> List[str]:
    """What is wrong with a provenance mapping (empty when nothing is): exactly
    :data:`PROVENANCE_KEYS`, each of its type."""
    if not isinstance(provenance, Mapping):
        return [f"the provenance is a mapping of {list(PROVENANCE_KEYS)}, got {provenance!r}"]
    missing = [key for key in PROVENANCE_KEYS if key not in provenance]
    unknown = sorted(key for key in provenance if key not in PROVENANCE_KEYS)
    errors = []
    if missing or unknown:
        errors.append(f"provenance keys are exactly {list(PROVENANCE_KEYS)}: missing {missing}, "
                      f"unknown {unknown}")
    return errors + _provenance_value_errors(provenance)


def _provenance_value_errors(provenance: Mapping[str, Any]) -> List[str]:
    """What is wrong with the provenance values present (their keys are checked
    by the caller)."""
    errors = []
    for key in ("reward", "training", "seeds"):
        if key in provenance and not isinstance(provenance[key], Mapping):
            errors.append(f"{key} is a JSON object, got {provenance[key]!r}")
    for key in ("cell_family", "trainer_commit"):
        value = provenance.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            errors.append(f"{key} is a non-empty string or None, got {value!r}")
    value = provenance.get("cell_family_sha256")
    if value is not None and (not isinstance(value, str) or not _SHA256_HEX.fullmatch(value)):
        errors.append(f"cell_family_sha256 is 64 lowercase hex digits or None, got {value!r}")
    if "dirty" in provenance and not isinstance(provenance["dirty"], bool):
        errors.append(f"dirty is a bool, got {provenance['dirty']!r}")
    value = provenance.get("episodes_trained", 0)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        errors.append(f"episodes_trained is an int >= 0, got {value!r}")
    if "validation" in provenance and not isinstance(provenance["validation"], (list, tuple)):
        errors.append(f"validation is a list (the validation curve), got "
                      f"{provenance['validation']!r}")
    errors += _held_out_errors(provenance.get("held_out"))
    try:
        _json_ready(dict(provenance), "provenance")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def _held_out_errors(value: Any) -> List[str]:
    """What is wrong with a held-out score: empty when it is None (not yet
    evaluated) or holds :data:`HELD_OUT_KEYS` as they are defined there."""
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"held_out is a JSON object or None (until evaluated), got {value!r}"]
    errors = []
    episodes = value.get("episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, numbers.Integral) or episodes < 1:
        errors.append(f"held_out['episodes'] counts the held-out episodes the score averages, "
                      f"an int >= 1, got {episodes!r}")
    score = value.get("return_mean")
    if (isinstance(score, bool) or not isinstance(score, numbers.Real)
            or not math.isfinite(float(score))):
        errors.append(f"held_out['return_mean'] is the held-out score (the mean undiscounted "
                      f"return), a finite number, got {score!r}")
    return errors


def _manifest_errors(manifest: Any) -> List[str]:
    """What is wrong with a format-2 manifest (empty when nothing is)."""
    if not isinstance(manifest, Mapping):
        return [f"a manifest is a JSON object, got {type(manifest).__name__}"]
    version = manifest.get("format")
    if (isinstance(version, bool) or not isinstance(version, int)
            or version != PAIR_Q_FORMAT_VERSION):
        return [f"format {version!r}: format {PAIR_Q_FORMAT_VERSION} only"]
    errors = []
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    unknown = sorted(key for key in manifest if key not in MANIFEST_KEYS)
    if missing or unknown:
        errors.append(f"keys are exactly {list(MANIFEST_KEYS)}: missing {missing}, unknown "
                      f"{unknown}")
    if manifest.get("kind") not in CHECKPOINT_KINDS:
        errors.append(f"kind is one of {CHECKPOINT_KINDS}, got {manifest.get('kind')!r}")
    if manifest.get("purpose") not in CHECKPOINT_PURPOSES:
        errors.append(f"purpose is one of {CHECKPOINT_PURPOSES}, got {manifest.get('purpose')!r}")
    for check, key in ((_sha, "sha256"), (_classes, "classes")):
        try:
            check(manifest.get(key), key)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    network = manifest.get("network")
    width = None
    try:
        if not isinstance(network, Mapping):
            raise TypeError(f"network is a JSON object, got {network!r}")
        config = dict(network)
        width = _int(config.pop("feature_dim", None), "network.feature_dim", 1)
        gamma = PairQConfig.from_json(config).gamma
        if manifest.get("gamma") != gamma:
            errors.append(f"gamma {manifest.get('gamma')!r} is the network's {gamma!r}")
    except (TypeError, ValueError) as exc:
        errors.append(f"network: {exc}")
    try:
        schema = _schema(manifest.get("schema"))
        if width is not None and schema["dim"] != width:
            errors.append(f"the schema's dim {schema['dim']} is the network's {width}")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    revision = manifest.get("learner_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        errors.append(f"learner_revision is an int >= 0, got {revision!r}")
    for key in ("numpy", "blas"):
        if not isinstance(manifest.get(key), str):
            errors.append(f"{key} is a string, got {manifest.get(key)!r}")
    errors += _provenance_value_errors(
        {key: value for key, value in manifest.items() if key in PROVENANCE_KEYS})
    try:
        _json_ready(dict(manifest), "manifest")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return list(dict.fromkeys(errors))


def read_manifest(path: PathLike) -> Dict[str, Any]:
    """The manifest of the checkpoint ``path``, its form checked; the arrays are
    not read (:func:`verify_checkpoint` reads them), so nothing here shows that
    its purpose or learner revision is still its header's. Judge a checkpoint
    by the verified manifest."""
    npz = _npz_path(path)
    where = manifest_path(npz)
    if not where.is_file():
        raise CheckpointError(
            f"{npz} has no manifest beside it ({where.name}): a checkpoint without one is "
            f"refused (the Phase 5 spec, other choices 6)")
    try:
        manifest = json.loads(where.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"{where}: not a JSON manifest: {exc}") from None
    errors = _manifest_errors(manifest)
    if errors:
        raise CheckpointError(f"{where}: " + "; ".join(errors))
    return manifest


def _read_arrays(npz: Path) -> Dict[str, np.ndarray]:
    """Every array of the archive ``npz``, read with object loading off."""
    if not npz.is_file():
        raise FileNotFoundError(f"pair checkpoint not found: {npz}")
    unreadable = (OSError, ValueError, EOFError, zipfile.BadZipFile)
    try:
        data = np.load(npz, allow_pickle=False)
    except unreadable as exc:
        raise CheckpointError(f"{npz}: not a format-2 checkpoint: {exc}") from None
    if not hasattr(data, "files"):
        raise CheckpointError(f"{npz}: not a format-2 checkpoint: one array, not an archive")
    try:
        with data:
            return {name: data[name] for name in data.files}
    except unreadable as exc:
        raise CheckpointError(f"{npz}: not a format-2 checkpoint: {exc}") from None


def _read_checkpoint(path: PathLike) -> Tuple[Dict[str, Any], Dict[str, Any], List[np.ndarray]]:
    """The manifest, the header and the weights of a format-2 checkpoint, all
    checked against each other."""
    npz = _npz_path(path)
    arrays = _read_arrays(npz)
    version = arrays.get(ARRAY_FORMAT)
    if version is None or version.shape != () or not np.issubdtype(version.dtype, np.integer):
        raise CheckpointError(f"{npz}: not a format-2 checkpoint (no format_version)")
    if int(version) != PAIR_Q_FORMAT_VERSION:
        legacy = " (the legacy DDQN's weight file, ddqn.py)" if int(version) == 1 else ""
        raise CheckpointError(
            f"{npz}: format {int(version)}{legacy}; the pair learner reads format "
            f"{PAIR_Q_FORMAT_VERSION} only")
    raw = arrays.get(ARRAY_HEADER)
    try:
        if raw is None or raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("no header")
        header = json.loads(raw.tobytes().decode("ascii"))
        if not isinstance(header, dict) or set(header) != set(HEADER_KEYS):
            # A header written before resolution R2 lacks the purpose and the revision.
            got = sorted(header) if isinstance(header, dict) else type(header).__name__
            raise ValueError(f"header keys are {list(HEADER_KEYS)}, got {got}")
        config = dict(header["network"])
        width = _int(config.pop("feature_dim"), "feature_dim", 1)
        hidden = PairQConfig.from_json(config).hidden
    except (TypeError, ValueError, KeyError, UnicodeDecodeError) as exc:
        raise CheckpointError(f"{npz}: a format-2 checkpoint with a broken header: {exc}") from None
    names = _param_names(len(hidden) + 1)
    if set(arrays) != set(names) | {ARRAY_FORMAT, ARRAY_HEADER}:
        raise CheckpointError(
            f"{npz}: the arrays are {sorted(arrays)}, not the header's network's "
            f"{sorted(set(names) | {ARRAY_FORMAT, ARRAY_HEADER})}")
    params = []
    fan_in = width
    for i, fan_out in enumerate(hidden + (1,)):
        for name, shape in ((names[2 * i], (fan_in, fan_out)), (names[2 * i + 1], (fan_out,))):
            array = arrays[name]
            if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
                raise CheckpointError(
                    f"{npz}: {name} is a finite float64 array of shape {shape}, got "
                    f"{array.dtype} {array.shape}")
            params.append(array)
        fan_in = fan_out
    manifest = read_manifest(npz)
    sha = arrays_sha256(arrays)
    if sha != manifest["sha256"]:
        raise CheckpointError(
            f"{npz}: its arrays' sha256 is {sha}, not its manifest's {manifest['sha256']}")
    # The sha covers the header, so these are the fields the sha binds: a
    # manifest relabelled (purpose, learner revision, kind, ...) fails here.
    for key in HEADER_KEYS:
        if _canonical(_json_ready(manifest[key], key)) != _canonical(header[key]):
            raise CheckpointError(f"{npz}: the manifest's {key} is not the arrays' header's")
    return manifest, header, params


def verify_checkpoint(path: PathLike) -> Dict[str, Any]:
    """The manifest of the checkpoint ``path``, once its arrays are checked
    against it (format, header, sha): its kind, purpose, learner revision,
    network, schema and classes are then the header's, which the sha binds.
    The runner reads the sha it then puts in the config from here, and judges
    this manifest (:func:`campaign_refusals`)."""
    return _read_checkpoint(path)[0]


def record_held_out(path: PathLike, held_out: Mapping[str, Any]) -> Dict[str, Any]:
    """Fill the manifest's held-out score (the evaluator's, unit U8a); return the
    new manifest.

    ``held_out`` is a JSON object holding at least :data:`HELD_OUT_KEYS`
    (the episodes scored and their mean undiscounted return); an empty or
    scoreless one is refused. The checkpoint is verified first, so a score is
    never written beside arrays that are not its manifest's, nor into a
    relabelled manifest. Only ``held_out`` changes, which is outside the sha;
    the manifest is rewritten whole or not at all.
    """
    manifest = verify_checkpoint(path)
    manifest["held_out"] = _json_object(held_out, "held_out")
    errors = _manifest_errors(manifest)
    if errors:
        raise ValueError("the held-out score is not a manifest's: " + "; ".join(errors))
    _write_atomic(manifest_path(path), lambda handle: handle.write(_manifest_bytes(manifest)))
    return manifest


def campaign_refusals(manifest: Mapping[str, Any], *, allow_dirty: bool = False) -> List[str]:
    """Why a campaign may not fly this checkpoint (critic B9); empty when it may.

    The runner's refusals (the spec, other choices 6): a checkpoint that is
    not ``trained``, has trained on no episode, has no held-out score, or
    was trained from a dirty tree without ``allow_dirty``
    (``--allow-dirty-checkpoint``). A held-out entry that holds no score
    (:data:`HELD_OUT_KEYS`) makes the manifest malformed, which raises
    :class:`CheckpointError` like any other malformed manifest. The loader
    applies none of the refusals, so FerrySim's bootstrap checkpoints load.

    Judge the manifest :func:`verify_checkpoint` returns: its purpose is then
    the header's, which the sha binds, so a bootstrap relabelled as trained
    never reaches this function. ``episodes_trained``, ``held_out`` and
    ``dirty`` stay outside the sha (the held-out score is written after the
    save), so those three refusals catch a mistake, not a hand edit.
    """
    errors = _manifest_errors(manifest)
    if errors:
        raise CheckpointError("not a format-2 manifest: " + "; ".join(errors))
    if not isinstance(allow_dirty, bool):
        raise TypeError(f"allow_dirty must be a bool, got {allow_dirty!r}")
    reasons = []
    if manifest["purpose"] != PURPOSE_TRAINED:
        reasons.append(f"its purpose is {manifest['purpose']!r}, not {PURPOSE_TRAINED!r}")
    if manifest["episodes_trained"] < 1:
        reasons.append(f"it trained on {manifest['episodes_trained']} episodes (at least 1)")
    if manifest["held_out"] is None:
        reasons.append("it has no held-out score (the evaluator fills it)")
    if manifest["dirty"] and not allow_dirty:
        reasons.append("it was trained from a dirty tree (allow it with "
                       "--allow-dirty-checkpoint)")
    return reasons


def manifest_provenance(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The provenance a mule announces for its checkpoint (``mule_ready.pair`` or
    ``.policy_checkpoint``; the spec, other choices 6; design §2.6): the sha,
    kind and purpose, the schema's version and the classes, γ, the reward,
    the seeds, the episodes, FerrySim's cell family and its hash, and the
    learner's revision. The tag is the config's, not the manifest's. From the
    loader's manifest, the kind, purpose, classes, schema, γ and revision
    announced are the ones its sha binds."""
    errors = _manifest_errors(manifest)
    if errors:
        raise CheckpointError("not a format-2 manifest: " + "; ".join(errors))
    keys = ("sha256", "kind", "purpose", "classes", "gamma", "reward", "seeds",
            "episodes_trained", "cell_family", "cell_family_sha256", "learner_revision")
    out = {key: _json_ready(manifest[key], key) for key in keys}
    out["schema"] = manifest["schema"]["version"]
    return out
