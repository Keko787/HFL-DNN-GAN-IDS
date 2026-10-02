"""FeRRy Phase 5 (unit U2): the pair learner's replay, with variable-length candidate sets.

**Why a new module.** The legacy ``ReplayBuffer`` (``selector/replay.py``)
stores one ``(state, reward, next_state, done)`` per step, where
``next_state`` is the single row the online policy picked at the next step
(``replay.py:24-37``), so the legacy target scores that one row and never
takes a max (``ddqn.py:171-186``). The pair learner's double-DQN target
maximises over the next decision's admitted pairs (the Phase 5 spec, other
choices 3; van Hasselt et al.), so a transition carries every candidate row
of the next decision and the mask it was taken under (Boutilier et al.).
The number of candidates changes from decision to decision (up to 15 pairs
at N = 12 and 21 at N = 24 in the design's probes, so the plan's "<= 18
pairs" is no bound: design §0.5 #4), so nothing here has a fixed width: a
batch concatenates the next rows and marks each with its transition
(:attr:`PairBatch.segment`). The legacy files stay untouched, because the
H2 golden pins them (``tests/golden/_mule_harness.py:600-602``).

**A transition** (:class:`PairTransition`) is one decision:

* ``x``: the row of the pair taken (the pair features, unit U1; for E3, the
  row of the stop taken, unit U6);
* ``reward``: r_k (the reward is FerrySim's, unit U8a);
* ``done``: True at the sortie's last decision, whose bootstrap is cut,
  because the flight Q's horizon is the sortie (memo L264; the Phase 5
  spec, other choices 3). A done transition has no next decision;
* otherwise ``next_rows`` and ``next_mask``: the next decision's candidate
  rows, in its order, and which of them ``fits_pair`` admitted. A decision
  whose mask was empty flew FX's pair, and its effective mask is that one
  pair (other choices 1), so a next mask admits at least one row.

The masked rows are kept as given, but nothing reads them: the target's max
runs over the admitted rows alone (``pair_q.PairQNet.targets``), so a
masked row need not even be finite.

**Sampling** is uniform without replacement from a FIFO ring of fixed
capacity (50,000 transitions by default; other choices 3), drawn from
``random.Random(seed)`` as the legacy buffer draws, so a run's batches
depend on its seed and its pushes alone (the spec's determinism convention).

**Layering.** Numpy and the standard library only (the Phase 5 spec,
conventions): E3 trains on this replay and never loads the plan package,
and ``selector/__init__`` does not import it, so no recorded arm loads it.
"""

from __future__ import annotations

import numbers
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_CAPACITY",
    "PairBatch",
    "PairReplay",
    "PairTransition",
]

#: The replay's capacity in transitions (the Phase 5 spec, other choices 3;
#: design D-C (a)). ``pair_q.LearnerSettings.replay_capacity`` restates it,
#: since neither module imports the other; a unit test keeps the two equal.
DEFAULT_CAPACITY = 50_000


def _count(value: Any, name: str) -> int:
    """``value`` as an int >= 0; a bool is refused (``True`` is no count)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def _real(value: Any, name: str) -> float:
    """``value`` as a finite float; a bool is refused."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _floats(value: Any, name: str, ndim: int) -> np.ndarray:
    """A float64 copy of ``value`` with ``ndim`` dimensions."""
    try:
        out = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array of real numbers: {exc}") from None
    if out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {out.shape}")
    return out


def _mask(value: Any, length: int, name: str) -> np.ndarray:
    """A bool copy of ``value``, one flag per row.

    Only a bool array is a mask: 0/1 integers are refused, because an index
    list passed by mistake would read as a mask of the wrong rows.
    """
    out = np.array(value)
    if out.dtype != np.bool_:
        raise TypeError(f"{name} must hold bools, got dtype {out.dtype}")
    if out.shape != (length,):
        raise ValueError(f"{name} holds one flag per row ({length}), got shape {out.shape}")
    return out


def _frozen(array: np.ndarray) -> np.ndarray:
    """``array``, read-only: a stored transition never changes under the replay."""
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class PairTransition:
    """One decision: the row taken, its reward, and the next decision's candidates.

    ``next_rows`` and ``next_mask`` are both None exactly when ``done``: the
    sortie's last decision bootstraps from nothing, and any other decision
    bootstraps from the next one (a ``home`` decision that a beacon insert
    follows bootstraps from the insert's decision, critic B12). The arrays
    are copied and stored read-only (float64 rows, a bool mask).
    """

    x: np.ndarray
    reward: float
    done: bool
    next_rows: Optional[np.ndarray] = None
    next_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        x = _floats(self.x, "x", 1)
        if x.size == 0:
            raise ValueError("x is the taken pair's row: it has at least one feature")
        if not np.isfinite(x).all():
            raise ValueError("x is the taken pair's row: its features are finite")
        reward = _real(self.reward, "reward")
        if not isinstance(self.done, bool):
            raise TypeError(f"done must be a bool, got {self.done!r}")
        rows = mask = None
        if self.done:
            if self.next_rows is not None or self.next_mask is not None:
                raise ValueError(
                    "a done transition is the sortie's last decision and bootstraps from "
                    "nothing: its next_rows and next_mask are None"
                )
        else:
            if self.next_rows is None or self.next_mask is None:
                raise ValueError(
                    "a transition that is not done carries the next decision's candidate rows "
                    "and mask (next_rows and next_mask)"
                )
            rows = _floats(self.next_rows, "next_rows", 2)
            if rows.shape[0] == 0 or rows.shape[1] != x.shape[0]:
                raise ValueError(
                    f"next_rows holds at least one candidate row of x's width {x.shape[0]}, "
                    f"got shape {rows.shape}"
                )
            mask = _mask(self.next_mask, rows.shape[0], "next_mask")
            if not mask.any():
                raise ValueError(
                    "next_mask admits at least one row: an empty mask flies FX's pair, which "
                    "is then the decision's one admitted pair (the Phase 5 spec, other choices 1)"
                )
            if not np.isfinite(rows[mask]).all():
                raise ValueError("the admitted next rows are finite")
            rows, mask = _frozen(rows), _frozen(mask)
        object.__setattr__(self, "x", _frozen(x))
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "next_rows", rows)
        object.__setattr__(self, "next_mask", mask)

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True, eq=False)
class PairBatch:
    """Transitions stacked for one update, their next candidates concatenated.

    * ``x``: (B, F), the rows taken; ``reward``: (B,); ``done``: (B,) bools;
    * ``next_rows``: (M, F), every candidate row of every next decision, the
      transitions' sets one after another in transition order;
    * ``next_mask``: (M,) bools, the masks concatenated the same way;
    * ``segment``: (M,) ints, the transition each next row belongs to: a
      non-decreasing sequence in [0, B), so each transition's rows are
      contiguous and in their decision's order.

    A done transition owns no rows and every other one owns an admitted row.
    The admitted rows are finite; the masked ones are never read.
    """

    x: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_rows: np.ndarray
    next_mask: np.ndarray
    segment: np.ndarray

    def __post_init__(self) -> None:
        x = _floats(self.x, "x", 2)
        size, width = x.shape
        if size == 0 or width == 0:
            raise ValueError(f"x holds at least one row of at least one feature, got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError("the rows taken are finite")
        reward = _floats(self.reward, "reward", 1)
        if reward.shape != (size,) or not np.isfinite(reward).all():
            raise ValueError(f"reward holds one finite reward per transition ({size})")
        done = _mask(self.done, size, "done")
        rows = _floats(self.next_rows, "next_rows", 2)
        if rows.shape[1] != width:
            raise ValueError(f"next_rows are rows of x's width {width}, got shape {rows.shape}")
        mask = _mask(self.next_mask, rows.shape[0], "next_mask")
        segment = np.array(self.segment)
        if segment.shape != (rows.shape[0],) or (
                segment.size and not np.issubdtype(segment.dtype, np.integer)):
            raise ValueError(
                f"segment holds one transition index per next row ({rows.shape[0]}), got "
                f"{segment.dtype} {segment.shape}"
            )
        segment = segment.astype(np.int64)
        if segment.size and (segment.min() < 0 or segment.max() >= size
                             or (np.diff(segment) < 0).any()):
            raise ValueError(
                f"segment is non-decreasing in [0, {size}): each transition's rows are "
                f"contiguous, in transition order"
            )
        owned = np.bincount(segment, minlength=size)
        admitted = np.bincount(segment[mask], minlength=size)
        if (owned[done] > 0).any():
            raise ValueError("a done transition owns no next rows: it bootstraps from nothing")
        if (admitted[~done] == 0).any():
            raise ValueError("every transition that is not done owns at least one admitted row")
        if not np.isfinite(rows[mask]).all():
            raise ValueError("the admitted next rows are finite")
        for name, value in (("x", x), ("reward", reward), ("done", done), ("next_rows", rows),
                            ("next_mask", mask), ("segment", segment)):
            object.__setattr__(self, name, _frozen(value))

    @classmethod
    def of(cls, transitions: Sequence[PairTransition]) -> "PairBatch":
        """The batch of ``transitions``, in their order."""
        items = list(transitions)
        if not items:
            raise ValueError("a batch holds at least one transition")
        for item in items:
            if not isinstance(item, PairTransition):
                raise TypeError(f"a batch holds PairTransition objects, got {item!r}")
        width = items[0].feature_dim
        if any(item.feature_dim != width for item in items):
            raise ValueError(
                f"a batch's rows share one width: {sorted({item.feature_dim for item in items})}"
            )
        live = [(i, item) for i, item in enumerate(items) if not item.done]
        if live:
            rows = np.concatenate([item.next_rows for _, item in live], axis=0)
            mask = np.concatenate([item.next_mask for _, item in live], axis=0)
            segment = np.concatenate(
                [np.full(item.next_rows.shape[0], i, dtype=np.int64) for i, item in live])
        else:
            rows = np.zeros((0, width), dtype=np.float64)
            mask = np.zeros(0, dtype=np.bool_)
            segment = np.zeros(0, dtype=np.int64)
        return cls(
            x=np.stack([item.x for item in items], axis=0),
            reward=np.array([item.reward for item in items], dtype=np.float64),
            done=np.array([item.done for item in items], dtype=np.bool_),
            next_rows=rows,
            next_mask=mask,
            segment=segment,
        )

    @property
    def size(self) -> int:
        """B, the number of transitions."""
        return int(self.x.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])


class PairReplay:
    """A FIFO ring of :class:`PairTransition`, sampled uniformly from a seeded stream.

    The first push fixes the row width; a transition of another width is
    refused, so a schema mix-up fails at the push instead of inside an
    update. Sampling draws ``batch_size`` distinct stored transitions with
    ``random.Random(seed).sample``, the legacy buffer's draw
    (``replay.py:65-72``); pushing draws nothing.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, *, seed: int):
        capacity = _count(capacity, "capacity")
        if capacity == 0:
            raise ValueError("capacity must be >= 1")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise TypeError(f"seed must be an int, got {seed!r}")
        self._capacity = capacity
        self._seed = int(seed)
        self._items: List[PairTransition] = []
        self._next = 0
        self._pushed = 0
        self._width: Optional[int] = None
        self._rng = random.Random(self._seed)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return (f"PairReplay(capacity={self._capacity}, seed={self._seed}, size={len(self)}, "
                f"pushed={self._pushed})")

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def pushed(self) -> int:
        """Transitions pushed since the start, overwritten ones included."""
        return self._pushed

    @property
    def feature_dim(self) -> Optional[int]:
        """The row width, fixed by the first push; None before it."""
        return self._width

    def push(self, transition: PairTransition) -> None:
        """Store ``transition``, overwriting the oldest once the ring is full."""
        if not isinstance(transition, PairTransition):
            raise TypeError(f"the replay stores PairTransition objects, got {transition!r}")
        if self._width is None:
            self._width = transition.feature_dim
        elif transition.feature_dim != self._width:
            raise ValueError(
                f"the replay's rows are {self._width} wide, got a transition of "
                f"{transition.feature_dim}"
            )
        if len(self._items) < self._capacity:
            self._items.append(transition)
        else:
            self._items[self._next] = transition
        self._next = (self._next + 1) % self._capacity
        self._pushed += 1

    def sample(self, batch_size: int) -> PairBatch:
        """``batch_size`` distinct stored transitions, uniformly, as one batch."""
        batch_size = _count(batch_size, "batch_size")
        if batch_size == 0:
            raise ValueError("batch_size must be >= 1")
        if batch_size > len(self._items):
            raise ValueError(f"batch_size={batch_size} > the replay's size {len(self._items)}")
        picks = self._rng.sample(range(len(self._items)), batch_size)
        return PairBatch.of([self._items[i] for i in picks])
