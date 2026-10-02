"""FeRRy Phase 5 (unit U6): arm E3, a numpy port of Chen et al.'s DQN recipe.

**What it is.** E3 is the learned competitor after Chen, Esrafilian,
Bayerlein, Gesbert and Caccamo (GLOBECOM Workshops 2023; build plan L1026,
L1304): a deep Q-network that flies a UAV to collect data, sees per device
its SNR, whether it is reachable, how much data is left and where it lies,
plus its own battery, and chooses among the actions a safety controller
leaves it. The user's decision 7 (a) builds it as a numpy re-implementation
of that recipe, trained in FerrySim, because the vendored drone_env DQN
needs PyTorch, sits where ``hermes`` may not import it and has no licence
(the Phase 5 design, finding 8).

**Where it sits** (the Phase 5 spec, other choices 11). E3 is a legacy-mode
whole-scheduler policy (``contact_policy = "chen_dqn"``), as D1-D5 are: the
scheduler hands it S3a's contacts and takes its route
(:meth:`ChenDQNPolicy.admit_and_order`), and the mule re-checks before each
stop what it declares (``in_flight_check``). Unlike D1-D5 it chooses each
next stop in flight, through the per-departure protocol of
``policies/next_stop.py`` (``chooses_next_stop``; the supervisor's hook,
unit U5):

* **Before takeoff, every contact.** :meth:`~ChenDQNPolicy.admit_and_order`
  admits every S3a contact, nearest the takeoff pose first, whatever the
  budget or the deadlines: Chen's safety controller acts at every step, not
  once before the flight, so nothing is dropped before takeoff and the
  scheduler reports no policy drop.
* **In flight, no check** (``in_flight_check = "none"``,
  ``budget_walk.IN_FLIGHT_NONE``): the departure check keeps the remainder as
  it stands and never re-plans (``FLScheduler.in_flight_rule`` is
  ``RULE_NONE``), since E3 itself chooses only stops that fit.
* **At takeoff and at every Pass-1 departure,** :meth:`~ChenDQNPolicy.next_stop`
  scores one row per stop still to fly (:func:`e3_rows`, from
  ``FerryRuntime.e3_observation``) and takes the admissible row with the
  highest Q, ties to the lowest row (``pair_q.masked_argmax``).
  ``admissible(i)`` is Chen's safety controller: S3b's single-contact
  predicate under ``RULE_BUDGET`` from the departure's state (transit, dwell,
  the return leg and the upload within the budget's end, and the energy
  clause; no deadline), which the supervisor binds. When no stop is
  admissible the answer is None: the pass ends, the mule flies home and the
  stops left are reported (``pass_1_e3_unvisited``), never widened. Never
  in Pass 2, which delivers to every slice stop nearest first
  (:func:`next_stop.pass_1_only`; critic B7 i).
* **One band, the cell's.** E3 flies the cell's ``contact_band`` at every
  stop (build plan L1026: FL-blind and band-fixed): the policy names no band,
  legacy mode never moves the runtime's, and a checkpoint is bound to the
  band it trained on (its class tuple), so a view on another band is
  refused rather than scored.

E3 flies none of FeRRy's machinery (build plan L1026): no deadline, no plan,
no coverage term, no age cap. It is rewarded in bytes, |C_k| / N, the
updates collected at the stop over the devices the mule answers for (the
design D-E; ``experiments/ferrysim/reward.py``'s ``BYTES``), since every
update has one size.

**The observation, per candidate stop** (the design D-J (a); :data:`E3_COLUMNS`).
Chen's per-device features (his o2, arXiv 2306.02029 section III: the SNR,
the reachable flag, the remaining data, the distance and the relative x and
y) are aggregated over the stop's members, and his own features (o4: the
battery and the battery needed to reach the destination) become the energy
left and the energy of the return from the stop, with the time left beside
them because FeRRy's sortie is bounded by a time budget as Chen's is by the
battery:

* ``remaining``: the share of the stop's updates not yet collected this
  mission, Chen's remaining data D. Every candidate is a stop not flown yet
  and each device holds one update of one size, so it is 1 on every row.
  Chen zeroes D when the device is out of reach; aggregated over a stop
  that would copy ``reachable``, so the column holds D itself;
* ``snr``: the median over the members of the SNR now from the mule's pose
  on the contact band, realized within reach and the mean ("radio map")
  SNR beyond, in units of :data:`SNR_SCALE_DB` (critic B7 iv);
* ``reachable``: the share of the members reachable now from the pose
  (within R_planar and at or above the SNR floor), Chen's ζ. Each candidate
  is a stop of its own, away from the pose, so from most poses none of its
  members is;
* ``dx``, ``dy``, ``distance``: where the stop lies from the pose (the stop
  less the pose; Chen's sign is the other) and the leg's length on the flight
  model's metric, in units of :data:`LENGTH_SCALE_M`;
* ``members``: the stop's members over N, the most the stop can earn;
* ``return_energy``: the energy of the return from the stop to the dock as a
  share of the sortie's energy reference, Chen's b_sc taken at the
  candidate; 0 without a reference;
* ``energy_left``: 1 less the energy spent this sortie over the reference,
  Chen's battery b; 1 without a reference (nothing binds);
* ``time_left``: the budget's end less the clock, over the budget; 1 without
  a budget. The last two are the same on every row of a call, as the pair
  features' pooled context is.

**Declared deviations from Chen** (decision 7 (a); the design D-J (a)):

* **stops rather than grid moves**: an action is a stop of the contact graph
  (S3a's contacts), served whole by the stack's contact, not one of Chen's
  six grid moves with a TDMA max-rate device per step. Chen's collection
  status q, which marks the device being read at a step, is 0 for every
  candidate at a departure and is left out;
* **one agent**: Chen's o3 (the other UAVs) and QMIX's mixing network are
  dropped. The learner is the pair learner's masked double DQN
  (``selector/pair_q.py``: Adam, the Huber loss, a hard target sync, the
  safety controller's mask in the target's max too), not Chen's QMIX/IQL
  settings. Chen's own learning rate is 5e-4 (Adam; section V); the trainer
  states its own (unit U8b);
* **no model-aided learning**: Chen's learned channel and device model and
  the simulated episodes it feeds are not built (build plan L1272); E3
  trains model-free in FerrySim;
* **constant or near-zero parts**: ``remaining`` is 1 on every row and
  ``reachable`` is 0 on most (critic B7 ii; about four rows in five on
  FerrySim-like cells, with a mean near 0.1). Both are kept for fidelity and
  declared (:data:`E3_DECLARED_CONSTANT`), so the test that no column is
  constant exempts them;
* **K**: E3 is trained at K = 1 and flown per mule at K = 3 in Study 5.3,
  each mule on a slice within the trained sizes (critic B7 v); the rows are
  per mule (N is the mule's) and relative to its pose, so a slice reads as
  a smaller cell.

**Checkpoints.** E3's weights are a format-2 checkpoint of the pair learner
(``pair_q``) of kind ``chen_dqn``, with :func:`e3_schema` as its feature
schema and the contact band as its one class, so the sha a config names
binds the weights, the rows they read, their scales and the band
(:func:`save_e3_checkpoint`, :func:`load_e3_network`). The mule builds the
policy from the config's checkpoint (:meth:`ChenDQNPolicy.from_checkpoint`,
``processes/mule.py``, unit U7) and refuses a sha, kind, schema or band
mismatch; a bootstrap checkpoint loads, as FerrySim's training starts from
one (the runner, not the mule, refuses it for campaigns, critic B9).

**Training (FerrySim only).** :meth:`ChenDQNPolicy.attach_trainer` hands the
policy the trainer's live network, ε, the episode's seeded stream and a
sink, before the policy's first call. Every decision is then ε-greedy over
the admissible stops (``pair_q.behaviour_row``; ε-greedy over the safety
controller's actions is Chen's rule, his Algorithm 1, line 13) on the live
network's online Q as it stands at that decision (never the target copy,
synced only every ``PairQConfig.target_sync`` updates, 500 by default),
with no reference phase (Chen's recipe has none), and each one reaches the
sink as an :class:`E3Step`. The supervisor makes no close call to a
next-stop policy, so the sink is called per decision, and a mission's steps
are the decisions of its Pass-1 stops flown, in order (see :class:`E3Step`).

**Layering.** Numpy, the standard library, ``hermes.types``, the protocol
module (``policies/next_stop.py``), ``policies/budget_walk.py`` and the pair
learner (``selector/pair_q.py``), and never the plan package, ``hermes.l1``,
``hermes.mule``, ``hermes.mission`` or ``experiments`` (the Phase 5 spec,
conventions): E3 flies none of the plan's machinery. ``policies/__init__``
does not import this module; only the mule builder does, for
``contact_policy = "chen_dqn"``, so no other arm loads it. Deterministic:
no wall time, and no draw but the trainer's seeded stream.
"""

from __future__ import annotations

import json
import math
import numbers
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hermes.scheduler.policies.budget_walk import IN_FLIGHT_NONE
from hermes.scheduler.policies.next_stop import E3View, pass_1_only
from hermes.scheduler.selector.pair_q import (
    KIND_CHEN_DQN,
    PairQConfig,
    PairQNet,
    behaviour_row,
    masked_argmax,
)
from hermes.types.scheduler import ContactWaypoint

__all__ = [
    "E3_COLUMNS",
    "E3_DECLARED_CONSTANT",
    "E3_DIM",
    "E3_SCHEMA_VERSION",
    "LENGTH_SCALE_M",
    "SNR_SCALE_DB",
    "ChenDQNPolicy",
    "E3Sink",
    "E3Step",
    "e3_rows",
    "e3_schema",
    "load_e3_network",
    "new_e3_network",
    "save_e3_checkpoint",
]

PathLike = Union[str, "os.PathLike[str]"]

#: The rows' schema name, stored in every E3 checkpoint's header (so its sha
#: binds it) and compared whole when one is loaded.
E3_SCHEMA_VERSION = "e3_v1"

#: One row per candidate stop, these columns in this order (module docstring).
E3_COLUMNS: Tuple[str, ...] = (
    "remaining", "snr", "reachable", "dx", "dy", "distance", "members",
    "return_energy", "energy_left", "time_left",
)
E3_DIM = len(E3_COLUMNS)

#: The columns kept for fidelity to Chen though they are constant or near zero
#: here (critic B7 ii): ``remaining`` is 1 on every row, ``reachable`` 0 on
#: most. Every other column varies over a FerrySim-like sample (a test).
E3_DECLARED_CONSTANT: Tuple[str, ...] = ("remaining", "reachable")

#: The rows' unit of length: the realism field's half-width, where the Exp 4
#: driver and FerrySim's cells scatter their devices (``h1_field_radius_m``,
#: 100 m, ``experiments/exp4/driver.py``), so ``dx``, ``dy`` and
#: ``distance`` stay within about +-2 there.
LENGTH_SCALE_M = 100.0

#: The rows' unit of SNR: a decade per unit. On FerrySim-like cells E3's
#: median SNR runs from about -13 to +15 dB (most of it the mean SNR beyond
#: reach), so ``snr`` stays within about +-1.5.
SNR_SCALE_DB = 10.0


def e3_schema() -> Dict[str, Any]:
    """E3's feature schema, as its checkpoints record it (a fresh dict).

    ``version`` and ``dim`` are what the pair learner requires of a schema;
    the columns, the declared columns and the two scales are E3's own, so a
    checkpoint trained on other rows or other units is refused on load
    (``PairQNet.load`` compares the schema whole). The band is not here: it
    is the checkpoint's class tuple.
    """
    return {
        "version": E3_SCHEMA_VERSION,
        "dim": E3_DIM,
        "columns": list(E3_COLUMNS),
        "declared_constant": list(E3_DECLARED_CONSTANT),
        "length_scale_m": LENGTH_SCALE_M,
        "snr_scale_db": SNR_SCALE_DB,
    }


def e3_rows(view: E3View) -> np.ndarray:
    """E3's rows for ``view``: (K, :data:`E3_DIM`) float64, one per stop in its order.

    Row i is the remainder's stop i, as ``admissible(i)`` and the policy's
    answer are. See the module docstring for each column and for what the
    sortie's columns read when the view has no energy reference
    (``energy_ref_j`` None: ``return_energy`` 0, ``energy_left`` 1) or no
    budget (``time_left`` 1). Every value is finite: the view's fields are
    checked finite, and its references are > 0 or None.
    """
    if not isinstance(view, E3View):
        raise TypeError(f"E3's rows are read from an E3View, got {view!r}")
    n = float(view.demand)
    e_ref = view.energy_ref_j
    energy_left = 1.0 if e_ref is None else 1.0 - view.energy_j / e_ref
    if view.budget_end is None or view.budget_s is None:
        time_left = 1.0
    else:
        time_left = (view.budget_end - view.clock_s) / view.budget_s
    out = np.empty((len(view.stops), E3_DIM), dtype=np.float64)
    for i, stop in enumerate(view.stops):
        values = {
            "remaining": stop.remaining,
            "snr": stop.snr_db / SNR_SCALE_DB,
            "reachable": stop.reachable,
            "dx": stop.dx_m / LENGTH_SCALE_M,
            "dy": stop.dy_m / LENGTH_SCALE_M,
            "distance": stop.distance_m / LENGTH_SCALE_M,
            "members": stop.members / n,
            "return_energy": 0.0 if e_ref is None else stop.return_energy_j / e_ref,
            "energy_left": energy_left,
            "time_left": time_left,
        }
        out[i] = [values[name] for name in E3_COLUMNS]
    return out


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def _band(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"E3 flies one contact band, a class name; got {value!r}")
    return value


def _e3_net(net: Any, what: str = "net") -> PairQNet:
    """``net`` as E3's network: a pair learner whose rows are E3's."""
    if not isinstance(net, PairQNet):
        raise TypeError(f"{what} must be a PairQNet (selector.pair_q), got {net!r}")
    if net.feature_dim != E3_DIM:
        raise ValueError(
            f"{what} scores rows {net.feature_dim} wide, and E3's rows are {E3_DIM} "
            f"({', '.join(E3_COLUMNS)})")
    return net


def _verdict(value: Any, index: int) -> bool:
    """``admissible(i)``'s answer, which must be a bool (the predicate's ``.ok``).

    Anything else is refused rather than read for its truth: a verdict passed
    whole would be truthy and admit every stop.
    """
    if not isinstance(value, bool):
        raise TypeError(f"admissible({index}) must answer a bool (the predicate's .ok), got "
                        f"{value!r}")
    return value


def _pose(value: Any) -> Tuple[float, ...]:
    try:
        pose = tuple(float(c) for c in value)
    except TypeError:
        raise TypeError(f"the mule's pose is a sequence of coordinates, got {value!r}") from None
    if not pose or not all(math.isfinite(c) for c in pose):
        raise ValueError(f"the mule's pose is finite coordinates, got {value!r}")
    return pose


def _nearest_key(pose: Tuple[float, ...], wp: ContactWaypoint) -> Tuple[Any, ...]:
    """Nearest the pose first (the flight model's metric), then the position and
    the members, so the order never depends on the order the contacts came in."""
    position = tuple(float(c) for c in wp.position)
    distance = sum((x - y) ** 2 for x, y in zip(pose, position)) ** 0.5
    return (distance, position, tuple(str(d) for d in wp.devices))


def _plain_manifest(manifest: Any, band: str) -> Dict[str, Any]:
    """A checkpoint manifest kept as a fresh JSON copy, checked to be an E3 one of ``band``."""
    if not isinstance(manifest, Mapping):
        raise TypeError(f"manifest is a checkpoint's manifest (a mapping), got {manifest!r}")
    try:
        out = json.loads(json.dumps(dict(manifest), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"manifest is not JSON: {exc}") from None
    if out.get("kind") != KIND_CHEN_DQN:
        raise ValueError(f"the manifest is a {out.get('kind')!r} checkpoint's, not "
                         f"{KIND_CHEN_DQN!r}")
    if out.get("classes") != [band]:
        raise ValueError(f"the manifest's checkpoint trained on {out.get('classes')}, and this "
                         f"policy flies {[band]}")
    if out.get("schema") != e3_schema():
        raise ValueError("the manifest's checkpoint reads other rows than E3's (its schema)")
    return out


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #

def new_e3_network(*, seed: int, config: Optional[PairQConfig] = None) -> PairQNet:
    """A fresh network over E3's rows, drawn from ``seed`` (``PairQNet``'s draws).

    For the trainer's bootstrap checkpoint (unit U8b) and for tests: the mule
    never flies one that a checkpoint does not hold (no random-init arm).
    """
    return PairQNet(E3_DIM, config, seed=seed)


def save_e3_checkpoint(net: PairQNet, path: PathLike, *, band: str, purpose: str,
                       provenance: Mapping[str, Any]) -> str:
    """Write ``net`` as an E3 checkpoint; return its sha256.

    ``PairQNet.save`` with E3's kind (``chen_dqn``), :func:`e3_schema` and
    ``[band]`` as the class tuple, so the loader the mule runs
    (:func:`load_e3_network`) reads exactly what this writes. ``purpose`` is
    ``bootstrap`` (FerrySim's training starts from one) or ``trained``;
    ``provenance`` is the trainer's (``pair_q.PROVENANCE_KEYS``).
    """
    return _e3_net(net).save(path, kind=KIND_CHEN_DQN, purpose=purpose, schema=e3_schema(),
                             classes=[_band(band)], provenance=provenance)


def load_e3_network(path: PathLike, *, expect_sha256: str, band: str
                    ) -> Tuple[PairQNet, Dict[str, Any]]:
    """The network and verified manifest of the E3 checkpoint ``path``.

    ``PairQNet.load`` expecting the sha the config names, E3's kind, its
    schema and ``[band]``: any other refuses with ``pair_q.CheckpointError``
    (a ``ValueError``), as does a checkpoint the manifest does not describe.
    No training state is checked, so a bootstrap checkpoint loads.
    """
    return PairQNet.load(path, expect_sha256=expect_sha256, expect_kind=KIND_CHEN_DQN,
                         expect_schema=e3_schema(), expect_classes=[_band(band)])


# --------------------------------------------------------------------------- #
# The trainer's view of a decision
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class E3Step:
    """One E3 decision as the trainer reads it (FerrySim only).

    ``view`` is what the policy saw; ``rows`` its rows (:func:`e3_rows`, as
    tuples, so steps compare by value); ``mask`` ``admissible``'s verdict on
    each, at least one True; ``scores`` the online Q of each row, the one the
    decision was taken on; ``row`` the stop flown; ``after_stop`` False at
    takeoff; ``devices`` the members of the stop flown.

    **Forming transitions** (unit U8b). A step is made at every call that
    names a stop, and each named stop is a Pass-1 stop flown, so a mission's
    steps are its Pass-1 stops flown, in order (``pass_1_flown``; FerrySim's
    ``SortieRecord.stops``), each step's ``devices`` that stop's; a mission
    with no stop flown has no step, and the first step of each mission is its
    takeoff call. Step k's transition has x = ``rows[row]``, the reward of
    stop k, and the next step's ``rows`` and ``mask`` as its next decision;
    the mission's last step is done (the flight Q's horizon is the sortie,
    memo L264), whether the pass ended because no stop was admissible or
    because none was left. ``mask`` always admits ``row``, so it is the
    effective mask too.
    """

    view: E3View
    rows: Tuple[Tuple[float, ...], ...]
    mask: Tuple[bool, ...]
    scores: Tuple[float, ...]
    row: int
    after_stop: bool
    devices: Tuple[str, ...]

    @property
    def x(self) -> np.ndarray:
        """The row flown, as a fresh float64 array."""
        return np.array(self.rows[self.row], dtype=np.float64)

    @property
    def matrix(self) -> np.ndarray:
        """Every row, (K, :data:`E3_DIM`), as a fresh float64 array."""
        return np.array(self.rows, dtype=np.float64)

    @property
    def effective_mask(self) -> Tuple[bool, ...]:
        """The stops the decision was taken among: the mask itself (it admits ``row``)."""
        return self.mask


#: The trainer's sink: called with each decision's :class:`E3Step`, in order.
E3Sink = Callable[[E3Step], None]


@dataclass(frozen=True)
class _Trainer:
    epsilon: float
    rng: Any
    sink: E3Sink


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #

class ChenDQNPolicy:
    """Arm E3 (``contact_policy = "chen_dqn"``): a learned next stop at every departure.

    A legacy-mode whole-scheduler policy that declares the per-departure
    protocol (``policies/next_stop.py``); see the module docstring. Built by
    the mule from a checkpoint (:meth:`from_checkpoint`) or around a network
    (tests, FerrySim); ``band`` is the cell's contact band, the one class it
    flies.
    """

    #: The config's switch value (``processes.config.CONTACT_POLICY_CHEN_DQN``)
    #: and the checkpoint kind (``pair_q.KIND_CHEN_DQN``) restate it; a test
    #: pins the three equal.
    name: str = "chen_dqn"
    #: Freeze Amendment 8: what the mule re-checks before each stop in flight,
    #: nothing, since the policy flies only stops its safety controller admits.
    in_flight_check: str = IN_FLIGHT_NONE
    #: E3 visits a stop for all its members (the config guard requires
    #: ``member_admission = "whole"``); the scheduler refuses member subsets to
    #: a whole-scheduler policy that does not take them.
    admits_member_subsets: bool = False
    #: The per-departure protocol (``next_stop.NextStopPolicy``): the
    #: supervisor asks :meth:`next_stop` for each Pass-1 stop, takeoff included.
    chooses_next_stop: bool = True

    def __init__(self, net: PairQNet, *, band: str,
                 manifest: Optional[Mapping[str, Any]] = None) -> None:
        self._net = _e3_net(net)
        self._band = _band(band)
        self._manifest = None if manifest is None else _plain_manifest(manifest, self._band)
        self._trainer: Optional[_Trainer] = None
        self._called = False

    @classmethod
    def from_checkpoint(cls, path: PathLike, *, expect_sha256: str,
                        band: str) -> "ChenDQNPolicy":
        """The policy flying the E3 checkpoint ``path`` (:func:`load_e3_network`).

        The mule's builder (``processes/mule.py``) calls this with the
        config's ``policy_checkpoint``, ``policy_checkpoint_sha256`` and
        ``contact_band``; every refusal is the loader's. The verified manifest
        is kept for the mule's provenance (``mule_ready.policy_checkpoint``).
        """
        net, manifest = load_e3_network(path, expect_sha256=expect_sha256, band=band)
        return cls(net, band=band, manifest=manifest)

    @property
    def net(self) -> PairQNet:
        """The network whose online copy scores the rows: the checkpoint's, or
        the trainer's live one once a trainer is attached."""
        return self._net

    @property
    def band(self) -> str:
        """The contact band E3 flies, its checkpoint's one class."""
        return self._band

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        """The verified manifest of the checkpoint flown (a fresh copy), None
        when the policy was built around a network."""
        return None if self._manifest is None else json.loads(json.dumps(self._manifest))

    @property
    def training(self) -> bool:
        """True once a trainer is attached (FerrySim only)."""
        return self._trainer is not None

    def admit_and_order(
        self,
        contacts: Sequence[ContactWaypoint],
        device_states: Any,
        env: Any,
        *,
        mission_deadline_ts: Optional[float] = None,
        feasibility_model: Any = None,
    ) -> List[ContactWaypoint]:
        """Every contact, nearest ``env.mule_pose`` first: E3's Pass-1 route.

        Chen's safety controller filters the actions at every step, so E3
        admits every S3a contact before takeoff and leaves the budget to the
        controller at each departure (:meth:`next_stop`): nothing is dropped
        here, whatever the budget (``mission_deadline_ts``) or the contacts'
        deadlines, and neither those nor ``device_states`` and
        ``feasibility_model`` are read. The order names the rows E3 scores
        (the lowest row wins a tie) and so is a tie-break only: nearest the
        takeoff pose first, then the position and the members, so it never
        depends on the order the contacts came in. Pass 2 builds its own
        order. The contacts are returned themselves (identity kept), so the
        scheduler reports no policy drop.
        """
        pose = _pose(getattr(env, "mule_pose", None))
        stops = list(contacts)
        for wp in stops:
            if not isinstance(wp, ContactWaypoint):
                raise TypeError(f"E3 admits S3a's contacts (ContactWaypoint), got {wp!r}")
        return sorted(stops, key=lambda wp: _nearest_key(pose, wp))

    def next_stop(self, remainder: Sequence[ContactWaypoint], state: Any, *, view: E3View,
                  admissible: Callable[[int], bool], pass_kind: Any,
                  after_stop: bool) -> Optional[int]:
        """The remainder index to fly next, or None when no stop is admissible.

        Pass 1 only (:func:`next_stop.pass_1_only`; critic B7 i). ``view`` is
        Chen's observation of the remainder from the departure, on the
        policy's band (another band is refused: E3 is band-fixed), one stop
        per remainder stop, members matching. ``admissible(i)`` is asked
        about every stop, and must answer a bool. When none is admissible
        the pass ends: None, and nothing is scored or drawn. Otherwise the
        network's online copy scores :func:`e3_rows` with its weights as they
        stand at this call (``PairQNet.q``: a DQN acts on the weights being
        learned, and its target copy enters only the learner's targets), and
        the answer is the admissible row with the highest Q, ties to the
        lowest row (``pair_q.masked_argmax``), or, with a trainer attached,
        ε-greedy over the admissible rows (``pair_q.behaviour_row``: two
        draws from the trainer's stream), and the decision goes to the
        trainer's sink (:class:`E3Step`). ``state`` is not read: the view
        holds what Chen's agent observes.
        """
        pass_1_only(pass_kind)
        self._called = True
        stops = list(remainder)
        if not stops:
            raise ValueError("E3 chooses among the stops still to fly: none is left")
        for wp in stops:
            if not isinstance(wp, ContactWaypoint):
                raise TypeError(f"the remainder holds ContactWaypoints, got {wp!r}")
        if not isinstance(view, E3View):
            raise TypeError(f"E3 reads its observation as an E3View, got {view!r}")
        if view.band != self._band:
            raise ValueError(
                f"E3 flies one band, {self._band!r} (its checkpoint's class), and the view "
                f"is on {view.band!r}: the band is fixed (build plan L1026)")
        if len(view.stops) != len(stops):
            raise ValueError(f"the view observes {len(view.stops)} stop(s) and "
                             f"{len(stops)} are left: one row per stop left")
        for i, (seen, wp) in enumerate(zip(view.stops, stops)):
            if seen.members != len(wp.devices):
                raise ValueError(f"the view's stop {i} has {seen.members} member(s) and the "
                                 f"remainder's {len(wp.devices)}: it observes other stops")
        if not isinstance(after_stop, bool):
            raise TypeError("after_stop is True at a departure from a stop served and False at "
                            f"takeoff, got {after_stop!r}")
        if not callable(admissible):
            raise TypeError("E3 needs admissible(i) -> bool, its safety controller: it never "
                            "flies an unchecked stop")
        mask = tuple(_verdict(admissible(i), i) for i in range(len(stops)))
        if not any(mask):
            return None
        rows = e3_rows(view)
        scores = self._net.q(rows)
        flags = np.array(mask, dtype=bool)
        trainer = self._trainer
        if trainer is None:
            row = masked_argmax(scores, flags)
        else:
            row = behaviour_row(scores, flags, epsilon=trainer.epsilon, rng=trainer.rng)
        if not mask[row]:
            raise RuntimeError(f"E3 picked stop {row}, which its safety controller refused")
        if trainer is not None:
            trainer.sink(E3Step(
                view=view,
                rows=tuple(tuple(float(v) for v in r) for r in rows),
                mask=mask,
                scores=tuple(float(q) for q in scores),
                row=int(row),
                after_stop=after_stop,
                devices=tuple(str(d) for d in stops[row].devices),
            ))
        return int(row)

    def attach_trainer(self, *, net: PairQNet, epsilon: float, rng: Any, sink: E3Sink,
                       around_reference: bool = False) -> None:
        """Fly every later decision ε-greedy with the trainer's live ``net``, and
        hand each one to ``sink``.

        FerrySim only (the Phase 5 spec, other choices 8; the design D-F
        (a)): the mule builds the policy from the run's bootstrap checkpoint
        (the config path), and the trainer, which keeps one network for the
        whole run, attaches it here, so every decision scores with its online
        weights as they stand at that call, updates made between decisions
        included (:meth:`next_stop`). With probability ``epsilon`` a decision
        is uniform over the admissible stops, otherwise the Q argmax
        (``pair_q.behaviour_row``, two draws from ``rng`` at every decision
        with an admissible stop, so runs that differ in ε draw alike); ε = 0
        flies as no trainer does and only collects the steps. E3 has no
        reference to fly around (Chen's recipe explores around its own Q;
        its behaviour schedule has no reference phase), so
        ``around_reference``, accepted so that a ``pair_q.Behaviour`` can be
        passed whole, must be False. Attach before the policy's first call,
        so every decision of an episode is trained alike: the mule builds a
        fresh policy per trial.
        """
        if self._called:
            raise RuntimeError(
                "attach the trainer before the policy's first call: a training episode flies "
                "a fresh policy")
        net = _e3_net(net, "the trainer's net")
        if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real):
            raise TypeError(f"epsilon must be a number in [0, 1], got {epsilon!r}")
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon!r}")
        if not callable(getattr(rng, "random", None)):
            raise TypeError(f"rng is the episode's seeded stream (random.Random), got {rng!r}")
        if not callable(sink):
            raise TypeError(f"sink takes each decision's E3Step, got {sink!r}")
        if not isinstance(around_reference, bool):
            raise TypeError(f"around_reference must be a bool, got {around_reference!r}")
        if around_reference:
            raise ValueError(
                "E3 has no reference to fly around: Chen's recipe explores around its own Q "
                "(a behaviour schedule with reference_episodes=0)")
        self._net = net
        self._trainer = _Trainer(epsilon=float(epsilon), rng=rng, sink=sink)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(band={self._band!r}, net={self._net!r})"
