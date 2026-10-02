"""FeRRy Phase 5 (unit U1): the pair features ``pair_v1``, and the learned score's adapter.

**What the rows are.** At each Pass-1 arrival at stop k the pair slot ranks
the (band, next stop) pairs its view offers (the Phase 5 spec, other choices
1; ``plan.types.PairView``), and the learned score gives one Q per pair from
shared weights (``selector.pair_q``, unit U2): a pointer network over a set of
rows that grows and shrinks with the remainder. :func:`pair_rows` builds that
set: one row per pair of ``view.pairs``, in that order (class-major in link
order, then the remainder's order; home alone), so row r is the slot's row r
and the network's lowest-row tie rule is the slot's. :class:`LearnedPairScorer`
is the slot's ``PairScorer`` over a network, and :func:`build_pair_slot` the
slot a mule flies from a verified checkpoint (resolution R8).

**The columns** (other choices 4: design D-D (b), corrected by critic A3, A10
(iv), C5 and C6). T is T_nom (``view.t_ref_s``, the reward's T, decision 4),
N the view's demand, b̄ the committed class, S the cap, P_c the interference
period. Each column depends on part of the pair only, which the tests pin:
``band`` columns vary with b alone, ``next`` columns with s alone, ``pair``
columns with both, and ``state`` columns are one value for every row of a
view (pooled context, design D-C).

==================  =====  ===============================================================
column              dep.   value
==================  =====  ===============================================================
band[c]             band   1 for the class b the stop is served on (one per link class)
snr_here            band   b's median realized SNR over k's members now, / 30 dB
dwell_here          band   b's dwell at k at that SNR (``arrival``), / T
gain_here           band   b's targets at k beyond b̄'s, / N (>= 0: b covers b̄)
reach_here          state  b̄'s targets at k, / N
travel              next   the leg from k to s (to the dock for home), / T
snr_next[c]         next   s's mean SNR on class c, median over its members, / 30 dB
dwell_next          next   s's predicted dwell on b̄ at the mean SNR, / T
slack_next          pair   sign(x) log1p(abs(x) / T), x = Deadline(s) - (now + dwell_here
                           + travel + dwell_next), unclipped; 0 when exempt_next, and home
exempt_next         next   1 when no deadline clause can bind s (exempt, or undated)
age_next            next   s's mean plan age (``PlanCommit.ages``), / S (/ 1 without a cap)
on_time_next        next   s's mean on-time rate (``features._on_time_rate``)
members_next        next   s's members, / N
capped_next         next   1 when some member of s is capped
home                next   1 for the home row
clock_left          pair   (budget end - (now + dwell_here + travel)) / budget, in [-1, 1];
                           1 without a budget
energy_left         state  1 - energy spent / reference (``l1_state``'s), in [-1, 1];
                           1 without a reference
remainder_share     state  the remainder's members, / N
least_slack         state  the least slack_next over the remainder's dated stops, each
                           priced after serving k on b̄ (0 when none)
weight_share        state  the remainder's committed weight / the demand's (0 when none)
offset[c]           state  the class's offset at k now: median of realized - mean per
                           link (critic C6), / 10 dB
prev_offset[c]      state  the previous Pass-1 arrival's offsets this trial, / 10 dB (0
                           at the trial's first)
has_prev            state  1 once a previous Pass-1 arrival was observed this trial
prev_age            state  its age, capped at 4 P_c, / P_c
prev_sin, prev_cos  state  sin and cos of 2 pi age / P_c (0, 0 at the first)
arrival_sin, _cos   pair   sin and cos of 2 pi (dwell_here + travel) / P_c
==================  =====  ===============================================================

The phase block (``offset`` to ``arrival_cos``) is present when the schema's
``phase`` flag is on, the default (design D-D (b); decision 6 (a)): without it
the score cannot anticipate the channel (design D-D (a)).

What the spec changed from the design's rows: feature 7 is s's mean SNR per
class, since s's band is chosen at s's own arrival; feature 9's slack is
log-scaled and unclipped, since the pilots' slack at collection was at least
582 s with a median of 1,143 s, about 2.9-5.7 T, which the design's [-1, 3]
clip saturated (critic A10 (iv)), with ``exempt_next`` beside it, which the
clip's ceiling stood for; feature 11, the value of s, is dropped (critic C5:
members / N under equal shards, noise under the stub's draws); the offsets
are differenced per link before the median (critic C6, ``FerryRuntime.
class_offsets_db``); the previous reading and its age are added, because one
snapshot of a sinusoid cannot tell a rising phase from a falling one (critic
A3). Three columns go beyond the spec's list, as encoding choices this unit
reports: ``reach_here``, because the reward's G_k counts the targets
collected at k (decision 4), so without the stop's own reach a row could not
carry the immediate gain that a γ > 0 target bootstraps from the next
decision; and ``prev_sin`` and ``prev_cos``, the phase of the previous
reading's age, because predicting the interference at s from two readings
needs the phase between them, which a 2 x 64 tanh network would otherwise
have to learn as a periodic function of the age (the design's risk R6), and
which the capped age no longer carries past 4 P_c.

**Causal.** Every column is a function of the view, which the mule takes at
the arrival instant from what it has observed by then (``FerryRuntime.
arrival_view``, ``observe``, ``class_offsets_db`` at the arrival, the
previous arrival's offsets kept by the mule, means for the rest of the route:
δ_obs = 0, Freeze L829). P_c is the channel's configuration, not its phases.
Nothing here reads the channel, the clock or any draw.

**NEW devices** need no default of their own: the view carries the mule's
(``_on_time_rate``'s 0.5 prior for a device never seen; the plan's age, 1 in
mission 1, or 0 and no weight for a beacon insert the plan never saw), and
every column is finite for them.

**The schema** (:class:`PairFeatureSchema`) is ``pair_v1`` over the link's
class tuple with the phase flag; its JSON form (version, dim, classes, phase,
columns, constants) is the checkpoint header's ``schema``, which U2's loader
compares whole. A checkpoint is read under its own schema, which must be this
module's for the mule's link classes, with the phase block or without as it
was trained (or as the caller asks); one trained on other classes, or under
other columns or scales, is refused (other choices 4). Changing any column or
constant here is a new schema version.

**Layering.** Plan mode only: this module reads the plan's types and imports
numpy and the pair learner; ``selector/__init__`` does not import it, and the
pair slot's module is loaded only inside :func:`build_pair_slot`. It imports
nothing from ``hermes.l1``, ``hermes.mule``, ``hermes.mission`` or
``experiments``. Deterministic: the same view gives the same rows, bit for bit.
"""

from __future__ import annotations

import collections.abc
import contextlib
import copy
import functools
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from hermes.scheduler.plan.types import ArrivalClass, ArrivalView, Pair, PairView, StopContext
from hermes.scheduler.selector.pair_q import (
    KIND_PAIR_Q,
    CheckpointError,
    PairQNet,
    verify_checkpoint,
)
from hermes.types.scheduler import ContactWaypoint

if TYPE_CHECKING:  # pragma: no cover - the slot's module loads in build_pair_slot only
    from hermes.scheduler.policies.pair_slot import PairQSlot

__all__ = [
    "DEPENDS",
    "DEPENDS_BAND",
    "DEPENDS_NEXT",
    "DEPENDS_PAIR",
    "DEPENDS_STATE",
    "OFFSET_SCALE_DB",
    "PAIR_FEATURE_SCHEMA",
    "PREVIOUS_AGE_CAP_PERIODS",
    "SCHEMA_CONSTANTS",
    "SHARE_CLIP",
    "SNR_SCALE_DB",
    "SPARSE_COLUMNS",
    "Column",
    "LearnedPairScorer",
    "PairFeatureSchema",
    "build_pair_slot",
    "covering_classes",
    "load_pair_scorer",
    "pair_rows",
]

#: The feature schema's version: the checkpoint header's ``schema.version``.
PAIR_FEATURE_SCHEMA = "pair_v1"

#: dB per unit of an SNR column: the design's "/30" (D-D), the scale of the
#: legacy selector's RF prior (``selector/features.py``) and of ChannelDDQN's
#: SNR slots (``FerryRuntime.l1_state``), so a mean SNR of 0-30 dB reads 0-1.
SNR_SCALE_DB = 30.0
#: dB per unit of an offset column. An offset is the interference plus the
#: members' median shadowing: in the jittery regime A = 5 dB and sigma_I =
#: 1.5 dB (``CONTACT_REGIMES``) with sigma_sh = 4 dB (``SHADOW_SIGMA_DB``,
#: ``hermes/l1/channel_model.py``), about 5.5 dB spread, and the classes'
#: difference spreads 6.4 dB (the research map, section 3.6), so 10 dB keeps
#: two spreads within about +-1.
OFFSET_SCALE_DB = 10.0
#: The previous reading's age is capped at 4 P_c (the spec, other choices 4).
PREVIOUS_AGE_CAP_PERIODS = 4.0
#: ``clock_left`` and ``energy_left`` are clipped to [-SHARE_CLIP, SHARE_CLIP]:
#: a pair that overruns by more than a whole budget, or a sortie past twice
#: its energy reference, is masked or failing already, and the clip keeps
#: those masked rows' inputs in range.
SHARE_CLIP = 1.0

#: The constants a schema's JSON records, so a checkpoint is refused under any
#: other scale (the loader compares the schema whole).
SCHEMA_CONSTANTS: Mapping[str, float] = MappingProxyType({
    "snr_scale_db": SNR_SCALE_DB,
    "offset_scale_db": OFFSET_SCALE_DB,
    "previous_age_cap_periods": PREVIOUS_AGE_CAP_PERIODS,
    "share_clip": SHARE_CLIP,
})

#: The columns that may be constant across a FerrySim sample, with why: the
#: spec's test holds that no column is constant there except as declared
#: (other choices 4). None is constant by construction on a link of several
#: classes; each of these marks an event the cells make rare, so a sample of
#: one cell or a few episodes can miss it.
SPARSE_COLUMNS: Mapping[str, str] = MappingProxyType({
    "gain_here": "a covering class reaching more targets than b̄ at the arrival SNR: at most "
                 "about 2 % of arrivals in the design's probes (finding 1; critic C8)",
    "exempt_next": "a remainder stop whose every member is capped: only from mission S on, "
                   "where the cap binds a whole stop",
})

#: What a column varies with across the rows of one view: the band b alone,
#: the next stop s alone, both, or neither (one value for every row).
DEPENDS_BAND = "band"
DEPENDS_NEXT = "next"
DEPENDS_PAIR = "pair"
DEPENDS_STATE = "state"
DEPENDS: Tuple[str, ...] = (DEPENDS_BAND, DEPENDS_NEXT, DEPENDS_PAIR, DEPENDS_STATE)

_TWO_PI = 2.0 * math.pi
_SCHEMA_KEYS = ("version", "dim", "classes", "phase", "columns", "constants")
#: A sha256 as ``hashlib`` writes it: the config's rule (``processes/config.py``),
#: which ``pair_q``'s loader applies too; the loader checks it before reading.
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class Column(NamedTuple):
    """One column of a ``pair_v1`` row: its name, what it varies with
    (:data:`DEPENDS`), its bounds (None: unbounded on that side) and whether
    it is a 0/1 flag."""

    name: str
    depends: str
    low: Optional[float] = None
    high: Optional[float] = None
    flag: bool = False


def _flag_column(name: str, depends: str) -> Column:
    return Column(name, depends, 0.0, 1.0, True)


@functools.lru_cache(maxsize=None)
def _column_specs(classes: Tuple[str, ...], phase: bool) -> Tuple[Column, ...]:
    """The columns of ``pair_v1`` over ``classes`` (link order), in row order."""
    clip = SHARE_CLIP
    out: List[Column] = [_flag_column(f"band[{c}]", DEPENDS_BAND) for c in classes]
    out += [
        Column("snr_here", DEPENDS_BAND),
        Column("dwell_here", DEPENDS_BAND, 0.0),
        Column("gain_here", DEPENDS_BAND, 0.0, 1.0),
        Column("reach_here", DEPENDS_STATE, 0.0, 1.0),
        Column("travel", DEPENDS_NEXT, 0.0),
    ]
    out += [Column(f"snr_next[{c}]", DEPENDS_NEXT) for c in classes]
    out += [
        Column("dwell_next", DEPENDS_NEXT, 0.0),
        Column("slack_next", DEPENDS_PAIR),
        _flag_column("exempt_next", DEPENDS_NEXT),
        Column("age_next", DEPENDS_NEXT, 0.0),
        Column("on_time_next", DEPENDS_NEXT, 0.0, 1.0),
        Column("members_next", DEPENDS_NEXT, 0.0, 1.0),
        _flag_column("capped_next", DEPENDS_NEXT),
        _flag_column("home", DEPENDS_NEXT),
        Column("clock_left", DEPENDS_PAIR, -clip, clip),
        Column("energy_left", DEPENDS_STATE, -clip, clip),
        Column("remainder_share", DEPENDS_STATE, 0.0, 1.0),
        Column("least_slack", DEPENDS_STATE),
        Column("weight_share", DEPENDS_STATE, 0.0, 1.0),
    ]
    if phase:
        out += [Column(f"offset[{c}]", DEPENDS_STATE) for c in classes]
        out += [Column(f"prev_offset[{c}]", DEPENDS_STATE) for c in classes]
        out += [
            _flag_column("has_prev", DEPENDS_STATE),
            Column("prev_age", DEPENDS_STATE, 0.0, PREVIOUS_AGE_CAP_PERIODS),
            Column("prev_sin", DEPENDS_STATE, -1.0, 1.0),
            Column("prev_cos", DEPENDS_STATE, -1.0, 1.0),
            Column("arrival_sin", DEPENDS_PAIR, -1.0, 1.0),
            Column("arrival_cos", DEPENDS_PAIR, -1.0, 1.0),
        ]
    return tuple(out)


def _class_names(classes: Any, name: str = "classes") -> Tuple[str, ...]:
    """The link's class tuple, in link order: distinct non-empty names, at least one."""
    if isinstance(classes, (str, bytes)) or not isinstance(classes, collections.abc.Iterable):
        raise TypeError(f"{name} is a sequence of class names, got {classes!r}")
    out = tuple(classes)
    if not out or any(not isinstance(c, str) or not c for c in out):
        raise ValueError(f"{name} holds non-empty class names, at least one: {classes!r}")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} names each class once: {classes!r}")
    return out


@dataclass(frozen=True)
class PairFeatureSchema:
    """``pair_v1`` over a link's classes, with or without the phase block.

    ``classes`` is the link's class tuple in link order (every class of the
    link, not only those covering at an arrival): the one-hot and the
    per-class columns follow it, and a view of another link is refused.
    ``phase`` keeps the phase block (design D-D (b), the default). The
    schema's identity is its JSON form (:meth:`to_json`), which a checkpoint's
    header records and the loader compares whole.
    """

    classes: Tuple[str, ...]
    phase: bool = True
    version: str = PAIR_FEATURE_SCHEMA

    def __post_init__(self) -> None:
        if self.version != PAIR_FEATURE_SCHEMA:
            raise ValueError(
                f"this module builds {PAIR_FEATURE_SCHEMA!r} rows, not {self.version!r}")
        if not isinstance(self.phase, bool):
            raise TypeError(f"phase is a bool (the phase block on or off), got {self.phase!r}")
        object.__setattr__(self, "classes", _class_names(self.classes))

    @classmethod
    def for_view(cls, view: PairView, *, phase: bool = True) -> "PairFeatureSchema":
        """The schema over the classes of ``view``'s link."""
        if not isinstance(view, PairView):
            raise TypeError(f"view must be a PairView, got {view!r}")
        return cls(classes=tuple(entry.name for entry in view.arrival.classes), phase=phase)

    @property
    def column_specs(self) -> Tuple[Column, ...]:
        return _column_specs(self.classes, self.phase)

    @property
    def columns(self) -> Tuple[str, ...]:
        """The column names, in row order."""
        return tuple(column.name for column in self.column_specs)

    @property
    def dim(self) -> int:
        """The row width: the network's ``feature_dim``."""
        return len(self.column_specs)

    def index(self, name: str) -> int:
        """The position of the column ``name`` in a row."""
        try:
            return self.columns.index(name)
        except ValueError:
            raise ValueError(f"{self.version} has no column {name!r}; its columns: "
                             f"{list(self.columns)}") from None

    def to_json(self) -> Dict[str, Any]:
        """The schema as a checkpoint's header records it (U2's ``version`` and
        ``dim``, and the rest of what fixes a row's meaning)."""
        return {
            "version": self.version,
            "dim": self.dim,
            "classes": list(self.classes),
            "phase": self.phase,
            "columns": list(self.columns),
            "constants": dict(SCHEMA_CONSTANTS),
        }

    @classmethod
    def from_json(cls, data: Any) -> "PairFeatureSchema":
        """The schema a checkpoint's ``schema`` records, if it is this module's.

        Refuses (ValueError, TypeError for a non-mapping) another version,
        other keys, and a dim, column list or constants other than this
        module's for its classes and phase flag: a row of another layout or
        scale would be read as this one.
        """
        if not isinstance(data, collections.abc.Mapping):
            raise TypeError(f"a feature schema is a JSON object, got {data!r}")
        if data.get("version") != PAIR_FEATURE_SCHEMA:
            raise ValueError(f"a {data.get('version')!r} schema is not {PAIR_FEATURE_SCHEMA!r}")
        keys, want_keys = set(data), set(_SCHEMA_KEYS)
        if keys != want_keys:
            raise ValueError(
                f"a {PAIR_FEATURE_SCHEMA} schema holds exactly {list(_SCHEMA_KEYS)}: missing "
                f"{sorted(want_keys - keys)}, unknown {sorted(map(str, keys - want_keys))}")
        classes = data["classes"]
        if not isinstance(classes, (list, tuple)):
            raise TypeError(f"the schema's classes are a list, got {classes!r}")
        schema = cls(classes=tuple(classes), phase=data["phase"])
        want = schema.to_json()
        differ = [key for key in ("dim", "columns", "constants")
                  if _plain(data[key]) != _plain(want[key])]
        if differ:
            raise ValueError(
                f"the schema's {differ} are not this module's {PAIR_FEATURE_SCHEMA} for the "
                f"classes {list(schema.classes)} with phase {schema.phase}: its rows would be "
                f"read under another layout or scale")
        return schema


def _plain(value: Any) -> Any:
    """``value`` with plain containers, numbers compared as numbers."""
    if isinstance(value, collections.abc.Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --------------------------------------------------------------------------- #
# The covering classes (FX's candidate set)
# --------------------------------------------------------------------------- #

def covering_classes(view: ArrivalView) -> Tuple[ArrivalClass, ...]:
    """The classes whose targets at the arrival include every target of b̄, in link order.

    The mask's classes (the spec, other choices 2; the user's decision 1 (a)):
    a pair's band reaches every device the committed class reaches at this
    stop at the arrival SNR, a capped one included. It is FX's candidate set
    (``cross_heuristic.fastest_covering_class``), so FX's band is always among
    them, and so is b̄; ``PairView.covering`` is this set, from which the
    view's pairs are formed.
    """
    if not isinstance(view, ArrivalView):
        raise TypeError(f"the covering classes are read from an ArrivalView, got {view!r}")
    need = set(view.committed_entry.targets)
    return tuple(sorted((entry for entry in view.classes if need.issubset(entry.targets)),
                        key=lambda entry: entry.index))


# --------------------------------------------------------------------------- #
# The rows
# --------------------------------------------------------------------------- #

def _slack(seconds: float, t_ref: float) -> float:
    """sign(x) log1p(abs(x) / T): the slack's transform, unclipped (critic A10 (iv))."""
    return math.copysign(math.log1p(abs(seconds) / t_ref), seconds)


def _clipped(value: float) -> float:
    return min(SHARE_CLIP, max(-SHARE_CLIP, value))


def _stop(ctx: StopContext) -> ContactWaypoint:
    """The remainder stop of ``ctx`` (never home: callers test ``is_home`` first)."""
    if ctx.stop is None:
        raise ValueError("home has no stop")
    return ctx.stop


def _free(ctx: StopContext) -> bool:
    """No deadline clause can bind the stop: it is exempt (every member capped,
    ``FLScheduler.plan_protected``) or no member has a date (S3a's ``inf``)."""
    return ctx.exempt or not math.isfinite(_stop(ctx).deadline_ts)


def _slack_s(ctx: StopContext, clock_s: float, ahead_s: float) -> float:
    """Seconds from the end of the service at the stop to its deadline: the stop
    reached ``ahead_s`` after ``clock_s`` and served for its predicted dwell."""
    return _stop(ctx).deadline_ts - (clock_s + ahead_s + ctx.pred_dwell_s)


def _state_values(view: PairView, schema: PairFeatureSchema, committed: ArrivalClass,
                  n: float, t_ref: float) -> Dict[str, float]:
    """The columns every row of ``view`` shares."""
    remainder = [ctx for ctx in view.stops if not ctx.is_home]
    least = min((_slack_s(ctx, view.clock_s, committed.dwell_s + ctx.travel_s)
                 for ctx in remainder if not _free(ctx)), default=None)
    out = {
        "reach_here": len(committed.targets) / n,
        "energy_left": (1.0 if view.energy_ref_j is None
                        else _clipped(1.0 - view.energy_j / view.energy_ref_j)),
        "remainder_share": sum(len(_stop(ctx).devices) for ctx in remainder) / n,
        "least_slack": 0.0 if least is None else _slack(least, t_ref),
        "weight_share": (sum(ctx.weight for ctx in remainder) / view.demand_weight
                         if view.demand_weight > 0.0 else 0.0),
    }
    if schema.phase:
        period = view.period_s
        previous = view.previous_offsets_db
        age = view.previous_age_s
        for i, name in enumerate(schema.classes):
            out[f"offset[{name}]"] = view.offsets_db[i] / OFFSET_SCALE_DB
            out[f"prev_offset[{name}]"] = (0.0 if previous is None
                                           else previous[i] / OFFSET_SCALE_DB)
        if age is None:
            out.update(has_prev=0.0, prev_age=0.0, prev_sin=0.0, prev_cos=0.0)
        else:
            turn = _TWO_PI * age / period
            out.update(has_prev=1.0,
                       prev_age=min(age, PREVIOUS_AGE_CAP_PERIODS * period) / period,
                       prev_sin=math.sin(turn), prev_cos=math.cos(turn))
    return out


def _next_values(ctx: StopContext, schema: PairFeatureSchema, n: float, t_ref: float,
                 cap: float) -> Dict[str, float]:
    """The columns of the candidate ``ctx`` that do not depend on the band."""
    out = {"travel": ctx.travel_s / t_ref}
    if ctx.is_home:
        out.update({f"snr_next[{name}]": 0.0 for name in schema.classes})
        out.update(dwell_next=0.0, exempt_next=0.0, age_next=0.0, on_time_next=0.0,
                   members_next=0.0, capped_next=0.0, home=1.0)
        return out
    out.update({f"snr_next[{name}]": snr / SNR_SCALE_DB
                for name, snr in zip(schema.classes, ctx.pred_snr_db)})
    out.update(
        dwell_next=ctx.pred_dwell_s / t_ref,
        exempt_next=1.0 if _free(ctx) else 0.0,
        age_next=ctx.age / cap,
        on_time_next=ctx.on_time,
        members_next=len(_stop(ctx).devices) / n,
        capped_next=1.0 if ctx.capped else 0.0,
        home=0.0,
    )
    return out


def pair_rows(view: PairView, schema: PairFeatureSchema
              ) -> Tuple[np.ndarray, Tuple[Pair, ...]]:
    """The ``pair_v1`` rows of ``view``: a (K, ``schema.dim``) float64 array and its pairs.

    One row per pair of ``view.pairs``, in that order, which is returned with
    the rows. ``schema.classes`` must be the view's link classes, in link
    order. Pure: the rows depend on the view alone, not on the mask, and the
    same view gives the same rows bit for bit.
    """
    if not isinstance(view, PairView):
        raise TypeError(f"the pair features read a PairView, got {view!r}")
    if not isinstance(schema, PairFeatureSchema):
        raise TypeError(f"schema must be a PairFeatureSchema, got {schema!r}")
    names = tuple(entry.name for entry in view.arrival.classes)
    if names != schema.classes:
        raise ValueError(
            f"the view's link has the classes {list(names)}, the schema {list(schema.classes)}: "
            f"a score trained on one link does not read another's rows")
    pairs = view.pairs
    n = float(view.demand)
    t_ref = view.t_ref_s
    cap = float(view.cap_s) if view.cap_s is not None else 1.0
    period = view.period_s
    committed = view.arrival.committed_entry
    entries = {entry.name: entry for entry in view.arrival.classes}
    observed = dict(zip(names, view.observed_snr_db))
    state = _state_values(view, schema, committed, n, t_ref)
    nexts = {ctx.index: _next_values(ctx, schema, n, t_ref, cap) for ctx in view.stops}
    contexts = {ctx.index: ctx for ctx in view.stops}
    bands: Dict[str, Dict[str, float]] = {}
    for entry in view.covering:
        values = {f"band[{name}]": 1.0 if name == entry.name else 0.0 for name in names}
        values.update(snr_here=observed[entry.name] / SNR_SCALE_DB,
                      dwell_here=entry.dwell_s / t_ref,
                      gain_here=(len(entry.targets) - len(committed.targets)) / n)
        bands[entry.name] = values
    columns = schema.columns
    rows = []
    for band, index in pairs:
        ctx = contexts[index]
        ahead = entries[band].dwell_s + ctx.travel_s
        if ctx.is_home or _free(ctx):
            slack = 0.0
        else:
            slack = _slack(_slack_s(ctx, view.clock_s, ahead), t_ref)
        if view.budget_end is None:
            clock_left = 1.0
        else:
            budget = view.budget_s if view.budget_s is not None else t_ref
            clock_left = _clipped((view.budget_end - (view.clock_s + ahead)) / budget)
        values = dict(state)
        values.update(bands[band])
        values.update(nexts[index])
        values.update(slack_next=slack, clock_left=clock_left)
        if schema.phase:
            turn = _TWO_PI * ahead / period
            values.update(arrival_sin=math.sin(turn), arrival_cos=math.cos(turn))
        rows.append([values[name] for name in columns])
    out = np.asarray(rows, dtype=np.float64).reshape(len(pairs), len(columns)) + 0.0
    if not np.isfinite(out).all():
        bad = sorted({columns[j] for j in np.argwhere(~np.isfinite(out))[:, 1]})
        raise ValueError(f"the pair features of this view are not finite in {bad}")
    return out, pairs


# --------------------------------------------------------------------------- #
# The learned score
# --------------------------------------------------------------------------- #

class LearnedPairScorer:
    """The learned score as the pair slot's ``PairScorer``: Q of each pair's row.

    ``score(view, mask=...)`` is ``net.q(pair_rows(view, schema))``, one
    number per pair of ``view.pairs`` in that order, the whole candidate set
    in one call so equal rows get one Q and the slot's lowest-row rule
    decides between them (U2's hand-off). The mask is not read: the slot
    picks among the admitted pairs. ``q_values`` is True, so the slot records
    the numbers as Q values, and ``name`` is the schema's version, which the
    slot's records carry as their ``scorer``. ``net`` is held, not copied: a
    trainer updating it is scored by its current weights (FerrySim, unit
    U8b). ``manifest`` is the verified checkpoint's (None for a network built
    in memory), from which a mule announces its provenance.
    """

    q_values: bool = True

    def __init__(self, net: Any, schema: PairFeatureSchema, *,
                 manifest: Optional[Mapping[str, Any]] = None) -> None:
        if not isinstance(schema, PairFeatureSchema):
            raise TypeError(f"schema must be a PairFeatureSchema, got {schema!r}")
        if not callable(getattr(net, "q", None)):
            raise TypeError(f"net scores rows with q(rows) (a PairQNet), got {net!r}")
        width = getattr(net, "feature_dim", None)
        if width != schema.dim:
            raise ValueError(
                f"the network reads rows {width!r} wide and the schema's are {schema.dim}: "
                f"they are not one schema's")
        if manifest is not None and not isinstance(manifest, collections.abc.Mapping):
            raise TypeError(f"manifest is the checkpoint's (a mapping) or None, got {manifest!r}")
        self._net = net
        self._schema = schema
        self._manifest = None if manifest is None else copy.deepcopy(dict(manifest))
        self.name: str = schema.version

    @property
    def net(self) -> Any:
        return self._net

    @property
    def schema(self) -> PairFeatureSchema:
        return self._schema

    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        """A copy of the verified checkpoint's manifest, or None."""
        return None if self._manifest is None else copy.deepcopy(self._manifest)

    def score(self, view: PairView, *, mask: Tuple[bool, ...]) -> Tuple[float, ...]:
        """The online network's Q of each pair of ``view.pairs``; ``mask`` is not read."""
        rows, pairs = pair_rows(view, self._schema)
        q = np.asarray(self._net.q(rows), dtype=np.float64)
        if q.shape != (len(pairs),):
            raise ValueError(f"the network gave {q.shape} numbers for {len(pairs)} pairs")
        return tuple(float(value) for value in q)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(schema={self._schema.version!r}, "
                f"classes={list(self._schema.classes)}, phase={self._schema.phase})")


@contextlib.contextmanager
def _refused_if_missing(path: Any) -> Iterator[None]:
    """``pair_q``'s reader raises FileNotFoundError for a path with no file
    behind it (a directory included); to a mule that is one more checkpoint
    it may not fly, so it is refused like the rest."""
    try:
        yield
    except FileNotFoundError as exc:
        raise CheckpointError(f"{exc}: there is no checkpoint to fly") from None


def load_pair_scorer(path: Any, *, expect_sha256: str, classes: Sequence[str],
                     phase: Optional[bool] = None) -> LearnedPairScorer:
    """The learned score of the ``pair_q`` checkpoint at ``path``, verified.

    ``expect_sha256`` is the config's (``pair_checkpoint_sha256``) and
    ``classes`` the mule's link classes in link order. The checkpoint is
    verified whole first (``pair_q.verify_checkpoint``: format, header,
    arrays against the manifest), then must be a ``pair_q`` checkpoint whose
    schema is this module's ``pair_v1`` (:meth:`PairFeatureSchema.from_json`)
    over exactly ``classes``, with the phase flag ``phase`` when one is asked
    (None takes the checkpoint's own); U2's loader then checks the sha
    against ``expect_sha256`` and the header's schema against this schema
    whole. No training state is checked (the runner's refusals,
    ``pair_q.campaign_refusals``), so FerrySim's bootstrap checkpoints load.

    Two kinds of error, so a mule can tell a refused checkpoint from a bug in
    its caller. Every refusal of what ``path`` names raises
    ``pair_q.CheckpointError`` (a ValueError): a path that is not an
    ``.npz``, no file there, and everything the checks above refuse, so that
    a config which passed ``mule_config_errors`` meets no other error from
    its checkpoint, and the mule refuses to run on it: it never flies a
    checkpoint whose rows it would read otherwise than it was trained on.
    Arguments that no valid config or link holds raise first, before any file
    is read: TypeError for ``classes``, ``phase`` or ``path`` of another
    type, ValueError for an empty or repeated class tuple or an
    ``expect_sha256`` that is not 64 lowercase hex digits (the config's rule).
    """
    names = _class_names(classes)
    if phase is not None and not isinstance(phase, bool):
        raise TypeError(f"phase is a bool or None (the checkpoint's own), got {phase!r}")
    if not isinstance(expect_sha256, str) or not _SHA256_HEX.fullmatch(expect_sha256):
        raise ValueError(f"expect_sha256 is the config's sha256, 64 lowercase hex digits as "
                         f"hashlib writes them, got {expect_sha256!r}")
    if Path(path).suffix != ".npz":
        raise CheckpointError(f"{path}: not a pair checkpoint, which is a format-2 .npz file")
    with _refused_if_missing(path):
        manifest = verify_checkpoint(path)
    if manifest["kind"] != KIND_PAIR_Q:
        raise CheckpointError(f"{path}: a {manifest['kind']!r} checkpoint, not {KIND_PAIR_Q!r}")
    try:
        schema = PairFeatureSchema.from_json(manifest["schema"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"{path}: its feature schema is not this module's: {exc}") from None
    if schema.classes != names:
        raise CheckpointError(
            f"{path}: trained on the classes {list(schema.classes)}, and the link's are "
            f"{list(names)}")
    if phase is not None and schema.phase is not phase:
        raise CheckpointError(
            f"{path}: its schema's phase block is {'on' if schema.phase else 'off'}, and "
            f"{'on' if phase else 'off'} was asked")
    with _refused_if_missing(path):  # the file can go between the two reads
        net, verified = PairQNet.load(path, expect_sha256=expect_sha256,
                                      expect_kind=KIND_PAIR_Q, expect_schema=schema.to_json(),
                                      expect_classes=list(names))
    return LearnedPairScorer(net, schema, manifest=verified)


def build_pair_slot(path: Any, *, expect_sha256: str, classes: Sequence[str],
                    phase: Optional[bool] = None) -> "PairQSlot":
    """The pair slot (``policies.pair_slot.PairQSlot``) a ``pair_q`` mule flies.

    Resolution R8: the slot around :func:`load_pair_scorer`'s learned score,
    so it flies only from a verified checkpoint, never a random-initialised
    network (other choices 5). It raises as :func:`load_pair_scorer` does: a
    ``CheckpointError`` is a checkpoint the mule refuses to run on, anything
    else a bug in its caller. The slot carries no provenance; its scorer
    does (``slot.scorer.manifest``, for ``mule_ready.pair``). The slot's
    module is imported here, on the ``pair_q`` path only.
    """
    from hermes.scheduler.policies.pair_slot import PairQSlot

    return PairQSlot(load_pair_scorer(path, expect_sha256=expect_sha256, classes=classes,
                                      phase=phase))
