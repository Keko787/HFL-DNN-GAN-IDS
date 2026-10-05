"""L3 merge rules — the registry behind ``ClusterConfig.aggregation``.

FeRRy build plan, Phase 1. Every rule answers two questions: how the mule merges
the updates it collected on one mission into a partial, and how the cluster
folds partials into the global θ.

``agg:plain`` is the merge every recorded run used: the num_examples-weighted
mean of full models at the mule, the same mean over partials at the cluster, and
the result replaces the global θ. It stays on its original code path
(:func:`~hermes.mission.partial_fedavg.partial_fedavg` and
:func:`~hermes.cluster.cross_mule_fedavg.cross_mule_fedavg`), so choosing it
changes nothing.

The other rules are age-aware and work in delta form. A device answers with
Δθ_i = θ_i − θ_{b_i}, its model minus the basis it trained from, and names the
version b_i of that basis (versions are cluster rounds). With v the version of
the θ the mule carried, the update's age is a_i = v − b_i. The mule forms::

    Δ_m = Σ_i w_i·Δθ_i / M_m,    w_i = n_i·v_i·s(a_i),    M_m = Σ_i n_i·v_i

with both sums over the updates the rule admits: w_i is exactly 0 past a
device's cutoff, and an excluded update adds nothing to M_m either. M_m is
staleness-free, so staleness shrinks the step instead of being normalised
away — two updates both of age 3 move θ by s(3) times their mean, not by the
full mean. The cluster, holding θ at version V, applies::

    θ ← θ + η · Σ_m M_m·s_m·Δ_m / Σ_{m live} M_m,    s_m = s(V − v_m)

where a partial is live when s_m > 0 and it is not empty. A single partial
gives θ + η·s_m·Δ_m, the FedAsync / Async-HFL mixing form. When no partial is
live (all past ``a_max`` at the cluster) the cluster takes no step and keeps
the round open; the outcome is ``expired``.

With every basis current (all a_i = 0), v_i = 1 and η = 1 this is exactly the
plain mean: θ_v + Σ (n_i/Σn)(θ_i − θ_v) = Σ (n_i/Σn)·θ_i. With every s_m = 1
the two levels compose into one flat merge over all the mules' devices, since
M_m·Δ_m = Σ_i w_i·Δθ_i.

* ``agg:cutoff`` — FeRRy's rule. s is FedAsync's hinge (Xie et al., 2019):
  1 up to age ``hinge_b``, then 1/(``hinge_a``·(a − ``hinge_b``) + 1). The
  weight is exactly zero past the device's cutoff a_max_j (after Yang et al.,
  JSAC 2025). The cutoff comes from the device's own deadline window,
  a_max_j = ⌊Φ_j / T⌋ with T the mission period (build-plan decision D5), or
  from a fixed ``a_max``, whichever is smaller. This is the deadline's third
  role in contribution C3.
* ``agg:asynchfl`` — exponential staleness decay exp(−λ·a) at the mule and
  again at the cluster, after Async-HFL (Yu et al., IoTDI 2023). Pair it with
  the devices' proximal term (``DeviceConfig.fedprox_rho``).
* ``agg:fedbuff`` — FedBuff (Nguyen et al., AISTATS 2022). The cluster buffers
  updates and, once K have arrived, applies their mean, each scaled by
  1/√(1 + a). Unweighted by n, as in the paper: at the mule w_i = s(a_i) and
  M_m is the update count. K is ``buffer_k`` when set, else the slice size of
  the mule whose partial opens the buffer. K is FedBuff's own quorum, so the
  cluster's ``min_participation`` does not gate it.

* ``agg:fedex`` — FedEx-Async's server step (Bian, Shen, Chen, Xu, IEEE TMC
  24(6), 2025), the merge of arm D4. When a transporter returns, the server
  applies x ← x − (1/N)·u_k with u_k = Σ_i m_i the cumulative local updates it
  carries and N the total number of clients; no server rate, no staleness
  weighting, no cutoff. In delta form (Δθ_i = −m_i) the mule sends the SUM
  Δ_m = Σ_i Δθ_i (every weight 1, normaliser 1, M_m the update count) and the
  cluster applies θ ← θ + η·Σ_m Δ_m / N over the pending partials, N =
  ``fedex_n`` or the registered devices. Faithful at η = 1 and
  ``min_participation`` = 1 (each return is its own step). Port note: a
  FedEx client's m_i accumulates every local step since its last visit; ours
  is the update against the basis it trained from, which is the same thing
  when the device trains once between visits (principle 14).

Refused by name until it exists: ``agg:seq`` (carries the model device to
device, so each device would train during its contact — a protocol change,
since principle 14 keeps sessions exchange-only).

For one mule the two-level form is exact: the mule's θ is always the cluster's
current θ, so V − v_m = 0. With several mules it is the usual hierarchical
reading — device staleness at the mule, partial staleness at the cluster.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from hermes.types import (
    DeviceID,
    GradientSubmission,
    MuleID,
    PartialAggregate,
    Weights,
)
from hermes.types.fl_messages import UPDATE_FORM_DELTA, UPDATE_FORM_WEIGHTS

from .partial_fedavg import PartialFedAvgError, partial_fedavg, partial_fedavg_delta

AGG_PLAIN = "agg:plain"
AGG_CUTOFF = "agg:cutoff"
AGG_ASYNCHFL = "agg:asynchfl"
AGG_FEDBUFF = "agg:fedbuff"
AGG_FEDEX = "agg:fedex"
AGG_SEQ = "agg:seq"

#: Rules this module can run.
IMPLEMENTED_RULES = (AGG_PLAIN, AGG_CUTOFF, AGG_ASYNCHFL, AGG_FEDBUFF, AGG_FEDEX)

#: Rules in the build plan that are not built yet, with the reason.
PLANNED_RULES: Dict[str, str] = {
    AGG_SEQ: (
        "carrying the model device to device needs in-session training, a "
        "protocol change (principle 14 keeps sessions exchange-only)"
    ),
}

VALUE_UNIFORM = "uniform"
VALUE_LOSS = "loss"
VALUE_PROXIES = (VALUE_UNIFORM, VALUE_LOSS)


class AggregationConfigError(ValueError):
    """Raised for an unknown rule or an invalid rule parameter."""


@dataclass(frozen=True)
class AggregationSpec:
    """One L3 merge rule and its parameters.

    Built from ``ClusterConfig``/``MuleConfig`` (``aggregation`` plus
    ``aggregation_params``); the driver sets both from one flag so the mule and
    the cluster always run the same rule.
    """

    rule: str = AGG_PLAIN
    #: Server rate η: the cluster mixes the merged update in at this rate.
    server_lr: float = 1.0
    #: Fixed age cutoff in cluster rounds (``agg:cutoff``); None = none.
    a_max: Optional[int] = None
    #: Mission period T in seconds. When set, each device's cutoff is also
    #: ⌊Φ_j / T⌋ with Φ_j its deadline window (decision D5).
    period_s: Optional[float] = None
    #: FedAsync hinge: s(a) = 1 for a ≤ b, else 1/(a_coef·(a − b) + 1).
    hinge_a: float = 1.0
    hinge_b: float = 0.0
    #: ``agg:asynchfl`` decay rate λ in s(a) = exp(−λ·a).
    decay: float = 0.5
    #: v_i: ``uniform`` (1) or ``loss`` (the update's raw local loss, taken
    #: after the cutoff; a missing or non-positive loss counts as the mean of
    #: the admitted updates' known losses, 1.0 if none). The mule divides by
    #: M_m = Σ n_i·v_i, so a loss level shared by one mule's updates cancels
    #: there, while across mules M_m = Σ n_i·loss_i weighs the partials and
    #: they combine exactly as one flat merge.
    value: str = VALUE_UNIFORM
    #: ``agg:fedbuff`` buffer size K; None = the slice size of the mule whose
    #: partial opens the buffer (all registered devices if that slice is empty).
    buffer_k: Optional[int] = None
    #: ``agg:fedex`` N, the total number of clients in x ← x + (1/N)·Σ Δθ;
    #: None = the devices registered at the cluster.
    fedex_n: Optional[int] = None

    def __post_init__(self) -> None:
        self.validate()

    # --------------------------------------------------------------- queries

    @property
    def is_plain(self) -> bool:
        return self.rule == AGG_PLAIN

    @property
    def update_form(self) -> str:
        """The form a device must answer in under this rule."""
        return UPDATE_FORM_WEIGHTS if self.is_plain else UPDATE_FORM_DELTA

    @property
    def uses_age_cap(self) -> bool:
        return self.rule == AGG_CUTOFF and (
            self.a_max is not None or self.period_s is not None
        )

    # ------------------------------------------------------------ validation

    def validate(self) -> None:
        if self.rule in PLANNED_RULES:
            raise AggregationConfigError(
                f"{self.rule} is not implemented yet: {PLANNED_RULES[self.rule]}"
            )
        if self.rule not in IMPLEMENTED_RULES:
            raise AggregationConfigError(
                f"unknown aggregation rule {self.rule!r}; "
                f"choose one of {IMPLEMENTED_RULES}"
            )
        if not self.server_lr > 0.0:
            raise AggregationConfigError(f"server_lr must be > 0, got {self.server_lr}")
        if self.a_max is not None and self.a_max < 0:
            raise AggregationConfigError(f"a_max must be >= 0, got {self.a_max}")
        if self.period_s is not None and not self.period_s > 0.0:
            raise AggregationConfigError(f"period_s must be > 0, got {self.period_s}")
        if self.hinge_a < 0.0 or self.hinge_b < 0.0:
            raise AggregationConfigError(
                f"hinge parameters must be >= 0, got a={self.hinge_a} b={self.hinge_b}"
            )
        if self.decay < 0.0:
            raise AggregationConfigError(f"decay must be >= 0, got {self.decay}")
        if self.value not in VALUE_PROXIES:
            raise AggregationConfigError(
                f"value must be one of {VALUE_PROXIES}, got {self.value!r}"
            )
        if self.fedex_n is not None and self.fedex_n < 1:
            raise AggregationConfigError(f"fedex_n must be >= 1, got {self.fedex_n}")
        if self.buffer_k is not None and self.buffer_k < 1:
            raise AggregationConfigError(f"buffer_k must be >= 1, got {self.buffer_k}")

    # ------------------------------------------------------------- config I/O

    def to_params(self) -> dict:
        """Everything but the rule name, for ``aggregation_params``."""
        raw = asdict(self)
        raw.pop("rule")
        return raw

    @classmethod
    def from_config(
        cls, rule: Optional[str], params: Optional[Mapping] = None
    ) -> "AggregationSpec":
        kwargs = dict(params or {})
        unknown = set(kwargs) - set(cls.__dataclass_fields__) - {"rule"}
        if unknown:
            raise AggregationConfigError(
                f"unknown aggregation parameter(s): {sorted(unknown)}"
            )
        kwargs.pop("rule", None)
        return cls(rule=rule or AGG_PLAIN, **kwargs)


# --------------------------------------------------------------------------- #
# Staleness and weights
# --------------------------------------------------------------------------- #

def staleness(spec: AggregationSpec, age: int) -> float:
    """s(age) for ``spec.rule``. Ages below 0 count as 0."""
    a = max(0, int(age))
    if spec.rule == AGG_CUTOFF:
        if a <= spec.hinge_b:
            return 1.0
        return 1.0 / (spec.hinge_a * (a - spec.hinge_b) + 1.0)
    if spec.rule == AGG_ASYNCHFL:
        return math.exp(-spec.decay * a)
    if spec.rule == AGG_FEDBUFF:
        return 1.0 / math.sqrt(1.0 + a)
    return 1.0


def age_cap(spec: AggregationSpec, window_s: Optional[float] = None) -> Optional[int]:
    """Device cutoff a_max_j in cluster rounds, or None for no cutoff.

    Decision D5: a device's deadline window Φ_j converts to rounds with the
    mission period T, a_max_j = ⌊Φ_j / T⌋; a fixed ``a_max`` also applies, and
    the smaller cap wins. Only ``agg:cutoff`` has a cutoff.
    """
    if spec.rule != AGG_CUTOFF:
        return None
    caps: List[int] = []
    if spec.a_max is not None:
        caps.append(int(spec.a_max))
    # An infinite window (unit U11's Oort-style ``pref`` law) cuts nothing.
    if spec.period_s is not None and window_s is not None and math.isfinite(float(window_s)):
        caps.append(max(0, int(math.floor(float(window_s) / spec.period_s))))
    return min(caps) if caps else None


def update_age(base_version: Optional[int], basis_version: Optional[int]) -> Optional[int]:
    """Age of an update in cluster rounds; None when either version is unknown.

    A basis newer than the mule's θ (possible when another mule delivered a
    later θ) counts as age 0.
    """
    if base_version is None or basis_version is None:
        return None
    return max(0, int(base_version) - int(basis_version))


def _known_loss(sub: GradientSubmission) -> Optional[float]:
    """The submission's local loss when it is usable as v_i, else None."""
    loss = sub.local_loss
    if loss is None or not math.isfinite(loss) or not loss > 0:
        return None
    return float(loss)


def _value_proxies(
    spec: AggregationSpec,
    subs: Sequence[GradientSubmission],
    admitted: Sequence[bool],
) -> List[float]:
    """v_i per submission: 1, or under ``loss`` the raw local loss.

    The raw loss, not the loss over a mean: the mule divides by
    M_m = Σ n_i·v_i, so any per-mule rescaling cancels there anyway, and the
    raw scale is what lets partials from different mules weigh against each
    other. The fallback for a missing loss is the mean over the ADMITTED
    updates only, so an update past its cutoff cannot shift anyone's weight.
    """
    if spec.value != VALUE_LOSS:
        return [1.0] * len(subs)
    losses = [_known_loss(s) for s in subs]
    known = [loss for loss, ok in zip(losses, admitted) if ok and loss is not None]
    fallback = sum(known) / len(known) if known else 1.0
    return [fallback if loss is None else loss for loss in losses]


@dataclass(frozen=True)
class UpdateWeight:
    """One update's age, cutoff, staleness-free mass and raw merge weight.

    ``mass`` is n_i·v_i (1.0 under ``agg:fedbuff``): what the update adds to
    the mule's normaliser M_m. ``weight`` is mass·s(a_i), exactly 0.0 past the
    cutoff. Keeping them apart is what lets staleness shrink the step.
    """

    device_id: DeviceID
    basis_version: Optional[int]
    age: Optional[int]
    cap: Optional[int]
    weight: float
    mass: float

    @property
    def admitted(self) -> bool:
        return self.weight > 0.0


def update_weights(
    spec: AggregationSpec,
    submissions: Sequence[GradientSubmission],
    *,
    base_version: Optional[int],
    age_caps: Optional[Mapping[DeviceID, Optional[int]]] = None,
) -> List[UpdateWeight]:
    """Mass n_i·v_i and raw weight w_i = mass·s(a_i) per submission.

    The weight is exactly 0.0 past the device's cutoff. ``agg:fedbuff`` has
    mass 1 and weight s(a_i) alone (FedBuff does not weight by n). An unknown
    age counts as 0. Admission is settled before the value proxies are formed,
    so an excluded update never moves an admitted one's weight.
    """
    caps = age_caps or {}
    rows = []
    for sub in submissions:
        age = update_age(base_version, sub.basis_version)
        a = 0 if age is None else age
        cap = caps.get(sub.device_id) if spec.rule == AGG_CUTOFF else None
        s = 0.0 if (cap is not None and a > cap) else staleness(spec, a)
        rows.append((sub, age, cap, s))
    values = _value_proxies(
        spec, submissions, [s > 0.0 and sub.num_examples > 0 for sub, _, _, s in rows],
    )
    out: List[UpdateWeight] = []
    for (sub, age, cap, s), v in zip(rows, values):
        # FedBuff and FedEx weigh every update alike (neither weights by n).
        mass = 1.0 if spec.rule in (AGG_FEDBUFF, AGG_FEDEX) else float(sub.num_examples) * v
        out.append(UpdateWeight(sub.device_id, sub.basis_version, age, cap, mass * s, mass))
    return out


# --------------------------------------------------------------------------- #
# Mule-side merge
# --------------------------------------------------------------------------- #

def merge_on_mule(
    spec: AggregationSpec,
    *,
    mule_id: MuleID,
    mission_round: int,
    submissions: Sequence[GradientSubmission],
    base_version: Optional[int] = None,
    age_caps: Optional[Mapping[DeviceID, Optional[int]]] = None,
) -> PartialAggregate:
    """Merge one mission's verified submissions into a partial.

    ``agg:plain`` runs :func:`partial_fedavg` unchanged and only annotates the
    result with versions and ages. The age-aware rules merge in delta form,
    Δ_m = Σ w_i·Δθ_i / M_m over the admitted updates; the partial carries M_m as
    ``weight_mass`` and w_i / M_m as ``device_weights``, which sum to the
    mass-weighted mean staleness (≤ 1, and 1 when no update is discounted).
    Raises :class:`PartialFedAvgError` when nothing carries weight, which the
    mule treats like a mission that collected nothing.
    """
    if spec.is_plain:
        agg = partial_fedavg(mule_id, mission_round, submissions)
        by_id = {s.device_id: s for s in submissions}
        versions = tuple(
            by_id[d].basis_version if d in by_id else None
            for d in agg.contributing_devices
        )
        return replace(
            agg,
            base_version=base_version,
            n_updates=len(agg.contributing_devices),
            device_basis_versions=versions,
            device_ages=tuple(update_age(base_version, v) for v in versions),
        )

    effective = [s for s in submissions if s.num_examples > 0]
    if not effective:
        raise PartialFedAvgError(
            "every submission had num_examples=0; nothing to aggregate"
        )
    weights = update_weights(
        spec, effective, base_version=base_version, age_caps=age_caps,
    )
    admitted = [(s, w) for s, w in zip(effective, weights) if w.admitted]
    excluded = tuple(w.device_id for w in weights if not w.admitted)
    if not admitted:
        raise PartialFedAvgError(
            f"no update carries weight under {spec.rule}: all "
            f"{len(effective)} were past their age cutoff"
        )
    raw = [w.weight for _, w in admitted]
    # Divide by the staleness-free mass M_m of the admitted updates, not by
    # Σ w_i: dividing by the weights' own sum would cancel a common staleness
    # factor and send a uniformly stale mission's full mean. Under FedBuff
    # every mass is 1, so M_m is the COUNT of updates, as in the paper.
    # Excluded updates add nothing, so they do not dilute the step.
    mass = float(sum(w.mass for _, w in admitted))
    # FedEx carries the SUM u_k of its clients' updates home; the cluster
    # divides by the total client count N, not by this mule's count.
    normalizer = 1.0 if spec.rule == AGG_FEDEX else mass
    agg = partial_fedavg_delta(
        mule_id,
        mission_round,
        [s for s, _ in admitted],
        weights=raw,
        normalizer=normalizer,
    )
    return replace(
        agg,
        rule=spec.rule,
        base_version=base_version,
        weight_mass=mass,
        device_basis_versions=tuple(w.basis_version for _, w in admitted),
        device_ages=tuple(w.age for _, w in admitted),
        device_weights=tuple(r / mass for r in raw),
        excluded_devices=excluded,
    )


# --------------------------------------------------------------------------- #
# Cluster-side fold
# --------------------------------------------------------------------------- #

def partial_age(partial: PartialAggregate, cluster_version: int) -> int:
    """Cluster rounds between the θ the mule carried and the cluster's θ."""
    if partial.base_version is None:
        return 0
    return max(0, int(cluster_version) - int(partial.base_version))


def partial_staleness(
    spec: AggregationSpec, partial: PartialAggregate, cluster_version: int
) -> float:
    """s(V − v_m) for one partial; zero past a fixed ``a_max`` under cutoff."""
    age = partial_age(partial, cluster_version)
    if spec.rule == AGG_CUTOFF and spec.a_max is not None and age > spec.a_max:
        return 0.0
    return staleness(spec, age)


def check_partial_form(spec: AggregationSpec, partial: PartialAggregate) -> None:
    """Refuse a partial built by a different rule than the cluster runs."""
    if partial.update_form != spec.update_form or (
        not spec.is_plain and partial.rule != spec.rule
    ):
        raise AggregationConfigError(
            f"partial from mule {partial.mule_id!r} was built by "
            f"{partial.rule} ({partial.update_form}); the cluster runs "
            f"{spec.rule} ({spec.update_form}). Set the same rule on both."
        )


@dataclass
class FedBuffBuffer:
    """Cluster-side FedBuff buffer: sum of staleness-scaled updates and a count.

    ``members`` names the partials behind the buffered sum as
    (mule_id, mission_round), so a trace can say which missions a deferred
    buffer holds and which ones a flush applied.
    """

    spec: AggregationSpec
    k: int
    total: Optional[List] = None
    count: int = 0
    members: List[Tuple[MuleID, int]] = field(default_factory=list)

    def add(self, partial: PartialAggregate, *, cluster_version: int) -> None:
        import numpy as np

        check_partial_form(self.spec, partial)
        if partial.is_empty() or partial.weight_mass <= 0.0:
            return
        scale = partial_staleness(self.spec, partial, cluster_version) * partial.weight_mass
        layers = [np.asarray(w, dtype=np.float64) * scale for w in partial.weights]
        if self.total is None:
            self.total = layers
        else:
            if len(layers) != len(self.total):
                raise AggregationConfigError("FedBuff: layer-count mismatch across partials")
            for acc, layer in zip(self.total, layers):
                if acc.shape != layer.shape:
                    raise AggregationConfigError("FedBuff: layer shape mismatch across partials")
                acc += layer
        self.count += int(round(partial.weight_mass))
        self.members.append((partial.mule_id, int(partial.mission_round)))

    @property
    def ready(self) -> bool:
        return self.total is not None and self.count >= self.k

    def apply(self, theta: Weights) -> Weights:
        """θ + η · (buffered sum / count); empties the buffer and its members."""
        import numpy as np

        if self.total is None or self.count <= 0:
            raise AggregationConfigError("FedBuff: apply() with an empty buffer")
        if len(theta) != len(self.total):
            raise AggregationConfigError("FedBuff: buffer and θ differ in layer count")
        out = [
            (np.asarray(t, dtype=np.float64) + self.spec.server_lr * acc / self.count).astype(t.dtype)
            for t, acc in zip(theta, self.total)
        ]
        self.total = None
        self.count = 0
        self.members = []
        return out
