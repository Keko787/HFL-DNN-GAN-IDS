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

    Δ_m = Σ_i w_i·Δθ_i / Σ_i w_i,        w_i = n_i · v_i · s(a_i)

and the cluster, holding θ at version V, applies::

    θ ← θ + η · Σ_m W_m·Δ_m / Σ_m W_m,   W_m = (Σ_i w_i)_m · s(V − v_m)

With every basis current (all a_i = 0), v_i = 1 and η = 1 this is exactly the
plain mean: θ_v + Σ (n_i/Σn)(θ_i − θ_v) = Σ (n_i/Σn)·θ_i.

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
  updates and, once K have arrived (K = slice size unless set), applies their
  mean, each scaled by 1/√(1 + a). Unweighted by n, as in the paper.

Refused by name until they exist: ``agg:fedex`` (FedEx-Async's 1/N merge,
which arrives with arm D4 in Phase 2) and ``agg:seq`` (carries the model device
to device, so each device would train during its contact — a protocol change,
since principle 14 keeps sessions exchange-only).

For one mule the two-level form is exact: the mule's θ is always the cluster's
current θ, so V − v_m = 0. With several mules it is the usual hierarchical
reading — device staleness at the mule, partial staleness at the cluster.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence

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
IMPLEMENTED_RULES = (AGG_PLAIN, AGG_CUTOFF, AGG_ASYNCHFL, AGG_FEDBUFF)

#: Rules in the build plan that are not built yet, with the reason.
PLANNED_RULES: Dict[str, str] = {
    AGG_FEDEX: "FedEx-Async's merge arrives with arm D4 in FeRRy Phase 2",
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
    #: v_i: ``uniform`` (1) or ``loss`` (the update's loss over the mean loss
    #: of the updates being merged, so it re-weights without rescaling).
    value: str = VALUE_UNIFORM
    #: ``agg:fedbuff`` buffer size K; None = the uploading mule's slice size.
    buffer_k: Optional[int] = None

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
    if spec.period_s is not None and window_s is not None:
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


def _value_proxies(spec: AggregationSpec, subs: Sequence[GradientSubmission]) -> List[float]:
    if spec.value != VALUE_LOSS:
        return [1.0] * len(subs)
    losses = [s.local_loss for s in subs if s.local_loss is not None and s.local_loss > 0]
    mean = sum(losses) / len(losses) if losses else 0.0
    if mean <= 0.0:
        return [1.0] * len(subs)
    return [
        (float(s.local_loss) / mean) if (s.local_loss is not None and s.local_loss > 0) else 1.0
        for s in subs
    ]


@dataclass(frozen=True)
class UpdateWeight:
    """One update's age, cutoff and raw merge weight."""

    device_id: DeviceID
    basis_version: Optional[int]
    age: Optional[int]
    cap: Optional[int]
    weight: float

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
    """Raw weight w_i per submission; exactly 0.0 past the device's cutoff.

    ``agg:fedbuff`` weights are s(a_i) alone (FedBuff does not weight by n);
    every other rule uses n_i·v_i·s(a_i). An unknown age counts as 0.
    """
    caps = age_caps or {}
    values = _value_proxies(spec, submissions)
    out: List[UpdateWeight] = []
    for sub, v in zip(submissions, values):
        age = update_age(base_version, sub.basis_version)
        a = 0 if age is None else age
        cap = caps.get(sub.device_id) if spec.rule == AGG_CUTOFF else None
        if cap is not None and a > cap:
            w = 0.0
        elif spec.rule == AGG_FEDBUFF:
            w = staleness(spec, a)
        else:
            w = float(sub.num_examples) * v * staleness(spec, a)
        out.append(UpdateWeight(sub.device_id, sub.basis_version, age, cap, w))
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
    result with versions and ages. The age-aware rules merge in delta form.
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
    # FedBuff averages over the COUNT of buffered updates (each already scaled
    # by its staleness); the weighted rules normalise by the weight mass.
    normalizer = float(len(raw)) if spec.rule == AGG_FEDBUFF else float(sum(raw))
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
        weight_mass=normalizer,
        device_basis_versions=tuple(w.basis_version for _, w in admitted),
        device_ages=tuple(w.age for _, w in admitted),
        device_weights=tuple(r / normalizer for r in raw),
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
    """Cluster-side FedBuff buffer: sum of staleness-scaled updates and a count."""

    spec: AggregationSpec
    k: int
    total: Optional[List] = None
    count: int = 0

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

    @property
    def ready(self) -> bool:
        return self.total is not None and self.count >= self.k

    def apply(self, theta: Weights) -> Weights:
        """θ + η · (buffered sum / count); empties the buffer."""
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
        return out
