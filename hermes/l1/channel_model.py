"""Channel models for the mule's radio links (FeRRy Phase 3, unit U2).

Two models live here.

Legacy: the mission-indexed backhaul (EX-4.3)
---------------------------------------------
:class:`ChannelModel`, :func:`loss_from_snr`, :class:`BackhaulPlan` and
:func:`backhaul_plan` were moved here VERBATIM from ``experiments/exp4/channel.py``,
which is now a re-export shim. ``hermes`` must not import ``experiments``
(finding A-01), and the ferry code that needs them lives in ``hermes``. They
model the mule->base-station backhaul as per-band SNR indexed by the MISSION
NUMBER, and the driver turns a band choice into a per-mission upload-loss
schedule for the cluster. The validity discipline behind them is in the shim's
docstring: bands cross over, H2 holds the best-average band, H2 and H3 read one
seeded trace, and clean links show ~no L1 benefit. ``tests/golden/test_golden_channel.py``
pins their output bit for bit at afa9526, including the 120 recorded C1/C2
trials. Do not edit them (Freeze Rule 1). They stay the default
(``backhaul_model="mission"``). The shim re-exports these objects rather than
wrapping them, so ``backhaul_plan`` looks up ``loss_from_snr``,
``best_average_band`` and ``AdaptiveChannelController`` in this module. Patch
them here: rebinding them on ``experiments.exp4.channel`` no longer reaches
``backhaul_plan``.

Seconds axis (Phase 3 design §1 D2)
-----------------------------------
On the simulated mission clock, every channel quantity is a function of the
simulated time ``t`` in seconds (``tau = t - epoch_s``):

* :class:`ContactChannel` gives the mule->device SNR on band class ``b``:
  ``SNR_b(d3D_j) + X_j(t) + I_b(t)``. The mean comes from the contact link
  (``hermes/l1/contact_link.py``, unit U3) as a callable, so neither module
  imports the other. ``X_j`` is per-link shadowing and ``I_b`` a per-class
  interference term.
* :class:`BackhaulChannel` gives the mule->base-station SNR per carrier at the
  dock. It is :class:`ChannelModel` with the mission index ``m/period`` replaced
  by ``tau/P_bh``.
* :func:`keyed_uniform` and :func:`keyed_bernoulli` give outcome draws keyed by
  (salt, key, round). Examples: the backhaul loss by (mule, mission_round) and
  a device's availability by (device, mission_round).

**Pairing by construction.** Nothing here is drawn from a sequential stream.
Every random quantity is a pure function of (salt, stream, key, time bin) or
(salt, key, round), computed with SHA-256 (:func:`smooth_normal`,
:func:`keyed_uniform`). Two arms that ask for the same (t, band, link) get the
same value, whatever else they asked for and in whatever order. A Bernoulli
outcome belongs to its mission, not to its position in a draw sequence. The
legacy per-upload stream paired draws by upload count instead: in 4 of the 40
recorded C1 pairs the two arms docked on different missions. Salts come from
the trial seed through :func:`trial_salt`, which is a copy of
``experiments/exp4/model_task._u32``.

**No horizon.** Each time bin's noise is hashed when it is needed, so there is
no precomputed table to run off the end of. (The prototype's ``drone_env``
noise table stops at ``2*max_steps+1`` entries, and beyond that it silently
has no noise.) The noise is defined for every finite ``t``.

**Reproducibility across platforms (critic A8-ii).** The SHA-256 digest, the
bin index and the conversion of digest bits to a uniform are exact on every
platform. Box-Muller then uses ``math.log`` and ``math.cos``, and the sinusoids
use ``math.sin``. These come from the operating system's C math library, and
IEEE 754 does not fix their last-bit rounding. The values are therefore
reproducible from run to run, and whatever the query order, on one platform.
Bit-identical values across operating systems or math libraries (Windows
against Linux, say) are NOT guaranteed. The differences are around 1e-16
relative, but they can still flip a comparison that sits exactly on a
threshold (SNR against the floor, ``u < p``).

**Backhaul loss magnitudes (critic A4).** Today's ``--realism`` cells without
``--l1-channel`` lose a flat 2 % of uploads when jittery and none when clean
(``Exp4Driver.jittery_backhaul_loss_pct`` and ``clean_backhaul_loss_pct``). A
fixed-carrier arm on the seconds model loses about 16 % per upload when
jittery and about 0.4 % when clean. The Phase 3 critic's probe
(``probe_backhaul_loss.py``: 400 seeds, 4 missions, the legacy model's gain
and phase draws with the regimes' amplitudes and sigma) finds these mean
losses per upload:

==================================  =======  =====
arm and model                       jittery  clean
==================================  =======  =====
seconds axis, fixed ``argmax g_c``  0.161    0.004
legacy, fixed best-average band     0.153    0.004
legacy, adaptive (H3)               0.018    0.004
==================================  =======  =====

This module's own hash-based gain and phase draws give the same picture. Over
400 seeds with 4 missions and T_nom = 219 s, the fixed carrier loses 0.163
jittery and 0.004 clean, averaged over one backhaul period (0.162 and 0.004
at one upload per mission, mid-mission). H3's controller at those uploads
loses 0.017 jittery and 0.004 clean. ``tests/unit/test_channel_model.py``
checks these ranges.

Turning the seconds model on for every mule arm is therefore like switching
``--l1-channel`` on for all of them: it moves every exit-gate number. The
choice is made per study (``backhaul_model``, spec Q7).
"""

from __future__ import annotations

import hashlib
import math
import operator
import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from hermes.l1.channel_utility import AdaptiveChannelController, best_average_band


# --------------------------------------------------------------------------- #
# Legacy mission-indexed backhaul (EX-4.3): moved verbatim from
# experiments/exp4/channel.py, pinned by tests/golden/test_golden_channel.py.
# --------------------------------------------------------------------------- #

@dataclass
class ChannelModel:
    """Per-band effective SNR (dB) over the mission sequence.

    ``snr(m, c) = base + g[c] + amplitude * sin(2*pi*(m/period + phase[c])) + noise``.
    Jittery lowers the base and raises the amplitude/noise, so bands dip into
    lossy troughs at different times.
    """

    n_bands: int = 3
    n_missions: int = 4
    seed: int = 0
    jittery: bool = False

    def __post_init__(self) -> None:
        rng = random.Random((self.seed ^ 0x0C0FFEE) & 0x7FFFFFFF)
        # Per-band static gain g(c) — modest spread so no band dominates.
        self._g = [rng.uniform(0.0, 3.0) for _ in range(self.n_bands)]
        # Distinct phases -> bands peak at different times (crossover).
        self._phase = [i / self.n_bands for i in range(self.n_bands)]
        rng.shuffle(self._phase)
        self._noise = [
            [rng.gauss(0.0, 1.5 if self.jittery else 0.4) for _ in range(self.n_bands)]
            for _ in range(self.n_missions)
        ]
        self._base = 6.0 if self.jittery else 12.0
        self._amp = 5.0 if self.jittery else 1.0
        self._period = max(2, self.n_missions)

    def snr(self, mission: int, band: int) -> float:
        wave = self._amp * math.sin(
            2.0 * math.pi * (mission / self._period + self._phase[band])
        )
        return self._base + self._g[band] + wave + self._noise[mission][band]

    def snr_by_mission(self) -> List[List[float]]:
        return [
            [self.snr(m, b) for b in range(self.n_bands)]
            for m in range(self.n_missions)
        ]


def loss_from_snr(snr_db: float, *, mid: float = 3.0, scale: float = 2.0) -> float:
    """Backhaul upload-loss probability from effective SNR (logistic).

    High SNR -> ~0 loss; SNR below ~``mid`` -> loss climbs toward 1. Tuned so
    a healthy channel (~12 dB) loses ~1% and a deep trough (~1 dB) loses
    ~70%+.
    """
    x = (snr_db - mid) / scale
    return 1.0 / (1.0 + math.exp(x))


@dataclass(frozen=True)
class BackhaulPlan:
    """Per-mission backhaul-loss schedule + the L1 trace behind it."""

    loss_schedule: List[float]
    chosen_bands: List[int]
    mean_chosen_snr_db: float
    adaptive: bool


def backhaul_plan(
    model: ChannelModel,
    *,
    adaptive: bool,
    switch_cost: float = 0.5,
    channel_use_cost: Optional[Tuple[float, ...]] = None,
) -> BackhaulPlan:
    """Turn the channel trace into a per-mission loss schedule.

    ``adaptive=True`` (arm H3) runs the ``U(c,t)`` controller, tracking the
    best band each mission; ``adaptive=False`` (arms H1/H2) holds the single
    best-average band. Both read the same SNR trace.
    """
    snr_by_m = model.snr_by_mission()
    chosen: List[int] = []
    losses: List[float] = []
    snrs: List[float] = []

    if adaptive:
        ctrl = AdaptiveChannelController(
            channel_use_cost=channel_use_cost or tuple(0.0 for _ in range(model.n_bands)),
            switch_cost=switch_cost,
        )
        current = -1
        for snr_per_band in snr_by_m:
            band = ctrl.select(snr_per_band, current)
            current = band
            chosen.append(band)
            snrs.append(snr_per_band[band])
            losses.append(loss_from_snr(snr_per_band[band]))
    else:
        fixed = best_average_band(snr_by_m)
        for snr_per_band in snr_by_m:
            chosen.append(fixed)
            snrs.append(snr_per_band[fixed])
            losses.append(loss_from_snr(snr_per_band[fixed]))

    mean_snr = sum(snrs) / len(snrs) if snrs else 0.0
    return BackhaulPlan(
        loss_schedule=losses,
        chosen_bands=chosen,
        mean_chosen_snr_db=mean_snr,
        adaptive=adaptive,
    )


# --------------------------------------------------------------------------- #
# Seconds axis (FeRRy Phase 3 design §1 D2)
# --------------------------------------------------------------------------- #

#: The channels' default epoch, equal to the mission clock's epoch
#: (``hermes.l1.mission_clock.SIM_EPOCH_S``). Channel time ``tau = t - epoch``
#: then starts at 0 when the trial's mule clocks start, as the legacy mission
#: index does. The value is copied rather than imported so that this module
#: does not depend on the clock module; a unit test checks that the two agree.
DEFAULT_EPOCH_S = 1.0e6

#: Stream names for :func:`ferry_salt`, one per independent quantity of a
#: trial. The contact channel's shadowing and interference share one salt,
#: because their noise streams differ.
SALT_CONTACT = "contact"
SALT_BACKHAUL = "backhaul"
SALT_AVAILABILITY = "availability"      # keyed_uniform(salt, device_id, mission_round)
SALT_BACKHAUL_LOSS = "backhaul_loss"    # keyed_uniform(salt, mule_key, mission_round)

#: Band classes on the one contact carrier, in index order. The index is what
#: ``MissionRoundCloseLine.band`` records (design §2.1). D1: wide 20 MHz,
#: medium 5 MHz, narrow 1.4 MHz.
CONTACT_BANDS: Tuple[str, ...] = ("wide", "medium", "narrow")

#: Contact interference regimes, as (amplitude A, noise sigma_I) in dB.
#: ``clean`` uses the legacy clean amplitude and noise (``ChannelModel``: 1 and
#: 0.4 dB). It is the default whatever the cell's regime, which keeps Exp 4's
#: "jitter hits the backhaul, not the short hop". ``jittery`` uses the legacy
#: jittery values (5 and 1.5 dB) and is meant for test (c) (design §1 D2, Q6).
CONTACT_REGIMES: Mapping[str, Tuple[float, float]] = MappingProxyType({
    "clean": (1.0, 0.4),
    "jittery": (5.0, 1.5),
})

#: Backhaul regimes, as (base, amplitude A, noise sigma) in dB. These are the
#: legacy ``ChannelModel``'s values: 12/1/0.4 clean and 6/5/1.5 jittery.
BACKHAUL_REGIMES: Mapping[str, Tuple[float, float, float]] = MappingProxyType({
    "clean": (12.0, 1.0, 0.4),
    "jittery": (6.0, 5.0, 1.5),
})

#: Shadowing standard deviation in dB. TR 36.777 Annex B, RMa-AV line of sight,
#: gives sigma_SF = 4.2*exp(-0.0046*h) = 3.66-3.83 dB at h = 20-30 m; 4 dB rounds
#: that up (research report §3A). The contact link's mean SNR includes a
#: 1.2816*sigma_sh margin, so this sigma must be the link's sigma_sh.
SHADOW_SIGMA_DB = 4.0
#: Shadowing correlation time in seconds: TR 38.901 Table 7.5-6 gives a 37 m
#: decorrelation distance for RMa line of sight, flown at the 5 m/s cruise
#: speed (research report §3D).
SHADOW_CORR_S = 7.4
#: Grid for position-keyed shadowing in metres: the same 37 m distance.
SHADOW_GRID_M = 37.0
#: Interference period P_c in seconds. This is an assumption (design §1 D2);
#: test (c) sweeps it as the median leg divided by rho.
INTERFERENCE_PERIOD_S = 60.0
#: Noise bin in seconds. Fast fading averages out within it: the coherence
#: time is about 7.3 ms at 5 m/s (AERIQ; research report §3D).
NOISE_BIN_S = 1.0
#: Upper bound of the backhaul carrier gain g_c ~ U(0, 3) dB, the legacy spread.
BACKHAUL_GAIN_MAX_DB = 3.0

# Internal hash streams. They start with "_", which public streams may not
# (see _stream), so a caller's stream never hashes the same text as these.
_U = "_u"
_PHASE = "_phase"
_GAIN = "_gain"
_SHADOW_POS = "_shadow_pos"

_TWO_POW_53 = 9007199254740992.0  # 2**53: a double holds any 53-bit integer exactly


# --- salts ------------------------------------------------------------------ #

def trial_salt(seed: int, *parts) -> int:
    """A 32-bit salt from the trial seed: the first 4 bytes of ``sha256("seed|part|...")``.

    This is a copy of ``experiments/exp4/model_task._u32``, the helper behind
    ``device_reliabilities`` and the D4 CARP split. It is copied because
    ``hermes`` must not import ``experiments`` (finding A-01). A unit test
    keeps the two identical.
    """
    payload = "|".join([str(seed), *(str(p) for p in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


def ferry_salt(seed: int, stream: str) -> int:
    """The design's salt for one seconds-axis quantity of a trial:
    ``trial_salt(seed, "ferry", stream)`` (design §1 D2)."""
    return trial_salt(seed, "ferry", stream)


# --- hashing ---------------------------------------------------------------- #

def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _unit(digest: bytes, offset: int = 0) -> float:
    """A uniform on [0, 1) from 8 digest bytes.

    The top 53 bits are divided by 2**53. The result is exact in a double, so
    it is identical on every platform.
    """
    return (int.from_bytes(digest[offset:offset + 8], "big") >> 11) / _TWO_POW_53


def _normal(text: str) -> float:
    """A standard normal from ``sha256(text)`` by Box-Muller (cosine branch).

    ``u1 = 1 - U`` lies in (0, 1], so ``log(u1)`` is finite and |z| <= 8.6.
    """
    d = _digest(text)
    u1 = 1.0 - _unit(d, 0)
    u2 = _unit(d, 8)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _salt(salt: Any) -> int:
    # A float salt would hash as "1.0" rather than "1" and silently give a
    # different stream, so only integers (Python or numpy) are accepted.
    if isinstance(salt, bool):
        raise TypeError(f"salt must be an integer, got {salt!r}")
    try:
        return operator.index(salt)
    except TypeError:
        raise TypeError(f"salt must be an integer, got {salt!r}") from None


def _stream(stream: Any) -> str:
    # The hashed text is "salt|stream|key|k". Salt and k are integers and the
    # stream has no "|", so the text parses one way only, whatever the key holds.
    if not isinstance(stream, str) or not stream or "|" in stream or stream.startswith("_"):
        raise ValueError(
            f"stream must be a non-empty str without '|' that does not start "
            f"with '_' (reserved), got {stream!r}"
        )
    return stream


def _finite(x: Any, name: str) -> float:
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {x!r}")
    return value


def _positive(x: Any, name: str) -> float:
    value = _finite(x, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x!r}")
    return value


def _non_negative(x: Any, name: str) -> float:
    value = _finite(x, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x!r}")
    return value


def _round_index(round_index: Any) -> int:
    if isinstance(round_index, bool):
        raise TypeError(f"round must be an integer, got {round_index!r}")
    try:
        return operator.index(round_index)
    except TypeError:
        raise TypeError(f"round must be an integer, got {round_index!r}") from None


# --- noise and keyed draws -------------------------------------------------- #

def smooth_normal(
    salt: int,
    stream: str,
    key: Hashable,
    t: float,
    bin_s: float,
    epoch: float,
) -> float:
    """Standard-normal noise as a pure, continuous function of time (design §1 D2).

    ``z(k)`` is a Box-Muller draw from ``sha256(salt|stream|key|k)`` for time bin
    ``k = floor((t - epoch)/bin_s)``. Within a bin the value moves linearly
    from ``z(k)`` to ``z(k+1)`` and is rescaled to unit variance::

        xi(t) = ((1-u)*z(k) + u*z(k+1)) / sqrt((1-u)**2 + u**2),
        where u = (t - epoch)/bin_s - k is the position within the bin.

    Properties (``tests/unit/test_channel_model.py``):

    * Pure. The same arguments give the same value in any process, whatever
      was computed before and in whatever order. No stream is consumed.
    * N(0, 1) at every ``t``. Without the rescaling, the variance would be 0.5
      mid-bin. The value is continuous in ``t``.
    * ``bin_s`` acts as the correlation time. The correlation is about 0.74 at
      half a bin and 0.29 at one bin, and exactly 0 from two bins on. It
      reaches 1/e at 0.9 bin, close to an exponential model with time
      constant ``bin_s``.
    * No horizon: the value is defined for every finite ``t``.

    ``key`` can be anything whose ``str()`` names the link or band: a device id
    or a band index, for instance. ``stream`` must not contain ``'|'`` or start
    with ``'_'`` (reserved). Every consumer of one stream must use the same
    ``epoch`` and ``bin_s``, or the bins do not line up and the values are not
    paired. ``epoch`` therefore has no default, as in the design's signature
    (§2.1). The channels pass their ``epoch_s``, :data:`DEFAULT_EPOCH_S` unless
    set; a default of 0.0 here would silently put a direct caller on other bins.
    Bit-identity across OS math libraries is not guaranteed (see the module
    docstring, critic A8-ii).
    """
    salt = _salt(salt)
    stream = _stream(stream)
    t = _finite(t, "t")
    bin_s = _positive(bin_s, "bin_s")
    epoch = _finite(epoch, "epoch")
    x = (t - epoch) / bin_s
    k = math.floor(x)
    u = x - k
    z0 = _normal(f"{salt}|{stream}|{key}|{k}")
    if u == 0.0:
        # This is the formula below at u = 0, bit for bit, minus one hash.
        return z0
    z1 = _normal(f"{salt}|{stream}|{key}|{k + 1}")
    a = 1.0 - u
    return (a * z0 + u * z1) / math.sqrt(a * a + u * u)


def keyed_uniform(salt: int, key: Hashable, round_index: int) -> float:
    """A uniform on [0, 1) keyed by (salt, key, round), not drawn from a stream.

    This is the design's ``u(salt, key, round)`` (§1 D2). Two uses:

    * The backhaul loss of a mule's mission is
      ``keyed_uniform(ferry_salt(seed, SALT_BACKHAUL_LOSS), mule_key, mission_round) < p``.
    * A device's availability at its Pass-1 contact is
      ``keyed_uniform(ferry_salt(seed, SALT_AVAILABILITY), device_id, mission_round) < rel_i``.

    The value depends only on its key. The outcome of mission ``m`` is
    therefore the same in every arm, whichever earlier missions docked, and it
    does not depend on the order in which uploads reach the cluster. The value
    is exact on every platform (integer arithmetic and one division by a power
    of two).
    """
    salt = _salt(salt)
    r = _round_index(round_index)
    return _unit(_digest(f"{salt}|{_U}|{key}|{r}"))


def keyed_bernoulli(p: float, salt: int, key: Hashable, round_index: int) -> bool:
    """True with probability ``p``: ``keyed_uniform(salt, key, round) < p``.

    ``p <= 0`` is never true and ``p >= 1`` always is.
    """
    return keyed_uniform(salt, key, round_index) < _finite(p, "p")


#: The design's name for :func:`keyed_uniform` (``u(salt, key, round)``).
u = keyed_uniform


def _shuffled_phases(salt: int, tag: str, n: int) -> Tuple[float, ...]:
    """Assign the phases {0, 1/n, ..., (n-1)/n} to ``n`` bands in a seeded order.

    The backhaul carriers use this. The legacy model shuffles ``[i/n]`` with
    its sequential RNG. Here the order is the sort order of
    ``sha256(salt|_phase|tag|i)``, a pure function of the salt. Distinct phases
    keep today's crossover: no two bands peak at the same time
    (``ChannelModel.__post_init__``). As in the legacy model, changing ``n``
    re-draws every phase.
    """
    order = sorted(range(n), key=lambda i: _digest(f"{salt}|{_PHASE}|{tag}|{i}"))
    phases = [0.0] * n
    for rank, band in enumerate(order):
        phases[band] = rank / n
    return tuple(phases)


def _contact_phases(salt: int, bands: Sequence[str]) -> Tuple[float, ...]:
    """The contact classes' interference phases ``phi_b``, keyed by class name, in ``bands`` order.

    * The three D1 classes (:data:`CONTACT_BANDS`) take a seeded shuffle of
      {0, 1/3, 2/3} (design §1 D2). They are ranked by
      ``sha256(salt|_phase|contact|name)`` over the three names, whichever of
      them the channel carries.
    * Any other class (unit U3's optional 10 MHz ``medium_wide``) takes the
      midpoint of one of the three gaps, {1/6, 1/2, 5/6}, picked by the same
      hash of its own name.

    A class's phase is therefore a function of the salt and its own name.
    Adding, dropping or reordering classes leaves every other class's phase
    alone, so the class-set sweep (design §1 D1: 3 classes, or 4 with 10 MHz)
    is paired on ``I_b(t)`` for the classes both sets carry. The phases stay
    distinct, which keeps the legacy crossover: an extra class peaks 1/6 of a
    period away from its neighbours. Two extra classes could share a midpoint,
    but the design defines only one.
    """
    ranked = sorted(CONTACT_BANDS, key=lambda name: _digest(f"{salt}|{_PHASE}|contact|{name}"))
    thirds = {name: rank / len(ranked) for rank, name in enumerate(ranked)}
    phases = []
    for name in bands:
        if name in thirds:
            phases.append(thirds[name])
        else:
            gap = int.from_bytes(_digest(f"{salt}|{_PHASE}|contact|{name}")[:8], "big") % 3
            phases.append((2 * gap + 1) / 6)
    return tuple(phases)


# --- the contact link ------------------------------------------------------- #

class ContactChannel:
    """Mule->device SNR on a band class at simulated time ``t`` (design §1 D2).

    ``snr_db(t, b, d, j) = mean(b, d) + X_j(t) + I_b(t)``, in dB. The terms:

    * ``mean(b, d_planar)`` is the contact link's mean SNR (unit U3's
      ``ContactLink.mean_snr_db``), passed in as a callable and called with the
      band's NAME. It includes the 1.2816*sigma_sh margin that makes R(b) the
      range with 90 % availability at the edge. So ``shadow_sigma_db`` must be
      the link's sigma_sh (:meth:`from_link` takes it from the link).
    * ``X_j`` is the shadowing of link ``j``, with standard deviation
      ``shadow_sigma_db``. All classes share it, because they share one carrier.

      - By default it is keyed by time:
        ``sigma_sh * smooth_normal(salt, "shadow", j, t, shadow_corr_s, epoch)``,
        with a 7.4 s correlation time.
      - ``shadow_keying="position"`` (critic C5/A6) keys it instead by (device,
        stop position quantized to ``shadow_grid_m``). The value is then fixed
        for a device and grid cell whatever the time: two arms hovering in the
        same cell see the same shadowing, and a hover does not change it. All
        the time variation is then in ``I_b``.
    * ``I_b(t) = A*sin(2*pi*(tau/(P_c*m_b) + phi_b)) + sigma_I*xi_b(t)`` is a
      declared per-class interference model, not a propagation claim.

      - ``phi_b``: the three D1 classes take a seeded shuffle of
        {0, 1/3, 2/3}, and an extra class (the 10 MHz option) takes a gap
        midpoint, {1/6, 1/2, 5/6}. Distinct phases keep the legacy crossover.
      - ``P_c`` is common to all bands unless ``period_multipliers`` say
        otherwise.
      - ``xi_b`` is the 1 s-bin noise keyed by the band name.
      - ``A`` and ``sigma_I`` come from ``regime``: clean by default, jittery
        for test (c).
      - ``phi_b`` and ``xi_b`` are keyed by the class name, not its index
        (:func:`_contact_phases`). A class therefore keeps its ``I_b(t)``
        when the class set changes (design §1 D1 sweeps it): in the
        four-class channel, in a subset, or in another order.

    Classes carry no random static gain; their offset is structural, through
    R_b. :meth:`pred_snr_db` is the mean alone, for planning: no noise and no
    time. A band can be given as an index into ``bands``, a name, or an object
    with a ``name`` (a ``BandClass``). It is resolved to its name before the
    mean is called or the noise is keyed, so all three forms give the same
    value.

    Every term is a pure function of its arguments (see the module docstring),
    so arms that share a salt are paired by construction. Treat an instance as
    immutable and build a new one to change a parameter.
    """

    def __init__(
        self,
        mean_snr_db: Callable[[str, float], float],
        *,
        salt: int,
        bands: Sequence[str] = CONTACT_BANDS,
        regime: str = "clean",
        interference_amp_db: Optional[float] = None,
        interference_sigma_db: Optional[float] = None,
        interference_period_s: float = INTERFERENCE_PERIOD_S,
        period_multipliers: Optional[Sequence[float]] = None,
        noise_bin_s: float = NOISE_BIN_S,
        shadow_sigma_db: float = SHADOW_SIGMA_DB,
        shadow_corr_s: float = SHADOW_CORR_S,
        shadow_keying: str = "time",
        shadow_grid_m: float = SHADOW_GRID_M,
        epoch_s: float = DEFAULT_EPOCH_S,
    ) -> None:
        if not callable(mean_snr_db):
            raise TypeError("mean_snr_db must be a callable (band name, d_planar) -> dB")
        self.mean_snr_db = mean_snr_db
        self.salt = _salt(salt)
        self.bands: Tuple[str, ...] = tuple(bands)
        if (not self.bands or len(set(self.bands)) != len(self.bands)
                or not all(isinstance(b, str) and b for b in self.bands)):
            raise ValueError(f"bands must be distinct non-empty names, got {bands!r}")
        self._index: Dict[str, int] = {name: i for i, name in enumerate(self.bands)}
        if regime not in CONTACT_REGIMES:
            raise ValueError(f"unknown contact regime {regime!r}; known: {sorted(CONTACT_REGIMES)}")
        self.regime = regime
        amp, sigma = CONTACT_REGIMES[regime]
        self.interference_amp_db = _non_negative(
            amp if interference_amp_db is None else interference_amp_db, "interference_amp_db")
        self.interference_sigma_db = _non_negative(
            sigma if interference_sigma_db is None else interference_sigma_db,
            "interference_sigma_db")
        self.interference_period_s = _positive(interference_period_s, "interference_period_s")
        if period_multipliers is None:
            mults: Tuple[float, ...] = tuple(1.0 for _ in self.bands)
        else:
            mults = tuple(_positive(m, "period_multipliers[i]") for m in period_multipliers)
            if len(mults) != len(self.bands):
                raise ValueError(
                    f"period_multipliers needs one entry per band ({len(self.bands)}), "
                    f"got {len(mults)}")
        self.period_multipliers = mults
        self.noise_bin_s = _positive(noise_bin_s, "noise_bin_s")
        self.shadow_sigma_db = _non_negative(shadow_sigma_db, "shadow_sigma_db")
        self.shadow_corr_s = _positive(shadow_corr_s, "shadow_corr_s")
        if shadow_keying not in ("time", "position"):
            raise ValueError(f"shadow_keying must be 'time' or 'position', got {shadow_keying!r}")
        self.shadow_keying = shadow_keying
        self.shadow_grid_m = _positive(shadow_grid_m, "shadow_grid_m")
        self.epoch_s = _finite(epoch_s, "epoch_s")
        self.phases: Tuple[float, ...] = _contact_phases(self.salt, self.bands)

    @classmethod
    def from_link(cls, link: Any, *, salt: int, **kwargs: Any) -> "ContactChannel":
        """Build a channel on a contact link object without importing its module.

        The link is duck-typed on unit U3's ``ContactLink`` (design §2.1):

        * Its ``mean_snr_db(band, d_planar)`` gives the mean.
        * Its ``names``, when present, become the band list, so the channel
          has the link's classes in the link's index order (the four-class
          option included).
        * Its ``shadow_sigma_db``, when present, becomes the shadowing sigma,
          so the shadowing matches the margin built into the mean.

        An explicit ``bands`` or ``shadow_sigma_db`` that disagrees with the
        link's is refused.
        """
        names = getattr(link, "names", None)
        if names is not None:
            names = tuple(names)
            if "bands" in kwargs and tuple(kwargs["bands"]) != names:
                raise ValueError(
                    f"bands={tuple(kwargs['bands'])!r} disagrees with the link's classes {names!r}")
            kwargs.setdefault("bands", names)
        sigma = getattr(link, "shadow_sigma_db", None)
        if sigma is not None:
            if "shadow_sigma_db" in kwargs and float(kwargs["shadow_sigma_db"]) != float(sigma):
                raise ValueError(
                    f"shadow_sigma_db={kwargs['shadow_sigma_db']!r} disagrees with the "
                    f"link's sigma_sh {sigma!r}, whose margin is in the mean SNR")
            kwargs.setdefault("shadow_sigma_db", float(sigma))
        return cls(link.mean_snr_db, salt=salt, **kwargs)

    def band_index(self, band: Any) -> int:
        """Resolve a band given as an index, a name, or an object with a ``name``."""
        if isinstance(band, bool):
            raise TypeError(f"band must be an index or a name, got {band!r}")
        name = getattr(band, "name", band)
        if isinstance(name, str):
            try:
                return self._index[name]
            except KeyError:
                raise ValueError(f"unknown band {name!r}; known: {self.bands}") from None
        try:
            i = operator.index(name)
        except TypeError:
            raise TypeError(f"band must be an index or a name, got {band!r}") from None
        if not 0 <= i < len(self.bands):
            raise ValueError(f"band index {i} out of range for {self.bands}")
        return i

    def pred_snr_db(self, band: Any, d_planar: float) -> float:
        """The planning SNR: the mean alone, with no shadowing, no interference and no time."""
        name = self.bands[self.band_index(band)]
        return float(self.mean_snr_db(name, _non_negative(d_planar, "d_planar")))

    def shadow_db(self, t: float, link_key: Hashable, *,
                  stop_pos: Optional[Sequence[float]] = None) -> float:
        """``X_j(t)``: the shadowing of link ``link_key``, in dB, the same for every class.

        With ``shadow_keying="position"``, ``stop_pos`` (the mule's planar
        (x, y, ...) position) is required, and ``t`` then only has to be finite.
        """
        t = _finite(t, "t")
        if self.shadow_keying == "position":
            if stop_pos is None:
                raise ValueError("position-keyed shadowing needs stop_pos")
            ix = math.floor(_finite(stop_pos[0], "stop_pos[0]") / self.shadow_grid_m)
            iy = math.floor(_finite(stop_pos[1], "stop_pos[1]") / self.shadow_grid_m)
            return self.shadow_sigma_db * _normal(
                f"{self.salt}|{_SHADOW_POS}|{link_key}|{ix}|{iy}")
        return self.shadow_sigma_db * smooth_normal(
            self.salt, "shadow", link_key, t, self.shadow_corr_s, self.epoch_s)

    def interference_db(self, t: float, band: Any) -> float:
        """``I_b(t)``: band ``b``'s interference term in dB, the same for every link."""
        i = self.band_index(band)
        t = _finite(t, "t")
        tau = t - self.epoch_s
        period = self.interference_period_s * self.period_multipliers[i]
        wave = self.interference_amp_db * math.sin(2.0 * math.pi * (tau / period + self.phases[i]))
        # Keyed by the class name, not its index, like the phase: a class keeps
        # its noise when the class set changes. (U3 appends the 10 MHz class, so
        # indices 0-2 do not shift there; a subset or a reordering would.)
        return wave + self.interference_sigma_db * smooth_normal(
            self.salt, "interference", self.bands[i], t, self.noise_bin_s, self.epoch_s)

    def snr_db(self, t: float, band: Any, d_planar: float, link_key: Hashable, *,
               stop_pos: Optional[Sequence[float]] = None) -> float:
        """The realized SNR of link ``link_key`` on ``band`` at time ``t``, in dB.

        This is ``pred_snr_db(band, d_planar) + shadow_db(t, link_key) +
        interference_db(t, band)``, added in that order.
        """
        return (self.pred_snr_db(band, d_planar)
                + self.shadow_db(t, link_key, stop_pos=stop_pos)
                + self.interference_db(t, band))

    def snr_by_band(self, t: float, d_planar: float, link_key: Hashable, *,
                    stop_pos: Optional[Sequence[float]] = None) -> Tuple[float, ...]:
        """:meth:`snr_db` on every class, in ``bands`` order (for the L1 state)."""
        return tuple(self.snr_db(t, i, d_planar, link_key, stop_pos=stop_pos)
                     for i in range(len(self.bands)))

    def describe(self) -> Dict[str, Any]:
        """The parameters as a JSON-able dict (for ``mule_ready.channel_params``)."""
        return {
            "model": "contact",
            "salt": self.salt,
            "bands": list(self.bands),
            "regime": self.regime,
            "interference_amp_db": self.interference_amp_db,
            "interference_sigma_db": self.interference_sigma_db,
            "interference_period_s": self.interference_period_s,
            "period_multipliers": list(self.period_multipliers),
            "phases": list(self.phases),
            "noise_bin_s": self.noise_bin_s,
            "shadow_sigma_db": self.shadow_sigma_db,
            "shadow_corr_s": self.shadow_corr_s,
            "shadow_keying": self.shadow_keying,
            "shadow_grid_m": self.shadow_grid_m,
            "epoch_s": self.epoch_s,
        }


# --- the backhaul ----------------------------------------------------------- #

def backhaul_period_s(n_missions: int, t_nom_s: float) -> float:
    """The backhaul period ``P_bh = n_missions * T_nom`` (design §1 D2).

    This is today's "one cycle per trial", in seconds. The legacy period is
    ``max(2, n_missions)`` missions; the floor of 2 is not carried over,
    because with a single mission the phase at the one upload is arbitrary
    either way.
    """
    n = _round_index(n_missions)
    if n < 1:
        raise ValueError(f"n_missions must be >= 1, got {n_missions!r}")
    return n * _positive(t_nom_s, "t_nom_s")


class BackhaulChannel:
    """Mule->base-station SNR per carrier at simulated time ``t`` (design §1 D2).

    This is the seconds analogue of :class:`ChannelModel`::

        SNR_c(t) = base + g_c + A*sin(2*pi*(tau/P_bh + phi_c)) + sigma*xi_c(t)

    * ``base``, ``A`` and ``sigma`` come from the legacy regimes: 12/1/0.4 dB
      clean and 6/5/1.5 dB jittery (:data:`BACKHAUL_REGIMES`).
    * ``g_c ~ U(0, 3)`` dB and ``phi_c``, a shuffle of {i/n}, are pure functions
      of the salt. The legacy model draws them from a sequential RNG: the
      distributions are the same, the values are not. As in the legacy model,
      changing ``n_carriers`` re-draws every ``phi_c`` (``g_c`` is keyed by
      carrier and stays); the design fixes 3 carriers.
    * ``P_bh = n_missions * T_nom`` (:func:`backhaul_period_s`) is passed in
      as ``period_s``.
    * ``xi_c`` is the 1 s-bin noise keyed by the carrier (:func:`smooth_normal`).

    It is evaluated at the dock at the simulated upload time. The upload-loss
    probability is :func:`loss_from_snr` of that SNR, unchanged. Carrier choice:

    * :meth:`fixed_band` is ``argmax_c g_c``, the noise-free long-run best
      carrier (the sinusoid and the noise both average to 0). It replaces the
      legacy ``best_average_band``, which averaged the realized trace and so
      used the future.
    * H3 runs :class:`AdaptiveChannelController` at every upload instead
      (:meth:`select_carrier`), holding ``current`` across missions.

    The loss magnitudes are in the module docstring (critic A4): about 16 %
    per upload jittery at the fixed carrier, against today's flat 2 %. Treat
    an instance as immutable.
    """

    def __init__(
        self,
        *,
        salt: int,
        period_s: float,
        regime: str = "clean",
        n_carriers: int = 3,
        base_db: Optional[float] = None,
        amp_db: Optional[float] = None,
        sigma_db: Optional[float] = None,
        gain_max_db: float = BACKHAUL_GAIN_MAX_DB,
        noise_bin_s: float = NOISE_BIN_S,
        epoch_s: float = DEFAULT_EPOCH_S,
    ) -> None:
        self.salt = _salt(salt)
        self.period_s = _positive(period_s, "period_s")
        if regime not in BACKHAUL_REGIMES:
            raise ValueError(f"unknown backhaul regime {regime!r}; known: {sorted(BACKHAUL_REGIMES)}")
        self.regime = regime
        n = _round_index(n_carriers)
        if n < 1:
            raise ValueError(f"n_carriers must be >= 1, got {n_carriers!r}")
        self.n_carriers = n
        base, amp, sigma = BACKHAUL_REGIMES[regime]
        self.base_db = _finite(base if base_db is None else base_db, "base_db")
        self.amp_db = _non_negative(amp if amp_db is None else amp_db, "amp_db")
        self.sigma_db = _non_negative(sigma if sigma_db is None else sigma_db, "sigma_db")
        self.gain_max_db = _non_negative(gain_max_db, "gain_max_db")
        self.noise_bin_s = _positive(noise_bin_s, "noise_bin_s")
        self.epoch_s = _finite(epoch_s, "epoch_s")
        self.gains_db: Tuple[float, ...] = tuple(
            self.gain_max_db * _unit(_digest(f"{self.salt}|{_GAIN}|backhaul|{c}"))
            for c in range(n))
        self.phases: Tuple[float, ...] = _shuffled_phases(self.salt, "backhaul", n)

    def _carrier(self, carrier: Any) -> int:
        if isinstance(carrier, bool):
            raise TypeError(f"carrier must be an integer, got {carrier!r}")
        try:
            c = operator.index(carrier)
        except TypeError:
            raise TypeError(f"carrier must be an integer, got {carrier!r}") from None
        if not 0 <= c < self.n_carriers:
            raise ValueError(f"carrier {c} out of range 0..{self.n_carriers - 1}")
        return c

    def pred_snr_db(self, carrier: int) -> float:
        """The long-run mean ``base + g_c``, with no sinusoid and no noise.

        This is the causal, noise-free SNR a planner may use to price an upload.
        """
        return self.base_db + self.gains_db[self._carrier(carrier)]

    def snr_db(self, t: float, carrier: int) -> float:
        """The realized SNR of ``carrier`` at simulated time ``t``, in dB."""
        c = self._carrier(carrier)
        t = _finite(t, "t")
        tau = t - self.epoch_s
        wave = self.amp_db * math.sin(2.0 * math.pi * (tau / self.period_s + self.phases[c]))
        return self.base_db + self.gains_db[c] + wave + self.sigma_db * smooth_normal(
            self.salt, "backhaul", c, t, self.noise_bin_s, self.epoch_s)

    def snr_all(self, t: float) -> Tuple[float, ...]:
        """:meth:`snr_db` of every carrier at ``t``: the controller's observation."""
        return tuple(self.snr_db(t, c) for c in range(self.n_carriers))

    def loss_probability(self, t: float, carrier: int) -> float:
        """The upload-loss probability ``loss_from_snr(snr_db(t, carrier))``."""
        return loss_from_snr(self.snr_db(t, carrier))

    def fixed_band(self) -> int:
        """``argmax_c g_c``, the carrier the non-adaptive arms hold (ties go to the lowest index)."""
        return max(range(self.n_carriers), key=lambda c: self.gains_db[c])

    def select_carrier(self, t: float, *, adaptive: bool, current: int = -1,
                       controller: Optional[AdaptiveChannelController] = None) -> int:
        """The carrier for an upload at ``t``.

        When not ``adaptive``, this is :meth:`fixed_band`. When ``adaptive``
        (arm H3), it is ``controller.select(snr_all(t), current)``. The
        default controller is the one ``backhaul_plan`` builds: zero use cost
        and a 0.5 switch cost. ``current`` is the carrier held since the last
        upload (-1 before the first).
        """
        if not adaptive:
            return self.fixed_band()
        ctrl = controller if controller is not None else AdaptiveChannelController(
            channel_use_cost=tuple(0.0 for _ in range(self.n_carriers)),
            switch_cost=0.5,
        )
        return ctrl.select(self.snr_all(t), current)

    def describe(self) -> Dict[str, Any]:
        """The parameters as a JSON-able dict (for ``mule_ready.channel_params``)."""
        return {
            "model": "backhaul",
            "salt": self.salt,
            "regime": self.regime,
            "n_carriers": self.n_carriers,
            "base_db": self.base_db,
            "amp_db": self.amp_db,
            "sigma_db": self.sigma_db,
            "period_s": self.period_s,
            "gain_max_db": self.gain_max_db,
            "gains_db": list(self.gains_db),
            "phases": list(self.phases),
            "fixed_band": self.fixed_band(),
            "noise_bin_s": self.noise_bin_s,
            "epoch_s": self.epoch_s,
        }


__all__ = [
    # legacy, verbatim
    "BackhaulPlan",
    "ChannelModel",
    "backhaul_plan",
    "loss_from_snr",
    # seconds axis
    "BACKHAUL_GAIN_MAX_DB",
    "BACKHAUL_REGIMES",
    "BackhaulChannel",
    "CONTACT_BANDS",
    "CONTACT_REGIMES",
    "ContactChannel",
    "DEFAULT_EPOCH_S",
    "INTERFERENCE_PERIOD_S",
    "NOISE_BIN_S",
    "SALT_AVAILABILITY",
    "SALT_BACKHAUL",
    "SALT_BACKHAUL_LOSS",
    "SALT_CONTACT",
    "SHADOW_CORR_S",
    "SHADOW_GRID_M",
    "SHADOW_SIGMA_DB",
    "backhaul_period_s",
    "ferry_salt",
    "keyed_bernoulli",
    "keyed_uniform",
    "smooth_normal",
    "trial_salt",
    "u",
]
