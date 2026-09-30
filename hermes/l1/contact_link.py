"""Contact link: band classes and the range-rate model (FeRRy Phase 3, D1).

The contact link is the short hop between the mule, hovering at a stop, and
the ground devices of that contact. This module prices that hop: how far each
band class reaches, what rate it carries at a given SNR, and how long a
payload keeps the mule hovering. That hover time is the dwell the mission
clock charges. The module is pure arithmetic, with no state, no randomness and
no clock. The time-varying part of the SNR (shadowing X_j(t) and the class
interference term I_b(t)) belongs to ``l1/channel_model.py``; ``mean_snr_db``
here is the mean that it varies around.

Nothing on a legacy path imports this module. With ``contact_band`` unset,
the pipeline keeps one ``rf_range_m`` and the fixed 1 s session (Freeze
Rule 1). The module uses only the standard library (no numpy) and imports
nothing from ``experiments`` (finding A-01).

Band classes
------------
All classes share one carrier at 3.32 GHz (``CHANNEL_FREQS_GHZ[0]``). The
repo's L1 carriers (3.32/3.34/3.90 GHz, the slide-26 bands of
``l1/channel_ddqn.py``; only 3.32 and 3.34 GHz lie in AERPAW's 3.3-3.45 GHz
band, AERPAW User Manual s3.5) differ by only 0.05-1.4 dB of free-space loss,
which is too little to carry a range trade, so the classes differ in LTE
channel bandwidth instead::

    class   bandwidth N_RB B_occ    kappa floor rate peak rate  R slant / planar (m)
                                          (CQI 1)    (CQI 15)   n = 2.2       n = 3.0
    wide    20 MHz    100  18 MHz   0.754 2.07 Mb/s  75.4 Mb/s   65.0 /  60.0  65.0 /  60.0
    medium  5 MHz     25   4.5 MHz  0.734 0.50 Mb/s  18.3 Mb/s  122.1 / 119.5 103.2 / 100.1
    narrow  1.4 MHz   6    1.08 MHz 0.732 0.12 Mb/s  4.39 Mb/s  233.5 / 232.2 166.0 / 164.1

    option: medium_wide  10 MHz, 50 RB, 9 MHz, 0.734, 1.01 / 36.7 Mb/s,
            89.1 / 85.5 m (n = 2.2), 81.9 / 78.0 m (n = 3.0)

The ranges are for h = 25 m with the wide anchor at 60 m planar. ``CLASSES``
is the default set. A class's position in the set is the band index that
trace lines carry (``MissionRoundCloseLine.band``). The optional 10 MHz class
is appended (``CLASSES_WITH_10MHZ``), so indices 0-2 mean the same in both
sets. Phase 3 re-baselines fly ``wide``, which leaves Exp 4's S3a geometry
unchanged; medium and narrow are for Study 5.4.

Formulas
--------
::

    B_occ,b         = N_RB,b * 180 kHz
    SE(s)           = efficiency of the highest CQI whose SNR threshold <= s
    rate(b, s)      = min(kappa_b * B_occ,b * SE(s), B_occ,b * log2(1 + 10^(s/10)))
                      for s >= floor, and 0 below the floor
    M_sh            = z_q * sigma_sh with z_q = Phi^-1(q); q = 0.9 and sigma_sh = 4 dB
                      give 1.2816 * 4 = 5.13 dB
    R_anchor        = hypot(anchor_planar_m, h)            (slant; the anchor is wide)
    R_b             = R_anchor * (B_occ,anchor / B_occ,b)^(1/n)
    R_planar(b)     = sqrt(R_b^2 - h^2), and exactly anchor_planar_m for the anchor
    d3D             = max(hypot(d_planar, h), 1 m)
    SNR_b(d3D)      = floor + M_sh + 10 * n * log10(R_b / d3D)             (mean)
    dwell(N, b, s)  = 8 * N / rate(b, s) seconds for N bytes

R_b follows from one EIRP shared by every class. Noise power scales with the
occupied bandwidth, so narrowing the channel from B_occ,anchor to B_occ,b
raises the SNR at every distance by 10*log10(B_occ,anchor / B_occ,b) dB. A
log-distance path loss with exponent n turns that gain into the range factor
above. The ratio uses the occupied bandwidth (100/6 PRB, 12.2 dB), not the
nominal one (20/1.4 MHz, 11.5 dB). The mean-SNR formula is that same budget
written relative to the edge (see ``implied_eirp_dbm``).

Altitude enters only through d3D. Flight legs, S3a and positions stay planar
(z = 0). The log-distance law is referenced to 1 m, the free-space reference
distance, and is not used inside it: a slant distance below 1 m, which only an
altitude below 1 m allows, is evaluated at 1 m.

kappa folds the LTE overheads (control region, reference signals, coding)
into one factor per class. The Shannon cap never binds with this table at the
default floor: SE / log2(1 + SNR) at the CQI thresholds is 0.545-0.736, so
kappa * SE stays below 0.56 of Shannon. The cap is kept so that a lower floor
(see "Other floors") can never price a rate above capacity.

Sources
-------
* N_RB per channel bandwidth: 3GPP TS 36.104 V12.10.0, Table 5.6-1.
  B_occ = N_RB x 180 kHz is its transmission bandwidth configuration (section
  3.2). The downlink's extra 15 kHz DC subcarrier is ignored.
* The CQI table (modulation, code rate x 1024, efficiency): 3GPP TS 36.213
  V8.8.0, Table 7.2.3-1. A CQI is the highest index whose transport block
  decodes with an error probability of at most 0.1. NR's TS 38.214 Table
  5.2.2.1-2 is identical.
* The SNR threshold per CQI: the AERPAW digital-twin abstraction of
  M. S. Hossen, A. Gurses, M. Sichitiu and I. Guvenc, "Accelerating
  Development in UAV Network Digital Twins with a Flexible Simulation
  Framework", arXiv:2503.07935 (2025), Algorithm 1. The paper does not print
  the table; the thresholds come from the authors' ``getSpectralEfficiency``
  (``Autonomous Trajectory/main_auto_traj.m``,
  github.com/mhossenece/UAVFlexSimFramework). The paper takes the mapping from
  S. Sesia, I. Toufik and M. Baker, "LTE - The UMTS Long Term Evolution",
  2nd ed., Wiley 2011 (not checked here). The code rounds efficiencies to two
  decimals, so this module pairs its thresholds with the 36.213 efficiencies.
* kappa_b = peak TBS / (B_occ,b x 5.5547), rounded to three decimals. The
  peak TBS is the single-layer transport block at I_TBS 26 per 1 ms TTI, from
  TS 36.213 V8.8.0 Table 7.1.7.2.1-1: 75,376, 36,696, 18,336 and 4,392 bits
  for 100, 50, 25 and 6 PRB.
* n = 2.2 and sigma_sh = 4 dB: 3GPP TR 36.777 V15.0.0 Annex B, whose
  line-of-sight aerial models at 20-30 m give n = 2.12-2.20 and
  sigma = 3.66-4.09 dB (taken from a third-party transcription). n = 3.0 is
  the sensitivity case: exponents measured at 4-16 m altitude, or
  non-line-of-sight links.
* h = 25 m: the altitude of the AERPAW AADM data-mule challenge (Hossen et
  al., arXiv:2602.16163).
* The shape "0 below the floor, attenuated and capped Shannon above" follows
  3GPP TR 36.942 V10.3.0 Annex A.1.

R(b) is the 90 % edge-availability range, not the floor-rate range
------------------------------------------------------------------
The build plan asks for "range R(b) at the rate floor". Here the mean SNR at
R(b) sits M_sh above the floor, so with shadowing of sigma_sh the link stays
above the floor with probability q = 0.9 at the edge. That is the usual
link-budget margin. (The small I_b(t) term of the channel model, with
A = 1 dB and sigma_I = 0.4 dB by default, lowers the figure to about 0.895.)
At its edge the wide class therefore runs at CQI 3 (5.12 Mb/s), not at the
floor rate. The floor-rate range is where the MEAN SNR reaches the floor
(availability 0.5): R_b * 10^(M_sh / (10 n)), about 111 m slant for wide at
n = 2.2, or 1.71 x R_wide (``floor_range_m``). This is recorded as a plan
deviation (critic A8-i).

The anchor and the link budget it implies
-----------------------------------------
No AERPAW measurement gives R(b), and physical budgets reach kilometres. The
published AERPAW setups (10 dBm, 10/2 dBi antennas) reach 1.5-6 km at the
floor, so a physical budget would never bind in a 100 m field. The anchor
R_planar(wide) = rf_range_m (60 m, Exp 4's default) is therefore an
assumption, chosen so the range gate binds where Exp 4's S3a already does.

Written as a budget, the anchor implies an EIRP of about -11.3 dBm at n = 2.2
(+3.2 dBm at n = 3.0), and the same EIRP for every class
(``implied_eirp_dbm``). The budget has these terms:
* free-space loss of 42.87 dB at 1 m (3.32 GHz), then exponent n beyond 1 m;
* a receiver noise figure of 9 dB (TR 36.942 V10.3.0 Table 4.8, UE);
* a receive antenna gain of 0 dBi;
* the margin M_sh.

The Phase 3 research's own low-altitude budget (0 dBm, 0 dBi, NF 9 dB, the
same margin) reaches 211 m slant on wide at n = 2.2. The anchor is therefore
about 11 dB more pessimistic than that budget. The gap stands for what free
space plus an exponent leaves out: antenna patterns towards the ground, body
and installation losses, and a low-power device radio. This is recorded as a
declared assumption (critic C6).

Below the floor (critic B12)
----------------------------
``rate_bps`` is 0 below the floor. ``dwell_s`` then returns None, never
infinity, because the link cannot carry the payload at all. Callers must not
charge it:
* a contact member below the floor is unreachable (TIMEOUT with
  answered=False, and no dwell);
* a backhaul upload below the floor is a lost upload, and the caller charges
  its own documented cap on the clock.
Test the result with ``is None``: a zero-byte payload above the floor has a
dwell of 0.0.

Other floors
------------
``snr_floor_db`` is a sweep parameter: -6.7 dB, or the -10 dB that TR 36.942's
form allows. With a floor below CQI 1's threshold, CQI 1 applies from the
floor up, so "rate > 0" and "at or above the floor" remain the same test. The
Shannon cap then binds only below about -10.8 dB. The cap is computed as
log1p(SNR) / ln 2, which stays accurate where 1 + SNR would round to 1, so the
capped rate is positive, and never above capacity, at every accepted floor.
A floor below -30 dB
(``MIN_SNR_FLOOR_DB``) is refused. That is 20 dB below the lowest floor any
source here uses (TR 36.942's -10 dB), so it is taken for a units mistake,
such as -67 typed for -6.7. With the anchor held, the floor also moves every
mean SNR, because the edge sits M_sh above it. A floor sweep at a fixed
anchor is therefore an EIRP sweep. To model a more sensitive receiver at a
fixed EIRP, scale R_anchor by 10^(delta / (10 n)) instead.

Worked numbers
--------------
At a stop (h = 25 m, n = 2.2) over 0-60 m planar:
* rates run from 20.0 to 5.1 Mb/s on wide, 9.0 to 3.9 Mb/s on medium, and
  3.6 to 1.9 Mb/s on narrow; no class reaches CQI 15;
* an IDS Pass-1 session (37,576 B: 18,820 pushed and 18,756 returned) takes
  0.015-0.059 s on wide and 0.084-0.158 s on narrow;
* 1 MB each way takes 0.8-3.1 s on wide and 4.5-8.4 s on narrow, and 53.7 s
  on narrow at 200 m.
Wide out-rates the narrower classes wherever it reaches, apart from CQI-step
artefacts. The class trade is therefore reach against dwell.

Not modelled: fast fading (coherence time about 7.3 ms at 5 m/s, so it
averages out within the channel model's 1 s step), HARQ, multi-antenna gains,
and any random per-class gain; the classes differ structurally, through R_b.
The results use the platform's log10 and pow, so they are not guaranteed to
be bit-identical across operating systems.
"""

from __future__ import annotations

import bisect
import math
import numbers
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple, Union

__all__ = [
    "BAND_CLASSES",
    "BandClass",
    "BandRef",
    "CLASSES",
    "CLASSES_WITH_10MHZ",
    "CONTACT_CARRIER_HZ",
    "CQIEntry",
    "CQI_TABLE",
    "ContactLink",
    "DEFAULT_BAND_CLASSES",
    "MEDIUM",
    "MEDIUM_WIDE",
    "MIN_SNR_FLOOR_DB",
    "NARROW",
    "PRB_BANDWIDTH_HZ",
    "REFERENCE_DISTANCE_M",
    "SNR_FLOOR_DB",
    "WIDE",
]


#: One LTE physical resource block: 12 subcarriers x 15 kHz (TS 36.104 s3.2).
PRB_BANDWIDTH_HZ: float = 180e3

#: The one carrier all contact classes share (``CHANNEL_FREQS_GHZ[0]``).
CONTACT_CARRIER_HZ: float = 3.32e9

#: CQI 1's SNR threshold: below it no MCS decodes (the default floor).
SNR_FLOOR_DB: float = -6.7

#: The lowest ``snr_floor_db`` a link accepts. It sits 20 dB below TR 36.942's
#: -10 dB, so a lower floor is taken for a units mistake (see "Other floors").
MIN_SNR_FLOOR_DB: float = -30.0

#: Reference distance of the log-distance law; nothing is evaluated closer.
REFERENCE_DISTANCE_M: float = 1.0

_SPEED_OF_LIGHT_M_S = 299_792_458.0
_THERMAL_NOISE_DBM_PER_HZ = -174.0
_LN2 = math.log(2.0)
# 10 ** (s / 10) overflows a float above about 3082 dB. From this SNR up,
# log2(1 + x) equals log2(x) to far better than double precision.
_SHANNON_ASYMPTOTE_DB = 3000.0


# --------------------------------------------------------------------------- #
# Helpers (defined first: the module-level band classes validate with them)
# --------------------------------------------------------------------------- #


def _real(value: Any, what: str) -> float:
    if type(value) is float:
        return value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{what} must be a real number, got {value!r}")
    return float(value)


def _finite(value: Any, what: str) -> float:
    x = _real(value, what)
    if not math.isfinite(x):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return x


def _snr(value: Any) -> float:
    # +/- infinity is a valid SNR (it prices as the peak or as unreachable);
    # NaN is not, since it would silently compare as "below the floor".
    s = _real(value, "snr_db")
    if math.isnan(s):
        raise ValueError("snr_db is NaN")
    return s


def _distance(value: Any) -> float:
    d = _real(value, "d_planar")
    if not math.isfinite(d) or d < 0.0:
        raise ValueError(f"d_planar must be finite and >= 0, got {value!r}")
    return d


def _shannon_se(snr_db: float) -> float:
    """log2(1 + SNR) in bit/s/Hz, the spec's expression, without overflow.

    It is computed as log1p(SNR) / ln 2. The literal log2(1.0 + SNR) rounds
    1 + SNR first, which quantizes the result at low SNR and makes it 0 once
    SNR < 2^-53 (about -160 dB). A capped rate could then be 0 at or above
    the floor, or exceed capacity.
    """
    if snr_db > _SHANNON_ASYMPTOTE_DB:
        return snr_db / 10.0 * math.log2(10.0)
    return math.log1p(10.0 ** (snr_db / 10.0)) / _LN2


def _fspl_db_at_1m(carrier_hz: float) -> float:
    """Free-space path loss at 1 m: 20 log10(4 pi f / c)."""
    return 20.0 * math.log10(4.0 * math.pi * carrier_hz / _SPEED_OF_LIGHT_M_S)


# --------------------------------------------------------------------------- #
# CQI table
# --------------------------------------------------------------------------- #


class CQIEntry(NamedTuple):
    """One row of TS 36.213 Table 7.2.3-1 with its AERPAW SNR threshold."""

    cqi: int
    modulation: str
    code_rate_x1024: int
    efficiency: float  # bits/s/Hz
    snr_threshold_db: float  # lowest SNR at which this CQI is reported


#: TS 36.213 V8.8.0 Table 7.2.3-1 (CQI 0 = "out of range" is implicit), with
#: the SNR thresholds of the AERPAW digital twin (Hossen et al. 2025; see the
#: module docstring). A CQI applies from its threshold up, inclusive.
CQI_TABLE: Tuple[CQIEntry, ...] = (
    CQIEntry(1, "QPSK", 78, 0.1523, -6.7),
    CQIEntry(2, "QPSK", 120, 0.2344, -4.7),
    CQIEntry(3, "QPSK", 193, 0.3770, -2.3),
    CQIEntry(4, "QPSK", 308, 0.6016, 0.2),
    CQIEntry(5, "QPSK", 449, 0.8770, 2.4),
    CQIEntry(6, "QPSK", 602, 1.1758, 4.3),
    CQIEntry(7, "16QAM", 378, 1.4766, 5.9),
    CQIEntry(8, "16QAM", 490, 1.9141, 8.1),
    CQIEntry(9, "16QAM", 616, 2.4063, 10.3),
    CQIEntry(10, "64QAM", 466, 2.7305, 11.7),
    CQIEntry(11, "64QAM", 567, 3.3223, 14.1),
    CQIEntry(12, "64QAM", 666, 3.9023, 16.3),
    CQIEntry(13, "64QAM", 772, 4.5234, 18.7),
    CQIEntry(14, "64QAM", 873, 5.1152, 21.0),
    CQIEntry(15, "64QAM", 948, 5.5547, 22.7),
)

_CQI_THRESHOLDS_DB: Tuple[float, ...] = tuple(e.snr_threshold_db for e in CQI_TABLE)
# Index k holds CQI k's efficiency; index 0 is "out of range".
_CQI_EFFICIENCY: Tuple[float, ...] = (0.0,) + tuple(e.efficiency for e in CQI_TABLE)
_MAX_CQI = len(CQI_TABLE)


# --------------------------------------------------------------------------- #
# Band classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BandClass:
    """An LTE channel bandwidth on the contact carrier.

    ``bandwidth_hz`` is the nominal channel bandwidth and ``n_prb`` its
    resource blocks (TS 36.104 Table 5.6-1). ``kappa`` is the class's
    efficiency factor on the occupied bandwidth (see the module docstring).
    """

    name: str
    bandwidth_hz: float
    n_prb: int
    kappa: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"band class name must be a non-empty string, got {self.name!r}")
        if (
            isinstance(self.n_prb, bool)
            or not isinstance(self.n_prb, numbers.Integral)
            or self.n_prb < 1
        ):
            raise ValueError(
                f"band class {self.name!r}: n_prb must be a positive integer, got {self.n_prb!r}"
            )
        bandwidth = _finite(self.bandwidth_hz, f"band class {self.name!r} bandwidth_hz")
        kappa = _finite(self.kappa, f"band class {self.name!r} kappa")
        if bandwidth <= 0.0:
            raise ValueError(f"band class {self.name!r}: bandwidth_hz must be positive")
        if not 0.0 < kappa <= 1.0:
            raise ValueError(f"band class {self.name!r}: kappa must be in (0, 1], got {kappa}")
        object.__setattr__(self, "n_prb", int(self.n_prb))
        object.__setattr__(self, "bandwidth_hz", bandwidth)
        object.__setattr__(self, "kappa", kappa)
        # An LTE channel never occupies more than its nominal bandwidth, so this
        # also catches a bandwidth given in MHz instead of Hz.
        if self.occupied_hz > bandwidth:
            raise ValueError(
                f"band class {self.name!r}: {self.n_prb} PRB occupy {self.occupied_hz:g} Hz, "
                f"more than the {bandwidth:g} Hz channel (bandwidth_hz is in Hz)"
            )

    @property
    def occupied_hz(self) -> float:
        """Occupied bandwidth B_occ = N_RB x 180 kHz."""
        return self.n_prb * PRB_BANDWIDTH_HZ


WIDE = BandClass("wide", 20e6, 100, 0.754)
MEDIUM = BandClass("medium", 5e6, 25, 0.734)
NARROW = BandClass("narrow", 1.4e6, 6, 0.732)
#: The optional fourth class (10 MHz). It is not in the default set.
MEDIUM_WIDE = BandClass("medium_wide", 10e6, 50, 0.734)

#: The default class set. A class's position is its trace band index.
CLASSES: Tuple[str, ...] = ("wide", "medium", "narrow")
#: The four-class option. The 10 MHz class is appended so indices 0-2 keep
#: their meaning.
CLASSES_WITH_10MHZ: Tuple[str, ...] = CLASSES + ("medium_wide",)
#: Every predefined class by name (read-only).
BAND_CLASSES: Mapping[str, BandClass] = MappingProxyType(
    {c.name: c for c in (WIDE, MEDIUM, NARROW, MEDIUM_WIDE)}
)
DEFAULT_BAND_CLASSES: Tuple[BandClass, ...] = (WIDE, MEDIUM, NARROW)

#: A band argument: the class name, its index in the link's class set, or the
#: class itself.
BandRef = Union[BandClass, str, int]


# --------------------------------------------------------------------------- #
# The link
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ContactLink:
    """Range, mean SNR, rate and dwell for the contact band classes (D1).

    ``anchor_planar_m`` is the run's ``rf_range_m``. The anchor class (wide)
    reaches exactly that planar distance, so S3a's geometry at wide stays
    Exp 4's. The anchor has no default, so a link can never silently assume
    60 m.

    ``shadow_sigma_db`` must equal the sigma of the shadowing term X_j(t)
    that ``l1/channel_model.py`` adds to ``mean_snr_db``. Only then is R(b)
    the range with ``margin_quantile`` availability at the edge.

    ``classes`` may list ``BandClass`` objects or the names of predefined
    ones (``CLASSES_WITH_10MHZ`` selects the four-class option). A band
    argument ``b`` may be a class name (``"wide"``), its index in ``classes``
    (the trace's band index) or the ``BandClass`` itself.

    Instances are immutable, and every method is a pure function of its
    arguments, so one link can be shared across threads and arms.
    """

    anchor_planar_m: float
    snr_floor_db: float = SNR_FLOOR_DB
    altitude_m: float = 25.0
    n_pl: float = 2.2
    shadow_sigma_db: float = 4.0
    margin_quantile: float = 0.9
    classes: Tuple[BandClass, ...] = DEFAULT_BAND_CLASSES
    anchor_class: str = "wide"

    def __post_init__(self) -> None:
        anchor = _finite(self.anchor_planar_m, "anchor_planar_m")
        floor = _finite(self.snr_floor_db, "snr_floor_db")
        altitude = _finite(self.altitude_m, "altitude_m")
        n_pl = _finite(self.n_pl, "n_pl")
        sigma = _finite(self.shadow_sigma_db, "shadow_sigma_db")
        quantile = _finite(self.margin_quantile, "margin_quantile")
        if anchor <= 0.0:
            raise ValueError(f"anchor_planar_m must be positive, got {anchor}")
        if floor < MIN_SNR_FLOOR_DB:
            raise ValueError(
                f"snr_floor_db must be >= {MIN_SNR_FLOOR_DB} dB, got {floor} "
                f"(the default is {SNR_FLOOR_DB} dB)"
            )
        if altitude < 0.0:
            raise ValueError(f"altitude_m must be >= 0, got {altitude}")
        if n_pl <= 0.0:
            raise ValueError(f"n_pl must be positive, got {n_pl}")
        if sigma < 0.0:
            raise ValueError(f"shadow_sigma_db must be >= 0, got {sigma}")
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"margin_quantile must be in (0, 1), got {quantile}")

        if isinstance(self.classes, (str, BandClass)):
            raise TypeError("classes must be a sequence of band classes or class names")
        classes = tuple(_as_band_class(c) for c in self.classes)
        if not classes:
            raise ValueError("classes must not be empty")
        names = tuple(c.name for c in classes)
        if len(set(names)) != len(names):
            raise ValueError(f"band class names must be unique, got {names}")
        if self.anchor_class not in names:
            raise ValueError(f"anchor_class {self.anchor_class!r} is not one of {names}")

        for attr, value in (
            ("anchor_planar_m", anchor),
            ("snr_floor_db", floor),
            ("altitude_m", altitude),
            ("n_pl", n_pl),
            ("shadow_sigma_db", sigma),
            ("margin_quantile", quantile),
            ("classes", classes),
        ):
            object.__setattr__(self, attr, value)

        margin = NormalDist().inv_cdf(quantile) * sigma
        anchor_index = names.index(self.anchor_class)
        anchor_occupied = classes[anchor_index].occupied_hz
        anchor_slant = math.hypot(anchor, altitude)
        slant: List[float] = []
        planar: List[float] = []
        rate_steps: List[Tuple[float, ...]] = []
        for i, cls in enumerate(classes):
            if i == anchor_index:
                reach, reach_planar = anchor_slant, anchor
            else:
                reach = anchor_slant * (anchor_occupied / cls.occupied_hz) ** (1.0 / n_pl)
                if reach <= altitude:
                    raise ValueError(
                        f"band class {cls.name!r} reaches {reach:.3f} m slant, which does not "
                        f"exceed the {altitude} m altitude, so it covers no ground device"
                    )
                reach_planar = math.sqrt(reach * reach - altitude * altitude)
            slant.append(reach)
            planar.append(reach_planar)
            peak = cls.kappa * cls.occupied_hz
            rate_steps.append(tuple(peak * se for se in _CQI_EFFICIENCY))

        # Derived values: plain attributes, not fields, so equality, hashing,
        # repr and dataclasses.replace() see only the parameters.
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_index", {name: i for i, name in enumerate(names)})
        object.__setattr__(self, "_anchor_index", anchor_index)
        object.__setattr__(self, "_margin_db", margin)
        object.__setattr__(self, "_edge_snr_db", floor + margin)
        object.__setattr__(self, "_slant", tuple(slant))
        object.__setattr__(self, "_planar", tuple(planar))
        object.__setattr__(self, "_rate_steps", tuple(rate_steps))

    # ------------------------------------------------------------------ #
    # Classes
    # ------------------------------------------------------------------ #

    @property
    def names(self) -> Tuple[str, ...]:
        """Class names in index order."""
        return self._names

    def index(self, b: BandRef) -> int:
        """The band index of ``b`` in this link's class set."""
        if isinstance(b, BandClass):
            i = self._index.get(b.name)
            if i is None or self.classes[i] != b:
                raise ValueError(f"{b!r} is not one of this link's classes {self._names}")
            return i
        if isinstance(b, str):
            i = self._index.get(b)
            if i is None:
                raise ValueError(f"unknown band class {b!r}; this link has {self._names}")
            return i
        if isinstance(b, bool) or not isinstance(b, numbers.Integral):
            raise TypeError(f"a band is a class name, index or BandClass, got {b!r}")
        i = int(b)
        if not 0 <= i < len(self._names):
            raise ValueError(f"band index {i} is out of range for {self._names}")
        return i

    def band(self, b: BandRef) -> BandClass:
        """The ``BandClass`` that ``b`` names."""
        return self.classes[self.index(b)]

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #

    @property
    def shadow_margin_db(self) -> float:
        """M_sh = Phi^-1(margin_quantile) * shadow_sigma_db (5.13 dB by default)."""
        return self._margin_db

    @property
    def edge_snr_db(self) -> float:
        """The mean SNR at every class's R(b): floor + M_sh."""
        return self._edge_snr_db

    def slant_m(self, d_planar: float) -> float:
        """Mule-to-device distance for a planar offset, at the link altitude."""
        return math.hypot(_distance(d_planar), self.altitude_m)

    def range_m(self, b: BandRef) -> float:
        """R(b), the slant range with ``margin_quantile`` edge availability.

        This is a 3D distance. Gates on planar distances (S3a's radius, and
        the ``range_m`` of ``ContactWaypoint`` and ``FerryPhysics``, which
        mean R_planar(b)) take ``range_planar_m``; this value would admit
        members up to R(b) - R_planar(b) too far (5 m at wide by default).
        """
        return self._slant[self.index(b)]

    def range_planar_m(self, b: BandRef) -> float:
        """R(b) on the ground; exactly ``anchor_planar_m`` for the anchor class.

        The anchor is returned as given rather than as
        sqrt(hypot(a, h)^2 - h^2), which is not always exactly ``a`` in
        floating point. S3a's radius at wide therefore equals Exp 4's
        ``rf_range_m`` bit for bit.
        """
        return self._planar[self.index(b)]

    def in_range(self, b: BandRef, d_planar: float) -> bool:
        """True iff a device ``d_planar`` metres from the stop is within R_planar(b).

        The comparison is inclusive, as S3a's membership test is.
        """
        return _distance(d_planar) <= self._planar[self.index(b)]

    def floor_range_m(self, b: BandRef) -> float:
        """Slant distance at which the MEAN SNR reaches the floor (availability 0.5).

        This is the plan's literal "range at the rate floor", which the model
        does not use as R(b) (critic A8-i): R_b * 10^(M_sh / (10 n)).
        """
        return self._slant[self.index(b)] * 10.0 ** (self._margin_db / (10.0 * self.n_pl))

    # ------------------------------------------------------------------ #
    # SNR, rate, dwell
    # ------------------------------------------------------------------ #

    def mean_snr_db(self, b: BandRef, d_planar: float) -> float:
        """Mean SNR on class ``b`` for a device ``d_planar`` metres from the stop.

        No shadowing and no interference: the channel model adds those. The
        slant distance is evaluated at no less than 1 m, the reference
        distance of the path-loss law.
        """
        i = self.index(b)
        d3d = max(self.slant_m(d_planar), REFERENCE_DISTANCE_M)
        return self._edge_snr_db + 10.0 * self.n_pl * math.log10(self._slant[i] / d3d)

    def above_floor(self, snr_db: float) -> bool:
        """True iff ``snr_db`` is at or above the floor (the floor is inclusive)."""
        return _snr(snr_db) >= self.snr_floor_db

    def cqi(self, snr_db: float) -> int:
        """The CQI that ``snr_db`` supports: 0 below the floor, otherwise 1-15."""
        s = _snr(snr_db)
        if s < self.snr_floor_db:
            return 0
        return self._cqi_above_floor(s)

    def rate_bps(self, b: BandRef, snr_db: float) -> float:
        """Rate in bit/s on class ``b`` at ``snr_db``; 0.0 below the floor."""
        i = self.index(b)
        s = _snr(snr_db)
        if s < self.snr_floor_db:
            return 0.0
        stepped = self._rate_steps[i][self._cqi_above_floor(s)]
        return min(stepped, self.classes[i].occupied_hz * _shannon_se(s))

    def dwell_s(self, nbytes: float, b: BandRef, snr_db: float) -> Optional[float]:
        """Hover seconds to move ``nbytes`` over class ``b`` at ``snr_db``.

        Returns ``8 * nbytes / rate_bps(b, snr_db)``, or None when the rate
        is 0 (below the floor). None means the member is unreachable, and
        callers must not charge it (critic B12; see the module docstring).
        The result is never infinite.
        """
        size = _real(nbytes, "nbytes")
        if not math.isfinite(size) or size < 0.0:
            raise ValueError(f"nbytes must be finite and >= 0, got {nbytes!r}")
        rate = self.rate_bps(b, snr_db)
        if rate <= 0.0:
            return None
        dwell = 8.0 * size / rate
        if math.isinf(dwell):
            raise OverflowError(f"dwell for {size:g} B at {rate:g} bit/s overflows")
        return dwell

    def _cqi_above_floor(self, s: float) -> int:
        # With a floor below CQI 1's threshold, CQI 1 applies from the floor
        # up, so rate > 0 exactly when s >= floor.
        return max(1, bisect.bisect_right(_CQI_THRESHOLDS_DB, s))

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #

    def implied_eirp_dbm(
        self,
        b: Optional[BandRef] = None,
        *,
        noise_figure_db: float = 9.0,
        rx_gain_dbi: float = 0.0,
        carrier_hz: float = CONTACT_CARRIER_HZ,
    ) -> float:
        """The EIRP (dBm) that puts the mean SNR at floor + M_sh exactly at R(b).

        The budget uses free-space loss at 1 m on ``carrier_hz``, log-distance
        loss with exponent ``n_pl`` beyond 1 m, thermal noise of -174 dBm/Hz
        over B_occ plus ``noise_figure_db``, and ``rx_gain_dbi``. By
        construction the result is the same for every class, so ``b`` only
        selects which class to compute it from (default: the anchor). The
        defaults give about -11.3 dBm, the assumption the anchor implies
        (critic C6).
        """
        i = self._anchor_index if b is None else self.index(b)
        noise_dbm = (
            _THERMAL_NOISE_DBM_PER_HZ
            + 10.0 * math.log10(self.classes[i].occupied_hz)
            + _finite(noise_figure_db, "noise_figure_db")
        )
        path_loss_db = _fspl_db_at_1m(_finite(carrier_hz, "carrier_hz")) + (
            10.0 * self.n_pl * math.log10(self._slant[i] / REFERENCE_DISTANCE_M)
        )
        return self._edge_snr_db + noise_dbm + path_loss_db - _finite(rx_gain_dbi, "rx_gain_dbi")

    def describe(self) -> Dict[str, Any]:
        """A JSON-ready record of the link and its classes, for run provenance.

        Each class row names its slant range ``range_slant_m`` (``range_m()``)
        rather than ``range_m``: ``ContactWaypoint.range_m`` is the planar
        R_planar(b), and one key must not mean both in a trace.
        """
        classes = []
        for i, cls in enumerate(self.classes):
            classes.append(
                {
                    "index": i,
                    "name": cls.name,
                    "bandwidth_mhz": cls.bandwidth_hz / 1e6,
                    "n_prb": cls.n_prb,
                    "occupied_mhz": cls.occupied_hz / 1e6,
                    "kappa": cls.kappa,
                    "range_slant_m": self._slant[i],
                    "range_planar_m": self._planar[i],
                    "floor_rate_bps": self.rate_bps(i, self.snr_floor_db),
                    "peak_rate_bps": self._rate_steps[i][_MAX_CQI],
                }
            )
        return {
            "snr_floor_db": self.snr_floor_db,
            "altitude_m": self.altitude_m,
            "n_pl": self.n_pl,
            "shadow_sigma_db": self.shadow_sigma_db,
            "margin_quantile": self.margin_quantile,
            "shadow_margin_db": self._margin_db,
            "anchor_class": self.anchor_class,
            "anchor_planar_m": self.anchor_planar_m,
            "carrier_hz": CONTACT_CARRIER_HZ,
            "classes": classes,
        }


def _as_band_class(value: Any) -> BandClass:
    if isinstance(value, BandClass):
        return value
    if isinstance(value, str):
        try:
            return BAND_CLASSES[value]
        except KeyError:
            raise ValueError(
                f"unknown band class {value!r}; predefined: {tuple(BAND_CLASSES)}"
            ) from None
    raise TypeError(f"a band class is a BandClass or a predefined name, got {value!r}")
