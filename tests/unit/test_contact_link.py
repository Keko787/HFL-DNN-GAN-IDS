"""FeRRy Phase 3, unit U3: the contact link's band classes and range-rate model.

Pins design section 1 D1 (binding spec D1):
* the class table and the CQI and threshold tables, against their sources;
* the rate formula and its invariants: never above Shannon (the cap never
  binds on this table), monotone in SNR, zero below the floor;
* R(b): the formula, the D1 table values, and the exact anchor at wide;
* dwell: it scales with bytes/rate, and below the floor it is None, never
  infinity (critic B12);
* the documented deviations: the floor-rate range (critic A8-i) and the
  implied EIRP of the anchor (critic C6);
* the design's 37,576 B Pass-1 dwell numbers, as regression anchors;
* the review fixes: the ``describe()`` record at non-default parameters,
  slant ``range_m`` against the planar ground radius, the -30 dB floor bound
  and an accurate Shannon cap at low SNR.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import random
from pathlib import Path
from statistics import NormalDist

import pytest

from hermes.l1 import contact_link as cl
from hermes.l1.contact_link import (
    BAND_CLASSES,
    CLASSES,
    CLASSES_WITH_10MHZ,
    CQI_TABLE,
    DEFAULT_BAND_CLASSES,
    MEDIUM,
    MEDIUM_WIDE,
    NARROW,
    WIDE,
    BandClass,
    ContactLink,
)

#: Ledger bytes of one IDS Pass-1 session: push 18,820 + update 18,756 (design D3).
PASS1_SESSION_BYTES = 37_576
#: 1 MB each way (design D1's large-payload figures).
ONE_MB_EACH_WAY = 2 * 1_000_000

_FSPL_1M_DB = 20.0 * math.log10(4.0 * math.pi * 3.32e9 / 299_792_458.0)


def _link(**kwargs) -> ContactLink:
    kwargs.setdefault("anchor_planar_m", 60.0)
    return ContactLink(**kwargs)


def _shannon_bps(occupied_hz: float, snr_db: float) -> float:
    """Computed independently of the module's own helper."""
    return occupied_hz * math.log2(1.0 + 10.0 ** (snr_db / 10.0))


def _stepped_bps(cls: BandClass, cqi: int) -> float:
    return cls.kappa * cls.occupied_hz * CQI_TABLE[cqi - 1].efficiency


def _grid(lo: float, hi: float, step: float):
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def _noise_dbm(cls: BandClass, noise_figure_db: float = 9.0) -> float:
    return -174.0 + 10.0 * math.log10(cls.occupied_hz) + noise_figure_db


# --------------------------------------------------------------------------- #
# Band classes (D1)
# --------------------------------------------------------------------------- #


def test_default_classes_are_the_d1_set():
    assert CLASSES == ("wide", "medium", "narrow")
    assert DEFAULT_BAND_CLASSES == (WIDE, MEDIUM, NARROW)
    assert tuple(c.name for c in DEFAULT_BAND_CLASSES) == CLASSES
    assert (WIDE.bandwidth_hz, WIDE.n_prb, WIDE.kappa) == (20e6, 100, 0.754)
    assert (MEDIUM.bandwidth_hz, MEDIUM.n_prb, MEDIUM.kappa) == (5e6, 25, 0.734)
    assert (NARROW.bandwidth_hz, NARROW.n_prb, NARROW.kappa) == (1.4e6, 6, 0.732)
    link = _link()
    assert link.names == CLASSES
    # The position in CLASSES is the trace's band index.
    assert [link.index(name) for name in CLASSES] == [0, 1, 2]


def test_occupied_bandwidth_is_prb_count_times_180_khz():
    for cls in BAND_CLASSES.values():
        assert cls.occupied_hz == cls.n_prb * 180e3
    assert (WIDE.occupied_hz, MEDIUM.occupied_hz, NARROW.occupied_hz) == (18e6, 4.5e6, 1.08e6)
    assert MEDIUM_WIDE.occupied_hz == 9e6


def test_kappa_is_the_peak_tbs_over_the_occupied_bandwidth():
    # TS 36.213 V8.8.0 Table 7.1.7.2.1-1: I_TBS 26, single layer, bits per 1 ms TTI.
    peak_tbs_bits = {100: 75_376, 50: 36_696, 25: 18_336, 6: 4_392}
    for cls in BAND_CLASSES.values():
        kappa = peak_tbs_bits[cls.n_prb] * 1000 / (cls.occupied_hz * 5.5547)
        assert round(kappa, 3) == cls.kappa


def test_the_10_mhz_class_is_optional_and_appended():
    assert "medium_wide" not in CLASSES
    assert CLASSES_WITH_10MHZ == CLASSES + ("medium_wide",)
    assert (MEDIUM_WIDE.bandwidth_hz, MEDIUM_WIDE.n_prb, MEDIUM_WIDE.kappa) == (10e6, 50, 0.734)
    four = _link(classes=CLASSES_WITH_10MHZ)
    assert [four.index(name) for name in CLASSES] == [0, 1, 2]
    assert four.index("medium_wide") == 3
    # Design D1: the 10 MHz class reaches 89.1 / 85.5 m at n = 2.2.
    assert round(four.range_m("medium_wide"), 1) == 89.1
    assert round(four.range_planar_m("medium_wide"), 1) == 85.5
    with pytest.raises(ValueError):
        _link().index("medium_wide")


# --------------------------------------------------------------------------- #
# CQI table and thresholds, against their sources
# --------------------------------------------------------------------------- #


def test_cqi_table_is_ts_36213_table_7_2_3_1():
    assert [e.cqi for e in CQI_TABLE] == list(range(1, 16))
    assert [e.modulation for e in CQI_TABLE] == ["QPSK"] * 6 + ["16QAM"] * 3 + ["64QAM"] * 6
    assert [e.code_rate_x1024 for e in CQI_TABLE] == [
        78, 120, 193, 308, 449, 602, 378, 490, 616, 466, 567, 666, 772, 873, 948,
    ]
    bits_per_symbol = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
    for e in CQI_TABLE:
        # The efficiency column is the modulation order times the code rate,
        # printed to four decimals.
        exact = bits_per_symbol[e.modulation] * e.code_rate_x1024 / 1024
        assert abs(exact - e.efficiency) <= 0.5e-4 + 1e-12
    assert CQI_TABLE[0].efficiency == 0.1523 and CQI_TABLE[-1].efficiency == 5.5547


def test_snr_thresholds_are_the_aerpaw_digital_twin_table():
    assert [e.snr_threshold_db for e in CQI_TABLE] == [
        -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7,
    ]
    assert cl.SNR_FLOOR_DB == CQI_TABLE[0].snr_threshold_db == -6.7
    assert _link().snr_floor_db == -6.7


def test_cqi_steps_are_inclusive_at_each_threshold():
    link = _link()
    for e in CQI_TABLE:
        assert link.cqi(e.snr_threshold_db) == e.cqi
        assert link.cqi(e.snr_threshold_db + 0.05) == e.cqi
        assert link.cqi(math.nextafter(e.snr_threshold_db, -math.inf)) == e.cqi - 1


# --------------------------------------------------------------------------- #
# Rate
# --------------------------------------------------------------------------- #


def test_rate_is_kappa_times_occupied_bandwidth_times_cqi_efficiency():
    link = _link()
    for name in CLASSES:
        cls = link.band(name)
        for e in CQI_TABLE:
            for snr in (e.snr_threshold_db, e.snr_threshold_db + 0.05):
                assert link.rate_bps(name, snr) == pytest.approx(_stepped_bps(cls, e.cqi), rel=1e-12)


def test_rate_never_exceeds_shannon_and_the_cap_never_binds_on_this_table():
    # The worst case inside a CQI step is its threshold, where Shannon is
    # lowest, so the thresholds are in the grid. Critic: kappa * SE / Shannon
    # stays at or below 0.555.
    link = _link(classes=CLASSES_WITH_10MHZ)
    grid = _grid(-6.7, 40.0, 0.01) + [e.snr_threshold_db for e in CQI_TABLE]
    worst = 0.0
    for name in link.names:
        cls = link.band(name)
        for snr in grid:
            rate = link.rate_bps(name, snr)
            shannon = _shannon_bps(cls.occupied_hz, snr)
            assert rate <= shannon
            # The min() never picks Shannon: the rate is the stepped CQI rate.
            assert rate == pytest.approx(_stepped_bps(cls, link.cqi(snr)), rel=1e-12)
            worst = max(worst, rate / shannon)
    assert worst < 0.56


def test_the_shannon_cap_is_live_below_about_minus_10_8_db():
    # A floor below CQI 1's threshold extends CQI 1 down to the floor; the cap
    # then keeps the rate under capacity.
    wide = WIDE
    stepped = _stepped_bps(wide, 1)
    low = _link(snr_floor_db=-15.0)
    assert low.rate_bps("wide", -14.0) == pytest.approx(_shannon_bps(wide.occupied_hz, -14.0), rel=1e-12)
    assert low.rate_bps("wide", -14.0) < stepped
    assert low.rate_bps("wide", -10.7) == pytest.approx(stepped, rel=1e-12)
    # At TR 36.942's -10 dB floor the cap still does not bind.
    ten = _link(snr_floor_db=-10.0)
    assert ten.cqi(-10.0) == 1
    assert ten.rate_bps("wide", -10.0) == pytest.approx(stepped, rel=1e-12)
    assert ten.rate_bps("wide", -10.0) < _shannon_bps(wide.occupied_hz, -10.0)


@pytest.mark.parametrize("floor", [-6.7, -10.0, -15.0])
def test_rate_is_monotone_non_decreasing_in_snr(floor):
    link = _link(snr_floor_db=floor, classes=CLASSES_WITH_10MHZ)
    edges = [e.snr_threshold_db for e in CQI_TABLE] + [floor]
    grid = sorted(
        set(
            _grid(-30.0, 45.0, 0.01)
            + edges
            + [math.nextafter(t, -math.inf) for t in edges]
            + [math.nextafter(t, math.inf) for t in edges]
            + [-math.inf, math.inf, 1e6]
        )
    )
    for name in link.names:
        rates = [link.rate_bps(name, snr) for snr in grid]
        assert all(a <= b for a, b in zip(rates, rates[1:]))
        assert rates[-1] == pytest.approx(_stepped_bps(link.band(name), 15), rel=1e-12)


@pytest.mark.parametrize("floor", [-6.7, -10.0, -5.0])
def test_rate_is_zero_below_the_floor_and_positive_from_it(floor):
    link = _link(snr_floor_db=floor)
    for name in CLASSES:
        for snr in (-math.inf, -60.0, floor - 1.0, math.nextafter(floor, -math.inf)):
            assert link.rate_bps(name, snr) == 0.0
            assert link.cqi(snr) == 0
            assert not link.above_floor(snr)
        # The floor itself is inclusive.
        assert link.rate_bps(name, floor) > 0.0
        assert link.above_floor(floor)
        assert link.cqi(floor) >= 1


@pytest.mark.parametrize("floor", [-30.0, -20.0])
def test_a_floor_far_below_cqi_1_prices_capacity_from_the_floor_up(floor):
    """Down to the lowest accepted floor, the capped rate is positive from the
    floor up, never above capacity, and its dwell is finite."""
    link = _link(snr_floor_db=floor, classes=CLASSES_WITH_10MHZ)
    below = math.nextafter(floor, -math.inf)
    for name in link.names:
        occupied = link.band(name).occupied_hz
        # The cap binds below about -10.95 dB on every class.
        for snr in (floor, math.nextafter(floor, math.inf), floor + 0.5, -15.0, -11.0):
            rate = link.rate_bps(name, snr)
            assert rate > 0.0 and link.above_floor(snr)
            assert rate == pytest.approx(_shannon_bps(occupied, snr), rel=1e-12)
            dwell = link.dwell_s(PASS1_SESSION_BYTES, name, snr)
            assert dwell is not None and math.isfinite(dwell) and dwell > 0.0
        assert link.rate_bps(name, below) == 0.0
        assert link.dwell_s(PASS1_SESSION_BYTES, name, below) is None


def test_floors_below_minus_30_db_are_refused():
    """A floor below MIN_SNR_FLOOR_DB is taken for a units mistake."""
    assert cl.MIN_SNR_FLOOR_DB == -30.0
    assert _link(snr_floor_db=-30.0).snr_floor_db == -30.0
    for floor in (math.nextafter(-30.0, -math.inf), -67.0, -160.0, -math.inf):
        with pytest.raises(ValueError):
            _link(snr_floor_db=floor)


def test_the_shannon_cap_is_accurate_at_low_snr():
    """log2(1.0 + x) rounds 1 + x first, so it is quantized from about -150 dB
    and 0 below about -160 dB. The cap uses log1p, which is accurate there."""
    for snr_db in (-200.0, -160.0, -157.0, -100.0, -30.0, -10.0):
        x = 10.0 ** (snr_db / 10.0)
        ln_1_plus_x = sum((-1) ** (k + 1) * x ** k / k for k in range(1, 40))
        # abs=0: approx's default absolute tolerance (1e-12) would pass a 0.
        expected = ln_1_plus_x / math.log(2.0)
        assert cl._shannon_se(snr_db) == pytest.approx(expected, rel=1e-13, abs=0.0)
    for snr_db in (0.0, 22.7, 60.0):
        exact = math.log2(1.0 + 10.0 ** (snr_db / 10.0))
        assert cl._shannon_se(snr_db) == pytest.approx(exact, rel=1e-14)


def test_d1_table_floor_and_peak_rates():
    link = _link(classes=CLASSES_WITH_10MHZ)
    expected_mbps = {
        "wide": (2.07, 75.4),
        "medium": (0.50, 18.3),
        "narrow": (0.12, 4.39),
        "medium_wide": (1.01, 36.7),
    }
    for name, (floor_mbps, peak_mbps) in expected_mbps.items():
        assert round(link.rate_bps(name, -6.7) / 1e6, 2) == floor_mbps
        peak = link.rate_bps(name, 22.7)
        assert peak == link.rate_bps(name, 60.0) == link.rate_bps(name, math.inf)
        assert round(peak / 1e6, 1 if peak > 10e6 else 2) == peak_mbps


# --------------------------------------------------------------------------- #
# Range R(b)
# --------------------------------------------------------------------------- #


def test_range_formula():
    for n_pl in (2.2, 3.0, 1.7):
        for altitude in (0.0, 25.0, 60.0):
            link = _link(n_pl=n_pl, altitude_m=altitude, classes=CLASSES_WITH_10MHZ)
            r_wide = math.hypot(60.0, altitude)
            assert link.range_m("wide") == pytest.approx(r_wide, rel=1e-15)
            for name in link.names:
                cls = link.band(name)
                r_b = r_wide * (WIDE.occupied_hz / cls.occupied_hz) ** (1.0 / n_pl)
                assert link.range_m(name) == pytest.approx(r_b, rel=1e-12)
                assert link.range_planar_m(name) == pytest.approx(
                    math.sqrt(r_b * r_b - altitude * altitude), rel=1e-12
                )
            # Narrower classes reach further.
            reach = [link.range_m(n) for n in ("wide", "medium_wide", "medium", "narrow")]
            assert reach == sorted(reach) and len(set(reach)) == 4


@pytest.mark.parametrize(
    "n_pl, expected",
    [
        (2.2, {"wide": (65.0, 60.0), "medium": (122.1, 119.5), "narrow": (233.5, 232.2)}),
        (3.0, {"wide": (65.0, 60.0), "medium": (103.2, 100.1), "narrow": (166.0, 164.1)}),
    ],
)
def test_d1_table_ranges(n_pl, expected):
    link = _link(n_pl=n_pl)
    for name, (slant, planar) in expected.items():
        assert round(link.range_m(name), 1) == slant
        assert round(link.range_planar_m(name), 1) == planar


def test_range_planar_of_wide_is_the_anchor_exactly():
    rng = random.Random(3)
    anchors = [60.0, 60, 50.0, 37.3, 45.0, 0.5, 1000.0] + [rng.uniform(1.0, 300.0) for _ in range(300)]
    naive_inexact = 0
    for anchor in anchors:
        for altitude in (0.0, 25.0, 30.0, 70.0):
            link = ContactLink(anchor_planar_m=anchor, altitude_m=altitude)
            assert link.range_planar_m("wide") == anchor  # exact, not approximate
            assert link.in_range("wide", anchor)
            naive = math.sqrt(math.hypot(anchor, altitude) ** 2 - altitude * altitude)
            naive_inexact += naive != anchor
    # Why the anchor is returned as given: the round trip is not exact.
    assert naive_inexact > 0


def test_a_non_wide_anchor_is_exact_on_its_own_class():
    link = _link(anchor_planar_m=100.0, anchor_class="medium")
    assert link.range_planar_m("medium") == 100.0
    assert link.range_m("wide") < link.range_m("medium") < link.range_m("narrow")


def test_in_range_is_inclusive_like_s3a():
    link = _link()
    assert link.in_range("wide", 60.0)
    assert not link.in_range("wide", math.nextafter(60.0, math.inf))
    assert link.in_range("narrow", 200.0) and not link.in_range("medium", 200.0)
    assert link.slant_m(60.0) == 65.0


def test_range_m_is_slant_and_range_planar_m_is_the_ground_radius():
    """range_m is the 3D R(b). Planar gates (S3a, and the range_m of
    ContactWaypoint and FerryPhysics) must take range_planar_m."""
    link = _link(classes=CLASSES_WITH_10MHZ)
    assert (link.range_m("wide"), link.range_planar_m("wide")) == (65.0, 60.0)
    for name in link.names:
        planar, slant = link.range_planar_m(name), link.range_m(name)
        assert link.slant_m(planar) == pytest.approx(slant, rel=1e-12)
        # A member just past the ground radius is out of range, although its
        # planar distance is still short of the slant range.
        beyond = math.nextafter(planar, math.inf)
        assert not link.in_range(name, beyond) and beyond < slant


# --------------------------------------------------------------------------- #
# Mean SNR
# --------------------------------------------------------------------------- #


def test_mean_snr_at_the_edge_is_the_floor_plus_the_shadow_margin():
    link = _link(classes=CLASSES_WITH_10MHZ)
    # Design D1: M_sh = 1.2816 * sigma_sh = 5.13 dB.
    assert link.shadow_margin_db == pytest.approx(1.2816 * 4.0, abs=1e-3)
    assert link.edge_snr_db == pytest.approx(-6.7 + link.shadow_margin_db, abs=1e-12)
    for name in link.names:
        assert link.mean_snr_db(name, link.range_planar_m(name)) == pytest.approx(link.edge_snr_db, abs=1e-9)
    # With shadowing N(0, sigma^2), the edge stays above the floor 90 % of the time.
    availability = 1.0 - NormalDist(link.edge_snr_db, link.shadow_sigma_db).cdf(link.snr_floor_db)
    assert availability == pytest.approx(0.9, abs=1e-12)


def test_mean_snr_formula_and_monotone_in_distance():
    link = _link()
    for name in CLASSES:
        previous = math.inf
        for i in range(3001):
            d_planar = i * 0.1
            d3d = math.hypot(d_planar, 25.0)
            expected = -6.7 + link.shadow_margin_db + 10.0 * 2.2 * math.log10(link.range_m(name) / d3d)
            snr = link.mean_snr_db(name, d_planar)
            assert snr == pytest.approx(expected, abs=1e-9)
            assert snr < previous
            previous = snr


def test_distances_below_one_metre_are_evaluated_at_one_metre():
    link = _link(altitude_m=0.0)
    at_one_metre = link.mean_snr_db("wide", 1.0)
    assert link.mean_snr_db("wide", 0.0) == at_one_metre
    assert link.mean_snr_db("wide", 0.4) == at_one_metre
    assert math.isfinite(at_one_metre)
    assert link.slant_m(0.4) == 0.4  # the geometry itself is not clamped


def test_margin_follows_the_quantile_and_sigma():
    assert _link(margin_quantile=0.5).shadow_margin_db == pytest.approx(0.0, abs=1e-12)
    assert _link(shadow_sigma_db=0.0).shadow_margin_db == 0.0
    assert _link(margin_quantile=0.95).shadow_margin_db == pytest.approx(1.6449 * 4.0, abs=1e-3)
    median_edge = _link(margin_quantile=0.5)
    assert median_edge.floor_range_m("wide") == pytest.approx(median_edge.range_m("wide"), rel=1e-12)
    # R(b) does not depend on the margin; the mean SNR does.
    assert _link(margin_quantile=0.95).range_m("narrow") == _link().range_m("narrow")


def test_a_floor_sweep_at_a_fixed_anchor_shifts_every_mean_snr():
    # Documented: at a fixed anchor, lowering the floor is an EIRP sweep.
    base, low = _link(), _link(snr_floor_db=-10.0)
    for name in CLASSES:
        assert low.range_m(name) == base.range_m(name)
        for d_planar in (0.0, 30.0, 60.0):
            assert low.mean_snr_db(name, d_planar) == pytest.approx(
                base.mean_snr_db(name, d_planar) - 3.3, abs=1e-9
            )
    assert low.implied_eirp_dbm() == pytest.approx(base.implied_eirp_dbm() - 3.3, abs=1e-9)


# --------------------------------------------------------------------------- #
# Dwell
# --------------------------------------------------------------------------- #


def test_dwell_is_eight_times_bytes_over_rate():
    link = _link(classes=CLASSES_WITH_10MHZ)
    for name in link.names:
        for snr in (-6.7, -2.0, 3.3, 10.0, 25.0):
            rate = link.rate_bps(name, snr)
            for nbytes in (0, 1, 52, 18_820, PASS1_SESSION_BYTES, 1_000_000, 10_000_000, 2.5e6):
                assert link.dwell_s(nbytes, name, snr) == pytest.approx(8 * nbytes / rate, rel=1e-15)


def test_dwell_scales_with_bytes_over_rate():
    """The plan's Phase 3 test: dwell scales with bytes/rate."""
    link = _link()
    for name in CLASSES:
        for d_planar in (0.0, 30.0, 60.0):
            snr = link.mean_snr_db(name, d_planar)
            base = link.dwell_s(PASS1_SESSION_BYTES, name, snr)
            for k in (2, 10, 1000):
                assert link.dwell_s(k * PASS1_SESSION_BYTES, name, snr) == pytest.approx(k * base, rel=1e-12)
        # At fixed bytes the dwell is inversely proportional to the rate.
        near, far = link.mean_snr_db(name, 0.0), link.mean_snr_db(name, 60.0)
        ratio = link.dwell_s(1e6, name, far) / link.dwell_s(1e6, name, near)
        assert ratio == pytest.approx(link.rate_bps(name, near) / link.rate_bps(name, far), rel=1e-12)
        # More SNR never lengthens a dwell.
        dwells = [link.dwell_s(PASS1_SESSION_BYTES, name, snr) for snr in _grid(-6.7, 30.0, 0.05)]
        assert all(a >= b for a, b in zip(dwells, dwells[1:]))


def test_dwell_below_the_floor_is_none_never_infinity():
    """Critic B12: a member below the floor is unreachable and is never charged."""
    link = _link()
    for name in CLASSES:
        for snr in (-math.inf, -30.0, math.nextafter(-6.7, -math.inf)):
            assert link.dwell_s(PASS1_SESSION_BYTES, name, snr) is None
            # Unreachable even with nothing to send.
            assert link.dwell_s(0, name, snr) is None
        # Above the floor a zero-byte dwell is 0.0, which is falsy: callers test "is None".
        assert link.dwell_s(0, name, -6.7) == 0.0
        for snr in (-6.7, 0.0, 50.0, math.inf):
            assert math.isfinite(link.dwell_s(10_000_000, name, snr))


# --------------------------------------------------------------------------- #
# Regression anchors: design section 1 D1 (h = 25 m, n = 2.2)
# --------------------------------------------------------------------------- #

# (class, planar distance m, CQI of the mean SNR, rate Mb/s to 0.1,
#  Pass-1 dwell s to 0.001, 1 MB each way s to 0.1). The design's prose gives
# wide 20.0 -> 5.1 Mb/s, medium 9.0 -> 3.9, narrow 3.6 -> 1.9 over 0-60 m;
# Pass-1 dwell 0.015-0.06 s on wide and 0.08-0.16 s on narrow; 1 MB each way
# 0.8-3.1 s on wide and 4.5-8.4 s on narrow. The medium dwell figures and the
# extra digits come from the design's probe (design_numbers.py).
_D1_ANCHORS = [
    ("wide", 0.0, 7, 20.0, 0.015, 0.8),
    ("wide", 60.0, 3, 5.1, 0.059, 3.1),
    ("medium", 0.0, 10, 9.0, 0.033, 1.8),
    ("medium", 60.0, 6, 3.9, 0.077, 4.1),
    ("narrow", 0.0, 13, 3.6, 0.084, 4.5),
    ("narrow", 60.0, 9, 1.9, 0.158, 8.4),
]


@pytest.mark.parametrize("name, d_planar, cqi, rate_mbps, pass1_s, one_mb_s", _D1_ANCHORS)
def test_design_d1_dwell_anchors(name, d_planar, cqi, rate_mbps, pass1_s, one_mb_s):
    link = _link()
    snr = link.mean_snr_db(name, d_planar)
    assert link.cqi(snr) == cqi
    rate = link.rate_bps(name, snr)
    assert rate == pytest.approx(_stepped_bps(link.band(name), cqi), rel=1e-12)
    assert round(rate / 1e6, 1) == rate_mbps
    pass1 = link.dwell_s(PASS1_SESSION_BYTES, name, snr)
    assert pass1 == pytest.approx(8 * PASS1_SESSION_BYTES / rate, rel=1e-12)
    assert round(pass1, 3) == pass1_s
    assert round(link.dwell_s(ONE_MB_EACH_WAY, name, snr), 1) == one_mb_s


def test_design_d1_narrow_at_200_m_and_no_peak_at_altitude():
    link = _link()
    snr = link.mean_snr_db("narrow", 200.0)
    assert link.cqi(snr) == 3
    assert round(link.dwell_s(ONE_MB_EACH_WAY, "narrow", snr), 1) == 53.7
    # No class reaches CQI 15 at 25 m, even directly overhead.
    assert all(link.cqi(link.mean_snr_db(name, 0.0)) < 15 for name in CLASSES)


# --------------------------------------------------------------------------- #
# Documented deviations: critic A8-i and C6
# --------------------------------------------------------------------------- #


def test_floor_rate_range_is_about_111_m_slant_for_wide():
    """A8-i: R(b) is the 90 % edge range; the floor-rate range is 1.71 x longer."""
    link = _link()
    floor_range = link.floor_range_m("wide")
    assert round(floor_range) == 111
    assert floor_range / link.range_m("wide") == pytest.approx(1.71, abs=0.005)
    # The mean SNR reaches the floor exactly there...
    d_planar = math.sqrt(floor_range ** 2 - 25.0 ** 2)
    assert link.mean_snr_db("wide", d_planar) == pytest.approx(-6.7, abs=1e-9)
    # ...while at R(b) itself wide runs at CQI 3 (5.12 Mb/s), not the floor rate.
    assert link.cqi(link.edge_snr_db) == 3
    assert round(link.rate_bps("wide", link.edge_snr_db) / 1e6, 2) == 5.12


def test_implied_eirp_of_the_anchor():
    """C6: the anchor implies about -11.3 dBm EIRP; the research budget reaches 211 m."""
    link = _link(classes=CLASSES_WITH_10MHZ)
    eirp = link.implied_eirp_dbm()
    assert round(eirp, 1) == -11.3
    for name in link.names:
        # The classes share one EIRP.
        assert link.implied_eirp_dbm(name) == pytest.approx(eirp, abs=1e-9)
    assert round(_link(n_pl=3.0).implied_eirp_dbm(), 1) == 3.2
    # An independent budget: the implied EIRP puts wide's edge at 65 m slant...
    noise = _noise_dbm(WIDE)
    max_path_loss = eirp - noise - link.edge_snr_db
    assert 10 ** ((max_path_loss - _FSPL_1M_DB) / 22.0) == pytest.approx(65.0, rel=1e-12)
    # ...and the research's low-altitude budget (0 dBm, 0 dBi, NF 9 dB, the same
    # margin) reaches 211 m.
    assert round(10 ** ((0.0 - noise - link.edge_snr_db - _FSPL_1M_DB) / 22.0)) == 211


def test_mean_snr_is_the_equal_eirp_link_budget():
    link = _link(classes=CLASSES_WITH_10MHZ)
    eirp = link.implied_eirp_dbm()
    for name in link.names:
        noise = _noise_dbm(link.band(name))
        for d_planar in (0.0, 10.0, 60.0, 150.0):
            path_loss = _FSPL_1M_DB + 22.0 * math.log10(math.hypot(d_planar, 25.0))
            assert link.mean_snr_db(name, d_planar) == pytest.approx(eirp - path_loss - noise, abs=1e-9)


# --------------------------------------------------------------------------- #
# Invariants over random parameters
# --------------------------------------------------------------------------- #


def test_invariants_hold_over_random_parameters():
    rng = random.Random(20260929)
    for _ in range(200):
        link = ContactLink(
            anchor_planar_m=rng.uniform(5.0, 300.0),
            snr_floor_db=rng.uniform(cl.MIN_SNR_FLOOR_DB, -3.0),
            altitude_m=rng.uniform(0.0, 120.0),
            n_pl=rng.uniform(1.6, 4.0),
            shadow_sigma_db=rng.uniform(0.0, 8.0),
            margin_quantile=rng.uniform(0.5, 0.99),
            classes=CLASSES_WITH_10MHZ,
        )
        assert link.range_planar_m("wide") == link.anchor_planar_m
        eirp = link.implied_eirp_dbm()
        for name in link.names:
            occupied = link.band(name).occupied_hz
            assert link.implied_eirp_dbm(name) == pytest.approx(eirp, abs=1e-6)
            edge_snr = link.mean_snr_db(name, link.range_planar_m(name))
            assert edge_snr == pytest.approx(link.edge_snr_db, abs=1e-6)
            snrs = sorted(rng.uniform(-35.0, 40.0) for _ in range(50))
            rates = [link.rate_bps(name, snr) for snr in snrs]
            assert all(a <= b for a, b in zip(rates, rates[1:]))
            for snr, rate in zip(snrs, rates):
                # Floors below about -10.8 dB let the cap bind; there the rate
                # IS Shannon, so allow for rounding in how it is computed.
                assert rate <= _shannon_bps(occupied, snr) * (1.0 + 1e-12)
                assert (rate > 0.0) == link.above_floor(snr)
                dwell = link.dwell_s(PASS1_SESSION_BYTES, name, snr)
                assert (dwell is None) == (rate == 0.0)
                assert dwell is None or math.isfinite(dwell)


# --------------------------------------------------------------------------- #
# Band arguments and validation
# --------------------------------------------------------------------------- #


def test_band_arguments_accept_a_name_an_index_or_the_class():
    link = _link()
    for i, name in enumerate(CLASSES):
        cls = link.band(name)
        assert link.band(i) is cls and link.band(cls) is cls
        assert link.index(name) == link.index(i) == link.index(cls) == i
        assert link.range_m(name) == link.range_m(i) == link.range_m(cls)


@pytest.mark.parametrize(
    "band, error",
    [
        ("ultra", ValueError),
        (3, ValueError),
        (-1, ValueError),
        (BandClass("wide", 20e6, 100, 0.7), ValueError),  # not this link's wide
        (True, TypeError),
        (1.0, TypeError),
        (None, TypeError),
    ],
)
def test_unknown_bands_are_refused(band, error):
    with pytest.raises(error):
        _link().range_m(band)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anchor_planar_m": 0.0},
        {"anchor_planar_m": -5.0},
        {"anchor_planar_m": math.inf},
        {"anchor_planar_m": math.nan},
        {"snr_floor_db": math.nan},
        {"altitude_m": -1.0},
        {"n_pl": 0.0},
        {"shadow_sigma_db": -0.1},
        {"margin_quantile": 0.0},
        {"margin_quantile": 1.0},
        {"classes": ()},
        {"classes": ("wide", "wide")},
        {"classes": ("medium", "narrow")},  # no anchor class
        {"classes": ("wide", "ultra")},
        {"anchor_class": "ultra"},
    ],
)
def test_invalid_link_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        _link(**kwargs)


@pytest.mark.parametrize("kwargs", [{"anchor_planar_m": "60"}, {"classes": "wide"}, {"n_pl": None}])
def test_mistyped_link_parameters_are_refused(kwargs):
    with pytest.raises(TypeError):
        _link(**kwargs)


def test_a_class_that_cannot_clear_the_altitude_is_refused():
    # Anchored on narrow at 10 m planar and 100 m altitude, wide would reach
    # only about 28 m slant: it covers no ground device.
    with pytest.raises(ValueError, match="covers no ground device"):
        ContactLink(anchor_planar_m=10.0, altitude_m=100.0, anchor_class="narrow")


@pytest.mark.parametrize(
    "args",
    [
        ("", 20e6, 100, 0.754),
        ("x", 20, 100, 0.754),  # bandwidth in MHz, not Hz
        ("x", -1.0, 6, 0.7),
        ("x", 20e6, 0, 0.754),
        ("x", 20e6, 100.0, 0.754),
        ("x", 20e6, True, 0.754),
        ("x", 20e6, 100, 0.0),
        ("x", 20e6, 100, 1.2),
    ],
)
def test_invalid_band_classes_are_refused(args):
    with pytest.raises(ValueError):
        BandClass(*args)


def test_method_inputs_are_validated():
    link = _link()
    with pytest.raises(ValueError):
        link.rate_bps("wide", math.nan)
    with pytest.raises(ValueError):
        link.dwell_s(-1, "wide", 10.0)
    with pytest.raises(ValueError):
        link.dwell_s(math.inf, "wide", 10.0)
    with pytest.raises(ValueError):
        link.mean_snr_db("wide", -1.0)
    with pytest.raises(ValueError):
        link.mean_snr_db("wide", math.inf)
    with pytest.raises(TypeError):
        link.dwell_s("37576", "wide", 10.0)
    with pytest.raises(TypeError):
        link.dwell_s(True, "wide", 10.0)


# --------------------------------------------------------------------------- #
# Value semantics, provenance record, module hygiene
# --------------------------------------------------------------------------- #


def test_link_is_immutable_and_replace_recomputes():
    link = _link()
    with pytest.raises(dataclasses.FrozenInstanceError):
        link.n_pl = 3.0
    assert round(dataclasses.replace(link, n_pl=3.0).range_m("medium"), 1) == 103.2
    assert link == _link() and hash(link) == hash(_link())
    assert [f.name for f in dataclasses.fields(ContactLink)] == [
        "anchor_planar_m",
        "snr_floor_db",
        "altitude_m",
        "n_pl",
        "shadow_sigma_db",
        "margin_quantile",
        "classes",
        "anchor_class",
    ]
    # Class names resolve to the predefined classes, and numbers normalise.
    assert _link(anchor_planar_m=60, classes=list(CLASSES)) == link


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        # Every parameter off its default, so no field can come from a module
        # constant. At -15 dB the floor rate sits on the Shannon cap.
        {
            "anchor_planar_m": 50.0,
            "snr_floor_db": -15.0,
            "altitude_m": 30.0,
            "n_pl": 3.0,
            "shadow_sigma_db": 3.0,
            "margin_quantile": 0.95,
            "anchor_class": "medium",
        },
        # A floor above CQI 1's threshold: CQI 1 still applies at the floor.
        {"snr_floor_db": -5.0},
    ],
)
def test_describe_is_json_ready_and_agrees_with_the_methods(kwargs):
    link = _link(classes=CLASSES_WITH_10MHZ, **kwargs)
    record = json.loads(json.dumps(link.describe()))
    for key in (
        "anchor_planar_m",
        "anchor_class",
        "snr_floor_db",
        "altitude_m",
        "n_pl",
        "shadow_sigma_db",
        "margin_quantile",
    ):
        assert record[key] == getattr(link, key)
    assert record["shadow_margin_db"] == link.shadow_margin_db
    assert record["carrier_hz"] == 3.32e9
    assert [row["name"] for row in record["classes"]] == list(CLASSES_WITH_10MHZ)
    floor = link.snr_floor_db
    for row in record["classes"]:
        name = row["name"]
        cls = link.band(name)
        assert row["index"] == link.index(name)
        assert row["bandwidth_mhz"] == cls.bandwidth_hz / 1e6
        assert (row["n_prb"], row["kappa"]) == (cls.n_prb, cls.kappa)
        assert row["occupied_mhz"] == cls.occupied_hz / 1e6
        # The slant range is named apart from a waypoint's planar range_m.
        assert "range_m" not in row
        assert row["range_slant_m"] == link.range_m(name)
        assert row["range_planar_m"] == link.range_planar_m(name)
        # The rate at THIS link's floor. Independently: each floor here lies in
        # CQI 1's step or below it, where the Shannon cap may bind.
        assert row["floor_rate_bps"] == link.rate_bps(name, floor)
        assert row["floor_rate_bps"] == pytest.approx(
            min(_stepped_bps(cls, 1), _shannon_bps(cls.occupied_hz, floor)), rel=1e-12
        )
        assert row["peak_rate_bps"] == link.rate_bps(name, math.inf)
        assert row["peak_rate_bps"] == pytest.approx(_stepped_bps(cls, 15), rel=1e-12)


def test_module_uses_only_the_standard_library():
    """Design section 2.1: the l1 modules are numpy-free; hermes never imports experiments."""
    tree = ast.parse(Path(cl.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports"
            roots.add(node.module.split(".")[0])
    stdlib = {"__future__", "bisect", "dataclasses", "math", "numbers", "statistics", "types", "typing"}
    assert roots <= stdlib
