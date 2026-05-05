"""EX-1.3 — calibration TOML loader + energy formulas."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.calibration import (
    Exp1Calibration,
    Exp3Calibration,
    exp1_energy_proxy,
    exp3_energy_proxy,
    load_calibration,
)


# --------------------------------------------------------------------------- #
# load_calibration
# --------------------------------------------------------------------------- #

def test_load_default_calibration_succeeds():
    """The TOML shipped with the repo loads without errors."""
    cal = load_calibration()
    assert cal.exp1.P_idle_W > 0
    assert cal.exp1.epsilon_bit_J_per_bit > 0
    assert cal.exp1.B_nominal_bps > 0
    assert cal.exp3.P_idle_W > 0
    assert cal.exp3.epsilon_prop_J_per_m > 0
    assert cal.status in ("placeholder", "verified")


def test_default_calibration_is_placeholder():
    """The shipped TOML must declare itself placeholder until paper run."""
    cal = load_calibration()
    assert cal.status == "placeholder"
    assert cal.is_paper_grade is False


def test_load_alternate_calibration_path(tmp_path):
    custom = tmp_path / "custom.toml"
    custom.write_text(
        '''
[exp1.aerpaw_usrp]
P_idle_W = 7.5
epsilon_bit_J_per_bit = 2.5e-9
B_nominal_bps = 5_000_000

[exp3.mule_platform]
P_idle_W = 7.5
epsilon_bit_J_per_bit = 2.5e-9
epsilon_prop_J_per_m = 15.0
mule_cruise_speed_m_s = 8.0

[provenance]
last_verified = "2026-12-31"
source = "test override"
status = "verified"
''',
        encoding="utf-8",
    )
    cal = load_calibration(custom)
    assert cal.exp1.P_idle_W == 7.5
    assert cal.exp3.epsilon_prop_J_per_m == 15.0
    assert cal.status == "verified"
    assert cal.is_paper_grade is True


def test_load_rejects_missing_required_table(tmp_path):
    incomplete = tmp_path / "bad.toml"
    incomplete.write_text(
        '[exp1.aerpaw_usrp]\nP_idle_W = 5.0\n', encoding="utf-8",
    )
    with pytest.raises(KeyError, match="missing required key"):
        load_calibration(incomplete)


def test_load_rejects_invalid_status(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '''
[exp1.aerpaw_usrp]
P_idle_W = 5.0
epsilon_bit_J_per_bit = 1.2e-9
B_nominal_bps = 10000000

[exp3.mule_platform]
P_idle_W = 5.0
epsilon_bit_J_per_bit = 1.2e-9
epsilon_prop_J_per_m = 10.0
mule_cruise_speed_m_s = 5.0

[provenance]
last_verified = "2026-05-01"
source = "test"
status = "definitely-not-real"
''',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status must be"):
        load_calibration(bad)


def test_load_rejects_non_numeric_constant(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '''
[exp1.aerpaw_usrp]
P_idle_W = "five point zero"
epsilon_bit_J_per_bit = 1.2e-9
B_nominal_bps = 10000000

[exp3.mule_platform]
P_idle_W = 5.0
epsilon_bit_J_per_bit = 1.2e-9
epsilon_prop_J_per_m = 10.0
mule_cruise_speed_m_s = 5.0

[provenance]
last_verified = "2026-05-01"
source = "test"
status = "placeholder"
''',
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="must be a number"):
        load_calibration(bad)


# --------------------------------------------------------------------------- #
# exp1_energy_proxy
# --------------------------------------------------------------------------- #

def test_exp1_energy_idle_only_when_no_bytes():
    cal = Exp1Calibration(
        P_idle_W=5.0, epsilon_bit_J_per_bit=1.2e-9, B_nominal_bps=1e7,
    )
    e = exp1_energy_proxy(T_proc_s=10.0, B_pw_bytes=0, cal=cal)
    assert e.idle_J == pytest.approx(50.0)
    assert e.tx_J == pytest.approx(0.0)
    assert e.total_J == pytest.approx(50.0)


def test_exp1_energy_tx_only_when_zero_proc_time():
    cal = Exp1Calibration(
        P_idle_W=5.0, epsilon_bit_J_per_bit=1.0e-9, B_nominal_bps=1e7,
    )
    # 1 MB = 8 Mbits = 8e6 bits → 8e6 * 1e-9 = 8e-3 J
    e = exp1_energy_proxy(T_proc_s=0.0, B_pw_bytes=1_000_000, cal=cal)
    assert e.idle_J == pytest.approx(0.0)
    assert e.tx_J == pytest.approx(8e-3)


def test_exp1_energy_combines_idle_and_tx():
    cal = Exp1Calibration(
        P_idle_W=5.0, epsilon_bit_J_per_bit=1.0e-9, B_nominal_bps=1e7,
    )
    e = exp1_energy_proxy(T_proc_s=10.0, B_pw_bytes=1_000_000, cal=cal)
    assert e.idle_J == pytest.approx(50.0)
    assert e.tx_J == pytest.approx(8e-3)
    assert e.total_J == pytest.approx(50.008)


# --------------------------------------------------------------------------- #
# exp3_energy_proxy
# --------------------------------------------------------------------------- #

def test_exp3_energy_three_components_independent():
    cal = Exp3Calibration(
        P_idle_W=5.0, epsilon_bit_J_per_bit=1.0e-9,
        epsilon_prop_J_per_m=10.0, mule_cruise_speed_m_s=5.0,
    )
    e = exp3_energy_proxy(
        T_mission_s=60.0, B_tx_bytes=1_000_000, L_path_m=200.0, cal=cal,
    )
    assert e.idle_J == pytest.approx(300.0)
    assert e.tx_J == pytest.approx(8e-3)
    assert e.prop_J == pytest.approx(2000.0)
    assert e.total_J == pytest.approx(2300.008)


def test_exp3_energy_propulsion_dominates_at_long_paths():
    """The propulsion term is what makes wasted hops show up."""
    cal = Exp3Calibration(
        P_idle_W=5.0, epsilon_bit_J_per_bit=1.0e-9,
        epsilon_prop_J_per_m=10.0, mule_cruise_speed_m_s=5.0,
    )
    short = exp3_energy_proxy(
        T_mission_s=60.0, B_tx_bytes=1_000_000, L_path_m=100.0, cal=cal,
    )
    long = exp3_energy_proxy(
        T_mission_s=60.0, B_tx_bytes=1_000_000, L_path_m=1000.0, cal=cal,
    )
    assert long.total_J > short.total_J
    # The delta is purely propulsion.
    assert (long.prop_J - short.prop_J) == pytest.approx(9000.0)
