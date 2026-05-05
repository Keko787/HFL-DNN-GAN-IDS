"""Calibration TOML loader + energy-proxy formulas.

The TOML file ``experiments/calibration.toml`` is the single source of
truth for every physical-model constant in the experiments. This module
loads it, validates required keys, and exposes the two energy formulas
(Eq. 3 from §IV-B for Experiment 1, Eq. 5 from §IV-D for Experiment 3)
as small pure functions.

Why a separate module:

* The experiment server scripts don't need to know about energy at
  trial time — the CSV records ``Tproc_s`` and ``Bpw_bytes`` raw, and
  the analysis script computes energy post-hoc with the calibration.
* Calibration constants change *less* often than the analysis logic,
  so keeping them in TOML lets a paper reviewer diff calibration
  bumps without slogging through Python.
* The status word (``placeholder`` / ``verified``) gates whether the
  analysis stamps figures with a watermark.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# tomllib is stdlib in Python 3.11+; tomli is the back-port for 3.10.
if sys.version_info >= (3, 11):  # pragma: no cover - branch by interpreter
    import tomllib
else:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "tomli is required on Python <3.11; "
            "install with `pip install tomli`"
        ) from e


@dataclass(frozen=True)
class Exp1Calibration:
    """Constants for Experiment 1's energy proxy (Eq. 3)."""

    P_idle_W: float
    epsilon_bit_J_per_bit: float
    B_nominal_bps: float


@dataclass(frozen=True)
class Exp3Calibration:
    """Constants for Experiment 3's mission-energy proxy (Eq. 5)."""

    P_idle_W: float
    epsilon_bit_J_per_bit: float
    epsilon_prop_J_per_m: float
    mule_cruise_speed_m_s: float


@dataclass(frozen=True)
class Calibration:
    """Aggregated view of the whole calibration TOML."""

    exp1: Exp1Calibration
    exp3: Exp3Calibration
    last_verified: str
    source: str
    status: str  # "placeholder" | "verified"

    @property
    def is_paper_grade(self) -> bool:
        return self.status == "verified"


def load_calibration(path: Optional[Path] = None) -> Calibration:
    """Load + validate ``experiments/calibration.toml``.

    Defaults to ``experiments/calibration.toml`` next to this module.
    Pass an explicit ``path`` to load an alternate calibration (e.g.
    a per-paper-run override).
    """
    if path is None:
        path = Path(__file__).resolve().parent / "calibration.toml"
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    exp1_raw = _require(raw, "exp1.aerpaw_usrp")
    exp3_raw = _require(raw, "exp3.mule_platform")
    prov_raw = _require(raw, "provenance")

    return Calibration(
        exp1=Exp1Calibration(
            P_idle_W=_float(exp1_raw, "P_idle_W"),
            epsilon_bit_J_per_bit=_float(exp1_raw, "epsilon_bit_J_per_bit"),
            B_nominal_bps=_float(exp1_raw, "B_nominal_bps"),
        ),
        exp3=Exp3Calibration(
            P_idle_W=_float(exp3_raw, "P_idle_W"),
            epsilon_bit_J_per_bit=_float(exp3_raw, "epsilon_bit_J_per_bit"),
            epsilon_prop_J_per_m=_float(exp3_raw, "epsilon_prop_J_per_m"),
            mule_cruise_speed_m_s=_float(exp3_raw, "mule_cruise_speed_m_s"),
        ),
        last_verified=str(_require(prov_raw, "last_verified")),
        source=str(_require(prov_raw, "source")),
        status=_validate_status(_require(prov_raw, "status")),
    )


# --------------------------------------------------------------------------- #
# Energy formulas
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Exp1EnergyDecomposition:
    """Eq. 3 split into its idle and transmission components."""

    idle_J: float
    tx_J: float

    @property
    def total_J(self) -> float:
        return self.idle_J + self.tx_J


def exp1_energy_proxy(
    *, T_proc_s: float, B_pw_bytes: float, cal: Exp1Calibration,
) -> Exp1EnergyDecomposition:
    """Eq. 3: ``E = T_proc · P_idle + B_pw · ε_bit``, kept decomposed.

    The paper specifies the two components are reported as a stacked
    bar rather than collapsed to a scalar; we return the decomposition
    and let the caller choose to sum or stack.
    """
    idle = float(T_proc_s) * float(cal.P_idle_W)
    tx = float(B_pw_bytes) * 8.0 * float(cal.epsilon_bit_J_per_bit)
    return Exp1EnergyDecomposition(idle_J=idle, tx_J=tx)


@dataclass(frozen=True)
class Exp3EnergyDecomposition:
    """Eq. 5 split into idle, transmission, and propulsion components."""

    idle_J: float
    tx_J: float
    prop_J: float

    @property
    def total_J(self) -> float:
        return self.idle_J + self.tx_J + self.prop_J


def exp3_energy_proxy(
    *,
    T_mission_s: float,
    B_tx_bytes: float,
    L_path_m: float,
    cal: Exp3Calibration,
) -> Exp3EnergyDecomposition:
    """Eq. 5: ``E = T_mission · P_idle + B_tx · ε_bit + L_path · ε_prop``."""
    idle = float(T_mission_s) * float(cal.P_idle_W)
    tx = float(B_tx_bytes) * 8.0 * float(cal.epsilon_bit_J_per_bit)
    prop = float(L_path_m) * float(cal.epsilon_prop_J_per_m)
    return Exp3EnergyDecomposition(idle_J=idle, tx_J=tx, prop_J=prop)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _require(d: Dict[str, Any], dotted: str) -> Any:
    """Walk a dotted path through nested dicts; raise if missing."""
    parts = dotted.split(".")
    cur: Any = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"calibration TOML missing required key: {dotted!r}")
        cur = cur[p]
    return cur


def _float(d: Dict[str, Any], key: str) -> float:
    if key not in d:
        raise KeyError(f"calibration TOML missing required key: {key!r}")
    val = d[key]
    if not isinstance(val, (int, float)):
        raise TypeError(
            f"calibration TOML key {key!r} must be a number, got {type(val).__name__}"
        )
    return float(val)


def _validate_status(s: Any) -> str:
    s = str(s)
    if s not in ("placeholder", "verified"):
        raise ValueError(
            f"provenance.status must be 'placeholder' or 'verified', got {s!r}"
        )
    return s
