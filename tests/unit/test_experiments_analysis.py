"""EX-1.5 — analysis tests.

Pins:

* :func:`paired_wilcoxon_with_cliffs_delta` — paired tests against
  hand-checked synthetic data.
* :func:`bootstrap_ci` — non-degenerate CIs from a known distribution.
* :func:`solve_crossover_round` — R* recovery from a synthetic linear
  trend.
* :func:`load_trials` + :func:`summarize` — round-trip a synthetic CSV.
* :func:`write_figures` — smoke test that figures land on disk without
  raising.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pytest

from experiments.analysis.exp1 import (
    TrialRow,
    load_trials,
    per_cell_crossovers,
    per_cell_paired_tests,
    per_row_energy,
    summarize,
    write_figures,
)
from experiments.analysis.stats import (
    bootstrap_ci,
    cliffs_delta,
    paired_wilcoxon_with_cliffs_delta,
    solve_crossover_round,
)
from experiments.calibration import Exp1Calibration


# --------------------------------------------------------------------------- #
# Cliff's delta + Wilcoxon
# --------------------------------------------------------------------------- #

def test_cliffs_delta_identical_arrays_zero():
    a = [1.0, 2.0, 3.0, 4.0]
    assert cliffs_delta(a, a) == pytest.approx(0.0, abs=1e-9)


def test_cliffs_delta_strictly_greater_one():
    a = [10.0, 20.0, 30.0]
    b = [1.0, 2.0, 3.0]
    assert cliffs_delta(a, b) == pytest.approx(1.0)


def test_cliffs_delta_strictly_less_minus_one():
    a = [1.0, 2.0, 3.0]
    b = [10.0, 20.0, 30.0]
    assert cliffs_delta(a, b) == pytest.approx(-1.0)


def test_paired_wilcoxon_significant_when_a_dominates():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=10.0, scale=1.0, size=30)
    b = rng.normal(loc=8.0, scale=1.0, size=30)
    res = paired_wilcoxon_with_cliffs_delta(a, b)
    assert res.significant
    assert res.cliffs_delta > 0.0
    assert res.delta_magnitude in ("medium", "large")


def test_paired_wilcoxon_handles_zero_diff():
    """All-zero diff returns a neutral p=1.0 rather than crashing."""
    a = [5.0, 5.0, 5.0]
    res = paired_wilcoxon_with_cliffs_delta(a, a)
    assert res.p_value == 1.0
    assert res.cliffs_delta == 0.0
    assert res.delta_magnitude == "negligible"


def test_paired_wilcoxon_rejects_size_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        paired_wilcoxon_with_cliffs_delta([1, 2, 3], [1, 2])


# --------------------------------------------------------------------------- #
# Bootstrap CI
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_brackets_true_mean():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=1.0, size=200)
    point, lo, hi = bootstrap_ci(
        samples, statistic=np.mean, n_bootstraps=500, seed=42,
    )
    assert lo < point < hi
    assert lo < 5.0 < hi


def test_bootstrap_ci_deterministic_under_seed():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_ci(samples, np.mean, n_bootstraps=100, seed=42)
    b = bootstrap_ci(samples, np.mean, n_bootstraps=100, seed=42)
    assert a == b


# --------------------------------------------------------------------------- #
# R* solver
# --------------------------------------------------------------------------- #

def test_solve_crossover_round_recovers_known_R_star():
    """T_proc_FL(R) = 0.5 + 0.1 R; centralized baseline = 1.0 → R* = 5."""
    fl_R = [1, 2, 3, 4, 5, 6]
    fl_T = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    cent_T = [1.0, 1.0, 1.0]
    est = solve_crossover_round(fl_R, fl_T, cent_T)
    assert est.R_star is not None
    assert est.R_star == pytest.approx(5.0, abs=1e-6)
    assert est.slope_per_round_s == pytest.approx(0.1, abs=1e-6)


def test_solve_crossover_round_degenerate_when_only_one_R():
    est = solve_crossover_round([5, 5, 5], [1.0, 1.1, 0.9], [2.0])
    assert est.R_star is None


def test_solve_crossover_round_degenerate_when_slope_non_positive():
    """If FL gets faster as R grows, no crossover by Tproc."""
    est = solve_crossover_round([1, 2, 3, 4], [4.0, 3.0, 2.0, 1.0], [2.5])
    assert est.R_star is None


# --------------------------------------------------------------------------- #
# CSV round-trip via load_trials
# --------------------------------------------------------------------------- #

def _write_synthetic_csv(path: Path, n_trials: int = 5) -> List[dict]:
    """Emit a CSV with the schema the experiment server produces."""
    rows: List[dict] = []
    cells = [
        ("Dpd=10MB|alpha=1.0|R=5", "10MB", 1.0, 5),
        ("Dpd=100MB|alpha=1.0|R=5", "100MB", 1.0, 5),
    ]
    for cell_id, Dpd, alpha, R in cells:
        for trial in range(n_trials):
            for arm, fl_R in (("FL", R), ("Centralized", 1)):
                Bpw = R * 2 * 200_000 if arm == "FL" else (
                    10 * 1024 * 1024 if Dpd == "10MB" else 100 * 1024 * 1024
                )
                rows.append({
                    "cell_id": cell_id,
                    "arm": arm,
                    "trial_index": trial,
                    "seed": trial * 31 + 7,
                    "param_Dpd": Dpd,
                    "param_alpha": alpha,
                    "param_R": R,
                    "Tproc_s": 0.05 + 0.01 * trial + (0.2 if arm == "FL" else 0.0),
                    "Ttx_s": 0.05 + 0.01 * trial,
                    "Bpw_bytes": Bpw,
                    "eta": 0.999 if arm == "Centralized" else 0.5,
                    "deadline_s": 1.0,
                    "Pcomplete": 1,
                    "n_rounds": fl_R,
                    "n_clients": 4,
                    "status": "ok",
                    "duration_s": 0.1,
                    "error": "",
                })

    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def test_load_trials_round_trip(tmp_path):
    csv_path = tmp_path / "exp1.csv"
    _write_synthetic_csv(csv_path, n_trials=3)
    rows = load_trials(csv_path)
    assert len(rows) == 2 * 3 * 2  # 2 cells × 3 trials × 2 arms
    assert all(isinstance(r, TrialRow) for r in rows)
    assert all(r.is_ok for r in rows)


def test_load_trials_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell_id", "arm"])
        w.writerow(["x", "FL"])
    with pytest.raises(ValueError, match="missing required columns"):
        load_trials(csv_path)


def test_summarize_emits_expected_blocks(tmp_path):
    csv_path = tmp_path / "exp1.csv"
    _write_synthetic_csv(csv_path, n_trials=10)
    rows = load_trials(csv_path)
    text = summarize(rows)
    assert "Experiment 1 — summary" in text
    # With only one R per cell, R* is degenerate; the line should say so.
    assert "R*" in text
    # Cliff's δ section appears when paired tests have ≥ 2 pairs.
    assert "Cliff" in text or "δ" in text


# --------------------------------------------------------------------------- #
# per_cell_paired_tests + per_cell_crossovers
# --------------------------------------------------------------------------- #

def test_per_cell_paired_tests_returns_one_entry_per_cell(tmp_path):
    csv_path = tmp_path / "exp1.csv"
    _write_synthetic_csv(csv_path, n_trials=5)
    rows = load_trials(csv_path)
    paired = per_cell_paired_tests(rows)
    # 2 cells, both with FL+Centralized at the same trial indices.
    assert len(paired) == 2
    for cell in paired:
        assert cell.n_pairs == 5
        # Centralized Bpw is much larger than FL's, so δ should be very negative.
        assert cell.Bpw_test.cliffs_delta < -0.5


def test_per_cell_crossovers_returns_one_per_dpd_alpha(tmp_path):
    """Even though synthetic CSV has only one R per cell, the function
    returns a degenerate CrossoverEstimate (R_star=None) per (Dpd, alpha)."""
    csv_path = tmp_path / "exp1.csv"
    _write_synthetic_csv(csv_path, n_trials=5)
    rows = load_trials(csv_path)
    cross = per_cell_crossovers(rows, n_bootstraps=100, seed=42)
    assert len(cross) == 2
    # Single R value → degenerate.
    assert all(c.R_star is None for c in cross)


# --------------------------------------------------------------------------- #
# Energy decomposition
# --------------------------------------------------------------------------- #

def test_per_row_energy_uses_calibration(tmp_path):
    csv_path = tmp_path / "exp1.csv"
    _write_synthetic_csv(csv_path, n_trials=2)
    rows = load_trials(csv_path)
    cal = Exp1Calibration(
        P_idle_W=10.0, epsilon_bit_J_per_bit=1.0e-9, B_nominal_bps=1e7,
    )
    e_rows = per_row_energy(rows, cal)
    assert len(e_rows) == len(rows)
    # idle_J = Tproc · 10 W; check at least one row.
    e0 = e_rows[0]
    r0 = rows[0]
    assert e0.idle_J == pytest.approx(r0.Tproc_s * 10.0)


# --------------------------------------------------------------------------- #
# Figure generation smoke
# --------------------------------------------------------------------------- #

def test_write_figures_smoke(tmp_path):
    csv_path = tmp_path / "exp1.csv"
    _write_synthetic_csv(csv_path, n_trials=4)
    rows = load_trials(csv_path)
    cal = Exp1Calibration(
        P_idle_W=5.0, epsilon_bit_J_per_bit=1.2e-9, B_nominal_bps=1e7,
    )
    figures_dir = tmp_path / "figures"
    written = write_figures(rows, cal, figures_dir=figures_dir,
                             placeholder_watermark=False)
    # Should produce at least the energy bar + Pcomplete heatmap +
    # paired-tests CSV; eta/Rstar may be skipped due to single-R synthetic.
    assert len(written) >= 3
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_figures_with_watermark(tmp_path):
    """Watermark path doesn't crash; just verify it runs."""
    csv_path = tmp_path / "exp1.csv"
    _write_synthetic_csv(csv_path, n_trials=3)
    rows = load_trials(csv_path)
    cal = Exp1Calibration(
        P_idle_W=5.0, epsilon_bit_J_per_bit=1.2e-9, B_nominal_bps=1e7,
    )
    figures_dir = tmp_path / "figures"
    written = write_figures(rows, cal, figures_dir=figures_dir,
                             placeholder_watermark=True)
    assert len(written) >= 1
