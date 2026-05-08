"""EX-3.5 — exp3 analysis tests.

Smoke-tests:

* CSV loader rejects schemas missing required columns.
* :func:`summarize` produces a non-empty string from a small CSV.
* :func:`write_figures` writes the expected file names without raising.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.analysis.exp3 import (
    load_trials,
    paired_test,
    summarize,
    write_figures,
)
from experiments.exp3.metrics import Exp3MetricSummary


def _required_cols():
    return [
        "cell_id", "arm", "trial_index", "seed",
        "param_N", "param_beta", "param_rrf", "param_deadline_het",
    ] + Exp3MetricSummary.csv_columns() + [
        "n_devices", "beta", "deadline_het", "rf_range_m",
        "status", "duration_s", "error",
    ]


def _write_csv(path: Path, rows):
    cols = _required_cols()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            full = {c: row.get(c, "") for c in cols}
            w.writerow(full)


def _row(arm: str, trial: int, **overrides):
    base = dict(
        cell_id="N=5|beta=1.0|deadline_het=False|rrf=60.0",
        arm=arm,
        trial_index=trial,
        seed=42,
        param_N=5,
        param_beta=1.0,
        param_rrf=60.0,
        param_deadline_het=False,
        update_yield=2.5,
        coverage=0.8,
        jains_fairness=0.9,
        participation_entropy=2.0,
        round_close_rate_kmin1=1.0,
        round_close_rate_kmin2=0.95,
        round_close_rate_kminhalf=0.8,
        round_close_rate_kminN=0.5,
        rho_contact="",
        pass2_coverage="",
        propulsion_energy_J="",
        propulsion_idle_J="",
        propulsion_tx_J="",
        propulsion_prop_J="",
        mission_completion_s="",
        path_length_m="",
        n_devices=5,
        beta=1.0,
        deadline_het=False,
        rf_range_m=60.0,
        status="ok",
        duration_s=0.1,
        error="",
    )
    base.update(overrides)
    return base


def test_load_trials_rejects_missing_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("cell_id,arm,trial_index\n")
    with pytest.raises(ValueError):
        load_trials(bad)


def test_summarize_handles_small_csv(tmp_path: Path):
    csv_path = tmp_path / "exp3_tiny.csv"
    rows = []
    for trial in range(4):
        rows.append(_row("A1", trial, update_yield=2.0,
                         jains_fairness=0.5))
        rows.append(_row("A2", trial, update_yield=3.0,
                         jains_fairness=0.8,
                         rho_contact=2.5,
                         pass2_coverage=0.6,
                         propulsion_energy_J=100.0,
                         mission_completion_s=120.0,
                         path_length_m=50.0))
        rows.append(_row("A3", trial, update_yield=3.5,
                         round_close_rate_kminhalf=0.9,
                         rho_contact=2.5,
                         pass2_coverage=0.7,
                         propulsion_energy_J=80.0,
                         mission_completion_s=110.0,
                         path_length_m=45.0))
        rows.append(_row("A4", trial, update_yield=4.0,
                         round_close_rate_kminhalf=0.95,
                         rho_contact=2.5,
                         pass2_coverage=0.85,
                         propulsion_energy_J=70.0,
                         mission_completion_s=100.0,
                         path_length_m=40.0))
    _write_csv(csv_path, rows)
    loaded = load_trials(csv_path)
    assert len(loaded) == 16
    text = summarize(loaded)
    # The summary mentions the comparisons we expect.
    assert "A2 vs A1" in text
    assert "A4 vs A3" in text


def test_paired_test_returns_none_when_too_few_pairs(tmp_path: Path):
    csv_path = tmp_path / "exp3_tiny.csv"
    rows = [_row("A2", 0), _row("A1", 0)]  # only 1 pair
    _write_csv(csv_path, rows)
    loaded = load_trials(csv_path)
    pt = paired_test(loaded, "A2", "A1", "update_yield")
    assert pt.test is None
    assert pt.n_pairs == 1


def test_write_figures_smoke(tmp_path: Path):
    csv_path = tmp_path / "exp3_smoke.csv"
    rows = []
    # Two cells × four arms × four trials so the box-plots have data.
    for rrf in (60.0, 120.0):
        for trial in range(4):
            rows.append(_row("A1", trial, param_rrf=rrf, rf_range_m=rrf,
                             update_yield=2.0 + 0.1 * trial,
                             jains_fairness=0.5))
            rows.append(_row("A2", trial, param_rrf=rrf, rf_range_m=rrf,
                             update_yield=3.0 + 0.1 * trial,
                             rho_contact=2.0))
            rows.append(_row("A3", trial, param_rrf=rrf, rf_range_m=rrf,
                             update_yield=3.5,
                             round_close_rate_kminhalf=0.9,
                             rho_contact=2.0,
                             propulsion_energy_J=80.0))
            rows.append(_row("A4", trial, param_rrf=rrf, rf_range_m=rrf,
                             update_yield=4.0,
                             round_close_rate_kminhalf=0.95,
                             rho_contact=2.0))
    _write_csv(csv_path, rows)

    figs_dir = tmp_path / "figures"
    loaded = load_trials(csv_path)
    written = write_figures(loaded, figures_dir=figs_dir)
    written_names = {p.name for p in written}
    # The paired-tests CSV plus at least three of the six figures.
    assert "exp3_paired_tests.csv" in written_names
    assert any("a4_vs_a3" in n for n in written_names)
    assert any("rho_contact" in n for n in written_names)
    # All-arms per-metric figures (one per metric) + mule-only energy.
    # fig0a now leads with the round-count-invariant headline metric;
    # update_yield is demoted to fig0f as a supporting panel.
    assert "exp3_fig0a_mission_completion_rate.png" in written_names
    assert "exp3_fig0b_round_close_rate_kmin2.png" in written_names
    assert "exp3_fig0c_completion_fairness.png" in written_names
    assert "exp3_fig0d_coverage.png" in written_names
    assert "exp3_fig0e_propulsion_energy.png" in written_names
    assert "exp3_fig0f_update_yield.png" in written_names
    # LaTeX caption sidecar — one \begin{figure} per metric figure.
    assert "exp3_fig_captions.tex" in written_names
    tex = (figs_dir / "exp3_fig_captions.tex").read_text(encoding="utf-8")
    for stem in (
        "exp3_fig0a_mission_completion_rate",
        "exp3_fig0b_round_close_rate_kmin2",
        "exp3_fig0c_completion_fairness", "exp3_fig0d_coverage",
        "exp3_fig0e_propulsion_energy",
        "exp3_fig0f_update_yield",
    ):
        assert stem in tex, f"caption .tex missing reference to {stem}"
    # Use \end{figure} as the canonical count — \begin{figure} appears
    # in the file's leading comment line as well. Seven figures:
    # fig0a-f (a=mission_completion_rate, b=round_close, c=fairness,
    # d=coverage, e=propulsion_energy, f=update_yield/supporting)
    # plus fig4 (consolidated β-sweep).
    assert tex.count(r"\end{figure}") == 7
    assert r"\caption" in tex and r"\label" in tex
    assert "fig:exp3:beta_sweep" in tex
