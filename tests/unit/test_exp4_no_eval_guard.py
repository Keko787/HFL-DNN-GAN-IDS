"""EX-4 regression — a trial that produced NO model evaluation is not `ok`.

The first H2/H3 dead-zone sweep recorded 22/190 trials as ``status=ok`` even
though their cluster emitted no ``model_evaluation`` event at all
(``rounds_evaluated=0``). Those rows carry **blank** convergence columns but
**hard-zero** federation columns, so the analysis dropped them from the AUC
means (``.dropna()``) while averaging their 0.0 into the participation means —
an asymmetry that flipped the sign of several H3−H2 differences.

``Exp4Driver`` now stamps such a row ``status="no_eval"`` so the canonical
``status == "ok"`` filter excludes it from *every* metric. These tests pin both
halves of that contract: the metric layer really does emit an empty trace as
``rounds_evaluated=0`` + blank AUCs, and the analysis loader really does drop
any non-``ok`` row.
"""

from __future__ import annotations

import pandas as pd

from experiments.analysis.exp4 import load_exp4
from experiments.exp4.events_consumer import Exp4Observation
from experiments.exp4.metrics import summarise_observation


def _empty_obs(n_devices: int = 6) -> Exp4Observation:
    """A trial where the stack came up but nothing ever aggregated."""
    return Exp4Observation(
        n_devices=n_devices,
        cluster_rounds_closed=0,
        up_bundles_ingested=0,
        cluster_ready=True,
        mule_ready=True,
        dock_bootstrapped=False,   # the observed failure mode
    )


def test_empty_trace_yields_zero_rounds_and_blank_auc():
    """The signature the guard keys on: no evals -> 0 rounds, blank AUCs."""
    summary = summarise_observation(
        _empty_obs(), n_devices=6, rf_range_m=60.0, n_missions_target=4, tau=0.9,
    )
    assert int(summary.rounds_evaluated) == 0

    row = summary.to_row()
    # Convergence columns are blank ...
    for col in ("init_auc", "final_auc", "best_auc"):
        assert row[col] == "", f"{col} should be blank on an empty trace, got {row[col]!r}"
    # ... while the federation columns are hard zeros. That asymmetry is
    # exactly why such a row must never be labelled `ok`.
    assert float(row["update_yield"]) == 0.0
    assert int(row["missions_completed"]) == 0


def test_load_exp4_drops_non_ok_rows(tmp_path):
    """`no_eval` rows are excluded from EVERY metric, not just the blank ones."""
    csv_path = tmp_path / "trials.csv"
    pd.DataFrame([
        # a real trial
        dict(arm="H2", param_regime="jittery", trial_index=0, param_N=6,
             param_rrf=60.0, param_n_missions=4, param_dead_zone=0.0,
             param_link_quality=0.5, final_auc=0.97, update_yield=2.0, status="ok"),
        # the pathology: no model ever trained, but participation reads 0.0
        dict(arm="H3", param_regime="jittery", trial_index=0, param_N=6,
             param_rrf=60.0, param_n_missions=4, param_dead_zone=0.0,
             param_link_quality=0.5, final_auc="", update_yield=0.0,
             status="no_eval"),
    ]).to_csv(csv_path, index=False)

    df = load_exp4(csv_path)

    assert len(df) == 1, "non-ok rows must be dropped"
    assert set(df["arm"]) == {"H2"}
    # The fabricated 0.0 must not reach any mean.
    assert 0.0 not in list(pd.to_numeric(df["update_yield"], errors="coerce"))


def test_driver_marks_no_eval_status():
    """The driver's guard is wired to rounds_evaluated (source-level pin).

    The full path needs a real orchestrator (covered by the slow integration
    suite); here we pin that the guard exists and keys on the right signal, so
    a refactor cannot silently drop it.
    """
    import inspect

    from experiments.exp4.driver import Exp4Driver

    src = inspect.getsource(Exp4Driver._run_topology)
    assert "no_eval" in src, "the no-eval guard was removed from _run_topology"
    assert "rounds_evaluated" in src, "the guard must key on rounds_evaluated"
