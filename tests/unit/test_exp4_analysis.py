"""EX-4.2 R4 — paired H1-vs-H0 statistical analysis (analysis/exp4.py).

Deterministic synthetic CSV with a known pattern (clean = tie, jittery =
H1 wins) pins the paired-Wilcoxon + bootstrap-CI verdict logic without
needing a real orchestrator sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analysis.exp4 import analyze, analyze_metric


def _make_df(n_seeds: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    common = dict(param_N=6, param_rrf=60.0, param_n_missions=4, status="ok")
    for t in range(n_seeds):
        # clean: both arms ~0.930 (tie)
        for arm in ("H0", "H1"):
            rows.append(dict(
                arm=arm, param_regime="clean", trial_index=t,
                final_auc=0.930 + rng.normal(0, 0.002),
                final_accuracy=0.90, update_yield=3.0,
                mission_completion_rate=0.9, round_close_rate_kmin2=0.7,
                **common,
            ))
        # jittery: H1 beats H0 consistently
        rows.append(dict(
            arm="H0", param_regime="jittery", trial_index=t,
            final_auc=0.905 + rng.normal(0, 0.002),
            final_accuracy=0.90, update_yield=0.5,
            mission_completion_rate=0.33, round_close_rate_kmin2=0.13,
            **common,
        ))
        rows.append(dict(
            arm="H1", param_regime="jittery", trial_index=t,
            final_auc=0.925 + rng.normal(0, 0.002),
            final_accuracy=0.90, update_yield=1.75,
            mission_completion_rate=0.75, round_close_rate_kmin2=0.75,
            **common,
        ))
    return pd.DataFrame(rows)


def test_clean_tie_jittery_h1_wins():
    df = _make_df(20)

    clean = analyze_metric(df, "clean", "final_auc")
    assert clean is not None and clean.n_pairs == 20
    assert clean.verdict == "tie"          # CI straddles 0

    jit = analyze_metric(df, "jittery", "final_auc")
    assert jit.verdict == "H1 > H0"        # CI strictly positive
    assert jit.ci_lo > 0.0
    assert jit.p_value < 0.05
    assert jit.mean_diff > 0.015
    assert jit.magnitude == "large"        # perfect separation -> delta=+1

    # A participation metric also favours H1 under jittery.
    assert analyze_metric(df, "jittery", "update_yield").verdict == "H1 > H0"


def test_analyze_runs_all_regimes_and_metrics():
    results = analyze(_make_df(10))
    verdicts = {(r.regime, r.metric): r.verdict for r in results}
    assert verdicts[("jittery", "final_auc")] == "H1 > H0"
    assert verdicts[("clean", "final_auc")] == "tie"


def test_too_few_pairs_returns_none():
    df = _make_df(1)   # 1 seed -> <2 pairs
    assert analyze_metric(df, "clean", "final_auc") is None
