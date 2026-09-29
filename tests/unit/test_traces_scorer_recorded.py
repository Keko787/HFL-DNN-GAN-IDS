"""The trace scorer on the recorded L1 cell (``results/exp4_matrix/C_traces``).

Pins what the scorer says about real traces, not just hand-built ones: the
reach counts reported in ``HERMES_Matrix_Results.md`` (τ = 0.82: H3 28/40 vs
H2 19/40; τ = 0.75: 39/40 vs 30/40), and agreement with the trial CSV the
same run wrote on every column Freeze Amendment 5 left alone. Round closure is
the exception: the recorded CSV counted backhaul-dropped rounds as closed.

Skipped where the recorded results are not checked out.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from experiments.analysis.traces_scorer import score_traces

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / "results" / "exp4_matrix" / "C_traces"
TRIAL_CSV = REPO / "results" / "exp4_matrix" / "C_h2h3.csv"

pytestmark = pytest.mark.skipif(
    not TRACES.is_dir(), reason="recorded traces results/exp4_matrix/C_traces not present",
)

#: Columns the scorer computes differently from the recorded CSV, by design,
#: besides ``round_close_rate_k*`` (Amendment 5): the cell id is sanitised in
#: the directory name, and the CSV's time to τ used τ = 0.9 where the scorer
#: uses its first τ.
NOT_COMPARABLE = {"cell_id", "tau", "t_at_tau_round"}


@pytest.fixture(scope="module")
def scores():
    return score_traces(TRACES, taus=(0.82, 0.75))


def test_every_arm_has_forty_trials(scores):
    by_arm = {}
    for s in scores:
        by_arm[s.key.arm] = by_arm.get(s.key.arm, 0) + 1
    assert by_arm == {"H2": 40, "H3": 40}


@pytest.mark.parametrize("arm, reached", [("H3", (28, 39)), ("H2", (19, 30))])
def test_reach_counts_match_the_matrix_report(scores, arm, reached):
    group = [s for s in scores if s.key.arm == arm]
    assert tuple(sum(s.tau[i].reached for s in group) for i in range(2)) == reached


@pytest.mark.skipif(not TRIAL_CSV.is_file(), reason="recorded trial CSV not present")
def test_unaffected_columns_match_the_recorded_trial_csv(scores):
    with open(TRIAL_CSV, newline="", encoding="utf-8") as f:
        recorded = {
            (r["arm"], int(r["trial_index"]), int(r["seed"])): r for r in csv.DictReader(f)
        }
    assert len(recorded) == len(scores)
    compared = set()
    for s in scores:
        row = s.to_row()
        want = recorded[(s.key.arm, s.key.trial_index, s.key.seed)]
        for col, value in row.items():
            if col not in want or col in NOT_COMPARABLE or col.startswith("round_close_rate_k"):
                continue
            compared.add(col)
            if want[col] == "" or value == "":
                assert (col, value) == (col, want[col])
            else:
                try:
                    expected = float(want[col])
                except ValueError:
                    assert (col, str(value)) == (col, want[col])
                else:
                    assert (col, float(value)) == (col, pytest.approx(expected, rel=1e-12))
    # The comparison covered the metric columns and the provenance/status ones.
    assert {"update_yield", "final_auc", "mission_budget_s", "status"} <= compared
