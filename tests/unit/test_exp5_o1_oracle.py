"""The O1 oracle (Study 5.4's optimality gap; experiments/analysis/o1_oracle.py).

Pinned:

* The hook wraps the scheduler's ``build_ferry_plan``: the flown plan is F's
  own, returned unchanged, and the oracle reads one gap per plan.
* The command line refuses a cell above N = 6, and the summary reads the rows.
* (slow) On one real mission of F in FerrySim: the oracle prices F's committed
  plan at F's own V; its best is never worse than F's under F's plan key, and
  its best V is never below F's; the bound that prunes branches changes
  neither optimum.
"""

from __future__ import annotations

import logging

import pytest

from experiments.analysis import o1_oracle as O


class _Sched:
    def __init__(self):
        self.calls = 0

    def _now(self):
        return 5.0

    def build_ferry_plan(self, *args, **kw):
        self.calls += 1
        return ["the plan"]


class _Service:
    def __init__(self):
        self.supervisor = type("S", (), {})()
        self.supervisor.scheduler = _Sched()


def test_the_hook_flies_fs_plan_and_reads_one_gap_per_plan(monkeypatch):
    seen = []
    monkeypatch.setattr(O, "capture", lambda sch, now: ("inputs", now))
    monkeypatch.setattr(O, "mission_gap", lambda inp, max_devices: seen.append(inp) or inp)
    sink = []
    service = _Service()
    O.oracle_hook(sink)(service)
    sch = service.supervisor.scheduler
    assert sch.build_ferry_plan() == ["the plan"] and sch.build_ferry_plan(now=7.0) == [
        "the plan"]
    assert sink == [("inputs", 5.0), ("inputs", 7.0)]


def test_the_command_line_refuses_n_above_six(tmp_path):
    with pytest.raises(SystemExit):
        O.main(["--cells", "jit-n12-90", "--episodes", "1", "--out", str(tmp_path / "o.json")])


def test_the_summary_reads_the_rows():
    rows = [dict(cell="c", gap_v=g, gap_v_at_key=g, gap_share_at_key=0.0, per_stop_bands=b,
                 wall_s=1.0) for g, b in ((0.0, False), (0.2, True), (0.1, True))]
    s = O.summarise(rows)["c"]
    assert s["missions"] == 3 and s["gap_v_max"] == 0.2
    assert s["share_at_zero_gap_v"] == pytest.approx(1 / 3)
    assert s["per_stop_band_share"] == pytest.approx(2 / 3)


@pytest.mark.slow
def test_on_a_real_mission_the_oracle_bounds_f_and_the_prune_is_exact(monkeypatch):
    from experiments.ferrysim import cells as C
    from experiments.ferrysim.episode import Policy, run_episode

    captured = []

    def grab(service):
        sch = service.supervisor.scheduler
        original = sch.build_ferry_plan

        def wrapped(*args, **kw):
            now = sch._now()
            route = original(*args, **kw)
            if not captured:
                captured.append(O.capture(sch, now=float(now)))
            return route

        sch.build_ferry_plan = wrapped

    cell = C.cell_named("jit-n6-75")
    logging.disable(logging.WARNING)
    try:
        run_episode(cell, C.stream_seeds(C.VAL_STREAM, cell.name, 1)[0], Policy.of_arm("F"),
                    hooks=(grab,))
    finally:
        logging.disable(logging.NOTSET)
    (inp,) = captured
    gap = O.mission_gap(inp)                 # raises if F's V is not re-priced exactly
    assert gap.f_v_repriced == pytest.approx(gap.f_v, abs=1e-9)
    assert gap.gap_v >= -1e-9 and gap.gap_share_at_key >= -1e-9
    pruned = O.OracleSearch(inp).run()
    monkeypatch.setattr(O.OracleSearch, "_hopeless", lambda self, *a: False)
    full = O.OracleSearch(inp).run()
    assert [s.key for s in pruned] == [s.key for s in full]
