"""FeRRy Phase 5 (unit U8a): FerrySim's runner, its cells, its reward and the headroom tool.

FerrySim (``experiments/ferrysim/``) is the stack's own trial run in one
process, with stand-ins for local training and the devices' service loops,
which do not run (the user's decision 2 (a); the orchestrator's resolution
R25). Pinned here (the Phase 5 spec's units table, row U8a):

* **The cells and the seed streams** (decision 3; critic A9, B14, C3): the six
  cells, their Phase 4 pilot flags, the S* tool's S at their budgets, the
  families, and the three seed streams, disjoint and checked.
* **Study 5.6's cells** (decision 6 (a); critic A3; orchestrator resolution
  R22): the four cells, each its Study 5.5 cell at P_c = 4 x or 2 x the pinned
  lag; the families ``jittery`` and ``clean`` and their hashes unchanged, and
  ``jittery56`` beside them; A3's arrival-to-arrival lag; the lag's
  measurement in the repository (``evaluate.fx_lags``, ``fx_lag_median``: FX's
  lags pooled over its first validation episodes; the Phase 5 repair round's
  RB-2), and, marked slow, the pinned lags measured again (minutes) and FX's
  lag / P_c on each cell near its quarter or half (about 40 s; RB-4).
* **Parity.** The in-process copy of the golden harness, with FerrySim's mission
  tap installed, reproduces UG5's oracle of the FX trials (every part, the
  flight slot's calls included) and UG4's of a legacy trial; and one stub FX
  trial through the real orchestrator (real processes and TCP) equals
  FerrySim's run on the row, the configs and every mule and cluster event
  (critic B8), bar what the wall clock and the OS decide (the stamps, the
  ports, the row's ``mission_duration_s_mean``) and the row's three
  device-serve columns: the real devices log ``device_served`` and FerrySim's
  only ``device_ready``.
* **The device-serve columns** (resolution R25; the final check's FS-1): every
  FerrySim episode's row, and the trace scorer's row of its kept trace, hold
  their harness artifacts (coverage 0.0, Jain's index 1.0, entropy 0.0),
  whatever flies it and under either device model.
* **T3 for episodes**: the same inputs give the same records, rewards, returns
  and trace bar the planner's wall time; other seeds, and another trainer
  stream, differ. The trainer's reference phase flies FX's pair.
* **How a flight is read**: N, the demand, the committed set, the coverage
  weights, T, the collected members, the last leg and the energy model's powers
  against the kept trace itself; a kept trace is filed under its FerrySim cell
  and policy, so no episode's trace overwrites another's.
* **T4**: on a hand-worked mission the gains sum to the merge's weights, each
  credited to its stop (``update_weights`` and ``merge_on_mule``); in a FerrySim
  mission at K = 1 a stop's gain is its collected n_i over n_ref N, and the
  mule's own closed records agree with FerrySim's reading stop for stop.
* **The reward**: the derived reward, F·hand and E3's bytes by hand; the
  expected-availability reward is the realized reward's mean over the keyed
  draw, exactly by enumeration, against the draw itself, and on FerrySim's
  returns.
* **The headroom tool**: the replay scorer and the leaf order, the leaf cap and
  the first leaf's check, the oracle at least every scripted policy on toy cells
  (exhausted searches), the headroom over pair choices against the FX arm with
  F's gain beside it, ε and the pause rule, and the report's reward and device
  model; and its plan (resolution R23; the repair round's A-1): every flight
  flies ``--plan-score-params`` as the runner's flag reads it, and the report
  records it, while on the cells' own plan no flight is overridden and nothing
  is recorded.
* **The evaluator**: every policy meets the same episodes, each flown with the
  task's reward and device model; worker processes give the same results; the
  held-out score lands in a checkpoint's manifest.
* **Layering and cost**: FerrySim imports nothing from ``tests``; smoke runs stay
  under 15 s.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import json
import logging
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.exp4.driver import F_COV_SCORE, Exp4Driver, trace_dir_name
from experiments.ferrysim import cells as C
from experiments.ferrysim import episode as E
from experiments.ferrysim import evaluate as EV
from experiments.ferrysim import headroom as HR
from experiments.ferrysim import inprocess as IP
from experiments.ferrysim import reward as R
from hermes.mission.aggregation_rules import AggregationSpec, merge_on_mule, update_weights
from hermes.types import DeviceID, GradientSubmission, MissionOutcome, MuleID
from hermes.types.fl_messages import UPDATE_FORM_DELTA, UPDATE_FORM_WEIGHTS

from tests.golden import _build_p3_sim as UG4
from tests.golden import _build_p4_plan as UG5
from tests.golden import _canon

REPO = Path(__file__).resolve().parents[2]
N12 = C.cell_named("jit-n12-120")
N6 = C.cell_named("jit-n6-90")
EXPECTED = dataclasses.replace(R.DERIVED, expected_availability=True)

#: Toy cells, chosen by a probe over six validation seeds of four toy
#: configurations: two missions, small enough that the oracle's search is
#: exhausted in a few seconds. On the first the oracle finds a gain no
#: scripted reference takes (0.54); on the second F and ``committed_pair`` beat
#: FX's pair rule, and one sortie's search has 14 leaves.
TOY = (
    (C.FerryCell("toy-n8-120", C.FAMILY_JITTERY, C.ROLE_DECISION_RICH, 8, 120.0,
                 C.BUDGET_STAND_IN, 2, "jittery", n_missions=2), 2796905889),
    (C.FerryCell("toy-n10-120", C.FAMILY_JITTERY, C.ROLE_DECISION_RICH, 10, 120.0,
                 C.BUDGET_STAND_IN, 2, "jittery", n_missions=2), 2341372912),
)


def _seed(cell=N12, index=0, stream=C.VAL_STREAM):
    return C.stream_seeds(stream, cell.name, 1, start=index)[0]


def _f(value):
    """A float of a canonical case (``"f:<repr>"``) as a float."""
    return float(value[2:]) if isinstance(value, str) and value.startswith("f:") else value


def _floats(record):
    """A canonical case's record of floats as a dict of floats."""
    return {k: _f(v) for k, v in record.items() if k != IP.TYPE_KEY}


@functools.lru_cache(maxsize=None)
def _episode(cell_name, index, label):
    """A cached episode of a cell on the validation stream, flown by a reference."""
    policy = {p.label: p for p in E.reference_policies()}[label]
    cell = C.cell_named(cell_name)
    return E.run_episode(cell, _seed(cell, index), policy, trial_index=index)


def _qualifying(pick, tries=12):
    """The first (cell, validation index) of the Study 5.5 cells, index by index and
    the stress budget first, whose episode ``pick(cell, index)`` accepts: a test
    that needs a decision-rich episode finds one at whatever budgets the cells are
    re-pinned to (scripts/exp5/repin.py)."""
    for index in range(tries):
        for cell in C.STUDY_5_5_CELLS:
            if pick(cell, index):
                return cell, index
    raise AssertionError(f"no Study 5.5 validation episode qualifies in {tries} tries")


@functools.lru_cache(maxsize=None)
def _decisions(cell, index):
    """The pair slot's decisions in a validation episode (fx_pair, a trainer at 0)."""
    ep = E.run_episode(cell, _seed(cell, index), E.Policy.scripted("fx_pair"),
                       trainer=E.Trainer())
    return sum(len(steps) for steps in ep.steps)


# --------------------------------------------------------------------------- #
# The cells and the seed streams
# --------------------------------------------------------------------------- #

def test_the_cells_are_decision_3s_on_phase_4s_pilot_flags():
    assert [c.name for c in C.CELLS] == [
        "jit-n6-45", "jit-n6-90", "jit-n12-120", "jit-n12-180", "cln-n12-120", "cln-n12-180"]
    table = {c.name: (c.family, c.role, c.n_devices, c.budget_s, c.budget_role,
                      c.contact_regime) for c in C.CELLS}
    (s6, k6), (s12, k12) = C.BUDGETS_S[6], C.BUDGETS_S[12]
    (rs6, rk6), (rs12, rk12) = C.BUDGET_ROLES[6], C.BUDGET_ROLES[12]
    assert table == {
        "jit-n6-45": ("jittery", "control", 6, s6, rs6, "jittery"),
        "jit-n6-90": ("jittery", "control", 6, k6, rk6, "jittery"),
        "jit-n12-120": ("jittery", "decision-rich", 12, s12, rs12, "jittery"),
        "jit-n12-180": ("jittery", "decision-rich", 12, k12, rk12, "jittery"),
        "cln-n12-120": ("clean", "negative-control", 12, s12, rs12, "clean"),
        "cln-n12-180": ("clean", "negative-control", 12, k12, rk12, "clean"),
    }
    # each name carries its size and budget, each size's stress budget first
    assert all(c.name == C.cell_label(c.name[:3], c.n_devices, c.budget_s) for c in C.CELLS)
    assert s6 < k6 and s12 < k12
    for c in C.CELLS:
        assert (c.payload_bytes, c.n_missions, c.network_regime, c.cap_s) == (
            1_000_000, 4, "jittery", C.CAP_S[(c.n_devices, c.contact_regime)])
        # UG5's pilot flags (Experiment_4_Run_Guide.md 2.7), the cell's own budget,
        # cap and contact channel on top
        assert c.driver_settings() == dict(
            UG5.PLAN_PILOT, mission_budget_s=c.budget_s, age_cap_missions=c.cap_s,
            ferry_physics={"contact_regime": c.contact_regime})
        cell = c.cell("FX", 123, 4)
        assert (cell.cell_id, cell.arm, cell.seed, cell.trial_index) == (
            f"N={c.n_devices}|n_missions=4|regime=jittery|rrf=60.0", "FX", 123, 4)
        assert dict(cell.params) == {"N": c.n_devices, "rrf": 60.0, "n_missions": 4,
                                     "regime": "jittery"}
    assert C.FAMILIES["jittery"] == C.CELLS[:4] and C.FAMILIES["clean"] == C.CELLS[4:]
    assert C.STUDY_5_5_CELLS == (C.cell_named("jit-n12-120"), C.cell_named("jit-n12-180"))
    assert C.family_sha256("jittery") != C.family_sha256("clean")
    assert C.family_sha256("jittery") == C.family_sha256("jittery")
    with pytest.raises(ValueError, match="no FerrySim cell"):
        C.cell_named("jit-n24-120")
    with pytest.raises(ValueError, match="role"):
        C.FerryCell("x", "jittery", "pilot", 6, 45.0, C.BUDGET_STAND_IN, 2, "jittery")


def test_a_cell_can_set_the_interference_period_for_study_5_6():
    cell = dataclasses.replace(N12, name="jit-n12-120-p30", interference_period_s=30.0)
    assert cell.driver_settings()["ferry_physics"] == {"contact_regime": "jittery",
                                                       "interference_period_s": 30.0}
    assert N12.driver_settings()["ferry_physics"] == {"contact_regime": "jittery"}
    periods = []
    E.run_episode(cell, _seed(), E.Policy.of_arm("FX"), stop_after=0, hooks=[
        lambda service: periods.append(
            service.supervisor.ferry.contact_channel.interference_period_s)])
    assert periods == [30.0]
    with pytest.raises(ValueError, match="interference_period_s"):
        dataclasses.replace(N12, interference_period_s=0.0)


@pytest.mark.parametrize("n, contact", [(6, "jittery"), (12, "jittery"), (12, "clean")])
def test_each_cells_cap_is_the_s_star_tools_at_its_budgets(n, contact):
    """S* at the size's two budgets, never below 2 (decision 1 of Phase 4)."""
    from experiments.analysis.age_cap_s_star import s_star_report

    budgets = C.BUDGETS_S[n]
    driver = Exp4Driver(mission_clock="sim", realism=True, contact_band="wide",
                        payload_bytes=1_000_000, ferry_physics={"contact_regime": contact})
    report = s_star_report(driver, n_devices=n, budgets=budgets, regime="jittery",
                           families=("F",))
    cells = [c for c in C.CELLS if c.n_devices == n and c.contact_regime == contact]
    assert sorted(c.budget_s for c in cells) == sorted(budgets)
    assert {c.cap_s for c in cells} == {max(2, report.s("F"))} == {C.CAP_S[(n, contact)]}


def test_the_seed_streams_are_disjoint_and_checked():
    groups = []
    for c in C.CELLS:
        groups.append((C.VAL_STREAM, C.stream_seeds(C.VAL_STREAM, c.name, 200)))
        groups.append((C.HELDOUT_STREAM, C.stream_seeds(C.HELDOUT_STREAM, c.name, 1000)))
    for run in range(3):
        for family in C.FAMILIES:
            groups.append((C.train_stream(run),
                           [C.train_episode(run, family, e)[1] for e in range(2000)]))
    C.check_disjoint(groups)
    kinds = collections.defaultdict(set)
    for stream, seeds in groups:
        kinds[C.stream_kind(stream)].update(seeds)
        assert {C.seed_kind(s) for s in seeds} == {C.stream_kind(stream)}
        assert all(0 <= s < 2 ** 32 for s in seeds)
    assert not kinds["val"] & kinds["heldout"]
    assert not kinds["train"] & (kinds["val"] | kinds["heldout"])
    held, val = groups[1][1], groups[0][1]
    with pytest.raises(ValueError, match="carries the tag"):
        C.check_disjoint([(C.VAL_STREAM, [held[0]])])
    with pytest.raises(ValueError, match="repeats"):
        C.check_disjoint([(C.HELDOUT_STREAM, [held[0], held[0]])])
    with pytest.raises(ValueError, match="carries the tag"):
        C.check_disjoint([(C.train_stream(0), [val[0]])])
    # the goldens' and the runner's small seeds are in no stream
    assert C.seed_kind(59) is None and C.seed_kind(26) is None
    with pytest.raises(ValueError, match="FerrySim stream"):
        C.stream_kind("ferrysim-train-x")
    with pytest.raises(ValueError, match="int >= 0"):
        C.train_stream(-1)


def test_a_stream_is_a_function_of_its_name_cell_and_index(monkeypatch):
    full = C.stream_seeds(C.HELDOUT_STREAM, "jit-n12-120", 50)
    assert len(set(full)) == 50
    assert C.stream_seeds(C.HELDOUT_STREAM, "jit-n12-120", 20, start=30) == full[30:]
    assert full != C.stream_seeds(C.HELDOUT_STREAM, "jit-n12-180", 50)
    assert full != C.stream_seeds(C.VAL_STREAM, "jit-n12-120", 50)
    assert C.trial_seed(C.HELDOUT_STREAM, "jit-n12-120", 7) == full[7]
    # a seed repeated within a stream is skipped, so its episodes stay distinct
    raw = [5, 7, 5, 9, 7, 11]
    monkeypatch.setattr(C, "trial_seed", lambda stream, cell, index: raw[index])
    assert C.stream_seeds(C.VAL_STREAM, "x", 4) == (5, 7, 9, 11)
    assert C.stream_seeds(C.VAL_STREAM, "x", 2, start=2) == (9, 11)


def test_a_training_run_draws_its_familys_cells_from_its_own_stream():
    drawn = [C.train_episode(4, "jittery", e) for e in range(400)]
    counts = collections.Counter(cell.name for cell, _ in drawn)
    assert set(counts) == {c.name for c in C.FAMILIES["jittery"]}
    assert min(counts.values()) > 60
    assert drawn[17] == C.train_episode(4, "jittery", 17)
    assert drawn != [C.train_episode(5, "jittery", e) for e in range(400)]
    assert {C.seed_kind(seed) for _, seed in drawn} == {"train"}
    assert {cell.family for cell, _ in (C.train_episode(4, "clean", e) for e in range(50))} == {
        "clean"}
    with pytest.raises(ValueError, match="family"):
        C.train_episode(4, "noisy", 0)


# --------------------------------------------------------------------------- #
# Study 5.6's cells (decision 6 (a); critic A3; resolution R22)
# --------------------------------------------------------------------------- #

# >>> re-pin pins: scripts/exp5/repin.py rewrites these with the cells' re-pin
# block (experiments/ferrysim/cells.py), from what it measures; they pin it.
#: The families' hashes (a manifest's ``cell_family_sha256``): ``jittery``,
#: ``clean`` and ``jittery56``.
JITTERY_SHA256 = "32b5cb6bc119f1e2b13423dd178cef81c4f7032f24bd199b732f53bd296d3e91"
CLEAN_SHA256 = "76955c9b637ab875c761bf0ce21181ce1983d9c5df02b6183e74abbfc2915902"
JITTERY56_SHA256 = "0079de11cdbe8b3fc71f7f6bdd1cf1d859e600031dd625b7996b35b7767fac66"
#: Study 5.6's lags (s) by Study 5.5 cell, and their periods (s): the quarter and
#: half at the stress budget, then at the knee.
LAGS_S = {"jit-n12-120": 26, "jit-n12-180": 34}
P_C_S = (104, 52, 136, 68)
#: The lags' sample: (lags, pooled median to 3 places) per Study 5.5 cell.
LAG_SAMPLE = {"jit-n12-120": (1397, 26.418), "jit-n12-180": (397, 34.106)}
#: The ratio check's bounds by budget (stress, knee): the 0.5 and 99.5 % points
#: of a RATIO_EPISODES-episode median in the lag's 200-episode measurement
#: (episodes resampled whole, since one layout's lags move together), as a
#: multiple of the measured median, widened by 5 % for the lag's rounding to the
#: second and its move with P_c, and rounded outward. At the stand-ins: 0.86 to
#: 1.17 x at 120 s and 0.70 to 1.38 x at 180 s before widening.
RATIO_BOUNDS_BY_BUDGET = ((0.81, 1.23), (0.66, 1.45))
# <<< re-pin pins

#: Study 5.6's cells: the name, the Study 5.5 cell it copies, and its P_c (s).
STUDY_5_6 = (("jit-n12-120-q", "jit-n12-120", float(P_C_S[0])),
             ("jit-n12-120-h", "jit-n12-120", float(P_C_S[1])),
             ("jit-n12-180-q", "jit-n12-180", float(P_C_S[2])),
             ("jit-n12-180-h", "jit-n12-180", float(P_C_S[3])))

#: The ratio check: FX's median lag / P_c over the first RATIO_EPISODES
#: validation episodes of a Study 5.6 cell lies within RATIO_BOUNDS of its
#: target (0.25 or 0.5). 24 is the fewest episodes probed at which the
#: stand-ins' 180 s cells' 99 % ranges of lag / P_c stayed apart (0.17 to 0.35
#: and 0.35 to 0.69); the re-pin checks the ranges stay apart at the new budgets.
RATIO_EPISODES = 24
RATIO_BOUNDS = dict(zip(C.BUDGETS_S[12], RATIO_BOUNDS_BY_BUDGET))


def test_study_5_6s_cells_are_study_5_5s_at_a_quarter_and_a_half_of_the_lag():
    """P_c = 4 x and 2 x the lag pinned from FX's median arrival-to-arrival time on
    the validation stream of the matching Study 5.5 cell; otherwise each is that
    Study 5.5 cell, the S* tool's cap at its own period included. The control is
    the clean N = 12 cells; the cells keep out of decision 3's, and fly their own
    seed streams."""
    from experiments.analysis.age_cap_s_star import s_star_report

    assert C.STUDY_5_6_LAGS_S == LAGS_S
    assert (C.P_C_QUARTER_STRESS_S, C.P_C_HALF_STRESS_S, C.P_C_QUARTER_KNEE_S,
            C.P_C_HALF_KNEE_S) == P_C_S
    assert [c.name for c in C.STUDY_5_6_CELLS] == [name for name, _, _ in STUDY_5_6]
    for (name, base, period), cell in zip(STUDY_5_6, C.STUDY_5_6_CELLS):
        assert period == {"q": 4, "h": 2}[name[-1]] * C.STUDY_5_6_LAGS_S[base]
        assert C.cell_named(name) is cell
        assert cell == dataclasses.replace(C.cell_named(base), name=name,
                                           interference_period_s=period)
        assert cell.driver_settings() == dict(
            C.cell_named(base).driver_settings(),
            ferry_physics={"contact_regime": "jittery", "interference_period_s": period})
        # a label by the spec's rule (ASCII, at most 28 characters, no "__"), and a
        # plain path component, since it names a kept trace's directory
        assert name.isascii() and len(name) <= 28 and "__" not in name
        assert E._path_part(name, "the cell's name") == name
        driver = Exp4Driver(mission_clock="sim", realism=True, contact_band="wide",
                            payload_bytes=1_000_000,
                            ferry_physics=cell.driver_settings()["ferry_physics"])
        report = s_star_report(driver, n_devices=12, budgets=C.BUDGETS_S[12],
                               regime="jittery", families=("F",))
        assert max(2, report.s("F")) == cell.cap_s == C.CAP_S[(12, "jittery")]
    assert not set(C.STUDY_5_6_CELLS) & set(C.CELLS)
    assert C.STUDY_5_6_CONTROL_CELLS == (C.cell_named("cln-n12-120"), C.cell_named("cln-n12-180"))
    C.check_disjoint([(stream, C.stream_seeds(stream, c.name, count))
                      for c in C.CELLS + C.STUDY_5_6_CELLS
                      for stream, count in ((C.VAL_STREAM, 200), (C.HELDOUT_STREAM, 1000))])
    assert C.stream_seeds(C.VAL_STREAM, "jit-n12-120-q", 50) != C.stream_seeds(
        C.VAL_STREAM, "jit-n12-120", 50)


def test_jittery56_adds_study_5_6s_cells_and_moves_no_other_family():
    """The jittery and clean families, and their hashes, are as they were; the
    second jittery family is the first and Study 5.6's four cells, and a run
    over it draws all eight from its own stream. Study 5.5 reads its two cells
    by name, since Study 5.6's are decision-rich N = 12 jittery cells too."""
    assert C.family_sha256("jittery") == JITTERY_SHA256
    assert C.family_sha256("clean") == CLEAN_SHA256
    assert [c.name for c in C.FAMILIES["jittery"]] == [
        "jit-n6-45", "jit-n6-90", "jit-n12-120", "jit-n12-180"]
    assert [c.name for c in C.FAMILIES["clean"]] == ["cln-n12-120", "cln-n12-180"]
    # The Exp 5 addendum's scale family (Study 5.11 (c)) follows them; it moves
    # none of them (tests/unit/test_exp5_ferrysim_scale.py).
    assert list(C.FAMILIES) == ["jittery", "clean", "jittery56", "scale"]
    assert C.FAMILIES["jittery56"] == C.FAMILIES["jittery"] + C.STUDY_5_6_CELLS
    assert C.family_sha256("jittery56") == JITTERY56_SHA256
    # the jittery regime's one score flies them, whichever family it practised over
    assert {c.family for c in C.FAMILIES["jittery56"]} == {"jittery"}
    assert [c.name for c in C.STUDY_5_5_CELLS] == ["jit-n12-120", "jit-n12-180"]
    drawn = [C.train_episode(4, "jittery56", e) for e in range(800)]
    counts = collections.Counter(cell.name for cell, _ in drawn)
    assert set(counts) == {c.name for c in C.FAMILIES["jittery56"]}
    assert min(counts.values()) > 60
    assert {C.seed_kind(seed) for _, seed in drawn} == {"train"}
    assert drawn[:50] != [C.train_episode(4, "jittery", e) for e in range(50)]


def test_arrival_lags_run_from_one_pass_1_arrival_to_the_next_in_a_sortie():
    """Critic A3's lag: the dwell at a stop and the leg to the next stop together;
    a sortie's last stop, whose span runs to the upload's end, starts none, and
    no lag crosses two sorties."""
    def sortie(*arrivals):
        return SimpleNamespace(stops=tuple(SimpleNamespace(t_s=t) for t in arrivals))

    assert C.arrival_lags([sortie(10.0, 25.0, 47.5), sortie(), sortie(90.0),
                           sortie(100.0, 130.0)]) == [15.0, 22.5, 30.0]
    cell, index = _qualifying(                   # an episode with two stops in a sortie
        lambda c, i: len(C.arrival_lags(_episode(c.name, i, "FX").sorties)) >= 2)
    ep = _episode(cell.name, index, "FX")
    lags = C.arrival_lags(ep.sorties)
    assert len(lags) >= 2
    assert lags == [s.stops[k].t_next_s - s.stops[k].t_s
                    for s in ep.sorties for k in range(len(s.stops) - 1)]
    assert all(b.t_s - a.end_s < b.t_s - a.t_s
               for s in ep.sorties for a, b in zip(s.stops, s.stops[1:]))


def test_the_lag_is_fxs_arrival_lags_pooled_over_its_first_validation_episodes():
    """R22's measurement in the repository (the repair round's RB-2): the FX arm
    flies the cell's first validation episodes, as an evaluation flies them, and
    every arrival-to-arrival lag of every sortie is pooled in episode order; the
    lag is their median, before it is rounded. 200 episodes by default, the
    sample ``cells.STUDY_5_6_LAGS_S`` was pinned on."""
    assert EV.LAG_EPISODES == 200
    flown = [lag for i in range(2) for lag in C.arrival_lags(
        _episode("jit-n12-120", i, "FX").sorties)]
    assert EV.fx_lags("jit-n12-120", episodes=2) == flown and len(flown) >= 2
    assert EV.fx_lag_median(N12, episodes=2) == statistics.median(flown)
    with pytest.raises(ValueError, match="no lag"):
        EV.fx_lag_median(N12, episodes=0)


@pytest.mark.slow
@pytest.mark.parametrize("name", [c.name for c in C.STUDY_5_5_CELLS])
def test_the_pinned_lags_are_measured_again_on_their_own_sample(name):
    """Resolution R22's pinned lags (RB-1, RB-2): FX's pooled median over the first
    200 validation episodes of each Study 5.5 cell, flown in worker processes, is
    the measurement the lags were pinned on, lag for lag in count, and rounds to
    the pinned integer. Slow: about 400 episodes in all."""
    lags = EV.fx_lags(name, workers=4)
    median = statistics.median(lags)
    assert (len(lags), round(median, 3)) == LAG_SAMPLE[name]
    assert round(median) == C.STUDY_5_6_LAGS_S[name] == LAGS_S[name]


@pytest.mark.slow
@pytest.mark.parametrize("name", [name for name, _, _ in STUDY_5_6])
def test_fx_flies_study_5_6s_cells_at_a_quarter_and_a_half_of_their_period(name):
    """FX's median lag / P_c on the cell's own first validation episodes is near
    its target, within RATIO_BOUNDS, which keep the quarter cell off the half
    cell's target and the half cell off 0.75 and 1, A3's aliasing points; and
    the cell's period is the channel's. Slow: 96 episodes, about 40 s."""
    cell = C.cell_named(name)
    target, other = {"q": (0.25, 0.5), "h": (0.5, 0.25)}[name[-1]]
    low, high = (target * f for f in RATIO_BOUNDS[cell.budget_s])
    assert not low <= other <= high and high < 0.75
    lags, periods = [], []
    for i, seed in enumerate(C.stream_seeds(C.VAL_STREAM, name, RATIO_EPISODES)):
        ep = E.run_episode(cell, seed, E.Policy.of_arm("FX"), trial_index=i, hooks=[
            lambda service: periods.append(
                service.supervisor.ferry.contact_channel.interference_period_s)])
        lags.extend(C.arrival_lags(ep.sorties))
    assert periods == [cell.interference_period_s] * RATIO_EPISODES
    ratio = statistics.median(lags) / cell.interference_period_s
    assert low <= ratio <= high, (name, ratio, len(lags))


def test_the_lag_checks_that_fly_tens_of_episodes_are_marked_slow():
    """The repair round's RB-4: the ratio check flies 96 FX episodes (about 40 s)
    and the pinned lags' check 400, so both carry the repository's ``slow`` mark
    (pytest.ini) and a ``-m "not slow"`` run leaves them out."""
    for test in (test_fx_flies_study_5_6s_cells_at_a_quarter_and_a_half_of_their_period,
                 test_the_pinned_lags_are_measured_again_on_their_own_sample):
        assert "slow" in [mark.name for mark in test.pytestmark], test.__name__


# --------------------------------------------------------------------------- #
# Parity: the copy, the tap and the real orchestrator
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["fx_45s", "fx_n12_120s"])
def test_a_ferrysim_fx_episode_with_the_stub_is_ug5s_oracle(name):
    settings, cell = UG5.TRIALS[name]
    with UG5.flight_slot_spy() as calls:
        ep = E.run_episode_on(Exp4Driver(**settings), cell, E.Policy.of_arm("FX"),
                              device_model=IP.DEVICE_MODEL_STUB, keep_case=True,
                              case_settings=settings)
    case = dict(ep.case)
    case["flight_slot"] = [UG5.slot_record(c) for c in calls]
    golden = UG5.load_golden()["cases"][name]
    for part in ("inputs",) + UG5.PARTS:
        assert UG5.compare_part(part, golden[part], case[part]) == [], part
    # every mission was read, one decision per Pass-1 stop flown
    completed = [e for e in ep.case["mission_completed"]["exp4-mule"]]
    assert len(ep.sorties) == cell.params["n_missions"] == len(completed)
    assert [len(s.stops) for s in ep.sorties] == [len(e["pass_1_flown"]) for e in completed]


def test_the_copy_with_no_hooks_is_ug4s_trial():
    settings, cell = UG4.TRIALS["h1_replan_trim_wide"]
    run = IP.run_trial(Exp4Driver(**settings), cell)
    case = IP.case_of(settings, cell, run.row, run.orch)
    golden = UG4.load_golden()["cases"]["h1_replan_trim_wide"]
    for part in ("inputs",) + UG4.PARTS:
        assert _canon.diff(golden[part], case[part]) == [], part


def test_the_equal_device_model_changes_only_the_examples_and_scores():
    settings, cell = UG5.TRIALS["fx_45s"]
    stub = E.run_episode_on(Exp4Driver(**settings), cell, E.Policy.of_arm("FX"),
                            device_model=IP.DEVICE_MODEL_STUB)
    equal = E.run_episode_on(Exp4Driver(**settings), cell, E.Policy.of_arm("FX"))
    # the same flight (the stub's n_i and scores steer nothing in plan mode) ...
    assert ([[(s.t_s, s.devices, s.collected) for s in x.stops] for x in stub.sorties]
            == [[(s.t_s, s.devices, s.collected) for s in x.stops] for x in equal.sorties])
    # ... merged at the equal shard's weight, which the stub's draws make vary
    equal_w = {w for x in equal.sorties for s in x.stops for w in s.w}
    stub_w = {w for x in stub.sorties for s in x.stops for w in s.w}
    assert equal_w == {float(R.EQUAL_SHARD_EXAMPLES)}
    assert len(stub_w) > 1
    with pytest.raises(ValueError, match="device_model"):
        IP.RoleHooks(device_model="real")


#: What a real process's wall clock and OS decide: the envelope stamps and
#: durations, the ports, the planner's wall time and the mule's metrics timer
#: of mission wall time (UG5's probe of the real orchestrator), and each flight
#: decision's wall time (Exp 5 addendum, Study 5.11 (a)).
_WALL_KEYS = {"ts", "duration_s", "rf_port", "dock_port", "mule_rf_port", "port",
              "timer.mission_duration_s", "plan_wall_s", *IP.DECISION_WALL_FIELDS}
#: The row's wall column: the mean mission duration, timed by the harness clock
#: in process and by the wall clock in a real process.
_ROW_WALL = {"mission_duration_s_mean"}
#: The row's device-serve columns, built from the devices' ``device_served``
#: events, which only the device process's service loop logs: FerrySim runs
#: none (resolution R25), so its row holds :data:`FERRYSIM_SERVES`, UG4's
#: harness artifacts, in every trial; a real trial's come from its serves.
_ROW_SERVES = {"coverage", "jains_fairness", "participation_entropy"}
FERRYSIM_SERVES = {"coverage": 0.0, "jains_fairness": 1.0, "participation_entropy": 0.0}


def _masked(value, keys):
    if isinstance(value, dict):
        return {k: ("<masked>" if k in keys else _masked(v, keys)) for k, v in value.items()}
    if isinstance(value, list):
        return [_masked(v, keys) for v in value]
    return value


def _device_events(case):
    """The names of every event a canonical case's devices logged."""
    return {e["event"] for events in case["device_events"].values() for e in events}


@pytest.mark.slow
def test_a_stub_fx_trial_through_the_real_orchestrator_is_ferrysims(tmp_path):
    """Critic B8: one stub FX trial run through the real orchestrator (real
    processes and TCP) against FerrySim's run of it, event by event: the row,
    the configs and every mule and cluster event, bar what the wall clock and
    the OS decide (:data:`_WALL_KEYS`, :data:`_ROW_WALL`) and the row's
    device-serve columns (:data:`_ROW_SERVES`). Those come from the devices'
    serves: the real devices log ``device_served``, FerrySim's only
    ``device_ready`` (resolution R25), so FerrySim's are the artifacts."""
    settings, cell = UG5.TRIALS["fx_n12_120s"]
    row = dict(Exp4Driver(**settings, trace_root=tmp_path, trial_budget_s=300.0).run_trial(cell))
    trace = tmp_path / trace_dir_name(cell)
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(trace.iterdir())
             if p.suffix in (".json", ".jsonl") and p.name != "trial_status.json"}
    devices = [SimpleNamespace(device_id=json.loads(text)["device_id"])
               for f, text in files.items() if f.startswith("device-") and f.endswith(".json")]
    real = IP.case_of(settings, cell, row, SimpleNamespace(
        topology=SimpleNamespace(devices=devices), files=files, exit_codes={}))
    sim = E.run_episode_on(Exp4Driver(**settings), cell, E.Policy.of_arm("FX"),
                           device_model=IP.DEVICE_MODEL_STUB, keep_case=True,
                           case_settings=settings).case
    for part in ("row", "configs", "mule_ready", "mission_started", "mission_completed",
                 "cluster_events"):
        keys = _WALL_KEYS | (_ROW_WALL | _ROW_SERVES if part == "row" else set())
        assert _canon.diff(_masked(sim[part], keys), _masked(real[part], keys)) == [], part
    for part in ("names", "other"):
        assert _canon.diff(_masked(sim["mule_events"][part], _WALL_KEYS),
                           _masked(real["mule_events"][part], _WALL_KEYS)) == [], part
    assert len(real["mission_completed"]["exp4-mule"]) == 4
    # the serve columns: the real devices served, FerrySim's only announced
    # themselves, so FerrySim's row holds the artifacts and the real row does not
    assert set(real["device_events"]) == set(sim["device_events"])
    assert _device_events(real) >= {"device_ready", "device_served"}
    assert _device_events(sim) == {"device_ready"}
    assert {k: _f(sim["row"][k]) for k in _ROW_SERVES} == FERRYSIM_SERVES
    assert _f(real["row"]["coverage"]) > 0.0 and _f(real["row"]["participation_entropy"]) > 0.0


@pytest.mark.parametrize("policy, device_model", [
    (E.Policy.of_arm("FX"), IP.DEVICE_MODEL_EQUAL),
    (E.Policy.of_arm("F"), IP.DEVICE_MODEL_STUB),
    (E.Policy.scripted("greedy_1"), IP.DEVICE_MODEL_EQUAL),
], ids=["FX-equal", "F-stub", "greedy_1-equal"])
def test_a_ferrysim_rows_device_serve_columns_are_harness_artifacts(policy, device_model,
                                                                    tmp_path):
    """Resolution R25 (the final check's FS-1): FerrySim runs no device service
    loop, so its devices log only ``device_ready``, and an episode's row holds
    the device-serve columns' harness artifacts (:data:`FERRYSIM_SERVES`),
    whatever flies it and under either device model; so does the trace
    scorer's row of its kept trace, which reads the same device events. No
    study reads them; the real values come only from real processes (the B8
    test above)."""
    from experiments.analysis.traces_scorer import score_traces

    cell = C.cell_named("jit-n6-45")
    ep = E.run_episode(cell, _seed(cell), policy, device_model=device_model, keep_case=True,
                       driver_overrides={"trace_root": tmp_path})
    assert ep.sorties and {k: ep.row[k] for k in _ROW_SERVES} == FERRYSIM_SERVES
    assert {k: _f(ep.case["row"][k]) for k in _ROW_SERVES} == FERRYSIM_SERVES
    assert len(ep.case["device_events"]) == cell.n_devices
    assert _device_events(ep.case) == {"device_ready"}
    (scored,) = score_traces(tmp_path / cell.name / policy.label)
    row = scored.to_row()
    assert {k: row[k] for k in _ROW_SERVES} == FERRYSIM_SERVES


# --------------------------------------------------------------------------- #
# Episodes (T3) and how they are read
# --------------------------------------------------------------------------- #

def _sorties(ep):
    return [s.to_json() for s in ep.sorties]


@pytest.mark.parametrize("policy, trainer", [
    (E.Policy.of_arm("FX"), None),
    (E.Policy.scripted("greedy_1"), E.Trainer(epsilon=0.5, rng_seed=11)),
])
def test_t3_an_episode_is_a_function_of_its_inputs(policy, trainer):
    cell, index = _qualifying(lambda c, i: _decisions(c, i) >= 6)
    seed = _seed(cell, index)
    a, b = (E.run_episode(cell, seed, policy, trainer=trainer, keep_case=True)
            for _ in range(2))
    assert a.summary() == b.summary()
    assert [list(s) for s in a.rewards] == [list(s) for s in b.rewards]
    assert a.ret == b.ret and a.pair_records == b.pair_records
    for part in a.case:
        assert _canon.diff(a.case[part], b.case[part]) == [], part
    # the planner's wall time is the only wall stamp, and it is masked
    for e in a.case["mission_completed"]["exp4-mule"]:
        assert e["plan_wall_s"] == IP.WALL_TOKEN
    other = E.run_episode(cell, _seed(cell, index + 1), policy, trainer=trainer)
    assert _sorties(other) != _sorties(a) and other.ret != a.ret
    if trainer is not None:
        explored = E.run_episode(cell, seed, policy,
                                 trainer=dataclasses.replace(trainer, rng_seed=12))
        assert explored.pair_records != a.pair_records


def test_a_trainer_at_epsilon_0_flies_as_none_and_collects_each_missions_decisions():
    cell, index = _qualifying(lambda c, i: _decisions(c, i) >= 6)
    plain = E.run_episode(cell, _seed(cell, index), E.Policy.scripted("fx_pair"))
    sunk = []
    trained = E.run_episode(cell, _seed(cell, index), E.Policy.scripted("fx_pair"),
                            trainer=E.Trainer(),
                            sink=lambda steps, records: sunk.append((steps, records)))
    assert trained.summary() == plain.summary()
    assert len(trained.steps) == len(trained.sorties) == len(sunk)
    for steps, sortie, records, (sunk_steps, sunk_records) in zip(
            trained.steps, trained.sorties, trained.pair_records, sunk):
        assert len(steps) == len(sortie.stops) == len(records) == len(sunk_records)
        assert steps == sunk_steps
        for step, stop, record in zip(steps, sortie.stops, records):
            assert step.view.clock_s == stop.t_s == record["t_s"]
            assert tuple(map(str, step.view.arrival.devices)) == stop.devices
            assert step.choice.describe()["band"] == record["band"] == stop.band
    assert sum(len(s) for s in trained.steps) >= 6
    with pytest.raises(ValueError, match="trainer"):
        E.run_episode(N12, _seed(), E.Policy.of_arm("FX"), trainer=E.Trainer())
    with pytest.raises(ValueError, match="sink"):
        E.run_episode(N12, _seed(), E.Policy.scripted("fx_pair"), sink=lambda *a: None)


def test_the_trainers_reference_phase_flies_fx_pair():
    """In the reference phase (critic C4) a decision at ε 0 flies FX's pair wherever
    the mask admits it, so ``greedy_1`` under that trainer flies as ``fx_pair``, on
    an episode where ``greedy_1`` alone flies otherwise."""
    cell = TOY[1][0]
    seed = _seed(cell)
    fx_pair = E.run_episode(cell, seed, E.Policy.scripted("fx_pair"))
    greedy = E.run_episode(cell, seed, E.Policy.scripted("greedy_1"))
    around = E.run_episode(cell, seed, E.Policy.scripted("greedy_1"),
                           trainer=E.Trainer(epsilon=0.0, rng_seed=3, around_reference=True))
    assert _sorties(greedy) != _sorties(fx_pair)
    assert _sorties(around) == _sorties(fx_pair)
    assert all(record["agrees_fx"] for records in around.pair_records for record in records)
    assert {record["scorer"] for records in around.pair_records for record in records} == {
        "greedy_1"}


def test_stop_after_ends_the_trial_once_that_mission_has_closed():
    full = _episode("jit-n12-120", 0, "FX")
    part = E.run_episode(N12, _seed(), E.Policy.of_arm("FX"), stop_after=1)
    assert len(full.sorties) == 4 and len(part.sorties) == 2
    assert _sorties(part) == _sorties(full)[:2]


def test_a_pair_slot_episodes_records_are_ferrysims_reading_and_a_difference_raises():
    ep = _episode("jit-n12-120", 0, "fx_pair")
    k = next(i for i, s in enumerate(ep.sorties) if len(s.stops) >= 2)
    sortie, records = ep.sorties[k], ep.pair_records[k]
    E._check_pair_records(sortie, records)
    for field, change in (("t_next_s", lambda r: r["t_next_s"] + 1.0),
                          ("collected", lambda r: list(r["collected"])[:-1] or ["x"]),
                          ("terminal", lambda r: not r["terminal"]),
                          ("w", lambda r: [w + 1.0 for w in r["w"]] or [1.0])):
        bad = [dict(r) for r in records]
        bad[0][field] = change(bad[0])
        with pytest.raises(AssertionError, match=field):
            E._check_pair_records(sortie, bad)
    with pytest.raises(AssertionError, match="record"):
        E._check_pair_records(sortie, records[:-1])
    # arms carry no pair record; a slot carries one per Pass-1 stop flown
    assert set(_episode("jit-n12-120", 0, "FX").pair_records) == {None}
    assert [len(r) for r in ep.pair_records] == [len(s.stops) for s in ep.sorties]


def test_a_sorties_last_decision_ends_with_the_upload_or_at_the_landing_when_empty():
    """The mule closes a pair record at the end of the Pass-1 upload, or at the
    landing on the empty round (critic B2); FerrySim's reading agrees on both
    exits (the tap compares every stop with the mule's records), for every arm.
    UG5's 45 s trial, under the stub, flies three empty rounds of six missions."""
    settings, cell = UG5.TRIALS["fx_45s"]
    exits = collections.Counter()
    for policy in (E.Policy.scripted("fx_pair"), E.Policy.of_arm("FX")):
        ep = E.run_episode_on(Exp4Driver(**settings), cell, policy,
                              device_model=IP.DEVICE_MODEL_STUB)
        for sortie in ep.sorties:
            if not sortie.stops:
                continue
            last = sortie.stops[-1]
            assert last.terminal and last.end_s <= last.flight_end_s
            if sortie.empty:
                assert last.t_next_s == last.flight_end_s               # the landing
                assert not last.collected or set(last.w) == {0.0}
                exits["empty", policy.label] += 1
            else:
                assert last.t_next_s > last.flight_end_s                # the upload after it
                exits["upload", policy.label] += 1
    for label in ("fx_pair", "FX"):
        assert exits["empty", label] >= 2 and exits["upload", label] >= 2


def test_a_reading_that_disagrees_with_the_mules_records_stops_the_episode(monkeypatch):
    real = E.raw_merge_weights
    monkeypatch.setattr(E, "raw_merge_weights",
                        lambda spec, agg, report: {d: 2 * w for d, w in real(spec, agg,
                                                                              report).items()})
    E.run_episode(N12, _seed(), E.Policy.of_arm("FX"))          # an arm has no record
    with pytest.raises(AssertionError, match="the mule's record and FerrySim's reading differ"):
        E.run_episode(N12, _seed(), E.Policy.scripted("fx_pair"))


def test_ferrysims_reading_of_a_flight_is_what_its_trace_records():
    """The plan and the sessions as the reward reads them, against the kept trace
    itself (``mission_completed`` and the mule's config): N is the pair view's and
    the plan's demand with any insert; the demand, the committed set U is taken
    over, the coverage weights and T are the plan's; a stop collects its members
    whose Pass-1 outcome was clean; the terminal term is c_cov U over those; the
    last leg ends at the dock the sortie took off from; and the powers are the
    ones the mule's energy ledger was charged at."""
    docks = []
    ep = E.run_episode(N12, _seed(), E.Policy.scripted("fx_pair"), trainer=E.Trainer(),
                       keep_case=True, hooks=[lambda service: docks.append(
                           tuple(service.supervisor.ferry.flight.dock))])
    hand = ep.rescored(R.HAND)
    t_nom = _f(ep.case["configs"]["mule-exp4-mule.json"]["t_nom_s"])
    done = ep.case["mission_completed"]["exp4-mule"]
    assert len(done) == len(ep.sorties) == len(ep.steps) == 4
    for sortie, event, steps, terms, hand_terms in zip(ep.sorties, done, ep.steps, ep.rewards,
                                                       hand.rewards):
        plan = event["plan"]
        weights = _floats(plan["weights"])
        inserted = {d for insert in event["inserts"] for d in insert["devices"]}
        clean = {o["device"] for o in event["pass_1_outcomes"] if o["outcome"] == "clean"}
        assert {step.view.demand for step in steps} == {sortie.n_demand}
        assert sortie.n_demand == len(set(plan["demand"]) | inserted)
        assert list(sortie.demand) == plan["demand"]
        assert sorted(sortie.committed) == sorted(plan["served"])
        assert dict(sortie.coverage_weights) == weights
        assert sortie.t_ref_s == _f(plan["score"]["t_ref_s"]) == t_nom
        assert [list(stop.collected) for stop in sortie.stops] == [
            [d for d in stop.devices if d in clean] for stop in sortie.stops]
        missed = sum(weights[d] for d in plan["served"] if d not in clean)
        assert terms[-1].coverage == pytest.approx(
            R.DERIVED.c_cov * missed / sum(weights[d] for d in plan["demand"]), rel=1e-12)
        takeoff = tuple(_f(c) for c in event["pass_1_flown"][0]["depart_pose"])
        last = sortie.stops[-1]
        assert last.next_position == docks[0] == takeoff
        assert hand_terms[-1].distance == pytest.approx(
            R.HAND_PER_METRE * math.dist(last.position, takeoff) / R.HAND_SCALE, rel=1e-12)
        ledger = _floats(event["sim_ledger"])
        assert _f(event["energy_j"]) == pytest.approx(
            sortie.p_move_w * (ledger["transit"] + ledger["return"])
            + sortie.p_hover_w * (ledger["dwell"] + ledger["listen"]), rel=1e-9)
    # an episode where a misreading would show: the plan commits to fewer devices
    # than it demands, and weighs them unequally
    assert any(len(s.committed) < len(s.demand) for s in ep.sorties)
    assert any(len(set(s.coverage_weights.values())) > 1 for s in ep.sorties)


def _kept_arrivals(trace):
    """Each mission's Pass-1 arrivals, as a kept trace's mule events record them."""
    (mule,) = sorted(trace.glob("mule-*.jsonl"))
    events = [json.loads(line) for line in mule.read_text(encoding="utf-8").splitlines()]
    return [[stop["arrival_s"] for stop in e["pass_1_flown"]] for e in events
            if e.get("event") == "mission_completed"]


def test_a_kept_trace_is_filed_under_its_cell_and_policy(tmp_path):
    """The driver names a trace by the runner's cell id, the arm, the trial and the
    seed, which the FX arm and every pair slot of one seed share (each flies FX's
    configuration), as do FerrySim cells of one size; under one trace root each
    episode's trace is kept whole, under its FerrySim cell and policy."""
    def differs(cell, index):                    # FX and committed_pair fly differently
        seed = _seed(cell, index)
        return _sorties(E.run_episode(cell, seed, E.Policy.of_arm("FX"))) != _sorties(
            E.run_episode(cell, seed, E.Policy.scripted("committed_pair")))

    rich, index = _qualifying(differs)
    other_cell = next(c for c in C.STUDY_5_5_CELLS if c != rich)
    seed = _seed(rich, index)
    kept = {}
    for cell, policy in ((rich, E.Policy.of_arm("FX")),
                         (rich, E.Policy.scripted("committed_pair")),
                         (other_cell, E.Policy.of_arm("FX"))):
        ep = E.run_episode(cell, seed, policy, trial_index=1, keep_case=True,
                           driver_overrides={"trace_root": tmp_path})
        root = tmp_path / cell.name / policy.label
        assert ep.case["inputs"]["settings"]["trace_root"] == str(root)
        (trace,) = root.iterdir()
        assert trace.name == trace_dir_name(cell.cell("FX", seed, 1))
        kept[cell.name, policy.label] = _kept_arrivals(trace)
        assert kept[cell.name, policy.label] == [[s.t_s for s in x.stops] for x in ep.sorties]
    assert kept[rich.name, "FX"] != kept[rich.name, "committed_pair"]
    assert len(list(tmp_path.rglob("mule-*.jsonl"))) == 3
    unsafe = E.Policy(label="greedy/1", scorer=E.Policy.scripted("greedy_1").scorer)
    with pytest.raises(ValueError, match="plain path component"):
        E.run_episode(N12, seed, unsafe, driver_overrides={"trace_root": tmp_path})


class _Refusing:
    name = "refusing"

    def score(self, view, *, mask):
        raise RuntimeError("this scorer refuses to rank")


def test_a_mule_that_fails_raises_with_its_traces_reasons():
    with pytest.raises(IP.TrialFailure, match="refuses to rank") as failure:
        E.run_episode(N12, _seed(), E.Policy(label="refusing", scorer=_Refusing))
    assert failure.value.reasons and "refuses to rank" in failure.value.reasons[0]


def test_the_policies_and_the_references():
    refs = E.reference_policies()
    assert [p.label for p in refs] == ["FX", "F", "fx_pair", "committed_pair", "hyb", "greedy_1"]
    assert [p.pair_slot for p in refs] == [False, False, True, True, True, True]
    assert all(p.arm == "FX" for p in refs if p.pair_slot)
    with pytest.raises(ValueError, match="plan arm"):
        E.Policy(label="x", arm="H1", scorer=refs[2].scorer)
    with pytest.raises(ValueError, match="scripted"):
        E.Policy.scripted("nearest")
    with pytest.raises(ValueError, match="arm"):
        E.run_episode_on(Exp4Driver(**N12.driver_settings()), N12.cell("F", _seed()),
                         E.Policy.of_arm("FX"))


# --------------------------------------------------------------------------- #
# T4: the reward is the merge's weight
# --------------------------------------------------------------------------- #

def _sub(did, n, basis, loss=0.2, form=UPDATE_FORM_DELTA):
    return GradientSubmission(device_id=DeviceID(did), mule_id=MuleID("m"), mission_round=3,
                              delta_theta=[np.full((2,), 0.1 * (1 + len(did)))],
                              num_examples=n, submitted_at=0.0, local_loss=loss,
                              basis_version=basis, update_form=form)


def _stop(t, collected, w, *, devices=None, terminal=False, t_next=None, dropped=(),
          targets=None, dwell=2.0, listen=0.0, flight_end=None, pos=(0.0, 0.0, 0.0),
          next_pos=(0.0, 0.0, 0.0)):
    devices = tuple(devices if devices is not None else collected)
    end = t + dwell + listen
    t_next = t + 20.0 if t_next is None else t_next
    return R.StopRecord(
        t_s=t, devices=devices, targets=tuple(targets if targets is not None else devices),
        collected=tuple(collected), w=tuple(w), uplink_dropped=tuple(dropped),
        t_next_s=t_next, terminal=terminal, end_s=end,
        flight_end_s=t_next if flight_end is None else flight_end, dwell_s=dwell,
        listen_s=listen, position=pos, next_position=next_pos)


def _sortie(stops, *, n=6, committed=None, weights=None, availability=None, t_ref=200.0,
            demand=None):
    demand = tuple(demand or [f"d{i}" for i in range(n)])
    return R.SortieRecord(
        mission_round=1, stops=tuple(stops), n_demand=n, demand=demand,
        committed=tuple(committed if committed is not None else demand),
        coverage_weights=weights or {d: 1.0 for d in demand}, t_ref_s=t_ref,
        availability=availability or {}, p_hover_w=168.5, p_move_w=143.6)


@pytest.mark.parametrize("rule, params", [
    ("agg:cutoff", {"a_max": 2}),
    ("agg:cutoff", {"a_max": 2, "value": "loss", "hinge_a": 0.5, "hinge_b": 1.0}),
    ("agg:asynchfl", {"decay": 0.3}),
])
def test_t4_a_hand_worked_missions_gains_are_the_merges_weights(rule, params):
    spec = AggregationSpec.from_config(rule, params)
    subs = [_sub("d0", 8, 5), _sub("d1", 12, 4, loss=0.5), _sub("d2", 6, 2),
            _sub("d3", 10, 5, loss=0.4), _sub("d4", 0, 5), _sub("d5", 9, 3)]
    caps = {DeviceID("d2"): 1}                       # age 3 past its cap: weight 0
    uw = update_weights(spec, [s for s in subs if s.num_examples > 0], base_version=5,
                        age_caps=caps)
    agg = merge_on_mule(spec, mule_id=MuleID("m"), mission_round=3, submissions=subs,
                        base_version=5, age_caps=caps)
    weights = R.raw_merge_weights(spec, agg, None)
    raw = {str(u.device_id): u.weight for u in uw}
    assert weights.keys() == {d for d, w in raw.items() if w > 0}
    for d, w in weights.items():
        assert w == pytest.approx(raw[d], rel=1e-12)
    from hermes.mule.mule_main import _raw_merge_weights

    assert weights == {str(d): w for d, w in _raw_merge_weights(spec, agg, None).items()}
    stops = [("d0", "d1"), ("d2", "d3", "d4"), ("d5",)]
    sortie = _sortie([
        _stop(100.0 + 20 * k, s, [weights.get(d, 0.0) for d in s], terminal=k == 2)
        for k, s in enumerate(stops)], n=7, demand=[f"d{i}" for i in range(7)])
    terms = R.sortie_rewards(R.RewardSpec(c_t=0.0, c_cov=0.0), sortie)
    n_ref_n = R.EQUAL_SHARD_EXAMPLES * 7
    total = sum(t.gain for t in terms) * n_ref_n
    assert total == pytest.approx(sum(u.weight for u in uw), rel=1e-12)
    assert total == pytest.approx(
        sum(w * agg.weight_mass for w in agg.device_weights), rel=1e-12)
    for t, s in zip(terms, stops):                    # each update credited to its stop
        assert t.gain * n_ref_n == pytest.approx(sum(raw.get(d, 0.0) for d in s), rel=1e-12)


def test_t4_plain_weights_are_the_collected_example_counts():
    spec = AggregationSpec.from_config("agg:plain", {})
    subs = [_sub(d, n, 1, form=UPDATE_FORM_WEIGHTS) for d, n in (("a", 4), ("b", 15), ("c", 7))]
    agg = merge_on_mule(spec, mule_id=MuleID("m"), mission_round=3, submissions=subs,
                        base_version=1)
    lines = [SimpleNamespace(device_id=DeviceID(d), outcome=MissionOutcome.CLEAN, num_examples=n)
             for d, n in (("a", 4), ("b", 15), ("c", 7))]
    report = SimpleNamespace(lines=lines + [SimpleNamespace(
        device_id=DeviceID("z"), outcome=MissionOutcome.TIMEOUT, num_examples=99)])
    assert R.raw_merge_weights(spec, agg, report) == {"a": 4.0, "b": 15.0, "c": 7.0}
    assert R.raw_merge_weights(spec, None, report) == {}
    from hermes.mule.mule_main import _raw_merge_weights

    assert R.raw_merge_weights(spec, agg, report) == {
        str(d): w for d, w in _raw_merge_weights(spec, agg, report).items()}


def test_t4_a_ferrysim_mission_at_k1_gains_its_collected_examples():
    """Under the stack's stub (n_i drawn in [4, 15]) a stop's G_k n_ref N is the sum
    of the n_i it collected, the merge's own weights at age 0, and the mule's
    closed record holds the same weights."""
    results = []

    def keep(service):
        run_one = service.supervisor.run_one_mission

        def kept():
            result = run_one()
            results.append(result)
            return result
        service.supervisor.run_one_mission = kept

    n_ref = R.STUB_MEAN_EXAMPLES
    spec = R.RewardSpec(c_t=0.0, c_cov=0.0, n_ref=n_ref)
    ep = E.run_episode(N12, _seed(), E.Policy.scripted("fx_pair"), reward=spec,
                       device_model=IP.DEVICE_MODEL_STUB, hooks=[keep])
    assert len(results) == len(ep.sorties) == 4
    checked = set()
    for result, sortie, terms, records in zip(results, ep.sorties, ep.rewards, ep.pair_records):
        n_i = {str(line.device_id): line.num_examples for line in
               (result.report.lines if result.report is not None else ())}
        agg = result.aggregate
        merged = {} if agg is None else {
            str(d): w * agg.weight_mass for d, w in zip(agg.contributing_devices,
                                                        agg.device_weights)}
        if agg is not None:
            assert set(agg.device_ages) <= {0}       # K = 1: every update is fresh
        for stop, term, record in zip(sortie.stops, terms, records):
            got = term.gain * n_ref * sortie.n_demand
            assert got == pytest.approx(sum(n_i[d] for d in stop.collected if d in merged))
            assert got == pytest.approx(sum(merged.get(d, 0.0) for d in stop.collected))
            assert list(record["w"]) == list(stop.w)
            checked.update(n_i[d] for d in stop.collected)
    assert len(checked) > 2                          # the n_i varied


# --------------------------------------------------------------------------- #
# The reward by hand
# --------------------------------------------------------------------------- #

def test_the_derived_reward_by_hand():
    spec = R.RewardSpec(c_t=0.1, c_e=0.5, c_cov=1.0)
    s1 = _stop(100.0, ("a",), (10.0,), devices=("a", "e"), dwell=5.0, listen=1.0, t_next=120.0)
    s2 = _stop(120.0, ("b", "c"), (10.0, 5.0), devices=("b", "c", "d"), dwell=8.0, terminal=True,
               t_next=141.0, flight_end=140.0)
    weights = {"a": 1.0, "b": 2.0, "c": 1.0, "d": 4.0, "e": 9.0, "f": 3.0}
    sortie = _sortie([s1, s2], n=4, demand=list(weights), weights=weights,
                     committed=["a", "b", "c", "d"])
    t1, t2 = R.sortie_rewards(spec, sortie)
    assert t1.gain == pytest.approx(10.0 / (10 * 4))
    assert t1.time == pytest.approx(0.1 * 20.0 / 200.0)
    assert t1.energy == pytest.approx(0.5 * (168.5 * 6.0 + 143.6 * 14.0) / (168.5 * 200.0))
    assert t1.coverage == 0.0 and t1.distance == 0.0
    assert t2.gain == pytest.approx(15.0 / 40.0)
    assert t2.time == pytest.approx(0.1 * 21.0 / 200.0)
    assert t2.energy == pytest.approx(0.5 * (168.5 * 8.0 + 143.6 * 12.0) / (168.5 * 200.0))
    assert t2.coverage == pytest.approx(4.0 / sum(weights.values()))     # d left uncollected
    assert R.sortie_return(spec, sortie).total == pytest.approx(t1.total + t2.total)
    assert t1.total == pytest.approx(t1.gain - t1.time - t1.energy)
    # the defaults: c_e = 0, so no energy term
    assert {t.energy for t in R.sortie_rewards(R.DERIVED, sortie)} == {0.0}
    # a sortie with no Pass-1 stop decides nothing; its shortfall is reported only
    empty = _sortie([], n=4, demand=list(weights), weights=weights, committed=["a", "b"])
    assert R.sortie_rewards(spec, empty) == ()
    assert empty.undecided_shortfall == pytest.approx(3.0 / 20.0)


def test_f_hand_and_e3s_bytes_by_hand():
    s1 = _stop(10.0, ("a", "b"), (10.0, 10.0), devices=("a", "b", "c"), t_next=40.0,
               pos=(0.0, 0.0, 0.0), next_pos=(30.0, 40.0, 0.0))
    s2 = _stop(40.0, (), (), devices=("d",), terminal=True, t_next=70.0,
               pos=(30.0, 40.0, 0.0), next_pos=(0.0, 0.0, 0.0))
    sortie = _sortie([s1, s2], n=5)
    h1, h2 = R.sortie_rewards(R.HAND, sortie)
    assert h1.total == pytest.approx((200 * 2 - 30.0 - 0.002 * 50.0) / 150)
    assert h2.total == pytest.approx((0 - 30.0 - 0.002 * 50.0) / 150)
    assert h2.coverage == 0.0
    b1, b2 = R.sortie_rewards(R.BYTES, sortie)
    assert (b1.total, b2.total) == (2 / 5, 0.0)
    with pytest.raises(ValueError, match="fixes its own weights"):
        R.RewardSpec(kind="hand")
    with pytest.raises(ValueError, match="kind"):
        R.RewardSpec(kind="paid")
    with pytest.raises(ValueError, match="n_ref"):
        R.RewardSpec(n_ref=0)
    assert R.RewardSpec.from_json(R.HAND.to_json()) == R.HAND
    grid = R.grid_specs()
    assert [(g.c_t, g.c_cov) for g in grid] == list(itertools.product(
        (0.03, 0.1, 0.3), (0.25, 1.0, 4.0)))
    assert {g.c_e for g in grid} == {0.0} and not any(g.expected_availability for g in grid)
    with pytest.raises(ValueError, match="T_nom"):
        R.sortie_rewards(R.DERIVED, dataclasses.replace(sortie, t_ref_s=None))


def test_a_stop_record_refuses_what_a_flight_cannot_hold():
    with pytest.raises(ValueError, match="targets"):
        _stop(0.0, ("a",), (10.0,), devices=("b",))
    with pytest.raises(ValueError, match="collected"):
        _stop(0.0, ("a",), (10.0,), devices=("a", "b"), targets=("b",))
    with pytest.raises(ValueError, match="one weight"):
        _stop(0.0, ("a",), (), devices=("a",))
    with pytest.raises(ValueError, match="arrival"):
        _stop(10.0, ("a",), (1.0,), t_next=5.0)
    with pytest.raises(ValueError, match="terminal"):
        _sortie([_stop(0.0, ("a",), (1.0,)), _stop(20.0, ("b",), (1.0,), terminal=True)][::-1])
    with pytest.raises(ValueError, match="next stop's arrival"):
        _sortie([_stop(0.0, ("a",), (1.0,), t_next=19.0), _stop(20.0, ("b",), (1.0,),
                                                                    terminal=True)])


# --------------------------------------------------------------------------- #
# Expected availability (critic C1)
# --------------------------------------------------------------------------- #

def _drawn(available, rel):
    """One draw of a two-stop sortie: each target collected when available."""
    s1_targets, s2_targets = ("a", "b", "c"), ("d",)
    stops = []
    for k, targets in enumerate((s1_targets, s2_targets)):
        got = tuple(d for d in targets if available[d])
        stops.append(_stop(100.0 + 30 * k, got, (10.0,) * len(got), devices=targets + (
            ("x",) if k == 0 else ()), targets=targets,
            dropped=tuple(d for d in targets if not available[d]), terminal=k == 1,
            listen=0.0 if len(got) == len(targets) else 1.0, t_next=130.0 + 40 * k))
    weights = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "x": 5.0, "y": 6.0}
    return _sortie(stops, n=6, demand=list(weights), weights=weights,
                   committed=["a", "b", "c", "d", "x"], availability=rel)


def test_the_expected_reward_is_the_realized_rewards_mean_over_the_draw_exactly():
    rel = {"a": 0.3, "b": 0.85, "c": 0.5, "d": 0.6}
    spec = R.RewardSpec(c_t=0.1, c_cov=1.0)
    mean = [0.0, 0.0]
    expected = set()
    for outcome in itertools.product((True, False), repeat=4):
        available = dict(zip("abcd", outcome))
        p = math.prod(rel[d] if available[d] else 1.0 - rel[d] for d in "abcd")
        sortie = _drawn(available, rel)
        for k, t in enumerate(R.sortie_rewards(spec, sortie)):
            mean[k] += p * t.total
        expected.add(tuple(round(t.total, 12) for t in R.sortie_rewards(
            dataclasses.replace(spec, expected_availability=True), sortie)))
    assert len(expected) == 1                       # the same whatever was drawn
    assert list(expected.pop()) == pytest.approx(mean, abs=1e-12)
    # a member the availability map does not list is always available, as the
    # mule's draw treats it (``FerryRuntime.uplink_drops``)
    unlisted = _drawn(dict.fromkeys("abcd", True), {"a": 0.3, "b": 0.85, "c": 0.5})
    listed = dataclasses.replace(unlisted, availability={**rel, "d": 1.0})
    assert R.sortie_rewards(EXPECTED, unlisted) == R.sortie_rewards(EXPECTED, listed)
    assert R.sortie_rewards(EXPECTED, unlisted) != R.sortie_rewards(
        EXPECTED, dataclasses.replace(unlisted, availability=rel))
    with pytest.raises(ValueError, match="availability"):
        R.sortie_rewards(EXPECTED, dataclasses.replace(_drawn(dict.fromkeys("abcd", True), rel),
                                                       availability={}))


def test_the_expected_credit_is_the_mules_keyed_draws_mean():
    from hermes.l1.channel_model import keyed_uniform

    rel = {"d0": 0.15, "d1": 0.4, "d2": 0.62, "d3": 0.9, "d4": 0.999}
    hits = [sum(keyed_uniform(salt, d, 3) < r for d, r in rel.items()) for salt in range(4000)]
    mean = statistics.fmean(hits)
    se = statistics.stdev(hits) / math.sqrt(len(hits))
    assert abs(mean - sum(rel.values())) < 4 * se


def test_ferrysims_realized_and_expected_returns_agree_in_the_mean():
    diffs, credit = [], []
    for i in range(20):
        ep = E.run_episode(N6, _seed(N6, i), E.Policy.of_arm("FX"), trial_index=i)
        ex = ep.rescored(EXPECTED)
        assert ex.sorties is ep.sorties              # nothing is flown again
        diffs.append(ep.ret - ex.ret)
        for s in ep.sorties:
            for stop in s.stops:
                credit += [(d in stop.collected) - s.availability[d] for d in stop.targets]
    for xs in (diffs, credit):
        mean, se = statistics.fmean(xs), statistics.stdev(xs) / math.sqrt(len(xs))
        assert abs(mean) < 4 * se, (mean, se)
    assert len(credit) > 200


# --------------------------------------------------------------------------- #
# The headroom oracle
# --------------------------------------------------------------------------- #

class _Base:
    name = "base"

    def score(self, view, *, mask):
        return tuple(-float(row) for row in range(len(view.pairs)))


def test_the_replay_scorer_takes_its_path_in_its_sortie_only():
    view = SimpleNamespace(pairs=(("wide", 0), ("wide", 1), ("medium", 0), ("medium", 1)))
    scorer = HR.ReplayScorer(1, path=(1,), base=_Base())
    with pytest.raises(RuntimeError, match="begin_mission"):
        scorer.score(view, mask=(True,) * 4)
    scorer.begin_mission(0)
    assert scorer.score(view, mask=(True,) * 4) == (0.0, -1.0, -2.0, -3.0)
    assert scorer.branching == []
    scorer.begin_mission(1)
    assert scorer.score(view, mask=(True, False, True, True)) == (0.0, 0.0, 1.0, 0.0)
    assert scorer.score(view, mask=(False, True, False, True)) == (0.0, 1.0, 0.0, 0.0)
    assert scorer.score(view, mask=(False,) * 4) == (0.0, -1.0, -2.0, -3.0)   # the fallback's
    assert scorer.branching == [3, 2, 0]
    assert scorer.chosen == [("medium", 0), ("wide", 1), None]
    lost = HR.ReplayScorer(0, path=(5,), base=_Base())
    lost.begin_mission(0)
    with pytest.raises(ValueError, match="left the flight"):
        lost.score(view, mask=(True, True, False, False))


def test_the_search_flies_every_sequence_once():
    """A tree whose second decision depends on the first: the leaf order
    finds every leaf, each once."""
    def branching(full):
        if full[0] == 0:
            return [2, 3]
        return [2, 1, 2]

    path, leaves = (), []
    while path is not None:
        full = list(path) + [0] * 3
        radix = branching(full)
        leaves.append(tuple(full[:len(radix)]))
        path = HR._next_path(path, radix)
    assert leaves == [(0, 0), (0, 1), (0, 2), (1, 0, 0), (1, 0, 1)]
    assert HR._next_path((), [0, 1]) is None          # an empty mask is one branch
    assert HR._next_path((), []) is None


def test_the_oracle_is_at_least_every_scripted_policy_on_toy_cells():
    from hermes.scheduler.policies.pair_slot import SCRIPTED_SCORERS, scripted_scorer

    gains = []
    for cell, seed in TOY:
        t0 = time.perf_counter()
        h = HR.episode_headroom(cell, seed, max_leaves=64)
        assert time.perf_counter() - t0 < 15.0
        assert not any(s.truncated for s in h.sorties)
        assert h.slot_gain >= 0.0 and h.gain >= 0.0
        assert h.value >= max(h.references[label] for label in HR.value_references())
        assert h.f_gain == h.references["F"] - h.references["FX"]
        # the decisions per sortie are those of fx_pair's flight, the first leaf
        assert [len(oracle.branching) for oracle in h.sorties] == list(h.decisions)
        for j, oracle in enumerate(h.sorties):
            assert oracle.fx_return == h.reference_sorties["fx_pair"][j]
            assert oracle.best_return >= oracle.fx_return
            for name in SCRIPTED_SCORERS:
                # the reference flying sortie j alone, FX's pair rule the rest
                policy = E.Policy(label=name, arm="FX", scorer=functools.partial(
                    HR.SortieScorer, j, scripted_scorer(name)))
                hybrid = E.run_episode(cell, seed, policy, stop_after=j)
                assert hybrid.sortie_returns[j] <= oracle.best_return
        gains.append(h.gain)
        assert sum(s.leaves for s in h.sorties) >= 4
        # the leaf cap: one leaf is FX's pair rule, flagged when more remained
        j = max(range(len(h.sorties)), key=lambda k: h.sorties[k].leaves)
        capped = HR.sortie_oracle(cell, seed, j, max_leaves=1)
        assert (capped.leaves, capped.truncated, capped.best_return) == (
            1, h.sorties[j].leaves > 1, h.sorties[j].fx_return)
    assert gains[0] > 0.1                              # a choice no reference takes


def test_the_leaf_cap_flags_only_a_search_it_cut_short():
    """A search exhausted at exactly ``max_leaves`` leaves is complete; one leaf
    fewer leaves it truncated (its value then a lower bound)."""
    cell = TOY[0][0]
    seed = _seed(cell, index=2)                        # sortie 1: one decision of two pairs
    flown = [HR.sortie_oracle(cell, seed, 1, trial_index=2, max_leaves=m) for m in (1, 2, 3)]
    assert [(o.leaves, o.truncated, o.branching) for o in flown] == [
        (1, True, (2,)), (2, False, (2,)), (2, False, (2,))]


def test_the_oracle_refuses_a_first_leaf_that_is_not_fx_pairs_flight(monkeypatch):
    """Each sortie's first leaf replays the ``fx_pair`` episode; a replay that left
    that flight is refused rather than reported."""
    cell, seed = TOY[0]
    real = HR.sortie_oracle

    def drifted(*args, **kwargs):
        oracle = real(*args, **kwargs)
        return dataclasses.replace(oracle, fx_return=oracle.fx_return + 1.0)

    monkeypatch.setattr(HR, "sortie_oracle", drifted)
    with pytest.raises(AssertionError, match="first leaf"):
        HR.episode_headroom(cell, seed, max_leaves=1)


def test_the_headroom_is_over_pair_choices_against_the_fx_arm():
    """The gain is V - R(FX arm) (resolution R6). V is the oracle's sum, FX's return
    or a scripted reference flown whole in the slot, whichever is largest; the F
    arm's return is not in V (it flies the committed band where the slot's mask
    refuses it, so no pair choice reaches its flight), and its gain over FX is
    reported beside the headroom. The slot gain is the oracle over ``fx_pair``.
    Every value here differs, so each definition is pinned."""
    refs = {"FX": 0.2, "F": 0.9, "fx_pair": 0.1, "committed_pair": 0.35, "hyb": 0.45,
            "greedy_1": 0.3}
    h = HR.EpisodeHeadroom(
        cell="c", seed=7, index=3,
        sorties=tuple(HR.SortieOracle(sortie=j, best_return=b, fx_return=f, leaves=3,
                                      truncated=False, best_path=(1,), branching=(2,))
                      for j, (b, f) in enumerate(((0.5, 0.3), (-0.1, -0.2)))),
        references=refs, reference_sorties={k: (v / 2, v / 2) for k, v in refs.items()},
        decisions=(1, 1))
    assert HR.value_references() == ("FX", "fx_pair", "committed_pair", "hyb", "greedy_1")
    assert h.v_dfs == pytest.approx(0.4)
    assert h.value == 0.45                             # hyb flown whole, not F's 0.9
    assert h.gain == pytest.approx(0.25)
    assert h.slot_gain == pytest.approx(0.3)
    assert h.f_gain == pytest.approx(0.7)
    out = h.to_json()
    assert (out["value"], out["gain"], out["slot_gain"], out["f_gain"]) == (
        h.value, h.gain, h.slot_gain, h.f_gain)
    assert HR.EpisodeHeadroom.from_json(json.loads(json.dumps(out))) == h


def test_epsilon_and_the_pause_rule_are_decisions_5_and_10s():
    assert HR.epsilon_from_headroom(0.0) == 0.01
    assert HR.epsilon_from_headroom(0.05) == 0.01
    assert HR.epsilon_from_headroom(0.3) == pytest.approx(0.03)
    assert HR.epsilon_from_headroom(-0.2) == 0.01
    with pytest.raises(ValueError):
        HR.epsilon_from_headroom(math.nan)
    assert HR.pause_rule({"a": {"headroom": 0.005}, "b": {"headroom": 0.0099}}) is True
    assert HR.pause_rule({"a": {"headroom": 0.005}, "b": {"headroom": 0.01}}) is False
    episodes = [{"cell": "c", "gain": g, "slot_gain": 0.05, "f_gain": f, "v_dfs": 1.0,
                 "value": 1.0 + g, "references": {"FX": 1.0, "F": 1.0 + f, "fx_pair": 0.95},
                 "sorties": [{"leaves": 3, "truncated": False}], "decisions": [1, 2]}
                for g, f in ((0.0, 0.5), (0.2, -0.1), (0.4, 0.1))]
    cell = HR.cell_headroom(episodes)
    assert cell["headroom"] == pytest.approx(0.2)
    assert cell["epsilon"] == pytest.approx(0.02)      # the gain's, not the slot gain's 0.01
    assert cell["slot_headroom"] == pytest.approx(0.05)
    assert cell["f_gain"] == pytest.approx(0.5 / 3)
    assert cell["episodes_f_above_value"] == 1         # F's 1.5 against V's 1.0
    assert cell["sorties_with_2_or_more_decisions"] == 0.5
    assert cell["episodes_with_gain"] == 2


def test_the_headroom_report_on_a_toy_cell():
    """The report flies its episodes with its own reward and device model."""
    cell, _ = TOY[0]
    other = R.RewardSpec(c_t=0.3, c_cov=4.0)
    report = HR.headroom_report([cell], episodes=1, max_leaves=8, reward=other,
                                device_model=IP.DEVICE_MODEL_STUB)
    assert report["stream"] == C.VAL_STREAM and set(report["cells"]) == {cell.name}
    assert (report["reward"], report["device_model"]) == (other.to_json(), "stub")
    (episode,) = report["episodes"]
    seed = _seed(cell)
    fx = E.Policy.of_arm("FX")
    flown = E.run_episode(cell, seed, fx, reward=other, device_model=IP.DEVICE_MODEL_STUB).ret
    assert (episode["seed"], episode["references"]["FX"]) == (seed, flown)
    assert flown != E.run_episode(cell, seed, fx, device_model=IP.DEVICE_MODEL_STUB).ret
    assert flown != E.run_episode(cell, seed, fx, reward=other).ret
    assert report["pause"] == (report["cells"][cell.name]["headroom"] < 0.01)
    assert "Pause" in HR.format_report(report)
    json.dumps(report)                                 # JSON-ready, no wall time
    assert "wall" not in json.dumps(report)


def test_the_headroom_flies_and_records_the_plan_it_is_given(monkeypatch):
    """Resolution R23 (the repair round's A-1): every flight of the report, the
    references' and each oracle leaf's, flies its plan score settings as a driver
    override, and the report records them, on its first line too, so a sweep
    trained under a pilot's plan reads ε on it; on the cells' own plan no flight
    is overridden and nothing is recorded, the report it always was. F-cov's
    settings move the flights."""
    cell, _ = TOY[0]
    calls = []
    fly = HR.run_episode

    def spied(*args, **kwargs):
        calls.append(kwargs.get("driver_overrides"))
        return fly(*args, **kwargs)

    monkeypatch.setattr(HR, "run_episode", spied)
    own = HR.headroom_report([cell], episodes=1, max_leaves=2)
    references = len(E.reference_policies())
    assert len(calls) > references and calls == [None] * len(calls)
    assert "plan_score_params" not in own
    assert "plan score" not in HR.format_report(own)
    calls.clear()
    planned = HR.headroom_report([cell], episodes=1, max_leaves=2, plan_score_params=F_COV_SCORE)
    assert len(calls) > references
    assert calls == [{"plan_score_params": F_COV_SCORE}] * len(calls)
    assert planned["plan_score_params"] == F_COV_SCORE
    assert HR.format_report(planned).splitlines()[0].endswith(
        'per sortie; plan score settings {"c_cov_per_device": 0.0, "c_link": 0.0})')
    assert planned["episodes"][0]["references"] != own["episodes"][0]["references"]
    assert json.loads(json.dumps(planned))["plan_score_params"] == F_COV_SCORE
    with pytest.raises(ValueError, match=r"unknown settings \['kappa'\]"):
        HR.headroom_report([cell], episodes=1, plan_score_params={"kappa": 1})


@pytest.fixture
def logging_put_back():
    """A command line's ``main`` turns HERMES' logs below errors off while it
    runs and puts the level back itself; this puts it back after the test too."""
    level = logging.root.manager.disable
    yield
    logging.disable(level)


def test_the_headroom_command_takes_the_plan_as_the_runner_reads_it(monkeypatch, capsys,
                                                                      tmp_path, logging_put_back):
    """``--plan-score-params`` (the runner's JSON object of ``PlanScoreParams``
    fields) reaches every episode's task and the saved report; with no flag the
    tasks carry the cells' own plan, {}; settings ``PlanScoreParams`` does not
    have, or a value it refuses, are a usage error before any episode flies."""
    built = []
    monkeypatch.setattr(HR, "parallel_map",
                        lambda fn, tasks, workers=1: built.append(list(tasks)) or [])
    monkeypatch.setattr(HR, "cell_headroom", lambda episodes: {"headroom": 1.0})
    monkeypatch.setattr(HR, "format_report", lambda report: "")
    out = tmp_path / "headroom.json"
    assert HR.main(["--cells", "jit-n12-120", "--episodes", "2", "--plan-score-params",
                    '{"c_cov_per_device": 0.25}', "--out", str(out)]) == 0
    assert [t.plan_score_params for t in built[0]] == [{"c_cov_per_device": 0.25}] * 2
    assert json.loads(out.read_text(encoding="utf-8"))["plan_score_params"] == {
        "c_cov_per_device": 0.25}
    assert HR.main(["--cells", "jit-n12-120", "--episodes", "1", "--out", str(out)]) == 0
    assert [t.plan_score_params for t in built[1]] == [{}]
    assert "plan_score_params" not in json.loads(out.read_text(encoding="utf-8"))
    for flag, why in (('{"kappa": 1}', "--plan-score-params: plan_score_params: unknown "
                                       "settings ['kappa']"),
                      ('{"dwell_in_delta": "no"}', "--plan-score-params: dwell_in_delta must "
                                                   "be a bool"),
                      ("[1]", "--plan-score-params must be a JSON object")):
        capsys.readouterr()
        with pytest.raises(SystemExit) as refused:
            HR.main(["--cells", "jit-n12-120", "--plan-score-params", flag])
        assert refused.value.code == 2 and why in capsys.readouterr().err, flag
    assert len(built) == 2


def test_the_headroom_and_evaluate_commands_make_their_out_folder_before_flying(
        monkeypatch, tmp_path, logging_put_back):
    """A long run never ends on a missing folder: the headroom and evaluate
    commands make ``--out``'s folder before any episode flies, as the command
    line's own files get theirs (the docs unit's review, C2: the Run Guide's
    ``results/exp5/headroom/`` did not exist after a 49-minute run)."""
    made = []
    out = tmp_path / "results" / "exp5" / "headroom" / "report.json"
    monkeypatch.setattr(HR, "parallel_map",
                        lambda fn, tasks, workers=1: made.append(out.parent.is_dir()) or [])
    monkeypatch.setattr(HR, "cell_headroom", lambda episodes: {"headroom": 1.0})
    monkeypatch.setattr(HR, "format_report", lambda report: "")
    assert HR.main(["--cells", "jit-n12-120", "--episodes", "1", "--out", str(out)]) == 0
    assert out.is_file()
    evaluated = tmp_path / "results" / "exp5" / "references" / "val.json"
    monkeypatch.setattr(EV, "evaluate",
                        lambda *args, **kwargs: made.append(evaluated.parent.is_dir()) or [])
    assert EV.main(["--cells", "jit-n12-120", "--episodes", "1", "--out", str(evaluated)]) == 0
    assert json.loads(evaluated.read_text(encoding="utf-8")) == {"summaries": [], "scores": []}
    assert made == [True, True]


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #

def test_every_policy_meets_the_same_episodes_in_workers_too():
    policies = [E.Policy.of_arm("FX"), E.Policy.scripted("greedy_1")]
    here = EV.evaluate([TOY[0][0]], policies, stream=C.VAL_STREAM, episodes=2)
    assert [(s["index"], s["policy"]) for s in here] == [
        (0, "FX"), (0, "greedy_1"), (1, "FX"), (1, "greedy_1")]
    assert here[0]["seed"] == here[1]["seed"] != here[2]["seed"] == here[3]["seed"]
    assert [s["seed"] for s in here[::2]] == list(C.stream_seeds(C.VAL_STREAM, "toy-n8-120", 2))
    workers = EV.evaluate([TOY[0][0]], policies, stream=C.VAL_STREAM, episodes=2, workers=2)
    assert workers == here
    score = EV.score_summary([s for s in here if s["policy"] == "FX"])
    assert score["episodes"] == 2 and score["return_mean"] == pytest.approx(
        statistics.fmean(s["return"] for s in here if s["policy"] == "FX"))
    assert set(EV.scores_by(here)) == {("toy-n8-120", "FX"), ("toy-n8-120", "greedy_1")}
    with pytest.raises(ValueError, match="one policy"):
        EV.score_summary(here)
    with pytest.raises(ValueError, match="training stream"):
        EV.evaluate([N12], policies, stream=C.train_stream(0), episodes=1)


def test_the_evaluator_flies_each_episode_with_its_reward_and_device_model():
    """Study 5.7 scores under its grid's rewards: an evaluation's reward and device
    model are each episode's, as :func:`run_episode` flies them."""
    cell = TOY[0][0]
    seed = _seed(cell)
    other = R.RewardSpec(c_t=0.3, c_cov=4.0)
    fx = E.Policy.of_arm("FX")
    (got,) = EV.evaluate([cell], [fx], stream=C.VAL_STREAM, episodes=1, reward=other,
                         device_model=IP.DEVICE_MODEL_STUB)
    want = E.run_episode(cell, seed, fx, reward=other, device_model=IP.DEVICE_MODEL_STUB)
    assert got == dict(EV.episode_summary(want), stream=C.VAL_STREAM)
    assert got["return"] != E.run_episode(cell, seed, fx, device_model=IP.DEVICE_MODEL_STUB).ret
    assert got["return"] != E.run_episode(cell, seed, fx, reward=other).ret


def test_a_checkpoints_held_out_score_lands_in_its_manifest(tmp_path):
    from hermes.scheduler.selector.pair_q import (
        HELD_OUT_KEYS,
        LearnerSettings,
        PairQConfig,
        PairQNet,
        verify_checkpoint,
    )

    path = tmp_path / "g0.9_s1.npz"
    PairQNet(5, PairQConfig(hidden=(8, 4), gamma=0.9), seed=1).save(
        path, kind="pair_q", purpose="trained", schema={"version": "pair_test", "dim": 5},
        classes=["wide"], provenance=dict(
            reward=R.DERIVED.to_json(), training={"learner": LearnerSettings().to_json()},
            seeds={"init": 1}, cell_family="jittery", cell_family_sha256=C.family_sha256(
                "jittery"), trainer_commit=None, dirty=True, episodes_trained=10,
            validation=[], held_out=None))
    summaries = [{"cell": "jit-n12-120", "stream": C.HELDOUT_STREAM, "policy": "FQ-g90",
                  "index": i, "return": r, "decisions": [2, 3],
                  "terms": {"gain": 1.0, "time": 0.1, "energy": 0.0, "distance": 0.0,
                            "coverage": 0.2}}
                 for i, r in enumerate((0.5, 0.75, 1.0))]
    manifest = EV.record_held_out_score(str(path), summaries)
    held = manifest["held_out"]
    assert set(HELD_OUT_KEYS) <= set(held)
    assert held["episodes"] == 3 and held["return_mean"] == pytest.approx(0.75)
    # the sample standard deviation and its standard error
    assert (held["return_sd"], held["return_se"]) == pytest.approx((0.25, 0.25 / math.sqrt(3)))
    assert verify_checkpoint(path)["held_out"] == held
    with pytest.raises(ValueError, match="held-out"):
        EV.record_held_out_score(str(path), [dict(s, stream=C.VAL_STREAM) for s in summaries])


# --------------------------------------------------------------------------- #
# Layering and cost
# --------------------------------------------------------------------------- #

def test_ferrysim_imports_nothing_from_tests_and_its_package_loads_no_module():
    code = (
        "import sys\n"
        "import experiments.ferrysim\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('experiments.ferrysim.'))\n"
        "import experiments.ferrysim.headroom, experiments.ferrysim.evaluate\n"
        "tests = sorted(m for m in sys.modules if m == 'tests' or m.startswith('tests.'))\n"
        "print(loaded, tests)\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "[] []"
    imports_tests = re.compile(r"^\s*(from|import)\s+tests\b", re.MULTILINE)
    imports_ferrysim = re.compile(r"^\s*(from|import)\s+experiments\.ferrysim\b", re.MULTILINE)
    for path in sorted((REPO / "experiments" / "ferrysim").glob("*.py")):
        assert not imports_tests.search(path.read_text(encoding="utf-8")), path.name
    for path in sorted((REPO / "hermes").rglob("*.py")):
        assert not imports_ferrysim.search(path.read_text(encoding="utf-8")), path


def test_smoke_runs_stay_under_15_s():
    for policy in E.reference_policies():
        t0 = time.perf_counter()
        ep = E.run_episode(N12, _seed(index=2), policy, trial_index=2)
        assert time.perf_counter() - t0 < 15.0, policy.label
        assert len(ep.sorties) == 4 and ep.decisions >= 4
