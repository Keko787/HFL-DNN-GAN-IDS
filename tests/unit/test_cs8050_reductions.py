"""CMP_SC 8050 Phase 2: the two conversions of T1, checked with the plan checker.

These tests check the yes-to-yes maps of Lemmas 1 and 2 and the facts the
no-to-no arguments rest on (a device is only served at its own point; a step
between grid points that are not joined costs more than one leg; time is the
knapsack's capacity). The full no-to-no check on small cases is step 3.7.
"""

from __future__ import annotations

import itertools
import math

import pytest

from experiments.cs8050.problem import (
    RULE_REACH,
    RULE_TIME_LIMIT,
    Plan,
    Setting,
    Stop,
    check_plan,
)
from experiments.cs8050.reductions import (
    from_grid_graph,
    from_knapsack,
    grid_device_id,
    is_grid_cycle,
    item_device_id,
    plan_from_cycle,
    plan_from_items,
)

# A 2-by-3 block of grid points and its Hamiltonian cycle.
GRID = ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1))
CYCLE = GRID
# Four points in a row: a path, with no Hamiltonian cycle.
PATH = ((0, 0), (1, 0), (2, 0), (3, 0))
# Figure 1 of T1: a 3-by-3 block missing its top-right corner, and the cycle drawn.
FIG1 = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2))
FIG1_CYCLE = ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 1))

QUANTA = [None, 1.0, 2.5, 3.0]


# --------------------------------------------------------------------------- #
# Lemma 1: grid graphs
# --------------------------------------------------------------------------- #

def test_cycle_helper():
    assert is_grid_cycle(GRID, CYCLE)
    assert not is_grid_cycle(GRID, GRID[:5])
    assert not is_grid_cycle(PATH, PATH)
    assert not is_grid_cycle(GRID, ((0, 0), (1, 1), (1, 0), (2, 0), (2, 1), (0, 1)))


@pytest.mark.parametrize("quantum", QUANTA)
def test_a_hamiltonian_cycle_gives_a_plan_serving_everyone(quantum):
    inst, target = from_grid_graph(GRID, quantum_s=quantum)
    v = check_plan(inst, plan_from_cycle(inst, CYCLE))
    leg = 1.0 if quantum is None else quantum
    assert v.ok and v.value == target == 6.0
    assert math.isclose(v.end_s, 6 * leg)
    assert math.isclose(inst.time_limit_s, 6 * leg)


@pytest.mark.parametrize("quantum", QUANTA)
@pytest.mark.parametrize("size", [0.0, 500.0])
def test_figure_1s_instance(quantum, size):
    assert is_grid_cycle(FIG1, FIG1_CYCLE)
    inst, target = from_grid_graph(FIG1, quantum_s=quantum, size_bytes=size)
    v = check_plan(inst, plan_from_cycle(inst, FIG1_CYCLE))
    leg = 1.0 if quantum is None else quantum
    assert v.ok and v.value == target == 8.0
    assert math.isclose(v.end_s, 8 * leg + 8 * size)
    # Taking the figure's red dashed step, (2, 1) -> (1, 2), costs more than
    # one leg (and forces a second diagonal, (1, 1) -> (0, 2)): over the limit.
    order = ((0, 0), (1, 0), (2, 0), (2, 1), (1, 2), (1, 1), (0, 2), (0, 1))
    plan = Plan("unit", tuple(Stop(p, (grid_device_id(p),)) for p in order))
    assert check_plan(inst, plan).rule == RULE_TIME_LIMIT


@pytest.mark.parametrize("quantum", QUANTA)
def test_the_cycle_may_start_anywhere(quantum):
    inst, _ = from_grid_graph(GRID, quantum_s=quantum)
    turned = CYCLE[3:] + CYCLE[:3]
    assert check_plan(inst, plan_from_cycle(inst, turned)).ok


@pytest.mark.parametrize("quantum", QUANTA)
def test_a_step_between_unjoined_points_breaks_the_time_limit(quantum):
    # (1, 0) -> (2, 1) and (2, 0) -> (1, 1) are diagonal steps, sqrt(2) apart:
    # over one leg exactly, and two legs once rounded up.
    inst, _ = from_grid_graph(GRID, quantum_s=quantum)
    order = ((0, 0), (1, 0), (2, 1), (2, 0), (1, 1), (0, 1))
    plan = Plan("unit", tuple(Stop(v, (grid_device_id(v),)) for v in order))
    v = check_plan(inst, plan)
    assert (v.ok, v.rule) == (False, RULE_TIME_LIMIT)


@pytest.mark.parametrize("quantum", QUANTA)
def test_a_path_has_no_plan_serving_everyone(quantum):
    # Smoke test of the no-to-no direction: every order of one-device stops
    # through four points in a row is over the limit. (Repeat visits and
    # shared stops only add legs or cannot happen; step 3.7 checks all plans.)
    inst, _ = from_grid_graph(PATH, quantum_s=quantum)
    for order in itertools.permutations(PATH):
        plan = Plan("unit", tuple(Stop(v, (grid_device_id(v),)) for v in order))
        assert check_plan(inst, plan).rule == RULE_TIME_LIMIT


def test_a_device_is_served_only_at_its_own_point():
    inst, _ = from_grid_graph(GRID)
    plan = Plan("unit", (Stop((1, 0), (grid_device_id((0, 0)),)),))
    assert check_plan(inst, plan).rule == RULE_REACH


def test_hardness_holds_with_equal_data_and_ferrys_three_reaches():
    # Grid spacing above the narrow setting's 232 m reach; every device sends
    # the same 500 kB, as in the planner's passes.
    settings = (
        Setting.constant("wide", 60, 100_000),
        Setting.constant("medium", 120, 40_000),
        Setting.constant("narrow", 232, 10_000),
    )
    inst, target = from_grid_graph(GRID, size_bytes=500_000, settings=settings, spacing_m=300)
    plan = plan_from_cycle(inst, CYCLE, spacing_m=300)
    assert plan.setting == "wide"
    v = check_plan(inst, plan)
    assert v.ok and v.value == target
    # The same route on a slower setting sends for longer and runs over.
    slow = Plan("narrow", plan.stops)
    assert check_plan(inst, slow).rule == RULE_TIME_LIMIT


def test_a_setting_reaching_the_next_point_is_refused():
    with pytest.raises(ValueError):
        from_grid_graph(GRID, settings=(Setting.constant("wide", 1.0, 1.0),))


# --------------------------------------------------------------------------- #
# Lemma 2: knapsack
# --------------------------------------------------------------------------- #

SIZES, VALUES, CAPACITY = (3, 4, 5), (4, 5, 6), 8


@pytest.mark.parametrize("quantum", QUANTA)
def test_a_subset_that_fits_gives_a_plan_worth_its_value(quantum):
    inst, target = from_knapsack(SIZES, VALUES, CAPACITY, 10, quantum_s=quantum)
    v = check_plan(inst, plan_from_items(inst, [0, 2]))
    assert v.ok and v.value == target == 10.0
    assert v.timings[0].arrival_s == 0.0 and v.timings[0].dwell_s == 8.0
    assert v.end_s == 8.0


@pytest.mark.parametrize("quantum", QUANTA)
def test_a_subset_over_the_capacity_is_over_the_time_limit(quantum):
    inst, _ = from_knapsack(SIZES, VALUES, CAPACITY, 10, quantum_s=quantum)
    v = check_plan(inst, plan_from_items(inst, [1, 2]))
    assert (v.ok, v.rule) == (False, RULE_TIME_LIMIT)
    assert v.timings[0].home_s == 9.0


def test_no_items_is_the_empty_plan():
    inst, _ = from_knapsack(SIZES, VALUES, CAPACITY, 10)
    v = check_plan(inst, plan_from_items(inst, []))
    assert v.ok and v.value == 0.0


def test_every_item_is_in_reach_of_the_one_site():
    inst, _ = from_knapsack(SIZES, VALUES, CAPACITY, 10)
    everything = Plan("one", (Stop(inst.dock, tuple(item_device_id(i) for i in range(3))),))
    v = check_plan(inst, everything)
    assert v.rule == RULE_TIME_LIMIT          # 12 s of sending, not out of reach
    assert inst.sites == (inst.dock,)


def test_bad_knapsack_instances_are_refused():
    with pytest.raises(ValueError):
        from_knapsack((), (), 1, 1)
    with pytest.raises(ValueError):
        from_knapsack((1, 2), (1,), 1, 1)
    with pytest.raises(ValueError):
        from_knapsack((1,), (0,), 1, 1)
