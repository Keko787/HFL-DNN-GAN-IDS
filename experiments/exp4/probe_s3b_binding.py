"""Find where the S3b deadline-feasibility gate starts to bind.

**Why this exists.** S3b only constrains anything if the mission budget is
tight relative to the flight cost of the queue. Discovering that threshold by
running the orchestrator would cost hours of TensorFlow subprocess time to learn
a fact that is pure geometry — so this probe drives the **real scheduler**
(`FLScheduler.build_contact_queue`, real S1/S3/S3a/S3b) over the **real device
layouts** produced by `build_exp4_topology`, with no subprocesses and no model.

It answers: for a given N / field radius / rf_range / cruise speed, at what
`mission_budget_s` do contacts begin to be dropped, and at what budget does the
queue collapse entirely? Use the knee it reports to choose the budgets for a
real sweep.

Note on clocks: the gate charges *simulated flight time* (straight-line distance
at `cruise_speed_m_s` plus a per-stop session time). That is a different clock
from the trial's wall-clock (~10 s per mission, dominated by TF compute), so the
budget should be read as a flight-time budget.

    python -m experiments.exp4.probe_s3b_binding
    python -m experiments.exp4.probe_s3b_binding --N 6 --seeds 20 --cruise 5.0
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from typing import List, Optional, Sequence

from hermes.scheduler.fl_scheduler import FLScheduler
from hermes.scheduler.stages.s3b_feasibility import FeasibilityModel
from hermes.types import DeviceID, MissionSlice, MuleID

from .topology_builder import build_exp4_topology


def _fixed_clock(t: float):
    return lambda: t


def _scheduler_with_layout(positions, *, now: float, budget: Optional[float],
                           model: FeasibilityModel) -> FLScheduler:
    """A real FLScheduler seeded with a real Exp-4 device layout."""
    sch = FLScheduler(
        now_fn=_fixed_clock(now),
        mission_budget_s=budget,
        feasibility_model=model,
    )
    ids = [DeviceID(f"exp4-dev-{i:03d}") for i in range(len(positions))]
    sch.ingest_slice(MissionSlice(
        mule_id=MuleID("probe-mule"),
        device_ids=tuple(ids),
        issued_round=1,
        issued_at=now,
    ))
    for did, pos in zip(ids, positions):
        sch.device_states[did].last_known_position = pos
    return sch


def probe(
    *,
    n_devices: int,
    seeds: Sequence[int],
    rf_range_m: float,
    field_radius_m: float,
    budgets: Sequence[float],
    cruise_speed_m_s: float,
    session_time_s: float,
) -> None:
    model = FeasibilityModel(
        cruise_speed_m_s=cruise_speed_m_s, session_time_s=session_time_s,
    )
    now = 1000.0

    # Real device layouts from the real topology builder.
    layouts: List[list] = []
    for s in seeds:
        topo = build_exp4_topology(
            n_devices=n_devices, rf_range_m=rf_range_m, n_missions=4, seed=s,
            device_reliability=True, field_radius_m=field_radius_m,
        )
        layouts.append([d.position for d in topo.devices])

    # Baseline: how many contacts does S3a form, and what does the queue cost?
    base_counts, base_costs = [], []
    for pos in layouts:
        sch = _scheduler_with_layout(pos, now=now, budget=None, model=model)
        q = sch.build_contact_queue(rf_range_m=rf_range_m, now=now)
        base_counts.append(len(q))
        pose = (0.0, 0.0, 0.0)
        total = 0.0
        for wp in q:
            _, c = model.cost(pose, wp.position)
            total += c
            pose = wp.position
        base_costs.append(total)

    print(f"Layout: N={n_devices} rf_range={rf_range_m} field_radius={field_radius_m} "
          f"| cruise={cruise_speed_m_s} m/s session={session_time_s}s | {len(seeds)} seeds")
    print(f"  contacts formed per mission : mean {st.mean(base_counts):.2f} "
          f"(min {min(base_counts)}, max {max(base_counts)})")
    print(f"  unconstrained queue cost (s): mean {st.mean(base_costs):.1f} "
          f"(min {min(base_costs):.1f}, max {max(base_costs):.1f})")
    print()
    print(f"{'budget_s':>9} {'kept/miss':>10} {'drop:deadline':>14} {'drop:budget':>12} "
          f"{'% dropped':>10}  status")
    print("-" * 78)

    budget_knee = None
    collapse = None
    deadline_floor = None
    for b in budgets:
        kept_n, d_overdue, d_budget = [], 0, 0
        for pos in layouts:
            sch = _scheduler_with_layout(pos, now=now, budget=b, model=model)
            q = sch.build_contact_queue(rf_range_m=rf_range_m, now=now)
            feas = sch.last_feasibility
            if feas is not None:
                d_overdue += len(feas.dropped_overdue)
                d_budget += len(feas.dropped_budget)
            kept_n.append(len(q))
        tot_possible = sum(base_counts)
        pct = 100.0 * (d_overdue + d_budget) / max(1, tot_possible)
        status = ""
        if d_budget and budget_knee is None:
            budget_knee = b
            status = "<-- BUDGET starts binding"
        if sum(kept_n) == 0 and collapse is None:
            collapse = b
            status = "<-- queue fully collapsed"
        if deadline_floor is None:
            deadline_floor = d_overdue
        print(f"{b:9.1f} {st.mean(kept_n):10.2f} {d_overdue:14d} {d_budget:12d} "
              f"{pct:9.1f}%  {status}")

    print()
    print(f"  DEADLINE floor: ~{deadline_floor} contacts/grid are dropped even at the "
          f"loosest budget — they cannot be REACHED before their own deadline at "
          f"{cruise_speed_m_s} m/s.")
    print("    That is the deadline binding, not the budget. It is independent of "
          "mission_budget_s;")
    print("    raise cruise speed or Phi, or shrink the field, to relieve it.")
    if budget_knee is None:
        print("  BUDGET never bound over this grid — every budget was slack.")
    else:
        print(f"  BUDGET first binds at ~{budget_knee:.1f} s"
              f"{'; queue collapses at ~%.1f s' % collapse if collapse else ''}.")
        print("  Suggested sweep budgets: one clearly slack, one near the knee, one tight.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.exp4.probe_s3b_binding")
    p.add_argument("--N", type=int, default=6, help="devices (paper run used 6)")
    p.add_argument("--seeds", type=int, default=20, help="how many layouts to average over")
    p.add_argument("--rrf", type=float, default=60.0, help="rf_range_m")
    p.add_argument("--field-radius-m", type=float, default=100.0)
    p.add_argument("--cruise", type=float, default=5.0, help="cruise_speed_m_s")
    p.add_argument("--session-time-s", type=float, default=1.0)
    p.add_argument("--budgets", nargs="+", type=float, default=None,
                   help="budget grid (s); default is an auto log-ish ladder")
    a = p.parse_args(argv)

    budgets = a.budgets or [200, 150, 120, 100, 80, 60, 50, 40, 30, 20, 15, 10, 5]
    probe(
        n_devices=a.N,
        seeds=list(range(a.seeds)),
        rf_range_m=a.rrf,
        field_radius_m=a.field_radius_m,
        budgets=sorted(budgets, reverse=True),
        cruise_speed_m_s=a.cruise,
        session_time_s=a.session_time_s,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
