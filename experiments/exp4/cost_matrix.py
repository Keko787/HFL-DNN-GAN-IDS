"""Cost the Phase-3 matrix in wall-clock before launching it.

Checklist §5.1 requires the matrix to be **costed before it runs**, because the
failure mode this whole document exists to prevent is discovering the price after
paying it. This is that calculation, as code rather than arithmetic in a
document, so the numbers can be re-derived when an input changes.

Per-arm costs come from **measured** committed sweeps wherever they exist; every
estimated figure is labelled and its basis stated. Run::

    python -m experiments.exp4.cost_matrix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Concurrent trials — i.e. how many runner PROCESSES run at once.
#:
#: CORRECTED 2026-08-13 after sweep A came in 2.44x over. The per-trial figures
#: below were accurate to 1.7 % (28.4 s actual vs 27.9 s predicted); the error was
#: entirely here. `run_matrix.sh` ran one runner at a time, so the real
#: concurrency was 1, not 3 — the runner executes a grid SEQUENTIALLY, and
#: parallelism only comes from sharding a sweep across several runner processes
#: writing DIFFERENT csv files.
#:
#: Set this to the number of shards you will actually launch. 3 is safe (at 5 the
#: box exhausted memory and ~30 % of trials failed); 1 is what an unsharded
#: script gives you.
CONCURRENCY = 1

#: Fraction added for startup and tail effects.
#:
#: CORRECTED to 0. The measured per-trial durations are END-TO-END trial times
#: taken from `duration_s`, which already include process-tree spawn and
#: teardown. Adding an overhead on top double-counted it. Measured multiplier on
#: sweep A was 1.017 — i.e. the per-trial mean alone predicts wall-clock.
OVERHEAD = 0.0


@dataclass
class ArmCost:
    """Per-trial wall-clock for one arm, with provenance."""

    arm: str
    seconds: float
    basis: str
    measured: bool = True

    def label(self) -> str:
        return "measured" if self.measured else "ESTIMATED"


# Per-trial seconds at the matrix operating point (real-model, n_missions=4,
# enforcement ON). Sources:
#   H0            - h0h1_all.csv, 520 ok rows. No mule, so the budget is a no-op.
#   H1            - exp4_s3b/*.csv, 100 ok rows, recorded WITH a budget. Note it
#                   is FASTER than H1 without one (51.4 s): the gate drops stops,
#                   so the mule flies less.
#   H2/H3         - h2h3_dz, 100 ok rows each, WITHOUT a budget. Scaled by the
#                   same ratio enforcement produced for H1 (37.8/51.4 = 0.735).
#   B1/B2         - calibrated against H1 at the matrix config.
ARMS: Dict[str, ArmCost] = {
    "H0": ArmCost("H0", 18.0, "h0h1_all.csv, n=520"),
    "H1": ArmCost("H1", 37.8, "exp4_s3b/*.csv, n=100, budget ON"),
    "H2": ArmCost("H2", 46.2, "h2h3_dz n=100 (62.9 s) x 0.735 enforcement ratio",
                  measured=False),
    "H3": ArmCost("H3", 45.3, "h2h3_dz n=100 (61.6 s) x 0.735 enforcement ratio",
                  measured=False),
    # B1/B2 costed at H1's rate, now CALIBRATED rather than assumed. A 9-trial
    # run at the matrix config (synthetic data — read the ratios, not the
    # absolutes) gave H1 39.0 s, B1 38.6 s, B2 39.6 s: all three within 3 %,
    # all completing 4/4 missions with 0 failures.
    #
    # An earlier calibration appeared to show B2 at 0.45x H1. That was not a
    # routing effect — B2 was dying at mission 2 on an over-eager guard, so it
    # was cheap because it was doing less. Checking WHAT the arms accomplished,
    # not just how long they took, is what caught it.
    "B1": ArmCost("B1", 37.8, "calibrated: 0.99x H1 (n=3, 4/4 missions)",
                  measured=False),
    "B2": ArmCost("B2", 37.8, "calibrated: 1.02x H1 (n=3, 4/4 missions)",
                  measured=False),
}


@dataclass
class Sweep:
    """One coherent sweep: a set of arms over a set of cells."""

    name: str
    arms: List[str]
    cells: int
    seeds: int
    rationale: str
    notes: List[str] = field(default_factory=list)

    @property
    def trials(self) -> int:
        return self.cells * self.seeds * len(self.arms)

    def seconds(self) -> float:
        per_cell_seed = sum(ARMS[a].seconds for a in self.arms)
        return self.cells * self.seeds * per_cell_seed

    def wall_clock_s(self) -> float:
        return self.seconds() * (1 + OVERHEAD) / CONCURRENCY


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #
#
# The design principle is Freeze D5: `dead_zone` and `link_quality` are consumed
# ONLY in the H0 branch. Sweeping them across the mule arms would be one
# configuration under different seeds — the exact error the freeze recorded. So
# the surface is swept for H0-vs-H1 and NOT for the baseline comparison.

MATRIX: List[Sweep] = [
    Sweep(
        name="A — architecture surface (H0 vs H1)",
        arms=["H0", "H1"],
        # clean (1 cell; dead_zone/link_quality do not apply) + jittery (4 x 3).
        cells=13,
        seeds=20,
        rationale=(
            "The headline participation claim, and the only comparison the "
            "dead_zone x link_quality surface is meaningful for."
        ),
        notes=["Re-run of the committed 13-cell design with enforcement ON."],
    ),
    Sweep(
        name="B — scheduling-policy comparison (H1 vs B1 vs B2)",
        arms=["H1", "B1", "B2"],
        # clean + jittery at ONE fixed (dead_zone, link_quality) point.
        cells=2,
        seeds=20,
        rationale=(
            "The reviewer-facing baseline comparison. One flag differs between "
            "arms, so it isolates the ranking policy."
        ),
        notes=[
            "dead_zone / link_quality FIXED, not swept: Freeze D5 — they are "
            "H0-only, so sweeping them here varies nothing (that error was "
            "already made once in the H2/H3 dead-zone sweep).",
            "B2 requires --real-model.",
        ],
    ),
    Sweep(
        name="C — L1 adaptivity (H2 vs H3)",
        arms=["H2", "H3"],
        cells=1,
        seeds=20,
        rationale="The L1 claim. Separate sweep: needs --l1-channel, which "
                  "changes the backhaul model in both arms.",
        notes=["NOT poolable with A or B — different backhaul model, "
               "non-aligned seeds."],
    ),
]


def main() -> int:
    print("Phase-3 matrix — wall-clock cost")
    print(f"concurrency={CONCURRENCY}  overhead={OVERHEAD:.0%}\n")

    print("Per-arm per-trial cost:")
    for a in ARMS.values():
        print(f"  {a.arm:<3} {a.seconds:>6.1f} s  [{a.label():>9}]  {a.basis}")
    print()

    hdr = f"{'sweep':<44} {'arms':>4} {'cells':>6} {'trials':>7} {'wall':>9}"
    print(hdr)
    print("-" * len(hdr))
    total_trials = 0
    total_s = 0.0
    for sw in MATRIX:
        total_trials += sw.trials
        total_s += sw.wall_clock_s()
        print(f"{sw.name:<44} {len(sw.arms):>4} {sw.cells:>6} {sw.trials:>7} "
              f"{sw.wall_clock_s()/60:>7.0f}m")
    print("-" * len(hdr))
    print(f"{'TOTAL':<44} {'':>4} {'':>6} {total_trials:>7} "
          f"{total_s/60:>7.0f}m  ({total_s/3600:.1f} h)")

    print()
    print(f"Traces at ~9.7 KB/trial: {total_trials * 9.7 / 1024:.1f} MB")

    print()
    print("Notes:")
    for sw in MATRIX:
        for n in sw.notes:
            print(f"  [{sw.name.split(' ')[0]}] {n}")

    naive = sum(
        sw.cells * sw.seeds * len(sw.arms) * max(ARMS[a].seconds for a in sw.arms)
        for sw in MATRIX
    ) * (1 + OVERHEAD) / CONCURRENCY
    print()
    print(f"Sanity check — costing every arm at its sweep's slowest arm: "
          f"{naive/3600:.1f} h (upper bound).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
