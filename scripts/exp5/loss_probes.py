"""Exploratory: the loss probes, each scored against FeRRy's F, and P9, the reach test
on every recorded comparison.

    py -3.11 scripts/exp5/loss_probes.py p9               # the recorded cells; no flights
    py -3.11 scripts/exp5/loss_probes.py P1 P4 [--score]  # a probe's cells, once flown
    py -3.11 scripts/exp5/loss_probes.py P10 [--score]    # the baselines on other bands
    py -3.11 scripts/exp5/loss_probes.py all [--score]

Added 3 Nov 2026; not pre-registered. The probes, their cells and the commands that fly
them are in DeveloperDocs/Experiment_5_Loss_Probes_Plan.md. Each probe is its own
family. F is the reference, and every other arm in a cell is compared with it on

- time to tau: the scorer's paired comparison (scoring.compare_pair), over the trials
  in which both arms reach tau;
- reach at tau: the exact McNemar test on the paired reached / not-reached outcomes,
  with a percentile-bootstrap CI on the paired difference in reach share.

Holm runs across each probe's comparisons, one family per metric, with FeRRy's other
arms (FX, FQ) against F in families of their own. A claim needs the CI to exclude 0 and
Holm p < alpha, as in the scorer. Time to tau leaves out every trial in which either arm
never reached tau, so it can favour the arm that reached tau less often; the reach test
is the check on that.

P9 runs the reach test at tau = 0.71, and both tests at the second tau, 0.82, on every
comparison the pre-registered studies make (batch 1, sens, batch 2 and batch 3), from
the scored files the campaign wrote; Holm across each study and test. It lists each
recorded time-to-tau claim whose favoured arm reached tau less often.

--score runs the trace scorer on a probe's trial CSVs that have no scored file yet, as
launch.py score does, one file at a time (scripts/exp5/ddil_score_probes.py scores a
probe's files in parallel with the launcher's job runner, then this writes the report). Never run launch.py score with a probe's --set changes: it would
overwrite the pre-registered score files. P7 and P8 join the registry once their builds
(B2, B3) exist. Outputs: results/exp5/scores/probes/<name>.{md,csv}.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib.util
import io
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

OUT = REPO / "results/exp5/scores/probes"
TAU = 0.71
SECOND_TAU = 0.82
REFERENCE = "F"
P9_STAGES = ("batch1", "sens", "batch2", "batch3")
NAMES = {"F": "F", "FX": "FX", "H1": "H1", "D1": "MAX-AoI", "D2": "Oort", "D3": "Whittle",
         "D4": "FedEx route", "D5": "FedCS", "E3": "DQN (Chen)", "D4fedex": "FedEx route + merge"}
EIGHT = "['F','FX','H1','D1','D2','D3','D4','D5']"


def is_ferry(variant: str) -> bool:
    """FeRRy's own arms besides F: compared with F apart from the baselines. F's variants
    with their own flags (step 2's F-rp and F-rpnl) are FeRRy's too."""
    return variant.startswith(("FX", "FQ", "F-"))


# --------------------------------------------------------------------- the probes
@dataclass
class Flight:
    """One launcher run of a probe: its stage, out-root and --set changes, the study
    whose cells it scores, and optionally the arms it keeps and its trials a cell."""

    label: str
    stage: str
    out_root: str
    sets: List[str]
    study: str
    arms: Optional[List[str]] = None
    trials: int = 20


@dataclass
class Probe:
    name: str
    title: str
    flights: List[Flight]
    #: cell -> the cell whose baselines stand in where this one has none flown (the
    #: baselines fly wide, so a setting that moves only the narrow band leaves them as is)
    ref_cells: Dict[str, str] = field(default_factory=dict)
    #: DDIL build D0: each arm against F also on hover and updates per mission, and on
    #: time to tau with a miss at the trial's end (D0_ENDPOINTS); the flights score with
    #: --ddil-columns (DDIL_SCORE_SET), which writes the last of those.
    ddil_endpoints: bool = False
    #: (arm, reference) pairs compared beside each arm against F, in each cell that holds
    #: both, on the same tests, in families of their own (<probe>/pairs/<kind>).
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    #: For a probe whose cells hold no F: the probe whose flights fly F in the same cells and
    #: on the same seeds, and the script (in scripts/exp5) that compares the two and writes
    #: this probe's report. main() leaves such a probe's report to that script.
    f_from: Optional[str] = None
    compared_by: str = ""


def _s53x(extra: Sequence[str]) -> List[str]:
    return ["s53x.K=[1]", "s53x.h0=false", *extra]


# DDIL build D1, step 1 of DeveloperDocs/FeRRy_DDIL_Unified_Plan.md (its NOMS slice): every
# environment switch on is the session model at t_out = 10 s with its retry, the re-poll and
# shared interference (nested bands; every class takes the narrow class's term, so an arm on
# narrow sees the recorded wave); FeRRy's arms plan from the causal estimate. Mission
# backhaul, N = 6, knee, 1 MB, 20 trials on P1's seeds, the 8 arms unless a cell says.
D1_ENV = ["--session-t-out-s", "10", "--contact-repoll", "true", "--interference-shared", "narrow"]
D1_EST = ["--channel-estimate", "causal"]
D1_ALL = D1_ENV + D1_EST
D1_AMP = {11: ["--interference-amp-db", "11"], 17: ["--interference-amp-db", "17"]}


def _bursts(duty: float, mean_s: float = 6.7) -> List[str]:
    """The burst process: 17 dB deep on every class (decision 11; mean 6.7 s from step 0)."""
    return ["--burst-duty", f"{duty:g}", "--burst-mean-s", f"{mean_s:g}", "--burst-depth-db", "17"]


def _without(args: Sequence[str], flag: str) -> List[str]:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == flag:
            skip = True
            continue
        out.append(a)
    return out


EIGHT_ARMS = ["F", "FX", "H1", "D1", "D2", "D3", "D4", "D5"]
#: (tag, args, arms or None for the 8, budget or None for the knee)
DDIL1_MAIN:List[Tuple[str, List[str], Optional[List[str]], Optional[str]]] = [
    ("_d1base", D1_ALL, None, None),
    ("_d1amp11", D1_ALL + D1_AMP[11], None, None),
    ("_d1ampharsh", D1_ALL + D1_AMP[17], None, None),
    ("_d1npl", D1_ALL + ["--n-pl", "2.7"], None, None),
    ("_d1sigma8", D1_ALL + ["--shadow-sigma-db", "8"], None, None),
    ("_d1burst10", D1_ALL + _bursts(0.1), None, None),
    ("_d1burst40", D1_ALL + _bursts(0.4), None, None),
    ("_d1ampharshpc", _without(D1_ALL, "--interference-shared") + D1_AMP[17], None, None),
    # F-o and FX-o: the environment switches on, the planner reading the channel's own sigmas.
    ("_d1baseo", D1_ENV, ["F", "FX"], None),
    ("_d1ampharsho", D1_ENV + D1_AMP[17], ["F", "FX"], None),
    ("_d1ampharsh", D1_ALL + D1_AMP[17], None, "stress"),
]
DDIL1_CHECKS: List[Tuple[str, List[str], Optional[List[str]], Optional[str]]] = [
    # Each switch alone on the recorded 17 dB channel (per-class phases, recorded pricing).
    ("_a1sess10", D1_AMP[17] + ["--session-t-out-s", "10"], None, None),
    ("_a1sess2", D1_AMP[17] + ["--session-t-out-s", "2"], None, None),
    ("_a1repoll", D1_AMP[17] + ["--contact-repoll", "true"], None, None),
    ("_a1burst40", D1_AMP[17] + _bursts(0.4), None, None),
    ("_a1est", D1_AMP[17] + D1_EST, None, None),
    ("_d1ampharshres", D1_ALL + D1_AMP[17] + ["--session-resumable", "true"], None, None),
    ("_d1burst10l", D1_ALL + _bursts(0.1, 20.0), None, None),
    ("_d1burst40l", D1_ALL + _bursts(0.4, 20.0), None, None),
]


def _d1_cells(cells) -> str:
    """``s515.d1_cells`` as a TOML array of inline tables, for --set. A cell is
    (tag, args, arms, budget), or with its size (..., n), or with flags for some of its
    arms alone (..., n, {arm: [flags]}), the launcher's ``arm_args``."""
    def lit(xs):
        return "[" + ",".join(f"'{x}'" for x in xs) + "]"
    rows = []
    for tag, args, arms, budget, *rest in cells:
        size = rest[0] if rest else None
        arm_args = rest[1] if len(rest) > 1 else None
        row = f"tag='{tag}',args={lit(args)},arms={lit(arms or EIGHT_ARMS)}"
        if budget:
            row += f",budget='{budget}'"
        if size:
            row += f",n={int(size)}"
        if arm_args:
            row += ",arm_args={" + ",".join(f"{arm}={lit(flags)}"
                                             for arm, flags in arm_args.items()) + "}"
        rows.append("{" + row + "}")
    return "s515.d1_cells=[" + ",".join(rows) + "]"


# DDIL build B10, the FQ track's cells for NOMS (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md, The
# FQ track; the task table's row 3): FQ-g75 (the recorded checkpoint, flown outside the physics
# it trained on, as a declared test) and FQ-s1 (jittery56 on step 1's physics, family
# jittery56-s1) in step 1's N = 6 cells, on the same tags and out-root as the DDIL1 probe; and
# N = 12 at 17 dB, where F and FX fly too (step 1 has no N = 12 cell). The narrow-jammed cell
# waits for build D8. FQ-s1 flies the runner's FQ-g75 arm with its own checkpoint; its job is
# blocked until FQ_S1_CHECKPOINT names it. Each N = 6 cell lists step 1's eight arms beside
# the FQ arms: their jobs have step 1's arguments and CSVs, so a flight skips them (every row
# is there) and the probe pairs FQ with them.
FQ_ARMS = ["FQ-g75", "FQ-s1"]
#: FQ-s1's checkpoint, repo-relative: trained 16 Nov 2026 from b0e84b8f (rl-sweep, out-root
#: results/exp5_fqs1, γ = 0.75, ten seeds), evaluated on the held-out stream in step 1's four
#: jittery cells, and picked as FQ-g75 was: the lower-median validation seed on Study 5.5's
#: cells (their step-1 twins jit-n12-90-s1 and jit-n12-180-s1), seed 3.
FQ_S1_CHECKPOINT: Optional[str] = "results/exp5_fqs1/checkpoints/5.5-jittery56-s1/g75/g0.75_s3.npz"
#: The per-class-phase cell is the plan's sensitivity cell: it flies shared interference off,
#: which FQ-s1 trained with on, so FQ-s1 flies it off its training conditions as a declared
#: test (user, 16 Nov), the one switch the runner lets it fly off there.
FQ_S1_OFF_SHARED = {"FQ-s1": ["--checkpoint-off-switches", "interference_shared"]}
DDILFQ1_CELLS = [
    ("_d1base", D1_ALL, EIGHT_ARMS + FQ_ARMS, None),
    ("_d1ampharsh", D1_ALL + D1_AMP[17], EIGHT_ARMS + FQ_ARMS, None),
    ("_d1ampharshpc", _without(D1_ALL, "--interference-shared") + D1_AMP[17],
     EIGHT_ARMS + FQ_ARMS, None, None, FQ_S1_OFF_SHARED),
    ("_d1burst40", D1_ALL + _bursts(0.4), EIGHT_ARMS + FQ_ARMS, None),
    ("_d1ampharsh", D1_ALL + D1_AMP[17], ["F", "FX"] + FQ_ARMS, None, 12),
]


def _learned_arms() -> str:
    """``s515.learned_arms`` for --set: FQ-s1 as the runner's FQ-g75 with its own checkpoint."""
    own = "arm='FQ-g75'" + (f",checkpoint='{FQ_S1_CHECKPOINT}'" if FQ_S1_CHECKPOINT else "")
    return "s515.learned_arms={FQ-s1={" + own + "}}"


# DDIL build D2, step 2 (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md; the task table's row 4): F
# with the risk-priced dwell, its reserve and the leave-by time (F-rp), and F-rp without the
# leave-by time (F-rpnl), as variants of the runner's F. On step 1's model in step 1's cells
# and out-root, so they pair with its F, FX and baselines; and on the recorded physics in
# P1's mission-backhaul cells and out-root (base, 17 dB and n = 2.7: the cells P1 flew there),
# so they pair with P1's recorded arms. Each cell lists the eight arms its out-root already
# holds: their jobs have the recorded arguments and CSVs, so a flight skips them. The reserve
# k = 1 is proposed, open to change before the flight.
D2_ARGS = ["--dwell-pricing", "expected", "--dwell-reserve-k", "1", "--leave-by", "true"]
D2_ARGS_NO_LEAVE = ["--dwell-pricing", "expected", "--dwell-reserve-k", "1"]
DDIL2_VARIANTS = ("s515.arm_variants={F-rp={arm='F',args=["
                  + ",".join(f"'{a}'" for a in D2_ARGS) + "]},F-rpnl={arm='F',args=["
                  + ",".join(f"'{a}'" for a in D2_ARGS_NO_LEAVE) + "]}}")
DDIL2_HONEST = [(tag, args, EIGHT_ARMS + ["F-rp"] + (["F-rpnl"] if tag == "_d1ampharsh" else []),
                 budget) for tag, args, arms, budget in DDIL1_MAIN if arms is None]
#: P1's mission-backhaul cells (recorded physics), by their axis tags and flags.
DDIL2_RECORDED = [
    ("_base", [], EIGHT_ARMS + ["F-rp"], None),
    ("_ampharsh", D1_AMP[17], EIGHT_ARMS + ["F-rp"], None),
    ("_npl", ["--n-pl", "2.7"], EIGHT_ARMS + ["F-rp"], None),
]


DDIL1_SETS =["s515.backhaul='mission'", "s515.arms=[]", "s515.amps_db=[]",
              "s515.shadow_sigmas_db=[]", "s515.mission_backhaul_arms=[]"]
#: DDIL build D0: the DDIL probes score with the scorer's --ddil-columns, into scored files
#: of their own (<csv stem>_scored_ddil.csv), so P1's and the campaign's are never rescored.
#: A score setting: no runner job's arguments move, so the flown jobs stay done.
DDIL_SCORE_SET = "score.s515.scorer_flags=['--ddil-columns']"

# The DDIL1N probe (17 Nov 2026, the user's go-ahead): Deadline-Heuristic (H1) and FedEx (D4)
# pinned to narrow, the band F chooses at N = 6, under step 1's model in its default and
# 17 dB cells, on step 1's seeds (they follow the cell and trial, not the arm), so they pair
# with step 1's F, FX and wide-band baselines. It asks how much of F's lead under the
# stricter contact model is the band and how much the plan, as P10 asked on the recorded
# physics. An out-root of its own: these jobs' names are step 1's wide-band ones.
DDIL1N_CELLS = [("_d1base", D1_ALL, ["H1", "D4"], None),
                ("_d1ampharsh", D1_ALL + D1_AMP[17], ["H1", "D4"], None)]


#: P6's 10 MB cells: the field's knee, and 12 x it with no limit in effect.
P6_10MB = ["s59x.budgets=['knee','relaxed']", "s59x.payload_bytes=10000000",
           "batch2.relaxed_factor=12"]


def _p6(half_width: int, extra: Sequence[str]) -> List[str]:
    """P6's N = 12 cells on a field: both fields' knee is 900 s (H1 pilots, 6 Nov 2026) and
    S* is 2. The 1 MB cells' F, FX, H1, D3 and D4 fly as batch 1's s59 in the same out-root."""
    return [f"campaign.field_half_width_m={half_width}", "pilot_outputs.knee_s.12=900",
            "pilot_outputs.stress_s.12=450", f"s59x.arms={EIGHT}", "s59x.N=[12]", "s59x.K=[1]",
            "s59x.contact_regimes=['jittery']", *extra]


PROBES: Dict[str, Probe] = {p.name: p for p in [
    Probe("P1", "Harsh radio, every baseline", [
        Flight("seconds backhaul", "batch3", "results/exp5",
               [f"s515.arms={EIGHT}", "s515.amps_db=[11.0]", "s515.mission_backhaul_arms=[]"],
               "s515"),
        Flight("mission backhaul", "batch3", "results/exp5_p1mbh",
               ["s515.backhaul='mission'", f"s515.arms={EIGHT}", "s515.amps_db=[]",
                "s515.shadow_sigmas_db=[]", "s515.mission_backhaul_arms=[]"], "s515"),
    ]),
    Probe("P2", "Budgets tighter than stress", [
        Flight("60 s", "batch2", "results/exp5_probe_tight40",
               _s53x(["batch2.relaxed_factor=0.4", "s53x.budgets=['relaxed']"]), "s53x"),
        Flight("45 s", "batch2", "results/exp5_probe_tight30",
               _s53x(["batch2.relaxed_factor=0.3", "s53x.budgets=['relaxed']"]), "s53x"),
    ]),
    Probe("P3", "MAX-AoI, Oort and FedCS at scale", [
        Flight("", "batch2", "results/exp5",
               ["s59x.arms=['F','D1','D2','D5']", "s59x.N=[12,24]", "s59x.K=[1]",
                "s59x.contact_regimes=['jittery']"], "s59x"),
    ]),
    Probe("P4", "Baselines in the far-device and short-reach cells", [
        Flight("", "batch2", "results/exp5",
               ["s54.arms=['F','H1','D1','D2','D3','D4','D5']", "s54.payloads=[1000000]",
                "s54.o1=false"], "s54"),
    ], ref_cells={"n6k1_knee_1mb_ratio2": "n6k1_knee_1mb",
                  "n6k1_knee_1mb_ratio3": "n6k1_knee_1mb"}),
    Probe("P5", "Bigger uploads at the N = 6 cell", [
        Flight(f"{mb} MB", "batch2", f"results/exp5_probe_{mb}mb",
               _s53x([f"s53x.payload_bytes={mb}000000", "s53x.budgets=['knee']",
                      "s53x.d4_faithful=false"]), "s53x",
               arms=["F", "FX", "H1", "D1", "D2", "D3", "D4", "D5"])
        for mb in (2, 5, 10)
    ]),
    Probe("P6", "Fields beyond the narrow band's reach", [
        Flight("200 m, 1 MB", "batch2", "results/exp5_probe_w200_1mb",
               _p6(200, ["s59x.budgets=['knee']"]), "s59x"),
        Flight("300 m, 1 MB", "batch2", "results/exp5_probe_w300r",
               _p6(300, ["s59x.budgets=['knee']"]), "s59x"),
        Flight("200 m, 10 MB", "batch2", "results/exp5_probe_w200_10mb",
               _p6(200, P6_10MB), "s59x"),
        Flight("300 m, 10 MB", "batch2", "results/exp5_probe_w300_10mb",
               _p6(300, P6_10MB), "s59x"),
        Flight("300 m, N = 24, 10 MB, 600 s", "batch2", "results/exp5_probe_w300_n24",
               ["campaign.field_half_width_m=300", "pilot_outputs.knee_s.24=600",
                "pilot_outputs.stress_s.24=300", "pilot_outputs.s_star.24=3",
                f"s59x.arms={EIGHT}", "s59x.N=[24]", "s59x.K=[1]",
                "s59x.contact_regimes=['jittery']", "s59x.budgets=['knee']",
                "s59x.payload_bytes=10000000"], "s59x"),
    ]),
    # DDIL plan step 5a (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md): P1's radio cells at the
    # measured payload (18.8 KB) and its knee (120 s), to see whether the 17 dB hover rise is
    # a 1 MB effect. Flown 16 Nov 2026; recorded physics, P1's seeds.
    Probe("DDIL5a", "Harsh radio at the measured payload", [
        Flight("seconds backhaul", "batch3", "results/exp5_dds",
               ["s515.payload_bytes='measured'", "s515.budget='knee_meas'", f"s515.arms={EIGHT}",
                "s515.mission_backhaul_arms=[]", "s515.amps_db=[11.0]"], "s515"),
        Flight("mission backhaul", "batch3", "results/exp5_ddm",
               ["s515.payload_bytes='measured'", "s515.budget='knee_meas'", f"s515.arms={EIGHT}",
                "s515.mission_backhaul_arms=[]", "s515.backhaul='mission'", "s515.amps_db=[]",
                "s515.shadow_sigmas_db=[]"], "s515"),
    ]),
    # DDIL step 1 (build D1), the NOMS slice: the main cells, then the checks, on P1's seeds.
    Probe("DDIL1", "Step 1: the honest DDIL channel", [
        Flight("main", "batch3", "results/exp5_ddil1",
               DDIL1_SETS + [_d1_cells(DDIL1_MAIN), DDIL_SCORE_SET], "s515"),
        Flight("checks", "batch3", "results/exp5_ddil1",
               DDIL1_SETS + [_d1_cells(DDIL1_CHECKS), DDIL_SCORE_SET], "s515"),
    ], ddil_endpoints=True),
    # DDIL build B10: the FQ track's NOMS cells, in step 1's out-root (FQ_ARMS above). The
    # track's reads: FQ-s1 against FQ-g75 (the physics change alone) and both against FX.
    Probe("DDILFQ1", "FQ-g75 and FQ-s1 on step 1's physics", [
        Flight("", "batch3", "results/exp5_ddil1",
               DDIL1_SETS + [_d1_cells(DDILFQ1_CELLS), _learned_arms(), DDIL_SCORE_SET],
               "s515"),
    ], ddil_endpoints=True, pairs=[("FQ-s1", "FQ-g75"), ("FQ-s1", "FX"), ("FQ-g75", "FX")]),
    # DDIL build D2, step 2: F-rp and F-rpnl beside step 1's arms and P1's recorded ones. The
    # leave-by time's own effect is F-rpnl against F-rp; FX is the other FeRRy reference.
    Probe("DDIL2", "Step 2: the risk-priced dwell and the leave-by time", [
        Flight("honest", "batch3", "results/exp5_ddil1",
               DDIL1_SETS + [_d1_cells(DDIL2_HONEST), DDIL2_VARIANTS, DDIL_SCORE_SET], "s515"),
        Flight("recorded", "batch3", "results/exp5_p1mbh",
               DDIL1_SETS + [_d1_cells(DDIL2_RECORDED), DDIL2_VARIANTS, DDIL_SCORE_SET],
               "s515"),
    ], ddil_endpoints=True, pairs=[("F-rpnl", "F-rp"), ("F-rp", "FX")]),
    # The baselines on narrow under step 1's model (DDIL1N_CELLS above); compared with step
    # 1's F, FX and wide-band runs by scripts/exp5/ddil_narrow_baselines.py.
    Probe("DDIL1N", "Step 1: the baselines pinned to narrow", [
        Flight("narrow", "batch3", "results/exp5_ddil1_bnarrow",
               DDIL1_SETS + [_d1_cells(DDIL1N_CELLS), "campaign.baseline_contact_band='narrow'",
                             DDIL_SCORE_SET], "s515"),
    ], ddil_endpoints=True, f_from="DDIL1", compared_by="ddil_narrow_baselines.py"),
    # The 4-mission cap lets F stop at about 335 s while the baselines fly about 800 s, so the
    # recorded compute cells compare unequal mission time. Ten missions gives every arm the
    # time; n_missions is in the seed hash, so these 40 trials are new seeds, not an extension.
    Probe("P11", "Compute cells flown to 10 missions", [
        Flight("", "batch3", "results/exp5_rd10",
               ["campaign.n_missions=10", "s512.arms=['F','FX','H1','D3','D4','D5']",
                "s512.payloads=[1000000]",
                "pilot_outputs.train_levels={spread=[68.0,0.5,0.0,1.0],"
                "stragglers=[68.0,0.5,0.2,5.0]}"], "s512", trials=40),
    ]),
]}


# --------------------------------------------------------------------- the tests
def mcnemar_exact(only_ref: int, only_var: int) -> float:
    """Two-sided exact McNemar p: the discordant pairs (only the reference reached tau,
    only the variant did) against Binomial(their sum, 1/2)."""
    n = only_ref + only_var
    if n == 0:
        return 1.0
    k = min(only_ref, only_var)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def compare_reach(ref, var, tau: float, family: str, plan, SC):
    """F (or a study's reference) against one arm on reaching tau, paired by trial and
    seed; mean_diff = reference share - variant share, so a claim favours the arm that
    reached tau more often."""
    import numpy as np
    from experiments.analysis.stats import bootstrap_ci

    col = f"reached_tau{tau:g}"
    c = SC.Comparison(family, var.cell, var.variant, ref.variant, col, "higher", 0)
    a, b = SC._values(ref, col), SC._values(var, col)
    xs: List[float] = []
    ys: List[float] = []
    for t in sorted(set(a) & set(b)):
        (seed_a, x), (seed_b, y) = a[t], b[t]
        if seed_a != seed_b:
            c.pairing_errors += 1
            continue
        if x is None or y is None:
            continue
        xs.append(1.0 if x else 0.0)
        ys.append(1.0 if y else 0.0)
    c.n_pairs = len(xs)
    notes = [f"{c.pairing_errors} trials with different seeds left out (pairing error)"] \
        if c.pairing_errors else []
    if c.n_pairs < 2:
        c.note = "; ".join(notes + ["fewer than 2 pairs"])
        return c
    only_ref = sum(1 for x, y in zip(xs, ys) if x > y)
    only_var = sum(1 for x, y in zip(xs, ys) if y > x)
    c.ref_mean = sum(xs) / len(xs)
    c.variant_mean = sum(ys) / len(ys)
    c.mean_diff, c.ci_low, c.ci_high = bootstrap_ci(
        [x - y for x, y in zip(xs, ys)], np.mean, n_bootstraps=plan.n_bootstraps)
    c.p_value = mcnemar_exact(only_ref, only_var)
    c.note = "; ".join(notes + [f"{only_ref} reached by {ref.variant} only, "
                                f"{only_var} by {var.variant} only"])
    return c


def compare_time(ref, var, tau: float, family: str, plan, SC):
    """The scorer's paired time-to-tau comparison (lower is better), in ``family``."""
    spec = SC.Spec(family, "sim_s_to_tau", "lower", reference=ref.variant)
    return SC.compare_pair(ref, var, f"sim_s_to_tau{tau:g}", spec, plan)


#: DDIL build D0's paired endpoints (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md): (kind, the
#: scored column, which way is better). Hover and updates per mission are the summary's
#: sim_dwell_s_mean and update_yield; the time to tau with a miss at the trial's end is the
#: scorer's --ddil-columns column, so no trial leaves that comparison.
D0_ENDPOINTS: Tuple[Tuple[str, str, str], ...] = (
    ("hover", "sim_dwell_s_mean", "lower"),
    ("updates", "update_yield", "higher"),
    ("time_or_end", "sim_s_to_tau{tau:g}_or_end", "lower"),
)
KIND_LABELS = {"time": "time to tau", "reach": "reach", "hover": "hover per mission",
               "updates": "updates per mission", "time_or_end": "time to tau, miss at the end"}


def compare_endpoint(ref, var, col: str, better: str, family: str, plan, SC):
    """The scorer's paired comparison of one D0 endpoint, in ``family``."""
    spec = SC.Spec(family, col, better, reference=ref.variant)
    return SC.compare_pair(ref, var, col, spec, plan)


def kind_of(metric: str) -> str:
    """The kind of a comparison, from its metric's column."""
    if metric.startswith("reached"):
        return "reach"
    if metric.endswith("_or_end"):
        return "time_or_end"
    return {"sim_dwell_s_mean": "hover", "update_yield": "updates"}.get(metric, "time")


def _compare_all(ref, var, tau: float, family: str, plan, SC, ddil: bool) -> List[Any]:
    """``var`` against ``ref`` on time and reach and, with ``ddil``, D0's endpoints; each
    kind in a family of its own (``family`` + "/" + kind)."""
    out = [fn(ref, var, tau, f"{family}/{kind}", plan, SC)
           for kind, fn in (("time", compare_time), ("reach", compare_reach))]
    if ddil:
        out += [compare_endpoint(ref, var, col.format(tau=tau), better, f"{family}/{kind}",
                                 plan, SC) for kind, col, better in D0_ENDPOINTS]
    return out


def reach_contradicts(favoured: str, reach) -> bool:
    """Whether the arm a time claim favours reached tau less often than the other arm
    (the reach comparison's point estimate)."""
    if reach is None or reach.ref_mean is None:
        return False
    mine, other = ((reach.ref_mean, reach.variant_mean) if favoured == reach.reference
                   else (reach.variant_mean, reach.ref_mean))
    return mine < other


# --------------------------------------------------------------------- probe scoring
def probe_comparisons(probe: Probe, cells: Dict[str, Dict[str, Any]], plan, SC,
                      tau: float = TAU) -> List[Any]:
    """Every arm against F in each cell, on time and on reach, and on D0's endpoints for a
    DDIL probe; then the probe's other pairs, in each cell that holds both. Holm per
    family: the baselines, FeRRy's other arms and the pairs apart, each kind apart."""
    comps = []
    for cell, variants in cells.items():
        pool = dict(variants)
        prefix, _, bare = cell.rpartition(": ")
        prefix = f"{prefix}: " if prefix else ""
        stand_in = probe.ref_cells.get(bare)
        if stand_in:
            for v, e in cells.get(prefix + stand_in, {}).items():
                if v != REFERENCE and (v not in pool or not pool[v].scored_rows):
                    pool[v] = e
        todo: List[Tuple[Any, Any, str]] = []
        ref = variants.get(REFERENCE)
        if ref is not None and ref.scored_rows:
            todo += [(ref, e, "ferry" if is_ferry(v) else "baselines")
                     for v, e in pool.items() if v != REFERENCE and e.scored_rows]
        for v, r in probe.pairs:
            a, b = pool.get(r), pool.get(v)
            if a is not None and b is not None and a.scored_rows and b.scored_rows:
                todo.append((a, b, "pairs"))
        for a, b, group in todo:
            for c in _compare_all(a, b, tau, f"{probe.name}/{group}", plan, SC,
                                  probe.ddil_endpoints):
                c.cell = cell
                comps.append(c)
    SC.holm(comps, plan)
    return comps


def _launcher():
    spec = importlib.util.spec_from_file_location("exp5_launch", HERE / "launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp5_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


def flight_cells(L, SC, flight: Flight, *, score: bool = False):
    """A flight's entries by cell (prefixed with the flight's label) and variant, and the
    plan. The whole stage is built, so a cell its study shares with another reads that
    study's CSV, as launch.py score reads it."""
    s, _ = L.load_settings(L.PARAMS)
    ns = argparse.Namespace(trials=flight.trials, set=list(flight.sets), dataset=None,
                            **{o: None for o, _, _ in L.LEVERS})
    with contextlib.redirect_stdout(io.StringIO()):
        L.apply_levers(s, ns)
        jobs = L.build(flight.stage, s, None, flight.out_root)
    runs, entries, _, plan = L.score_plan(flight.stage, s, jobs)
    mine = [e for e in entries.get(flight.study, [])
            if flight.arms is None or e.arm in flight.arms]
    if score:
        wanted = {str(e.scored) for e in mine}
        for j in runs:
            if str(REPO / j.out) in wanted and not L.done(j):
                print(f"scoring {j.out}")
                subprocess.run([sys.executable] + j.args, cwd=REPO, check=True)
    cells: Dict[str, Dict[str, Any]] = {}
    for e in mine:
        SC.load(e)
        key = f"{flight.label}: {e.cell}" if flight.label else e.cell
        cells.setdefault(key, {})[e.variant] = e
    return cells, plan


def _f(x: Optional[float], nd: int = 1) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def _diff(c, nd: int = 1, scale: float = 1.0) -> str:
    if c is None:
        return ""
    if c.mean_diff is None:
        return c.note
    return (f"{c.mean_diff * scale:.{nd}f} [{c.ci_low * scale:.{nd}f}, "
            f"{c.ci_high * scale:.{nd}f}]")


def _verdict(c) -> str:
    if c is None or c.p_holm is None:
        return ""
    return NAMES.get(c.favours, c.favours) if c.claim else "no claim"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    names: List[str] = []
    for r in rows:
        names += [k for k in r if k not in names]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def _mean_of(entry, col: str, SC) -> Optional[float]:
    vals = [SC._num(r.get(col)) for r in entry.scored_rows]
    vals = [x for x in vals if x is not None]
    return sum(vals) / len(vals) if vals else None


def _holm(c) -> str:
    return "" if c is None or c.p_holm is None else f"{c.p_holm:.4f}"


def _d0_table(cell: str, variants, borrowed, stand_in, by, SC, tau: float = TAU) -> List[str]:
    """A cell's D0 endpoints, each arm against F (DDIL build D0)."""
    cols = {kind: col.format(tau=tau) for kind, col, _ in D0_ENDPOINTS}
    lines = ["", f"D0's endpoints in {cell}, F − arm:", "",
             "| arm | hover per mission (s) | hover: F − arm [95% CI] | Holm p | verdict | "
             "updates per mission | updates: F − arm [95% CI] | Holm p | verdict | "
             "time to τ, miss at the end (s) | time: F − arm [95% CI] | Holm p | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, e in variants.items():
        name = NAMES.get(v, v)
        if not e.scored_rows and v != REFERENCE and v in borrowed:
            e, name = borrowed[v], f"{name} ({stand_in})"
        if not e.scored_rows:
            lines.append(f"| {name} | not flown |" + " |" * 11)
            continue
        row = f"| {name} |"
        for kind, nd in (("hover", 1), ("updates", 2), ("time_or_end", 1)):
            c = by.get((cell, v, REFERENCE, kind))
            row += (f" {_f(_mean_of(e, cols[kind], SC), nd)} | {_diff(c, nd)} | {_holm(c)} | "
                    f"{_verdict(c)} |")
        lines.append(row)
    return lines


def _pairs_table(comps) -> List[str]:
    """The probe's other pairs, every kind, in every cell that holds both arms."""
    pairs = [c for c in comps if c.study.split("/")[1:2] == ["pairs"]]
    lines = ["", "## Other pairs", "",
             "Difference = the second arm − the first (time and hover: negative when the "
             "second is faster or hovers less; reach and updates: positive when the second "
             "reaches τ more often or merges more).", "",
             "| cell | arm | against | measure | arm | against | against − arm [95% CI] | "
             "pairs | Holm p | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in pairs:
        kind = kind_of(c.metric)
        nd = 2 if kind in ("reach", "updates") else 1
        lines.append(f"| {c.cell} | {NAMES.get(c.variant, c.variant)} | "
                     f"{NAMES.get(c.reference, c.reference)} | {KIND_LABELS[kind]} | "
                     f"{_f(c.variant_mean, nd)} | {_f(c.ref_mean, nd)} | {_diff(c, nd)} | "
                     f"{c.n_pairs} | {_holm(c)} | {_verdict(c)} |")
    return lines if pairs else lines[:3] + ["- none flown (so far)"]


def probe_report(L, SC, probe: Probe, *, score: bool = False) -> List[str]:
    cells: Dict[str, Dict[str, Any]] = {}
    plan = None
    for flight in probe.flights:
        got, plan = flight_cells(L, SC, flight, score=score)
        cells.update(got)
    comps = probe_comparisons(probe, cells, plan, SC)
    by = {(c.cell, c.variant, c.reference, kind_of(c.metric)): c for c in comps}
    arms = [(cell, v, e) for cell, vs in cells.items() for v, e in vs.items()]
    flown = sum(1 for *_, e in arms if e.scored_rows)
    # Hover is a measure of what F's fixes act on, not an outcome: a baseline that leaves
    # the members below the floor unserved hovers less and merges less, so its hover
    # claims stay in the tables and the CSV, out of this list.
    losses = [c for c in comps if c.claim and c.reference == REFERENCE
              and c.favours != REFERENCE and not is_ferry(c.variant)
              and kind_of(c.metric) != "hover"]
    lines = [f"# {probe.name}: {probe.title}", "",
             "Exploratory, not pre-registered (DeveloperDocs/Experiment_5_Loss_Probes_Plan.md). "
             f"Reference F. Time to tau = {TAU:g} over the trials both arms reach it, and reach "
             "at tau (exact McNemar on the paired outcomes), each with a 95% bootstrap CI. "
             "Holm per family: the baselines on time, on reach, and FX on each apart. "
             "Difference = F - arm: time, negative when F is faster; reach, positive when F "
             "reaches tau more often.", ""]
    if probe.ddil_endpoints:
        lines += ["DDIL build D0's endpoints (DeveloperDocs/FeRRy_DDIL_Unified_Plan.md), paired "
                  "by trial and seed, each in a Holm family of its own: hover per mission "
                  "(sim_dwell_s_mean, lower is better), updates merged per mission "
                  "(update_yield, higher), and time to tau with a trial that missed it given "
                  "the trial's end, so no trial is dropped (lower).", ""]
    if probe.pairs:
        lines += ["Other pairs, in families of their own: "
                  + ", ".join(f"{NAMES.get(v, v)} against {NAMES.get(r, r)}"
                              for v, r in probe.pairs)
                  + "; the difference is the second arm minus the first.", ""]
    lines += [f"Flown and scored: {flown} of {len(arms)} arm-cells.", ""]
    lines += ["## Where a baseline beats F", ""]
    if probe.ddil_endpoints:
        lines += ["On time to τ, reach, updates per mission, or time with misses at the end. "
                  "A baseline's lower hover is listed in the cells' tables, not here: it comes "
                  "from leaving the members below the floor unserved.", ""]
    lines += ([f"- {c.cell}: {NAMES.get(c.variant, c.variant)} "
               f"({KIND_LABELS[kind_of(c.metric)]}; Holm p {c.p_holm:.4f})" for c in losses]
              or ["- none (so far)"])
    for cell, variants in cells.items():
        prefix, _, bare = cell.rpartition(": ")
        prefix = f"{prefix}: " if prefix else ""
        stand_in = probe.ref_cells.get(bare)
        borrowed = cells.get(prefix + stand_in, {}) if stand_in else {}
        lines += ["", f"## {cell}", ""]
        if stand_in:
            lines += [f"An arm not flown here is compared on its {stand_in} trials (same seeds), "
                      "marked with that cell's name.", ""]
        lines += ["| arm | trials scored | reach τ | mean time to τ (s) | time: F − arm [95% CI] | "
                  "Holm p | verdict | reach: F − arm [95% CI] | discordant | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v, e in variants.items():
            name = NAMES.get(v, v)
            if not e.scored_rows and v != REFERENCE and v in borrowed:
                e, name = borrowed[v], f"{name} ({stand_in})"
            n = len(e.scored_rows)
            if not n:
                lines.append(f"| {name} | not flown | | | | | | | | | |")
                continue
            reach = [SC._num(r.get(f"reached_tau{TAU:g}")) for r in e.scored_rows]
            reach = [x for x in reach if x is not None]
            times = [SC._num(r.get(f"sim_s_to_tau{TAU:g}")) for r in e.scored_rows]
            times = [x for x in times if x is not None]
            t = by.get((cell, v, REFERENCE, "time"))
            r = by.get((cell, v, REFERENCE, "reach"))
            lines.append(
                f"| {name} | {n} | {_f(sum(reach) / len(reach) if reach else None, 2)} | "
                f"{_f(sum(times) / len(times) if times else None)} | {_diff(t)} | "
                f"{'' if t is None or t.p_holm is None else f'{t.p_holm:.4f}'} | {_verdict(t)} | "
                f"{_diff(r, 2)} | {'' if r is None else r.note.split('; ')[-1]} | "
                f"{'' if r is None or r.p_holm is None else f'{r.p_holm:.4f}'} | {_verdict(r)} |")
        if probe.ddil_endpoints:
            lines += _d0_table(cell, variants, borrowed, stand_in, by, SC)
    if probe.pairs:
        lines += _pairs_table(comps)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{probe.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / f"{probe.name}.csv", [c.row() for c in comps])
    return lines


# --------------------------------------------------------------------- P11
#: P11's other questions. Every arm reaches tau in every trial, so its time test counts every
#: trial; the horizon check asks the same thing with the tail cut off at a time all arms flew.
#: The recorded no-delay cell has 20 trials and other seeds, so the cost is compared unpaired.
P11_TIMES = (150, 200, 250, 300, 400, 500)
P11_NO_DELAY = Flight("", "batch3", "results/exp5",
                      ["s512.arms=['F','FX','H1','D3','D4','D5']", "s512.payloads=[1000000]"], "s512")
P11_NO_DELAY_CELL = "n6k1_knee_1mb_none"


def trial_flights(entry) -> List[Tuple[float, int, int]]:
    """Each trial's flight time (first takeoff to last landing), updates merged and missions,
    from its mule trace."""
    out: List[Tuple[float, int, int]] = []
    for trace in Path(str(entry.csv_path)[:-4] + "_traces").glob("*/mule-exp4-mule.jsonl"):
        missions = [e for e in map(json.loads, trace.read_text(encoding="utf-8").splitlines())
                    if e.get("event") == "mission_completed"]
        if missions:
            out.append((max(m["sim_end_s"] for m in missions) -
                        min(m["sim_start_s"] for m in missions),
                        sum(m.get("pass_1_merged_updates") or 0 for m in missions),
                        len(missions)))
    return out


def as_of(entry, horizon: float, SC, tau: float = TAU):
    """The entry as seen at ``horizon``: a trial that reached tau later, or never, reached it
    not at all and counts as the horizon, so the mean is a restricted mean with no misses
    dropped."""
    time_col, reach_col = f"sim_s_to_tau{tau:g}", f"reached_tau{tau:g}"
    out = copy.copy(entry)
    out.scored_rows = []
    for row in entry.scored_rows:
        t = SC._num(row.get(time_col))
        made_it = t is not None and t <= horizon
        out.scored_rows.append({**row, time_col: t if made_it else horizon,
                                reach_col: 1.0 if made_it else 0.0})
    return out


def _times(entry, SC, tau: float = TAU) -> List[Optional[float]]:
    return [SC._num(r.get(f"sim_s_to_tau{tau:g}")) for r in entry.scored_rows]


def delay_cost(with_delay: Sequence[float], without: Sequence[float],
               n: int = 10000) -> Tuple[float, float, float]:
    """Mean(with_delay) - mean(without) and its 95% bootstrap CI, unpaired: the two runs have
    different seeds."""
    import numpy as np

    a, b = np.asarray(with_delay, float), np.asarray(without, float)
    rng = np.random.default_rng(0)
    sims = (rng.choice(a, (n, a.size)).mean(axis=1) - rng.choice(b, (n, b.size)).mean(axis=1))
    return float(a.mean() - b.mean()), *map(float, np.quantile(sims, [0.025, 0.975]))


def p11_report(L, SC, *, score: bool = False) -> List[str]:
    probe = PROBES["P11"]
    lines = probe_report(L, SC, probe, score=score)
    cells, plan = flight_cells(L, SC, probe.flights[0])
    flown = {c: {v: trial_flights(e) for v, e in vs.items()} for c, vs in cells.items()}
    horizons = {c: min(t[0] for fs in vs.values() for t in fs) for c, vs in flown.items()}
    seen = {c: {v: as_of(e, horizons[c], SC) for v, e in vs.items()} for c, vs in cells.items()}
    comps = probe_comparisons(Probe("P11 horizon", "", []), seen, plan, SC)
    by = {(c.cell, c.variant, c.metric.startswith("reached")): c for c in comps}
    no_delay = flight_cells(L, SC, P11_NO_DELAY)[0][P11_NO_DELAY_CELL]
    for cell, variants in cells.items():
        h = horizons[cell]
        lines += ["", f"## {cell}: at equal mission time", "",
                  f"The horizon, {h:.0f} s, is the shortest flight of any arm in any trial here, "
                  "so every arm has flown to it in every trial. Restricted mean: time to tau "
                  "capped at the horizon, with a miss counted as the horizon. Holm across "
                  "both cells, one family per metric.", "",
                  "| arm | flight mean (s) | merged per mission | restricted mean (s) | "
                  "F − arm [95% CI] | Holm p | verdict | reached by horizon: F − arm [95% CI] | "
                  "discordant | Holm p | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for v, e in variants.items():
            fs = flown[cell][v]
            t, r = by.get((cell, v, False)), by.get((cell, v, True))
            mean = sum(x for x in _times(seen[cell][v], SC)) / len(seen[cell][v].scored_rows)
            lines.append(
                f"| {NAMES.get(v, v)} | {sum(f[0] for f in fs) / len(fs):.0f} | "
                f"{sum(f[1] for f in fs) / sum(f[2] for f in fs):.2f} | {mean:.1f} | {_diff(t)} | "
                f"{'' if t is None or t.p_holm is None else f'{t.p_holm:.4f}'} | {_verdict(t)} | "
                f"{_diff(r, 2)} | {'' if r is None else r.note.split('; ')[-1]} | "
                f"{'' if r is None or r.p_holm is None else f'{r.p_holm:.4f}'} | {_verdict(r)} |")
        lines += ["", "Share of trials at tau by each time:", "",
                  "| arm | " + " | ".join(f"{x} s" for x in P11_TIMES) + " |",
                  "|---|" + "---|" * len(P11_TIMES)]
        for v, e in variants.items():
            ts = _times(e, SC)
            lines.append(f"| {NAMES.get(v, v)} | " + " | ".join(
                f"{sum(t is not None and t <= x for t in ts) / len(ts):.2f}"
                for x in P11_TIMES) + " |")
        lines += ["", f"What the delay costs: time to tau here against the recorded "
                  f"{P11_NO_DELAY_CELL} cell (no training delay, {P11_NO_DELAY.trials} trials, "
                  "4 missions, every arm reaching tau in every trial; other seeds, so unpaired "
                  "and for description, not a claim).", "",
                  "| arm | no delay (s) | here (s) | cost [95% CI] |", "|---|---|---|---|"]
        for v, e in variants.items():
            if v not in no_delay:
                continue
            base = [t for t in _times(no_delay[v], SC) if t is not None]
            here = [t for t in _times(e, SC) if t is not None]
            d, lo, hi = delay_cost(here, base)
            lines.append(f"| {NAMES.get(v, v)} | {sum(base) / len(base):.1f} | "
                         f"{sum(here) / len(here):.1f} | {d:.1f} [{lo:.1f}, {hi:.1f}] |")
    (OUT / "P11.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / "P11_horizon.csv", [c.row() for c in comps])
    return lines


# --------------------------------------------------------------------- P10
#: P10: the baselines on the bands F can choose, at N = 6 and 40 trials (the NOMS review's
#: comment 8). Every recorded baseline flew wide, so these runs ask how much of F's gain
#: the band alone gives. Each band has an out-root of its own, and the runs use batch 1's
#: study s53: under batch 2's s53x the launcher reads the cells as batch 1's.
P10_BANDS = ("narrow", "medium")
P10_ARMS = ("H1", "D4")
P10_WIDE = Flight("", "batch1", "results/exp5", ["s53.K=[1]", "s53.d4_faithful=false"], "s53",
                  arms=list(P10_ARMS), trials=40)
#: F's recorded N = 6 cells. s53's F shares its job with s514's capS, which owns the
#: files, and a stage built here resolves the pair to s53's path, which was never flown.
P10_F_DIR = "results/exp5/b1/s514"
P10_FLOWN = {band: Flight("", "batch1", f"results/exp5_probe_b{band}",
                          [f"campaign.baseline_contact_band='{band}'", "s53.K=[1]",
                           "s53.d4_faithful=false"], "s53", arms=list(P10_ARMS), trials=40)
             for band in P10_BANDS}
P10_PINNED_SETS = ["s54.arms=['FB+narrow','FB+medium']", "s54.narrow_range_ratios=[]",
                   "s54.far_shares=[]", "s54.payloads=[1000000]", "s54.o1=false"]
P10_PINNED = (Flight("", "batch2", "results/exp5", P10_PINNED_SETS, "s54", trials=40),
              Flight("", "batch2", "results/exp5_probe_fbs",
                     [*P10_PINNED_SETS, "s54.budget='stress'"], "s54", trials=40))


def p10_f(SC, cell: str):
    """F's recorded 40 trials in ``cell``, read from the files that hold them."""
    base = REPO / P10_F_DIR
    entry = SC.Entry("s514", cell, REFERENCE, REFERENCE, base / f"{cell}__capS.csv",
                     base / f"{cell}__capS_scored.csv", 40)
    SC.load(entry)
    return entry


def _renamed(entry, name: str):
    """A copy of an entry under the name P10 compares it by (one arm flies three bands)."""
    out = copy.copy(entry)
    out.variant = name
    return out


def p10_comparisons(arms: Dict[str, Dict[str, Any]], plan, SC, tau: float = TAU) -> List[Any]:
    """Each baseline on each band against F choosing its band, F pinned to that band, and
    itself on wide, on time and on reach. Holm per family; ``arms`` is budget -> name ->
    entry, named F, F@<band> and <arm>@<band>."""
    comps = []
    for budget, pool in arms.items():
        for band in P10_BANDS:
            for arm in P10_ARMS:
                var = pool.get(f"{arm}@{band}")
                if var is None or not var.scored_rows:
                    continue
                for family, name in (("against F", REFERENCE), ("against pinned F", f"F@{band}"),
                                     ("against itself on wide", f"{arm}@wide")):
                    ref = pool.get(name)
                    if ref is None or not ref.scored_rows:
                        continue
                    for kind, fn in (("time", compare_time), ("reach", compare_reach)):
                        c = fn(ref, var, tau, f"P10/{family}/{kind}", plan, SC)
                        c.cell = budget
                        comps.append(c)
    SC.holm(comps, plan)
    return comps


def p10_report(L, SC, *, score: bool = False) -> List[str]:
    arms: Dict[str, Dict[str, Any]] = {"knee": {}, "stress": {}}
    plan = None

    def take(flight: Flight, name, *, scores: bool = False) -> None:
        nonlocal plan
        got, plan = flight_cells(L, SC, flight, score=score and scores)
        for cell, variants in got.items():
            budget = next((b for b in arms if f"_{b}" in cell), None)
            for v, e in variants.items():
                if budget is not None and name(v):
                    arms[budget][name(v)] = _renamed(e, name(v))

    # Only the probe's own out-roots are scored: the recorded cells stay as they are.
    take(P10_WIDE, lambda v: f"{v}@wide")
    for band, flight in P10_FLOWN.items():
        take(flight, lambda v, band=band: f"{v}@{band}", scores=True)
    take(P10_PINNED[0], lambda v: f"F@{v[len('FBp'):]}")
    take(P10_PINNED[1], lambda v: f"F@{v[len('FBp'):]}", scores=True)
    for budget in arms:
        arms[budget][REFERENCE] = p10_f(SC, f"n6k1_{budget}")
    comps = p10_comparisons(arms, plan, SC)
    lines = ["# P10: The baselines on the bands F can choose", "",
             "Exploratory, not pre-registered (the NOMS review's comment 8). N = 6, one mule, "
             "1 MB, 40 trials a cell, paired by seed. Time to tau = 0.71 over the trials both "
             "arms reach it, and reach at tau (exact McNemar), each with a 95% bootstrap CI. "
             "Holm per family: against F choosing its band, against F pinned to the same band, "
             "and against the arm itself on wide, on each metric apart. Difference = reference "
             "- arm: time, negative when the reference is faster.", ""]
    for budget, pool in arms.items():
        lines += [f"## {budget}", "", "| arm | trials scored | reach τ | mean time to τ (s) |",
                  "|---|---|---|---|"]
        for name, e in sorted(pool.items()):
            n = len(e.scored_rows)
            if not n:
                lines.append(f"| {name} | not flown | | |")
                continue
            reach = [SC._num(r.get(f"reached_tau{TAU:g}")) for r in e.scored_rows]
            reach = [x for x in reach if x is not None]
            times = [SC._num(r.get(f"sim_s_to_tau{TAU:g}")) for r in e.scored_rows]
            times = [x for x in times if x is not None]
            lines.append(f"| {name} | {n} | {_f(sum(reach) / len(reach) if reach else None, 2)} | "
                         f"{_f(sum(times) / len(times) if times else None)} |")
        lines += ["", "| arm | reference | metric | pairs | difference [95% CI] | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|"]
        for c in comps:
            if c.cell != budget:
                continue
            is_reach = c.metric.startswith("reached")
            p = "" if c.p_holm is None else f"{c.p_holm:.4f}"
            lines.append(f"| {c.variant} | {c.reference} | {'reach' if is_reach else 'time'} | "
                         f"{c.n_pairs} | {_diff(c, 2 if is_reach else 1)} | {p} | {_verdict(c)} |")
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "P10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / "P10.csv", [c.row() for c in comps])
    return lines


# --------------------------------------------------------------------- P9
def recorded_claims(stage_dir: Path, study: str) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """The pre-registered time-to-tau claims of one study, by (cell, variant, reference)."""
    path = stage_dir / f"{study}_comparisons.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["cell"], r["variant"], r["reference"]): r for r in rows
            if r.get("claim") == "True" and r.get("metric", "").startswith("sim_s_to_tau")}


def p9_report(L, SC) -> List[str]:
    comps: List[Any] = []
    stage_of: Dict[int, str] = {}
    claims: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    plan = None
    for stage in P9_STAGES:
        s, _ = L.load_settings(L.PARAMS)
        with contextlib.redirect_stdout(io.StringIO()):
            jobs = L.build(stage, s, None, "results/exp5")
        _, entries, _, plan = L.score_plan(stage, s, jobs)
        stage_dir = REPO / "results/exp5/scores" / L.STAGE_DIR[stage]
        for study, members in entries.items():
            spec = plan.specs.get(study)
            if spec is None or not spec.metric:
                continue
            for key, row in recorded_claims(stage_dir, study).items():
                claims[(stage, study, *key)] = row
            by_cell: Dict[str, Dict[str, Any]] = {}
            for e in members:
                SC.load(e)
                if e.scored_rows:
                    by_cell.setdefault(e.cell, {})[e.variant] = e
            for cell, variants in by_cell.items():
                for name, e in variants.items():
                    ref = spec.reference_of(name)
                    if ref is None or ref not in variants:
                        continue
                    for kind, tau, fn in (("reach", TAU, compare_reach),
                                          ("time", SECOND_TAU, compare_time),
                                          ("reach", SECOND_TAU, compare_reach)):
                        c = fn(variants[ref], e, tau, f"p9/{stage}/{study}/{kind}{tau:g}",
                               plan, SC)
                        stage_of[id(c)] = stage
                        comps.append(c)
    SC.holm(comps, plan)

    def study_of(c) -> str:
        return c.study.split("/")[2]

    def kind_of(c) -> str:
        return c.study.split("/")[3]

    reach71 = {(stage_of[id(c)], study_of(c), c.cell, c.variant, c.reference): c
               for c in comps if kind_of(c) == f"reach{TAU:g}"}
    lines = ["# P9: reach at tau, and the second tau, on every recorded comparison", "",
             "Exploratory, not pre-registered (3 Nov 2026). Every comparison the pre-registered "
             "studies make, from the scored files of batch 1, sens, batch 2 and batch 3 "
             "(results/exp5). Reach: exact McNemar on the paired reached / not-reached outcomes, "
             "with a 95% bootstrap CI on the paired difference in reach share (reference - "
             "variant). Time: the scorer's paired comparison at the second tau. Holm across "
             "each study and test. The studies' own comparisons (scores/<batch>/) are unchanged.",
             "", "## Summary", "",
             f"| batch | study | comparisons | reach at {TAU:g}: claims for the reference / the variant | "
             f"time to {SECOND_TAU:g}: comparisons with 2+ pairs | claims | "
             f"reach at {SECOND_TAU:g}: claims |",
             "|---|---|---|---|---|---|---|"]
    keys = sorted({(stage_of[id(c)], study_of(c)) for c in comps},
                  key=lambda k: (P9_STAGES.index(k[0]), k[1]))
    for stage, study in keys:
        mine = [c for c in comps if stage_of[id(c)] == stage and study_of(c) == study]
        r71 = [c for c in mine if kind_of(c) == f"reach{TAU:g}"]
        t82 = [c for c in mine if kind_of(c) == f"time{SECOND_TAU:g}"]
        r82 = [c for c in mine if kind_of(c) == f"reach{SECOND_TAU:g}"]
        lines.append(
            f"| {L.STAGE_DIR[stage]} | {study} | {len(r71)} | "
            f"{sum(c.claim and c.favours == c.reference for c in r71)} / "
            f"{sum(c.claim and c.favours == c.variant for c in r71)} | "
            f"{sum(c.p_value is not None for c in t82)} | {sum(c.claim for c in t82)} | "
            f"{sum(c.claim for c in r82)} |")

    lines += ["", "## Recorded time-to-tau claims whose favoured arm reached tau less often", "",
              "A recorded claim (the study's own score file) whose favoured arm reached "
              f"tau = {TAU:g} in fewer of the paired trials than the other arm.", "",
              "| batch | study | cell | favoured | other | time diff (ref − variant) | "
              "reach favoured | reach other | reach Holm p | reach verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    flagged = 0
    for (stage, study, cell, variant, ref), row in sorted(claims.items()):
        r = reach71.get((stage, study, cell, variant, ref))
        fav = row["favours"]
        if not reach_contradicts(fav, r):
            continue
        flagged += 1
        mine, other = ((r.ref_mean, r.variant_mean) if fav == r.reference
                       else (r.variant_mean, r.ref_mean))
        lines.append(f"| {L.STAGE_DIR[stage]} | {study} | {cell} | {fav} | "
                     f"{variant if fav == ref else ref} | {float(row['mean_diff']):.1f} | "
                     f"{mine:.2f} | {other:.2f} | {r.p_holm:.4f} | {_verdict(r)} |")
    if not flagged:
        lines.append("| | | none | | | | | | | |")

    for kind, title in ((f"reach{TAU:g}", f"Reach claims at tau = {TAU:g}"),
                        (f"time{SECOND_TAU:g}", f"Time-to-tau claims at tau = {SECOND_TAU:g}"),
                        (f"reach{SECOND_TAU:g}", f"Reach claims at tau = {SECOND_TAU:g}")):
        rows = [c for c in comps if kind_of(c) == kind and c.claim]
        scale, nd = (1.0, 1) if kind.startswith("time") else (1.0, 2)
        lines += ["", f"## {title}", "",
                  "| batch | study | cell | reference | variant | n pairs | reference | variant | "
                  "difference [95% CI] | Holm p | favours | note |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in rows:
            lines.append(f"| {L.STAGE_DIR[stage_of[id(c)]]} | {study_of(c)} | {c.cell} | "
                         f"{c.reference} | {c.variant} | {c.n_pairs} | {_f(c.ref_mean, nd)} | "
                         f"{_f(c.variant_mean, nd)} | {_diff(c, nd, scale)} | {c.p_holm:.4f} | "
                         f"{NAMES.get(c.favours, c.favours)} | {c.note} |")
        if not rows:
            lines.append("| | | none | | | | | | | | | |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "p9.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(OUT / "p9.csv", [{"batch": L.STAGE_DIR[stage_of[id(c)]], "test": kind_of(c),
                                 **c.row()} for c in comps])
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("which", nargs="+",
                    help="p9, P10, a probe (" + ", ".join(PROBES) + ") or all")
    ap.add_argument("--score", action="store_true",
                    help="score a probe's trial CSVs that have no scored file yet")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    names = {k.lower(): k for k in list(PROBES) + ["p9", "P10"]}
    which = [names.get(w.lower(), w.upper()) for w in a.which]
    if "ALL" in which:
        which = ["p9"] + list(PROBES) + ["P10"]
    unknown = [w for w in which if w not in ("p9", "P10") and w not in PROBES]
    if unknown:
        sys.exit(f"unknown probe(s) {', '.join(unknown)}: p9, {', '.join(PROBES)} or all")
    L = _launcher()
    SC = L._scoring()
    for w in which:
        probe = PROBES.get(w)
        if probe is not None and probe.compared_by:
            # Its cells hold no F; its own script compares it with f_from's F and writes its
            # report, which a report from here would overwrite.
            print(f"{w}: compared with {probe.f_from}'s F by scripts/exp5/{probe.compared_by}; "
                  f"run that (score first with ddil_score_probes.py {w})\n")
            continue
        lines = (p9_report(L, SC) if w == "p9"
                 else p10_report(L, SC, score=a.score) if w == "P10"
                 else p11_report(L, SC, score=a.score) if w == "P11"
                 else probe_report(L, SC, PROBES[w], score=a.score))
        print("\n".join(lines[:40]) + ("\n…" if len(lines) > 40 else ""))
        print(f"\nwritten: {OUT / (w + '.md')}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
