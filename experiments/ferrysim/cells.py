"""FerrySim's cells and seed streams (FeRRy Phase 5, unit U8a).

**The cells** (the user's decision 3 (a); critic A1, A9 and C3). The pair score
matters only where it has real choices: at Exp 5's defaults (the clean contact
channel and the measured 18.8 KB model) FX flies exactly as F
(Experiment_4_Run_Guide.md section 2.7), and with 6 devices a collection flight
makes one stop in 66 to 72 of 72 flights (the Phase 5 design, finding 1), so
looking ahead cannot matter there. Each cell therefore pre-registers its
contact regime and its payload (critic A9), and the cells are:

* the **control** at N = 6, Exp 5's size, with the jittery contact channel at
  1 MB: declared as the cell where looking ahead cannot matter (critic A1);
  Study 5.5 is not read on it. Its budgets are Phase 4's priors, the knee
  90 s and the stress budget 45 s (the Phase 5 design D-G (a);
  HERMES_Configuration_Reference.md section 18.7), until the Phase 3/4 pilots
  measure them;
* the **decision-rich** cell at N = 12, jittery, 1 MB, with 2 to 5 Pass-1
  stops per flight, where Studies 5.5 and 5.6 are read. Its knee and stress
  budget need a pilot (critic B6), so the stand-in budgets 120 s and 180 s
  are flown until one measures them (decision 3);
* the **negative control**: the clean contact channel at 1 MB and N = 12, the
  same budgets (critic C3: at the measured payload the slot would be vacuous,
  so that control could not fail).

The cap S of every cell is the S* tool's at the cell's two budgets
(``experiments.analysis.age_cap_s_star``; the user's decision 1 of Phase 4:
F's S* on 90 % of 30 layouts at every budget given, never below 2), run at
planning level during the build (the Phase 5 spec, "What needs your
go-ahead", item 5):

    --N 6  --budgets 90 45   --contact-band wide --payload-bytes 1000000
      --regime jittery --ferry-physics '{"contact_regime": "jittery"}'   -> S = 2
    --N 12 --budgets 120 180 (same flags, contact regime jittery or clean) -> S = 2

(``python -m experiments.analysis.age_cap_s_star ... --families F``; a test
re-runs it.) The rest of a cell is Phase 4's pilot configuration
(Experiment_4_Run_Guide.md section 2.7, as UG5's trials fly it,
``tests/golden/_build_p4_plan.py``): the simulated clock, wide as the
reference class, realism, the T_nom deadline unit, ``replan`` with the
``trim`` fallback (the pair slot needs ``replan``: orchestrator resolution
R3), ``agg:cutoff``, the channel reliability source, 4 missions. The cell's
network regime is ``jittery`` in every cell, so only the contact regime
differs between the families.

**One score per contact regime** (decision 3): :data:`FAMILIES` groups the
cells a score practises over, both sizes and both budgets of the jittery
regime, and both budgets of the clean control.

**Study 5.6's cells** (the user's decision 6 (a); critic A3; the
orchestrator's resolution R22): the decision-rich cell at each stand-in budget,
but for its name and its contact channel's interference period P_c, set to 4 x
and 2 x the lag between decisions, so that the lag is a quarter (``-q``) or a
half (``-h``) of the period; the clean N = 12 cells are their control. The lag
is A3's, from one Pass-1 arrival to the next in the same sortie
(:func:`arrival_lags`), not the leg: FX's median on the validation stream of
the matching Study 5.5 cell at the default P_c, pooled over the lags of its
first 200 episodes and rounded to the nearest second (:data:`STUDY_5_6_LAGS_S`;
``evaluate.fx_lag_median`` measures it); the cells are re-pinned with the
budgets after the N = 12 pilot, on the same sample and statistic. They stay
out of :data:`CELLS` (decision 3's cells, the headroom report's default) and
out of the families ``jittery`` and ``clean``, whose cells and hashes are
unchanged. The family ``jittery56`` is the jittery cells and these four, for a
jittery score that practises at Study 5.6's periods too; whether it does is the
user's choice at the campaign, made before the 5.5 sweep, since a manifest
records its family's hash (R22). Study 5.5 is read on its own two cells
(:data:`STUDY_5_5_CELLS`), whichever family trained the score.

**The scale family** (the build plan's Exp 5 addendum, Study 5.11 (c):
FerrySim beyond the stack). The jittery decision-rich cell at N = 24, 48 and
96, one mule, on a field that grows with N at N = 6's density (half-width
``100 * sqrt(N / 6)`` m: 200, 282.8 and 400 m;
``experiments.exp4.topology_builder.grown_field_radius_m``), since the
realism field's fixed 100 m would raise the density with N. Each size flies
two stand-in budgets until its budget pilot (``python -m experiments.ferrysim
pilot``) measures the knee: the binding edge, the largest budget on a 10 s grid
at which F's S* on 90 % of the S* tool's 30 layouts is still 2 (one mission
can no longer serve every servable device on more than a tenth of them), and
1.5 times it. The rule reproduces the N = 12 stand-ins exactly (edge 120 s,
and 180 s); at N = 6 its edge is 80 s, beside the priors 45 and 90 s. Found by
bisection with the S* tool at planning level (``--families F --field-ref-n 6``
with the cells' flags), on 2 Oct 2026:

    N = 24, field 200.0 m:  edge  350 s, 1.5 x edge  525 s   -> S = 2
    N = 48, field 282.8 m:  edge  680 s, 1.5 x edge 1020 s   -> S = 2
    N = 96, field 400.0 m:  edge 1330 s, 1.5 x edge 1995 s   -> S = 2

(S* on 90 % of layouts is 2 at the edge and 1 at 1.5 x, so S takes decision
1's floor, 2, as everywhere.) These cells stay out of :data:`CELLS` and out of
every other family, whose cells and hashes are unchanged: the cell's field
(``field_radius_m``) is left out of a cell's JSON when it is the driver's
default (None), as every other cell's is. The learned score trains at N = 6
and 12, so a score flown here is out of practice (its /N features shift), a
declared test.

**The seed streams** (the spec, other choices 8; critic B14). A FerrySim
episode is one trial of a cell, and its trial seed (the driver's
``Cell.seed``: the layout, the availability, the channel's phases and every
keyed draw) comes from one of three streams: ``ferrysim-train-<seed>`` for
training run ``<seed>``, ``ferrysim-val`` for validation (the headroom report
and ε use it) and ``ferrysim-heldout`` for Study 5.5's held-out judgement,
shared by every checkpoint and reference (common random numbers). They are
disjoint by construction: a trial seed is 32 bits, as the runner's are
(``experiments/runner/grid.py``), its top two bits name its stream kind
(:data:`STREAM_TAGS`) and the other 30 are a SHA-256 of the stream, the cell and
the episode index; within a stream a repeated seed is skipped, so a stream's
episodes are distinct. :func:`check_disjoint` checks any set of seeds against
their streams. A seed is a pure function of its stream, cell and index, so a
stream never moves when another is extended.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from experiments.runner import Cell

#: Every cell's payload per direction (bytes): 1 MB, declared (decision 3).
PAYLOAD_BYTES = 1_000_000
#: Missions per episode: one trial of four missions (the Phase 5 design D-G).
N_MISSIONS = 4
#: The RF range (m), the grid's default and wide's planar range.
RF_RANGE_M = 60.0
#: The cells' network (backhaul) regime: the same in every cell, so the
#: families differ in their contact channel only (critic A9).
NETWORK_REGIME = "jittery"

ROLE_CONTROL = "control"
ROLE_DECISION_RICH = "decision-rich"
ROLE_NEGATIVE_CONTROL = "negative-control"
ROLES: Tuple[str, ...] = (ROLE_CONTROL, ROLE_DECISION_RICH, ROLE_NEGATIVE_CONTROL)

#: Where a cell's budget comes from: Phase 4's priors at N = 6, the stand-in
#: budgets at N = 12 until a pilot measures that cell's knee (critic B6).
BUDGET_KNEE_PRIOR = "knee-prior"
BUDGET_STRESS_PRIOR = "stress-prior"
BUDGET_STAND_IN = "stand-in"

FAMILY_JITTERY = "jittery"
FAMILY_CLEAN = "clean"
#: The jittery regime's second family: its cells and Study 5.6's (resolution R22).
FAMILY_JITTERY_56 = "jittery56"
#: Study 5.11 (c)'s cells at N = 24, 48 and 96 on a growing field (the Exp 5
#: addendum): :data:`SCALE_CELLS`.
FAMILY_SCALE = "scale"
#: The reference size whose density the scale family keeps (N = 6 in the
#: realism field's 100 m).
SCALE_REF_N = 6


@dataclasses.dataclass(frozen=True)
class FerryCell:
    """One FerrySim cell: the trial a FerrySim episode runs, but for its seed.

    ``family`` is the contact regime the cell's score practises in (one score
    per regime; Study 5.6's cells are the jittery regime's, whichever of its
    families the score practises over), ``role`` what the cell is for
    (:data:`ROLES`), ``cap_s`` the S* tool's S at the family's budgets for this
    size. :meth:`driver_settings` and :meth:`cell` give exactly what
    ``Exp4Driver(**settings).run_trial(cell)`` takes, the arm's own
    configuration aside (the arm is the episode's). ``interference_period_s``
    sets the contact channel's interference period P_c (None: the regime's own,
    60 s; ``CONTACT_REGIMES`` in ``hermes/l1/channel_model.py``), for Study
    5.6's cells (decision 6 (a); :data:`STUDY_5_6_CELLS`). ``field_radius_m``
    sets the realism field's half-width (None: the driver's 100 m), for the
    scale family (:data:`SCALE_CELLS`); at None it is left out of
    :meth:`to_json`, so the other cells' JSON, and their families' hashes,
    are unchanged.
    """

    name: str
    family: str
    role: str
    n_devices: int
    budget_s: float
    budget_role: str
    cap_s: int
    contact_regime: str
    n_missions: int = N_MISSIONS
    payload_bytes: int = PAYLOAD_BYTES
    rf_range_m: float = RF_RANGE_M
    network_regime: str = NETWORK_REGIME
    interference_period_s: Optional[float] = None
    field_radius_m: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "|" in self.name:
            raise ValueError(f"a cell's name is a non-empty string without '|', got {self.name!r}")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")
        for field_name in ("n_devices", "cap_s", "n_missions", "payload_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an int >= 1, got {value!r}")
        if not float(self.budget_s) > 0.0:
            raise ValueError(f"budget_s must be > 0, got {self.budget_s!r}")
        period = self.interference_period_s
        if period is not None and not (isinstance(period, (int, float))
                                       and not isinstance(period, bool) and period > 0.0):
            raise ValueError(f"interference_period_s must be > 0 or None, got {period!r}")
        field_m = self.field_radius_m
        if field_m is not None and not (isinstance(field_m, (int, float))
                                        and not isinstance(field_m, bool) and field_m > 0.0):
            raise ValueError(f"field_radius_m must be > 0 or None, got {field_m!r}")

    def driver_settings(self) -> Dict[str, Any]:
        """The ``Exp4Driver`` settings of this cell's trials (Phase 4's pilot flags).

        As UG5's ``PLAN_PILOT`` (``tests/golden/_build_p4_plan.py``) with this
        cell's budget, cap and contact regime (and its interference period and
        field, when set).
        """
        physics: Dict[str, Any] = {"contact_regime": str(self.contact_regime)}
        if self.interference_period_s is not None:
            physics["interference_period_s"] = float(self.interference_period_s)
        settings = dict(
            mission_clock="sim", realism=True, contact_band="wide",
            deadline_time_scale="t_nom", in_flight_response="replan",
            replan_fallback="trim", aggregation="agg:cutoff",
            contact_reliability_source="channel", payload_bytes=int(self.payload_bytes),
            mission_budget_s=float(self.budget_s), age_cap_missions=int(self.cap_s),
            ferry_physics=physics,
        )
        if self.field_radius_m is not None:
            settings["h1_field_radius_m"] = float(self.field_radius_m)
        return settings

    def cell_params(self) -> Dict[str, Any]:
        """The runner's grid axes of this cell (``Cell.params``)."""
        return {"N": int(self.n_devices), "rrf": float(self.rf_range_m),
                "n_missions": int(self.n_missions), "regime": str(self.network_regime)}

    @property
    def cell_id(self) -> str:
        """The runner's cell id of these axes (``key=value|...``, sorted), so a
        kept trace's directory parses as the trace scorer expects."""
        p = self.cell_params()
        return "|".join(f"{k}={p[k]}" for k in sorted(p))

    def cell(self, arm: str, seed: int, trial_index: int = 0) -> Cell:
        """The driver's ``Cell`` of one episode of this cell, flown by ``arm``."""
        return Cell(cell_id=self.cell_id, arm=str(arm), trial_index=int(trial_index),
                    seed=int(seed), params=self.cell_params())

    def to_json(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if out["field_radius_m"] is None:
            # The driver's field: left out, so a cell from before the scale
            # family hashes as it did.
            del out["field_radius_m"]
        return out


#: The cells (decision 3 (a)). S = 2 everywhere: the S* tool's S at each
#: size's two budgets (module docstring).
CELLS: Tuple[FerryCell, ...] = (
    FerryCell("jit-n6-45", FAMILY_JITTERY, ROLE_CONTROL, 6, 45.0, BUDGET_STRESS_PRIOR, 2,
              "jittery"),
    FerryCell("jit-n6-90", FAMILY_JITTERY, ROLE_CONTROL, 6, 90.0, BUDGET_KNEE_PRIOR, 2,
              "jittery"),
    FerryCell("jit-n12-120", FAMILY_JITTERY, ROLE_DECISION_RICH, 12, 120.0, BUDGET_STAND_IN, 2,
              "jittery"),
    FerryCell("jit-n12-180", FAMILY_JITTERY, ROLE_DECISION_RICH, 12, 180.0, BUDGET_STAND_IN, 2,
              "jittery"),
    FerryCell("cln-n12-120", FAMILY_CLEAN, ROLE_NEGATIVE_CONTROL, 12, 120.0, BUDGET_STAND_IN, 2,
              "clean"),
    FerryCell("cln-n12-180", FAMILY_CLEAN, ROLE_NEGATIVE_CONTROL, 12, 180.0, BUDGET_STAND_IN, 2,
              "clean"),
)

#: Study 5.6's lags (s), A3's from one Pass-1 arrival to the next in the same
#: sortie (:func:`arrival_lags`), by the Study 5.5 cell they were measured on
#: (decision 6 (a); critic A3's rule; resolution R22): FX's median over the
#: first 200 episodes of that cell's validation stream (``ferrysim-val``) at the
#: default P_c of 60 s, pooled over every lag of the sample, rounded to the
#: nearest second. ``experiments.ferrysim.evaluate.fx_lag_median(cell)`` measures
#: it (the FX arm itself, no training; a slow test re-measures both cells). The
#: Phase 5 fix round measured, at jit-n12-120, 26.42 s over 1,397 lags (IQR 20.1
#: to 34.0 s; the sample's halves 26.76 and 26.27 s) and, at jit-n12-180, 34.11 s
#: over 397 lags (IQR 26.6 to 47.6 s; halves 32.85 and 36.06 s). The integers
#: are a pre-registration convention fixed by this sample and statistic, not a
#: measurement to the second: the pooled median's 95 % interval (episodes
#: resampled whole) is about 25.4 to 27.5 s and 32.2 to 36.9 s, and the first
#: 400 episodes give 26.80 s and 35.12 s, which round to 27 and 35. The cells are
#: re-pinned with the N = 12 pilot's budgets, measured again on the same sample
#: and statistic.
STUDY_5_6_LAGS_S: Dict[str, int] = {"jit-n12-120": 26, "jit-n12-180": 34}
#: Study 5.6's interference periods P_c (s): the quarter cells fly 4 x the lag
#: (lag / P_c = 0.25), the half cells 2 x (0.5). The half cells follow the rule
#: too, though their 52 s and 68 s lie near the default 60 s.
P_C_QUARTER_120_S = 4 * STUDY_5_6_LAGS_S["jit-n12-120"]     # 104 s
P_C_HALF_120_S = 2 * STUDY_5_6_LAGS_S["jit-n12-120"]        # 52 s
P_C_QUARTER_180_S = 4 * STUDY_5_6_LAGS_S["jit-n12-180"]     # 136 s
P_C_HALF_180_S = 2 * STUDY_5_6_LAGS_S["jit-n12-180"]        # 68 s

#: Study 5.6's cells (decision 6 (a); resolution R22): each is the Study 5.5 cell
#: of its budget, but for its name and its interference period, the quarter
#: cell (``-q``) and the half cell (``-h``).
STUDY_5_6_CELLS: Tuple[FerryCell, ...] = (
    FerryCell("jit-n12-120-q", FAMILY_JITTERY, ROLE_DECISION_RICH, 12, 120.0, BUDGET_STAND_IN, 2,
              "jittery", interference_period_s=float(P_C_QUARTER_120_S)),
    FerryCell("jit-n12-120-h", FAMILY_JITTERY, ROLE_DECISION_RICH, 12, 120.0, BUDGET_STAND_IN, 2,
              "jittery", interference_period_s=float(P_C_HALF_120_S)),
    FerryCell("jit-n12-180-q", FAMILY_JITTERY, ROLE_DECISION_RICH, 12, 180.0, BUDGET_STAND_IN, 2,
              "jittery", interference_period_s=float(P_C_QUARTER_180_S)),
    FerryCell("jit-n12-180-h", FAMILY_JITTERY, ROLE_DECISION_RICH, 12, 180.0, BUDGET_STAND_IN, 2,
              "jittery", interference_period_s=float(P_C_HALF_180_S)),
)

#: The scale family's field half-widths (m), N = 6's density in the realism
#: field's 100 m: ``topology_builder.grown_field_radius_m(100.0, N, 6)``,
#: written out so that the registry is a table (a test checks them).
SCALE_FIELD_M: Dict[int, float] = {24: 200.0, 48: 282.8, 96: 400.0}


def _scale_cell(n: int, budget_s: float) -> FerryCell:
    return FerryCell(f"scl-n{n}-{budget_s:g}", FAMILY_SCALE, ROLE_DECISION_RICH, n, budget_s,
                     BUDGET_STAND_IN, 2, "jittery", field_radius_m=SCALE_FIELD_M[n])


#: Study 5.11 (c)'s cells (the Exp 5 addendum; the module docstring): N = 24,
#: 48 and 96 at N = 6's density, each at its binding edge and 1.5 times it,
#: stand-ins until the size's budget pilot; S = 2.
SCALE_CELLS: Tuple[FerryCell, ...] = (
    _scale_cell(24, 350.0), _scale_cell(24, 525.0),
    _scale_cell(48, 680.0), _scale_cell(48, 1020.0),
    _scale_cell(96, 1330.0), _scale_cell(96, 1995.0),
)

#: Every cell by name: decision 3's, Study 5.6's and the scale family's.
CELLS_BY_NAME: Dict[str, FerryCell] = {
    c.name: c for c in CELLS + STUDY_5_6_CELLS + SCALE_CELLS}

#: One score per contact regime, practised over these cells (decision 3): the
#: jittery family, both sizes and both budgets at the default P_c, and the clean
#: control; or, for the jittery regime's score, ``jittery56``, the jittery
#: family's cells and Study 5.6's (resolution R22). ``jittery56`` changes
#: neither ``jittery`` nor ``clean``, and ``scale`` (Study 5.11 (c), flown and
#: not trained on: the learned score practises at N = 6 and 12) none of them.
FAMILIES: Dict[str, Tuple[FerryCell, ...]] = {
    FAMILY_JITTERY: tuple(c for c in CELLS if c.family == FAMILY_JITTERY),
    FAMILY_CLEAN: tuple(c for c in CELLS if c.family == FAMILY_CLEAN),
    FAMILY_JITTERY_56: tuple(c for c in CELLS if c.family == FAMILY_JITTERY) + STUDY_5_6_CELLS,
    FAMILY_SCALE: SCALE_CELLS,
}

#: The cells Study 5.5 is read on (decision 5: N = 12, the jittery regime), by
#: name: Study 5.6's cells are decision-rich N = 12 jittery cells too, and the
#: rule reads these two whichever family trained the score (resolution R22).
STUDY_5_5_CELLS: Tuple[FerryCell, ...] = (CELLS_BY_NAME["jit-n12-120"],
                                         CELLS_BY_NAME["jit-n12-180"])

#: Study 5.6's control: the clean N = 12 cells, at the default P_c (resolution R22).
STUDY_5_6_CONTROL_CELLS: Tuple[FerryCell, ...] = (CELLS_BY_NAME["cln-n12-120"],
                                                 CELLS_BY_NAME["cln-n12-180"])


def cell_named(name: Union[str, FerryCell]) -> FerryCell:
    """The cell called ``name`` (:data:`CELLS`, :data:`STUDY_5_6_CELLS`,
    :data:`SCALE_CELLS`); a :class:`FerryCell` is itself."""
    if isinstance(name, FerryCell):
        return name
    try:
        return CELLS_BY_NAME[name]
    except (KeyError, TypeError):
        raise ValueError(f"no FerrySim cell {name!r}; the cells are {sorted(CELLS_BY_NAME)}") \
            from None


def family_sha256(family: str) -> str:
    """SHA-256 of a family's cells as JSON: the cell family's hash a manifest
    records (the spec, other choices 6)."""
    try:
        cells = FAMILIES[family]
    except KeyError:
        raise ValueError(f"no cell family {family!r}; the families are {sorted(FAMILIES)}") \
            from None
    blob = json.dumps([c.to_json() for c in cells], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def arrival_lags(sorties: Iterable[Any]) -> List[float]:
    """Critic A3's lag, the base of Study 5.6's ratio (resolution R22): the time
    from each Pass-1 arrival to the next arrival in the same sortie, in seconds.

    ``sorties`` are an episode's sortie records (``episode.EpisodeResult``'s
    ``sorties``, each a ``reward.SortieRecord``), whose ``stops`` are the
    Pass-1 stops flown, in order (the mule's ``pass_1_flown``), each with its
    arrival ``t_s``. The lag holds the dwell at the first stop and the leg to
    the next. A sortie's last stop starts no lag: its decision's span runs to
    the end of the upload, not to an arrival.
    """
    return [float(b.t_s) - float(a.t_s) for s in sorties for a, b in zip(s.stops, s.stops[1:])]


# --------------------------------------------------------------------------- #
# Seed streams
# --------------------------------------------------------------------------- #

STREAM_TRAIN = "train"
STREAM_VAL = "val"
STREAM_HELDOUT = "heldout"
STREAM_KINDS: Tuple[str, ...] = (STREAM_TRAIN, STREAM_VAL, STREAM_HELDOUT)

#: The validation and held-out streams' names (the spec, other choices 8).
VAL_STREAM = "ferrysim-val"
HELDOUT_STREAM = "ferrysim-heldout"
_TRAIN_PREFIX = "ferrysim-train-"

#: A trial seed's top two bits: its stream kind. 0 is no FerrySim stream (the
#: runner's and the goldens' seeds, e.g. UG5's 59 and 26).
STREAM_TAGS: Dict[str, int] = {STREAM_TRAIN: 1, STREAM_VAL: 2, STREAM_HELDOUT: 3}
_TAG_SHIFT = 30
_LOW_MASK = (1 << _TAG_SHIFT) - 1


def train_stream(run_seed: int) -> str:
    """Training run ``run_seed``'s stream, ``ferrysim-train-<run_seed>``."""
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed < 0:
        raise ValueError(f"a training run's seed is an int >= 0, got {run_seed!r}")
    return f"{_TRAIN_PREFIX}{run_seed}"


def stream_kind(stream: str) -> str:
    """The kind of a stream name: ``train``, ``val`` or ``heldout``."""
    if stream == VAL_STREAM:
        return STREAM_VAL
    if stream == HELDOUT_STREAM:
        return STREAM_HELDOUT
    if isinstance(stream, str) and stream.startswith(_TRAIN_PREFIX):
        tail = stream[len(_TRAIN_PREFIX):]
        if tail.isdigit() and str(int(tail)) == tail:
            return STREAM_TRAIN
    raise ValueError(
        f"a FerrySim stream is {VAL_STREAM!r}, {HELDOUT_STREAM!r} or "
        f"'{_TRAIN_PREFIX}<seed>', got {stream!r}")


def trial_seed(stream: str, cell: str, index: int) -> int:
    """The trial seed of episode ``index`` of ``cell`` in ``stream``, repeats included.

    ``(tag << 30) | SHA-256(stream|cell|index)`` mod 2**30, with the stream
    kind's tag (:data:`STREAM_TAGS`). :func:`stream_seeds` skips a repeat
    within a stream; this is the raw draw.
    """
    tag = STREAM_TAGS[stream_kind(stream)]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"an episode index is an int >= 0, got {index!r}")
    digest = hashlib.sha256(f"{stream}|{cell}|{index}".encode("utf-8")).digest()
    return (tag << _TAG_SHIFT) | (int.from_bytes(digest[:4], "big") & _LOW_MASK)


def seed_kind(seed: int) -> Optional[str]:
    """The stream kind a trial seed's tag names, or None (no FerrySim stream)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 32):
        raise ValueError(f"a trial seed is a 32-bit int, got {seed!r}")
    tag = seed >> _TAG_SHIFT
    for kind, value in STREAM_TAGS.items():
        if value == tag:
            return kind
    return None


def stream_seeds(stream: str, cell: str, count: int, *, start: int = 0) -> Tuple[int, ...]:
    """Episodes ``start`` .. ``start + count - 1`` of ``cell`` in ``stream``: distinct seeds.

    Episode e is the e-th distinct seed of the raw draws 0, 1, 2, ...
    (:func:`trial_seed`), so a seed repeated within the stream is skipped and
    a stream's episodes never repeat; the result does not depend on ``start``
    and ``count`` beyond the slice they select.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count is an int >= 0, got {count!r}")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError(f"start is an int >= 0, got {start!r}")
    seen = set()
    out: List[int] = []
    raw = 0
    while len(out) < start + count:
        seed = trial_seed(stream, cell, raw)
        raw += 1
        if seed in seen:
            continue
        seen.add(seed)
        out.append(seed)
    return tuple(out[start:])


def check_disjoint(groups: Iterable[Tuple[str, Iterable[int]]]) -> None:
    """Refuse seeds that put two streams' episodes together (critic B14).

    ``groups`` holds (stream name, trial seeds) pairs, one per cell of a
    stream, say. Raises ValueError when a seed's tag is not its stream's kind
    (so it may be another stream's), when streams of different kinds share a
    seed, or when a validation or held-out group repeats one (a cell's
    episodes must be distinct).
    """
    owner: Dict[int, str] = {}
    for stream, seeds in groups:
        kind = stream_kind(stream)
        listed = list(seeds)
        if kind != STREAM_TRAIN and len(set(listed)) != len(listed):
            raise ValueError(f"stream {stream!r} repeats an episode's seed")
        for seed in listed:
            if seed_kind(seed) != kind:
                raise ValueError(
                    f"seed {seed} of stream {stream!r} carries the tag of "
                    f"{seed_kind(seed)!r}, not {kind!r}")
            other = owner.get(seed)
            if other is not None and stream_kind(other) != kind:
                raise ValueError(f"seed {seed} is in both {other!r} and {stream!r}")
            owner.setdefault(seed, stream)


def train_episode(run_seed: int, family: str, index: int) -> Tuple[FerryCell, int]:
    """Episode ``index`` of training run ``run_seed`` over ``family``'s cells.

    The cell is drawn uniformly from the family by a keyed draw of (run,
    index), and the seed from the run's stream for that cell, so a training
    run's episodes are a pure function of its seed and never meet the
    validation or held-out streams. A seed may repeat within a training run
    (about one pair in 2**30 draws), which is harmless to training.
    """
    cells = FAMILIES.get(family)
    if not cells:
        raise ValueError(f"no cell family {family!r}; the families are {sorted(FAMILIES)}")
    stream = train_stream(run_seed)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"an episode index is an int >= 0, got {index!r}")
    digest = hashlib.sha256(f"{stream}|cell|{family}|{index}".encode("utf-8")).digest()
    cell = cells[int.from_bytes(digest[:4], "big") % len(cells)]
    return cell, trial_seed(stream, cell.name, index)


__all__ = [
    "CELLS",
    "CELLS_BY_NAME",
    "FAMILIES",
    "FAMILY_CLEAN",
    "FAMILY_JITTERY",
    "FAMILY_JITTERY_56",
    "FerryCell",
    "HELDOUT_STREAM",
    "N_MISSIONS",
    "PAYLOAD_BYTES",
    "P_C_HALF_120_S",
    "P_C_HALF_180_S",
    "P_C_QUARTER_120_S",
    "P_C_QUARTER_180_S",
    "ROLES",
    "STREAM_HELDOUT",
    "STREAM_KINDS",
    "STREAM_TAGS",
    "STREAM_TRAIN",
    "STREAM_VAL",
    "STUDY_5_5_CELLS",
    "STUDY_5_6_CELLS",
    "STUDY_5_6_CONTROL_CELLS",
    "STUDY_5_6_LAGS_S",
    "VAL_STREAM",
    "arrival_lags",
    "cell_named",
    "check_disjoint",
    "family_sha256",
    "seed_kind",
    "stream_kind",
    "stream_seeds",
    "train_episode",
    "train_stream",
    "trial_seed",
]
