"""Capture of Phase 4's plan arms at 386c275 (FeRRy Phase 5, unit UG5).

Freeze Rule 1 has three legacy faces in Phase 5: the wall clock (the afa9526
goldens beside this module), Phase 3's simulated clock with
``plan_mode=legacy`` (UG4's ``p3_sim.json``, 6e6f92d) and Phase 4's plan arms,
which until now only Phase 4's own property tests covered
(``tests/integration/test_p4_plan_trials.py`` checks what a trial carries, not
what it flies). This module records the third face as oracles, captured on the
untouched tree at 386c275: eight stub trials of the plan arms, each run end to
end through the code a real trial runs, and every later Phase 5 unit must
reproduce them with its switches at their defaults.

**How a trial runs.** Exactly as UG4's (``_build_p3_sim``, imported and not
edited): ``Exp4Driver.run_trial`` unchanged; the real orchestrator writes the
per-role JSON; the real cluster, mule and device services then run in this
process from that JSON, on synchronous in-process links, with a wall clock
only the harness moves. One mule: UG4's ``run_roles`` refuses more (no
simulated-order gate in process), so there is no K = 2 trial here (Phase 5
critic A4); Phase 4's K = 2 loopback in ``tests/integration/test_p4_plan_missions.py``
stays that pin. A trial is a pure function of its cell and settings, with one
exception: ``mission_completed.plan_wall_s`` is the planner's wall time
(``time.perf_counter``, ``fl_scheduler.py`` ``build_ferry_plan``), kept beside
the plan for that reason (Phase 4 critic B12). The fixture keeps the key and
masks its value (:data:`WALL_TOKEN`). Two of the trials (``fx_n12_120s`` and
``f_45s``) were also run once through the real orchestrator at 386c275 (real
subprocesses and TCP): the row, the per-role JSON, ``mule_ready``,
``mission_started``, ``mission_completed`` and every cluster event agreed,
bar what the wall clock and the OS decide there (as for UG4's trials: the
envelope ``ts`` and ``duration_s``, the ports, the planner's wall time, the
row columns built from the devices' serves, and the ``metrics_snapshot``
timer of mission wall time).

**What is compared**, per trial and part: UG4's parts (the row, the per-role
JSON, every mule event in order, ``mule_ready``, ``mission_started``,
``mission_completed``, every cluster and device event), each on the key sets
it had at 386c275 (critic A3's rule, ``_canon.diff``: a field added with a
default passes, a removed or renamed one fails; values, lists and event
sequences match exactly; a row's JSON cells such as ``ferry_params`` are
strings and match exactly), and one part of UG5's own, ``flight_slot``: every
call the mule made to its flight slot, recorded by :func:`flight_slot_spy`,
which wraps the two fillings' methods and changes nothing. A ``next_stop`` call
records the slot, the pass, ``after_stop``, the remainder (each
``ContactWaypoint``), the departure's ``FlightState``, every order the slot
tried with its ``fits`` verdict (the departure check's own fold) and the index
it picked; a ``band_at_arrival`` call records the ``ArrivalView`` it was given
(None when the slot reads none) and the class it named. Phase 5's pair slot is
a third filling; the committed and FX slots must keep their calls and their
arguments (the Phase 5 spec, other choices 1), and this part is what pins them,
under a rule stricter than critic A3's (:func:`compare_part`): an argument a
386c275 call did not pass fails, even one with a default that leaves the
flight unchanged, and so does one it passed that the call leaves out.

**What the trials cannot reach.** The beacon hook has no source in a trial
run through the driver (at 386c275 nothing in ``hermes/`` or ``experiments/``
calls ``MuleSupervisor.offer_contact``; only tests do): it runs at every
Pass-1 departure with no offer queued, so no trial shows its place between
the departure check and the slot's pick. And the plan's exempt stops
(``FLScheduler.plan_protected``) are protected in seven of the eight trials but
never decisively: with ``plan_protected`` returning nothing, every trial is
the same, part for part. Phase 4's own tests pin both
(``tests/integration/test_p4_plan_missions.py``,
``tests/unit/test_p4_fl_scheduler_plan.py``), gated by the 386c275 baseline.

**The trials** (:data:`TRIALS`; the Phase 5 spec's units table, row UG5): F,
FX, FB+medium, F-cov, F-cap and F-prio on one layout at N = 6, 1 MB, the
jittery contact channel, a 45 s budget, S = 2 and 6 missions; FX on the same
layout at 60 s; and FX at N = 12, 120 s, S = 3 and 4 missions, where a sortie
has several stops, so that FX's next-stop rule and the departure check after a
stop are pinned. The seeds were chosen by a probe over seeds 1-60 for N = 6 and
1-40 for N = 12 (see :data:`SEED` and :data:`N12_SEED`).

    py -3.11 tests/golden/_build_p4_plan.py            # compare, write nothing
    py -3.11 tests/golden/_build_p4_plan.py --write    # only at 386c275, clean tree
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.exp4.driver import Exp4Driver  # noqa: E402
from experiments.runner import Cell  # noqa: E402

from tests.golden import _build_p3_sim as UG4  # noqa: E402
from tests.golden import _canon  # noqa: E402

#: The commit whose plan-arm behaviour the fixture pins (main: Phase 4's code,
#: 69b551f, and its docs), captured before any Phase 5 change.
BASE_COMMIT = "386c27552e249da07550bc9042b4a907c7e8e684"
FIXTURE = "p4_plan"

TYPE_KEY = _canon.TYPE_KEY

#: What the fixture stores for ``mission_completed.plan_wall_s``, the planner's
#: wall seconds (``build_ferry_plan`` measures it with ``time.perf_counter``):
#: the only wall time in a plan-mode trace, which Phase 4 keeps beside ``plan``
#: so that determinism comparisons can drop it (critic B12;
#: Experiment_4_Run_Guide.md section 2.7, "Reading a plan-mode trace"). The key
#: stays in the fixture, so removing or renaming it fails; any finite
#: non-negative number of seconds is stored as this token, anything else as
#: itself.
WALL_TOKEN = "wall-seconds"


# --------------------------------------------------------------------------- #
# The trials
# --------------------------------------------------------------------------- #

def _cell(arm: str, seed: int, **params: Any) -> Cell:
    p = {"N": 6, "rrf": 60.0, "n_missions": 6, "regime": "jittery"}
    p.update(params)
    return Cell(cell_id="|".join(f"{k}={v}" for k, v in sorted(p.items())), arm=arm,
                trial_index=0, seed=seed, params=p)


#: The Phase 4 pilots' common driver settings (Experiment_4_Run_Guide.md
#: section 2.7, "The pilot plan": the simulated clock, wide as the reference
#: class, the T_nom deadline unit, the re-plan with the trim fallback,
#: agg:cutoff, the channel reliability source, realism), with the Phase 5
#: cells' declared 1 MB and jittery contact channel (the user's decision 3 of
#: 2026-10-01, ``contact_regime="jittery"``: 5 dB of interference on a 60 s
#: period, ``CONTACT_REGIMES`` in hermes/l1/channel_model.py, the regime in
#: which the fastest class at a stop changes over a sortie), the
#: stress budget of 45 s and the cap S = 2 (the S* tool's S for F at 1 MB at
#: the prior budgets of 90 and 45 s, Experiment_4_Run_Guide.md section 2.7).
#: T_nom is computed per cell over the driver's default 20 reference layouts,
#: as a campaign computes it.
PLAN_PILOT: Dict[str, Any] = dict(
    mission_clock="sim", realism=True, contact_band="wide", deadline_time_scale="t_nom",
    in_flight_response="replan", replan_fallback="trim", aggregation="agg:cutoff",
    contact_reliability_source="channel", payload_bytes=1_000_000, mission_budget_s=45.0,
    age_cap_missions=2, ferry_physics={"contact_regime": "jittery"},
)

#: The N = 6 layout of the six 45 s trials and FX at 60 s, as in one CSV. Seed
#: 59 was chosen by a probe over seeds 1-60 of all seven trials because it
#: exercises the most of what Phase 4 added, and every one of these: the cap
#: binds in five of six missions of every capped arm; violations of three
#: causes (``crowded``, ``dropped_in_flight``, ``not_merged``); member subsets
#: reduce an S3a stop in every arm (the complement dropped where the stop
#: flies); hover stops of capped devices in F, FX, FB+medium, F-cov and
#: F-prio; an in-flight member trim after a stop in F, FX and F-prio; FX
#: switching class at arrival at both budgets; and an empty F-cov mission
#: (cap-only service). F-prio flies exactly as F does on this layout (and its
#: probe counts equal F's on every seed probed); it differs in its plans'
#: weights and its row.
SEED = 59

#: FX's budget at the Phase 4 exit gate's stub smoke (30 s and 60 s,
#: Experiment_4_Run_Guide.md section 2.7) and the plan's 60 s knee (build plan
#: L1009), on the 45 s trials' layout.
FX_GATE_BUDGET_S = 60.0

#: N = 12, Study 5.5's decision-rich cell (the user's decision 3): 2-5 Pass-1
#: stops per sortie at the stand-in budget of 120 s with S = 3 (the Phase 5
#: spec's units table, row UG5). At N = 6 a sortie mostly has one stop (66 to
#: 72 of 72 in the Phase 5 design's probes), so FX's next-stop rule rarely has
#: a choice there, and on seed 59 it has none. Seed 26 was chosen by a probe
#: over seeds 1-40: FX re-orders after a stop in two missions, and in the
#: first of them the departure checks after that re-order fail and trim a
#: stop's members twice (``arm_trimmed``); its ``fits`` refuses the nearest
#: candidate order at six departures; the last stop of mission 1 is dropped at
#: the departure check; and FX switches class at arrival.
N12_SEED = 26

#: name -> (driver settings, cell).
TRIALS: Dict[str, Tuple[Dict[str, Any], Cell]] = {
    "f_45s": (dict(PLAN_PILOT), _cell("F", SEED)),
    "fx_45s": (dict(PLAN_PILOT), _cell("FX", SEED)),
    "fb_medium_45s": (dict(PLAN_PILOT), _cell("FB+medium", SEED)),
    "f_cov_45s": (dict(PLAN_PILOT), _cell("F-cov", SEED)),
    "f_cap_45s": (dict(PLAN_PILOT), _cell("F-cap", SEED)),
    "f_prio_45s": (dict(PLAN_PILOT), _cell("F-prio", SEED)),
    "fx_60s": (dict(PLAN_PILOT, mission_budget_s=FX_GATE_BUDGET_S), _cell("FX", SEED)),
    "fx_n12_120s": (dict(PLAN_PILOT, mission_budget_s=120.0, age_cap_missions=3),
                    _cell("FX", N12_SEED, N=12, n_missions=4)),
}
TRIAL_NAMES = tuple(TRIALS)

#: The parts of a case, each compared on its own (one test each): UG4's, then
#: the flight slot's calls.
PARTS = UG4.PARTS + ("flight_slot",)


# --------------------------------------------------------------------------- #
# The flight slot's calls
# --------------------------------------------------------------------------- #

#: The arguments the mule passes each method of the flight slot at 386c275
#: (``mule_main.py``: ``_ferry_next_stop`` and ``_ferry_band_at_arrival``). A
#: call record keeps each one it was passed under its own key (``pass_kind``
#: as ``pass``, ``fits`` as the orders it was asked about); any other argument
#: goes under :data:`ADDED_ARGUMENTS`.
SLOT_ARGUMENTS: Dict[str, Tuple[str, ...]] = {
    "next_stop": ("remainder", "state", "fits", "pass_kind", "after_stop"),
    "band_at_arrival": ("view", "pass_kind"),
}

#: The key under which a call record keeps, by name, the arguments its 386c275
#: call did not pass. The fixture has none. Critic A3's rule alone would let
#: such a key pass as an added field; :func:`compare_part` fails it, because
#: the committed and FX slots must keep their arguments (the Phase 5 spec,
#: other choices 1).
ADDED_ARGUMENTS = "added_arguments"

_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def _pass_name(pass_kind: Any) -> Any:
    return getattr(pass_kind, "value", pass_kind)


def _passed(sig: inspect.Signature, bound: inspect.BoundArguments) -> Dict[str, Any]:
    """What a bound call passed, by argument name, the slot itself left out.

    A keyword the method takes through ``**kwargs`` is listed under its own
    name, and extra positional values through ``*args`` under ``*`` and that
    parameter's name, so that every argument has a name to be recorded by.
    """
    out: Dict[str, Any] = {}
    for i, (name, value) in enumerate(bound.arguments.items()):
        kind = sig.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            if value:
                out["*" + name] = value
        elif i > 0:
            out[name] = value
    return out


def _replace(sig: inspect.Signature, bound: inspect.BoundArguments, name: str, value: Any) -> None:
    """Pass ``value`` as argument ``name``, wherever the call bound it."""
    if name in bound.arguments and sig.parameters[name].kind not in _VAR_KINDS:
        bound.arguments[name] = value
        return
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD and name in bound.arguments.get(p.name, {}):
            bound.arguments[p.name] = dict(bound.arguments[p.name], **{name: value})


def _recordable(value: Any) -> Any:
    """``value`` in canonical form, or its type's name when it has none."""
    try:
        return _canon.canon(value)
    except TypeError:
        return f"<{type(value).__name__}>"


@contextlib.contextmanager
def flight_slot_spy() -> Iterator[List[Dict[str, Any]]]:
    """Record every call to the flight slot's two fillings, changing none.

    ``CommittedSlot`` and ``CrossHeuristic`` (``hermes.scheduler.policies.
    cross_heuristic``) get their ``next_stop`` and ``band_at_arrival`` wrapped
    for the duration. A wrapper takes whatever the method takes: it binds the
    call to the method's own signature and passes it on unchanged, and returns
    what the method returned, so the trial flies exactly as it would; a call
    the method refuses goes to the method as it came, which refuses it with
    its own error. The ``fits`` a ``next_stop`` call is given is wrapped too,
    to record each order tried and its verdict in the order the slot tried
    them. A record keeps the arguments of :data:`SLOT_ARGUMENTS` the call
    passed, each under its own key, and every other argument under
    :data:`ADDED_ARGUMENTS`. So a 386c275 argument the mule stops passing, or
    a new one it starts passing (with a default in the slot, say, so that the
    flight is unchanged), is a readable mismatch (:func:`compare_part`), not a
    crash inside the mule. The yielded list fills as the trial runs, one plain
    mapping per call; the originals are restored on exit, an error included.
    """
    from hermes.scheduler.policies import cross_heuristic as ch

    calls: List[Dict[str, Any]] = []
    saved: List[Tuple[type, str, Any]] = []

    def spy(orig, method: str):
        sig = inspect.signature(orig)

        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):
            try:
                bound = sig.bind(self, *args, **kwargs)
            except TypeError:
                return orig(self, *args, **kwargs)
            passed = _passed(sig, bound)
            tried: List[Dict[str, Any]] = []
            fits = passed.get("fits")
            if method == "next_stop" and callable(fits):
                def recorded(order):
                    ok = fits(order)
                    tried.append({"order": [[str(d) for d in wp.devices] for wp in order],
                                  "ok": ok})
                    return ok

                _replace(sig, bound, "fits", recorded)
            result = orig(*bound.args, **bound.kwargs)
            rec: Dict[str, Any] = {"call": method, "slot": self.name}
            if "pass_kind" in passed:
                rec["pass"] = _pass_name(passed["pass_kind"])
            if method == "next_stop":
                if "after_stop" in passed:
                    rec["after_stop"] = passed["after_stop"]
                if "remainder" in passed:
                    rec["remainder"] = list(passed["remainder"])
                if "state" in passed:
                    rec["state"] = passed["state"]
                if "fits" in passed:
                    rec["fits"] = tried
                rec["index"] = result
            else:
                if "view" in passed:
                    rec["view"] = passed["view"]
                rec["band"] = result
            added = {k: _recordable(v) for k, v in passed.items()
                     if k not in SLOT_ARGUMENTS[method]}
            if added:
                rec[ADDED_ARGUMENTS] = added
            calls.append(rec)
            return result
        return wrapper

    try:
        for cls in (ch.CommittedSlot, ch.CrossHeuristic):
            for method in SLOT_ARGUMENTS:
                orig = cls.__dict__[method]
                saved.append((cls, method, orig))
                setattr(cls, method, spy(orig, method))
        yield calls
    finally:
        for cls, method, orig in reversed(saved):
            setattr(cls, method, orig)


def slot_record(call: Mapping[str, Any]) -> Dict[str, Any]:
    """One flight-slot call in canonical form: a record at every level.

    The waypoints, the flight state and the arrival view are dataclasses, so
    ``canon`` stores them as records already; each tried order is made a record
    here, so that a field added anywhere passes and a removed one fails.
    """
    out = dict(call)
    if "fits" in out:
        out["fits"] = [_canon.record("slot.fits", t) for t in out["fits"]]
    return _canon.record(f"slot.{call['call']}", out)


def added_slot_arguments(current: Any, limit: int = 25) -> List[str]:
    """The calls in a ``flight_slot`` part that were passed an argument their
    386c275 call does not pass, one line each (at most ``limit``)."""
    out: List[str] = []
    if not isinstance(current, list):
        return out
    for i, c in enumerate(current):
        if len(out) >= limit:
            break
        if isinstance(c, dict) and c.get(ADDED_ARGUMENTS):
            out.append(f"$[{i}].{ADDED_ARGUMENTS}: {c.get('call')} of the {c.get('slot')} slot "
                       f"({c.get('pass')}) was passed {sorted(c[ADDED_ARGUMENTS])}, which its "
                       f"386c275 call does not pass")
    return out


def compare_part(part: str, golden: Any, current: Any) -> List[str]:
    """The mismatches of one part of a trial with its golden.

    Every part: critic A3's rule (``_canon.diff``), so a key added with a
    default passes. ``flight_slot`` also: the Phase 5 spec's rule that the
    committed and FX slots keep their calls and their arguments (other choices
    1), so an argument the 386c275 call did not pass fails, with a default or
    without (:func:`added_slot_arguments`), and so does one it passed and the
    call no longer does (``_canon.diff``: its key is missing). A field added
    with a default inside an argument (a ``FlightState``, ``ContactWaypoint``
    or ``ArrivalView`` field) passes, as in every other part.
    """
    problems = _canon.diff(golden, current)
    if part == "flight_slot":
        problems += added_slot_arguments(current)
    return problems


def masked_wall(value: Any) -> Any:
    """``plan_wall_s`` as the fixture stores it (:data:`WALL_TOKEN`)."""
    if isinstance(value, str) and value.startswith("f:"):
        try:
            x = float(value[2:])
        except ValueError:
            return value
        if x == x and x != float("inf") and x >= 0.0:
            return WALL_TOKEN
    return value


def mask_wall_times(case: Dict[str, Any]) -> Dict[str, Any]:
    """``case`` with every ``mission_completed.plan_wall_s`` masked, in place."""
    for events in case["mission_completed"].values():
        for e in events:
            if "plan_wall_s" in e:
                e["plan_wall_s"] = masked_wall(e["plan_wall_s"])
    return case


# --------------------------------------------------------------------------- #
# A trial
# --------------------------------------------------------------------------- #

def case_of(settings: Mapping[str, Any], cell: Cell, row: Mapping[str, Any],
            orch: "UG4.InProcessOrchestrator", calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One trial's canonical record: UG4's parts, the slot's ``calls``
    (:func:`flight_slot_spy`'s list) and the planner's wall time masked."""
    case = UG4.case_of(settings, cell, row, orch)
    case["flight_slot"] = [slot_record(c) for c in calls]
    return mask_wall_times(case)


@functools.lru_cache(maxsize=None)
def capture(name: str) -> Dict[str, Any]:
    """Run trial ``name`` in this process; its canonical record (cached per process)."""
    settings, cell = TRIALS[name]
    driver = Exp4Driver(**settings)
    with UG4.in_process_orchestrator(), flight_slot_spy() as calls:
        UG4.InProcessOrchestrator.last = None
        row = dict(driver.run_trial(cell))
        orch = UG4.InProcessOrchestrator.last
    if orch is None:
        raise AssertionError(f"trial {name}: the driver never built the orchestrator")
    return case_of(settings, cell, row, orch, calls)


def build_cases() -> Dict[str, Any]:
    return {name: capture(name) for name in TRIAL_NAMES}


def load_golden() -> Dict[str, Any]:
    return _canon.load(FIXTURE)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"rewrite data/{FIXTURE}.json")
    ap.add_argument("--force", action="store_true",
                    help="allow --write away from a clean 386c275 tree (a golden that was wrong)")
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    logging.basicConfig(level=logging.ERROR, format="%(name)s: %(message)s")
    if args.write:
        head = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--", "hermes", "experiments")
        if (head != BASE_COMMIT or dirty) and not args.force:
            print(f"refusing to write: the fixture pins {BASE_COMMIT[:7]}, but HEAD is "
                  f"{head[:7]}" + (f" with changes in hermes/ or experiments/:\n{dirty}"
                                   if dirty else "")
                  + "\n(--force overrides; only for a golden that was itself wrong)",
                  file=sys.stderr)
            return 2
    t0 = time.time()
    cases = build_cases()
    took = time.time() - t0
    if args.write:
        meta = {"base_commit": BASE_COMMIT, "unit": "UG5",
                "what": "Phase 4 plan-arm stub trials run in process: rows, per-role JSON, "
                        "every role's events and the flight slot's calls"}
        path = _canon.dump(FIXTURE, meta, cases)
        print(f"wrote {path.relative_to(REPO)}: {len(cases)} trials, "
              f"{path.stat().st_size / 1024:.0f} KiB ({took:.1f} s)")
        return 0
    golden = load_golden()["cases"]
    status = 0
    for name in TRIAL_NAMES:
        for part in ("inputs",) + PARTS:
            g = golden.get(name, {}).get(part, "<missing>")
            c = cases[name][part]
            problems = compare_part(part, g, c)
            added = UG4.added_keys(g, c)
            if problems:
                status = 1
            if problems or added:
                print(f"{name}.{part}: {len(problems)} mismatch(es), {len(added)} added key(s)")
                for p in problems[:8]:
                    print(f"    {p}")
                for a in added[:8]:
                    print(f"    added: {a}")
    print(f"{len(TRIAL_NAMES)} trials x {len(PARTS)} parts compared "
          f"({'differ' if status else 'same'}; {took:.1f} s)")
    return status


if __name__ == "__main__":
    sys.exit(main())
