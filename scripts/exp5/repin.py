"""Re-pin FerrySim's cells to the stack's measured budgets (Run Guide 2.8, "The re-pin").

FerrySim's N = 6 and N = 12 cells fly placeholder budgets (Phase 4's priors at
N = 6, the stand-ins at N = 12) until the stack's knee and S* pilots measure
them; the learned score must train on the budgets the stack flies. This moves
them, and everything that follows from them, together (R29):

1. The budgets: params.toml [pilot_outputs] stress_s and knee_s at N = 6 and 12
   become cells.BUDGETS_S, each size's stress budget first, their source
   BUDGET_STRESS and BUDGET_KNEE.
2. The names, which carry the budget (jit-n12-120 -> jit-n12-<stress>, and
   Study 5.6's -q and -h cells with them): renamed in one pass over the code,
   the tests, the launcher and params.toml. DeveloperDocs is listed, not
   edited: its records are history.
3. The caps: the S* tool's S for F at each size's two budgets, per contact
   regime, never below 2 (cells' module docstring), as the cap test computes it.
4. Study 5.6's lags: FX's arrival-to-arrival lags over the first 200 validation
   episodes of each new N = 12 jittery cell at the default P_c, pooled; the
   median rounded to the second is the lag, the quarter and half cells' P_c 4 x
   and 2 x it. The ratio check's bounds come from the same sample: the 0.5 and
   99.5 % points of a 24-episode pooled median (episodes resampled whole) over
   the full median, widened by 5 % and rounded outward; refused unless the
   quarter and half cells' ranges stay apart.
5. The pins: cells.py's re-pin block and test_p5_ferrysim_runner.py's re-pin
   pins (the families' hashes, recomputed in a fresh interpreter; the lags,
   periods, lag sample and ratio bounds), and a check that each Study 5.6 cell's
   S* at its own period is its base cell's cap.

    python scripts/exp5/repin.py --dry-run         # the plan, nothing written
    python scripts/exp5/repin.py --workers 8       # measure and write

Then: run the FerrySim and Exp 5 tests (the script prints the commands), review
the docs it lists, commit, and set [rl] repinned = true. Run it from a clean
tree on the commit that holds the pilots' outputs; `git diff` shows every edit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.version_info < (3, 11):
    sys.exit("repin.py needs Python 3.11+ (tomllib)")
import tomllib  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CELLS_PY = REPO / "experiments" / "ferrysim" / "cells.py"
PINS_PY = REPO / "tests" / "unit" / "test_p5_ferrysim_runner.py"
#: Where the cells' names are renamed (DeveloperDocs is only listed).
RENAME_GLOBS = ("experiments/**/*.py", "scripts/exp5/*.py", "scripts/exp5/params.toml",
                "tests/**/*.py")
DOC_GLOBS = ("DeveloperDocs/**/*.md", "DeveloperDocs/**/*.html", "README.md")
SIZES = {6: ("jit",), 12: ("jit", "cln")}
REGIMES = {(6, "jittery"), (12, "jittery"), (12, "clean")}
LAG_EPISODES = 200
RATIO_EPISODES = 24
BLOCK_START, BLOCK_END = "# >>> re-pin block", "# <<< re-pin block"
PINS_START, PINS_END = "# >>> re-pin pins", "# <<< re-pin pins"


def _setup_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def label(prefix: str, n: int, budget: float) -> str:
    return f"{prefix}-n{n}-{float(budget):g}"


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #

def pilot_budgets(params: Path) -> Dict[int, Tuple[float, float]]:
    data = tomllib.loads(params.read_text(encoding="utf-8-sig"))
    po = data.get("pilot_outputs", {})
    out, missing = {}, []
    for n in (6, 12):
        try:
            stress, knee = float(po["stress_s"][str(n)]), float(po["knee_s"][str(n)])
        except (KeyError, TypeError, ValueError):
            missing.append(f"pilot_outputs.knee_s/stress_s at N = {n}")
            continue
        if not 0 < stress < knee:
            sys.exit(f"N = {n}: stress {stress} s must be above 0 and below the knee {knee} s")
        out[n] = (stress, knee)
    if missing:
        sys.exit("params.toml lacks " + "; ".join(missing) + " (run the knee pilot and "
                 "`exp5 report knee --apply`)")
    return out


def name_map(old: Dict[int, Tuple[float, float]], new: Dict[int, Tuple[float, float]]
             ) -> Dict[str, str]:
    """Every cell name a budget move changes, Study 5.6's included."""
    mapping: Dict[str, str] = {}
    for n, prefixes in SIZES.items():
        for prefix in prefixes:
            for ob, nb in zip(old[n], new[n]):
                o, w = label(prefix, n, ob), label(prefix, n, nb)
                mapping[o] = w
                if (prefix, n) == ("jit", 12):
                    for suffix in ("q", "h"):
                        mapping[f"{o}-{suffix}"] = f"{w}-{suffix}"
    return {k: v for k, v in mapping.items() if k != v}


def rename_pattern(mapping: Dict[str, str]) -> "re.Pattern[str]":
    keys = sorted(mapping, key=len, reverse=True)       # jit-n12-120-q before jit-n12-120
    return re.compile(r"(?<![\w-])(" + "|".join(map(re.escape, keys)) + r")(?!\w)")


def files(globs: Tuple[str, ...]) -> List[Path]:
    seen = []
    me = Path(__file__).resolve()
    for g in globs:
        for p in sorted(REPO.glob(g)):
            if (p.is_file() and "results" not in p.parts and p not in seen
                    and p.resolve() != me):
                seen.append(p)
    return seen


def rename_everywhere(mapping: Dict[str, str], write: bool) -> Dict[Path, int]:
    """One pass, every name at once, so a swap (120 -> 180, 180 -> 360) is safe."""
    if not mapping:
        return {}
    pattern = rename_pattern(mapping)
    counts = {}
    for p in files(RENAME_GLOBS):
        raw = p.read_bytes().decode("utf-8")
        new, n = pattern.subn(lambda m: mapping[m.group(1)], raw)
        if n:
            counts[p] = n
            if write:
                p.write_bytes(new.encode("utf-8"))
    return counts


def docs_to_review(mapping: Dict[str, str]) -> Dict[Path, int]:
    if not mapping:
        return {}
    pattern = rename_pattern(mapping)
    out = {}
    for p in files(DOC_GLOBS):
        n = len(pattern.findall(p.read_text(encoding="utf-8", errors="replace")))
        if n:
            out[p] = n
    return out


# --------------------------------------------------------------------------- #
# Measurements (each in a fresh interpreter, on the cells as written)
# --------------------------------------------------------------------------- #

def _child(*args: str) -> Any:
    proc = subprocess.run([sys.executable, str(Path(__file__)), *args], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8")
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if proc.returncode != 0 or not lines:
        sys.exit(f"repin {args[0]} failed:\n{proc.stderr[-3000:]}")
    return json.loads(lines[-1])


def _cap(n: int, budgets: Tuple[float, float], contact: str, period: Any = None) -> int:
    """S* for F at ``budgets``, never below 2: the cap test's own computation."""
    _setup_path()
    from experiments.analysis.age_cap_s_star import s_star_report
    from experiments.exp4.driver import Exp4Driver
    physics: Dict[str, Any] = {"contact_regime": contact}
    if period is not None:
        physics["interference_period_s"] = float(period)
    driver = Exp4Driver(mission_clock="sim", realism=True, contact_band="wide",
                        payload_bytes=1_000_000, ferry_physics=physics)
    report = s_star_report(driver, n_devices=n, budgets=budgets, regime="jittery",
                           families=("F",))
    return max(2, int(report.s("F")))


def child_caps(budgets_json: str) -> Dict[str, int]:
    budgets = {int(k): tuple(v) for k, v in json.loads(budgets_json).items()}
    return {f"{n}|{c}": _cap(n, budgets[n], c) for n, c in sorted(REGIMES)}


def child_lags(names: List[str], episodes: int, workers: int, boot: int) -> Dict[str, Any]:
    """FX's lags per episode on each cell's validation stream, the pooled median,
    and the ratio check's bounds from the same sample."""
    _setup_path()
    import numpy as np
    from experiments.ferrysim import cells as C
    from experiments.ferrysim import evaluate as EV
    from experiments.ferrysim.episode import Policy
    out = {}
    for name in names:
        cell = C.cell_named(name)
        fx = Policy.of_arm(EV.ARM_FX)
        tasks = [EV.EpisodeTask(cell=cell, stream=C.VAL_STREAM, index=e, seed=seed, policy=fx)
                 for e, seed in enumerate(C.stream_seeds(C.VAL_STREAM, cell.name, episodes))]
        per_episode = [list(l) for l in EV.parallel_map(EV.lags_task, tasks, workers=workers)]
        pooled = [x for l in per_episode for x in l]
        if len(pooled) < 2:
            sys.exit(f"FX flew fewer than 2 lags on {name}: Study 5.6's lag is undefined there")
        median = statistics.median(pooled)
        q1, _, q3 = statistics.quantiles(pooled, n=4)
        rng = np.random.default_rng(0)
        ratios = []
        for _ in range(boot):
            pick = rng.integers(0, len(per_episode), size=RATIO_EPISODES)
            sample = [x for i in pick for x in per_episode[i]]
            if sample:
                ratios.append(statistics.median(sample) / median)
        lo, hi = (float(np.quantile(ratios, q)) for q in (0.005, 0.995))
        out[name] = {"lags": len(pooled), "median": median, "q1": q1, "q3": q3,
                     "ratio_99": [lo, hi],
                     "bounds": [math.floor(lo * 0.95 * 100) / 100,
                                math.ceil(hi * 1.05 * 100) / 100]}
    return out


def child_check() -> Dict[str, Any]:
    """The families' hashes, and each Study 5.6 cell's S* at its own period."""
    _setup_path()
    from experiments.ferrysim import cells as C
    caps = {c.name: _cap(12, C.BUDGETS_S[12], "jittery", c.interference_period_s)
            for c in C.STUDY_5_6_CELLS}
    return {"sha": {f: C.family_sha256(f) for f in ("jittery", "clean", "jittery56", "scale")},
            "caps_5_6": caps, "cap_12": C.CAP_S[(12, "jittery")],
            "p_c": [C.P_C_QUARTER_STRESS_S, C.P_C_HALF_STRESS_S, C.P_C_QUARTER_KNEE_S,
                    C.P_C_HALF_KNEE_S]}


# --------------------------------------------------------------------------- #
# Writing the blocks
# --------------------------------------------------------------------------- #

def _replace_block(path: Path, start: str, end: str, body: str) -> None:
    raw = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    i, j = raw.find(start), raw.find(end)
    if i < 0 or j < i:
        sys.exit(f"{path} has no {start!r} ... {end!r} block")
    j += len(end)
    raw = raw[:i] + body.replace("\n", nl) + raw[j:]
    path.write_bytes(raw.encode("utf-8"))


def cells_block(budgets: Dict[int, Tuple[float, float]], caps: Dict[Tuple[int, str], int],
                lags: Dict[str, int], measured: Dict[str, Any], today: str) -> str:
    (s6, k6), (s12, k12) = budgets[6], budgets[12]
    note = "; ".join(f"{name}: {m['median']:.2f} s over {m['lags']:,} lags (IQR "
                     f"{m['q1']:.1f} to {m['q3']:.1f} s)" for name, m in measured.items())
    lag_items = ", ".join(f'"{k}": {v}' for k, v in lags.items())
    return f'''{BLOCK_START}
#: The (stress, knee) budgets (s) of each size's cells, and where each comes from
#: (BUDGET_*): the stack's knee and stress pilots (params.toml [pilot_outputs]),
#: re-pinned {today} by scripts/exp5/repin.py.
BUDGETS_S: Dict[int, Tuple[float, float]] = {{6: ({s6!r}, {k6!r}), 12: ({s12!r}, {k12!r})}}
BUDGET_ROLES: Dict[int, Tuple[str, str]] = {{
    6: (BUDGET_STRESS, BUDGET_KNEE), 12: (BUDGET_STRESS, BUDGET_KNEE)}}
#: The cap S per (size, contact regime): the S* tool's S at the size's two
#: budgets, never below 2 (module docstring).
CAP_S: Dict[Tuple[int, str], int] = {{(6, "jittery"): {caps[(6, "jittery")]}, (12, "jittery"): {caps[(12, "jittery")]}, (12, "clean"): {caps[(12, "clean")]}}}
#: Study 5.6's lags (s), A3's from one Pass-1 arrival to the next in the same
#: sortie (:func:`arrival_lags`), by the Study 5.5 cell they were measured on
#: (decision 6 (a); critic A3's rule; resolution R22): FX's median over the
#: first 200 episodes of that cell's validation stream (``ferrysim-val``) at the
#: default P_c of 60 s, pooled over every lag of the sample, rounded to the
#: nearest second; ``experiments.ferrysim.evaluate.fx_lag_median(cell)``
#: measures it, and a slow test measures it again. The integers are a
#: pre-registration convention fixed by this sample and statistic, not a
#: measurement to the second. Measured {today}: {note}.
STUDY_5_6_LAGS_S: Dict[str, int] = {{{lag_items}}}
{BLOCK_END}'''


def pins_block(sha: Dict[str, str], lags: Dict[str, int], p_c: List[int],
               measured: Dict[str, Any]) -> str:
    sample = ", ".join(f'"{k}": ({m["lags"]}, {round(m["median"], 3)!r})'
                       for k, m in measured.items())
    bounds = ", ".join(f"({m['bounds'][0]!r}, {m['bounds'][1]!r})" for m in measured.values())
    raw = "; ".join(f"{m['ratio_99'][0]:.2f} to {m['ratio_99'][1]:.2f} x at {k}"
                    for k, m in measured.items())
    lag_items = ", ".join(f'"{k}": {v}' for k, v in lags.items())
    return f'''{PINS_START}: scripts/exp5/repin.py rewrites these with the cells' re-pin
# block (experiments/ferrysim/cells.py), from what it measures; they pin it.
#: The families' hashes (a manifest's ``cell_family_sha256``): ``jittery``,
#: ``clean`` and ``jittery56``.
JITTERY_SHA256 = "{sha["jittery"]}"
CLEAN_SHA256 = "{sha["clean"]}"
JITTERY56_SHA256 = "{sha["jittery56"]}"
#: Study 5.6's lags (s) by Study 5.5 cell, and their periods (s): the quarter and
#: half at the stress budget, then at the knee.
LAGS_S = {{{lag_items}}}
P_C_S = ({p_c[0]}, {p_c[1]}, {p_c[2]}, {p_c[3]})
#: The lags' sample: (lags, pooled median to 3 places) per Study 5.5 cell.
LAG_SAMPLE = {{{sample}}}
#: The ratio check's bounds by budget (stress, knee): the 0.5 and 99.5 % points
#: of a RATIO_EPISODES-episode median in the lag's 200-episode measurement
#: (episodes resampled whole, since one layout's lags move together), as a
#: multiple of the measured median, widened by 5 % for the lag's rounding to the
#: second and its move with P_c, and rounded outward. Before widening: {raw}.
RATIO_BOUNDS_BY_BUDGET = ({bounds})
{PINS_END}'''


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: List[str]) -> int:
    if argv[:1] == ["_caps"]:
        print(json.dumps(child_caps(argv[1])))
        return 0
    if argv[:1] == ["_lags"]:
        names, episodes, workers, boot = json.loads(argv[1]), int(argv[2]), int(argv[3]), int(argv[4])
        print(json.dumps(child_lags(names, episodes, workers, boot)))
        return 0
    if argv[:1] == ["_check"]:
        print(json.dumps(child_check()))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--params", type=Path, default=HERE / "params.toml")
    ap.add_argument("--dry-run", action="store_true", help="Show the plan; write nothing.")
    ap.add_argument("--workers", type=int, default=1, help="FX's lag episodes in parallel.")
    ap.add_argument("--episodes", type=int, default=LAG_EPISODES,
                    help="Validation episodes per lag (the pinned rule: 200).")
    ap.add_argument("--boot", type=int, default=20000, help="Bootstrap resamples.")
    a = ap.parse_args(argv)
    _setup_path()
    from experiments.ferrysim import cells as C

    old = {n: tuple(C.BUDGETS_S[n]) for n in (6, 12)}
    new = pilot_budgets(a.params)
    mapping = name_map(old, new)
    print("budgets (stress, knee) s: " + "; ".join(
        f"N = {n}: {old[n]} -> {new[n]}" for n in (6, 12)))
    print("names: " + (", ".join(f"{k} -> {v}" for k, v in mapping.items()) or "unchanged"))
    others = set(C.CELLS_BY_NAME) - set(mapping)
    clash = [v for v in mapping.values() if v in others]
    if clash:
        sys.exit(f"new names {clash} would collide with existing cells")
    renames = rename_everywhere(mapping, write=False)
    for p, n in renames.items():
        print(f"  rename {n:3d} in {p.relative_to(REPO)}")
    caps = {tuple(k.split("|")): v for k, v in _child(
        "_caps", json.dumps({str(n): list(b) for n, b in new.items()})).items()}
    caps = {(int(n), c): v for (n, c), v in caps.items()}
    print("caps (S* for F, never below 2): " + ", ".join(
        f"N = {n} {c}: {v}" for (n, c), v in sorted(caps.items())))
    data = tomllib.loads(a.params.read_text(encoding="utf-8-sig"))
    stack = (data.get("pilot_outputs", {}).get("s_star") or {})
    for n in (6, 12):
        if str(n) in stack and int(stack[str(n)]) != caps[(n, "jittery")]:
            print(f"  note: the stack's S* pilot gave S = {stack[str(n)]} at N = {n}; "
                  f"the cells' cap rule (never below 2) gives {caps[(n, 'jittery')]}")
    docs = docs_to_review(mapping)
    if a.dry_run:
        for p, n in docs.items():
            print(f"  review {n:3d} mention(s) in {p.relative_to(REPO)}")
        print("dry run: nothing written")
        return 0

    today = dt.date.today().isoformat()
    rename_everywhere(mapping, write=True)
    names12 = [label("jit", 12, b) for b in new[12]]
    held = {name: C.STUDY_5_6_LAGS_S[k] for name, k in zip(names12, C.STUDY_5_6_LAGS_S)}
    placeholder = {name: {"median": float(v), "lags": 0, "q1": 0.0, "q3": 0.0}
                   for name, v in held.items()}
    _replace_block(CELLS_PY, BLOCK_START, BLOCK_END,
                   cells_block(new, caps, held, placeholder, today))
    print(f"measuring FX's lags on {', '.join(names12)} ({a.episodes} validation episodes "
          f"each, {a.workers} workers)...")
    measured = _child("_lags", json.dumps(names12), str(a.episodes), str(a.workers),
                      str(a.boot))
    lags = {name: int(round(m["median"])) for name, m in measured.items()}
    for name, m in measured.items():
        print(f"  {name}: {m['median']:.3f} s over {m['lags']} lags -> {lags[name]} s; ratio "
              f"bounds {m['bounds']} (99 % range {m['ratio_99'][0]:.2f}-"
              f"{m['ratio_99'][1]:.2f})")
        lo, hi = m["bounds"]
        if not (lo > 0.5 and hi < 1.5):
            sys.exit(f"{name}: the ratio bounds {m['bounds']} let the quarter and half cells' "
                     f"ranges meet (they need lo > 0.5 and hi < 1.5): the ratio check needs "
                     f"more than {RATIO_EPISODES} episodes. Nothing further written; "
                     f"`git checkout -- .` undoes the rename.")
    _replace_block(CELLS_PY, BLOCK_START, BLOCK_END,
                   cells_block(new, caps, lags, measured, today))
    check = _child("_check")
    off = {k: v for k, v in check["caps_5_6"].items() if v != check["cap_12"]}
    if off:
        sys.exit(f"Study 5.6 cells whose S* at their own period is not their base's cap "
                 f"{check['cap_12']}: {off}. Their cap rule needs a decision.")
    _replace_block(PINS_PY, PINS_START, PINS_END,
                   pins_block(check["sha"], lags, check["p_c"], measured))
    print(f"wrote {CELLS_PY.relative_to(REPO)} and {PINS_PY.relative_to(REPO)}; families' "
          f"hashes: " + ", ".join(f"{k} {v[:12]}" for k, v in check["sha"].items()))
    for p, n in docs.items():
        print(f"  review {n:3d} mention(s) in {p.relative_to(REPO)}")
    print("\nnext:\n"
          "  python -m pytest tests/unit -q -p no:cacheprovider -m \"not slow\" -k \"ferrysim or "
          "p5_ or exp5\"\n"
          "  python -m pytest tests/unit/test_p5_ferrysim_runner.py -q -p no:cacheprovider "
          "-m slow\n"
          "  then review the docs above, commit, and set [rl] repinned = true")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
