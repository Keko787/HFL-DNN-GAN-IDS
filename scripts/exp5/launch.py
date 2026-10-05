"""Exp 5 launcher: one command per stage, from scripts/exp5/params.toml.

Exp 5 runs on the Exp 4 harness (python -m experiments.exp4.runner_main) with
the FeRRy flags. This launcher turns a stage into jobs, one runner process per
(study, cell, arm), and runs them side by side (exp5.cmd and exp5.sh call it
with the right interpreter):

    check     is the machine ready (interpreter, packages, dataset, code, memory,
              disk), and what each stage still needs; status with no stage
              gives the second half alone
    plan      list the jobs, what each still needs, the trial count and the cost
    validate  run every job's arguments through the runner's own usage checks
              (runner_main.main with the trial loop stubbed out: no trial runs)
    run       validate, write a manifest, then run the jobs in parallel
    status    count the rows each job's CSV holds against what it should
    report    a finished pilot's pilot_outputs lines for params.toml, by the
              pre-registered rules (ttl: 2 x p95 rounded up; knee: params
              [knee] metric and plateau_share; sstar: the tool's S for F)
    score     a batch's trials through the trace scorer, then each study's
              paired comparisons by params.toml [score] (scoring.py), written
              to <out_root>/scores/<stage dir>/ (index.md first)
    pack      a stage's kept traces (which git leaves out) into one archive,
              <out_root>/archives/<stage dir>_traces.tar.gz, its SHA-256 in
              archives/SHA256SUMS; unpack checks it and puts them back
    campaign  every stage in order, skipping those done; it stops before a stage
              whose settings are unset, after a stage whose report a person
              reads (the pilots, the calibration, Study 5.5's verdict), and at
              a failure. Run it again after setting what it asked for.

``--smoke`` (with run, plan or campaign) shrinks every job to its smallest (one
trial; RL at 30 episodes) and writes beside the repository (../exp5_smoke): a
dry run that checks every job starts and ends, never a result.

A stage may be a group instead: pilots (ttl knee sstar), rl (the five RL
stages), batches (batch1 sens batch2 pilot3 batch3) or all (the campaign);
`run` on a group goes through it as the campaign does. `report <pilot>
--apply` writes a finished, clean pilot's outputs into params.toml.

Levers change a setting for one run without editing params.toml: --trials,
--seed, --missions, --contact-regime, --tau, --dataset, --jobs, --mem-gb,
--devices, and --set KEY=VALUE for any other key. Each is printed, and
recorded in the manifest of every stage the run starts. --study and --arms
narrow which jobs run.

Stages, in order (each fills in settings the next one needs; see params.toml):

    ttl      the session-TTL pilot: fit_time_probe.py at each N
    knee     H1 budget sweeps on wide, per N and payload
    sstar    the S* tool per N at the knee and the stress budget
    batch1   5.3 core, 5.9 core, 5.11 (a) and (b), 5.14 (the 8 Oct split)
    sens     5.3's core arms at the session TTL x 0.75 and x 1.5 (x 1 is batch 1's)

The learned score's FerrySim campaign (Run Guide 2.8), after the re-pin
(params.toml [rl] repinned; every stage refuses until it is set):

    rl-headroom   the headroom report, which every verdict reads epsilon from
    rl-calibrate  gamma in {0, 0.9} x 3 seeds per family, evaluate, report
    rl-sweep      Study 5.5: 6 gammas x 10 seeds, evaluate --record, the verdict
    rl-e3         E3's trainings and evaluate --reward bytes --record
    rl-s57        Study 5.7's scores (F-hand, the dwell and cov ablations, the
                  reward grid), only if the 5.5 verdict keeps the learned score

Then the studies that wait for the verdict, and those with pilots of their own:

    batch2   5.1, 5.2, the rest of 5.3, 5.4 (with O1), 5.5's stack check, 5.6,
             5.7, 5.8, the rest of 5.9, 5.13
    pilot3   5.15's interference levels, 5.12's training-time levels (p512) and
             5.11 (c)'s FerrySim budget sweeps
    batch3   5.12, 5.15, 5.11 (c)

And, apart from the campaign, a reviewer's reproduction of batch 1:

    quick    5.3's core at the knee, batch 1's own jobs at fewer trials, in a
             folder of their own; `report quick` sets each trial beside the
             recorded one with the same seed

Each training is one `ferrysim train` job with an explicit tag, so a stopped
stage resumes by skipping the checkpoints that exist (`ferrysim sweep`
refuses outright once any of its checkpoints exists). A stage's evaluate and
report jobs wait for its trainings. Trainings run with one BLAS thread.

    scripts\\exp5\\exp5 check
    scripts\\exp5\\exp5 plan batch1
    scripts\\exp5\\exp5 run pilots
    scripts\\exp5\\exp5 report knee --apply
    scripts\\exp5\\exp5 run quick --trials 5
    scripts\\exp5\\exp5 status

What it keeps to:

* Paired seeds. Every trial's seed is sha256(base_seed | cell | trial), not
  the arm's, so one process per arm with the same base seed and cell flies
  the same layouts and draws as one invocation with every arm.
* One CSV per setting. No FeRRy flag is part of a trial's key, so every job
  writes its own CSV under results/exp5/<stage>/<study>/, and a sidecar
  <csv>.argv.json holds the arguments that wrote it; a resume under other
  arguments is refused. A killed run resumes by running the same stage again:
  the runner skips the (cell, arm, trial) rows already in each CSV. A trial
  that failed keeps its row (status error or timeout) and is not retried;
  `status` counts those rows as "not ok".
* Shared cells run once. Jobs whose arguments differ only in --csv and
  --n-trials are one job at the largest --n-trials; the others read its first
  n trials (same seeds), and the manifest lists each alias.
* Provenance. Trial rows do not record the commit, so each run writes
  _launcher/manifest_<time>.json: the commit and any change under hermes/ or
  experiments/ (refused unless --allow-dirty), the interpreter and package
  versions, the thread caps, the dataset's fingerprint, the host, this file's
  and params.toml's hashes and text, and every job's arguments. Each job's
  start and end go to _launcher/jobs.jsonl, its output to _logs/<job>.log.
* The real interpreter. A Windows venv's python.exe is a launcher stub that
  starts the real interpreter as a child, which doubles every process and
  hides it from the 5.11 footprint probe; run this through ..\\py311.cmd.
* One run at a time. `run` holds results/exp5/.launcher.lock while it works, so
  a second stage started beside it (stack trials and trainings would share the
  cores and skew both) is refused rather than slowing the first.

Nothing here decides a study's design: the arms, cells, budgets and counts are
params.toml's, and a stage whose settings are unset is refused.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.version_info < (3, 11):
    sys.exit("launch.py needs Python 3.11+ (tomllib)")
import tomllib  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARAMS = HERE / "params.toml"
RUNNER = ["-m", "experiments.exp4.runner_main"]
RL_STAGES = ("rl-headroom", "rl-calibrate", "rl-sweep", "rl-e3", "rl-s57")
STAGES = (("ttl", "knee", "sstar", "batch1", "sens") + RL_STAGES
          + ("batch2", "pilot3", "batch3", "quick"))
STAGE_DIR = {"ttl": "ttl", "knee": "knee", "sstar": "sstar", "batch1": "b1", "sens": "sens",
             **{st: "rl/" + st[3:] for st in RL_STAGES},
             "batch2": "b2", "pilot3": "p3", "batch3": "b3", "quick": "quick"}
#: The stages whose jobs may read batch 1's CSV for a cell they share with it.
#: Never quick: a reproduction flies its trials again.
REUSES_BATCH1 = ("sens", "batch2", "batch3")
#: The stages `score` reads (the pilots have `report`); each scores into
#: <out_root>/scores/<stage dir>/, with its scorer runs' logs there too.
SCORED = ("batch1", "sens", "batch2", "batch3", "quick")
STAGE_DIR.update({f"score-{st}": f"scores/{STAGE_DIR[st]}" for st in SCORED})
FERRYSIM = ["-m", "experiments.ferrysim"]

#: The plan arms take the age cap (--age-cap-missions); the H and D arms do not.
PLAN_ARMS = {"F", "FX", "FB+wide", "FB+medium", "FB+narrow", "F-cov", "F-prio", "F+L1",
             "F-round", "F-pref"}

#: Unit U11's arms fly their own deadline law (the round's, after FedCS or Oort),
#: whatever --deadline-law says, so the campaign's law is not passed to them.
OWN_LAW_ARMS = {"F-round", "F-pref"}


def is_ferry_arm(arm: str) -> bool:
    """The arms that fly campaign.ferry_arm_deadline_law: the plan arms, F-cap
    (F with no cap) and the learned FQ arms, less unit U11's own-law arms."""
    if arm in OWN_LAW_ARMS:
        return False
    return arm in PLAN_ARMS or arm == "F-cap" or arm.startswith("FQ")

#: The Phase 4 pilots' common settings (Run Guide 2.7), every runner job's base.
COMMON_FLAGS = [
    "--real-model", "--realism",
    "--mission-clock", "sim", "--contact-band", "wide", "--deadline-time-scale", "t_nom",
    "--in-flight-response", "replan", "--replan-fallback", "trim",
    "--aggregation", "agg:cutoff", "--contact-reliability-source", "channel",
    "--keep-event-traces",
]

#: Study 5.14: F against F with one switch flipped (build plan, 5.14). Each is
#: (job tag, arm, extra flags, cap) with cap "S", "S+1" or None (off).
LOCAL_SEARCH = '{"exact_max_devices": 0, "exhaustive_max_stops": 0}'
S514_VARIANTS: Tuple[Tuple[str, str, Tuple[str, ...], Optional[str]], ...] = (
    ("capS", "F", (), "S"),
    ("whole", "F", ("--member-admission", "whole"), "S"),
    ("local", "F", ("--plan-search-params", LOCAL_SEARCH), "S"),
    ("wcov", "F", ("--plan-score-params", '{"coverage_rank": "weighted"}'), "S"),
    ("capS1", "F", (), "S+1"),
    ("capoff", "F", (), None),
    ("hoveroff", "F", ("--plan-search-params", '{"hover_stops": false}'), "S"),
    ("secF", "F", ("--backhaul-model", "seconds"), "S"),
    ("secFL1", "F+L1", ("--backhaul-model", "seconds"), "S"),
)

#: Study 5.11 (a): the planner's mode, forced through --plan-search-params.
S511A_MODES = {
    "auto": None,
    "subsets": '{"exact_max_devices": 0}',
    "local": LOCAL_SEARCH,
}

#: Windows without long paths: a kept trace's deepest file must stay under this.
MAX_PATH = 250
#: Measured on results/ (5 Oct 2026): trace folders up to 93 characters
#: (N=6-dead_zone=0.6-...-rrf=60.0__H2__t0__s2191267877), files up to 26
#: (cluster-exp4-cluster.jsonl); these leave room for N=18, F+L1, t39.
TRACE_DIR_CHARS = 100
TRACE_FILE_CHARS = 30


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

class Settings:
    """params.toml, read by dotted key; a missing key is recorded, not raised."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.missing: List[str] = []
        self.overrides: List[str] = []     # the command line's changes, for the manifest

    def get(self, key: str, default: Any = KeyError) -> Any:
        node: Any = self.data
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is KeyError:
                    self.missing.append(key)
                    return None
                return default
        return node

    def per_n(self, key: str, n: int) -> Any:
        """A value keyed by N (``{"6" = ...}``)."""
        return self.get(f"{key}.{n}")


def load_settings(path: Path) -> Tuple[Settings, str]:
    # utf-8-sig: Windows editors (Notepad, PowerShell 5.1) may write a BOM.
    text = path.read_text(encoding="utf-8-sig")
    return Settings(tomllib.loads(text)), text


# --------------------------------------------------------------------------- #
# Levers: the command line's changes to params.toml, for this run only
# --------------------------------------------------------------------------- #

def toml_value(raw: str) -> Any:
    """A value as TOML reads it (5, 0.5, true, [1, 2], { "6" = 90.0 }, "x");
    anything TOML refuses is taken as a bare string (jittery)."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def set_dotted(data: Dict[str, Any], key: str, value: Any) -> None:
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"{key}: {part} is a value, not a table")
    node[parts[-1]] = value


def set_everywhere(data: Dict[str, Any], name: str, value: Any) -> int:
    """Set ``name`` in every table that has it; returns how many."""
    count = 0
    for v in data.values():
        if isinstance(v, dict):
            if name in v:
                v[name] = value
                count += 1
            count += set_everywhere(v, name, value)
    return count


#: Named levers: (option, params.toml key, type). Each is shorthand for
#: --set KEY=VALUE, and the manifest records it the same way.
LEVERS: Tuple[Tuple[str, str, Any], ...] = (
    ("seed", "campaign.base_seed", int),
    ("missions", "campaign.n_missions", int),
    ("contact_regime", "campaign.contact_regime", str),
    ("tau", "score.tau", float),
    ("mem_gb", "machine.mem_budget_gb", float),
    ("devices", "machine.max_device_processes", int),
)


def apply_levers(s: Settings, a: argparse.Namespace) -> None:
    """The levers and --set changes, applied to the settings this run reads.

    params.toml stays as it is; the manifest of every stage run records each
    change (and the settings as changed), and a CSV written under other
    arguments is still refused on resume, so a run that changes a job's
    arguments needs its own --out-root."""
    changes: List[str] = []
    if a.trials is not None:
        n = set_everywhere(s.data, "n_trials", int(a.trials))
        changes.append(f"--trials {a.trials} ({n} tables' n_trials)")
    for option, key, kind in LEVERS:
        value = getattr(a, option, None)
        if value is None:
            continue
        value = [kind(v) for v in value] if isinstance(value, list) else kind(value)
        set_dotted(s.data, key, value)
        changes.append(f"{key} = {json.dumps(value)} (--{option.replace('_', '-')})")
    for item in a.set or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            sys.exit(f"--set takes KEY=VALUE (a dotted params.toml key), not {item!r}")
        if key.split(".")[0] not in s.data:
            sys.exit(f"--set {key}: params.toml has no [{key.split('.')[0]}] table")
        value = toml_value(raw.strip())
        set_dotted(s.data, key, value)
        changes.append(f"{key} = {json.dumps(value)} (--set)")
    if a.dataset:
        d = Path(a.dataset).resolve()
        os.environ["HERMES_CICIOT_DIR"] = str(d)        # the jobs inherit it
        changes.append(f"HERMES_CICIOT_DIR = {d} (--dataset)")
    s.overrides = changes
    for c in changes:
        print(f"[lever] {c}")


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    stage: str
    study: str
    name: str                     # unique within the stage: <study>/<tag>
    args: List[str]               # after the interpreter
    out: str                      # the CSV, JSON or checkpoint it writes, relative to REPO
    kind: str                     # "runner", "tool", "rl-train" or "score"
    n: int = 0                    # devices (for cost)
    mem_gb: float = 0.0
    trials: Optional[int] = None  # rows it should write (runner jobs)
    blocked: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    alias_of: Optional[str] = None
    deps: List[str] = field(default_factory=list)   # jobs of this stage it waits for
    slots: int = 1                # scheduler slots (an evaluate with W workers takes W)


def _fmt(x: Any) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _payload_tag(p: Any) -> str:
    if p == "measured":
        return "meas"
    p = int(p)
    return f"{p // 1_000_000}mb" if p % 1_000_000 == 0 else f"{p}b"


def _arm_tag(arm: str) -> str:
    return arm.replace("+", "p")


def _ferrysim_cells():
    """FerrySim's cell registry (Study 5.6's cells and periods follow the re-pin)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from experiments.ferrysim import cells
    return cells


class Builder:
    """Expands one stage of params.toml into jobs."""

    def __init__(self, s: Settings, root: str = "results/exp5"):
        self.s = s
        self.root = root.replace("\\", "/").rstrip("/")
        self.jobs: List[Job] = []

    # -- shared pieces -------------------------------------------------- #

    def _mem_runner(self, n: int, k: int) -> float:
        tf = float(self.s.get("machine.gb_per_tf_process"))
        light = float(self.s.get("machine.gb_per_light_process"))
        return tf * (1 + n) + light * (1 + k)

    def concurrency(self, n: int, k: int = 1) -> int:
        """Runner jobs of this size the scheduler runs at once (jobs, memory, devices)."""
        cap = int(self.s.get("machine.max_jobs"))
        mem = float(self.s.get("machine.mem_budget_gb"))
        devices = int(self.s.get("machine.max_device_processes"))
        return max(1, min(cap, int(mem // self._mem_runner(n, k)), devices // max(1, n)))

    def _runner(self, stage: str, study: str, tag: str, *, arm: str, n: int, k: int,
                budget_s: Any, payload: Any, n_trials: int, extra: Sequence[str] = (),
                cap: Any = None, footprint: bool = False,
                n_missions: Optional[int] = None, ttl_factor: float = 1.0) -> Job:
        s = self.s
        before = len(s.missing)
        out = f"{self.root}/{STAGE_DIR[stage]}/{study}/{tag}.csv"
        ttl = s.per_n("pilot_outputs.session_ttl_s", n)
        if ttl is not None and ttl_factor != 1.0:      # the sensitivity stage's TTLs
            ttl = float(ttl) * ttl_factor
        seed = s.get("campaign.base_seed")
        missions = n_missions if n_missions is not None else s.get("campaign.n_missions")
        args = RUNNER + [
            "--csv", out, "--arms", arm, "--N", str(n),
            "--n-missions", _fmt(missions),
            "--regime", str(s.get("campaign.regime")),
            "--n-trials", str(n_trials), "--base-seed", _fmt(seed),
            "--mission-budget-s", _fmt(budget_s), "--session-ttl-s", _fmt(ttl),
        ] + COMMON_FLAGS
        contact = s.get("campaign.contact_regime", None)
        if contact and "--contact-regime" not in extra:
            args += ["--contact-regime", str(contact)]
        if payload != "measured":
            args += ["--payload-bytes", _fmt(payload)]
        if k > 1:
            args += ["--n-mules", str(k)]
        ref_n = int(s.get("campaign.field_ref_n", 0) or 0)
        if ref_n:
            args += ["--h1-field-ref-n", str(ref_n)]
        if cap is not None:
            args += ["--age-cap-missions", _fmt(cap)]
        law = s.get("campaign.ferry_arm_deadline_law", None)
        if law and is_ferry_arm(arm) and "--deadline-law" not in extra:
            args += ["--deadline-law", str(law)]
        # agg:cutoff cuts by age only with a merge period (Configuration
        # Reference 14 and 20.12); every stage after the knee pilot passes T_nom,
        # to every job whose merge is agg:cutoff (not, say, D4's own agg:fedex).
        merge = (list(extra)[list(extra).index("--aggregation") + 1]
                 if "--aggregation" in extra else "agg:cutoff")
        if (stage != "knee" and merge == "agg:cutoff"
                and s.get("campaign.merge_period_t_nom", False)):
            args += ["--agg-period-t-nom"]
        if footprint:
            args += ["--footprint-probe"]
        args += list(extra)
        job = Job(stage, study, f"{study}/{tag}", args, out, "runner", n=n,
                  mem_gb=self._mem_runner(n, k), trials=n_trials)
        job.blocked = sorted(set(s.missing[before:]))
        del s.missing[before:]
        return job

    def _budget(self, level: str, n: int) -> Any:
        if level == "relaxed":              # 5.3's third budget, a multiple of the knee
            knee = self.s.per_n("pilot_outputs.knee_s", n)
            factor = self.s.get("batch2.relaxed_factor")
            return None if knee is None or factor is None else float(knee) * float(factor)
        key = {"knee": "knee_s", "stress": "stress_s", "knee_meas": "knee_meas_s",
               "stress_meas": "stress_meas_s"}[level]
        return self.s.per_n(f"pilot_outputs.{key}", n)

    def _cap(self, cap: Optional[str], n: int) -> Any:
        if cap is None:
            return None
        s_star = self.s.per_n("pilot_outputs.s_star", n)
        if s_star is None:
            return None
        return int(s_star) + (1 if cap == "S+1" else 0)

    def _arm_job(self, stage: str, study: str, *, arm: str, n: int, k: int, level: str,
                 payload: Any, n_trials: int, extra: Sequence[str] = (),
                 cap: Optional[str] = "S", tag_arm: Optional[str] = None,
                 footprint: bool = False, n_missions: Optional[int] = None,
                 tag_extra: str = "", budget_s: Any = None, ttl_factor: float = 1.0) -> Job:
        before = len(self.s.missing)
        budget = self._budget(level, n) if budget_s is None else budget_s
        cap_value = self._cap(cap, n) if arm in PLAN_ARMS or arm.startswith("FQ") else None
        learned, learned_missing = self._learned(arm)
        missing = self.s.missing[before:] + learned_missing
        del self.s.missing[before:]
        tag = f"n{n}k{k}_{level}{tag_extra}__{tag_arm or _arm_tag(arm)}"
        job = self._runner(stage, study, tag, arm=arm, n=n, k=k, budget_s=budget,
                           payload=payload, n_trials=n_trials, extra=list(extra) + learned,
                           cap=cap_value, footprint=footprint, n_missions=n_missions,
                           ttl_factor=ttl_factor)
        job.blocked = sorted(set(job.blocked) | set(missing))
        return job

    # -- learned arms (batch 2): the checkpoint each flies ---------------------- #

    _PAIR_TAG = {"FQ": "main", "FQ-hand": "hand", "FQ-dwell": "dwell", "FQ-cov": "cov"}

    def keeps(self, arm: str) -> Optional[bool]:
        """Whether a study flies ``arm``: an FQ arm only when the 5.5 verdict kept
        the learned score (None: not decided yet); every other arm always."""
        if not arm.startswith("FQ"):
            return True
        return self.s.get("rl.keep_learned", None)

    def _learned(self, arm: str) -> Tuple[List[str], List[str]]:
        """The checkpoint flag ``arm`` flies with, and the settings it still needs."""
        s = self.s
        if arm == "E3":
            path = s.get("rl.checkpoints.e3", None)
            return (["--policy-checkpoint", f"E3={path}"], []) if path else (
                [], ["rl.checkpoints.e3"])
        if not arm.startswith("FQ"):
            return [], []
        if arm in self._PAIR_TAG:
            tag = self._PAIR_TAG[arm]
            key = "main" if tag == "main" else tag
        else:                               # FQ-g<X>: the 5.5 stack check's picks
            tag = arm[3:]
            key = "g0" if tag == "g0" else "best"
        path = s.get(f"rl.checkpoints.{key}", None)
        return (["--pair-checkpoint", f"{tag}={path}"], []) if path else (
            [], [f"rl.checkpoints.{key}"])

    def _arms(self, arms: Sequence[str]) -> Tuple[List[str], List[str]]:
        """The arms a batch-2 study flies, and what decides the rest: FQ arms are
        dropped when the verdict did not keep the learned score, blocked while
        it is undecided."""
        flown, undecided = [], []
        for arm in arms:
            keep = self.keeps(arm)
            if keep is None:
                undecided.append(arm)
                flown.append(arm)
            elif keep:
                flown.append(arm)
        return flown, (["rl.keep_learned"] if undecided else [])

    def _study_jobs(self, stage: str, study: str, *, arms: Sequence[str], n: int, k: int,
                    level: str, payload: Any, n_trials: int, extra: Sequence[str] = (),
                    tag_extra: str = "", n_missions: Optional[int] = None,
                    budget_s: Any = None, fp: bool = True) -> None:
        """One job per arm of one cell, with the batch-2 arm rules applied."""
        flown, undecided = self._arms(arms)
        for arm in flown:
            job = self._arm_job(stage, study, arm=arm, n=n, k=k, level=level, payload=payload,
                                n_trials=n_trials, extra=extra, tag_extra=tag_extra,
                                n_missions=n_missions, budget_s=budget_s, footprint=fp,
                                tag_arm=None)
            if arm.startswith("FQ") and undecided:
                job.blocked = sorted(set(job.blocked) | set(undecided))
            self.jobs.append(job)

    # -- stages --------------------------------------------------------- #

    def ttl(self) -> None:
        s = self.s
        fits = int(s.get("ttl.fits"))
        for n in s.get("ttl.N") or []:
            side = self.concurrency(int(n))
            out = f"{self.root}/ttl/fit_time_n{n}.json"
            args = ["scripts/exp5/fit_time_probe.py", "--N", str(n), "--fits", str(fits),
                    "--trials", str(side), "--out", out]
            seed = s.get("campaign.base_seed", None)
            if seed is not None:
                args += ["--seed", _fmt(seed)]
            # The probe is the load it measures: it runs alone.
            self.jobs.append(Job("ttl", "ttl", f"ttl/n{n}", args, out, "tool", n=int(n),
                                 mem_gb=float(s.get("machine.mem_budget_gb"))))

    def knee(self) -> None:
        s = self.s
        n_trials = int(s.get("knee.n_trials"))
        for n, payload in s.get("knee.cells") or []:
            n = int(n)
            budgets = s.per_n("knee.budgets_s", n)
            if budgets is None:
                job = self._runner("knee", "knee", f"n{n}_{_payload_tag(payload)}_b_unset",
                                   arm="H1", n=n, k=1, budget_s=None, payload=payload,
                                   n_trials=n_trials)
                job.blocked = sorted(set(job.blocked) | {f"knee.budgets_s.{n}"})
                self.jobs.append(job)
                continue
            for b in budgets:
                self.jobs.append(self._runner(
                    "knee", "knee", f"n{n}_{_payload_tag(payload)}_b{int(b):04d}",
                    arm="H1", n=n, k=1, budget_s=b, payload=payload, n_trials=n_trials))

    def sstar(self) -> None:
        s = self.s
        payload = s.get("sstar.payload_bytes")
        ref_n = int(s.get("campaign.field_ref_n", 0) or 0)
        for n in s.get("sstar.N") or []:
            n = int(n)
            before = len(s.missing)
            knee, stress = self._budget("knee", n), self._budget("stress", n)
            out = f"{self.root}/sstar/s_star_n{n}.json"
            args = ["-m", "experiments.analysis.age_cap_s_star", "--N", str(n),
                    "--budgets", _fmt(knee), _fmt(stress), "--payload-bytes", _fmt(payload),
                    "--contact-band", "wide", "--regime", str(s.get("campaign.regime")),
                    "--json", out]
            if ref_n:
                args += ["--field-ref-n", str(ref_n)]
            job = Job("sstar", "sstar", f"sstar/n{n}", args, out, "tool", n=n,
                      mem_gb=float(s.get("machine.gb_per_light_process")))
            job.blocked = sorted(set(s.missing[before:]))
            del s.missing[before:]
            self.jobs.append(job)

    def batch1(self) -> None:
        s = self.s
        fp = bool(s.get("campaign.footprint", True))
        # 5.3, core cells.
        for k in s.get("s53.K") or []:
            for n in s.get("s53.N") or []:
                for level in s.get("s53.budgets") or []:
                    for arm in s.get("s53.arms") or []:
                        self.jobs.append(self._arm_job(
                            "batch1", "s53", arm=arm, n=int(n), k=int(k), level=level,
                            payload=s.get("s53.payload_bytes"),
                            n_trials=int(s.get("s53.n_trials")), footprint=fp))
                    if s.get("s53.d4_faithful", False):
                        self.jobs.append(self._arm_job(
                            "batch1", "s53", arm="D4", n=int(n), k=int(k), level=level,
                            payload=s.get("s53.payload_bytes"),
                            n_trials=int(s.get("s53.n_trials")),
                            extra=("--aggregation", "agg:fedex"), tag_arm="D4fedex",
                            footprint=fp))
        # 5.9, core cells.
        for k in s.get("s59.K") or []:
            for n in s.get("s59.N") or []:
                for level in s.get("s59.budgets") or []:
                    for arm in s.get("s59.arms") or []:
                        self.jobs.append(self._arm_job(
                            "batch1", "s59", arm=arm, n=int(n), k=int(k), level=level,
                            payload=s.get("s59.payload_bytes"),
                            n_trials=int(s.get("s59.n_trials")), footprint=fp))
        # 5.11 (b), weak and strong scaling with mules.
        for n, k in s.get("s511b.cells") or []:
            for level in s.get("s511b.budgets") or []:
                for arm in s.get("s511b.arms") or []:
                    self.jobs.append(self._arm_job(
                        "batch1", "s511b", arm=arm, n=int(n), k=int(k), level=level,
                        payload=s.get("s511b.payload_bytes"),
                        n_trials=int(s.get("s511b.n_trials")), footprint=fp))
        # 5.14, F against F with one switch flipped.
        n = int(s.get("s514.N"))
        for level in s.get("s514.budgets") or []:
            for tag, arm, extra, cap in S514_VARIANTS:
                self.jobs.append(self._arm_job(
                    "batch1", "s514", arm=arm, n=n, k=1, level=level,
                    payload=s.get("s514.payload_bytes"), n_trials=int(s.get("s514.n_trials")),
                    extra=extra, cap=cap, tag_arm=tag, footprint=fp))
        # 5.11 (a), decision cost in FerrySim (in process; no TensorFlow).
        workers = int(s.get("s511a.workers"))
        for mode in s.get("s511a.modes") or []:
            base = f"{self.root}/b1/s511a/{mode}"
            args = ["-m", "experiments.ferrysim", "pilot",
                    "--cells", *[str(c) for c in s.get("s511a.cells") or []],
                    "--policies", *[str(p) for p in s.get("s511a.policies") or []],
                    "--episodes", str(int(s.get("s511a.episodes"))),
                    "--workers", str(workers), "--trace-root", base + "_traces",
                    "--out", base + ".json"]
            if S511A_MODES[mode] is not None:
                args += ["--plan-search-params", S511A_MODES[mode]]
            light = float(s.get("machine.gb_per_light_process"))
            self.jobs.append(Job("batch1", "s511a", f"s511a/{mode}", args, base + ".json",
                                 "tool", n=0, mem_gb=light * (1 + 2 * workers)))


    # -- batch 2: the studies after the RL verdict ----------------------------- #

    def batch2(self) -> None:
        s = self.s
        st = "batch2"
        # 5.1: the merge rules, each on a fixed route.
        rho = s.get("s51.fedprox_rho", None)
        n51 = int(s.get("s51.N"))
        for route in s.get("s51.routes") or []:
            for rule in s.get("s51.rules") or []:
                extra: List[str] = []
                missing: List[str] = []
                if rule == "agg:cutoff+fedprox":
                    extra = ["--aggregation", "agg:cutoff", "--fedprox-rho", _fmt(rho)]
                    if rho is None:
                        missing = ["s51.fedprox_rho"]
                elif rule == "agg:fedbuff":
                    extra = ["--aggregation", rule, "--agg-buffer-k", str(n51)]
                elif rule != "agg:cutoff":
                    extra = ["--aggregation", rule]
                rtag = rule.replace("agg:", "").replace("+", "_")
                for p2 in (s.get("s51.pass_2") or []) if route == "H1" else ["unbudgeted"]:
                    p2_extra = ["--pass-2-budget"] if p2 == "budgeted" else []
                    for level in s.get("s51.budgets") or []:
                        # The cell is the route and Pass 2; the variant compared is the rule.
                        job = self._arm_job(
                            st, "s51", arm=route, n=n51, k=1, level=level,
                            payload=s.get("s51.payload_bytes"),
                            n_trials=int(s.get("s51.n_trials")), extra=extra + p2_extra,
                            tag_extra=f"_{_arm_tag(route)}_{p2[:5]}", tag_arm=rtag,
                            footprint=True)
                        job.blocked = sorted(set(job.blocked) | set(missing))
                        self.jobs.append(job)
        # 5.2: the deadline forms (F-add is F with the additive law).
        for level in s.get("s52.budgets") or []:
            for arm in s.get("s52.arms") or []:
                real, extra = (("F", ["--deadline-law", "additive"]) if arm == "F-add"
                               else (arm, []))
                self.jobs.append(self._arm_job(
                    st, "s52", arm=real, n=int(s.get("s52.N")), k=1, level=level,
                    payload=s.get("s52.payload_bytes"), n_trials=int(s.get("s52.n_trials")),
                    extra=extra, tag_arm=_arm_tag(arm), footprint=True))
        # 5.3 beyond batch 1's core (cells it shares with batch 1 read batch 1's CSV).
        for k in s.get("s53x.K") or []:
            for level in s.get("s53x.budgets") or []:
                self._study_jobs(st, "s53x", arms=s.get("s53x.arms") or [],
                                 n=int(s.get("s53x.N")), k=int(k), level=level,
                                 payload=s.get("s53x.payload_bytes"),
                                 n_trials=int(s.get("s53x.n_trials")))
                if s.get("s53x.d4_faithful", False):
                    self.jobs.append(self._arm_job(
                        st, "s53x", arm="D4", n=int(s.get("s53x.N")), k=int(k), level=level,
                        payload=s.get("s53x.payload_bytes"),
                        n_trials=int(s.get("s53x.n_trials")),
                        extra=("--aggregation", "agg:fedex"), tag_arm="D4fedex",
                        footprint=True))
        if s.get("s53x.h0", False):
            self.jobs.append(self._h0_job(st, "s53x", n=int(s.get("s53x.N")),
                                          n_trials=int(s.get("s53x.n_trials"))))
        # 5.4: F against FB+ pinned to each class, one U10 axis at a time.
        n54 = int(s.get("s54.N"))
        settings: List[Tuple[str, List[str], Dict[str, Any], Dict[str, Any]]] = [
            ("", [], {}, {})]
        for r in s.get("s54.narrow_range_ratios") or []:
            settings.append((f"_ratio{_fmt(r)}", ["--narrow-range-ratio", _fmt(r)],
                             {"narrow_range_ratio": float(r)}, {}))
        for f in s.get("s54.far_shares") or []:
            settings.append((f"_far{int(round(float(f) * 100))}", ["--far-share", _fmt(f)],
                             {}, {"far_share": float(f)}))
        for payload in s.get("s54.payloads") or []:
            level = s.get("s54.budget") + ("_meas" if payload == "measured" else "")
            for tag, extra, _phys, _drv in settings:
                for arm in s.get("s54.arms") or []:
                    self.jobs.append(self._arm_job(
                        st, "s54", arm=arm, n=n54, k=1, level=level, payload=payload,
                        n_trials=int(s.get("s54.n_trials")), extra=extra,
                        tag_extra=f"_{_payload_tag(payload)}{tag}", footprint=True))
        if s.get("s54.o1", False):
            cells6 = [c.name for c in _ferrysim_cells().FAMILIES["jittery"] if c.n_devices == 6]
            w = int(s.get("s54.o1_workers"))
            for tag, _extra, phys, drv in settings:
                out = f"{self.root}/b2/s54/o1{tag or '_base'}.json"
                args = ["-m", "experiments.analysis.o1_oracle", "--cells", *cells6,
                        "--episodes", str(int(s.get("s54.o1_episodes"))), "--workers", str(w),
                        "--out", out]
                if phys:
                    args += ["--physics", json.dumps(phys)]
                if drv:
                    args += ["--driver", json.dumps(drv)]
                self.jobs.append(Job(st, "s54", f"s54/o1{tag or '_base'}", args, out, "tool",
                                     mem_gb=float(s.get("machine.gb_per_light_process")) * (1 + w),
                                     slots=w))
        # 5.5's stack check: the verdict's best gamma and gamma = 0, beside FX and F.
        best = s.get("rl.checkpoints.best_tag", None)
        arms55 = [f"FQ-{best}" if best else "FQ-gbest", "FQ-g0", "FX", "F"]
        for level in s.get("s55.budgets") or []:
            for arm in arms55:
                job = self._arm_job(st, "s55", arm=arm, n=int(s.get("s55.N")), k=1,
                                    level=level, payload=s.get("s55.payload_bytes"),
                                    n_trials=int(s.get("s55.n_trials")), footprint=True)
                if arm == "FQ-gbest":
                    job.blocked = sorted(set(job.blocked) | {"rl.checkpoints.best_tag"})
                self.jobs.append(job)
        # 5.6: the cells at each interference period, and the clean control.
        n56 = s.get("s56.n_trials", None)
        cells = _ferrysim_cells().STUDY_5_6_CELLS
        for cell in cells:
            self._study_jobs(st, "s56", arms=s.get("s56.arms") or [], n=int(s.get("s56.N")),
                             k=1, level="cell", payload=s.get("s56.payload_bytes"),
                             n_trials=int(n56 or 1), budget_s=cell.budget_s,
                             extra=["--interference-period-s",
                                    _fmt(float(cell.interference_period_s))],
                             tag_extra=f"_{_fmt(cell.budget_s)}s_p{_fmt(cell.interference_period_s)}")
        for budget in sorted({c.budget_s for c in cells}):
            self._study_jobs(st, "s56", arms=s.get("s56.arms") or [], n=int(s.get("s56.N")),
                             k=1, level="cell", payload=s.get("s56.payload_bytes"),
                             n_trials=int(n56 or 1), budget_s=budget,
                             extra=["--contact-regime", "clean"],
                             tag_extra=f"_{_fmt(budget)}s_clean")
        if n56 is None:
            for j in self.jobs:
                if j.study == "s56":
                    j.blocked = sorted(set(j.blocked) | {"s56.n_trials"})
        # 5.7: the plan terms and the reward (learned score only).
        for level in s.get("s57.budgets") or []:
            self._study_jobs(st, "s57", arms=s.get("s57.arms") or [], n=int(s.get("s57.N")),
                             k=1, level=level, payload=s.get("s57.payload_bytes"),
                             n_trials=int(s.get("s57.n_trials")))
        # 5.8: fairness under cost, at 4 and 8 missions.
        for missions in s.get("s58.missions") or []:
            for level in s.get("s58.budgets") or []:
                self._study_jobs(st, "s58", arms=s.get("s58.arms") or [], n=int(s.get("s58.N")),
                                 k=1, level=level, payload=s.get("s58.payload_bytes"),
                                 n_trials=int(s.get("s58.n_trials")), n_missions=int(missions),
                                 tag_extra=f"_m{int(missions)}")
        # 5.9 beyond batch 1's core.
        for regime in s.get("s59x.contact_regimes") or []:
            extra = [] if regime == s.get("campaign.contact_regime", None) else [
                "--contact-regime", str(regime)]
            for n in s.get("s59x.N") or []:
                for k in s.get("s59x.K") or []:
                    for level in s.get("s59x.budgets") or []:
                        self._study_jobs(st, "s59x", arms=s.get("s59x.arms") or [], n=int(n),
                                         k=int(k), level=level,
                                         payload=s.get("s59x.payload_bytes"),
                                         n_trials=int(s.get("s59x.n_trials")), extra=extra,
                                         tag_extra="" if not extra else f"_{regime}")
        # 5.13: data heterogeneity, with family labels throughout.
        n513 = int(s.get("s513.N"))
        for part, alpha in s.get("s513.partitions") or []:
            extra = ["--family-labels"]
            if part != "iid":
                extra += ["--partition", str(part), "--dirichlet-alpha", _fmt(alpha)]
            ptag = f"_{part}" + ("" if part == "iid" else f"{_fmt(alpha)}")
            self._study_jobs(st, "s513", arms=s.get("s513.route_arms") or [], n=n513, k=1,
                             level=s.get("s513.budget"), payload=s.get("s513.payload_bytes"),
                             n_trials=int(s.get("s513.n_trials")), extra=extra, tag_extra=ptag)
            route = s.get("s513.plain_on_route")
            self.jobs.append(self._arm_job(
                st, "s513", arm=route, n=n513, k=1, level=s.get("s513.budget"),
                payload=s.get("s513.payload_bytes"), n_trials=int(s.get("s513.n_trials")),
                extra=extra + ["--aggregation", "agg:plain"], tag_extra=ptag,
                tag_arm=f"{_arm_tag(route)}plain", footprint=True))

    def _h0_job(self, stage: str, study: str, *, n: int, n_trials: int) -> Job:
        """H0, the live-link reference: no mule, the wall clock only (Phase 3
        deviations), the network regime and the real model as the rest."""
        s = self.s
        before = len(s.missing)
        out = f"{self.root}/{STAGE_DIR[stage]}/{study}/n{n}_wall__H0.csv"
        args = RUNNER + ["--csv", out, "--arms", "H0", "--N", str(n),
                         "--n-missions", _fmt(s.get("campaign.n_missions")),
                         "--regime", str(s.get("campaign.regime")),
                         "--n-trials", str(n_trials),
                         "--base-seed", _fmt(s.get("campaign.base_seed")),
                         "--real-model", "--realism", "--keep-event-traces"]
        job = Job(stage, study, f"{study}/n{n}_wall__H0", args, out, "runner", n=n,
                  mem_gb=self._mem_runner(n, 0), trials=n_trials)
        job.blocked = sorted(set(s.missing[before:]))
        del s.missing[before:]
        return job

    # -- batch 3's pilots, and batch 3 ----------------------------------------- #

    def pilot3(self) -> None:
        s = self.s
        # 5.15's harsher interference: H1's service by amplitude, at the knee.
        n = int(s.get("p515.N"))
        for amp in s.get("p515.amps_db") or []:
            self.jobs.append(self._arm_job(
                "pilot3", "p515", arm="H1", n=n, k=1, level="knee", payload=1000000,
                n_trials=int(s.get("p515.n_trials")),
                extra=["--interference-amp-db", _fmt(amp)], tag_extra=f"_amp{_fmt(amp)}",
                footprint=False))
        # 5.11 (c): FerrySim's budget sweep per scale cell.
        w = int(s.get("p511c.workers"))
        for cell in s.get("p511c.cells") or []:
            budgets = s.get(f"p511c.budgets_s.{cell}", None)
            out = f"{self.root}/p3/p511c/{cell}.json"
            args = FERRYSIM + ["pilot", "--cells", str(cell),
                               "--budgets", *[_fmt(b) for b in (budgets or [])],
                               "--policies", *[str(p) for p in s.get("p511c.policies") or []],
                               "--episodes", str(int(s.get("p511c.episodes"))),
                               "--workers", str(w), "--out", out]
            job = Job("pilot3", "p511c", f"p511c/{cell}", args, out, "tool",
                      mem_gb=float(s.get("machine.gb_per_light_process")) * (1 + w), slots=w)
            if budgets is None:
                job.blocked = [f"p511c.budgets_s.{cell}"]
            self.jobs.append(job)
        # 5.12's training-time levels: H1 at the knee, the median fit time at multiples
        # of the mission cycle (the knee budget plus the turnaround), a fixed spread.
        n = int(s.get("p512.N"))
        knee = s.get(f"pilot_outputs.knee_s.{n}", None)       # unset: _arm_job blocks
        cycle = None if knee is None else float(knee) + float(s.get("p512.turnaround_s"))
        for factor in s.get("p512.cycle_factors") or []:
            median = None if cycle is None else float(round(float(factor) * cycle))
            self.jobs.append(self._arm_job(
                "pilot3", "p512", arm="H1", n=n, k=1, level="knee",
                payload=s.get("p512.payload_bytes"), n_trials=int(s.get("p512.n_trials")),
                extra=["--train-time-s", _fmt(median),
                       "--train-time-sigma", _fmt(float(s.get("p512.sigma")))],
                tag_extra=f"_cyc{_fmt(float(factor))}", footprint=False))

    def batch3(self) -> None:
        s = self.s
        st = "batch3"
        # 5.12: training time on the simulated clock x payload (x model), at the
        # levels pilot3's p512 set.
        levels = s.get("pilot_outputs.train_levels", None)
        n512 = int(s.get("s512.N"))
        for arch in s.get("s512.model_archs") or ["ciciot"]:
            arch_extra = [] if arch == "ciciot" else ["--model-arch", str(arch)]
            for name, vals in (levels or {"unset": [0, 0, 0, 1]}).items():
                median, sigma, share, factor = (float(v) for v in vals)
                extra = arch_extra + ["--train-time-s", _fmt(median)]
                if median > 0:
                    extra += ["--train-time-sigma", _fmt(sigma), "--straggler-share",
                              _fmt(share), "--straggler-factor", _fmt(factor)]
                for payload in s.get("s512.payloads") or []:
                    level = s.get("s512.budget") + ("_meas" if payload == "measured" else "")
                    for arm in s.get("s512.arms") or []:
                        job = self._arm_job(
                            st, "s512", arm=arm, n=n512, k=1, level=level, payload=payload,
                            n_trials=int(s.get("s512.n_trials")), extra=extra,
                            tag_extra=f"_{_payload_tag(payload)}_{name}"
                                      + ("" if arch == "ciciot" else f"_{arch}"),
                            footprint=True)
                        if levels is None:
                            job.blocked = sorted(set(job.blocked)
                                                 | {"pilot_outputs.train_levels"})
                        self.jobs.append(job)
        # 5.15: one radio axis at a time from the jittery default, on the seconds backhaul.
        n515 = int(s.get("s515.N"))
        base = ["--backhaul-model", str(s.get("s515.backhaul"))]
        harsher = s.get("s515.harsher_amp_db", None)
        lossier = s.get("s515.lossier_n_pl", None)
        axes: List[Tuple[str, List[str], List[str]]] = [("_base", [], [])]
        for a in s.get("s515.amps_db") or []:
            axes.append((f"_amp{_fmt(a)}", ["--interference-amp-db", _fmt(a)], []))
        axes.append(("_ampharsh", ["--interference-amp-db", _fmt(harsher)],
                     [] if harsher is not None else ["s515.harsher_amp_db"]))
        axes.append(("_npl", ["--n-pl", _fmt(lossier)],
                     [] if lossier is not None else ["s515.lossier_n_pl"]))
        for sg in s.get("s515.shadow_sigmas_db") or []:
            axes.append((f"_sigma{_fmt(sg)}", ["--shadow-sigma-db", _fmt(sg)], []))
        for tag, extra, missing in axes:
            for arm in s.get("s515.arms") or []:
                job = self._arm_job(st, "s515", arm=arm, n=n515, k=1,
                                    level=s.get("s515.budget"),
                                    payload=s.get("s515.payload_bytes"),
                                    n_trials=int(s.get("s515.n_trials")), extra=base + extra,
                                    tag_extra=tag, footprint=True)
                job.blocked = sorted(set(job.blocked) | set(missing))
                self.jobs.append(job)
        for arm in s.get("s515.mission_backhaul_arms") or []:
            self.jobs.append(self._arm_job(
                st, "s515", arm=arm, n=n515, k=1, level=s.get("s515.budget"),
                payload=s.get("s515.payload_bytes"), n_trials=int(s.get("s515.n_trials")),
                tag_extra="_missionbh", footprint=True))
        # 5.11 (c): FerrySim at each scale cell's knee, every policy.
        knees = s.get("s511c.knee_s", None)
        w = int(s.get("s511c.workers"))
        for cell in s.get("p511c.cells") or []:
            knee = None if knees is None else knees.get(cell)
            out = f"{self.root}/b3/s511c/{cell}.json"
            args = FERRYSIM + ["pilot", "--cells", str(cell), "--budgets", _fmt(knee),
                               "--policies", *[str(p) for p in s.get("s511c.policies") or []],
                               "--episodes", str(int(s.get("s511c.episodes"))),
                               "--workers", str(w), "--trace-root", out[:-5] + "_traces",
                               "--out", out]
            job = Job(st, "s511c", f"s511c/{cell}", args, out, "tool",
                      mem_gb=float(s.get("machine.gb_per_light_process")) * (1 + w), slots=w)
            if knee is None:
                job.blocked = [f"s511c.knee_s.{cell}"]
            self.jobs.append(job)

    # -- the quick reproduction, and the sensitivity to a pilot's value -------- #

    def quick(self) -> None:
        """A reproduction of 5.3's core in a few hours, for an artifact reviewer:
        batch 1's own jobs (same arguments, hence the same seeds) at fewer
        trials, written to CSVs of their own, never batch 1's. `report quick`
        then sets each trial beside the recorded one with the same seed."""
        s = self.s
        fp = bool(s.get("campaign.footprint", True))
        n, payload, trials = int(s.get("quick.N")), s.get("quick.payload_bytes"), int(
            s.get("quick.n_trials"))
        for k in s.get("quick.K") or []:
            for level in s.get("quick.budgets") or []:
                for arm in s.get("quick.arms") or []:
                    self.jobs.append(self._arm_job(
                        "quick", "s53", arm=arm, n=n, k=int(k), level=level, payload=payload,
                        n_trials=trials, footprint=fp))
                if s.get("quick.d4_faithful", False):
                    self.jobs.append(self._arm_job(
                        "quick", "s53", arm="D4", n=n, k=int(k), level=level, payload=payload,
                        n_trials=trials, extra=("--aggregation", "agg:fedex"),
                        tag_arm="D4fedex", footprint=fp))

    def sens(self) -> None:
        """5.3's core arms at the session TTL times each factor. Factor 1 is batch
        1's own cell, which this stage reads from batch 1 (REUSES_BATCH1)."""
        s = self.s
        fp = bool(s.get("campaign.footprint", True))
        for factor in s.get("sens.ttl_factors") or []:
            for level in s.get("sens.budgets") or []:
                for arm in s.get("sens.arms") or []:
                    self.jobs.append(self._arm_job(
                        "sens", "ttl", arm=arm, n=int(s.get("sens.N")), k=int(s.get("sens.K")),
                        level=level, payload=s.get("sens.payload_bytes"),
                        n_trials=int(s.get("sens.n_trials")), footprint=fp,
                        ttl_factor=float(factor), tag_extra=f"_ttl{_fmt(float(factor))}"))

    # -- the learned score's FerrySim campaign (Run Guide 2.8) --------------- #

    def _rl_gate(self) -> List[str]:
        """Every RL stage waits for the re-pin (Run Guide 2.8, go-ahead item 5)."""
        if self.s.get("rl.repinned", False) is not True:
            return ["rl.repinned (set true after the re-pin commit)"]
        return []

    def _ckpt_root(self) -> str:
        return f"{self.root}/checkpoints"

    def _train(self, stage: str, study: str, *, kind: str, family: str, gamma: float,
               seed: int, tag: Optional[str] = None, ablation: Optional[str] = None,
               extra: Sequence[str] = ()) -> Job:
        s = self.s
        own_tag = ablation or tag
        ckpt = f"{self._ckpt_root()}/{study}/{own_tag}/g{float(gamma):g}_s{seed}.npz"
        args = FERRYSIM + ["train", "--kind", kind, "--family", family, "--study", study,
                           "--gamma", _fmt(float(gamma)), "--seed", str(seed),
                           "--root", self._ckpt_root()]
        args += ["--ablation", ablation] if ablation else ["--tag", str(tag)]
        val = s.get(f"rl.val_episodes.{family}", None)
        if val is not None:
            args += ["--val-episodes", str(int(val))]
        # FerrySim's own defaults (10,000 episodes, a validation every 1,000) unless set;
        # a smaller count is for testing the launcher, never for a campaign checkpoint.
        for key, flag in (("rl.episodes", "--episodes"), ("rl.eval_every", "--eval-every")):
            value = s.get(key, None)
            if value is not None:
                args += [flag, str(int(value))]
        args += list(extra)
        job = Job(stage, study, f"{study}/{own_tag}/g{float(gamma):g}_s{seed}", args, ckpt,
                  "rl-train", mem_gb=float(s.get("rl.gb_per_training")))
        job.blocked = self._rl_gate()
        return job

    def _rl_tool(self, stage: str, study: str, name: str, args: List[str], out: str,
                 deps: Sequence[str] = (), workers: int = 1) -> Job:
        episodes = self.s.get("rl.eval_episodes", None)   # evaluate's default: 1,000 per cell
        if episodes is not None and args[:1] == ["evaluate"]:
            args = args + ["--episodes", str(int(episodes))]
        job = Job(stage, study, f"{study}/{name}", FERRYSIM + args, out, "tool",
                  mem_gb=float(self.s.get("rl.gb_per_training")) * (1 + workers),
                  deps=list(deps), slots=workers)
        job.blocked = self._rl_gate()
        return job

    def _headroom(self) -> str:
        return f"{self.root}/rl/headroom/headroom.json"

    def _verdict_inputs(self, job: Job) -> None:
        """A report reads epsilon from the headroom report, which rl-headroom writes."""
        if not (REPO / self._headroom()).exists():
            job.blocked = sorted(set(job.blocked) | {"the rl-headroom stage's report"})

    @staticmethod
    def _gtag(gamma: float) -> str:
        return f"g{round(float(gamma) * 100)}"

    def rl_headroom(self) -> None:
        s = self.s
        w = int(s.get("rl.workers"))
        self.jobs.append(self._rl_tool(
            "rl-headroom", "headroom", "headroom",
            ["headroom", "--episodes", str(int(s.get("rl.headroom_episodes"))),
             "--workers", str(w), "--out", self._headroom()],
            self._headroom(), workers=w))

    def rl_calibrate(self) -> None:
        s = self.s
        w = int(s.get("rl.workers"))
        for fam in s.get("rl.calibration_families") or []:
            study = f"5.5-calibration-{fam}"
            names = []
            for g in s.get("rl.calibration_gammas") or []:
                for seed in s.get("rl.calibration_seeds") or []:
                    job = self._train("rl-calibrate", study, kind="pair_q", family=fam,
                                      gamma=g, seed=int(seed), tag=self._gtag(g))
                    self.jobs.append(job)
                    names.append(job.name)
            cells = (["--cells", "cln-n12-120", "cln-n12-180"] if fam == "clean" else [])
            ev = f"{self.root}/rl/calibration/{fam}_evaluation.json"
            evaluate = self._rl_tool(
                "rl-calibrate", study, "evaluate",
                ["evaluate", "--checkpoints", f"{self._ckpt_root()}/{study}", *cells,
                 "--workers", str(w), "--out", ev], ev, deps=names, workers=w)
            self.jobs.append(evaluate)
            verdict = f"{self.root}/rl/calibration/{fam}_verdict.json"
            report = self._rl_tool(
                "rl-calibrate", study, "report",
                ["report", "--evaluation", ev, "--headroom", self._headroom(), *cells,
                 "--out", verdict], verdict, deps=[evaluate.name])
            self._verdict_inputs(report)
            self.jobs.append(report)

    def rl_sweep(self) -> None:
        s = self.s
        w = int(s.get("rl.workers"))
        fam = str(s.get("rl.family"))
        study = f"5.5-{fam}"
        names = []
        for g in s.get("rl.sweep_gammas") or []:
            for seed in s.get("rl.sweep_seeds") or []:
                job = self._train("rl-sweep", study, kind="pair_q", family=fam, gamma=g,
                                  seed=int(seed), tag=self._gtag(g))
                self.jobs.append(job)
                names.append(job.name)
        ev = f"{self.root}/rl/s55/evaluation.json"
        evaluate = self._rl_tool(
            "rl-sweep", study, "evaluate",
            ["evaluate", "--checkpoints", f"{self._ckpt_root()}/{study}", "--record",
             "--workers", str(w), "--out", ev], ev, deps=names, workers=w)
        self.jobs.append(evaluate)
        verdict = f"{self.root}/rl/s55/verdict.json"
        report = self._rl_tool(
            "rl-sweep", study, "report",
            ["report", "--evaluation", ev, "--headroom", self._headroom(), "--out", verdict],
            verdict, deps=[evaluate.name])
        self._verdict_inputs(report)
        self.jobs.append(report)

    def rl_e3(self) -> None:
        s = self.s
        w = int(s.get("rl.workers"))
        before = len(s.missing)
        fam = str(s.get("rl.e3.family"))
        gamma = s.get("rl.e3.gamma")
        settings = s.get("rl.e3.settings")
        missing = sorted(set(s.missing[before:]))
        del s.missing[before:]
        if settings not in (None, "pair", "chen"):
            missing.append("rl.e3.settings (\"pair\" or \"chen\")")
        extra = ["--lr", "5e-4", "--epsilon-start", "1.0"] if settings == "chen" else []
        study = "5.3-e3"
        names = []
        for seed in s.get("rl.e3.seeds") or []:
            job = self._train("rl-e3", study, kind="chen_dqn", family=fam,
                              gamma=float(gamma or 0.0), seed=int(seed), tag="e3", extra=extra)
            job.blocked = sorted(set(job.blocked) | set(missing))
            self.jobs.append(job)
            names.append(job.name)
        ev = f"{self.root}/rl/e3/evaluation.json"
        evaluate = self._rl_tool(
            "rl-e3", study, "evaluate",
            ["evaluate", "--checkpoints", f"{self._ckpt_root()}/{study}", "--reward", "bytes",
             "--record", "--workers", str(w), "--out", ev], ev, deps=names, workers=w)
        evaluate.blocked = sorted(set(evaluate.blocked) | set(missing))
        self.jobs.append(evaluate)

    def rl_s57(self) -> None:
        s = self.s
        w = int(s.get("rl.workers"))
        fam = str(s.get("rl.family"))
        before = len(s.missing)
        gamma = s.get("rl.gamma_star")
        keep = s.get("rl.keep_learned")
        missing = sorted(set(s.missing[before:]))
        del s.missing[before:]
        if keep is False:
            print("rl-s57: the 5.5 verdict did not keep the learned score (rl.keep_learned = "
                  "false); 5.7 then compares FX-dwell and FX-cov, which are not built")
            return
        study = f"5.7-{fam}"
        g = float(gamma or 0.0)

        def add(job: Job) -> Job:
            job.blocked = sorted(set(job.blocked) | set(missing))
            self.jobs.append(job)
            return job

        groups: Dict[str, List[str]] = {"hand": [], "dwell": [], "cov": []}
        for seed in s.get("rl.s57.seeds") or []:
            groups["hand"].append(add(self._train(
                "rl-s57", study, kind="pair_q", family=fam, gamma=g, seed=int(seed),
                tag="hand", extra=["--reward", "hand"])).name)
            for ab in ("dwell", "cov"):
                groups[ab].append(add(self._train(
                    "rl-s57", study, kind="pair_q", family=fam, gamma=g, seed=int(seed),
                    ablation=ab)).name)
        ev = f"{self.root}/rl/s57"
        add(self._rl_tool("rl-s57", study, "evaluate-hand",
                          ["evaluate", "--checkpoints", f"{self._ckpt_root()}/{study}/hand",
                           "--reward", "hand", "--record", "--workers", str(w),
                           "--out", f"{ev}/hand_evaluation.json"],
                          f"{ev}/hand_evaluation.json", deps=groups["hand"], workers=w))
        add(self._rl_tool("rl-s57", study, "evaluate-ablations",
                          ["evaluate", "--checkpoints", f"{self._ckpt_root()}/{study}/dwell",
                           f"{self._ckpt_root()}/{study}/cov", "--no-references", "--record",
                           "--workers", str(w), "--out", f"{ev}/ablations_evaluation.json"],
                          f"{ev}/ablations_evaluation.json",
                          deps=groups["dwell"] + groups["cov"], workers=w))
        # The reward grid: one tag per (c_t, c_cov) point that no arm flies.
        default = (0.1, 1.0)
        for c_t in s.get("rl.s57.grid_c_t") or []:
            for c_cov in s.get("rl.s57.grid_c_cov") or []:
                if (float(c_t), float(c_cov)) == default:
                    continue   # the derived reward's own weights: 5.5's checkpoints
                tag = f"ct{float(c_t):g}-cc{float(c_cov):g}"
                names = [add(self._train(
                    "rl-s57", study, kind="pair_q", family=fam, gamma=g, seed=int(seed),
                    tag=tag, extra=["--c-t", _fmt(float(c_t)), "--c-cov", _fmt(float(c_cov))])).name
                    for seed in s.get("rl.s57.grid_seeds") or []]
                add(self._rl_tool("rl-s57", study, f"evaluate-{tag}",
                                  ["evaluate", "--checkpoints", f"{self._ckpt_root()}/{study}/{tag}",
                                   "--c-t", _fmt(float(c_t)), "--c-cov", _fmt(float(c_cov)),
                                   "--workers", str(w), "--out", f"{ev}/{tag}_evaluation.json"],
                                  f"{ev}/{tag}_evaluation.json", deps=names, workers=w))


def build(stage: str, s: Settings, studies: Optional[Sequence[str]] = None,
          root: str = "results/exp5") -> List[Job]:
    b = Builder(s, root)
    getattr(b, stage.replace("-", "_"))()
    jobs = b.jobs
    if studies:
        jobs = [j for j in jobs if j.study in studies]
    jobs = dedupe(jobs)
    if stage in REUSES_BATCH1:
        reuse_batch1(jobs, s, root)
    return jobs


def reuse_batch1(jobs: List[Job], s: Settings, root: str) -> None:
    """A later batch's cell that batch 1 already flies (the same arguments but
    --csv and --n-trials) is not flown again: with no more trials than batch 1
    it reads batch 1's CSV; with more, it writes the extra trials into that same
    CSV (the runner skips the rows there), so every trial of the cell stays in
    one file with the same seeds."""
    b1 = Builder(Settings(s.data), root)
    b1.batch1()
    primary = {_key(j): j for j in dedupe(b1.jobs)
               if j.kind == "runner" and not j.blocked and j.alias_of is None}
    for j in jobs:
        if j.kind != "runner" or j.blocked or j.alias_of is not None:
            continue
        p = primary.get(_key(j))
        if p is None:
            continue
        if (j.trials or 0) <= (p.trials or 0):
            j.alias_of = f"batch1:{p.name}"
            j.out = p.out
        else:
            j.args[j.args.index("--csv") + 1] = p.out
            j.out = p.out
            j.aliases.append(f"extends batch1:{p.name} ({p.trials} -> {j.trials} trials)")


def _key(job: Job) -> Tuple[str, ...]:
    """A runner job's arguments without --csv and --n-trials."""
    args, out, skip = job.args, [], 0
    for a in args:
        if skip:
            skip -= 1
            continue
        if a in ("--csv", "--n-trials"):
            skip = 1
            continue
        out.append(a)
    return tuple(out)


def dedupe(jobs: List[Job]) -> List[Job]:
    """Shared cells run once, at the largest --n-trials."""
    groups: Dict[Tuple[str, ...], List[Job]] = collections.OrderedDict()
    for j in jobs:
        if j.kind != "runner" or j.blocked:
            groups[(j.name,)] = [j]
            continue
        groups.setdefault(_key(j), []).append(j)
    result = []
    for members in groups.values():
        primary = max(members, key=lambda j: (j.trials or 0))
        for j in members:
            if j is not primary:
                j.alias_of = primary.name
                primary.aliases.append(f"{j.name} (first {j.trials} trials)")
        result.extend(members)
    return result


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def interpreter_problem() -> Optional[str]:
    if sys.prefix != sys.base_prefix:
        return ("this is a venv interpreter (" + sys.executable + "). On Windows its "
                "python.exe is a launcher stub that hides each process behind a second "
                "one: run through ..\\py311.cmd (the conda environment ferry311).")
    return None


def dataset_fingerprint() -> Dict[str, Any]:
    env = os.environ.get("HERMES_CICIOT_DIR")
    candidates = ([Path(env)] if env else []) + [REPO.parent / "datasets" / "CICIOT2023",
                                                 REPO / "datasets" / "CICIOT2023"]
    for d in candidates:
        files = sorted(d.glob("*.csv")) if d.is_dir() else []
        if files:
            h = hashlib.sha256()
            total = 0
            for f in files:
                size = f.stat().st_size
                total += size
                h.update(f"{f.name}\t{size}\n".encode())
            return {"dir": str(d), "csv_files": len(files), "bytes": total,
                    "names_and_sizes_sha256": h.hexdigest()}
    return {"dir": None, "csv_files": 0}


def long_paths_enabled() -> bool:
    if os.name != "nt":
        return True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


def deepest_path(job: Job) -> int:
    if job.kind != "runner":
        return len(str(REPO / job.out))
    stem = (REPO / job.out).with_suffix("")
    return len(str(stem)) + len("_traces/") + TRACE_DIR_CHARS + 1 + TRACE_FILE_CHARS


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8").stdout


def provenance() -> Dict[str, Any]:
    status = git("status", "--porcelain", "--untracked-files=normal")
    code = git("status", "--porcelain", "--untracked-files=normal", "--", "hermes", "experiments")
    return {"commit": git("rev-parse", "HEAD").strip(),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "code_dirty": [l for l in code.splitlines() if l.strip()],
            "tree_status": [l for l in status.splitlines() if l.strip()]}


def environment(threads: Dict[str, str]) -> Dict[str, Any]:
    from importlib import metadata
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tensorflow",
                 "keras", "protobuf", "flwr", "psutil"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    host: Dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(),
                            "processor": platform.processor(), "logical_cpus": os.cpu_count()}
    try:
        import psutil
        host["physical_cpus"] = psutil.cpu_count(logical=False)
        host["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return {"python": sys.version, "executable": sys.executable, "packages": versions,
            "threads": threads, "host": host}


def validate(jobs: List[Job]) -> Dict[str, Dict[str, Any]]:
    """Each runner job's arguments through runner_main's usage checks (no trial)."""
    todo = [j for j in jobs if j.kind in ("runner", "rl-train") and not j.blocked
            and j.alias_of is None and not done(j)]
    if not todo:
        return {}
    payload = json.dumps([{"name": j.name, "args": j.args, "kind": j.kind,
                           "out": str((REPO / j.out).resolve())} for j in todo])
    proc = subprocess.run([sys.executable, str(Path(__file__)), "_validate"], cwd=REPO,
                          input=payload, capture_output=True, text=True, encoding="utf-8")
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        sys.exit("validation did not complete:\n" + proc.stderr[-3000:])


def _validate_child() -> int:
    """Runs in a subprocess: runner_main.main per job with TrialRunner stubbed."""
    import contextlib
    import io
    import logging
    import tempfile

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    jobs = json.loads(sys.stdin.read())
    from experiments.exp4 import runner_main as RM

    seen: Dict[str, Any] = {}

    class _NoTrials:
        def __init__(self, grid, **_):
            seen["trials"] = grid.total()

        def run(self, _fn):
            return 0

    RM.TrialRunner = _NoTrials
    results = {}
    with tempfile.TemporaryDirectory(prefix="exp5_validate_") as tmp:
        for job in jobs:
            if job["kind"] == "rl-train":
                results[job["name"]] = _validate_training(job)
                continue
            argv = list(job["args"][len(RUNNER):])
            argv[argv.index("--csv") + 1] = str(Path(tmp) / "v.csv")
            seen.clear()
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    logging.disable(logging.CRITICAL)
                    RM.main(argv)
                results[job["name"]] = {"ok": True, "trials": seen.get("trials")}
            except SystemExit as e:
                lines = [l for l in err.getvalue().splitlines() if l.strip()]
                results[job["name"]] = {"ok": False,
                                        "error": lines[-1] if lines else f"exit {e.code}"}
            except Exception as e:  # noqa: BLE001 - reported per job
                results[job["name"]] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(results))
    return 0


def _validate_training(job: Dict[str, Any]) -> Dict[str, Any]:
    """A training job through `ferrysim train`'s own checks, up to the training:
    the parser, the run's spec (tag, reward, plan), the tree and the path's
    refusals; then the path it would save to must be the one the launcher
    skips on resume."""
    import contextlib
    import io

    from experiments.ferrysim import __main__ as FM

    argv = list(job["args"][len(FERRYSIM):])
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            ap = FM.parser()
            args = ap.parse_args(argv)
            spec = FM._spec(args, args.gamma, args.seed, FM._run_plan(args, ap))
            path = FM._path(args, spec)
            FM._tree(args, ap)
    except SystemExit as e:
        lines = [l for l in err.getvalue().splitlines() if l.strip()]
        return {"ok": False, "error": lines[-1] if lines else f"exit {e.code}"}
    except (TypeError, ValueError) as e:
        return {"ok": False, "error": str(e)}
    if Path(path).resolve() != Path(job["out"]):
        return {"ok": False, "error": f"saves to {path}, not {job['out']}"}
    return {"ok": True, "trials": None}


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #

def csv_rows(path: Path) -> Tuple[int, int]:
    """(rows, rows whose status is not ok) of a runner CSV; (0, 0) if absent."""
    if not path.exists():
        return 0, 0
    rows = bad = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            if row.get("status", "ok") != "ok":
                bad += 1
    return rows, bad


def done(job: Job) -> bool:
    path = REPO / job.out
    if job.kind == "tool":
        return path.exists()
    if job.kind == "rl-train":
        # A training saves its arrays and then the manifest beside them.
        return path.exists() and path.with_suffix(".json").exists()
    if job.kind == "score":
        # Current: newer than its trial CSV, and scored under these arguments.
        src = REPO / job.args[job.args.index("--status-csv") + 1]
        side = path.with_suffix(path.suffix + ".argv.json")
        return (path.exists() and src.exists() and side.exists()
                and path.stat().st_mtime >= src.stat().st_mtime
                and json.loads(side.read_text(encoding="utf-8")) == job.args)
    rows, _ = csv_rows(path)
    return rows >= (job.trials or 0)


def seconds_per_trial(n: int) -> float:
    """Rough: probes of 5 Oct 2026 (19/29/42 s at N = 6/12/24, synthetic data)
    plus about 6 s to load the real data."""
    return 25.0 + 1.0 * n


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_plan(stage: str, s: Settings, jobs: List[Job], show: bool,
             checked: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    b = Builder(s)
    by_study: Dict[str, List[Job]] = collections.OrderedDict()
    for j in jobs:
        by_study.setdefault(j.study, []).append(j)
    total_trials = 0
    hours = 0.0
    blocked = set()
    print(f"Stage {stage}: {len(jobs)} jobs")
    for study, members in by_study.items():
        primary = [j for j in members if j.alias_of is None]
        trials = sum(j.trials or 0 for j in primary if j.kind == "runner")
        total_trials += trials
        print(f"\n  {study}: {len(members)} jobs, {len(primary)} to run, {trials} trials")
        for j in members:
            state = "done" if done(j) else ""
            note = ""
            if j.blocked:
                note = "BLOCKED: unset " + ", ".join(j.blocked)
                blocked.update(j.blocked)
            elif j.alias_of:
                note = f"= {j.alias_of}"
            elif any(a.startswith("extends") for a in j.aliases):
                note = next(a for a in j.aliases if a.startswith("extends"))
            elif checked and j.name in checked and not checked[j.name]["ok"]:
                note = "INVALID: " + checked[j.name]["error"]
            if j.kind == "runner" and not j.alias_of:
                conc = b.concurrency(j.n) if j.n else 1
                hours += (j.trials or 0) * seconds_per_trial(j.n) / conc / 3600
            if j.kind == "rl-train" and not done(j):
                # Rough: 1-2 h per training on one core (5 Oct 2026 probe), side by side.
                hours += 1.5 / max(1, int(s.get("rl.max_jobs", 1)))
            size = {"runner": f"{j.trials} trials", "rl-train": "training"}.get(j.kind, "tool")
            print(f"    {j.name:34s} {size:>10s} {j.mem_gb:6.1f} GB  {state:4s} {note}")
            if show and not j.alias_of:
                print("      " + subprocess.list2cmdline(["python"] + j.args))
    print(f"\n  total: {total_trials} trials to run; rough compute {hours:.1f} h "
          f"at the scheduler's concurrency")
    deepest = max((deepest_path(j) for j in jobs), default=0)
    if deepest:
        limit = "no limit (long paths on)" if long_paths_enabled() else f"limit {MAX_PATH}"
        print(f"  deepest kept-trace path: about {deepest} characters ({limit})")
    if blocked:
        print("  unset in params.toml: " + ", ".join(sorted(blocked)))
    return 0


def cmd_validate(jobs: List[Job]) -> Dict[str, Dict[str, Any]]:
    checked = validate(jobs)
    bad = {k: v for k, v in checked.items() if not v["ok"]}
    for j in jobs:
        if j.name in checked and checked[j.name]["ok"] and checked[j.name]["trials"] is not None:
            j.trials = int(checked[j.name]["trials"])
    print(f"validated {len(checked)} jobs: {len(checked) - len(bad)} ok, {len(bad)} refused")
    for name, v in bad.items():
        print(f"  {name}: {v['error']}")
    return checked


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.kill()


def cmd_run(stage: str, s: Settings, params_text: str, jobs: List[Job], *,
            allow_dirty: bool, skip_blocked: bool, yes: bool, max_jobs: Optional[int],
            out_root: str = "results/exp5") -> int:
    problem = interpreter_problem()
    if problem:
        sys.exit(problem)
    blocked = [j for j in jobs if j.blocked]
    if blocked and not skip_blocked:
        names = sorted({b for j in blocked for b in j.blocked})
        sys.exit(f"{len(blocked)} jobs need unset settings ({', '.join(names)}); set them in "
                 f"params.toml, or pass --skip-blocked to run the rest")
    jobs = [j for j in jobs if not j.blocked]
    prov = provenance()
    if prov["code_dirty"] and not allow_dirty:
        sys.exit("hermes/ or experiments/ has uncommitted changes:\n  "
                 + "\n  ".join(prov["code_dirty"]) + "\ncommit them, or pass --allow-dirty")
    data = dataset_fingerprint()
    needs_data = stage == "ttl" or any("--real-model" in j.args for j in jobs)
    if needs_data and not data["csv_files"]:
        sys.exit("CICIoT2023 not found (../datasets/CICIOT2023 or HERMES_CICIOT_DIR): the "
                 "loader would fall back to a synthetic task")
    if not long_paths_enabled():
        deep = [j for j in jobs if deepest_path(j) > MAX_PATH]
        if deep:
            sys.exit(f"{len(deep)} jobs would write paths over {MAX_PATH} characters with "
                     f"Windows long paths off, e.g. {deep[0].out}")
    if any("--footprint-probe" in j.args for j in jobs):
        try:
            import psutil  # noqa: F401
        except ImportError:
            sys.exit("--footprint-probe needs psutil")

    if any(j.kind in ("runner", "rl-train") for j in jobs):
        checked = cmd_validate(jobs)
        if any(not v["ok"] for v in checked.values()):
            sys.exit("the runner refuses the jobs above; nothing ran")

    # Resume guard: a CSV written under other arguments is refused.
    for j in jobs:
        if j.alias_of or j.kind == "score":
            continue
        side = (REPO / j.out).with_suffix((REPO / j.out).suffix + ".argv.json")
        if side.exists():
            before = json.loads(side.read_text(encoding="utf-8"))
            if _key_args(before) != _key_args(j.args):
                changed = sorted(set(_key_args(j.args)) ^ set(_key_args(before)))
                sys.exit(f"{j.out} was written under other arguments ({side.name}; they "
                         f"differ in {' '.join(changed[:6])}). A run with changed "
                         f"settings writes to a fresh place: give it --out-root.")

    todo = [j for j in jobs if j.alias_of is None and not done(j)]
    runner_trials = sum(j.trials or 0 for j in todo if j.kind == "runner")
    print(f"\nstage {stage}: {len(todo)} jobs to run ({runner_trials} trials), "
          f"{len(jobs) - len(todo)} done or shared; commit {prov['commit'][:10]}")
    if not todo:
        return 0
    if not yes:
        if input("start? [y/N] ").strip().lower() not in ("y", "yes"):
            print("nothing started")
            return 1
    lock = acquire_lock(REPO / out_root, stage)
    try:
        return _run_jobs(stage, s, params_text, jobs, todo, prov, data, max_jobs, out_root)
    finally:
        lock.unlink(missing_ok=True)


def acquire_lock(out_root: Path, stage: str) -> Path:
    """results/exp5/.launcher.lock, held while a stage runs; a stale one is replaced."""
    out_root.mkdir(parents=True, exist_ok=True)
    lock = out_root / ".launcher.lock"
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                held = {}
            pid = int(held.get("pid", -1))
            alive = False
            try:
                import psutil
                alive = pid > 0 and psutil.pid_exists(pid)
            except ImportError:
                alive = True
            if alive:
                sys.exit(f"stage {held.get('stage')} is running (pid {pid}, since "
                         f"{held.get('since')}): one stage at a time, so stack trials and "
                         f"trainings never share the cores. Wait for it, or stop it.")
            lock.unlink(missing_ok=True)   # stale: its launcher is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "stage": stage,
                       "since": dt.datetime.now().isoformat(timespec="seconds")}, f)
        return lock
    sys.exit(f"could not take {lock}")


def _run_jobs(stage: str, s: Settings, params_text: str, jobs: List[Job], todo: List[Job],
              prov: Dict[str, Any], data: Dict[str, Any], max_jobs: Optional[int],
              out_root: str) -> int:
    root = (REPO / out_root / STAGE_DIR[stage]).resolve()
    (root / "_launcher").mkdir(parents=True, exist_ok=True)
    (root / "_logs").mkdir(parents=True, exist_ok=True)
    threads = {k: str(v) for k, v in (s.get("machine.threads") or {}).items()}
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = root / "_launcher" / f"manifest_{stamp}.json"
    manifest = {
        "stage": stage, "started": dt.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv, "git": prov, "environment": environment(threads), "dataset": data,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "params_sha256": hashlib.sha256(params_text.encode()).hexdigest(),
        "params_toml": params_text,
        # The command line's levers and --set changes, and the settings as changed.
        "overrides": s.overrides,
        "settings": json.loads(json.dumps(s.data, default=str)),
        "jobs": [asdict(j) for j in jobs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    events = (root / "_launcher" / "jobs.jsonl").open("a", encoding="utf-8")

    def log_event(**kw):
        kw["at"] = dt.datetime.now().isoformat(timespec="seconds")
        events.write(json.dumps(kw) + "\n")
        events.flush()

    env = dict(os.environ)
    env.update(threads)
    env["PYTHONIOENCODING"] = "utf-8"
    default_cap = "rl.max_jobs" if stage in RL_STAGES else "machine.max_jobs"
    cap_jobs = max_jobs or int(s.get(default_cap))
    mem_budget = float(s.get("machine.mem_budget_gb"))
    cap_devices = int(s.get("machine.max_device_processes"))
    pending = sorted(todo, key=lambda j: -j.mem_gb)   # largest first: no long tail
    running: Dict[str, Tuple[Job, subprocess.Popen, float, Any]] = {}
    failed: List[str] = []
    skipped: List[str] = []
    todo_names = {j.name for j in todo}
    finished_ok = {j.name for j in jobs if j.name not in todo_names}
    try:
        while pending or running:
            used = sum(r[0].mem_gb for r in running.values())
            devices = sum(r[0].n for r in running.values() if r[0].kind == "runner")
            slots = sum(r[0].slots for r in running.values())
            for j in list(pending):
                if any(d in failed or d in skipped for d in j.deps):
                    pending.remove(j)
                    skipped.append(j.name)
                    log_event(event="skipped", job=j.name, reason="a job it waits for failed")
                    print(f"[{time.strftime('%H:%M:%S')}] skip {j.name}: a job it waits for failed")
                    continue
                if any(d not in finished_ok for d in j.deps):
                    continue
                if running and slots + j.slots > cap_jobs:
                    continue
                if running and used + j.mem_gb > mem_budget:
                    continue
                if running and j.kind == "runner" and devices + j.n > cap_devices:
                    continue
                out = REPO / j.out
                out.parent.mkdir(parents=True, exist_ok=True)
                if j.kind == "score":
                    out.unlink(missing_ok=True)    # never an old score under new arguments
                if j.kind in ("runner", "score"):
                    side = out.with_suffix(out.suffix + ".argv.json")
                    side.write_text(json.dumps(j.args, indent=1), encoding="utf-8")
                logf = (root / "_logs" / (j.name.replace("/", "__") + ".log")).open(
                    "a", encoding="utf-8")
                logf.write(f"\n=== {dt.datetime.now().isoformat(timespec='seconds')} "
                           f"{subprocess.list2cmdline(['python'] + j.args)}\n")
                logf.flush()
                flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                proc = subprocess.Popen([sys.executable] + j.args, cwd=REPO, env=env,
                                        stdout=logf, stderr=subprocess.STDOUT,
                                        creationflags=flags)
                running[j.name] = (j, proc, time.monotonic(), logf)
                pending.remove(j)
                used += j.mem_gb
                devices += j.n if j.kind == "runner" else 0
                slots += j.slots
                log_event(event="start", job=j.name, pid=proc.pid)
                print(f"[{time.strftime('%H:%M:%S')}] start {j.name} "
                      f"({len(running)} running, {len(pending)} waiting)")
            if pending and not running:
                # Nothing runs and nothing could start: what is left waits for a job
                # that is not in this run (left out with --study, or blocked).
                for j in pending:
                    skipped.append(j.name)
                    log_event(event="skipped", job=j.name, reason="waits for a job not run")
                    print(f"[{time.strftime('%H:%M:%S')}] skip {j.name}: it waits for "
                          f"{', '.join(d for d in j.deps if d not in finished_ok)}")
                pending.clear()
                break
            time.sleep(2.0)
            for name, (j, proc, t0, logf) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                logf.close()
                wall = time.monotonic() - t0
                del running[name]
                rows, bad = csv_rows(REPO / j.out) if j.kind == "runner" else (None, None)
                log_event(event="end", job=name, rc=rc, wall_s=round(wall, 1),
                          rows=rows, rows_not_ok=bad)
                mark = "ok" if rc == 0 else f"FAILED rc={rc}"
                extra = f", {rows} rows ({bad} not ok)" if rows is not None else ""
                print(f"[{time.strftime('%H:%M:%S')}] {mark} {name} in {wall / 60:.1f} min{extra}")
                if rc != 0:
                    failed.append(name)
                else:
                    finished_ok.add(name)
    except KeyboardInterrupt:
        print("\ninterrupted: stopping the running jobs (each CSV resumes on the next run)")
        for name, (j, proc, _, logf) in running.items():
            _kill_tree(proc)
            logf.close()
            log_event(event="interrupted", job=name)
        events.close()
        return 130
    events.close()
    manifest["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    manifest["failed"] = failed
    manifest["skipped"] = skipped
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    ok = len(todo) - len(failed) - len(skipped)
    print(f"\nstage {stage}: {ok} jobs ok, {len(failed)} failed, {len(skipped)} skipped; "
          f"manifest {manifest_path}")
    for name in failed:
        print(f"  failed: {name} (log in {root / '_logs'})")
    return 1 if failed or skipped else 0


def _key_args(args: Sequence[str]) -> Tuple[str, ...]:
    return _key(Job("", "", "", list(args), "", "runner"))


def cmd_status(stage: str, jobs: List[Job]) -> int:
    by_study: Dict[str, List[Job]] = collections.OrderedDict()
    for j in jobs:
        by_study.setdefault(j.study, []).append(j)
    for study, members in by_study.items():
        primary = [j for j in members if j.alias_of is None and not j.blocked]
        n_done = sum(1 for j in primary if done(j))
        rows = bad = expected = 0
        for j in primary:
            if j.kind == "runner":
                r, b = csv_rows(REPO / j.out)
                rows, bad, expected = rows + r, bad + b, expected + (j.trials or 0)
        blocked = sum(1 for j in members if j.blocked)
        line = f"{study:8s} jobs {n_done}/{len(primary)} done"
        if expected:
            line += f", trials {rows}/{expected} ({bad} not ok)"
        if blocked:
            line += f", {blocked} blocked"
        print(line)
    return 0


def _round_to(x: float, step: float) -> float:
    return max(step, step * round(x / step))


def trial_best_accuracy(traces: Path) -> Dict[int, float]:
    """trial_index -> the highest accuracy any evaluation after the first (the
    initial model) reached, from each kept trial's cluster trace: a trial
    reaches tau exactly when this is at least tau, as the scorer counts it."""
    out: Dict[int, float] = {}
    if not traces.is_dir():
        return out
    for trial in traces.iterdir():
        m = re.search(r"__t(\d+)__s", trial.name)
        log = next(trial.glob("cluster-*.jsonl"), None) if trial.is_dir() else None
        if m is None or log is None:
            continue
        best = None
        for line in log.read_text(encoding="utf-8").splitlines():
            if '"model_eval"' not in line:
                continue
            e = json.loads(line)
            if int(e.get("cluster_round", 0)) == 0:
                continue
            best = float(e["accuracy"]) if best is None else max(best, float(e["accuracy"]))
        if best is not None:
            out[int(m.group(1))] = best
    return out


def pilot_tau(best_by_n: Dict[int, List[float]], share: float, step: float
              ) -> Tuple[Optional[float], Dict[int, float]]:
    """The largest tau on the grid (multiples of ``step``) that at least ``share``
    of the trials reach, per N, and the smallest of those over N: the tau every
    N's knee reaches in at least that share of its trials."""
    per_n: Dict[int, float] = {}
    for n, values in best_by_n.items():
        if not values:
            continue
        k = math.ceil(share * len(values))              # trials that must reach tau
        kth = sorted(values, reverse=True)[k - 1]       # tau <= this keeps k of them
        per_n[n] = round(math.floor(round(kth / step, 9)) * step, 6)
    return (min(per_n.values()) if per_n else None), per_n


def pick_spread(levels: Sequence[Tuple[float, Optional[float]]], band: Sequence[float]
                ) -> Optional[float]:
    """p512's rule: of (median_s, mean not-ready share) per level, the median whose
    share lies in ``band``, the one nearest the band's middle if several."""
    lo, hi = float(band[0]), float(band[1])
    inside = [(abs(share - (lo + hi) / 2), median) for median, share in levels
              if share is not None and lo <= share <= hi]
    return min(inside)[1] if inside else None


def p512_levels(s: Settings, jobs: List[Job], problems: List[str]) -> Optional[str]:
    """5.12's training-time levels from pilot3's p512 trials, as the TOML inline
    table pilot_outputs.train_levels takes (None while there is no pick)."""
    p512 = [j for j in jobs if j.study == "p512" and not j.blocked]
    if not p512:
        return None
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from experiments.analysis.traces_scorer import load_status_csv, score_traces
    band = s.get("p512.not_ready_band")
    print(f"  p512: share of H1's Pass-1 contacts that found no update ready, by median "
          f"fit time (band {band[0]:g}-{band[1]:g})")
    levels: List[Tuple[float, Optional[float]]] = []
    for j in p512:
        median = float(j.args[j.args.index("--train-time-s") + 1])
        csv_path, traces = REPO / j.out, REPO / (j.out[:-4] + "_traces")
        shares: List[float] = []
        if csv_path.exists() and traces.is_dir():
            for sc in score_traces(traces, compute_columns=True,
                                   status_csv=load_status_csv(csv_path)):
                if sc.compute is not None and sc.compute.not_ready_share is not None:
                    shares.append(float(sc.compute.not_ready_share))
        share = sum(shares) / len(shares) if shares else None
        levels.append((median, share))
        print(f"    median {median:6.0f} s: " + ("no scored trials" if share is None else
                                                 f"{share:5.1%} not ready (n = {len(shares)})"))
    median = pick_spread(levels, band)
    if median is None:
        problems.append(f"p512: no level's not-ready share lies in {band}; change "
                        f"p512.cycle_factors and run pilot3 again")
        return None
    sigma = float(s.get("p512.sigma"))
    share, factor = float(s.get("p512.straggler_share")), float(s.get("p512.straggler_factor"))
    print(f"    -> spread: median {median:.0f} s, sigma {sigma:g}; stragglers: the same with "
          f"{share:.0%} of devices at {factor:g}x")
    return (f"{{ none = [0.0, 0.0, 0.0, 1.0], spread = [{median:.1f}, {sigma}, 0.0, 1.0], "
            f"stragglers = [{median:.1f}, {sigma}, {share}, {factor}] }}")


#: The trial-CSV columns `report quick` sets beside the recorded run's.
REPRO_COLUMNS = ("final_accuracy", "final_auc", "update_yield", "round_close_rate_kmin1",
                 "rounds_closed", "pass1_contacts_mean", "mission_duration_s_mean")


def _ok_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """(trial_index, seed) -> row, for a trial CSV's ok rows."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["trial_index"], r["seed"]): r for r in csv.DictReader(f)
                if r.get("status", "ok") == "ok"}


def report_quick(s: Settings, jobs: List[Job], recorded_root: str = "results/exp5") -> int:
    """Each reproduced trial beside the recorded batch-1 trial with the same seed.

    quick's jobs are batch 1's own (the same arguments), so trial i of each
    is the same draw: the same layout, shards and channel. Across hosts the
    numbers may still differ in their last bits (the math library) and
    through scheduling (the real processes run over TCP); the report says how
    many trials match to 1e-9 and how far apart the rest are, per column."""
    rec = Builder(Settings(s.data), recorded_root)
    rec.batch1()
    recorded = {_key(j): j for j in dedupe(rec.jobs)
                if j.kind == "runner" and not j.blocked and j.alias_of is None}
    out: Dict[str, Any] = {"recorded_root": recorded_root, "columns": list(REPRO_COLUMNS),
                           "jobs": {}}
    print(f"  each reproduced trial beside the recorded batch-1 trial with the same seed "
          f"(recorded under {recorded_root})")
    for j in jobs:
        if j.kind != "runner":
            continue
        p = recorded.get(_key(j))
        if p is None:
            print(f"\n  {j.name}: batch 1 has no job with these arguments")
            continue
        mine, theirs = _ok_rows(REPO / j.out), _ok_rows(REPO / p.out)
        common = sorted(set(mine) & set(theirs), key=lambda k: int(k[0]))
        same = 0
        cols: Dict[str, Dict[str, Any]] = {}
        for c in REPRO_COLUMNS:
            pairs = []
            for key in common:
                try:
                    pairs.append((float(theirs[key][c]), float(mine[key][c])))
                except (KeyError, ValueError):
                    continue
            if pairs:
                cols[c] = {"recorded_mean": sum(a for a, _ in pairs) / len(pairs),
                           "reproduced_mean": sum(b for _, b in pairs) / len(pairs),
                           "max_abs_diff": max(abs(a - b) for a, b in pairs), "n": len(pairs)}

        def close(a: str, b: str) -> bool:
            try:
                x, y = float(a), float(b)
            except ValueError:
                return a == b
            return abs(x - y) <= 1e-9 * max(1.0, abs(x))
        for key in common:
            if all(close(theirs[key].get(c, ""), mine[key].get(c, "")) for c in REPRO_COLUMNS):
                same += 1
        out["jobs"][j.name] = {"recorded": p.out, "reproduced": j.out, "pairs": len(common),
                               "identical": same, "columns": cols}
        print(f"\n  {j.name}: {len(common)} trials paired with {p.name}, {same} identical "
              f"in every column")
        if not common:
            print(f"    (reproduced rows: {len(mine)} in {j.out}; recorded rows: "
                  f"{len(theirs)} in {p.out})")
        for c, v in cols.items():
            print(f"    {c:24s} recorded {v['recorded_mean']:9.4f}  reproduced "
                  f"{v['reproduced_mean']:9.4f}  largest |difference| {v['max_abs_diff']:.3g}")
    runner = [j for j in jobs if j.kind == "runner"]
    if runner:
        path = (REPO / runner[0].out).parent.parent / "reproduction.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\n  written to {path}; `score quick` compares the arms as batch 1's 5.3 does")
    return 0


def _inline(per_n: Any) -> str:
    """A pilot output as params.toml writes it: an inline table per N
    ({ "6" = 90.0, ... }), or one value (tau = 0.7)."""
    if not isinstance(per_n, dict):
        return f"{per_n:g}" if isinstance(per_n, float) else str(per_n)
    return "{ " + ", ".join(
        f'"{k}" = {v:.1f}' if isinstance(v, float) else f'"{k}" = {v}'
        for k, v in per_n.items()) + " }"


def write_pilot_outputs(path: Path, values: Dict[str, str]) -> List[Tuple[str, str]]:
    """Set keys of params.toml's [pilot_outputs] in place, every other line kept:
    a key's line (set, or commented out as unset) is replaced; a new key goes at
    the table's end. Returns (old, new) per key; refuses a file that would no
    longer parse."""
    raw = path.read_bytes().decode("utf-8-sig")
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)
    start = next((i for i, l in enumerate(lines) if l.strip() == "[pilot_outputs]"), None)
    if start is None:
        sys.exit(f"{path} has no [pilot_outputs] table")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
               len(lines))
    insert_at = end
    while insert_at > start + 1 and (not lines[insert_at - 1].strip()
                                     or lines[insert_at - 1].lstrip().startswith("#")):
        insert_at -= 1
    changes = []
    for key, value in values.items():
        new = f"{key} = {value}"
        pattern = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=")
        hit = next((i for i in range(start + 1, end) if pattern.match(lines[i])), None)
        if hit is None:
            lines.insert(insert_at, new)
            insert_at += 1
            end += 1
            changes.append(("", new))
        else:
            changes.append((lines[hit], new))
            lines[hit] = new
    text = nl.join(lines)
    try:
        parsed = tomllib.loads(text)["pilot_outputs"]
    except (tomllib.TOMLDecodeError, KeyError) as e:
        sys.exit(f"not written: the edited {path.name} would not parse ({e})")
    for key, value in values.items():
        if parsed.get(key) != tomllib.loads(f"v = {value}")["v"]:
            sys.exit(f"not written: {key} would not read back as {value}")
    path.write_bytes(text.encode("utf-8"))
    return changes


def cmd_report(stage: str, s: Settings, jobs: List[Job],
               apply_to: Optional[Path] = None) -> int:
    """A finished pilot's pilot_outputs lines for params.toml, by the pre-registered
    rules; with ``apply_to`` (report --apply), written into that file."""
    out: Dict[str, Any] = {"stage": stage}
    outputs: Dict[str, Any] = collections.OrderedDict()   # key -> {N: value}, or a value
    problems: List[str] = []
    if stage == "ttl":
        ttl = {}
        for j in jobs:
            path = REPO / j.out
            if not path.exists():
                print(f"  {j.name}: no report yet ({j.out})")
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            ttl[str(j.n)] = float(math.ceil(r["ttl_floor_s"]))
            print(f"  N = {j.n:2d}: fit p50 {r['fit_s']['p50']:.2f} s, p95 {r['fit_s']['p95']:.2f} s, "
                  f"max {r['fit_s']['max']:.2f} s over {r['workers']} workers "
                  f"-> TTL {ttl[str(j.n)]:.0f} s")
        out["session_ttl_s"] = ttl
        if ttl:
            outputs["session_ttl_s"] = ttl
    elif stage == "knee":
        metric = str(s.get("knee.metric"))
        share = float(s.get("knee.plateau_share"))
        cells: Dict[Tuple[int, str], Dict[float, List[float]]] = collections.OrderedDict()
        # Beside the knee: the final accuracy, and how many trials reach each fixed
        # tau of the scoring plan within the trial (the highest accuracy after any
        # round, from the kept traces: the scorer's own "reached"). Time to tau is
        # blank in a trial that never gets there. The knees' highest accuracies
        # also give the pilot's tau ([score] pilot_tau_*).
        accuracy: Dict[Tuple[int, str, float], List[float]] = {}
        reached: Dict[Tuple[int, str, float], List[float]] = {}
        taus = s.get("score.tau", [0.82])
        fixed_taus = [float(t) for t in (taus if isinstance(taus, list) else [taus])
                      if t != "pilot"]
        not_ok = 0
        for j in jobs:
            path = REPO / j.out
            if not path.exists() or j.blocked:
                continue
            budget = float(j.args[j.args.index("--mission-budget-s") + 1])
            payload = (j.args[j.args.index("--payload-bytes") + 1]
                       if "--payload-bytes" in j.args else "measured")
            values = cells.setdefault((j.n, payload), {}).setdefault(budget, [])
            accs = accuracy.setdefault((j.n, payload, budget), [])
            highs = reached.setdefault((j.n, payload, budget), [])
            best = trial_best_accuracy(REPO / (j.out[:-4] + "_traces"))
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("status", "ok") != "ok":
                        not_ok += 1
                        continue
                    try:
                        values.append(float(row[metric]))
                    except (KeyError, ValueError):
                        pass
                    try:
                        accs.append(float(row["final_accuracy"]))
                    except (KeyError, ValueError):
                        pass
                    t = int(row.get("trial_index", -1))
                    if t in best:
                        highs.append(best[t])
        knees: Dict[str, Dict[str, float]] = {}
        for (n, payload), by_budget in cells.items():
            means = {b: sum(v) / len(v) for b, v in sorted(by_budget.items()) if v}
            if not means:
                continue
            best = max(means.values())
            knee = min(b for b, m in means.items() if m >= share * best)
            stress = _round_to(knee / 2, 5.0)
            tag = "1mb" if payload == "1000000" else _payload_tag(payload)
            knees.setdefault(tag, {})[str(n)] = knee
            knees.setdefault(tag + "_stress", {})[str(n)] = stress
            print(f"\n  N = {n}, payload {payload}: {metric} by budget "
                  f"(knee: first mean >= {share:.0%} of the best, {best:.3f})")
            for b, v in sorted(by_budget.items()):
                if not v:
                    print(f"    {b:7.0f} s   no ok rows")
                    continue
                m = sum(v) / len(v)
                sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
                half = 1.96 * sd / math.sqrt(len(v))
                mark = "  <- knee" if b == knee else ""
                accs = accuracy.get((n, payload, b), [])
                highs = reached.get((n, payload, b), [])
                acc = f"; final accuracy {sum(accs) / len(accs):.3f}" if accs else ""
                for t in fixed_taus if highs else []:
                    acc += f", {sum(1 for x in highs if x >= t)}/{len(highs)} reach {t:g}"
                print(f"    {b:7.0f} s   {m:6.3f} ± {half:5.3f}  (n = {len(v)}, served share "
                      f"{m / n:5.1%}{acc}){mark}")
            budgets = sorted(by_budget)
            if knee == budgets[-1]:
                problems.append(f"N = {n}, payload {payload}: the knee is the grid's largest "
                                f"budget; the grid may not reach the plateau. Extend it.")
                print("    WARNING: the knee is the grid's largest budget; the grid may not "
                      "reach the plateau. Extend it before using this knee.")
            if knee == budgets[0]:
                problems.append(f"N = {n}, payload {payload}: the knee is the grid's smallest "
                                f"budget. Extend the grid down.")
                print("    WARNING: the knee is the grid's smallest budget; extend the grid down.")
        out["knees"] = knees
        out["rows_not_ok"] = not_ok
        print(f"\n  rows not ok (left out): {not_ok}")
        for tag, key in (("1mb", "knee_s"), ("1mb_stress", "stress_s"),
                         ("meas", "knee_meas_s"), ("meas_stress", "stress_meas_s")):
            if tag in knees:
                outputs[key] = {k: float(v) for k, v in knees[tag].items()}
        # The pilot's tau, by the rule in [score] (decided 5 Oct 2026, before the
        # knee pilot finished): the largest tau on the grid that at least `share`
        # of H1's trials reach at each N's knee at 1 MB, the smallest over N.
        tau_share = float(s.get("score.pilot_tau_share", 0.8))
        tau_step = float(s.get("score.pilot_tau_step", 0.01))
        at_knee = {n: reached.get((n, p, float(knees["1mb"][str(n)])), [])
                   for (n, p) in cells if p == "1000000" and str(n) in knees.get("1mb", {})}
        tau, per_n = pilot_tau(at_knee, tau_share, tau_step)
        if at_knee:
            print(f"\n  the pilot's tau: the largest tau (steps of {tau_step:g}) that at least "
                  f"{tau_share:.0%} of H1's trials reach at each N's knee, the smallest over N")
            for n, vals in at_knee.items():
                print(f"    N = {n:2d}: {len(vals):2d} trials with traces -> "
                      + (f"tau {per_n[n]:.2f}" if n in per_n else "no traces"))
            missing = [n for n, vals in at_knee.items() if not vals]
            if missing:
                problems.append(f"no kept traces at the knee for N = {missing}: the pilot's "
                                f"tau needs every N")
            elif tau is not None:
                print(f"    -> tau = {tau:.2f}")
                outputs["tau"] = tau
    elif stage == "sstar":
        stars = {}
        for j in jobs:
            path = REPO / j.out
            if not path.exists():
                print(f"  {j.name}: no report yet ({j.out})")
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            stars[str(j.n)] = int(r["families"]["F"]["s"])
            print(f"  N = {j.n:2d}: S = {stars[str(j.n)]} at budgets {r['budgets']}")
        out["s_star"] = stars
        if stars:
            outputs["s_star"] = stars
    elif stage == "pilot3":
        print("  p515: H1's served share by interference amplitude (5.15's harsher level)")
        for j in jobs:
            if j.study != "p515":
                continue
            path = REPO / j.out
            amp = j.args[j.args.index("--interference-amp-db") + 1]
            vals: List[float] = []
            if path.exists():
                with path.open(newline="", encoding="utf-8") as f:
                    vals = [float(r["update_yield"]) for r in csv.DictReader(f)
                            if r.get("status", "ok") == "ok" and r.get("update_yield")]
            share = (sum(vals) / len(vals) / j.n) if vals else None
            print(f"    amp {amp:>5} dB: " + ("no ok rows" if share is None else
                                               f"served share {share:5.1%} (n = {len(vals)})"))
        print("  p511c: each scale cell's table (served share by budget) is in its JSON:")
        for j in jobs:
            if j.study == "p511c":
                print(f"    {j.name}: {'written' if (REPO / j.out).exists() else 'not yet'} "
                      f"({j.out})")
        levels = p512_levels(s, jobs, problems)
        if levels is not None:
            outputs["train_levels"] = levels
        print("\n  set [s515] harsher_amp_db, lossier_n_pl and [s511c] knee_s in params.toml "
              "by hand; --apply writes 5.12's levels")
    elif stage == "quick":
        return report_quick(s, jobs)
    elif stage in RL_STAGES:
        trained = [j for j in jobs if j.kind == "rl-train"]
        print(f"  trainings: {sum(1 for j in trained if done(j))}/{len(trained)} saved")
        for j in jobs:
            if j.kind != "tool":
                continue
            path = REPO / j.out
            if not path.exists():
                print(f"  {j.name}: not written yet")
                continue
            if j.name.endswith("/report"):
                r = json.loads(path.read_text(encoding="utf-8"))
                keys = ("outcome", "replace_fx", "greedy_1_beats_fx", "stack_check", "epsilon")
                shown = {k: r[k] for k in keys if k in r}
                print(f"  {j.name}: " + (json.dumps(shown) if shown else
                                         f"see {j.out} (keys: {', '.join(list(r)[:10])})"))
            else:
                print(f"  {j.name}: written ({j.out})")
    else:
        print(f"no report for stage {stage}")
        return 1
    if outputs:
        print("\n[pilot_outputs]")
        for key, per_n in outputs.items():
            print(f"{key} = {_inline(per_n)}")
    if apply_to is None:
        return 0
    if stage not in ("ttl", "knee", "sstar", "pilot3"):
        print(f"\n--apply: stage {stage}'s values are a reading, not a rule; set them by hand")
        return 1
    unfinished = [j.name for j in jobs if not j.blocked and j.alias_of is None and not done(j)]
    if unfinished:
        problems.append(f"{len(unfinished)} jobs not finished (e.g. {unfinished[0]})")
    if problems or not outputs:
        print("\n--apply: nothing written:\n  " + "\n  ".join(problems or ["no outputs yet"]))
        return 1
    for old, new in write_pilot_outputs(apply_to, {k: _inline(v) for k, v in outputs.items()}):
        print(f"\n  {apply_to.name}: {old.strip() or '(new)'}\n      -> {new}")
    print(f"\nwritten to {apply_to}; review the change and commit it before the next stage")
    return 0


# --------------------------------------------------------------------------- #
# Scoring (scoring.py says what each step does)
# --------------------------------------------------------------------------- #

def _scoring():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import scoring
    return scoring


def score_plan(stage: str, s: Settings, jobs: List[Job]):
    """(the scorer jobs, each study's entries, the tool outputs, the plan) of a stage.

    A job that aliases another reads the other's CSV (batch 1's, or the
    stage's own primary), and keeps its own cell and variant in its study.
    Scorer runs are one per scored file: studies that share a CSV and score
    it alike share the run."""
    SC = _scoring()
    plan, problems = SC.plan_from(s.data)
    if problems:
        sys.exit("params.toml [score]: " + "; ".join(problems))
    # An alias's primary may itself read batch 1's CSV (reuse_batch1 set its out).
    by_name = {j.name: j for j in jobs}
    entries: Dict[str, List[Any]] = collections.OrderedDict()
    runs: Dict[str, Job] = collections.OrderedDict()
    tools: List[Tuple[str, str, bool]] = []
    light = float(s.get("machine.gb_per_light_process"))
    for j in jobs:
        if j.kind == "tool":
            tools.append((j.name, j.out, (REPO / j.out).exists()))
        if j.kind != "runner" or j.blocked:
            continue
        src = j
        if j.alias_of and not j.alias_of.startswith("batch1:"):
            src = by_name[j.alias_of]
        spec = plan.specs.get(j.study)
        s_star = s.get(f"pilot_outputs.s_star.{j.n}", None)
        args, scored, missing = SC.scorer_call(src.out, plan, spec, s_star)
        arm = j.args[j.args.index("--arms") + 1]
        cell, variant = SC.labels(j.name, arm)
        entries.setdefault(j.study, []).append(SC.Entry(
            j.study, cell, variant, arm, REPO / src.out, REPO / scored, int(j.trials or 0)))
        if scored in runs or missing:
            continue
        rows, _ = csv_rows(REPO / src.out)
        if rows and (REPO / SC.traces_dir(src.out)).is_dir():
            runs[scored] = Job(f"score-{stage}", j.study, f"{j.study}/{Path(scored).stem}",
                               args, scored, "score", mem_gb=2 * light)
    return list(runs.values()), entries, tools, plan


def cmd_score(stage: str, s: Settings, text: str, jobs: List[Job], a: argparse.Namespace) -> int:
    """Score a stage's trials, then write each study's comparisons."""
    if stage not in SCORED:
        sys.exit(f"score reads the studies' trials: one of {', '.join(SCORED)} "
                 f"(a pilot has `report`)")
    SC = _scoring()
    runs, entries, tools, plan = score_plan(stage, s, jobs)
    rc = 0
    if not a.compare_only:
        if a.rescore:
            for j in runs:
                p = REPO / j.out
                p.with_suffix(p.suffix + ".argv.json").unlink(missing_ok=True)
        todo = [j for j in runs if not done(j)]
        print(f"score {stage}: {len(runs)} scored files, {len(todo)} to (re)score")
        if todo:
            # A file that failed to score leaves its study's rows out; the
            # comparisons below use what is scored, and say how much that is.
            rc = cmd_run(f"score-{stage}", s, text, todo, allow_dirty=a.allow_dirty,
                         skip_blocked=True, yes=a.yes, max_jobs=a.max_jobs,
                         out_root=a.out_root)
    out_dir = REPO / a.out_root / "scores" / STAGE_DIR[stage]
    results = collections.OrderedDict()
    comparisons = []
    for study, members in entries.items():
        spec = plan.specs.get(study) or SC.Spec(study, None)
        arms, cmps = SC.study_results(study, members, spec, plan)
        results[study] = (spec, arms, cmps)
        comparisons += cmps
    SC.holm(comparisons, plan)
    index = []
    for study, (spec, arms, cmps) in results.items():
        SC.write_study(out_dir, study, spec, plan, arms, cmps)
        index.append({"study": study,
                      "metric": SC.column(spec.metric, plan.tau) if spec.metric else "",
                      "variants": len(arms), "comparisons": len(cmps),
                      "claims": sum(1 for c in cmps if c.claim),
                      "scored": sum(a_["scored"] for a_ in arms),
                      "trials": sum(a_["trials"] for a_ in arms)})
        print(f"  {study:6s} {len(arms):3d} variants, {len(cmps):3d} comparisons, "
              f"{sum(1 for c in cmps if c.claim):3d} claims; "
              f"{sum(a_['scored'] for a_ in arms)}/{sum(a_['trials'] for a_ in arms)} trials scored")
    path = SC.write_index(out_dir, stage, index, tools, plan)
    print(f"\nscores in {out_dir} (start at {path.name})")
    return rc


# --------------------------------------------------------------------------- #
# Trace archives: the kept traces stay out of git (decided 5 Oct 2026); each
# stage's go into one archive whose checksum is committed, for a release
# --------------------------------------------------------------------------- #

ARCHIVES = "archives"          # under the out root
SUMS = "SHA256SUMS"


def archive_name(stage: str) -> str:
    return STAGE_DIR[stage].replace("/", "_") + "_traces.tar.gz"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_sums(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = digest
    return sums


def cmd_pack(stage: str, a: argparse.Namespace) -> int:
    """Every kept-trace folder of a stage into <out_root>/archives/<stage
    dir>_traces.tar.gz, its SHA-256 into archives/SHA256SUMS (committed) and a
    listing beside it (folders, trials, files, bytes, the commit)."""
    import tarfile
    root = REPO / a.out_root
    src = root / STAGE_DIR[stage]
    lock = root / ".launcher.lock"
    if lock.exists():
        try:
            held = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            held = {}
        if held.get("stage") == stage:
            sys.exit(f"stage {stage} is running (pid {held.get('pid')}): pack it once it ends")
    dirs = sorted(p for p in src.rglob("*_traces") if p.is_dir()) if src.is_dir() else []
    if not dirs:
        print(f"{stage}: no kept traces under {src}")
        return 0
    out = root / ARCHIVES
    out.mkdir(parents=True, exist_ok=True)
    name = archive_name(stage)
    part = out / (name + ".part")
    listing = []
    with tarfile.open(part, "w:gz", compresslevel=9) as tar:
        for d in dirs:
            files = [f for f in d.rglob("*") if f.is_file()]
            listing.append({"path": d.relative_to(root).as_posix(),
                            "trials": sum(1 for x in d.iterdir() if x.is_dir()),
                            "files": len(files), "bytes": sum(f.stat().st_size for f in files)})
            tar.add(d, arcname=d.relative_to(root).as_posix())
    part.replace(out / name)
    digest = _sha256(out / name)
    sums = _read_sums(out / SUMS)
    sums[name] = digest
    (out / SUMS).write_text("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())),
                            encoding="utf-8")
    size = (out / name).stat().st_size
    raw = sum(x["bytes"] for x in listing)
    (out / (name[: -len(".tar.gz")] + ".json")).write_text(json.dumps({
        "stage": stage, "archive": name, "sha256": digest, "bytes": size,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD").strip(), "folders": listing}, indent=1),
        encoding="utf-8")
    print(f"{stage}: {len(dirs)} trace folders, {sum(x['trials'] for x in listing)} trials, "
          f"{raw / 1e6:.1f} MB -> {out / name} ({size / 1e6:.1f} MB), sha256 {digest[:16]}...")
    return 0


def cmd_unpack(stage: str, a: argparse.Namespace) -> int:
    """A stage's trace archive back into place, after checking its SHA-256
    against archives/SHA256SUMS."""
    import tarfile
    root = REPO / a.out_root
    arch = root / ARCHIVES / archive_name(stage)
    if not arch.exists():
        print(f"{stage}: {arch} is not here (download it from the release into "
              f"{arch.parent})")
        return 1
    want = _read_sums(root / ARCHIVES / SUMS).get(arch.name)
    if want is None:
        sys.exit(f"{arch.name} has no checksum in {ARCHIVES}/{SUMS}")
    if _sha256(arch) != want:
        sys.exit(f"{arch.name} does not match its recorded SHA-256: a damaged or other file")
    prefix = STAGE_DIR[stage] + "/"
    with tarfile.open(arch, "r:gz") as tar:
        bad = [m.name for m in tar.getmembers() if not m.name.startswith(prefix)]
        if bad:
            sys.exit(f"{arch.name} holds files outside {prefix}: {bad[0]}")
        tar.extractall(root, filter="data")
        n = sum(1 for m in tar.getmembers() if m.isfile())
    print(f"{stage}: {n} trace files into {root / STAGE_DIR[stage]}")
    return 0


# --------------------------------------------------------------------------- #
# Smoke runs and the campaign
# --------------------------------------------------------------------------- #

#: Where --smoke writes unless told otherwise: beside the repository, never in it.
SMOKE_ROOT = REPO.parent / "exp5_smoke"

#: The stages in the order the campaign runs them. quick is not one: it is a
#: reviewer's reproduction of batch 1, run on its own.
CAMPAIGN = ("ttl", "knee", "sstar", "rl-headroom", "rl-calibrate", "rl-sweep", "rl-e3",
            "rl-s57", "batch1", "sens", "batch2", "pilot3", "batch3")

#: Stages after which a person reads a report and sets params.toml: the pilots'
#: outputs, the calibration's sanity check, Study 5.5's verdict and batch 3's pilots.
PAUSE_AFTER = ("ttl", "knee", "sstar", "rl-calibrate", "rl-sweep", "pilot3")


def _set_flag(args: List[str], flag: str, value: str) -> None:
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]


def smoke(jobs: List[Job]) -> None:
    """Every job at its smallest: a check that each starts and ends, not a result."""
    for j in jobs:
        a = j.args
        if j.kind == "runner":
            _set_flag(a, "--n-trials", "1")
            j.trials = 1
        elif j.kind == "rl-train":
            for flag, value in (("--episodes", "30"), ("--eval-every", "15"),
                                ("--val-episodes", "8")):
                _set_flag(a, flag, value)
        elif a and a[0].endswith("fit_time_probe.py"):
            _set_flag(a, "--fits", "1")
            _set_flag(a, "--trials", "1")
        elif "experiments.analysis.age_cap_s_star" in a:
            _set_flag(a, "--layouts", "3")
        elif "pilot" in a:
            _set_flag(a, "--episodes", "1")
        elif "evaluate" in a:
            _set_flag(a, "--episodes", "3")
        elif "headroom" in a:
            _set_flag(a, "--episodes", "4")


#: Named groups of stages; any command that takes a stage takes a group. `run`
#: on a group goes through its stages as the campaign does.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "pilots": ("ttl", "knee", "sstar"),
    "rl": RL_STAGES,
    "batches": ("batch1", "sens", "batch2", "pilot3", "batch3"),
    "all": CAMPAIGN,
}


def jobs_for(stage: str, s: Settings, a: argparse.Namespace) -> List[Job]:
    """A stage's jobs with the command line's selection: --study, then --arms
    (stack trials of those arms only; every other kind of job is left out),
    then --smoke."""
    jobs = build(stage, s, a.study, a.out_root)
    if a.arms:
        jobs = [j for j in jobs if j.kind == "runner"
                and j.args[j.args.index("--arms") + 1] in a.arms]
    if a.smoke:
        smoke(jobs)
    return jobs


def cmd_sequence(name: str, stages: Sequence[str], s: Settings, text: str,
                 a: argparse.Namespace) -> int:
    """Run stages in order, skipping those done, until a decision point.

    It stops before a stage whose settings are unset (and names them; with
    --skip-blocked it runs that stage's other jobs first), after a stage whose
    report a person reads (the pilots, the calibration, Study 5.5's verdict,
    batch 3's pilots: it prints the report), and at any failure. Run it again
    once the settings are in params.toml and committed."""
    for stage in stages:
        jobs = jobs_for(stage, s, a)
        if not jobs:
            print(f"[{name}] {stage}: no jobs (left out by --study or --arms)")
            continue
        live = [j for j in jobs if j.alias_of is None]
        blocked = sorted({b for j in jobs for b in j.blocked})
        if live and not blocked and all(done(j) for j in live):
            print(f"[{name}] {stage}: done")
            continue
        if blocked and not a.skip_blocked:
            print(f"[{name}] stops before {stage}: params.toml needs {', '.join(blocked)}")
            return 2
        print(f"[{name}] {stage}: running")
        rc = cmd_run(stage, s, text, jobs, allow_dirty=a.allow_dirty, skip_blocked=True,
                     yes=a.yes, max_jobs=a.max_jobs, out_root=a.out_root)
        if rc != 0:
            print(f"[{name}] stops: stage {stage} did not finish cleanly (rc {rc})")
            return rc
        if blocked:
            print(f"[{name}] stops after {stage}'s runnable jobs: params.toml needs "
                  f"{', '.join(blocked)} for the rest")
            return 2
        if stage in PAUSE_AFTER:
            cmd_report(stage, s, jobs)
            how = (f"`report {stage} --apply` writes them" if stage in ("ttl", "knee", "sstar")
                   else "set them by hand")
            print(f"[{name}] pauses after {stage}: put what the report gives in params.toml "
                  f"({how}), commit, and run again")
            return 0
    print(f"[{name}] every stage is done")
    return 0


def cmd_status_all(s: Settings, a: argparse.Namespace) -> int:
    """One line per stage: jobs done, trials written, what is unset."""
    for stage in CAMPAIGN + ("quick",):
        jobs = jobs_for(stage, s, a)
        live = [j for j in jobs if j.alias_of is None and not j.blocked]
        n_done = sum(1 for j in live if done(j))
        rows = expected = 0
        for j in live:
            if j.kind == "runner":
                rows += min(csv_rows(REPO / j.out)[0], j.trials or 0)
                expected += j.trials or 0
        blocked = sorted({b for j in jobs for b in j.blocked})
        line = f"  {stage:13s} jobs {n_done:3d}/{len(live):<3d}"
        if expected:
            line += f"  trials {rows:5d}/{expected:<5d}"
        if blocked:
            line += f"  needs {', '.join(blocked[:3])}" + (" ..." if len(blocked) > 3 else "")
        print(line)
    return 0


def cmd_check(s: Settings, a: argparse.Namespace) -> int:
    """Is this machine ready, and what does each stage still need?"""
    from importlib import metadata
    import shutil
    failed = []

    def line(state: Optional[bool], what: str, detail: str) -> None:
        mark = {True: " ok ", None: "warn", False: "FAIL"}[state]
        if state is False:
            failed.append(what)
        print(f"  [{mark}] {what:22s} {detail}")

    print("Machine")
    line(sys.version_info >= (3, 11), "Python", f"{sys.version.split()[0]} ({sys.executable})")
    problem = interpreter_problem()
    line(problem is None, "interpreter", problem or "the interpreter itself, not a venv stub")
    req = REPO / "AppSetup" / "requirements_exp4.txt"
    for raw in req.read_text(encoding="utf-8").splitlines():
        spec = raw.split("#", 1)[0].strip()
        if "==" not in spec:
            continue
        name, pinned = (x.strip() for x in spec.split("==", 1))
        try:
            have: Optional[str] = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        line(False if have is None else (True if have == pinned else None), name,
             f"{have or 'missing'} (requirements_exp4.txt pins {pinned})")
    data = dataset_fingerprint()
    line(bool(data["csv_files"]), "CICIoT2023",
         f"{data['csv_files']} CSV files, {data.get('bytes', 0) / 1e9:.1f} GB in {data['dir']}"
         if data["csv_files"] else "not found: ../datasets/CICIOT2023, or --dataset DIR")
    line(True if long_paths_enabled() else None, "long paths",
         "on" if long_paths_enabled() else f"off: kept traces must stay under {MAX_PATH} chars")
    prov = provenance()
    line(None if prov["code_dirty"] else True, "code",
         f"commit {prov['commit'][:10]} on {prov['branch']}"
         + (f"; uncommitted under hermes/ or experiments/: {len(prov['code_dirty'])} files "
            f"(run refuses without --allow-dirty)" if prov["code_dirty"] else ""))
    try:
        import psutil
        ram = psutil.virtual_memory().total / 2**30
        cores = psutil.cpu_count(logical=False) or os.cpu_count()
        budget = float(s.get("machine.mem_budget_gb"))
        line(True if budget <= 0.9 * ram else None, "memory",
             f"{ram:.0f} GB; params machine.mem_budget_gb {budget:g} (--mem-gb to change)")
        line(True, "cores", f"{cores} physical, {os.cpu_count()} logical; machine.max_jobs "
                            f"{s.get('machine.max_jobs')} (--jobs to change)")
    except ImportError:
        line(False, "psutil", "missing (the footprint probe and the lock need it)")
    root = REPO / a.out_root
    probe = root if root.exists() else REPO
    free = shutil.disk_usage(probe).free / 2**30
    line(True if free > 20 else None, "disk", f"{free:.0f} GB free at {probe} "
                                              f"(the whole campaign keeps about 1 GB of traces)")
    lock = root / ".launcher.lock"
    if lock.exists():
        try:
            held = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            held = {}
        line(None, "running", f"stage {held.get('stage')} holds the lock (pid {held.get('pid')}, "
                              f"since {held.get('since')})")
    print("\nStages (in campaign order; quick is a reviewer's reproduction)")
    cmd_status_all(s, a)
    print("\n" + ("ready" if not failed else "not ready: " + ", ".join(failed)))
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_validate"]:
        return _validate_child()
    # A stage runs for hours, often with its output going to a file: show each line.
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("command", choices=("check", "plan", "validate", "run", "status", "report",
                                        "score", "pack", "unpack", "campaign"))
    groups = "; ".join(f"{k}: {' '.join(v)}" for k, v in GROUPS.items() if k != "all")
    ap.add_argument("stage", choices=STAGES + tuple(GROUPS), nargs="?", default=None,
                    metavar="stage",
                    help=f"A stage ({', '.join(STAGES)}) or a group of them ({groups}; "
                         f"all: the campaign). check, campaign and status need none.")
    g = ap.add_argument_group("what runs")
    g.add_argument("--study", nargs="+", default=None, metavar="STUDY",
                   help="Only these studies of the stage (e.g. s53 s514).")
    g.add_argument("--arms", nargs="+", default=None, metavar="ARM",
                   help="Only the stack trials of these arms (e.g. F FX H1); every other job "
                        "(FerrySim, tools, trainings) is left out.")
    g.add_argument("--smoke", action="store_true",
                   help="Every job at its smallest (1 trial; RL at 30 episodes), written to "
                        "../exp5_smoke unless --out-root says otherwise: a check that every "
                        "job starts and ends, never a result.")
    g.add_argument("--out-root", default="results/exp5", metavar="DIR",
                   help="Where the stages write, relative to the repository (default "
                        "results/exp5). A run with changed settings needs its own.")
    g.add_argument("--params", type=Path, default=PARAMS, metavar="FILE",
                   help="The settings file (default scripts/exp5/params.toml).")
    g = ap.add_argument_group(
        "levers (this run only: params.toml is not changed, and every stage's manifest "
        "records each one)")
    g.add_argument("--trials", type=int, default=None, metavar="N",
                   help="Trials per cell, in every study (n_trials). Trial i keeps its seed "
                        "whatever the count, so fewer trials are the first ones of a full run.")
    g.add_argument("--seed", type=int, default=None, metavar="N",
                   help="campaign.base_seed: a fresh draw of every layout and channel (a "
                        "replication; give it its own --out-root).")
    g.add_argument("--missions", type=int, default=None, metavar="N", help="campaign.n_missions.")
    g.add_argument("--contact-regime", choices=("clean", "jittery"), default=None,
                   help="campaign.contact_regime: the contact channel of every job that "
                        "does not set its own.")
    g.add_argument("--tau", type=float, nargs="+", default=None, metavar="TAU",
                   help="score.tau: the accuracy thresholds; the first is the primary.")
    g.add_argument("--dataset", default=None, metavar="DIR",
                   help="Where CICIoT2023's CSVs are (default ../datasets/CICIOT2023).")
    g.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                   help="Any params.toml setting by its dotted key, the value as TOML "
                        "(--set s58.missions=[4,8] --set s53.n_trials=40); repeatable.")
    g = ap.add_argument_group("the machine")
    g.add_argument("--jobs", "--max-jobs", dest="max_jobs", type=int, default=None, metavar="N",
                   help="Jobs side by side (machine.max_jobs; rl.max_jobs for RL stages).")
    g.add_argument("--mem-gb", type=float, default=None, metavar="GB",
                   help="machine.mem_budget_gb: the memory the running jobs may take.")
    g.add_argument("--devices", type=int, default=None, metavar="N",
                   help="machine.max_device_processes: training processes at once.")
    g = ap.add_argument_group("run control")
    g.add_argument("--yes", action="store_true", help="Start without asking.")
    g.add_argument("--skip-blocked", action="store_true",
                   help="Run the jobs whose settings are set, and leave out the rest.")
    g.add_argument("--allow-dirty", action="store_true",
                   help="Run with uncommitted changes under hermes/ or experiments/.")
    g.add_argument("--show-commands", action="store_true", help="plan: print each command.")
    g.add_argument("--apply", action="store_true",
                   help="report (ttl, knee, sstar, pilot3): write the pilot's outputs into "
                        "the params file, once every job is finished and the report is clean.")
    g.add_argument("--rescore", action="store_true",
                   help="score: score every file again, current or not.")
    g.add_argument("--compare-only", action="store_true",
                   help="score: only rewrite the comparisons from the scored files there are.")
    a = ap.parse_args(argv)
    if a.smoke and a.out_root == "results/exp5":
        a.out_root = str(SMOKE_ROOT)           # a smoke run never writes the results tree

    s, text = load_settings(a.params)
    apply_levers(s, a)
    if a.command == "check":
        return cmd_check(s, a)
    if a.command == "campaign":
        return cmd_sequence("campaign", CAMPAIGN, s, text, a)
    if a.command == "status" and a.stage is None:
        return cmd_status_all(s, a)
    if a.stage is None:
        ap.error(f"{a.command} needs a stage or a group: one of "
                 f"{', '.join(STAGES + tuple(GROUPS))}")
    if a.command == "run" and a.stage in GROUPS:
        return cmd_sequence(a.stage, GROUPS[a.stage], s, text, a)
    stages = GROUPS.get(a.stage, (a.stage,))
    if a.command == "score" and a.stage in GROUPS:
        stages = tuple(st for st in stages if st in SCORED)
    rc = 0
    for stage in stages:
        if a.command in ("pack", "unpack"):
            rc |= (cmd_pack if a.command == "pack" else cmd_unpack)(stage, a)
            continue
        if len(stages) > 1:
            print(f"\n=== {stage} ===")
        jobs = jobs_for(stage, s, a)
        if a.command == "plan":
            rc |= cmd_plan(stage, s, jobs, a.show_commands)
        elif a.command == "validate":
            checked = cmd_validate(jobs)
            cmd_plan(stage, s, jobs, False, checked)
            rc |= 1 if any(not v["ok"] for v in checked.values()) else 0
        elif a.command == "status":
            rc |= cmd_status(stage, jobs)
        elif a.command == "report":
            rc |= cmd_report(stage, s, jobs, apply_to=a.params if a.apply else None)
        elif a.command == "score":
            rc |= cmd_score(stage, s, text, jobs, a)
        else:
            rc = cmd_run(stage, s, text, jobs, allow_dirty=a.allow_dirty,
                         skip_blocked=a.skip_blocked, yes=a.yes, max_jobs=a.max_jobs,
                         out_root=a.out_root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
