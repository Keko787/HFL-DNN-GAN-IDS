"""Exp 5 launcher: one command per stage, from scripts/exp5/params.toml.

Exp 5 runs on the Exp 4 harness (python -m experiments.exp4.runner_main) with
the FeRRy flags. This launcher turns a stage into jobs, one runner process per
(study, cell, arm), and runs them side by side:

    plan      list the jobs, what each still needs, the trial count and the cost
    validate  run every job's arguments through the runner's own usage checks
              (runner_main.main with the trial loop stubbed out: no trial runs)
    run       validate, write a manifest, then run the jobs in parallel
    status    count the rows each job's CSV holds against what it should
    report    a finished pilot's pilot_outputs lines for params.toml, by the
              pre-registered rules (ttl: 2 x p95 rounded up; knee: params
              [knee] metric and plateau_share; sstar: the tool's S for F)

Stages, in order (each fills in settings the next one needs; see params.toml):

    ttl      the session-TTL pilot: fit_time_probe.py at each N
    knee     H1 budget sweeps on wide, per N and payload
    sstar    the S* tool per N at the knee and the stress budget
    batch1   5.3 core, 5.9 core, 5.11 (a) and (b), 5.14 (the 8 Oct split)

    ..\\py311.cmd scripts\\exp5\\launch.py plan batch1
    ..\\py311.cmd scripts\\exp5\\launch.py run ttl
    ..\\py311.cmd scripts\\exp5\\launch.py status knee

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
STAGES = ("ttl", "knee", "sstar", "batch1")
STAGE_DIR = {"ttl": "ttl", "knee": "knee", "sstar": "sstar", "batch1": "b1"}

#: The plan arms take the age cap (--age-cap-missions); the H and D arms do not.
PLAN_ARMS = {"F", "FX", "FB+wide", "FB+medium", "FB+narrow", "F-cov", "F-prio", "F+L1"}


def is_ferry_arm(arm: str) -> bool:
    """The arms that fly campaign.ferry_arm_deadline_law: the plan arms, F-cap
    (F with no cap) and the learned FQ arms."""
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
# Jobs
# --------------------------------------------------------------------------- #

@dataclass
class Job:
    stage: str
    study: str
    name: str                     # unique within the stage: <study>/<tag>
    args: List[str]               # after the interpreter
    out: str                      # the CSV or JSON it writes, relative to REPO
    kind: str                     # "runner" or "tool"
    n: int = 0                    # devices (for cost)
    mem_gb: float = 0.0
    trials: Optional[int] = None  # rows it should write (runner jobs)
    blocked: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    alias_of: Optional[str] = None


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
                cap: Any = None, footprint: bool = False) -> Job:
        s = self.s
        before = len(s.missing)
        out = f"{self.root}/{STAGE_DIR[stage]}/{study}/{tag}.csv"
        ttl = s.per_n("pilot_outputs.session_ttl_s", n)
        seed = s.get("campaign.base_seed")
        args = RUNNER + [
            "--csv", out, "--arms", arm, "--N", str(n),
            "--n-missions", _fmt(s.get("campaign.n_missions")),
            "--regime", str(s.get("campaign.regime")),
            "--n-trials", str(n_trials), "--base-seed", _fmt(seed),
            "--mission-budget-s", _fmt(budget_s), "--session-ttl-s", _fmt(ttl),
        ] + COMMON_FLAGS
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
        if footprint:
            args += ["--footprint-probe"]
        args += list(extra)
        job = Job(stage, study, f"{study}/{tag}", args, out, "runner", n=n,
                  mem_gb=self._mem_runner(n, k), trials=n_trials)
        job.blocked = sorted(set(s.missing[before:]))
        del s.missing[before:]
        return job

    def _budget(self, level: str, n: int) -> Any:
        key = {"knee": "knee_s", "stress": "stress_s"}[level]
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
                 footprint: bool = False) -> Job:
        before = len(self.s.missing)
        budget = self._budget(level, n)
        cap_value = self._cap(cap, n) if arm in PLAN_ARMS else None
        missing = self.s.missing[before:]
        del self.s.missing[before:]
        tag = f"n{n}k{k}_{level}__{tag_arm or _arm_tag(arm)}"
        job = self._runner(stage, study, tag, arm=arm, n=n, k=k, budget_s=budget,
                           payload=payload, n_trials=n_trials, extra=extra,
                           cap=cap_value, footprint=footprint)
        job.blocked = sorted(set(job.blocked) | set(missing))
        return job

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


def build(stage: str, s: Settings, studies: Optional[Sequence[str]] = None,
          root: str = "results/exp5") -> List[Job]:
    b = Builder(s, root)
    getattr(b, stage)()
    jobs = b.jobs
    if studies:
        jobs = [j for j in jobs if j.study in studies]
    return dedupe(jobs)


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
    todo = [j for j in jobs if j.kind == "runner" and not j.blocked and j.alias_of is None]
    if not todo:
        return {}
    payload = json.dumps([{"name": j.name, "args": j.args} for j in todo])
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
            elif checked and j.name in checked and not checked[j.name]["ok"]:
                note = "INVALID: " + checked[j.name]["error"]
            if j.kind == "runner" and not j.alias_of:
                conc = b.concurrency(j.n) if j.n else 1
                hours += (j.trials or 0) * seconds_per_trial(j.n) / conc / 3600
            size = f"{j.trials} trials" if j.kind == "runner" else "tool"
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
    print(f"validated {len(checked)} runner jobs: {len(checked) - len(bad)} ok, {len(bad)} refused")
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
    if stage in ("ttl", "knee", "batch1") and not data["csv_files"]:
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

    checked = cmd_validate(jobs)
    if any(not v["ok"] for v in checked.values()):
        sys.exit("the runner refuses the jobs above; nothing ran")

    # Resume guard: a CSV written under other arguments is refused.
    for j in jobs:
        if j.alias_of:
            continue
        side = (REPO / j.out).with_suffix((REPO / j.out).suffix + ".argv.json")
        if side.exists():
            before = json.loads(side.read_text(encoding="utf-8"))
            if _key_args(before) != _key_args(j.args):
                sys.exit(f"{j.out} was written under other arguments ({side.name}); "
                         f"write the new setting to a fresh path")

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
    cap_jobs = max_jobs or int(s.get("machine.max_jobs"))
    mem_budget = float(s.get("machine.mem_budget_gb"))
    cap_devices = int(s.get("machine.max_device_processes"))
    pending = sorted(todo, key=lambda j: -j.mem_gb)   # largest first: no long tail
    running: Dict[str, Tuple[Job, subprocess.Popen, float, Any]] = {}
    failed: List[str] = []
    try:
        while pending or running:
            used = sum(r[0].mem_gb for r in running.values())
            devices = sum(r[0].n for r in running.values() if r[0].kind == "runner")
            for j in list(pending):
                if len(running) >= cap_jobs:
                    break
                if running and used + j.mem_gb > mem_budget:
                    continue
                if running and j.kind == "runner" and devices + j.n > cap_devices:
                    continue
                out = REPO / j.out
                out.parent.mkdir(parents=True, exist_ok=True)
                if j.kind == "runner":
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
                log_event(event="start", job=j.name, pid=proc.pid)
                print(f"[{time.strftime('%H:%M:%S')}] start {j.name} "
                      f"({len(running)} running, {len(pending)} waiting)")
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
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nstage {stage}: {len(todo) - len(failed)} jobs ok, {len(failed)} failed; "
          f"manifest {manifest_path}")
    for name in failed:
        print(f"  failed: {name} (log in {root / '_logs'})")
    return 1 if failed else 0


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


def cmd_report(stage: str, s: Settings, jobs: List[Job]) -> int:
    """A finished pilot's pilot_outputs lines for params.toml, by the pre-registered rules."""
    out: Dict[str, Any] = {"stage": stage}
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
        print("\n[pilot_outputs]\nsession_ttl_s = { "
              + ", ".join(f'"{k}" = {v:.1f}' for k, v in ttl.items()) + " }")
    elif stage == "knee":
        metric = str(s.get("knee.metric"))
        share = float(s.get("knee.plateau_share"))
        cells: Dict[Tuple[int, str], Dict[float, List[float]]] = collections.OrderedDict()
        not_ok = 0
        for j in jobs:
            path = REPO / j.out
            if not path.exists() or j.blocked:
                continue
            budget = float(j.args[j.args.index("--mission-budget-s") + 1])
            payload = (j.args[j.args.index("--payload-bytes") + 1]
                       if "--payload-bytes" in j.args else "measured")
            values = cells.setdefault((j.n, payload), {}).setdefault(budget, [])
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("status", "ok") != "ok":
                        not_ok += 1
                        continue
                    try:
                        values.append(float(row[metric]))
                    except (KeyError, ValueError):
                        pass
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
                print(f"    {b:7.0f} s   {m:6.3f} ± {half:5.3f}  (n = {len(v)}, served share "
                      f"{m / n:5.1%}){mark}")
            budgets = sorted(by_budget)
            if knee == budgets[-1]:
                print("    WARNING: the knee is the grid's largest budget; the grid may not "
                      "reach the plateau. Extend it before using this knee.")
            if knee == budgets[0]:
                print("    WARNING: the knee is the grid's smallest budget; extend the grid down.")
        out["knees"] = knees
        out["rows_not_ok"] = not_ok
        print(f"\n  rows not ok (left out): {not_ok}")
        if "1mb" in knees:
            print("\n[pilot_outputs]")
            print("knee_s   = { " + ", ".join(f'"{k}" = {v:.1f}' for k, v in knees["1mb"].items()) + " }")
            print("stress_s = { " + ", ".join(
                f'"{k}" = {v:.1f}' for k, v in knees["1mb_stress"].items()) + " }")
        if "meas" in knees:
            print("knee_meas_s   = { " + ", ".join(
                f'"{k}" = {v:.1f}' for k, v in knees["meas"].items()) + " }")
            print("stress_meas_s = { " + ", ".join(
                f'"{k}" = {v:.1f}' for k, v in knees["meas_stress"].items()) + " }")
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
        print("\n[pilot_outputs]\ns_star = { " + ", ".join(f'"{k}" = {v}' for k, v in stars.items())
              + " }")
    else:
        print(f"no report for stage {stage}")
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_validate"]:
        return _validate_child()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("command", choices=("plan", "validate", "run", "status", "report"))
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--study", nargs="+", default=None,
                    help="Only these studies of the stage (e.g. s53 s514).")
    ap.add_argument("--params", type=Path, default=PARAMS)
    ap.add_argument("--show-commands", action="store_true", help="plan: print each command.")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run: allow uncommitted changes under hermes/ or experiments/.")
    ap.add_argument("--skip-blocked", action="store_true",
                    help="run: leave out the jobs whose settings are unset.")
    ap.add_argument("--max-jobs", type=int, default=None, help="run: override machine.max_jobs.")
    ap.add_argument("--yes", action="store_true", help="run: start without asking.")
    ap.add_argument("--out-root", default="results/exp5",
                    help="Where the stages write (default results/exp5; for a dry test, "
                         "a scratch folder).")
    a = ap.parse_args(argv)

    s, text = load_settings(a.params)
    jobs = build(a.stage, s, a.study, a.out_root)
    if a.command == "plan":
        return cmd_plan(a.stage, s, jobs, a.show_commands)
    if a.command == "validate":
        checked = cmd_validate(jobs)
        cmd_plan(a.stage, s, jobs, False, checked)
        return 1 if any(not v["ok"] for v in checked.values()) else 0
    if a.command == "status":
        return cmd_status(a.stage, jobs)
    if a.command == "report":
        return cmd_report(a.stage, s, jobs)
    return cmd_run(a.stage, s, text, jobs, allow_dirty=a.allow_dirty,
                   skip_blocked=a.skip_blocked, yes=a.yes, max_jobs=a.max_jobs,
                   out_root=a.out_root)


if __name__ == "__main__":
    sys.exit(main())
