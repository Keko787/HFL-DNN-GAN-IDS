# Experiment 5 — Reproducibility Guide

How to set up a machine for Experiment 5 (FeRRy, Studies 5.1–5.15), run it in full or in part, change its settings from the command line, score it, and check a reproduction against the recorded results.

There are two ways in:

- **Path A, reproduce the headline (a few hours).** Rerun Study 5.3's headline cell with the committed settings, then compare your trials one by one with the recorded ones. This is the path for artifact reviewers. See [§4](#4-path-a--reproduce-the-headline).
- **Path B, run the campaign (several days).** Run every stage from the pilots to scoring. See [§5](#5-path-b--run-the-campaign).

Both use one command, `exp5`, described in [§3](#3-the-exp5-command).

**Contents**
1. [What the experiment is](#1-what-the-experiment-is)
2. [Setup](#2-setup)
3. [The `exp5` command](#3-the-exp5-command)
4. [Path A — reproduce the headline](#4-path-a--reproduce-the-headline)
5. [Path B — run the campaign](#5-path-b--run-the-campaign)
6. [Levers: changing a run without editing files](#6-levers-changing-a-run-without-editing-files)
7. [Scoring and the analysis plan](#7-scoring-and-the-analysis-plan)
8. [Outputs and provenance](#8-outputs-and-provenance)
9. [Determinism: what matches across machines](#9-determinism-what-matches-across-machines)
10. [Troubleshooting](#10-troubleshooting)
11. [Status and open decisions](#11-status-and-open-decisions)

---

## 1. What the experiment is

Exp 5 evaluates FeRRy, the HERMES scheduler built from contributions C1–C5, in fifteen studies. The studies and their arms are in [`FeRRy_Build_Plan.html`](FeRRy_Build_Plan.html) (Experiments, Study overview and Addendum).

Every trial runs on the real-process stack: one cluster, K mules and N device processes over loopback TCP, with real Keras training on CICIoT2023, on a simulated mission clock. A trial is one run of Exp 4's harness, `python -m experiments.exp4.runner_main`, with the FeRRy flags. Every flag is documented in the [Exp 4 Run Guide](Experiment_4_Run_Guide.md) §2.6–2.8 and the [Configuration Reference](HERMES_Configuration_Reference.md) §17–20.

The launcher, [`scripts/exp5/launch.py`](../scripts/exp5/launch.py), turns each **stage** of the campaign into runner jobs (one process per study, cell and arm). It reads every setting from [`scripts/exp5/params.toml`](../scripts/exp5/params.toml): arms, cells, budgets, trial counts, machine limits, the pilots' outputs, the RL decisions and the analysis plan. Nothing about a study's design lives in code.

### The stages

| Stage | What it runs | Trials | Launcher's estimate¹ | Needs first |
|---|---|---|---|---|
| `ttl` | Session-timeout pilot: real-model fit time at N = 6, 12, 18, 24 | (4 probes) | minutes | — |
| `knee` | H1 budget sweeps per N and payload, which set the knee and stress budgets | 600 | 3.1 h | `ttl` |
| `sstar` | The age cap S\* per N at the knee and stress budgets | (4 tool runs) | minutes | `knee` |
| `rl-headroom` … `rl-s57` | The learned score's FerrySim campaign: headroom, calibration, Study 5.5's γ sweep and verdict, E3's trainings, 5.7's scores | 77–116 trainings | 12 h | `knee` and the re-pin ([§5.3](#53-the-re-pin)) |
| `batch1` | 5.3 core, 5.9 core, 5.11 (a) and (b), 5.14 | 1,760 | 4.6 h | `sstar` |
| `sens` | 5.3's F, FX and H1 at the session timeout × 0.75 and × 1.5 (× 1 is batch 1's own cell) | 120 | 0.2 h | `batch1` |
| `batch2` | 5.1, 5.2, the rest of 5.3, 5.4 with the O1 oracle, 5.5's stack check, 5.6, 5.7, 5.8, the rest of 5.9, 5.13 | about 9,100–10,100² | 35 h | the RL verdict |
| `pilot3` | Batch 3's own pilots: 5.15's interference levels, 5.12's training-time levels, 5.11 (c)'s FerrySim budget sweeps | 80 | under an hour | `knee` |
| `batch3` | 5.12, 5.15, 5.11 (c) | 1,060 | 1.5 h | `pilot3` |
| `quick` | **Not part of the campaign:** a reviewer's reproduction of 5.3's headline cell ([§4](#4-path-a--reproduce-the-headline)) | 320 | 0.5 h | committed pilot outputs |

¹ From `exp5 plan`, on the reference machine ([§2.1](#21-the-machine)). These are rough, and they run short: the knee pilot took about 1.5 times its estimate. RL trainings are estimated at 1–2 h each, 14 at a time.
² Depends on the RL verdict: the learned (FQ) arms fly only if 5.5 keeps the learned score.

Study 5.10 (AERPAW's real radios) is not a launcher stage; it waits for testbed access.

The groups `pilots` (ttl, knee, sstar), `rl` (the five RL stages), `batches` (batch1, sens, batch2, pilot3, batch3) and `all` (the campaign) can be used anywhere a stage name is.

---

## 2. Setup

### 2.1 The machine

The campaign was sized on one Windows 10 Pro machine with 8 physical cores (16 logical) and 95 GB of RAM, where these limits ran cleanly:
- 6 jobs at once (`max_jobs`);
- 36 training processes at once (`max_device_processes`; that is six N = 6 trials, or one N = 24 trial with an N = 12 one);
- a 75 GB memory budget (`mem_budget_gb`);
- 14 RL trainings at once (`[rl] max_jobs`).

These limits are `"auto"` in `params.toml`, which scales them to the host that runs:

| Limit | `"auto"` | Sized-on host | 12 cores (20 logical), 64 GB |
|---|---|---|---|
| `max_device_processes` | the smaller of 4.5 × physical cores and 2.25 × logical CPUs | 36 | 45 |
| `max_jobs` | `max_device_processes` ÷ 6 (whole N = 6 jobs) | 6 | 7 |
| `mem_budget_gb` | 0.8 × RAM, rounded down | 75 | 51 |
| `[rl] max_jobs` | logical CPUs − 2 | 14 | 18 |

`check` prints what each limit comes to on your machine, and every manifest records it under `auto`. To fix a limit, give a number in `params.toml`, or use a lever for one run ([§6](#6-levers-changing-a-run-without-editing-files)). The limits only pace the scheduler. No trial's arguments depend on them, except the TTL probe's count of trials side by side, which is the load it measures. On a smaller machine the jobs are the same; they just take longer. No GPU is used.

**Run every stage of a campaign on one machine.** A few planner values computed through `math.erfc` differ in the last bit between Windows builds, which can flip rare near-ties ([§9](#9-determinism-what-matches-across-machines)). The session timeouts are wall-clock fit times on the CPU that ran the TTL pilot. Arms are compared only within one host. `check` warns (`pilot host`) when the TTL pilot's manifest names a different host, and `run` repeats the warning before any stage with stack trials. For a campaign of the new host's own, run `ttl` and `knee` there under a fresh `--out-root`. During a long stage, pause operating-system updates.

**Windows power throttling.** Windows 11 throttles a process it judges to be in the background (EcoQoS), and on a hybrid CPU that confines it to the efficiency cores. A stage left running in a terminal, or with the screen off, qualifies: on the second host its jobs ran at about 28% CPU (6 Oct 2026), and every wall-clock measurement slowed with them. So on Windows the launcher opts itself and every process under it (each runner, its device, cluster and mule processes, each trainer) out of power throttling as they appear, and keeps the machine awake while a stage runs. `[machine] unthrottle = false` turns this off; `check` prints it, and each manifest records it under `power` (the processes opted out, and any refused). It needs psutil to reach the processes under the jobs.

**Disk:** the dataset takes 13.8 GB. The kept event traces take roughly 60–120 KB per trial, about 1 GB in some 300,000 files for the whole campaign ([§8.1](#81-trace-archives)).

### 2.2 Python

Use Python 3.11.9 with the pinned Exp 4 packages (TensorFlow 2.21, Keras 3.14, Flower 1.39 and the rest in [`AppSetup/requirements_exp4.txt`](../AppSetup/requirements_exp4.txt)):

```bash
conda create -n ferry311 python==3.11.9
conda activate ferry311
python -m pip install -r AppSetup/requirements_exp4.txt pytest
```

**On Windows, use a conda environment, not a venv.** A venv's `python.exe` there is a launcher stub that starts the real interpreter as a child process, so every trial process appears twice and Study 5.11's footprint probe measures the stub. The launcher refuses a venv interpreter.

### 2.3 The dataset

Unzip all 169 `part-*.csv` files of CICIoT2023 (12.8 GiB), flat, into `../datasets/CICIOT2023/` beside the repository. Alternatively, put them anywhere and pass `--dataset DIR` to every command, or set `HERMES_CICIOT_DIR`.

Each trial draws 3 training files and 1 test file by its seed, so the whole set is needed. Without the dataset, the model loader would quietly fall back to a synthetic task; the launcher refuses to start instead. Each manifest records the dataset's fingerprint: file count, bytes, and a hash of the names and sizes.

### 2.4 Long paths (Windows)

Kept traces are deep: `results/exp5/<stage>/<study>/<cell>__<arm>_traces/<trial>/<file>`. With Windows long paths off, keep the repository path short. `plan` prints the deepest path a stage will write, and `run` refuses a stage that would exceed 250 characters.

### 2.5 Pointing `exp5` at the interpreter

The `exp5` scripts start the launcher with this interpreter:
- `EXP5_PYTHON`, if set: a `python` executable, or (on Windows) a `.cmd` that starts one;
- on Windows, otherwise, `..\py311.cmd` beside the repository, if it exists;
- otherwise `python` on PATH (Windows) or `python3` (Linux, macOS).

```bash
export EXP5_PYTHON="$HOME/miniconda3/envs/ferry311/bin/python"
```

On Windows: `set EXP5_PYTHON=C:\Users\<you>\miniconda3\envs\ferry311\python.exe`.

### 2.6 Check the machine

```bash
scripts/exp5/exp5.sh check
```

`check` reports, each marked ok, warn or FAIL:
- the interpreter, and every package against its pin;
- the dataset, with its file count and size;
- Windows long paths;
- uncommitted code;
- memory and cores against `[machine]`, and what each `"auto"` limit comes to;
- whether the session timeouts were measured on this host (`pilot host`);
- free disk space;
- a stage already running (it holds `results/exp5/.launcher.lock`).

It then lists every stage's progress and the settings each stage still needs. It ends with `ready` or with the failed items.

### 2.7 The code gate (Path B, and before reporting results)

Run the full test suite against the three recorded baselines. It takes about 23 minutes on 8 cores:

```bash
python -m pytest tests -p no:cacheprovider -q -rfE --junitxml=run.xml
python tests/golden/make_baseline.py compare run.xml
```

The comparison must show only:
- the five known failures, with their recorded signatures (the selector's decision-dense cell and four `test_mode_switch` subprocess tests);
- the load-sensitive real-model smoke test, which the comparison allows;
- on any host other than the one that recorded the baselines, five last-bit `erfc` differences: `test_golden_feasibility::test_random_instances`, three `test_golden_p4_plan` `mission_completed` pins, and `test_p3_legacy_equivalence::test_the_one_predicate_reproduces_every_recorded_walk`.

---

## 3. The `exp5` command

Below, `exp5` stands for `scripts/exp5/exp5.sh` (Linux, macOS) or `scripts\exp5\exp5` (Windows). Both run `scripts/exp5/launch.py`, which can also be called directly with the right interpreter. Run everything from the repository root.

```
exp5 <command> [stage or group] [options]
```

| Command | What it does |
|---|---|
| `check` | Machine readiness and every stage's needs ([§2.6](#26-check-the-machine)). |
| `plan <stage>` | Lists the jobs, what each still needs, the trial count, the rough compute time and the deepest trace path. `--show-commands` prints each runner command. |
| `validate <stage>` | Passes every job's arguments through the runner's own checks without running a trial. |
| `run <stage>` | Validates, writes a manifest, asks before starting (or `--yes`), then runs the jobs side by side. On a group, it goes through the group's stages as the campaign does. |
| `status [stage]` | Jobs done and trials written. With no stage, one line per stage. |
| `report <stage>` | A pilot's outputs by the pre-registered rules (`ttl`, `knee`, `sstar`). Also: what `pilot3` measured, the RL verdicts, and the reproduction check (`quick`). `--apply` writes a pilot's outputs into `params.toml`. |
| `score <stage>` | The studies' paired comparisons ([§7](#7-scoring-and-the-analysis-plan)). |
| `pack <stage>` / `unpack <stage>` | A stage's kept traces into one checksummed archive, and back ([§8.1](#81-trace-archives)). |
| `campaign` | The same as `run all`. |

**Stopping and resuming.** Stop a run with Ctrl+C: the running jobs are stopped, and running the same command again resumes. Each runner job skips the trials already in its CSV, and RL stages skip checkpoints that exist. A trial that failed keeps its row (status `error` or `timeout`) and is not retried; `status` counts those rows as "not ok".

**One stage at a time.** `run` holds `results/exp5/.launcher.lock`, so a second stage started beside it is refused. Two stages sharing the cores would skew each other's timing.

**Shared cells run once.** Jobs whose arguments differ only in their CSV path and trial count are one job, run at the largest trial count; the others read its first n trials, which have the same seeds. For example, 5.3's F at N = 6 reads the first 20 of 5.14's 40 trials. `plan` and the manifest list each alias.

**Group runs stop at decision points.** `run <group>` (and `campaign`) skips stages that are done. It stops before a stage whose settings are unset, and names them. It pauses after a stage whose report a person must read: the pilots, the RL calibration, Study 5.5's verdict and batch 3's pilots. It also stops at any failure. Fill in what it asked for, commit, and run the same command again.

**Smoke runs.** `--smoke` runs every job at its smallest (one trial; RL trainings at 30 episodes) into `../exp5_smoke`, outside the results tree. It checks that every job starts and ends; it never produces a result.

```bash
exp5 run batch1 --smoke --yes
```

---

## 4. Path A — reproduce the headline

Path A reruns Study 5.3's headline cell: the core arms F, FX, H1, D1, D2, D3, D4, and the faithful D4, at N = 6 with one mule, at the knee and stress budgets, 20 trials each (320 trials). It runs in about half an hour on the reference machine, and a few hours on a laptop. It needs the committed `params.toml`, whose pilot outputs are filled in, and the recorded batch 1 CSVs under `results/exp5/b1/`. It does not need the pilots or the RL campaign.

```bash
exp5 check
exp5 run quick --yes
exp5 report quick
exp5 score quick
```

**Why the trials can be compared one by one.** `quick`'s jobs are batch 1's own jobs, with the same arguments and therefore the same seeds. Each trial's seed is `sha256(base_seed | cell | trial)`, so reproduced trial *i* is the same draw as recorded trial *i*: the same device layout, data shards and channel. `quick` writes to `results/exp5/quick/`, never over batch 1's CSVs.

**`report quick`** pairs every reproduced trial with the recorded trial that has the same `(trial_index, seed)`. For each arm it reports:
- how many trials match in every compared column to 1e-9;
- for each column (final accuracy, AUC, update yield, round closure, rounds closed, contacts per mission, mission duration), the recorded mean, the reproduced mean and the largest difference.

The result is written to `results/exp5/quick/reproduction.json`. On the recording machine, expect close or identical values. On another machine, expect small differences ([§9](#9-determinism-what-matches-across-machines)); what should hold is the comparison between arms.

**`score quick`** runs 5.3's analysis on the reproduction and writes `results/exp5/scores/quick/s53.md`, to compare with the recorded `results/exp5/scores/b1/s53.md`.

**Shorter or longer.**
- `--trials 5` runs the first 5 trials of each cell, still the same seeds, so they pair with the recorded trials 0–4.
- `--set quick.K=[1,3]` adds the three-mule cells, which makes it the whole of batch 1's 5.3 core.
- `--arms F FX H1` runs only those arms.

```bash
exp5 run quick --trials 5 --yes
```

---

## 5. Path B — run the campaign

The order matters. The pilots set the budgets every study flies. The learned score's verdict (Study 5.5) decides F's in-flight slot before the studies that fly F in batch 2. Commit `params.toml` after each decision: every manifest records the commit, and a run refuses uncommitted changes under `hermes/` or `experiments/`.

### 5.1 Settings

Review [`scripts/exp5/params.toml`](../scripts/exp5/params.toml) before the first stage: the campaign seed, regimes and laws (`[campaign]`), the machine limits (`[machine]`), each stage's arms, cells and trial counts, and the analysis plan (`[score]`). A setting that is commented out is **unset**. Every job that needs it is refused, and `plan`, `check` and the refusal name it.

### 5.2 The pilots

```bash
exp5 run pilots --yes
```

This runs `ttl`, then pauses for its report. Each pilot fills in what the next one needs:

| Pilot | Rule (pre-registered in `params.toml`) | Writes |
|---|---|---|
| `ttl` | 2 × the p95 fit time, rounded up | `session_ttl_s` |
| `knee` | The smallest budget whose mean update yield reaches 95% of the grid's best; stress is half of it, rounded to 5 s. τ: the largest τ (in steps of 0.01) that at least 80% of H1's trials reach within the trial at each N's knee at 1 MB, the smallest over N | `knee_s`, `stress_s` (and `knee_meas_s`, `stress_meas_s` at the measured payload), `tau` |
| `sstar` | The S\* tool's S for F | `s_star` |

After each pilot, write its outputs and commit, then run the group again:

```bash
exp5 report knee --apply
git commit -am "Exp 5: knee pilot outputs"
exp5 run pilots --yes
```

`--apply` replaces only those keys' lines in `[pilot_outputs]` and keeps every other line. It refuses while a job is unfinished, or when the report warns (for example, a knee at the edge of its budget grid, which means the grid should be extended).

The knee pilot also sets **τ**, the accuracy threshold of most studies' primary metric (time to τ). At 4 missions, few trials reach the build plan's τ = 0.82. So τ is set from the pilot, by the rule in the table, fixed on 5 Oct 2026 before the pilot finished; 0.82 stays as a second τ, and its reach rate is still reported. A trial "reaches" τ when its highest accuracy after any round is at least τ, which is how the scorer counts it. The report reads this from the kept traces, and for each budget it also gives the mean final accuracy and how many trials reach 0.82.

### 5.3 The re-pin

FerrySim's training cells fly placeholder budgets until the stack's pilots measure them: Phase 4's priors at N = 6, stand-ins at N = 12. The re-pin moves them to the measured stress and knee budgets, so the learned score trains on the budgets the stack flies. It runs once, after the knee and S\* pilots, and is committed with the repository. A reproduction checks out a commit that already contains it; a fresh campaign with different knees redoes it.

```bash
python scripts/exp5/repin.py --dry-run
python scripts/exp5/repin.py --workers 8
```

From `params.toml`'s pilot outputs, `repin.py`:
- rewrites the re-pin block in `experiments/ferrysim/cells.py` (each size's budgets, the caps and Study 5.6's lags), from which every cell is built;
- renames the cells across the code and tests, since each name carries its budget (`jit-n12-120` becomes `jit-n12-<stress>`);
- computes each cell's cap with the S\* tool;
- re-measures Study 5.6's lag with FX on the same 200-episode sample, and derives the lag ratio test's bounds from the same sample;
- rewrites the tests' pinned values (the family hashes, lags, periods and lag sample).

It stops if the quarter-period and half-period cells' ratio ranges would meet, or if a Study 5.6 cell's S\* differs from its base cell's cap. About 3 minutes with 2 workers (26 s with 8 on the second host).

Then run the FerrySim tests it prints, review the documents it lists (their records are history, so it leaves them alone; references such as the Configuration Reference's cell table and the guides' commands are updated by hand), commit, and set `[rl] repinned = true`. A test that holds a renamed cell's budget as a bare number fails here, since the script renames names, not numbers. Every RL stage refuses until that is set. See also the [Exp 4 Run Guide](Experiment_4_Run_Guide.md) §2.8 ("The re-pin").

### 5.4 The learned score (RL)

```bash
exp5 run rl --yes
```

The RL stages are listed below, with the decisions they need from the Scheduler Freeze §5l:
- `rl-headroom`;
- `rl-calibrate`: γ ∈ {0, 0.9} × 3 seeds per family; pauses for the sanity check;
- `rl-sweep`: Study 5.5, 6 γ × 10 seeds, `evaluate --record` and the pre-registered verdict; pauses;
- `rl-e3`: E3's trainings, at Chen's settings: γ = 0.99, lr 5e-4, ε from 1.0, the defaults of his published code;
- `rl-s57`: 5.7's scores, only if the verdict keeps the learned score.

Each training is one `ferrysim train` job with one BLAS thread. Train from a clean tree: a checkpoint's manifest records its commit, and the runner refuses to fly a checkpoint trained from a dirty tree.

After the verdict, set `[rl] gamma_star` and `keep_learned`, and the checkpoint paths in `[rl.checkpoints]` (`report rl-sweep` prints the picks). Commit.

### 5.5 The batches

```bash
exp5 run batches --yes
```

This runs `batch1`, `sens`, `batch2`, `pilot3` and `batch3` in order:
- A cell that batch 1 already ran is not run again: a later batch reads batch 1's CSV, or appends its extra trials to it with the same seeds.
- `pilot3` pauses for its report:
  - `report pilot3 --apply` writes 5.12's training-time levels (`pilot_outputs.train_levels`) by p512's rule. p512 flies H1 at the N = 6 knee with the median fit time at 0.25, 0.5, 1 and 2 times the mission cycle (the knee budget plus the 30 s turnaround). "Spread" is the level at which 10–30% of contacts find no update ready, the one nearest 20% if several; "stragglers" is spread with 20% of the devices at 5×.
  - Set `[s515] harsher_amp_db` and `lossier_n_pl`, and `[s511c] knee_s`, by hand from the same report.
- `--skip-blocked` runs a stage's other jobs first.

The study settings once left open are decided: FedProx's ρ is 0.01 (`[s51]`), 5.6 flies 40 trials per cell (`[s56]`), and 5.12's levels come from the pilot.

### 5.6 Scoring

```bash
exp5 score batches
```

See [§7](#7-scoring-and-the-analysis-plan). Score a batch any time after it runs. Scoring again redoes only the files whose trial CSV changed.

---

## 6. Levers: changing a run without editing files

Levers change a setting **for one command only**. `params.toml` is not edited. Each change is printed (`[lever] …`) and recorded in the manifest of every stage the command starts, under `overrides`, along with the settings as changed.

| Lever | Sets | Use |
|---|---|---|
| `--trials N` | `n_trials` in every study | Fewer or more trials. Trial *i* keeps its seed whatever the count. |
| `--seed N` | `campaign.base_seed` | A replication with fresh layouts and channels. |
| `--missions N` | `campaign.n_missions` | Missions per trial. |
| `--contact-regime clean\|jittery` | `campaign.contact_regime` | The contact channel, for every job that does not set its own. |
| `--tau T [T …]` | `score.tau` | The accuracy thresholds; the first is the primary. |
| `--dataset DIR` | `HERMES_CICIOT_DIR` | Where CICIoT2023 is. |
| `--jobs N` | `machine.max_jobs` (`rl.max_jobs` for RL stages) | Jobs side by side. This lever and the next two replace the `"auto"` value ([§2.1](#21-the-machine)). |
| `--mem-gb GB` | `machine.mem_budget_gb` | The memory the running jobs may take. |
| `--devices N` | `machine.max_device_processes` | Training processes at once. |
| `--set KEY=VALUE` | Any dotted key; the value is read as TOML | Anything else, for example `--set s58.missions=[4,8]`, `--set s53.arms='["F","H1"]'` or `--set score.family=cell`. Repeatable. A table `params.toml` does not have is refused. |

Two options narrow what runs:
- `--study S [S …]` keeps only those studies of the stage, for example `--study s53 s514`.
- `--arms A [A …]` keeps only the stack trials of those arms, and leaves out every other kind of job (FerrySim, tools, trainings).

**Changed settings need their own output folder.** A job's CSV carries a sidecar (`<csv>.argv.json`) with the arguments that wrote it. Resuming a CSV under different arguments is refused, and the refusal names the arguments that differ. So a run whose levers change a job's arguments (`--seed`, `--missions`, `--contact-regime`, most `--set` changes) writes to a fresh place with `--out-root`:

```bash
exp5 run batch1 --study s53 --seed 7 --out-root ../exp5_seed7 --yes
exp5 score batch1 --study s53 --out-root ../exp5_seed7
```

`--trials`, `--tau`, `--jobs`, `--mem-gb`, `--devices` and `--dataset` do not change a job's arguments in a way the resume guard checks, so they can share the default folder. For `--trials`, fewer trials are simply the first trials of a full run.

**On a smaller machine:**

```bash
exp5 run quick --jobs 2 --devices 12 --mem-gb 16 --yes
```

---

## 7. Scoring and the analysis plan

`exp5 score <stage>` works in two steps.

1. **Score.** The trace scorer ([`experiments/analysis/traces_scorer.py`](../experiments/analysis/traces_scorer.py)) runs once per trial CSV over its kept traces. It writes `<csv stem>_scored*.csv` beside the CSV, with the scorer's arguments in a sidecar file. It always adds the cost and pair columns. Some studies add more: 5.12 the compute columns, 5.13 the detection columns, and 5.8 and 5.14 an age cap at the study's S.
2. **Compare** ([`scripts/exp5/scoring.py`](../scripts/exp5/scoring.py)). Within each study:
   - Trials are grouped by cell (the job's tag before `__`) and by variant (after it: the arm, or the study's own label, such as 5.14's switch or 5.1's merge rule).
   - In each cell, every variant is compared with the study's reference variant on the study's primary metric. The comparison is trial by trial on the paired seeds: the mean paired difference with a bootstrap 95% CI, and a paired Wilcoxon test with Cliff's δ.
   - Then Holm's adjustment is applied across the study.
   - **A claim needs the CI to exclude zero and a Holm-adjusted p below α.**

The analysis plan is `[score]` in `params.toml`, fixed before the studies run. Its `tau = ["pilot", 0.82]` makes the knee pilot's τ (`pilot_outputs.tau`) the primary threshold and keeps 0.82 as the second; `score` refuses until the pilot's τ is set. The plan also gives α, the Holm family (`study` or `cell`), and per study: the primary metric and which direction is better, the reference variant, any variant compared with a different reference (`versus`), the columns reported beside the primary metric (`also`), and what the scorer adds.

**Missing values.** Pairs are complete cases, as in Exp 4's analysis:
- a trial whose status is not ok drops out of every comparison it is in;
- a blank metric drops the pair out of that comparison (time to τ is blank in a trial that never reached τ);
- each comparison reports its `n_pairs`, and the arms table gives the reach rate at τ beside the time;
- a trial index whose seed differs between the two sides is a pairing error: it is counted and left out, never compared.

**Outputs**, under `results/exp5/scores/<stage dir>/`:
- `index.md`: every study, its metric, and how many comparisons and claims it has;
- `<study>.md`: per cell, every variant's n, mean, median and reach rate, and its comparison with the reference;
- `<study>_arms.csv` and `<study>_comparisons.csv`: every column;
- `plan.json`: the analysis plan as applied.

FerrySim and tool outputs (5.11 (a) and (c), 5.4's O1 oracle) are JSON reports of their own, listed in `index.md`. Questions that compare across cells, such as the 5.9 scale trend, 5.11's scaling with mules, 5.4's curve and 5.6's interaction, are read from the arms tables.

---

## 8. Outputs and provenance

```
results/exp5/
  <stage dir>/                     ttl, knee, sstar, b1, sens, b2, p3, b3, quick, rl/…
    <study>/<cell>__<variant>.csv          one runner job's trials
    <study>/<cell>__<variant>.csv.argv.json   the arguments that wrote it
    <study>/<cell>__<variant>_traces/      kept event traces, one folder per trial
    <study>/<cell>__<variant>_scored*.csv  the trace scorer's rows
    _launcher/manifest_<time>.json         one per run
    _launcher/jobs.jsonl                   every job's start and end
    _logs/<job>.log                        every job's output
  checkpoints/<study>/<tag>/g<γ>_s<seed>.npz (+ .json)   RL checkpoints with manifests
  scores/<stage dir>/                      the comparisons (§7)
  archives/<stage dir>_traces.tar.gz       a stage's kept traces (exp5 pack; not in git)
  archives/<stage dir>_traces.json         what each archive holds
  archives/SHA256SUMS                      every archive's checksum
```

**The manifest** of every run records:
- the commit, and any uncommitted change;
- the interpreter and package versions, and the thread caps;
- the dataset's fingerprint;
- the host;
- the launcher's hash, and the full `params.toml`;
- the command line's overrides, and the settings as changed;
- every job's command.

To reproduce any stage exactly: check out its commit, recreate the environment ([§2.2](#22-python)), check the dataset's fingerprint, and run the same stage with the manifest's `params.toml` and overrides.

**What is committed:**
- `params.toml` with its decisions;
- the trial CSVs and their sidecars, the scored CSVs, the manifests and the scores;
- the RL checkpoints;
- the archives' checksums and listings.

The kept event traces are not committed: about 1 GB in some 300,000 files over the campaign, against the 20 MB Exp 4 committed. Git ignores them (`.gitignore`); they go into per-stage archives instead ([§8.1](#81-trace-archives)). The analysis needs only the scored CSVs. The traces are needed only to re-score, for example with another τ or a new metric.

### 8.1 Trace archives

```bash
exp5 pack batch1
```

`pack` puts every kept-trace folder of a stage (or a group: `exp5 pack all`) into `results/exp5/archives/<stage dir>_traces.tar.gz`. The traces compress about 6×. It records the archive's SHA-256 in `archives/SHA256SUMS` and writes a listing beside it, `<stage dir>_traces.json`, with the folders, trial counts, sizes and the commit. Commit the checksums and listings; the archives themselves go to a release of the repository, and to a Zenodo deposit with a DOI for the paper's artifact.

To re-score from someone else's traces, download the archive into `results/exp5/archives/`, then:

```bash
exp5 unpack batch1
exp5 score batch1 --rescore
```

`unpack` refuses an archive whose checksum doesn't match `SHA256SUMS`, or one that holds files outside its stage's folder.

---

## 9. Determinism: what matches across machines

- **Seeds.** Every trial's seed is `sha256(base_seed | cell | trial)` and does not depend on the arm. So every arm of a cell runs the same layouts, data shards and channel, and the same trial in another run (or in `quick`) is the same draw.
- **Thread caps** (`[machine.threads]`: OMP 2, TF intra 2, inter 1, OpenBLAS 1) are set for every job and recorded. Keep them fixed for a campaign: they change wall-clock fit time, which the TTL pilot measured, and TensorFlow's thread count can change the model's last bits.
- **Across hosts.** A few planner values computed through `math.erfc` differ in the last bit between Windows builds. They flip rare near-ties, which the golden tests show as five known differences ([§2.7](#27-the-code-gate-path-b-and-before-reporting-results)). Trials also run as real processes over TCP, so scheduling on another machine can change timing-dependent outcomes. Expect a reproduction on another host to match the recorded trials closely but not bit for bit. What should hold is the comparison between arms, which is why `report quick` reports differences per column rather than demanding identity.
- **On one host**, with the same commit, settings and environment, a stage reproduces its trials.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `this is a venv interpreter` | On Windows, a venv's `python.exe` is a stub. Use the conda environment ([§2.2](#22-python)) and point `EXP5_PYTHON` at it. |
| `CICIoT2023 not found` | Put the CSVs in `../datasets/CICIOT2023`, or pass `--dataset DIR`. |
| `N jobs need unset settings (...)` | Set the named keys in `params.toml` (a pilot's: `report <pilot> --apply`), or `--skip-blocked` to run the rest. |
| `… was written under other arguments` | You changed a setting that changes a job's arguments. Give the run its own `--out-root` ([§6](#6-levers-changing-a-run-without-editing-files)). |
| `stage X is running (pid …)` | One stage at a time. Wait for it, or stop it. A stale lock from a dead process is replaced automatically. |
| `hermes/ or experiments/ has uncommitted changes` | Commit them: the manifest must name the code that ran. `--allow-dirty` overrides this for tests only. |
| `would write paths over 250 characters` | Windows long paths are off. Use a shorter repository path, or turn long paths on. |
| A job FAILED | Its output is in `results/exp5/<stage>/_logs/<job>.log`. Running the stage again resumes; failed trials keep their rows and are not retried. |
| `--apply: nothing written` | The pilot is unfinished, or the report warned. Read the report: a knee at the edge of the grid means extending `[knee] budgets_s`. |
| `the runner refuses the jobs above` | A job's arguments fail the runner's own checks. The message names the job and the check. |

---

## 11. Status and open decisions

*As of 6 Oct 2026.* What each stage still waits on, and the steps left before the sweep, are in [Experiment_5_Readiness.md](Experiment_5_Readiness.md).

- **Done:**
  - the TTL pilot: `session_ttl_s` 36, 34, 34 and 23 s at N = 6, 12, 18 and 24;
  - the knee pilot, on the jittery contact channel (600 trials, none failed): knees 150, 180, 240 and 262 s at N = 6, 12, 18 and 24 (stress half of each), 120 s at N = 6 with the measured payload, and τ = 0.71. N = 6's knee is the grid's largest budget; it was accepted as is rather than extending the grid. At N = 18 and 24 accuracy levels off near 0.71 at every budget;
  - the S\* pilot (`6e4b6c3`): `s_star` 2 at every N. One mission covers 90% of layouts at each knee, and two at each stress budget; N ≥ 12 by the tool's greedy bound. It ran on a second host (12 physical cores, 20 logical, 64 GB). The TTL and knee pilots ran on the first (8, 16, 95 GB). S\* is a planning-level calculation with no timing in it, so the host does not change it.
- **Host.** From S\* on, the campaign runs on the second host, by choice (5 Oct 2026). Its machine limits are `"auto"` ([§2.1](#21-the-machine)). The session timeouts and knees are still the first host's; `check` and `run` warn about it.
- **The re-pin, 5 Oct 2026 (`77880dc`):** FerrySim's N = 6 cells moved to 75 and 150 s (`jit-n6-75`, `jit-n6-150`), and N = 12 to 90 and 180 s (`jit-n12-90`, `cln-n12-90` and Study 5.6's `jit-n12-90-q`/`-h`; the 180 s cells kept their names). Caps 2; Study 5.6's lags 27 and 34 s, so P_c 108/54 and 136/68 s; ratio bounds 0.83–1.26 and 0.67–1.48.
- **The smoke run and the code gate, 5–6 Oct 2026:** every stage after the pilots passed a smoke run, every trial row ok, once two smoke-only problems were fixed (`b54185a`: smoke trainings now pass the learner's warm-up, and `report` shows an RL verdict). The code gate on `b54185a` matched all three baselines (6,375 tests, the five known failures). Details in the readiness doc's steps 3 and 4.
- **Next:**
  - batch 1, and the RL campaign;
  - p512's band, to decide before `pilot3` (the readiness doc's open decisions);
  - the batches.
- **Decided: τ.** In the knee pilot's first cells, H1 reached τ = 0.82 within 4 missions in few trials: 5 of the first 49 at N = 12, and none at N = 24, where accuracy levels off near 0.71 at every budget. The missions stay at 4, and τ is set from the knee pilot by the rule in [§5.2](#52-the-pilots), with 0.82 kept as a second τ. The rule gave 0.71 (0.82 at N = 6, 0.72 at N = 12, 0.71 at N = 18 and 24).
- **Decided: the traces.** Per-stage archives (`exp5 pack`), with their checksums committed; the archives go to a release and to Zenodo. Nothing has been published yet.
- **Decided: the study settings.**
  - E3's γ is 0.99, Chen's published code default.
  - FedProx's ρ in 5.1 is 0.01.
  - 5.6 flies 40 trials per cell, as 5.5's stack check and 5.7 do.
  - 5.12's training-time levels are set by a pilot (p512, in `pilot3`).
- **Built 7 Oct 2026, before batch 2 (`aedae8b`; code gate as the baselines):** FX-dwell and FX-cov for Study 5.7 (5.5 kept FX); `agg:asynchfl` switched to Async-HFL's polynomial staleness, q = 0.5; the launcher's power-throttling opt-out (§2.1).
- **Not built:** 5.1's `agg:seq` (decided out: it needs a protocol change); 5.1's hand-set merge weights; M1 (5.5 did not keep the learned score); Study 5.10 (needs AERPAW access).
