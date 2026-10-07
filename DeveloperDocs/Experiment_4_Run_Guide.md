# Experiment 4 — Run Guide

**Status:** Shipped. EX-4.0 → EX-4.3 are implemented on the real
[`MultiProcessOrchestrator`](../hermes/processes/orchestrator.py); the paper-grade
≥20-seed paired sweep has been run and analysed. This guide is the reproducer:
environment → smoke → the parallel paper sweep → analysis → figures.

**Companion docs:**
- [`HERMES_Experiment4_Methodology_and_Implementation.md`](HERMES_Experiment4_Methodology_and_Implementation.md) — **start here**: the holistic account of what Exp 4 is, how it is built, what it measured, and what it does not cover.
- [`HERMES_Experiment4_L1_RF_Layer.md`](HERMES_Experiment4_L1_RF_Layer.md) — Layer 1 in depth: channel model, `U(c,t)` controller, backhaul-loss schedule.
- [`HERMES_Experiment4_L2_Scheduling_Layer.md`](HERMES_Experiment4_L2_Scheduling_Layer.md) — Layer 2 in depth: the gated scheduler and the bounded RL tie-break.
- [`HERMES_Experiment4_Integrated_Design_and_Plan.html`](HERMES_Experiment4_Integrated_Design_and_Plan.html) — design, the arm-ablation ladder, and per-stage build status.
- [`HERMES_Experiment4_Jittery_Methodology.md`](HERMES_Experiment4_Jittery_Methodology.md) — the validity record: the crossover surface (§6), the L1 audit + integrated result (§7), and the adversarial-review remediation table (§8).
- [`HERMES_Operations_Runbook.md`](HERMES_Operations_Runbook.md) §0 — base environment setup this guide builds on.
- [`Experiment_1_Run_Guide.md`](Experiment_1_Run_Guide.md) / [`Experiment_3_Run_Guide.md`](Experiment_3_Run_Guide.md) — sibling guides.

---

## 0. What the experiment measures

**The measured realization of Algorithm 2 — L1 + L2 + L3 composed end-to-end,
against traditional flat FL, with a real DNN-IDS in the loop.** Unlike Exp 3
(which runs policy objects against the abstracted `Exp3Sim`), every Exp 4 arm runs
through the **real subprocess orchestrator** (real TCP, real two-pass Pass-1 → dock
→ Pass-2 cross-mule FedAvg, real Keras training on each device).

The arms are a **layer-ablation ladder**, all paired-seeded:

| Arm | What it turns on | Runs where |
|---|---|---|
| **H0** | Traditional flat FL: no mule, all clients each round, single-hop backhaul | in-process (`Exp4Driver._run_h0`) |
| **H1** | Mule + Four-Stage Gated Scheduler + two-pass HFL, deterministic distance ranking | real orchestrator |
| **H2** | H1 + `TargetSelectorRL` DDQN in the S3.5 tie-break | real orchestrator |
| **H3** | H2 + real L1 adaptive channel (`U(c,t)` controller feeding `rf_prior`) | real orchestrator |

**Headline results** (see the Methodology doc for the full tables):
- **H1 vs H0** is an honest *crossover*: under a clean backhaul H0 wins (the mule is
  overhead); under jittery, H1's participation advantage grows with the terrain
  dead-zone — a tie near the well-connected corner, decisive at `dead_zone≥0.4`
  (Cliff's δ up to +0.98, p<0.001).
- **H3 vs H2** (isolated L1): clean null, small significant jittery gain in the
  converged model (final AUC +0.012, accuracy +0.035, p<0.05).

---

## 1. Environment

Exp 4 uses the same library set as the rest of HERMES — **no Exp 4-specific
package**. Everything it imports (`numpy`, `pandas`, `matplotlib`, `tensorflow`/
`keras`, plus the stdlib) is already pinned in
[`AppSetup/requirements_core.txt`](../AppSetup/requirements_core.txt).

Two ways to get a working environment:

**A. The documented base venv** (Python 3.10, per [Runbook §0](HERMES_Operations_Runbook.md)):
```bash
python3.10 -m venv .venv310
source .venv310/Scripts/activate            # Git Bash on Windows
pip install -r AppSetup/requirements_core.txt
pip install pytest
```

**B. The exact environment that produced the committed results**, pinned in
[`AppSetup/requirements_exp4.txt`](../AppSetup/requirements_exp4.txt):
```bash
python3.11 -m venv .venv311
source .venv311/Scripts/activate            # Git Bash on Windows
pip install -r AppSetup/requirements_exp4.txt
pip install pytest
```
The `results/exp4_paper/*.csv` in this repo were generated on Python 3.11.9 with:

| | version |
|---|---|
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.8.4 |
| tensorflow | 2.21.0 |
| keras | 3.14.1 |

The code is tolerant of both (it runs on the 3.10/tf-2.15 base venv and on this
newer 3.11/tf-2.21/keras-3 stack). If you need bit-identical reproduction of the
committed CSVs, use `requirements_exp4.txt` (table B); for a fresh run, either works.

**Dataset.** The `canonical` data source reads the real CICIOT-2023 CSVs. They are
**gitignored** — place them at `../datasets/CICIOT2023/` (one level above the repo)
or point `HERMES_CICIOT_DIR` at them. No dataset? Use `--data-source synthetic`
(a real-shaped separable task, deterministic, needs nothing on disk).

**Verify the install** (fast unit suite — pure-Python, no subprocesses/TF fits):
```bash
pytest tests/unit/test_exp4_channel.py tests/unit/test_exp4_metrics.py \
       tests/unit/test_exp4_analysis.py tests/unit/test_exp4_model_task.py -q
```

---

## 2. The runner CLI

One entry point drives every arm: [`experiments.exp4.runner_main`](../experiments/exp4/runner_main.py).
`--help` for the canonical list; the load-bearing flags:

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | Per-trial CSV (created if missing; **resumable**). |
| `--arms` | `H0 H1 H2 H3 D1 D2 D3 D4 D5` (`driver.DEFAULT_ARMS`) | Subset of arms. **H0 and D2 need `--real-model`**; on `--mission-clock sim` H0 is dropped from the default list and refused when named. The plan arms run only when named (§2.7), and so do the Phase 5 arms (§2.8). See §2.3, §2.5. |
| `--N` | `2` | Device-population sweep. |
| `--rrf` | `60` | `rf_range_m` sweep. |
| `--n-missions` | `2` | Missions (FL rounds) per trial. |
| `--n-trials` | `1` | Paired seeds per cell. Use `20` for paper-grade. |
| `--regime` | `clean` | `clean` and/or `jittery`. |
| `--real-model` | off | Run the real canonical DNN-IDS (EX-4.1+). Omit → EX-4.0 noise stub. |
| `--data-source` | `canonical` | `canonical` (real CICIOT) or `synthetic` (no dataset). |
| `--realism` | off | Per-device short-range contact reliability + recoverable jittery backhaul loss. Required for the participation claim. |
| `--dead-zone` | `0.6` | **Sweep axis** — H0 jittery unreachable-client fraction. |
| `--link-quality` | `0.4` | **Sweep axis** — H0 jittery per-round success prob for a reachable client. |
| `--l1-channel` | off | Arm H3: adaptive channel → per-mission backhaul-loss schedule + `rf_prior`. Use with `--realism`. |
| `--selector-weights` | (none) | Trained DDQN `.npz` (from `experiments.exp3.train_a4`) for H2/H3. Omit → random-init selector (smoke only). |
| `--local-epochs` | `1` | Local training epochs per device per round. |
| `--trial-budget-s` | `120` | Hard per-trial wall-clock; the process tree is killed on overrun and the row recorded `status=error`. |

**H0 needs `--real-model`** (it is a real-model convergence baseline); it is dropped
with a warning from a stub run.

### 2.3 The arms

| Arm | What it is | Notes |
|---|---|---|
| `H0` | Traditional flat FL, no mule | **needs `--real-model`** |
| `H1` | + mule, gated scheduler, two-pass HFL, deterministic ranking | |
| `H2` | + `TargetSelectorRL` in the S3.5 tie-break | random-init unless `--selector-weights` |
| `H3` | + L1 adaptive channel | use with `--l1-channel` |
| `D1` | **SOTA baseline** — MAX-AoI, as a whole scheduler | stub or real-model |
| `D2` | **SOTA baseline** — Oort's statistical-utility selection, as a whole scheduler | **needs `--real-model`** |

D1 and D2 replace S3, S3b and S3.5 with their own rule, so they own admission as well as order.
They supersede the ordering-only B1/B2, which the driver no longer runs. D3–D5 are in §2.5,
the plan arms in §2.7, and the Phase 5 arms (FQ and its variants, E3 and `H1+L1`) in §2.8. H2 and
H3 leave Exp 5 (FeRRy Phase 5, decision 8), and `H1+L1` keeps the adaptive backhaul's reference;
`--require-trained` refuses an H2 or H3 without `--selector-weights`.

**Valid pairings.** `D1`/`D2`/`H2` vs `H1` isolate the policy (D1 and D2 the whole scheduler, H2
the ranking) — same transport, same realism, same seeds, one thing different. `H1` vs `H0` is the
architecture comparison. `H3` vs `H2` is the L1 comparison. **`H2`/`H3` must not be compared
against `H0`/`H1`** — they run with `--l1-channel`, which changes the backhaul model in both, and
their seeds do not line up.

> **Why `D2` refuses without `--real-model`.** Oort ranks on each device's training loss. The stub
> reports a *random* loss, so ranking on it would be a random ordering wearing Oort's name — a
> result-shaped artefact. The driver raises, and the policy raises `OortUnusableError` if devices
> were served but no loss arrived. Do not work around it; run `D2` with real training or not at all.

### 2.1 The two scheduler toggles — both off, and every committed result is an "off" run

These change what the scheduler *does*, so a run with either one on is **not comparable** with the
committed CSVs. Both default off; leaving them off reproduces the recorded behaviour exactly.

| Flag | Default | Meaning |
|---|---|---|
| `--mission-budget-s` | (none) | **Enforce the deadline.** Without it `Deadline(j)` is only a sort key. With it the S3b gate drops contacts that cannot be reached in time, the mule **aborts** a remainder it can no longer serve and returns with what it has, and skipped devices get their window widened. Measured cost at a slack budget: **mission completion 0.767 → 0.542**. |
| `--mission-window-adaptation` | off | **S3c mission-level widening.** Tracks `served/planned` across missions and widens *every* device's window while the mule is below target. Tunables: `--mission-window-target` (`0.8`), `--mission-window-gain` (`2.0`), `--mission-window-history` (`5`), `--mission-window-max-scale` (`4.0`). |

Two things to know before using them:

* **Adaptation without a budget should be a no-op.** If the deadline never binds, a wider window
  rescues nothing. Run S3c *with* `--mission-budget-s`; an adaptation-only arm is a negative
  control, not a result.
* **Start a new CSV.** Rows now carry `mission_budget_s` and `mission_window_adaptation`, so a
  results file is self-describing. A pre-existing CSV therefore **cannot be resumed** — the runner
  stops with `pass allow_schema_change=True to override`. Do not override: that error is the guard
  against pooling toggled rows with historical ones. Write to a new path instead.

```bash
python -m experiments.exp4.runner_main --csv results/exp4_s3c/on.csv --arms H1 --N 6 --rrf 50 --n-missions 8 --n-trials 20 --realism --mission-budget-s 120 --mission-window-adaptation --keep-event-traces
```

### 2.2 `--keep-event-traces` — pass this on anything you might want to re-analyse

Each trial's per-contact event stream normally lives in a temp run dir, gets folded into the
aggregate metrics, and is **deleted at teardown**. That is why a finished sweep cannot be re-scored
against a new scheduling baseline — there is nothing left to replay, so "how would policy X have
done?" costs a full re-run.

`--keep-event-traces` copies each trial's raw events next to the CSV instead
(`<csv-stem>_traces/`, or `--trace-dir`). It changes **no** trial behaviour.

| | |
|---|---|
| **Cost** | ~9.7 KB per trial — about **2.3 MB for a 240-trial matrix** |
| **What you get** | `device_served` / `device_serve_failed` with timestamps, plus **device positions** (kept from the configs — the events do not carry them, and no spatial policy can be scored without them) |
| **Failed trials** | captured too — traces are taken *before* the timeout check, so timed-out runs keep theirs |
| **Since 2026-09-28** | `device_served` carries `mission_round` and `pass_kind` (`collect` / `deliver`); `mission_completed` carries `pass_1_plan` (each contact's devices and deadline) and `pass_1_outcomes` (each session's device, outcome and contact time) |
| **Older traces** | none of those fields; attribute events to missions by joining timestamps against `mission_started` / `mission_completed`, which the scorer below does |

**Rule of thumb: if a run is expensive enough that you would not want to repeat it, pass this flag.**

**Scoring retained traces.** `experiments/analysis/traces_scorer.py` re-scores any trace root
without a re-run: the standard summary columns with round closure corrected (traces from before
Freeze Amendment 5 counted backhaul-dropped rounds as closed), time to τ in missions, cluster
rounds and wall-clock seconds, per-device update age and Network AoU, and the deadline-miss rate
where the trace records the Pass-1 plan.

```bash
python -m experiments.analysis.traces_scorer --traces results/exp4_matrix/C_traces --tau 0.82 0.75 --csv c_scored.csv
```

### 2.4 FeRRy Phase 1 — merge rules, FedProx, budgeted Pass 2 (mule arms)

All off by default; the defaults reproduce the recorded runs. H0 ignores them (flat FL keeps its
own mean). Every value is described in `HERMES_Configuration_Reference.md` §14–15, and each row
records `aggregation`, `aggregation_params` (JSON), `fedprox_rho`, `pass_2_budget`,
`deadline_law`, `deadline_params` (JSON) and `miss_priority`, so, as in §2.1, start a new CSV.

| Flag | Default | Meaning |
|---|---|---|
| `--aggregation` | `agg:plain` | The L3 merge rule, set on the cluster and the mule together. `agg:cutoff` (FeRRy: n·v·hinge(age), zero past the cutoff), `agg:asynchfl` (Async-HFL's polynomial (age + 1)^−q at the mule and the cluster; `--agg-asynchfl-form exponential` for exp(−λ·age), this rule's form before 7 Oct 2026), `agg:fedbuff` (apply the mean after K updates). |
| `--agg-server-lr`, `--agg-a-max`, `--agg-period-s`, `--agg-hinge-a`, `--agg-hinge-b`, `--agg-asynchfl-form`, `--agg-poly-q`, `--agg-decay`, `--agg-value`, `--agg-buffer-k` | see §14 | The rule's parameters. `--agg-period-s` turns each device's deadline window into its cutoff (decision D5). |
| `--fedprox-rho` | 0 | FedProx weight on every device. |
| `--pass-2-budget` | off | Walk Pass 2 against `--mission-budget-s` (required); skipped devices keep their older basis, so ages spread. Pass-1 devices train ahead on the basis they adopt, so one skipped in Pass 2 ships its next update one round old. |
| `--deadline-law` | `additive` | `multiplicative`: Φ ← clamp(β·clamp(Φ)), β_on after an on-time delivery, β_partial after a miss by a device that answered, β_timeout after one by a device that did not, one-shot cluster overrides. Tunables `--deadline-beta-on` (0.8), `--deadline-beta-partial` (1.25), `--deadline-beta-timeout` (1.5), `--deadline-phi-min` (5), `--deadline-phi-max` (300); see §15 of the configuration reference. |
| `--miss-priority` | off | S3b admits contacts by their members' consecutive misses before their deadline. Only acts with `--mission-budget-s`, since S3b does nothing without a budget. |

Each `mission_completed` event now carries `pass_1_merge` (rule, base version, per-device age and
weight share w_i / M_m under the age-aware rules, empty under `agg:plain`, updates excluded past
the cutoff), `pass_1_merged_devices` and
`pass_1_merged_updates` (the CLEAN devices whose update the merge used), `pass_2_skipped`, and
`deadline_state` (each device's window Φ in seconds and miss streak after the mission); each
`pass_1_plan` contact carries `device_deadlines` (each member's own deadline) and each
`pass_1_outcomes` entry the update's `basis_version` and `age`. A mission whose every update was
past its cutoff is reported empty but keeps its `pass_1_outcomes`. `mule_ready` records the
settings the mule actually runs (rule and parameters, deadline law and parameters, miss priority,
Pass-2 budget, mission budget) and `device_ready` the device's `fedprox_rho`. Age-aware rules add
cluster events carrying the uploading `mission_round` and the `partials` involved:
`cluster_merge` for a step, `cluster_merge_deferred` while FedBuff fills, and
`cluster_merge_expired` when every pending partial was past `a_max` (no step, round left open).
With `--keep-event-traces`, each kept trace also gets a `trial_status.json` (status, error,
run time, the trial's budget and, on the mission clock, the runner's soft cap), which
`experiments/analysis/traces_scorer.py` uses to leave failed trials out.

A Study 5.1 cell (H1 routes, budgeted Pass 2 so ages spread; repeat per rule with the same seeds):

```bash
python -m experiments.exp4.runner_main --csv results/exp5_s51/cutoff_b60.csv --arms H1 --N 6 --n-missions 4 --regime jittery --n-trials 40 --real-model --realism --mission-budget-s 60 --pass-2-budget --aggregation agg:cutoff --agg-a-max 2 --keep-event-traces
```

---

### 2.5 Several mules and the Phase 2 baselines

FeRRy Phase 2. Every default is the one-mule topology; see
`HERMES_Configuration_Reference.md` §16 for each value and Freeze §5i for what changed.

| Flag | Default | Meaning |
|---|---|---|
| `--arms D3 D4 D5` | — | D3 Whittle index (Cui et al., TMC 2024), D4 FedEx-Async with CARP (TMC 2025), D5 FedCS Algorithm 3 degraded (ICC 2019). Whole schedulers, like D1/D2. |
| `--n-mules` | 1 | Mules on one cluster; `--N` stays the total device count. |
| `--min-participation` | 1 | Partials per merge: 1 or `--n-mules`. `agg:plain` at several mules needs `--n-mules`. |
| `--dock-on-empty / --no-dock-on-empty` | on at several mules | An empty mission still docks with an empty partial. |
| `--down-wait-s` | the trial budget at several mules | How long a docked mule waits for its DOWN before flying on. |
| `--whittle-variant`, `--whittle-weights` | `expected`, `uniform` | D3's port choice and its ω. `oort` weights need `--real-model`. |
| `--fedcs-value` | `unit` | D5's greedy key. |
| `--agg-fedex-n` | registered devices | N in `agg:fedex`'s θ + Σ Δθ / N. |

A Study 5.3 channel-free cell (3 mules, 60 s budget; repeat per arm with the same seeds; D4
faithful uses `--aggregation agg:fedex`, every other arm an age-aware rule so a quorum of 1 is
sound):

```bash
python -m experiments.exp4.runner_main --csv results/exp5_s53/k3_b60_d4.csv --arms D4 --N 18 --n-mules 3 --n-missions 4 --regime clean --n-trials 40 --real-model --mission-budget-s 60 --aggregation agg:fedex --keep-event-traces
```

### 2.6 The mission clock and the contact link (FeRRy Phase 3)

Every default is the recorded wall-clock run; see `HERMES_Configuration_Reference.md` §17 for each
value and Freeze §5j for what changed. Every flag below needs `--mission-clock sim` except a numeric
`--deadline-time-scale`, `--initial-window-s`, `--session-ttl-s` and `--rf-link-token`, which the
wall clock takes too (`--t-nom-layouts` is accepted there and unused); `--deadline-time-scale t_nom`
and `--initial-window-missions` need `--mission-clock sim`. Write every Phase 3 run to a fresh CSV:
rows now carry 13 more provenance columns and 15 simulated ones, so an older CSV cannot be resumed.

| Flag | Default | Meaning |
|---|---|---|
| `--mission-clock` | `wall` | `sim` flies every mule arm on the simulated mission clock: flight, airtime, missed replies, the upload and the dock turnaround are charged to it, and nothing waits for them in wall time. H0 has no simulated round time: named, it is refused; in the default arm list it is dropped. |
| `--contact-band` | none | `wide` (the Phase 3 re-baselines), `medium` or `narrow`. Omit it for the channel-free control: every contact then costs 1 s, plus 1 s when a reply is missing. |
| `--contact-band-classes` | `wide medium narrow` | Add `medium_wide` for the optional 10 MHz class. |
| `--in-flight-response` | `abort` | `replan`: re-check the whole remainder at every departure and repair it instead of abandoning it. |
| `--replan-fallback` | `reorder` | For our arms under `replan`. `reorder` flies H1, H2 and H3 the same route whenever the pre-flight check fires; `trim` keeps each arm's own order and serves fewer stops. The pilot plan flies `trim` (below). |
| `--backhaul-model` | `mission` | `seconds`: the seconds-axis backhaul (H3 adaptive, every other arm the fixed carrier), with a loss draw keyed by seed, mule and mission. Jittery cells then lose about 16 % of uploads at the fixed carrier, against `--realism`'s flat 2 %. Not with `--l1-channel`. |
| `--contact-reliability-source` | `origin` | `channel`: the SNR gate at the stop plus the device's availability drawn on the mule. Needs `--contact-band`. |
| `--payload-bytes` | measured | Bytes per direction for the dwell and upload charge, e.g. `1000000` or `10000000` (decision D3). |
| `--deadline-bounds` | `collection` | What Deadline(j) bounds: `collection` (default), the collection, arrival + dwell; `delivery_per_stop`, each stop's own return to the dock plus the upload, checked per stop (it does not bound when the earlier stops' updates actually reach the cluster; this was `delivery` at `ef1faa1`); or `delivery`, route-level: the route's landing plus the upload meets the Deadline(j) of every update collected on it, and a stop that would land an update already on board late is refused as `delivery`. The bound is on the priced route: a contact that overruns its priced time at the stop where Pass 1 ends is not re-checked, and `mission_completed.delivery_overrun_s` records any late landing. Only H1–H3 are held to either delivery value (D1–D5 have no deadline clause). |
| `--deadline-time-scale` | `1.0` | The deadline law's time unit: a number on either clock, or `t_nom`, T_nom / 10 s, on the mission clock only (spec Q1). |
| `--initial-window-s`, `--initial-window-missions` | 60 s | Φ₀ in the law's recorded unit (either clock), or in nominal mission periods (mission clock only). |
| `--t-nom-s`, `--t-nom-layouts` | computed, 20 | T_nom, the cell's median nominal mission period, computed only when a setting needs it. |
| `--backhaul-period-s` | `n_missions` × T_nom | The seconds backhaul's period. |
| `--agg-period-t-nom` | off | `agg:cutoff`: set D5's `period_s` to T_nom. |
| `--session-ttl-s` | 3 s | The mule's wall-clock session TTL. A fit that outlasts it becomes a missed reply, so ferry cells set it to at least twice the measured real-model fit time (the pilot plan below takes its 95th percentile with N devices training at once). |
| `--rf-link-token` / `--no-rf-link-token` | on exactly with `--mission-clock sim` | One RF link token per trial (Freeze Amendment 10). |
| `--expected-input-dim` | 21 on the canonical data | A real-model ferry cell whose model has another input width is refused. |
| `--snr-floor-db`, `--altitude-m`, `--n-pl`, `--shadow-sigma-db`, `--margin-quantile`, `--contact-regime`, `--interference-period-s`, `--noise-bin-s`, `--shadow-corr-s`, `--shadow-keying`, `--cruise-speed-m-s`, `--turnaround-s`, `--listen-s`, `--energy-capacity-j`, `--p-move-w`, `--p-hover-w` | the design's | The D1–D3 physics (configuration reference §17.1–17.3). Energy figures are SIMULATED. |

On the mission clock the trial's hard kill is re-costed from the session TTL (562 s at the default
3 s with N = 6 and 4 missions), so `--trial-budget-s` no longer kills a slow but healthy ferry
trial, and the `--timeout-s` label follows it. That label is one cap for the whole run: the
largest re-costed budget over the grid, unless `--timeout-s` is given, so a trial can run past its
own budget and still be `ok`. Each mission-clock `trial_status.json` records the cap as
`soft_cap_s`, and `traces_scorer.py` without `--status-csv` applies it, so it gives the runner's
verdict. On the wall clock, pass `--status-csv` when the run used `--timeout-s`, as before.
Mission-clock traces kept before this change have no `soft_cap_s`, so score them with
`--status-csv` too.

Each `mission_completed` carries the mission's simulated record (start and end, the clock's
ledger, the stops flown, re-plans, the SIMULATED energy, the backhaul upload), and rows gain the 15
`sim_*` columns. `traces_scorer.py` scores a mission-clock trace in simulated seconds
(`sim_s_to_τ`), and refuses one whose clocks disagree. `mission_duration_s_mean` and `wall_s_to_τ`
stay wall time. On the mission clock the serve counts, coverage and Jain's index count member
contacts only, so they do not compare with wall-clock rows.

**Reading a mission-clock trace.**

- `mission_completed.pass_1_preflight_drops` lists each contact that H1–H3 dropped before takeoff,
  with its reason (`overdue`, `budget`, `energy` or, under `--deadline-bounds delivery`,
  `delivery`, listed in that order). It is always [] for D1–D5; since Phase 4, D1–D3 and D5
  report what their walk left out before takeoff in `pass_1_policy_drops` instead, never widened
  (§2.7). Under `--deadline-bounds delivery` the in-flight records can also read `delivery`:
  `aborts[].reason`, and `replans[].rejected[].reason` and `replans[].dropped[].reason`, for a
  stop that would land an update already on board after its Deadline(j), and
  `mission_completed.delivery_overrun_s` says how far the Pass-1 upload, or the landing when
  nothing was uploaded, ended past the earliest deadline of the updates on board; no CSV column
  reads it. A `mission_empty` means only that Pass 1 aggregated no update: either the plan was
  empty, or its contacts were flown and answered nothing. `mission_empty` carries only
  `mission_round` (and, under `--dock-on-empty`, `docked`); the plan fields are on the same round's
  `mission_completed`. If that `mission_completed` has `pass_1_contacts` 0 (an empty
  `pass_1_plan`) and a non-empty `pass_1_preflight_drops` whose entries all read `budget`, no
  contact fits the budget on its own: each was priced from the dock at takeoff, and the budget is
  below each contact's predicted home time. With `pass_1_contacts` above 0, the drops say nothing
  about the contacts that were flown.
- A flown stop's `snr_db` and `rate_bps` are read at arrival, while its `dwell_s` is priced at each
  target's own session start (critic C2), so they do not reproduce `dwell_s`.

**The pilot plan** (decided by the user on 2026-09-29; no pilot has run yet).

- *Deadline unit:* `--deadline-time-scale t_nom` (T_nom / 10 s). With the default Φ₀ each device
  then starts with about six missions' worth of window, as in the recorded runs (spec Q1).
- *Session TTL:* `--session-ttl-s` at least 2× the 95th percentile of the real model's
  `train_offline` time, measured at the exit gate's concurrency (N devices training at once).
- *Budget knee:* an H1 sweep of `--mission-budget-s` on `--contact-band wide` with the measured
  payload and the `t_nom` unit, for each N of the gate's grid. The knee is where the served fraction
  stops rising. Narrow and medium knees wait for Study 5.4 (the cliff, below).
- *In-flight response:* `--in-flight-response replan --replan-fallback trim`. Each arm keeps its own
  order, as the D arms do; `reorder` would make H1–H3 fly the same route whenever the pre-flight
  check fires.
- *An idea on record, not in the plan:* a pilot that runs the knee sweep under both `trim` and
  `reorder` and chooses by how often the pre-flight check fires and what each costs in coverage.
- *Open follow-up:* the real-model smoke test (`test_exp4_real_model_synthetic_converges`) fails
  with `rounds_closed` 0 under load and passes on an idle host; the user signed off the baseline
  that records it on 2026-09-29. If the session-TTL pilot shows the cause is a device's fit
  outrunning the 3 s TTL under load, the test is fixed then.

**Pilot notes** (final cross-cutting check, Freeze §5j).

- *Set the deadline unit.* At the default `--deadline-time-scale 1.0` the law's constants (Φ₀ =
  60 s, −5 s / +10 s steps) are sized for missions of about 10 s of wall clock, while a simulated
  mission lasts minutes. In a 14-mission probe (narrow, 1 MB, N = 8, a 200 s budget) H1 served the
  field-wide contact once and then found it overdue at every later takeoff, by 20 s more each
  mission: the +10 s widening per miss never catches up with the clock. Set the unit on
  simulated-clock cells; the pilot plan sets `t_nom`, as the example below does (spec Q1).
- *The narrow-band cliff.* With `--contact-band narrow` (or medium), `--payload-bytes` set and a
  `--mission-budget-s` below the field-wide contact's predicted home time, every gated arm flies
  empty missions (30 s turnarounds, `rounds_closed` 0). For H1–H3, `pass_1_preflight_drops` shows
  the budget drop; D1–D5 record [] there, and since Phase 4 D1–D3 and D5 name the contact in
  `pass_1_policy_drops` (§2.7). The budget is not the only clause that can empty them. At
  `--deadline-time-scale 1.0` (Φ₀ = 60 s) the contact's predicted finish (94.9 s after takeoff at
  N = 8, 1 MB) can be past Deadline(j): H1–H3 then drop it as `overdue` at any budget, above the
  knee too, until missed missions widen its window past that finish, while D1–D3 and D5 keep the
  budget cliff. With `--deadline-time-scale t_nom` (Φ₀ = 6 × T_nom, 1500 s in trial T2) the
  budget binds first. Measure the knee per band, payload, N and deadline unit (Configuration
  Reference §17.1). Decided 2026-09-29: the cliff waits for Phase 4's member-subset admission, and
  until then narrow and medium cells are not compared under budgets below it. Phase 4 lands it
  behind `--member-admission` (§2.7): the plan arms fly `subset` by default, and H1–H3, D1–D3 and
  D5 fly it when a run asks; `whole`, their default, keeps the cliff.
- *Missions run longer than planned.* The planner prices the mean SNR and dwell is convex in it,
  so realized missions ran longer than predicted by +0.8 % on average on wide at 1 MB, +7.1 % on
  wide at 10 MB and +19.2 % on narrow at 1 MB. Budgets bind in flight more often on narrow bands
  and large payloads.
- *Do not restart a mule by hand during an ordered trial* (several mules on `--mission-clock sim`
  below a full quorum, or under `agg:fedbuff`). The cluster ignores the restarted mule's stale
  dock markers, but an upload the crashed mule sent can still make the restarted mule's uploads
  fold late (`sim_order_late`), and an upload it left held can stall the trial until a mule's DOWN
  wait runs out, which under the driver is the whole trial's wall budget. The driver never
  restarts a mule.

A Phase 3 exit-gate cell (H1 on the simulated clock, wide band, a budget; repeat it for D1, D2, D3
and D4 with the same seeds, and at a stress budget). The pilots come first: they re-measure the
budget knee with the ferry model and the real-model fit time, so set `KNEE_S` and `TTL_S` from
them. The cell flies the pilot plan's `--in-flight-response replan --replan-fallback trim` and
keeps the other defaults (the `mission` backhaul, the `origin` reliability source, the measured
payload, `--deadline-bounds collection`); add `--backhaul-model seconds` or
`--contact-reliability-source channel` where a study chooses them (the pilot plan sets neither).
No Phase 3 result has been recorded yet.

```bash
# KNEE_S: the budget knee from the Phase 3 pilot's H1 sweep; TTL_S: at least 2x the 95th-percentile real-model fit time with N devices training at once.
python -m experiments.exp4.runner_main --csv results/exp5_p3/h1_sim_wide_knee.csv --arms H1 --N 6 --n-missions 4 --regime jittery --n-trials 40 --real-model --realism --mission-budget-s "$KNEE_S" --mission-clock sim --contact-band wide --deadline-time-scale t_nom --in-flight-response replan --replan-fallback trim --session-ttl-s "$TTL_S" --keep-event-traces
```

### 2.7 The plan clock (FeRRy Phase 4)

Every default is the recorded run; see `HERMES_Configuration_Reference.md` §18 for each value and
Freeze §5k for what changed. The plan arms run only when named with `--arms`, and only on
`--mission-clock sim`: the default arm list is still the nine Phase 3 arms (H0 is dropped from it on
the simulated clock). The trial CSV header is unchanged, but write every Phase 4 run to a fresh CSV
path, and every setting of a sweep to its own: the runner skips (cell, arm, trial) keys already in a
file, and none of the Phase 4 flags is part of the key, so a second setting written to the same file
would silently keep the first's rows.

| Flag | Default | Meaning |
|---|---|---|
| `--arms F FX FB+wide FB+medium FB+narrow F-cov F-cap F-prio` | the nine Phase 3 arms | The plan arms (below). The runner refuses one the driver cannot run (on the wall clock, without a band, with `--pass-2-budget`, with `abort` and a cap, ...) before any trial, as a usage error. |
| `--member-admission {whole,subset}` | each arm's own | `subset` lets a stop that fails whole admit the members that still fit. The plan arms fly `subset` unless the run says `whole` (F under `whole` keeps the narrow-band cliff, for comparison); H1–H3, D1–D3 and D5 fly `whole` unless the run says `subset`; D4 always flies `whole`. The setting applies to every arm of the run. |
| `--age-cap-missions S` | off | The age cap: a device whose update has not reached the model for S of its mule's missions must be served. Set it from the S\* tool (below). F-cap always runs without it. |
| `--age-cap-lookahead L` | 0 | A device is capped from age S − L. Leave it at 0. |
| `--plan-score-params JSON` | {} | The plan score's settings as a JSON object, e.g. `'{"c_cov_per_device": 0.25, "c_energy": 0, "coverage_rank": "weighted"}'` for one cell of the κ sweep. Unknown keys are refused. F-cov sets its coverage term off on top. |
| `--plan-search-params JSON` | {} | The search's bounds (`exact_max_devices`, `exhaustive_max_stops`, `heuristic_max_passes`, `heuristic_max_evaluations`). Leave them at the defaults: at N = 6 the search is exact. |
| `--base-seed` | 42 | The salt of every trial's seed. A pilot takes a base seed of its own (decision 7), or its trials are the headline's first ones. |

| Arm | What it is | Notes |
|---|---|---|
| `F` | The plan search over every band class, with the committed order in flight | Phase 4's F: the learned pair choice flies as FQ (§2.8). |
| `FX` | F with the cross-heuristic in flight: after each Pass-1 stop the nearest stop that keeps the rest feasible, and on arrival the fastest class that still reaches every device b̄ reaches | The exit gate's arm. At the measured payload it flies exactly as F (critic B6). |
| `FB+wide`, `FB+medium`, `FB+narrow` | The plan search pinned to one class (Path B+) | Fly their own class whatever `--contact-band` says. |
| `F-cov` | F without the coverage term | Serves only what the cap forces: report it as "cap-only service" (decision 3). |
| `F-cap` | F without the age cap | |
| `F-prio` | F whose coverage weight is the age alone, without the miss streak | `--miss-priority` does not reach the plan arms: each sets its own, and the row records it. |

**What a plan arm needs.** `--mission-clock sim` and a `--contact-band`, which is the reference
class for the arms that search the classes (and the band of every H and D arm in the same CSV).
Every plan arm flies the `trim` fallback, whatever `--replan-fallback` says, and gets T_nom per cell
(`--t-nom-s` gives it instead), since its score measures the whole mission against T_nom. A cap
needs `--in-flight-response replan`: `abort` gives up capped stops and is refused with a cap.
`--pass-2-budget` is refused.

**The pilot plan** (decision 7 of 2026-09-30; no pilot has run). Nothing runs before the user's
go-ahead, and not before the Phase 3 pilot has set the session TTL and the knee (§2.6).

- *Common settings:* N = 6, jittery; `--mission-clock sim --contact-band wide --deadline-time-scale
  t_nom --in-flight-response replan --replan-fallback trim --aggregation agg:cutoff
  --contact-reliability-source channel`; a base seed of the pilots' own; fresh CSV paths; n = 20
  for every stub pilot (the exit gate's).
- *The real-model FX smoke* (the design's pilot table, with decision 7's payload): FX alone, at
  1 MB, at the knee, 4 missions, n = 5, once the Phase 3 pilot has set the session TTL.
- *Budgets:* the stub FX smoke at the gate's 30 s and 60 s; the 5.4 and 5.8 pilots at the knee and
  at a stress budget, half the knee rounded to 5 s. The knee is the Phase 3 pilot's H1 sweep on wide,
  plus the same sweep at 1 MB, both under the pilots' own `channel` reliability source (critic C4).
- *Payloads:* 5.4 at the measured payload and at 1 MB; 5.8 and the cap check at 1 MB (at the
  measured payload F serves every device every mission, so the cap and the coverage term never
  bind); the real-model FX smoke at 1 MB.
- *Missions:* 5.8 at 4, the headline's, and at 8, because the cap binds only from mission S on (the
  headline's count is fixed after it); the cap check at 8; everything else at 4.
- *Arms* (the design's pilot table): the stub FX smoke F and FX; 5.4 F, FB+wide, FB+medium and
  FB+narrow; 5.8 F, F-cov, F-cap, F-prio, FB+wide, D1, D3 (uniform weights, the default) and D4
  route-only (`agg:cutoff`, as the common flags give it); the cap check (build-plan decision D4) F
  at S − 1, S and S + 1, one CSV each.
- *The cap:* S comes from the S\* tool at the knee and the stress budget, before the pilots (below).
- *Cap violations* are reported by cause, with no pass mark: device availability alone makes about
  15 % of device-missions miss at S = 3 (critic A2).
- *"Behaves" means:* every trial ends ok; the predicate holds at every departure; FB+c flies only
  class c; F's plan key is never above the best FB+'s (critic A3); a repeated trial gives identical
  traces bar wall stamps (`plan_wall_s` is one); planning takes at most 1 s per mission
  (`plan_wall_s`).
- *Cost:* about 1,400 stub trials, to be re-costed before the go-ahead
  (`experiments/exp4/cost_matrix.py` knows no plan arm).

**The κ sweep** (decision 2): κ in {0.15, 0.25, 1} and c₄ in {0, 0.1}, reporting the plans that fly
empty, under `"coverage_rank": "weighted"`. Under the default `lexicographic` rank κ only orders
plans that serve the same weight share, so the sweep would not show the trade decision 2 priced;
`weighted` ranks by V alone (R11). Each (κ, c₄) is its own run with its own CSV; the setting is not
a grid axis, so the seeds are the same across the sweep and the cells stay paired.

**The S\* tool** (`experiments/analysis/age_cap_s_star.py`, planning level only: nothing is flown).
It prints, per arm family and budget, the S\* that 90 % of its 30 reference layouts need, each
family's S (i) and S + 1 (ii), and the cell's S, F's, never below 2 (decision 1). Give it the cell's
settings: the pilots' budgets, the payload, the band classes, and `--theta-bytes 18756` for a
real-model cell at the measured payload (its default is the stub's θ). It prices the realism field,
as runs with `--realism` lay devices out (`--no-realism` prices the tight cluster). An (ii) it
cannot vouch for carries its caveat: for a pinned class, at a budget outside the 45–90 s measured,
or under whole admission. Today, at 1 MB with the spec's prior budgets of 90 and 45 s, it gives F
an S of 2 (FB+wide 4, FB+medium 3, FB+narrow 3), and at the measured payload F's S\* is 1, so S is
2 by the floor; the pilots re-run it at the measured knee.

```bash
# KNEE_S and STRESS_S: the Phase 3 pilot's knee at 1 MB, and half of it rounded to 5 s.
python -m experiments.analysis.age_cap_s_star --budgets "$KNEE_S" "$STRESS_S" --payload-bytes 1000000 --contact-band wide --regime jittery --json results/exp5_p4/s_star_1mb.json
```

A 5.8 pilot cell at the stress budget (S from the tool; PILOT_SEED a base seed of the pilots' own;
TTL_S the Phase 3 pilot's session TTL). Repeat it at the knee, and at 4 missions; the plan arms and
the D arms of one invocation share the cell's seeds:

```bash
python -m experiments.exp4.runner_main --csv results/exp5_p4/s58_pilot_stress_8m.csv --arms F F-cov F-cap F-prio FB+wide D1 D3 D4 --N 6 --n-missions 8 --regime jittery --n-trials 20 --base-seed "$PILOT_SEED" --realism --mission-budget-s "$STRESS_S" --payload-bytes 1000000 --mission-clock sim --contact-band wide --deadline-time-scale t_nom --in-flight-response replan --replan-fallback trim --aggregation agg:cutoff --contact-reliability-source channel --age-cap-missions "$S" --session-ttl-s "$TTL_S" --keep-event-traces
```

One point of the κ sweep (κ = 0.25, c₄ = 0):

```bash
python -m experiments.exp4.runner_main --csv results/exp5_p4/kappa_0.25_c4_0.csv --arms F --N 6 --n-missions 4 --regime jittery --n-trials 20 --base-seed "$PILOT_SEED" --realism --mission-budget-s "$STRESS_S" --payload-bytes 1000000 --mission-clock sim --contact-band wide --deadline-time-scale t_nom --in-flight-response replan --replan-fallback trim --aggregation agg:cutoff --contact-reliability-source channel --age-cap-missions "$S" --plan-score-params '{"c_cov_per_device": 0.25, "c_energy": 0, "coverage_rank": "weighted"}' --session-ttl-s "$TTL_S" --keep-event-traces
```

**Reading a plan-mode trace.**

- `mule_ready` states the plan settings the scheduler runs, the score's and search's resolved.
  `mission_completed.plan` is the mission's closed plan: `band` (b̄, the class committed), `search`
  (the search mode), `per_class` (each class's best: `v`, `served`, `cap_key`, `served_share`, and
  the rank applied), `score` (`v`, the predicted whole mission `mission_s`, and the terms),
  `demand`, `weights`, `served`, `cap` (S, the ages, the capped devices and the violations, each
  with its device, planning age and cause) and `visited`. `plan_wall_s`, beside it, is wall time:
  leave it out of any determinism comparison.
- `mission_completed.band` is b̄, and `pass_1_flown[].band` the class each stop was flown on: under
  FX a Pass-1 stop can differ from b̄, while Pass 2 always flies b̄. The scorer's `band_shares`
  counts per stop.
- `pass_1_preflight_drops` can read `plan`: a demanded device the plan left out by choice, whose
  offered stop fits alone from the dock at takeoff. It is widened like any drop but stays out of
  S3c's planned count. A clause (`overdue`, `budget`, `energy`) means the device does not fit even
  alone there; under `whole` the clause judges its whole stop.
- A hover stop is not marked: it is a one-device Pass-1 stop away from its device, often at the dock
  or at the class's reach edge.
- D1–D3 and D5 on the simulated clock record what their walk left out before takeoff in
  `pass_1_policy_drops` (`"widened": false`), only when they left something out; those devices are
  never widened.
- The violations, by cause: `unplannable` means no class the arm may fly serves the device alone
  within the budget, even at its best hover point, which is physics for the time budget (under an
  energy capacity it reflects the time-minimising point; the pilots set none); `crowded` means some
  class could but the plan chose otherwise, the planner's myopia or partition drift (below);
  `dropped_in_flight` means its stop was dropped or trimmed in flight; `not_merged` means it was
  flown but its update did not reach the merge (no reply, an availability failure, a cutoff).
- `plan.score.mission_s` is a prediction: Pass 2 is priced on the class's S3a stops at the dock, and
  the mule rebuilds Pass 2 after Pass 1, so the flown mission (`sim_end_s − sim_start_s`) can differ
  by a few seconds either way.
- Score every arm at one S with `traces_scorer.py --age-cap-s S`: `cap_violations` counts (device,
  mission) pairs aged at least S after the mission, for every arm, H and D arms and F-cap included,
  while `cap_violation_events` is the plan-mode mule's own log by cause. The two differ by lost
  backhaul uploads and merges the cluster defers. `plan_served_share_mean` counts devices, not
  weights; `far_served_share` counts the devices beyond `--rrf` of the dock.

**Pilot notes** (from the build, its review and the hover decision; Freeze §5k).

- *The hover point's place.* It minimises time, so a capped far device is often served from the dock
  or from the class's reach edge, where the noisy link is weakest. Expect more in-flight misses
  there: in the pilots' configuration FB+medium's `not_merged` at 45 s rose from 228 to 245 with the
  hover stops, and F's `dropped_in_flight` at 30 s from 0 to 12. A margin inside the reach is a
  pilot-time choice.
- *Empty missions before the cap binds.* Uncapped devices keep S3a's stops, so a mission can fly
  empty while nothing is capped, most at 30 s (on the S\* tool's layouts at 30 s: 5, 18 and 6 empty
  plans for FB+wide, FB+medium and FB+narrow, none with a capped device). F under
  `--member-admission whole` can fly empty at 30 s while a capped device fits alone at its hover
  point.
- *Pass 2* still delivers at S3a's stops, so a far device is visited at its own position on the way
  back out.
- *Partition drift.* S3a re-clusters every mission, so a capped device's stop can serve it alone but
  not beside another capped device: at S\*+1 FB+medium crowded 7 times in 291 missions at 45 s and
  FB+wide twice in 360, and at 30 s every family crowds (F 20 times in 309). Read the FB+ arms'
  `crowded` counts at the stress budget, and every arm's at 30 s, with this in mind.
- *Longer missions under the default rank.* Time only breaks ties among plans that serve the same
  weight, so a plan can take a much longer Pass 2 to serve one more device (layout 18, mission 4:
  182 s against 118 s under `weighted`). The served share is nominal: a member at a class's edge
  counts fully, though its outage on the jittery channel is about 0.15–0.2.
- *Deadline windows span more of F's missions.* T_nom, the deadline unit, stays priced on wide for
  every arm (critic C4), so an arm with shorter missions fits more of them into a deadline window
  (the critic's estimate: F's narrow missions about 55 s against a T_nom of 172–250 s, about 4×
  as many as a wide arm's). Documented, not corrected: compare deadline-driven counts between F
  and the wide arms with this in mind.
- *Planning time.* At N = 6 the search is exact and a plan takes at most about 0.25 s; the "1 s per
  mission" mark does not extend to large N (at N = 96 on a 500 m field a plan took 3.1–3.4 s).
- *Not yet built:* Study 5.4's sweep knobs (unit U10, before the 5.4 headline; the pilots run the
  default layouts, where 0.68 of the devices already lie beyond 60 m at N = 6), and F·round and
  F·pref (unit U11, with Study 5.2).

### 2.8 The flight clock's pair score, FerrySim and E3 (FeRRy Phase 5)

Every default is the recorded run; see `HERMES_Configuration_Reference.md` §19 for each value and
Freeze §5l for what landed. The Phase 5 arms run only when named with `--arms`, and only on
`--mission-clock sim`: the default arm list is still the nine Phase 3 arms. A learned arm flies only
a trained checkpoint that the runner has checked before any trial; no learned arm flies random
weights. The trial CSV header is unchanged, and none of the Phase 5 flags is part of the (cell, arm,
trial) key, so write every setting to its own CSV (each checkpoint, each Study 5.6 period, each
Study 5.7 grid point): a second setting written to the same file would silently keep the first's
rows.

**Nothing in this section has run.** Each command below was checked with a parse-only probe (the
argument parsers and the runner's pre-trial checks, on stand-in checkpoints, with no trial, no
episode and no training), and none was run. What needs the user's go-ahead (Freeze §5l); nothing
below runs, and nothing is committed, without it:

1. *The training campaigns:* 77–119 trainings, about 11–22 h plus 1–2 h of held-out evaluation
   (R20 puts Study 5.5's held-out evaluation at about 3.3 h at 8 workers with the N = 6 control),
   run in batches, the calibration and the controls first.
2. *E3's training settings* (R16): the pair learner's defaults, or Chen's lr 5e-4 and ε from 1.0;
   `--gamma` is required.
3. *The jittery score's family* (R22, R29): `jittery` or `jittery56`; the orchestrator recommends
   `jittery56` with `--val-episodes 400`.
4. *The LICENSE* (MIT, as the README names) before any checkpoint commit, and each checkpoint
   commit (decision 9).
5. *The pilots:* the N = 12 knee and stress budgets, after the Phase 3/4 pilots; then the cells,
   the 5.6 lags and the 5.6 periods are re-pinned.
6. *The stack trials* (Study 5.5's check, 5.6, 5.7, 5.3's E3 cells), re-costed first.
7. *The campaign gate,* in decision 10 (ii)'s order: the headroom report (run during the build),
   the sweep and its evaluation, the committed checkpoints, Study 5.5's verdict, the stack trials.
8. *R14's candidate amendment.*

**The two gates** (decision 10 (ii)). The code gate (every test, FerrySim's parity tests among them)
is met once the full suite is the same as all three baselines (Freeze §5l, "Test baselines and the
compare rule"). The campaign gate follows the go-ahead, in item 7's order.

| Flag | Default | Meaning |
|---|---|---|
| `--arms FQ FQ-hand FQ-dwell FQ-cov FQ-g0 FQ-g25 FQ-g50 FQ-g75 FQ-g90 FQ-g99 E3 H1+L1` | the nine Phase 3 arms | The Phase 5 arms (below). The runner refuses one the driver cannot run (on the wall clock, without its checkpoint, an FQ arm without `replan`, `H1+L1` without the adaptive backhaul) before any trial, as a usage error. |
| `--pair-checkpoint TAG=PATH` | none | Repeatable. The checkpoint (the `.npz`, its manifest `.json` beside it) that the FQ arm of TAG flies: `main` (FQ), `hand`, `dwell`, `cov`, `g0`, `g25`, `g50`, `g75`, `g90`, `g99` (FQ-g0 … FQ-g99). |
| `--policy-checkpoint E3=PATH` | none | E3's checkpoint. |
| `--allow-dirty-checkpoint` | off | Fly a checkpoint trained from a dirty tree: for development and tests, not for a campaign. It lifts only that refusal. |
| `--require-trained` | off | Refuse H2 and H3 without `--selector-weights` (decision 8 (a)); they stay in the default arm list though they leave Exp 5. |
| `--interference-period-s P` | the design's 60 s | Phase 3's contact-channel period P_c (§2.6): Study 5.6's setting on the stack (below). |

| Arm | What it is | Its checkpoint | Notes |
|---|---|---|---|
| `FQ` | F with the learned (band, next stop) score in the flight slot | `main`: the derived reward, at whatever weights its manifest records (R28) | The build plan's F; Phase 4's F keeps the committed slot, and the paper may call FQ "F". |
| `FQ-g0` … `FQ-g99` | FQ at γ = 0, 0.25, 0.5, 0.75, 0.9 or 0.99 | `gX`: γ = X/100, the derived reward at decision 4 (a)'s weights (c_t 0.1, c_cov 1) | Study 5.5's stack check flies the verdict's two picks. |
| `FQ-hand` | FQ trained on F·hand | `hand`: an F·hand checkpoint, which no other tag takes | Study 5.7. |
| `FQ-dwell`, `FQ-cov` | FQ with the plan score's dwell term, or its coverage term, off | `dwell`, `cov`: trained on that plan (`train --ablation dwell` or `cov`) | Study 5.7, only if Study 5.5 keeps the learned score (critic C2). It did not (6 Oct 2026), so 5.7 flies FX-dwell and FX-cov. |
| `FX-dwell`, `FX-cov` | FX with the plan score's dwell term, or its coverage term, off | none | Study 5.7 (added 7 Oct 2026, as Study 5.5 kept FX). Plan arms (FQ-dwell's and FQ-cov's score changes, FX's slot); `--mission-clock sim`. |
| `E3` | Chen et al.'s DQN as a whole scheduler that names each next stop in flight | `E3` (tag `e3`): trained on E3's bytes reward | Legacy mode: whole stops whatever `--member-admission` says, a `--contact-band`, the run's in-flight response. |
| `H1+L1` | H1 with H3's adaptive backhaul | none | Needs `--backhaul-model seconds` or `--l1-channel` (R27); it replaces H3 as the adaptive backhaul's reference (decision 8). |

**What a learned arm needs.** An FQ arm needs what F needs (§2.7: `--mission-clock sim` and a
`--contact-band`; it flies the `trim` fallback and T_nom per cell) and also `--in-flight-response
replan`, which is not the runner's default (R3: its mask folds the whole rest of the flight, which
only the re-plan's departure check folds next). E3 needs `--mission-clock sim` and a
`--contact-band`. Each checkpoint given is checked before any trial, whichever arms run:

- it is verified (the arrays against the manifest's sha, which binds the kind, the purpose, the
  learner's revision, the schema and the classes) and is of the flag's kind (`pair_q`, `chen_dqn`);
- it is trained (purpose `trained`, at least one episode), scored on the held-out runs (`evaluate
  --record`) and from a clean tree, unless `--allow-dirty-checkpoint` (critic B9);
- it is its tag's (R24): `gX`'s γ, the tag's reward (F·hand for `hand`, bytes for `e3`, the derived
  reward for the rest), decision 4 (a)'s weights under `gX`, `dwell` and `cov`, and kept weights that
  took an update;
- a pair checkpoint trained on the plan its arm flies under this run's flags (R23: FQ-dwell's or
  FQ-cov's change, and `--plan-score-params`).

The driver then reads each file once, keeps its sha for the run, and before each trial loads it as
the mule will, so a checkpoint re-saved at the same path is refused before the next trial: save new
weights under a new path. The row records the tag and the sha (`pair_tag` and `pair_sha256` in
`ferry_params`; `policy_tag` and `policy_sha256` in `policy_params` for E3), never the path; the sha
finds the manifest, which holds γ, the reward and the rest.

**FerrySim** (`python -m experiments.ferrysim`; configuration reference §19.9). The commands below
are NOT RUN: each waits for the go-ahead, and `train` and `sweep` write under the repository's
`results/exp5/checkpoints/` unless `--root` says otherwise. A development smoke run outside the
repository (SCRATCH: a directory of your own):

```bash
python -m experiments.ferrysim train --kind pair_q --family jittery --study smoke --gamma 0.9 --seed 0 --episodes 50 --eval-every 25 --val-episodes 8 --root "$SCRATCH/ferrysim-checkpoints" --allow-dirty
```

The runner refuses that checkpoint: it has no held-out score, one trained from a dirty tree (which
`--allow-dirty` records) needs `--allow-dirty-checkpoint`, and if its replay never warmed (1,000
transitions) its kept weights took no update (R24; an open item asks for a `--warmup-transitions`
flag).

*The headroom report* (decision 10 (i)(a)), on the validation stream. The build ran it (Freeze §5l:
no pause, ε = 0.01 in every cell; 49 min at 12 workers, written outside the repository). Re-run it
into the results tree after the re-pin, and with `--plan-score-params` when a sweep trains on a
pilot's plan, since `report` refuses a headroom report flown on another plan than the evaluation's:

```bash
python -m experiments.ferrysim headroom --episodes 200 --workers 12 --out results/exp5/headroom/headroom.json
```

*The calibration* (other choices 12): γ ∈ {0, 0.9} × 3 seeds per family, before the sweep, each
family under a study of its own (R19), the jittery one on the family its sweep will fly (R29). Shown
for `jittery56` with `--val-episodes 400`, the recommended setting of go-ahead item 3; for `jittery`,
name that family and study and leave `--val-episodes` at its default:

```bash
python -m experiments.ferrysim sweep --study 5.5-calibration-jittery56 --family jittery56 --gammas 0 0.9 --seeds 0 1 2 --val-episodes 400 --workers 6
python -m experiments.ferrysim sweep --study 5.5-calibration-clean --family clean --gammas 0 0.9 --seeds 0 1 2 --workers 6
python -m experiments.ferrysim evaluate --checkpoints results/exp5/checkpoints/5.5-calibration-jittery56 --workers 8 --out results/exp5/calibration/jittery56_evaluation.json
python -m experiments.ferrysim report --evaluation results/exp5/calibration/jittery56_evaluation.json --headroom results/exp5/headroom/headroom.json --out results/exp5/calibration/jittery56_verdict.json
python -m experiments.ferrysim evaluate --checkpoints results/exp5/checkpoints/5.5-calibration-clean --cells cln-n12-90 cln-n12-180 --workers 8 --out results/exp5/calibration/clean_evaluation.json
python -m experiments.ferrysim report --evaluation results/exp5/calibration/clean_evaluation.json --headroom results/exp5/headroom/headroom.json --cells cln-n12-90 cln-n12-180 --out results/exp5/calibration/clean_verdict.json
```

A calibration verdict reads "NOT pre-registered" (R26) and is decided by the same rule; with 3 seeds
it cannot read "rising". If the γ = 0 sanity check fails, the learner may be revised once, on the
control cells, before the sweep (decision 5).

*Study 5.5's sweep*, its held-out evaluation (`--record` writes each checkpoint's held-out score
into its manifest, which the runner needs; about 3.3 h at 8 workers with the N = 6 control, R20) and
the verdict:

```bash
python -m experiments.ferrysim sweep --study 5.5-jittery56 --family jittery56 --gammas 0 0.25 0.5 0.75 0.9 0.99 --seeds 0 1 2 3 4 5 6 7 8 9 --val-episodes 400 --workers 8
python -m experiments.ferrysim evaluate --checkpoints results/exp5/checkpoints/5.5-jittery56 --record --workers 8 --out results/exp5/s55/evaluation.json
python -m experiments.ferrysim report --evaluation results/exp5/s55/evaluation.json --headroom results/exp5/headroom/headroom.json --out results/exp5/s55/verdict.json
```

The verdict gives the outcome (`rising`, `flat`, `inconclusive` or `sanity-failed`), whether FX is
replaced, whether `greedy_1` beat the FX arm by ε (then the user decides, critic A2), and the stack
check's picks (`stack_check`: `best_gamma`, `best_gamma_seed` and `gamma0_seed`, each γ's
median-validation seed).

*E3* (go-ahead item 2). E3_GAMMA is the γ chosen there; for Chen's settings add `--lr 5e-4
--epsilon-start 1.0`. E3's own score is read in bytes (R18):

```bash
python -m experiments.ferrysim sweep --kind chen_dqn --study 5.3-e3 --family jittery --gammas "$E3_GAMMA" --seeds 0 1 2 3 4 --workers 5
python -m experiments.ferrysim evaluate --checkpoints results/exp5/checkpoints/5.3-e3 --reward bytes --record --workers 8 --out results/exp5/e3/evaluation.json
```

*Study 5.7's scores* (only if Study 5.5 keeps the learned score; GAMMA_STAR is the verdict's best
γ): F·hand, the two plan-term ablations, and one point of the weight grid (c_t ∈ {0.03, 0.1, 0.3} ×
c_cov ∈ {0.25, 1, 4}, each point under a tag of its own that no arm flies; a grid checkpoint flies as
`main`). The ablations train on their arms' plans (R23), so they are evaluated without the
references, which fly the cells' own plan (or beside references given that plan with
`--plan-score-params`):

```bash
python -m experiments.ferrysim sweep --study 5.7-jittery56 --family jittery56 --reward hand --gammas "$GAMMA_STAR" --seeds 0 1 2 3 4 --val-episodes 400 --workers 5
python -m experiments.ferrysim sweep --study 5.7-jittery56 --family jittery56 --ablation dwell --gammas "$GAMMA_STAR" --seeds 0 1 2 3 4 --val-episodes 400 --workers 5
python -m experiments.ferrysim sweep --study 5.7-jittery56 --family jittery56 --ablation cov --gammas "$GAMMA_STAR" --seeds 0 1 2 3 4 --val-episodes 400 --workers 5
python -m experiments.ferrysim sweep --study 5.7-jittery56 --family jittery56 --c-t 0.3 --c-cov 4 --tag ct0.3-cc4 --gammas "$GAMMA_STAR" --seeds 0 1 2 --val-episodes 400 --workers 3
python -m experiments.ferrysim evaluate --checkpoints results/exp5/checkpoints/5.7-jittery56/hand --reward hand --record --workers 8 --out results/exp5/s57/hand_evaluation.json
python -m experiments.ferrysim evaluate --checkpoints results/exp5/checkpoints/5.7-jittery56/dwell results/exp5/checkpoints/5.7-jittery56/cov --no-references --record --workers 8 --out results/exp5/s57/ablations_evaluation.json
```

*The re-pin* (go-ahead item 5). After the N = 12 pilot, the cells' budgets, `STUDY_5_6_LAGS_S`, the
four P_c constants, the `jittery56` hash and the test literals move together (R29), and the lag is
re-measured on the same sample and statistic (a code edit, not a flag). `scripts/exp5/repin.py`
makes it from params.toml's pilot outputs: it rewrites `experiments/ferrysim/cells.py`'s re-pin
block (the N = 6 and N = 12 budgets, the caps, the lags) and the tests' re-pin pins, renames the
cells (whose names carry the budget) across the code and tests, re-measures the lag with
`evaluate.fx_lag_median`'s sample and statistic, derives the ratio check's bounds from the same
sample, and checks each Study 5.6 cell's S* at its own period; `--dry-run` shows the plan:

```bash
python scripts/exp5/repin.py --workers 8
```

**Stack trials** (go-ahead item 6; NOT RUN, each study re-costed first: the spec estimated 320
trials for Study 5.5's check, about 1,300 for 5.6 and 400 for 5.7). They fly FerrySim's cell
settings on the stack, here at N = 12 and the 120 s stand-in (repeat at 180 s, to its own CSV), with
stub devices as in §2.7 (`--real-model` for a real-model run). TTL_S is the Phase 3 pilot's session
TTL. Study 5.5's check, with the verdict's picks (here γ = 0.9 from seed 3 and γ = 0 from seed 5):

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s55/stack_120.csv --arms FQ-g90 FQ-g0 FX F --N 12 --n-missions 4 --regime jittery --n-trials 40 --realism --mission-budget-s 120 --payload-bytes 1000000 --mission-clock sim --contact-band wide --contact-regime jittery --deadline-time-scale t_nom --in-flight-response replan --replan-fallback trim --aggregation agg:cutoff --contact-reliability-source channel --age-cap-missions 2 --session-ttl-s "$TTL_S" --pair-checkpoint g90=results/exp5/checkpoints/5.5-jittery56/g90/g0.9_s3.npz --pair-checkpoint g0=results/exp5/checkpoints/5.5-jittery56/g0/g0_s5.npz --keep-event-traces
```

*Study 5.6* (decision 6 (a); R22): the same cell with the contact channel's period at 4 × or 2 × the
lag, `--interference-period-s` 104 (the quarter cell) or 52 (the half) at 120 s, and 136 or 68 at
180 s, beside the clean control (`--contact-regime clean`, no period), each to its own CSV. The
period is not a grid axis, so the runner's seeds pair the cells. N_56 is the trials per cell, fixed
when 5.6 is re-costed. The quarter cell at 120 s, the kept score flying as FQ (if FX stays, 5.6 is E3
against FX):

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s56/q_120.csv --arms FQ FX E3 --N 12 --n-missions 4 --regime jittery --n-trials "$N_56" --realism --mission-budget-s 120 --payload-bytes 1000000 --mission-clock sim --contact-band wide --contact-regime jittery --interference-period-s 104 --deadline-time-scale t_nom --in-flight-response replan --replan-fallback trim --aggregation agg:cutoff --contact-reliability-source channel --age-cap-missions 2 --session-ttl-s "$TTL_S" --pair-checkpoint main=results/exp5/checkpoints/5.5-jittery56/g90/g0.9_s3.npz --policy-checkpoint E3=results/exp5/checkpoints/5.3-e3/e3/g0.9_s0.npz --keep-event-traces
```

*Study 5.7* (FQ against FQ-dwell, FQ-cov and FQ-hand, with D4 as the travel-only reference):

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s57/ablations_120.csv --arms FQ FQ-dwell FQ-cov FQ-hand D4 --N 12 --n-missions 4 --regime jittery --n-trials 40 --realism --mission-budget-s 120 --payload-bytes 1000000 --mission-clock sim --contact-band wide --contact-regime jittery --deadline-time-scale t_nom --in-flight-response replan --replan-fallback trim --aggregation agg:cutoff --contact-reliability-source channel --age-cap-missions 2 --session-ttl-s "$TTL_S" --pair-checkpoint main=results/exp5/checkpoints/5.5-jittery56/g90/g0.9_s3.npz --pair-checkpoint dwell=results/exp5/checkpoints/5.7-jittery56/dwell/g0.9_s0.npz --pair-checkpoint cov=results/exp5/checkpoints/5.7-jittery56/cov/g0.9_s0.npz --pair-checkpoint hand=results/exp5/checkpoints/5.7-jittery56/hand/g0.9_s0.npz --keep-event-traces
```

*Study 5.3's E3 cells* (a 3-mule, 60 s cell like §2.5's, here jittery and on the simulated clock,
with E3 and `H1+L1`):

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s53/k3_b60_e3.csv --arms E3 H1 H1+L1 D1 D3 --N 18 --n-mules 3 --n-missions 4 --regime jittery --n-trials 40 --real-model --realism --mission-budget-s 60 --mission-clock sim --contact-band wide --contact-regime jittery --backhaul-model seconds --aggregation agg:cutoff --session-ttl-s "$TTL_S" --policy-checkpoint E3=results/exp5/checkpoints/5.3-e3/e3/g0.9_s0.npz --keep-event-traces
```

E3 is trained at K = 1 and flown per mule at K = 3, on slices within the trained sizes (critic B7 v).

**Reading a Phase 5 trace.**

- `mule_ready.pair` (an FQ mule) and `mule_ready.policy_checkpoint` (E3) state the verified
  checkpoint's provenance: the sha, kind, purpose, classes, γ, reward, seeds, episodes, the cell
  family and its hash, the learner's revision, the schema's version, and the config's tag; no path.
- `mission_completed.pass_1_pairs` (FQ): one record per Pass-1 stop flown, in order. At the arrival:
  `t_s`, the stop's `devices`, b̄ (`committed`), the pair flown (`band`; `next_index`, where 0 keeps
  the plan's order and null is home; `next`), the pairs offered (`pairs`) and admitted (`feasible`,
  `admitted_pairs`), `fallback` (`mask_empty` when no pair fitted and FX's pair flew), FX's pair by
  FX's own rule (`fx_band`, `fx_next`, `agrees_fx`) and the scorer with its Q values (`scorer`, `q`,
  `q_fx`). Closed when the mission ends: `collected` and their merge weights `w`, `late`,
  `t_next_s` (the next arrival, or the end of the Pass-1 upload), `terminal`, and `trimmed_next`
  (the departure check after the stop did not keep the pair's order). `pass_1_flown[].band` agrees
  with `band`.
- `mission_completed.pass_1_e3` (E3): one entry per next-stop call, with `t_s`, `after_stop` (false
  at takeoff), `stops`, `admissible` (one bool per stop), `next_index` and `next`;
  `pass_1_e3_unvisited`: the stops left when none was admissible, never widened. E3's N is not
  recorded per call.
- Each of these is left out when empty, and none appears for any other arm.

**The scorer's pair columns.** `--pair-columns` adds seven columns after the τ columns
(configuration reference §19.8): `pair_decisions`, `pair_feasible_mean`, `pair_mask_empty` (a
count), `pair_fx_agree_share` (an empty-mask decision counts as agreeing), `pair_band_off_bbar_share`,
`pair_reorder_share` and `e3_unvisited_mean` (stops, not devices, over every mission). The shares
count choices, not flights (R21); they are reported diagnostics, and no Study 5.5 step reads them.
Without the flag the row is the Phase 4 one, byte for byte.

```bash
python -m experiments.analysis.traces_scorer --traces results/exp5/s55/stack_120_traces --pair-columns --csv results/exp5/s55/stack_120_scored.csv
```

**The refusals you will meet**, each before any trial or training, as a usage error (exit 2):

- *R3:* an FQ arm without `--in-flight-response replan` (the runner's default is `abort`).
- *No checkpoint:* a learned arm whose tag has none; a tag no learned arm flies (a 5.7 grid tag: fly
  that checkpoint as `main`).
- *B9:* a bootstrap checkpoint, one trained on no episode, one without a held-out score (run
  `evaluate --record`), or one trained from a dirty tree without `--allow-dirty-checkpoint`.
- *R24:* a checkpoint that is not its tag's, for example `g25=` a γ = 0.9 checkpoint, `hand=` a
  derived-reward one or `main=` an F·hand one, `g90=` one trained at a 5.7 grid point's weights,
  `E3=` one not trained on bytes, or kept weights that took no update.
- *R23:* `dwell=` or `cov=` a checkpoint not trained by `train --ablation`, `main=` an ablation's
  checkpoint, or any pair checkpoint trained under other `--plan-score-params` than the run gives.
- *A dirty tree:* `train` and `sweep` refuse it unless `--allow-dirty` (git sees a change under
  `hermes/` or `experiments/`, untracked files included), and record it in the manifest.
- *R19:* `train` and `sweep` refuse to replace another family's checkpoint, even with
  `--overwrite`: one study per family.
- *R12, the paths:* a checkpoint path is read against the working directory, and one that names no
  checkpoint, or none with its manifest beside it, is refused. The mule gets the path repo-relative
  for a file inside the repository and absolute otherwise, so a relative path given outside the
  repository never names another file. (During a run, a checkpoint rewritten since the runner
  checked it is refused before its next trial.)
- *R27:* `H1+L1` without `--backhaul-model seconds` or `--l1-channel`.
- *`--require-trained`:* H2 or H3 without `--selector-weights`.

## 3. Smoke run (one trial, no dataset)

Prove the stack end-to-end in ~1–2 min without CICIOT:
```bash
python -m experiments.exp4.runner_main \
    --csv results/exp4_smoke.csv \
    --arms H0 H1 --N 3 --rrf 60 --n-missions 3 --n-trials 1 \
    --regime jittery --dead-zone 0.4 --link-quality 0.5 \
    --real-model --data-source synthetic --realism --local-epochs 4
```
Expect a resumable CSV with a per-round convergence trace (`init_auc` → `final_auc`)
and the federation metrics from the real two-pass orchestrator.

---

## 4. The paper-grade sweep (parallel)

The paper run is 20 seeds over the full `dead_zone × link_quality` jittery surface +
a clean reference (H0/H1), plus the H2/H3 L1 comparison. Serially that is ~6 h; the
committed results were produced in ~1 h by
[`experiments/exp4/run_paper_sweep_parallel.sh`](../experiments/exp4/run_paper_sweep_parallel.sh),
which fans the grid into **6 concurrent shards** (one per dead-zone + clean + L1),
each an independent resumable CSV, with per-trial TF threads capped
(`OMP_NUM_THREADS=2`) so the shards share cores instead of oversubscribing.

```bash
# Edit the PY path at the top of the script to your interpreter first, then:
bash experiments/exp4/run_paper_sweep_parallel.sh
```

Outputs under `results/exp4_paper/`:
- `h0h1_surface.csv` + `h0h1_dz02/04/06.csv` + `h0h1_clean.csv` — the H0/H1 shards.
- `h2h3_l1.csv` — the H3-vs-H2 L1 comparison (`--l1-channel`, `n_missions=6`).

**Resume:** every shard consults the CSV before each trial and skips done rows, so a
killed run just re-runs the same command (or re-launches the script) and continues.

**Right-size for your box:** the script assumes ~20 cores. On fewer cores, reduce the
number of concurrent shards (run the dead-zone shards in two waves) or drop the shard
count; on a single serial machine, the equivalent is a plain
`runner_main --dead-zone 0.0 0.2 0.4 0.6 --link-quality 0.3 0.5 0.7 --regime jittery`
invocation.

---

## 5. Analysis

[`experiments.analysis.exp4`](../experiments/analysis/exp4.py) forms the paired
`(treatment − baseline)` differences per regime × metric and reports paired Wilcoxon
+ Cliff's δ + a bootstrap 95% CI. A verdict is claimed only when the CI excludes 0.

**Merge the H0/H1 shards, then analyse the participation surface:**
```bash
python - <<'PY'
import pandas as pd
files = ["h0h1_surface","h0h1_dz02","h0h1_dz04","h0h1_dz06","h0h1_clean"]
df = pd.concat([pd.read_csv(f"results/exp4_paper/{f}.csv") for f in files], ignore_index=True)
df.to_csv("results/exp4_paper/h0h1_all.csv", index=False)
print(len(df), "rows -> h0h1_all.csv")
PY

python -m experiments.analysis.exp4 --csv results/exp4_paper/h0h1_all.csv \
    --metrics mission_completion_rate update_yield round_close_rate_kmin2 final_auc \
    --surface --surface-metric mission_completion_rate
```

**The L1 comparison** uses the generalised `--treatment/--baseline` (defaults H1/H0):
```bash
python -m experiments.analysis.exp4 --csv results/exp4_paper/h2h3_l1.csv \
    --treatment H3 --baseline H2 \
    --metrics mission_completion_rate round_close_rate_kmin2 final_auc final_accuracy
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | Per-trial CSV (a merged surface, or a single shard). |
| `--metrics` | 5 defaults | Higher-is-better metrics to test (treatment − baseline). |
| `--treatment` / `--baseline` | `H1` / `H0` | Arm pair. Use `H3` / `H2` for the L1 claim. |
| `--surface` | off | Also print the per-`(dead_zone,link_quality)` jittery verdict. |
| `--surface-metric` | `final_auc` | Metric for the surface breakdown. |

> **Pairing note:** the pair keys include `dead_zone`/`link_quality`, so a merged
> multi-cell CSV pairs within each cell. The `--surface` breakdown is the honest
> view; the top-level table *pools* across the surface (more power, but mixes
> operating points — read both).

---

## 6. Figures

The two rebuttal figures (grayscale-safe, hatches + greys) and their generators:

```bash
python DeveloperDocs/exp4_figure.py         # -> results/exp4_paper/fig_exp4_crossover.png
python DeveloperDocs/exp4_figure_layer1.py  # -> results/exp4_paper/fig_exp4_layer1.png
python DeveloperDocs/exp4_analysis.py       # pure-stdlib Cliff's-δ tables to stdout
```

`exp4_figure.py` reads `h0h1_all.csv` (the crossover); `exp4_figure_layer1.py` reads
`h2h3_dz_*.csv` (H2/H3 across the dead-zone sweep, produced by
[`run_l1_deadzone_sweep.sh`](../experiments/exp4/run_l1_deadzone_sweep.sh)). Both
hard-code absolute `results/exp4_paper/` paths at the top — edit for a different
checkout.

Both figures draw uncertainty as a **percentile bootstrap 95 % CI** (bounded in
[0,1] by construction) with per-seed points overlaid, and assert that nothing is
drawn above AUC = 1.0. Do **not** reintroduce a symmetric ±SD whisker: `final_auc`
is bimodal (a session either trains or stays at its untrained init), so ±SD both
implies a spread that does not exist and renders above the metric's ceiling.

---

## 7. Where the results + code live

| What | Path |
|---|---|
| Committed result CSVs | [`results/exp4_paper/`](../results/exp4_paper/) (`h0h1_all.csv`, `h0h1_*.csv`, `h2h3_l1.csv`, `h2h3_dz_*.csv`) |
| Figures | `results/exp4_paper/fig_exp4_crossover.png`, `fig_exp4_layer1.png` |
| Runner CLI | [`experiments/exp4/runner_main.py`](../experiments/exp4/runner_main.py) |
| Driver (per-trial logic) | [`experiments/exp4/driver.py`](../experiments/exp4/driver.py) |
| Real DNN-IDS task | [`experiments/exp4/model_task.py`](../experiments/exp4/model_task.py) |
| L1 channel model + `U(c,t)` controller | [`experiments/exp4/channel.py`](../experiments/exp4/channel.py), [`hermes/l1/channel_utility.py`](../hermes/l1/channel_utility.py) |
| Metrics / events consumer | [`experiments/exp4/metrics.py`](../experiments/exp4/metrics.py), [`events_consumer.py`](../experiments/exp4/events_consumer.py) |
| Paired analysis | [`experiments/analysis/exp4.py`](../experiments/analysis/exp4.py) |
| Parallel sweep script | [`experiments/exp4/run_paper_sweep_parallel.sh`](../experiments/exp4/run_paper_sweep_parallel.sh) |
| Unit tests | `tests/unit/test_exp4_*.py` |
| Integration tests | `tests/integration/test_exp4_realmodel_smoke.py` (marked `slow`) |

---

## 8. Troubleshooting

**`arm H0 ... run with real_model=True`.** H0 is a real-model baseline; add `--real-model`.

**`CICIOT-2023 not found`.** Set `HERMES_CICIOT_DIR`, place the CSVs at
`../datasets/CICIOT2023/`, or switch to `--data-source synthetic`.

**Trials `status=error` under heavy parallelism.** Startup contention with many
concurrent shards can trip `--startup-timeout-s`. Bump it (the parallel script uses
90 s) or reduce concurrent shards. One dropped trial just drops that seed from its
pair; the analysis tolerates it.

**Shards look stalled.** Each trial spawns a real subprocess tree (1 cluster + 1 mule
+ N devices). On Windows always use finite `--n-missions`. The mules exit on their own and
keep their final `metrics_snapshot`. The driver stops the cluster and the devices with
`terminate()` (TerminateProcess), so their kept traces never hold `metrics_snapshot` or
`service_stopped`. Registry counters such as `sim_order_late_uploads`, `backhaul_unpriced_uploads`
or `rf_reconnects` are therefore absent from them: read their per-event equivalents instead
(Configuration Reference §17.6).

**Analysis prints `(no paired results)`.** Fewer than 2 paired seeds for that
regime/cell, or the treatment/baseline arms aren't both present in the CSV.

**A jittery surface cell shows `H0 > H1`.** Expected at the well-connected corner
(`dead_zone=0.0, link_quality=0.7`) — the documented flip point where the mule is
overhead. See Methodology §6; report the surface, not a single cell.
