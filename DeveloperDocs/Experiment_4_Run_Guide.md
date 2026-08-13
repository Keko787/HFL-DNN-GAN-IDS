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
| `--arms` | `H0 H1 H2 H3` | Subset of arms. H0 needs `--real-model`. |
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
| **Caveat** | `device_served` has no `mission_round`; attribute rounds by joining timestamps against `mission_started` / `mission_completed` |

**Rule of thumb: if a run is expensive enough that you would not want to repeat it, pass this flag.**

---

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
+ N devices). On Windows always use finite `--n-missions` and let the tree exit
naturally — `terminate()` can skip the final metrics snapshot.

**Analysis prints `(no paired results)`.** Fewer than 2 paired seeds for that
regime/cell, or the treatment/baseline arms aren't both present in the CSV.

**A jittery surface cell shows `H0 > H1`.** Expected at the well-connected corner
(`dead_zone=0.0, link_quality=0.7`) — the documented flip point where the mule is
overhead. See Methodology §6; report the surface, not a single cell.
