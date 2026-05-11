# Experiment 3 — Run Guide

**Status:** Experiment 3 chunks are in flight. This guide documents the current shipping path: train the A4 selector, then walk the A1×A2×A3×A4 grid against [`Exp3Sim`](../experiments/exp3/sim_env.py) and write a per-trial CSV.

**Companion docs:**
- [`HERMES_Experiments_Implementation_Plan.md`](HERMES_Experiments_Implementation_Plan.md) — design and Definition of Done per chunk.
- [`HERMES_Operations_Runbook.md`](HERMES_Operations_Runbook.md) — environment setup that this guide assumes is already done.
- [`HERMES_Configuration_Reference.md`](HERMES_Configuration_Reference.md) — every tunable in the selector, sim, and reward shape.
- [`Exp3_Future_Energy_Models.md`](Exp3_Future_Energy_Models.md) — Option B (retry/revisit) and Option C (adaptive in-cluster positioning) propulsion-energy extensions.
- [`Experiment_1_Run_Guide.md`](Experiment_1_Run_Guide.md) — sibling guide for the federated-vs-centralized experiment.

---

## 0. What the experiment measures

**Centralized FL vs mule heuristics vs HERMES scheduling — the A1–A4 ablation.**

| Arm | What it represents | Where |
|---|---|---|
| **A1** | Centralized FL (no mule, long-range link, classical FedAvg) | [`experiments/exp3/arm_a1.py`](../experiments/exp3/arm_a1.py) |
| **A2** | Round-robin mule (visit every device in declaration order) | [`experiments/exp3/arm_mule.py`](../experiments/exp3/arm_mule.py) |
| **A3** | Deadline-feasibility mule (filter by EDF feasibility, then nearest-first) | [`experiments/exp3/arm_mule.py`](../experiments/exp3/arm_mule.py) |
| **A4** | HERMES `TargetSelectorRL` (DDQN intra-bucket selection) | [`experiments/exp3/arm_mule.py`](../experiments/exp3/arm_mule.py) + [`hermes/scheduler/selector/`](../hermes/scheduler/selector/) |

All four arms run against the same simulator ([`Exp3Sim`](../experiments/exp3/sim_env.py)), so cross-arm comparisons share the world model. The grid sweeps `N` (bucket size), `β` (deadline tightness), `rrf` (RF range), `deadline_het` (homogeneous vs heterogeneous deadlines), and `jittery` (clean vs degraded network).

The harness produces a per-trial CSV that is **checkpoint-and-resume safe** — the runner consults `CSVTrialLog.already_done` before every trial, so multi-session AERPAW reservations don't lose work.

---

## 1. Prerequisites

1. Python 3.10 venv at `.venv310/` with `requirements_core.txt` installed (see [`HERMES_Operations_Runbook.md` §0](HERMES_Operations_Runbook.md)). Exp 3 is sim-only — no `flwr`, no real subprocesses, no network shaping needed.
2. Repo cloned; `cwd = /path/to/FL-DNN-GAN-IDS`.

Verify the install before the first run:
```bash
pytest tests/unit/test_exp3_sim_env.py tests/unit/test_exp3_drivers.py \
       tests/unit/test_exp3_metrics.py tests/unit/test_exp3_policies.py \
       tests/unit/test_exp3_train_a4.py tests/unit/test_exp3_analysis.py -v
```

---

## 2. Two-step flow

A4 needs trained DDQN weights to be paper-grade. The runner has a smoke-only fallback to random-init weights, but flatly refuses to use it under `--require-trained-a4`. So:

1. **Train the selector** with [`experiments.exp3.train_a4`](../experiments/exp3/train_a4.py) → `weights/a4_selector.npz`.
2. **Run the trial grid** with [`experiments.exp3.runner_main`](../experiments/exp3/runner_main.py) → `results/exp3.csv`.

Step 1 is a one-time cost per reward-shape change. Weights transfer directly to the evaluation environment because the [`ContactSim`](../hermes/scheduler/selector/sim_env.py) trainer shares the contact-event reward shape with [`Exp3Sim`](../experiments/exp3/sim_env.py) (per implementation plan §4.2).

---

## 3. Step 1 — train A4

```bash
mkdir -p weights
python -m experiments.exp3.train_a4 \
    --episodes 400 \
    --rf-range-m 60 \
    --output weights/a4_selector.npz
```

Common knobs (full list via `python -m experiments.exp3.train_a4 --help`):

| Flag | Default | Purpose |
|---|---|---|
| `--output` | required | `.npz` path for the trained DDQN weights. |
| `--episodes` | 400 | Training episodes. 400 is the implementation-plan default. |
| `--n-devices` | 8 | Devices per training episode bucket. |
| `--rf-range-m` | 60.0 | S3a clustering radius for training. Must match the eval-time `--rrf` you care about. |
| `--mission-budget` | 200.0 | Per-episode mission budget seconds. |

Training prints per-episode reward to stderr. A trained `.npz` is portable across runner invocations — keep one canonical file under `weights/` so paper runs always pull the same selector.

---

## 4. Step 2 — run the trial grid

**Paper-grade run (full grid, A1–A4, trained A4):**
```bash
python -m experiments.exp3.runner_main \
    --csv results/exp3.csv \
    --selector-weights weights/a4_selector.npz \
    --require-trained-a4 \
    --n-trials 20
```

**Smoke run (subset of arms / axes, faster turnaround):**
```bash
python -m experiments.exp3.runner_main \
    --csv results/exp3_smoke.csv \
    --arms A1 A4 \
    --N 5 --beta 1.0 --rrf 60 --deadline-het 0 --jittery 0 \
    --n-trials 2 \
    --selector-weights weights/a4_selector.npz
```

If you omit `--selector-weights`, A4 silently falls back to a random-init DDQN — useful for runner / pipeline smoke tests but the rows are not paper-grade. `--require-trained-a4` makes that fallback a hard error.

---

## 5. Sweep knobs

`python -m experiments.exp3.runner_main --help` for the canonical list. The five independent variables are the grid axes:

| Flag | Default | Meaning |
|---|---|---|
| `--N` | `5 10 20` | Bucket-size sweep. Number of devices per A4 bucket. |
| `--beta` | `0.25 0.5 1.0 2.0` | Deadline tightness — scales `mission_budget * β`. `β=0.25` is the budget-tight regime where A3's feasibility filter actually fires; `β ≥ 1.0` is budget-rich (filter never triggers). |
| `--rrf` | `30 60 120` | `rf_range_m` sweep. |
| `--deadline-het` | `0 1` | `0` = uniform deadlines; `1` = heterogeneous. |
| `--jittery` | `0 1` | `0` = clean (no packet loss, no latency jitter); `1` = jittery (2 % packet loss + 30 % latency jitter, matching Exp 1's `--jittery` cell). Pass `--jittery 0` to skip jittery cells. |

**Network model knobs** (defaults are calibrated to put A3's feasibility filter under genuine pressure in jittery cells but not in clean cells):

| Flag | Default | Meaning |
|---|---|---|
| `--clean-upload-bytes` | `1.0e6` | Per-contact upload payload in clean cells (1 MB). |
| `--clean-upload-bps` | `1.0e7` | Nominal upload rate in clean cells (10 Mbps). Yields `upload_s ≈ 0.8 s` per contact — minimal network pressure. |
| `--jittery-upload-bytes` | `1.0e7` | Per-contact upload payload in jittery cells (10 MB — typical FL gradient). |
| `--jittery-upload-bps` | `1.0e6` | Nominal upload rate in jittery cells (1 Mbps). Yields `upload_s ~ 80 s` per contact. The simulator also applies inverse-distance falloff and ±20 % rate jitter on top, plus 2 % packet loss and 30 % latency jitter. |

**Dead-zone knobs** (model correlated, not i.i.d., failure modes — terrain blockage, range-edge SNR collapse):

| Flag | Default | Meaning |
|---|---|---|
| `--clean-a1-dead-zone-pct` | `0.0` | Fraction of A1 clients marked unreachable in clean cells. |
| `--jittery-a1-dead-zone-pct` | `60.0` | Fraction of A1 clients marked persistently long-range-unreachable in jittery cells. With N=20 FedAvg rounds, an i.i.d. 40 % per-round failure unions to ~100 % `mission_completion_rate`; a 60 % dead zone caps it at 40 %. Tune lower for a milder jittery story, higher (e.g. 80) to put A1 below the mule arms in jittery on cumulative metrics. Mule arms have no analogous knob — short-range device↔mule RF is treated as reliable. |

**Other:**

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | Per-trial CSV path. |
| `--n-trials` | 20 | Trials per cell, paired across arms. |
| `--base-seed` | 42 | Seed root for paired-seed reproducibility. |
| `--arms` | `A1 A2 A3 A4` | Subset of arms to run. |
| `--timeout-s` | 300.0 | Soft per-trial wall-clock budget — records `status=timeout`, never aborts the loop. |
| `--selector-weights` | (none) | `.npz` from `train_a4`. If omitted, A4 uses random-init weights (smoke-only). |
| `--require-trained-a4` | off | Refuse to run when A4 would use random-init weights. Set on every paper-grade run. |

---

## 6. Calibration sensitivity

The grid above varies the experiment-level independent variables. Calibration constants in the sim env (`ENERGY_W`, `COMPLETION_BONUS`, idle-time prior, etc.) live in [`experiments/calibration.toml`](../experiments/calibration.toml) and the sensitivity-sweep TOMLs under [`experiments/calibration_sensitivity/`](../experiments/calibration_sensitivity/):

- `eps_high.toml` / `eps_low.toml` — energy-weight prior swings.
- `p_idle_high.toml` / `p_idle_low.toml` — idle-time prior swings.

Point the runner at one of these to produce the sensitivity panel; the sweep keeps every other knob fixed at the headline calibration.

See [`HERMES_Configuration_Reference.md`](HERMES_Configuration_Reference.md) §4 (selector reward weights) and §6 (selector feature scaling) for the source of these defaults.

---

## 7. Resume semantics

The trial CSV is the resume index. On every trial the runner consults `CSVTrialLog.already_done({cell_id, arm, trial_index})` and skips matching rows. Crash recovery is just "re-run the same command":

```bash
# Initial run gets interrupted at trial 712/1920:
python -m experiments.exp3.runner_main --csv results/exp3.csv \
    --selector-weights weights/a4_selector.npz --require-trained-a4

# Resume — picks up from trial 713 automatically:
python -m experiments.exp3.runner_main --csv results/exp3.csv \
    --selector-weights weights/a4_selector.npz --require-trained-a4
```

Status columns on each row: `status ∈ {ok, error, timeout}`, `duration_s`, `error` (last traceback line on `error`, soft-cap message on `timeout`, empty on `ok`). Soft `timeout` rows are recorded but the loop never aborts — one bad cell can't stop the grid.

---

## 8. CSV schema

Runner writes one row per `(cell_id, arm, trial_index)`. Columns:

| Group | Columns |
|---|---|
| Cell key | `cell_id`, `arm`, `trial_index`, `seed` |
| Sweep params | `param_N`, `param_beta`, `param_rrf`, `param_deadline_het`, `param_jittery` |
| Sim summary | `Exp3MetricSummary.csv_columns()` (per-trial completions, energy, latency, etc.) |
| Echo | `n_devices`, `beta`, `deadline_het`, `rf_range_m`, `jittery` |
| Status | `status`, `duration_s`, `error` |

The exact metric set lives in [`experiments/exp3/metrics.py`](../experiments/exp3/metrics.py). Adding columns to a partial CSV is safe — the resume key is the cell tuple, not the column set.

---

## 9. Analysis

The analysis driver and the cross-arm sanity check both consume the trial CSV directly (stdlib `csv` + `numpy` + `matplotlib`, same `requirements_core.txt` venv that ran the experiment).

### 9.1 The full panel — `experiments.analysis.exp3`

Loads the CSV, applies the calibration TOML, runs the paired tests, prints a one-page text summary, and writes the six paper figures + a paired-tests CSV.

```bash
python -m experiments.analysis.exp3 \
    --csv results/exp3.csv \
    --figures-dir DeveloperDocs/figures/exp3
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | Per-trial CSV from `experiments.exp3.runner_main`. |
| `--figures-dir` | `figures/exp3` | Output directory for the panel figures. |
| `--calibration` | (default TOML) | Override path to `calibration.toml`. |
| `--no-figures` | off | Skip figure generation; print summary only. |
| `--jittery-filter` | `all` | One of `clean`, `jittery`, `all`. `clean` keeps only `r.jittery=False` rows (single box per arm); `jittery` keeps only `r.jittery=True`; `all` renders paired clean+jittery boxes per arm so the reader sees clusters within and across regimes. |

The six figures (per implementation plan §4.2 EX-3.5) plus one CSV companion:

| # | What | Comparison |
|---|---|---|
| 1 | A2 vs A1 | paired Wilcoxon on update yield + Jain's fairness |
| 2 | A3 vs A2 | paired Wilcoxon on round close rate + propulsion energy |
| 3 | A4 vs A3 | paired Wilcoxon on update yield + round close rate (the experiment's primary novelty) |
| 4 | β-sweep curve | update yield vs β with one curve per arm, faceted by N — slope-vs-cliff figure |
| 5 | rrf-sweep curve | update yield vs rrf with one curve per arm at `β=1.0, N=10` |
| 6 | ρ_contact bar chart | faceted by rrf, comparing A2 / A3 / A4 |
| – | `exp3_paired_tests.csv` | paper-table reproducibility |

### 9.2 Cross-arm sanity check — `audit_arm_agreement`

Before publishing a "the arms are indistinguishable" finding (rather than calling it a bug), run the audit script. It re-runs the simulator with each arm against the same seeds and reports per-seed decision sequences plus aggregate statistics covering decision counts, pairwise agreement rates, and A3 filter activation. Three failure modes it flags:

1. **Decision count too low** — if a typical mission only makes 2–3 ranking decisions before the budget runs out, the arms have very little room to differ.
2. **A3's feasibility filter never triggers** — `EdfFeasibilityPolicy` reduces to "EDF over the candidate list" when the filter never drops anything.
3. **A4 picks identically to A2 or A3** — degenerate trained policy, or a candidate list too short for ranking to matter.

```bash
python -m experiments.exp3.audit_arm_agreement \
    --selector-weights weights/a4_selector.npz \
    --seeds 20 \
    --n-devices 10 --beta 1.0 --rrf 60.0
```

| Flag | Default | Meaning |
|---|---|---|
| `--selector-weights` | required | A4 weights from `train_a4`. |
| `--seeds` | (required) | Number of seeds to audit. |
| `--n-devices` | 10 | Devices per audit episode. |
| `--beta` | 0.25 | Deadline tightness for the audit. |
| `--rrf` | 60.0 | RF range for the audit. |
| `--mission-budget-s` | 600.0 | Per-mission budget. |
| `--cruise-speed-m-s` | 5.0 | Mule cruise speed. |

No CSV is written; the output is for human inspection. Run the audit whenever A4 looks suspiciously close to A2 / A3 in the headline figures.

### 9.3 End-to-end procedure (paper-grade)

```bash
# Step A — sanity-check the arms differ before trusting the panel:
python -m experiments.exp3.audit_arm_agreement \
    --selector-weights weights/a4_selector.npz \
    --seeds 20 --n-devices 10 --beta 0.25 --rrf 60.0

# Step B — render the full panel for the all-regimes view:
python -m experiments.analysis.exp3 \
    --csv results/exp3.csv \
    --figures-dir DeveloperDocs/figures/exp3 \
    --jittery-filter all

# Step C — regime-specific panels for the paper appendix:
python -m experiments.analysis.exp3 \
    --csv results/exp3.csv \
    --figures-dir DeveloperDocs/figures/exp3_clean \
    --jittery-filter clean

python -m experiments.analysis.exp3 \
    --csv results/exp3.csv \
    --figures-dir DeveloperDocs/figures/exp3_jittery \
    --jittery-filter jittery
```

The audit comes first — if A4 is degenerate against A2 / A3 at the audit's `(N, β, rrf)` cell, the panel figures will look uninformative for a defensible reason and the paper text needs to disclose it.

### 9.4 Interactive / notebook use

Loaders + helpers are importable from [`experiments/analysis/exp3.py`](../experiments/analysis/exp3.py):

```python
from experiments.analysis.exp3 import load_trials
from experiments.calibration import load_calibration

cal = load_calibration(None)
rows = load_trials("results/exp3.csv")

clean = [r for r in rows if r.is_ok and not r.jittery]
jittery = [r for r in rows if r.is_ok and r.jittery]
print(f"{len(clean)} clean, {len(jittery)} jittery trials")
```

Stats primitives (paired Wilcoxon + Cliff's δ, bootstrap CI) live in [`experiments/analysis/stats.py`](../experiments/analysis/stats.py) — shared with Exp 1.

---

## 10. Troubleshooting

**`A4 is in --arms but --selector-weights was not provided`.** You hit the warn (without `--require-trained-a4`) or hard error (with it) saying the runner would use random-init DDQN weights. Train weights first via §3.

**A4 results look identical to a random policy.** Likely either (a) the `.npz` you loaded was trained against a different `--rf-range-m` than the eval `--rrf`, or (b) `--epsilon` isn't 0 at eval time (the runner forces `epsilon=0` automatically — if you bypassed it, it's on you). Confirm with `audit_arm_agreement`.

**Trials all status=timeout at the default 300 s cap.** Soft cap only — rows are recorded. Either bump `--timeout-s` if your cell legitimately needs more wall-clock, or shrink the sweep (fewer `--N` / `--beta` / `--rrf` points). Real trials should land well under 300 s on a laptop.

**A1 dominates A4 in jittery cells.** Check `--jittery-a1-dead-zone-pct`. The default is 60 — terrain-blocked clients in jittery cells. If you set it to 0, A1 enjoys an i.i.d.-only failure model and trivially beats the mule arms over enough rounds (the FedAvg union effect).

**A3's feasibility filter never fires.** Expected at `β ≥ 1.0` with the default clean-network model (upload_s is much smaller than collect_s, so feasibility is trivial). Either run `--beta 0.25 --jittery 1` to put genuine pressure on the filter, or tighten `--clean-upload-bps` / loosen `--mission-budget`.

**Resume re-runs already-done rows.** Two possible causes: (a) you changed the cell key (e.g. a sweep-param value), so the new cell ID doesn't match the CSV's; (b) two runner processes share one CSV path — the file isn't locked across processes.

---

## 11. Where the code lives

| What | File |
|---|---|
| Runner CLI | [`experiments/exp3/runner_main.py`](../experiments/exp3/runner_main.py) |
| A4 trainer CLI | [`experiments/exp3/train_a4.py`](../experiments/exp3/train_a4.py) |
| Driver (per-trial logic) | [`experiments/exp3/driver.py`](../experiments/exp3/driver.py) |
| Sim env | [`experiments/exp3/sim_env.py`](../experiments/exp3/sim_env.py) |
| Per-trial metrics summary | [`experiments/exp3/metrics.py`](../experiments/exp3/metrics.py) |
| A1 arm | [`experiments/exp3/arm_a1.py`](../experiments/exp3/arm_a1.py) |
| A2 / A3 / A4 mule arms | [`experiments/exp3/arm_mule.py`](../experiments/exp3/arm_mule.py) |
| Cross-arm audit | [`experiments/exp3/audit_arm_agreement.py`](../experiments/exp3/audit_arm_agreement.py) |
| Selector (DDQN) | [`hermes/scheduler/selector/`](../hermes/scheduler/selector/) — `target_selector_rl.py`, `ddqn.py`, `features.py`, `replay.py`, `selector_train.py`, `sim_env.py` (`ContactSim`) |
| Calibration | [`experiments/calibration.toml`](../experiments/calibration.toml), [`experiments/calibration_sensitivity/`](../experiments/calibration_sensitivity/) |
| Trial harness (shared with Exp 1) | [`experiments/runner/`](../experiments/runner/) |
