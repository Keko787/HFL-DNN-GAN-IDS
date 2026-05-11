# Experiment 1 — Run Guide

**Status:** Experiment 1 complete (2026-05-01). This guide documents how to reproduce the runs whose CSVs sit under [`results/exp1_chameleon*.csv`](../results/).

**Companion docs:**
- [`HERMES_Experiments_Implementation_Plan.md`](HERMES_Experiments_Implementation_Plan.md) — design and Definition of Done for the experiment.
- [`HERMES_Operations_Runbook.md`](HERMES_Operations_Runbook.md) — environment setup that this guide assumes is already done.
- [`Experiment_3_Run_Guide.md`](Experiment_3_Run_Guide.md) — sibling guide for the A1–A4 ablation.

---

## 0. What the experiment measures

**Federated vs Centralized at fixed radio.** Compares an FL arm (per-round uplink + downlink between server and `N` clients) against a Centralized arm (one bulk uplink per client) on a shared, shaped link, holding the radio constant at a 10 Mbps cap. The trial grid sweeps payload size `D_pd`, round count `R`, and the two arms; per-trial wall-clock is bracketed by the **server's `time.perf_counter()`** (the server is the sole clock authority — clients do not report their own durations).

The harness produces a per-trial CSV with one row per `(cell_id, arm, trial_index)`. Re-running with the same `--output` skips already-completed rows, so multi-session runs (e.g., AERPAW reservations) are checkpoint-and-resume safe.

---

## 1. Prerequisites

1. Python 3.10 venv at `.venv310/` with `requirements_core.txt` + `requirements_edge.txt` installed (see [`HERMES_Operations_Runbook.md` §0](HERMES_Operations_Runbook.md)).
2. Repo cloned; `cwd = /path/to/FL-DNN-GAN-IDS`.
3. CIC-IoT-2023 unzipped under `$HOME/datasets/CICIOT2023` (loader: [`Config/SessionConfig/datasetLoadProcess.py`](../Config/SessionConfig/datasetLoadProcess.py)).
4. For **shaped runs only:** root access on each client node so `tc` / `netem` can install qdiscs.

Verify the install before the first run:
```bash
pytest tests/unit/test_exp1_partition.py tests/unit/test_exp1_protocol.py tests/unit/test_exp1_topology.py -v
pytest tests/integration/test_exp1_server_client.py -v
```

---

## 2. Topology

Default topology: **1 server + 4 clients**, each client owning one disjoint partition of CIC-IoT-2023 (`partition ∈ {0, 1, 2, 3}`). The server binds a TCP listener; clients dial in, REGISTER with `(client_id, data_partition)`, and either fill their pre-declared slot or get rejected.

Three configuration modes are supported (server side):

| Mode | How to invoke | When |
|---|---|---|
| Explicit JSON | `--topology configs/exp1_aerpaw.json` | Paper runs. Reproducible. |
| Explicit CLI | `--client d1@192.168.1.11:partition=0 --client d2@…` | One-off runs. |
| Discovery | `--discover --n-clients 4` | Dev-mode convenience — accepts the first N clients with unique partitions covering `0..N-1`. |

A pre-built Chameleon topology lives at [`experiments/exp1/setup/configs/exp1_chameleon.json`](../experiments/exp1/setup/configs/exp1_chameleon.json) (substitute the `${IP}` placeholders with your reservation's addresses).

---

## 3. Single-host quick run

The fastest way to get a CSV. Spins up 1 server + 4 clients all on `127.0.0.1`, runs the full trial grid, then shuts everything down.

```bash
# Smoke run (2 trials per cell, single filter cell):
N_TRIALS=2 FILTER=Dpd=10MB,R=5 bash experiments/exp1/setup/launch_local.sh

# Full grid → results/exp1.csv:
bash experiments/exp1/setup/launch_local.sh

# Custom output path (e.g., the jittery cell):
OUT=results/exp1_jittery.csv bash experiments/exp1/setup/launch_local.sh
```

Environment knobs accepted by [`launch_local.sh`](../experiments/exp1/setup/launch_local.sh):

| Var | Default | Meaning |
|---|---|---|
| `OUT` | `results/exp1.csv` | CSV output path. |
| `N_TRIALS` | `20` | Trials per cell (paired across arms). |
| `BASE_SEED` | `42` | Seed root for reproducibility. |
| `SERVER_PORT` | `9000` | TCP bind port for the server. |
| `N_CLIENTS` | `4` | Number of client subprocesses. |
| `FILTER` | (empty) | Subset of cells to run, e.g. `Dpd=10MB,R=5`. |

Server stdout + stderr stream to `logs/exp1_server.log`; each client to `logs/exp1_d{i}.log`. The script traps `EXIT INT TERM` and SIGTERMs every client on cleanup.

---

## 4. Multi-host run (Chameleon / AERPAW / generic)

When clients live on different hosts than the server (the actual paper-run shape). Run the server on one node and one client per data-partition on the others.

**4.1 Server (one node):**
```bash
python -m experiments.exp1.server \
    --client d1@192.168.1.11:partition=0 \
    --client d2@192.168.1.12:partition=1 \
    --client d3@192.168.1.13:partition=2 \
    --client d4@192.168.1.14:partition=3 \
    --bind-host 0.0.0.0 \
    --bind-port 9000 \
    --output results/exp1.csv \
    --n-trials 20 \
    --base-seed 42
```

Or load the topology from JSON:
```bash
python -m experiments.exp1.server \
    --topology experiments/exp1/setup/configs/exp1_chameleon.json \
    --output results/exp1_chameleon.csv \
    --n-trials 20
```

**4.2 Clients (one per node):**
```bash
# Node hosting d1:
python -m experiments.exp1.client \
    --client-id d1 --server 192.168.1.10:9000 --data-partition 0

# Node hosting d2 (and so on):
python -m experiments.exp1.client \
    --client-id d2 --server 192.168.1.10:9000 --data-partition 1
```

The server blocks in `accept_clients` until every slot is claimed (or `--registration-timeout-s` elapses, default 60 s). It then walks the trial grid, broadcasting `TRIAL_BEGIN` / `TRIAL_END` / `SHUTDOWN` control frames over the same socket.

`--strict-ip` rejects clients whose source IP doesn't match the slot's declared `host` (default: warn-only).

---

## 5. Network shaping

Runs against a 10 Mbps shared link. `shape_link.sh` installs the `tc` / `netem` qdiscs on each client node:

```bash
sudo bash experiments/exp1/setup/shape_link.sh apply               # 10 Mbps TBF cap
sudo bash experiments/exp1/setup/shape_link.sh apply --jittery     # +30% delay jitter + 2% loss
sudo bash experiments/exp1/setup/shape_link.sh status              # show current qdiscs
sudo bash experiments/exp1/setup/shape_link.sh remove --jittery    # drop netem only (TBF stays)
sudo bash experiments/exp1/setup/shape_link.sh remove              # drop ALL shaping
```

Auto-detects the interface (tries `eno1np0`, then `eno1`, then the device holding the default route). The `remove --jittery` path is implemented as a full teardown + TBF reinstall (not a child-qdisc delete) to avoid a netlink race that would otherwise drop SSH keepalives mid-operation.

**Three canonical cells:**
- **Clean:** `apply` only.
- **Jittery:** `apply --jittery` (matches the `results/exp1_chameleon_jittery.csv` cell).
- **Unshaped:** `remove` (matches `results/exp1_chameleon_no1gb.csv` for the unshaped baseline).

The `B_nominal_mbps` field in the topology is documentation-only; actual shaping is the operator's job.

---

## 6. Resume semantics

The trial CSV is the resume index. On every trial the runner consults `CSVTrialLog.already_done({cell_id, arm, trial_index})` and skips matching rows. Crash recovery is just "re-run the same command":

```bash
# Initial run gets interrupted at trial 47/240:
bash experiments/exp1/setup/launch_local.sh

# Resume — picks up from trial 48 automatically:
bash experiments/exp1/setup/launch_local.sh
```

Status columns on each row: `status ∈ {ok, error, timeout}`, `duration_s`, `error` (last traceback line on `error`, soft-cap message on `timeout`, empty on `ok`). Soft `timeout` rows are recorded but the loop never aborts — one bad cell can't stop the grid.

---

## 7. CSV schema

Server writes one row per `(cell_id, arm, trial_index)`. Columns:

| Group | Columns |
|---|---|
| Cell key | `cell_id`, `arm`, `trial_index`, `seed` |
| Sweep params | `param_*` (one per independent variable, e.g. `param_Dpd`, `param_R`) |
| Metrics | server-side per-trial timing + bytes (see `experiments/exp1/server.py`) |
| Status | `status`, `duration_s`, `error` |

Already-done detection keys on `(cell_id, arm, trial_index)`, so adding new columns to a partial CSV is safe.

---

## 8. Analysis

Driver scripts under [`experiments/analysis/`](../experiments/analysis/) consume the trial CSVs directly. They use stdlib `csv` (no pandas dependency) plus `numpy` + `matplotlib` for the figures, so the same `requirements_core.txt` venv that ran the experiment can render the analysis.

### 8.1 The full panel — `exp1.py`

The top-level driver. Loads the CSV, applies the calibration TOML, runs the paper's stats (paired Wilcoxon + Cliff's δ on `Bpw` / `Ttx`, bootstrap CI on R\*), prints a one-page text summary, and writes the five paper figures + a paired-tests CSV.

```bash
python -m experiments.analysis.exp1 \
    --csv results/exp1_chameleon_no1gb.csv \
    --figures-dir DeveloperDocs/figures/exp1
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | Per-trial CSV from the server. |
| `--figures-dir` | `figures/exp1` | Output directory for the panel figures. |
| `--calibration` | (default TOML) | Override path to `calibration.toml`. |
| `--no-figures` | off | Skip figure generation; print summary only. |

Outputs into `--figures-dir`: `exp1_Pcomplete_heatmap.png`, `exp1_eta_heatmap.png`, `exp1_Rstar_regression.png`, `exp1_energy_stacked_bar.png`, `exp1_paired_tests.csv`. Re-running overwrites in place.

### 8.2 Cleaner / single-purpose figures

Each script below replaces or augments one panel from `exp1.py` with a publication-cleaner alternative. Run them after `exp1.py` if you need the cleaner figure for the paper draft.

```bash
# 3-bar energy figure (FL pooled across Dpd, Centralized broken out by Dpd):
python -m experiments.analysis.exp1_energy_clean \
    --csv results/exp1_chameleon_no1gb.csv \
    --out DeveloperDocs/figures/exp1/exp1_energy_clean.png \
    --alpha 1.0

# R* regression in the clean cell + residuals + per-cell-means table:
python -m experiments.analysis.exp1_rstar_clean \
    --csv results/exp1_chameleon_no1gb.csv \
    --out-dir DeveloperDocs/figures/exp1
# Writes: exp1_Rstar_regression_clean.png, exp1_Rstar_residuals.png,
#         exp1_Rstar_with_data.png, exp1_Rstar_per_cell_means.csv

# Jittery vs primary deep-dive (3 figures: walltime, energy, per-trial scatter):
python -m experiments.analysis.exp1_jittery_deep_dive \
    --csv-primary results/exp1_chameleon_no1gb.csv \
    --csv-jittery results/exp1_chameleon_jittery.csv \
    --out-dir DeveloperDocs/figures/exp1_jittery \
    --Dpd 100MB --alpha 1.0 --R 20
# Writes: jittery_walltime.png, jittery_energy.png, jittery_per_trial.png

# Sensitivity grid — 5 calibrations side by side (P_idle ±50%, eps_bit ±50%, baseline):
python -m experiments.analysis.exp1_sensitivity_panel \
    --csv results/exp1_chameleon_no1gb.csv \
    --out DeveloperDocs/figures/exp1/sensitivity_grid.png

# Combined 2×5 sensitivity grid (rows = primary / jittery, cols = 5 calibrations):
python -m experiments.analysis.exp1_sensitivity_combined \
    --csv-primary results/exp1_chameleon_no1gb.csv \
    --csv-jittery results/exp1_chameleon_jittery.csv \
    --out DeveloperDocs/figures/exp1/sensitivity_combined.png
```

`exp1_sensitivity_panel` and `exp1_sensitivity_combined` both default `--variants-dir` to [`experiments/calibration_sensitivity/`](../experiments/calibration_sensitivity/) and read `p_idle_low.toml`, `p_idle_high.toml`, `eps_low.toml`, `eps_high.toml` from there. Override with `--variants-dir` or `--baseline-toml` if you have a custom calibration.

### 8.3 End-to-end procedure (paper-grade)

```bash
# Step A — render the full panel + summary for each CSV:
python -m experiments.analysis.exp1 \
    --csv results/exp1_chameleon.csv \
    --figures-dir DeveloperDocs/figures/exp1

python -m experiments.analysis.exp1 \
    --csv results/exp1_chameleon_jittery.csv \
    --figures-dir DeveloperDocs/figures/exp1_jittery

# Step B — replace the energy + R* panels with the cleaner versions:
python -m experiments.analysis.exp1_energy_clean \
    --csv results/exp1_chameleon_no1gb.csv \
    --out DeveloperDocs/figures/exp1/exp1_energy_clean.png

python -m experiments.analysis.exp1_rstar_clean \
    --csv results/exp1_chameleon_no1gb.csv \
    --out-dir DeveloperDocs/figures/exp1

# Step C — primary-vs-jittery deep dive:
python -m experiments.analysis.exp1_jittery_deep_dive \
    --csv-primary results/exp1_chameleon_no1gb.csv \
    --csv-jittery results/exp1_chameleon_jittery.csv \
    --out-dir DeveloperDocs/figures/exp1_jittery

# Step D — calibration sensitivity:
python -m experiments.analysis.exp1_sensitivity_combined \
    --csv-primary results/exp1_chameleon_no1gb.csv \
    --csv-jittery results/exp1_chameleon_jittery.csv \
    --out DeveloperDocs/figures/exp1/sensitivity_combined.png
```

The order above (full panel first, cleaner figures second, deep-dives last) mirrors the order the paper draft uses. Each step is independent — re-run any single script after a CSV refresh without redoing the others.

### 8.4 Interactive / notebook use

Every script above is also importable. Loaders + helpers live in [`experiments/analysis/exp1.py`](../experiments/analysis/exp1.py):

```python
from experiments.analysis.exp1 import load_trials, summarize, per_row_energy
from experiments.calibration import load_calibration

cal = load_calibration(None).exp1
rows = load_trials("results/exp1_chameleon_no1gb.csv")
print(summarize(rows))

energy = per_row_energy([r for r in rows if r.alpha == 1.0], cal)
```

Stats primitives (paired Wilcoxon + Cliff's δ, R\* bootstrap CI, crossover-round solver) live in [`experiments/analysis/stats.py`](../experiments/analysis/stats.py).

### 8.5 Existing inputs and outputs

CSVs to compare against:
- [`results/exp1_chameleon.csv`](../results/exp1_chameleon.csv) — clean cell, Chameleon hardware.
- [`results/exp1_chameleon_jittery.csv`](../results/exp1_chameleon_jittery.csv) — jittery cell.
- [`results/exp1_chameleon_no1gb.csv`](../results/exp1_chameleon_no1gb.csv) — unshaped variant.

Pre-rendered figures live under [`DeveloperDocs/figures/exp1/`](figures/exp1/), [`DeveloperDocs/figures/exp1_jittery/`](figures/exp1_jittery/), and the `exp1_sens_*` / `exp1_jittery_sens_*` siblings — use them as a sanity-check target when re-rendering.

---

## 9. Troubleshooting

**Registration times out.** The server blocks until every slot is claimed; if a client never connects, you see `TimeoutError: registration timed out … missing slots: [...]`. Check:
- The server is reachable from the client (`telnet <server-host> 9000`).
- The client's `--client-id` matches a slot in the topology and `--data-partition` matches that slot's partition.
- In explicit-IP mode without `--strict-ip`, host mismatch is a warning, not an error — but in `--strict-ip` mode it's a rejection.

**A trial errors but the grid keeps going.** Expected. The runner records `status=error` + the last traceback line and moves on. Re-run after fixing the bug; only error rows that you delete will be retried (already-done detection keys on the row, not on `status`).

**`status=timeout` rows.** Soft cap only — the driver returned, but late. The row's `error` column carries the soft-cap message. Bump `--timeout-s` if your cell legitimately needs more wall-clock; the cap defaults to whatever the server hard-coded.

**Shaping survives a reboot / breaks SSH.** Always pair `apply` with `remove` (or `remove --jittery` for the netem overlay only). The `remove --jittery` path is deliberately not implemented as a single `tc qdisc del … parent 1:1` because that operation is not netlink-atomic on a busy interface and can drop SSH keepalives long enough to take the node off the network.

**CSV has duplicate rows.** Shouldn't happen — `CSVTrialLog.already_done` is checked before every append. If you see dupes, you've probably pointed two server processes at the same CSV; the file isn't locked across processes.

---

## 10. Where the code lives

| What | File |
|---|---|
| Server | [`experiments/exp1/server.py`](../experiments/exp1/server.py) |
| Client | [`experiments/exp1/client.py`](../experiments/exp1/client.py) |
| Wire protocol (control + data frames) | [`experiments/exp1/protocol.py`](../experiments/exp1/protocol.py) |
| Topology config | [`experiments/exp1/topology.py`](../experiments/exp1/topology.py) |
| Data partitioning | [`experiments/exp1/data_partition.py`](../experiments/exp1/data_partition.py) |
| Local launcher | [`experiments/exp1/setup/launch_local.sh`](../experiments/exp1/setup/launch_local.sh) |
| Link shaping | [`experiments/exp1/setup/shape_link.sh`](../experiments/exp1/setup/shape_link.sh) |
| Chameleon JSON | [`experiments/exp1/setup/configs/exp1_chameleon.json`](../experiments/exp1/setup/configs/exp1_chameleon.json) |
| Trial harness (shared with Exp 3) | [`experiments/runner/`](../experiments/runner/) |
