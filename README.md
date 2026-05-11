# HFL-DNN-GAN-NIDS / HERMES

A **Hierarchical Federated Learning + GAN-based Network Intrusion Detection System** for private and IoT networks, deployed over a mobile-mule-assisted four-tier architecture. The scheduling/transport substrate that drives mules, devices, edge servers, and the cloud is called **HERMES**.

The repo ships:
- The legacy single-server Flower training pipeline (`App/TrainingApp/`), kept for backwards compatibility.
- The HERMES multi-process system (`hermes/`) with a deterministic FL scheduler, a DDQN-based intra-bucket target selector, partial cross-mule FedAvg, RF / dock / cloud transports, and a TCP-backed orchestrator that brings the whole topology up on a single laptop.
- AC-GAN / DNN-IDS model code with the mode-collapse fixes from the September 2025 analysis.
- A paper-experiment harness (`experiments/`) that drives Experiments 1 and 3 (federated-vs-centralized and the A1–A4 scheduler ablation) and writes resumable per-trial CSVs.

## Table of Contents

- [Team](#team)
- [Status](#status)
- [Architecture](#architecture)
  - [Tiers](#tiers)
  - [Programs (HERMES)](#programs-hermes)
  - [Two-pass mission flow](#two-pass-mission-flow)
- [Models](#models)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Datasets](#datasets)
- [Quick start (HERMES local emulation)](#quick-start-hermes-local-emulation)
- [Legacy training pipeline](#legacy-training-pipeline)
- [Experiments](#experiments)
- [Tests](#tests)
- [Operations & Configuration](#operations--configuration)
- [Developer documentation](#developer-documentation)
- [License](#license)

---

## Team

**Faculty Advisors (2026)**
- Dr. Chenqi Qu
- Dr. Prasad Calyam

**Graduate Students (2026)**
- Kevin Kostage
- Bishwas
- Chandra
- Kiran

---

## Status

- **Phases 0–7 of the HERMES build are closed** (Sprint 2 chunks A–O + Phase 7 chunks P1–P5).
- Final test count: **410 passed, 22 deselected** (the deselects are the two known flaky tests — stochastic A/B at `rf=60 m` and the Flower-mode subprocess timeouts).
- **Experiment 1** (federated vs centralized at fixed radio) is complete (results under `results/`); **Experiment 3** (A1–A4 ablation) chunks are in flight.
- The next milestones are paper-experiment scaffolding and AERPAW deployment when the testbed returns.

See [`DeveloperDocs/HERMES_FL_Scheduler_Implementation_Plan.md`](DeveloperDocs/HERMES_FL_Scheduler_Implementation_Plan.md) for phase-by-phase status.

---

## Architecture

HERMES is **symmetric server/client at every tier-boundary** — every server has a matching client on the other side of the link. The mule's NUC is the only host that runs both a server role and a client role simultaneously: server-to-devices in-field (`HFLHostMission`) and client-to-server at dock (`ClientCluster`).

### Tiers

| Tier | Host | Programs resident | Scope |
|---|---|---|---|
| **Tier 1** — Edge Device | Device CPU | `ClientMission` (Flagger + Discriminator trainer + FL-client) | local data only |
| **Tier 2** — Edge Server | Stationary server | `HFLHostCluster` (registry, θ_gen, cross-mule FedAvg) + `ClusterCloudClient` | cluster |
| **Tier 2-mobile** — Mule NUC | Intel NUC on UAV/UGV | `L1 RL` channel actor + `FLScheduler` (with `TargetSelectorRL`) + `HFLHostMission` + `ClientCluster` | per-mission |
| **Tier 3** — Cloud | Chameleon / AERPAW | `Tier3Coordinator` (θ_gen refinement, cross-cluster rhythm) | global |

Within Layer 2 the scheduler is decoupled into **four stages**: S1 Eligibility → S2A Readiness (on-contact) → S2B FL Readiness Flag (on-device) → S3 Deadline & Priority, with an S3.5 intra-bucket selector and an S3a contact-clustering pass.

### Programs (HERMES)

Seven cooperating programs replace the original monolithic Flower server/client pair:

1. **`FLScheduler`** ([`hermes/scheduler/fl_scheduler.py`](hermes/scheduler/fl_scheduler.py)) — L2 scheduler on the mule's NUC. Owns *who* gets visited and per-device deadlines.
2. **`TargetSelectorRL`** ([`hermes/scheduler/selector/target_selector_rl.py`](hermes/scheduler/selector/target_selector_rl.py)) — DDQN intra-bucket next-target selector. Sub-model of S3.5; only ranks contact positions the deterministic stages already admitted.
3. **`HFLHostMission`** ([`hermes/mission/host_mission.py`](hermes/mission/host_mission.py)) — Mission-scope FL server on the mule's NUC. Runs the COLLECT (Pass 1) and DELIVER (Pass 2) sub-sessions and per-contact partial FedAvg.
4. **`ClientCluster`** ([`hermes/mule/client_cluster.py`](hermes/mule/client_cluster.py)) — Dock-handoff client on the mule's NUC. Owns the entire dock lifecycle (UP partial-aggregate / DOWN next-mission bundle); never trains.
5. **`ClientMission`** ([`hermes/mission/client_mission.py`](hermes/mission/client_mission.py)) — Edge-device FL client + flagger. Trains the discriminator **offline between mule visits**, computes a utility score (`w₁·perf + w₂·diversity_adjusted`), and beacons `FL_OPEN` when worth federating.
6. **`HFLHostCluster`** ([`hermes/cluster/host_cluster.py`](hermes/cluster/host_cluster.py)) — Cluster FL coordinator on Tier 2. Authoritative `DeviceRegistry`, slices missions per-mule, hosts θ_gen and the synth sample generator, runs cross-mule FedAvg, dispatches dock bundles.
7. **`L1 RL Module`** ([`hermes/l1/channel_ddqn.py`](hermes/l1/channel_ddqn.py)) — RF channel selector (DDQN, channel-only). The original "trajectory" head moved up to `TargetSelectorRL` because mule navigation is mechanical between known device positions.

These are coordinated by four information flows: **intra-NUC** (L1 ↔ L2 ↔ HFL-Mission ↔ ClientCluster), **in-field RF link** (HFLHostMission ↔ ClientMission), **dock handoff** (ClientCluster ↔ HFLHostCluster), and **cloud sync** (HFLHostCluster ↔ Tier 3).

### Two-pass mission flow

Each mission is **two-pass**:

- **Pass 1 — COLLECT.** Mule walks scheduler-ranked contact positions; `HFLHostMission` pushes `θ_disc + synth` and pulls each device's pre-prepared `Δθ_disc`; per-contact partial FedAvg merges in-range deltas before folding into the running mission aggregate.
- **Inter-pass dock.** `ClientCluster` ships the mission aggregate to `HFLHostCluster`, which runs cross-mule FedAvg → `θ_disc'`, regenerates the synth batch, and emits cluster amendments. The DOWN bundle comes back to the mule.
- **Pass 2 — DELIVER.** Mule walks every contact greedily and pushes `θ_disc' + synth'` to all in-range devices; no Δθ is requested. Devices immediately start the next round of offline local training against `θ_disc'`.

Why two passes: every Δθ collected in Pass 1 was trained against the θ the cluster delivered in the previous mission's Pass 2, so cross-mule FedAvg is mathematically exact — async-FL drift becomes structurally impossible.

A textual diagram and the full happy-path / exception-path walkthrough live in [`DeveloperDocs/HERMES_FL_Scheduler_Design.md`](DeveloperDocs/HERMES_FL_Scheduler_Design.md) §§3–4.

---

## Models

- **NIDS (canonical)** — DNN binary classifier defined by `create_CICIOT_Model` in [`Config/modelStructures/NIDS/NIDS_Struct.py`](Config/modelStructures/NIDS/NIDS_Struct.py). 5-layer Dense stack (`64 → 32 → 16 → 8 → 4 → 1`) with `BatchNormalization`, `Dropout(0.4)`, and L2-regularized kernels — ~4.7K params, ~18.8 KB at float32. This is what `--mode hermes` and Experiment 4 wire in. Conv1D → GRU → LSTM hybrids and IoT variants in the same module are alternatives, not the project default.
- **Discriminator** — multi-class real-vs-synthetic classifier ([`Config/modelStructures/GAN/discriminatorStruct.py`](Config/modelStructures/GAN/discriminatorStruct.py)).
- **Generator** — GAN-based traffic synthesizer ([`Config/modelStructures/GAN/generatorStruct.py`](Config/modelStructures/GAN/generatorStruct.py)).
- **GAN / AC-GAN / WGAN** — combined adversarial models ([`Config/modelStructures/GAN/ganStruct.py`](Config/modelStructures/GAN/ganStruct.py)). The AC-GAN central trainer carries the September-2025 mode-collapse fixes (3:1 D-to-G ratio, label-smoothing rebalance, generator-LR slowdown, discriminator health monitoring). See [`DeveloperDocs/ACGAN_Mode_Collapse_Fix_Implementation_Plan.md`](DeveloperDocs/ACGAN_Mode_Collapse_Fix_Implementation_Plan.md) and [`DeveloperDocs/GAN_Training_Mode_Collapse_Analysis.md`](DeveloperDocs/GAN_Training_Mode_Collapse_Analysis.md).

Trained checkpoints are kept under [`ModelArchive/`](ModelArchive/) (NIDS, GAN, AC-GAN, WGAN versions).

---

## Repository layout

```
FL-DNN-GAN-IDS/
├── hermes/                  # NEW — HERMES system
│   ├── cluster/             # HFLHostCluster, cross_mule_fedavg, device_registry
│   ├── mission/             # HFLHostMission, ClientMission, partial_fedavg, utility
│   ├── mule/                # MuleSupervisor, ClientCluster
│   ├── scheduler/           # FLScheduler + S1/S2A/S2B/S3/S3a/S3.5 stages
│   │   ├── stages/          # s1_eligibility, s2a_readiness, s2b_flag, s3_deadline, s3a_cluster, s35_selector
│   │   ├── selector/        # TargetSelectorRL (DDQN), features, replay, sim_env, scope_guard
│   │   └── policies/        # arrival_order, edf_feasibility
│   ├── l1/                  # ChannelDDQN, RF prior
│   ├── transport/           # rf_link, dock_link, cloud_link, tcp_*, channel_emulator, wire
│   ├── types/               # bundles, fl_messages, fl_state, ids, registry, signatures, ...
│   ├── processes/           # MultiProcessOrchestrator + cluster/mule/device entrypoints
│   └── observability/       # JSONL events + MetricsRegistry
├── App/
│   ├── TrainingApp/         # Legacy Flower path (HFLHost, Client) + --mode hermes hook
│   └── InferenceApp/        # Detection (live + adversarial) + Generation (AC-GAN eval)
├── Config/
│   ├── modelStructures/     # NIDS_Struct.py, GAN/ (generator, discriminator, gan)
│   ├── ModelTrainingConfig/ # Client (Central, HFL) + Host (FitOnEnd, ModelManagement)
│   ├── SessionConfig/       # ArgumentConfigLoad, datasetLoadProcess, modelCreateLoad, hyperparameterLoading
│   └── DatasetConfig/
├── experiments/             # NEW — paper experiment harness
│   ├── runner/              # grid, csv_log, runner
│   ├── exp1/                # client/server, data_partition, protocol, topology
│   ├── exp3/                # arm_a1, arm_mule, train_a4, driver, sim_env, metrics
│   ├── analysis/            # exp1.py, exp1_*, exp3.py, stats
│   ├── calibration_sensitivity/  # eps / p_idle high/low TOMLs
│   └── calibration.{py,toml}
├── tests/
│   ├── unit/                # ~370+ unit tests across hermes + experiments
│   └── integration/         # e2e topology, faults, two-pass contact, TCP links, observability, ...
├── AppSetup/
│   ├── AERPAW_node_Setup.py
│   ├── Chameleon_node_Setup.py
│   ├── DockerSetup/         # Dockerfile.client, Dockerfile.server, docker-compose.yml
│   ├── requirements_core.txt
│   └── requirements_edge.txt
├── FlightFramework/         # Flight strategies / runtime reused for partial FedAvg
├── Analysis/                # FLMetrics, TestAnalysis, logs
├── ModelArchive/            # Trained NIDS / GAN / AC-GAN / WGAN checkpoints
├── results/                 # Experiment 1 trial CSVs (chameleon, jittery, no1gb)
├── DeveloperDocs/           # Design + plan + runbook + AC-GAN analyses
└── pytest.ini
```

---

## Prerequisites

- Ubuntu 22.04 LTS with CUDA 12 drivers (P100 / M40 supported) for GPU runs; CPU works for the multi-process orchestrator.
- **Python 3.10** (the dev venv lives at `.venv310/`).
- For Windows development the runbook lists Git Bash, PowerShell, and `cmd.exe` activation paths.

---

## Installation & Setup

```bash
# Clone
git clone https://github.com/Keko787/HFL-DNN-GAN-IDS.git
cd HFL-DNN-GAN-IDS

# Create the venv (one-time)
python3.10 -m venv .venv310

# Activate
source .venv310/bin/activate                # Linux / macOS
# source .venv310/Scripts/activate          # Git Bash on Windows
# .venv310\Scripts\Activate.ps1             # PowerShell

# Core HERMES + test deps (always required):
pip install -r AppSetup/requirements_core.txt
pip install pytest

# Edge / training deps (legacy Flower path, AC-GAN, dataset loaders):
pip install -r AppSetup/requirements_edge.txt
```

If you only intend to run the multi-process HERMES path, `requirements_core.txt` + pytest is enough — `flwr` is only needed for the legacy path and the M3–M5 mode-switch tests.

**Testbed-node bootstrap (optional):**
```bash
python3 AppSetup/AERPAW_node_Setup.py        # AERPAW node
python3 AppSetup/Chameleon_node_Setup.py     # Chameleon node
```

**Containerized deploy:** Dockerfiles for client + server and a `docker-compose.yml` are under [`AppSetup/DockerSetup/`](AppSetup/DockerSetup/).

---

## Datasets

1. Download **CIC IoT2023** from the [CIC website](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
2. Upload `CICIoT2023.zip` to `$HOME/datasets/`, then:

```bash
unzip $HOME/datasets/CICIoT2023.zip -d $HOME/datasets/CICIOT2023
```

Loader: [`Config/SessionConfig/datasetLoadProcess.py`](Config/SessionConfig/datasetLoadProcess.py). IoTBotnet is also supported as an alternate dataset for the legacy client.

---

## Quick start (HERMES local emulation)

The multi-process orchestrator brings the whole topology up on a single laptop via real subprocesses + TCP. End-to-end runtime: under 30 s for the smallest viable topology (1 cluster + 1 mule + 1 device).

**1. Verify the artefact:**
```bash
pytest tests/integration/test_e2e_topology.py -v
```

**2. Smallest viable smoke topology** (save as `scripts/run_smoke.py`):
```python
from hermes.processes import (
    ClusterConfig, DeviceConfig, MuleConfig,
    MultiProcessOrchestrator, TopologyConfig,
)

def main() -> None:
    topo = TopologyConfig(
        cluster=ClusterConfig(
            cluster_id="c-smoke",
            dock_host="127.0.0.1", dock_port=0,        # 0 = ephemeral
            synth_batch_size=2, min_participation=1,
        ),
        mules=[MuleConfig(
            mule_id="m-smoke",
            rf_host="127.0.0.1", rf_port=0,
            rf_range_m=60.0, session_ttl_s=3.0,
            n_missions=1,
        )],
        devices=[DeviceConfig(
            device_id="d-smoke", position=(0.0, 0.0, 0.0),
        )],
    )
    orch = MultiProcessOrchestrator(topo, capture_output=True)
    try:
        orch.start_all(timeout=20.0)
        orch.mule_handles["m-smoke"].proc.wait(timeout=30.0)
    finally:
        orch.shutdown_all(timeout=5.0, cleanup_tmpdir=False)
        print(f"\nrun dir = {orch.tmpdir}")

if __name__ == "__main__":
    main()
```

```bash
python scripts/run_smoke.py
```

JSONL events (`cluster-c-smoke.jsonl`, `mule-m-smoke.jsonl`, `device-d-smoke.jsonl`) land in the printed run dir. Expected event sequence: `cluster_ready` → `mule_ready` → `device_ready` → `mule_bootstrapped` → `dock_bootstrapped` → `mission_started` → `device_served`×N (Pass 1) → `up_bundle_ingested` → `cluster_round_closed` → `device_served`×N (Pass 2) → `mission_completed` → `metrics_snapshot` → `service_stopped`.

**Sprint-2 DoD topology** (1 cluster + 2 mules + 5 devices, 3 stationary + 2 mobile):
```python
mules = [
    MuleConfig(mule_id="m1", expected_devices=["d1", "d2", "d3"],
               rf_host="127.0.0.1", rf_port=0, rf_range_m=60.0),
    MuleConfig(mule_id="m2", expected_devices=["d4", "d5"],
               rf_host="127.0.0.1", rf_port=0, rf_range_m=60.0),
]
devices = [DeviceConfig(device_id=f"d{i+1}", position=(i*10.0, 0.0, 0.0))
           for i in range(5)]
```

`rf_range_m` controls how aggressively S3a clusters in-range devices into a single contact event: `30` → mostly N=1 contacts, `60` (validated default) → 2–3 devices per contact, `120` → most of a slice falls into one or two large contacts.

The full cold-start guide, diagnostics, and dock/cloud topology recipes are in [`DeveloperDocs/HERMES_Operations_Runbook.md`](DeveloperDocs/HERMES_Operations_Runbook.md).

---

## Legacy training pipeline

The original Flower-based binaries are preserved for backwards compatibility and to provide a zero-risk rollback path. They default to `--mode legacy`; pass `--mode hermes` to dispatch into the new code.

### Federated Training (Host)
```bash
python3 App/TrainingApp/HFLHost/HFLHost.py --help
python3 App/TrainingApp/HFLHost/HFLHost.py <usual-args>                  # legacy Flower server
python3 App/TrainingApp/HFLHost/HFLHost.py <usual-args> --mode hermes    # HERMES path
```

### Localized & Federated Training (Client)
```bash
python3 App/TrainingApp/Client/TrainingClient.py --help
python3 App/TrainingApp/Client/TrainingClient.py <usual-args>                  # legacy Flower client
python3 App/TrainingApp/Client/TrainingClient.py <usual-args> --mode hermes    # HERMES path
# Default uses CICIOT2023 dataset; use --dataset IOTBOTNET for IoTBotnet.
```

### Demo (centralized AC-GAN)
```bash
python3 TrainingClient.py --model_type AC-GAN --model_training Both \
                          --trainingArea Central --dataset CICIOT --save_name Test1
```

The mode-switch contract (M1–M7) is enforced by [`tests/unit/test_mode_switch.py`](tests/unit/test_mode_switch.py): argparse defaults, subprocess smoke under both modes, a repo-wide grep that hermes imports stay inside the `if args.mode == "hermes":` branch, and shared-venv compatibility.

---

## Experiments

The paper-experiment harness is under [`experiments/`](experiments/). It owns the seed schedule, writes per-trial CSV rows, and is checkpoint-and-resume safe so multi-session AERPAW reservations don't lose work.

| # | What it measures | Status |
|---|---|---|
| **1** | Federated vs Centralized at fixed radio (10 Mbps shaped link) | **complete** (results in [`results/exp1_chameleon*.csv`](results/)) |
| **2** | Traditional vs adaptive radio (RL channel selection) | deferred — telemetry-side, not FL |
| **3** | Centralized FL vs mule heuristics vs HERMES scheduling (A1–A4 ablation) | in-flight |
| **4** | Integrated HERMES vs traditional FL — all metrics | blocked on Exp 1 + 3 closing |

Run a resumable experiment grid:
```bash
python -m experiments.runner --exp 1 --resume
python -m experiments.runner --exp 3 --resume
```

Per-experiment drivers, sim envs, arm definitions (`arm_a1.py`, `arm_mule.py`, `train_a4.py`), metrics, and analysis notebooks live under `experiments/exp1/`, `experiments/exp3/`, and `experiments/analysis/`. See [`DeveloperDocs/HERMES_Experiments_Implementation_Plan.md`](DeveloperDocs/HERMES_Experiments_Implementation_Plan.md) for the full design and Definition of Done per chunk; [`DeveloperDocs/Exp3_Future_Energy_Models.md`](DeveloperDocs/Exp3_Future_Energy_Models.md) covers the Option B / Option C propulsion-energy extensions kept in the back pocket.

---

## Tests

```bash
# Verify the install (~370+ unit tests, < 15 s):
python -m pytest tests/unit -q

# Full fast suite (skipping the two pre-existing flaky tests):
pytest tests/ -q --deselect tests/integration/test_contact_selector_ab.py \
                 --deselect tests/unit/test_mode_switch.py

# Slow integration suite (real subprocesses, ~30 s):
pytest tests/ -m slow

# A single test:
pytest tests/integration/test_e2e_faults.py::test_pass_2_unreachable_device_marked_undelivered -v
```

Integration coverage spans the e2e topology, two-pass contact, fault injection, TCP RF / dock / cloud links, scheduler re-rank, contact selector A/B, observability JSONL, and the mule supervisor (single + two-pass).

---

## Operations & Configuration

- **Cold-start, mode switching, diagnostics, AERPAW deployment** → [`DeveloperDocs/HERMES_Operations_Runbook.md`](DeveloperDocs/HERMES_Operations_Runbook.md).
- **Every tunable** (utility weights, FL_Threshold, min-participation, selector reward weights, DDQN hyperparameters, feature scaling, deadline floors, beacon window, bucket priority order) with `file:line`, default, surface (config field / constructor arg / module constant), and rationale → [`DeveloperDocs/HERMES_Configuration_Reference.md`](DeveloperDocs/HERMES_Configuration_Reference.md).
- **Observability**: each process emits JSONL events under the orchestrator's run dir; counters/gauges/timers go through `MetricsRegistry` and surface in `metrics_snapshot` events at shutdown.

---

## Developer documentation

Authoritative design / planning docs (under [`DeveloperDocs/`](DeveloperDocs/)):

- [`HERMES_FL_Scheduler_Design.md`](DeveloperDocs/HERMES_FL_Scheduler_Design.md) — tier/program responsibilities, two-pass mission flow, state machines, message schemas.
- [`HERMES_FL_Scheduler_Implementation_Plan.md`](DeveloperDocs/HERMES_FL_Scheduler_Implementation_Plan.md) — phase-by-phase build plan with DoD per chunk.
- [`HERMES_Configuration_Reference.md`](DeveloperDocs/HERMES_Configuration_Reference.md) — single source of truth for every tunable.
- [`HERMES_Operations_Runbook.md`](DeveloperDocs/HERMES_Operations_Runbook.md) — first-boot guide, diagnostics, deployment.
- [`HERMES_Experiments_Implementation_Plan.md`](DeveloperDocs/HERMES_Experiments_Implementation_Plan.md) — paper-experiment scaffolding (Exp 1, Exp 3, shared trial harness).
- [`Exp3_Future_Energy_Models.md`](DeveloperDocs/Exp3_Future_Energy_Models.md) — Option B (retry/revisit) and Option C (adaptive in-cluster positioning) propulsion-energy extensions.
- AC-GAN analyses & fix plans:
  - [`GAN_Training_Mode_Collapse_Analysis.md`](DeveloperDocs/GAN_Training_Mode_Collapse_Analysis.md)
  - [`ACGAN_Mode_Collapse_Fix_Implementation_Plan.md`](DeveloperDocs/ACGAN_Mode_Collapse_Fix_Implementation_Plan.md)
  - [`ACGAN_Discriminator_Freezing_Analysis.md`](DeveloperDocs/ACGAN_Discriminator_Freezing_Analysis.md)
  - [`ACGAN_Additional_Training_Flaws_Analysis.md`](DeveloperDocs/ACGAN_Additional_Training_Flaws_Analysis.md)
  - [`Validity_Training_NaN_Issue_Analysis_And_Solutions.md`](DeveloperDocs/Validity_Training_NaN_Issue_Analysis_And_Solutions.md)
  - [`ac-gan arch.md`](DeveloperDocs/ac-gan%20arch.md) and [`ac-gan batch processing.md`](DeveloperDocs/ac-gan%20batch%20processing.md)

---

## License

This project is licensed under the [MIT License](LICENSE).
