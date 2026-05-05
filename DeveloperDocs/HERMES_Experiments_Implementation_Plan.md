# HERMES Experiments — Implementation Plan

**Status:** Experiment 1 complete (2026-05-01). Drafted post-Sprint-2 closeout; revised after the testbed-agnostic redesign and EX-0..EX-1.5 land. Experiment 3 chunks remain (see status table at §1).
**Companion to:** `HERMES_FL_Scheduler_Implementation_Plan.md` (which covers
the system build through Phase 7) and `HERMES_FL_Scheduler_Design.md`.

This document captures what is needed to actually *run* the four experiments
the paper describes (§IV of the manuscript). Phase 0–7 of the system plan
ship the *artefact* HERMES; this plan ships the *measurements* that exercise
that artefact and produce paper figures. They are sequenced in parallel:
the trial harness can be built while Phase 7 hardening lands.

---

## 0. Scope

Four experiments are described in the paper:

| # | What it measures | Status of paper section |
|---|---|---|
| **1** | Federated vs Centralized at fixed radio (10 Mbps shaped link) | full draft, this doc plans it |
| **2** | Traditional vs adaptive radio (RL channel selection) | full draft, deferred — telemetry-side, not FL |
| **3** | Centralized FL vs mule heuristics vs HERMES scheduling (A1–A4 ablation) | full draft, this doc plans it |
| **4** | Integrated HERMES vs traditional FL, all metrics | section stub (`§??`); blocked on Exp 1 + 3 closing first |

Experiments 1 and 3 are the user-priority targets. Experiment 2 is RF-link
telemetry work that lives in the L1 / channel-DDQN layer and is largely
disjoint from the scheduling experiments planned here. Experiment 4 reuses
the trial harness from Exp 1 + 3 and folds the integrated metrics together;
it cannot be planned in detail until 1 and 3 close so we know what
"integrated" actually means against measured baselines.

This doc therefore plans **Experiment 1 + Experiment 3 + the shared trial
harness**. It also surfaces the AERPAW availability caveat that gates real
hardware runs of all four.

---

## 1. What's already in place

The Sprint-2 artefact gives us a real running system; what's missing is
not the system but the experiment scaffolding around it.

**System pieces ready to drive:**

* **A4 (HERMES RL scheduler)** — `TargetSelectorRL` + two-pass mission +
  per-contact partial-FedAvg, exercised end-to-end via
  `MultiProcessOrchestrator` (Sprint 2 chunks A–O).
* **DNN-IDS classifier (canonical)** — [`create_CICIOT_Model`](Config/modelStructures/NIDS/NIDS_Struct.py:214) at `Config/modelStructures/NIDS/NIDS_Struct.py:214`. 5-layer Dense stack (`64 → 32 → 16 → 8 → 4 → 1`) with `BatchNormalization` + `Dropout(0.4)` and L2-regularized kernels; ~4.7K params, ~18.8 KB at float32. Instantiated by the project from [`modelCreateLoad.py:60-63`](Config/SessionConfig/modelCreateLoad.py:60). This is the model `--mode hermes` and Experiment 4 will wire in; see [HERMES_Configuration_Reference.md §13](HERMES_Configuration_Reference.md#13-canonical-model-choice--ciciot-nids) for the authoritative entry. *Other architectures in the same module (Conv1D → GRU → LSTM hybrids, IoT variants) are alternatives, not the project default.*
* **CICIOT-2023 loader** — `Config/SessionConfig/datasetLoadProcess.py`.
* **rrf sweep harness** — `tests/integration/test_contact_selector_ab.py`
  already runs `rf_range_m ∈ {30, 60, 120}` at one β; the parametric
  template extends.
* **Selector A/B sim env** — `hermes/scheduler/selector/sim_env.py`
  (planar field, per-device reliability, completion model). Needs
  extensions for Experiment 3 but the bones are there.
* **Observability** — JSONL events per process under the orchestrator's
  run dir, plus `MetricsRegistry` counter/gauge/timer primitives. The
  data the experiments need flows through the system; the gap is
  aggregating it into trial-level CSV rows.

---

## 2. Shared trial harness (Chunk EX-0)

**Goal.** A reusable runner that drives both Experiment 1 and Experiment 3
trials, owns the seed schedule, writes a per-trial CSV row on completion,
and is checkpoint-and-resume so multi-session AERPAW reservations don't
lose work.

**Why first.** Both experiments need it. Building it once unblocks the
parallel implementation of arms / metrics for either experiment.

**Where it lives.** New top-level `experiments/` directory, sibling to
`hermes/`. Sub-dirs `experiments/runner/`, `experiments/exp1/`,
`experiments/exp3/`, `experiments/analysis/` (Jupyter notebooks).

**Tasks**

1. **Trial-grid spec** — a small Python class taking the cartesian product
   of independent variables and emitting a deterministic `(cell_id, seed)`
   stream. Cells include trial index 0..19 paired across arms.
2. **CSV writer** with idempotent append and a `cell_id + trial_index +
   arm` unique key. Resume = read existing CSV, skip cells already done.
3. **Per-trial driver protocol** — each experiment supplies a
   `run_trial(arm, cell, seed) -> dict[str, Any]` callable; the runner
   wraps it in timing + exception capture, writes the row.
4. **Multi-process integration** — for arms that need a real topology
   (Exp 1 FL arm, Exp 3 A4), the driver instantiates a
   `MultiProcessOrchestrator` and reads back JSONL events at trial close.
5. **Trial timeout + cleanup** — every trial must terminate within a
   bounded budget; on timeout, mark `status=timeout` and reclaim
   resources (kill subprocesses, cleanup tmpdir).

**Definition of Done**
- One driver script `python -m experiments.runner --exp 1 --resume` runs
  twenty trials per cell of the Experiment-1 grid, writes a CSV, and is
  resumable across sessions.
- A unit test covers grid generation + paired-seed reproducibility +
  resume-skip-already-done semantics.
- Failing one trial does not break subsequent trials.

**Size:** ~1 sprint (1 engineer).

---

## 3. Experiment 1 — Federated vs Centralized at fixed radio — ✅ done (2026-05-01)

After the testbed-agnostic redesign, EX-1.1 became a **server + client pair** rather than a monolithic centralized driver — same scripts run on local subprocesses, AERPAW, or Chameleon by config. Both FL and Centralized arms are driven from one server entry point (`python -m experiments.exp1.server`); the client (`python -m experiments.exp1.client`) is dumb byte-shipper bound to a single CSR (CICIOT shard role).

### 3.1 Status against paper spec

| Need | Status | Where it landed |
|---|---|---|
| FedAvg / FL arm | ✅ | `Exp1ServerDriver._run_fl` in `experiments/exp1/server.py` (testbed-agnostic; clients ship per-round uplink + receive per-round downlink) |
| Centralized arm (raw data → server, no FL) | ✅ | `Exp1ServerDriver._run_centralized` (one-shot bulk uplink per client) |
| DNN-IDS model | ✅ | unchanged at `Config/modelStructures/NIDS/NIDS_Struct.py`; the wire-level metrics don't depend on the model running |
| 4-way fixed-seed CICIOT partition | ✅ | `experiments/exp1/data_partition.py` — deterministic SHA-256-seeded `numpy.random.Generator` permutation; supports both index-only and serialized shard outputs |
| Trial harness with CSV | ✅ | EX-0: `experiments/runner/` |
| 6 metrics: Tproc, Bpw, Ttx, η, E, Pcomplete | ✅ | Tproc/Ttx/Bpw/η/Pcomplete recorded by the server per row; E (idle + tx decomposition) computed post-hoc by `experiments/calibration.exp1_energy_proxy` |
| AERPAW USRP calibration constants Pidle, εbit | ✅ structurally; ⏳ values are placeholders | `experiments/calibration.toml`. Status field `placeholder` triggers a watermark on every figure until replaced with `verified` and the real AERPAW spec values |
| tc/netem 10 Mbps shaping runbook | ✅ | `experiments/exp1/setup/shape_link.sh` (apply / remove / status; supports `--jittery` ablation); operations runbook §6.5 |
| Stats: Wilcoxon + Cliff's δ + bootstrap CI on R* | ✅ | `experiments/analysis/stats.py` + `experiments/analysis/exp1.py` |

### 3.2 Chunk-by-chunk record

| Chunk | Scope | Status |
|---|---|---|
| **EX-1.1** | **Server + client pair** (testbed-agnostic). Three arg-parsing modes (explicit JSON, explicit CLI, discovery). Wire protocol with CONTROL (JSON) + BULK (raw bytes) frames. Server is the single clock authority via `time.perf_counter()`. Both FL and Centralized arms drive through the same scripts. | ✅ |
| **EX-1.2** | Deterministic CICIOT partition utility — `partition_indices(N, n, seed)` produces disjoint, complete, reproducible shards via SHA-256-seeded numpy RNG. `materialize_shard` supports both index-only (filler bytes) and source-array (real CICIOT) modes. | ✅ |
| **EX-1.3** | `experiments/calibration.toml` with `[exp1.aerpaw_usrp]`, `[exp3.mule_platform]`, and `[provenance]` tables; `experiments/calibration.py` loader returns typed `Exp1Calibration` / `Exp3Calibration` dataclasses; `exp1_energy_proxy` returns the (idle_J, tx_J) decomposition the paper's stacked bar requires. Status field gates the watermark on figures. | ✅ |
| **EX-1.4** | `experiments/exp1/setup/shape_link.sh` (apply / remove / status; `--jittery` adds ±30% + 2% loss for the ablation cell); `experiments/exp1/setup/launch_local.sh` orchestrates one server + 4 clients on a single Linux box; runbook §6.5 covers all three deployment recipes (local, AERPAW, Chameleon). | ✅ |
| **EX-1.5** | `experiments/analysis/stats.py` (paired Wilcoxon + Cliff's δ + bootstrap CI + R* regression) + `experiments/analysis/exp1.py` (CSV loader, per-cell paired tests, energy decomposition, summary text, five paper figures). Watermark stamped automatically when the calibration is `placeholder`. | ✅ |

### 3.3 Definition of Done (Experiment 1) — met

- ✅ All 5 sub-chunks landed and tested. **66 new tests** (24 EX-0 harness + 27 EX-1.1 protocol/topology/integration + 17 EX-1.2 partition + 11 EX-1.3 calibration + 19 EX-1.5 analysis), all green.
- ✅ A full sweep of the factorial grid (framework × |D|pd × α × R + the jittery-link ablation cell) runs to completion on local emulation. Smoke proven by `tests/integration/test_exp1_server_client.py` (1 cell × 2 arms × 2 trials = 4 trials in <1 second).
- ✅ The 5 paper figures (η heatmap, R* regression, energy stacked-bar, Pcomplete heatmap, Bpw/Ttx significance CSV) reproduce from any trial CSV via `python -m experiments.analysis.exp1 --csv <path> --figures-dir <dir>`. Pinned by `test_write_figures_smoke`.
- ✅ AERPAW / Chameleon deployment is config-only — same Python scripts; only the topology JSON's IPs change, and bandwidth shaping is the operator's responsibility (`tc/netem` for Linux dev, AERPAW's wireless emulator on the testbed).

**Size:** landed in one focused session. The `placeholder` calibration values must be replaced with real AERPAW USRP datasheet numbers before the paper run — that's a one-line edit to `calibration.toml` plus updating the `status` field to `verified`.

---

## 4. Experiment 3 — A1 vs A2 vs A3 vs A4 scheduling ablation

### 4.1 Status against paper spec

| Need | Status | Action |
|---|---|---|
| **A1**: centralized FL no mule, uniform per-round client sampling | ❌ legacy Flower path exists but isn't wired as a measured arm | Chunk EX-3.1 |
| **A2**: arrival-order, no skip / no reorder | ❌ only distance-sort fallback exists | Chunk EX-3.2 |
| **A3**: heuristic EDF with feasibility skip | ❌ deadline math exists, EDF policy doesn't | Chunk EX-3.2 |
| **A4**: RL DQN scheduler | ✅ shipping | — |
| Kostage sim env (planar field, 3 base stations, fixed-speed mule, time-varying upload rates) | ⚠ planar field + reliability exist; 3 BSes + per-channel rates + speed model missing | Chunk EX-3.3 |
| Sweep knobs N ∈ {5,10,20}, β ∈ {0.5,1.0,2.0}, deadline-het mode, rrf ∈ {30,60,120} | ⚠ only rrf is swept today | Chunk EX-3.3 |
| 9 metrics — round close rate, update yield, coverage, **Jain's**, **entropy**, mission completion, **propulsion energy (Eq 5)**, **ρ_contact**, Pass-2 coverage | ⚠ ~half exist; Jain's, entropy, propulsion energy, ρ_contact missing | Chunk EX-3.4 |
| Trial harness + CSV | ❌ shared with Exp 1 | Chunk EX-0 |
| Stats analysis | ❌ shared notebook structure | Chunk EX-3.5 |

### 4.2 Chunks

#### EX-3.1 — A1 driver: centralized FL with uniform client sampling

**Goal.** A measured-arm wrapper around the existing Flower path that
records the experiment-level metrics (round close rate, update yield,
coverage, Jain's, entropy) in the same CSV format A2–A4 use.

**Note.** A1 has no mule by design, so mule-specific metrics (mission
completion time, propulsion energy, ρ_contact, Pass-2 coverage) are
recorded as `N/A` per the paper's spec.

**Where.** `experiments/exp3/arm_a1.py`. Imports from
`hermes.cluster.HFLHostCluster` only — no mule, no scheduler, no
selector — and uses Flower's centralized FL strategy with uniform
random client sampling each round.

#### EX-3.2 — A2 + A3 scheduler policies

**Goal.** Two new policies added to `hermes/scheduler/policies/` (new
sub-package) that plug into the supervisor's contact-queue path the same
way the existing distance-sort fallback does:

* **A2 — `ArrivalOrderPolicy`:** services the FL-readiness candidate
  registry in registration order; no skipping, no reordering. ~30 LOC.
* **A3 — `EdfFeasibilityPolicy`:** earliest-deadline-first with a
  feasibility check that skips a candidate if `transit + collect +
  return + upload > remaining_budget`. ~50 LOC.

Both policies expose the same `rank_contacts(...) → List[ContactWaypoint]`
shape `TargetSelectorRL.rank_contacts` does, so the supervisor
constructor takes one of {A2, A3, A4, None} via the existing
`target_selector` arg without further refactor.

**Tests.** Each policy gets a unit test pinning its ordering rule:
A2 keeps registration order; A3 produces EDF with the documented skip.

#### EX-3.3 — Sim env extensions + sweep knobs

**Goal.** Extend `hermes/scheduler/selector/sim_env.py` (or sibling) so
the experiment's parametric sweep is real:

1. **3 base stations** along the field's far edge — currently the sim
   has implicit "return to origin"; needs explicit BS positions.
2. **Time-varying per-channel upload rates** — the cluster-edge link
   bandwidth varies during a trial; today the sim has constant rates.
3. **Fixed-speed mule** — currently the mule "teleports" between
   waypoints with `time_to_complete = SESSION_TIME + TIME_PER_DIST · dist`.
   The paper specifies a fixed-speed mule which materially changes
   propulsion energy at long path lengths; reconcile or update Eq 5.
4. **Configurable N, β, deadline-heterogeneity** — make the existing
   `bucket_size`, `mission_budget`, and deadline-distribution parameters
   into experiment-level knobs the trial grid can sweep over.

**Trade-off.** The existing sim was designed for the smart-vs-dumb
selector A/B at one set of params. Extending it costs more than writing
a focused Experiment-3 sim, but reusing it preserves the A/B test as a
sanity check on the extensions.

#### EX-3.4 — Missing metrics

**Goal.** Add aggregators in `experiments/exp3/metrics.py` for:

* **Jain's fairness index** — `J = (Σxᵢ)² / (N · Σxᵢ²)` over per-device
  service counts. ~10 LOC.
* **Participation entropy** — Shannon entropy of the service share
  distribution. ~10 LOC.
* **Propulsion energy (Eq 5)** — `Tmission · Pidle + Btx · εbit + Lpath ·
  εprop`. The path length `Lpath` is the sum of per-contact transit
  distances, already accumulated by the supervisor; just needs surfacing
  + the εprop calibration constant.
* **ρ_contact** — `Σ |c.devices| / |contacts|` summed across the Pass-1
  contact list, reported per-cell. ~5 LOC.
* **Round close rate at multiple kmin thresholds** — fraction of rounds
  hitting `kmin ∈ {1, N/2, N}` aggregated updates within deadline. ~10 LOC.
* **Update yield** — mean count of updates aggregated per round. ~5 LOC.
* **Coverage ratio** — fraction of scheduled devices serviced ≥ once. ~5 LOC.

All of these read from the JSONL event stream and the supervisor's
`MissionRoundCloseReport` / `MissionDeliveryReport` — the data is
already flowing.

#### EX-3.5 — Statistical analysis notebook

**Goal.** `experiments/analysis/exp3.ipynb` produces:

1. **Pair-wise paired Wilcoxon** on update yield + Jain's fairness for
   A2 vs A1 (slow-deadline claim).
2. **A3 vs A2** on round close rate + mission energy-proxy cost.
3. **A4 vs A3** on update yield + round close rate (the experiment's
   primary novelty).
4. **β-sweep curve**: update yield vs β with one curve per arm,
   faceted by N. The slope-vs-cliff figure.
5. **rrf-sweep curve**: update yield vs rrf with one curve per arm at
   `β=1.0, N=10`. The contact-event-aware-scheduling figure.
6. **ρ_contact bar chart** faceted by rrf, comparing A2/A3/A4.

### 4.3 Definition of Done (Experiment 3)

- A1 / A2 / A3 / A4 each run as measured arms in the trial harness.
- All 9 metrics emit per trial.
- Sweep grid (arm × N × β × het mode × rrf) runs to completion on
  local emulation, 20 trials per cell.
- The 6 paper figures reproduce from the CSV via the analysis notebook.
- AERPAW deployment is config-only (the simulator handles all four arms
  locally; A1 dominates AERPAW wall-clock because it ships real
  Flower over real radios).

**Size:** ~1 sprint. EX-3.1 + EX-3.2 + EX-3.4 are independent and small;
EX-3.3 (sim env extensions) is the longest single task.

---

## 5. Experiments 2 and 4 — out of scope here

* **Experiment 2** (adaptive radio) lives in the L1 / channel-DDQN layer.
  It uses RF telemetry rather than FL traffic, ground base stations
  rather than an edge server, and link-layer rather than session-level
  metrics. A separate plan when L1 work resumes.
* **Experiment 4** (integrated HERMES end-to-end) reuses the Exp-1+3
  trial harness and folds metrics across all of them. Plan when Exp 1
  and Exp 3 close so we know what "integrated" looks like against
  measured baselines.

---

## 6. AERPAW availability caveat

The paper says experiments run "on the AERPAW wireless digital twin."
The system implementation plan notes the testbed is currently down.
Three honest paths:

1. **Wait for AERPAW.** All experiments run on real hardware as
   designed. The local-emulation work this plan describes is dev-mode
   validation of the harness; the paper numbers come from AERPAW.
2. **Run on local emulation, document.** Sprint 2's emulation maps
   1:1 onto AVNs; the wire format, scheduler, and metrics are
   identical. The paper would need a "we ran on local emulation
   because AERPAW was unavailable" note. The selector A/B + scheduler
   ablation results are still valid at the algorithmic level.
3. **Hybrid.** Run cheap arms (A2/A3/A4 sim trials) locally; reserve
   AERPAW time for the costly arms (A1 Flower, Exp 1 centralized
   bulk transfer) when the testbed returns.

The hybrid path is the practical default. Build everything to run
locally first; AERPAW is then a config swap for the arms that need
real RF.

---

## 7. Suggested sequencing

| Sprint | Chunk | Engineer | Notes |
|---|---|---|---|
| 1 | **EX-0** trial harness | both | unblocks 1+3 |
| 2 | **EX-3.1**, **EX-3.2**, **EX-3.4** in parallel | both | A1 driver, A2/A3 policies, missing metrics |
| 2 | **EX-1.2**, **EX-1.3** in parallel | one engineer | partition + energy formula + calibration |
| 3 | **EX-3.3** sim env extensions + retrain A4 on extended sim | one engineer | longest single task |
| 3 | **EX-1.1** centralized driver | one engineer | biggest Exp-1 gap |
| 4 | **EX-1.4** tc/netem runbook | one engineer | OS-level work, can land any time after EX-1.1 |
| 4 | **EX-1.5**, **EX-3.5** analysis notebooks | one engineer | once data exists |
| 5 | Full local-emulation dry run of both experiments | both | catches integration drift |
| (TBD) | AERPAW deployment + real-hardware sweep | both | gated on testbed return |

**Calendar estimate:** ~4–5 sprints to "ready to run on local emulation,"
plus AERPAW deployment effort once the testbed is live.

---

## 8. Open decisions

These need resolving before code lands:

1. **Centralized arm framework choice** — write a 50-line PyTorch
   training loop, or reuse Flower's centralized strategy by setting
   `min_fit_clients = num_clients` and one round? The Flower reuse is
   smaller and reads cleaner; the PyTorch path has fewer moving parts.
2. **Where lives the energy calibration?** Single `experiments/calibration.toml`
   used by both Exp 1 + Exp 3, or per-experiment? Single file is more
   honest (the constants are the same physics).
3. **Sim env reuse vs. rewrite for Experiment 3.** Extending the
   existing selector-A/B sim preserves backward compatibility but adds
   knob complexity. A focused Exp-3 sim is cleaner but means
   maintaining two sims. Decision: extend the existing one unless a
   concrete blocker appears.
4. **Is the existing trained A4 actor still valid after EX-3.3?**
   The DDQN was trained on the unchanged sim env; if EX-3.3 changes the
   reward distribution materially, A4 needs retraining. Design EX-3.3
   to preserve the existing A/B's reward shape so retraining is
   optional, not mandatory.
5. **Should we report local-emulation numbers as "preliminary" in the
   paper while AERPAW is down?** Tied to the §6 path choice.

---

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| AERPAW remains unavailable through paper deadline | All four experiments lose their "real testbed" framing | Hybrid path: dev locally, swap config for AERPAW when it returns; document local-emulation provenance honestly |
| Sim env extensions break the existing selector A/B | Sprint-1.5 DoD test starts flaking | Keep the existing sim as the default; gate the EX-3.3 extensions behind explicit constructor flags |
| Centralized arm wall-clock dominates Experiment 1 budget | A few cells in the grid never finish in 20 trials | Smaller `|D|pd` cells run first; per-cell timeout in the trial harness; resume across multiple AERPAW reservations |
| Trial-harness CSV grows unbounded across 20 × N cells × N arms | Disk + analysis-notebook load times | One CSV per experiment, append-only; analysis notebook reads via pandas with column dtypes pinned |
| εprop (propulsion energy per metre) is hard to source for AERPAW UAVs | Eq 5 has a fudge factor | Ship it as a configurable constant with a sensitivity analysis in the notebook; the paper claim is the *ratio* between arms, not the absolute number |
| A4's existing trained weights diverge from the EX-3.3 extended sim | Retraining sprint silently slips into the schedule | Sanity-test A4 against the extended sim early in EX-3.3; trigger retrain only if the A/B drops below the existing 5% margin |

---

## 10. Exit criteria

The experiments are "ready to run" when:

1. ✅ Trial harness produces a resumable CSV from a parametric grid.
2. ✅ Experiment 1 has both arms (FL, Centralized) wired to the harness;
   all 6 metrics emit; analysis notebook reproduces the 5 paper figures.
3. ✅ Experiment 3 has all four arms (A1–A4) wired to the harness; all
   9 metrics emit; analysis notebook reproduces the 6 paper figures.
4. ✅ Local-emulation full dry-run completes, producing CSVs the
   notebooks can ingest end-to-end.
5. ⏳ AERPAW deployment runbook documents the swap. (Gated on testbed.)
6. ⏳ Final paper-grade run executed on AERPAW (or honest local-emulation
   note in the paper). (Gated on testbed.)

Items 1–4 are achievable in the 4–5 sprint estimate. Items 5–6 are
gated on AERPAW availability, which is outside the project's control.
