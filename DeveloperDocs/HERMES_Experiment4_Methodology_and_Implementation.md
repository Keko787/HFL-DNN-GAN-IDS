# Experiment 4 — Methodology & Implementation

**The measured realization of Algorithm 2.** Where Experiments 1–3 each exercise one layer
against an abstracted model, Experiment 4 composes **L1 + L2 + L3 in a single trial on the real
multi-process stack**, with a real DNN-IDS training in the loop, and measures whether federation
survives when the backhaul does not.

**Companion documents**

| Document | Covers |
|---|---|
| [`HERMES_Experiment4_L1_RF_Layer.md`](HERMES_Experiment4_L1_RF_Layer.md) | Layer 1 — channel model, `U(c,t)` controller, backhaul-loss schedule |
| [`HERMES_Experiment4_L2_Scheduling_Layer.md`](HERMES_Experiment4_L2_Scheduling_Layer.md) | Layer 2 — the gated scheduler, buckets, the bounded RL tie-break |
| [`HERMES_Experiment4_Jittery_Methodology.md`](HERMES_Experiment4_Jittery_Methodology.md) | **The validity record** — physical model, statistics, results, every caveat |
| [`Experiment_4_Run_Guide.md`](Experiment_4_Run_Guide.md) | How to reproduce it |
| [`HERMES_Experiment4_Integrated_Design_and_Plan.html`](HERMES_Experiment4_Integrated_Design_and_Plan.html) | Design rationale and per-stage build status |

---

## 1. The question, and why it needed a new experiment

The SEC26 reviewers' sharpest surviving criticism, which the authors conceded:

> "The three layers are exercised together only in **integration tests**, not in one end-to-end
> experiment."

Algorithm 2 in the paper is literally titled *Integrated HERMES Mission Round*. It threads all
three layers through one round:

```
L1 →  chan_idx  = L1CHANNELSELECT()          ; navigate to c*.pos
L2 →  c*        = SCHEDULERROUND(M, A)       (Algorithm 1: gated + RL tie-break)
L3 →  batch_agg = PARTIALFEDAVG({Δθ_j})      (Pass 1 · COLLECT)
      θ'        = CROSSMULEFEDAVG({mission_agg})  (dock)
      push θ' to devices                     (Pass 2 · DELIVER)
```

Experiment 4 is that algorithm, **measured**. Two things make it different in kind from Exp 3:

1. **Real processes, not a simulator.** Every trial spawns a cluster, a mule and N device
   subprocesses that talk over real TCP; the two-pass Pass-1 → dock → Pass-2 cycle and the
   cross-mule FedAvg are the production code paths, not models of them.
2. **A real model, not a proxy.** Each device runs the canonical `create_CICIOT_Model` on a real
   CIC-IoT-2023 shard, and the cluster scores each aggregated θ on a held-out set — so the
   experiment can report *learning outcomes*, which is what reviewer 74D asked for.

---

## 2. The arm ladder

Each arm adds exactly one layer to the one before, so a difference is attributable.
All arms are **paired-seeded**: the same `(cell, trial_index)` produces the same device layout,
shards, initial θ, and per-device reliabilities in every arm.

| Arm | What it turns on | Where it runs |
|---|---|---|
| **H0** | Traditional flat FL — no mule, all clients each round, single-hop backhaul | in-process |
| **H1** | + mule, gated scheduler, two-pass hierarchical FL, **deterministic** ranking | real orchestrator |
| **H2** | + `TargetSelectorRL` (DDQN) in the S3.5 tie-break | real orchestrator |
| **H3** | + real L1 adaptive channel (`U(c,t)`) | real orchestrator |

> **Do not compare H2/H3 against H0/H1.** They are run with `--l1-channel`, which replaces the flat
> backhaul loss with the RF channel model in *both* arms, and their per-trial seeds do not line up
> with the H0/H1 grid (the H0/H1 cells carry three `link_quality` values, which shifts the derived
> seeds). H1-vs-H0 and H3-vs-H2 are each valid; cross-pair comparisons are not.

> **No RL claim follows from Experiment 4.** The committed runs used a **random-init** selector
> (no trained `.npz` was supplied). That does not confound H3-vs-H2 — both arms carry the same
> selector — but it means H2-vs-H1 measures nothing about learned scheduling. The RL question
> belongs to Experiment 3.

---

## 3. Implementation

### 3.1 The stack under test

`MultiProcessOrchestrator` spawns one cluster + one mule + N device subprocesses, each writing
JSONL events. The trial is **finite and self-terminating**: the mule is capped at `n_missions`, so
the process tree exits on its own and the driver reads the logs afterwards. This matters on
Windows, where `terminate()` skips the `finally` block that emits the final metrics snapshot — so
every trial is designed to exit naturally, with a hard wall-clock budget as a backstop that
records `status=error` rather than hanging the sweep.

| Component | Role in a trial |
|---|---|
| `experiments/exp4/driver.py` | builds the topology, runs the arm, rolls JSONL up into one metric row |
| `experiments/exp4/topology_builder.py` | device placement, contact reliabilities, per-arm config |
| `experiments/exp4/model_task.py` | the real learning task — data, model, train/eval functions |
| `experiments/exp4/events_consumer.py` | JSONL → `Exp4Observation` |
| `experiments/exp4/metrics.py` | `Exp4Observation` → the 32-column metric row |
| `experiments/analysis/exp4.py` | paired statistics over the trial CSV |

### 3.2 The learning task

**"Driver-prepares-once."** Rather than have each device subprocess load and preprocess CIC-IoT
independently — which would be slow and would risk the arms training on different data — the
driver prepares the task **once per trial**, serializes each device's shard, the shared held-out
test set, and the real seeded initial weights, then points the subprocesses at those files.

* **Model:** the canonical `create_CICIOT_Model` (TF/Keras), sigmoid output.
* **Data:** the production CICIOT pipeline (balanced, 21 canonical features), or a real-shaped
  separable synthetic task for dataset-free smoke runs.
* **Initial θ:** one fixed seed shared by every arm, so all arms start from an identical model.

### 3.3 The physical model

The impairment model is **symmetric by construction** — an early version was not, and the
resulting result was invalid (§6). Full justification is in the validity record §2.

| Mechanism | Applies to | Nature |
|---|---|---|
| Per-device contact reliability `U(0.15,1.0) × rf_factor` | **both** H0 and the mule arms | short-range, jitter-**immune** |
| Dead-zone (fraction of devices with no long-range path) | **H0 only** | permanent unreachability; the mule physically flies to them |
| Long-range backhaul loss (mule→BS) | mule arms | **recoverable per-mission** — round does not close, θ carried forward |

The asymmetry is the physics, not a thumb on the scale: **jitter degrades long-range links, not the
short device↔mule hop** — which is the entire point of routing collection through a mule. The
mule's compensating costs (contact failures, backhaul loss) are modelled and *do* bite.

---

## 4. Metrics

The row is 32 columns. They are not equally trustworthy, and the document is explicit about which
carry the claim:

| Class | Metrics | Status |
|---|---|---|
| **Headline — participation** | `mission_completion_rate`, `update_yield`, `round_close_rate@{1,2,N/2,N}` | **report these** |
| **Headline — learning** | `final_auc`, `final_accuracy`, `final_loss`, `best_auc`, `delta_auc`, `t_at_tau_round` | report, with §5.2 caveat |
| Fairness / distribution | `jains_fairness`, `participation_entropy`, `completion_fairness` | secondary |
| Diagnostics | `rounds_closed`, `missions_completed`, `mission_failures`, `pass{1,2}_contacts_mean`, `mission_duration_s_mean`, `rho_contact` | debugging |
| **Descriptors — not evidence** | `coverage`, `pass2_coverage` | **do not headline** |

**Why `coverage` is not evidence.** It counts `device_served` events, which include Pass-2
deliveries — and Pass 2 reaches every device regardless of whether Pass 1 collected anything from
it. So coverage is ≈1.0 for any working mule, whatever the collection outcome. `update_yield` and
`mission_completion_rate` are the honest measures of collection.

**`round_close_rate` is honest.** A round closes only if it produced ≥1 update **and** its backhaul
upload was not dropped. Empty and backhaul-dropped rounds count as *not closed*, so the mule's
jittery penalty is visible rather than hidden.

---

## 5. Results

Canonical CICIOT, N=6, `n_missions=4`, **20 paired seeds**, the full
`dead_zone ∈ {0,0.2,0.4,0.6} × link_quality ∈ {0.3,0.5,0.7}` jittery surface plus a clean
reference. Paired Wilcoxon + Cliff's δ + bootstrap 95 % CI; **a verdict is claimed only when the
CI excludes 0 and p<0.05**.

### 5.1 The crossover — the paper's payoff

> **Units note.** *Participation* here is the normalized form, `update_yield ÷ N` (N=6 devices), so
> it sits in [0,1] alongside the other metrics — this is what the figures plot. The validity record
> quotes the same quantity as raw `update_yield`; e.g. clean H0 0.573 ≡ 3.44 updates/round, H1
> 0.217 ≡ 1.30. The two documents do not disagree.

| Regime | Metric | H0 | H1 | H1−H0 | 95 % CI | Verdict |
|---|---|---|---|---|---|---|
| clean | participation | 0.573 | 0.217 | −0.356 | [−0.429, −0.294] | **H0 > H1** |
| clean | final AUC | 0.985 | 0.958 | −0.027 | [−0.039, −0.017] | **H0 > H1** |
| jittery dz=0.4 | participation | 0.197 | 0.272 | +0.075 | [+0.039, +0.113] | **H1 > H0** |
| jittery dz=0.6 | participation | 0.093 | 0.272 | +0.179 | [+0.154, +0.206] | **H1 > H0** |
| jittery dz=0.6 | final AUC | 0.846 | 0.963 | +0.117 | [+0.067, +0.177] | **H1 > H0** (δ +0.98) |

**Reading.** Under a healthy backhaul, flat FL is strictly better — the mule is overhead, and the
paper says so up front. As devices lose their long-range path the mule's advantage grows
**monotonically**, from a tie to decisive. The crossover boundary sits around **dead-zone ≈ 0.2–0.4**.

**The flip test happened and is reported.** At the well-connected corner (dz=0.0, link=0.7) the
surface *flips*: H0 significantly beats H1. The review asked whether such a point exists; it does.

### 5.2 What the AUC difference actually measures

`final_auc` is **bimodal**: a session either trains (~0.98) or receives no aggregated update and
ends at its untrained initialization (~0.25) — with *nothing in between* (0 of 519 rows fall in
[0.30, 0.65]). Decomposing the pooled jittery gap of +0.023:

| Component | H1 | H0 | Contribution |
|---|---|---|---|
| Session-collapse rate | 0.84 % | 3.35 % | **+0.0177 (76 %)** |
| AUC given the session trained | 0.9646 | 0.9577 | +0.0055 (24 %) |
| Conditional paired test | — | — | +0.0070, CI **[−0.0009, +0.0147]** → **tie** |

**So the honest sentence is "the mule's sessions survive," not "the mule trains a better model."**
Conditional on training at all, the arms are statistically indistinguishable. Lead with
participation; report AUC as the downstream consequence.

### 5.3 Layer 1 — a reported null

Across the H3-vs-H2 dead-zone sweep (200/200 valid trials, 20/20 paired seeds per cell),
**all 5 conditions × 4 metrics are ties**. The channel-model-level effect is real
(§ L1 doc), but it does not resolve end-to-end at this sample size. **No end-to-end accuracy
benefit is claimed for L1.**

---

## 6. Validity record

This experiment has been through **two adversarial audits**, and both changed the result. That
history is kept deliberately — it is the evidence that the numbers survived scrutiny.

| Audit | Finding | Outcome |
|---|---|---|
| Jittery remediation | The first jittery result was **manufactured** — the regime never reached the mule arm, so "H1 jittery" was H1 clean relabelled | Invalid result withdrawn; symmetric physical model, honest `deadline_met`, paired statistics and the sensitivity surface all added |
| 20-skeptic L1 audit | **No rigging found** across rigging/fair-baseline/loss-map/wiring/robustness lenses; 6 framing caveats confirmed | §7.2 caveats recorded; magnitude reported as calibration-dependent |
| AUC>1.0 investigation | Data always in range (max 0.9940); the **figure** misrepresented it via ±SD whiskers on a bounded, bimodal metric | Figure standards now enforced in code (§5.1 of the validity record) |
| — same investigation | **Data-integrity bug:** trials that produced no model were recorded `status=ok`, their blank AUC dropped from one mean while their fabricated 0.0 participation was averaged into another | Driver now records `status=no_eval`; contaminated sweep re-run clean (0/200 failures); the apparent L1 effect **retracted** |

**Two lessons worth carrying into every future experiment**, because both defects were *silent*:

1. A trial must **assert it produced what it claims to have produced** — a run that trained no
   model must not be able to masquerade as a success.
2. A figure must draw its numbers from **the same code that produces the tables**, or the two will
   eventually disagree — as they did, by ~0.19 AUC.

---

## 7. Scope boundary — stated, not hidden

Experiment 4 models the **network and computation** layers with real fidelity. It does **not**
model:

* **Mule flight physics** — propulsion energy, mission time under a tight flight budget β, or
  coverage limited by that budget. That is Experiment 3's territory, and the two are complementary.
* **Unified cross-layer energy** — L1 switching + L2 flight + L3 compute in one ledger. `ε_bit` is
  now reconciled (7.0×10⁻¹⁰ J/bit, verified) but `ε_prop` remains a placeholder, so **only
  normalized energy is reportable**.
* **Real RF** — the channel is modelled, with perfect cost-free sensing (see the L1 document).
* **Scale** — N=6 devices and a single mule; multi-mule behaviour is untested here.

Consequently Exp 4 validates **participation/convergence resilience** (Observation 3), not the
budget-scheduling claim (Observation 4).

---

## 8. Reproducing it

Full instructions: [`Experiment_4_Run_Guide.md`](Experiment_4_Run_Guide.md).
Pinned environment: `AppSetup/requirements_exp4.txt` (Python 3.11 stack that produced the CSVs).

```bash
bash experiments/exp4/run_paper_sweep_parallel.sh      # H0/H1 surface + clean reference
bash experiments/exp4/run_l1_deadzone_sweep.sh          # H2/H3 L1 dead-zone sweep
```

```bash
python -m experiments.analysis.exp4 --csv results/exp4_paper/h0h1_all.csv --surface --surface-metric mission_completion_rate
```

```bash
python -m experiments.analysis.exp4 --csv results/exp4_paper/h2h3_dz_dz06.csv --treatment H3 --baseline H2
```

Sweeps are **resumable** — a re-run skips rows already present — and shard concurrency is capped
(`MAX_PAR`), because at 5 concurrent shards the TensorFlow process trees exhausted memory and ~30 %
of trials failed to bootstrap.

| Artifact | Path |
|---|---|
| H0/H1 surface (519 ok rows) | `results/exp4_paper/h0h1_all.csv` |
| H2/H3 L1 sweep (200 ok rows, 20/20 paired per cell) | `results/exp4_paper/h2h3_dz_*.csv` |
| Figures + generators | `results/exp4_paper/fig_exp4_*.png`, `DeveloperDocs/exp4_figure*.py` |
| Shared figure style + guards | `experiments/analysis/figstyle.py` |
| Tests | `tests/unit/test_exp4_*.py`, `tests/integration/test_exp4_realmodel_smoke.py` |
