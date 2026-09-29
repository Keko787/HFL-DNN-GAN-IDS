# HERMES Configuration Reference

**Status:** Phase 7 living document. Companion to
[HERMES_FL_Scheduler_Design.md](HERMES_FL_Scheduler_Design.md) and
[HERMES_FL_Scheduler_Implementation_Plan.md](HERMES_FL_Scheduler_Implementation_Plan.md).

This is the **single source of truth** for every tunable in HERMES — every
weight, threshold, timeout, learning-rate, and calibration constant.
A deployment engineer or paper reviewer landing on the codebase should be
able to answer "what value am I using and why?" without grepping the source.

Each entry lists:

* **Symbol** — the name in code.
* **Where** — `file:line` of the definition.
* **Default** — current value.
* **Surface** — *config field* (changeable without code edit, via
  `TopologyConfig` / `ClusterConfig` / etc.), *constructor arg* (passed in
  at object creation), or *module constant* (requires a code edit + new
  release to change).
* **Rationale** — why this value, and what would move it.

Entries marked **(open)** are tunables the design doc still has open
decisions about — see §8 of the implementation plan and §9 of the design
doc.

---

## 1. Utility scoring weights (S2B device readiness)

The scheduler's S2B gate ranks devices by a composite utility score
combining performance, diversity, and freshness. Weights are passed as
function arguments so per-experiment overrides don't require a code
release.

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `w1` (perf weight) | [hermes/mission/utility.py:104](hermes/mission/utility.py:104) | 0.7 | function param | Performance dominates utility — devices with strong recent rounds rank higher. Reset between experiments via the calling function's signature. |
| `w2` (diversity weight) | [hermes/mission/utility.py:105](hermes/mission/utility.py:105) | 0.3 | function param | Diversity bonus — keeps the cluster from converging on a single client's update. **(open)** §8 #2 keeps the FL_Threshold-vs-adaptive question open; same applies to whether `w1+w2` should be learned per cluster. |
| `w_acc` (accuracy weight, perf sub-score) | [hermes/mission/utility.py:62](hermes/mission/utility.py:62) | 0.5 | function param | Accuracy is the headline metric in CICIOT-2023 binary classification. |
| `w_auc` (AUC weight, perf sub-score) | [hermes/mission/utility.py:63](hermes/mission/utility.py:63) | 0.3 | function param | AUC catches threshold-blind cases where accuracy alone is misleading. |
| `w_loss` (loss weight, perf sub-score) | [hermes/mission/utility.py:64](hermes/mission/utility.py:64) | 0.2 | function param | Loss as a tiebreaker; clipped at `loss_cap` so an outlier round can't dominate. |
| `loss_cap` | [hermes/mission/utility.py:65](hermes/mission/utility.py:65) | 10.0 | function param | Ceiling before normalisation; raise if your model's typical loss is >10. |

## 2. FL_Threshold — S2B utility cutoff

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `DEFAULT_FL_THRESHOLD` | [hermes/scheduler/stages/s2b_flag.py:23](hermes/scheduler/stages/s2b_flag.py:23) | 0.60 | constructor arg via `FLScheduler(fl_threshold=...)` | Cutoff above which a device's `FL_READY_ADV` passes S2B. Set at the AC-GAN quality bar — 0.6 reflects "this device's local model is contributing meaningful signal." **(open)** §8 #2: should this be adaptive per cluster? Currently static. |

## 3. Min-participation threshold

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `min_participation` | [hermes/cluster/host_cluster.py:117](hermes/cluster/host_cluster.py:117) | 1 | `ClusterConfig.min_participation` field | Minimum mules whose UP must arrive before cross-mule FedAvg fires. Default 1 = **partial-FedAvg** (aggregates with whoever's there). Set to `len(mules)` for full-FedAvg semantics (lockstep, no aggregation until everyone reports). Sprint 2 chunk-L wired this through `ClusterConfig`. |

## 4. Selector reward weights (offline sim env)

These drive the DDQN training reward in
`hermes/scheduler/selector/sim_env.py`. Constants today; if a future
chunk swaps the reward calibration, lift them into a `RewardConfig` dataclass.

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `ENERGY_W` | [hermes/scheduler/selector/sim_env.py:54](hermes/scheduler/selector/sim_env.py:54) | 0.002 | module constant | Energy penalty per unit. Set so that energy term roughly matches one-tenth of `COMPLETION_BONUS` for typical episodes. |
| `COMPLETION_BONUS` | [hermes/scheduler/selector/sim_env.py:59](hermes/scheduler/selector/sim_env.py:59) | 200.0 | module constant | Reward for a single completed FL session. Calibrated so episode total reward sits in roughly [-200, +600] range. |
| `SESSION_TIME` | [hermes/scheduler/selector/sim_env.py:67](hermes/scheduler/selector/sim_env.py:67) | 30.0 | module constant | Base FL exchange duration in seconds (sim units). |
| `TIME_PER_DIST` | [hermes/scheduler/selector/sim_env.py:68](hermes/scheduler/selector/sim_env.py:68) | 0.1 | module constant | Travel time penalty per distance unit. Tunes the trade-off between flying further to a reliable device vs. visiting a flaky one nearby. |
| `TIME_NOISE_STD` | [hermes/scheduler/selector/sim_env.py:69](hermes/scheduler/selector/sim_env.py:69) | 1.0 | module constant | Gaussian σ on per-step duration. Stops the policy from over-fitting to a deterministic timeline. |

## 5. Selector DDQN hyperparameters

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `hidden` | [hermes/scheduler/selector/ddqn.py:88](hermes/scheduler/selector/ddqn.py:88) | 16 | constructor arg | Hidden-layer width. Small enough to fit on the NUC alongside L1 channel actor. |
| `lr` | [hermes/scheduler/selector/ddqn.py:89](hermes/scheduler/selector/ddqn.py:89) | 0.01 | constructor arg | SGD learning rate; standard DDQN range. |
| `gamma` | [hermes/scheduler/selector/ddqn.py:90](hermes/scheduler/selector/ddqn.py:90) | 0.5 | constructor arg | Discount factor — short horizon; we care about per-contact reward, not multi-mission credit assignment. |
| `target_sync_every` | [hermes/scheduler/selector/ddqn.py:91](hermes/scheduler/selector/ddqn.py:91) | 200 | constructor arg | Steps between target-network syncs. |
| `buffer_capacity` | [hermes/scheduler/selector/replay.py:41](hermes/scheduler/selector/replay.py:41) | 10 000 | constructor arg | Replay buffer size — enough for ~50 episodes of contact transitions. |
| `epsilon_start` | [hermes/scheduler/selector/selector_train.py:53](hermes/scheduler/selector/selector_train.py:53) | 0.9 | `TrainConfig` field | ε-greedy starting value. |
| `epsilon_end` | [hermes/scheduler/selector/selector_train.py:54](hermes/scheduler/selector/selector_train.py:54) | 0.05 | `TrainConfig` field | ε-greedy floor. |
| `epsilon_decay_episodes` | [hermes/scheduler/selector/selector_train.py:55](hermes/scheduler/selector/selector_train.py:55) | 300 | `TrainConfig` field | Episodes over which ε linearly decays. |

## 6. Selector feature scaling

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `_DISTANCE_SCALE` | [hermes/scheduler/selector/features.py:55](hermes/scheduler/selector/features.py:55) | 100.0 | module constant | Divides distance feature so inputs sit ≈ ±3 for the tanh-activated DDQN. Matches the 100 m world radius in sim. |
| `_POS_SCALE` | [hermes/scheduler/selector/features.py:56](hermes/scheduler/selector/features.py:56) | 100.0 | module constant | Same idea for x/y/z position features. |
| `FEATURE_DIM` | [hermes/scheduler/selector/features.py:49](hermes/scheduler/selector/features.py:49) | 11 | module constant | Selector feature vector width. Bumping this requires retraining the actor. |

## 7. Scheduler stages

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `MIN_DEADLINE_FULFILMENT_S` | [hermes/scheduler/stages/s3_deadline.py:48](hermes/scheduler/stages/s3_deadline.py:48) | 5.0 | module constant | Floor on the rolling fulfilment window. Stops fast-phase shrinks from collapsing the deadline to zero. |
| `FAST_PHASE_ON_TIME_SHRINK_S` | [hermes/scheduler/stages/s3_deadline.py:44](hermes/scheduler/stages/s3_deadline.py:44) | 5.0 | module constant | How much a CLEAN delta tightens the next deadline window. |
| `FAST_PHASE_MISSED_WIDEN_S` | [hermes/scheduler/stages/s3_deadline.py:45](hermes/scheduler/stages/s3_deadline.py:45) | 10.0 | module constant | How much a TIMEOUT/PARTIAL delta loosens the next deadline window. Asymmetric (widen > shrink) to err on the side of attempt rather than skip. These three constants are the recorded **additive** law; FeRRy's multiplicative law is configured in §15. |
| `beacon_window_s` | [hermes/scheduler/fl_scheduler.py:79](hermes/scheduler/fl_scheduler.py:79) | 30.0 | constructor arg | How recent a beacon must be to count as "active." |
| `BUCKET_PRIORITY` | [hermes/types/scheduler.py](hermes/types/scheduler.py) | `[NEW, SCHEDULED_THIS_ROUND, BEACON_ACTIVE]` | enum order | Bucket-walking order in `build_target_queue` / `build_contact_queue`. |

## 8. Mission + ClientMission timeouts

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `solicit_timeout_s` | [hermes/mission/client_mission.py:116](hermes/mission/client_mission.py:116) | 30.0 | `ClientMission` constructor arg | How long a device waits for an FL_OPEN solicit before treating the contact as missed. |
| `disc_push_timeout_s` | [hermes/mission/client_mission.py:117](hermes/mission/client_mission.py:117) | 30.0 | `ClientMission` constructor arg | How long a device waits for the discriminator push after it sends FL_READY_ADV. |
| `session_ttl_s` | `MuleConfig.session_ttl_s` | 5.0 | `MuleConfig` field | Per-contact TTL on the supervisor side. Sprint-2 multi-process tests use 2-3 s; AERPAW deployment may need longer if RF is slow. |
| `synth_batch_size` | `ClusterConfig.synth_batch_size` | 4 | `ClusterConfig` field | Number of synthetic samples per DOWN bundle. |

## 9. Transport (TCP + channel emulator)

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `accept_timeout_s` (RF) | [hermes/transport/tcp_rf_link.py:109](hermes/transport/tcp_rf_link.py:109) | 0.25 | `TCPRFLinkServer` constructor | Listener `select()` budget — short so the accept loop can respond to shutdown. |
| `send_timeout_s` (RF) | [hermes/transport/tcp_rf_link.py:110](hermes/transport/tcp_rf_link.py:110) | 30.0 | `TCPRFLinkServer` constructor | Per-message `sendall` cap. RF messages are small (kilobytes); 30 s is a generous ceiling for slow links. |
| `accept_timeout_s` (Dock) | [hermes/transport/tcp_dock_link.py:88](hermes/transport/tcp_dock_link.py:88) | 0.25 | `TCPDockLinkServer` constructor | Same role as RF, dock-side. |
| `send_timeout_s` (Dock) | [hermes/transport/tcp_dock_link.py:89](hermes/transport/tcp_dock_link.py:89) | 60.0 | `TCPDockLinkServer` constructor | Higher than RF — UP/DOWN bundles can be hundreds of MB for real models. |
| `connect_timeout_s` | [hermes/transport/tcp_dock_link.py:369](hermes/transport/tcp_dock_link.py:369) | 5.0 | client constructor | Mule client reach-cluster window. |
| `drop_prob` | [hermes/transport/channel_emulator.py:51](hermes/transport/channel_emulator.py:51) | 0.0 | `ChannelEmulator` field | Synthetic packet-drop probability. Production transports leave this at 0; experiments can dial in fault scenarios. |
| `mean_delay_s` | [hermes/transport/channel_emulator.py:52](hermes/transport/channel_emulator.py:52) | 0.0 | `ChannelEmulator` field | Mean per-message latency. Use AERPAW link characterisation when available. |
| `jitter_s` | [hermes/transport/channel_emulator.py:53](hermes/transport/channel_emulator.py:53) | 0.0 | `ChannelEmulator` field | ±half-range around `mean_delay_s`. |

## 10. Cloud link (Tier-3)

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `tier3_url` | `ClusterConfig.tier3_url` | None | `ClusterConfig` field | When None, no cloud link is wired. When set, cluster polls Tier-3 every 5 s and folds returned `GeneratorRefinement` into the local generator (Phase 7 Chunk P2). |
| `_TIER3_POLL_INTERVAL_S` | [hermes/processes/cluster.py](hermes/processes/cluster.py) | 5.0 | module constant on `ClusterService` | Throttles the poll loop. Tier-3 is best-effort, so polling more aggressively only burns network. **(open)** §8 #4: cadence can move once Tier-3's actual refinement rate is known. |
| `request_timeout_s` (HTTP) | [hermes/transport/cloud_link.py:127](hermes/transport/cloud_link.py:127) | 10.0 | constructor arg | Per-HTTP-call timeout. |

## 11. Process orchestration

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `_BOOTSTRAP_TICK_S` | [hermes/processes/mule.py](hermes/processes/mule.py) | 1.0 | module constant on `MuleService` | Mule's bootstrap-wait granularity — short so a SIGTERM during startup is honoured within ~1 second instead of hanging on the 60 s device-wait loop. |
| `_STDERR_TAIL_LINES` | [hermes/processes/orchestrator.py:67](hermes/processes/orchestrator.py:67) | 200 | module constant | Ring-buffer depth for the orchestrator's stderr drainer (chunk L-L8). Surfaces the last 200 lines of a crashed subprocess in `OrchestratorError`. |
| `shutdown_all` default `timeout` | [hermes/processes/orchestrator.py](hermes/processes/orchestrator.py) | 15.0 | method kwarg | Generous enough to swallow a mule mid-mission when SIGTERM arrives (chunk L-M5). |
| `accept_timeout_s` (orchestrator port-out poll) | [hermes/processes/orchestrator.py](hermes/processes/orchestrator.py) | 0.05 (50 ms) | inline constant | How often `_wait_for_port` checks the port-out file. |

## 12. AERPAW calibration constants (Experiment-time)

These ship with the experiments plan, not the system plan — they belong
in [HERMES_Experiments_Implementation_Plan.md](HERMES_Experiments_Implementation_Plan.md)
chunks EX-1.3 and EX-3.4. Listed here for completeness so a paper
reviewer can find every numeric input in one document.

| Symbol | Source | Status |
|---|---|---|
| `Pidle` | AERPAW USRP front-end spec sheet | **not yet sourced** — Phase 7 deployment work, recorded in `experiments/calibration.toml` per the experiments plan |
| `εbit` | AERPAW USRP front-end spec sheet | same as above |
| `εprop` | AERPAW UAV propulsion spec | same as above; sensitivity analysis in `exp3.ipynb` |
| `Bnominal` | tc/netem shaped link | 10 Mbps for Experiment 1; configured at OS level |

---

## How to change a value

The order of preference, cheapest first:

1. **Config field** (e.g. `ClusterConfig.min_participation`) — change the
   topology JSON; no rebuild.
2. **Constructor arg** (e.g. `DDQN(lr=0.005)`) — change the calling code;
   no library edit.
3. **Module constant** — last resort. Requires a `hermes/` source edit
   and a release. Document why in the commit message; if a value is
   tuned often enough that a code edit is friction, lift it into a
   config field in a follow-up chunk.

If you change any of these for a paper run, record the changed value
in your experiment's `calibration.toml` (or wherever the experiments
plan lands the audit trail) so the paper reproduces.

---

## 13. Canonical model choice — CICIOT NIDS

The IDS model used **across the board for this project** when the
target dataset is CICIOT-2023 is `create_CICIOT_Model` from
[Config/modelStructures/NIDS/NIDS_Struct.py:214](Config/modelStructures/NIDS/NIDS_Struct.py:214).
Anywhere the experiments or the integrated system instantiate the IDS
model, this is the function.

| Property | Value |
|---|---|
| Architecture | 5 Dense layers (`64 → 32 → 16 → 8 → 4 → 1`), each with `BatchNormalization` + `Dropout(0.4)`; ReLU activations + L2-regularized kernels; sigmoid output for binary classification |
| Source | [Config/modelStructures/NIDS/NIDS_Struct.py:214 `create_CICIOT_Model`](Config/modelStructures/NIDS/NIDS_Struct.py:214) |
| Instantiation site | [Config/SessionConfig/modelCreateLoad.py:60-63](Config/SessionConfig/modelCreateLoad.py:60) — `if dataset_used == "CICIOT": nids = create_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha)` |
| Hyperparameters | Per [Config/SessionConfig/hyperparameterLoading.py:34-50](Config/SessionConfig/hyperparameterLoading.py:34) — `input_dim = X_train.shape[1]` (typically 21 for CICIOT post-preprocessing), `BATCH_SIZE = 64`, `learning_rate = 0.0001`, `l2_alpha = 0.0001` |
| Approximate parameter count | ~4,700 trainable params; ~18.8 KB at float32 |
| On-the-wire size for FL | `|θ| ≈ 18.8 KB` per round per direction (paper-faithful value for `--theta-bytes` in Experiment 1's FL arm; the 200,000-byte default in `experiments/exp1/server.py` is a round-number placeholder for smoke runs) |

### Project-wide invariants on the two flags

`create_CICIOT_Model` takes two boolean flags (`regularizationEnabled`,
`DP_enabled`) that select between four architectural variants. The
project locks them as follows for every run:

| Flag | Setting | Why |
|---|---|---|
| `DP_enabled` | **`False` — always, until further notice** | The differential-privacy code path (TF-Privacy `DPKerasAdamOptimizer` in `nidsModelCentralTrainingConfig.py` and `NIDSModelClientConfig.py`) is **currently broken** and not part of any paper run. Treat `DP_enabled = True` as a known-bad code path; do not turn it on without first fixing the underlying TF-Privacy integration. |
| `regularizationEnabled` | **`True` is the typical setting; `False` is optional** | L2 regularization on the Dense kernels (`l2_alpha = 1e-4`) plus the BatchNorm + Dropout layers. Disable only for ablation runs that explicitly want to study the effect of regularization on the IDS classifier. |

**Operative branch in `create_CICIOT_Model`.** Because `DP_enabled` is
locked to `False`, the canonical instantiation always lands in the
**first** `if regularizationEnabled:` branch (the 64→32→16→8→4→1
stack with L2 + BN + Dropout(0.4)). The other three branches —
including the `elif regularizationEnabled and DP_enabled` branch, which
is dead code under Python's `if/elif` ordering anyway — are not
exercised in any current run.

> **Watch when DP is fixed.** When the DP-broken status is resolved,
> two things need to land together: (1) the underlying TF-Privacy
> integration, and (2) a fix to the conditional ordering in
> `create_CICIOT_Model` so the `regularizationEnabled and DP_enabled`
> branch is actually reachable (today it is shadowed by the unguarded
> `if regularizationEnabled:` above it). Re-enabling DP without
> re-ordering the branches would silently use the regularization-only
> architecture — which would not be the intended DP+reg combined
> variant.

### Variants in `NIDS_Struct.py` — informational

The same module ships several other architectures (`create_high_performance_nids`, `create_balanced_nids`, `create_lightweight_nids`, `create_optimized_NIDS_model`, `create_optimized_model`, `cnn_lstm_gru_model_*` for IoT). These are alternatives — **not the project default**. The `modelCreateLoad.py` factory currently routes CICIOT explicitly to `create_CICIOT_Model`; switching to one of the alternatives is its own decision and should land as a separate code review, not a silent default change.

### Where this matters for the experiments

* **Experiment 1 (FL vs Centralized)** — does *not* call the model at training time (paper §IV-B excludes SGD compute from `Tproc`). The IDS model only matters for `--theta-bytes`: pass `--theta-bytes 18800` for a paper-faithful FL byte count, or compute it precisely from `model.count_params() * 4` once the data is loaded so `input_dim` is known.
* **Experiment 3 (scheduling ablation)** — A1 (centralized FL) needs the model for the actual Flower training round. A2/A3/A4 are scheduling sims — they don't touch the model.
* **Experiment 4 (integrated end-to-end)** — calls the model at every device's local-train step. This is the primary consumer.
* **HERMES `--mode hermes` path** — when wired, `ClientMission`'s `local_train` callback should call into this model's `.fit` step. Sprint-1B left a stub at [App/TrainingApp/Client/TrainingClient.py:192-197](App/TrainingApp/Client/TrainingClient.py:192) (`_stub_train`) that explicitly raises pending Sprint-1.5 / Sprint-2 wiring; replace with a wrapper around `create_CICIOT_Model` + `model.fit()` when integrating.

---

## 14. L3 merge rules and update age (FeRRy Phase 1)

The merge rule is chosen once per run and must be the same on the cluster and
the mule; the Exp 4 driver sets both from `--aggregation`. Every default below
reproduces the recorded runs. Code: [hermes/mission/aggregation_rules.py](../hermes/mission/aggregation_rules.py).

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `aggregation` | `ClusterConfig`, `MuleConfig` | `agg:plain` | config field; `--aggregation` | `agg:plain` is the num_examples-weighted mean of full models that every recorded run used, on its original code path. `agg:cutoff` (FeRRy), `agg:asynchfl` and `agg:fedbuff` merge deltas weighted by age. `agg:fedex` (Phase 2, arm D4) is FedEx-Async's server step: the mule sends the sum of its updates and the cluster applies θ + η·Σ/N on each return, unweighted by n or age. `agg:seq` (needs in-session training) is refused by name. |
| `server_lr` (η) | `aggregation_params` | 1.0 | config field; `--agg-server-lr` | The cluster adds η × the merged update to θ. At η = 1 with every basis current and `value = uniform`, the age-aware rules equal `agg:plain`, which the regression test pins. |
| `hinge_a`, `hinge_b` | `aggregation_params` | 1.0, 0 | config field; `--agg-hinge-a/-b` | FedAsync's hinge s(a) = 1 for a ≤ b, else 1/(a·(a − b) + 1) (`agg:cutoff`). To be replaced by the constants the theory track derives. |
| `a_max` | `aggregation_params` | None | config field; `--agg-a-max` | Fixed cutoff in cluster rounds; weight is exactly 0 past it. At the cluster it also cuts whole partials older than `a_max`; a fold whose every partial is cut takes no step, leaves the round open and sends every waiting mule its DOWN (event `cluster_merge_expired`). An update whose age is unknown (no basis version) counts as age 0 and is never cut. |
| `period_s` (T) | `aggregation_params` | None | config field; `--agg-period-s` | Mission period for the per-device cutoff of decision D5 (below). |
| `decay` (λ) | `aggregation_params` | 0.5 | config field; `--agg-decay` | `agg:asynchfl`: s(a) = exp(−λ·a) on each device's age at the mule, and on each partial's age at the cluster. |
| `value` | `aggregation_params` | `uniform` | config field; `--agg-value` | v_i in w_i = n_i·v_i·s(a_i): 1, or the update's raw training loss (`loss`), taken over the updates admitted past the cutoff; an update with no loss gets the mean of the known ones. Raw rather than divided by a mean, so partials from several mules combine exactly as one merge over all their devices. |
| `buffer_k` (K) | `aggregation_params` | the slice size of the mule whose partial first opens the buffer (all registered devices if that slice is empty), fixed for the run | config field; `--agg-buffer-k` | `agg:fedbuff`: updates buffered per server step. K is FedBuff's own quorum, so `min_participation` does not gate it. While the buffer fills the round stays open and θ unchanged; the service still sends the mule its DOWN (event `cluster_merge_deferred`). A buffer still filling when the trial ends never reaches θ, and the mean is not n-weighted, so FedBuff at the default K does not tie with `agg:plain` even with every basis current; run it with K = 1 for that check. |
| `fedex_n` (N) | `aggregation_params` | the cluster's registered devices | config field | `agg:fedex`: N in x ← x + (1/N)·Σ_i Δθ_i, the total number of clients (FedEx-Async, TMC 2025). Faithful at η = 1 and `min_participation` = 1, so every return is its own step. |
| `fedprox_rho` (ρ) | `DeviceConfig` | 0.0 | config field; `--fedprox-rho` | Local loss + (ρ/2)·‖θ − θ_received‖² over trainable weights, in a custom loop. 0 keeps the plain Keras `fit`. |
| `pass_2_budget` | `MuleConfig` | False | config field; `--pass-2-budget` | Walks Pass 2 against `mission_budget_s` as a second sortie with the full budget, priced with the S3b cost model from the mule's tracked pose (its last Pass-1 stop until Phase 3 returns the pose to the dock), skipping rather than stopping at a contact that does not fit. Skipped devices get a `SKIPPED` delivery line and keep their older basis; without it every basis is current and ages never spread. With it on, the Pass-1 push also asks each device to train ahead on the basis it adopts, on a background thread, so a device collected in Pass 1 and skipped in Pass 2 ships its next update one round old instead of training in session on the new θ. Needs `mission_budget_s`. |

**How staleness enters the step.** The mule merges its admitted updates as

    Δ_m = Σ_i n_i·v_i·s(a_i)·Δθ_i / M_m,   M_m = Σ_i n_i·v_i   (admitted updates only)

and the cluster, holding θ at version V, folds the live partials (those with
s(V − v_m) > 0) as

    θ ← θ + η · Σ_m M_m·s(V − v_m)·Δ_m / Σ_m M_m

The normaliser M_m is staleness-free, so staleness shrinks the step instead of
only redistributing weight: a mission whose updates are all one round old moves
θ by s(1) × the mean update, and a single partial folds as θ + η·s(V − v_m)·Δ_m,
the FedAsync / Async-HFL mixing form. With every age 0, `value = uniform` and
η = 1 both reduce to the plain mean. `pass_1_merge.weights` in the trace are
w_i / M_m, which sum to the mass-weighted mean staleness: 1 only when no update
is discounted (s(a_i) = 1 for every i, e.g. every age ≤ `hinge_b` under
`agg:cutoff`). `agg:plain` records no shares, so its `weights` list is empty.

**Age and the cutoff (decision D5).** Model versions are cluster rounds: the
DOWN bundle's `mission_slice.issued_round` is the version of its θ. An update's
age is the version of the θ the mule carried minus the version its device
trained from, so ages are global and comparable across mules. Under
`agg:cutoff` each device's cutoff comes from its own deadline window:

    a_max_j = ⌊ Φ_j · s / T ⌋,   Φ_j = max(MIN_DEADLINE_FULFILMENT_S, deadline_fulfilment_s_j)

with s the S3c window scale (1.0 unless S3c is on) and T = `period_s`. If
`a_max` is also set, the smaller cap wins. Φ_j and s are the values the device
was planned under: the mule computes the caps once per mission, right after the
Pass-1 plan, before any session folds its outcome into Φ_j (a CLEAN would
tighten it) or S3c moves its scale. Φ_j and T must be on the same clock.
On the Exp 4 harness that is the wall clock, where one mission cycle takes
about 9–10 s and Φ starts at 60 s, so a period near the measured cycle gives a
cutoff of about 6 rounds and does not bind in a 4-mission trial; report the T
used with every run. From FeRRy Phase 3 both are simulated seconds.

---

## 15. Deadline law and miss priority (FeRRy Phase 1)

Set on the mule (`MuleConfig`); the Exp 4 driver sets them from `--deadline-law`
and `--miss-priority`. The defaults are the recorded behaviour. Code:
`DeadlineLaw` in [hermes/scheduler/stages/s3_deadline.py](../hermes/scheduler/stages/s3_deadline.py).

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `deadline_law` | `MuleConfig` | `additive` | config field; `--deadline-law` | `additive` is the recorded law (§7): −5 s on time, +10 s on any miss, 5 s floor, no ceiling, sticky cluster overrides. `multiplicative` is Φ ← clamp(β·clamp(Φ)): the step applies to the clamped window the deadline uses, so a stored Φ outside the clamps still moves on its first outcome. |
| `beta_on` | `deadline_params` | 0.8 | config field; `--deadline-beta-on` | Factor after an on-time delivery (< 1: tighten). |
| `beta_partial` | `deadline_params` | 1.25 | config field; `--deadline-beta-partial` | Factor after a miss by a device that answered (its advert arrived) — a PARTIAL, or a TIMEOUT after the advert such as Exp 4's lost uplink. The device was reachable, so it relaxes less. The mule marks this with `RoundCloseDelta.answered`. |
| `beta_timeout` | `deadline_params` | 1.5 | config field; `--deadline-beta-timeout` | Factor after a miss by a device that never answered, including the synthetic TIMEOUTs for devices S3b dropped or an abort abandoned (marked `synthetic`). |
| `phi_min`, `phi_max` | `deadline_params` | 5 s, 300 s | config field; `--deadline-phi-min/-max` | Clamps. The ceiling is what the additive law lacks. |
| `expire_overrides` | `deadline_params` | follows the law (on for multiplicative) | config field | A cluster deadline override stops applying once its time passes or the device's next outcome arrives. Under the recorded law it is never cleared (SEC26_Code_Audit.md). |
| `miss_priority` | `MuleConfig` | False | config field; `--miss-priority` | S3b's admission walk orders contacts by their members' consecutive misses (`DeviceSchedulerState.miss_streak`, reset by a CLEAN) before their deadline, so the wider window a miss earns no longer sends the device to the back of the queue. The visit order of what is admitted is unchanged. |

**Why this form, and the defaults.** log Φ moves by log β per outcome, so a
device on time with probability p (misses being timeouts) drifts by
p·log β_on + (1 − p)·log β_timeout per contact (β_partial replaces β_timeout
for the misses of a device that answered). That is zero at
p* = log β_timeout / (log β_timeout − log β_on) = 0.645 with the defaults,
close to the additive law's break-even of 2/3. Devices more reliable than p*
tighten toward Φ_min, less reliable ones relax toward Φ_max, and Φ is bounded
either way. The β values and clamps are placeholders until the theory track
supplies them; sweep them rather than tune them per result.

**One interaction to know.** A device that was once on time has deadline
t_last_on_time + Φ, so once it goes unserved for longer than Φ, S3b drops it as
overdue, and each drop widens Φ. Under the additive law Φ grows 10 s per drop,
about as fast as Exp 4's clock advances per mission, so it rarely catches up.
The multiplicative law catches up geometrically, but not past Φ_max: a device
unserved for more than Φ_max after its last on-time contact stays overdue until
the age cap (FeRRy Phase 4, which exempts capped devices from the overdue check)
brings it back. A device never on time has deadline now + Φ and is overdue only
if the flight alone exceeds Φ. At Exp 4's time scales (trials of about a minute)
the Φ_max case does not arise.

---

## 16. Several mules and the Phase 2 baselines (FeRRy Phase 2)

Every default reproduces the one-mule topology every recorded run used, byte for
byte (Freeze §5i). The Exp 4 driver sets these from its flags; `N` stays the
TOTAL device count, split across the mules. Code:
[experiments/exp4/topology_builder.py](../experiments/exp4/topology_builder.py),
[hermes/processes/cluster.py](../hermes/processes/cluster.py),
[hermes/mule/client_cluster.py](../hermes/mule/client_cluster.py).

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `n_mules` (K) | driver, topology builder | 1 | `--n-mules` | At K > 1 the mules are `exp4-mule-<k>`, the seeded devices are split into K contiguous angular sectors around the dock (sizes within one) so each mule tours its own area, every mule starts at the dock, and backhaul loss draws from one stream per mule so paired seeds hold whatever the upload order. Arm D4 replaces the sectors with its CARP assignment. |
| `min_participation` | `ClusterConfig` | 1 | `--min-participation` | Partials per cluster merge. With several mules it must be 1 (asynchronous: each return is a merge; needs an age-aware rule or `agg:fedex`) or K (synchronous rounds). `agg:plain` at K > 1 needs K: at 1 each merge would overwrite θ with one mule's mean. A value strictly between is refused because the last partial of the run can be stranded; FedBuff is exempt (K is its quorum). Under a quorum above 1 a lost backhaul upload holds its mule's place with an empty partial, so the mules stay in step. |
| `dock_on_empty` | `MuleConfig` | off; on at K > 1 | `--dock-on-empty` | An empty mission still docks, uploading an empty partial (with its round report and Pass-2 ledger) that counts toward the quorum and adds nothing to θ; it takes the current θ and skips Pass 2. Without it a quorum of K deadlocks as soon as one mule collects nothing. |
| `down_wait_s` | `MuleConfig` | None (one 10 s wait whose expiry ends the loop); the trial budget at K > 1 | `--down-wait-s` | How long a docked mule waits for its DOWN. Set, a timeout is survived: `dock_down_timeout`, restage the Pass-1 θ and version, skip Pass 2, fly the next mission. Queued stale DOWNs are also drained before each upload. |
| `contact_policy` | `MuleConfig` | None (our pipeline) | arm | `max_aoi` (D1), `oort` (D2), `whittle` (D3), `fedex` (D4), `fedcs` (D5): whole-scheduler baselines that replace S3, S3b and S3.5. The in-flight re-check is the budget only for D1–D3 and D5, and none for D4 (Freeze §5h). |
| `whittle_variant` | `MuleConfig` | `expected` | `--whittle-variant` | D3: `expected` is Cui's eq. 48 index in expectation over the unobserved connection, ρ·I(x, 1) = ω[ρ·x(x−1)/2 + x]; `literal` is I(x, 1), which makes unreachable devices budget sinks. x = missions since the device's last merged update + 1; ρ = (answered + 1)/(attempts + 2), clamped to ≥ 0.05. |
| `whittle_weights` | `MuleConfig` | `uniform` | `--whittle-weights` | D3's ω: 1, or `oort` (Oort's statistical utility normalised to mean 1; never-measured devices get the mean). `oort` needs `--real-model`. |
| `fedcs_value` | `MuleConfig` | `unit` | `--fedcs-value` | D5 (FedCS Algorithm 3, degraded): the greedy key is value/total seconds; `unit` is the paper's argmin marginal time, `devices` weighs a contact by its device count. |
| D4 assignment | driver | CARP at K > 1 | arm D4 | Computed once per trial with `carp_assign` (Gibbs sampling over single-client moves minimising Σ R_k·Δ_k², FedEx-Async Theorem 2): dock at the origin, 5 m/s, 1 s per client, seeded from the trial. Static because each device is wired to one mule. The tour is a closed 2-OPT tour from and back to the dock; it never skips. Run faithful with `--aggregation agg:fedex`, route-only with `agg:cutoff`. |

**Trace fields at K > 1.** `mule_ready` records `down_wait_s` and
`dock_on_empty`; `mission_empty` gains `docked` when the switch is on (true
once the empty partial is uploaded, even if its DOWN then timed out, which
`dock_down_timeout` records); the
cluster emits `mule_bootstrapped`, and `backhaul_upload_lost.awaits_quorum`,
`up_bundle_ingested.partial_refused`/`held_mission_round` and
`cluster_round_closed.mule_id` where they apply. Rows gain the provenance
columns `n_mules`, `min_participation`, `dock_params` and `policy_params`
(blank at their one-mule defaults), so start a new CSV.

**Scoring at K > 1.** Each device ages in its own mule's missions; Network AoU is
sampled after every mission of any mule; Pass-2 coverage divides by the mule's
own slice; an upload the cluster refused (`partial_refused`) or never folded
is not credited, and a mission whose upload was lost under a quorum is counted
once, as lost, though the fold lists the place held for it with the expired
partials; `missions_to_τ` is the reaching mission's place among its own mule's
missions (it counts mission periods, so it compares across fleet sizes).

---

## Cross-references

* [HERMES_FL_Scheduler_Design.md](HERMES_FL_Scheduler_Design.md) §6
  documents what each constant *means* in the system architecture.
* [HERMES_FL_Scheduler_Implementation_Plan.md](HERMES_FL_Scheduler_Implementation_Plan.md)
  §8 lists open decisions tied to several of these values
  (deadline clock semantics, FL_Threshold tuning, beacon channel,
  Tier-3 cadence).
* [HERMES_Experiments_Implementation_Plan.md](HERMES_Experiments_Implementation_Plan.md)
  ties experiment-time constants (Pidle, εbit, εprop, Bnominal) into the
  trial harness via `experiments/calibration.toml`.
