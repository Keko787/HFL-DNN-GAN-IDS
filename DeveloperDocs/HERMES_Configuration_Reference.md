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
| `session_ttl_s` | `MuleConfig.session_ttl_s` | 5.0 | `MuleConfig` field | Per-contact TTL on the supervisor side. Sprint-2 multi-process tests use 2-3 s; AERPAW deployment may need longer if RF is slow. The Exp 4 builder sets 3 s (`SESSION_TTL_S`); `--session-ttl-s` overrides it, and ferry cells set it from the measured real-model fit time (§17.5). It stays a wall-clock timer on the mission clock. |
| `synth_batch_size` | `ClusterConfig.synth_batch_size` | 4 | `ClusterConfig` field | Number of synthetic samples per DOWN bundle. |

## 9. Transport (TCP + channel emulator)

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `accept_timeout_s` (RF) | [hermes/transport/tcp_rf_link.py:148](hermes/transport/tcp_rf_link.py:148) | 0.25 | `TCPRFLinkServer` constructor | Listener `select()` budget — short so the accept loop can respond to shutdown. |
| `send_timeout_s` (RF) | [hermes/transport/tcp_rf_link.py:149](hermes/transport/tcp_rf_link.py:149) | 30.0 | `TCPRFLinkServer` constructor | Bound on each blocking send, set with `SO_SNDTIMEO` (packed per OS by `sndtimeo_optval`). Reads have no timeout since Freeze Amendment 10: before it this value also bounded reads, and a device silent for 30 s was dropped (finding P-02). RF messages are small (kilobytes); 30 s is a generous ceiling for slow links. The device side (`TCPRFLinkClient`, [tcp_rf_link.py:637](hermes/transport/tcp_rf_link.py:637)) bounds its sends at 60 s the same way. |
| `accept_timeout_s` (Dock) | [hermes/transport/tcp_dock_link.py:202](hermes/transport/tcp_dock_link.py:202) | 0.25 | `TCPDockLinkServer` constructor | Same role as RF, dock-side. |
| `send_timeout_s` (Dock) | [hermes/transport/tcp_dock_link.py:203](hermes/transport/tcp_dock_link.py:203) | 60.0 | `TCPDockLinkServer` constructor | Higher than RF — UP/DOWN bundles can be hundreds of MB for real models. Until Freeze Amendment 10 it was packed as a POSIX `timeval` on every OS, so Windows read it as 60 ms. |
| `connect_timeout_s` | [hermes/transport/tcp_dock_link.py:622](hermes/transport/tcp_dock_link.py:622) | 5.0 | client constructor | Mule client reach-cluster window. The RF client has the same default, and a device's re-dial must be acknowledged within it (§17.4). |
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
used with every run. From FeRRy Phase 3 both are simulated seconds on the
mission clock, and `--agg-period-t-nom` sets T to the cell's T_nom (§17.5).

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
| D4 assignment | driver | CARP at K > 1 | arm D4 | Computed once per trial with `carp_assign` (Gibbs sampling over single-client moves minimising Σ R_k·Δ_k², FedEx-Async Theorem 2): dock at the origin, 5 m/s, 1 s per client, seeded from the trial. On the mission clock (FeRRy Phase 3, §17) the per-client time is one client's predicted Pass-1 airtime at R_planar(b)/2 at the band's mean SNR, with the trial's payload (1 s without a band), and the speed is the cell's cruise speed. Static because each device is wired to one mule. The tour is a closed 2-OPT tour from and back to the dock; it never skips. Run faithful with `--aggregation agg:fedex`, route-only with `agg:cutoff`. |

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

## 17. Mission clock and contact link (FeRRy Phase 3)

`mission_clock = "wall"` is every recorded run, and every default in this section reproduces it
(Freeze §5g, Rule 1; §5j). `"sim"` flies the mule arms on a simulated mission clock, with a contact
link that has a band, a range and a rate, and a backhaul priced in seconds. The physics are the
build plan's decisions D1–D3 as the user accepted them on 2026-09-29 (Phase 3 design, §1). Code:
[hermes/l1/mission_clock.py](../hermes/l1/mission_clock.py),
[hermes/l1/contact_link.py](../hermes/l1/contact_link.py),
[hermes/l1/channel_model.py](../hermes/l1/channel_model.py),
[hermes/mule/ferry.py](../hermes/mule/ferry.py),
[hermes/processes/config.py](../hermes/processes/config.py),
[experiments/exp4/driver.py](../experiments/exp4/driver.py).

**What the clock charges.** Only the mule's supervisor advances the clock, and nothing sleeps for
simulated time, so a leg of any length costs no wall time. Every charge names a kind, and each
mission keeps a ledger of them:

| Kind | When | Amount |
|---|---|---|
| `transit` | before each stop | leg length / cruise speed |
| `dwell` | once per contact, at the host's commit | Σ over the answered targets of 8 · bytes / rate(b, SNR), each target's SNR taken at its own session start (critic C2); without a band, 1 s per contact (`session_time_s`) |
| `listen` | at the commit, when any expected advert, gradient or ack is missing, uplink-dropped devices included (critic C7) | `listen_s`, once per contact |
| `return` | after the last stop of each pass, also after an abort, a re-plan to nothing or an empty pass | leg to the dock / cruise speed |
| `upload` | at the inter-pass dock | 8 · UP bytes / rate(wide, backhaul SNR) (§17.2) |
| `turnaround` | once per mission, after Pass 1's return, upload or not | `turnaround_s` |
| `dock_wait` | on the DOWN | `advance_to(max(upload end + turnaround, cluster_sim_ts))`: the Lamport sync with the cluster |

The clock starts at `SIM_EPOCH_S` = 1e6 s, above the 0.0 that scheduler state reads as "never",
and refuses to reach `SIM_CEILING_S` = 1e9 s, which wall time passed in 2001: a stamp's size says
which clock made it, and `advance_to` refuses a wall stamp. The clock is never reset (deadlines and
`last_clean_ts` are absolute). At each takeoff the pose returns to the dock, the ledger restarts and
the S3b budget is stamped; a budgeted Pass 2 runs from its own takeoff. Transport and coordination
timers stay on the wall clock (session TTLs and joins, the receipt TTL, busy flags, dock and
bootstrap waits, device timers, sockets, the driver's budgets), and so do the events' envelope `ts`
and `duration_s`.

### 17.1 D1 — band classes and the range–rate model

All classes share one carrier at 3.32 GHz and differ in LTE channel bandwidth: the repo's
3.32/3.34/3.90 GHz carriers differ by at most 1.4 dB of free-space loss, too little to carry a range
trade.

| Class (band index) | Bandwidth, N_RB | B_occ = N_RB × 180 kHz | κ | Floor rate (CQI 1) | Peak rate (CQI 15) | R slant / planar, n = 2.2 | R slant / planar, n = 3.0 |
|---|---|---|---|---|---|---|---|
| `wide` (0) | 20 MHz, 100 | 18 MHz | 0.754 | 2.07 Mb/s | 75.4 Mb/s | 65.0 / 60.0 m | 65.0 / 60.0 m |
| `medium` (1) | 5 MHz, 25 | 4.5 MHz | 0.734 | 0.50 Mb/s | 18.3 Mb/s | 122.1 / 119.5 m | 103.2 / 100.1 m |
| `narrow` (2) | 1.4 MHz, 6 | 1.08 MHz | 0.732 | 0.12 Mb/s | 4.39 Mb/s | 233.5 / 232.2 m | 166.0 / 164.1 m |
| `medium_wide` (3, optional) | 10 MHz, 50 | 9 MHz | 0.734 | 1.01 Mb/s | 36.7 Mb/s | 89.1 / 85.5 m | 81.9 / 78.0 m |

Ranges are for h = 25 m with the wide anchor at 60 m planar. A class's index is the `band` a trace
line records; the optional 10 MHz class is appended (`--contact-band-classes wide medium narrow
medium_wide`), so indices 0–2 keep their meaning. The Phase 3 re-baselines fly `wide`, which keeps
Exp 4's S3a geometry (R_planar(wide) is `rf_range_m` exactly); medium and narrow are for Study 5.4.

    B_occ,b      = N_RB,b × 180 kHz
    SE(s)        = efficiency of the highest CQI whose SNR threshold ≤ s
    rate(b, s)   = min(κ_b · B_occ,b · SE(s), B_occ,b · log2(1 + 10^(s/10)))   for s ≥ floor; 0 below
    M_sh         = Φ⁻¹(q) · σ_sh = 1.2816 × 4 dB = 5.13 dB
    R_wide       = hypot(rf_range_m, h)                                        (slant; the anchor)
    R_b          = R_wide · (B_occ,wide / B_occ,b)^(1/n)
    R_planar(b)  = sqrt(R_b² − h²), and exactly rf_range_m for wide
    d3D          = max(hypot(d_planar, h), 1 m)
    SNR_b(d3D)   = floor + M_sh + 10 · n · log10(R_b / d3D)                   (the mean)
    dwell        = 8 · bytes / rate(b, SNR)                                   (None below the floor)

Altitude enters only through d3D: legs, positions and S3a stay planar. The Shannon cap never binds
at the default floor (SE / log2(1 + SNR) is 0.545–0.736 at the CQI thresholds), which the tests use
as an invariant.

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `contact_band` | `MuleConfig` | None | config field; `--contact-band` | The class every stop flies. None on the mission clock is the channel-free control (critic A1): every member is a target, and a contact costs `session_time_s` (1 s) plus the listen window when a reply is missing. |
| `contact_band_classes` | `MuleConfig`, `ClusterConfig` | None (wide, medium, narrow) | config field; `--contact-band-classes` | The link's classes in index order. The cluster reads each report line's band index with it, so the two must match. |
| `snr_floor_db` | `MuleConfig` | −6.7 dB | config field; `--snr-floor-db` | CQI 1 of the AERPAW digital-twin thresholds (Hossen et al., arXiv:2503.07935, from the authors' `getSpectralEfficiency`). TR 36.942's form allows −10 dB, the sweep value. With the anchor held, the floor moves every mean SNR, so a floor sweep at a fixed anchor is an EIRP sweep. A floor below −30 dB (`MIN_SNR_FLOOR_DB`) is refused as a units mistake. |
| `altitude_m` (h) | `MuleConfig` | 25 m | config field; `--altitude-m` | The altitude of the AERPAW AADM data-mule challenge (arXiv:2602.16163). |
| `n_pl` (n) | `MuleConfig` | 2.2 | config field; `--n-pl` | 3GPP TR 36.777 V15.0.0 Annex B: the line-of-sight aerial models give n = 2.12–2.20 at 20–30 m. n = 3.0 is the sensitivity case (exponents measured at 4–16 m altitude, or non-line-of-sight links). |
| `shadow_sigma_db` (σ_sh) | `MuleConfig` | 4 dB | config field; `--shadow-sigma-db` | TR 36.777 RMa-AV line of sight: 3.66–3.83 dB at 20–30 m. The margin M_sh and the channel's shadowing use the same σ. |
| `margin_quantile` (q) | `MuleConfig` | 0.9 | config field; `--margin-quantile` | R(b) is the range with q link availability at the edge. |

**R(b) is the 90 % edge-availability range, not the floor-rate range** (a plan deviation, critic
A8-i). At R(b) the mean SNR sits M_sh above the floor, so with σ_sh of shadowing the link stays
above the floor 90 % of the time at the edge (about 89 % with the default interference term of
§17.2). At its edge the wide class therefore runs at CQI 3 (5.12 Mb/s), not at the floor rate. The
floor-rate range, where the mean SNR reaches the floor, is R_b · 10^(M_sh / (10 n)): 111.2 m slant
for wide at n = 2.2, 1.71 × R (`ContactLink.floor_range_m`).

**The anchor is an assumption, and so is the link budget it implies** (critic C6). No AERPAW
measurement gives R(b), and the published AERPAW budgets (10 dBm, 10 and 2 dBi antennas) reach
1.5–6 km at the floor, so a physical budget would never bind in a 100 m field. R_planar(wide) =
`rf_range_m` (60 m, Exp 4's default) is chosen so that the range gate binds where Exp 4's S3a
already does. Written as a budget (42.87 dB of free-space loss at 1 m at 3.32 GHz, exponent n
beyond; a receiver noise figure of 9 dB, TR 36.942 V10.3.0 Table 4.8; 0 dBi; the margin M_sh) it
implies an EIRP of −11.3 dBm at n = 2.2 (+3.2 dBm at n = 3.0), the same for every class
(`ContactLink.implied_eirp_dbm`). The Phase 3 research's own low-altitude budget (0 dBm, 0 dBi,
NF 9 dB, the same margin) reaches 211 m slant on wide at n = 2.2, so the anchor is about 11 dB more
pessimistic. The gap stands for what free space and an exponent leave out: antenna patterns towards
the ground, body and installation losses, and a low-power device radio.

**Worked numbers** (h = 25 m, n = 2.2, 0–60 m planar). The rate runs 20.0 → 5.1 Mb/s on wide,
9.0 → 3.9 Mb/s on medium and 3.6 → 1.9 Mb/s on narrow; no class reaches CQI 15 at this altitude. An
IDS Pass-1 session (37,576 B) takes 0.015–0.059 s on wide and 0.084–0.158 s on narrow; 1 MB each way
takes 0.8–3.1 s on wide and 4.5–8.4 s on narrow, and 53.7 s on narrow at 200 m. Wide out-rates the
narrower classes wherever it reaches, apart from CQI-step artefacts, so the class trade is reach
against dwell, and it only matters for far devices or large payloads.

**Under a mission budget it is also a matter of contact granularity** (final check, E2E1-01).
S3a clusters at R_planar(b), so on narrow the realism field forms one field-wide contact in about
98 % of layouts (on medium the largest contact holds about 70 % of the devices). S3b, the re-plan
and every D-arm walk admit or refuse a contact whole. With a declared payload, a budget below that
contact's predicted home time empties every mission of every gated arm (D4 still flies
everything). At or above it, all N devices are served from one stop, provided the deadline clause
does not bind first. At N = 8, seed 777, narrow and 1 MB, with T2's `--deadline-time-scale t_nom`
(Deadline(j) = takeoff + 1500 s in its first mission), the contact's home is 99.12 s after takeoff:
0 devices are admitted at 60 s and at 99.0 s, all 8 at 99.5 s. The median cliff at 1 MB is 76.5 s
at N = 6, 98.2 s at N = 8 and 120.5 s at N = 10. At the default unit 1.0 (Φ₀ = 60 s) the same
contact's predicted finish, 94.9 s after takeoff, is already past Deadline(j). S3b reads the
deadline clause first, so H1–H3 drop it as `overdue` at any budget, above the knee too, until
missed missions widen its window past that finish; the budget-only walks (D1–D3, D5) keep the
budget cliff. Measure the budget knee per (band, payload, N, deadline unit). Phase 4 removes the
cliff by admitting a member subset or capping S3a's contacts by predicted dwell;
`tests/unit/test_p3_final_fixes_mule.py` pins today's behaviour at both units, so that change must
be made deliberately. Decided 2026-09-29: the cliff waits for Phase 4's member-subset admission,
and until then narrow and medium cells are not compared under budgets below it; their budget knees
wait for Study 5.4 (the Run Guide's §2.6 has the pilot plan).

**Below the floor** (critic B12) the rate is 0 and `dwell_s` returns None, never infinity: a member
below the floor at arrival is unreachable (not solicited, TIMEOUT with `answered=False`, no
airtime); a target whose SNR has fallen below the floor by its own session start is priced at the
floor rate; a backhaul upload below the floor is lost (§17.2).

**Sources.** N_RB: 3GPP TS 36.104 V12.10.0 Table 5.6-1. CQI efficiencies: TS 36.213 V8.8.0 Table
7.2.3-1. κ = peak TBS / (B_occ × 5.5547), the single-layer I_TBS 26 blocks of TS 36.213 V8.8.0 Table
7.1.7.2.1-1. The shape "0 below the floor, attenuated and capped Shannon above": TR 36.942 V10.3.0
Annex A.1. The research report's `p3_research.md` has the derivations; the TR 36.777 and TR 38.901
values come from a third-party transcription.

### 17.2 D2 — the seconds-axis channel

**Contact link.** For device j on class b at simulated time t:

    SNR_b,j(t) = SNR_b(d3D_j) + X_j(t) + I_b(t)
    I_b(t)     = A · sin(2π (τ / P_c + φ_b)) + σ_I · ξ_b(t),   τ = t − epoch

X_j is the link's shadowing, shared by every class because they share one carrier. I_b is a
declared interference model, not a propagation claim. The planner uses the mean alone
(`pred_snr_db`); the contact uses the realized SNR, at arrival for the range and floor gate and at
each target's own session start for its dwell. Dwell is convex in SNR, so pricing the mean (δ_obs
= 0) under-predicts: in the final check realized missions ran longer than predicted by +0.8 % on
average on wide at 1 MB, +7.1 % on wide at 10 MB and +19.2 % on narrow at 1 MB, and a member's
expected dwell was 1.01–1.39 × its predicted dwell. Budgets therefore bind in flight more often on
narrow bands and large payloads, which bears on Study 5.4, the payload sweep and the budget-knee
pilot.

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `shadow_corr_s` | `MuleConfig` | 7.4 s | config field; `--shadow-corr-s` | TR 38.901 Table 7.5-6: a 37 m decorrelation distance (RMa, line of sight) flown at 5 m/s. The interpolated noise below gives a correlation of 0.29 at one correlation time and exactly 0 from two on (TR 38.901's exponential: 0.37 and 0.14). |
| `shadow_keying` | `MuleConfig` | `time` | config field; `--shadow-keying` | `position` keys X_j by (device, stop cell on a 37 m grid) instead, fixed in time, so two arms hovering in one cell see the same shadowing (critic C5); the time variation then lives in I_b alone. |
| `contact_regime` | `MuleConfig` | `clean`: A = 1 dB, σ_I = 0.4 dB | config field; `--contact-regime` | The contact link is regime-independent by default, whatever the cell's regime, which keeps Exp 4's "jitter hits the backhaul, not the short hop". `jittery` (A = 5 dB, σ_I = 1.5 dB) is for test (c). |
| `interference_period_s` (P_c) | `MuleConfig` | 60 s | config field; `--interference-period-s` | An assumption. Test (c) sets P_c = median leg / ρ for ρ ∈ {0.25, 0.5, 1, 2}. One period for every class (a plan deviation from "a period per band"); φ_b is a seeded shuffle of {0, 1/3, 2/3} (the 10 MHz class takes a gap midpoint), so the classes still cross over. No random per-class gain: the classes differ structurally, through R_b. `ContactChannel` also takes per-class period multipliers (default 1), which no configuration field sets. |
| `noise_bin_s` (Δ) | `MuleConfig` | 1 s | config field; `--noise-bin-s` | Fast fading averages out within it: the coherence time is about 7.3 ms at 5 m/s (AERIQ). |

**The noise is a pure function of time.** z(k) is a Box–Muller draw from the SHA-256 digest of
(salt, stream, key, k), for the bin k = ⌊(t − epoch) / Δ⌋, and ξ(t) = ((1 − u)·z_k + u·z_{k+1}) /
√((1 − u)² + u²) with u the position within the bin, so the variance is 1 at every t and there is
no horizon. Two arms that ask for the same (t, band, link) get the same value, whatever else they
asked for and in whatever order: they are paired by construction. Bit-identical values across
operating systems are not guaranteed, because Box–Muller and the sinusoids use the platform's math
library (critic A8-ii); the differences are about 1e-16 relative.

**Salts and keyed draws.** `ferry_salt(seed, stream)` is `trial_salt(seed, "ferry", stream)`, the
first 4 bytes of SHA-256 of the trial seed, "ferry" and a stream name (`contact`, `backhaul`,
`availability`, `backhaul_loss`). `trial_salt` is a copy of `experiments/exp4/model_task._u32`,
since `hermes` must not import `experiments`. Outcomes are keyed, not drawn from a stream:
`keyed_uniform(salt, key, round)`. A device's availability at its Pass-1 contact fails when
`keyed_uniform(ferry_salt(seed, "availability"), device, mission_round)` ≥ rel_i; an upload is
lost when `keyed_uniform(ferry_salt(seed, "backhaul_loss"), mule, mission_round)` < p_loss. Every
arm therefore faces the same uniform for a given (device or mule, mission round), whichever
missions docked before it: common random numbers. The availability draw comes out the same in every
arm that contacts the device in that mission round, since rel_i is fixed per trial. A backhaul loss
can still differ between arms, because p_loss is each arm's own: read at its own upload start and,
for H3, on the carrier its controller picked (the loss table below: 0.162 at the fixed carrier,
0.017 for H3, when jittery).

**Backhaul.** Three carriers, evaluated at the dock at the simulated upload time:

    SNR_c(t) = base + g_c + A · sin(2π (τ / P_bh + φ_c)) + σ · ξ_c(t)

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `backhaul_model` | `MuleConfig`, `ClusterConfig` | `mission` | config field; `--backhaul-model` | `mission`: the recorded loss, the flat `--realism` percentage or the `--l1-channel` schedule by mission round, drawn by the cluster from a stream. On the mission clock the upload is still charged, timed at `base + max_c g_c` of the seconds model for the trial's seed and regime (`mule_ready.backhaul_timing_snr_db`), the same for every arm, H3 included; no backhaul outcome is recorded. `seconds` (mission clock only): the model above, with the keyed loss draw; the driver sets the flat percentage to 0 and refuses `--l1-channel` with it. Chosen per study (spec Q7). |
| `backhaul_regime` | `MuleConfig` | `clean`; the driver sets the cell's regime | driver | base, A and σ: 12, 1 and 0.4 dB clean; 6, 5 and 1.5 dB jittery (the legacy model's). g_c ~ U(0, 3) dB and φ_c, a shuffle of {0, 1/3, 2/3}, are hashed from the salt: the same distributions as the legacy model, other values. |
| `backhaul_policy` | `MuleConfig` | `fixed`; the driver sets `adaptive` for H3 | driver | `fixed`: argmax_c g_c, the noise-free long-run best carrier. The legacy `best_average_band` averaged the realized trace, so it used the future. `adaptive`: H3's `AdaptiveChannelController` at every upload, holding its carrier across missions. |
| `backhaul_period_s` (P_bh) | `MuleConfig` | None: `n_missions` × `t_nom_s` | config field; `--backhaul-period-s` | Today's "one cycle per trial", in seconds (without the legacy floor of 2 missions). |
| `t_nom_s` | `MuleConfig` | None | config field; `--t-nom-s` | T_nom (§17.5). |
| `rf_prior_schedule_db` | `MuleConfig` | None | driver, with `--l1-channel` | The causal RF prior under the `mission` model (critic B4): entry r − 1 is the L1 trace's SNR on the carrier chosen for mission r's upload, and after each mission that docked the mule sets the planner's `rf_prior_snr_db` to it (`mission_schedule_index` picks the entry, as the cluster's loss draw does). Refused on the wall clock and with the seconds model, whose own producer (`RFPriorProducer`) sets the prior to the SNR last observed at an upload on the carrier used. 20 dB before the first upload either way. The driver's trial-mean `rf_prior_snr_db` is never handed to a mission-clock cell. |

**The upload.** The mule picks the carrier (fixed, or H3's controller) at the upload's start,
reads its SNR there, and charges 8 · UP bytes / rate(wide, SNR) as `upload`; p_loss =
`loss_from_snr(SNR)` = 1 / (1 + exp((SNR − 3) / 2)). The probability is read at the upload's start,
not at its completion (`sim_upload_ts`): a plan deviation bounded by the upload's duration. Below
the SNR floor the upload is lost, charged the floor-rate time for its bytes, and recorded with
p_loss 1.0 (critic B12). The UP carries `UpBundle.backhaul`, a `BackhaulUpload` (carrier, SNR,
p_loss, start, duration, bytes, below-floor flag). The cluster draws the loss for every UP, an empty
partial's included; an UP the mule did not price is never lost and is counted as
`backhaul_unpriced_uploads`.

**Loss magnitudes** (critic A4), mean loss per upload over 400 seeds, 4 missions, T_nom = 219 s:

| Arm and model | Jittery | Clean |
|---|---|---|
| Today: `--realism` without `--l1-channel` | 0.02 (flat) | 0 |
| Seconds axis, fixed carrier argmax g_c (critic's probe) | 0.161 | 0.004 |
| Legacy model, fixed best-average band (critic's probe) | 0.153 | 0.004 |
| Legacy model, H3 adaptive (critic's probe) | 0.018 | 0.004 |
| This implementation, fixed carrier, one upload per mission | 0.162 | 0.004 |
| This implementation, H3's controller | 0.017 | 0.004 |

Switching the seconds model on for every mule arm is like switching `--l1-channel` on for all of
them: it moves every exit-gate number, and round closure becomes mostly a function of the L1 gap.

### 17.3 D3 — payload and energy

| Quantity | Bytes | Note |
|---|---|---|
| θ of the canonical 21-input model | 18,756 | 4,689 float32 values, BatchNorm statistics included (Phase 3 design, finding 1; the 46-input fallback model is 25,156 B and is refused in ferry cells, design R8) |
| Pass-1 session | 37,576 | the push (θ plus a 64 B synthetic batch, 18,820 B) and the update (18,756 B) |
| Pass-2 session | 18,820 | the push |
| UP partial | about 18,756 | an empty partial is 0 B |
| Stub θ | 52 | stub runs have negligible airtime unless a payload is declared |

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `payload_bytes` | `MuleConfig` | None: measured | config field; `--payload-bytes` | Bytes per direction, declared (D3's sweep: measured, 1 MB, 10 MB). A Pass-1 session is priced at 2 × the payload, a Pass-2 session and the upload at 1 × (an empty partial stays 0 B). The real θ still crosses the link; only the charge changes. |
| `cruise_speed_m_s` | `MuleConfig` | 5 m/s | config field; `--cruise-speed-m-s` | Freeze decision D2; it still awaits a platform citation. The flight power follows the speed. |
| `turnaround_s` | `MuleConfig` | 30 s | config field; `--turnaround-s` | Exp 3's `dock_time_s` (`experiments/exp3/sim_env.py:157`), once per mission, so every mission advances the clock. |
| `listen_s` | `MuleConfig` | 1 s | config field; `--listen-s` | The frozen 1 s session, charged once per contact with a missing reply. |
| `energy_capacity_j` | `MuleConfig` | None | config field; `--energy-capacity-j` | SIMULATED battery capacity. It switches the energy clause of the predicate on: e + P_move·(transit + return) + P_hover·dwell ≤ capacity, which binds only with a budget. Pass 2's energy counts from its own takeoff (a recharge or swap during the turnaround). The L1 state's energy slot (slot 7, 1 − E/E_ref, recorded only) counts the same sortie energy: E restarts at 0 at Pass 2's takeoff, and E_ref is this capacity or, without one, P_hover × `mission_budget_s`; with neither, the slot stays 1.0. |
| `p_move_w`, `p_hover_w` | `MuleConfig` | None: the Zeng model's 143.6 W at 5 m/s and 168.5 W | config field; `--p-move-w`, `--p-hover-w` | SIMULATED powers; override the model for a sensitivity case (a DJI M100 class is about 83–100 J/m at 5 m/s, an AERPAW LAM6 class about 233–391 J/m). |
| `deadline_bounds` | `MuleConfig` | `collection` | config field; `--deadline-bounds` | What Deadline(j) bounds in the ferry predicate (spec Q2). It has three values since the user's decision of 2026-09-29 to keep both delivery readings (commit `d175afa` after `ef1faa1`). `collection` (the default): arrival + dwell ≤ Deadline(j), which matches the scorer's miss metric. `delivery_per_stop`: the plan's single-contact predicate. At each stop, the finish plus that stop's own return to the dock plus the upload must be ≤ its Deadline(j), i.e. the update could have been delivered in time had the mule flown home right after that stop. It is checked per stop, so it does not bound when an update actually reaches the cluster: that is the route's landing plus the upload, which is later for every stop but the last. This is what `delivery` meant at `ef1faa1`. `delivery`: route-level. Every update collected on the route must reach the cluster (the route's landing plus the upload) by its own Deadline(j). Each stop keeps the per-stop check and must also be home, upload done, by `deliver_by`, the earliest Deadline(j) of the updates already on board (before takeoff S3b assumes every admitted stop's members answer; in flight only members collected CLEAN count, each with its own Deadline(j)). A stop that is not refused for its own deadline but fails this check is refused with the reason `delivery`: the check comes after the stop's own deadline and before the budget and energy clauses, so a stop over the budget too is still reported as `delivery`. The bound is on the priced route, like the budget: the mule re-checks it at every Pass-1 departure, but a contact that runs longer than priced (a silent member's listen window, a noisy band) at the stop where Pass 1 ends is never re-checked, so an update already on board can still land late. `mission_completed.delivery_overrun_s` records by how much (§17.6). Only our arms' deadline rule reads either delivery value; Pass 2 and D1–D5 have no deadline clause. `delivery` changed meaning after `ef1faa1` with no rename shim: the setting is sim-only and new in Phase 3, and no recorded run or committed trace used it. The legacy predicate tests the arrival. |

**Energy, labelled SIMULATED everywhere:** E = P(v) · t_move + P_hover · (t_dwell + t_listen),
from the clock's ledger (transit and return at P(v); dwell and listen at P_hover; upload, turnaround
and dock wait on the ground, not charged). The model is Zeng, Xu and Zhang, "Energy Minimization
for Wireless Communication With Rotary-Wing UAV", IEEE Trans. Wireless Commun. 18(4), 2019
(arXiv:1804.02238), eq. (6), with its 2 kg-class parameter set (read through secondary sources, as
the `EnergyModel` docstring details): P(5 m/s) = 143.6 W, which is 28.7 J/m, and P_hover = 168.5 W.
Declared assumptions (critic C8): 5 m/s is below the model's minimum-power speed of about 10.2 m/s
(126 W, 12.3 J/m); the climb and descent to 25 m are not charged; there is one shared dock and
queueing at it is not modelled; the radio is not charged. Exp 3's placeholder ε_prop = 10 J/m stays
in Exp 3. Traces and rows carry `energy_status = "simulated"`.

### 17.4 Switches and configuration fields

Every field defaults to the recorded run, and old per-role JSON loads unchanged. On the wall clock
every field that only means something on the mission clock must keep its default
(`mule_config_errors`); the deadline time unit (`deadline_time_scale`, `initial_window_s`), the
session TTL and the RF link token are valid on either clock. The driver's `t_nom` scale and
`--initial-window-missions` are not: they need T_nom, so the mission clock (§17.5).

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `mission_clock` | `MuleConfig`, `ClusterConfig` | `wall` | config field; `--mission-clock` | `sim`: one `MissionClock` per mule process, and the cluster's simulated-time bookkeeping (it echoes the latest `sim_upload_ts` it ingested as `DownBundle.cluster_sim_ts`). Every role of a trial runs one clock. |
| `trial_seed` | `MuleConfig`, `ClusterConfig` | None | set by the builder | The salt of the contact channel, the backhaul and the keyed draws, so every arm of a trial sees one channel. Required on the mission clock. |
| `in_flight_response` | `MuleConfig` | `abort` | config field; `--in-flight-response` | `abort`: Amendment 8's rule (the next stop with its return-and-upload tail), priced on the clock. `replan`: the whole remainder checked at every departure (`FLScheduler.fold_remainder`) and repaired (`replan_remainder`); a budgeted Pass 2 is checked in flight only under `replan`. The pilot plan, decided 2026-09-29, flies `replan` (critic B5; Run Guide §2.6). |
| `replan_fallback` | `MuleConfig`, `FLScheduler` | `reorder` | config field; `--replan-fallback` | What our arms' Pass-1 re-plans do when the arm's own order over the re-admitted stops does not fit. `reorder`: 2-OPT to the dock, then S3b's admission order; before takeoff that makes H1, H2 and H3 fly the same route whenever the check fires, and the check never drops a stop. `trim`: keep the arm's order and drop the stops it cannot serve (`order_used = "arm_trimmed"`). Our arms' Pass 1 only; the baselines keep their policy's order and take no 2-OPT fallback by default. The pilot plan, decided 2026-09-29, flies `trim`, so each arm keeps its own order, as the D arms do (critic C3). |
| `validate_flown_order` | `FLScheduler` | False | constructor arg (the mule sets it under `replan`) | The pre-flight order check (design §3.3): the order the arm will fly is folded before takeoff and repaired by the fallback; drops join `last_feasibility` by reason, energy included (critic B10), so the pre-flight widening covers them. Needs the ferry model and a budget. |
| `refuse_deadline_overrides` | `FLScheduler` | False | constructor arg (the mule sets it on the mission clock) | Cluster deadline overrides are wall-clock stamps (critic B3). A sim-mode cluster never issues one; a sim-mode mule that receives one fails, with `dock_bootstrap_failed` and exit code 5 at the bootstrap, or `mission_failed` and exit code 3 at a later dock. |
| `contact_reliability_source` | `MuleConfig` | `origin` | config field; `--contact-reliability-source` | `origin`: each device's own draw of rel × rf_factor (`--realism`). `channel` (needs a band): the SNR gate at the stop, and the availability rel_i ~ U(0.15, 1) (`device_reliabilities(seed, N)`) drawn on the mule, keyed; devices are then built with `contact_reliability=None`, and under `origin` they keep their draw (nulling it would remove every contact failure). Pass 2 faces the SNR gate only. |
| `device_availability` | `MuleConfig` | {} | set by the builder | The ground truth {device_id: rel_i} for the keyed draw, each mule its own slice. Kept out of the scheduler, the policies, the L1 state and every event (`mule_ready` shows only its size, `device_availability_n`), or D3's ρ would become an oracle (critic B16). The mule's per-role JSON holds it. |
| `deadline_time_scale` | `MuleConfig`, `FLScheduler` | 1.0 | config field; `--deadline-time-scale` (a number, or `t_nom` for T_nom / 10 s on the mission clock only) | Spec Q1: the recorded constants were set against missions of about 10 s of wall clock (`LEGACY_MISSION_PERIOD_S`), and a two-pass mission on the clock takes minutes. Every time constant of the law moves by this factor: the additive −5 s / +10 s steps and 5 s floor, the multiplicative 5 s / 300 s clamps, and Φ₀; the β factors do not. At 1.0 it is left out of `deadline_params`. On the mission clock the law cannot keep up at 1.0: in a 14-mission probe (narrow, 1 MB, N = 8, a 200 s budget) H1 served the field-wide contact once and then found it overdue at every later takeoff, by 20 s more each mission, since the additive law's +10 s per miss never catches up with the clock (an empty mission alone costs the 30 s turnaround). Simulated-clock cells should set the unit; the pilot plan, decided 2026-09-29, sets `t_nom` (with the default Φ₀, about six missions' worth of window, as in the recorded runs), as the Run Guide's example does. The driver refuses a `time_scale` key in its own `deadline_params` (`Exp4Driver.deadline_params`, which `runner_main` builds from the five multiplicative-law flags `--deadline-beta-*` and `--deadline-phi-*`) with `DeadlineLawError`, a `ValueError`, as afa9526 did, in any form `dict()` accepts (a mapping or a list of pairs). The unit is set only through this field, which the trial CSV's `deadline_time_scale` column records. |
| `initial_window_s` | `MuleConfig`, `FLScheduler` | None (60 s) | config field; `--initial-window-s`, or `--initial-window-missions` (mission clock only) | Φ₀, stated in the law's recorded unit and multiplied by the scale, so None and an explicit 60 give the same Φ₀ at any scale. `--initial-window-missions m` gives Φ₀ = m nominal mission periods (critic A7: Φ₀ = 60 s was a placeholder), passed as m × T_nom / scale; at the scale T_nom / 10 the recorded 60 s is 6 missions. |
| `input_dim` | `MuleConfig` | None | set by the builder | The model's input width, for `mule_ready` (the payload's provenance, design R8). |
| `rf_link_token` | `MuleConfig`, `DeviceConfig` | None | `--rf-link-token` / `--no-rf-link-token` | Freeze Amendment 10: a mule's RF server with a token refuses registrations carrying another. The driver derives one per trial (the first 16 hex digits of SHA-256 of cell, arm, trial index and seed), on by default exactly on the mission clock. All of a trial's mules and devices share it. |
| `newest_solicit_only` | `DeviceConfig` | False | set by the builder (on exactly on the mission clock) | Critic B1: a device answers only its newest queued solicit. The ferry mule accepts only adverts whose `in_reply_to` names the solicit it is gathering for, so a stale answer would cost the device its push wait through the next gather. |
| `link_token`, `send_timeout_s`, `newest_solicit_only` | `TCPRFLinkClient` | None, 60 s, False | constructor args | The device side of the three rows above; sends bounded by `SO_SNDTIMEO`, reads unbounded (§9). |
| `sim_markers` | `TCPDockLinkServer` | False | constructor arg (on exactly when the cluster orders uploads) | Unit U9: the server queues `DockClockMarker`s with the UPs (`registered`, `departed`, and a mule's `clock` or `done`), read with `recv_dock_event()`; `session_of(mule)` names a mule's current connection. `client_send_clock(mule_id, sim_ts, *, done=False)` sends a marker; the mule does not call it today, and the end of its dock connection counts as done. Import the marker from `hermes.transport.dock_link`. |
| `_RECONNECT_INITIAL_S`, `_RECONNECT_MAX_S`, `_RECONNECT_HOLD_S` | `DeviceService` | 0.5 s, 10 s, 30 s | class constants | Amendment 10's device re-dial: the wait doubles from 0.5 s to 10 s, and a link that drops again within 30 s of a re-dial resumes from twice its last wait, so two devices evicting each other stay at the cap. A re-dial counts only when the mule acknowledges it within `connect_timeout_s`. |

**Refused combinations** (the config layer, `TopologyConfig.validate`, the supervisor and the
driver; each fires only on a non-default setting):

- any mission-clock-only setting on the wall clock, the seconds backhaul included (critic B16);
- on the mission clock: no `rf_range_m` (the single-pass path is not ported), no trial seed, the
  channel reliability source without a band (critic B16), an availability map under `origin`, the
  seconds backhaul without a period (or T_nom), `rf_prior_schedule_db` with the seconds model or of
  another length than the cluster's loss schedule;
- roles of one trial on different clocks, backhaul models or band classes, or, under the seconds
  backhaul, a mule whose trial seed is not the cluster's (the keyed loss draw and the channel are
  salted with it); a device with a link token other than its mule's; a device that draws its own
  reliability under the channel source (the failure would be drawn twice);
- several mules on the mission clock below a full quorum or under FedBuff without `down_wait_s` on
  every mule: the cluster may hold an upload while the others catch up, and the recorded 10 s DOWN
  wait would end the run;
- a supervisor with a clock and a `now_fn`, a `FerrySpec` without the clock, a start pose other
  than the dock, a cruise speed other than the flight model's, a contact link not anchored at
  `rf_range_m`;
- in the driver: H0 on the mission clock (critic A5: its simulated round time is outside Phase 3;
  the runner drops it from the default arm list and refuses it when named); `--l1-channel` with the
  seconds model; Φ₀ given both ways; `--agg-period-t-nom` without `agg:cutoff` or with a
  `period_s`; a real-model mission-clock cell whose model width is not the declared one (design R8;
  21 on the canonical data, `--expected-input-dim` overrides).

### 17.5 Exp 4 driver and runner

The runner's flags map onto `Exp4Driver` fields of the same name (`experiments/exp4/runner_main.py`,
`_add_phase_3_flags`); the Run Guide's §2.6 has an example cell.

| Flag | Default | Meaning |
|---|---|---|
| `--mission-clock` | `wall` | `sim` runs every mule arm on the mission clock; fresh CSV paths are required (§17.7). |
| `--contact-band`, `--contact-band-classes` | none; wide medium narrow | §17.1. |
| `--in-flight-response`, `--replan-fallback` | `abort`, `reorder` | §17.4. |
| `--backhaul-model`, `--backhaul-period-s` | `mission`; n_missions × T_nom | §17.2. The driver gives H3 the adaptive carrier policy and every other arm the fixed one, and the backhaul regime is the cell's. |
| `--contact-reliability-source` | `origin` | §17.4. |
| `--payload-bytes`, `--deadline-bounds` | measured, `collection` | §17.3. |
| `--t-nom-s`, `--t-nom-layouts` | computed when needed; 20 | T_nom, below. |
| `--deadline-time-scale` | 1.0 | A number, valid on either clock, or `t_nom` for T_nom / 10 s, on the mission clock only (§17.4). |
| `--initial-window-s`, `--initial-window-missions` | none | Φ₀ in the law's recorded unit (either clock), or in nominal mission periods (mission clock only; §17.4). |
| `--agg-period-t-nom` | off | `agg:cutoff` only: D5's `period_s` = the cell's T_nom (§14). |
| `--session-ttl-s` | 3 s (the builder's) | The mule's wall-clock session TTL. A fit that outlasts it becomes a missed reply with a listen charge, so ferry cells set it to at least twice the measured real-model `train_offline` time (spec Q12). The pilot plan, decided 2026-09-29, takes the 95th percentile of that time, measured with N devices training at once (the exit gate's concurrency). |
| `--rf-link-token` / `--no-rf-link-token` | on exactly with `--mission-clock sim` | §17.4. |
| `--expected-input-dim` | 21 on the canonical data | Design R8: a missing CICIoT dataset silently builds the 46-input model (25,156 B, not 18,756 B), which a real-model mission-clock cell refuses. |
| `--snr-floor-db`, `--altitude-m`, `--n-pl`, `--shadow-sigma-db`, `--margin-quantile`, `--contact-regime`, `--interference-period-s`, `--noise-bin-s`, `--shadow-corr-s`, `--shadow-keying`, `--cruise-speed-m-s`, `--turnaround-s`, `--listen-s`, `--energy-capacity-j`, `--p-move-w`, `--p-hover-w` | the design's | The D1–D3 physics (§17.1–17.3); only the flags given are passed. |
| `--l1-channel` | off | On the mission clock (with the `mission` backhaul model only), the loss schedule is kept and the selector's RF prior becomes the chosen band's SNR at the last upload made, not the trial's mean SNR (`rf_prior_schedule_db`, critic B4). |

**T_nom, the nominal mission period** (spec Q1). One value per cell, the same for every arm and
every trial: the median, over `--t-nom-layouts` reference layouts, of
`fl_scheduler.nominal_mission_period_s`. That is Pass 1 (each leg's transit and predicted dwell,
the return leg, the upload), plus the turnaround, plus Pass 2 (legs, dwell, return), for one plan
with no budget and no selector from the dock, with nothing failing (no listen charge). The layouts
are drawn as a trial's are (the builder's positions, the cell's N and spread) from their own seeds,
`_u32(N, "t_nom", k)`, so neither the grid's base seed nor its trial count moves T_nom and a resumed
CSV keeps it. Each layout is priced with a wide-band spec on its own seed's channel, the trial's
payload (the seed θ and synthetic batch) and a placeholder backhaul period, which the planner never
reads; with several mules, each layout is priced as its slowest angular slice. T_nom is computed
only when a setting needs it (the seconds backhaul without a period, `--deadline-time-scale t_nom`,
`--initial-window-missions`, `--agg-period-t-nom`), unless `--t-nom-s` gives it, and cached per
cell. For orientation, a stub cell at N = 6 gives 194.2 s with `--realism` and 34.8 s without; the
pilot computes it per exit-gate cell.

**Wall budgets** (critic B14). On the wall clock the trial is killed at `--trial-budget-s` (120 s),
as recorded. On the mission clock the kill is the larger of that and `ferry_wall_bound_s`, built
from the waits the code caps rather than from a typical run: 90 s of startup waits (devices 60 s,
bootstrap DOWN 30 s), then per mission two passes of at most one contact per device of the largest
slice, each at most one TTL of gathering plus a 2 × TTL join, and a 10 s DOWN wait; doubled per
mission with several mules, or multiplied by max(2, K) when the cluster orders the uploads (the
mules may then run one at a time). That is 562 s at the 3 s TTL with N = 6, one mule and 4
missions, and 4,450 s at a 30 s TTL; 642 s at K = 3 in simulated order, against 458 s at a full
quorum. The largest slice is taken as ceil(N/K) devices. Angular or CARP slices can be unbalanced,
so at K ≥ 3, when the cluster does not order the uploads (a full quorum, not FedBuff), an
all-timeouts worst case can exceed the bound (slices of 7, 1 and 1 devices at a 3 s TTL: 136 s per
mission against the bound's 128 s). In simulated order the K× factor covers any split, and a
healthy trial never comes near it. With several mules `down_wait_s` follows the re-costed budget,
and the runner's soft cap defaults to the largest budget over the grid. `trial_status.json`
records the trial's own budget (`trial_budget_s`), and on mission-clock trials `t_nom_computed`.
On mission-clock trials run by `runner_main` it also records `soft_cap_s`: the soft cap the runner
applied to every trial of the run (the largest budget over the grid, or `--timeout-s`), which
`runner_main` passes to the driver as `Exp4Driver.soft_cap_s`; a driver used without `runner_main`
is told no cap and writes no `soft_cap_s`. When a trace is scored without its trial CSV, a marker
`ok` is relabelled `timeout` only if its run time exceeds `soft_cap_s`, or `trial_budget_s` when
the marker has no `soft_cap_s`, so `traces_scorer.py` gives the runner's verdict on a grid with
several `--N` or `--n-missions` values (final check, legacy F1). Wall-clock markers keep their five
recorded keys (`status`, `error`, `n_missions_target`, `run_s`, `trial_budget_s`) on the error path
as on the ok path; there, as recorded, only the trial CSV knows about a `--timeout-s`.

### 17.6 Trace fields on the mission clock

Additive, and only on the mission clock unless stated; the envelope `ts` stays wall time. A
wall-clock trace at the defaults keeps its recorded key sets.

- **`mule_ready`:** `mission_clock` ("sim"), `clock_epoch_s`; the ferry settings
  (`FerrySpec.describe`): `contact_band`, `band_classes` (floor, altitude, n, σ_sh, quantile,
  margin, anchor, carrier, and per class its index, bandwidth, N_RB, occupied bandwidth, κ,
  `range_slant_m`, `range_planar_m`, floor and peak rates), `channel_params` (contact and backhaul),
  `backhaul_model`, `backhaul_policy`, `backhaul_timing_snr_db` (under `mission`),
  `in_flight_response`, `replan_fallback`, `contact_reliability_source`, `device_availability_n`,
  `payload`, `energy_params` (powers, capacity, speed, `status: "simulated"`), `cruise_speed_m_s`,
  `dock`, `dock_turnaround_s`, `listen_s`, `deadline_bounds`; and `payload_bytes`,
  `deadline_time_scale`, `initial_window_s`, `effective_initial_window_s` (Φ₀ in clock seconds),
  `t_nom_s`, `trial_seed`, `input_dim`, `rf_link_token_set` and `rf_prior_source`
  (`seconds_backhaul`, `mission_schedule` or `constant`). On the wall clock the three time-unit
  fields appear only when the unit is not the recorded one.
- **`mission_started`:** `sim_start_s` and `rf_prior_snr_db`, the prior the Pass-1 plan gets.
- **`mission_completed`:** `sim_start_s`, `sim_end_s`, `sim_ledger`, `sim_pass_2_start_s`,
  `pass_1_flown` and `pass_2_flown`, `replans`, `aborts`, `inserts`, `offers_refused`,
  `pass_1_preflight_drops` (each contact S3b or the pre-flight order check refused before takeoff,
  which the mule widened then: `position`, `devices`, `deadline_ts` and `reason` = `overdue`,
  `budget`, `energy` or, under `deadline_bounds = delivery`, `delivery`, listed in that order; []
  for the whole-scheduler baselines D1–D5, whose walks report no drops; no predicted home time;
  final check E2E1-01), `budget_overrun_s` (how far the Pass-1 upload, or the landing when nothing
  was uploaded, ended past the budget; 0 within it, none without a budget), `delivery_overrun_s`
  (under `deadline_bounds = delivery` only, and only then present in the event, so the other
  values keep their key sets: how far the Pass-1 upload, or the landing when nothing was uploaded,
  ended past the earliest Deadline(j) of the updates on board, counted as the in-flight check
  counts them, i.e. members collected CLEAN, each with its own Deadline(j); 0 when every one was on
  time or nothing was on board; D1–D5 fill it too, though no clause holds them to it),
  `pass_2_budget_overrun_s`, `energy_j`, `band`, `backhaul` (carrier, `snr_db`, `p_loss`,
  `t_start_s`, `upload_s`, `t_upload_s` = the completion, `bytes`, `below_floor`) and
  `energy_status`. Each flown stop records its position,
  devices, `deadline_ts`, the departure state (`depart_s`, `depart_pose`, `depart_energy_j`),
  `transit_s`, `arrival_s`, `end_s`, `band`, `targets`, `unreachable`, per-member `snr_db` and
  `rate_bps` at arrival, `dwell_s`, `listen_s`, `missing`, `uplink_dropped` and `l1_choice` (the L1
  actor's choice, logged and never acted on; none in processes). The arrival `snr_db` and
  `rate_bps` do not reproduce `dwell_s`, which is priced at each target's own session start
  (critic C2); the session-start SNRs are not in the mule's trace. `pass_1_plan[].deadline_ts` and
  `pass_1_outcomes[].contact_ts` keep their names and are simulated seconds, as
  `mule_ready.mission_clock` says. Under `deadline_bounds = delivery` the in-flight records can
  also read `delivery`: `aborts[].reason`, and `replans[].rejected[].reason` and
  `replans[].dropped[].reason`, for a stop that would land an update already on board after its
  Deadline(j).
- **Mule failures:** `dock_bootstrap_failed` (exit code 5) when the bootstrap DOWN is refused.
- **Devices, either clock:** `device_reconnected` (`attempts`, `down_s`), only when a re-dial
  succeeds (Amendment 10). The registry counter `rf_reconnects` counts the same re-dials but
  reaches a trace only in the end-of-run `metrics_snapshot`, which a device the orchestrator stops
  on Windows never writes: count `device_reconnected` instead.
- **Cluster:** `cluster_ready` gains `mission_clock`, `backhaul_model` and, when it orders uploads,
  `sim_order = "conservative"`; `up_bundle_ingested` and `backhaul_upload_lost` gain
  `sim_upload_ts`, `carrier`, `snr_db` and `p_loss`, and in ordered cells `sim_order_seq`,
  `held_wall_s` and `sim_order_late`; `cluster_round_closed` and `model_eval` gain `sim_ts`, the
  latest simulated upload ingested. An upload is `sim_order_late` when it completed strictly
  before one already folded (a tie is not late); that happens after a mule's `down_wait_s` ran
  out, or when a mule joins behind an upload lost at a quorum of 1 or under FedBuff, whose time the
  cluster does not echo. An UP without a simulated time is folded at once and counted as
  unordered. New: the event `mule_departed`. The cluster also keeps the registry metrics
  `backhaul_unpriced_uploads`, `mules_departed`, `sim_order_late_uploads` and
  `sim_order_unordered_uploads`, and the timer `sim_order_held_s`. These are not trace fields
  (final check, E2E2-1): they reach a trace only in the end-of-run `metrics_snapshot`, which a
  cluster that the Exp 4 orchestrator stops on Windows (`terminate()`, which is TerminateProcess)
  never writes. A kept trace carries the same facts per event:
  - `mules_departed` is the number of `mule_departed` events;
  - `sim_order_late_uploads` is the number of fold events (`up_bundle_ingested`,
    `backhaul_upload_lost`) with `sim_order_late` true;
  - `sim_order_unordered_uploads` is the number of fold events carrying `sim_order_seq` whose
    `sim_upload_ts` is null or at least 1e9;
  - the `sim_order_held_s` samples are the `held_wall_s` values of every fold event carrying
    `sim_order_seq` (each UP the gate released);
  - `backhaul_unpriced_uploads` is the number of fold events with `p_loss` null when
    `cluster_ready.backhaul_model` is `"seconds"`.

  The one divergence is an UP whose ingest raised (counted in `ingest_failures`): it is counted
  and timed but has no fold event.
- **A restarted mule in an ordered cell** (final check, protocol F1). A mule restarted under the
  same id is tracked under its live dock session once the cluster knows of that session, from the
  mule's new bootstrap or its `registered` marker, whichever the cluster reads first. Dock sessions
  are numbered in the order connections register. From then on, a `registered`, `clock`, `done` or
  `departed` marker of an older session changes nothing and is not traced, even when the cluster
  reads it only after the restart (a restart during the startup wait). Two gaps remain, both
  because an UP carries no session:
  - an upload the crashed process sent, read after the live session is tracked, raises the live
    mule's bound to its time. Other mules' later uploads may then be folded before the live mule's
    next upload, which is folded with `sim_order_late` true and answered with a later
    `cluster_sim_ts` than its own;
  - an upload of the crashed process still held when the mule restarts stays ahead of the
    restarted mule's uploads and does not wait for them. It may be folded first, and the restarted
    mule's first upload is then folded with `sim_order_late` true. Otherwise, when another mule's
    upload falls between the two times, the fold stalls: nothing is released until a waiting
    mule's `down_wait_s` runs out, which under the Exp 4 driver is the whole trial's wall budget.

  The Exp 4 orchestrator spawns each mule once and never restarts one, so only a manual or
  fault-injected restart reaches either gap.
- **Messages and lines:** `FLOpenSolicit.solicit_id`; `in_reply_to` on `FLReadyAdv`,
  `GradientSubmission` and `DeliveryAck`; `DiscPush.uplink_drop`; `MissionRoundCloseLine.snr_db`;
  `MissionDeliveryLine.band` and `bytes_sent`; `UpBundle.sim_upload_ts` (the upload's completion,
  critic B8) and `backhaul`; `DownBundle.cluster_sim_ts`; `SpectrumSig.contact_class_snr_db` and
  `DeviceSchedulerState.spectrum_snr_db`. All default to the recorded value; the bundle signatures
  do not cover them.

### 17.7 Scoring on simulated time

`experiments/exp4/events_consumer.py`, `metrics.py` and `experiments/analysis/traces_scorer.py`.

- **The clock domain** is read from the mules' `mule_ready.mission_clock`, else the cluster's
  `cluster_ready`; absent means wall, which is every recorded trace. A trace whose clocks disagree
  raises `ClockDomainError` (a `ValueError`) instead of being scored: mules that disagree with each
  other or with the cluster, an unknown clock, a simulated trace with a mission lacking
  `sim_start_s`/`sim_end_s` or a stamp at or above 1e9 s (a plan deadline, a `contact_ts`, a
  cluster `sim_ts` or `sim_upload_ts`), or a wall trace with simulated fields. Scored anyway, such a
  trace would read 0 % or 100 % deadline misses. The driver records such a trial as an error.
- **Ordering.** On the mission clock missions complete, for the ages, Network AoU, merge credits and
  time to τ, in the order of their simulated ends (`sim_end_s`); the wall envelope `ts` still places
  cluster events in the missions' windows. Deadline misses compare simulated contact times with
  simulated deadlines.
- **Time to τ:** `sim_s_to_τ` = the reaching evaluation's `model_eval.sim_ts` minus the fleet's first
  `sim_start_s`, beside `missions_to_τ`, `rounds_to_τ` and `wall_s_to_τ`.
- **15 columns** after `tau`, blank on the wall clock: `sim_mission_duration_s_mean` (takeoff to the
  Pass-2 landing); the per-mission ledger means `sim_transit_s_mean`, `sim_dwell_s_mean`,
  `sim_listen_s_mean`, `sim_return_s_mean`, `sim_upload_s_mean`, `sim_turnaround_s_mean` and
  `sim_dock_wait_s_mean`, which sum to the duration; `sim_energy_j_mean` with `energy_status`
  ("simulated"); `sim_budget_overrun_s_mean` and `sim_budget_overrun_rate` (the share of budgeted
  missions that overran, design R11); the trial's totals `sim_replans`, `sim_aborts` and
  `sim_inserts`. No column reads `mission_completed.delivery_overrun_s` (§17.6), since a new
  column would change every CSV header; read it from the trace.
- **What stays wall (critic B13):** `mission_duration_s_mean` and `wall_s_to_*`, which on the
  mission clock measure host compute and TTL waits, not flight. Use `sim_mission_duration_s_mean`
  and `sim_s_to_*`.
- **What the serve counts mean (critic B13).** On the wall clock every solicit is a broadcast: every
  eligible device answers, and a non-member times out waiting for a push, which still counts as a
  serve. On the mission clock solicits are targeted, so `per_device_serves`, coverage, Jain's index
  and participation entropy count member contacts only. Ferry and wall-clock rows do not compare on
  those four.
- **Provenance:** 13 columns, blank at the driver's defaults. Ten are blank at their recorded
  values: `mission_clock`, `contact_band`, `in_flight_response`, `backhaul_model`,
  `contact_reliability_source`, `deadline_time_scale` (the resolved number), `initial_window_s`,
  `t_nom_s`, `session_ttl_s` (blank at 3 s) and `ferry_params` (JSON of every other ferry field the
  mules ran, `backhaul_period_s` resolved to P_bh, and `t_nom_computed`); a wall-clock row fills
  the scale and the TTL when they differ from 1.0 and 3 s, and Φ₀ when it is given. Three settings
  no CSV recorded before are filled whenever they are on, on either clock: `l1_channel` (1),
  `realism` (1) and `input_dim` (the model's width; blank on the stub). The scorer derives all 13
  from a trace's configs in the driver's format; on the 600 kept traces that gives `realism` 1 on
  all, `l1_channel` 1 on the 120 C/C2 traces, `input_dim` 21 (blank on the 40 stub S3c traces) and
  the other ten blank. `t_nom_computed` is read from
  `trial_status.json`; for a trace without it, it is inferred, exactly unless `--t-nom-s` was given
  beside a setting that needs T_nom or beside any Φ₀.
- **Fresh CSV paths** (critic D3). The 13 provenance and 15 simulated columns change every CSV
  header, recorded arms included, so the runner refuses to append to a file written before Phase 3.

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
