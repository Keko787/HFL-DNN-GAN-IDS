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
| `asynchfl_form` | `aggregation_params` | `polynomial` | config field; `--agg-asynchfl-form` | `agg:asynchfl`'s staleness function, on each device's age at the mule and on each partial's age at the cluster. `polynomial` is Async-HFL's s(a) = (a + 1)^−q, which it adopts from FedAsync (decided 7 Oct 2026; [related work §6a](HERMES_Related_Work_Notes.md), item 10). `exponential` is s(a) = exp(−λ·a), this rule's form before then, which is not Async-HFL's. Written to `aggregation_params` under `agg:asynchfl` only, so every other rule's rows and traces read as before. |
| `poly_q` (q) | `aggregation_params` | 0.5 | config field; `--agg-poly-q` | `agg:asynchfl`, polynomial form: q in (a + 1)^−q. Async-HFL does not report its q; 0.5 is FedAsync's polynomial, and FedBuff's (1 + τ)^−0.5. |
| `decay` (λ) | `aggregation_params` | 0.5 | config field; `--agg-decay` | `agg:asynchfl`, exponential form: λ in exp(−λ·a). Given without a form, it selects the exponential: a recorded `agg:asynchfl` dict names `decay` and no form, and reads as the exponential it flew. |
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
wait for Study 5.4 (the Run Guide's §2.6 has the pilot plan). *Phase 4 (2026-09-30) lands it
behind `member_admission` (§18.5): under `subset` a contact that fails whole is re-issued with the
members that still fit, for the plan arms by default and for H1–H3, D1–D3 and D5 when a run asks
(never D4). On this instance, at T2's unit, every gated arm then admits 5 devices at 60 s and 7 at
99.0 s. `whole`, the default, keeps the cliff, and the pins above are its side.*

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
  final check E2E1-01; Phase 4 adds the reason `plan` in plan mode and reports what D1–D5 leave
  out in `pass_1_policy_drops`, §18.6), `budget_overrun_s` (how far the Pass-1 upload, or the landing when nothing
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

## 18. The plan clock (FeRRy Phase 4)

`plan_mode = "legacy"` is every recorded run, and every default in this section reproduces it
(Freeze §5g, Rule 1; §5k). `"ferry"` plans each mission at the dock: the band class b̄ and the
Pass-1 route as one decision, ranked by an age cap and a plan score, with stops reduced to the
members that fit. The settings are the user's decisions of 2026-09-30 (build-plan decision D4 and
the seven decisions of the Phase 4 spec) with the resolutions R1–R14 and the hover decision recorded
in Freeze §5k. Code: [hermes/scheduler/plan/](../hermes/scheduler/plan/__init__.py) (`types.py`,
`plan_score.py`, `member_subset.py`, `plan_search.py`, `hover.py`),
[hermes/scheduler/stages/s3d_age_cap.py](../hermes/scheduler/stages/s3d_age_cap.py),
[hermes/scheduler/policies/cross_heuristic.py](../hermes/scheduler/policies/cross_heuristic.py),
[hermes/scheduler/fl_scheduler.py](../hermes/scheduler/fl_scheduler.py) (`build_ferry_plan`),
[hermes/processes/config.py](../hermes/processes/config.py),
[experiments/analysis/age_cap_s_star.py](../experiments/analysis/age_cap_s_star.py).

**One plan-mode mission** (`FLScheduler.build_ferry_plan`, then `MuleSupervisor`):

1. S1 and S3, as the recorded pipeline runs them. The demand is the devices S3 bucketed and dated,
   in S3's order.
2. The age cap (§18.2): each demanded device's age and the capped set; then each device's coverage
   weight (§18.3).
3. For each class the arm may fly (every class of the link under `search`, the pinned one under
   `fixed:<class>`): S3a at the class's radius R_planar(c), then the hover rule (§18.4). Pass 2 is
   priced once per class on its S3a stops, as T_nom prices it: folded from the dock, no budget,
   delivering.
4. The search (§18.4) returns the candidate with the smallest plan key over every class searched.
5. A guard fold: the route must pass S3b's predicate as flown (`RULE_DEADLINE_BUDGET`, the exempt
   stops protected, the set computed on the route itself). A failure raises `FLSchedulerError` and
   commits nothing: it would be a bug in the search.
6. The commit: b̄'s physics become the scheduler's model, so the departure check, the re-plan and
   Pass 2 price b̄; `last_feasibility` holds the route and every demanded device left out, labelled
   (§18.6); `last_plan` holds the `PlanCommit`.
7. The mule sets its runtime's band to b̄ (`FerryRuntime.set_band`), annotates the queue, widens and
   records the drops, flies Pass 1 with the flight slot (§18.5), flies Pass 2 on b̄, and closes the
   plan right after `record_merged` (with nothing merged on the empty path).

Without a budget nothing is gated (S3b's opt-in contract): a plan cell without `--mission-budget-s`
is the control, serving whom the plan key prefers.

### 18.1 Switches and configuration fields

The eight plan fields are `PLAN_MULE_FIELDS`, inside `SIM_ONLY_MULE_FIELDS` and outside
`FERRY_SPEC_FIELDS`, so on the wall clock each must keep its default and Phase 3's `ferry_params`
strings are unchanged. Every field after `plan_mode` keeps the name of
`hermes.scheduler.plan.PlanOptions.from_config`'s keyword (`PLAN_OPTION_FIELDS`). The driver sets
them per arm (§18.8).

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `plan_mode` | `MuleConfig`, `MuleSupervisor`, `FLScheduler` | `legacy` | config field; the plan arms | `ferry`: the plan clock (above). Needs the simulated clock, a `contact_band` (the reference class; the channel-free control has no classes to choose among) and `t_nom_s` (T in the score, decision 2 (b)). At the defaults (`legacy`, `whole`) nothing loads the plan package; an H or D arm under `subset` loads its member walk (`plan.member_subset`). |
| `band_class_policy` | `MuleConfig`, `PlanOptions` | `search` | config field; arms F and FB+c | `search` searches every class of the link each mission (arm F); `contact_band` is then only the reference class, which the ferry spec, T_nom, D4's CARP split and the H and D arms of the same CSV use, and the provenance column reads `search`. `fixed:<class>` (Path B+, the FB+ arms) must name the run's `contact_band` and flies only the `committed` slot: FB+c flies only class c (the commit refuses anything else). |
| `member_admission` | `MuleConfig`, `MuleSupervisor`, `FLScheduler`, `PlanOptions` | `whole` | config field; `--member-admission` | `whole` admits a stop with all its members or none, the recorded rule and its narrow-band cliff (§17.1). `subset` may admit the members that still fit (§18.5): the plan arms' default (the driver passes it), and for H1–H3, D1–D3 and D5 only when a run asks; never D4. Valid on the simulated clock only. The scheduler's value is the one source: plan mode refuses options that disagree. |
| `flight_slot` | `MuleConfig`, `PlanOptions` | `committed` | config field; arm FX | `committed`: the plan's next stop on b̄ (F, the FB+ arms). `cross_heuristic`: FX's rule (§18.5, decision 5). Refused with a pinned band. |
| `age_cap_missions` | `MuleConfig`, `AgeCapSpec.s_missions` | None (off) | config field; `--age-cap-missions` | The cap S (§18.2), an int ≥ 1, plan mode only. Set per cell from the S\* tool (§18.7): the smallest value covering 90 % of layouts at both pilot budgets, never below 2. S = 1 stays legal, since the D4 check flies F at S − 1. Arm F-cap runs without it. Not the merge cutoff a_max (`aggregation_rules.age_cap`, `MuleSupervisor._age_caps`), which counts cluster rounds (§14). |
| `age_cap_lookahead` | `MuleConfig`, `AgeCapSpec.lookahead` | 0 | config field; `--age-cap-lookahead` | L: a device is capped from age S − L. Kept at 0 (R9): L = 1 does not remove the miss of critic probe A3, and with L > 0 the mule logs devices that the scorer, at S, does not count. |
| `plan_score_params` | `MuleConfig`, `PlanScoreParams` | {} (the defaults of §18.3) | config field; `--plan-score-params` (JSON) | The score's settings by field name; an unknown key or a value the type refuses is refused, so a misspelt setting cannot keep its default in silence. `mule_ready` records them resolved. |
| `plan_search_params` | `MuleConfig`, `PlanSearchParams` | {} (the defaults of §18.4) | config field; `--plan-search-params` (JSON) | The search's bounds by field name; unknown keys are refused. |

**Refused combinations** (each fires only on a non-default value):

- *The configuration* (`mule_config_errors`, which `TopologyConfig.validate`, the mule process and
  the driver's `check_arm` all run). On the wall clock: any plan field other than its default. On
  the simulated clock outside plan mode: an unknown `plan_mode` or `member_admission`; `subset` with
  `contact_policy = "fedex"` (D4's tour has no gate); and any plan field other than
  `member_admission`, the cap included, which is refused rather than ignored. In plan mode: no
  `contact_band`; no `t_nom_s`; a band-class policy other than `search` or `fixed:<the run's
  contact_band>`; a `replan_fallback` other than `trim`, under either in-flight response (critic
  B11); `pass_2_budget` (critic B8: the budgeted Pass-2 walk admits whole stops, so a narrow Pass 2
  at 1 MB, 45 s at the median, would deliver nothing under a short budget); a cap that is not an int
  ≥ 1 or a lookahead that is not an int ≥ 0; a `contact_policy` or `use_rl_selector` (the plan owns
  admission and order); and `in_flight_response = abort` together with a cap (critic A10: abort
  gives up the whole tail, capped stops that would fit alone included). Then the options are built
  as the mule builds them (`PlanOptions.from_config`, the only place this module loads the plan
  package), which refuses an unknown flight slot, `cross_heuristic` with a pinned band, and unknown
  score or search settings.
- *The scheduler* (`FLScheduler`). `subset` without the ferry physics (the wall clock), or with a
  whole-scheduler policy that does not declare `admits_member_subsets` (D4); in plan mode a target
  selector, the `reorder` fallback, a `member_admission` other than the options', or classes that do
  not share one dock; `build_ferry_plan` before `set_mission_round` (with a cap, critic B9; without
  one too, since the age weights read the ages) or at a pose other than the dock.
- *The supervisor* (`MuleSupervisor`). `subset` or plan mode without the mission clock; plan options
  or `t_nom_s` in legacy mode; in plan mode the channel-free control, no `t_nom_s`, `pass_2_budget`,
  or `abort` with a cap.
- *The driver.* A plan arm on the wall clock, and any plan arm whose mule configuration or ferry
  spec the guards above refuse (`Exp4Driver.check_arm`, which checks the plan arms only and passes
  every other known arm); the runner calls it for each plan arm named and turns a refusal into a
  usage error before any trial.

### 18.2 The age cap

**The age** (decision 1). a_j(m) = m − U_j, where m is the mission being planned
(`FLScheduler.mission_round`, set by the mule before every plan; the first mission is 1) and U_j is
the mission whose merge last used j's update (`last_merged_round`, set by `record_merged` when the
mission closes), 0 if none has. So a device never merged has age m, one merged in the previous
mission age 1, and one merged in the mission being planned age 0 (which cannot happen before a
plan). There is no clamp to 1 (the score floors the weight instead, §18.3) and no fallback to
`last_clean_round`: a CLEAN update the merge cutoff excluded is not service. This is the scorer's
unit: a device capped at L = 0 is exactly one the scorer counts at age ≥ S after mission m unless m
merges it, and a never-merged device is capped from mission S on.

**Capped, and the three kinds of stop.** A device is capped when its age reaches S − L. On any
route, whole or reduced, computed on the route actually folded (critic B1):

- an *exempt* stop (every member capped) skips its own deadline clause, the predicate's `protected`;
  it keeps its members' earliest deadline, which stays finite for the trace;
- a *mixed* stop (some members capped) is not exempt: it carries the earliest deadline among its
  uncapped members (critic B2), so the capped members' lateness is excused and the others' is not;
- a *priority* stop (any member capped) goes first in a trim and sheds its uncapped members first.

**The cap key** is the ages of the capped devices a plan leaves out, largest first, and the plan key
compares it before anything else (§18.3): a plan that leaves an older capped device out loses to any
plan that keeps it. It minimises the oldest unserved age, not the number of violations: (4, 4, 4)
beats (5, 3) (critic C5), the plan's fallback that "keeps the oldest capped devices that fit".

**Violations** (other choices 6), one per capped device a mission fails, with its planning age,
reported by cause with no pass mark. Device availability alone makes about 15 % of device-missions
miss at S = 3 (critic A2).

| Cause | When | Means |
|---|---|---|
| `unplannable` | the plan is made | The plan left the device out, and no class the arm may fly serves it alone within the budget, from the dock at takeoff, at its offered stop: its S3a stop when that serves it alone, else its best hover point (§18.4), which no other point beats. So it is physics for the time budget, under either admission. Under an energy capacity it reflects the time-minimising point: a slightly slower point with a shorter dwell can need less energy (hovering costs 168.5 W, flight 143.6 W), and the label does not look for it. |
| `crowded` | the plan is made | Some class could serve the device alone there, but the chosen plan does not. Under `whole`, a capped device left out with its whole stop is `crowded` when it fits alone: the admission rule, not physics, left it out. |
| `dropped_in_flight` | the mission closes | The plan served the device, but no Pass-1 stop flown held it: its stop was dropped, or trimmed without it, in flight. |
| `not_merged` | the mission closes | A flown stop held the device, but its update did not reach the mule's merge: no reply, a failed availability draw (critic C3: the cap can then pin a device whose draw keeps failing), a member not solicited on arrival (out of range or below the floor), or an update the merge excluded (under `agg:cutoff`, past a_max; the close reads the merge's exclusions, not the round report). |

The mule's log (`plan.cap.violations`) and the scorer's count (§18.7) differ by what the mule cannot
see: a lost backhaul upload, a merge the cluster defers (critic B10: the mule credits its own merge
at its close, before a quorum closes the cluster's round), and with L > 0 the devices aged S − L to
S − 1 that the mule logs.

### 18.3 The plan score and the rank

    V(b̄, π | demand) = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T)
    U     = 1 − Σ_served w / Σ_demand w
    L     = Σ_served w·p_out / Σ_demand w
    p_out = Φ((floor − SNR_b̄(d)) / σ_eff),   σ_eff = √(σ_sh² + σ_I² + A²/2)

- **Δ** is the whole mission on b̄ (decision 2 (b)): Pass 1 (transit, dwell at the class's predicted
  rate, the return leg, the upload), the dock turnaround, and Pass 2, which flies b̄ too. The band
  sets Pass 2's length: at 1 MB and N = 6 its median is 94 s on wide, 62 s on medium and 45 s on
  narrow (critic C2). A plan that serves nobody flies neither pass and pays the turnaround only.
  **T** is the cell's T_nom (`t_nom_s`).
- **U** is the weighted coverage shortfall over the demand (the devices left after S1 and S3); the
  served devices are the members of the plan's stops, reduced or not. U = L = 0 for an empty demand.
- **L** is the expected link loss, the design's link option (ii): a served member at planar distance
  d misses its contact with the chance that its SNR falls below the floor around the mean SNR the
  planner prices (δ_obs = 0), with the contact channel's spread. The moment match is exact for the
  shadowing and the noise but not for the sine: against the exact outage it agrees to 4 decimals on
  the clean channel and within 0.008 on the jittery one (at a class's edge 0.1776 against 0.1849). A
  member beyond R_planar(b̄) has outage 1, and a noise-free channel gives a step. The formula has
  two definitions, the runtime's (`FerryRuntime.outage_probability`, which the planner calls) and
  the score's (`plan_score.outage_by_distance`), tied by a test within 1e-15 (R3).
- **E** is the SIMULATED energy of both passes, return legs included; over P_hover·T it reads as a
  share of a mission spent hovering.
- **The weights** (decision 3): `age` gives max(a_j, 1) × (1 + m_j) with the arm's `miss_priority`
  on (F: about a_j², since every device left out is widened as a miss, critic A5) and max(a_j, 1)
  alone with it off (F-prio); `uniform` gives 1 × (1 + m_j) or 1, the plan's letter 1 − served/N.
  The floor at 1 is in the weight only; the cap reads the raw age. Every weight is ≥ 1.

V of the empty plan is −c₁(t_turn/T)² − c₂. With Δ and E held and c₃ ≤ c₂, serving one more device
never lowers V; it raises V when c₂ > 0, unless c₃ = c₂ and the device's outage is 1. At c₂ = c₃ = 0
(F-cov) V does not see coverage at all, so F-cov serves only what the cap forces ("cap-only
service"). V is declared, not derived: the repo holds no derivation from the theory track, so V is
hand-set in FedEx-Async's form (Theorem 2, eq. 24) with the convex-surrogate caveat, and swept.

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `c_time` | `PlanScoreParams` | 1.0 | `plan_score_params` | c₁ (spec, other choices 7). |
| `c_cov_per_device` | `PlanScoreParams` | 1.0 | `plan_score_params` | κ, with c₂ = κ·N_demand: one average device is worth κ full T² of time. κ = 1 (decision 2): serve every device the budget allows, and let time break ties. The pilot sweeps κ in {0.15, 0.25, 1} (`plan_score.PILOT_KAPPAS`), under the `weighted` rank. At κ = 0.15, 2 of 30 plans already fly empty at 30 s and 1 MB, and at 0.1, 1 to 9 of 30, which is why the critic's 0.1–0.25 (C1) was not taken. |
| `c_link` | `PlanScoreParams` | None (c₃ = c₂) | `plan_score_params` | With c₃ = c₂, c₂U + c₃L = c₂(1 − the expected weighted served share). |
| `c_energy` | `PlanScoreParams` | 0.1 | `plan_score_params` | c₄. E is nearly collinear with Δ (143.6 W flying, 168.5 W hovering), so energy breaks ties. The pilot sweeps it in {0, 0.1} (`PILOT_C_ENERGIES`). P_hover = 0 is refused while c₄ > 0. |
| `coverage_weights` | `PlanScoreParams` | `age` | `plan_score_params` | `age` or `uniform` (above). |
| `dwell_in_delta` | `PlanScoreParams` | True | `plan_score_params` | False takes both passes' dwell out of Δ in the score only (arm F-dwell, Study 5.7); the predicate still prices it and E still counts the hovering. `score.delta_s` is then the Δ that V prices, not the predicted mission, which is `score.mission_s` for every arm (R4). |
| `coverage_rank` | `PlanScoreParams` | `lexicographic` | `plan_score_params` | How the candidates that tie on the cap key are ranked (R11). `lexicographic`: the served weight share, Σ_served w / Σ_demand w (1 for an empty demand), then V. `weighted`: V alone (the pilot's κ sweep). With κ = 0 (F-cov) `weighted` applies whatever it says. The rank changes neither V nor its terms. |

**The plan key**, the smallest wins: under `lexicographic` (cap key, −round(share, 9), −round(V, 9),
class index, each stop's (position, devices)); under `weighted` and for F-cov (cap key, −round(V,
9), class index, stops), which is `Candidate.key` itself. Rounding to 9 decimals makes the same plan
priced along two float paths tie, and the tie then falls to the class and the stops, so the order is
total and the pick never depends on the order candidates are met. Under `lexicographic` the empty
plan wins only when no plan that serves anyone is admitted. Why the share comes first: V alone lets
the empty plan outscore a device that fits, since every serving plan pays a whole Pass 2 and the
empty plan pays only the turnaround (U7's probe: −3.85 for serving one device against −3.02 for
flying empty; Freeze §5k). The share is nominal: a served member counts fully, though on the pilots'
jittery channel its outage at a class's edge is about 0.15–0.2. Missions can run longer under it,
since time only breaks ties (Freeze §5k gives the measured cost).

### 18.4 The plan search and the hover rule

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `exact_max_devices` | `PlanSearchParams` | 6 | `plan_search_params` | A demand this small is searched exactly (`exact`): every ordered sequence of distinct offered stops, each reduced to every non-empty member subset, depth first, a failing prefix pruning its extensions and, at one flight state, a failing member set its supersets (every clause of the predicate is monotone in a stop's members). This is the optimum over the member subsets V prices (critic A6), checked against an independent brute force. |
| `exhaustive_max_stops` | `PlanSearchParams` | 6 | `plan_search_params` | Above the exact demand, a class with at most this many stops is searched depth first over ordered stop subsets (`stop_subsets`), each stop whole if it fits, else reduced greedily in the F member order, skip not stop; a stop that admits nobody prunes its branch. The plan's threshold (L829). |
| `heuristic_max_passes` | `PlanSearchParams` | 50 | `plan_search_params` | Above both, `local`: a 2-OPT tour of the stops from the dock and back, its member trim (under `whole`, its priority stops first, each kept whole if it fits, exempt stops protected), then first-improvement scans. A pass scans the current route's neighbours in a fixed order and moves to the first with a smaller key: drop a stop; insert an unrouted stop with all its members, at every place from the front; reverse a segment of two stops or more; and, under `subset`, drop one served member of a stop that serves several, least worth first. The order is part of the contract: it decides which local optimum a scan reaches and how many walks the trace records. A move counts only when every stop it lists admits someone, and a neighbour walked before is not walked again. The search stops at a local optimum, after this many passes, or at the walk bound. |
| `heuristic_max_evaluations` | `PlanSearchParams` | 2,000 | `plan_search_params` | Walks per class (a walk folds a whole route; the start's counts), every scan included. A count, never wall time, so a repeated trial plans the same (critic C7). It buys determinism, not a time bound: a walk costs more on a longer route, and at N = 96 on a 500 m field at 1 MB with no budget every class reaches it and a plan took 3.1–3.4 s. |

**Which mode runs.** Per class: `exact` when the demand has at most `exact_max_devices` devices,
else `stop_subsets` when the class has at most `exhaustive_max_stops` offered stops, else `local`.
The empty plan is one candidate per class: it flies nothing, and the score leaves Pass 2 out. With
several classes the empty plan's band is the first searched class (R8), and an empty demand gives
mode `exact` and the empty plan on that class. Under `whole` every mode flies whole stops, and the
drop-member move is off (R5). At N = 6, the pilots' size, the search is exact: a whole plan took at
most about 0.25 s with no budget (a synthetic worst case) and about 20 ms under 30–120 s budgets.

**Above 6 devices the search is a heuristic** (a plan deviation). The stop-subset family never sheds
a member from a stop that fits whole, and the local search has no move that swaps one stop for
another, so where two capped stops compete for one slot it keeps the one the trim took first: forced
onto small capped problems it ended with a worse cap key than the exact search in 6.1 % (R7). Under
`lexicographic` a scan on the plan key never takes a drop or a drop-member move, since each serves
less weight, so the local search runs up to three scans: `weighted`'s own (on `Candidate.key`, from
the trim's route), then the plan key's from the best plan met, then the plan key's from the trim's
route when that differs; a later scan reuses an earlier one's walks without counting them, and all
share the class's bounds. The class's best is never below `weighted`'s plan under the plan key and,
unless a bound ends the search, it is a local optimum of every move. The A3 guarantee (F's plan key
is never above any FB+c's) holds because each class is searched on its own with its own bounds; a
bound shared across the classes searched would break it.

**The hover rule** (the user's decision of 2026-09-30; `plan/hover.py`). In plan mode, under a
budget and a cap, each capped device that its own S3a stop cannot serve alone within the budget
(U1's `servable_alone`, from the dock at takeoff, deadline-exempt) leaves that stop for a one-device
stop at its best hover point on the class. The stop it leaves keeps its position and takes its
remaining members' bucket and B2 deadline; an emptied stop goes; every other stop is S3a's own
object, so each device stays in exactly one stop. Uncapped devices, and capped devices their S3a
stop serves alone, keep S3a's stops; without a budget, or with the cap off, nothing moves; Pass 2 is
priced and flown on S3a's stops. The best hover point is the point p of the segment from the dock to
the device that minimises its alone mission (the flight to p, the dwell at |device − p| at the
class's predicted rate, the flight back, the Pass-1 upload), priced by the class's own
`FeasibilityModel`, among the points within the class's planar reach of the device by both distance
formulas the code uses (the model's `** 0.5`, which decides whether the dwell is charged, and the
`math.sqrt` of S3a and the contact gate, which decides whether the device is solicited) and where
its predicted dwell is finite. The dwell never falls with the distance on the contact link, so no
point of the plane serves the device alone sooner, and a device none of the class's points serves is
out of reach of the class within the budget. The point depends on neither the clock, the budget nor
the partition, and is found deterministically:

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `GRID_CELLS` | `plan/hover.py` | 64 | module constant | The first grid over the distances from the device, [0, min(the device's distance from the dock, R)], cut back by bisection (`REACH_STEPS` = 64) to the farthest finite dwell when the floor comes first. |
| `ROUNDS`, `KEEP_CELLS` | `plan/hover.py` | 64, 8 | module constants | At most 64 rounds halve at most 8 cells each, the lowest bounds first. A cell's bound is the flight at its end nearer the dock plus the dwell at its end nearer the device, plus the upload. The dwell is a step function of the distance, so only cells that hold a step edge survive, and the halving brackets the best edge. |
| `TOLERANCE_S` | `plan/hover.py` | 1e-9 s | module constant | A cell is halved while it could still beat the best point by more than this. A tie goes to the point nearer the device. On the critic's 30 layouts a point took 65 to 118 pricings, and a test holds it to an exact minimum within 1e-6 s at 1 MB, 8 MB, and 2 MB with a 100 m range. |

The point minimises time, not energy. It often sits at the dock or at the class's reach edge, where
the noisy link is weakest, so a hover stop misses more often in flight (Run Guide §2.7, pilot
notes). The per-class summary's `stops` counts the stops offered (S3a's with the hover rule), and
traces do not mark a hover stop: it appears in `pass_1_flown` as a one-device stop, recognisable
only by its position.

### 18.5 Member admission per arm family, and the flight slot

**Reduced stops.** A stop reduced to some of its members is an ordinary `ContactWaypoint`: the same
position (S3a placed it within R of every member), the subset in the stop's own order, the worst of
their buckets (the stop's own when none has one), and the earliest of their deadlines (B2's under a
cap); its `band`, `range_m` and `pred_snr_db` stay None until the mule annotates it. A stop that
fits whole is kept as the same object; a stop none of whose members fits is dropped whole with the
reason `fold` gives, except that a priority stop that keeps no one drops each member with the reason
that refused it. Feasibility is down-closed in a stop's members (dwell adds up, the deadline is a
minimum, energy grows with dwell), so one pass in member order is final, and the admitted set is
maximal.

| Arm family | Member order inside a stop | What happens to the rest |
|---|---|---|
| F, FX, FB+c, F-cov, F-cap, F-prio (`subset` by default) | capped members first, then weight per second of the member's predicted dwell alone, descending, then device id (a member with zero dwell and a positive weight counts as infinitely cheap; a member without a weight weighs 1) | The search reduces stops (§18.4). Each demanded device left out is dropped with the first clause that refuses it alone at its offered stop on b̄ from the dock at takeoff, in the predicate's order (`overdue`, never for a capped device; `budget`; `energy`; `delivery` cannot fire at takeoff), else `plan`, a choice; drops are widened like any pre-flight drop, and `plan` drops stay out of S3c's planned count (critic C6). Under `whole` a stop's left-out members are judged together as that stop (R12). In flight the trim reduces stops under `subset` and keeps or drops whole stops under `whole`. |
| H1–H3 (S3b), with `--member-admission subset` | the member's own deadline, then its Pass-1 dwell at the SNR offset, then id; the miss streak first when the arm's miss priority is on | The members left out are dropped by the clause that refused each, one reduced contact per reason, widened, recorded and counted in S3c's planned count as any S3b drop. |
| D1–D3 (`greedy_budget_walk`), with `--member-admission subset` | the arm's own contact key on the one-member contact (D1 the age, D2 Oort's utility with its staleness bonus, D3 the Whittle index), then id | Reported in `pass_1_policy_drops` (§18.6) and never widened (decision 6). |
| D5 (FedCS), with `--member-admission subset` | Algorithm 3's selection key on the one-member pick (the least marginal time, under both `fedcs_value`s), then id | As D1–D3. |
| D4 | never: its tour has no gate | — |

For H1–H3, D1–D3 and D5 only the pre-flight admission passes the member-subset carrier
(`MemberSubsets`, built by `build_contact_queue`). The in-flight check, the re-plan and the trim
never create reduced stops for them, and the budget walk refuses the carrier in Pass 2; Pass 2 is
never member-reduced for any arm (R10). Inside the pre-flight order check S3b's re-admission
re-sorts the kept stops by their own deadlines, and a reduced stop's deadline can be later than its
original's: in a random probe it dropped a stop in 17 of the 1,231 checks that fired under `subset`
(0 under `whole`), and such drops join `last_feasibility` and are widened.

**The plan-mode re-plan** (`FLScheduler.replan_remainder` in plan mode, Pass 1) is a trim of the
committed plan: the remainder is kept if it passes as flown with its exempt stops protected;
otherwise the priority stops fly first, then the rest, each part in its flight order. Under `subset`
it is U3's member trim (priority stops shed their uncapped members first; a capped member is dropped
only when the protected-only trim cannot hold it); under `whole` each stop is kept whole if it fits,
else dropped whole with its reason. It never re-orders otherwise, so `reorder` is refused in plan
mode (critic B11): re-ordering belongs to the flight slot. The plan's own members are dated by the
plan, a beacon insert's by the mule's record, and a member dated by neither by its stop's deadline.
In flight, in Pass 1 only, the exempt set is recomputed at each departure and given to the departure
check, abort's head fold, the re-plan, the beacon hook's fold and FX's `fits`; a capped member never
lowers `deliver_by`, in a mixed stop too; and the beacon hook groups an offer within b̄'s range.

**The flight slot** (`policies/cross_heuristic.py`; decision 5). The mule calls the slot at every
stop of both passes, after the departure check and the beacon hook:

- `committed` (F, the FB+ arms): the remainder's first stop on b̄, exactly the recorded
  `remainder.pop(0)`.
- `cross_heuristic` (FX), in Pass 1 only. *Next stop*: after each stop served (never at takeoff,
  where the plan's first stop is flown), the stop nearest the mule whose move to the front keeps the
  rest of the plan feasible (`fits`: the departure check's fold of the reordered rest from the
  departure state, updates on board included), else the plan's next stop; ties go to the stop's
  position, then its devices, then its place in the plan. *Band on arrival*: the fastest class whose
  targets include every target of b̄ there, priced at the arrival SNR (`FerryRuntime.arrival_view`:
  per class, the members it would solicit, within R_planar(c) and at or above the floor, and the
  dwell of serving them all); ties go to more targets, then b̄, then the class index. b̄ itself
  always qualifies, so FX never dwells longer than F would at the arrival SNR and never reaches
  fewer devices (R6); a contact charges each target at its own session start (critic C2), so the
  realized dwell can occasionally exceed F's. The departure check prices the rest on b̄, which stays
  conservative for whatever class FX flies. The design's first rule, the class that reaches the most
  members, added up to 94 s of dwell and overran the budget at the last stop, where nothing
  re-checks (critic A7).
- `pair_q` (the FQ arms; FeRRy Phase 5, §19.2), in Pass 1 only: one decision at each Pass-1
  arrival, the class to serve the stop on and the next stop, the masked argmax of a learned score
  over the covering classes times the remainder; the chosen stop is moved to the front after the
  stop, so the slot's call at the departure returns 0.
- In Pass 2 every slot flies the queue's order on b̄, and `FerryRuntime.contact_plan` refuses a
  Pass-2 contact plan on any other class.

### 18.6 Trace fields

Additive. The trace-event fields below appear on the simulated clock only; a recorded trace keeps
its key sets, and the D arms' report is the one trace-event field that can appear at the defaults.

- **Per-role JSON** (either clock): every mule's carries the eight plan fields, at their defaults
  when unset, so it gains keys at their defaults, as it did in Phase 3 (Freeze §5j).
- **`mule_ready`:** in plan mode `plan_mode` and the options the scheduler runs, read back from it:
  `band_class_policy`, `flight_slot`, `member_admission`, `age_cap_missions`, `age_cap_lookahead`,
  `plan_score_params` (the seven settings, defaults included) and `plan_search_params` (the four).
  Outside plan mode only `member_admission`, and only when it is `subset`.
- **`mission_completed.plan`** (plan mode): the closed `PlanCommit.describe()`, JSON-ready and free
  of wall time, so a repeated trial describes the same plan: `mission_round`; `band` (b̄) and
  `band_index`; `band_class_policy`; `search` (the committed class's mode) and `candidates` (scored
  over every class); `per_class`, one summary per class searched (`band`, `index`, `mode`, `stops`,
  `candidates`, and that class's best's `v`, `served` (a count), `cap_key`, `served_share`, with
  `coverage_rank`, the rank applied; a local search adds `evaluations`, `passes` and `bounded`, true
  when a bound rather than a local optimum ended it, and under `lexicographic` `weighted_passes` and
  `weighted_evaluations`, its first scan's); `budget_end` (absolute, on the mission clock); `score`
  (`v`, `delta_s`, `time`, `coverage`, `link`, `energy_j`, `energy`, `served_weight`,
  `demand_weight`, `mission_s`, and the constants `c` and `t_ref_s`); `demand`, `weights` and
  `served`; `cap` (`s`, `lookahead`, `ages`, `capped` and `violations`, each `{device, age,
  reason}`); and `visited`, the devices of the Pass-1 stops actually flown. The queue's stops are
  not in it: their positions show only in `pass_1_flown` for the stops flown.
- **`mission_completed.plan_wall_s`** (plan mode): the wall seconds the plan took, the only wall
  time a result holds, outside `plan` so determinism comparisons leave it out (critic B12).
- **`mission_completed.band`** is b̄ in plan mode; **`pass_1_flown[].band`** is the class each
  Pass-1 stop was flown on, which under FX can differ from b̄, and `pass_2_flown[].band` is always
  b̄.
- **`pass_1_preflight_drops`** gains a fifth reason in plan mode, `plan`, listed after `overdue`,
  `budget`, `energy` and `delivery`: demand the plan left out by choice, since the device fits from
  the dock at takeoff alone at its offered stop (under `whole`, with the rest of its stop), widened
  like any drop and left out of S3c's planned count. A drop's `position` is its offered stop's, a
  hover stop's included.
- **`pass_1_policy_drops`** (D1–D5 on the simulated clock, decision 6): each contact the baseline's
  walk left out before takeoff (under `subset`, the rest of a contact it served in part), as
  `position`, `devices`, `deadline_ts`, `reason` and `"widened": false`. The reason is the clause
  that refuses the contact alone from takeoff under the arm's in-flight rule (budget only for D1–D3
  and D5), else `budget`. Written only when non-empty, and never widened; `pass_1_preflight_drops`
  stays [] for these arms.
- **The consumer** (`experiments/exp4/events_consumer.py`): `MissionRecord` gains `band`,
  `flown_bands`, `plan_band`, `plan_search`, `plan_demand`, `plan_served`, `plan_v`,
  `plan_mission_s` (from `score.mission_s`, never `delta_s`), `cap_s`, `cap_capped`,
  `cap_violations` (`(device, age, reason)`), `plan_wall_s` and `policy_drops`, each None where a
  mission does not record it, and the property `has_plan`.

### 18.7 Scoring and the S\* tool

**Nine scorer columns** after `deadline_basis` (`traces_scorer.PHASE_4_COLUMNS`), blank where a
trace cannot say, so every trace recorded before Phase 4 scores blank in all of them at the defaults
and keeps every other column. They are scorer-only: the trial CSV's header is unchanged.

| Column | What it holds | Filled |
|---|---|---|
| `cap_s` | The S the cap columns count at | with `--age-cap-s`, else the trace's own S (the plans' `cap.s`, else a plan-mode mule config's `age_cap_missions`) |
| `cap_violations` | (device, own-mule mission) pairs whose age after the mission is at least S: a device capped when the mission was planned that the mission did not merge. Needs no plan, so it scores every arm alike, H and D arms and F-cap included | as `cap_s` |
| `cap_violation_devices` | the devices with any such pair | as `cap_s` |
| `cap_violation_events` | the mule's own log, by cause (all four causes listed, 0 included; JSON), at the S the mule ran | when a plan ran a cap |
| `band_shares` | the share of Pass-1 stops flown on each band class, per stop, so FX's switches count (JSON) | simulated-clock traces a Phase 4 build recorded |
| `plan_served_share_mean` | the mean over plan-mode missions with a demand of \|served\| / \|demand\|, counted in devices, not weights (the weights differ between F, F-prio and the uniform option) | missions that recorded a plan |
| `plan_v_mean` | the mean V of the committed plans | missions that recorded a plan |
| `far_served_share` | the far devices' merged updates over their own missions; a device is far when its planar distance from the dock exceeds the mule config's `rf_range_m` (R_planar(wide), Study 5.4) | simulated-clock traces a Phase 4 build recorded |
| `policy_drops` | the devices a whole-scheduler baseline left out before takeoff, summed over missions (0 when it left nothing out) | D arms on simulated-clock traces a Phase 4 build recorded |

A trace counts as recorded by a Phase 4 build when its mule config carries `plan_mode`, as every
Phase 4 mule's JSON does, at the defaults too. `--age-cap-s S` (an int ≥ 1) scores every arm of a
study at one S and warns about a trace whose mules ran another; with it, older traces fill the cap
columns too. A trace whose plans ran two different caps is refused (`ValueError` naming the trial),
with or without `--age-cap-s`. The provenance columns follow the driver's rule (§18.8), through the
driver's own `plan_ferry_params` and `contact_band_column`, and a plan-mode trace counts as needing
T_nom when the status marker is missing.

```bash
python -m experiments.analysis.traces_scorer --traces results/exp4_p4/s58_traces --age-cap-s 3 --csv scored.csv
```

**The S\* tool** (`experiments/analysis/age_cap_s_star.py`; the Phase 4 spec, other choices 13;
decision 1). For one layout, one budget and one arm family, S\* is the fewest Pass-1 missions that
together serve every device a mission can serve at all: the set-cover number of those devices by the
device sets one mission serves within the budget. Below S\* some capped device goes unserved
whatever the plan does; at S\* a covering schedule exists, but the plan, which looks one mission
ahead, can still crowd a device (critic A4, layout 25).

- *Planning level only.* Nothing is flown or drawn: each mission is priced by the predicate the mule
  plans with, at the mean SNR, deterministically, under the budget rule (the budget clause, and the
  energy clause when a capacity is configured), from the dock at time 0, deadlines left out (a
  capped device is exempt from its own).
- *The plan's stop family.* Each class's stops are S3a's at R_planar(c) on a fresh plan (every
  device new), with the hover rule applied to every device as if it were capped. `layout_s_star(...,
  hover=False)` prices S3a's stops alone, the family before the hover decision, which reproduces the
  design probe's table.
- *The trial's physics.* Every class is priced by the model the plan-mode mule builds for it
  (`FerryRuntime.plan_classes`), from the spec the driver would give the cell's mule, with the
  declared payload, else the measured θ and synthetic batch (the stub's unless `--theta-bytes` says
  otherwise: pass 18756 for the canonical model in real-model cells).
- *Its own reference layouts.* Layout k is drawn as a trial's is, from the seed `_u32(N, "s_star",
  k)`, so S does not move with the grid's base seed or trial count (T_nom's precedent).
- *Unservable devices.* A device that no class of the family serves alone within the budget, even at
  its best hover point (the plan's `unplannable`), is left out of the cover and reported, never
  counted as a missing mission. Without an energy capacity that is physics; with one it need not be
  (§18.2). Under `whole` the tool also leaves out a device whose whole stop no mission flies, which
  the plan labels `crowded` when it fits alone.
- *Exact, then greedy.* Up to `--exact-max-devices` (6) devices S\* is exact; above, it is a greedy
  upper bound (each mission, per class, the 2-OPT tour of the stops of the devices still uncovered,
  folded with the F member walk; the class serving the most taken). In U9's probe the bound equalled
  the exact S\* in 88–100 % of cases per family and was never more than 2 above it.
- *Decision 1's S.* Per family and budget, the ⌈0.9 n⌉-th smallest S\* over the n layouts (each
  layout's S\* counting its servable devices only); S (i) is the largest over the budgets given,
  never below 2; the cell's S is F's. The tool prints S + 1 as (ii) with the caveats it cannot vouch
  for: "for a pinned class" (partition drift, below), "at 30 s" (each budget given outside the
  45–90 s that was measured, `S_PLUS_1_MEASURED_S`), and "under whole admission".
- *What S + 1 gives, as measured* (each of the tool's 30 layouts at its own S\*+1, 3S missions,
  critic B4's loopback, 1 MB): no `unplannable` at 45, 60 or 90 s in any family, and no violation at
  all for F and FB+narrow there. A pinned class can still crowd a capped device at the stress
  budget, because S3a re-clusters every mission on the devices' buckets and deadlines, so a capped
  device's stop can drift from the tool's fresh partition to one that serves it alone but not beside
  another capped device: FB+medium crowded 7 times in 291 missions at 45 s, FB+wide twice in 360. At
  30 s every family crowds (F 20 times in 309 missions). The cell's S + 1 is at least a layout's
  S\*+1 only where S covers that layout: with the budgets 90 and 45 s F's S + 1 = 3 crowded 5 times
  in 270 missions at 45 s, all on layout 7, whose S\* (3) is above S.
- *What it gives today* (the default cell: N = 6, wide reference, jittery, 30 layouts). At 1 MB with
  the budgets 90 and 45 s: F S = 2, FB+wide 4, FB+medium 3, FB+narrow 3, every device servable at
  45 s; with 90, 60, 45 and 30 s: F 3, FB+wide 4, FB+medium 4, FB+narrow 5. At the measured payload
  (`--theta-bytes 18756`) F's S\* is 1 on every layout, so S = 2 by the floor. The pilots set S from
  the measured knee and stress budgets.
- *Limits.* It computes one mule over the whole layout (no mule-count option), which fits the
  one-mule pilots. A budget must be a finite number of seconds above 0: an infinite budget would
  admit a mission the energy clause prices at infinity, so for the energy clause alone give a large
  finite budget such as 1e9.

```bash
python -m experiments.analysis.age_cap_s_star --budgets "$KNEE_S" "$STRESS_S" --payload-bytes 1000000 --contact-band wide --regime jittery --json s_star.json
```

Flags: `--budgets` (required), `--N` (6), `--rrf` (60), `--regime` (`jittery`), `--layouts` (30),
`--contact-band` (`wide`), `--contact-band-classes`, `--backhaul-model` (`mission`; prices the
Pass-1 upload the budget includes), `--payload-bytes` (omit for measured), `--theta-bytes`,
`--ferry-physics` (JSON of physics overrides), `--no-realism` (the tight EX-4.0 cluster),
`--member-admission` (`subset`), `--families` (default F and FB+c for every class),
`--exact-max-devices` (6), `--layout-tag` (`s_star`) and `--layout-offset` (0; the design probe used
tag `t_nom`, offset 1000), `--json` (the full report).

### 18.8 Exp 4 driver and runner

The plan arms (`driver.PLAN_ARMS`) run only on the simulated clock and only when named with
`--arms`; the runner's default arm list stays the nine Phase 3 arms (`driver.DEFAULT_ARMS`), and the
driver runs both lists (`ARMS`). Labels are ASCII with no `__`, so a kept trace's directory holds
them whole.

| Arm | Plan fields over F's | Notes |
|---|---|---|
| `F` | `plan_mode = ferry`, `band_class_policy = search`, `flight_slot = committed`, `member_admission = subset` unless `--member-admission` says otherwise, the run's cap, lookahead, score and search settings | The plan search with the committed slot; the learned pair score flies as FQ (§19.7). |
| `FX` | `flight_slot = cross_heuristic` | §18.5. |
| `FB+wide`, `FB+medium`, `FB+narrow` | `band_class_policy = fixed:<class>`, and the mule's `contact_band` is the class (in its ferry settings, critic A8) | Path B+: the C1 ablation. |
| `F-cov` | `plan_score_params` gains `{"c_cov_per_device": 0, "c_link": 0}` over the run's | Cap-only service (decision 3); the κ sweep does not reach it. |
| `F-cap` | `age_cap_missions = None`, `age_cap_lookahead = 0` | No cap, whatever `--age-cap-missions` says. |
| `F-prio` | `miss_priority` off | Weights by age alone. Every other plan arm runs `miss_priority` on, whatever `--miss-priority` says, and the row records the arm's own value. |

Every plan arm gets the `trim` fallback, whatever `--replan-fallback` says (the scheduler refuses
`reorder` in plan mode), and T_nom, computed per cell unless `--t-nom-s` gives it, since the score
measures the mission against it; T_nom stays priced on wide for every arm, so the deadline unit is
one constant of the cell (critic C4). One consequence is documented, not corrected (the Phase 4
spec): an arm whose missions are shorter fits more of them into a deadline window, so F's windows
span more missions than a wide arm's (critic C4's estimate: F's narrow missions last about 55 s
against a T_nom of 172–250 s, so about 4× as many fit). An H or D arm in the same CSV flies
`--contact-band`.

| Flag | Default | Meaning |
|---|---|---|
| `--arms F FX FB+wide FB+medium FB+narrow F-cov F-cap F-prio` | the nine Phase 3 arms | The plan arms, with `--mission-clock sim`; the runner refuses one the driver cannot run (`check_arm`) before any trial. |
| `--member-admission {whole,subset}` | each arm's own: `subset` for the plan arms, `whole` for H1–H3, D1–D3 and D5 | Applies to every arm of the run; D4 always runs `whole`. Refused on the wall clock unless `whole`. |
| `--age-cap-missions S`, `--age-cap-lookahead L` | off, 0 | The plan arms' cap and lookahead (§18.2); F-cap runs without the cap. |
| `--plan-score-params JSON`, `--plan-search-params JSON` | {} | A JSON object of `PlanScoreParams` or `PlanSearchParams` fields, e.g. `'{"c_cov_per_device": 0.25, "c_energy": 0, "coverage_rank": "weighted"}'` for one cell of the κ sweep; unknown keys are refused. |
| `--base-seed` | 42 | The salt of every trial's seed (SHA-256 of the base seed, the cell and the trial index). A pilot takes its own, so its seeds are not the headline's (decision 7). |

None of the Phase 4 flags is a grid axis, so they move no seed (`--base-seed` does, on purpose). On
the wall clock the driver refuses each of them unless it keeps its default (`--member-admission
whole` counts as the default).

**Provenance.** No trial CSV column is added. In plan mode `ferry_params` gains all eight plan
fields as the mule config holds them, and `contact_band` reads `search` for an arm that searches the
classes (FB+c keeps its class); outside plan mode `ferry_params` gains `member_admission` only when
it is `subset`. A Phase 3 row's strings are unchanged. `miss_priority` records each arm's own value.
The trace scorer derives the same strings from a kept trace.

## 19. The flight clock's pair score, FerrySim and E3 (FeRRy Phase 5)

`flight_slot = "pair_q"` and `contact_policy = "chen_dqn"` are new values, and every default in this
section reproduces the recorded runs (Freeze §5g, Rule 1; §5l): no trace event gains a field at the
defaults, and every mule's per-role JSON gains six keys at null. The FQ arms fill Phase 4's flight
slot with a learned (band, next stop) score; E3 is a numpy port of Chen et al.'s DQN, a legacy-mode
whole scheduler that names each next stop in flight; `H1+L1` is H1 with H3's adaptive backhaul;
FerrySim (`experiments/ferrysim/`) trains and judges the learned fillings in process. The settings
are the user's decisions of 2026-10-01 with the resolutions R1–R29 recorded in Freeze §5l. Code:
[hermes/scheduler/policies/pair_slot.py](../hermes/scheduler/policies/pair_slot.py),
[hermes/scheduler/selector/pair_features.py](../hermes/scheduler/selector/pair_features.py),
[pair_q.py](../hermes/scheduler/selector/pair_q.py) and
[pair_replay.py](../hermes/scheduler/selector/pair_replay.py),
[hermes/scheduler/policies/chen_dqn.py](../hermes/scheduler/policies/chen_dqn.py) and
[next_stop.py](../hermes/scheduler/policies/next_stop.py),
[hermes/scheduler/plan/types.py](../hermes/scheduler/plan/types.py) (the pair types),
[hermes/scheduler/fl_scheduler.py](../hermes/scheduler/fl_scheduler.py) (`fits_after_service`),
[hermes/mule/mule_main.py](../hermes/mule/mule_main.py),
[hermes/processes/config.py](../hermes/processes/config.py),
[experiments/ferrysim/](../experiments/ferrysim/__init__.py).

**One decision of the pair slot** (a Pass-1 arrival at stop k; `MuleSupervisor._ferry_pair_at_arrival`):

1. The view (`PairView`, §19.2) is read at the arrival instant, after the transit is charged and
   before the contact; every read is pure.
2. The mask: `fits_pair(b, s)`, `FLScheduler.fits_after_service` bound by
   `pair_slot.bind_fits_pair`, is asked about every pair the view offers.
3. The scope guard (`scope_guard.assert_pairs_admitted`) checks every pair against the plan.
4. The scorer gives one finite number per pair (`check_pair_scores`), and the slot picks the masked
   argmax, ties to the lowest row; with no admitted pair it flies FX's pair (`mask_empty`).
5. The contact plan is built on the chosen class b at once. After the stop the chosen stop s is
   moved to the front (`cross_heuristic.moved_to_front`); the next departure runs the departure
   check on that order (the record notes `trimmed_next` when it does not keep it), then the beacon
   hook, then the slot's `next_stop`, which returns 0.

At takeoff the plan's first stop is flown (the decision still applies on arrival there), and Pass 2
flies b̄ in the queue's order (`FerryRuntime.contact_plan` refuses any other class there): the slot
decides nothing at either. Each mission's decisions are closed when it ends, on each of its three
exits (§19.2).

### 19.1 Switches and configuration fields

The six checkpoint fields are `CHECKPOINT_MULE_FIELDS` (`PAIR_CHECKPOINT_FIELDS` and
`POLICY_CHECKPOINT_FIELDS`), declared just before `plan_mode` (a test pins the plan fields as
`MuleConfig`'s last), inside `SIM_ONLY_MULE_FIELDS` and outside `FERRY_SPEC_FIELDS` and
`PLAN_MULE_FIELDS`: on the wall clock each must keep its default, and no Phase 3 or Phase 4
`ferry_params` string changes. A recorded per-role JSON loads with them at None.

| Symbol | Where | Default | Surface | Rationale |
|---|---|---|---|---|
| `flight_slot = pair_q` | `MuleConfig`, `PlanOptions` (`plan.types.FLIGHT_SLOT_PAIR_Q`) | `committed` | config field; the FQ arms | The learned (band, next stop) score in the flight slot (§19.2). Plan mode only; needs the three pair fields and `in_flight_response = replan` (R3); refused with a pinned band (`PlanOptions`, as `cross_heuristic` is: the pair chooses the class on arrival). |
| `pair_checkpoint` | `MuleConfig` | None | the FQ arms (`--pair-checkpoint TAG=PATH`) | The format-2 checkpoint the slot's score loads (§19.5), a non-empty string: repo-relative with `/` when the driver finds the file inside the repository, else absolute; a relative path is read under the repository root (R12). |
| `pair_checkpoint_sha256` | `MuleConfig` | None | the driver, from the verified manifest | The sha256 of the checkpoint's arrays, 64 lowercase hex digits: the mule flies no other arrays. |
| `pair_checkpoint_tag` | `MuleConfig` | None | the driver, from the arm (`driver.CHECKPOINT_TAGS`) | `main`, `hand`, `dwell`, `cov`, `g0` … `g99`: ASCII letters, digits, `_` and `-`, a letter or digit first, at most 28 characters. With the sha, the row's provenance. |
| `contact_policy = chen_dqn` | `MuleConfig` (`CONTACT_POLICY_CHEN_DQN`) | None | arm E3 | Chen et al.'s DQN as a whole scheduler (§19.6), on the simulated clock in legacy mode; needs a `contact_band`, `member_admission = whole` and the three policy fields. |
| `policy_checkpoint`, `policy_checkpoint_sha256`, `policy_checkpoint_tag` | `MuleConfig` | None | arm E3 (`--policy-checkpoint E3=PATH`) | As the pair fields, for E3's `chen_dqn` checkpoint; its tag is `e3`. |
| `chooses_next_stop` | a whole-scheduler policy's class attribute | absent (False) | E3 declares `True` | Read with `getattr`, and only `True` counts: the supervisor then asks the policy for the next stop at takeoff and at every Pass-1 departure (§19.6). No other policy, slot or selector declares it. |
| `pair_slot` | `MuleSupervisor` | None | the mule process (`build_pair_slot`) | The `PairQSlot` a `pair_q` mule flies; required with `pair_q`, refused otherwise. |
| `install_flight_slot(slot)` | `MuleSupervisor` | never called on a recorded path | FerrySim | Installs a `PairQSlot` in plan mode before the first mission; it flies mission for mission as the config path's slot does, while `mule_ready.flight_slot` still names the configured slot. |
| `pair_checkpoints`, `policy_checkpoints` | `Exp4Driver` | {} | the runner's checkpoint flags | Tag to path, simulated clock only. A tag no learned arm flies, or an empty path, is refused when the driver is built; the files are read when an arm that flies one is checked (§19.5). |
| `pair_columns` | `traces_scorer.score_trial`, `score_traces` | False | `--pair-columns` | The seven Phase 5 columns after the τ columns (§19.8). |

**Refused combinations** (each fires only on a value other than the recorded one):

- *The configuration* (`mule_config_errors`, through `_plan_config_errors` and
  `_learned_config_errors`). On the wall clock: any of the six fields set, and `contact_policy =
  chen_dqn` (E3 chooses each next stop on the simulated clock). On the simulated clock: `pair_q`
  outside plan mode (a plan field off its default in legacy mode); in plan mode `pair_q` without
  all three pair fields, with an `in_flight_response` other than `replan` (R3; with `abort` and a
  cap, critic A10's reason is the one given), or with a pinned band (`PlanOptions`); `chen_dqn` in
  plan mode (which takes no `contact_policy`); `chen_dqn` in legacy mode without a `contact_band`,
  with `member_admission = subset`, or without all three policy fields; a checkpoint field beside
  another slot or policy; a sha that is not 64 lowercase hex digits, a tag outside the pattern
  above, a path that is not a non-empty string. Where a switch cannot run at all (`pair_q` outside
  plan mode, `chen_dqn` in plan mode or on the wall clock), only that refusal is reported, not its
  requirements as well.
- *The supervisor* (`MuleSupervisor`, `MuleSupervisorError`). `pair_q` without a `pair_slot`; a
  `pair_slot` that is not a `PairQSlot`, or beside the `committed` or `cross_heuristic` slot, or on
  a legacy mule. `install_flight_slot` on a legacy mule, once a mission has started, under a pinned
  band, and for anything but a `PairQSlot` (TypeError).
- *The scheduler* (`fits_after_service`). Legacy mode, or no commit yet (`FLSchedulerError`); Pass
  2, a pose other than the stop's, `collected` outside the stop, a `rest` holding the stop's members
  (ValueError); a dwell that is not a finite number ≥ 0.
- *The mule process.* A checkpoint its loader refuses (a sha, kind, schema, class or band mismatch,
  no file, a path that is not an `.npz`), before the process binds anything: `CheckpointRefused`,
  exit 1, no port written.
- *The driver* (`Exp4Driver.check_arm`, before any trial and again before each trial). A learned
  arm on the wall clock; a learned arm whose tag has no checkpoint (no random-init arm); an FQ arm
  without `in_flight_response = replan` (R3); a checkpoint of the other kind, or one its mule would
  refuse (§19.5); `H1+L1` where it would fly as H1 (R27). No training state is checked here, so
  FerrySim's bootstrap checkpoints fly.
- *The runner.* A checkpoint a campaign may not fly (§19.5), and under `--require-trained` H2 or H3
  without `--selector-weights`, each a usage error (exit 2) before any trial.

### 19.2 The pair slot

**The view** (`plan.types.PairView`), read at the arrival at stop k:

| Field | What it holds |
|---|---|
| `arrival` | `FerryRuntime.arrival_view`: per class of the link, in link order, the members it would solicit at k (within R_planar(c), at or above the SNR floor) and the dwell of serving them, at the SNR observed now |
| `pose`, `clock_s` | the mule's pose (k's position) and the arrival instant |
| `observed_snr_db` | per class, the median realized SNR over k's members now (`observe(...).class_snr_db`) |
| `offsets_db` | per class, the median over k's members of realized minus mean SNR, each link differenced before the median (`FerryRuntime.class_offsets_db`; critic C6) |
| `previous_offsets_db`, `previous_age_s` | the previous Pass-1 arrival's offsets in this trial and how long ago it was, carried across stops and sorties and reset each trial; both None at the trial's first (critic A3) |
| `period_s` | P_c, the contact channel's interference period (its configuration, not its phases) |
| `budget_end`, `budget_s` | the mission budget's end and its length (None without one) |
| `t_ref_s` | T_nom, the plan's T |
| `energy_j`, `energy_ref_j` | the energy spent this sortie, and `l1_state`'s reference: the capacity, else P_hover × the budget (a 0 reference is stored as None) |
| `stops` | one `StopContext` per stop of the remainder, in its order, or home alone when none is left: the leg from k (`travel_s`), the dwell on b̄ at the mean SNR (`pred_dwell_s`), per class the median of the members' mean SNR (`pred_snr_db`), `capped` (some member capped), `exempt` (the stop is in `plan_protected`), the members' mean plan `age`, mean `on_time` rate and summed coverage `weight` |
| `demand`, `demand_weight`, `cap_s` | N, the plan's demand with any beacon insert (R9); the demand's summed coverage weights; the cap S |

**The pairs** are the covering classes (`PairView.covering`: those whose targets at k include every
target of b̄ at the arrival SNR; FX's candidate set, with b̄ always among them) times the
candidates, class-major in link order and then in the remainder's order; home stands alone. A pair
is (b, the index of s in the remainder, or None for home).

**The mask** (`FLScheduler.fits_after_service`; the user's decision 1 (a)). The state after serving
k on b is the clock plus b's dwell at the observed SNR and the energy plus P_hover × that dwell;
under route-level `deadline_bounds = delivery` its `deliver_by` is lowered to the own deadline of
each target of b the plan did not cap, dated by the plan, else the mule's record of an insert, else
the stop's deadline (conservatively, as if every target answered; R4). Then:

- *a stop pair* is admitted when `fold_remainder` of the rest, s first and the others in plan order,
  passes from that state on b̄'s model at the mean SNR (δ_obs = 0), under the arm's in-flight rule,
  with the plan's exempt stops protected: the whole rest of the flight still fits. It is the fold
  FX's `fits` runs, and the one the `replan` departure check runs next; under `abort` the check
  folds the next stop alone, so the mask would be the stricter, which is why `pair_q` needs `replan`
  (R3);
- *home*, offered only once the remainder is empty, is admitted when the end of the dwell plus the
  return leg and the Pass-1 upload is no later than `deliver_by` (route-level `delivery` only) and
  the budget's end, and the energy clause holds (with a capacity).

Without a budget nothing is gated. Neither pair tests the served stop's own deadline clause, since
the mule is there whichever pair it picks, so under `collection` and `delivery_per_stop` a slower
covering band can make that stop's own collection late unmasked (R4). The mask prices neither the
1 s listen window (charged only when a reply is missing) nor each target's session-start pricing:
the departure check after the stop catches both (in the final check's replay, 123 of 2,484
decisions were followed by a re-plan at the next departure). `bind_fits_pair` answers only for the
view's pairs, and the slot refuses a verdict that is not a bool.

**The pick.** The masked argmax of the scorer's numbers, ties to the lowest row (R5), so with one
class the ties follow the remainder's order. A scorer (`plan.types.PairScorer`) gives one finite
number per pair, higher better; the slot refuses a wrong count or a non-finite number before
anything is drawn. With a trainer attached (FerrySim only) the pick is `pair_q.behaviour_row`
instead: ε-greedy over the admitted pairs, around FX's pair in the reference phase. **On an empty
mask** the mule flies FX's pair, its fastest covering class (least dwell, then more targets, then
b̄, then the class index) with no reorder (index 0, or home), recorded `fallback = mask_empty`; its
effective mask is that one pair, which the learner stores and bootstraps through.

**The scope guard** (`scope_guard.assert_pairs_admitted`, `SelectorScopeViolation`): every pair's band
is a covering class; its next stop is a stop of the remainder (home only once that is empty; an
index that is not an int in range, a bool included, is refused); and every member of the served stop
and of the remainder is admitted this mission (the plan's served devices and the beacon hook's
inserts, never the pre-flight drops). A violation is a wiring bug and fails loudly.

**The scripted references** (`pair_slot.scripted_scorer(name)`; critic A2 and B5). Each ranks every
pair totally, so the tie rule never decides for it, and each flies inside the slot's rules (the
mask, the fallback, the reorder after the stop), so none is the fixed arm of its name:

| Name | Ranking | Where it parts from the arm of its name |
|---|---|---|
| `fx_pair` | FX's band order (least dwell, then more targets, then b̄, then the class index), then nearest first | Under `abort`, only where the mask refuses every stop on FX's band and admits another band's pair. Under `replan` the FX arm re-plans first wherever (FX's band, 0) is refused, and may drop stops, while the slot reorders to FX's nearest admitted stop and drops nothing, or on an empty mask flies the head of the same re-plan |
| `committed_pair` | b̄ first, then FX's band order, then the plan's order | Flies F's flight wherever (b̄, 0) is admitted; elsewhere it reorders, and on an empty mask flies FX's pair, which is F's only when FX's band is b̄ |
| `hyb` | FX's band order, then the plan's order (critic B5's HYB, FX's band with the planned order) | Flies a fixed HYB's flight wherever (FX's band, 0) is admitted or no pair is; elsewhere it reorders. No fixed HYB filling exists |
| `greedy_1` | the most targets at the arrival SNR, then the least dwell plus travel, then FX's tie-breaks | The one-step greedy rule under decision 4 (critic A2), the "most devices" band rule Phase 4 rejected: a reference to beat, not a filling |

**The records** (`mission_completed.pass_1_pairs`, `pair_q` missions only), one per Pass-1 stop
flown, in order, written at the arrival and closed when the mission ends; JSON-ready, with no wall
time:

| Key | Meaning |
|---|---|
| `t_s` | the arrival on the mission clock |
| `devices`, `committed` | the stop's members; b̄ |
| `band`, `next_index`, `next` | the pair flown: the class, the remainder's index (0 keeps the plan's order, null is home), and the next stop's members or `"home"` |
| `pairs`, `feasible`, `admitted_pairs` | how many pairs were offered and how many the mask admitted, and those as `[band, index]` in row order |
| `fallback` | `"mask_empty"` when no pair fitted, else null |
| `fx_band`, `fx_next`, `agrees_fx` | FX's pair by FX's own rules at this arrival, on the remainder as it stands and priced as the mask prices it (FX's band; the nearest stop whose pair on that band the mask admits, else 0; null for home), and whether the pair flown is it. That is FX's rule, not the FX arm's flight: under `replan` the arm re-plans first exactly where `[fx_band, 0]` is missing from `admitted_pairs` with stops left |
| `scorer`, `q`, `q_fx` | the scorer's name (`pair_v1` for the learned score, else the reference's); the scores of the pair flown and of FX's pair, to 6 places, for a scorer whose numbers are Q values, else null |
| `collected`, `w` | added at the close: the members collected CLEAN at the stop, in member order, and the raw L3 weight the merge gave each (under `agg:plain` n_i from the CLEAN report line; under the age-aware rules `device_weights` × `weight_mass`; 0 for an update the merge left out, and on the empty round) |
| `late` | the collected members whose stamp is strictly after their own deadline (the plan's, or the insert's) |
| `t_next_s` | the next Pass-1 arrival, or for the sortie's last decision the end of the Pass-1 upload (the landing on the empty round), so Δt_k = `t_next_s` − `t_s` |
| `terminal` | no later Pass-1 decision follows in the sortie: a `home` that a beacon insert follows is not terminal (critic B12) |
| `trimmed_next` | the departure check after this stop did not keep the order the pair set: it re-planned it under `replan`, or gave up the pass under `abort` |

The records close in `_ferry_result`, which every simulated-clock exit returns through (the empty
round, no DOWN, the normal path); the slot's `close_mission` is called at every close, with
`((), ())` when no decision was made. A beacon insert can fly ahead of the chosen stop, and `next`
then names a stop not flown next, with `trimmed_next` False; nothing on a recorded or FerrySim path
offers a beacon.

### 19.3 The pair features (`pair_v1`)

`selector.pair_features.pair_rows(view, schema)` gives one row per pair, in `view.pairs` order: on
the three-class link 36 columns with the phase block and 24 without (4C + 24 and 2C + 18 for C
classes). T is T_nom (`view.t_ref_s`), N the view's demand, b̄ the committed class, S the cap (1
without one), P_c the interference period. Each column depends on the band b alone (`band`), the
next stop s alone (`next`), both (`pair`) or neither (`state`: one value for every row of a view),
as a test pins.

| Column | Dep. | Value | Bounds |
|---|---|---|---|
| `band[c]`, one per class | band | 1 for the class b, else 0 | flag |
| `snr_here` | band | b's median realized SNR over k's members now, / 30 dB | — |
| `dwell_here` | band | b's dwell at k at that SNR, / T | ≥ 0 |
| `gain_here` | band | b's targets at k beyond b̄'s, / N | [0, 1] |
| `reach_here` | state | b̄'s targets at k, / N (R11) | [0, 1] |
| `travel` | next | the leg from k to s (to the dock for home), / T | ≥ 0 |
| `snr_next[c]`, one per class | next | s's mean SNR on class c, the median over its members, / 30 dB (0 for home) | — |
| `dwell_next` | next | s's predicted dwell on b̄ at the mean SNR, / T | ≥ 0 |
| `slack_next` | pair | sign(x)·log1p(abs(x) / T) with x = Deadline(s) − (now + `dwell_here` + travel + s's dwell), unclipped; 0 when `exempt_next`, and for home | — |
| `exempt_next` | next | 1 when no deadline clause can bind s: exempt (every member capped), or undated | flag |
| `age_next` | next | s's mean plan age, / S | ≥ 0 |
| `on_time_next` | next | s's mean on-time rate (`features._on_time_rate`: 0.5 for a device never seen) | [0, 1] |
| `members_next` | next | s's members, / N | [0, 1] |
| `capped_next` | next | 1 when some member of s is capped | flag |
| `home` | next | 1 for the home row | flag |
| `clock_left` | pair | (budget end − (now + `dwell_here` + travel)) / budget; 1 without a budget | [−1, 1] |
| `energy_left` | state | 1 − energy spent / the reference; 1 without one | [−1, 1] |
| `remainder_share` | state | the remainder's members, / N | [0, 1] |
| `least_slack` | state | the least slack over the remainder's dated, non-exempt stops, each priced after serving k on b̄ (the least `slack_next` on b̄'s rows); 0 when none | — |
| `weight_share` | state | the remainder's committed weight / the demand's (0 when that is 0) | [0, 1] |
| `offset[c]`, one per class | state | the class's offset at k now, / 10 dB | — |
| `prev_offset[c]`, one per class | state | the previous Pass-1 arrival's offsets this trial, / 10 dB (0 at the trial's first) | — |
| `has_prev` | state | 1 once a previous Pass-1 arrival was observed this trial | flag |
| `prev_age` | state | its age, capped at 4·P_c, / P_c (0 at the first) | [0, 4] |
| `prev_sin`, `prev_cos` | state | sin and cos of 2π·age / P_c, the age uncapped (0 at the first; R11) | [−1, 1] |
| `arrival_sin`, `arrival_cos` | pair | sin and cos of 2π·(`dwell_here` + travel) / P_c | [−1, 1] |

The phase block runs from `offset` to `arrival_cos` (design D-D (b); decision 6 (a)). What changed
from the design's rows (the Phase 5 spec, other choices 4): feature 7 is s's mean SNR per class,
since s's band is chosen at s's own arrival; the slack is log-scaled and unclipped, since the pilots'
slack at collection was at least 582 s with a median of 1,143 s, which the design's [−1, 3] clip
saturated (critic A10 (iv)), with `exempt_next` beside it; feature 11, the value of s, is dropped
(critic C5: members / N under equal shards, noise under the stub's draws); the offsets are
differenced per link (C6); the previous reading and its age are added (A3). R11 keeps three columns
beyond the spec's list: `reach_here`, so that a row carries its own decision's reward level, which a
γ > 0 target bootstraps from; and `prev_sin` and `prev_cos`, the phase between the two readings,
which the capped age loses past 4·P_c (U1 measured that on jit-n12-120 at 1.3 % of decisions at
P_c = 60 s, 16.9 % at 45 s and 24.7 % at 30 s, each a sortie's first). Only `gain_here` and
`exempt_next` may be constant on a FerrySim sample (`SPARSE_COLUMNS`: a covering class that reaches
more than b̄ is rare, and a stop whose members are all capped appears only from mission S on); every
value is finite. A row reads no channel, clock or draw: the view is taken at the arrival from what
the mule has observed by then. The phase block is tested at P_c = 30, 45 and 60 s.

**The schema** (`PairFeatureSchema(classes, phase=True)`; the constants `SNR_SCALE_DB` 30,
`OFFSET_SCALE_DB` 10, `PREVIOUS_AGE_CAP_PERIODS` 4 and `SHARE_CLIP` 1): its JSON (`version`
`pair_v1`, `dim`, `classes`, `phase`, `columns`, `constants`) is the checkpoint header's `schema`,
which the loader compares whole, so a checkpoint is read only under its own columns and scales, over
exactly the link's classes in link order. Changing a column or a constant is a new schema version.

**The learned score** (`LearnedPairScorer(net, schema)`): `score(view, mask=...)` is the online
network's Q of `pair_rows(view)`, the whole candidate set in one call (equal rows get one Q), in
`view.pairs` order; it does not read the mask; `q_values` is True and `name` is `pair_v1`. The
network is held, not copied, so a trainer's updates are scored at once. `load_pair_scorer(path,
expect_sha256=, classes=, phase=None)` verifies the checkpoint whole, then checks its kind
(`pair_q`), its schema (this module's `pair_v1` over exactly `classes`; the phase flag only when one
is asked, else the checkpoint's own) and the sha. Every refusal of what `path` names is a
`CheckpointError` (a ValueError), no file and a path that is not an `.npz` included; arguments that
no valid config holds raise TypeError or ValueError before any file is read. `build_pair_slot(...)`
is the `PairQSlot` around it, the slot's module imported only there (R8).

### 19.4 The learner (`selector/pair_q.py`, `selector/pair_replay.py`)

A masked pointer double DQN in numpy: one scalar Q per row from shared weights (the legacy DDQN's
pointer form), so the candidate set can grow and shrink with the remainder. These are new modules:
the legacy `DDQN` refuses γ = 0, trains by SGD on a squared loss and scores one stored next row, and
the H2 golden pins it.

| Symbol | Where | Default | Rationale |
|---|---|---|---|
| `hidden`, `activation` | `PairQConfig` | (64, 64), tanh | Other choices 3; tanh is the only activation taken. W ~ N(0, 1/fan_in) and b = 0, float64, from `numpy.random.default_rng(seed)`; the target network starts as a copy. |
| `gamma` | `PairQConfig` | 0.9 | γ ∈ [0, 1]. Study 5.5 sweeps {0, 0.25, 0.5, 0.75, 0.9, 0.99}; a trainer always states its own. |
| `lr`, `adam_beta1`, `adam_beta2`, `adam_eps` | `PairQConfig` | 1e-3, 0.9, 0.999, 1e-8 | Adam with bias correction; β1, β2 and ε are Kingma and Ba's defaults, which the spec does not set. |
| `huber_delta` | `PairQConfig` | 1.0 | The Huber loss of Q_online(s, a) − y, averaged over the batch, with y held fixed. |
| `grad_clip` | `PairQConfig` | 10.0 | The gradient's global norm is clipped to it before Adam's step. |
| `target_sync` | `PairQConfig` | 500 | A hard copy of the online weights every 500 updates. |
| `n_step` | `PairQConfig` | 1 | The only value taken. |
| `batch`, `replay_capacity`, `warmup_transitions` | `LearnerSettings` | 64, 50,000, 1,000 | No update until the replay holds 1,000 transitions; then one update per decision on a batch of 64 (`PairQLearner.observe`). |
| `reference_episodes`, `epsilon_start`, `epsilon_end`, `decay_fraction` | `BehaviourSchedule` | 500, 0.3, 0.05, 0.5 | Critic C4: episodes 0–499 fly ε-greedy around FX's pair at 0.3; from episode 500, ε-greedy on Q, ε falling linearly from 0.3 to 0.05 at half the run, and 0.05 after. ε never starts at 1.0 for the pair score (E3's schedule has no reference phase, §19.6). |
| `LEARNER_REVISION` | `pair_q` | 0 | Written into every checkpoint's header (R2). The one learner revision other choices 12 allows bumps it to 1, and the report refuses a sweep trained by two revisions or by one past 1. Settings changed by command-line flags are not counted; the training-spec check refuses a sweep that mixes them. |

- *The target* is y = r + γ · Q_target(s′, a*), where a* is the argmax of Q_online over the next
  decision's admitted rows, ties to the lowest row (double DQN, van Hasselt et al.). Only admitted
  rows are ever forwarded, so a masked row cannot enter a target; γ = 0 and a done transition give
  y = r exactly. A transition is done at the sortie's last decision: the flight Q's horizon is the
  sortie (memo L264).
- *A transition* (`PairTransition(x, reward, done, next_rows, next_mask)`) is the row of the pair
  taken, its reward and, unless done, every candidate row of the next decision with the mask it was
  taken among (on an empty mask, FX's row alone), so at least one next row is admitted. A batch
  concatenates the next rows, each marked with its transition. The replay (`PairReplay(capacity,
  seed=)`) is a FIFO ring sampled uniformly without replacement from `random.Random(seed)`.
- *Choosing a row:* `masked_argmax` (ties to the lowest row; an empty mask raises) and, while
  training, `behaviour_row`: with probability ε a uniform admitted row, else the reference when it
  is given and admitted, else the masked argmax; exactly two draws per call, every argument checked
  before them. Equal rows get one Q, bit for bit (`q` forwards each distinct row once), so the
  lowest-row rule decides between them; the batched target's single pass agrees to rounding.
- *Determinism:* every draw comes from a seeded stream, and with `OPENBLAS_NUM_THREADS=1` a training
  run is reproducible to the byte (T3).

### 19.5 Checkpoints

**Format 2** (`PairQNet.save`, `.load`). An `.npz` of `format_version` (2), `header` (the canonical
JSON, as bytes, of `format`, `kind` (`pair_q` or `chen_dqn`), `purpose` (`bootstrap` or `trained`),
`learner_revision`, `network` (the row width and every `PairQConfig` setting, γ included), `schema`
and `classes`) and the online weights (`layer<i>_W`, `layer<i>_b`, float64), with a JSON manifest
beside it (the same name, `.json`; sorted keys, ASCII, no key naming a wall time). The sha256 is
taken over every array in the file, the header included (each array's name, little-endian dtype,
shape and bytes, in name order), so the sha a config names binds the weights, what they read, the
kind, the purpose and the learner's revision (R2): the same weights saved as bootstrap, as trained,
and under another revision have three shas. The legacy `DDQN.load` refuses format 2 by its own
check, and archives are read with numpy's object loading off.

**The manifest** holds exactly 21 keys: the header's seven (`HEADER_KEYS`, which the sha binds); the
writer's four (`sha256`; `gamma`, checked against the header's network; `numpy`; `blas`); and the
trainer's ten (`PROVENANCE_KEYS`, outside the sha): `reward` (the reward spec), `training` (the
training spec and its outcome), `seeds`, `cell_family` and `cell_family_sha256`, `trainer_commit`
and `dirty`, `episodes_trained` (the episodes the kept weights trained on), `validation` (the curve:
one entry per validation, with its score, each cell's mean and the updates taken by then), and
`held_out`, None until the evaluator fills it (`pair_q.record_held_out`) and then holding at least
`episodes` (an int ≥ 1) and `return_mean` (a finite number), the mean undiscounted held-out return
(`HELD_OUT_KEYS`; R10). `held_out` is written after the save, so it stays outside the sha, and
`episodes_trained`, `held_out` and `dirty` catch mistakes, not hand edits.

**The loader refuses** (`pair_q.CheckpointError`, a ValueError) a missing or malformed manifest; any
format but 2 (the legacy DDQN's format 1 included); a broken header (one written before R2 lacks the
purpose and the revision); arrays that disagree with the manifest's sha or its header (a relabelled
purpose or revision included); extra or missing arrays, another dtype or shape, non-finite weights;
and a sha, kind, class tuple (order counts) or schema other than expected. It checks no training
state, so a bootstrap checkpoint loads, and it loads a checkpoint of another learner revision.
`read_manifest` checks the manifest's form only: judge a checkpoint by `verify_checkpoint`'s
manifest.

**Where a checkpoint is checked:**

- *The runner*, a campaign's entry (critic B9). For every checkpoint given, whichever arms run
  (R15): `verify_checkpoint`, the flag's kind, then `campaign_refusals`: a purpose other than
  `trained`, `episodes_trained` below 1, no held-out score, or a dirty tree without
  `--allow-dirty-checkpoint`. Then as its tag (R24; `ferrysim.checkpoints.tag_refusals`): `gX`
  needs γ = X/100; `hand` needs the F·hand reward, and every other pair tag the derived reward, at
  decision 4 (a)'s weights (c_t 0.1, c_cov 1) under `gX`, `dwell` and `cov` (`main` takes any
  weights its manifest records; R28); `e3` needs E3's bytes reward; and the kept weights must have
  taken an update (the validation entry at `episodes_trained` records more than 0, and a manifest
  without that record is refused). Then, once the driver is built, each pair checkpoint against the
  plan its arm flies under the run's flags (R23: the manifest's `training.spec.plan_score_params`,
  none recorded being the default plan, against `Exp4Driver.plan_settings(arm)`, with
  `PlanScoreParams`' defaults filled in on both sides). Each refusal is a usage error, exit 2,
  before any trial.
- *The driver* (`Exp4Driver.checkpoint_settings`, `check_arm`). It reads each path once against the
  working directory, verifies it, keeps the verified sha for the run, and writes the path into the
  mule's config repo-relative with `/` when the file lies under the repository root, else as its
  resolved absolute path (R12). Before each trial it loads the checkpoint exactly as the mule will
  (`load_pair_scorer` over the link's classes, or `load_e3_network` on the contact band, from
  `processes.mule.checkpoint_path`), so a file rewritten since the first check is refused before
  anything is spawned; a checkpoint re-saved at the same path is therefore refused at the next
  trial (use a new path). No training state is checked.
- *The mule* (`processes/mule.py`). It reads a relative path under `REPO_ROOT`, whatever its working
  directory, and loads the checkpoint against the config's sha before it binds anything; any
  refusal is `CheckpointRefused`, and the process exits 1 without writing its port.

**The layout** (decision 9 (a); `ferrysim.checkpoints.checkpoint_path`):
`results/exp5/checkpoints/<study>/<tag>/g<γ>_s<seed>.npz`, with the manifest beside each; a study
and a tag are plain path components. The layout has no cell family, so each family gets a study of
its own, and a run never replaces another family's checkpoint, even with `--overwrite` (R19).
Nothing is committed during the build: each study's final checkpoints are committed with the user's
consent, after a LICENSE file is added (decision 9: the README names MIT, and the repository has no
LICENSE file yet). Tests write their checkpoints under `tmp_path`.

**Trained weights for the H arms.** `--require-trained` (the runner; decision 8 (a)) refuses H2 and
H3, both in the default arm list, without `--selector-weights`, as Exp 3's `--require-trained-a4`
does; it checks only that weights are given (R15). The learned arms need no such flag: the runner
always refuses an untrained checkpoint.

### 19.6 E3 (`contact_policy = chen_dqn`)

`policies.chen_dqn.ChenDQNPolicy` (decision 7 (a); after Chen et al., GLOBECOM Workshops 2023) is a
legacy-mode whole scheduler, as D1–D5 are, that chooses each next stop in flight.

- *Declarations:* `name = "chen_dqn"`, `in_flight_check = "none"`, `admits_member_subsets = False`,
  `chooses_next_stop = True`.
- *Before takeoff* (`admit_and_order`): every S3a contact, nearest the takeoff pose first, then by
  position, then by members; the order only names the rows and decides ties. It reads no budget,
  deadline or device state, so no policy drop is reported.
- *In flight, no check:* the departure check keeps the remainder as it stands (`RULE_NONE`), never
  re-planning or trimming it, so E3's Pass 1 is the same under `abort` and `replan`. E3 flies the
  driver's configured in-flight response, as D1–D5 do (R27), and its Pass 2 follows it as every
  arm's does.
- *The hook* (`MuleSupervisor._ferry_e3_next_stop`; the protocol of `policies/next_stop.py`): at
  takeoff and at every Pass-1 departure, after the departure check and the beacon hook, never in
  Pass 2, the supervisor calls `next_stop(remainder, state, view=, admissible=, pass_kind=,
  after_stop=)`. `admissible(i)` is Chen's safety controller: S3b's single-contact predicate under
  `RULE_BUDGET` from the departure's state (transit, dwell, the return leg and the Pass-1 upload
  within the budget's end, and the energy clause; no deadline). The policy returns the admissible
  row with the highest online Q, ties to the lowest row, and None exactly when no stop is admissible
  (`checked_choice`), which ends the pass: the mule flies home, and the stops left go to
  `pass_1_e3_unvisited`, never widened. Beacon inserts are offered to it as any stop is.
- *The observation* (`FerryRuntime.e3_observation`, an `E3View`): per candidate stop, on the contact
  band, from the departure pose; N is the slice with every planned or inserted device outside it
  (R9).

| Column (`e3_v1`) | Value |
|---|---|
| `remaining` | the share of the stop's updates not collected this mission: 1 on every row (declared constant) |
| `snr` | the median over the members of the SNR now from the pose, realized within reach and the mean ("radio map") SNR beyond (critic B7 iv), / 10 dB |
| `reachable` | the share of members within reach and at or above the floor: 0 on most rows (declared) |
| `dx`, `dy`, `distance` | the stop less the pose, and the leg on the flight model's metric, / 100 m |
| `members` | the stop's members, / N |
| `return_energy` | the energy of the return from the stop to the dock, / the energy reference (0 without one) |
| `energy_left` | 1 − energy spent / the reference (1 without one) |
| `time_left` | (budget end − clock) / budget (1 without one) |

The schema (`e3_v1`: the 10 columns, the two declared constant, `length_scale_m` 100 and
`snr_scale_db` 10) and the contact band, the checkpoint's one class, are bound by the sha. **Declared
deviations from Chen** (decision 7 (a)): stops rather than grid moves (Chen's collection status q,
0 for every candidate at a departure, is left out); one agent (no other UAVs, no QMIX); no
model-aided learning; `remaining` and `reachable` constant or near zero, kept for fidelity; the pair
learner's masked double DQN rather than Chen's settings (his Adam learning rate is 5e-4); dx and dy
the stop less the pose, Chen's sign being the other; trained at K = 1 and flown per mule at K = 3
in Study 5.3, on slices within the trained sizes (critic B7 v). E3 is rewarded in bytes, |C_k| / N.
It acts on the live network's online Q at each call, never on the target copy. Its training defaults
are the pair learner's (lr 1e-3, ε from 0.3 to 0.05, no reference phase), with `--gamma` required;
Chen's lr 5e-4 and ε from 1.0 are the user's choice at the E3 training go-ahead (R16), and the
manifest records which. Checkpoints: `save_e3_checkpoint` and `load_e3_network` (kind `chen_dqn`,
schema `e3_v1`, classes `[band]`); the mule builds `ChenDQNPolicy.from_checkpoint`, which refuses a
sha, kind, schema or band mismatch; a bootstrap checkpoint loads.

### 19.7 Arms, tags and the runner

`driver.PHASE_5_ARMS` run only when named with `--arms`, on the simulated clock; the runner's default
arm list stays `DEFAULT_ARMS`, and the driver runs `ARMS` = `DEFAULT_ARMS` + `PLAN_ARMS` +
`PHASE_5_ARMS`. Labels are ASCII with no `__`, so a kept trace's directory holds them whole.

| Arm | Tag | Settings over its base | Notes |
|---|---|---|---|
| `FQ` | `main` | F's, with `flight_slot = pair_q` | The plan's F (the paper may call FQ "F"); Phase 4's F keeps the committed slot. `main` flies any derived-reward weights its manifest records (R28). |
| `FQ-g0`, `FQ-g25`, `FQ-g50`, `FQ-g75`, `FQ-g90`, `FQ-g99` | `g0` … `g99` | as FQ | Study 5.5's γ sweep: tag `gX` flies γ = X/100 at decision 4 (a)'s weights (R24). |
| `FQ-hand` | `hand` | as FQ | The score trained on F·hand (decision 4): only an F·hand checkpoint, and an F·hand checkpoint only here (R24). |
| `FQ-dwell` | `dwell` | FQ, its `plan_score_params` gaining `{"dwell_in_delta": false}` (`FQ_DWELL_SCORE`) | Study 5.7's dwell ablation, with a checkpoint trained on that plan (R23). |
| `FQ-cov` | `cov` | FQ, its `plan_score_params` gaining F-cov's `{"c_cov_per_device": 0, "c_link": 0}` | Study 5.7's coverage ablation, likewise. |
| `E3` | `e3` | `contact_policy = chen_dqn`, whole stops, the run's in-flight response | §19.6. |
| `H1+L1` | — | H1 with `backhaul_policy = adaptive` (simulated clock), or `backhaul_plan(adaptive=True)` (`--l1-channel`) | Decision 8 (a): the adaptive backhaul's reference once H2 and H3 leave Exp 5. Refused without `--l1-channel` or `--backhaul-model seconds` (R27); member subsets as H1. |

An FQ arm is a plan arm (`is_plan_arm`) and gets F's settings wherever F has them: the `trim`
fallback, `subset` admission by default, the miss priority on, T_nom per cell and the pre-trial
check; a test asserts that each FQ arm's mule config equals F's but for `flight_slot`, the
checkpoint fields and its ablation's own field (critic A5). It also needs `--in-flight-response
replan` (R3). FQ-dwell, FQ-cov and FQ-hand train their checkpoints only if Study 5.5 keeps the
learned score (critic C2). It did not (6 Oct 2026), so Study 5.7's plan-term ablations fly FX:
`FX-dwell` and `FX-cov` (added 7 Oct 2026) are FX's slot with FQ-dwell's and FQ-cov's score
changes, addendum plan arms (`ADDENDUM_PLAN_ARMS`) with F's settings wherever F has them.

| Flag | Default | Meaning |
|---|---|---|
| `--arms FQ FQ-hand FQ-dwell FQ-cov FQ-g0 ... FQ-g99 E3 H1+L1` | the nine Phase 3 arms | The Phase 5 arms, with `--mission-clock sim`; a learned arm needs its tag's checkpoint. |
| `--pair-checkpoint TAG=PATH` | none | Repeatable; tags `main`, `hand`, `dwell`, `cov`, `g0`, `g25`, `g50`, `g75`, `g90`, `g99`. A tag no learned arm flies, a tag given twice, or an empty path is a usage error. |
| `--policy-checkpoint E3=PATH` | none | E3's checkpoint (`e3=PATH` too). |
| `--allow-dirty-checkpoint` | off | Fly a checkpoint trained from a dirty tree, for development and tests, not a campaign; it lifts only the dirty refusal. |
| `--require-trained` | off | Refuse H2 and H3 without `--selector-weights` (§19.5). |

None of these flags is a grid axis, so they move no seed, and each setting needs a CSV of its own:
the runner skips (cell, arm, trial) keys already in a file. The pair learner's module is imported
only when a checkpoint flag is given.

**Provenance.** No trial CSV column is added, and no path appears in a row. For a `pair_q` mule
`ferry_params` gains `pair_tag` and `pair_sha256` (`plan_ferry_params`); for E3 `policy_params` is
`{"policy_sha256": ..., "policy_tag": ...}` as sorted JSON (`learned_policy_params`); every other row
reads as before. The trace scorer derives the same strings from a kept trace's per-role JSON with the
same two functions. The sha finds the manifest, which holds γ, the reward and the rest.

### 19.8 Trace fields and the scorer's pair columns

Additive, on the simulated clock and only on the mules that fly a learned filling; at the defaults no
trace event gains a field.

- **Per-role JSON** (either clock): every mule's carries the six checkpoint fields, null unless it
  flies a learned filling.
- **`mule_ready.pair`** (a `pair_q` mule) and **`mule_ready.policy_checkpoint`** (E3): the verified
  manifest's provenance (`pair_q.manifest_provenance`: `sha256`, `kind`, `purpose`, `classes`,
  `gamma`, `reward`, `seeds`, `episodes_trained`, `cell_family`, `cell_family_sha256`,
  `learner_revision` and `schema`, the schema's version) and the config's `tag`; no path. A mule
  whose slot FerrySim installed keeps the configured FX slot's `flight_slot` and has no `pair`.
- **`mission_completed.pass_1_pairs`** (a `pair_q` mission that flew a Pass-1 stop): the closed
  decision records (§19.2), left out when empty.
- **`mission_completed.pass_1_e3`** (E3): one entry per call, with `t_s`, `after_stop` (false at
  takeoff), `stops` (each stop's devices), `admissible` (one bool per stop), `next_index` and `next`
  (`"home"` for None). E3's N is not recorded per call; a reader recomputes it as the slice with the
  planned and inserted devices. **`pass_1_e3_unvisited`**: each stop left when nothing was
  admissible, with `position`, `devices`, `deadline_ts` and `"widened": false`. Each is left out
  when empty.
- **The consumer** (`experiments/exp4/events_consumer.py`): `MissionRecord.pair_decisions`
  (`PairDecision`, field for field, `next` as `next_devices`, with `mask_empty` and `reorders`),
  `e3_calls` (`E3Call`) and `e3_unvisited` (each stop's members), None where a mission has no such
  field. Each field is read only in the JSON form the mule writes it (a number is a finite int or
  float, never a bool or a string; an index or count an int ≥ 0; a flag a bool; a name a string; an
  id list a list of strings); anything else reads as None or empty, FX's pair is read whole, and a
  record whose choice (`band` and `next_index`) cannot be read is skipped.

**The pair columns** (`traces_scorer.PHASE_5_COLUMNS`) appear only with `pair_columns`
(`--pair-columns`), after the τ columns; without it the row is the Phase 4 one, byte for byte, and
the trial CSV's header is unchanged either way.

| Column | What it holds |
|---|---|
| `pair_decisions` | the decisions, pooled over every mission and mule |
| `pair_feasible_mean` | the mean number of pairs the mask admitted |
| `pair_mask_empty` | a count: the decisions that found no admissible pair and fell back on FX's pair (the share is this over `pair_decisions`) |
| `pair_fx_agree_share` | the share whose chosen pair was FX's by FX's own rule at that arrival; an empty-mask decision counts as agreeing |
| `pair_band_off_bbar_share` | the share served on a class other than b̄ |
| `pair_reorder_share` | the share whose pair chose a stop other than the remainder's head |
| `e3_unvisited_mean` | the mean over the trial's missions of the stops E3's pass left unvisited: stops, not devices; a mission without the field left none |

A trial flew the pair slot when its mule config names `pair_q` or any of its missions recorded a
decision (FerrySim installs its slots on FX's configuration, so its traces name FX's slot); with no
decision the two counts are 0 and the means and shares blank, and on any other trial every pair
column is blank. `e3_unvisited_mean` applies when the config names `chen_dqn` or a mission recorded
E3's calls, and is blank otherwise. The agreement and re-order shares count choices, not flights
(R21): the band is flown at once, but the stop flown next can differ when `trimmed_next` is set or
a beacon stop is inserted ahead of the chosen one. They are reported diagnostics, and no Study 5.5
step reads them. `mule_ready.pair` is not parsed: provenance comes from the per-role JSON, as the
driver's does.

```bash
python -m experiments.analysis.traces_scorer --traces results/exp5/s55/stack_120_traces --pair-columns --csv results/exp5/s55/stack_120_scored.csv
```

### 19.9 FerrySim (`experiments/ferrysim/`)

FerrySim is the stack's own trial, run in one process (decision 2 (a)): `Exp4Driver.run_trial`, its
per-role JSON and the real cluster, mule and device services, on synchronous in-process links and a
virtual wall clock, simulated time being the mule's own mission clock. `inprocess.py` is a copy of
UG4's in-process orchestrator (`tests/golden/_build_p3_sim.py`) with the helpers it takes from
`tests` (`experiments` never imports `tests`; critic C7), one mule per trial, and two hooks: `on_mule`
callables (FerrySim installs an episode's pair slot there) and the device model. It lives in
`experiments/`, not `hermes/scheduler/selector/`: `hermes/` may not import `experiments/`, and the
scheduler may not import `hermes.l1`, while FerrySim needs both.

**What is stood in for** (R25). Beyond the links and the clock: the devices' local training, by the
`equal` device model (`inprocess.equal_shard_trainer`: the same noisy update with a constant example
count n_i = 10 and constant scores, so every update weighs the same in the merge) or `stub`, the
stack's own stub trainer (n_i drawn afresh in [4, 15] at every training call), which the parity
tests use; and the devices' service loops, which do not run, so device traces hold only
`device_ready`. A FerrySim row (`EpisodeResult.row`, and the trace scorer's row of a kept trace)
therefore has `coverage` 0.0, `participation_entropy` 0.0 and `jains_fairness` 1.0, and a
`mission_duration_s_mean` of the harness clock: harness artifacts in every FerrySim trial, which no
study reads, since FerrySim reads flights from the mule's own records. *Parity:* with the stub,
FerrySim equals UG5's oracles in all nine parts; stub FX and FQ trials through the real orchestrator
equal FerrySim's runs on every mule and cluster event, wall stamps masked, and on the row bar
`mission_duration_s_mean` and the three serve columns (critic B8); FQ's install path equals its
config path on every mission. A process runs one episode at a time (the patches are process-wide);
parallel runs use spawned workers, each with `OPENBLAS_NUM_THREADS=1`, results in task order.

**The cells** (decision 3 (a); `cells.CELLS` and `STUDY_5_6_CELLS`). Every cell is Phase 4's pilot
configuration, as UG5's trials fly it: the simulated clock, realism, wide as the reference class, the
`t_nom` deadline unit, `replan` with the `trim` fallback, `agg:cutoff`, the `channel` reliability
source, 1 MB per direction, 4 missions, `rf_range_m` 60, the jittery network regime, and S = 2, the
S\* tool's S at each size's two budgets.

| Cell | Family | Role | N | Budget | Contact channel |
|---|---|---|---|---|---|
| `jit-n6-75` | jittery | control, where looking ahead cannot matter | 6 | 75 s, the stack's stress budget | jittery, P_c 60 s |
| `jit-n6-150` | jittery | control | 6 | 150 s, the stack's knee | jittery, P_c 60 s |
| `jit-n12-90` | jittery | decision-rich: Study 5.5 | 12 | 90 s, the stack's stress budget | jittery, P_c 60 s |
| `jit-n12-180` | jittery | decision-rich: Study 5.5 | 12 | 180 s, the stack's knee | jittery, P_c 60 s |
| `cln-n12-90` | clean | negative control (critic C3) | 12 | 90 s, stress | clean |
| `cln-n12-180` | clean | negative control | 12 | 180 s, knee | clean |
| `jit-n12-90-q`, `jit-n12-90-h` | `jittery56` | Study 5.6: lag/P_c a quarter, a half | 12 | 90 s, stress | jittery, P_c 108 s and 54 s |
| `jit-n12-180-q`, `jit-n12-180-h` | `jittery56` | Study 5.6 | 12 | 180 s, knee | jittery, P_c 136 s and 68 s |

**Re-pinned 5 Oct 2026** (`scripts/exp5/repin.py`; Exp 5 Reproducibility Guide §5.3). The budgets
are the stack's knee and stress pilots (`scripts/exp5/params.toml` `[pilot_outputs]`), so the learned
score trains on the budgets the stack flies (critic B6). Before that date N = 6 flew Phase 4's priors
(45 and 90 s) and N = 12 stand-ins (120 and 180 s), and records from then name the cells `jit-n6-45`,
`jit-n6-90`, `jit-n12-120`, `cln-n12-120`, `jit-n12-120-q` and `jit-n12-120-h` (P_c 104 and 52 s);
the 180 s cells kept their names. Study 5.6's periods (R22) are 4 × and 2 × `STUDY_5_6_LAGS_S` (27 s
at 90 s, 34 s at 180 s): critic A3's lag from one Pass-1 arrival to the next in a sortie
(`cells.arrival_lags`), FX's median over every lag of the first 200 episodes of the matching Study 5.5
cell's validation stream at the default P_c, rounded to the nearest second (`evaluate.fx_lag_median`;
the re-pin measured 26.94 s over 672 lags and 34.11 s over 397; at the stand-ins the fix round had
measured 26.42 s over 1,397 lags at 120 s). The integers are a pre-registration convention, not a
measurement to the second. The ratio check's bounds come from the same sample (a 24-episode pooled
median's 0.5 and 99.5 % points over the full median, widened by 5 % and rounded outward): 0.83–1.26
at 90 s and 0.67–1.48 at 180 s, so the quarter and half cells' ranges stay apart, if narrowly at
180 s. The half cells follow the rule though their 54 s and 68 s lie near 60 s. The 5.6 cells stay
out of `CELLS`, the headroom report's default; their control is the clean N = 12 cells
(`STUDY_5_6_CONTROL_CELLS`). A later re-pin moves them, the four P_c constants, the hashes and the
test literals together, on the same sample and statistic.

**Families**, one score per contact regime: `jittery`, the four jittery cells (sha256
`deaa4e08…052e`; `32b5cb6b…3e91` before the re-pin); `clean`, the two clean cells (`97d119d6…f98f`;
`76955c9b…5902` before); and `jittery56`, the jittery cells and Study 5.6's four (`affb41aa…2a47`;
`0079de11…ac66` before); and, since the Exp 5 addendum, `scale` (§20.3, `d410f0d1…70ad`), which
moves none of them and which the re-pin left alone. A manifest records its family and the hash
(`cells.family_sha256`). Study 5.5 is read on `STUDY_5_5_CELLS` (jit-n12-90 and jit-n12-180),
whichever family trained the score; which family the jittery score practises on is the user's
choice before the 5.5 sweep (R29).

**Seed streams** (other choices 8; critic B14). An episode is one trial of a cell, its trial seed
drawn from `ferrysim-train-<seed>` (training run `<seed>`), `ferrysim-val` (validation, the
headroom report and ε) or `ferrysim-heldout` (Study 5.5's held-out judgement, shared by every
checkpoint and reference: common random numbers). A seed is 32 bits: its top two bits are the
stream kind (1 train, 2 val, 3 held-out) and the other 30 a SHA-256 of the stream, the cell and the
episode index. So the streams are disjoint by construction (`check_disjoint` checks it), a stream's
episodes are distinct (a repeat is skipped), and a stream never moves when another is extended. A
training run draws each episode's cell uniformly from its family by a keyed draw
(`cells.train_episode`), so every γ of one seed meets the same episodes. The 5.6 cells' streams are
keyed by their names, so their quarter and half cells do not share layouts in FerrySim; in stack
trials the runner's seeds pair them.

**The reward** (decision 4 (a); `reward.RewardSpec(kind, c_t, c_e, c_cov, n_ref,
expected_availability)`). A decision is each Pass-1 stop flown, for every arm, so FX's, F's and a
slot's returns are read alike:

    r_k = G_k − c_t·Δt_k/T − c_e·ΔE_k/(P_hover·T)   (− c_cov·U at the sortie's last decision)
    G_k = Σ w_i / (n_ref·N), over the updates collected CLEAN at k
    U   = Σ ω_j over the committed devices left uncollected / Σ ω_j over the demand

w_i is the raw L3 weight the mule's merge gave the update (`update_weights`: n_i·v_i·s(a_i), 0 past
the cutoff); n_ref is the reference example count (10, the equal model's); N the mission's demand
with any beacon insert; Δt_k runs from the arrival at k to the next arrival, or after the sortie's
last decision to the end of the Pass-1 upload (the landing on the empty round); T is T_nom; ΔE_k is
the mule's own energy model over that span; ω are the plan's coverage weights, over its committed
(`served`) devices. Defaults: c_t = 0.1, c_e = 0, c_cov = 1; no lateness term (none of 1,148 probed
collections was late, critic B3) and no energy term (energy tracks time). At the training cells
every collected update weighs n_ref, so G_k is the stop's count over N, and "derived against
hand-set" compares weights, not what the merge weight contains (critic B4). A sortie with no Pass-1
stop makes no decision and adds nothing; its shortfall is reported only. FerrySim credits an update
whose backhaul upload was lost (Freeze §5l, recorded). `hand` is F·hand, "today's reward" ported
from ContactSim and declared as such: (200·|C_k| − Δt_k − 0.002·metres_k)/150, with no terminal
term. `bytes` is E3's: |C_k| / N. Study 5.7's grid is `reward.grid_specs()`: c_t ∈ {0.03, 0.1, 0.3}
× c_cov ∈ {0.25, 1, 4}. *Expected availability* (critic C1): training replaces each targeted
member's keyed availability draw by its probability rel_j (a collected member is credited rel_j·w_j,
a dropped one rel_j·n_ref, and U is taken in expectation); the policy never reads rel_j, and
validation, the held-out evaluation and every reported number use the realized draw.

**The episode API** (`episode.py`). `run_episode(cell, seed, policy, *, trial_index=0,
reward=DERIVED, device_model="equal", trainer=None, sink=None, hooks=(), stop_after=None,
keep_case=False, driver_overrides=None) -> EpisodeResult`. `Policy(label, arm="FX", scorer=None)`:
with no scorer the arm's own slot flies (`Policy.of_arm`); a scorer factory flies a fresh
`PairQSlot` on a plan arm's configuration, installed through `install_flight_slot`
(`Policy.scripted(name)` for the references). `reference_policies()` gives the FX and F arms (R6),
then `fx_pair`, `committed_pair`, `hyb` and `greedy_1`. `Trainer(epsilon, rng_seed,
around_reference)` attaches a trainer (ε = 0 with no reference flies as none does). `EpisodeResult`
holds the sortie records, each decision's reward terms, the mule's closed pair records, the steps
(with a trainer), the driver's row (with R25's artifacts), and `.ret`, `.terms`, `.rescored(reward)`
and `.summary()`. With `driver_overrides={"trace_root": d}` the trial's traces are kept under
`d/<cell name>/<policy label>/`: a pair slot's trace and row name arm FX, and the directory names the
policy. Under a pair slot the mule's own records are compared with FerrySim's reading stop by stop,
and any difference raises.

**Training** (`train.py`; other choices 3 and 12). `TrainSpec(kind, seed, family, cells, network,
learner, reward, episodes=10000, eval_every=1000, val_episodes=200, patience=3, phase=True,
plan_score_params={})` is one run: one learner, at one γ, from one seed, over one family's cells.
The pair score flies FX's configuration with a fresh slot around the live network; E3 flies its arm
through the config path from a bootstrap checkpoint the run writes to a temporary directory and
deletes after. Episode e flies `BehaviourSchedule.at(e, episodes)` from its own seeded stream. An
episode flies with the network as it stood at its start; then its transitions are pushed in
decision order, each followed by one update once the replay is warm, so the network moves between
episodes, never inside one. Every `eval_every` episodes and after the last, the network flies the
validation episodes greedily (`val_episodes`, spread over the family's cells, the first of each
cell's validation stream) at the realized draw; the score is the mean of the cells' means; a
strictly higher score keeps the weights, and the run stops after `patience` validations without
one. The kept weights are saved as `trained` (§19.5), recording the training spec and its outcome,
the seeds (the initial weights' and the replay's seeds are hashes of the run's stream, so every γ
of one seed starts alike), the family and its hash, the commit and the dirty flag,
`episodes_trained`, the validation curve, and no held-out score yet. `TrainSpec` refuses a reward at
the realized draw (critic C1), a kind and a reward that do not go together (`pair_q` trains on the
derived reward or F·hand, `chen_dqn` on bytes), an E3 schedule with a reference phase, fewer
validation episodes than cells, a plan for E3, and cells that set a plan themselves.
`plan_score_params` (R23) is the plan the pair score trains and validates under, recorded in the
spec only when set. A tree is dirty when git reports any change under `hermes/` or `experiments/`
(untracked files included, ignored ones not); without git it counts as dirty.

**The command line** (`python -m experiments.ferrysim <command>`). Nothing runs unless a user runs
it, and the campaigns wait for the user's go-ahead (Freeze §5l). Each command puts the caller's
logging level back when it ends (R28).

| Command | Flags (default) |
|---|---|
| `train` | `--kind` (`pair_q`, or `chen_dqn`); `--family` (`jittery`; `clean`, `jittery56`); `--study` (required); `--tag` (the arm's: `g<100γ>` on the derived reward, `dwell` or `cov` with `--ablation`, `hand` on F·hand, `e3`); `--root` (the repository's `results/exp5/checkpoints`); `--gamma`, `--seed` (both required); `--episodes` (10000); `--eval-every` (1000); `--val-episodes` (200); `--patience` (3); `--reward` (derived; `bytes` for `chen_dqn`; or `hand`); `--c-t` (0.1); `--c-cov` (1); `--lr` (1e-3); `--epsilon-start` (0.3); `--epsilon-end` (0.05); `--reference-episodes` (500; none for `chen_dqn`); `--no-phase`; `--ablation` (`dwell` or `cov`); `--plan-score-params` (JSON, the runner's format); `--allow-dirty`; `--overwrite` |
| `sweep` | as `train`, with `--gammas` and `--seeds` (both required) in place of `--gamma` and `--seed`, and `--workers` (1); every path is checked before the first run |
| `evaluate` | `--checkpoints` (files, or directories searched for `.npz` files at any depth); `--cells` (the jittery family's four); `--stream` (`heldout`, or `val`); `--episodes` (1000); `--start` (0); `--references` (FX, F and the four scripted ones) or `--no-references`; `--reward` (derived; `hand`, `bytes`); `--c-t`; `--c-cov`; `--workers` (1); `--plan-score-params` (the references' plan); `--record`; `--out` (required) |
| `report` | `--evaluation` (required); `--epsilon` or `--headroom` (exactly one); `--cells` (jit-n12-90 jit-n12-180); `--out` |
| `headroom` | `--cells` (decision 3's six); `--episodes` (200); `--start` (0); `--max-leaves` (512); `--workers` (1); `--plan-score-params`; `--out` |

- `train` and `sweep` refuse a dirty tree unless `--allow-dirty` is given (the manifest records it),
  and an existing checkpoint unless `--overwrite` is given, and never replace another family's (R19).
  A derived reward at other weights needs a `--tag` of its own (Study 5.7's grid: free tags that no
  arm flies, which the runner refuses), and an explicit learned arm's tag takes only a run that arm
  flies: its γ, its reward, decision 4 (a)'s weights under `gX`, `dwell` and `cov`, and an
  ablation's plan (R23, R24). `--ablation dwell` or `cov` trains FQ-dwell's or FQ-cov's score under
  the driver's own plan settings, under that tag, at decision 4 (a)'s weights; `--plan-score-params`
  may not repeat its keys. `--lr`, `--epsilon-start` and `--epsilon-end` reach the manifest.
- `evaluate` flies every checkpoint and reference on the same held-out episodes of each cell, each
  policy read under one reward at the realized draw (`--reward bytes` for E3's own score; R18), and
  keeps no traces. `--record` (the held-out stream only) writes each checkpoint's held-out score into
  its manifest: `episodes`, `return_mean` (the mean of the cells' means), the stream, the reward and
  each cell's summary. A pair checkpoint flies the plan it trained under; beside the references it
  must have trained on theirs (`--plan-score-params`), and with `--no-references` that flag is
  refused. The N = 6 control is among the default cells (R20: about 3.3 h for Study 5.5's
  evaluation at 8 workers); a clean study names `--cells cln-n12-90 cln-n12-180`, and Study 5.6's
  cells fly only when named.
- `report` applies Study 5.5's rule (below) to an evaluation file on `--cells`, with ε given or read
  from a headroom report flown on the evaluation's plan. It refuses an evaluation not on the
  held-out stream, a checkpoint that is not `pair_q` or not trained, two learner revisions or one
  past 1, mixed families, rewards or training specs, a duplicate (γ, seed), mismatched episode
  counts, cells not flown, and a table missing a reference the rule reads, each as a usage error.
- `python -m experiments.ferrysim.evaluate` (`--cells`, by default Study 5.5's two; `--stream`,
  `--episodes`, `--start`, `--policies`, `--workers`, `--out`) evaluates the references alone.

**Study 5.5's rule** (decision 5 (a); `report.decide`), in this order, with nothing looked at twice.
The training seed is the unit: a seed's score is its checkpoint's mean held-out return over the
cells read.

1. *Sanity* (critic A2): the γ = 0 mean must be at least max(FX, `greedy_1`) − ε, FX being the arm
   itself (R6); a point comparison (R18). If it fails, the outcome is `sanity-failed`, the curve is
   not read and FX stays; the learner may be revised once, on the control cells, before the sweep.
2. *Rising:* the best γ > 0, picked on the kept checkpoints' validation scores on the cells read
   (ties to the lower γ; the same entries for every γ, R18), beats γ = 0 by at least ε, with the
   bootstrap CI of the gain excluding 0 and the Holm-adjusted p below 0.05: an exact paired Wilcoxon
   of each γ > 0 against γ = 0 on the seed means, Holm over the five contrasts
   (`stats.compare_to_reference`; with 10 seeds the exact floor is 0.002, and 0.0098 after Holm).
3. *Flat:* every γ > 0 is within ±ε of γ = 0 by TOST on the seed means, an intersection-union test,
   so no Holm.
4. *Inconclusive:* anything else.

FX is replaced only if the outcome is rising and the best γ also beats the best fixed rule by ε
under the same claim rule; the fixed rules are decision 5's four (the FX and F arms, `hyb` and
`greedy_1`; R17), the best being the one with the highest held-out mean, and `fx_pair` and
`committed_pair` are reported beside them. If `greedy_1` beats the FX arm by ε on the held-out runs,
by the claim rule with the episode as the unit (R18), the verdict says so and the user decides
(critic A2). Reported alongside, never decided on: Page's trend test with Spearman's ρ, the learning
curves, each cell's share of sorties with two or more decisions (critic A1), every cell's means (the
N = 6 control included) and the stack check's picks (the best γ and γ = 0, each from its
median-validation seed). *The pre-registered grid* (R26): γ ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99}, 10
seeds per γ and 1,000 held-out episodes per cell (`GAMMAS`, `SEEDS_PER_GAMMA`,
`HELD_OUT_EPISODES`); a sweep whose γ set, seeds per γ or held-out count differs is labelled "NOT
pre-registered" on the verdict's second line and in its JSON, with the reasons, and is decided by
the same rule (the calibration's 2 γ × 3 seeds is one such sweep). Below 8 seeds "rising" is out of
reach after Holm over five contrasts, while "flat" is not.

**Headroom and ε** (decision 10 (i)(a); `headroom.py`; critic B14). Per sortie, a depth-first search
over the sequences of admitted pairs, each leaf a deterministic replay with `fx_pair` flying the
other sorties and the mule stopped once the sortie closes, at most `MAX_LEAVES` = 512 leaves
(beyond that the sortie is `truncated` and its value a lower bound). V is the largest of the
sorties' best returns summed and the whole-episode returns of the FX arm and the four scripted
references (`value_references`); the headroom is the mean of V − R(FX arm) over the validation
episodes (R6), and F's gain over FX is reported beside it, not in it. ε = max(0.01, 0.1 × the
headroom) (`epsilon_from_headroom`); the build pauses only if the headroom is below 0.01 in every
cell (`pause_rule`). `report` takes ε as max(0.01, 0.1 × the mean headroom of the cells it reads).
Every flight flies the cells' own plan, or `--plan-score-params`, which the report records and the
rule checks against the evaluation's plan (R23). ε belongs to Study 5.5, under decision 4's default
reward: the headroom command has no `--reward`, and Study 5.7 has no pre-registered ε (R28). The
build's report (Freeze §5l) gives ε = 0.01 in every cell.

**The statistics** (`experiments/analysis/stats.py`). `tost_paired(a, b, *, margin, alpha=0.05)`:
Schuirmann's two one-sided tests on the paired differences a − b; `.equivalent` when the larger
one-sided p is below α, which is the same as the (1 − 2α) interval lying inside (−ε, ε); with zero
variance p is 0 when the difference lies strictly inside the bound and 1 otherwise; it refuses a
margin that is not a finite number above 0, an α outside (0, 0.5), unpaired or non-finite inputs and
fewer than 2 seeds. `trend_test({level: per-seed values}, *, alternative="increasing",
method="auto", n_bootstraps=2000, confidence=0.95, seed=42)`: Page's L with the seeds as blocks and
the levels sorted ascending; `rho` is the mean over the seeds of Spearman's ρ, with a seeded
percentile bootstrap over the seeds; the exact p comes from the permutation null with tied ranks
kept as they are, the asymptotic one from the tie-corrected variance; `auto` is exact up to 8
levels (`EXACT_MAX_LEVELS`) and 200 seeds (`EXACT_MAX_SEEDS`), and `exact` refuses more than 8
levels. Both were checked against scipy (`ttest_1samp`, `page_trend_test`).

---

## 20. The Exp 5 addendum: Studies 5.11–5.15 (2 Oct 2026)

The build plan's addendum of 2 Oct 2026 adds five studies (5.11 decision cost and scaling, 5.12
compute and model-size heterogeneity, 5.13 data heterogeneity, 5.14 component ablations, 5.15 the
radio layer) and lists what each must build before it runs. This section records each build as it
lands. Freeze Rule 1 holds throughout: every switch defaults to the recorded run, every new trace
field is additive and left out where it does not apply, and the trial CSV's header is unchanged.
**Nothing in this section has run**; each study waits for the user's go-ahead.

### 20.1 Decision cost (Study 5.11 (a))

**Trace fields.** On the simulated clock, beside the records of §19.8 and never inside them (the
records stay free of wall time, critic B12):

- **`mission_completed.pass_1_pairs_wall`** (a `pair_q` mission that decided something): one entry
  per record of `pass_1_pairs`, in its order. **`mission_completed.pass_1_e3_wall`** (E3): one
  entry per call of `pass_1_e3`. Each entry is `{"decide_s": ..., "mask_s": ...}`, wall seconds
  from `time.perf_counter`: `decide_s` is the whole decision (the mask's predicate, the scorer or
  policy and the pick) and `mask_s` the predicate's share of it. For the pair slot the supervisor
  times from binding the predicate to the slot's answer and wraps the predicate it hands the slot,
  so the slot itself still reads no wall clock; for E3, the predicate over every stop left plus
  the policy's answer and its check. The view (`PairView`) or observation (`e3_observation`) a
  decision reads is built before it and is not timed.
- Both follow `pass_1_e3_unvisited` in `SIM_MISSION_OPTIONAL_FIELDS` and are left out when None or
  empty, so no other mission gains a key; F, FX and every H and D arm record neither.
- Wall times, so every determinism comparison drops them as it drops `plan_wall_s`: FerrySim's
  `inprocess.mask_wall_times` masks each value (`DECISION_WALL_FIELDS`), and the parity and
  repeat tests mask or drop them by name.
- **The consumer:** `MissionRecord.pair_walls` and `e3_walls`, tuples of `DecisionWall(decide_s,
  mask_s)`, None where a mission has no such field. A time that is not a finite number ≥ 0 reads
  as None, and an entry that is not a record as two Nones, so the rest stay in step.

**The cost columns** (`traces_scorer.COST_COLUMNS`) appear only with `cost_columns`
(`--cost-columns`), after the τ columns and, when both are asked for, after the pair columns;
without it the row is unchanged. Scorer-only: the trial CSV's header is unchanged.

| Column | What it holds |
|---|---|
| `plan_wall_s_mean`, `plan_wall_s_p95` | the planner's wall time per plan-mode mission (`plan_wall_s`, the whole of `build_ferry_plan`, every class searched), mean and 95th percentile (numpy's linear rule, as `age_p95`), over every mule's missions |
| `plan_search_shares` | the share of the plan-mode missions whose committed class ran each search mode (`plan.search`: `exact`, `stop_subsets`, `local`, all listed; JSON), so a sweep that forces a mode through `--plan-search-params` can check it held |
| `pair_wall_s_mean`, `pair_wall_s_p95`, `pair_mask_wall_s_mean` | the pair slot's `decide_s` per decision (mean, p95) and its `mask_s` (mean), pooled over the trial's decisions |
| `e3_wall_s_mean`, `e3_wall_s_p95`, `e3_mask_wall_s_mean` | the same for E3's calls |
| `flight_decisions_per_mission` | the mean over the trial's missions of the decisions timed in flight, both kinds together, on a trial whose mule config names the pair slot or E3's policy or whose missions recorded a wall (a mission without one made none) |

Each is blank where the trial has nothing to average. Forcing the search into one mode
(`--plan-search-params`, `PlanSearchParams`): exact with `exact_max_devices` ≥ N; stop subsets
with `{"exact_max_devices": 0}` and `exhaustive_max_stops` ≥ the stops; local with both 0.

```bash
python -m experiments.analysis.traces_scorer --traces results/exp5/s511a_traces --cost-columns --pair-columns --csv results/exp5/s511a_scored.csv
```

### 20.2 The footprint (Study 5.11)

A real-process trial runs 1 + K + N processes, about 1.6 GB apiece under `--real-model`, so
memory sets how large an N the stack can run (the build plan, 5.11's caveats). The footprint probe
(`experiments/exp4/footprint.py`) measures it per trial.

| Setting | Default | Meaning |
|---|---|---|
| `Exp4Driver.footprint_probe` (`--footprint-probe`) | off | Sample every process the orchestrator started (the cluster, the mules, the devices, each with its children) from a daemon thread, from just after `start_all` to just before `shutdown_all`, and write `footprint.json` (`FOOTPRINT_FILE`) beside the kept trace, a timed-out trial's included. Needs a trace root (`--keep-event-traces`; the runner refuses it otherwise, as a usage error) and `psutil`, which only this path imports. Reads only, so no trial changes; the runner passes the two settings to the driver only when the flag is given. |
| `Exp4Driver.footprint_interval_s` (`--footprint-interval-s`) | 0.5 | The sampling interval, seconds (> 0). |

`footprint.json` (schema 1): `processes` and `processes_by_role`; `peak_rss_bytes_total`, the
largest summed resident memory over one sample (the concurrent peak, what has to fit in the
host's memory); `peak_rss_bytes_by_role`, each role's largest per-process peak, where a process's
peak is the OS's high-water mark (`VmHWM` on Linux, the peak working set on Windows) or, on a
platform that records none, its largest sample (`peak_source`: `os`, `sampled` or `mixed`);
`peak_rss_bytes_sum`, every process's peak summed, an upper bound on the concurrent peak;
`samples`, `interval_s` and `probe` (the psutil version and platform). H0 runs in process and
writes none.

**Scorer columns**, in the cost group (`--cost-columns`, §20.1) after the decision cost:
`trial_processes`, `peak_rss_mib_total` and `peak_rss_mib_cluster`, `peak_rss_mib_mule`,
`peak_rss_mib_device` (MiB), blank for a trace without the file.

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s511b/fp.csv --arms F FX --N 6 12 24 --keep-event-traces --footprint-probe ...
```

### 20.3 A field that grows with N, the scale family and the pilot (Studies 5.9 and 5.11 (c))

**The field rule.** The realism field is a fixed 100 m half-width, so the device density rises with
N. `topology_builder.grown_field_radius_m(radius, N, ref_n)` keeps the density of `ref_n` devices
in `radius`: `radius * sqrt(N / ref_n)`, rounded to 0.1 m (at N = ref_n, `radius` itself). From
100 m at N = 6: 141.4 m at 12, 200 m at 24, 282.8 m at 48, 400 m at 96.

| Setting | Default | Meaning |
|---|---|---|
| `Exp4Driver.h1_field_ref_n` (`--h1-field-ref-n`) | None | Grow the realism field with N at this size's density: a trial's devices and T_nom's reference layouts are drawn on `field_radius_m(N)`. Needs `--realism` (refused otherwise). None keeps the recorded fixed field; the runner passes it only when given. The row does not record it (the kept traces hold the positions): write each setting to its own CSV. |
| S\* tool `--field-radius-m`, `--field-ref-n` | 100, None | The same field for the S\* tool's layouts. |
| `FerryCell.field_radius_m` | None | A FerrySim cell's field half-width (`h1_field_radius_m` in its driver settings); at None it is left out of the cell's JSON, so the other families' hashes are unchanged. |

T_nom at N = 6's density (the cells' flags, 1 MB, wide): 203 s at 6, 442 s at 12, 1,066 s at 24,
2,384 s at 48 and 6,123 s at 96, against 298, 378, 458 and 638 s at 12 to 96 in the fixed 100 m
field. Which field Studies 5.9 and 5.11 fly is the user's choice; the scale family below is the
constant-density one.

**The scale family** (`cells.SCALE_CELLS`, family `scale`): Study 5.11 (c)'s FerrySim cells beyond
the stack, the jittery decision-rich configuration of §19.9 at N = 24, 48 and 96 on the grown field
(`cells.SCALE_FIELD_M`), one mule, S = 2.

| Cell | N | Field | Budget |
|---|---|---|---|
| `scl-n24-350`, `scl-n24-525` | 24 | 200 m | 350 s (the binding edge), 525 s (1.5 ×) |
| `scl-n48-680`, `scl-n48-1020` | 48 | 282.8 m | 680 s, 1,020 s |
| `scl-n96-1330`, `scl-n96-1995` | 96 | 400 m | 1,330 s, 1,995 s |

The budgets are stand-ins until each size's budget pilot: the **binding edge** is the largest budget
on a 10 s grid at which F's S\* on 90 % of the S\* tool's 30 layouts is still 2 (one mission can no
longer serve every servable device on more than a tenth of them), found by bisection at planning
level on 2 Oct 2026, and the second budget is 1.5 × the edge. The rule reproduces the N = 12
stand-ins exactly (edge 120 s, 1.5 × 180 s); its N = 6 edge is 80 s, beside the priors 45 and 90 s.
S\* on 90 % of layouts is 2 at each edge and 1 at 1.5 ×, so S takes decision 1's floor, 2. The cells
stay out of `CELLS` (the headroom report's default) and every other family. A score trained at
N = 6 and 12 flies here out of practice (its /N features shift): a declared test; so does E3, whose
observation divides distances by the 100 m field (`chen_dqn.LENGTH_SCALE_M`, part of its schema). One FX episode as
a build check (2 Oct 2026, this container, not a measurement): 1.5 s at N = 24, 2.8 s at 48 and
9.8 s at 96, the planner 0.33, 0.61 and 2.26 s per mission.

**The pilot** (`python -m experiments.ferrysim pilot`, `experiments/ferrysim/pilot.py`): every
(cell, budget, policy) on the first `--episodes` episodes of each cell's validation stream (never
the held-out one), every budget and policy on the same layouts, each `--budgets` value overriding
the cell's own. Each episode's summary is `evaluate`'s plus `served_share` per mission (updates
collected over N) and `served_of_demand` (over the plan's demand), and its wall times: the
episode's (`wall_s`), the planner's per mission and each flight decision's (`decide_s`, `mask_s`,
from `EpisodeResult.walls`, which is never part of an episode's equality or summary). The table
(`pilot_table`) folds them per (cell, budget, policy), means and 95th percentiles; the knee is read
off it as the stack's pilot reads its own (where the served share stops rising): the module names
none. `--plan-search-params` forces the planner's mode as the runner's flag does (Study 5.11 (a)
in process), `--device-model stub` flies the stack's stub trainer, and `--trace-root` keeps each
episode's traces under `budget=<b>/<cell>/<policy>/` for `traces_scorer --cost-columns`.
Policies are the references' labels (FX, F, `fx_pair`, `committed_pair`, `hyb`, `greedy_1`) or any
driver arm; the default is FX.

```bash
python -m experiments.ferrysim pilot --cells scl-n96-1330 --budgets 1000 1330 1700 2000 --policies FX greedy_1 --episodes 20 --workers 4 --out results/exp5/s511c/pilot_n96.json
```

### 20.4 Interference strength (Study 5.15)

The contact channel's interference term is `A·sin(2π(t/P_c + φ)) + σ_I·n(t)` per class (§17.2);
the contact regime fixes A and σ_I as a pair (`CONTACT_REGIMES`: clean 1 and 0.4 dB, jittery 5 and
1.5 dB). Study 5.15 sweeps them one axis at a time, so each can now be set on its own.

| Setting | Default | Meaning |
|---|---|---|
| `MuleConfig.interference_amp_db` (`--interference-amp-db`, `ferry_physics`) | None | The amplitude A (dB, ≥ 0) in place of the regime's; None keeps the regime's. |
| `MuleConfig.interference_sigma_db` (`--interference-sigma-db`) | None | The noise σ_I (dB, ≥ 0) in place of the regime's. |

Both are ferry-spec fields (`FERRY_SPEC_FIELDS` → `FerrySpec.from_config` → `ContactChannel`),
simulated-clock only (refused on the wall clock when set), and in the driver's
`FERRY_PHYSICS_FIELDS`, so a FerrySim cell's `ferry_physics` takes them too. The planner prices
what flies: the outage's σ is `sqrt(σ_sh² + σ_I² + A²/2)` (`FerryRuntime.outage_probability`,
`plan_score.sigma_eff_db`), and `mule_ready.channel_params.contact` records both values, as it
always has. **`ferry_params`** leaves each out while it is None (`FERRY_PARAMS_OMITTED_AT_NONE`),
so every recorded row keeps its string, and shows it when set; the scorer's provenance does the
same. `ADDENDUM_MULE_FIELDS` names every `MuleConfig` field the addendum adds, so the tests that pin
what a recorded trial's per-role JSON gains (at the defaults: these keys, at None) name them.

The radio flags that already existed for Study 5.15: `--n-pl`, `--shadow-sigma-db`,
`--shadow-corr-s`, `--interference-period-s`, `--contact-regime`, `--backhaul-model` and
`--l1-channel` (§17.4, §17.5).

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s515/amp8.csv --arms F FX H1 --mission-clock sim --contact-band wide --contact-regime jittery --interference-amp-db 8 ...
```

### 20.5 The F+L1 arm (Studies 5.14 and 5.15)

`F+L1` (`driver.ADDENDUM_ARMS`) is F's plan with H3's adaptive backhaul controller and no learned
selector, as `H1+L1` is H1's scheduler with it (§19.7): the plan arms' reference for the adaptive
backhaul. It is a plan arm (`ADDENDUM_PLAN_ARMS`, `is_plan_arm`), so it has F's settings wherever
F has them (member subsets, the miss priority, the `trim` fallback, T_nom, the plan settings and
the pre-trial check), while `PLAN_ARMS` keeps its pinned value. It flies the controller where
H1+L1 does: on the simulated clock's seconds-axis backhaul (`backhaul_policy="adaptive"`, the
carrier picked at every upload) or with `--l1-channel` (the cluster's adaptive per-mission loss
schedule and H3's RF prior schedule); anywhere else, and on the wall clock (a plan arm), it is
refused before any trial. Its row differs from F's only in `ferry_params`' `backhaul_policy`. It
runs only when named.

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s515/fl1.csv --arms F F+L1 H1 H1+L1 --mission-clock sim --contact-band wide --backhaul-model seconds ...
```

### 20.6 The hover-stop switch (Study 5.14)

The hover rule (§18.4, `plan/hover.py`) was unconditional in plan mode. `PlanSearchParams.hover_stops`
switches it:

| Setting | Default | Meaning |
|---|---|---|
| `plan_search_params.hover_stops` (`--plan-search-params '{"hover_stops": false}'`) | true | True: the search runs over S3a's stops with the hover rule applied, as recorded. False: over S3a's stops alone, so a capped device its S3a stop cannot serve alone stays there, and the cap's `unplannable` reads S3a's stops. |

It lives in the search settings, the stop family the search runs over, so it needs no new plan
field: `PlanSearchParams.as_dict` leaves it out at True, so a plan's `mule_ready` and every pinned
dict keep their keys, and the row's `ferry_params` records `plan_search_params` as given (`{}` at the
defaults, `{"hover_stops": false}` when off). The build plan expected the switch to change every
plan arm's `ferry_params` string; this way only a run that sets it shows it, and no recorded string
moves. Study 5.14's "hover off" arm is F with this setting. The S\* tool prices the hover rule's
stops (§18.7); a cap set for a hover-off run is the user's call (`layout_s_star(..., hover=False)`
prices S3a's stops alone).

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s514/hover_off.csv --arms F --mission-clock sim --contact-band wide --plan-search-params '{"hover_stops": false}' ...
```

### 20.7 Data heterogeneity and the detector's metrics (Study 5.13)

Every recorded trial split its training rows IID (`partition_indices`: an even cut of a seeded
permutation) and scored the global model by accuracy, AUC and loss only. All of the below is opt-in
and needs `--real-model` (the stub trainer has no data); at the defaults the task, its files, the
events and the rows are the recorded ones.

| Setting | Default | Meaning |
|---|---|---|
| `Exp4Driver.partition` (`--partition`) | `iid` | `iid`: the recorded split, exactly. `dirichlet`: label skew, each class's rows shared over the devices by Dir(α·1_N) (Hsu et al. 2019; the NIID-Bench rule), over the attack families with `family_labels`, else over the binary label. `quantity`: shard sizes Dir(α·1_N), each shard a mix as in IID. |
| `Exp4Driver.dirichlet_alpha` (`--dirichlet-alpha`) | None | α > 0, finite (1 moderate, 0.1 strong); needed by `dirichlet` and `quantity`, refused with `iid` (α = ∞). |
| `Exp4Driver.family_labels` (`--family-labels`) | off | Keep each row's CICIoT2023 attack family beside the binary label (`partition.FAMILIES`: Benign = 0 and the legacy loader's seven `DICT_7CLASSES` families), in the task and as a `family` array in the shard and test `.npz` files. The model stays binary and the devices never read the family. |

**The partitioner** (`experiments/exp4/partition.py`). Every draw is a function of the trial seed,
the partition and α (paired arms hold the same shards), and a skewed partition re-cuts the IID
rows: the test set and the training rows are the IID task's, so an α sweep is paired too. **No shard
is empty:** a draw that leaves a device short is drawn again (up to `MAX_DRAWS`, 1,000), and a
partition that still cannot fill every device is refused; `prepare_trial` also refuses any task with
an empty shard, which would train nothing and report zero metrics silently.

**The family label on the canonical data.** `load_and_balance_data_stratified` keeps its
`original_label` column when asked (`keep_original_label`, off by default, when it drops it as
recorded); the loader maps it through `DICT_7CLASSES` and carries it by row index through the
canonical `preprocess_dataset`, which shuffles and splits by position and keeps the index, so the
family never enters the features. The synthetic task draws a family per row from a stream of its
own (Benign for class 0, an attack family uniformly for class 1), moving no row.

**The detector's metrics.** When the cluster's test set carries the families, `model_eval` adds
`detection` (`model_task.detection_metrics`): the confusion counts (`tp`, `fp`, `tn`, `fn`), `tpr`
(recall on attacks), `fpr`, `precision` and `f1` (null where a denominator is 0), and per family
present its rows (`n_by_family`) and recall (`recall_by_family`: flagged for an attack family,
passed for Benign, so Benign's is 1 − FPR). Without the families the event is the recorded one. The
consumer reads it as `ModelEvalPoint.detection` (`Detection`). H0's in-process evaluation does not
compute it.

**The status marker** of a kept trace records a non-default data setting as `data`:
`partition`, `dirichlet_alpha`, `family_labels` and `shard_rows`, each device's training rows by its
id. The trial CSV's header is unchanged and its row does not record the setting: write each setting
to its own CSV.

**Scorer columns** (`traces_scorer.DETECTION_COLUMNS`, `--detection-columns`, last):
`data_partition` and `data_alpha` (from the marker), `network_aou_shard_weighted_mean` (Network
AoU with each device weighted by its shard's rows), and the final evaluation's `tpr_final`,
`fpr_final`, `precision_final`, `f1_final`, `recall_by_family_final` (JSON) and
`recall_family_min` (the worst attack family's, Benign left out). Blank where the trace cannot say;
the final accuracy and AUC are the summary's own columns.

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s513/dir01.csv --arms F FX H1 D3 --real-model --family-labels --partition dirichlet --dirichlet-alpha 0.1 --keep-event-traces ...
python -m experiments.analysis.traces_scorer --traces results/exp5/s513/dir01_traces --detection-columns --csv results/exp5/s513/dir01_scored.csv
```

### 20.8 The model's architecture (Study 5.12)

`build_ids_model` built `create_CICIOT_Model` only. `model_task.MODEL_ARCHS` makes the architecture a
setting; each is a binary classifier over the canonical inputs with one sigmoid output, so local
training (FedProx included), the evaluation (the detection metrics included) and the merge treat
them alike, and each is deterministic from the seed. θ at 21 inputs, float32:

| `--model-arch` | Builder (`Config/modelStructures/NIDS/NIDS_Struct.py`) | θ |
|---|---|---|
| `ciciot` (default, None) | `create_CICIOT_Model`: dense 64-32-16-8-4-1, every recorded run's | 18,756 B |
| `balanced` | `create_balanced_nids`: separable Conv1D + GRU | 41,360 B |
| `optimized` | `create_optimized_model`: a dense residual stack | 92,676 B |
| `high_performance` | `create_high_performance_nids`: Conv1D + GRU + LSTM | 347,396 B |

| Setting | Default | Meaning |
|---|---|---|
| `Exp4Driver.model_arch` (`--model-arch`) | None | The architecture the seed θ (`initial_theta`), every device's trainer (`make_local_train_fn`) and the cluster's evaluation (`evaluate_theta`) build. Real model only (refused otherwise, and an unknown name). The measured payload follows θ's size on the simulated clock, so the larger models' dwell and upload grow with them; `--payload-bytes` still declares one. |
| `DeviceConfig.model_arch`, `ClusterConfig.model_arch` | None | Set by the topology builder; written to the per-role JSON only when set (`CONFIG_FIELDS_OMITTED_AT_NONE`), so a recorded trial's JSON keeps its keys. |

At None every builder is called exactly as recorded (no `arch` keyword is passed). The row does not
record the architecture: write each to its own CSV (the kept per-role JSON names it).

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s512/hp.csv --arms F FX H1 D5 D4 --real-model --model-arch high_performance --mission-clock sim --contact-band wide ...
```

### 20.9 Training time on the simulated clock, and the device energy (Study 5.12)

Every recorded run charged a device's local fit nothing on the simulated clock: an update was always
ready when the mule came. With `--train-time-s`, each device's fit takes simulated time and a Pass-1
contact can find no update ready.

**The draw** (`experiments/exp4/compute.py`). Each device's fit time `T_j` comes from four settings
and the trial seed. A device's draws are keyed by the seed and its id, so every arm of a trial holds
the same times:

| Setting (flag) | Default | Meaning |
|---|---|---|
| `median_s` (`--train-time-s`) | off | The median fit time, simulated s (>= 0). 0 is the sweep's "none" level. It flies exactly as a recorded run (no contact is ever not ready), but it records the fits and uplinks for the device energy. |
| `sigma` (`--train-time-sigma`) | 0 | The log-normal spread: `T_j = median_s * exp(sigma * z_j)`, with `z_j` a standard normal. |
| `straggler_share` (`--straggler-share`) | 0 | Exactly `round(share * N)` devices (half up) straggle; the plan's "20 % stragglers at 5×" is 0.2 with factor 5. |
| `straggler_factor` (`--straggler-factor`) | 1 | The stragglers' time multiple (>= 1). |

`Exp4Driver.train_time_params` holds them (the simulated clock only; refused otherwise). The topology
builder draws `T_j` for every device and gives each mule its slice's times in two fields.
`MuleConfig.device_train_time_s` holds the per-device times, the ground truth, never shown in
`ferry_params`, as the availability is not. `MuleConfig.train_time_params` holds the settings, and
the row's `ferry_params` shows them as `train_time_params`.

**The model** (`hermes/mule/fit_clock.py`, the mule's `FitClock`). It lives on the mule, as the
availability draw does:

* Each device starts its first fit at the mule's first takeoff: it was deployed with the seed model.
* A device starts a new fit each time a model reaches it: a Pass-1 push (collected, uplink dropped
  or unanswered) or a Pass-2 delivery, at that session's stamp (`ContactCommit.pushed`).
* A Pass-1 contact at time t finds an update ready iff `t >= start_j + T_j`.

A target that is not ready is named in the solicit (`FLOpenSolicit.not_ready`) and answers with its
advert. Nothing is pushed to it, so its fit runs on and its basis is kept. It costs no airtime and no
listen window, and is recorded as a TIMEOUT (`ContactPlan.not_ready`, `ContactCommit.not_ready`). The
device code trains after a delivery, and after a Pass-1 push when asked to train ahead, so the model
times its fits exactly there. One case is optimistic: a device that adopts a Pass-1 basis without
training ahead (an unbudgeted Pass 2) and then misses its delivery. It fits at its next contact, on
that contact's model, but the model times that fit from the basis it last received.

**The records**, only when the mule has train times, so no recorded trace gains a key:

* `mission_completed.train_fits`: every fit the mission started, `[device, start_s]`.
* Each Pass-1 stop's `not_ready`: the targets found with no update ready.
* Each Pass-1 stop's `uplink_s`: each collected update's own uplink airtime (`ContactCommit.uplink_dwell_s`: its bytes at its session's SNR; empty without a band).
* `mule_ready.train_time_params` and `train_time_n`: the settings and how many devices the clock times (never the per-device times).

**The scorer** (`traces_scorer --compute-columns`, last in the row; `compute_report`):

| Column | Meaning |
|---|---|
| `train_time_median_s`, `train_time_sigma`, `straggler_share`, `straggler_factor` | The settings, from the mule config. |
| `pass_1_target_contacts`, `not_ready_contacts`, `not_ready_share` | Pass-1 targets solicited, those that found no update ready, and their share. |
| `pass_1_target_contacts_after_first`, `not_ready_contacts_after_first`, `not_ready_share_after_first` | The same over each mule's missions after its first. Every fit starts at the first takeoff, so the first mission's contacts, seconds later, find few updates ready at any train time. `pilot3`'s p512 rule reads this share (decided 6 Oct 2026). |
| `pass_1_clean_share` | CLEAN Pass-1 sessions per target contact. The summary's own `update_yield` is the plan's update yield. |
| `policy_not_ready_drops` | The devices arm D5's readiness test left out before takeoff (0 for every other arm). |
| `device_train_busy_s` | The devices' fit seconds. Each fit runs `T_j`, or until a newer model restarts it, or until its mule's last mission ends, whichever comes first. |
| `device_uplink_s` | The collected updates' uplink airtime. |
| `device_p_comp_w`, `device_p_tx_w` | The powers used: `--device-p-comp-w` (default 5.0 W) and `--device-p-tx-w` (default 1.0 W). These are placeholders for the modelled device. |
| `device_energy_j_total`, `device_energy_j_max` | Per device, `P_comp × busy + P_tx × uplink`, as the total and for the most loaded device. |

The study's "none" level is `--train-time-s 0`, so its rows carry the same columns.

**The baselines that read the update times** (the user's decisions of 2026-10-03). With train times,
the mule process binds its fit clock to a whole-scheduler baseline that reads it
(`bind_fit_clock`), and `mule_ready.train_time_policy` names it:

* **D5, FedCS: a readiness test, no waiting** (deviation 3 of `policies/fedcs_degraded.py`). FedCS
  counts each client's update time in the round. A mule cannot wait at a stop for an update without a
  capability no other arm has, so the term becomes a test. At each step of Algorithm 3 the candidates
  are the contacts with a member whose update is ready at the predicted arrival (`ready_at <= clock +
  transit`). A contact none of whose members is ready stays for later steps, and the walk ends when no
  remaining contact has one. Whole stops are priced whole; under member subsets a candidate is reduced
  to its ready members. What the test left out is reported in `pass_1_policy_drops` with reason
  `not_ready`; the scorer counts those devices as `policy_not_ready_drops`.
* **D2, Oort: the system-speed term restored** (deviation 1 of `policies/oort.py`). Each explored
  member's utility, staleness bonus included, is multiplied by `(T / t_i) ** alpha` when `t_i > T`
  (Oort's Eq. 2 and Algorithm 1). `t_i` is the device's fit time plus its predicted dwell at the
  contact: the shared feasibility model's per-member dwell, one session without a band, and `inf` for
  a member predicted unreachable. `T` is the cell's T_nom, and `alpha` is Oort's default, 2. The
  driver computes T_nom for D2 whenever train times are set, and `mule_config_errors` refuses D2 with
  train times but no `t_nom_s`. Oort does not test readiness, so D2 still flies to a device whose
  update is not ready; the term only ranks slow devices lower.

Without train times nothing is bound, and both policies are the recorded ones.

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s512/strag.csv --arms F FX H1 D5 D4 --mission-clock sim --contact-band wide --train-time-s 60 --straggler-share 0.2 --straggler-factor 5 --keep-event-traces results/exp5/s512/traces ...
python -m experiments.analysis.traces_scorer --traces results/exp5/s512/traces --compute-columns --csv results/exp5/s512/scored.csv
```

### 20.10 Study 5.4's sweep knobs (unit U10)

Study 5.4 sweeps the classes' range–rate trade and the share of devices beyond the widest class's
reach (build plan, 5.4). Unit U10 adds one setting for each, decided on 2026-10-05; at their
defaults every range, layout, row and trace is the recorded one.

| Setting | Default | Meaning |
|---|---|---|
| `MuleConfig.narrow_range_ratio` (`--narrow-range-ratio`, `ferry_physics`) | None | The narrow class's planar reach as a multiple of `rf_range_m` (wide's), in place of D1's derivation (232.2 / 60 = 3.87 at the defaults, §17.1). Study 5.4 sweeps 2, 3 and the derivation. |
| `Exp4Driver.far_share` (`--far-share`; the S\* tool's `--far-share`) | None | Exactly `far_count(N, share)` = ⌊share·N + ½⌋ devices of every realism layout lie beyond `rf_range_m` of the dock and the rest within it. Study 5.4 sweeps 0.25, 0.5 and 0.75. |

**The narrow reach.** `ContactLink.narrow_range_ratio` sets the class's planar reach; its slant
reach follows (`hypot(reach, altitude)`), and so does its mean SNR curve, because a class's edge is
defined as the floor plus the 90 % shadowing margin at its reach (§17.1). So a ratio below the
derivation's lowers the narrow class's implied EIRP below the others'; the edge keeps its meaning,
and wide and medium are untouched. It needs a narrow class that is not the anchor. It is a
ferry-spec field (`FERRY_SPEC_FIELDS` → `FerrySpec.from_config` → `ContactLink`), simulated-clock
only, in `FERRY_PHYSICS_FIELDS` and `ADDENDUM_MULE_FIELDS`, and `ferry_params` leaves it out at None
(`FERRY_PARAMS_OMITTED_AT_NONE`) as it does §20.4's fields.

**The far share.** `topology_builder.device_positions(..., far_share=, far_radius_m=)` chooses the
far devices with the layout's own seeded generator, then draws each device uniformly on the field
until it lands on its side of `far_radius_m` (the dock is at the origin; the trace scorer's
`far_devices` reads the same set, so `far_served_share` is its served share). The driver needs
realism and applies it to the trial's devices and to T_nom's reference layouts (and T_nom's cache
key), and the S\* tool's reference layouts take it, so the knee, S\* and T_nom of a far-share cell
are measured on its own layouts. A share that the field cannot hold (no point beyond the radius) is
refused. The row does not record it: write each setting to its own CSV (the kept traces hold the
positions; the launcher's `.argv.json` holds the flag).

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s54/far50_ratio2.csv --arms F FB+wide FB+medium FB+narrow --mission-clock sim --contact-band wide --realism --far-share 0.5 --narrow-range-ratio 2 ...
python -m experiments.analysis.age_cap_s_star --budgets 90 45 --far-share 0.5 --ferry-physics '{"narrow_range_ratio": 2}' --payload-bytes 1000000 --contact-band wide --regime jittery
```

Tests: `tests/unit/test_exp5_u10.py`.

### 20.11 The O1 oracle (Study 5.4's optimality gap)

`experiments/analysis/o1_oracle.py` is the build plan's O1 (after Zhai et al., TWC 2025; decided
2026-10-05): an offline, planning-level search that is never flown. F flies FerrySim episodes in
process, and an `on_mule` hook wraps the scheduler's `build_ferry_plan`: each mission F plans as it
always does, then the oracle searches on exactly that plan's inputs (`capture`: the demand,
weights, S3 deadlines, the age cap, the start state, the budget's end, each class's model and Pass
2, the score settings). So the gap is read on F's own mission states, ages and caps included.

| Setting | Default | Meaning |
|---|---|---|
| `--cells` | `jit-n6-75 jit-n6-150` | FerrySim cells at N ≤ 6 (the build plan's bound; a larger cell is refused). |
| `--episodes`, `--stream` | 30, `ferrysim-val` | Episodes per cell, from the validation stream. |
| `--arm` | F | The arm whose plans are judged. |
| `--physics`, `--driver` | none | Overrides on the cells, e.g. Study 5.4's `'{"narrow_range_ratio": 2}'` or `'{"far_share": 0.5}'` (§20.10). |

**The family.** A plan is an ordered sequence of disjoint groups of demanded devices, each served
at one position on one class whose planar reach covers every member from there (a member beyond it
would count as served at outage 1 with no dwell, so it is refused). A group is offered its centroid,
each member's position (S3a's two rules), and the position of every stop F was offered on any class
that holds all its members (S3a's and the hover rule's). So F's own family, the offered stops and
their member subsets on one class, lies inside the oracle's, and per-stop bands, any grouping and
any order are added. Each stop is admitted by its class's predicate (S3b's deadline-and-budget
rule, an exempt stop protected) from the state the route has reached; the classes share the dock,
speed and hover power, so one state threads through them. The committed class b̄ prices Pass 2.

**The score and the gap.** Exactly F's: `plan_score.score` and `cap_key`, each member's outage on
its own stop's class. Two optima per mission: the best under F's own plan key (the cap key, then
the served share under the default lexicographic rank, then V), and the best V. The report gives
`gap_v` (best V − F's V ≥ 0), `gap_v_at_key` and `gap_share_at_key`, whether the key-best flies a
stop off its committed class, and the search's size. Two checks run on every mission: the oracle
prices F's committed plan at F's own V (else it raises: the two disagree on pricing), and its best
is never worse than F's under F's key. A branch is pruned when one already met served the same
devices from the same position no later, on no more energy and no more link loss, or when a bound
on every extension's V (its home, energy and link with nothing left uncovered, on the cheapest Pass
2; F's default Δ only) cannot beat either optimum; a test checks that the bound changes neither.
About 1 minute per N = 6 mission on one core.

```bash
python -m experiments.analysis.o1_oracle --cells jit-n6-75 jit-n6-150 --episodes 30 --workers 8 --out results/exp5/s54/o1_gap.json
```

Tests: `tests/unit/test_exp5_o1_oracle.py`.

### 20.12 Study 5.2's deadline forms: F-round and F-pref (unit U11)

Study 5.2 asks whether FeRRy's per-device deadline (admit, order, cut off) beats the deadlines the
selection literature uses. F flies the multiplicative law and F·add the additive one (§15); unit U11
adds the other two arms, decided on 2026-10-05 with the whole deadline form varied (option A): the
admission, the order and the merge cutoff all follow the arm's form.

| Arm | Deadline law | Admission and order | Merge cutoff (`agg:cutoff`) | Weights |
|---|---|---|---|---|
| `F-round` (after FedCS) | `round`, `round_s` = the cell's mission budget | every device's deadline is the plan's time + `round_s`, the round's end; S3a's anchoring ties, so it keeps S1's order | Φ = `round_s`, so `a_max = ⌊round_s / T⌋`: about one round | F's |
| `F-pref` (after Oort) | `pref`, `round_s` = the budget | as F-round: no per-device cutoff (the round's end, which the budget clause already enforces) | Φ = ∞: no cutoff (`age_cap` reads an infinite window as none) | × Oort's `(T / t_j)^α` for `t_j > T` |

**The laws** (`hermes/scheduler/stages/s3_deadline.py`): `LAW_ROUND` and `LAW_PREF` take
`round_s` (refused elsewhere, left out of `to_params` when None, so every recorded and multiplicative
row keeps its string); `compute_deadline` returns `now + round_s` for every device; neither moves a
window with outcomes; `effective_window` is `round_s` or ∞. The arms set their own law
(`Exp4Driver.arm_deadline_law`; `_ARM_LAW`), whatever `--deadline-law` says, and need
`--mission-budget-s`; every other arm keeps the driver's law. The row's `deadline_law` and
`deadline_params` are the arm's, and the scorer rebuilds them the same way.

**Oort's factor** (`plan_score.oort_speed_factors`, D2's `speed_penalty` with α = 2): T is the plan's
T_nom and `t_j` the device's round time as D2 reads it, its fit time (none without
`--train-time-s`) plus its predicted dwell, here at its own position on the reference class (the
plan weighs devices before it groups them). A device that never finishes keeps a weight of
`MIN_SPEED_FACTOR` (1e-12), as a weight must stay positive. `MuleConfig.plan_speed_alpha` (F-pref:
2; sim-only, plan mode only, needs `t_nom_s`) makes the mule bind the factor as the scheduler's
`plan_speed`, and `demand_weights(..., speed=)` multiplies each weight by it; `ferry_params` shows
`plan_speed_alpha` only when set. Without training times `t_j` is a dwell of seconds, under T, so
the factor is 1: in Study 5.2's default cells F-pref then differs from F-round in its cutoff alone,
as D2's speed term is inert there too; with `--train-time-s` (Study 5.12's settings) it binds.

**The merge period.** `agg:cutoff` cuts by age only with a period T (`--agg-period-t-nom`, or
`--agg-period-s`); without one the window term is off for every arm. The campaign's stages after
the knee pilot pass `--agg-period-t-nom` (`scripts/exp5/params.toml`, `campaign.merge_period_t_nom`).

**The change inside a pinned definition.** `build_ferry_plan` passes
`speed=getattr(self, "plan_speed", None)` to `demand_weights`; at None, every arm but F-pref, the
weights are the recorded ones. `tests/unit/test_p5_fits_after_service.py` strips exactly that
keyword before comparing the definition with 386c275's (Scheduler Freeze §5n).

```bash
python -m experiments.exp4.runner_main --csv results/exp5/s52/round.csv --arms F-round --mission-clock sim --contact-band wide --mission-budget-s 90 --aggregation agg:cutoff --agg-period-t-nom ...
```

Tests: `tests/unit/test_exp5_u11.py`.

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
