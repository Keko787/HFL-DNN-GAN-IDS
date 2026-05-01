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
| `FAST_PHASE_MISSED_WIDEN_S` | [hermes/scheduler/stages/s3_deadline.py:45](hermes/scheduler/stages/s3_deadline.py:45) | 10.0 | module constant | How much a TIMEOUT/PARTIAL delta loosens the next deadline window. Asymmetric (widen > shrink) to err on the side of attempt rather than skip. |
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
