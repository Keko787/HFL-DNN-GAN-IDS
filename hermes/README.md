# `hermes/` — the FeRRy code map

`hermes` is the Python package that runs FeRRy. In FeRRy a UAV data mule flies from a dock, collects
federated-learning updates from edge devices, and carries them back to the edge server. Each mission
makes two passes. Pass 1 collects updates. Pass 2 delivers the merged model.

A mule makes its decisions on two clocks:

- **Plan clock** (at the dock, before takeoff): picks a band class and a Pass-1 route as one
  decision, under one feasibility test.
- **Flight clock** (in Pass 1, in the air): at each stop, picks the band to serve on and the next
  stop to fly to.

Every arm runs on simulated **mission time**: time since takeoff, with flight and energy priced by
a model.

This page shows where each part lives. It names symbols, not line numbers. To find one, search for
`def <name>` or `class <Name>`. Links are relative to this folder. Code outside `hermes/`
(`experiments/`, `scripts/`) is linked with `../`.

## Quick index

| Looking for | File | Symbol |
|---|---|---|
| Plan clock entry point | [scheduler/fl_scheduler.py](scheduler/fl_scheduler.py) | `FLScheduler.build_ferry_plan` |
| Route + band search | [scheduler/plan/plan_search.py](scheduler/plan/plan_search.py) | `plan_search`, `search_class` |
| Plan score V, plan key | [scheduler/plan/plan_score.py](scheduler/plan/plan_score.py) | `score`, `plan_key`, `coverage_weight` |
| Feasibility test | [scheduler/stages/s3b_feasibility.py](scheduler/stages/s3b_feasibility.py) | `FeasibilityModel.admit`, `filter_feasible` |
| Flight clock loop | [mule/mule_main.py](mule/mule_main.py) | `MuleSupervisor._ferry_fly_pass` |
| Departure check | [mule/mule_main.py](mule/mule_main.py) | `MuleSupervisor._ferry_departure` |
| Re-plan trim | [scheduler/fl_scheduler.py](scheduler/fl_scheduler.py) | `FLScheduler._trim_plan`, `replan_remainder` |
| Cross heuristic (arm FX) | [scheduler/policies/cross_heuristic.py](scheduler/policies/cross_heuristic.py) | `CrossHeuristic` |
| Learned pair score (arm FQ) | [scheduler/policies/pair_slot.py](scheduler/policies/pair_slot.py) | `PairQSlot` |
| RL model (double DQN) | [scheduler/selector/pair_q.py](scheduler/selector/pair_q.py) | `PairQNet`, `PairQLearner` |
| RL features (36 columns) | [scheduler/selector/pair_features.py](scheduler/selector/pair_features.py) | `PairFeatureSchema`, `pair_rows` |
| RL replay buffer | [scheduler/selector/pair_replay.py](scheduler/selector/pair_replay.py) | `PairReplay` |
| Reward | [../experiments/ferrysim/reward.py](../experiments/ferrysim/reward.py) | `RewardSpec`, `sortie_rewards`, `episode_return` |
| Training | [../experiments/ferrysim/train.py](../experiments/ferrysim/train.py) | `train`, `pair_transitions`, `validate` |
| Routing primitives | [scheduler/routing/](scheduler/routing/) | `two_opt`, `best_order`, `replan_route` |
| Baselines D1–D5, E3 | [scheduler/policies/](scheduler/policies/) | see [Arms](#4-arms-paper-label--code) |
| Arm label → settings | [../experiments/exp4/driver.py](../experiments/exp4/driver.py) | `PLAN_ARMS`, `_PLAN_ARM`, `_ARM_POLICY` |
| Merge rules (mule + server) | [mission/aggregation_rules.py](mission/aggregation_rules.py) | `merge_on_mule`, `update_weights`, `staleness` |
| Mission time, energy | [l1/mission_clock.py](l1/mission_clock.py) | `MissionClock`, `EnergyModel`, `FlightModel` |
| Band classes, radio | [l1/contact_link.py](l1/contact_link.py), [l1/channel_model.py](l1/channel_model.py) | `BandClass`, `ContactLink`, `ContactChannel` |
| Campaign launcher | [../scripts/exp5/launch.py](../scripts/exp5/launch.py) | stages from [params.toml](../scripts/exp5/params.toml) |

## Folder tree

```
hermes/
├── scheduler/            L2: the plan clock, and each arm's admission and order
│   ├── fl_scheduler.py     FLScheduler: plan entry, H1's queue, Pass-2 queue, re-plan
│   ├── plan/               plan clock: search, score V, member subsets, hover stops, types
│   ├── stages/             pipeline stages S1, S3, S3a, S3b (feasibility), S3d (age cap)
│   ├── routing/            2-opt tour building, re-planning the rest of a pass
│   ├── policies/           flight-slot fillings (F, FX, FQ) and the baselines D1–D5, E3
│   └── selector/           the learned pair score: network, features, replay
├── mule/                 the mule: flight loop, ferry runtime, dock handler, device fit clock
├── mission/              one mission's FL: mule-side server, device client, contact plan, merges
├── cluster/              edge server (Tier 2): device registry, server merge
├── l1/                   radio and physics: band classes, channels, mission time, energy
├── processes/            multi-process topology: orchestrator, cluster/mule/device processes, config
├── transport/            RF and dock links (loopback and TCP), wire format
├── types/                shared dataclasses: messages, bundles, plan commit, round reports
└── observability/        JSONL event stream and metrics registry
```

Code outside the package:

```
experiments/ferrysim/     FerrySim: the stack run in one process; RL training, held-out evaluation, scale
experiments/exp4/         trial driver and CLI: arm → config, field, DNN-IDS task, data shards
experiments/analysis/     trace scorer (time to τ, reach, ages, cap violations) and statistics
experiments/runner/       shared trial grid with paired seeds and a resumable CSV log
scripts/exp5/             Exp 5 campaign: launcher, parameters, scoring, paper tables and figures
tests/                    unit, integration and golden tests
```

## 1. The two clocks

### Plan clock (at the dock)

Entry: `FLScheduler.build_ferry_plan` in [scheduler/fl_scheduler.py](scheduler/fl_scheduler.py).
The steps run in this order:

1. **Eligibility and deadlines.** [stages/s1_eligibility.py](scheduler/stages/s1_eligibility.py)
   `filter_eligible` keeps the eligible devices. [stages/s3_deadline.py](scheduler/stages/s3_deadline.py)
   then gives each one a deadline: `DeadlineLaw`, `DeadlineLaw.next_window`, `compute_deadline`,
   `classify_bucket`.
2. **Age cap and coverage weights.** [stages/s3d_age_cap.py](scheduler/stages/s3d_age_cap.py)
   provides `evaluate_cap`, `stop_deadline` and `priority_first`.
   [plan/plan_score.py](scheduler/plan/plan_score.py) provides `coverage_weight` and `demand_weights`.
3. **Stops, for each band class.** [stages/s3a_cluster.py](scheduler/stages/s3a_cluster.py)
   `cluster_by_rf_range` groups devices into stops at the class's radius.
   [plan/hover.py](scheduler/plan/hover.py) `offer_hover_stops` gives each capped far device a stop
   of its own.
4. **Search.** [plan/plan_search.py](scheduler/plan/plan_search.py) `plan_search` runs
   `search_class` for each class. It picks one of three modes with `search_mode`: exact, stop
   subsets or local search (`_ClassSearch._exact`, `_stop_subsets`, `_local`). `price_pass_2`
   prices Pass 2.
5. **Feasibility test.** Every candidate is admitted against one test: the mission budget, with
   flight, dwell, upload and the return leg priced on mission time.
   [stages/s3b_feasibility.py](scheduler/stages/s3b_feasibility.py) provides `FeasibilityModel`
   (`admit`, `fold`, `leg`, `home_at`) and `FerryPhysics` (dwell and upload times).
   [plan/member_subset.py](scheduler/plan/member_subset.py) admits part of a stop: `member_order`,
   `admit_stop`, `trim_members`.
6. **Choice.** The best candidate wins by `plan_key` over the score `score` (V), both in
   [plan/plan_score.py](scheduler/plan/plan_score.py). The commitment is a `PlanCommit` in
   [types/scheduler.py](types/scheduler.py). Shared plan types (`Candidate`,
   `PlanScoreParams`, `PlanOptions`, `ArrivalView`, `PairView`) are in
   [plan/types.py](scheduler/plan/types.py).

### Flight clock (Pass 1, in the air)

Entry: `MuleSupervisor._run_ferry_mission` in [mule/mule_main.py](mule/mule_main.py). It runs one
two-pass mission: takeoff, plan, Pass 1, upload, Pass 2, dock. Each pass is flown by
`_ferry_fly_pass`. At each departure it does these steps:

| Step | Symbol |
|---|---|
| Departure check: does the rest of the pass still fit? | `_ferry_departure` |
| If not, re-plan the rest (plan arms trim the committed plan) | `FLScheduler.replan_remainder` → `_trim_plan` |
| Flight slot picks the next stop | `_ferry_next_stop` |
| On arrival, the slot picks the band (FX) or the (band, next stop) pair (FQ) | `_ferry_band_at_arrival`, `_ferry_pair_at_arrival` |
| Serve the contact | `_ferry_stop` → `HFLHostMission.run_contact` ([mission/host_mission.py](mission/host_mission.py)) |

The **flight slot** is set by `MuleConfig.flight_slot` and has three fillings:

| Arm | `flight_slot` | Class | File |
|---|---|---|---|
| F | `committed` | `CommittedSlot`: the plan's next stop, on the committed band | [policies/cross_heuristic.py](scheduler/policies/cross_heuristic.py) |
| FX | `cross_heuristic` | `CrossHeuristic`: on arrival, the fastest band that still reaches every member; at departure, the nearest stop whose move to the front keeps the plan feasible | [policies/cross_heuristic.py](scheduler/policies/cross_heuristic.py) |
| FQ | `pair_q` | `PairQSlot`: the learned score over masked (band, next stop) pairs, chosen on arrival, with FX as the fallback | [policies/pair_slot.py](scheduler/policies/pair_slot.py) |

`flight_slot_policy` builds F's and FX's slot. `selector/pair_features.build_pair_slot` builds FQ's,
called from [processes/mule.py](processes/mule.py) `_build_pair_slot`. The slot protocol is in
[policies/next_stop.py](scheduler/policies/next_stop.py). Pass 2 has no slot: it uses
`FLScheduler.build_pass_2_queue`, nearest first, on the committed band.

### Mission time

- [l1/mission_clock.py](l1/mission_clock.py): `MissionClock` (simulated time and energy ledger),
  `FlightModel`, `EnergyModel`, `zeng_power_w` (rotary-wing power).
- [mule/ferry.py](mule/ferry.py): `FerrySpec` and `FerryRuntime` connect the clock, the link and
  the payload for one mule.
- [mule/fit_clock.py](mule/fit_clock.py): `FitClock` gives devices training time on mission time.
  This is the paper's training-time study.

## 2. Routing

| What | File | Symbols |
|---|---|---|
| Plan search over band classes and stops | [plan/plan_search.py](scheduler/plan/plan_search.py) | `plan_search`, `search_class`, `_ClassSearch` |
| Tour building and 2-opt | [routing/two_opt.py](scheduler/routing/two_opt.py) | `nearest_neighbour`, `cheapest_insertion`, `two_opt`, `best_order`, `order_contacts` |
| Re-plan the rest of a pass | [routing/replan.py](scheduler/routing/replan.py) | `replan_route`, `ReplanResult` |
| Member subsets within a stop | [plan/member_subset.py](scheduler/plan/member_subset.py) | `admit_members`, `admit_stop`, `fold_members`, `trim_members` |
| Hover points for capped devices | [plan/hover.py](scheduler/plan/hover.py) | `best_hover_point`, `hover_stop` |
| Stops by RF range; Pass-2 order | [stages/s3a_cluster.py](scheduler/stages/s3a_cluster.py) | `cluster_by_rf_range`, `order_pass_2_greedy` |
| D4's CARP routing (baseline) | [policies/fedex_carp.py](scheduler/policies/fedex_carp.py) | `carp_search`, `carp_assign`, `carp_cost` |

## 3. The learned score (RL)

| Part | File | Symbols |
|---|---|---|
| Model: a masked pointer double DQN in numpy (2×64 tanh; Adam 1e-3; Huber loss; hard target sync every 500 updates) | [selector/pair_q.py](scheduler/selector/pair_q.py) | `PairQNet` (`q`, `q_target`, `targets`, `masked_argmax`), `PairQLearner.observe`, `BehaviourSchedule`, `PairQConfig` |
| Features `pair_v1`: one row per (band, next stop) pair | [selector/pair_features.py](scheduler/selector/pair_features.py) | `PairFeatureSchema`, `pair_rows`, `covering_classes`, `LearnedPairScorer` |
| Replay with variable-size candidate sets (50k transitions, batches of 64) | [selector/pair_replay.py](scheduler/selector/pair_replay.py) | `PairReplay`, `PairTransition`, `PairBatch` |
| Slot that flies the score (mask, fallback, closing record) | [policies/pair_slot.py](scheduler/policies/pair_slot.py) | `PairQSlot`, `PairStep`, `closed_record` |
| Reward: for each Pass-1 decision, merge weight collected − c_t·time (− c_cov·uncovered weight at the sortie's end) | [../experiments/ferrysim/reward.py](../experiments/ferrysim/reward.py) | `RewardSpec`, `StopRecord`, `SortieRecord`, `sortie_rewards`, `episode_return`, `raw_merge_weights` |
| Training loop: episodes, behaviour (ε-greedy around FX, then on Q), validation, checkpoint | [../experiments/ferrysim/train.py](../experiments/ferrysim/train.py) | `train`, `TrainSpec`, `pair_spec`, `fly_pair_episode`, `pair_transitions`, `validate` |
| One episode (one trial flown by one policy) | [../experiments/ferrysim/episode.py](../experiments/ferrysim/episode.py) | `run_episode`, `Policy`, `EpisodeResult` |
| Cells and seed streams (train / validation / held-out kept apart) | [../experiments/ferrysim/cells.py](../experiments/ferrysim/cells.py) | `FerryCell`, `train_episode`, `stream_seeds`, `check_disjoint` |
| Real stack run in one process | [../experiments/ferrysim/inprocess.py](../experiments/ferrysim/inprocess.py) | `run_trial`, `InProcessOrchestrator`, `World` |
| Checkpoints, manifests, held-out scoring | [../experiments/ferrysim/checkpoints.py](../experiments/ferrysim/checkpoints.py), [selector/pair_q.py](scheduler/selector/pair_q.py) | `evaluate_checkpoints`, `held_out_score`, `verify_checkpoint`, `read_manifest` |
| Held-out evaluation; headroom over FX; verdict report | [evaluate.py](../experiments/ferrysim/evaluate.py), [headroom.py](../experiments/ferrysim/headroom.py), [report.py](../experiments/ferrysim/report.py) | `evaluate`, `headroom_report`, `decide` |

E3 (Chen et al.'s DQN) is in [policies/chen_dqn.py](scheduler/policies/chen_dqn.py) as
`ChenDQNPolicy` with `e3_rows`. It trains with the same learner on its own per-stop rows and a
bytes reward: `train.e3_spec`, `fly_e3_episode`, `e3_transitions`.

Command line: `python -m experiments.ferrysim <command>`, where the command is one of `train`,
`sweep`, `evaluate`, `report`, `headroom` or `pilot`. The parser is in
[../experiments/ferrysim/\_\_main\_\_.py](../experiments/ferrysim/__main__.py).

## 4. Arms: paper label → code

[../experiments/exp4/driver.py](../experiments/exp4/driver.py) turns each arm label into mule
settings through these tables:

- `PLAN_ARMS`, `PAIR_ARMS`, `ADDENDUM_PLAN_ARMS` and `is_plan_arm`: which arms fly the plan clock.
- `_PLAN_ARM`: each plan arm's plan fields.
- `_ARM_POLICY`: each whole-scheduler baseline's `contact_policy`.
- `_ARM_SCORE` and `_ARM_LAW`: ablations of the plan score and of the deadline law.

The mule process builds the policy in [processes/mule.py](processes/mule.py), in
`_build_target_selector` and `_learned_fillings`.

| Arm | What it is | Code |
|---|---|---|
| **F** | Plan clock, committed flight slot | `FLScheduler.build_ferry_plan` + `CommittedSlot` |
| **FX** | F with the cross heuristic | `CrossHeuristic` ([cross_heuristic.py](scheduler/policies/cross_heuristic.py)) |
| **FQ** | F with the learned pair score | `PairQSlot` + `PairQNet` ([pair_slot.py](scheduler/policies/pair_slot.py), [pair_q.py](scheduler/selector/pair_q.py)) |
| H1 | HERMES scheduler: deadline buckets, S3b gate, nearest first | `FLScheduler.build_contact_queue` + `filter_feasible` |
| H0 | Live-link reference, no mule | driver (`arm == "H0"`) |
| D1 | MAX-AoI | `MaxAoIPolicy` ([max_aoi.py](scheduler/policies/max_aoi.py)) + `greedy_budget_walk` ([budget_walk.py](scheduler/policies/budget_walk.py)) |
| D2 | Oort | `OortPolicy` ([oort.py](scheduler/policies/oort.py)) |
| D3 | Whittle index over age | `WhittlePolicy` ([whittle.py](scheduler/policies/whittle.py)) |
| D4 | FedEx route (CARP), with FeRRy's merge | `FedExCarpPolicy` ([fedex_carp.py](scheduler/policies/fedex_carp.py)) |
| D4_FedEx | D4 with FedEx-Async's own merge | the same policy + `--aggregation agg:fedex` (launcher tag `D4fedex`) |
| D5 | FedCS, degraded to a data mule | `FedCSDegradedPolicy` ([fedcs_degraded.py](scheduler/policies/fedcs_degraded.py)) |
| E3 | Chen et al.'s DQN | `ChenDQNPolicy` ([chen_dqn.py](scheduler/policies/chen_dqn.py)) |

These are the ablation and study arms, all in `driver.py`:

- `FB+wide`, `FB+medium`, `FB+narrow`: band pinned.
- `F-cov`, `F-cap`, `F-prio`: one plan term removed.
- `F-round`, `F-pref`: alternative deadline forms. `F-add` is a launcher label for F with
  `--deadline-law additive`.
- `FX-dwell`, `FX-cov`: objective terms.
- `F+L1`, `H1+L1`: adaptive backhaul.
- `FQ-g*`: the γ sweep.

## 5. Merges

[mission/aggregation_rules.py](mission/aggregation_rules.py) is the registry behind
`ClusterConfig.aggregation`. Its module docstring gives the equations.

| Rule | Used by | Notes |
|---|---|---|
| `agg:cutoff` | FeRRy, and every arm unless stated | FedAsync hinge `staleness`; weight 0 past the device's cutoff (`age_cap`) |
| `agg:fedex` | D4_FedEx | FedEx-Async's server step |
| `agg:plain`, `agg:asynchfl`, `agg:fedbuff` | Merge-rules study | the `agg:cutoff+fedprox` variant adds the devices' proximal term |

- **Mule merge:** `merge_on_mule` and `update_weights` in
  [aggregation_rules.py](mission/aggregation_rules.py). They are called from
  `HFLHostMission` ([mission/host_mission.py](mission/host_mission.py)), with the plain path in
  [mission/partial_fedavg.py](mission/partial_fedavg.py).
- **Server merge:** `HFLHostCluster.aggregate_pending` and `_aggregate_age_aware`
  ([cluster/host_cluster.py](cluster/host_cluster.py)), plus
  [cluster/cross_mule_fedavg.py](cluster/cross_mule_fedavg.py).

## 6. Radio (L1)

| What | File | Symbols |
|---|---|---|
| Band classes, range–rate model, CQI table | [l1/contact_link.py](l1/contact_link.py) | `BandClass`, `ContactLink`, `CQIEntry` |
| Per-contact SNR (shadowing, interference), backhaul channel, loss | [l1/channel_model.py](l1/channel_model.py) | `ContactChannel`, `BackhaulChannel`, `loss_from_snr`, `backhaul_plan` |
| Adaptive backhaul controller (`+L1` arms) | [l1/channel_utility.py](l1/channel_utility.py) | `AdaptiveChannelController` |
| RF prior from past uploads | [l1/rf_prior.py](l1/rf_prior.py) | `RFPriorStore`, `RFPriorProducer` |

## 7. Mission, processes, transport

- **One mission's FL** ([mission/](mission/)):
  - `HFLHostMission`, the mule-side server: `run_contact`, `open_pass_2`, `deliver_contact`,
    `close_round`.
  - `ClientMission`, the device.
  - `ContactPlan`, what one stop needs: members, SNR, dwell.
- **Mule** ([mule/](mule/)):
  - `MuleSupervisor` runs the mission.
  - `ClientCluster` handles the dock: the UP and DOWN bundles.
- **Edge server** ([cluster/](cluster/)): `HFLHostCluster` and `DeviceRegistry`.
- **Processes** ([processes/](processes/)):
  - `MultiProcessOrchestrator` starts one cluster process, K mule processes and N device processes.
  - [config.py](processes/config.py) holds `TopologyConfig`, `ClusterConfig`, `MuleConfig` and
    `DeviceConfig`, with their validation.
- **Transport** ([transport/](transport/)):
  - RF link (mule ↔ device) and dock link (mule ↔ server), each loopback or TCP.
  - [wire.py](transport/wire.py) frames the messages.
- **Types** ([types/](types/)): FL messages, bundles, `PlanCommit`, round reports.
  [signatures.py](types/signatures.py) checks bundle integrity only.
- **Observability** ([observability/](observability/)): `JsonEventEmitter`, the per-process event
  stream that the trace scorer reads, and `MetricsRegistry`.

## 8. Experiments and the paper's comparisons

| What | File |
|---|---|
| Trial CLI | [../experiments/exp4/runner_main.py](../experiments/exp4/runner_main.py) |
| Trial driver: arm → config, the trial clock, kept traces | [../experiments/exp4/driver.py](../experiments/exp4/driver.py) (`Exp4Driver`) |
| Field (device and dock positions) | [../experiments/exp4/topology_builder.py](../experiments/exp4/topology_builder.py) |
| DNN-IDS learning task, data and model | [../experiments/exp4/model_task.py](../experiments/exp4/model_task.py), [prep.py](../experiments/exp4/prep.py) |
| Non-IID shards (label and quantity skew) | [../experiments/exp4/partition.py](../experiments/exp4/partition.py) |
| Device compute time | [../experiments/exp4/compute.py](../experiments/exp4/compute.py) |
| Trial footprint (processes, memory) | [../experiments/exp4/footprint.py](../experiments/exp4/footprint.py) |
| Trace scorer (time to τ, reach, ages, cap violations, band shares) | [../experiments/analysis/traces_scorer.py](../experiments/analysis/traces_scorer.py) |
| Statistics (paired Wilcoxon, Cliff's δ, Holm, bootstrap) | [../experiments/analysis/stats.py](../experiments/analysis/stats.py) |
| Campaign launcher: one command per stage | [../scripts/exp5/launch.py](../scripts/exp5/launch.py), [params.toml](../scripts/exp5/params.toml) |
| Study scoring: paired comparisons against F | [../scripts/exp5/scoring.py](../scripts/exp5/scoring.py) |
| Paper tables and figures | [paper_tables.py](../scripts/exp5/paper_tables.py), [paper_figures.py](../scripts/exp5/paper_figures.py) |
| FQ vs E3 (exploratory) | [../scripts/exp5/fq_vs_e3.py](../scripts/exp5/fq_vs_e3.py) |

Each of the paper's studies is a section of [params.toml](../scripts/exp5/params.toml), scored
under `[score.<section>]`:

| # | Study | Section(s) |
|---|---|---|
| 1 | Merge rules | `s51` |
| 2 | Deadline forms | `s52` |
| 3 | Scheduler comparison | `s53`, `s53x` |
| 4 | Band pinning | `s54` |
| 5 | Learned score | `rl` stages (`rl-headroom`, `rl-calibrate`, `rl-sweep`), `s55`, `s56` |
| 6 | Objective terms | `s57`, `rl.s57` |
| 7 | Fairness | `s58` |
| 8 | Scale | `s59`, `s59x` |
| 9 | Decision cost | `s511a`, `s511b`, `s511c`, `p511c` |
| 10 | Training time | `s512`, `p512` |
| 11 | Non-IID | `s513` |
| 12 | Component ablations | `s514` |
| 13 | Radio | `s515`, `p515` |
| – | FQ vs E3 (exploratory) | `rl.e3`, `fq_vs_e3.py` |

## 9. Paper term → code term

| Paper | Code |
|---|---|
| mission time | `MissionClock` (the "simulated clock" in docstrings) |
| feasibility test | S3b: `FeasibilityModel` |
| stop | `ContactWaypoint` (one position and the devices in range of it) |
| band class | `BandClass`, named `wide` / `medium` / `narrow` |
| plan score V, plan key | `plan_score.score`, `plan_score.plan_key` |
| departure check | `MuleSupervisor._ferry_departure` |
| re-plan trim | `FLScheduler._trim_plan` |
| age cap S | S3d: `s3d_age_cap` |
| Pass 1 / Pass 2 | `MissionPass.COLLECT` / `MissionPass.DELIVER` |
| mule merge / server merge | `merge_on_mule` / `HFLHostCluster.aggregate_pending` |

## 10. Not on the FeRRy path

These modules are kept for the earlier HERMES experiments, and no Exp 5 arm uses them:

- `scheduler/selector/`: `ddqn.py`, `target_selector_rl.py`, `selector_train.py`, `sim_env.py`,
  `features.py` and `replay.py`, the HERMES-era selector of arm H2. `sim_env.py`'s reward
  constants are ported as the reward of the `FQ-hand` arm.
- `scheduler/stages/`: `s2a_readiness.py`, `s2b_flag.py` and `s35_selector.py`, HERMES pipeline
  stages that are not on the plan path. `s3c_mission_window.py` adapts the deadline window per
  mission and is off by default (`mission_window_adaptation = False`).
- `scheduler/policies/`: `arrival_order.py` and `edf_feasibility.py`, the Experiment-3 ablation
  arms (used by `experiments/exp3`).
- `l1/channel_ddqn.py`: the channel-only DDQN. If one is wired, the mule consults it at each
  visit, but only records its choice. No arm acts on it.
- `transport/channel_emulator.py`: a no-op stub.
- `transport/cloud_link.py`: the Tier-3 cloud link. It connects only when
  `ClusterConfig.tier3_url` is set, and the Exp 5 runs leave that unset.
- Each package's `__main__.py`: a small phase demo, for example `python -m hermes.mule`.
