# HERMES Paper vs. Code Audit (SEC 2026 submission #74)

**Generated:** 2026-07-21, during the SEC 2026 rebuttal window (Jul 18-22).
**Scope:** every factual claim in sec26-paper74.pdf about the implementation, checked
against the working tree at FL-DNN-GAN-IDS/. Produced by an 18-agent audit; the
highest-impact findings (Exp 1 platform, Exp 3 scheduler scope, A1 dead-zone default,
deadline sign error) were re-verified by hand against source.

**Companion document:** SEC26_Rebuttal_Draft.md

**Health warning:** sections E and H flag artifacts whose producing code is NOT in this
repo (Table II / Figs 3-4, Table VI / Fig 7). Those rows are inference from the paper
text, not from code. Do not cite them as established.

---

# HERMES #74 — Rebuttal Consolidation Brief (T1 + T2)

Every code claim below was re-verified against the working tree at `D:\networkIntrusionDetectionSystem\FL-DNN-GAN-IDS`. Paper quotes are from `Presentation Documents\sec26-paper74.pdf` (13 pp., text extracted).

---

## A. The deadline formula, as actually implemented

**True formula** — `hermes/scheduler/stages/s3_deadline.py:68-78`:

```
if state.deadline_override_ts is not None:      # cluster amendment wins, no arithmetic
    return state.deadline_override_ts
fulfilment = max(5.0, state.deadline_fulfilment_s)
return now + fulfilment - compute_idle_time(state, now)
```

with `compute_idle_time` = `0.0` if `idle_time_ref_ts <= 0.0` else `max(0.0, now - idle_time_ref_ts)` (lines 63-65).

| Symbol in paper | Actual code object | Unit | Sign | Default / bounds |
|---|---|---|---|---|
| `t_base` | `now`, from `now_fn=time.time` (`fl_scheduler.py:80`), re-sampled at **every queue build** (lines 209/241/340/457) | epoch s | + | not a per-round base timestamp; design doc §9 Q1 (`HERMES_FL_Scheduler_Design.md:684`) still lists this as unresolved |
| `Φ(j)` | `DeviceSchedulerState.deadline_fulfilment_s` — a **per-device response-time window in seconds**, not a reliability statistic | s | + | init **60.0** for every device (`hermes/types/scheduler.py:92`); read-time clamp `max(5.0, ·)` |
| `ι(j)` | `now - idle_time_ref_ts`, where the ref is written **only** in the CLEAN branch (line 149 — the sole write in all of `hermes/`) | s | − | never-contacted device gets `ι = 0` via a `<= 0.0` sentinel, not a large idle |

**Fast-phase constants** (`s3_deadline.py:44-48, 147-159`): CLEAN → `Φ = max(5.0, Φ − 5.0)`; PARTIAL **or** TIMEOUT (one shared `else`) → `Φ = Φ + 10.0`. `FAST_PHASE_ON_TIME_SHRINK_S = 5.0`, `FAST_PHASE_MISSED_WIDEN_S = 10.0`, `MIN_DEADLINE_FULFILMENT_S = 5.0`, **no upper cap**. CLEAN also sets `is_new = False` and re-anchors `idle_time_ref_ts = delta.contact_ts`; the non-clean branch deliberately does not (`s3_deadline.py:132-135`: *"a failed attempt doesn't reset the 'when were you last reliable' clock"*).

**Does a clean contact move the deadline earlier or later? Later, at any realistic revisit interval.** Once `idle_time_ref_ts > 0`, the `now` terms cancel: **deadline = `idle_time_ref_ts + Φ`**. A CLEAN at time `t` sets ref = `t` and `Φ → Φ−5`, so the new deadline is `t + Φ − 5`. The shift relative to the previous deadline is `(gap − 5) s`, where `gap` = seconds since the last clean contact. Measured in the repo venv: device with ref=1000, Φ=60, evaluated at now=1120 → deadline 1060.0 (60 s overdue); after a CLEAN at 1120 → Φ=55, ref=1120, deadline **1175.0** (+115 s, slack flips −60 → +55). It moves *earlier* only when `gap < 5 s`. Conversely five consecutive TIMEOUTs at 120 s spacing grow Φ 70→110 but leave the anchor stale, so slack-vs-now goes −50 → −490: **a persistently failing device becomes monotonically more overdue, i.e. more urgent.** The correct statement is that a CLEAN tightens the *next service interval* (from Φ to Φ−5, measured from the contact), not that it moves the absolute deadline earlier.

**Path dependence.** The 5 s floor discards shrink credit that the uncapped widen retains, so Φ is a lossy, order-dependent function of history: two devices with identical records (20 CLEAN + 1 TIMEOUT) end at Φ = 15.0 s (cleans first) vs Φ = 5.0 s (timeout first) — a 10 s deadline spread at identical reliability.

**Other side conditions the printed equation omits.** (i) `deadline_override_ts` short-circuits everything and **nothing in `hermes/` ever clears it back to `None`** (only write: `s3_deadline.py:188`) — the bypass is sticky. (ii) The cluster can overwrite Φ wholesale via `registry_deltas['deadline_fulfilment_s']` (`s3_deadline.py:198-203`), so Φ is not purely mule-local. (iii) A device that passes S1 on an override alone but has no bucket is silently dropped with a warning (`fl_scheduler.py:256-259`).

### What the paper says that is wrong

| Paper location | Text | Problem | Replacement |
|---|---|---|---|
| §III-B, p.6 | "Φ(j) captures historical participation reliability" | Φ is a seconds-valued window, history-independent at init, path-dependent thereafter. The repo *does* compute a real reliability statistic — `on_time_count/(on_time_count+missed_count)` in `selector/features.py:80-94` — and `compute_deadline` never reads it. | "Φ(j) is an adaptive per-device fulfilment window (seconds), initialised to 60 s and updated per contact outcome by Δ_shrink = 5 s (floored at 5 s) and Δ_widen = 10 s (unbounded); ι(j) is the elapsed time since that device's last *clean* contact." |
| §III-B, p.6 | "successful contacts shorten future deadlines while failed or partial sessions increase deadline slack" | True of Φ, **inverted** for the emitted deadline timestamp (CLEAN re-anchors ι in the same branch). | "a clean contact tightens the device's next service interval from Φ to Φ−5 s measured from the contact; a failed or partial contact widens Φ by 10 s but leaves the reliability anchor stale, so the device's urgency continues to rise." |
| Alg. 1 L19 / L21, p.7 | `Φ(j) += ∆+` on clean; `Φ(j) −= ∆−` otherwise | **Signs inverted** vs the code and vs the paper's own p.6 prose. | `Φ(j) ← max(Φ_min, Φ(j) − ∆_shrink)` on clean; `Φ(j) ← Φ(j) + ∆_widen` otherwise. Rename the symbols so sign is not carried by superscripts. State ∆_shrink = 5 s, ∆_widen = 10 s, Φ_min = 5 s, no upper bound. |
| Alg. 1 L1, p.7 | `Φ, ι ← FOLDAMENDMENTS(A)` | `fold_cluster_amendment` (`s3_deadline.py:184-215`) never writes `idle_time_ref_ts`. The only write is in the fast-phase CLEAN branch. | `Φ ← FOLDAMENDMENTS(A)`; move the ι update into the clean branch of the fast-phase loop (L18-19). |
| Alg. 1 L5, p.7 | `Deadline(c) ← t_base + Φ(c) − ι(c)` (unconditional, per cluster) | Two-case in code (override wins); computed **per device**, then a contact inherits the `min` over members (`s3a_cluster.py:169`). | Print it as a two-case definition and add one line: "a contact region inherits the tightest member deadline." |
| §III-B / Alg. 1 | `t_base` implied to be a per-round base timestamp | Wall-clock `time.time()` re-sampled per queue build; the team's own design doc still lists the clock as open question Q1. | Define it as wall-clock sampled at queue-build time, or resolve Q1 before camera-ready. |

Also disclose that PARTIAL and TIMEOUT share one branch (`round_report.py:26-28` defines three outcomes; the deadline logic distinguishes two).

---

## B. The buckets, as actually implemented

**Three**, defined as a str-Enum (`hermes/types/scheduler.py:31-33`); the classifier `classify_bucket` (`s3_deadline.py:85-110`) is the only bucket classifier in the package and is a nine-line short-circuiting chain with a hard `ValueError` fallthrough:

| Order | Bucket | Admission test | Semantics |
|---|---|---|---|
| 1 | `NEW` | `state.is_new` | registered but never served |
| 2 | `SCHEDULED_THIS_ROUND` | `state.is_in_slice` | assigned to the current mission slice |
| 3 | `BEACON_ACTIVE` | `last_beacon_ts > 0 and (now − last_beacon_ts) <= beacon_window_s` (default 30 s) | opportunistic RF beacon heard |

Priority order is positional in `BUCKET_PRIORITY = (NEW, SCHEDULED_THIS_ROUND, BEACON_ACTIVE)` (`scheduler.py:48-53`); lower index drains first. The signature takes only `state`, `now`, `beacon_window_s` — **no deadline, no idle time, no utility is passed in or read.** Because `is_new` is tested first, an urgently-overdue slice member that is still `is_new` files under NEW. A `PARTIAL`/`TIMEOUT` never clears `is_new`, so a device that has only ever timed out stays in NEW indefinitely. A contact waypoint inherits the **highest-priority** (worst) bucket among its members (`s3a_cluster.py:72-74`) and the **minimum** member deadline (`:169`).

**How buckets combine with the deadline sort:** bucket-first outer loop over `BUCKET_PRIORITY`; intra-bucket ordering is Euclidean distance + `device_id` (`s35_selector.py:39-45`), or descending Q-value under the learned selector (`target_selector_rl.py:291-295`). The computed deadline is written into `TargetWaypoint.deadline_ts` as payload and is **never a sort key** in `build_target_queue` (`fl_scheduler.py:296-303`). Its only ordering influence on the default path is the third and last element of the S3a anchor key `(-delivery_priority, bucket_priority, deadline_ts)` (`s3a_cluster.py:96`). The selector's 11-dim feature vector contains no deadline and no idle-time term. The one component that genuinely sorts on deadline is the A3 `EdfFeasibilityPolicy` (`policies/edf_feasibility.py:141-144`) — and it is injected into the same per-bucket slot, so it is EDF *within* bucket, not global EDF.

**FL utility is not a partition axis and not a ranking term** — it is a binary gate, `adv.utility > 0.60` (`s2b_flag.py:23,34`), on a different code path (contact time), never called from any queue builder. `DeviceSchedulerState.last_utility` is written (`s3_deadline.py:145`) and never read as an input to any decision.

### What the paper says that is wrong

- §III-B, p.6: *"devices are subsequently partitioned into priority buckets according to urgency and expected FL contribution quality."* Neither factor appears in `classify_bucket`. Replacement: **"devices are partitioned into three hard-rank tiers by participation state — never-served (NEW), assigned to the current slice (SCHEDULED_THIS_ROUND), and opportunistically beacon-detected (BEACON_ACTIVE). Deadline urgency and FL utility are handled outside this partition: the deadline is computed per device in the same stage and enters ordering as the final tiebreaker in contact-anchor selection, while the FL-utility threshold is an admit/reject gate applied at contact time."**
- Alg. 1 L7 places `BUCKETCLASSIFY(C)` **after** `CLUSTERBYRANGE` (L3). Not executable: `cluster_by_rf_range` raises `ValueError` if buckets are unset (`s3a_cluster.py:165-167`). Swap L3 and L7. (The stale module docstring at `s3a_cluster.py:5-7` is the likely source of the error and contradicts the function docstring 100 lines below it.)
- Internal inconsistency to fix at the same time: `HERMES_FL_Scheduler_Design.md:543` defines the tiers as *"1=new, 2=shortened-deadline on-time, 3=in-session awaiting"* — not what the code implements.

---

## C. The stage pipeline

**Six stage modules exist**: `hermes/scheduler/stages/{s1_eligibility, s2a_readiness, s2b_flag, s3_deadline, s3a_cluster, s35_selector}.py`. There are two competing "four"s, and the paper never enumerates either.

- **The design doc's four** (`HERMES_FL_Scheduler_Design.md:45`, echoed at `README.md:75`): S1 → S2A → S2B → S3, *"with an S3.5 intra-bucket selector and an S3a contact-clustering pass"* as extras.
- **The four the per-round scheduler actually runs** (`fl_scheduler.py:320` docstring and `:342-373` body): **S1 → S3 (bucket + deadline) → S3a → S3.5**. This is the set Algorithm 1 labels.

| Stage | Gates on | Kind |
|---|---|---|
| **S1** `filter_eligible` | `is_in_slice OR deadline_override_ts is not None OR fresh beacon` — deliberately coarse | hard rule |
| **S3** `classify_bucket` + `compute_deadline` | three lifecycle booleans; deadline computed and cached | hard rule |
| **S3a** `cluster_by_rf_range` | groups eligible devices within `rf_range_m` into contact waypoints; anchor key `(-delivery_priority, bucket_priority, deadline_ts)`; contact inherits worst bucket, min deadline | hard rule |
| **S3.5** `TargetSelectorRL.rank_contacts` (or `ArrivalOrder` / `EdfFeasibility`) | ranks **within** the current bucket only; skipped entirely when a bucket has < 2 candidates (`fl_scheduler.py:412-416`) | **the only learned stage** |
| **S2A** `is_on_contact_ready` — advert eligible + 5 s freshness | **not on any queue path**; sole caller `FLScheduler.ingest_ready_adv`, whose only callers repo-wide are `tests/unit/test_fl_scheduler.py:145,157,169` |
| **S2B** `passes_fl_threshold` — strict `utility > 0.60` | same; `DEFAULT_FL_THRESHOLD = 0.60` is dead at runtime. The live gate is an inline duplicate at `hermes/mission/host_mission.py:249` and `:442`, `adv.utility < min_utility` with `min_utility: float = 0.0` (`:217`, `:347`) and no caller overriding it — **the effective utility gate in every run is `utility >= 0.0`, a no-op** |

**Bounds on the learned component.** Design principle 12 (`HERMES_FL_Scheduler_Design.md` §7): *"`TargetSelectorRL` is bounded to intra-bucket ordering… It cannot promote a gated-out device, cannot reorder buckets, and cannot override a deadline."* Enforcement in code is partial: `_enforce_collect_pass` raises `SelectorScopeViolation` if any selector entry point is called with `pass_kind=DELIVER` (`target_selector_rl.py:60-66`, pinned by four tests in `tests/unit/test_selector_pass_gate.py`) — this is real and non-vacuous. `assert_candidates_admitted` is a set-membership check that defaults `admitted = candidates`, i.e. tautological unless a caller supplies a narrower set (`target_selector_rl.py:154-156, 285-287`). "Cannot reorder buckets" is enforced by the *caller's* per-bucket loop, not by the guard. `select_server` bypasses both guards (`target_selector_rl.py:326-332`).

**Critical scope note for T2:** in the paper's headline scheduling experiment (Table VII, Figs 8-10) **only S3a runs**. `experiments/exp3/sim_env.py:54,453` imports and calls `cluster_by_rf_range` and nothing else from the scheduler; `FLScheduler`, `filter_eligible`, `classify_bucket`, `compute_deadline`, and `fold_round_close_delta` are never invoked from `experiments/`. Deadlines in Exp 3 are static per-device windows, `self._deadlines[did] = now + mission_budget_s*0.5*β*(0.5|1.5)` (`sim_env.py:426-447`), and every device is stamped `is_new=False`, `bucket = Bucket.SCHEDULED_THIS_ROUND` (`:433-441`) — **one bucket, no adaptive deadline, no fast-phase fold in the reported results.**

---

## D. Design rationale a reviewer would accept

| Item | One-sentence engineering reason | Grounding |
|---|---|---|
| **Fulfilment-window direction** (clean shrinks, miss widens) | A device that just delivered cleanly has demonstrated it can be served on the current cadence, so its window tightens; a device that timed out gets a wider window so the mule stops burning flight time returning to a flaky node every round. | **NOT stated anywhere in code or docs — authors must supply.** The only adjacent comment is the ι semantics at `s3_deadline.py:132-135` ("when were you last reliable"). Supply this rationale or the sign-flip fix in §A looks arbitrary. |
| **Asymmetric 5 s / 10 s constants** | Magnitudes are deliberately small "so the window drifts, not whipsaws" (`s3_deadline.py:42-43`). | Comment covers *smallness only*. **The 2:1 widen bias and the absence of an upper cap have no stated rationale — authors must supply.** Defensible framing: widening on evidence of failure must outpace tightening on evidence of success because a missed contact costs a wasted flight leg, whereas an over-wide window costs only latency. If they cannot defend an unbounded Φ, add a cap and say so. |
| **5 s floor** | A zero or negative window would place the deadline at or before `now`, making every device permanently overdue and collapsing the S3a deadline tiebreaker and the A3 EDF ordering into noise. | Grounded: `s3_deadline.py:47-48` — *"Floor on the fulfilment window — never let the formula drive it to zero."* |
| **Bucket taxonomy (provenance, not urgency)** | The tiers encode non-negotiable coverage obligations — never-served devices first (anti-starvation), then devices the cluster contractually assigned to this mission slice, then beacons as a free bonus — and are the only hard rank tiers precisely so a learned policy cannot trade them away. | Grounded: `HERMES_FL_Scheduler_Design.md:62` ("these are the only hard rank tiers"), principle 6 (beacons are *"a bonus when the mule is already in range, never a summon signal"*), principle 12. |
| **RL scope bound** | Learning is confined to one explicit sub-stage after every deterministic gate so hard rules stay hard and the learned policy is auditable; a bounded action space is also what makes the complexity claim true (209 parameters, < 200 ops per decision). | Grounded: principle 12 verbatim; `ddqn.py:20-21`; runtime `SelectorScopeViolation` on Pass-2 misuse with four pinning tests. **Caveat to disclose:** the admitted-set guard is vacuous by default, and in the Exp-3 driver the selector is handed the full unvisited contact list (`experiments/exp3/arm_mule.py:117-129`) — intra-bucket confinement holds there only because the sim populates a single bucket. |

---

## E. Provenance table

| Paper artifact | Where it actually ran | Measured vs modeled |
|---|---|---|
| Fig. 1, Fig. 2 (architecture) | n/a | Diagrams. |
| Table I (feature comparison) | n/a | Literature survey. |
| **Eq. (1) `U(c,t)` RF utility model** | **No implementation in this repo.** `hermes/l1/` contains only `channel_ddqn.py` (inference-only `ChannelDDQN`, 8-feature state, 3 bands 3.32/3.34/3.90 GHz, **no `update`, no trainer**) and `rf_prior.py` (read-only SNR store). No file computes `R(·)`, `g(c)`, `κ(c)`, or `λ(c,t)`. | Unverifiable from code. |
| **Table II** (offload MB, avg rate, avg eff. SNR, switches for Fixed CH1/CH2/CH3, HERMES-Heuristic, static oracle) | **Not reproducible from this repo** — no L1 experiment script, no SNR/mobility trace file, no offload log, no analysis script. Repo-wide search for `*.csv/*.npy/*.npz` returns only Exp-1 artifacts. Paper text says gains and costs are *"calibrated from AERPAW telemetry collected during preliminary measurement passes"* and that four policies are compared *"under the same UAV mobility trace and SNR sequence."* | Reads as **trace-driven offline replay**: one recorded SNR/mobility sequence, four policies scored against a rate-tier mapping. Under that reading, telemetry (SNR sequence, mobility) is measured; offload MB, rate tier, and switch counts are **computed from a model**, not bytes counted on a radio. **The authors must state which parts were real AERPAW SDR passes (dates/nodes/reservation) and which were replay — nothing in the repo settles it.** |
| Fig. 3 (cumulative offload), Fig. 4 (rate-tier occupancy) | Same source as Table II; same gap. | Same. |
| **Table V, Fig. 5, Fig. 6** (FL vs Centralized comm baseline) | **Chameleon Cloud, multi-host, real TCP over a `tc/netem`-shaped 10 Mbps link.** Topology `experiments/exp1/setup/configs/exp1_chameleon.json` pins four client hosts at `129.114.108.91 / .191 / 109.6 / 108.11` (Chameleon address space); results are `results/exp1_chameleon*.csv`; `AppSetup/` ships both `Chameleon_node_Setup.py` and `AERPAW_node_Setup.py` (the latter is an apt/pip installer only). **No AERPAW artifact of any kind.** | **Measured:** end-to-end wall-clock per trial, server-side `time.perf_counter()` (`server.py:285-301, 349-388`), 1 server + 4 clients, n=20 paired trials/cell, 1080 rows. **Modeled/synthetic:** the payloads — clients ship a **zero-filled filler buffer** (`client.py:56-58, 191-195`, docstring: *"we're measuring transport, not training"*); `θ` size is a constant `--theta-bytes` default **200 000 B**, ~10× the project's canonical CIC-IoT NIDS model (~4.7 K params, ~18.8 KB); **no training and no aggregation occur**, so `T_agg` in Table IV is structurally zero; `D = α·T_c` and the energy proxy come from `experiments/calibration.toml` (P_idle = 5 W and P_tx = 12 W are explicitly **modeled** within the Ettus 18 W envelope, ε_bit derived). |
| **Fig. 7** (RL training dynamics), **Table VI** (Return / Jobs Done / Jobs Failed, 5 jobs, 100 m × 100 m, mean over two seeds) | **No code in this repo produces these metrics.** Repo-wide search for `jobs_done`, `jobs_failed`, `n_jobs`, `overloaded` returns nothing; `experiments/exp3/metrics.py` has no return/job fields; the repo's EDF policy is a within-bucket contact ranker, not a job scheduler. The nearest artifact is `TrainMetrics.mean_reward_by_episode` (`selector_train.py:68-70`) from a 400-episode, 8-device `ContactSim` run — different scenario, different metrics. | **Provenance unknown — authors must identify the producing codebase.** |
| **Table VII, Fig. 8, Fig. 9, Fig. 10** (A1-A4 sweep, 8 640 paired trials) | **Local pure-Python simulator.** All mule arms run `experiments/exp3/sim_env.py::Exp3Sim`, which composes `hermes/scheduler/selector/sim_env.py::ContactSim`; A1 is *also* simulated (`arm_a1.py`, docstring: *"Why a sim rather than driving the real Flower path"*). No AERPAW, no ROS/Gazebo, no network I/O, no radios. The paper's own §IV-A wording — *"an abstracted mobile-relay simulation adapted from prior drone-relay frameworks"* — is the accurate description. | **Measured:** nothing physical. **Modeled:** transit/collect times, Bernoulli completion (`reliability ~ U(0.15,1.0)`), upload rate with ±20 % Gaussian jitter, jittery regime (2 % loss + 30 % latency jitter applied to the long-range mule→BS link only), and the A1 collapse mechanism (`long_range_link_quality = 0.4` **plus** a persistent dead-zone fraction). Energy in J/Δθ is `T·P_idle + B·ε_bit + L·ε_prop` with `ε_prop = 10.0 J/m` labelled `REPLACE-FROM-PLATFORM-SPEC` in `calibration.toml`. **No per-trial CSV for Exp 3 exists in the repo** and **no trained A4 weight file (`.npz`) exists anywhere.** |
| **A4 selector training** | `experiments/exp3/train_a4.py` → `train_selector_contact` → `ContactSim`. That file's own docstring: *"The real training domain is the AERPAW digital twin (Phase 6)… This is not physics."* | Modeled end to end; single seed (default 0), and the DDQN weight init is **unseeded** (`TargetSelectorRL(rng_seed=…)` seeds only ε-greedy; `DDQN(feature_dim=FEATURE_DIM)` gets no seed). |

---

## F. Overclaims the authors must walk back

Ordered most damaging first.

1. **"The experiment is conducted in the AERPAW digital twin environment using four edge devices and one edge server connected through a shared wireless segment."** (§IV-A, p.8, for the Table V / Fig 5 / Fig 6 baseline.)
   *Unsupported:* every artifact says Chameleon Cloud — `exp1_chameleon.json` with `129.114.108.x` hosts, `results/exp1_chameleon*.csv`, and a `Chameleon_node_Setup.py` sibling to the AERPAW installer. The team's own `HERMES_Experiments_Implementation_Plan.md` §6 is titled **"AERPAW availability caveat"** and states: *"The paper says experiments run 'on the AERPAW wireless digital twin.' The system implementation plan notes the testbed is currently down,"* then lists "run on local emulation, document" as an option requiring a disclosure note. One `grep` of the results directory ends this.
   *Replacement:* "The communication baseline is executed on four bare-metal Chameleon Cloud nodes plus one server, over real TCP on a `tc/netem`-shaped 10 Mbps link; AERPAW deployment is future work." Add that payloads are fixed-size synthetic buffers (200 KB per FL round-leg) so the measurement isolates transport, and that no model fitting or aggregation is performed in this experiment — then delete "vanilla FedAvg aggregation" and "fixed CICIOT-2023 data-partition seeds" from the factor list, since neither is exercised.

2. **"The selector is implemented as a DDQN agent trained under a CTDE paradigm using the AERPAW digital twin environment."** (§III-B, p.6.)
   *Unsupported on two of three counts.* "CTDE" appears in exactly two docstrings (`replay.py:1`, `selector_train.py:5`) and zero lines of executable code; the system is single-agent (one mule pose, one energy scalar, `step(action_idx: int)`, one Q-network, no critic, no joint action space; the only other RL module, `l1/channel_ddqn.py`, has no trainer at all). "AERPAW digital twin" is contradicted by `sim_env.py:7-9`. DDQN itself **survives** — `_online`/`_target` split, target sync every 200 steps, `r + γ·Q_target(s', argmax_a Q_online(s',a))`.
   *Replacement:* "The selector is a Double-DQN agent trained offline in ContactSim, a lightweight in-process contact-event simulator; the trained 209-parameter policy is shipped to the onboard NUC for inference-only execution. AERPAW-testbed training is future work." Do not hedge with "AERPAW-inspired" and do not redefine CTDE as "trained offline, deployed at the edge" — retrofitting the definition costs more than the correction.

3. **Algorithm 1 lines 19 and 21 have the sign backwards**, contradicting both the code and the paper's own p.6 prose.
   *Fix, don't defend* — see §A. Same edit pass must fix L1 (`ι` is never written by the slow phase) and swap L3/L7 (the printed order raises `ValueError` in `s3a_cluster.py:165-167`).

4. **"devices are subsequently partitioned into priority buckets according to urgency and expected FL contribution quality."** (§III-B, p.6.)
   *Unsupported:* `classify_bucket` is nine lines over three booleans with no deadline and no utility term. Replacement wording in §B. This is the single most quotable confirmation of the "under-specified" charge, because the claimed two-factor prioritisation turns out to be a three-way provenance tag.

5. **"Four-Stage Gated Federation Scheduler"** (§III-B, p.6), with the gating never enumerated.
   *Two problems:* the numeral collides with the team's own doc, which defines a *different* four (S1/S2A/S2B/S3); and the two stages that justify "Gated" at the FL-threshold level have **no production caller** — `ingest_ready_adv` is invoked only from unit tests, and the live path uses a duplicated inline check with `min_utility = 0.0` and a non-strict comparison, making the utility gate a no-op in every run. Any result attributed to `FL_Threshold = 0.60` filtering did not have that filter active.
   *Replacement:* enumerate as "S1 eligibility, S3 deadline + bucket classification, S3a RF contact clustering, S3.5 bounded RL selection" (what Algorithm 1 shows and `build_contact_queue` runs), and scope "gated" to S1 eligibility plus bucket admission, stating that the on-contact readiness/utility gates (S2A/S2B) are specified and unit-tested but were not exercised in the reported experiments.

6. **Table III, arm A4: "A DQN policy selects the next device based on feasibility-aware observations."**
   *Two errors:* it is a DDQN (§III-B says so — the paper contradicts itself), and the action is a **contact region**, not a device (design principle 15; `rank_contacts` ranks `ContactWaypoint`s). Also worth one sentence: A4's action space is variable-size K with a pointer-style scalar-Q scorer (11→16→1, tanh), not a fixed per-action head — a reviewer expecting a standard Q-head will otherwise think the implementation is broken.

7. **"In addition, 80% of A1 clients are marked as persistently unreachable from the central server…"** (§IV-A, p.9.)
   *Code default is 60%* (`--jittery-a1-dead-zone-pct`, default `60.0`, `runner_main.py:155`). Worse, `DeveloperDocs/Experiment_3_Run_Guide.md:131` documents the knob as: *"Tune lower for a milder jittery story, higher (e.g. 80) to put A1 below the mule arms in jittery on cumulative metrics."* A reviewer who reads that line will conclude the A1 collapse in Table VII and Fig 8 is a chosen parameter, not a finding.
   *Replacement:* state the value actually used, present the dead-zone fraction and the 0.4 long-range link-quality multiplier as **explicit modelling assumptions about correlated terrain blockage**, justify them (an i.i.d. per-round failure model unions to ~100 % success over 20 FedAvg rounds and therefore cannot represent terrain occlusion — this is a legitimate argument, made in the code comments at `arm_a1.py:82-100`), and report sensitivity across at least two dead-zone values.

8. **Implied claim that the adaptive deadline and the bucket taxonomy drive the reported scheduling results.**
   *Unsupported:* Exp 3 never calls `FLScheduler`, `classify_bucket`, `compute_deadline`, or the fast-phase fold; deadlines are static per-device windows and every device sits in one bucket. Say plainly that Table VII/Figs 8-10 isolate **contact-event clustering plus Pass-1 ranking policy**, and that the Φ/ι adaptation machinery is evaluated separately (unit + integration tests) rather than swept.

9. **Minor but checkable:** the sweep is described as `β ∈ {0.5,1.0,2.0}` giving `3×3×3×2×2 = 108` cells; the shipped default is `--beta 0.25 0.5 1.0 2.0`. State the grid actually run. Relatedly, for all mule arms Table VII's MCR and "Round part." columns are identical by construction — `arm_mule.py` emits exactly one `Exp3RoundLog` per mission ("one mission ≡ one FL aggregation cycle"), so the two metrics are the same quantity. Either drop one column or explain the identity before a reviewer asks why two independent metrics agree to three decimals in six of six rows.

---

## G. Ammunition found

**Reproducibility of the communication baseline — strong.** Table V reproduces *exactly* from a committed per-trial CSV. Running the means over `results/exp1_chameleon.csv` (1080 rows, n=20 per cell, both arms) yields 10 MB: 4.201 / 16.681 / 41.636 s and 100 MB: 4.162 / 16.639 / 41.595 s at R = 5/20/50 — matching the paper to three decimals. Offer the CSV as supplementary material.

**Statistics already implemented and committed.** `experiments/analysis/stats.py` implements paired Wilcoxon with Cliff's delta, non-parametric bootstrap CIs (2000 resamples), a crossover-round solver `R*`, and `bootstrap_R_star_ci`. Results are checked in: `DeveloperDocs/figures/exp1/exp1_paired_tests.csv` (18 cells, all W = 0, p = 1.9e-6, |δ| = 1.0 "large", n = 20 pairs each) and the jittery equivalent. This directly answers "are your differences significant."

**Sensitivity analysis already exists.** `experiments/calibration_sensitivity/{eps_high,eps_low,p_idle_high,p_idle_low}.toml` plus eight rendered figure directories (`figures/exp1_sens_*`, `figures/exp1_jittery_sens_*`) implement the ±50 % energy-constant sweep that `calibration.toml` promises. If a reviewer challenges P_idle = 5 W or ε_bit, the answer already exists.

**Real model size for the FL-vs-centralized argument.** Canonical CIC-IoT NIDS classifier: 5-layer dense `64→32→16→8→4→1` with BatchNorm + Dropout(0.4), **~4.7 K parameters, ~18.8 KB at float32** (`HERMES_Experiments_Implementation_Plan.md` §1, `Config/modelStructures/NIDS/NIDS_Struct.py:214`). Use this — not the 200 KB placeholder — when arguing bytes-on-wire. (And disclose the placeholder, see §F.1.)

**Bounded-inference claim is genuinely defensible.** The selector DDQN is 11→16→1 with tanh: **209 trainable parameters**, pure NumPy, no framework, documented as "< 200 ops per bucket — dwarfed by the RF link" (`ddqn.py:20-21`). This is unusually strong support for the §III-B complexity paragraph and should be quantified in the rebuttal.

**Real, tested safety invariants.** `SelectorScopeViolation` on any Pass-2 selector call, pinned by four tests (`tests/unit/test_selector_pass_gate.py`); `assert_candidates_admitted` set-membership guard; design principles 1-15 in `HERMES_FL_Scheduler_Design.md` §7 with principle-by-principle assertion tests (`tests/unit/test_design_principles.py`). Total suite: **612 test functions** across 45 unit + 19 integration files (the design doc's closeout note cites "410 passed, 22 deselected" at Phase 7). The 18 tests in `tests/unit/test_s3_deadline.py` pin the exact deadline semantics — useful to show the implemented behaviour is intentional, not accidental.

**An arm-divergence auditor already exists.** `experiments/exp3/audit_arm_agreement.py` re-runs the simulator per seed and reports decision sequences, pairwise agreement rates, and A3 filter-activation counts. If a reviewer asks "do A2/A3/A4 actually behave differently," this can be run and reported.

**Re-sliceable variance data.** The Exp-3 harness writes one CSV row per `(cell_id, arm, trial_index, seed)` with paired seeds (`base_seed = 42`) and echoes all five sweep params — so Table VII's ±SD can be decomposed by N, β, r_rf, deadline heterogeneity, and regime **if the CSV still exists** (see H.5).

**Empty categories — state plainly:** (i) **no accuracy, loss, or convergence metric for the IDS model is measured anywhere in Exp 1 or Exp 3** — neither experiment trains a model, so "enhanced collaborative training efficiency" in the abstract has no learning-quality evidence behind it; (ii) **no multi-seed variance for A4 training** — one seed, and weight init is unseeded, so even that seed is not reproducible; (iii) **no per-trial Exp-3 CSV and no A4 `.npz` weight file in the repo**; (iv) **no RF telemetry trace, no L1 experiment script**; (v) no ablation isolating the deadline adaptation or the bucket tiers (both are inert in Exp 3).

---

## H. Open questions only the authors can answer

1. **Table II / Fig. 3 / Fig. 4:** what physically ran? Real AERPAW SDR passes (which nodes, which reservation dates, which vehicle profile), a replay of recorded AERPAW telemetry through an offline policy comparison, or an emulator? The repo contains no L1 experiment code, no SNR/mobility trace, and no analysis script.
2. Within that experiment, which quantities were **counted** (bytes actually transferred over a radio) versus **derived** from the rate-tier mapping `R(·)` and the gain/cost constants `g(c)`, `κ(c)`, `λ(c,t)`? Was "Avg. Eff. SNR" measured per interval or imputed as `γ₁(t) + g(c)`?
3. Was the heuristic controller of Eq. (1) ever implemented in a runnable artifact, and where? `hermes/l1/` contains only an inference-only `ChannelDDQN` (which the paper correctly defers to future work) and a read-only `RFPriorStore`.
4. **Fig. 7 and Table VI:** which codebase produced the 5-job / 100 m × 100 m overloaded scenario with Return / Jobs Done / Jobs Failed over two seeds? Nothing in this repo emits those metrics, and the repo's EDF policy is a within-bucket contact ranker, not a job scheduler. Is this an earlier prototype, and does it share any component with the shipped system?
5. **Table VII / Figs. 8-10:** where is the per-trial CSV? Which exact command line — specifically which `--beta` grid (was 0.25 included?), which `--jittery-a1-dead-zone-pct` (60 or 80?), and was `--require-trained-a4` set?
6. Which A4 weight file produced Table VII? No `.npz` exists in the repo, and `driver.py:106-110` silently falls back to an **untrained, unseeded** `TargetSelectorRL` when weights are absent unless `--require-trained-a4` is passed.
7. Given that `TargetSelectorRL(rng_seed=…)` seeds only the ε-greedy RNG while `DDQN` is constructed without a seed, is any A4 result reproducible from `--seed`? Should this be fixed and re-run before camera-ready?
8. For Φ: fix the paper to describe the implemented seconds-window, or change the code to consume the real reliability statistic `on_time_count/(on_time_count+missed_count)` that already exists in `selector/features.py`? These give different papers; pick one before the camera-ready.
9. Was the 60.0 s default fulfilment window ever calibrated, or is it a placeholder? Its inline comment cites "design §9 Q1 open," but §9 Q1 is about the deadline *clock*, not the window magnitude.
10. Is the 2:1 widen-to-shrink ratio intentional tuning, and is the absent upper bound on Φ intentional? If not, add a cap before publication.
11. Is `deadline_override_ts` supposed to be cleared at mission start? No code path anywhere clears it, so one cluster amendment permanently disables fast-phase adaptation for that device.
12. Should PARTIAL and TIMEOUT fold identically? They share one `else` branch, while the design doc treats them as distinct outcomes.
13. Is `--theta-bytes = 200 000` intended to represent the 18.8 KB canonical NIDS model, or a different (e.g. GAN discriminator) payload? The paper reports `B_pw = R·2|θ|` without ever stating |θ|.
14. Does the paper intend to claim any learning-quality result? Neither Exp 1 nor Exp 3 trains a model or measures accuracy, so the abstract's "enhanced collaborative training efficiency" currently has no supporting measurement.
15. Should `Experiment_3_Run_Guide.md:131` ("tune… higher (e.g. 80) to put A1 below the mule arms") be rewritten before the repository is made public alongside the paper? As written it reads as story-fitting, independent of whether the modelling choice is sound.