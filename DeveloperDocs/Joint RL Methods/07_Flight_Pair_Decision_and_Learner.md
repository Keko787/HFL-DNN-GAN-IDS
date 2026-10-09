# The flight-time pair decision and its learner — end to end

*7 Oct 2026. Document 07 of the Joint RL Methods series; the overview is [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) (J2 and J5 there). This is the one place where reinforcement learning is actually run in FeRRy: the (band, next stop) choice at each Pass-1 arrival, the network that scores it, the simulator that trains it, and what the pre-registered study found. It restates code and records and decides nothing. File:line references were checked against the working tree on 7 Oct 2026. Numbers come from the code, from `results/exp5/rl/` and from the named documents; a number marked **(derived)** was computed from those JSON files for this document, and **unverified** marks a claim no record or code backs. Where this document disagrees with the overview or with another record, section 13 says so and why.*

---

## 0. Short version

At every Pass-1 arrival the mule makes one decision, a **pair (band b, next stop s)**. A mask admits only pairs that keep the plan whole and the rest of the flight feasible; a scorer ranks the admitted pairs; the highest wins. Four kinds of scorer fill the slot: none (F, the committed order), a fixed rule (FX), a learned pair score (FQ), and four scripted references. The learned score is a numpy masked pointer double DQN (36 → 64 → 64 → 1) trained in FerrySim, the real stack in one process.

Study 5.5 asked whether looking ahead (γ > 0) helps over a one-step score. Pre-registered answer, 6 Oct 2026: **flat**. No γ beat γ = 0 by ε = 0.01; the best learned score sits at FX's level and 0.0059 below the one-step rule `greedy_1`; **FX stays** as FeRRy's in-flight rule. The learned competitor E3 (DQN over stops, no FeRRy machinery) scores below FX in bytes (section 9). Why FQ at γ = 0 landed on FX and not on `greedy_1` is held as a hypothesis (it "copied FX"); the screen that would test it was written down and **not run** (section 8).

---

## 1. The decisions made, by clock and by who decides

| # | Decision | Clock | Inputs | Output | Who decides | Learned? | Code |
|---|---|---|---|---|---|---|---|
| D1 | Band class b̄ and route π | Plan, once per mission at the dock | Demand, mean SNR, ages, deadlines | Committed (b̄, queue, budget) | Plan search on hand-set score V | No | `hermes/scheduler/plan/plan_search.py`, `plan_score.py` (not covered here) |
| D2 | **(band, next stop) pair** | Flight, every Pass-1 arrival | `PairView`: per-class SNR now, offsets, remainder, clock, energy, ages | Band for the stop just reached; next stop or home | The pair slot: mask, then a scorer's masked argmax | **FQ only** (tested); FX, `greedy_1` etc. fixed | `policies/pair_slot.py:762` (`pair_at_arrival`) |
| D3 | Takeoff stop, Pass-2 order | Takeoff; Pass 2 | The committed plan | Plan's first stop; b̄ in queue order | Nobody: slot returns index 0 and never reads the view | No | `pair_slot.py:716`, `749` |
| D4 | Departure check after the stop | Flight, after each stop | Remainder in the order the pair set, observed rate | Keep, trim, re-plan (`replan`) or abort | The S3b predicate, one shared fold | No | `mule/mule_main.py:1976-1986` |
| D5 | Fallback when no pair is admitted | Flight | The mask | FX's band, index 0 (home if none left), logged `mask_empty` | The slot | No | `pair_slot.py:797-799` |
| D6 | E3's next stop | Takeoff and every Pass-1 departure | Chen-style per-stop rows (10 columns) | Index of an admissible stop, or None (pass ends) | A DQN among stops S3b's budget predicate admits | Yes | `policies/chen_dqn.py`, `next_stop.py` |
| D7 | Backhaul band (mule to base station) | Once per mission | Utility U(c, t) | Backhaul channel | Deterministic utility; `ChannelDDQN` retired | No | outside this document |

D2 is the joint decision. The order of operations at an arrival is in `mule_main.py:2225-2257` (transit charged, pair decided at the arrival instant, contact plan built on the chosen band) and `:2016-2021` (after the stop the chosen stop is moved to the front, then D4 folds that order).

---

## 2. The decision and its mask

### 2.1 What the slot sees: `PairView` (`plan/types.py:996`)

Taken at the arrival instant, after transit is charged and before the contact. Scheduler-side data built by the mule (`FerryRuntime.arrival_view`), so the scheduler imports nothing from the mule.

| Field | Meaning |
|---|---|
| `arrival` (`ArrivalView`, `:728`) | Stop's members, committed class b̄, and per class an `ArrivalClass` (`:706`): `targets` reached at the realised SNR, `dwell_s` of serving them all |
| `observed_snr_db`, `offsets_db` | Per class, median realised SNR over the members now; realised minus mean, differenced per link before the median |
| `previous_offsets_db`, `previous_age_s` | The previous Pass-1 arrival's offsets and their age; both None at the trial's first |
| `period_s` | P_c, the contact channel's interference period (configuration, not phases) |
| `stops` (`StopContext`, `:902`) | One per remainder stop, in order, or the single `home` once none is left: leg `travel_s`, predicted dwell on b̄ at mean SNR, per-class mean SNR, `capped`, `exempt`, mean plan `age`, `on_time`, coverage `weight` |
| `budget_end`, `budget_s`, `t_ref_s` (T_nom), `energy_j`, `energy_ref_j`, `demand` (N), `demand_weight`, `cap_s` (S) | Clock, energy and plan context |

**Pairs** (`PairView.pairs`, `:1147`): the *covering* classes (`:1125`: those whose targets include every target of b̄ at this stop) times the candidates, class-major in link order, then in the remainder's order; home stands alone and is offered only when the remainder is empty. The row order is the tie order.

### 2.2 The mask: `fits_after_service`

The mask is one predicate, `FLScheduler.fits_after_service` (`fl_scheduler.py:1966`), bound per arrival by `bind_fits_pair` (`pair_slot.py:465`). A pair is admitted only if all three hold:

1. **The band covers the plan**: it reaches every device b̄ reaches here (the covering set; enforced when the pairs are formed and again by the scope guard).
2. **The next stop is in the plan**: an index of the remainder, never a new stop. Home only once the remainder is empty.
3. **The rest of the flight still fits**: after serving here on that band (dwell priced at the observed SNR, its targets collected) the remainder, with the chosen stop first and the others in plan order, is folded (`fold_remainder`) from the state after service, with the plan's exempt stops protected, on b̄'s model at the mean SNR for later stops (δ_obs = 0), under the arm's in-flight rule. For the home pair the landing is checked directly: dwell + return leg + Pass-1 upload by `deliver_by` (route-level `delivery` only) and by `budget_end`, then the energy clause.

Not tested by the mask: the served stop's own deadline clause (the mule is there whichever pair is picked) and the listen window (charged only when a reply is missing; the departure check sees it). Without a budget nothing is gated. Pass 1 and plan mode only; a legacy mule or a Pass-2 call raises. Under `abort` the departure check folds the chosen stop alone, whose verdict is the mask's first fold, so the mask is the stricter; `flight_slot = pair_q` needs `in_flight_response = replan` (`processes/config.py:931-938`).

### 2.3 The scope guard

`assert_pairs_admitted` (`selector/scope_guard.py:57`) runs before the mask is asked. It raises `SelectorScopeViolation` (a wiring bug, not a data issue) if any pair's band is outside the covering classes, any index is outside the remainder, home is offered with stops left, or any member of the served stop or of the remainder is not in the `admitted` set (the plan's devices plus the beacon hook's inserts). Freeze principle 12: learning chooses only among admitted options. The legacy selector's `assert_candidates_admitted` (`:39`) is untouched.

### 2.4 The pick, the fallback and the record

`pair_at_arrival` (`pair_slot.py:762-839`) in order: pass guard (Pass 1 only); scope guard; the mask is asked of every pair (`:789`, answers must be bools, `:242`); FX's pair is computed for the record (`:791-792`); the scorer returns one finite number per pair (`check_pair_scores`, `types.py:1189`); if any pair is admitted the **masked argmax**, ties to the lowest row (`pair_q.masked_argmax`, `:525`), or with a trainer attached ε-greedy over the admitted pairs (`behaviour_row`, `:545`); a pick the mask refused raises (`:807-809`).

**Fallback `mask_empty`** (`types.py:785`, `pair_slot.py:797-799`): when no pair is admitted the mule flies **FX's band with no reorder** (index 0, or home), not FX's nearest-stop pick. `PairChoice` enforces that the fallback happens exactly when no pair fits and that it flies FX's band with `next_index` 0 or None (`types.py:1248-1258`). Frequency, from the module docstring: FX's own pair overruns the budget at the arrival SNR at about 19 of 71 last-stop arrivals at N = 6 and 1 MB (the Phase 5 design, finding 3; not re-measured here). During training an empty mask stores the flown pair as the one admitted row (`PairStep.effective_mask`, `:650`).

**Record** (`DECISION_KEYS`, `:197`, then `CLOSE_KEYS`, `:215`, added at mission end): `t_s, devices, committed, band, next_index, next, pairs, feasible, admitted_pairs, fallback, fx_band, fx_next, agrees_fx, scorer, q, q_fx`, then `collected, w, late, t_next_s, terminal, trimmed_next`. Written to `mission_completed.pass_1_pairs`. `agrees_fx` compares with FX's rule at this arrival (the nearest stop whose pair on FX's band the mask admits, else 0), not with the FX arm's flight (the two part under `replan` when (FX's band, 0) is refused).

---

## 3. The fillings of the slot

All fillings share the pass guard, scope guard, mask, fallback and reorder; they differ only in ranking. The slot's tie rule never decides for a scripted scorer, because each ranks every pair totally (`_ranked`, `:324`).

| Filling | Arm / policy label | Ranking | Code | Learned |
|---|---|---|---|---|
| Committed | **F** | None: the plan's order on b̄; never reads the view | `cross_heuristic.py:186` (`CommittedSlot`) | No |
| Cross-heuristic | **FX** | Next stop: the nearest remaining stop whose move to the front keeps the rest feasible, else index 0. Band: the fastest covering class at the arrival SNR (ties: more targets, then the committed class, then lower index) | `cross_heuristic.py:221`, `:168` | No |
| Learned pair score | **FQ** (FQ-g0 … FQ-g99 for the γ sweep) | Q of each admitted pair's `pair_v1` row | `selector/pair_features.py:584` (`LearnedPairScorer`) | Yes |
| `fx_pair` | scripted | FX's band rank, then the nearest stop | `pair_slot.py:362` | No |
| `committed_pair` | scripted | b̄ first, then FX's band order, then the plan's order | `:378` | No |
| `hyb` | scripted | FX's band, plan's order (critic B5) | `:400` | No |
| `greedy_1` | scripted | Most targets at the arrival SNR; then least (dwell here + travel); then FX's band order and nearest-first | `:418` | No |
| E3 | `contact_policy = chen_dqn` | Next stop only, DQN over per-stop rows, bytes reward, no plan | `policies/chen_dqn.py`, `next_stop.py` | Yes |
| M1 | after Ho et al. | One network over every pair, no plan, no mask | `monolithic_ho.py`, **not built** (only if 5.5 kept a learned score) | n/a |

Notes on the scripted references:

- They are FX, F and HYB *inside the slot's rules*, not the fixed arms. Where they part from the arm is documented per scorer in the module docstring (`pair_slot.py:49-92`); the study reads FX and F as the arms themselves (resolution R6) and reports `fx_pair` and `committed_pair` beside them, outside the rule.
- `greedy_1` ranks by targets first because one device is worth 1/N of the reward, more than any time a pair can save at c_t = 0.1. It is the "most devices" band rule Phase 4 rejected as a filling (it can land past the budget after the last stop); the mask now prices that landing, but realised dwell still runs longer than priced, so it is a reference to beat, not a filling. It is a function of FQ's inputs, which is why it is the right one-step sanity reference.
- FX's band rule never dwells longer than F at the arrival SNR and never reaches fewer devices, since b̄ is always among its candidates (`cross_heuristic.py:44-56`).
- FX and F inside FerrySim: F's slot reads neither the channel nor the predicate; FX acts in Pass 1 only, its band half at every arrival and its stop half after each stop.

---

## 4. The 36-column `pair_v1` features

`selector/pair_features.py`. T = T_nom (`t_ref_s`), N = demand, S = cap, P_c = interference period. One row per pair of `view.pairs`, in that order (`pair_rows`, `:515`). Every column is a function of the view alone (causal, deterministic); the schema's JSON form (version, dim, classes, phase flag, columns, constants) is the checkpoint header's `schema` and is compared whole on load. Counts below were confirmed by building the schema over (wide, medium, narrow): dim 36, 24 without the phase block; by dependence: 6 band, 11 next, 4 pair, 15 state.

| Group | Column | Depends on | Value |
|---|---|---|---|
| Band (3) | `band[c]` ×3 | band | One-hot of the class the stop is served on |
| Band | `snr_here` | band | b's median realised SNR over the stop's members now, ÷ 30 dB |
| Band | `dwell_here` | band | b's dwell at k at that SNR, ÷ T |
| Band | `gain_here` | band | b's targets beyond b̄'s, ÷ N (≥ 0) |
| Stop | `reach_here` | state | b̄'s targets at k, ÷ N |
| Next | `travel` | next | Leg from k to s (to the dock for home), ÷ T |
| Next | `snr_next[c]` ×3 | next | s's mean SNR on each class, median over its members, ÷ 30 dB |
| Next | `dwell_next` | next | s's predicted dwell on b̄ at the mean SNR, ÷ T |
| Next | `slack_next` | pair | sign(x)·log1p(\|x\|/T), x = Deadline(s) − (now + dwell_here + travel + dwell_next); unclipped; 0 for home or an exempt/undated stop |
| Next | `exempt_next`, `capped_next`, `home` | next | 0/1 flags (no deadline clause can bind; some member capped; the home row) |
| Next | `age_next` | next | s's mean plan age ÷ S (÷ 1 without a cap) |
| Next | `on_time_next` | next | Mean on-time rate (0.5 prior for a device never seen) |
| Next | `members_next` | next | s's members ÷ N |
| Situation (5) | `clock_left` | pair | (budget end − (now + dwell_here + travel)) ÷ budget, clipped to [−1, 1]; 1 without a budget |
| Situation | `energy_left` | state | 1 − energy spent ÷ reference, clipped to [−1, 1]; 1 without a reference |
| Situation | `remainder_share` | state | Remainder's members ÷ N |
| Situation | `least_slack` | state | Least `slack_next` over the dated remainder, each priced after serving k on b̄ |
| Situation | `weight_share` | state | Remainder's committed weight ÷ the demand's |
| Phase (12) | `offset[c]` ×3 | state | Class offset now (realised − mean, per-link median), ÷ 10 dB |
| Phase | `prev_offset[c]` ×3 | state | Previous Pass-1 arrival's offsets this trial, ÷ 10 dB (0 at the first) |
| Phase | `has_prev` | state | 1 once a previous reading exists |
| Phase | `prev_age` | state | Its age, capped at 4 P_c, ÷ P_c |
| Phase | `prev_sin`, `prev_cos` | state | sin and cos of 2π·age/P_c |
| Phase | `arrival_sin`, `arrival_cos` | pair | sin and cos of 2π·(dwell_here + travel)/P_c |

Constants (`:177-191`): `SNR_SCALE_DB` 30, `OFFSET_SCALE_DB` 10, `PREVIOUS_AGE_CAP_PERIODS` 4, `SHARE_CLIP` 1. Changing any column or constant is a new schema version.

**Why the phase block exists.** The contact channel's interference is a regular wave of period P_c (jittery regime: amplitude A = 5 dB, σ_I = 1.5 dB, shadowing σ = 4 dB, per the docstring at `:177-184`; default P_c 60 s, `cells.py:173`). One reading cannot tell a rising signal from a falling one, so the row carries two readings and the timing between them. `reach_here`, `prev_sin` and `prev_cos` go beyond the design's list as encoding choices (`:73-81`). Dropped from the design: a per-stop "value" (equal to members ÷ N under equal data) and a clipped slack (it saturated).

**Sparse columns** (`SPARSE_COLUMNS`, `:207`): `gain_here` (a covering class reaching more than b̄ at the arrival SNR: at most about 2 % of arrivals in the design's probes) and `exempt_next` (a whole stop capped, from mission S on). A learner therefore has almost no signal on band-for-coverage; the band choice that matters is the rate/dwell trade (`snr_here`, `dwell_here`, `arrival_*`).

---

## 5. The network and the update

`selector/pair_q.py`. numpy only (no torch); `LEARNER_REVISION = 0` (`:153`).

### 5.1 Network (`PairQNet`, `:691`)

A pointer score: shared weights give one scalar Q per row, so the candidate set can grow and shrink. Hidden (64, 64), tanh, 36 → 64 → 64 → 1; initial W ~ N(0, 1/fan_in), b = 0, float64; target network a copy. `_scores` (`:601`) gives equal rows one Q bit for bit so the lowest-row tie rule, not rounding, decides between symmetric pairs.

### 5.2 Settings (`PairQConfig` `:353`, `LearnerSettings` `:488`)

| Setting | Value |
|---|---|
| γ | In [0, 1]; the default 0.9 is the design's; a trainer always states its own; Study 5.5 sweeps {0, 0.25, 0.5, 0.75, 0.9, 0.99} |
| Optimiser | Adam, lr 1e-3, β1 0.9, β2 0.999, ε 1e-8 |
| Loss | Huber, δ = 1, batch mean, y held fixed (semi-gradient); global gradient-norm clip 10 |
| Target | Hard copy of the online weights every 500 updates |
| n-step | 1 only (refused otherwise) |
| Batch, warm-up, replay | 64; no update until 1,000 transitions are stored, then one update per decision; 50,000 (FIFO) |

### 5.3 The target (`targets`, `:795`)

y = r + γ · Q_target(s′, a\*), with a\* = argmax of **Q_online** over the next decision's **admitted** rows (ties to the lowest row): double DQN, with the online network choosing and the target valuing. Only admitted rows are forwarded, so a masked row cannot enter a target. A done transition (the sortie's last decision; the Q's horizon is the sortie) and every transition at γ = 0 get y = r exactly. A transition that is not done must own at least one admitted next row.

### 5.4 Behaviour (`BehaviourSchedule`, `:434`; `behaviour_row`, `:545`)

Episodes 0–499: ε-greedy **around FX's pair** at ε = 0.3 (the greedy action is FX's pair if admitted). After that: ε-greedy on the network's own Q, ε falling linearly from 0.3 to 0.05 at episode `decay_fraction` × run length (the first half), then 0.05. ε never starts at 1.0 for the pair score (a Q over pairs never trained is noise). Exploration is uniform over admitted pairs only. Every decision with an admitted pair draws exactly two numbers from the episode's seeded stream, so runs differing in ε or weights draw alike. **E3 uses the same schedule with no reference phase and ε from 1.0** (`train.py:187`, and the manifests: `epsilon_start` 1.0, `reference_episodes` 0).

### 5.5 Replay (`selector/pair_replay.py`)

A `PairTransition` (`:118`) is one decision: the taken row x, the reward, `done`, and, unless done, the **next decision's every candidate row and the mask it was taken under** (a done transition carries none; a next mask must admit at least one row; masked rows may be non-finite). A `PairBatch` (`:182`) concatenates the next rows and marks each with its transition in `segment` (non-decreasing); `PairReplay` (`:288`) is a FIFO ring sampled uniformly without replacement from `random.Random(seed)` (`DEFAULT_CAPACITY` 50,000, `:64`). Variable width is the point: up to 15 pairs at N = 12 and 21 at N = 24 in the design's probes, so the plan's "at most 18" is no bound. A `home` decision that a beacon insert follows bootstraps from the insert's decision (critic B12).

### 5.6 Checkpoints, format 2

`.npz` of the online weights (float64), `format_version` = 2, and a header (canonical JSON bytes): `format, kind, purpose, learner_revision, network, schema, classes`. A JSON manifest sits beside it. **sha256 is taken over every array including the header** (`arrays_sha256`, `:1109`: name, little-endian dtype, shape, C-order bytes), so the sha a config names pins the weights, what they read, why they were written (`bootstrap` | `trained`) and which learner revision wrote them (resolution R2). The manifest repeats the header and adds what the sha does not bind: γ, numpy/BLAS versions, reward and training specs, seeds, cell family and its hash, trainer commit and dirty flag, `episodes_trained`, the validation curve, and `held_out` (None until the evaluator fills it, `record_held_out`, `:1395`). No wall time enters a manifest.

**Refusals.** The loader (`PairQNet.load`, `:989`; `_read_checkpoint`, `:1330`) refuses: a missing or malformed manifest; any format but 2 (the legacy DDQN's format 1 is named); arrays not matching the manifest (sha) or the header (a relabelled purpose or revision fails here); a sha other than the config's; a kind, class tuple or schema other than expected; non-float64 or non-finite weights; pickled content (object loading is off). `pair_features.load_pair_scorer` (`:656`) adds: not `.npz`, no file, not `pair_v1`, other classes, other phase flag. The **runner's** refusals, applied to a verified manifest (`campaign_refusals`, `:1415`): purpose not `trained`, no episode trained, no held-out score, dirty tree (unless `--allow-dirty-checkpoint`). The loader applies none of these so FerrySim's bootstrap checkpoints load. `checkpoints.tag_refusals` (per the module docstring) also ties a checkpoint to its arm's tag: γ, reward, a network that took an update, and, for an FQ arm, only on the plan it trained under.

---

## 6. FerrySim (`experiments/ferrysim/`)

**What it is.** The Exp 4 driver's own trial run in one process: the real cluster, mule and device services, synchronous in-process links, a virtual wall clock; only the devices' local training is a stand-in (`equal` shards for training: every update weighs n_ref = 10) and the device service loops do not run (`inprocess.py:1-80`). Parity tests compare it with the stack's oracles. An episode is one trial of a cell (4 missions, 1 MB payload, wide reference class, `replan` with `trim`, `agg:cutoff`, simulated clock, S\* cap 2).

| Module | Role |
|---|---|
| `inprocess.py` | The in-process trial; `RoleHooks.on_mule` installs the episode's `PairQSlot` through `MuleSupervisor.install_flight_slot` |
| `cells.py` | Cells, families, seed streams, re-pin block |
| `episode.py` | One episode under a `Policy` (an arm's own slot, or a pair slot around a scorer factory), read as the reward reads it; the mule's closed records are compared stop for stop |
| `reward.py` | The reward (below) |
| `train.py` | One training run to one checkpoint |
| `checkpoints.py` | Layout `results/exp5/checkpoints/<study>/<tag>/g<γ>_s<seed>.npz`, tree state, flying, scoring |
| `evaluate.py` | Held-out and validation returns on common random numbers, workers, Study 5.6's lag measure |
| `headroom.py` | Clairvoyant oracle and ε |
| `report.py` | Study 5.5's rule |
| `pilot.py` | Study 5.11's budget and decision-cost pilot |

### 6.1 Cells, families and the re-pin (`cells.py:273-293`; re-pinned 5 Oct 2026)

| Cell | Role | N | Budget (s) | Contact regime | P_c (s) |
|---|---|---|---|---|---|
| `jit-n6-75`, `jit-n6-150` | control | 6 | 75 (stress), 150 (knee) | jittery | 60 (default) |
| `jit-n12-90`, `jit-n12-180` | decision-rich | 12 | 90, 180 | jittery | 60 |
| `cln-n12-90`, `cln-n12-180` | negative control | 12 | 90, 180 | clean | n/a |
| `jit-n12-90-q`, `-h` | Study 5.6 | 12 | 90 | jittery | 108, 54 |
| `jit-n12-180-q`, `-h` | Study 5.6 | 12 | 180 | jittery | 136, 68 |
| `scl-n24/48/96-*` (6 cells) | Study 5.11 (c), flown not trained | 24, 48, 96 | stand-ins (350/525, 680/1020, 1330/1995) | jittery | 60 |

**Families** (`FAMILIES`, `:369`; one score per contact regime): `jittery` = the four N = 6 and N = 12 jittery cells; `clean` = the two clean cells; `jittery56` = the four jittery cells plus Study 5.6's four (8 cells; the family the sweep trains on, `params.toml [rl] family`); `scale`. The cell for each training episode is drawn uniformly from the family by a keyed draw of (run, index) (`train_episode`, `:544`). The cap S = 2 everywhere (`CAP_S`). Study 5.6's lags 27 s (at 90 s) and 34 s (at 180 s) are FX's median Pass-1 arrival-to-arrival time on the validation stream, measured 5 Oct 2026 as 26.94 s over 672 lags and 34.11 s over 397 lags, rounded; P_c is 4× (quarter, `-q`) or 2× (half, `-h`) the lag. The repinning script derives the lag-ratio check's bounds (0.83–1.26 and 0.67–1.48) from the same sample and stops if the quarter and half cells' ratio ranges would meet.

### 6.2 Seed streams

A trial seed is 32 bits: the top two bits name the stream kind (`train` 1, `val` 2, `heldout` 3; `STREAM_TAGS`), the other 30 are a SHA-256 of (stream, cell, episode index) (`trial_seed`, `:468`). Streams are `ferrysim-train-<seed>`, `ferrysim-val` (headroom, ε, in-training validation) and `ferrysim-heldout` (Study 5.5's judgement, shared by every checkpoint and reference: common random numbers). Disjoint by construction; `check_disjoint` verifies; within a validation or held-out stream a repeated seed is skipped. A stream never moves when another is extended.

### 6.3 The reward (`reward.py:139`)

```
r_k = G_k − c_t · Δt_k / T − c_e · ΔE_k / (P_hover · T)      (− c_cov · U at the sortie's last decision)
G_k = Σ w_i over updates collected CLEAN at k  /  (n_ref · N)
U   = Σ ω_j over committed devices left uncollected  /  Σ ω_j over the demand
```

w_i is the raw L3 merge weight (n·v·s(age), zero past the cutoff; `aggregation_rules.update_weights`), so one fresh device of reference size is worth 1/N; under the training cells' equal-shard model G_k is the stop's collected count over N. Δt_k runs from the arrival at k to the next Pass-1 arrival, or after the last decision to the end of the Pass-1 upload. Defaults c_t = 0.1, **c_e = 0**, c_cov = 1; no lateness term (none of 1,148 probed collections was late). Study 5.7's grid, `GRID_C_T` × `GRID_C_COV` = {0.03, 0.1, 0.3} × {0.25, 1, 4} (`:89-90`), is now unused (it was a learned-score grid).

**Expected availability while training.** Whether a device answers is a keyed draw independent of the pair, with noise 10–100× the time signal a decision moves, so each targeted member is credited its probability rel_j (a dropped member at n_ref, a collected one at its realised weight) and U counts it uncollected with probability 1 − rel_j. Validation, held-out evaluation and every reported number use the realised draw. Other kinds: `hand` (F·hand, ContactSim's reward ported: (200·|C_k| − Δt − 0.002·metres)/150), `bytes` (E3's: |C_k|/N).

### 6.4 One training run (`train.py`)

One call = one learner, one γ, one seed, one family, ending in one checkpoint.

| Setting | Value (`train.py:176-187`) |
|---|---|
| Episodes | Up to 10,000; each episode is a 4-mission trial |
| Validation | Every 1,000 episodes, `val_episodes` = 200 greedy episodes spread over the family's cells, the first of each cell's validation stream (the same at every validation and run); the score is the mean over cells of each cell's mean realised return. The sweep's calibration used 400 for `jittery56` (`params.toml [rl.val_episodes]`) |
| Keep / stop | Keep the weights of the best validation (a strictly higher score is a new best); stop after `patience` = 3 validations without one |
| Order of events | An episode is flown with the network as it stood at its start; then its transitions are pushed in decision order, each followed by one update once warm. The network moves between episodes, never inside one |
| Devices | Equal shards (`TRAINING_DEVICE_MODEL`) |
| Determinism | Same seed, same weights and manifest, bit for bit, with `OPENBLAS_NUM_THREADS=1`; refuses a dirty tree unless `--allow-dirty` |

Transitions per episode, from the manifests **(derived)**: the `g0.75_s3` run flew 8,000 episodes and pushed 49,667 transitions (about 6 per episode).

### 6.5 Headroom and ε (`headroom.py`)

Before any training, a clairvoyant per-sortie oracle bounds what any pair choice could gain. FerrySim cannot fork a mission, so it replays: for sortie j a depth-first search over every sequence of admitted pairs (each decision's admitted pairs in FX's ranking, so the first leaf is FX's choices); each leaf is a deterministic replay with `fx_pair` flying every other sortie. At most `MAX_LEAVES` = 512 leaves per sortie, beyond which the sortie is `truncated` and its value a lower bound. The search sees the realised channel and availability draws, so the headroom is an upper bound no causal policy reaches. V = the largest of the oracle sum, FX's return and the scripted references' whole-slot returns; headroom = V − R(FX) (never negative); `slot_gain` is the oracle over `fx_pair`. F is reported beside (`f_gain`), not in V. **ε = max(0.01, 0.1 × the validation headroom)** (`epsilon_from_headroom`, `:91`); the build pauses only if the headroom is below 0.01 in every cell (it did not).

---

## 7. Study 5.5 and Study 5.6

### 7.1 Study 5.5's pre-registered rule (`report.py`; decision 5 (a))

Read on N = 12 at both budgets (`jit-n12-90`, `jit-n12-180`; `STUDY_5_5_CELLS`), whichever family trained the score; N = 6 is reported as the control. The unit is the training seed: a seed's score is its checkpoint's mean undiscounted held-out return over the cells read, on one shared set of held-out episodes per cell. Grid (`:106-112`): γ ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99}, 10 seeds each, 1,000 held-out episodes per cell. `decide` (`:466`) applies, in order, with nothing looked at twice:

1. **Sanity check.** The γ = 0 mean must be at least max(FX, `greedy_1`) − ε (FX the arm itself). If not: `sanity-failed`, the curve is not read, FX stays. The learner may be revised **once** before the sweep (the report refuses a sweep trained by two revisions or a revision past the first, `MAX_LEARNER_REVISION = 1`).
2. **Rising.** The best γ > 0, picked on **validation** (mean over seeds of the kept checkpoints' validation scores on the cells read; ties to the lower γ, never on held-out), beats γ = 0 by ≥ ε with the bootstrap CI above 0 and Holm-adjusted p < 0.05 (exact paired Wilcoxon of every γ > 0 against γ = 0 on seed means, Holm over the five contrasts; with 10 seeds the exact floor is 0.002, 0.0098 after Holm).
3. **Flat.** Every γ > 0 within ±ε of γ = 0 by Schuirmann's TOST on seed means (an intersection-union test, so no Holm).
4. **Inconclusive** otherwise. No second look.

**Replace FX** only if the outcome is rising **and** the best γ beats the best fixed rule (FX, F, `hyb`, `greedy_1`; the one with the highest held-out mean) by ε under the same claim rule (gain ≥ ε, CI above 0, p < 0.05). Otherwise FX stays and the null is published. **`greedy_1` against FX** is reported separately: if it beats FX by ε (episode as the unit), the user decides (critic A2). Reported, never decided on: Page's trend test, learning curves, each cell's share of sorties with 2+ decisions, per-cell means, and the **stack check's picks** (the best γ and γ = 0, each at its median-validation seed). A sweep off the grid (the calibration's) is labelled, not refused (R26). Rule fixed before any training; the claim rule is gain ≥ ε AND CI excluding 0 AND p < 0.05.

The stack check (batch 2): the best γ and γ = 0 each from its median-validation seed, beside FX and F, 40 seeds at each of the two N = 12 budgets (320 trials, build plan), scored on time to τ against FX with `pair_fx_agree_share` among the reported columns (`params.toml [score.s55]`).

### 7.2 Study 5.6 as rebuilt

Design (decision 6 (a); resolutions R22, R29): the question "when does learning help" was to sweep transit time over channel period. As rebuilt it uses **two lag/P_c ratios** (quarter and half) instead of the grid {0.25, 0.5, 1, 2}, which aliases, at both N = 12 budgets, with the clean N = 12 cells as the control; "FX tuned separately per regime" was dropped because FX has no parameters. Cells: the four `jit-n12-{90,180}-{q,h}` and `cln-n12-{90,180}` (periods in 6.1). Because 5.5 did not keep the learned score, **M1 is not built and 5.6 is E3 against FX**. Settings (`params.toml [s56]`): arms listed `["FQ", "FX", "E3"]`, FQ dropped by the launcher when `rl.keep_learned = false` (`launch.py:735-775`), 40 trials per cell, N = 12, 1 MB; E3 flies `rl.checkpoints.e3` (`g0.99_s0`). That is 6 cells × 2 arms × 40 = 480 trials **(derived; the `params.toml` comment still counts 3 arms, 720)**. Primary metric time to τ against FX; also reached τ, round closure, Pass-1 contacts, update yield. **Status: built, batch 2, not run.** The memo's claim that the transit/period axis "earns no credit on its own" (settled by construction) still stands; 5.6 measures how much, not whether.

---

## 8. Results

All values are `results/exp5/rl/*` unless noted. Reward is the derived one at the realised draw; higher is better.

### 8.1 Headroom (`headroom/headroom.json`, 200 validation episodes per cell, 6 Oct 06:42–07:17)

| Cell | Decisions per sortie | Sorties with 2+ decisions | Headroom (V − FX) | Slot headroom | FX | `greedy_1` | `hyb` | F | Oracle truncated sorties |
|---|---|---|---|---|---|---|---|---|---|
| `cln-n12-180` | 2.07 | 64.6 % | 0.0126 | 0.0107 | 0.1127 | 0.1162 | 0.1136 | 0.1020 | 29 of 800 |
| `cln-n12-90` | 1.86 | 66.6 % | 0.0321 | 0.0050 | −0.2620 | −0.2592 | −0.2587 | −0.2694 | 0 |
| `jit-n12-180` | 1.50 | 32.8 % | 0.0229 | 0.0191 | 0.0590 | 0.0663 | 0.0578 | 0.0404 | 4 |
| `jit-n12-90` | 1.84 | 62.0 % | 0.0420 | 0.0154 | −0.2935 | −0.2822 | −0.2922 | −0.3167 | 0 |
| `jit-n6-150` | 1.03 | 2.3 % | 0.0123 | 0.0108 | 0.0857 | 0.0879 | 0.0857 | 0.0514 | 0 |
| `jit-n6-75` | 1.14 | 12.6 % | 0.0238 | 0.0128 | 0.1401 | 0.1485 | 0.1411 | 0.0964 | 0 |

ε = 0.01 in every cell (the largest 0.1 × headroom is 0.0042). The room over FX is thin (0.012–0.042; 0.005–0.019 for the in-flight slot) and is a clairvoyant bound. The N = 6 cells barely decide. `greedy_1` beats FX in all six cells, by more than ε only at `jit-n12-90` (+0.0113). F trails FX in every cell (`f_gain` −0.007 to −0.044).

### 8.2 Calibration (not pre-registered; γ ∈ {0, 0.9} × 3 seeds per family; 6 Oct 07:17–09:17)

Verdicts read on the two N = 12 cells of each family (`calibration/*_verdict.json`, `cells`); the N = 6 cells appear only in `per_cell`.

| Family | Outcome | FQ γ = 0 | FQ γ = 0.9 | FX | `greedy_1` | Floor | γ = 0 seeds below floor |
|---|---|---|---|---|---|---|---|
| `jittery56` | flat | −0.0778 | −0.0780 | −0.0786 | −0.0716 | −0.0816 | 0 of 3 (passed) |
| `clean` | **sanity-failed** | −0.0989 | −0.1005 | −0.0914 | −0.0884 | −0.0984 | 2 of 3 (failed by 0.0005) |

`jittery56`: γ = 0.9 against γ = 0 is −0.0002 (CI [−0.0042, 0.0038]), equivalent within ±ε (TOST p = 0.0094). `greedy_1` against FX: +0.0070 (CI [0.0045, 0.0098]) on 2,000 paired episodes on `jittery56`, +0.0029 (CI [0.0016, 0.0044]) on `clean`; both below ε, so the user-decides flag is not raised. Training: validation scores bounce within about ±0.005–0.01 of one level (clean −0.14 to −0.16; `jittery56` −0.04 to −0.06), the best check lands anywhere from the first to the sixth, TD loss hardly moves (about 0.006 clean, 0.009 `jittery56`). One run's row was re-checked against its manifest: `clean g0_s0` validation −0.146, −0.155, −0.152, −0.162; best at 1,000; stopped at 4,000; loss 0.0059 → 0.0060. The full 12-run table is in the findings document.

### 8.3 The sweep: Study 5.5's verdict (`s55/verdict.json`; trainings 13:58–19:53, evaluation 197 min, 6 Oct; pre-registered: `preregistered = true`)

| γ | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 0.99 |
|---|---|---|---|---|---|---|
| Mean held-out return | −0.0795 | −0.0776 | −0.0779 | **−0.0774** | −0.0778 | −0.0779 |
| Gain over γ = 0 | n/a | 0.0019 | 0.0015 | 0.0020 | 0.0017 | 0.0016 |
| Raw p / Holm p | n/a | 0.232 / 0.480 | 0.160 / 0.480 | 0.027 / **0.137** | 0.193 / 0.480 | 0.105 / 0.422 |
| TOST p (equivalence within ±0.01) | n/a | 1.4e-4 | 2.8e-6 | 1.3e-6 | 4.1e-5 | 2.7e-7 |

- **Outcome: flat.** Best γ (picked on validation) = 0.75; its Holm p = 0.137 and CI of the gain [0.0007, 0.0036] sit below ε in gain, so not rising. All five TOST contrasts are equivalent. Page's trend test: ρ = 0.234, p = 0.051 (reported, not decided on).
- **Sanity check passed:** γ = 0 mean −0.0795 against floor −0.0816, with 3 of 10 seeds below the floor and the shortfall from `greedy_1` 0.0079.
- **Fixed rules** (the same held-out episodes): `greedy_1` −0.0716, FX −0.0786 (FX arm), `hyb` −0.0790, `fx_pair` −0.0786, `committed_pair` −0.0923, F −0.1146. Best γ (0.75, −0.0774) against the best fixed rule `greedy_1`: gain −0.0059, CI [−0.0066, −0.0050], p = 0.00195 (the 10-seed exact floor). `replace_fx = false`.
- **`greedy_1` against FX:** +0.0070, CI [0.0045, 0.0098], Cliff's δ 0.005, flag false (below ε).
- **Per cell** (`per_cell`, γ = 0.75 / FX / `greedy_1` / F): `jit-n12-90` −0.2448 / −0.2445 / −0.2392 / −0.2883; `jit-n12-180` +0.0899 / +0.0873 / +0.0960 / +0.0591; N = 6 controls: `jit-n6-75` 0.0546 / 0.0515 / 0.0550 / 0.0001; `jit-n6-150` 0.1083 / 0.1060 / 0.1116 / 0.0630.
- **Share of sorties with 2+ decisions** (`decisions`): `jit-n12-90` 0.63, `jit-n12-180` 0.37, `jit-n6-75` 0.10, `jit-n6-150` 0.03.
- **Stack check picks:** γ = 0.75 at seed 3, γ = 0 at seed 5 (`params.toml [rl.checkpoints]`).

**What follows by the rule** (applied in `params.toml`, 6 Oct): `keep_learned = false`; FX stays as FeRRy's in-flight rule; `gamma_star = 0.75` recorded only; the FQ arms leave batch 2 except 5.5's stack check (FQ-g75, FQ-g0); 5.7's learned scores (FQ-hand, FQ-dwell, FQ-cov, the reward grid) do not train; M1 is not built. 5.7 flies FX, FX-dwell (dwell term out of the plan score), FX-cov (coverage term out) and D4 (`driver.py:186-190`; built 7 Oct).

**Context for the size of these numbers** (from the same file): F, which has no in-flight slot, trails FX by 0.036 (−0.1146 against −0.0786) on the two cells read, about 18 times the learned-versus-fixed gap that was tested (0.002). The fixed in-flight rule already captures what the slot offers over the committed order; the question 5.5 asked is about the further step to a learned score, whose possible headroom is bounded by 8.1.

### 8.4 The "learner copied FX" diagnosis

**Observation.** greedy_1 sees only what FQ sees (`Greedy1Scorer` reads the `PairView` the features are built from; section 3). A working one-step learner (γ = 0) should therefore match it. FQ at γ = 0 does not: −0.0795 against −0.0716 in the sweep (0.0079 below, three of ten seeds under the floor), about 0.006 below on calibration `jittery56` and 0.0105 below on `clean`. It lands instead at FX's level. A pure ceiling would leave FQ near `greedy_1`, not near FX.

**Four candidate causes** (calibration findings; each with a fix the code already supports):

| # | Cause | Why it fits | Fix (existing flags) |
|---|---|---|---|
| 1 | Learns from data that mostly follows FX | The first 500 episodes fly FX's pair at ε = 0.3, then ε-greedy at 0.3 tapering to 0.05 by mid-run: little that differs from FX is tried | `ferrysim train --epsilon-start 0.5–1.0 --epsilon-end --reference-episodes` fewer or 0 |
| 2 | Learning rate too coarse | Differences that matter are about 0.01; large steps overshoot, matching flat bouncing curves | `--lr 3e-4` |
| 3 | Noisy checks stop training early | Bounce about as large as ε; "3 without a new best" can end a run by chance and pick a lucky checkpoint | `--val-episodes`, `--patience` |
| 4 | Network too small (64 × 64) | Less likely: greedy_1's rule is simple | Code change (`PairQConfig.hidden` is not a flag) |

**Caveats on cause 1 found in the code (not in the findings).** The reference phase is only the first 500 episodes (about 3,000 transitions, **derived** from about 6 transitions per episode: roughly 6 % of the replay by the end of a typical run). After it the greedy action is the network's own argmax, not FX's. "Mostly follows FX" therefore needs the network itself to agree with FX, which would be self-reinforcing, but **no agreement-with-FX share of the trained scores is recorded**: the evaluation files hold returns and decisions per sortie, not `agrees_fx`. The batch-2 stack check reports `pair_fx_agree_share` and is the first place the hypothesis can be measured against data. Until then the diagnosis rests on the greedy_1 comparison alone, which the findings document itself states.

**The unrun option-B screen** (fixed in the findings before it runs; rule allows one revision of the learner on the control cells before the sweep). Cells: the `clean` family, γ = 0, seeds 0 and 1, beside the calibration's default runs. Variants: V1 explore more (`--epsilon-start 1.0 --epsilon-end 0.05 --reference-episodes 0`); V2 finer steps (`--lr 3e-4`); V3 both; V4 steadier selection (`--val-episodes 400 --patience 6`, which changes only which checkpoint is kept). Pick: a variant qualifies if its γ = 0 clean-cell mean reaches the floor (greedy_1 − ε = −0.0984); take the highest of those; if none qualifies keep the defaults (the gap is a ceiling). The chosen settings would apply to every later pair-score training (E3 keeps Chen's). Cost: about 1.5–2 h for the screen plus about 30 min for a launcher change (a learner block under `[rl]`), then about 10 h for the sweep. **Decided 6 Oct 13:58: option A** (the sweep as pre-registered, default learner). Any use of the screen would bump `LEARNER_REVISION` to 1, which `report.py` allows. It was not run; the paper can say the learned score matched FX, not that no learner could do better.

---

## 9. E3 and its results

**What it is** (`policies/chen_dqn.py`, `next_stop.py`): a numpy port of Chen et al.'s DQN recipe (GLOBECOM Workshops 2023), labelled "DQN over the contact graph, after Chen et al.", never FedQMIX. It is a **legacy-mode whole-scheduler policy** (`contact_policy = "chen_dqn"`) that, unlike D1–D5, picks each next stop in flight. Before takeoff it admits every S3a contact, nearest first, dropping nothing; in flight there is no departure check (`in_flight_check = "none"`); at takeoff and at every Pass-1 departure it scores one row per remaining stop and takes the admissible row with the highest Q. `admissible(i)` is Chen's safety controller: S3b's single-contact predicate under `RULE_BUDGET` (transit, dwell, return leg and upload within the budget end, and the energy clause; **no deadline**). When none is admissible the answer is None, the pass ends and the stops left are reported (`pass_1_e3_unvisited`), never widened. It never acts in Pass 2. It flies the cell's one band (the checkpoint's class tuple is `["wide"]`), with no plan, deadline, coverage term or age cap.

**Observation** (10 columns, `e3_v1`, `chen_dqn.py:193`): `remaining` (constant 1), `snr` (÷ 10 dB), `reachable` (near 0), `dx`, `dy`, `distance` (÷ 100 m), `members` (÷ N), `return_energy`, `energy_left`, `time_left`. `remaining` and `reachable` are declared constant or near-constant, kept for fidelity.

**Declared deviations from Chen:** stops, not grid moves; one agent (QMIX with one agent is DQN); no learned digital twin or model-aided episodes; the pair learner's masked double DQN rather than QMIX/IQL settings; trained at K = 1 and flown per mule at K = 3 in Study 5.3. **Reward: bytes**, |C_k|/N. Scored on update yield and round closure in the paper, not raw bytes.

**Training** (`params.toml [rl.e3]`; manifests): family `jittery`, 5 seeds, Chen's settings: γ = 0.99 (the default of his published code), lr 5e-4, ε from 1.0 to 0.05 over the first half, no reference phase; otherwise the pair learner's defaults (replay 50,000, batch 64, warm-up 1,000, Adam, Huber, target sync 500, 200 validation episodes, patience 3). Stage `rl-e3`, 6 Oct 23:10 to 7 Oct 00:10; trained from commit `5b9a7aec`, clean tree. Checkpoints `results/exp5/checkpoints/5.3-e3/e3/g0.99_s{0..4}`; `rl.checkpoints.e3` is seed 0.

**Results** (`e3/evaluation.json`, 1,000 held-out episodes on each of the four `jittery` cells, bytes reward, mean over the four cells, **derived**):

| Policy | Bytes return |
|---|---|
| `greedy_1` | 2.0695 |
| FX (and `fx_pair`) | 2.0660 |
| `hyb` | 2.0659 |
| `committed_pair` | 2.0608 |
| F | 2.0521 |
| E3 seeds 0–4 | 1.9330, 1.9406, 1.9407, 1.9379, 1.9364 |

The five E3 seeds are within 0.4 % of each other (1.933–1.941, matching the readiness document), kept at episodes 2,000 / 5,000 / 7,000 / 5,000 / 3,000. Seed 0 has the lowest held-out score of the five and the median validation (1.9642). On the very reward E3 was trained on, E3 is about 6 % **below** FX and F on this stream, in every cell **(derived)**: for example `jit-n12-180` 2.200 against FX 2.276, `jit-n12-90` 1.643 against 1.687, `jit-n6-75` 1.690 against 2.055. E3 serves more stops per sortie (3.2 to 5.1 decisions per sortie in these cells, against 1.0–2.1 for the plan arms) but loses updates relative to the plan. These are FerrySim numbers on the bytes metric only; the stack comparison (time to τ, round closure, update yield) is Study 5.3's and 5.6's and has **not been run** for E3.

---

## 10. The older learner and the lineage

### 10.1 The legacy selector DDQN (`selector/ddqn.py`), by contrast

| | Legacy `TargetSelectorRL` / `DDQN` | Pair learner (`pair_q.py`) |
|---|---|---|
| Decision | Order devices within a bucket (intra-bucket) | (band, next stop) pair at each Pass-1 arrival |
| Runs when | A bucket holds at least two candidates | Every arrival with an admitted pair |
| Network | 11 → 16 → 1, tanh (`:62`), float32 | 36 → 64 → 64 → 1, tanh, float64 |
| Loss / optimiser | MSE, plain SGD (lr 0.01 default) | Huber (δ = 1), Adam (lr 1e-3), clip 10 |
| γ | Must lie in (0, 1) (`:103`); default 0.5; cannot sweep from 0 | [0, 1] |
| Target | Scores the single next row stored with the transition; **never takes a max** | Maximum over the next decision's admitted rows, double-DQN argmax by the online net |
| Replay | `ReplayBuffer`, default 10,000, one next row | 50,000, every next row plus its mask |
| Target sync | Every 200 steps | Every 500 updates |
| Training | `selector_train.TrainConfig`: 400 episodes, batch 32, warm-up 64, buffer 4,000, ε 0.9 → 0.05 over 300 episodes, rewards scaled by 1/150, in `ContactSim`/`BucketSim` | FerrySim, up to 10,000 episodes, ε 0.3 → 0.05 |
| Checkpoint | Format 1 weight file; the new loader refuses it by name | Format 2 with manifest and sha |
| Status | H2/H3 use the random-init path when no weights are supplied; H2 and H3 leave Exp 5; untouched because the H2 golden pins its seeded init and forward pass | The FQ arms |

The old selector is a data point: (selection-only, FL-aware, learned) tied (selection-only, FL-aware, fixed), H2 against H1. Learning did not help while the policy could only reorder visits that all happen anyway.

### 10.2 The prototype: `hermes_rl/` and `experiments/sim/drone_env/`

**Environment** (`hermes_rl/drone_env.py`): a Gymnasium environment (with a shim if Gymnasium is absent); the drone moves among fixed waypoints, transit takes time with no transfer, sensors within a radius are collected at 0.05 MB per step, uploads go to base stations over 3 channels with `R_k(d, t) = max(0, α·sin(ω·t + φ_k) + β/(1 + γ·d))` (`:217-224`); defaults α 0.5, β 0.15, γ 0.05, ω 0.15 (period 2π/ω ≈ 41.9 steps), phases 0, 2π/3, 4π/3. The **joint action** is one discrete index (waypoint × base station × channel): 5 × 3 × 3 = 45 in the default scenario (`:150-165`). The default scenario cannot be completed (its jobs need 600–1,600 collection steps against a 300-step cap, vendored README). The training scenario (`train_dqn.make_env_config`, `:492`, moved unchanged into the vendored `overloaded_config()`): 5 jobs, 7 waypoints (so the full action space is 7 × 3 × 3 = 63), ω = 0.25, 700 steps.

**Trainer** (`hermes_rl/train_dqn.py`): defaults (`:37`) 1,500 episodes, lr 5e-5, γ 0.99, batch 64, replay 50,000, target update 1,000, ε 1.0 → 0.05 over 1,000 episodes, two 128-unit ReLU layers, evaluation every 50 episodes on 20 episodes; Adam, Huber, clip 10, double-DQN target. **What the hybrid actually learns:** `train_hybrid` (`:584`) trains a Double DQN whose **action is which job to focus on (5 actions)**; a heuristic then executes the waypoint, base station and channel (`execute_for_job`, `:558`). Exploration is feasibility-guided (`feasibility_job`, `:291`), and observations are augmented with per-job feasibility scores (26 inputs). The module docstring still describes an older design (heuristic picks the waypoint, the DQN picks base station and channel, 45 → 9 actions); the code and the vendored README ("a Double DQN picks the job, a heuristic picks waypoint, base station and channel") disagree with it. `--mode dqn` trains a standalone DQN over the full joint action with a shaped reward (`train_dqn`, `:694`; it prints "45 actions" but the action space on this scenario is 63). The hybrid's heuristic reads the exact channel rate at the predicted arrival time, an oracle (vendored README, known issues).

**Vendored copy** (`experiments/sim/drone_env/`, 1,992 lines, 47 tests in `tests/unit/test_drone_env.py`): from github.com/FyneappleJuice/hermes_rl commit `a8a453f` (2026-04-28, the nested repo's only commit, author FyneappleJuice), copied 2026-09-28. Seeded, opt-in `ScenarioRandomization` (position jitter, random phases, deadline jitter, rate noise as a table over time; all off by default), and three golden rollouts recorded from the original files pin the default behaviour to `a8a453f`. `hermes_rl/` itself is an untracked nested git repository in the working tree (outer `git status`: `?? hermes_rl/`); the outer `.gitignore` does not cover it (finding G-01).

**Caveat on SEC'26 Table VI's 76.40.** The environment ignores seeds by default, so the trainer's periodic evaluations (seeds 2000+) and its final evaluation (seeds 9000+) all play the same deterministic episode. `train_hybrid` restores the best of about 30 evaluations (1,500 episodes at one every 50) and then re-measures it on that same instance. So 76.40 is a best-of-about-30 score on one instance, not a held-out mean. The committed plot (`hermes_rl/dqn_results.png`, viewed for this document) shows the hybrid's evaluation return at about +76 only at episodes 50, 100, 200 and 250, then at about −5 (the heuristic's level; 2 jobs done, 3 failed) from episode 300, with dips to about −85 (1 job done, 4 failed) later. The vendored README matches Table VI to this plot "by numbers, scenario and metrics" and says training was not re-run; `SEC26_Code_Audit.md` (section E, line 135) records that no code in the audited repository produced Table VI (provenance unknown). The README says that audit's question H4 records the provenance; **unverified**: no "H4" appears in the audit file.

**Licence (open item, verified state).** The author's repository has no licence file (vendored README; no LICENSE file in `hermes_rl/` or in the vendored directory). This repository has a root MIT `LICENSE` (commit `7d52958d`), which cannot grant rights the repository does not hold for the vendored code. The vendored directory is tracked on `main`. Open: ask the author, or remove `experiments/sim/drone_env/`. Phase 5 does not use it; E3 was written from scratch for exactly this reason (it needs PyTorch, sits where `hermes` may not import it and has no licence, `chen_dqn.py:8-11`).

**What the prototype does and does not show.** It established the time-axis channel and the joint-action framing that Phase 3 and the pair slot descend from. The memo's arithmetic for the anticipation gap (reproduced for this document with the memo's snippet: −0.14 committed against +0.75 best-available over four stops; period 41.9 steps) uses the default constants, a hand-picked tour W0→W3→W4→W1→W2, equal amplitudes and cosine phases where the environment uses sine; it is an illustration, not an environment rollout. It supports "committing at all is the loss", not any learning claim. The prototype's one learning result (Table VI) is a job-selection result on a different problem, with the caveats above.

---

## 11. Layer interfaces

| Layer | Supplies to the decision | Receives from it | Where |
|---|---|---|---|
| **L1** (RF) | Per-class SNR now (shadowing plus interference wave), per-class mean SNR for later stops, rate → dwell = 8N/rate(b, SNR), flight legs, energy model, P_c | The chosen band, actuated for the contact; dwell charged to the simulated clock | `hermes/l1/contact_link.py`, `mission_clock.py`; `FerryRuntime.arrival_view`, `observe`, `class_offsets_db`, `stop_contexts` |
| **L2** (scheduling) | The plan (b̄, queue, exempt stops, ages, weights), S3b predicate (mask, departure check), `replan_remainder` | The reordered remainder; `trimmed_next` and `mask_empty` records | `plan/types.py`, `fl_scheduler.py:1966`, `policies/pair_slot.py`, `routing/replan.py` |
| **L3** (hierarchical FL) | Merge weights (the reward's G_k), plan ages and `capped` flags, the cap S, coverage weights, deadlines | Misses widen ages for the next mission's demand | `aggregation_rules.update_weights`, `stages/s3d_age_cap.py` |

The scorer sees L3 only through columns (`age_next`, `capped_next`, `weight_share`, `on_time_next`, `slack_next`) and the reward. The hard gates always run before anything learned ranks anything; the slot imports nothing from `hermes.l1`, `hermes.mule`, `hermes.mission` or `experiments` (the runtime reaches it as a `PairView` and a callable).

---

## 12. Failure modes

| Mode | What happens | Guard |
|---|---|---|
| No pair admitted | Fly FX's band, index 0; recorded `mask_empty` | `PairChoice` invariants; common at last stops (about 19 of 71 at N = 6) |
| Scorer returns wrong length or non-finite | Refused | `check_pair_scores` |
| Pick the mask refused | RuntimeError | `pair_slot.py:807` |
| Wiring offers a non-admitted device or stop | `SelectorScopeViolation` | `assert_pairs_admitted` |
| `fits_pair` answers a non-bool | TypeError (a fold result passed whole would admit every pair) | `_verdict` |
| Slot called in Pass 2 or `band_at_arrival` called in Pass 1 | ValueError / RuntimeError | pass guards |
| Slot with `abort` | Refused at config | `config.py:931-938` |
| Untrained, unscored, dirty, wrong-tag, wrong-plan, relabelled or other-schema checkpoint | Refused (mule exits 1; runner refuses) | sections 5.6, `tag_refusals` |
| Divergence | `FloatingPointError` on non-finite loss or gradient | `PairQNet.update` |
| Symmetric layouts offer equal rows | Equal Q, lowest row wins, not rounding | `_scores` |
| Validation bounce as large as ε | Patience rule can stop a run by chance; an improvement under ε is invisible | findings; cause 3 |
| E3's mask is budget-only | E3 may serve stops with missed deadlines; no widening of dropped stops | `chen_dqn.py` |

---

## 13. Discrepancies found while writing this document

1. **Overview section 5.1 on the hybrid.** The overview (and `train_dqn.py`'s module docstring, and the vendored copy's docstring) say a heuristic picks the waypoint and the DQN picks (base station, channel), cutting 45 actions to 9. The code (`train_hybrid`, `execute_for_job`) and the vendored README say the DQN picks the **job** (5 actions) and the heuristic picks waypoint, base station and channel. The standalone mode's action space on the training scenario is 63, not 45. The overview's description of the prototype as the first demonstration that the *band decision* belongs on the flight clock should read as a statement about the environment's structure and the memo's arithmetic, not about anything the hybrid learned.
2. **Table VI.** The overview's "best-of-about-30 on one fixed instance" is confirmed (vendored README; 1,500 episodes / 50-episode evaluations). Added here: the committed plot shows the +76 at the first evaluations and about the heuristic's level afterward.
3. **Calibration findings, cells read.** The findings text says the calibration means are over "the four jittery cells above" and "the two clean ones". The verdict JSONs read the two N = 12 cells in each family (`cells` = `jit-n12-90`, `jit-n12-180`; `cln-n12-90`, `cln-n12-180`); the N = 6 cells are in `per_cell` only. The numbers quoted match the JSON.
4. **TOST bound.** The overview and the findings say TOST p ≤ 1.3e-4; the largest is 1.35e-4 (γ = 0.25).
5. **Fallback pair.** The overview says the mule "flies FX's pair" on an empty mask; the code flies **FX's band with no reorder** (index 0), not FX's nearest-stop choice (`types.py:785`, `:1253`).
6. **Stale pre-re-pin text.** `FeRRy_Learned_Pair_Score.html` (3 Oct, "not yet trained") lists N = 6 budgets 45/90 s and N = 12 budgets 120/180 s; the build plan's "As built" for 5.5 says "Not run" at 120/180 s, and for 5.6 lists P_c 104/52 s at 120 s; `cells.py`'s module docstring still names the priors. The re-pin block (`cells.py:277`, 75/150 and 90/180 s, P_c 108/54 and 136/68 s) and the records dated 5–7 Oct are current.
7. **Study 5.6 trial count.** `params.toml [s56]` counts 6 cells × 3 arms = 720 trials; with `keep_learned = false` the launcher drops FQ, leaving 480 (derived from `launch.py`, not from running `exp5 plan`).
8. **"Replay mostly follows FX."** The findings' cause 1 is a hypothesis, and the code qualifies it (section 8.4): the reference phase is 500 episodes of up to 10,000, and no agreement share of the trained scores has been recorded. Not a contradiction with the overview, which already calls it a hypothesis.
9. **The learned-versus-F size.** New here, not a discrepancy: in the sweep file F trails FX by 0.036, which puts the tested learned-versus-fixed gap (0.002) in context.

---

## 14. Open items

- [ ] **Option-B screen** (section 8): only if reviewers press on "the learner copied FX"; on the `clean` cells, not the family the sweep flies; first measure `pair_fx_agree_share` in the 5.5 stack check, which is in batch 2.
- [ ] **Study 5.5's stack check** (FQ-g75 seed 3, FQ-g0 seed 5, FX, F; 320 trials) and **Study 5.6** (E3 against FX, 480 trials): batch 2, ready, not run.
- [ ] **Study 5.7** (FX, FX-dwell, FX-cov, D4; 40 trials per cell at N = 12): built 7 Oct, not run; no pre-registered ε.
- [ ] **M1** is not built; build only if a later result keeps a learned score. **Irregular interference** likewise.
- [ ] **A plan-time test of delayed consequence** (γ across missions) is untested; the memo's rule says a hierarchy is not entered without it.
- [ ] **`drone_env` licence**: ask FyneappleJuice or remove `experiments/sim/drone_env/`; decide what to do about the untracked nested repo `hermes_rl/`.
- [ ] **`train_dqn.py` docstring** in `hermes_rl/` and the vendored copy describes a design the code does not implement; the vendored copy's README documents the real behaviour, so only the docstring needs a note (the vendored files are otherwise unchanged by policy).
- [ ] **Citations**: Chen et al. 2023 and Bayerlein et al. 2021 into `HERMES_Related_Work_Notes.md`.
- [ ] **Refresh stale text** listed in discrepancy 6 (build plan "As built", Learned Pair Score note, `cells.py` docstring) when those documents are next edited; this series edits none of them.

---

## 15. Sources

`hermes/scheduler/policies/{pair_slot,cross_heuristic,chen_dqn,next_stop}.py` · `hermes/scheduler/plan/types.py` · `hermes/scheduler/fl_scheduler.py` (`fits_after_service`) · `hermes/scheduler/selector/{scope_guard,pair_features,pair_q,pair_replay,ddqn,replay,features,target_selector_rl,selector_train}.py` · `hermes/mule/mule_main.py` · `hermes/processes/config.py` · `experiments/ferrysim/{cells,episode,inprocess,reward,train,checkpoints,evaluate,headroom,report,__main__}.py` · `experiments/exp4/driver.py` · `scripts/exp5/{params.toml,launch.py,repin.py}` · `results/exp5/rl/{headroom,calibration,s55,e3}/*.json` and `results/exp5/checkpoints/*/*.json` · `hermes_rl/{drone_env,train_dqn,dqn_results.png}` · `experiments/sim/drone_env/README.md` · `tests/unit/test_drone_env.py` · [Experiment_5_RL_Calibration_Findings.md](../Experiment_5_RL_Calibration_Findings.md) · [Experiment_5_Readiness.md](../Experiment_5_Readiness.md) · [Experiment_5_Reproducibility_Guide.md](../Experiment_5_Reproducibility_Guide.md) (section 5.3, 5.4) · [FeRRy_Build_Plan.html](../FeRRy_Build_Plan.html) (Phase 5, Studies 5.5 to 5.7) · [FeRRy_Learned_Pair_Score.html](../FeRRy_Learned_Pair_Score.html) · [Layer redefinition and RL decision memo](../architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md) · [SEC26_Code_Audit.md](../SEC26_Code_Audit.md) · [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md).

*Nothing was re-run for this document except the memo's four-stop arithmetic, the schema's column counts, and sums over the committed JSON files (marked derived).*
