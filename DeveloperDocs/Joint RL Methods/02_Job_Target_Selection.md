# Job target selection — which devices and contacts are served, and in what order

*7 Oct 2026. Part of the series that starts with [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md). A reading document for the Layer-2 "Who" decision: it restates the code and the records, decides nothing, and says "unverified" where a number could not be checked against its source. File:line references were read against the working tree on 7 Oct 2026; the Configuration Reference's own line numbers have drifted (see §13). Numbers are copied from the named records; none were re-run. The companion for the gate that bounds this decision is [03_Feasibility_Gate_and_Mission_Clock.md](03_Feasibility_Gate_and_Mission_Clock.md).*

---

## 0. The short version

"Job selection" here means one question: **given the devices a mule's slice holds, which are served this mission, as which stops, and in what order?** It is the Layer-2 job (design principle 1: RL is *How*, the scheduler is *Who*, HFL is *When*). Two implementations answer it:

| Path | Who answers | Arms | Learned? |
|---|---|---|---|
| **Legacy pipeline** — S1 → S3 → S3a → S3b → S3.5 (`FLScheduler.build_contact_queue`) | Hard rules; then a distance sort or the DDQN selector inside each bucket | H1 (H2 = H1 + selector; H3 = H2 + adaptive backhaul) | Only the S3.5 selector (random-init in every recorded run) |
| **Whole-scheduler baselines** — a policy that owns S3, S3b and S3.5 (`admit_and_order`) | The policy's own key, admitted through a shared budget walk | D1 MAX-AoI, D2 Oort, D3 Whittle, D4 FedEx-CARP, D5 FedCS | No |
| **Plan mode** — `build_ferry_plan` (the F family) | A search over (band class, route, member subsets) ranked by the cap key, served weight share, then V | F, FX, FB+, ablations | No |

Four things to carry away.

1. **Learning never decides who is eligible.** S1, the S3b predicate and the age cap run first; whatever ranks after them only permutes (S3.5) or picks among masked options (the flight slot). This is design principle 12, enforced by `selector/scope_guard.py`, and it holds for every arm.
2. **The deadline admits, but in the legacy path it does not order.** S3b walks contacts in earliest-deadline order to decide who fits, then the queue is regrouped by bucket and sorted by distance from the mule (`fl_scheduler.py:992-1022`). The build plan records the same fact ("H1 flies bucket-then-distance order and discards S3b's EDF order").
3. **A ranking policy only matters when the budget truncates the queue.** Exp 4's sweep B found H1, B1 (MAX-AoI), B2 (Oort), H2 and H3 *byte-identical* in `final_auc` and `update_yield` at N = 6 for every budget from 120 s down to 15 s. Batch 1 shows the same shape on the new clock: H1 and D1 are both 282 s at N = 6, one mule, knee; with three mules H1, D1, D2 and D3 all sit at 64–65 s (§10).
4. **FeRRy's lead over the selection-only arms comes from the plan, not from a better ranking.** F reaches τ = 0.71 in 194 s against 266–282 s for H1, D1, D2, D3 at the N = 6 knee, but *which* mechanism in F earns that (band choice, member subsets, the age-weighted objective) is not isolated yet: Study 5.4 has not run, and the one batch-1 ablation that touches the cap (5.14) finds nothing at N = 6.

---

## 1. Decisions made in this layer

Clock: **plan** = at the dock, once per mission; **flight** = at a stop or departure; **dock** = on the slow-phase handoff.

| # | Decision | Clock | Inputs | Output | Who decides | Learned? | Code path |
|---|---|---|---|---|---|---|---|
| J1 | Is a device a candidate? | Plan | `is_in_slice`, `deadline_override_ts`, `last_beacon_ts` | Eligible ids | S1 rule | No | `stages/s1_eligibility.py:47-66`; called at `fl_scheduler.py:830`, `1581` |
| J2 | Is an advert admitted at contact? | Contact | `FLReadyAdv` state, `issued_at`, utility | Admit / reject | S2A, S2B (device computes, mule verifies) | No | `s2a_readiness.py:92`, `s2b_flag.py:139`, `FLScheduler.ingest_ready_adv` `:676`. **No runtime caller** (§3.2) |
| J3 | When is a device due? | Plan | Φ, idle reference, S3c scale, override | `Deadline(j)` | S3 law | No | `s3_deadline.py:369` |
| J4 | How does Φ move? | Flight (fast), dock (slow) | Outcome, `answered`, amendment | New Φ | S3 law | No | `s3_deadline.py:472`, `:569` |
| J5 | Which tier? | Plan | `is_new`, `missed_count`, slice flag, beacon age | NEW / SCHEDULED / BEACON_ACTIVE | S3 | No | `s3_deadline.py:427` |
| J6 | Are all windows too tight? | Per mission | served / planned over the last 5 missions | Window scale 1.0–4.0 | S3c (off by default) | No | `s3c_mission_window.py:41-115` |
| J7 | Which devices share a stop? | Plan | Positions, deadlines, buckets, radius R | `ContactWaypoint`s | S3a | No | `s3a_cluster.py:99` |
| J8 | Which stops fit? | Plan, flight | Stops, clock, budget, model | Kept / dropped with a reason | S3b predicate | No | `s3b_feasibility.py:527` (see doc 03) |
| J9 | In what order inside a bucket? | Plan (Pass 1) | Per-contact features (11) | Permutation | Distance sort, or `TargetSelectorRL` | Selector: yes, offline | `fl_scheduler.py:992-1022`, `target_selector_rl.py:267` |
| J10 | Who and in what order, as a whole | Plan, in-flight re-plan | Contacts, device states, budget, model | Route | D1–D5 `admit_and_order` | No (E3 is learned) | `fl_scheduler.py:899-937`, `1311-1339` |
| J11 | Which devices are protected by age? | Plan | `last_merged_round`, mission round, S | Ages, capped set, cap key | S3d | No | `s3d_age_cap.py:148`, `:182` |
| J12 | Who does the plan serve? | Plan (F family) | Demand, weights, cap, classes | (b̄, route, members) | `plan_search` by the plan key | No | `fl_scheduler.py:1454`, `plan/plan_search.py` |
| J13 | Which stop next, in flight? | Flight (F family) | Remaining stops, observed SNR | Next stop (and band) | FX rule, or FQ score | FQ yes — flat | `policies/pair_slot.py`, `cross_heuristic.py` (overview §2) |

J1–J9 are the Exp 4 pipeline; J10 is Amendment 4; J11–J13 are FeRRy Phases 4–5.

---

## 2. The legacy pipeline, stage by stage

```
S1  eligibility          HARD GATE   slice membership OR deadline override OR a fresh beacon
S3  deadline + bucket    RANK TIER   Deadline(j); NEW > SCHEDULED_THIS_ROUND > BEACON_ACTIVE
S3a RF clustering        REGROUP     devices within R of an anchor -> ContactWaypoints
S3b feasibility          HARD GATE   drop what cannot be reached in time or afford (opt-in: needs a budget)
S3.5 intra-bucket order  ORDER ONLY  distance sort, or the DDQN selector when a bucket holds >= 2 contacts
```

Entry points: `build_contact_queue` (Pass 1) at `fl_scheduler.py:789` and `build_pass_2_queue` at `:1367`. `build_target_queue` (`:695`, the per-device single-pass API the design text describes) is never called in Exp 4 (L2 record §1); S3 runs per device there and S3.5's `select_order` (`s35_selector.py:175`) is the distance placeholder.

### 2.1 S1 — eligibility
`eligible(i) = has_active_deadline(i) ∨ beacon_heard(i)` (`s1_eligibility.py:47`). `has_active_deadline` is `is_in_slice or deadline_override_ts is not None` (`:29`); a beacon counts only within `beacon_window_s` = 30 s (`:34`, default `:50`). In Exp 4 S1 admits on slice membership alone and rejected zero devices in 7,200 device-missions of probe (L2 record §4.4), because no beacon source is wired.

### 2.2 S2A and S2B — designed, not exercised
S2A (`s2a_readiness.py:92`) admits an advert whose state can open a session and whose `issued_at` is within 5.0 s; S2B (`s2b_flag.py:139`) is the strict `utility > FL_Threshold`, default 0.60 (`:136`). `ingest_ready_adv` has no runtime caller; the gate that runs is an inline check in `HFLHostMission.run_contact` with `min_utility = 0.0`, which cannot reject. Scheduler Freeze **D3** removes readiness gating from the contribution claims rather than wiring it. Study 5.12 (not run) reintroduces readiness through D5's `not_ready` test, a different mechanism.

### 2.3 S3 — the deadline and the bucket

```
Deadline(j) = now + Φ(j)·scale − idle(j)          idle(j) = max(0, now − idle_time_ref_ts), 0 if never served
            = idle_time_ref_ts + Φ(j)·scale        when the device has been served (scale = S3c scale, 1.0 off)
```
(`compute_deadline`, `s3_deadline.py:369-401`.) The second form is why a device left unserved longer than Φ has a deadline in the past, and S3b drops it as overdue (Config Reference §15, "One interaction to know"). A never-served device has `idle = 0`, so its first deadline is `now + Φ`. `deadline_override_ts` short-circuits the formula (`:389`).

**Bucket** (`classify_bucket`, `:427`): `NEW` if `is_new` and `missed_count < NEW_BUCKET_ATTEMPT_LIMIT` (3, `:76`) or if no lower tier would accept it; else `SCHEDULED_THIS_ROUND` if in slice; else `BEACON_ACTIVE` if a beacon is fresh; else `ValueError`. The probation limit exists because an unreachable device never clears `is_new` and would hold the top tier forever. In Exp 4 every round holds one non-empty bucket (`NEW` ×6 in round 1, `SCHEDULED` ×6 after; L2 record §4.3a), so the tier walk never discriminates. In the mission-clock campaign a device that failed twice could still sit in `NEW` while others are `SCHEDULED`; whether batch 1 ever held two buckets is **unverified**.

### 2.4 S3a — RF-range clustering (`s3a_cluster.py:99`)
Greedy, anchor-based, deterministic:

1. Sort the remaining devices by `(−delivery_priority, bucket index, deadline)` (`_device_priority_key`, `:77`) and take the first as the anchor.
2. The cluster is the anchor plus every un-clustered device **within R of the anchor** (not of each other).
3. The stop is the cluster centroid if every member is within R of it, else the anchor's own position.
4. The stop inherits the best (lowest-index) bucket among members and the **earliest** member deadline.

Properties: lossless (every input device appears once); order-dependent; the radius is a parameter, so plan mode runs it once per band class at that class's radius (`fl_scheduler.py:1614-1620`). Design principle 15 describes the members as served "in parallel"; the ferry model charges the **sum** of member dwells (one shared channel, `FerryPhysics.dwell_s`, `s3b_feasibility.py:399`). `delivery_priority` is a cluster-side carry-over copied onto the mule's state at `ingest_slice` (`fl_scheduler.py:610-623`) so the anchor order sees the current value. Pass 2 clusters the whole slice, ignoring S1, and orders nearest-first from the mule's pose (`order_pass_2_greedy`, `s3a_cluster.py:187`; no selector).

### 2.5 S3b, then the bucket walk and S3.5
S3b is the gate (doc 03). Its survivors are regrouped by inherited bucket and walked in `BUCKET_PRIORITY` order (`fl_scheduler.py:970-1022`). Inside a bucket:

- **≥ 2 contacts and a selector wired:** `rank_contacts(members, states, env, pass_kind=COLLECT, admitted=bucketed)` (`:1004-1019`). The `admitted` argument is the S1/S3-admitted set so the scope guard can fire.
- **Otherwise:** `sorted(members, key=distance from mule_pose)` (`:987`, `:1021`). The design text carves out the single-candidate short-circuit explicitly (design §2.7).

If `validate_flown_order` is on (the `replan` response), S3b's EDF-admitted set is then folded in the order about to be flown and repaired when it fails (`_validate_order`, `:1038`; doc 03 §5).

### 2.6 The priority key
`FLScheduler(miss_priority=True)` passes `_contact_miss_priority` (the longest `miss_streak` among a contact's members, `:505`) to `filter_feasible`, which then sorts by `(−priority, deadline, position, devices)` (`s3b_feasibility.py:811-817`). Its purpose: a missed device's wider window pushes its deadline later, so without the key it is also sent to the back of S3b's queue. **It is off in every batch-1 arm that runs the legacy path**: the H1 and D1–D3 job arguments carry no `--miss-priority`. In the F family the driver sets it on for every plan arm but F-prio (`experiments/exp4/driver.py:1190-1200`), where it multiplies the coverage weight (§6.2) instead of reordering S3b, because plan mode never calls `filter_feasible`.

---

## 3. The deadline law

State per device: Φ (`deadline_fulfilment_s`, initial Φ₀ = 60 s, `types/scheduler.py:33`), `idle_time_ref_ts`, `miss_streak`, `last_clean_ts`, `last_clean_round`, `last_merged_round`, `reach_attempts`, `reach_answered`. The law is a `DeadlineLaw` (`s3_deadline.py:118`); `None` is the recorded additive law with its original arithmetic.

### 3.1 Additive (the recorded law; `s3_deadline.py:62-66`)
- CLEAN: Φ ← max(5, Φ − 5). Any other outcome: Φ ← Φ + 10. Floor 5 s, no ceiling. Break-even on-time rate 10 / (10 + 5) = 2/3.
- Cluster overrides are sticky: nothing ever clears one (SEC26_Code_Audit).
- Constants are hand-set, sized "against missions of about 10 s of wall clock".

### 3.2 Multiplicative and clamped (FeRRy; `DeadlineLaw.next_window`, `:271`)
```
Φ ← min(Φ_max, max(Φ_min, β·clamp(Φ)))        clamp applied BEFORE the step, so Φ·β holds from outside the clamps
β = β_on 0.8 (CLEAN) | β_partial 1.25 (miss by a device that answered) | β_timeout 1.5 (never answered)
Φ_min = 5 s, Φ_max = 300 s      (Config Reference §15; defaults at `:165-169`)
```
`answered` is `RoundCloseDelta.answered`; the synthetic TIMEOUTs the mule feeds devices it dropped carry `answered=False, synthetic=True` and do not count as reachability observations (`:521`). Expected drift per contact is zero at p\* = ln β_timeout / (ln β_timeout − ln β_on) = 0.645 (Config Reference §15; the class docstring says "about 0.65"). More reliable devices tighten toward Φ_min, less reliable relax toward Φ_max. The β values and clamps are placeholders "until the theory track supplies them". Overrides expire once their time passes or the next outcome arrives (`expire_overrides`, default on for this law).

### 3.3 Time unit
All constants are stated in the recorded unit and multiplied by `time_scale` (default 1.0, the recorded law exactly). The Q1 setting is `time_scale = T_nom / 10 s` (`time_scale_for_period`, `:676`; `LEGACY_MISSION_PERIOD_S = 10.0`, `:673`), so the recorded Φ₀ = 60 s is six missions of window. The pilot probe that motivated it: at scale 1.0 a 14-mission narrow-band run served the field-wide contact once and then found it overdue at every later takeoff, "by 20 s more each mission" (build plan, Phase 3 pilot notes; T_nom was 172–250 s in the final check's trials). Batch 1 passes `--deadline-time-scale t_nom` to every arm; **only the F family also passes `--deadline-law multiplicative`** — H1 and D1–D3 run the additive law.

### 3.4 Fast phase, slow phase, S3c
- **Fast** (`fold_round_close_delta`, `:472`): per contact outcome. CLEAN clears `is_new`, refreshes `idle_time_ref_ts`, `last_clean_ts`/`last_clean_round`, resets `miss_streak`, shrinks Φ. Any miss widens Φ, extends `miss_streak`, and does **not** refresh the idle reference or the last-clean fields.
- **Slow** (`fold_cluster_amendment`, `:569`): at the dock. Applies `deadline_overrides` (refused on the simulated clock — absolute wall stamps; none are issued today), clamps a cluster-supplied Φ, folds `spectrum_sig` per-class SNR (nothing decides on it), and `delivery_priority`.
- **S3c** (`MissionWindowAdapter`): window 5, target success 0.8, gain 2.0, scale 1.0–4.0; `scale = min(max_scale, 1 + gain·max(0, target − rate))` where rate is pooled Σserved/Σplanned; widen-only, a pure function of history, off by default, and its denominator includes S3b's own pre-flight drops. Exp 4 expected it to do nothing without a budget; it was never part of a recorded headline.
- **Unit U11 laws** `round` (after FedCS: every device due at `now + round_s`, merge window = the round) and `pref` (after Oort: no per-device cutoff; slow devices weigh less). Neither moves Φ. Built for Study 5.2; not run.

---

## 4. S3.5 — the intra-bucket DDQN selector

**What it is.** A pointer-style scalar-Q network: each candidate contact is scored on its own and sorted by descending Q, ties on `(position, devices)` (`target_selector_rl.py:267-304`). `q = tanh(xW1 + b1)W2 + b2`, 11 → 16 → 1, numpy (`selector/ddqn.py:82`); defaults `hidden=16`, `lr=0.01`, `gamma=0.5`, `target_sync_every=200` (`:92-98`). No fixed output head, so one network ranks a variable number of contacts.

**The 11 features** (`features.py:11-25` for devices; `extract_features_for_contact`, `:181`, for contacts). Slot 5 changes meaning for contacts, so a per-device model cannot be reused per contact.

| Slot | Per device | Per contact (what Exp 4 ranks) |
|---|---|---|
| 0 | distance mule–device / 100 | distance mule–stop / 100 |
| 1–3 | x, y, z / 100 | stop x, y, z / 100 |
| 4 | on-time rate (0.5 if never seen) | mean on-time rate over members |
| 5 | beacon fresh (0/1) | member count / 5, clamped to 1 |
| 6–8 | bucket one-hot (NEW, SCHEDULED, BEACON_ACTIVE) | same, the contact's inherited bucket |
| 9 | mule energy (0–1) | same; frozen at 1.0 in Exp 4 (L2 record §4.3) |
| 10 | `rf_prior_snr_db / 30` | same; one trial-wide scalar in Exp 4 |

**Guards.** Pass-kind guard: raises `SelectorScopeViolation` outside Pass 1 (`_enforce_collect_pass`, `target_selector_rl.py:52`). Scope guard: `assert_candidates_admitted` re-checks every member against the admitted set (`scope_guard.py:39`). The selector returns a permutation of its input; it cannot add, drop or re-bucket. `assert_pairs_admitted` (`:57`) extends the same principle to the flight slot's (band, next stop) pairs.

**Training** (`selector_train.py`; offline only — Exp 4 never imports it). A single agent on `BucketSim` / `ContactSim` (`sim_env.py`): reward `−time − w·energy + 200·[completed]`, `SESSION_TIME` 30 s, `TIME_PER_DIST` 0.1, noise σ 1.0, bucket 6, budget 150; `TrainConfig`: 400 episodes, ε 0.9 → 0.05 over 300, batch 32, warm-up 64, buffer 4,000, reward scale 1/150. The next-state bootstrap takes the online argmax over the remaining candidates and the target network's value of it, i.e. double-Q. The design says "CTDE on AERPAW digital twin"; as built nothing is multi-agent and no twin was used (`selector_train.py` docstring; SEC26 audit divergence D-1). The sim is "not physics": the completion bonus is set to dominate distance so the policy can differ from the placeholder at all.

**Why it tied H1 — five reasons that stack, from the records.**

1. *It was never trained.* Every committed H2 run used a random-init network (Freeze D4; Methodology §2). H2 vs H1 "measures nothing about learned scheduling". (A second defect, L2 record §5.2: `rng_seed` seeded only ε, so even the random weights were not reproducible; fixed, but the on-disk rows predate the fix.)
2. *It only reorders.* Both arms visit every contact in the queue once; "the same work in a different order" (L2 record §3).
3. *There is nothing to reorder across.* One non-empty bucket per round (§4.3a), so the walk is a no-op and the selector orders the whole round, but ordering a queue that is visited whole changes no recorded metric. Sweep B: H1, B1, B2, H2, H3 byte-identical at N = 6, `rrf` 60, every budget 120 s → 15 s; ordering can only matter when the queue is truncated in flight, and the binding band is narrow (identical at N = 6 over 100–150 m and 15–120 s; *different* at N = 12 / 300 m / 45 s, B1 yield 0.75 vs H1 0.25).
4. *Its inputs were thin.* Several feature columns are constant within a single-bucket batch (L2 record §4.6) and `rf_prior_snr` is one scalar per trial.
5. *The outcome was saturated.* H1's `final_auc` is 0.922–0.933 in all twelve A2 cells; the mule leaves almost no variance for a scheduler to explain (decision memo §0).

**Status.** H2 and H3 leave Exp 5 (build-plan decision 8; Freeze §5l); the runner's `--require-trained` refuses them without `--selector-weights`. The S3.5 code, the guards and the pair-slot extensions stay.

---

## 5. The age cap S and S\* (`stages/s3d_age_cap.py`)

**Why.** Nothing else bounds how long a device waits: Φ has no ceiling under the additive law and is clamped at Φ_max under the multiplicative one, and a device unserved for longer than Φ_max after its last on-time contact stays overdue. In plan mode only; no legacy module imports the file (Freeze Rule 1).

```
a_j(m) = m − U_j       m = mission being planned (first mission = 1), U_j = last_merged_round, 0 if never
capped_j  ⇔  a_j ≥ S − L        (L = lookahead, 0; L = 1 does not remove critic probe A3's miss)
```
(`device_age`, `:148`; `evaluate_cap`, `:182`.) Counted in the device's **own mule's missions** since its last *merged* update — the scorer's unit, so a CLEAN the merge cutoff excluded is not service, and a never-merged device is capped from mission S on. Decision 1 of 30 Sep (build plan D4: "not cluster rounds"); D5's cluster-round unit is for the merge cutoff a_max, a different quantity.

| Stop kind (on the route actually folded) | Rule | Function |
|---|---|---|
| exempt — every member capped | skips its own deadline clause (`protected`) | `is_exempt` `:215`, `cap_stops` `:259` |
| mixed — some members capped | not exempt; carries the earliest deadline of its uncapped members | `stop_deadline` `:278` |
| priority — any member capped | flies first in a trim and sheds uncapped members first | `priority_first` `:326`, `capped_first` `:342` |

**The cap key** (`cap_key`, `:361`) is the ages of the capped devices a plan leaves out, largest first, compared first in the plan key: (4, 4, 4) beats (5, 3). **S\*** is the smallest S that covers 90 % of 30 reference layouts at both pilot budgets (knee and stress), never below 2 (`analysis/age_cap_s_star.py`, Freeze §5g). Pilot result: **S\* = 2 at N = 6, 12, 18, 24** (one mission covers the knee, two the stress budget; N ≥ 12 by the tool's greedy bound). **Violations** are reported by cause with no pass mark: `unplannable`, `crowded` (at plan time), `dropped_in_flight`, `not_merged` (at close); device availability alone makes about 15 % of device-missions miss at S = 3 (critic A2). A capped device whose S3a stop cannot serve it alone gets a one-device **hover stop** at its best point (`plan/hover.py`, decided 30 Sep).

**What the evidence says.** At N = 6, 40 trials per cell, network AoU: cap S\* 0.715, cap off 0.715 (difference 0, Holm p = 1) at the knee; 0.81 and 0.81 at the stress budget; S\* + 1 also identical (Study 5.14, `results/exp5/scores/b1/s514.md`). So **the cap is not detectably doing anything at N = 6 on this metric.** Why is **unverified**; one reading is that the age-weighted objective already ranks old devices first, so the cap key rarely changes the plan. Study 5.8 (F−cap, F−prio, D3, D1; N, budgets from the pilots) is where the cap would have to show; it has not run.

---

## 6. Plan mode: who the plan serves

### 6.1 Demand, then search
`build_ferry_plan` (`fl_scheduler.py:1454`) runs S1 and S3 exactly as `build_contact_queue` does (`:1581-1601`; repeated rather than refactored out of the frozen method), evaluates the cap (`:1603`), weights demand (`:1605`), then for each class the arm may fly runs S3a at that class's radius, the hover rule, and prices Pass 2 once (`:1613-1633`). `plan_search` returns the best candidate; a guard fold re-checks it under S3b's predicate with exempt stops protected, and a failure raises (`:1641-1650`).

### 6.2 Coverage weights and the plan key
```
w_j = max(a_j, 1) · (1 + miss_streak_j)     "age" mode with miss_priority on (arm F; about a_j², critic A5)
w_j = max(a_j, 1)                            F-prio ; "uniform" mode: 1, or (1 + m_j)
```
(`coverage_weight`, `plan/plan_score.py:334`; `demand_weights`, `:406`). The plan key is a total order, so the pick never depends on enumeration order: **(cap key, −served weight share, −V, class index, stops)** under the default `coverage_rank = lexicographic`; `weighted` drops the share term (`plan/types.py:640-670`). V is in the overview §1.3 and not repeated here. The empty plan is a candidate and is chosen only when nothing that serves anyone is admitted. Search size: `exact` ≤ 6 devices, `stop_subsets` ≤ 6 stops per class, `local` above (overview §1.2).

### 6.3 Drops by reason
A device the plan leaves out is labelled with the first clause of S3b that refuses it alone from the dock (`_plan_drops`, `:1707`), else the plan-level reason `plan` (a choice, not a shortfall; excluded from S3c's planned count). The mule widens every one as a synthetic TIMEOUT at takeoff time (`mule_main.py:1671`).

---

## 7. The baselines that compete

**The fairness rule** (Amendments 4 and 8): a baseline replaces *our policy*, not *our physics*. S1 and S3a still run; the budget and our `FeasibilityModel` are handed through unchanged; a baseline owns S3's order, S3b's admission and S3.5's tie-break by exposing `admit_and_order` (`fl_scheduler.py:899-937`). Each declares `in_flight_check`: `budget` (D1, D2, D3, D5; the default) or `none` (D4). All of D1–D3 and D5 admit through `greedy_budget_walk` (`policies/budget_walk.py:75`): sort by the arm's key, fold under `RULE_BUDGET`, **skip rather than stop** at a contact that does not fit ("strictly more favourable to the baseline"). Under `member_admission="subset"` a contact that fails whole is re-issued with its best members that fit, each ranked by the arm's key on its one-member contact then by device id (`_walk_subsets`, `:136`); only the plan before takeoff does this.

| Arm | Source | Rank signal and key (descending unless noted) | Admission | What it replaces | Main deviations | Code |
|---|---|---|---|---|---|---|
| **D1** MAX-AoI | The standard AoI comparator | Contact age = **stalest member**'s `now − last_clean_ts`; never served = ∞ (first); ties by distance to the mule, then ids | Budget walk | S3 tiers, S3b, S3.5 | Age from last CLEAN, not last attempt (Amendment 5) | `max_aoi.py:65`, `:135`, `:149` |
| **D2** Oort | Lai et al., OSDI 2021 | Σ over members of `n·|loss| + 0.1·ln R / √L`; any unexplored member → ∞ (first). R = 1 + max `last_served_round`; L = `last_clean_round` | Budget walk | same | Mean loss, not RMS; rounds, not wall time; **bonus is `0.1·ln R/√L`, not Oort's `√(0.1·ln R/L)`, and the utility is not normalised, so the bonus is about 1e-4 of the utility** (Freeze §5e, an undocumented fidelity deviation); system-speed term restored only with the fit clock (Study 5.12, α = 2); needs `--real-model` | `oort.py:90`, `:105`, `:302` |
| **D3** Whittle | Cui et al., TMC 2024, eq. 48 | `expected` index: `ω·[ρ·x(x−1)/2 + x]`, x = R − `last_merged_round` (Cui's off-by-one), ρ̂ = (answered + 1)/(attempts + 2) clamped to [0.05, 1]; contact index = Σ members; ties `(position, devices)` | Budget walk | same | Λ unobserved at planning, so the paper's I(x,1) is replaced by an **expectation of our own** (`literal` kept for sensitivity); ω uniform by default | `whittle.py:209-231`, `:398` |
| **D4** FedEx-CARP | Bian et al., TMC 2025 | A 2-OPT **visit-all** tour closed at the depot; no ranking | None; never truncated | same | One transporter per mission, so the Gibbs assignment never runs; no energy gate; `in_flight_check = none`; reports its own overrun. Runs as `D4` (route only, `agg:cutoff`) and `D4fedex` (its merge, `agg:fedex`) | `fedex_carp.py:811` |
| **D5** FedCS (degraded) | Nishio and Yonetani, ICC 2019 | Greedy argmin marginal time (`unit`) or max devices/time (`devices`) from the current end of the route | Admit if the budget fits; line 4's unconditional removal | same | No Resource Request (last-known state only), C = 1, t^UD = 0 until the fit clock exists; **not run** (Study 5.12, batch 3) | `fedcs_degraded.py:230` |
| A2 | Exp 3 ablation | S3a's registration order, no skipping | None | S3.5 | "Do nothing" baseline | `arrival_order.py:212` |
| A3 | Exp 3 ablation | EDF on `deadline_ts` | Skip when `transit + collect + return + upload > remaining budget` | S3.5 | Own `FeasibilityModel` (session 30 s, nearest base station); not S3b's | `edf_feasibility.py:303` |

`policies/` is **not** frozen: adding a comparator cannot change a recorded HERMES arm, but a baseline that needs new state does (Oort's loss, Amendment 3; the last-CLEAN fields, Amendment 5).

**Reading D3 honestly.** Cui's index is optimal per decoupled single-arm problem and Cui's own simulations rank it best of four; there is no optimality theorem for the coupled policy, and the "expected" variant is not a theorem of the paper (Related Work §6a, rows 1–2). **Batch 1 flew D3 with the `expected` variant and `ω` uniform**: no `--whittle-weights` appears in any batch-1 job (manifest `20261006_003651`, e.g. job `s53/n6k1_knee__D3`), and the runner default is `uniform` (`runner_main.py:934`).

---

## 8. The 2 × 2, and where each arm sits

Two axes (decision memo §6.2): **decision scope** (permute a given route, or choose the next stop) and **objective denomination** (FL units, or bytes and age).

| | **FL-blind** (bytes, age) | **FL-aware** (updates, rounds, deadlines) |
|---|---|---|
| **Selection-only** — permutes within a given route | D1 MAX-AoI | H1 (our S3 + S3b + S3.5); D2 Oort; D3 Whittle; D5 FedCS (degraded); the old H2 |
| **Trajectory + selection** — chooses the next stop | E3, a DQN over the contact graph after Chen et al. | F (committed), FX (fixed rule), FQ (learned) — one architecture, three fillings |

D4 is outside the grid on this axis: it chooses the tour but ignores every deadline and gate. Three tests fall out: *scope* (read down), *awareness* (read across), and the **interaction** — whether trajectory control is worth more when the objective is FL-aware (difference of differences on paired seeds). The RL question is nested in the bottom-right cell. Batch 1 filled the F, FX, H1, D1–D4 cells; **E3, H0 and O1 are in batch 2, D5 in batch 3**, so the interaction test cannot be read yet. The old H2 is the one data point already on the grid: (selection-only, FL-aware, learned) tied (selection-only, FL-aware, fixed) — for the reasons in §4, not because learning was tried and failed.

---

## 9. Layer interfaces

| Edge | What crosses | Where |
|---|---|---|
| **L1 → L2** | Band class → range R → S3a's radius. In legacy runs R is the constant `rf_range_m` = 60 m (Exp 4's two-pass switch). Observed per-band SNR enters plan mode's flight slot, not S3. `rf_prior_snr_db` reaches S3.5 as slot 10: a trial-wide scalar in Exp 4, the causal last-observed SNR on the mission clock | `s3a_cluster.py:99`; `features.py:133`; Freeze §5j "Causal RF prior" |
| **L2 → L1** | Only the waypoint (design principle 5). Plan mode additionally commits b̄ | `fl_scheduler.py:1688` |
| **L3 → L2** | `RoundCloseDelta` (outcome, `answered`, loss, examples) → Φ, ages, miss streak. `last_merged_round` (`record_merged`, `:553`) → the age cap. Merge cutoff a_max_j = ⌊Φ_j·s/T⌋ uses the same Φ and is snapshotted once per mission right after planning (`_age_caps`, `mule_main.py:3094`) so the deadline and the cutoff never disagree | `s3_deadline.py:404` `effective_window` |
| **L2 → L3** | Devices a plan drops or an abort abandons are fed a synthetic TIMEOUT, so their Φ widens (`_widen_abandoned`, `mule_main.py:3149`). Coverage weights and the flight reward reuse the merge weight (overview §4) | |
| **Cluster → L2** | `MissionSlice` + `ClusterAmendment` at the dock → `ingest_slice` (`:569`) | |

---

## 10. Evidence

### 10.1 Batch 1, scored 6 Oct (`results/exp5/scores/b1/`; time to τ = 0.71, simulated s, lower is better; 20 paired trials; Holm within study; "claim" = CI excludes 0 and Holm p < 0.05)

| Cell | F | FX | H1 | D1 | D2 | D3 | D4 |
|---|---|---|---|---|---|---|---|
| N = 6, 1 mule, knee 150 s | **194** | 178 (no claim) | 282 | 282 | 266 | 280 | 264 |
| N = 6, 1 mule, stress 75 s | 178 | 175 | 220 | 238 | 229 | 209 | 264 |
| N = 6, 3 mules, knee and stress | 52.3 | 56.4 | 64.6 / 64.0 | 64.6 / 64.0 | 64.6 / 64.0 | 64.6 / 64.0 | 55.0 |
| N = 12, 1 mule, knee 180 s | 486 | 424 | 589 | — | — | 507 | 459 |
| N = 24, 1 mule, knee 262 s | **749** | 789 | 951 | — | — | 911 | 752 |

Claims against F: at the N = 6 knee all of H1, D1, D2, D3, D4 (Holm p ≤ 0.008); at the stress budget only D4; with three mules only D4fedex (101 s, p ≈ 1e-3); at N = 24 H1 and D3 (p ≈ 2e-5, 4e-5), not D4 or FX. FX is 16 s ahead of F at the knee, not a claim (Holm p 0.146 in 5.3's family).

**Read this as a statement about plans, not rankings.** H1 and D1 differ only in the ranking key yet land on the same 282 s; with three mules the budget never binds (each mule serves two devices) and four different selection rules give the same mean. D3's 280 s is no better than H1's. The only arms that differ are those that plan differently: F and FX (a plan over classes and member subsets) and D4 (no gate at all). The SOTA document draws the same line ("FeRRy beats every SOTA arm only at N = 6 with one mule at the knee budget, by 26–31 %").

### 10.2 Gate activity in the same trials (trial means of the scorer's `sim_*` columns; per trial of 4 missions; units per `HERMES_Configuration_Reference.md` §17.7)

| Arm, N = 6, 1 mule | Replans, knee | Replans, stress | Empty missions, stress | Budget overrun rate, stress |
|---|---|---|---|---|
| H1 | 0.05 | 0.10 | 0.45 | 0.01 |
| D1 / D2 / D3 | 0.00 / 0.10 / 0.00 | 0.30 / 0.35 / 0.35 | 0.40 / 0.40 / 0.25 | 0.03 / 0.04 / 0.03 |
| D4 | 0.00 | 0.00 | 0.05 | **0.85** (mean overrun 18.96 s) |
| FX | 0.00 | 0.10 | 0.10 | 0.24 (mean overrun 4.86 s) |
| F (`s514/…__capS`, first 20 trials) | 0.00 | 0.05 | 0.10 | 0.34 (mean overrun 11.6 s) |

The selection-only arms re-plan rarely (at most 0.35 per trial) but at the stress budget lose 0.25–0.45 missions per trial to emptiness; D4 flies its whole tour and overruns in most missions. F and FX serve more of the queue and so overrun more than H1 and D1–D3 do: the gate prices the mean SNR, so a realised mission is longer than planned (doc 03 §3.4). F's `cap_violations` are 4.6 (knee) and 5.0 (stress) per trial of 4 missions × 6 devices. F's N = 6 one-mule files live in `s514/` as `capS` (the manifest aliases `s53/n6k1_knee__F` to it).

### 10.3 Earlier records and why they are not used above
- **Exp 4 matrix (13 Aug).** H1 vs H0 under jitter: AUC 0.927 vs 0.873, completion +0.191 (A2); H0 wins every clean metric (A1); H3 vs H2 ties end to end (C). There is **no H2-vs-H1 row** in `HERMES_Matrix_Results.md`; "H2 tied H1" is the decision memo's statement, resting on the sweep-B probe recorded in the Holistic Revision Plan (§4, reason 3).
- **SOTA pilot (13 Aug): D1/D2 vs H1** — at 120 s the baselines collect more (`update_yield` +0.225, +0.200); at 60 s H1 wins accuracy (+0.073 over D1, p = 0.0087; +0.088 over D2, p = 0.0033, n = 40), "the advantage is conditional on the budget binding". **Superseded for citation:** every budgeted cell ran under the stale budget stamp (Amendment 6), every D1/D2 cell under the age-reset defect (Amendment 5) and the in-flight deadline re-check (Amendment 8). The build plan says re-run before citing.

---

## 11. Failure modes and recorded defects

| What | Effect | Fixed by |
|---|---|---|
| S3 computed a deadline nothing compared to a clock | ~34 % of contacts (26/76) dropped at any budget once S3b was on; −0.225 mission completion at 120 s | S3b (opt-in); L2 record §4.2 |
| Gate-dropped and abandoned devices got no feedback | Φ never widened; the same devices dropped forever (a starvation loop made by the gate) | `_widen_abandoned`, Amendment 1 |
| Every loop was per device | A systemically tight schedule looked like bad luck | S3c, Amendment 2 (off by default) |
| **D1/D2 age reset by any outcome** | Real TIMEOUTs/PARTIALs and the synthetic TIMEOUT reset the age; in `b60` traces only 34 of 608 scheduled D1 Pass-1 devices came back CLEAN, so D1's route changed in most missions of every `--realism` cell | Amendment 5: `last_clean_ts`/`last_clean_round`, set only by a CLEAN |
| **Stale budget stamp** | Stamped only in `ingest_slice` (DOWN); an empty mission skips the dock, so the next mission planned against the old stamp; 72–81 % of `b60` missions were empty; one D1 trial's budget fell from 60 s to ~39 s | Amendment 6: `start_mission()` at every mission start (`fl_scheduler.py:526`, `mule_main.py:1115`, `:1604`) |
| In-flight re-check held D1/D2 to our per-device deadline | MAX-AoI puts overdue devices first, so the check refused its own first choice and aborted the route | Amendment 8: `in_flight_check` declared per policy |
| A plan's diagnostics outlived the plan | An early return left the last mission's `last_feasibility`, so devices were widened twice | Reset on entry (`:823-828`, `:1529-1534`) |
| Flown order never validated | Probe P1: a 70 s walk flown as 150 s | `_validate_order` under `replan` (`:1038`) |
| H2/H3 dead-zone sweep did not vary the dead zone | Five "conditions" were one configuration | Matrix design; L2 record §5.1 |
| Untrained selector not reproducible | `rng_seed` seeded ε only | Threaded to the network; rows on disk predate it |
| S3a re-clusters every mission | A device the plan keeps leaving out lands at its own far position, where the cap could never serve it (final check PLAN-1) | Hover stops |
| Narrow-band 0-or-N cliff | One field-wide contact admitted whole or not at all | `member_admission="subset"` (F default; H/D opt-in) |
| Partition drift | At S\* + 1 FB+medium crowded 7 times in 291 missions at 45 s and FB+wide twice in 360; at 30 s every family crowds | Reported, not corrected |

---

## 12. Open items

- [ ] **Study 5.4 and O1** — the first test that can attribute F's lead to the band-class choice; batch 2.
- [ ] **Study 5.8** — F−cap, F−prio, D3 (fairness reference), D1 at the pilot budgets; batch 2. It is where the cap and the priority key are tested; 5.14 found nothing at N = 6.
- [ ] **Study 5.2** — F against F·add, F·round, F·pref (unit U11 built; not run).
- [ ] **D5 / Study 5.12** and **E3, H0, O1** — outside batch 1, so the 2 × 2's bottom-left cell and the interaction test are unread.
- [ ] **The Oort staleness-bonus form** (`0.1·ln R/√L` vs `√(0.1·ln R/L)`) — recorded as "not changed here" in Amendment 5; declare it wherever D2 is reported or fix it before batch 2.
- [ ] **D3's ω** — decide between uniform (what ran) and Oort-weighted (what two documents say), and state it.
- [ ] **Manuscript sign** — Algorithm 1 and the deadline equation print the update with the sign reversed; the code is correct (Freeze §4).
- [ ] **Whether batch 1 ever held two buckets** — would show whether the tier walk is exercised on the new clock (unverified).
- [ ] **Cluster-round vs mission-round age** — the cap counts the mule's missions; the merge cutoff a_max counts cluster rounds. Both are documented, but they are different units for the same device.

---

## 13. Doc and code discrepancies found while writing

1. **D3's ω.** `FeRRy_Build_Plan.html` (baselines table, Phase 2 "done" line) and `HERMES_SOTA_Baseline_Candidates.md` §0.2 say ω comes from Oort's statistical utility. The code default (`whittle.py:395-398`, `driver.py:722`, `runner_main.py:934`) and every batch-1 D3 job use `uniform`.
2. **D3 at N = 12.** SOTA Candidates §0.4 gives 526 s (n.s.); the scored `s59_arms.csv` and `s59.md` give 507 s for D3 at `n12k1_knee`.
3. **"H2 tied H1".** Stated in the decision memo and the overview, but `HERMES_Matrix_Results.md` records H1 vs H0 and H3 vs H2 only; the support is the sweep-B probe (byte-identical outputs), which says the arms did not differ, not that a learned selector failed to help.
4. **"Four of the eleven slots are constant"** (L2 record §4.6; decision memo cut 2). By code, within a single-bucket batch the three bucket one-hots, mule energy and `rf_prior_snr` are all constant (five), plus z when devices are planar. The count does not change the conclusion; I did not re-run the Exp 4 batch.
5. **Stale line references in the Configuration Reference.** §7 cites `s3_deadline.py:44-48` (now 62-66) and `fl_scheduler.py:79` for `beacon_window_s` (now 111); §5 cites `ddqn.py:88-91` (now 92-98). §5 also lists the replay default (10,000) but not `TrainConfig`'s own 4,000, warm-up 64 and batch 32.
6. **Arm naming drift.** `oort.py`'s docstring still says "arm B2" and `max_aoi.py` "B1" in places; the arms are D2 and D1 since Phase 2.
7. **Build plan "Gate" line** says S3b checks "transit + session + return + upload" against `min(Deadline(j), budget)`. The code default bounds the *collection* (`arrival + dwell ≤ Deadline(j)`) and applies return and upload only to the budget clause; the min form is `deadline_bounds = delivery_per_stop` (the plan's own "Deviations" list says so). Detail in doc 03 §2.
8. **`selector_train.py` / Design §2.7** say CTDE on an AERPAW twin; as built it is one agent in a toy sim (acknowledged in the module docstring and SEC26 audit D-1).

---

## 14. Sources

`hermes/scheduler/{fl_scheduler.py, stages/s1_eligibility, s2a_readiness, s2b_flag, s3_deadline, s3a_cluster, s3b_feasibility, s3c_mission_window, s3d_age_cap, s35_selector}.py` · `hermes/scheduler/selector/{ddqn, features, target_selector_rl, selector_train, sim_env, scope_guard}.py` · `hermes/scheduler/policies/{max_aoi, oort, whittle, fedex_carp, fedcs_degraded, edf_feasibility, arrival_order, budget_walk}.py` · `hermes/scheduler/plan/{plan_score, types}.py` · `hermes/mule/mule_main.py` · [FL Scheduler Design](../HERMES_FL_Scheduler_Design.md) (§2.1, §2.7, §6.8, §7) · [Scheduler Freeze](../HERMES_Scheduler_Freeze.md) (§1–§5g, Amendments 1–8, 11) · [Exp 4 L2 record](../HERMES_Experiment4_L2_Scheduling_Layer.md) · [Configuration Reference](../HERMES_Configuration_Reference.md) (§4–§7, §15, §18.2) · [SOTA Baseline Candidates](../HERMES_SOTA_Baseline_Candidates.md) · [SOTA Results](../HERMES_SOTA_Results.md) · [Matrix Results](../HERMES_Matrix_Results.md) · [Related Work §6a](../HERMES_Related_Work_Notes.md) · [FeRRy Build Plan](../FeRRy_Build_Plan.html) · [Decision memo](../architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md) · [Experiment 5 readiness](../Experiment_5_Readiness.md) · `results/exp5/scores/b1/{s53, s59, s514}.md`, `results/exp5/b1/_launcher/manifest_20261006_003651.json`, `scripts/exp5/params.toml`.
