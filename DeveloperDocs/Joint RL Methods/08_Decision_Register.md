# HERMES / FeRRy — decision register: every runtime decision, by layer and clock

*7 Oct 2026. A reference table. It lists each decision the system makes while it runs, what feeds it, what bounds it, who decides it (a rule, a search, a trained model, or a number a person set), and where the code is. It restates the code and the records; it decides nothing, and it re-ran nothing. Companion to [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) (what is optimised jointly, and the evidence) and [09_Work_Record.md](09_Work_Record.md) (how the work went). Status of every study: [Experiment_5_Readiness.md](../Experiment_5_Readiness.md).*

*How this was checked. Each row was read against the module named in its "code path" column (all under `hermes/` unless a path starts with `experiments/`). Facts of the form "has no caller" come from a repo-wide search of `hermes/`, `experiments/` and `scripts/` for the function's name, 7 Oct 2026. Values for constants come from the code default or from `scripts/exp5/params.toml`, whichever the campaign flies. Where a row rests on a document alone it says so.*

---

## 0. The short version

**There are 39 decisions in the register.** Counted by who decides them in the arms Exp 5 calls FeRRy (**F**, and **FX**, which differs from F in the in-flight rule only):

| Decider | Count | Which |
|---|---|---|
| **Fixed** (a formula, predicate or table with no free choice) | 27 | P01–P03, P08–P14, F01, F02, F05, F06, F09–F11, M01–M04, K01–K06 |
| **Heuristic** (a greedy or ordering rule that picks among options) | 9 | P04, P05, P15 (the H and D arms' order), F03 and F04 (FX's rule), F07, F08, K07, K08 |
| **Search** (enumerates or hill-climbs against a hand-set score) | 2 | P06 (band class b̄), P07 (route π) |
| **Learned** (a trained model) | **0 active in F or FX** (K09 is the one learned row; it is unwired) |  Two slots can be filled by a learned filling: F03 and F04, by the pair score FQ; F04 also by E3; P15 by the legacy selector H2 |

**The only learned runtime component in FeRRy is the in-flight pair score FQ** (`hermes/scheduler/selector/pair_q.py`, flown through `policies/pair_slot.py` when `flight_slot = "pair_q"`). It fills the same slot as the fixed rule FX (rows F03 and F04), inside the same mask. Study 5.5 found it flat against FX, and `keep_learned = false` was applied on 6 Oct, so **no arm that Exp 5 calls FeRRy runs a learned decision.** FQ survives only as a stack check in batch 2 (FQ-g75, FQ-g0). Section 6 lists every learned component in the repository and whether anything runs it.

Three things the register shows that are easy to miss:

1. **The plan clock is search, not learning, by design.** b̄ and π (P06, P07) are chosen by exhaustive or local search against a hand-set score V (P08). No plan-time value is learned.
2. **Two gates the design documents list do not run.** The on-contact readiness threshold of 0.60 and its 5 s freshness window (`ingest_ready_adv`) have no runtime caller; the live utility check admits everything with utility ≥ 0 (F01). And the cluster's slow-phase deadline update (K05) is inert for Φ, since the cluster issues no deadline overrides.
3. **Many "decisions" are numbers.** Section 7 lists 54 constants that act as decisions. Most were set by a person and swept once or not at all; a few were measured by a pilot (budgets, τ, S\*, session timeouts).

---

## 1. How to read it

**Layers.** L1 = RF communication. L2 = mobility-aware scheduling. L3 = hierarchical federated learning. A decision that spans two is written `L1+L2`.

**Tiers.** *device* (ground node), *mule* (the drone's onboard computer), *cluster* (the edge server). Cloud (tier 3) makes no decision in the register.

**Clocks.** *plan* = at the dock, once per mission. *flight* = at each stop and arrival of Pass 1, plus the contact itself. *merge* = at the mule when Pass 1 closes. *dock* = at the cluster, the inter-pass dock, and Pass 2.

**Arms.** F = plan search, committed in-flight slot. **FX** = F with the fixed cross-heuristic in the in-flight slot. FQ = F with the learned pair score in that slot. FB+c = band class pinned to c. F-cov, F-cap, F-prio, F-add, F-round, F-pref, F+L1, FX-dwell, FX-cov = one mechanism of F switched. H1 = HERMES as submitted. D1 MAX-AoI, D2 Oort, D3 Whittle, D4 FedEx with CARP, D5 FedCS (degraded) = published schedulers ported. E3 = DQN over the contact graph after Chen et al. H0 = flat FL on a live link (wall clock only).

**Hard gates always run first.** S1, the S3b predicate and the age cap run before anything ranks; a learned or heuristic choice only picks among what they admit (Freeze principle 12; `selector/scope_guard.py`). Each row's "gates" column says what bounds that row.

---

### 1.1 One mission, in decision order

The order in which a plan-mode mission (F or FX) takes its decisions, with the row of each. Legacy arms (H, D) skip the bracketed steps and use P15.

```
 DOCK, before takeoff (plan clock)
   P01 eligible set -> P02 deadlines -> P03 tiers -> P11 ages and capped set -> P09 weights
   for each band class c:  P04 stops at R(c) -> P05 hover stops for capped devices
                           -> P07 route search, each stop through P10 admission
   P08 rank the classes' best plans  -> P06 commit b-bar  -> P12 guard fold, commit, label drops
   P13 widen what the plan left out
 FLIGHT, Pass 1 (flight clock) — repeats for each stop
   F06 departure check -> [F07 re-plan, or abort] -> F08 beacon insert (inert)
   F04 next stop  (FX: nearest that fits; FQ: with F03 and F05)
   fly -> F03 band at arrival -> F02 who is solicited -> F01 readiness -> F09 update ready?
   F10 session outcome -> F11 deadline-law update
 MERGE (merge clock), Pass 1 closes
   M02 cutoffs (snapshot taken at planning) -> M01 merge weights -> M03 merged set and ages
   M04 empty round?
 DOCK (dock clock)
   K01 backhaul carrier and upload -> K02 cluster merge at rate eta (K03 quorum)
   K04 close the round, DOWN -> [K05 slow-phase deadline update] -> K06 priority bumps
   K08 slices (fixed at seeding) -> K07 Pass 2: nearest-first delivery at b-bar
   (K09 dock server choice: never reached)
```

### 1.2 Which arm decides what

A cell shows who decides the row in that arm: **rule** (Fixed), **greedy** (Heuristic), **search**, **learned**, or a dash where the arm has no such decision. D4 and H1 are compared on the same stack, with their own admission.

| Arm | Band class (P06) | Route (P07 / P15) | Per-arrival band (F03) | Next stop (F04) | Re-plan (F07) | Merge (M01) | Backhaul (K01) |
|---|---|---|---|---|---|---|---|
| F | search | search | rule (keep b̄) | rule (plan order) | greedy (trim) | rule (`agg:cutoff`) | rule (fixed carrier) |
| FX | search | search | greedy (fastest covering class) | greedy (nearest that fits) | greedy (trim) | rule | rule |
| FQ | search | search | **learned** | **learned** | greedy (trim) | rule | rule |
| FB+c | pinned | search | rule | rule | greedy (trim) | rule | rule |
| F+L1 | search | search | rule | rule | greedy (trim) | rule | rule (adaptive) |
| H1 | none (wide) | greedy (EDF admit, tier, distance) | none | none | greedy (trim) | rule | rule |
| D1 / D2 / D3 / D5 | none | greedy (their own admit-and-order) | none | none | their own | rule | rule |
| D4 | none | greedy (CARP, 2-OPT; no gate) | none | none | none | rule (`agg:cutoff`, or `agg:fedex`) | rule |
| E3 | none | none | none | **learned** (admissible stops only) | none | rule | rule |

---

## 2. Plan clock — at the dock, once per mission

All plan-mode rows run inside `FLScheduler.build_ferry_plan` (`fl_scheduler.py`) unless a row says otherwise. Plan-mode arms are F, FX, FQ, FB+c and the F-ablations; H and D arms plan with `build_contact_queue`.

| ID | Decision | Layer / tier | Inputs | Output | Decider | Hard gates that bound it | Code path | Arms that vary it |
|---|---|---|---|---|---|---|---|---|
| P01 | Eligibility (S1) | L2, mule | slice membership, deadline override, last beacon time, beacon window | eligible device ids | Fixed | is the first gate | `stages/s1_eligibility.py:is_eligible` | none. In Exp 5 membership alone admits: no beacon source is wired |
| P02 | Deadline(j) (S3) | L2 (L3 reads it) | Φ_j, idle time since last CLEAN, S3c scale, override | one timestamp per device | Fixed | Φ floor 5 s (additive) or clamp [Φ_min, Φ_max] (multiplicative); an override expires under the multiplicative law | `stages/s3_deadline.py:compute_deadline`, `DeadlineLaw` | additive: H1, D1–D5, F-add; multiplicative: F, FX, FB+, F-cov, F-cap, F-prio, F+L1; `round`: F-round; `pref`: F-pref |
| P03 | Bucket tier (NEW > SCHEDULED > BEACON) | L2, mule | is_new, missed count, in-slice flag, beacon age | tier tag per device | Fixed | a device no tier accepts is dropped and logged; NEW probation ends after 3 failed attempts | `stages/s3_deadline.py:classify_bucket` | H1 and H2 queue by tier, then distance. Plan arms: tier enters only through S3a's anchor key (the plan key ignores it) |
| P04 | Clustering into stops (S3a) | L2, mule | eligible ids, positions, deadlines, delivery_priority, radius R(b) | stops: centroid if within R of every member, else the anchor's position; stop tier = worst, deadline = tightest | Heuristic (greedy anchor sweep) | every member lies within R of its stop | `stages/s3a_cluster.py:cluster_by_rf_range` (once per class in plan mode) | radius 60 m (wide) in legacy; per class in F, FX, FQ; one class in FB+c |
| P05 | Hover stop for a capped device | L1+L2 | capped devices their S3a stop cannot serve alone, class model, budget | a one-device stop at the best point on the dock–device segment | Heuristic (closed form) | within the class's reach and above the SNR floor; only capped devices move | `scheduler/plan/hover.py:offer_hover_stops` | off in 5.14's `hoveroff`; on elsewhere |
| P06 | Contact band class b̄ | L1+L2 (+L3 weights) | demand, per-class stops, per-class feasibility model (rate, dwell), Pass-2 price, cap, weights | the committed class, flown in both passes (`set_band`) | **Search** (smallest plan key over classes) | every stop passes S3b under that class; cap key ranks first; the empty plan is a candidate | `fl_scheduler.py:build_ferry_plan` → `plan/plan_search.py:plan_search`; `mule/mule_main.py:_run_ferry_mission` | F, FX, FQ search wide, medium, narrow; FB+wide, FB+medium, FB+narrow pin one (`band_class_policy`); H, D, E3 fly `contact_band` with no decision |
| P07 | Route π, with member subsets | L2 (L1 prices) | the class's stops, deadlines, budget, cap, weights | an ordered list of stops, each reduced to the members that fit | **Search**: `exact` (≤ 6 devices), `stop_subsets` (≤ 6 stops), `local` (2-OPT plus scans) | S3b predicate from takeoff; guard fold must pass | `plan/plan_search.py:_exact`, `_stop_subsets`, `_local`; `plan/member_subset.py`; `routing/two_opt.py` | `whole` (5.14), `local` forced (5.14; 5.11 (a) modes `subsets`, `local`) |
| P08 | Plan ranking | L2 ↔ L3 | each candidate's V terms, served weight share, cap key | the pick | Fixed (a total order over the search) | the order is (cap key, −share, −V, class index, stops), so no pick depends on enumeration order | `plan/plan_score.py:plan_key`, `score` | `weighted` rank (5.14 `wcov`, the κ sweep); F-cov (κ = 0); FX-dwell; FX-cov |
| P09 | Coverage weight of each demanded device | L3 → L2 | age a_j, miss streak, speed factor | w_j | Fixed | every demanded device weighs more than 0 | `plan/plan_score.py:coverage_weight`, `demand_weights` | F-prio (age alone); F-pref (Oort speed factor); `uniform` mode |
| P10 | Admission predicate (S3b) | L2, mule | stop, flight state (pose, clock, energy, deliver_by), budget end, rule, band model | admit or reject with a reason (overdue, budget, energy, delivery); the next state | Fixed | is the gate: nothing ranks before it | `stages/s3b_feasibility.py:FeasibilityModel.admit`, `fold`, `filter_feasible` | rule: deadline+budget (F family, H1–H3), budget only (D1–D3, D5, Pass 2), none (D4); `deadline_bounds`; `member_admission` |
| P11 | Age cap and capped set | L3 → L2 | mission m, last merged round, S, lookahead L | capped set, exempt and priority stops, cap key | Fixed (S is a constant, section 7.2) | capped devices come first in the plan key; an all-capped stop skips only its own deadline clause | `stages/s3d_age_cap.py:evaluate_cap`, `cap_stops`, `cap_key` | F-cap off; 5.14 `capS1`, `capoff`; H and D arms have none |
| P12 | Plan commit | L2 | the winning candidate | `PlanCommit`; the mule's model becomes b̄'s; every demanded device the plan leaves out is labelled | Fixed | a guard fold must pass S3b as flown, else `FLSchedulerError` and nothing commits; wall time is recorded, never used | `fl_scheduler.py:build_ferry_plan`, `_plan_drops`, `_plan_servable`, `close_plan` | plan arms only |
| P13 | Widening of devices the plan left out | L2 → L3 | drops by reason; devices abandoned in flight | a synthetic TIMEOUT: Φ widens, miss streak + 1 | Fixed | D arms' drops are reported, never widened (user's decision 6) | `mule/mule_main.py:_widen_abandoned`; `stages/s3_deadline.py:fold_round_close_delta` | D arms do not widen |
| P14 | Mission-window scale (S3c) | L2 | served over planned for the last 5 missions | a multiplier ≥ 1 on every window | Fixed | only widens; capped at 4 | `stages/s3c_mission_window.py:MissionWindowAdapter` | **off** in every Exp 5 cell (no flag set) |
| P15 | Pass-1 order in the legacy arms (S3.5) | L2 | admitted contacts, pose | the queue | Heuristic (nearest first); **Learned in H2** (11→16→1 DDQN, only for buckets of 2 or more); the policy's own for D1–D5 | scope guard; pass guard (Pass 2 refused); S3b ran first | `stages/s35_selector.py:select_order`; `fl_scheduler.py:build_contact_queue`; `selector/target_selector_rl.py:rank_contacts`; `policies/*` | H1, H2 (left Exp 5), D1–D5 |

---

## 3. Flight clock — at each arrival and contact

| ID | Decision | Layer / tier | Inputs | Output | Decider | Hard gates that bound it | Code path | Arms that vary it |
|---|---|---|---|---|---|---|---|---|
| F01 | Readiness at contact (S2A, S2B) | L3 device, L2 mule | the device's state and utility (0.7·performance + 0.3·diversity); advert | session opens, or the device is refused and nothing is pushed | Fixed (live threshold 0.0) | state must be `FL_OPEN`; `utility < min_utility` refuses, and `min_utility` is 0.0 | `mission/host_mission.py` (the `adv.is_eligible()` and `min_utility` tests in `_collect_session` and `_ferry_collect_worker`); `mission/client_mission.py:serve_once`; `mission/utility.py` | none. The 0.60 threshold and 5 s freshness in `stages/s2a_readiness.py`, `s2b_flag.py` run only via `FLScheduler.ingest_ready_adv`, which only tests call |
| F02 | Who is solicited at a stop | L1+L2 | stop, members' positions, class, arrival time, channel SNR, fit clock | targets, and unreachable members (never solicited, no airtime) | Fixed | planar distance ≤ R_planar(b) and SNR ≥ the floor (−6.7 dB); a member whose fit is unfinished gets an advert and no push | `mission/contact_plan.py:ContactPlan.at_arrival`; `mule/ferry.py:contact_plan` | class per arrival (F03) |
| F03 | Band for this stop | L1+L2 (+L3 reward for FQ) | SNR per class now, targets and dwell per class, the committed class; FQ also the 36-column pair rows | the class this stop is served on | Fixed (F, FB+: keep b̄); **Heuristic (FX)**; **Learned (FQ, opt-in)** | must reach every device b̄ reaches here; FQ: the mask F05; Pass 1 only, Pass 2 flies b̄ | `policies/cross_heuristic.py:fastest_covering_class`, `CrossHeuristic.band_at_arrival`; `policies/pair_slot.py:PairQSlot.pair_at_arrival`; `mule/mule_main.py:_ferry_band_at_arrival`, `_ferry_pair_at_arrival` | F, FB+: committed; FX: fastest covering class; FQ, FQ-g0 … FQ-g99; FerrySim references `fx_pair`, `committed_pair`, `hyb`, `greedy_1` |
| F04 | Next stop (or home) | L2 (+L3 reward for FQ) | remainder, pose, `fits(order)`; FQ rows; E3's per-stop observation and admissibility | index of the stop to fly next | Fixed (F: index 0); **Heuristic (FX: nearest stop that keeps the rest feasible)**; **Learned (FQ; E3)** | only stops of the plan; the whole remainder must still fit; decided after the departure check and the beacon hook; not at takeoff, not in Pass 2 (E3 also acts at takeoff) | `mule/mule_main.py:_ferry_next_stop`, `_ferry_e3_next_stop`; `policies/cross_heuristic.py:CrossHeuristic.next_stop`; `policies/chen_dqn.py`, `next_stop.py` | F, FB+: plan order; FX; FQ; E3 (legacy mode, no plan) |
| F05 | Pair mask and fallback | L2 | every (class, stop) pair at this arrival | the admitted pairs; with none, FX's pair, logged `mask_empty` | Fixed | admitted only while the whole rest of the flight still fits at the observed SNR (S3b predicate) | `fl_scheduler.py:fits_after_service`; `policies/pair_slot.py:bind_fits_pair`; `selector/scope_guard.py:assert_pairs_admitted` | FQ and its references |
| F06 | Departure check | L2 | remainder, flight state, budget end, in-flight rule, protected stops, δ_obs = 0 | keep, re-plan, or abort | Fixed | the same predicate as P10 | `mule/mule_main.py:_ferry_departure`; `fl_scheduler.py:fold_remainder`, `in_flight_rule` | `in_flight_response`: `abort` (code default), `replan` (every Exp 5 cell; FQ requires it); D4 none |
| F07 | Re-plan, or abort | L2 | the failed remainder | a new remainder; every dropped stop is final for the mission and widened | Heuristic (trim, or reorder by 2-OPT then EDF) | the result must pass the predicate; protected stops first | `fl_scheduler.py:replan_remainder`, `_trim_plan`; `routing/replan.py:replan_route`; `routing/two_opt.py` | `replan_fallback`: `trim` (Exp 5; plan mode always trims), `reorder` (code default); D1–D3, D5 re-admit by their own rule |
| F08 | Beacon insertion | L2 | an offered contact, remainder, flight state | the cheapest insertion that passes the fold, else refused with a reason | Heuristic | never evicts a planned stop; the fold must pass without skipping | `mule/mule_main.py:offer_contact`, `_ferry_try_insert` | **inert**: `offer_contact` has no caller outside tests |
| F09 | Device-side training and "not ready" | L3 device (mule's fit clock) | pushed basis θ, local shard, 1 epoch, batch 64, FedProx ρ; Study 5.12: fit time T_j | a prepared Δθ with its basis version, or no update | Fixed | a Pass-1 contact finds an update ready only if t ≥ fit start + T_j | `mission/client_mission.py:train_offline`, `_handle_collect_push`; `mule/fit_clock.py:FitClock.not_ready` | training-time levels (5.12, batch 3); every other cell is always ready |
| F10 | Session outcome | L2 mule | receipt: round, byte count, checksum, form, age | CLEAN, PARTIAL or TIMEOUT | Fixed | a failed check is PARTIAL; receipt age over 2 × `session_ttl_s` is PARTIAL | `mission/host_mission.py:_verify_receipt` | none |
| F11 | Deadline-law update, fast phase | L2 → L3 | the outcome, Φ, whether the device answered | the new Φ_j; `is_new`, `miss_streak`, last-CLEAN fields | Fixed | floor 5 s, or [Φ_min, Φ_max] | `stages/s3_deadline.py:fold_round_close_delta`, `DeadlineLaw.next_window` | law (see P02) |

---

## 4. Merge clock and dock — at the mule, then at the cluster

| ID | Decision | Layer / tier | Inputs | Output | Decider | Hard gates that bound it | Code path | Arms that vary it |
|---|---|---|---|---|---|---|---|---|
| M01 | Mule-side merge weight | L3, mule | n_i, value v_i, age a_i = v − b_i, cutoff a_max_j | Δ_m = Σ w_i Δθ_i / M_m with w_i = n_i·v_i·s(a_i); M_m is staleness-free | Fixed | w_i is exactly 0 past a_max_j; if none carries weight the round is empty | `mission/aggregation_rules.py:update_weights`, `merge_on_mule`; `mission/host_mission.py:close_round` | `agg:plain` (n-weighted mean of models), `agg:cutoff` (every Exp 5 stack cell by default), `+fedprox`, `agg:fedbuff`, `agg:asynchfl`, `agg:fedex` (D4 faithful) |
| M02 | Age cutoff a_max_j | L3 ← L2 | Φ_j × S3c scale, merge period T | ⌊Φ_j / T⌋ rounds (snapshot at planning) | Fixed | the smaller of this and a fixed `a_max`; an infinite window cuts nothing | `mule/mule_main.py:_age_caps`; `mission/aggregation_rules.py:age_cap`; `stages/s3_deadline.py:effective_window` | `round` and `pref` laws (F-round cuts by the round; F-pref none) |
| M03 | Merged set and age anchor | L3 → L2 | the merge's contributors | `last_merged_round` for each; the plan closes with its violations by cause | Fixed | a CLEAN update the cutoff excluded does not reset the age | `fl_scheduler.py:record_merged`, `close_plan`; `mule/mule_main.py:_merged_device_ids`; `stages/s3d_age_cap.py:close_commit` | plan arms close a plan; others record only |
| M04 | Dock after an empty Pass 1 | L2/L3, mule | whether anything merged; `dock_on_empty` | dock with an empty partial, or skip the dock | Fixed (a flag) | an empty partial counts toward the quorum and merges nothing | `mule/mule_main.py:_run_ferry_mission` (empty path), `_dock_empty` | off for one mule (every recorded run); on when the quorum exceeds 1 |
| K01 | Backhaul carrier (band) U(c, t) and upload charge | L1, mule → cluster | SNR per carrier at the upload time, current carrier, switch cost λ, use cost κ(c) | the carrier; upload time 8 · bytes / rate; loss probability | Fixed (adaptive: argmax of U; fixed: argmax g_c) | below the floor the upload is a capped, lost charge with p_loss = 1 | `l1/channel_utility.py:AdaptiveChannelController.select`; `l1/channel_model.py:select_carrier`; `mule/ferry.py:charge_upload` | adaptive: H3, H1+L1, F+L1; fixed: F, FX, H1, the rest |
| K02 | Cluster merge and server mixing rate η | L3, cluster | partials, ages s_m, masses M_m, θ, η | θ ← θ + η · Σ M_m s_m Δ_m / Σ_live M_m | Fixed | a partial is live only if s_m > 0 and non-empty; none live: no step, the round stays open (`expired`) | `cluster/host_cluster.py:_aggregate_age_aware`; `cluster/cross_mule_fedavg.py:apply_weighted_deltas`; `mission/aggregation_rules.py:partial_staleness` | `agg:plain` overwrites θ with the n-weighted mean; `agg:fedex` θ + η Σ Δ / N; `agg:fedbuff` applies after K updates |
| K03 | Round closure by quorum (`min_participation`) | L3, cluster | partials received, the quorum | merge now, or wait | Fixed | `agg:plain` with several mules needs a full quorum; FedBuff's K replaces it | `cluster/host_cluster.py:aggregate_pending` | **1 in every Exp 5 cell** (the launcher passes only `--n-mules`; the driver default is 1) |
| K04 | Round close and what the DOWN carries | L3/L2, cluster | the closing round | version + 1; a DOWN with the slice, θ, a synthetic batch, positions, delivery_priority, spectrum signature | Fixed | on the simulated clock a deadline override is refused | `cluster/host_cluster.py:close_cluster_round`, `dispatch_down_bundle` | none |
| K05 | Deadline-law update, slow phase (the amendment fold at the dock) | L2/L3 | amendment overrides, registry deltas | an override timestamp or a patched Φ (none is issued); position, delivery_priority and spectrum signature (these are carried) | Fixed | **inert for Φ**: the cluster issues no overrides and patches no Φ; overrides are refused on the simulated clock | `stages/s3_deadline.py:fold_cluster_amendment` | none |
| K06 | Delivery priority after Pass 2 | L2/L3, cluster | each delivery outcome | `delivery_priority` + 1 on a miss, reset to 0 on delivery; reaches the mule through K05's fold | Fixed | read by S3a's anchor key only | `cluster/device_registry.py:update_after_delivery` | none |
| K07 | Pass-2 order and budget | L2, mule | the whole slice (not S1's set), pose, b̄, budget | nearest-first stops; with a budget, a skip-don't-stop fold | Heuristic | skipped devices keep an older basis; the in-flight check runs only under `replan` with a budget | `fl_scheduler.py:build_pass_2_queue`; `stages/s3a_cluster.py:order_pass_2_greedy`; `mule/mule_main.py` (Pass 2 block) | `pass_2_budget` off in Exp 5's stack cells (plan mode refuses it); on for 5.1's budgeted cells on H1's route |
| K08 | Mission slicing across mules | L2, cluster | device positions, K mules | disjoint slices | Heuristic (static) | slices differ in size by at most one | `experiments/exp4/topology_builder.py:angular_slices`; `cluster/device_registry.py:rebalance` | angular sectors; D4's CARP assignment overrides them |
| K09 | Dock server choice | L2, mule | reachable servers, pose, energy, RF prior | the server to dock at | **Learned** (argmax of the selector's DDQN) | none beyond the selector's guards, which `select_server` bypasses | `scheduler/selector/target_selector_rl.py:select_server` | **not wired**: only tests call it; every run has one cluster |

---

## 5. Notes by row

**P01, F01 — the readiness gates.** The Scheduler Freeze decided on 13 Aug (D3) to *remove* S2A/S2B from the contribution claims, not wire them. The code agrees: `FLScheduler.ingest_ready_adv` (0.60 threshold, 5 s freshness) is called only from `tests/unit/test_fl_scheduler.py`. What runs is an inline copy in `host_mission.py`, whose `min_utility` is 0.0 and whose caller never passes another value (no `min_utility` keyword is passed anywhere in `hermes/` or `experiments/`). The device-side `fl_threshold` is also 0.0. So, in every Exp 5 trial, a device is refused only if its state is not `FL_OPEN`.

**P02, F11, K05 — the deadline law, three phases.** Φ is set at 60 s × `deadline_time_scale` for a new device (`DEFAULT_FULFILMENT_WINDOW_S`), moved after each contact by the fast phase (F11), and could be overwritten by the slow phase (K05). The slow phase does nothing to Φ in practice: `dispatch_down_bundle` patches only position, delivery_priority and spectrum signature, and `close_cluster_round` is called with no overrides. Φ's whole adaptation is therefore the fast phase. F, FX and the F-ablations fly the multiplicative law; H1 and every D arm fly the additive law (`launch.py`, `is_ferry_arm`).

**P03 — what the tiers do now.** `classify_bucket` never reads a deadline or a utility (SEC26 audit §B; confirmed in `s3_deadline.py`). In plan mode the plan key ignores the tier, so tier matters only inside S3a's anchor choice. For H1 it still decides the queue.

**P06, P07, P08 — the one search.** `plan_search` runs one search per class and keeps the class whose best candidate has the smallest `plan_key`. Only `exact` (≤ 6 devices) is optimal over the member subsets V prices; above that the two families are heuristics (build plan, Phase 4 deviations). All Exp 5 N = 6 cells are exact. Counts bound the search, never wall time, so a repeated trial plans the same.

**P08, P09 — V is declared, not derived.** V = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T), with c₂ = κ·N_demand, c₃ = c₂, κ = 1, c₄ = 0.1. The repo holds no derivation from the theory track; Δ² is a convex surrogate for staleness, not a bound. By default the plan key puts the served weight share before V (resolution R11), because V alone let the empty plan beat a device that fits.

**P10 — one predicate, many callers.** S3b, the departure check, the re-plan, FX's `fits`, the pair mask and Pass 2 all fold the same `FeasibilityModel.admit`; the baselines differ in the *rule* they hold it to, not in the arithmetic.

**F03, F04, F05 — the in-flight slot.** One code path serves every filling (`pair_slot.py`): same pass guard, scope guard, mask and fallback. FX's pair overruns the budget at about 19 of 71 last-stop arrivals at N = 6, so `mask_empty` is common. The slot never acts at takeoff or in Pass 2.

**F08 — beacon insertion.** Built and tested, but nothing in `hermes/`, `experiments/` or `scripts/` calls `offer_contact`. S1's beacon clause and the BEACON tier are likewise unexercised (Freeze D6).

**M01, K02 — the merge.** With every basis current, value uniform and η = 1, the age-aware merge equals the plain mean (the regression test). The age weights are FedAsync's hinge with hand-set constants (a = 1, b = 0); the "derived from the bound" form is an open decision (Readiness, "Decisions still open").

**K03 — closure.** Exp 5 flies asynchronous merges everywhere: `min_participation = 1`. `agg:plain` with several mules would need a full quorum and is refused otherwise (`driver.py`).

**K07 — Pass 2.** Pass 2 clusters the entire slice at b̄'s radius, not S1's eligible set, so every device is offered the new θ. The plan prices Pass 2 once per class as T_nom does. An unbudgeted Pass 2 (every Exp 5 stack cell) is flown as ordered, with no in-flight check.

**K08 — slicing.** The process runtime slices once, at seeding: the registry's round-robin is overridden by the topology builder's explicit assignment, and `rebalance_for` has a caller only in the demo `cluster/__main__.py`.

**K09 — dock server choice.** Defined in the design as the selector's second use; never wired. With one cluster the choice is empty. It is the only learned decision in the register with no way to run.

---

## 6. Every learned component, and whether anything runs it

| Component | Where | What it decides | Runs in Exp 5? | Status |
|---|---|---|---|---|
| **FQ**, the pair score (masked pointer double DQN, 36 → 64 → 64 → 1, tanh, numpy) | `selector/pair_q.py`, `pair_features.py`, `pair_replay.py`; `policies/pair_slot.py` | F03 and F04 (band and next stop) | **Only as a stack check in batch 2** (FQ-g75, FQ-g0). Not part of FeRRy | Study 5.5: flat against FX; matched FX (−0.0774 against −0.0786) and trails `greedy_1` (−0.0716). `keep_learned = false` |
| **E3**, DQN after Chen et al. | `policies/chen_dqn.py`, `next_stop.py` | F04 (next stop, no plan) | Yes: 5.3 and 5.6 in batch 2, from `g0.99_s0.npz` | a competitor, scored on update yield and round closure, never on bytes |
| **H2**, intra-bucket DDQN (11→16→1) | `selector/target_selector_rl.py`, `ddqn.py` | P15 (order inside a bucket) | **No.** H2 and H3 left Exp 5 (Phase 5 decision 8) | every committed H2 run used random-init weights; H2 tied H1 |
| **ChannelDDQN** (8→16→3) | `l1/channel_ddqn.py` | a contact band at a stop | **No.** No process passes a `channel_actor`; the parameter exists in `MuleSupervisor` only | never trained; retired from the plan, code and logging kept |
| `select_server` | `target_selector_rl.py` | K09 | **No.** No caller outside tests | defined, unwired |
| the prototype's DQN | `hermes_rl/` (a separate repository, untracked here) | a joint (waypoint, base station, channel) action | **No.** Not imported by `hermes/` | the lineage, not a runtime part |

What bounds any of them: S1, the S3b predicate and the age cap run before the slot is called; `assert_pairs_admitted` checks every pair against the plan's admitted set; a pass guard refuses a selector call in Pass 2; and `PairQSlot` raises if it picks a pair the mask refused.

---

### 6.1 What the records show about the non-learned decisions

Which of the searched and rule-based decisions has been tested against an alternative, as of 7 Oct 2026. "Not run" means the study is built and waiting for batch 2.

| Decision | Alternative tested | Status | Result |
|---|---|---|---|
| P06 band class b̄ | FB+ pinned to each class; O1 oracle for the gap | Study 5.4: **not run** | none |
| P07 route search, member subsets | whole stops (`whole`), forced local search (`local`) | 5.14, batch 1 | at N = 6 the knee cell shows no difference; at stress `whole` differs most (0.897 against 0.810 network AoU), not a claim (Holm p = 0.078) |
| P05 hover stops | `hoveroff` | 5.14, batch 1 | no difference at N = 6 |
| P11 age cap S\* | S\* + 1, off (F-cap) | 5.14, batch 1; 5.8 not run | no difference at N = 6 (identical to F's 0.715 at the knee and 0.810 at stress); why is not established here |
| P08 coverage rank | `weighted` (`wcov`) | 5.14 | 0.002 at the knee, not a claim |
| P08 the score's terms | FX-dwell, FX-cov | Study 5.7: **not run** | none |
| P02 deadline law | additive (F-add), round (F-round), pref (F-pref) | Study 5.2: **not run** | none |
| F03 / F04 in-flight filling | FQ (learned) against FX (fixed) | Study 5.5: **done** | flat; FX stays |
| F03 / F04 committed against FX | F against FX | 5.3, 5.9, 5.11 (b) | mixed: FX ahead at the N = 6 knee, at N = 12 and at N = 12 with three mules, F ahead at N = 24; the only claim is FX at N = 12, K = 3 |
| F07 trim against reorder | not ablated | by design: reorder is refused in plan mode | none |
| M01 merge rule | `agg:plain`, `fedbuff`, `asynchfl`, `+fedprox` | Study 5.1: **not run** | none |
| K01 adaptive backhaul | F+L1 against F on the seconds backhaul | 5.14, batch 1 | **a claim**: network AoU 0.689 against 0.883 (knee) and 0.786 against 0.970 (stress) |
| K02 server rate η | none | not swept | η = 1 throughout |
| K03 quorum | none | not varied | `min_participation` = 1 throughout |

## 7. Constants that act as decisions

Hand-set numbers that steer a row above. "Chosen by": **hand** = set by a person in the code or a spec; **swept** = tried at more than one value; **pilot** = measured by a pilot campaign stage; **rule** = computed from a pilot by a rule fixed before it ran; **decided** = a recorded decision.

### 7.1 Plan and search

| Constant | Value | Steers | Where set | Chosen by |
|---|---|---|---|---|
| c₁, `c_time` | 1 | P08 | `plan/types.py:PlanScoreParams` (`--plan-score-params`) | hand (FedEx's form); not swept |
| κ, `c_cov_per_device` | 1; c₂ = κ·N_demand, c₃ = c₂ | P08 | same | hand, decision 2 (30 Sep); the pilot sweeps {0.15, 0.25, 1} |
| c₄, `c_energy` | 0.1 | P08 | same | hand; sweeps {0, 0.1} |
| `coverage_rank` | `lexicographic` | P08 | same | decided (R11, 30 Sep); `weighted` flown in 5.14 |
| `coverage_weights` | `age` (× (1 + miss streak)) | P09 | same | decided (decision 3, 30 Sep) |
| `exact_max_devices`, `exhaustive_max_stops` | 6, 6 | P07 | `plan/types.py:PlanSearchParams` | hand (the plan's threshold) |
| `heuristic_max_passes`, `heuristic_max_evaluations` | 50, 2,000 | P07 | same | hand; counts so a repeated trial plans the same |
| `hover_stops` | on | P05 | same | decided (30 Sep, after the final check) |
| `member_admission` | `subset` for the F family | P07, P10 | `processes/config.py:MuleConfig` | decided (decision 4 (b); not the recommendation) |
| `band_class_policy` | `search` (F, FX, FQ) or `fixed:c` | P06 | `MuleConfig` | decided per arm |
| T_nom (the plan's T, the deadline unit) | median over 20 reference layouts, priced on wide | P08, P02, M02 | `fl_scheduler.py:nominal_mission_period_s`; the driver | rule; priced on wide for every arm (documented, not corrected) |

### 7.2 Deadline, age cap and budget

| Constant | Value | Steers | Where set | Chosen by |
|---|---|---|---|---|
| Φ₀ | 60 s × `deadline_time_scale` | P02 | `types/scheduler.py:DEFAULT_FULFILMENT_WINDOW_S` | hand; six missions at the Q1 scale |
| `deadline_time_scale` | T_nom / 10 s | P02, F11 | `--deadline-time-scale t_nom` (`launch.py`) | decided (29 Sep, spec Q1) |
| β_on, β_partial, β_timeout | 0.8, 1.25, 1.5 | F11 | `stages/s3_deadline.py:DeadlineLaw` | hand; break-even on-time rate p\* ≈ 0.65 |
| Φ_min, Φ_max | 5 s, 300 s (× time scale) | F11 | same | hand |
| additive law | −5 s on CLEAN, +10 s on a miss, floor 5 s | F11 | `s3_deadline.py` module constants | hand (original); no ceiling |
| law per arm | multiplicative for the F family, additive for H and D | P02 | `params.toml [campaign] ferry_arm_deadline_law` | decided 5 Oct |
| NEW probation | 3 failed attempts | P03 | `NEW_BUCKET_ATTEMPT_LIMIT` | hand |
| beacon window | 30 s | P01 | `FLScheduler(beacon_window_s)` | hand; unexercised |
| S3c window, target, gain, max | 5 missions, 0.8, 2.0, 4.0 | P14 | `MuleConfig.mission_window_*` | hand; off |
| S\* (the age cap S) | 2 at N = 6, 12, 18, 24; lookahead L = 0 | P11 | `params.toml [pilot_outputs] s_star` | rule: smallest S covering 90 % of 30 layouts at the knee and stress budgets, never below 2 (5 Oct) |
| mission budget (knee / stress) | 150 / 75, 180 / 90, 240 / 120, 262 / 130 s at N = 6, 12, 18, 24; 120 / 60 s at N = 6 with the measured payload | P10 | `params.toml [pilot_outputs]` | rule: the smallest budget reaching 95 % of the largest mean update yield; stress = half, to 5 s. N = 6's knee is the grid's edge, accepted as is. Measured on the first host |
| session timeout | 36, 34, 34, 23 s at N = 6, 12, 18, 24 (code default 5 s) | F10 | `params.toml [pilot_outputs] session_ttl_s` | rule: 2 × p95 fit time, rounded up. First host |
| busy-flag TTL | 45 s | F10 | `HFLHostMission(busy_ttl_s)` | hand |
| τ | 0.71 primary, 0.82 second | scoring | `params.toml [pilot_outputs] tau` | rule: largest τ in 0.01 steps that 80 % of H1 trials reach at the knee; the smallest over N |

### 7.3 Merge and cluster

| Constant | Value | Steers | Where set | Chosen by |
|---|---|---|---|---|
| hinge a, b | 1.0, 0.0 | M01 | `aggregation_rules.py:AggregationSpec` | hand (FedAsync's form); sensitivity check is an open decision |
| value proxy v_i | `uniform` | M01 | same | hand |
| merge period T | T_nom (`--agg-period-t-nom`) | M02 | `params.toml [campaign] merge_period_t_nom` | decided 5 Oct |
| server rate η | 1.0 | K02 | `AggregationSpec.server_lr` | hand |
| `agg:asynchfl` q | 0.5 (polynomial (a + 1)^−q) | M01, K02 | same | hand (FedAsync's; Async-HFL does not report q); form changed 7 Oct |
| FedProx ρ | 0.01 | F09 | `params.toml [s51] fedprox_rho` | decided 5 Oct |
| FedBuff K | the opening mule's slice size | K02 | `buffer_k` or the slice | hand |
| `min_participation` | 1 | K03 | driver default | hand |
| `pass_2_budget` | off in plan mode | K07 | `MuleConfig` | hand |
| `in_flight_response`, `replan_fallback` | `replan`, `trim` | F06, F07 | `launch.py:COMMON_FLAGS` | decided 29 Sep (pilot plan) |
| mission slicing | angular sectors | K08 | `topology_builder.py:angular_slices` | hand |
| device utility weights | w₁ = 0.7, w₂ = 0.3; performance 0.5 / 0.3 / 0.2 | F01 | `mission/utility.py`, `client_mission.py` | hand (design) |
| S2 constants | threshold 0.60, freshness 5 s (unused); `min_utility` 0.0 (live) | F01 | `stages/s2b_flag.py`; `host_mission.py` | hand; see note P01 |

### 7.4 Physics the planner prices

| Constant | Value | Steers | Where set | Chosen by |
|---|---|---|---|---|
| band classes | wide 20 MHz (100 RB), medium 5 MHz (25), narrow 1.4 MHz (6); one carrier at 3.32 GHz | P06, F02 | `l1/contact_link.py` | decided 29 Sep (D1) |
| wide-class planar range | `rf_range_m` = 60 m; medium about 119 m, narrow about 232 m (n = 2.2) | P04, P06 | `MuleConfig.rf_range_m`, `contact_link.py` | hand; an assumed anchor, no measured AERPAW range |
| path-loss exponent, shadowing σ | 2.2, 4 dB (correlation 7.4 s) | F02, P08 | `MuleConfig.n_pl`, `shadow_sigma_db`, `shadow_corr_s` | hand, cited (3GPP TR 36.777); swept in 5.15 |
| edge availability, SNR floor, altitude | 0.9, −6.7 dB, 25 m | F02 | `margin_quantile`, `snr_floor_db`, `altitude_m` | hand, cited |
| interference (jittery) | amplitude 5 dB, period P_c 60 s | F03, F04 | `contact_regime`, `interference_period_s` | decided 5 Oct (whole campaign flies jittery) |
| flight and energy | cruise 5 m/s, turnaround 30 s, listen 1 s, 143.6 W flying, 168.5 W hovering, no capacity | P10 | `MuleConfig`, `l1/mission_clock.py:EnergyModel` | hand (Zeng–Xu–Zhang 2019 powers); labelled simulated |
| payload | 1 MB per direction (18.8 KB measured) | P06, P10 | `--payload-bytes` | decided 29 Sep (D3); 1 MB so dwell is visible |
| backhaul U(c, t) | switch cost λ = 0.5; use cost κ(c) = 0 | K01 | `l1/channel_utility.py` | hand; no physical derivation (EX-4 record §8.3) |

### 7.5 Learner and campaign

| Constant | Value | Steers | Where set | Chosen by |
|---|---|---|---|---|
| FQ network and update | 64 × 64 tanh; Adam lr 1e-3; Huber δ 1; clip 10; target sync 500; batch 64; replay 50,000; warm-up 1,000 | F03, F04 | `selector/pair_q.py:PairQConfig`, `LearnerSettings` | hand (design D-C); not tuned |
| FQ exploration ε (training) | 500 episodes around FX's pair at 0.3, then 0.3 → 0.05 over the first half | training only | `pair_q.py:BehaviourSchedule` | hand; the calibration findings name it the likeliest cause of "copying FX" |
| FQ episodes, validation, patience | up to 10,000; every 1,000; stop after 3 without a best | training only | `experiments/ferrysim/train.py` | hand |
| FQ discount γ | {0, 0.25, 0.5, 0.75, 0.9, 0.99} × 10 seeds | F03, F04 | `params.toml [rl]` | pre-registered sweep; γ\* = 0.75 recorded only |
| equivalence margin ε (verdict) | max(0.01, 0.1 × headroom) = 0.01 in every cell | the verdict | `experiments/ferrysim/report.py` | rule: from the headroom report |
| FQ reward | r_k = G_k − c_t·Δt_k/T, c_t = 0.1; last decision − c_cov·U, c_cov = 1 | training | `experiments/ferrysim/reward.py` | decided (Phase 5 decision 4); the 3 × 3 grid left with the learned scores |
| E3 | γ 0.99, lr 5e-4, ε from 1.0, 5 seeds | F04 | `params.toml [rl.e3]` | decided 5 Oct (Chen's published code) |
| campaign | base seed 20261005; 4 missions; jittery; α = 0.05; 2,000 bootstraps; Holm per study | scoring | `params.toml [campaign]`, `[score]` | decided |

---

## 8. What this pass found

Found while building the register; each was checked against code.

- **S2A/S2B are specified but do not run** (P01, F01). Documented already in Freeze D3 and the SEC26 audit; restated here because the design documents' "Four-stage gated scheduler" still lists them.
- **The slow-phase deadline update is inert** (K05). The two-phase deadline mechanism of principle 4 therefore has one working phase.
- **`select_server`, `offer_contact`, `ChannelDDQN` as `channel_actor`, and `rebalance_for` (outside the demo) have no runtime caller.** Four decisions the design describes are not exercised in any recorded or Exp 5 run.
- **The bucket tiers do nothing in plan mode** except in S3a's anchor key (P03).
- **`min_participation` is 1 everywhere in Exp 5** (K03), so the cluster's quorum decision is trivially "merge now"; only FedBuff (5.1, batch 2) changes when θ moves.

## 9. Sources

Code as read on 7 Oct 2026 (HEAD `c5b1b88f`): the modules in each row. Records: [HERMES_Scheduler_Freeze.md](../HERMES_Scheduler_Freeze.md) (§1, §2 D3 and D6, §5e–5n); [FeRRy_Build_Plan.html](../FeRRy_Build_Plan.html) (module map, phases 3–5, studies); [Experiment_5_Readiness.md](../Experiment_5_Readiness.md); [Experiment_5_RL_Calibration_Findings.md](../Experiment_5_RL_Calibration_Findings.md); [SEC26_Code_Audit.md](../SEC26_Code_Audit.md) (§B, §C); `scripts/exp5/params.toml` and `scripts/exp5/launch.py`; module docstrings in `stages/`, `plan/`, `policies/`, `mission/aggregation_rules.py`, `mule/mule_main.py`.

*Numbers are copied from the code defaults and the records as of 7 Oct 2026; nothing was re-run for this document except a repo-wide search for callers.*
