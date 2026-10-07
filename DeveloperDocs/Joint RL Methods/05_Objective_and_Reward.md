# One objective — the plan score V, the flight reward, and what "one" means in the code

*7 Oct 2026. A reading document in the Joint RL Methods series: it restates the code and the records and decides nothing. It is the detail behind J4 ("one objective across layers") in [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) §4, and it leans on [04_Age_Staleness_Weight.md](04_Age_Staleness_Weight.md) for the merge weight and the ages. Code references are file:line as of commit `39b20b84`; numbers are copied from the named record, and the only things executed for this document are the test runs named in §9. Anything I could not check is marked "unverified".*

---

## 0. The short version

Contribution C2 says the plan score, the flight reward and the merge weight are one derived objective, so L2's decisions and L3's aggregation optimise the same thing. What the code actually does:

| Place | Quantity | Built from | Hand-set or derived |
|---|---|---|---|
| **Plan score V** (L2, plan clock) | −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T) | mission time, a coverage shortfall U weighted by **age × (1 + miss streak)**, expected link loss, simulated energy | **Hand-set** constants, in FedEx's form; "declared, not derived" |
| **Flight reward r_k** (L2, trained from L3) | G_k − c_t·Δt_k/T (− c_cov·U at the sortie's last decision) | G_k is the **merge weight** n·v·s(age) of what was collected at stop k, over n_ref·N; U uses V's coverage weights | **Hand-set** c_t = 0.1, c_cov = 1; form chosen, not derived |
| **Merge** (L3) | w ∝ n·v·s(age), zero past a_max | [04](04_Age_Staleness_Weight.md) | **Hand-set** hinge (a_h = 1, b = 0), q = 0.5 |

So the coupling is **by construction, not by derivation**: the reward reuses the merge's weight, and V's coverage term reuses the plan's age. There is no derivation of V's constants or of the reward's form from FedEx-Async's bound, and a HERMES device trains once per visit, so that bound does not carry over directly (§6). Two further points, both from reading the code and flagged as my reading in the text:

- the merge weight is a *discount that falls with age*; V's coverage weight is a *claim that rises with age*, and they use different ages (§4);
- in every cell flown or trained so far the merge weight collapses to a device count, so the reward's "merge weight" is nominally L3's and numerically just "devices collected over N" (§3.3).

**Where it stands.** The plan score, the reward and Study 5.7's arms are built and unit-tested. Study 5.5's null (6 Oct) removed the learned scores, so **5.7 now flies FX, FX-dwell, FX-cov and D4**, built 7 Oct and **not run**; the reward grid and F·hand left with the learned scores. Study 5.1, which decides whether the merge weights themselves can be defended, has not run either.

---

## 1. Decisions and quantities

| Name | Defined where | Inputs | Output | Hand-set or derived | Code path |
|---|---|---|---|---|---|
| Plan score V | `plan_score.score` | weights, outages, pass times, dwell, energy, T_nom, P_hover | V and its terms (`ScoreTerms`) | form declared; constants hand-set | `plan_score.py:500-616` |
| c₁ (`c_time`) | `PlanScoreParams` | — | 1.0 | hand-set | `plan/types.py:385` |
| κ (`c_cov_per_device`), c₂ = κ·N_demand | `PlanScoreParams.constants` | N_demand | 1.0; c₂ = N_demand | hand-set | `types.py:386`, `:403-407` |
| c₃ (`c_link`) | same | — | None, meaning c₃ = c₂ | hand-set | `types.py:387` |
| c₄ (`c_energy`) | same | — | 0.1 | hand-set | `types.py:388` |
| Δ | `plan_score.score` | Pass 1 + turnaround + Pass 2 (on b̄) | seconds, less dwell under F-dwell | derived from the physics | `plan_score.py:597-601` |
| U, L | `score` | weights, served set, p_out | shortfall; expected link loss | derived | `:604-610` |
| p_out, σ_eff | `mean_snr_outage`, `sigma_eff_db` | mean SNR, floor, shadowing, interference | Φ((floor − mean)/σ_eff) | derived (moment match) | `:243-284` |
| E | mission clock ledger | flight, dwell, listen time | simulated joules (Zeng-Xu-Zhang) | derived from a declared model | `hermes/l1/mission_clock.py:315-380` |
| Coverage weight w_j | `coverage_weight` | plan age, miss streak | max(a_j,1)·(1 + m_j) | hand-set form | `plan_score.py:334-366` |
| Plan key | `plan_key` | cap key, served share, V | ordering of candidates | hand-set rule (R11) | `:659-682` |
| Reward r_k | `sortie_rewards` | `SortieRecord`, `RewardSpec` | one `RewardTerms` per Pass-1 stop | form chosen (decision 4) | `reward.py:491-528` |
| c_t, c_cov, c_e | `RewardSpec` | — | 0.1, 1.0, 0 | hand-set; grid {0.03, 0.1, 0.3} × {0.25, 1, 4} defined | `reward.py:151-156`, `:89-90` |
| G_k | `_credits` | collected updates' raw merge weights w_i | Σ w_i / (n_ref·N) | derived from the merge | `reward.py:445-458`, `:523` |
| Expected availability | `RewardSpec.expected_availability` | availability rel_j | credit rel_j × weight for a targeted member | derived | `reward.py:445-458`, `train.py:170` |
| F·hand | `HAND` | collected count, dt, metres | (200·\|C_k\| − dt − 0.002·m)/150 | hand-set (ContactSim's own) | `reward.py:95-97`, `:505-510` |
| Bytes | `BYTES` | collected count | \|C_k\|/N | hand-set (E3's) | `reward.py:502-504` |

---

## 2. The plan score V

```
V(b̄, π | demand) = −[ c₁(Δ/T)² + c₂·U + c₃·L ] − c₄·E / (P_hover·T)
U     = 1 − Σ_served w / Σ_demand w
L     = Σ_served w·p_out / Σ_demand w
p_out = Φ((floor − SNR_b̄(d)) / σ_eff),    σ_eff = √(σ_sh² + σ_I² + A²/2)
```

Verified against `plan_score.py:11-14` (docstring) and `:597-612` (the code). V ≤ 0; the empty plan scores −c₁(t_turn/T)² − c₂ (Configuration Reference §18.3).

### 2.1 The terms

- **Δ is the whole mission on b̄**: Pass 1 (transit, dwell at the class's predicted rate, the return leg, the upload), the dock turnaround, and Pass 2, which flies b̄ too (decision 2 (b)). The band sets Pass 2's length: at 1 MB and N = 6 its median is 94 s on wide, 62 s on medium and 45 s on narrow (critic C2). A plan that serves nobody flies neither pass (a mission that collects nothing skips Pass 2) and pays only the turnaround (`_mission_s`, `:473`). T is the cell's nominal mission period T_nom, which plan mode requires. Under **F-dwell** (`dwell_in_delta = False`) both passes' dwell leaves Δ in the score only: the predicate still prices it and E still counts the hovering (`:599-600`).
- **U, the coverage shortfall,** is weighted by the coverage weights over the demand (devices left after S1 and S3). **L, the expected link loss,** is the chance a served member's SNR falls below the floor around the *mean* SNR the planner prices (δ_obs = 0): pricing the seeded phase would be an oracle. At a class's edge R(b) the mean SNR sits Φ⁻¹(0.9)·σ_sh above the floor, so with shadowing alone the outage there is 0.1 and the interference raises it. The Gaussian moment match is exact for shadowing and noise, not for the interference sine: against the exact outage it agrees to 4 decimals on the clean channel and within 0.008 on the jittery one (0.1776 against 0.1849 at a class's edge; Configuration Reference §18.3). A member beyond R_planar(b̄) has outage 1.
- **E is the simulated energy of both passes,** return legs included, over P_hover·T, so it reads as a share of a mission spent hovering. The model is Zeng, Xu and Zhang (2019), eq. (6), Table I set: **143.6 W flying at 5 m/s, 168.5 W hovering** (`mission_clock.py:345-347`, `:379-380`), declared **simulated**, with the capacity clause off. It is nearly collinear with Δ, which is why c₄ is small.
- The outage formula has two definitions, the runtime's (`FerryRuntime.outage_probability`, which the planner calls) and the score's (`outage_by_distance`); the U2 tests tie them within 1e-15.

### 2.2 The constants, as set and why

| Constant | Value | Rationale in the record |
|---|---|---|
| c₁ | 1 | spec, other choices 7 |
| c₂ = κ·N_demand, κ | 1 | one average device is worth κ full T² of time, "serve every device the budget allows; time breaks ties" (decision 2) |
| c₃ | = c₂ | so c₂U + c₃L = c₂(1 − the expected weighted served share) |
| c₄ | 0.1 | E is nearly collinear with Δ, so energy breaks ties (design D-B) |

The pilot's sweep is defined as `PILOT_KAPPAS = (0.15, 0.25, 1.0)` and `PILOT_C_ENERGIES = (0.0, 0.1)` (`plan_score.py:170-171`), under the `weighted` rank. Lower κ was probed and left out: with Δ over the whole mission, 2 of 30 plans already fly empty at κ = 0.15 (30 s, 1 MB), and 1 to 9 of 30 at κ = 0.1, which is why the critic's 0.1-0.25 range was not taken. **I found no stage in `scripts/exp5` that flies a κ or c₄ sweep and no scored output of one**, so "hand-set and swept" in the code's docstring is, as far as I can verify, "hand-set, with a sweep defined"; unverified whether an earlier Phase 4 pilot ran it.

When c₃ ≤ c₂, serving one more device at the same Δ and E never lowers V (it gains (w_j/Σw)(c₂ − c₃·p_out,j)); with c₂ = c₃ = 0 (F-cov) V does not see coverage at all (`:37-45`).

### 2.3 The coverage weights

`age` mode (the default): max(a_j, 1)·(1 + m_j) with the arm's `miss_priority` on, max(a_j, 1) alone with it off (F-prio); `uniform`: 1·(1 + m_j) or 1, the plan's letter 1 − served/N. a_j is the *plan age* (own mule's missions since last merged; [04](04_Age_Staleness_Weight.md) §2.2). Every device a plan leaves out is widened as a miss and a clean contact clears the streak, so m_j = a_j − 1 and **F's weight is about a_j²**: F's objective is declared quadratic in age (critic A5). The floor at 1 keeps every demanded device in U (a zero weight would remove it). Arm F-pref multiplies each weight by Oort's speed factor (`oort_speed_factors`, α = 2.0).

### 2.4 The rank, and why V alone is not what chooses (resolution R11)

V alone does not keep the "serve everyone the budget allows" promise: every plan that serves anyone pays a whole Pass 2, and the empty plan pays none, so V can prefer flying empty to serving a device that fits. The probe (u and v 60 m either side of the dock, z 400 m out, FB+wide, 1 MB, a 37.6 s budget, T = 200 s, cap off): serving v scores −3.85 (a predicted 264 s mission), the empty plan −3.02, and the plan flew empty mission after mission. So the default rank is `lexicographic`: (cap key, −round(served share, 9), −round(V, 9), class index, stops), where the share Σ_served w / Σ_demand w is 1 − U. The `weighted` rank is V alone and is what the κ sweep flies; with κ = 0 (F-cov) `weighted` applies whatever the setting says (`applied_rank`, `:641`). The rank never changes V or its terms.

Consequences worth stating: under the default rank **time and energy only break ties** among plans serving the same weight, so a plan can fly a much longer mission to serve one more device; and the share counts a member at a class's edge fully although its jittery-channel outage there is about 0.15-0.2 (Configuration Reference §18.3).

### 2.5 The arms that switch a term of V on or off

Each ablation changes one setting of `PlanScoreParams` or of the arm, and nothing else in the score. This is the whole of the "one switch" design behind Studies 5.2, 5.7, 5.8 and 5.14.

| Arm | What changes | Setting | Study | Run? |
|---|---|---|---|---|
| F | nothing: age weights × (1 + miss streak), share-first rank, cap S, hover stops | defaults | all | batch 1 (cells shared with 5.3, 5.9, 5.14) |
| F-cov | coverage and link terms off; serves capped devices only (the empty plan otherwise scores best) | `c_cov_per_device = 0`, `c_link = 0`; `weighted` rank applies | 5.8 | no |
| F-dwell | both passes' dwell out of Δ in the score only | `dwell_in_delta = False` | 5.7 (as FQ-dwell) | no |
| F-prio | weight by age alone, no miss-streak factor | the arm's `miss_priority` off | 5.8 | no |
| F-cap | no age cap | `age_cap_missions = None` | 5.8, 5.14 (`capoff`) | `capoff` in 5.14 only |
| F-pref | Oort's speed factor on the weights, no per-device cutoff | `plan_speed_alpha = 2.0`, `pref` law | 5.2 | no |
| FX-dwell, FX-cov | F-dwell and F-cov with FX in the flight slot | as above, `flight_slot = cross_heuristic` | 5.7 | no |
| `wcov` (5.14) | V alone ranks instead of share-first | `coverage_rank = weighted` | 5.14 | batch 1, null at N = 6 |

Two limits worth keeping in mind when reading any of these. (i) With Δ and E held and c₃ ≤ c₂, V is monotone in coverage, so a plan never gets worse by serving one more device at the same time and energy; what V trades is coverage against the *extra* time and energy a device costs, at the exchange rate κ. (ii) At the measured 18.8 KB payload F serves every device every mission, so V's trade has nothing to decide there; the cap, the coverage term and member subsets bind at 1 MB and at the knee and stress budgets (build plan, Study 5.8 and 5.14 held-fixed notes).

---

## 3. The flight reward

Each Pass-1 arrival at stop k is a decision (the pair: the class k is served on and the stop flown next). Decision 4 (a):

```
r_k = G_k − c_t · Δt_k / T − c_e · ΔE_k / (P_hover·T)      (− c_cov · U   at the sortie's last decision)
G_k = Σ w_i over the updates collected CLEAN at k  /  (n_ref · N)
U   = Σ ω_j over committed devices left uncollected  /  Σ ω_j over the demand
```

(`reward.py:7-10`, `:491-528`.) w_i is the raw L3 weight the mule's merge gave the update, `update_weights`' n_i·v_i·s(a_i), 0 past the cutoff (`raw_merge_weights`, `:551-570`); n_ref is the cell's declared example count and N the mission's demand, so **one fresh device of standard size is worth 1/N**. Δt_k runs from arrival at k to the next Pass-1 arrival, or, after the last decision, to the end of the Pass-1 upload. T is T_nom. ω_j are **the plan's coverage weights** (`PlanCommit.weights`), not the merge weights; the committed set is the plan's `served`.

Defaults and what is absent: **c_t = 0.1, c_e = 0, c_cov = 1.** No energy term (energy tracks time) and no lateness term (none of 1,148 probed collections was late, critic B3). A sortie (one Pass 1, dock to dock) is the Q's horizon. A sortie that flies no Pass-1 stop makes no decision and adds nothing: its shortfall is reported (`undecided_shortfall`), never charged.

To see the scale: at N = 12 one collected device is worth 1/12 ≈ 0.083, and a stop that takes 0.2 T of flying and dwell costs 0.1 × 0.2 = 0.02. This is arithmetic from the formula, not a measured quantity.

### 3.1 Expected availability, the training credit

Whether a device answers is a keyed draw independent of the pair, with noise 10 to 100 times the time signal a decision moves (critic C1). Training therefore replaces each targeted member's draw by its probability rel_j, holding the realized flight: a member collected at k, or one whose uplink the draw dropped, is credited rel_j times its weight (its realized w_j when collected, n_ref when dropped, exact for an age-0 update), and U counts it uncollected with probability 1 − rel_j (`_credits`, `coverage_shortfall`, `:445-481`). The policy never reads rel_j. **Training uses it (`TRAINING_REWARD`, `train.py:170`); validation and every reported number use the realized draw** (`train.py:323-325`). Given the flight up to the arrival at k, the expected credit is the realized credit's mean; `test_the_expected_reward_is_the_realized_rewards_mean_over_the_draw_exactly` checks it. One declared departure from "the reward the merge actually pays": an update whose backhaul upload was lost is still credited (build plan, Phase 5 deviations).

### 3.2 The two older rewards, kept for comparison

- **F·hand**, "today's reward", ContactSim's own ported and declared as such (`reward.py:51-56`): r_k = (200·|C_k| − Δt_k − 0.002·m_k)/150, with m_k the metres flown over Δt_k and no terminal term. The constants are `COMPLETION_BONUS = 200` and `ENERGY_W = 0.002` per metre (`selector/sim_env.py:54-59`, `:600`) and `reward_scale = 1/150` (`selector_train.py:69`). The "energy" in the older reward is therefore distance, not joules. It was the selector's reward for the original ContactSim DDQN (build plan: "200·completed − time − energy, scaled by 1/150"); arm FQ-hand trained on it.
- **Bytes**, E3's reward, |C_k|/N: the payload is fixed, so bytes are updates (decision 7). E3 uses no deadline, plan, coverage term or age cap, and is scored on update yield and round closure, never raw bytes.

### 3.3 What G_k is in practice

At FerrySim's training cells (one mule, `agg:cutoff`, `value = uniform`, the equal device model, 10 examples per device) every collected update weighs n_ref, because the age is 0 (an unbudgeted Pass 2 delivers the current basis to every device, [04](04_Age_Staleness_Weight.md) §2.4) and v = 1. So **G_k is the stop's collected count over N** (`reward.py:17-20`, critic B4). The merge weight is wired in, and a test pins that on hand-worked missions the gains are the merge's weights (`test_t4_a_hand_worked_missions_gains_are_the_merges_weights`), but on the flown cells it carries no more information than a head count. It only differs where weights differ: skewed shard sizes (Study 5.13, "decision 4's reward stops counting devices", critic B4), the `loss` value proxy, or a budgeted Pass 2 that spreads ages, which plan arms refuse. The build plan lists "a 5.7 cell where the merge weight varies" as an open item.

---

## 4. How the three uses relate

| Aspect | Merge (L3) | V's coverage term (L2) | Reward gain G_k (L2) |
|---|---|---|---|
| Weight | n·v·s(a) | max(a_j, 1)·(1 + m_j) | merge weight, over n_ref·N |
| Age | **merge age** a_i, cluster rounds (basis version) | **plan age** a_j, own-mule missions since last merged | merge age (via the merge weight) |
| Direction in age | falls as age grows (a discount) | rises as age grows (a claim to be served) | falls with merge age |
| Includes n, v | yes | no | yes (through w_i) |
| Zero region | exactly 0 past a_max_j | never (floor 1) | 0 past the cutoff |
| Penalty for skipped devices | none | U, weighted by the same plan weights | c_cov·U with V's weights ω_j |

My reading, not a claim in the repo: V and the merge share the *idea* "a device that has gone long unserved matters", but not a single quantity. They are consistent in spirit (serve what is old; do not over-trust what is stale) and they point opposite ways in the same variable name. The only literal sharing is the reward's use of the merge's raw weight for G_k and V's weights for U, so the reward mixes the two. "One objective" is a design discipline here: every term in V and r_k has a counterpart in the merge, and the constants are tied to nothing by derivation. The paper-safe wording is "an objective assembled from the same ingredients", not "derived".

---

## 5. Study 5.7 — one objective

**Question** (build plan): do the plan's dwell and coverage terms, and a reward derived from the merge weight, change outcomes compared with hand-set choices?

**Original design.** F against F−dwell, F−cov and F·hand; D4 as the travel-only reference; the derived-against-hand-set weights from 5.1; a reward-weight grid reported as the front of round closure against energy and deadline misses; ~400 trials plus the grid in FerrySim; answering "the fixed-weights objection a reviewer raised against A2FeD (#28)".

**As built after Study 5.5's null (6 Oct 2026).** Study 5.5 kept FX (`replace_fx = false`), so the learned-score arms cannot be trained:

| Item | Status |
|---|---|
| **Arms flown** | FX, **FX-dwell**, **FX-cov**, D4 (`params.toml [s57]`) |
| FX-dwell | FX's slot with the plan score's dwell out of Δ (`dwell_in_delta = False`): `driver.py:254`, `_ARM_SCORE` `:259` |
| FX-cov | FX's slot with the coverage term off (`c_cov_per_device = 0`, `c_link = 0`, "cap-only service"): `driver.py:249` |
| D4 | the travel-only reference; FedEx's tour with FeRRy's merge by default |
| Cells | N = 12, stress and knee budgets, 40 trials per cell, 1 MB; primary metric **round closure** (`round_close_rate_kmin1`, higher is better) against FX; also time to τ, reached τ, Network AoU |
| Pre-registered ε | **none** (R28) |
| **Left with the learned scores** | F·hand (FQ-hand), FQ-dwell and FQ-cov, and the c_t × c_cov reward grid (`[rl.s57]`, `grid_c_t = [0.03, 0.1, 0.3]`, `grid_c_cov = [0.25, 1, 4]`; (0.1, 1) is 5.5's own). The grid is still defined in `reward.grid_specs` but nothing flies it |
| Built | 7 Oct 2026 (`aedae8b`); 10 tests in `test_exp5_fx_ablations.py` pass (run 7 Oct) |
| Status | **Not run** (batch 2) |

What this study can and cannot say once it runs. It can say whether turning off the dwell or the coverage term changes round closure at N = 12 with FX in the flight slot, and where D4's travel-only plan stands. It **cannot** say anything about the reward's constants, the derived-versus-hand-set reward, or whether any c_t × c_cov setting would have changed Study 5.5's ranking, because those arms were learned scores. The 5.5 verdict is conditional on the one reward it was trained and judged under.

---

## 6. Declared, not derived — and the FedEx caveat

- **Status in the code.** `plan_score.py:47-63`: "The repo holds no derivation from the theory track, so V is hand-set in FedEx's form with the convex-surrogate caveat." The build plan's theory track says: start from FedEx-Async Theorem 2 (eq. 24), whose error term is proportional to (1/N)·Σ_k R_k·Δ_k²; substitute Δ_k(b̄, π) = Σ travel + Σ dwell(b̄, t_j) + upload; add Zhai's coverage term for devices the deadline excludes; optionally Cui's sequence form for `agg:seq`; state the result on a convex surrogate and say so. Output: V's constants and the derivation of the L3 weight. The build plan's risk table expects it may land late: "V's constants are hand-set; sweep the constants and present V in FedEx's form with the convex-surrogate caveat". It did land late, and it has not landed.
- **The caveat to carry into the text.** FedEx's bound is for clients that train without pause between visits, so every update is one tour stale. A HERMES device trains **once per visit** (principle 14: sessions are exchange-only; offline training between visits). Δ² in V is therefore a *convex surrogate for staleness*, not a bound. With one mule the sum over K is Δ² itself, so the exponent matters only in the trade against coverage and energy. The extension (dwell, upload, coverage, band-dependent Pass 2) is the contribution's novelty and also the part with no proof behind it.
- **What FedEx itself has and lacks** (Related Work Notes §3.1): a fixed line-of-sight rate (every transfer takes the same time), no deadlines, no time budget, no band choice, no admission or skipping, no re-planning. Its merge is delta accumulation with no staleness weighting; FeRRy's merge weights are therefore also not FedEx's.
- **The open decision** (Readiness): drop the derivation and add a ~240-trial sensitivity check on the hand-set constants; drop it with nothing added; or do the derivation, after which it becomes a weight mode and a 5.1 variant. It is "needed before" batch 2's 5.1 and 5.7. Given the null that removed 5.7's reward arms, the sensitivity check is the only route by which the V constants (κ, c₄) or the reward constants would be examined at all.

---

## 7. Layer interfaces

| Edge | What crosses | Where |
|---|---|---|
| **L1 → V** | rate(b, SNR) → dwell; per-class mean SNR and channel spread → p_out; the Zeng energy ledger → E | `plan_score` primitives (floats and callables); `FerryRuntime.plan_classes` |
| **L1 → reward** | the realized clock: Δt_k, dwell and listen seconds, energy over the span; the keyed availability draw | `StopRecord.t_next_s`, `dwell_s`, `uplink_dropped` |
| **L3 → V** | plan ages → coverage weights; the cap → the plan key's first component | `demand_weights`, `s3d_age_cap.cap_key` |
| **L3 → reward** | each collected update's raw merge weight w_i, read off `device_weights × weight_mass` | `raw_merge_weights`, `reward.py:551` |
| **L2 → L3** | the committed plan decides which updates exist to be merged | [04](04_Age_Staleness_Weight.md) §11 |
| **L2 internal** | V chooses (b̄, π) at the dock; the reward trains the flight slot; the hard gates (S1, S3b predicate, age cap) run before either | Freeze principle 12; `selector/scope_guard.py` |

---

## 8. Failure modes

- **V alone flies empty** when every serving plan pays a whole Pass 2 (the R11 probe). The default rank prevents it; the `weighted` rank and F-cov reproduce it, by design (F-cov is "cap-only service").
- **κ too low** flies empty: 2 of 30 at κ = 0.15, 1 to 9 of 30 at 0.1 (30 s, 1 MB). The sweep would report them; it has not been flown (§2.2).
- **The planner prices the mean SNR** (δ_obs = 0) and dwell is convex in SNR, so a realized mission runs longer than predicted: +0.8 % on wide at 1 MB, +7.1 % on wide at 10 MB, +19.2 % on narrow at 1 MB (per member up to 1.39×; build plan, Phase 3 pilot notes). The prediction is recorded per arm as `mission_s` for comparison with the realized ledger.
- **T_nom is priced on wide for every arm**, so an arm with shorter missions fits more of them into a deadline window and "wins" on Δ/T without a better plan (critic C4, documented, not corrected). F's narrow missions last about 55 s against a T_nom of 172-250 s.
- **The share-first rank counts edge members fully** although they miss about 15-20 % of the time at a class's edge on the jittery channel.
- **Weight and reward both go degenerate under equal data** (§3.3): the merge weight reduces to a count, so the reward cannot reward anything the head count does not.
- **Reward noise.** Without the expected-availability credit the availability draw's noise is 10 to 100 times the signal; with it, held-out and reported numbers still use the realized draw, so training and reporting differ by construction.
- **The 5.5 verdict is conditional on one reward.** The sweep trained and judged under c_t = 0.1, c_cov = 1; whether another point of the grid would have lifted the learner above FX was not tested.
- **Dwell may be inert.** At the measured 18.8 KB payload dwell is a fraction of a second and the dwell term only bites on narrow classes; Study 5.7 flies 1 MB (D3 of the build plan). F-dwell and F-cov are also the two ablations whose arms were built last and have the least history.

---

## 9. Evidence

| Question | Evidence | Status |
|---|---|---|
| Is V's arithmetic as documented? | The code matches the docstring and Configuration Reference §18.3 line for line (`plan_score.py:597-612`). `test_p4_plan_score` among the 440 tests I ran on 7 Oct passes | Verified here |
| Is the reward's arithmetic as documented? | 7 reward tests in `test_p5_ferrysim_runner.py` pass (7 Oct): the derived reward by hand, F·hand and bytes by hand, hand-worked merge weights as gains, the expected reward equals the realized reward's mean over the draw exactly | Verified here |
| Does the empty-plan problem exist? | The R11 probe (−3.85 against −3.02), recorded in the code and Freeze §5k | Recorded, from Phase 4 |
| How much can any in-flight choice gain, in reward units? | Headroom over FX, per episode: **0.012-0.042** overall, **0.005-0.019** for the in-flight slot; greedy_1 beats FX in all six cells (by more than ε only at `jit-n12-90`, +0.011); ε = max(0.01, 0.1 × headroom) = 0.01 (calibration findings, 6 Oct) | Scored. The units are the derived reward's; `evaluate` defaults to `DERIVED` (`evaluate.py:112`); that no override was passed in the sweep is unverified |
| Does the learned score beat FX under the derived reward? | Study 5.5 sweep: mean held-out returns −0.0795, −0.0776, −0.0779, −0.0774, −0.0778, −0.0779 for γ = 0 ... 0.99; flat (TOST p ≤ 1.3e-4); best FQ ≈ FX (−0.0786); trails greedy_1 (−0.0716) by 0.0059, p = 0.002 | Scored; null |
| Is the weighted rank (V alone) any different from share-first? | Study 5.14, Network AoU, N = 6: `wcov` against `capS`, knee 0.717 against 0.715, stress 0.805 against 0.810; neither a claim | Scored; null at N = 6 |
| Do the dwell and coverage terms matter? | **Study 5.7** (FX-dwell, FX-cov), N = 12 | **Not run** |
| Does F's weight (about a²) beat age alone or a uniform weight? | F against F-prio in 5.8; `uniform` mode has no study | Not run |
| Does a reward derived from the merge weight beat F·hand? | The F·hand arm and the grid left with the learned scores | **Cannot be answered** by the current plan |

---

## 10. Open items and discrepancies

**Open items.**

- [ ] **The derivation decision** (§6), before batch 2. The sensitivity-check option is the cheapest way for the V and reward constants to be examined at all.
- [ ] **Run Study 5.7** (batch 2): FX, FX-dwell, FX-cov, D4. Decide beforehand whether the paper may claim anything about the reward from it; per §5 it may not.
- [ ] **A cell where the merge weight varies** (build plan Phase 5 open items; Study 5.13's skew), so that the reward's "merge weight" differs from a count.
- [ ] **A κ and c₄ sweep** (`PILOT_KAPPAS`, `PILOT_C_ENERGIES`) is defined and not flown (§2.2).
- [ ] **Decide the paper's wording**: "assembled from the same ingredients" (§4) against "one derived objective" (C2). The second needs the derivation.
- [ ] **FQ-dwell and FQ-cov** exist only as arm names and `_ARM_SCORE` entries; nothing trains them under the current verdict.

**Doc and code discrepancies found while writing.**

1. **Contribution C2's wording** (build plan, "What the code does today"; Related Work §0) says "one derived objective ... used as plan score, reward and merge weight". The code has three differently-built weights (§4) and no derivation. The overview already says "declared, not derived"; the contribution ledger's word "derived" should go or wait for the theory track.
2. **Build plan, "One mission end to end", step 5 (Log)** gives the original reward r_k = Σ w_dl·s(age)·v − c_t·Δt − c_e·ΔE − c_miss·[miss]; the parenthetical beside it gives the as-built one. The as-built reward has c_e = 0, no miss term, and adds c_cov·U at the sortie's end. Both texts are in the plan; the second governs.
3. **Build plan Study 5.7 "Sweep"** ("reward weights (c_t, c_e, c_miss)") against the as-built grid c_t × c_cov with no c_miss; the plan's own as-built paragraph already says so.
4. **The overview's J4 table** lists V's constants as "hand-set and swept" (§4 of the overview, summary matrix). I could not find the sweep run (§2.2), so "swept" should read "sweep defined" unless a record I did not see exists.
5. **The value proxy.** The build plan's Phase 5 and the learned-pair-score note say the per-stop value proxy was dropped from the *features* (critic C5), and `plan_score.py` has no v_j; the build plan's "Build demand" step still lists a value proxy v_j per device. The value proxy exists only as the merge's `loss` mode, which no Exp 5 stage selects.
6. **F·hand's "energy"** in the build plan's table ("200·completed − time − energy") is distance times 0.002, not joules (`sim_env.py:54`); the overview's "(200*completed - time - energy)/150" is the same shorthand.

---

## 11. Sources

[FeRRy Build Plan](../FeRRy_Build_Plan.html) (C2, Phase 4 and 5, decisions 2 to 4, Study 5.7, theory track, risks) · [FeRRy Learned Pair Score](../FeRRy_Learned_Pair_Score.html) (reward section) · [Configuration Reference](../HERMES_Configuration_Reference.md) §18.3 · [Related Work Notes](../HERMES_Related_Work_Notes.md) §3.1, §5a · [Experiment 5 Readiness](../Experiment_5_Readiness.md) · [RL calibration findings](../Experiment_5_RL_Calibration_Findings.md) · [Joint RL Methods overview](../HERMES_Joint_RL_Methods.md) · [04 Age and staleness weight](04_Age_Staleness_Weight.md) · `hermes/scheduler/plan/{plan_score,types}.py` · `experiments/ferrysim/{reward,train,evaluate}.py` · `hermes/l1/mission_clock.py` · `hermes/scheduler/selector/{sim_env,selector_train}.py` · `experiments/exp4/driver.py` · `scripts/exp5/params.toml` · `results/exp5/scores/b1/s514.md`.

*Numbers are copied from those records as of 7 Oct 2026. The test runs named in §9 are the only things executed for this document.*
