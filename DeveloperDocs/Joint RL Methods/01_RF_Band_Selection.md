# RF band selection in HERMES / FeRRy — every radio and band decision

*7 Oct 2026. A reading document in the same series as [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md). It restates the code and the records; it decides nothing. Code references are `file:line` as of the working tree on this date. Numbers are copied from the record named beside them; none were re-run, except the one table marked "computed here" in §9.2, which is a re-reading of scored CSVs. Anything not verified is marked **unverified**.*

---

## 1. Purpose

FeRRy makes **radio decisions in three places**, and the repo's history uses the word "band" for all of them. They are not the same decision, they sit on different links, and only one of them is learned (and that one, the learned in-flight score FQ, was tested and kept out). This document separates them:

1. **The contact link** (mule ↔ ground device, a short hop at a stop): which *band class* the stop is served on. Decided **twice**: a commitment at the dock (b̄), and a per-arrival choice in flight.
2. **The backhaul** (mule ↔ base station, at the dock): which *carrier* the upload rides. Decided once per upload, by a deterministic utility.
3. **The retired learner** (`ChannelDDQN`): a third, never-trained decision that survives only as code and a log line.

**The short version.** The band-class decision is real, built and coupled to routing through *range*: a narrower class reaches farther, so S3a clusters into fewer stops. It is **searched, not learned**, at plan time; in flight it is a **fixed rule (FX)**, because the learned alternative (FQ) matched FX and fell short of the one-step rule `greedy_1` (Study 5.5, pre-registered, 6 Oct). The test of whether choosing reach at plan time pays off against a fixed class (**Study 5.4**) is built but **has not run**. The backhaul controller is a utility, not a network; its benefit is real in the model by construction and shows up in the Exp 5 stack once (Study 5.14's F+L1 against F on the seconds backhaul).

---

## 2. The decisions made

| # | Decision | Clock | Inputs | Output | Who decides | Learned? | Code path |
|---|---|---|---|---|---|---|---|
| **R0** | The class set, anchor and range–rate physics | Configuration, per run | `rf_range_m` (60 m), h = 25 m, n = 2.2, σ_sh = 4 dB, floor −6.7 dB | Three classes `wide` / `medium` / `narrow`; R(b), rate(b, SNR) | A declared model, accepted 29 Sep (D1) | No | `hermes/l1/contact_link.py:399-414, 426-657` |
| **R1** | **Plan-time band class b̄** | Dock, once per mission | Demand, weights, deadlines, cap; S3a stops per class; mean SNR | b̄ (committed for both passes), the stop set and route | Exhaustive / local search over every class, smallest `plan_key` | **No** | `hermes/scheduler/plan/plan_search.py:836-881`; `fl_scheduler.py:1454-1700`; commit `mule_main.py:1626-1628` |
| **R1′** | Pinned class (arm FB+c) | Dock | As R1, one class | b̄ = c | The experiment (`band_class_policy = fixed:<c>`) | No | `plan/types.py:1302-1317, 1424-1427` |
| **R2a** | Flight-time band, **committed** (arm F) | Every Pass-1 arrival | — | b̄ again | Nobody | No | `policies/cross_heuristic.py:186-218` |
| **R2b** | Flight-time band, **FX rule** | Every Pass-1 arrival | Per-class targets and dwell at the arrival SNR | The fastest class that covers b̄'s targets | Fixed rule | No | `cross_heuristic.py:168-183, 261-274` |
| **R2c** | Flight-time band **and** next stop, **FQ** | Every Pass-1 arrival | 36-column pair rows, mask | Pair (band, next stop) | Masked pointer double DQN | **Yes — tested, not kept** | `policies/pair_slot.py:762-839`; `selector/pair_features.py` |
| **R3** | Pass-2 band | — | — | b̄ | Not a decision: the runtime refuses any other class | No | `mule/ferry.py:1149-1154` |
| **R4** | **Backhaul carrier** | Each upload (dock) | Per-carrier SNR at upload start | One of three carriers | Fixed `argmax g_c`, or the U(c,t) utility (H3, H1+L1, F+L1) | No (utility) | `hermes/l1/channel_utility.py:46-88`; `channel_model.py:882-902`; `ferry.py:1294-1340` |
| **R5** | `ChannelDDQN` contact-band pick | Per contact (logging) | 8-feature state | Argmax over 3 carriers' slide-26 frequencies | A random-init network | Never trained; **retired from the plan** | `hermes/l1/channel_ddqn.py:56-102`; `mule_main.py:2229-2239` |

Two fixed points hold across R1 to R3: **range is a commitment** (b̄ sets R_planar(b̄), which sets S3a's radius, the stops and the route) and **rate is revisable** (the class that serves a stop may differ in Pass 1 under FX or FQ). Pass 2 flies b̄ in queue order.

---

## 3. What a band class is (R0)

### 3.1 The classes, and why bandwidth rather than carrier

All three classes share **one carrier, 3.32 GHz** (`contact_link.py:230`). The repo's three L1 carriers (3.32 / 3.34 / 3.90 GHz, `channel_ddqn.py:38`) differ by 0.05–1.4 dB of free-space loss, which cannot carry a range trade; the classes differ in **LTE channel bandwidth** instead (`contact_link.py:19-24`, `Build_Plan` D1).

| Class (index) | Channel, N_RB | B_occ | κ | Peak rate (CQI 15) | R slant / planar, n = 2.2 | n = 3.0 |
|---|---|---|---|---|---|---|
| `wide` (0) | 20 MHz, 100 | 18 MHz | 0.754 | 75.4 Mb/s | 65.0 / 60.0 m | 65.0 / 60.0 m |
| `medium` (1) | 5 MHz, 25 | 4.5 MHz | 0.734 | 18.3 Mb/s | 122.1 / 119.5 m | 103.2 / 100.1 m |
| `narrow` (2) | 1.4 MHz, 6 | 1.08 MHz | 0.732 | 4.39 Mb/s | 233.5 / 232.2 m | 166.0 / 164.1 m |

Source: `contact_link.py:19-33, 399-403`. I re-evaluated the ranges with `ContactLink(anchor_planar_m=60.0)` at both exponents; they match. A fourth optional class, `medium_wide` (10 MHz), is appended at index 3 so indices 0–2 never move (`:403, 409`). **The peak rates are never reached at h = 25 m**: over 0–60 m planar the realised rate is 20.0 → 5.1 Mb/s on wide, 9.0 → 3.9 on medium, 3.6 → 1.9 on narrow (`contact_link.py:175-179`, reproduced here).

### 3.2 Formulas and constants (`contact_link.py:42-76`)

```
B_occ,b        = N_RB,b · 180 kHz
SE(s)          = efficiency of the highest CQI whose SNR threshold ≤ s      (CQI table, :322-338)
rate(b, s)     = min(κ_b · B_occ,b · SE(s), B_occ,b · log2(1 + 10^(s/10)))   for s ≥ floor;  0 below   (:670-677)
M_sh           = Φ⁻¹(q) · σ_sh = 1.2816 · 4 dB = 5.13 dB      (q = 0.9)
R_b            = R_anchor · (B_occ,anchor / B_occ,b)^(1/n)      R_anchor = hypot(60, 25)             (:521-543)
SNR_b(d3D)     = floor + M_sh + 10 n log10(R_b / d3D)           (the *mean*; d3D ≥ 1 m)               (:648-657)
dwell(N, b, s) = 8N / rate(b, s) seconds; None below the floor, never ∞                               (:679-696)
```

Constants: floor −6.7 dB (CQI 1), h = 25 m, n = 2.2 (sensitivity n = 3.0), σ_sh = 4 dB, q = 0.9, anchor 60 m planar (`:233, 456-461`).

**Why bandwidth gives range.** One EIRP is shared by every class, and noise scales with occupied bandwidth, so narrowing from 18 MHz to 1.08 MHz raises SNR at every distance by 10·log10(100/6) = 12.2 dB; a log-distance exponent turns that into the factor above. The ratio uses *occupied* bandwidth (12.2 dB), not nominal (11.5 dB) (`:59-64`). The anchor implies an EIRP of −11.3 dBm at n = 2.2 (`implied_eirp_dbm`, `:707-734`; I re-evaluated −11.27).

**R(b) is a 90 % edge-availability range, not the floor-rate range.** At R(b) the mean SNR sits M_sh above the floor; wide therefore runs at CQI 3 (5.12 Mb/s) at its own edge, not at the floor rate (`:111-122`, recorded as a plan deviation, critic A8-i). The floor-rate range is 111.2 m slant on wide (`floor_range_m`, `:636-642`).

**Sweep knob (unit U10).** `narrow_range_ratio` sets narrow's planar reach as a multiple of the anchor's instead of the derivation's 3.87 (`:447-453, 463, 531-535`); the class's mean-SNR curve moves with it, so its implied EIRP then differs from the others' (checked: ratio 2.0 gives 120 m and −17.4 dBm). Study 5.4 sweeps 2 and 3 beside the derivation (`scripts/exp5/params.toml`, `[s54]`).

**The anchor is an assumption.** No AERPAW measurement gives R(b). 60 m is `rf_range_m`, Exp 4's default, chosen so the range gate binds where S3a already does; a physical budget (10 dBm, 10/2 dBi) would reach kilometres (`:124-145`).

### 3.3 Failure modes of the model

- **Below the floor** the rate is 0 and `dwell_s` returns `None`. A member below the floor at arrival is unreachable (TIMEOUT, `answered=False`, no dwell); a backhaul upload below the floor is lost and charged the floor-rate time (`:147-157`; `ferry.py:876-894`).
- **Mean-SNR pricing under-predicts dwell** because dwell is convex in SNR: realised missions ran **+0.8 %** longer on wide at 1 MB, **+7.1 %** on wide at 10 MB, **+19.2 %** on narrow at 1 MB (`HERMES_Configuration_Reference.md` §17.2, final check). The plan prices the mean; nothing in the plan prices the spread except V's link term (§4.4).
- **The narrow-band cliff.** S3a on narrow forms one field-wide contact in about 98 % of realism layouts; admitting it whole is a 0-or-N decision. F and the FB+ arms fly member subsets by default to avoid it (`HERMES_Configuration_Reference.md` §17.1, §18.5).
- **Bit-identity across operating systems is not guaranteed** (platform `log10`, `pow`, `sin`, `cos`; `contact_link.py:190-191`, `channel_model.py:57-66`).

---

## 4. R1 — The plan-time band class b̄

### 4.1 What is decided and what it commits

At the dock, before takeoff, `FLScheduler.build_ferry_plan` (`fl_scheduler.py:1454`) chooses **b̄ and the Pass-1 route π as one decision** and holds b̄ for both passes. The steps relevant to the radio:

1. For each class the arm may fly (`PlanSetup.searched`, `plan/types.py:1424-1427`): S3a at **that class's R_planar** (`cluster_by_rf_range(..., rf_range_m=c.radius_m)`, `fl_scheduler.py:1615-1620`), then the hover rule for capped devices, then Pass 2 priced on that class (`:1630-1632`).
2. `plan_search` searches each class on its own and returns the smallest plan key over classes (`plan_search.py:872-881`; `best = min(results, key=lambda r: plan_key(params, r.best))`).
3. The guard fold, then the commit: the class's model becomes the scheduler's model, so the in-flight check, the re-plan and Pass 2 price b̄ (`fl_scheduler.py:1641-1688`). The mule then calls `fx.set_band(commit.band)` (`mule_main.py:1626-1628`), which recomputes the runtime's range and band index (`ferry.py:805-824`).

**Choice rule** (`plan_search.py:28-41`): the age-cap key first, then (default `coverage_rank = lexicographic`) the served weight share, then V, then the class index and the stops. A total order, so the pick never depends on enumeration order. Consequence worth stating: **the plan key puts coverage before time, so a plan can fly a much longer mission to serve one more device** (Build Plan §5.4 caveats).

### 4.2 The search, by size

| Search | When | What it is |
|---|---|---|
| `exact` | demand ≤ 6 devices | Whole family, depth first; optimal over the member subsets V prices |
| `stop_subsets` | more devices, ≤ 6 stops in the class | Ordered subsets of stops; each whole if it fits, else reduced greedily |
| `local` | more stops | 2-OPT tour, member trim, up to three first-improvement scans, bounded by counts not wall time |

(`plan_search.py:43-117`.) Each class is searched independently, which is why **F's plan key is never above any FB+c's** (critic A3, `:121-124`). At N = 6 a whole plan took at most about 0.25 s with nothing pruned and about 20 ms under 30–120 s budgets (`:54-57`). Above 6 devices the search is a heuristic and only `exact` is optimal.

### 4.3 What the planner sees of the channel

**The mean SNR only** (`ContactChannel.pred_snr_db`, `channel_model.py:689-692`: no shadowing, no interference, no time). Dwell is priced as `link.dwell_s(session_bytes, band, mean_snr + offset)` (`ferry.py:854-865`, offset 0 at plan time, "δ_obs = 0"). Pricing the seeded phase would be an oracle. The upload at the dock is priced on the **anchor class** (`wide`) at the held carrier's mean backhaul SNR (`ferry.py:896-913`).

### 4.4 The link term in V

V's expected-link-loss term L uses the outage probability of each served member on b̄ at its stop:

```
P_out(b, d) = Φ((floor − SNR_b(d)) / σ_eff),    σ_eff = sqrt(σ_sh² + σ_I² + A²/2)
P_out = 1 beyond R_planar(b)
```

(`ferry.py:966-998`; copied by `plan_score.outage_by_distance`.) A²/2 is the variance of a sinusoid over a uniform phase; it is a normal approximation, since the sinusoid is not normal. With the clean contact regime (A = 1, σ_I = 0.4) σ_eff ≈ 4.08 dB, with the jittery regime (A = 5, σ_I = 1.5) ≈ 5.55 dB (computed from the formula; constants in §7.1). V's constants (c₁ = 1, c₂ = κ·N_demand, c₃ = c₂, c₄ = 0.1) are hand-set and swept, and are described in [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) §1.3; they are not restated here.

### 4.5 Interfaces

- **L1 → L2:** the planner receives physics only as floats and callables, never `hermes.l1` (`ferry.py:27-31`). `FerryRuntime.plan_classes()` hands it one `PlanClass` per class: radius = R_planar(b), a model whose physics is bound to the class (critic B3), the class index, and `outage` (`ferry.py:1000-1037`).
- **L2 → L1:** `set_band(b̄)` after the commit; nothing else.
- **L3 → L2 (into V):** coverage weights, age cap, deadlines. L3 never touches the radio.
- **Pinned classes (R1′):** `band_class_policy = fixed:<c>` searches class c only and refuses any flight slot but `committed` (`plan/types.py:1313-1318`): FB+c "flies only class c", so FX or the pair would not be FB+c.

### 4.6 Failure modes

- **b̄ is priced at the mean but flown at the realised SNR.** A member whose realised SNR is below the floor at arrival is not solicited at all.
- **Range gate is planar and inclusive** (`ContactLink.in_range`, `:629-634`); `range_m` is slant and would admit members up to R − R_planar too far (5 m on wide), so planar gates must use `range_planar_m` (`:609-617`).
- **T_nom, the deadline unit, is priced on wide for every arm**, so an arm with shorter missions fits more of them into a deadline window (documented, not corrected; critic C4).
- **Where b̄ matters little.** At N = 6, one mule, 150 s, one stop serves all six devices in most missions, and every pair FX's slot weighed over 24 episodes was the dock (`Experiment_5_Readiness.md`, "What could still go wrong").

---

## 5. R2 — The flight-time band

### 5.1 The slot, and the three fillings that touch the band

At each Pass-1 arrival the mule's supervisor builds an `ArrivalView` (`ferry.py:1180-1239`): for each class, **the members it would solicit now** (within R_planar(c), inclusive, and `snr ≥ floor` on the realised channel at the arrival time) and **the dwell of serving them all**, each priced at its own realised SNR. The slot acts in **Pass 1 only**, at every arrival including the first; at takeoff and in Pass 2 the mule flies the committed order on b̄ (`cross_heuristic.py:22-37`).

| Filling | Band rule | Next-stop rule | Learned |
|---|---|---|---|
| `committed` (F, FB+) | b̄ | The plan's next stop (index 0) | No |
| `cross_heuristic` (FX) | The fastest class that still reaches every target b̄ reaches, priced at the arrival SNR | After a stop, the nearest remaining stop whose move to the front keeps the rest feasible; else index 0 | No |
| `pair_q` (FQ) | One (band, next stop) decision: masked argmax of a learned score | (same decision) | **Yes** |

Dispatch: `flight_slot_policy` (`cross_heuristic.py:290-300`); `MuleSupervisor._init_plan` and `_ferry_fly_contact` (`mule_main.py:2225-2257`).

### 5.2 FX's band rule (`cross_heuristic.py:168-183`)

```
need       = targets of b̄ at this arrival
candidates = { c : need ⊆ targets(c) }                       (b̄ is always one)
FX band    = argmin_c ( dwell_s(c), −|targets(c)|, [c ≠ b̄], index(c) )
```

Because b̄ is always a candidate, FX **never dwells longer than F would at the arrival SNR and never reaches fewer devices**. The rule replaced an earlier "reach the most members" rule after critic probe E: at a 60 s budget, 8 of 50 switches went to a narrower class for one more member, adding a median 60.5 s and up to 94.1 s of dwell, and 6 last-stop switches landed past a budget b̄ would have met, where nothing re-checks (`:44-57`, critic A7). The departure check still prices the rest on b̄, which stays conservative for whatever class FX flies (`:55-57`). `greedy_1`, a scripted reference in the pair slot, *is* the "most targets" rule, and it **beat FX** in the 5.5 headroom cells and the sweep (§9.4); the mask now prices the landing at the observed rate, but realised dwell still runs longer than priced (`pair_slot.py:84-92`).

### 5.3 FQ's band decision, mask and fallback

The pair view holds, for each pair, class-major rows. A pair (b, s) is admitted only if (`pair_slot.py:19-45`; `fl_scheduler.py:1966-2025`):

1. b reaches every device b̄ reaches at this stop (the scope guard, `selector/scope_guard.py`);
2. s is a stop of the plan (or home, only when none is left);
3. the whole rest of the flight still fits after serving here on b at the **observed** dwell and flying to s, folded on b̄'s model at the mean SNR under the arm's in-flight rule (the S3b predicate; `fits_after_service`).

If **no pair fits**, the slot flies FX's pair and records `mask_empty` (`pair_slot.py:797-799`; `plan/types.py:785`). That is common: FX's own pair overruns the budget at about 19 of 71 last-stop arrivals at N = 6 (`plan/types.py:779-784`; "Phase 5 design, finding 3"). The learned score reads 36 columns (`PairFeatureSchema(("wide","medium","narrow"), phase=True).dim == 36`, checked here) from `pair_v1`: per-class SNR now, dwell, gain, the next stop's mean SNR per class, and a 12-column **phase block** (per-class offset now and at the previous arrival, its age, and sin/cos of the age and of the arrival lag over P_c) (`selector/pair_features.py:25-62`). The network is a pointer double DQN with hidden (64, 64), tanh, one score per row (`selector/pair_q.py:356-365`). Training, replay, ε schedule and checkpoint format are in [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md) §2.5 and are not repeated.

### 5.4 Why the band belongs on this clock

The interference term is a sinusoid of period P_c = 60 s with a per-class phase (§7.1), and legs run roughly 0.6–1.3 of a period in the prototype's constants, so the band that is best at takeoff lands in a trough at two of four stops: mean **−0.14 committed against +0.75 best-available at arrival** (`HERMES_Layer_Redefinition_and_RL_Decision.md` §2, from `hermes_rl/drone_env.py`'s constants; equal amplitudes, so *none of that gap comes from committing to the wrong band, all of it from committing at all*). That table is the arithmetic case for a flight-time decision. It does not say a *learned* flight-time decision beats a one-step rule, and §9.4 shows the learned score did not.

### 5.5 Interfaces

- **L1 → L2 (observation):** `ArrivalView` (per-class targets, dwell at the arrival SNR); `class_offsets_db` (realised minus mean SNR per link, median across links, critic C6; `ferry.py:1399`); `stop_contexts` (the next stop's leg, dwell on b̄ and per-class mean SNR; `ferry.py:1350`). All pure readers; they charge nothing and move no band.
- **L2 → L1 (actuation):** `contact_plan(..., band=decision.band)` builds the stop's plan on the chosen class (`mule_main.py:2249-2257`, `ferry.py:1121-1178`); the contact then prices each target at its own session start and charges `dwell` to the clock.
- **L3 → L2:** the pair rows' `age_next`, `capped_next`, `weight_share`, `on_time_next`, `slack_next` and the reward's merge weight (J4 in the overview). The band decision itself never sees L3 except through those columns.

### 5.6 Failure modes

- **Realised dwell differs from the view's.** The view prices each target at the arrival SNR; the contact prices each at its own session start (critic C2), so realised dwell can occasionally exceed F's (`HERMES_Configuration_Reference.md` §18.5).
- **No check after the last Pass-1 stop** (`mule_main.py` flies home), so a last-stop overrun lands past the budget. FX's design limits it; the mask prices it for FQ.
- **The pair mask folds the rest on b̄ at the mean SNR**, so it can admit a pair the realised channel then breaks; the departure check, trim and re-plan catch that after the stop.
- **FQ's learner may have copied FX** (a hypothesis, not a result; §9.4).

---

## 6. R4 — The backhaul carrier controller (L1)

This is the only radio decision the original Exp 4 evaluated (arm H3). It is on a **different link** from R1–R3: mule to base station, three carriers, once per mission (legacy) or once per upload (seconds axis).

### 6.1 The controller

```
U(c, t) = R(γ₁(t) + g(c)) − κ(c) − λ(c, t)
```

- `R(·)` = `rate_tier(snr)` = `log2(1 + 10^(snr/10))`, and **0 for snr ≤ 0 dB** (`channel_utility.py:29-42`).
- `γ₁(t) + g(c)` is the carrier's effective SNR passed in as `snr_per_band[c]`. The code takes a per-carrier effective SNR; it does not add g(c) itself.
- `κ(c)` = `channel_use_cost[c]`, default `(0, 0, 0)`; `backhaul_plan` and `BackhaulChannel.select_carrier` both pass **zero** use cost (`channel_model.py:197-200, 898-901`).
- `λ(c, t)` = `switch_cost` = **0.5**, charged only if a band is already held (`current ≥ 0`) and differs; units are rate-tier units, not a physical retune cost (`channel_utility.py:57-69`).
- `select` evaluates the incumbent first and replaces it only on a **strictly** greater utility, so ties keep the held band; with no held band (−1) it starts from band 0 (`:71-88`).

Deterministic, causal (the current observation only), no randomness. The comment block at `:57-64` says the 0.5 default "sits on a plateau where adaptive never worse under jittery holds for switch_cost in [0, ~1]" and that at λ → 0 the controller is a per-instant argmax-SNR oracle, so the superlative is near-tautological; it asks for a retune-time model "before publishing". That caveat is the file's own.

The fixed comparator is **`best_average_band`** in the legacy model (the band with the best mean SNR over the whole realised trace; `channel_utility.py:91-107`), and **`argmax_c g_c`** on the seconds axis (`channel_model.py:882-884`). The legacy one uses the future; the seconds-axis one does not (`:798-803`).

### 6.2 Two models of the backhaul channel

| | Legacy `mission` model | Seconds model |
|---|---|---|
| Axis | Mission index m | Simulated seconds t (τ = t − epoch) |
| SNR | `base + g_c + A·sin(2π(m/period + φ_c)) + noise` | `base + g_c + A·sin(2π(τ/P_bh + φ_c)) + σ·ξ_c(t)`, P_bh = n_missions·T_nom |
| Regimes (base, A, σ) | clean 12 / 1 / 0.4; jittery 6 / 5 / 1.5 dB | The same (`channel_model.py:260-263`) |
| g_c, φ_c | `U(0, 3)` dB and a shuffle of {i/n} from a sequential RNG | Same distributions, hashed from the trial salt |
| Loss | `loss_from_snr(snr)` = 1/(1 + e^((snr−3)/2)), drawn by the cluster against a **per-mission schedule** with a stream | `p_loss = loss_from_snr` of the SNR at the upload's start; below the floor `p_loss = 1`; drawn **keyed** by (seed, mule, mission round) |
| Code | `channel_model.py:115-221` (verbatim from `experiments/exp4/channel.py`, pinned by goldens) | `channel_model.py:779-921`; `ferry.py:1294-1340`; `cluster.py:1391-1420` |

Backhaul loss magnitudes per upload over 400 seeds, 4 missions, T_nom = 219 s (`channel_model.py:68-95`): seconds axis fixed carrier **0.161** jittery / 0.004 clean; legacy fixed best-average **0.153** / 0.004; legacy adaptive (H3) **0.018** / 0.004; this module's own draws: fixed 0.163 / 0.004, H3 0.017 / 0.004. Against the flat 2 % the recorded `--realism` runs apply.

The upload time is charged on the **anchor class** (`wide`) rate table: `8 · UP bytes / rate(wide, backhaul SNR)` (`ferry.py:889-894`). The planner prices the same upload at the held carrier's *noise-free* mean (`predicted_upload_s`, `ferry.py:896-913`).

### 6.3 Where it runs, and where it does not

The controller flies only when an arm asks: `_ADAPTIVE_BACKHAUL_ARMS = ("H3", "H1+L1", "F+L1")` (`experiments/exp4/driver.py:274`). H1+L1 and F+L1 are **refused** unless the cell has `--l1-channel` or, on the simulated clock, `--backhaul-model seconds` (`driver.py:1380-1400`), because anywhere else they would fly as H1 or F under another label. **Most Exp 5 cells fly neither** (only 4 of batch 1's 142 `.argv.json` files carry `--backhaul-model seconds`, none carries `--l1-channel`; counted here): the batch-1 argv for F's `capS` cell has `--realism` and no `--backhaul-model`, so the backhaul loss is the flat recorded percentage and the upload is timed at the fixed carrier's mean (`results/exp5/b1/s514/n6k1_knee__capS.csv.argv.json`; `HERMES_Configuration_Reference.md` §17.2, `backhaul_model`). The adaptive backhaul is exercised in Exp 5 by the `secF` / `secFL1` pair of Study 5.14 (done) and by Study 5.15 (batch 3, not run).

### 6.4 `rf_prior.py` and the L1 → L2 edge

`RFPriorStore` is a thread-safe per-band last-SNR store with a read-only scheduler API (`snapshot`, `read`, `mean_snr_db`) and an L1-internal `_record` (`rf_prior.py:39-81`). `RFPriorProducer` is its first production writer (Phase 3, critic B4): after each backhaul upload the mule calls `observe_upload(carrier, snr, t)`, and `prior_snr_db()` returns 20 dB (`DEFAULT_RF_PRIOR_SNR_DB`, `:86`) before the first upload, then the mean over carriers seen, or one carrier's last observation (`:89-160`; wired at `mule_main.py:2587-2592`, `ferry.py:1330-1331`). It replaces the driver's trial-mean prior, which under `--l1-channel` used the future.

The prior's consumer is slot 10 of the **legacy S3.5 selector's** feature vector, reached through `build_contact_queue` → `rank_contacts` (`HERMES_Experiment4_L1_RF_Layer.md` §4). **Plan mode does not read it:** `build_ferry_plan` takes no `rf_prior_snr_db` argument (`fl_scheduler.py:1454-1459`; grep of that file shows the parameter only on the legacy queue builders at `:701, 796`). So in FeRRy's plan arms the backhaul controller affects round closure (through the loss draw), not any scheduling decision.

### 6.5 Interfaces and failure modes

- **L1 → L3 (via the cluster):** `p_loss` rides the `UpBundle.backhaul` record; the cluster draws the loss, a lost upload leaves an empty stand-in partial and **the round does not close** (`cluster.py:153-170, 1391-1420`). That is the L1 → L3 edge, and it is why the backhaul decision shows up in round closure and not in the model's endpoint.
- **L2 → L1:** none for the carrier (policy is configuration: `backhaul_policy` ∈ {`fixed`, `adaptive`}, `ferry.py:181-183`).
- **Perfect, cost-free sensing:** the controller sees every carrier's SNR at the upload instant (`snr_all(t)`, `channel_model.py:874-876`).
- **Switching is not charged in the realised loss**, only in the decision, so the gain is an upper bound assuming lossless retuning (`HERMES_Experiment4_L1_RF_Layer.md` §5).
- **The sign is near baked in.** SNR → loss is monotone on a shared trace, so a better pick cannot lose; the magnitude is calibration-dependent (≈ 0.02–0.29 across defensible constants; same document §5).
- **Tier-0 plateau (observation from the code, unverified against a run).** `rate_tier` is 0 at or below 0 dB while the contact link's floor is −6.7 dB; for two carriers at or below 0 dB the utility is the same (0 − switch cost), so the held carrier wins by incumbency. The backhaul regimes' bases (12 and 6 dB) sit above that region except in deep jittery troughs.

---

## 7. The channel model, mission clock and dwell

### 7.1 Contact channel (`channel_model.py:534-760`)

```
SNR_b,j(t) = SNR_b(d3D_j) + X_j(t) + I_b(t)                       (snr_db, :725-734; the mean is pred_snr_db)
X_j(t)     = σ_sh · smooth_normal(salt, "shadow", j, t, 7.4 s, epoch)    same for every class (one carrier)
I_b(t)     = A · sin(2π (τ/(P_c·m_b) + φ_b)) + σ_I · ξ_b(t)        (interference_db, :712-723)
```

| Constant | Value | Source |
|---|---|---|
| σ_sh | 4 dB | `SHADOW_SIGMA_DB`, `:269` (TR 36.777 Annex B, 3.66–3.83 dB at 20–30 m, rounded up) |
| Shadowing correlation time | 7.4 s | `:273` (37 m decorrelation at 5 m/s) |
| P_c | 60 s, one period for all classes | `:278`; an assumption, swept in Study 5.6 |
| Noise bin | 1 s | `:281` |
| Regimes (A, σ_I) | **clean** 1 dB, 0.4 dB; **jittery** 5 dB, 1.5 dB | `CONTACT_REGIMES`, `:253-256` |
| φ_b | seeded shuffle of {0, 1/3, 2/3} over the three D1 classes; the 10 MHz class takes a gap midpoint | `_contact_phases`, `:501-529` |
| m_b | 1 for every class (a field exists, no config sets it) | `:620-628` |

**Seeded and paired by construction.** Nothing is drawn from a stream. `smooth_normal(salt, stream, key, t, bin_s, epoch)` is a Box–Muller draw from `sha256(salt|stream|key|k)` for bin k = ⌊(t − epoch)/bin⌋, interpolated linearly between bins and rescaled to unit variance (`:395-448`); correlation ≈ 0.74 at half a bin, ≈ 0.29 at one bin, exactly 0 from two (docstring). Outcomes (a device's availability, a backhaul loss) use `keyed_uniform(salt, key, round)` (`:451-469`). Two arms that ask for the same (t, class, link) get the same value whatever else they asked, which is what keeps arms paired. Phases and noise are keyed by class **name**, so a class keeps its I_b(t) when the class set changes (`:520-529`). `shadow_keying = "position"` keys X_j by (device, 37 m grid cell) instead, so two arms hovering in one cell see the same shadowing; default is `time` (`:694-710`).

**Jittery versus clean.** Two different knobs share the word.
- *Contact regime* (this table): clean A = 1 dB, σ_I = 0.4; jittery A = 5, σ_I = 1.5. The default is clean in the code; **Exp 5 flies the jittery contact channel everywhere** (decided; `Experiment_5_Readiness.md` "Decided already"; `--contact-regime jittery` in the batch-1 argv). FerrySim's `clean` family is the negative control (`experiments/ferrysim/cells.py:1-60`).
- *Network (backhaul) regime*: clean base 12 / A 1 / σ 0.4 dB; jittery 6 / 5 / 1.5 (`:260-263`). FerrySim sets it to jittery in every cell so the families differ in the contact channel only (`cells.py:154-160`).

Study 5.15 also exposes `--interference-amp-db`, `--interference-sigma-db`, `--n-pl`, `--shadow-sigma-db` (`HERMES_Configuration_Reference.md` §20.4; `Build_Plan` 5.15's "To build" is out of date, the flags exist, §10).

### 7.2 Mission clock and dwell (`mission_clock.py`)

The mule's mission time is a simulated clock starting at `SIM_EPOCH_S = 1e6` s, refusing to reach `1e9` s so a stamp's magnitude says which clock made it (`:84-89`). It is charged by kind (`LEDGER_KINDS`, `:97-105`), and nothing sleeps for simulated time.

| Kind | Charge |
|---|---|
| `transit` | leg / cruise speed (5 m/s) |
| `dwell` | once per contact: Σ over answered targets of `8·bytes / rate_bps(band, SNR)`, each SNR taken at that target's own session start; without a band, the cost model's 1 s |
| `listen` | `listen_s` = 1 s, once per contact missing any expected reply |
| `return` | leg to the dock |
| `upload` | `8·UP bytes / rate(wide, backhaul SNR)` |
| `turnaround` | 30 s once per mission |
| `dock_wait` | Lamport sync to the cluster's simulated time |

(`mission_clock.py:13-31, 476-478`.) Energy is a pure function of the ledger: 143.6 W flying, 168.5 W hovering (dwell and listen), ground time free; labelled simulated (`:315-377`, `EnergyModel`). `FlightModel.leg_s` uses exactly the arithmetic of the S3b cost model, so the clock charges the leg the planner predicted (`:514-521`).

**Dwell = bytes / rate** replaces Exp 4's constant `session_time = 1.0 s`. That constant was cut 3 of the decision memo (a slow and a fast band cost the same at a stop). Payloads: θ is 18,756 B, a Pass-1 session 37,576 B, and Exp 5's declared payload is 1 MB each way (`HERMES_Configuration_Reference.md` §17.3). Worked numbers at h = 25 m, n = 2.2, 0–60 m planar: a 37,576 B session takes 0.015–0.059 s on wide and 0.084–0.158 s on narrow; 1 MB each way takes 0.8–3.1 s on wide and 4.5–8.4 s on narrow, and 53.7 s on narrow at 200 m (`contact_link.py:175-189`; I reproduced 0.8 / 3.13 s on wide and 4.47 / 8.41 / 53.68 s on narrow). **Wide out-rates the narrower classes wherever it reaches**, so the trade is reach against dwell, and it matters only for far devices or large payloads.

---

## 8. History: how the decision got here

| Version | Radio decision | Source |
|---|---|---|
| **Prototype** `hermes_rl/drone_env.py` | One joint discrete action (waypoint × base station × channel), channel phases at evenly spaced offsets; time-varying term dominates distance | Overview §5.1; decision memo §2 |
| **MA-P-DQN (original design)** | A joint (Δposition, channel) action: a continuous trajectory head plus a discrete channel head | `HERMES_FL_Scheduler_Design.md:132-143`; April presentation slides cited in Freeze §"Records corrected" |
| **Reframe** | L1 becomes a **channel-only DDQN** (8 → 16 → 3, three slide-26 carriers); "navigation is mechanical"; the trajectory head becomes L2's `TargetSelectorRL` | `HERMES_FL_Scheduler_Design.md:132-148`, `channel_ddqn.py:1-27` |
| **What Exp 4 actually ran** | The deterministic utility U(c,t) on the **backhaul**, once per mission; `ChannelDDQN` has no trainer, the process runtime passes no channel actor, and where a test passes one its choice is logged, not actuated; the contact link had no band | Freeze, "Records corrected" (`HERMES_Scheduler_Freeze.md` ~L512-522); `ferry.py:1262-1269` |
| **Decision memo (cuts 1–3)** | Introduce a second band decision on the *contact* link that sets range; put the band decision on the flight clock; close the return path | `HERMES_Layer_Redefinition_and_RL_Decision.md` §1.1 |
| **FeRRy Phase 3** | Band classes with a range–rate model, seconds-axis channel, mission clock | `contact_link.py`, `channel_model.py`, `mission_clock.py` |
| **FeRRy Phase 4** | Plan-time b̄ and the FX flight slot | `plan_search.py`, `cross_heuristic.py` |
| **FeRRy Phase 5** | The learned pair FQ in the slot; **ChannelDDQN retired from the plan** (decision 8): "not trained, its code and logging kept"; H2 and H3 leave Exp 5, H1+L1 added as the adaptive backhaul's reference | Build Plan Phase 5 (decision 8); Freeze decision 8 (`:1534-1536`) |
| **Exp 5 addendum** | F+L1 (F with H3's backhaul) for Studies 5.14 and 5.15; U10 sweep knobs for 5.4; interference flags for 5.15 | `HERMES_Configuration_Reference.md` §20.4, §20.5, §20.10 |

The pattern: a learned joint action was reframed to a learned channel pick, which was never trained, replaced by a deterministic utility on a different link (the backhaul), and the contact-band question was reopened as a **range decision at plan time** plus a **rate decision in flight**, the latter tested as a learned score and found no better than a fixed rule.

---

## 9. What the evidence says

### 9.1 Summary

| Question | Evidence | Status |
|---|---|---|
| Is choosing reach (b̄) at plan time worth it against one fixed class? | Study 5.4: F against FB+wide / medium / narrow, metric Network AoU, 4 arms × 6 settings × 2 payloads × 20 trials = 960 (`params.toml` `[s54]`), O1 oracle at N = 6 beside it | **Not run** (batch 2; no `results/exp5/b2`) |
| Does the planner use the classes? | Computed here from batch 1 (§9.2) | Descriptive only |
| Does a flight-time band rule beat committing? | Batch 1, F against FX (§9.3) | **No claim** |
| Does a *learned* flight-time score beat the rule? | Study 5.5, pre-registered (§9.4) | **Flat; FX kept** |
| Does adaptive backhaul selection pay off? | Exp 4 channel model; Exp 4 end-to-end ties; C1 confirmation; Study 5.14 `secFL1` (§9.5) | Model: yes by construction. End-to-end: mixed. Exp 5: one claim |

### 9.2 What the planner commits (computed here from batch 1's scored CSVs)

Unweighted mean over trials of `band_shares`, "the share of Pass-1 stops flown on each band class" (`traces_scorer.py:104-105`). Files: `results/exp5/b1/s514/n6k1_knee__capS_scored.csv` (F, S = 2), `…/s53/n6k1_knee__FX_scored.csv`, and the stress files. F's rows are the first 20 trials of the 40-trial file to match FX's 20; that they share seeds is **unverified** (same `--base-seed` in the argv, not cross-checked row by row).

| Cell (N = 6, one mule) | Arm | wide | medium | narrow |
|---|---|---|---|---|
| knee (150 s) | F, first 20 trials | 0 | 0.208 | 0.792 |
| knee (150 s) | F, all 40 | 0 | 0.174 | 0.826 |
| knee (150 s) | FX | 0.046 | 0.463 | 0.491 |
| stress (75 s) | F, first 20 | 0.020 | 0.212 | 0.767 |
| stress (75 s) | FX | 0.136 | 0.474 | 0.390 |

Reading: at the measured knee budgets the planner picks `narrow` or `medium` essentially never `wide`, and FX shifts a large share of stops to a faster class than b̄ (note FX's per-stop switches count, `traces_scorer.py:104`). That shows the plan-time decision **changes what is flown**; it does not show it is worth it, which is 5.4's question. With three mules the picture differs (`s53/n6k3_knee__F`: 0.30 wide, 0.60 medium, 0.10 narrow).

### 9.3 F against FX, whole-scheduler comparison (`results/exp5/scores/b1/s53.md`)

N = 6, one mule, time to τ = 0.71 (lower is better), 20 paired seeds, Holm across the study: **knee** F 194 s, FX 178 s, difference 16 s, CI [3.93, 31.8], Holm p = 0.146, **no claim**; **stress** F 178, FX 175, Holm p = 0.799, no claim. Three mules: F 52.3 and FX 56.4, no claim. So a fixed flight-time rule matches the committed slot on this metric at this size. (F against the baselines is in the overview §1.4; not repeated.)

### 9.4 Study 5.5: FX against the learned FQ (`Experiment_5_RL_Calibration_Findings.md`, "The sweep's verdict")

γ ∈ {0, 0.25, 0.5, 0.75, 0.9, 0.99} × 10 seeds on `jittery56`, 1,000 held-out episodes per cell. Mean held-out returns −0.0795, −0.0776, −0.0779, **−0.0774**, −0.0778, −0.0779. **Not rising** (best gain 0.002, γ = 0.75, Holm p = 0.14); **flat** (every γ equivalent to γ = 0 within ±ε = 0.01, TOST p ≤ 1.3e-4). The best learned score (−0.0774) sits at FX's level (−0.0786) and trails `greedy_1` (−0.0716) by 0.0059, CI [−0.0066, −0.0050], p = 0.002. `replace_fx = false`, `keep_learned = false`; FX stays as FeRRy's in-flight rule; `gamma_star = 0.75` is recorded only. The calibration found the learned score at FX's level and the `clean` control **sanity-failed by 0.0005**; headroom over FX was thin (0.012–0.042).

**Held as a hypothesis, not a result:** the learner copied FX (a replay that mostly follows FX, validation bounce as large as ε, TD loss barely moving). The one-revision screen on the clean control cells was *not run* (option A was chosen). The paper can say the learned score matched FX, not that no learner could do better. This radio-specific reading follows: the in-flight band choice has a **one-step** structure on these cells, because looking ahead (γ > 0) added nothing.

### 9.5 The backhaul controller

1. **Channel-model level (Exp 4).** Mean per-mission backhaul-loss reduction, fixed − adaptive, 30 seeds, n_missions = 6, 3 bands: clean **+0.0003**, jittery **+0.1379** (adaptive ≤ fixed in 30/30 seeds, 1000/1000 at 1000 seeds). The strongest fair fixed baseline, the loss-optimal band, still loses by +0.1344 (`HERMES_Experiment4_L1_RF_Layer.md` §6.1).
2. **End-to-end (Exp 4, 20 paired seeds, n_missions = 4).** All 5 conditions × 4 metrics tie (`§6.2`). A separate n_missions = 6 sweep gives jittery final AUC +0.012 (p = 0.044) and accuracy +0.035 (p = 0.035), clean a tiny H2 > H3 on round closure (`HERMES_Experiment4_Jittery_Methodology.md` §7.3). The document's own defensible claim: a **backhaul-robustness mechanism, not an accuracy driver**.
3. **The recorded L1 confirmation (matrix sweep C; `HERMES_Matrix_Results.md`).** C1 (n = 40, jittery, 120 s): H3 against H2 final AUC **+0.046** [+0.0207, +0.0734], p = 0.0016; accuracy +0.030, p = 0.0026; participation metrics ties. C2 (n_missions = 6): AUC +0.023, p = 0.0125. From the traces, the whole effect is **reachability**: at τ = 0.82, 28/40 vs 19/40 reach (McNemar p = 0.0117); at τ = 0.75, 39/40 vs 30/40 (p = 0.0039). Conditional on reaching τ, rounds are identical.
4. **The backhaul-loss-index bug and Amendment 5** (`HERMES_Scheduler_Freeze.md` §5e; `cluster.py:140-150`). `processes/cluster.py` passed `up.mission_round`, which `UpBundle` does not have, so every UP drew against `schedule[0]`. **Every `--l1-channel` run applied mission 1's loss to every mission**: H3 got the loss of its mission-1 best band for the whole trial, H2 its best-average band's mission-1 loss. The C1 comparison therefore measured the difference between two mission-1 picks, **not per-mission adaptation**. `backhaul_lost_rounds` was always empty, so round closure counted backhaul-dropped rounds as closed. Re-scored from the C1 traces, closure at k = 1 falls **0.831 → 0.675 for H2** and **0.838 → 0.813 for H3**; accuracy and yield re-score identically (Freeze §5e table; `HERMES_PreRerun_Checklist.md` header). The fix is in the tree (`cluster.py:1371`, `mission_schedule_index`). **The re-runs the checklist orders (C1, C2) are not in the repo:** `results/exp4_matrix/C_h2h3.csv` and `C2_nm6.csv` are dated 14 Aug (**no re-run found; unverified that none exists elsewhere**). H2 and H3 left Exp 5 by decision 8, so the L1 confirmation is a recorded Exp 4 result with the caveat above, not an Exp 5 one.
5. **F+L1 against F on the seconds backhaul (Study 5.14, batch 1; `results/exp5/scores/b1/s514.md`).** N = 6, one mule, jittery contact and network, 1 MB, age cap S = 2, 40 paired seeds, `--backhaul-model seconds` for both, Network AoU (lower is better):

   | Budget | F (`secF`) | F+L1 (`secFL1`) | Difference [95 % CI] | Holm p | Round closure k = 1 (F → F+L1) | Mean time to τ = 0.71 (F → F+L1) |
   |---|---|---|---|---|---|---|
   | knee, 150 s | 0.883 | 0.689 | 0.195 [0.126, 0.277] | 1.68e-4 | 0.7875 → 0.9875 | 206.3 s → 165.1 s |
   | stress, 75 s | 0.970 | 0.786 | 0.183 [0.128, 0.244] | 1.68e-4 | 0.800 → 0.969 | 213.7 s → 160.8 s |

   Both cells are **claims** in the scorer's sense (CI excludes 0 and Holm p < 0.05). Budgets from the argv (`results/exp5/b1/s514/n6k1_{knee,stress}__secF*.csv.argv.json`); round closure and time to τ from `s514_arms.csv`. **This is a model result:** the seconds backhaul at a fixed carrier loses about 16 % of uploads when jittery against about 1.7 % for the controller (§6.2), so the sign is set by the channel model, as §6.5 notes. It is a claim about FeRRy's coupling to a modelled backhaul, not about a radio. The `capS` cells (default `mission` backhaul) show F at 0.715 and 0.810 on the same metric, so the seconds backhaul at a fixed carrier is itself a harsher setting than the recorded flat 2 % loss.

### 9.6 Caveats in one place

- Every radio number is **simulated**: no AERPAW measurement gives R(b) or the interference model; I_b(t) is "a declared interference model, not a propagation claim" (`channel_model.py:555-556`).
- Interference is a **regular sinusoid** with one period for all classes, which a learner can fit with phase features (`pair_features.py`, `prev_sin`/`prev_cos`). Irregular interference is built only if 5.5 kept the learned score; it did not.
- The learned in-flight score was trained and evaluated on FerrySim cells where one stop often serves all six devices; the N = 6 cells "add little training signal" (Readiness).
- Study 5.4's conclusion is unknown. If F's gap over the best FB+ excludes zero only in a narrow region of the range–rate trade, the "reach is a decision" claim (C1) is a curve, not a number (the study is designed to read as a curve).

---

## 10. Open items

- [ ] **Study 5.4 and O1** (batch 2): F against FB+wide / medium / narrow, with `--narrow-range-ratio` ∈ {2, 3} and `--far-share` ∈ {0.25, 0.5, 0.75} beside the base cell, at the measured and 1 MB payloads, plus the O1 optimality gap at N = 6 on FerrySim cells (30 episodes, 8 workers) (`params.toml` `[s54]`; `experiments/analysis/o1_oracle.py`). The study decides whether C1 is a claim.
- [ ] **Study 5.15** (batch 3, blocked on `pilot3`): interference amplitude, path-loss exponent, shadowing σ and the seconds backhaul, with H1+L1 against H1 and F+L1; `harsher_amp_db` and `lossier_n_pl` are unset in `params.toml` until `exp5 report pilot3` reads them.
- [ ] **A plan-time (b) test** (γ across missions) is untested; the memo's rule says do not build a learned planner without it (overview §3.1).
- [ ] **A retune-time model for the backhaul switch cost** (`channel_utility.py:57-64`), and a decision whether to charge switching in the realised loss; today the gain is an upper bound.
- [ ] **Re-run C1 and C2** (H3 against H2, post-Amendment 5) **or** state in `HERMES_Matrix_Results.md` that the confirmation predates the fix; the file's text still reads "CONFIRMED" with no in-file caveat (§11).
- [ ] **The anchor's link-budget gap**: 60 m implies −11.3 dBm, about 11 dB more pessimistic than the Phase 3 research's own low-altitude budget; a platform measurement (Study 5.10, AERPAW) is the only fix (`contact_link.py:140-145`).
- [ ] **The one-revision screen** for the learned score on the clean cells, only if reviewers press (overview §7).
- [ ] **Delete or document `ChannelDDQN`**: retired from the plan but still imported by `mule_main.py:87` and `ferry.py:120` (`L1_STATE_DIM`); its state vector is built per stop only for logging.

---

## 11. Discrepancies noted while writing (docs against code or records)

1. **`FeRRy_Build_Plan.html`, 5.4 caveats:** "The sweep knobs (unit U10) are not built." They are: `narrow_range_ratio` (`contact_link.py:463`), `--far-share` (`driver.py:827`), and the launcher settings (`launch.py:973-989`); commit `20d14d65`. `HERMES_Configuration_Reference.md` §20.10 is current.
2. **`FeRRy_Build_Plan.html`, 5.15:** lists "Interference amplitude and noise as flags" and the F+L1 arm under "To build". Both exist (`--interference-amp-db`, `--interference-sigma-db`; `F+L1` in `driver.ADDENDUM_ARMS`, commit `fa6131a7`).
3. **`HERMES_Joint_RL_Methods.md` §1.1** says wide "carries up to 75.4 Mb/s" and narrow "up to 4.39 Mb/s". Those are CQI-15 peak rates that no class reaches at h = 25 m (`contact_link.py:178`: "no class reaches CQI 15"); the realised ceilings are 20.0 and 3.6 Mb/s at a stop directly below.
4. **`HERMES_Joint_RL_Methods.md` §1.4:** "each a claim (Holm p ≤ 0.008)" for F against H1, D1–D4 at the knee. The scorer gives D4's Holm p as 0.00829 (`s53.md`), so the bound is "≤ 0.0083".
5. **`HERMES_Matrix_Results.md`** reports the L1 confirmation as "CONFIRMED" with no in-file note that Amendment 5 changed what C1 measured; the caveat lives in `HERMES_PreRerun_Checklist.md` and the Freeze §5e. The doc is modified in the working tree (header count only).
6. **`HERMES_Experiment4_L1_RF_Layer.md` §7** points the RF-prior consumer at `build_target_queue`; its own correction box in §4 says the live path is `build_contact_queue`. Only the §7 row is stale.
7. **`rate_tier` against the link's floor** (`channel_utility.py:29-42` against `contact_link.py:233`): the utility's 0 dB threshold is not the −6.7 dB CQI-1 floor. Not a bug (the backhaul and the contact link are separate models), but a reader who assumes one rate table will be misled.

---

## 12. Sources

Code: `hermes/l1/{contact_link,channel_model,mission_clock,channel_utility,rf_prior,channel_ddqn}.py`, `hermes/scheduler/policies/{cross_heuristic,pair_slot}.py`, `hermes/scheduler/plan/{plan_search,types}.py`, `hermes/scheduler/fl_scheduler.py`, `hermes/mule/{ferry,mule_main}.py`, `hermes/processes/cluster.py`, `hermes/scheduler/selector/{pair_features,pair_q}.py`, `experiments/exp4/{channel,driver}.py`, `experiments/ferrysim/cells.py`, `experiments/analysis/{o1_oracle,traces_scorer}.py`, `scripts/exp5/{launch.py,params.toml}`.

Records: [Experiment 4 L1 layer](../HERMES_Experiment4_L1_RF_Layer.md) · [Configuration reference](../HERMES_Configuration_Reference.md) §17, §18.5, §20.4–20.11 · [FeRRy build plan](../FeRRy_Build_Plan.html) (D1–D2, Phases 3–5, Studies 5.4, 5.14, 5.15) · [Layer redefinition and RL decision memo](../architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md) · [Scheduler Freeze](../HERMES_Scheduler_Freeze.md) §5e and decision 8 · [Pre-rerun checklist](../HERMES_PreRerun_Checklist.md) · [Matrix results](../HERMES_Matrix_Results.md) · [Experiment 5 readiness](../Experiment_5_Readiness.md) · [Experiment 5 RL calibration findings](../Experiment_5_RL_Calibration_Findings.md) · `results/exp5/scores/b1/{index,s53,s514}.md` and `s514_arms.csv`.

*Numbers are copied from those sources as of 7 Oct 2026. The re-evaluations marked "checked here" (ranges, implied EIRP, rates, dwell, the 36-column schema width) were run against the working tree; the §9.2 band shares are my aggregation of the scored CSVs.*
