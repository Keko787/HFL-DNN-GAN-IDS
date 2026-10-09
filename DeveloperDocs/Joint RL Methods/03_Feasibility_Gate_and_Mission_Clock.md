# The feasibility gate and the mission clock

*7 Oct 2026. Part of the series that starts with [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md); the companion for who is served is [02_Job_Target_Selection.md](02_Job_Target_Selection.md). A reading document: it restates the code and the records, decides nothing, and says "unverified" where a number could not be checked against its source. File:line references were read against the working tree on 7 Oct 2026. Numbers are copied from the named records; none were re-run. Nothing here changes a recorded sweep.*

---

## 0. The short version

Every decision that spends time in HERMES/FeRRy is checked by **one predicate**, `FeasibilityModel.admit` (`hermes/scheduler/stages/s3b_feasibility.py:527`), on **one clock**, `MissionClock` (`hermes/l1/mission_clock.py:144`). Before Phase 3 the same single-contact check was copied into four walks (S3b, the D-arm budget walk, FedCS, the mule's in-flight check) and priced a contact as `transit + 1 s`; now they are all folds over the predicate, and a contact costs what its bytes cost at the rate the band gives.

Four things to carry away.

1. **Hard gates run before anything learned** (design principle 12; the Freeze's architectural guarantee). The gate is a *removal*, not an ordering: it sits before S3.5 so a learned selector cannot resurrect what it drops, and before the flight slot's score so a learned pair can only be one the predicate admits.
2. **The planner prices the mean SNR; the flight does not.** Dwell is convex in SNR, so realised missions run systematically longer than planned: +0.8 % on wide at 1 MB, +7.1 % on wide at 10 MB, +19.2 % on narrow at 1 MB (build plan, Phase 3 pilot notes). The gate re-checks at every *departure* but not during a contact, so a budget can still be overrun at the stop where Pass 1 ends. In batch 1 the F-family and D4 overrun the budget in 24–85 % of missions at the stress budget (§9.2).
3. **The decision memo's three cuts are closed in code**; what the closing bought is a different matter. Cut 1 (a contact-band decision) is tested by Study 5.4, not run. Cut 2 (band on the arrival clock) was tested by Study 5.5 and came back flat. Cut 3 (rate-dependent dwell, re-plan instead of abort) has no isolating study; its tests are the 1,238 Phase 3 tests and the goldens (§8).
4. **The session TTL and the budget knee are pilots, not results.** The TTL is a wall-clock transport timer, and batch 1's sensitivity cells at TTL × 0.75 and × 1.5 reproduce the × 1 cell to the digit (F 194 s, FX 178 s, H1 282 s). The knee (150, 180, 240, 262 s at N = 6, 12, 18, 24) is where mean update yield reaches 95 % of the grid's best, and at the knee the pilot report's served share is 53–58 %, so the budget still binds at the "plateau" (§7).

---

## 1. Decisions made in this part of the system

| # | Decision | Clock | Inputs | Output | Who decides | Learned? | Code path |
|---|---|---|---|---|---|---|---|
| G1 | May this stop be served next? | Plan, flight | `FlightState`, stop, rule, budget end | `Verdict` (ok, reason, times, next state) | The predicate | No | `s3b_feasibility.py:527` |
| G2 | Which stops survive admission? | Plan | EDF-ordered stops, start state | Kept + dropped by reason | S3b walk (`filter_feasible`) | No | `:758` |
| G3 | Re-issue a stop that fails whole with the members that fit | Plan only | Stop, member order, state | Reduced stop(s) | `fold_subsets` (`member_admission="subset"`) | No | `:874` |
| G4 | Is the order about to be flown feasible? | Plan (pre-flight) | Kept queue in the arm's order | Keep, repair, or trim | `_validate_order` | No | `fl_scheduler.py:1038` |
| G5 | Does the rest of the pass still fit? | Flight, every departure | Remainder, state at the observed clock | ok / rejected stops | `fold_remainder` | No | `fl_scheduler.py:1172`; `mule_main.py:2080` |
| G6 | What to do when it does not | Flight | Remainder, arm, `replan_fallback` | New route + dropped (final) | `replan_route` / `_trim_plan` | No | `routing/replan.py:146`; `fl_scheduler.py:1204`, `:1808` |
| G7 | Abort the tail (legacy / `abort`) | Flight | First remaining stop | fly / break | `_remaining_is_feasible` | No | `mule_main.py:3038` |
| G8 | Which pairs does the flight slot's mask admit? | Flight, each Pass-1 arrival | (band, next stop) pairs, observed dwell | Mask bit per pair | `fits_after_service` | No | `fl_scheduler.py:1966` |
| G9 | May a beacon-offered stop join the route? | Flight, Pass-1 departure | Offer, remainder | Insert or refuse | `_ferry_try_insert` (inert: no source) | No | `mule_main.py:2834` |
| G10 | How much of Pass 2 fits? | Plan (Pass 2) | Nearest-first queue, budget from `t2` | fly / skip | `_budget_pass_2` / `fold` under `RULE_BUDGET` | No | `mule_main.py:3118` |
| G11 | What does each step cost in time and energy? | Both | Distance, bytes, rate, powers | Seconds charged per kind | `MissionClock` + `FlightModel` | No | `l1/mission_clock.py:179` |
| G12 | Where does each mission's budget start? | Takeoff | Clock | Budget stamp | `start_mission()` | No | `fl_scheduler.py:526` |
| G13 | Session TTL, budget knee, τ, S\* | Before the campaign | Pilot sweeps | `params.toml` values | Pre-registered rules | No | `scripts/exp5/params.toml:101-122` |

---

## 2. The mission clock

**Why it exists.** The mule used one wall clock for three jobs: mission time (planning `now`, Deadline(j), the S3b budget stamp, the in-flight check), transport timers (session TTLs, socket, dock and bootstrap waits) and measurement (event `ts`, `duration_s`). It never flew: its pose jumped from stop to stop, so the in-flight budget check counted host compute and TTL waits, not flight and airtime (`mission_clock.py:1-31`). Phase 3 puts mission time on a clock charged with the physics of each step and leaves the other two on the wall clock.

**What charges it** (`LEDGER_KINDS`, `:97-105`; one entry per kind, kept per mission):

| Kind | Charged by | Amount | Energy |
|---|---|---|---|
| `transit` | Supervisor, before each stop | `|pose − stop| / v` | P_move |
| `dwell` | Host's commit, once per contact | Σ over answered targets of `8·bytes / rate(band, SNR)`, each at its own session start; no band: `session_time_s` | P_hover |
| `listen` | Host's commit | `listen_s` once per contact that misses any expected reply (a dropped uplink looks like silence) | P_hover |
| `return` | Supervisor, after the last stop of a pass (also after an abort or a re-plan to nothing) | `|pose − dock| / v` | P_move |
| `upload` | Supervisor, at the inter-pass dock | `8·UP bytes / rate(wide, backhaul SNR)` | ground, free |
| `turnaround` | Supervisor, once per mission after Pass 1's return | 30 s | ground, free |
| `dock_wait` | Supervisor, on the DOWN | `advance_to` the cluster's simulated time (Lamport sync across mules) | ground, free |

**Invariants** (`:33-48`): *monotone* (a negative charge raises; `advance_to` is a max); *finite* (NaN or ∞ raises, so a rate of 0 below the SNR floor must become a finite decision first: an unreachable member costs no dwell, a backhaul below the floor is a lost upload charged at the floor rate, `ferry.py` `upload_cap_s`); *never paced* (a charge is an addition; nothing sleeps, so a 5 km leg cannot open finding P-02's 30 s silence); *absolute* (never reset; Deadline(j) and `last_clean_ts` are absolute stamps compared across missions, only the ledger resets, at takeoff).

**The epoch.** Starts at `SIM_EPOCH_S` = 1e6 s, above 0.0 because scheduler state uses 0.0 for "never" (`idle_time_ref_ts`, `last_clean_ts`, `last_beacon_ts`), and the clock refuses to reach `SIM_CEILING_S` = 1e9 s (`:84`, `:89`), which wall time passed in 2001 — so any stamp's value says which clock made it, and `advance_to` refuses a wall stamp. Mixing the two would clamp every idle term to 0 or make every served device overdue.

**Flight constants** (`FlightModel`, `:448-512`): cruise 5 m/s (Freeze D2: "still awaits a platform citation"), turnaround 30 s (Exp 3's `dock_time_s`), listen 1 s, one dock shared by every mule (queueing not modelled). `leg_s` uses exactly the arithmetic of `FeasibilityModel.cost`, "bit for bit", so the clock charges the leg the planner predicted.

**Energy is a pure function of the ledger** (`EnergyModel.energy_j`, `:427`): `E = P_move·(transit + return) + P_hover·(dwell + listen)`. SIMULATED, not measured: Zeng–Xu–Zhang 2019 eq. (6) with the Table I quadrotor, P_move = P(5 m/s) = 143.6 W, P_hover = P(0) = 168.5 W, 28.7 J/m at 5 m/s; 5 m/s sits below the model's minimum-power speed (about 10.2 m/s, 126 W), at 85 % of hover power. Climb, ground time and the radio are not charged. **The capacity clause is off by default (`capacity_j=None`) and no pilot sets one**; batch 1's job arguments carry no capacity flag.

**What stays on the wall clock.** `mission_duration_s_mean` and `wall_s_to_*` measure host compute and TTL waits; the sim columns (`sim_s_to_τ`, `sim_mission_duration_s_mean`) read the clock (Config Reference §17.7). A trace whose clocks disagree is refused (`ClockDomainError`).

**The deadline's time unit.** The recorded Φ constants (−5 s / +10 s, 5 s floor, 60 s Φ₀) were set against missions of about 10 s; a simulated two-pass mission takes minutes. `--deadline-time-scale t_nom` multiplies them by T_nom / 10 s (doc 02 §3.3). T_nom is the median over 20 reference layouts of one nominal mission (Pass 1 + turnaround + Pass 2, `nominal_mission_period_s`, `fl_scheduler.py:2135`), the same for every arm.

---

## 3. The predicate

### 3.1 Three rules
| Rule | Who | Clauses |
|---|---|---|
| `RULE_DEADLINE_BUDGET` ("deadline+budget") | Our arms (S3b, F family); Pass-1 in-flight check | The contact's own deadline, the budget, and (with a capacity) energy |
| `RULE_BUDGET` ("budget") | D1–D3 and D5 in flight and in their walks; Pass 2 for every arm | Budget only. The per-device deadline is S3b's rule, not theirs (Amendment 8) |
| `RULE_NONE` ("none") | D4 | No gate: always admitted, times still reported so the overrun can be measured |

With `budget_end = None` every clause is off, the energy clause included: **no budget, no gate** (the opt-in contract, which keeps every recorded run reproducible). `in_flight_rule` (`fl_scheduler.py:1145`) chooses the rule: `DELIVER` pass → budget; a whole-scheduler policy → what it declares (`budget` or `none`); otherwise deadline+budget.

### 3.2 Legacy arithmetic (`ferry` is None; `:593-606`)
```
transit = |pose − stop| / cruise_speed_m_s            (5.0, frozen)       finish = clock + transit + session_time_s (1.0, frozen)
overdue  iff  deadline rule, not protected, and  clock + transit > deadline_ts      (compares ARRIVAL)
budget   iff  finish > budget_end
next state = (stop, finish, energy, deliver_by)
```
No return leg, no upload, no energy. "Pinned by `tests/golden/test_golden_feasibility.py`: every walk returns exactly what it returned at 9147211."

### 3.3 Ferry arithmetic (`:608-635`)
```
arrival = clock + transit                finish = arrival + dwell                home = finish + return + upload   (upload: Pass 1 only)
dwell   = Σ over members within range_m and above the SNR floor of member_dwell(d, pass, δ_obs)
          (members' times ADD: one shared channel, spec Q9; no band: session_time_s once)
clauses, first failure is the reason:
 1 own deadline   (deadline rule, not protected)   collection:  finish ≤ Deadline(j)     [default]
                                                   delivery_per_stop, delivery: home ≤ Deadline(j)       → "overdue"
 2 on board       (deadline rule, delivery only)   home ≤ deliver_by (earliest deadline of updates already collected)  → "delivery"
 3 budget         (deadline and budget rules)      home ≤ budget_end                                       → "budget"
 4 energy         (only with a capacity)           E + P_move·(transit + return) + P_hover·dwell ≤ capacity → "energy"
next state = (stop, finish, E + P_move·transit + P_hover·dwell, deliver_by′)
```
Notes: (a) the two deadline clauses precede the budget, so a contact failing both is reported `overdue`; (b) `protected` (every member capped, doc 02 §5) exempts only the stop's **own** deadline — it is still held to the updates on board, and its own deadline does not lower `deliver_by`; (c) δ_obs, the observed-rate adjustment, is 0 in every call (critic C1): the planner prices the mean SNR; (d) `listen` is **not** priced — the clock charges it only when a reply is missing, and the next departure check sees it; (e) the legacy `overdue` test compares arrival and the ferry default compares finish, so the same stop can pass in one mode and fail in the other.

### 3.4 What the predicate does not know
Dwell uses the *mean* SNR, not the seeded phase (that would be an oracle), so a realised mission is longer than planned: the figures in §0. The re-check happens only at departures, so a contact that runs long (a silent member's listen window, a noisy band) at the stop where Pass 1 ends is never re-checked; the updates are on board and the only way left is home. The mission records the miss (`MissionRunResult.delivery_overrun_s`, and `sim_budget_overrun_*`).

### 3.5 Who calls it
| Caller | Rule | How |
|---|---|---|
| S3b admission, before takeoff | deadline+budget | `filter_feasible` → `fold(skip=True)` (`:758`) |
| Plan search and its guard fold | deadline+budget, exempt stops protected | `build_ferry_plan` (`fl_scheduler.py:1641`) |
| Pre-flight order check | per arm | `_validate_order` → `replan_remainder` |
| Departure check | `in_flight_rule` | `fold_remainder` (`skip=False`: does the route pass *as flown*) |
| Re-plan admission and its guard fold | per arm | `replan_route` (`routing/replan.py:146`) |
| D1–D3, D5 walks | budget | `greedy_budget_walk` → `fold(skip=True)` |
| Pass 2 | budget | `_budget_pass_2`; re-plan admission `RULE_BUDGET` |
| Pair mask (F family, plan mode) | in-flight rule | `fits_after_service` |
| E3's admissible set | budget | `mule_main.py:2531` |
| Beacon insert | in-flight rule | whole edited remainder must pass without skipping |
| T_nom | none | `fold(RULE_NONE)` for the nominal period |
| Cap servability | deadline+budget, protected | `servable_alone` (`s3d_age_cap.py:385`) |

The scheduler must not import `hermes.l1`: every piece of physics reaches the module as a float or a callable the mule builds (`FerryPhysics`, `:286`).

---

## 4. Member subsets: whole versus subset

`member_admission="whole"` admits a stop with all its members or none; it is the recorded rule and keeps the **narrow-band cliff**: at narrow (less so medium) with a declared payload the one field-wide contact is admitted whole, so under a budget below its predicted home time every gated arm flies empty missions (0 or N; finding E2E1-01). `"subset"` (F family default; opt-in for H1–H3 and D1–D3, D5; never D4) re-issues a stop that fails whole with the members that still fit, **before takeoff only** — the in-flight check and re-plan keep or drop contacts whole (`routing/replan.py`'s identity check) except plan mode's own trim.

`fold_subsets` (`:874`) tries each stop whole; refused, it walks members in the arm's order and admits each if the reduced stop still passes (skip, not stop). **One pass is final** because every clause is monotone in the member set: dwell is a sum of non-negative member times, the stop's deadline is its members' minimum, arrival, return and upload do not depend on members, energy grows with dwell. Member order for our arms (`_filter_subsets`, `:953`): own deadline first (EDF inside the contact, so service rotates), then predicted dwell, then device id; with `miss_priority`, miss streak leads. D arms rank members by their own key (doc 02 §7). Members left out join the drop list of the clause that refused them.

---

## 5. The departure check, the re-plan, and what "trim" and "reorder" mean

### 5.1 Responses (`MuleConfig.in_flight_response`; code default `abort`, batch 1 flies `replan` with `--replan-fallback trim`)

| | `abort` | `replan` |
|---|---|---|
| Check | Next stop only, with its return and upload (`_ferry_departure`, `mule_main.py:2116`; legacy `_remaining_is_feasible`, `:3038`) | The whole remainder folded as flown (`fold_remainder`) |
| Failure | Abandon the tail; widen it; close and dock | `replan_remainder` repairs it; dropped stops are **final for the mission** |
| Legacy path | `_remaining_is_feasible` re-runs S3b on `remaining[:1]` from the mule's *current* pose on the wall clock; D4 (`none`) always flies on; baselines get `greedy_budget_walk` on that one stop | n/a |

`_remaining_is_feasible` is the seed Amendment 1 added; it can foresee running out of **time**, a deterministic function of clock and geometry, never a random link failure ("a property of the model, not a gap").

### 5.2 The re-plan algorithm (`replan_route`, `routing/replan.py:146`; design §3.4 with critic C3)
1. If the current order passes the fold without skipping, keep it.
2. **Admission.** Protected stops first, in their current relative order; then the arm's own admission from where that prefix ends: S3b `filter_feasible` for our arms (miss priority if on, else EDF); the policy's own `admit_and_order` for D1–D3 and D5 (no 2-OPT, since 2-OPT would give a baseline a FeRRy mechanism, critic B5); a nearest-first budget walk for Pass 2; D4 never re-plans.
3. **Order.** Keep the arm's relative order over the admitted stops if it passes. Otherwise `reorder`: 2-OPT to the dock (`two_opt.order_contacts(admitted, pose, end=dock)`), then the admission order, which passes by construction. Or `trim`: keep the arm's order and leave out the stops it cannot serve in that order (a skip fold; protected stops keep their place in front).
4. Return `ReplanResult(route, dropped[(stop, reason)], order_used)` with `order_used` one of `current, arm, arm_trimmed, two_opt, admission, none`.

**Where order cannot help.** Before takeoff S3b has just admitted the whole queue from the same state, so the arm's own order over the admitted set is the order that just failed. Under `reorder` the repaired route is 2-OPT's or EDF's, both functions of the admitted *set*, so H1, H2 and H3 fly the same route whenever the check fires and the check never drops a stop; under `trim` each arm keeps its order at the price of serving fewer. The pilot plan accepted on 29 Sep flies `trim`, so each arm keeps its order "as the D arms do". No recorded run compares the two; the idea of a pilot that would is "on record, not in the plan".

**Plan mode** (F family) never calls `replan_route`: `_trim_plan` (`fl_scheduler.py:1808`) keeps the remainder if it passes as flown with exempt stops protected, else flies priority stops (any member capped) first and the rest after, each in flight order, and **never re-orders** — re-ordering belongs to the flight slot. Under `subset` it is `trim_members`: priority stops shed uncapped members first and each stop is kept whole, reduced, or dropped; under `whole` `_trim_whole` (`:1873`) keeps whole stops only, which retains the narrow-band cliff on purpose. `reorder` and `pass_2_budget` are refused in plan mode (`_bind_plan`, `:417`).

### 5.3 What happens to dropped devices
In Pass 1 each dropped device is fed a synthetic TIMEOUT (`answered=False, synthetic=True`) at the simulated drop time, so Φ widens (`_widen_abandoned`, `mule_main.py:3149`); in Pass 2 it is recorded as a skipped delivery. A stop refused by the on-board clause counts like a budget drop, since neither device was late itself.

### 5.4 2-OPT (`routing/two_opt.py`)
Cheapest-insertion start, first-improvement 2-OPT over every segment reversal, three shapes (closed tour; open path; **fixed-end path** to the dock, which is what the re-plan needs). A 2-OPT move must improve by more than 1e-9 m; passes capped at 1,000; optional seeded restarts; positions are plain tuples and the module imports no ML stack. Ties break by index; `order_contacts` sorts by `(position, devices)` first so the route does not depend on listing order. Not optimal, "and not claimed to be" (checked against brute force for n ≤ 7).

---

## 6. Hard gates before anything learned (Freeze principle 12)

Design principle 12: the selector "runs after the deterministic gates and after the deadline math. It cannot promote a gated-out device, cannot reorder buckets, and cannot override a deadline. Hard rules stay hard; learned rules stay inside one explicit sub-stage."

| Mechanism | What it guarantees | What it does not |
|---|---|---|
| **Position.** S3b is before S3.5 and before the pair score | A learned ranking only ever sees the survivors; the gate would also be skipped for single-candidate buckets if it sat inside the selector (`s3b_feasibility.py:11-17`) | Nothing checks it at runtime — it is call order |
| **Scope guard.** `assert_candidates_admitted` (`selector/scope_guard.py:39`) and `assert_pairs_admitted` (`:57`) | Every member of every candidate (and every pair's band and next stop) is in the admitted set; a violation is a wiring bug and fails loudly | The admitted set the legacy call passes is `bucketed`, i.e. S1/S3-admitted (`fl_scheduler.py:1018`), **not** post-S3b. A contact S3b dropped is still in that set, so the guard cannot catch it re-appearing; position (row 1) is what protects S3b |
| **Pass guard.** `_enforce_collect_pass` (`target_selector_rl.py:52`) | The selector raises in Pass 2 (universal delivery) | — |
| **Pair mask.** `fits_after_service` (`fl_scheduler.py:1966`) | A pair is admitted only if the band covers the plan, the next stop is in the plan, and the whole rest of the flight still fits at the observed SNR; if none is admitted the mule flies FX's pair and logs `mask_empty` | The served stop's own deadline is not tested (the mule is there whichever pair it picks) |
| **Plan guard fold.** `build_ferry_plan` (`:1641`) | The committed route passes the predicate as flown; a failure raises and commits nothing | — |
| **Plan mode owns admission.** `_bind_plan` (`:456`) refuses a `target_selector`, a `contact_policy`, `reorder`, and (with a cap) `abort` | No learned selector or baseline can own admission in the F family | — |
| **E3's safety controller** (`mule_main.py:2531`) | E3 names the next stop only among those S3b's budget rule admits (Chen's own gate-then-learn pattern) | E3 has no deadline, plan or cap |

The principle held through every experiment because learning was always placed *inside* an already-admitted set. It is also why the learned pair score's null (overview §2.6) is a null about ranking and not about feasibility.

---

## 7. The pilots: session TTL, budget knee, S\*, τ

All run on the first host (8 physical cores, 16 logical, 95 GB, Windows 10) except S\*, which is planning-level and ran on the second (12 physical, 20 logical, 64 GB, Windows 11). Pilots are fixed before the studies by pre-registered rules in `scripts/exp5/params.toml`.

### 7.1 Session TTL (`ttl`, 5 Oct 2026; `results/exp5/ttl/report.txt`)
Rule: **2 × the p95 of the real model's `train_offline` time**, rounded up, measured with N devices training at once.

| N | Fit p50 | Fit p95 | Max | Workers | TTL |
|---|---|---|---|---|---|
| 6 | 1.83 s | 17.99 s | 18.23 s | 36 | **36 s** |
| 12 | 1.46 s | 16.74 s | 17.64 s | 36 | **34 s** |
| 18 | 1.21 s | 16.64 s | 17.12 s | 36 | **34 s** |
| 24 | 0.69 s | 11.14 s | 12.40 s | 24 | **23 s** |

The p95 is the first fit (TensorFlow's tracing, about 17 s with 36 processes at once); later fits take about 1.6 s. The TTL is a wall-clock transport timer (the recorded 3 s default is far below it), so it never enters simulated time. **Sensitivity (`sens`, 6 Oct):** F, FX and H1 at N = 6, knee, TTL 27, 36 and 54 s: means 194.048, 178.059 and 281.838 s in all three, identical to the digit (`results/exp5/scores/sens/ttl_arms.csv`; the × 1 cell is batch 1's own). Identical is the expected outcome while no session times out and the clock is never paced; it shows the TTL is not a hidden parameter in this range, and says nothing about a TTL below the fit time. **Caveat:** the TTLs are fits on the first host's CPU, and the campaign's later stages ran on a faster second host, so the timeouts are probably more lenient than measured. A guess, not a measurement (Readiness, "What could still go wrong").

### 7.2 Budget knee and stress (`knee`, 5 Oct; 600 trials, jittery contact channel, 1 MB payload, `results/exp5/knee/report.txt`)
Rule: the knee is the **first budget whose mean update yield reaches 95 % of the grid's best**; stress is **half the knee, rounded to 5 s**.

| N | Best yield (grid) | Knee | Yield at knee | Served share at knee | Stress |
|---|---|---|---|---|---|
| 6 | 3.462 (150 s) | **150 s** | 3.462 | 57.7 % | 75 s |
| 12 | 6.650 (240 s) | **180 s** | 6.338 | 52.8 % | 90 s |
| 18 | 10.188 (300 s) | **240 s** | 10.062 | 55.9 % | 120 s |
| 24 | 13.188 (350 s) | **262 s** | 13.137 | 54.7 % | 130 s |

At the measured payload (18.8 KB), N = 6: knee 120 s (yield 3.312 of 3.450), stress 60 s. N = 6's knee is the **grid's largest budget**, so the report warned the plateau may lie beyond it; it was accepted as is (decided 5 Oct) and written by hand because `--apply` refuses on a warning (yield 3.263 at 120 s, 3.462 at 150 s). At N = 18 and 24 the yield is flat from 300 s and 350 s up.

**τ** is the largest τ (steps of 0.01) that at least 80 % of H1's trials reach at each N's knee, the smallest over N: 0.82 at N = 6, 0.72 at N = 12, 0.71 at N = 18 and 24, so **τ = 0.71**; 0.82 is kept as a second τ with its reach rate reported. Accuracy levels off near 0.71 at N = 18 and 24 at every budget (0 of 20 trials reach 0.82 there). A consequence recorded in Readiness: at N = 6, where 80 % of trials reach 0.82, most trials reach 0.71 early, so time to τ separates the arms less there.

**What the knee does not mean.** It is a yield plateau, not "the budget stops binding": served share is 53–58 % at every knee, so the gate still removes close to half the device-missions. The N = 6 stress budget (75 s) is not a grid point; the pilot's H1 sweep served 28.1 % at 60 s and 44.6 % at 90 s.

### 7.3 S\* (`sstar`, 5 Oct; `results/exp5/sstar/report.txt`)
S = 2 at every N, at budgets (150, 75), (180, 90), (240, 120), (262, 130). One mission covers 90 % of layouts at each knee and two at each stress budget (N ≥ 12 by the tool's greedy bound); the floor of 2 is what binds at the knee. Doc 02 §5 has the rule and the (null) N = 6 ablation.

### 7.4 The re-pin
FerrySim's training cells flew placeholder budgets until the knee and S\* pilots measured the stack's: N = 6 cells re-pinned to (75, 150) s and N = 12 to (90, 180) s, caps 2, Study 5.6's lags 27 and 34 s. Done; its own check passed narrowly (at 180 s the ratio bound is 1.48 against 1.5).

---

## 8. The decision memo's three cuts, and how each was closed

The memo (`rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md` §1.1) named three places where the two-clock coupling was severed in code. The build closed each; the **test** is what the memo said the closing should be tested by.

| Cut | Was (memo, with its own correction) | Closed by | Commit (Freeze §5g) | The test the memo asked for | Status of that test |
|---|---|---|---|---|---|
| **1 — no contact-band decision** (plan time) | L1 selected a band for the *backhaul* only; the device link had no band and `rf_range_m` was a fixed 60 m | `l1/contact_link.py`: band classes `wide` 20 MHz, `medium` 5 MHz, `narrow` 1.4 MHz on one 3.32 GHz carrier with a range–rate model; `MuleConfig.contact_band`; S3a at R_planar(b); numbered solicits gated by range and the SNR floor; plan mode commits b̄ (`build_ferry_plan`) | Phase 3 `7c7a269`; plan mode Phase 4 `48deb74` | "Cut the band→range edge and see whether the result moves": F against FB+ pinned to each class, Study 5.4, with O1 for the optimality gap | **Not run** (batch 2) |
| **2 — band on the wrong clock** (flight time) | L1 re-selected per *mission*; the prior reaching the selector was one mission-mean scalar | Per-arrival band: FX's fastest-covering-class rule (`cross_heuristic.py:168`) and FQ's masked (band, next stop) score (`pair_slot.py`, 36 columns of `pair_v1` with per-class SNR and a phase block); the causal RF prior replaces the trial mean | Phase 4 `48deb74`; Phase 5 `9c91818` | Sweep γ across stops within one sortie (property (b), flight clock) | **Run, flat** (Study 5.5): the learned score matched FX (−0.0774 vs −0.0786) and trailed `greedy_1` (−0.0716) |
| **3 — the return path** (flight → plan) | Dwell was a constant `session_time = 1.0 s`; the only upward edge, `_remaining_is_feasible()`, could only abort | Rate-dependent dwell `8·bytes/rate(b, SNR)` charged on the simulated clock; return leg and upload in the budget clause; `replan_remainder` repairs instead of aborting; pre-flight order check | Phase 3 `7c7a269` | None isolating; "Q2 and cut 3 are one piece of work; either alone leaves the loop open" | Covered by tests (1,238 new Phase 3 tests, the 144 goldens, the 2,798-test baseline match on 29 Sep); **no study compares abort with replan** |

**The memo's own correction to cut 3** matters here: S3b *did* model travel; what was constant was **dwell**. That is the exact constant (`session_time`) that blocked rate from feeding back into feasibility, and it is an uncited platform placeholder besides (Freeze D2: "needs a platform citation before publication").

---

## 9. Evidence

### 9.1 The gate, measured in Exp 4 (legacy model; L2 record §4.2; N = 6, `rf_range` 60 m, field radius 100 m, cruise 5 m/s, 20 layouts)
- **Deadline floor:** about 34 % of contacts (26/76) cannot be reached before their own deadline at 5 m/s at *any* budget.
- **Budget knee ≈ 60 s:** below it budget drops appear (46 % at 50 s, 58 % at 40 s, 72 % at 30 s, 93 % at 5 s).
- **End to end** (H1, jittery, 5 shards, 100/100 valid, paired by seed): gate off 0.767 mission completion; 120 s (deadline only) 0.542 (−0.225); 60 s 0.483; 30 s 0.308; 15 s 0.283. "The deadline was never slack in Experiment 4": enforcement reveals a constraint that was always there.
- All of this predates Amendments 5, 6 and 8; budgeted cells ran under the stale stamp and are due for re-run before citing.

### 9.2 Gate activity in batch 1 (N = 6, one mule, trial means of the scorer's `sim_*` columns, 20 trials; per trial of 4 missions)

| Arm | Budget | Replans | Aborts | Empty missions | Budget overrun rate (share of missions) | Mean overrun |
|---|---|---|---|---|---|---|
| H1 | knee 150 s | 0.05 | 0 | 0.05 | 0.03 | 0.02 s |
| H1 | stress 75 s | 0.10 | 0 | 0.45 | 0.01 | 0.04 s |
| F (`capS`) | knee | 0.00 | 0 | 0.05 | 0.05 | 2.04 s |
| F (`capS`) | stress | 0.05 | 0 | 0.10 | 0.34 | 11.6 s |
| FX | stress | 0.10 | 0 | 0.10 | 0.24 | 4.86 s |
| D4 | stress | 0.00 | 0 | 0.05 | **0.85** | 18.96 s |

Reading: no abort fires in these cells (all arms run `replan`); the arms whose plan fills the budget (F, FX) overrun it more than the arms the gate leaves slack, which is the mean-SNR pricing and the unchecked last-stop dwell (§3.4); D4's visit-all tour, with no gate, overruns in most missions, which is the attribution the D4 row was built to show. Empty missions at stress (0.45 per trial for H1) are the narrow cliff's cousin: a field-wide whole stop that does not fit.

### 9.3 FX's mask
FX's own pair overruns the budget at about 19 of 71 last-stop arrivals at N = 6 and the mule then flies FX's pair and logs `mask_empty` (overview §2.3; that count is from the calibration findings and was not re-derived here).

---

## 10. Layer interfaces

| Edge | What crosses | Where |
|---|---|---|
| **L1 → L2** | The cost: band class → range R_planar(b) → S3a's radius; rate(b, SNR) → dwell → the clock → feasibility. `FerryPhysics.member_dwell_s` is a callable the mule builds from the contact link; `upload_s` is the predicted Pass-1 upload on the wide class at the held carrier's mean SNR | `s3b_feasibility.py:286`; `l1/contact_link.py` |
| **L2 → L1** | The stop and the arrival time fix the phase the next observation lands in; plan mode commits b̄, and the mule actuates it (`fx.set_band`, `mule_main.py:1628`) | `mule_main.py` |
| **L3 → L2** | The age cap exempts all-capped stops from their own deadline clause; `deliver_by` carries the earliest deadline of updates on board (route-level `delivery` only); the merge cutoff a_max reads the same Φ the deadline used | `s3d_age_cap.py`; `s3_deadline.py:404` |
| **L2 → L3** | Drops widen Φ through synthetic TIMEOUTs; `close_plan(flown, merged)` records visited and merged sets and the close-time cap violations | `fl_scheduler.py:1937` |
| **Mule ↔ cluster** | The clock's `dock_wait` Lamport sync to the cluster's simulated time; the upload charge; the cluster orders uploads by simulated time when several mules share it (`SimOrderGate`) | `mission_clock.py:207`; Freeze §5g |

---

## 11. Failure modes and recorded defects

| What | Effect | Status |
|---|---|---|
| **Stale budget stamp** (Amendment 6) | Stamped only on a DOWN; an empty mission skips the dock, so the next planned against the old stamp; 72–81 % of `b60` missions were empty (H1 116/160, D1 126/160, D2 129/160); one D1 trial's budget fell from 60 s to ~39 s by mission 4 | Fixed: `start_mission()` at every mission start, after the ledger reset in ferry mode |
| Abort-only edge | The mule kept flying doomed queues, burning budget and delaying delivery | `replan` (Phase 3) |
| In-flight deadline check on baselines (Amendment 8) | MAX-AoI puts overdue devices first; the check refused them and aborted | `in_flight_check` per policy |
| Flown order never checked | Probe P1: a 70 s walk flown as 150 s | Pre-flight order check under `replan` |
| Mean-SNR pricing | Realised missions longer than planned (§0); budgets bind in flight more on narrow bands and large payloads | Declared; δ_obs = 0 in every call |
| Last-stop dwell unchecked | Budget (and, under `delivery`, `deliver_by`) can be overrun where Pass 1 ends | Recorded per mission (`delivery_overrun_s`, `sim_budget_overrun_*`); not corrected |
| Narrow-band 0-or-N cliff (E2E1-01) | Empty missions under a budget below the field-wide contact's home time | `subset` admission; `whole` keeps it on purpose (F-`whole`, R5) |
| Deadline unit 1.0 on the simulated clock | A 14-mission narrow probe: the field-wide contact found overdue at every later takeoff, "by 20 s more each mission" | `--deadline-time-scale t_nom` |
| FedCS "skip = stop" and "no return leg" | Hold only for the legacy model | Documented; the ferry predicate tests the time home |
| Mixed stops (critic B2) | Protecting a stop whole would let its uncapped members miss deadlines | A mixed stop carries its uncapped members' earliest deadline |
| Which drops are widened | The wall-clock path (`_run_two_pass_mission`) widens `dropped_overdue` and `dropped_budget`; the mission-clock path (`_run_ferry_mission`) widens those plus `dropped_energy`, `dropped_delivery` and, in plan mode, `dropped_plan` (`mule_main.py:1651-1672`) | Not a defect: energy and delivery drops exist only on the mission clock. The `plan` drops are left out of S3c's planned count |
| Session-TTL host | Fits measured on another CPU | Open (§12) |
| Energy clause | Declared simulated; capacity off in every pilot | Open |

---

## 12. Open items

- [ ] **Study 5.4 and O1** — the cut-1 test; batch 2 (about 50 h, ready).
- [ ] **A trim-versus-reorder comparison** — "on record, not in the plan"; only if reviewers ask what the replan response costs in coverage.
- [ ] **An abort-versus-replan study** — nothing isolates cut 3; the evidence is tests, not outcomes.
- [ ] **Re-measure the TTL and knee on the second host** — about 4 h (the knee took 4 h, the TTL pilot 5 min on the first host); then S\* again and the re-pin.
- [ ] **A platform citation for cruise speed 5 m/s and the 1 s session** (Freeze D2); the 30 s turnaround is Exp 3's `dock_time_s`; the energy figures are simulated and a capacity is not set.
- [ ] **Extend the N = 6 knee grid** if the plateau matters: the knee is the grid's largest budget, 150 s.
- [ ] **Whether to price the realised-over-planned gap** (a δ_obs > 0 or a margin) — the planner is optimistic by +0.8 % to +19.2 % depending on band and payload; no decision on record.
- [ ] **Beacon hook** — built and inert (no source); exercised only by unit tests.

---

## 13. Doc and code discrepancies found while writing

1. **Readiness stage table.** `Experiment_5_Readiness.md` lists `sens` as "⏳ After batch 1 | Batch 1's CSVs", while its own header and step 5 say batch 1 and `sens` are done and scored (commits `6f6f2bd`, `c784699`; `results/exp5/scores/sens/` exists). The `quick` row is the only stage still waiting on batch 1 in practice.
2. **Build plan "Gate" line** (FeRRy_Build_Plan, Phase 4 pipeline): S3b "keeps a candidate only if every stop's transit + session + return + upload fits min(Deadline(j), budget)". The code default bounds the *collection* (`finish ≤ Deadline(j)`) and applies return and upload only to the budget clause; the plan's min form is `deadline_bounds = delivery_per_stop`, which the plan's own "Deviations from this plan" list states. The Phase 4 line is a summary, not the default.
3. **S3b module docstring** says "There is no propulsion-energy, upload-rate or return-leg term"; that is the legacy model's cost model only (the same docstring's FeRRy Phase 3 section describes the ferry terms). Not wrong, but easy to misread as current.
4. **Legacy `overdue` compares arrival, the ferry default compares finish** (`:598` vs `:614-617`). Documented in the code; the Configuration Reference does not say so in one place.
5. **Config Reference line references** to `s3_deadline.py`, `ddqn.py` and `fl_scheduler.py` have drifted (see doc 02 §13); this document's references are current as of 7 Oct.

---

## 14. Sources

`hermes/scheduler/stages/{s3b_feasibility, s3d_age_cap}.py` · `hermes/scheduler/fl_scheduler.py` · `hermes/scheduler/routing/{replan, two_opt}.py` · `hermes/scheduler/selector/{scope_guard, target_selector_rl}.py` · `hermes/mule/mule_main.py` (`_remaining_is_feasible`, `_ferry_departure`, `_ferry_fly_pass`, `_widen_abandoned`, `_ferry_try_insert`) · `hermes/l1/mission_clock.py` · `hermes/l1/contact_link.py` · `hermes/mule/ferry.py` · [Scheduler Freeze](../HERMES_Scheduler_Freeze.md) (§1, §2 D1–D6, Amendments 1, 6–8, §5g table) · [Exp 4 L2 record](../HERMES_Experiment4_L2_Scheduling_Layer.md) (§4.2, §4.8) · [Configuration Reference](../HERMES_Configuration_Reference.md) (§17, §18) · [FeRRy Build Plan](../FeRRy_Build_Plan.html) (Phases 3 and 4, pilot notes, deviations) · [Decision memo](../architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md) (§1.1, §1.2, §4, §5, §7) · [Experiment 5 readiness](../Experiment_5_Readiness.md) · [Experiment 5 reproducibility guide](../Experiment_5_Reproducibility_Guide.md) (§5.2, §5.3) · `scripts/exp5/params.toml`, `results/exp5/{ttl,knee,sstar}/report.txt`, `results/exp5/scores/{b1,sens}/`, `results/exp5/b1/_launcher/manifest_20261006_003651.json`.
