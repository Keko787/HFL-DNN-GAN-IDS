# 06 — Route and plan search: the plan-time joint decision (band class × clustering × route)

*7 Oct 2026. One document in the joint-methods series; the overview is [HERMES_Joint_RL_Methods.md](../HERMES_Joint_RL_Methods.md), where this is **J1**. A reading and reference document: it restates the code and the records, decides nothing, and says "unverified" where a claim could not be checked against either. Numbers come from the files named beside them; none was re-run, except three tallies over the batch 1 per-trial CSVs, marked "tallied here". Status of every stage: [Experiment_5_Readiness.md](../Experiment_5_Readiness.md).*

---

## 0. Purpose and the short version

**What "navigation" means in HERMES.** FeRRy does no motion planning. A leg is a straight line at cruise speed (transit time is distance over speed in the shared `FeasibilityModel`; the flight model prices 143.6 W flying against 168.5 W hovering, `plan_score.py` module docstring). The mule's "trajectory" is therefore three discrete things chosen **once per mission, at the dock**:

1. which **band class** b̄ the whole mission flies (wide / medium / narrow), which fixes the radius a stop can reach;
2. which **stops** exist and which **members** each serves (clustering at that radius, with per-stop member subsets);
3. the **order** of the stops (the Pass-1 route π).

These three are one decision, not a chain: class sets range, range sets the clustering, clustering fixes the stops and so the route, and the route sets arrival times and the clock. The planner prices every class and commits the best (`plan/plan_search.py`, `FLScheduler.build_ferry_plan`). This is claim **C1, "reach is a decision"**, and the test for system-versus-stack is to cut the band→range edge (arm **FB+**, b̄ pinned).

**It is not learned.** Plan time is exhaustive search up to 6 devices and a bounded local search above, on a hand-set score V. The only learned component anywhere in the stack is the in-flight pair score FQ (J2, document 05), and it matched the fixed rule FX (Study 5.5, flat). Nothing in this document is a learning method; it is the *non-learned* half of the joint design, and the thing a learned planner would have to beat.

**What the evidence says so far.** The plan layer is built and its structure shows in batch 1: at N = 6 F commits narrow or medium and serves every demanded device at the knee budget, and beats every whole-scheduler baseline there (time to τ = 0.71: F 194 s, against 264–282 s for H1 and D1–D4). The attribution to *reach* is **not yet tested**: Study 5.4 (F against FB+ pinned to each class) and the optimality gap O1 are built and not run (batch 2). Batch 1's own plan-component ablations (5.14) show **no difference at N = 6** except whole-stop admission at the stress budget (not a claim). The lead over FedEx's tour (D4) is gone by N = 12 and N = 24. Sections 12 and 13 give the detail.

---

## 1. The decisions made, in one table

| # | Decision | Clock | Inputs | Output | Who decides | Learned? | Code |
|---|---|---|---|---|---|---|---|
| 1 | **Who is slice-mates** (device → mule) | Dock / trial launch | Device positions, K mules | Disjoint `MissionSlice` per mule | Registry round-robin by ID; the harness overrides with angular sectors (D4: CARP Gibbs) | No | `cluster/device_registry.py:123`; `exp4/topology_builder.py:171`; `exp4/driver.py:516` |
| 2 | **Demand** (who is plannable) | Plan | S1 eligibility, S3 bucket and deadline | `demand`, `deadlines` | Fixed rules | No | `fl_scheduler.py:1580-1601` |
| 3 | **Age cap and coverage weights** | Plan | Ages (mule's missions since last merge), miss streak | `CapState`, weights w_j | Fixed rules, S* from a pilot | No | `fl_scheduler.py:1603-1608`; `plan_score.py:334` |
| 4 | **Clustering → stops**, per class | Plan | Positions, class radius R_planar(c), S3 deadlines | S3a stops, then hover stops for capped devices | Greedy S3a; hover rule | No | `s3a_cluster.py:99`; `plan/hover.py:317` |
| 5 | **Member subset per stop** ("skip, not stop") | Plan (and re-plan) | Stop, flight state, budget, member order | A stop reduced to the members that fit | Predicate walk in F member order | No | `plan/member_subset.py:415` |
| 6 | **Route π** | Plan | The class's stops | Ordered stops, each with its members | Exact / stop-subset / local search | No | `plan/plan_search.py:573-801` |
| 7 | **Band class b̄** | Plan | Every class's best candidate | One class | Smallest plan key over classes | No | `plan_search.py:878` |
| 8 | **Commit** (b̄, queue, budget, score, cap) | Plan | The winner | `PlanCommit` | Scheduler, after a guard fold | No | `fl_scheduler.py:1665`; `types/scheduler.py:451` |
| 9 | **Pass 2 order** (delivery) | Plan (priced), flight (flown) | Whole slice at R(b̄) | Nearest-first order, on b̄ | Greedy nearest-first, not searched | No | `s3a_cluster.py:187`; `fl_scheduler.py:1367` |
| 10 | **Re-plan** after an overrun | Flight | Remainder, observed state | Trimmed remainder (never re-ordered in plan mode) | The one S3b predicate | No | `fl_scheduler.py:1204` |
| 11 | *Next stop and band in flight* | Flight | Observed SNR | (band, next stop) | FX (fixed) or FQ (learned) | **FQ only** | Document 05 (J2); not this one |

Rows 4 to 8 are the plan-time joint decision. Row 11 is listed only to mark the boundary: the plan commits an order and a class, and the flight slot may reorder and re-band within it.

---

## 2. The candidate family and the plan key

### 2.1 Inputs the search is handed

`build_ferry_plan` (`fl_scheduler.py:1454`) gathers, in this order: S1 and S3 (demand and deadlines, 1580-1601), the age cap and weights (1603-1608), and for **each class the arm may fly** S3a at that class's radius, then the hover rule, then Pass 2 priced once (1613-1633). It then calls `plan_search` (1634), runs a guard fold (1641), derives drops and cap violations (1652-1659) and commits (1665). Each class carries its own `FeasibilityModel` and an `outage(d)` callable (`PlanClass`, `plan/types.py:569`); the search module is numpy-free and imports nothing from L1, the mule, the policies or the scheduler (`plan_search.py` docstring, "Layering"). L1's contribution reaches it only as those callables.

### 2.2 What a candidate is

A candidate is a class c plus an **ordered sequence of distinct stops** offered on c (S3a(c) with the hover rule), each reduced to a **non-empty subset of its members**, such that every stop in turn passes S3b's predicate (`RULE_DEADLINE_BUDGET`) from the state the previous one left (`_admit`, `plan_search.py:491`). The age cap's stop rules are applied by value to every stop as reduced: a stop whose members are all capped is exempt from its own deadline clause; a mixed stop carries its uncapped members' earliest deadline. **The empty plan is a candidate** on every class: it flies nothing and pays only the dock turnaround (`run`, 542-547). Under `member_admission = whole` each stop is flown whole or not at all, which keeps the narrow-band cliff for comparison.

### 2.3 The plan key (the choice rule)

The smallest key wins (`plan_score.py:659-682`):

| Rank setting | Key (smallest wins) |
|---|---|
| `coverage_rank = lexicographic` (**default**) | (cap key, −round(served share, 9), −round(V, 9), class index, each stop's (position, devices)) |
| `weighted`, and F-cov under either setting | (cap key, −round(V, 9), class index, stops) — this is `Candidate.key` (`plan/types.py:646`) |

- **Cap key** (`s3d_age_cap.py:361`): the ages of the capped devices the plan leaves out, largest first. It minimises the oldest unserved age first, then the next; (4, 4, 4) beats (5, 3). A plan that serves every capped device another serves, and more, has a strictly smaller key.
- **Served share** (`plan_score.py:623`): Σ_served w / Σ_demand w, which is 1 − U. Every demanded device weighs more than 0, so under `lexicographic` **the empty plan wins only when no plan that serves anyone is admitted**.
- **V** and the class index and stops are the tie-breakers. Rounding to 9 decimals makes the same plan priced along two float paths tie; the final tie falls to class index and stop positions. The order is **total**, so the pick never depends on enumeration order, and F's key is never above any FB+c's (each class is searched on its own, then `min` over classes: `plan_search.py:878`; critic A3).
- `applied_rank` (`plan_score.py:641`) forces `weighted` when κ = 0 (F-cov), because a share-first rank would make F-cov serve everyone the budget allows, while F-cov is meant to be cap-only service.

**Why the share comes first.** V alone does not keep the "serve everyone the budget allows; time breaks ties" promise: every serving plan pays a whole Pass 2 and the empty plan pays none. The recorded probe: u and v 60 m either side of the dock, z 400 m out, FB+wide, 1 MB, a 37.6 s budget, T = 200 s, cap off: serving v scores −3.85 (predicted 264 s mission, its Pass 2 flying out to z), the empty plan −3.02 (30 s turnaround), and with the weights growing together the planner flew empty mission after mission (`plan_score.py` docstring, "The rank"; resolution R11). The cost is documented: under the share-first rank time only breaks ties, so a plan can fly a much longer mission to serve one more device (build plan 5.4 caveats), and the share is nominal (a served member counts fully, though on the pilots' jittery channel its outage at a class's edge is about 0.15 to 0.2, Configuration Reference §18.3).

### 2.4 The score V, recapped

V(b̄, π | demand) = −[c₁(Δ/T)² + c₂U + c₃L] − c₄E/(P_hover·T) (`plan_score.py:11`; defaults `plan/types.py:385-391`: c₁ = 1, κ = 1 so c₂ = κ·N_demand, c₃ = c₂, c₄ = 0.1). Δ is the whole mission on b̄: Pass 1, the turnaround, **and Pass 2, which also flies b̄**; T is the cell's T_nom. The planner prices only the *mean* SNR (δ_obs = 0): pricing the seeded phase would be an oracle. Pass 2 is priced once per class by `price_pass_2` (`plan_search.py:270`) exactly as T_nom prices it: the class's Pass-2 queue folded from the dock at clock 0, no budget, delivering. The constants are hand-set and swept (κ over {0.15, 0.25, 1}, c₄ over {0, 0.1}: `PILOT_KAPPAS`, `plan_score.py:170`); there is no derivation from the theory track, and Δ² is a convex surrogate for staleness, not a bound. Overview §1.3 and §4 cover V as a cross-layer objective; this document only needs it as a ranking.

Under the default weights, F's weight is about a_j² (age times 1 + miss streak, and every device a plan leaves out is widened as a miss); F-prio (miss priority off) weighs by age alone (`coverage_weight`, `plan_score.py:334`).

---

## 3. The search: three modes by size

`search_mode` (`plan_search.py:368`) picks per class, from the demand size and the class's stop count; defaults in `PlanSearchParams` (`plan/types.py:487-491`).

| Mode | When | What it enumerates | Optimal? |
|---|---|---|---|
| `exact` | demand ≤ `exact_max_devices` = 6 | Every ordered sequence of distinct stops, each reduced to every non-empty member subset (`_exact`, 573-618). A failing prefix prunes all its extensions; at one flight state a failing member set prunes its supersets (every predicate clause is monotone in a stop's members) | **Yes, over the member subsets V prices** (checked against an independent brute force in the tests, build plan Phase 4) |
| `stop_subsets` | demand > 6 and the class has ≤ `exhaustive_max_stops` = 6 stops | Depth first over ordered subsets of stops, each whole if it fits from its prefix, else reduced greedily in the F member order (`_stop_subsets`, 620-639) | No: never sheds a member from a stop that fits whole |
| `local` | more stops | 2-OPT tour → member trim → first-improvement scans (`_local`, 643-801) | No: a heuristic, bounded |

**The local search in detail.**

- *Start:* `order_contacts(stops, dock, end=dock)` (cheapest insertion then 2-OPT, no restarts, §5), then U3's member trim of it (`trim_members`, `member_subset.py:604`); under `whole`, the tour with priority stops first, each kept whole if it fits.
- *Neighbourhood, in a fixed order (the order is part of the contract):* drop a stop; insert an unrouted stop with all its members at every place; reverse a segment of two or more stops; under `subset`, drop one served member of a multi-member stop, least worth first (`moves`, 703-722). A move counts only when every stop it lists admits someone; a neighbour walked before is not walked again.
- *Scans:* under `weighted` one scan on the plan key. Under `lexicographic` a scan on the plan key never takes a drop or a drop-member move (each serves less weight), so up to **three** scans run: `weighted`'s own on `Candidate.key` from the trim's route, then one on the plan key from the best plan so far, then, if that route differs, one on the plan key from the trim's route (788-801). Later scans reuse earlier walks without counting them. The class's best is never below `weighted`'s plan under the plan key.
- *Bounds (counts, never wall time, so a repeated trial plans the same):* stop at a local optimum, after `heuristic_max_passes` = 50 passes, or when a walk would exceed `heuristic_max_evaluations` = 2,000 walks per class (start included, all scans together). The counts buy determinism, not a time bound (§10).
- *Known gaps:* no move swaps one stop for another, so where two capped stops compete for one slot it can keep the younger one the trim took first. Alone, the plan-key scan ended below `weighted`'s plan under its own key on 3 of 600 random local problems at κ = 1 (the 2026-09-30 review); forced onto small capped problems the heuristic ended with a worse cap key than the exact search in 6.1 % (R7, Configuration Reference §18.4).

**Above 6 devices the search is a heuristic and a declared plan deviation** (Freeze and the plan record). Only `exact` is optimal.

---

## 4. Member-subset admission: "skip, not stop"

**The problem it removes.** S3a puts every device within R_planar(b) of a stop into one contact, and every gate before Phase 4 admits a contact with all its members or none. On narrow (and medium) with a declared payload one stop covers the whole field, so under a budget below that stop's time the plan is empty: a **0-or-N cliff** set by contact granularity, not by the reach-against-dwell trade the band choice is about (finding E2E1-01; `member_subset.py` docstring; pinned in `tests/unit/test_p3_final_fixes_mule.py`).

**The rule** (`member_subset.py`). A reduced stop is an ordinary `ContactWaypoint` (`reduce_stop`, 160): the stop's own position, the subset as its devices, the worst of their buckets, the earliest of their deadlines. `admit_stop` (415) tries the stop **whole first**; if it fits it is admitted as the same object. Otherwise `admit_members` (305) walks the members in a given order, admitting each if the stop with it still passes the predicate and **skipping it otherwise, then going on**. Because every predicate clause is monotone in a stop's member set (dwell adds, the deadline is a minimum, energy grows with dwell, and arrival, return and upload do not depend on members), one pass in order is final and the admitted set is maximal for that order; a stop that fits whole is admitted whole by it.

**The F member order** (`member_order`, 236): capped members first; then w_j / dwell_j descending (the greedy knapsack ratio: weight per second of the member's predicted dwell alone); then device id. A member beyond range or below the SNR floor has zero dwell and, if it weighs anything, leads its group.

**With a cap:** an all-capped stop is exempt from its own deadline clause; a mixed stop is held to its uncapped members' deadline (critic B2), so a priority stop (any capped member) sheds its uncapped members first, and a priority stop none of whose members fits is dropped member by member, each with the reason that refused it, never labelled `overdue` for a capped member. Exemption is decided on each stop **as reduced, every time**, because a reduced stop is a new waypoint (critic B1).

**Where it applies.** The F family (F, FX, FB+c, F-cov, F-cap, F-prio) under `subset`, which is the plan arms' default (`PlanOptions`, `plan/types.py:1303`). For H1–H3 and D1–D3, D5 only pre-flight and only when a run asks; never D4, whose tour has no gate (Configuration Reference §18.5). `MuleConfig`'s own default is `whole`; the driver passes `subset` for plan arms.

**What it buys, in the data (tallied here from `results/exp5/b1/s514`, 40 trials per variant, 4 missions each).** N = 6 stress budget: whole-stop admission serves a mean share 0.878 against 0.918 for subsets, commits wide in 20.7 % of missions against 1.0 %, and has network AoU 0.897 against 0.810 (diff −0.0865, 95 % CI [−0.149, −0.0333], Holm p = 0.0775: **not a claim**). At the knee budget the two are identical (0.715, both). The cliff matters where a budget bites; at N = 6 and the knee, one stop serves everyone anyway.

---

## 5. Hover points

`plan/hover.py`, a user decision of 2026-09-30 after final-check findings PLAN-1 and E2E2-01. **The failure it fixes:** the mule widens every device a plan leaves out, so its deadline recedes, S3a anchors it last, and once it lies far from the rest it becomes a one-device stop **at its own position**, the dearest place to serve it from. The check found a device needing 47.4 s alone at its own position under a 45 s budget and 13.25 s hovering at the dock. No plan could serve it; the cap labelled it `unplannable`, which reads as physics; the mule widened it again; the loop starved it, often for good.

**The rule** (`offer_hover_stops`, 317). Under a budget and a cap, each **capped** device that its own S3a stop cannot serve alone from the dock within the budget (`servable_alone`, `s3d_age_cap.py:385`) leaves that stop for a one-device stop at its **best hover point** on that class. Uncapped devices, and capped devices their S3a stop serves alone, keep S3a's stops. Without a budget or without a cap nothing moves. Pass 2 is priced and flown on S3a's stops, not hover stops.

**The point** (`best_hover_point`, 212): on the segment from the dock to the device, the point that minimises that device's alone mission (flight there, dwell at |device − p|, flight back, Pass-1 upload) among points within the class's planar reach where the predicted dwell is finite. The segment suffices because dwell does not fall with distance on the contact link. Deterministic: a 64-cell grid (`GRID_CELLS`), then at most 64 halving rounds (`ROUNDS`) of at most 8 cells (`KEEP_CELLS`), tolerance 1e-9 s; 65 to 118 pricings a point on the critic's 30 layouts.

**Caveats.** It minimises *time*, not energy: under an energy capacity the cap's `unplannable` is physics only for the time budget (34 of 1,080 device-class pairs have a better-for-energy point at 1 MB; no pilot sets a capacity). The point often sits at the dock or at the class's reach edge, where the noisy link is weakest, so a hover stop misses more often in flight. Traces do not mark a hover stop (only its one-device shape and position). The switch `hover_stops` (default true) exists for Study 5.14's "hover stops off".

---

## 6. Routing primitives: cheapest insertion and 2-OPT

`routing/two_opt.py` is shared by the plan search, the Phase 3 re-plan and the D4 baseline. Three route shapes: closed tour, open path, **fixed-end path** (start fixed, every stop once, finish at `end`); a fixed end equal to the start is solved as the closed tour (`best_order`, 291).

- **Cheapest insertion** (`_cheapest_insertion`, 174): insert, one at a time, the point whose cheapest insertion adds least length; ties go to the cheaper delta, then lower index, then the *later* position. O(n³), "immaterial at mission sizes".
- **2-OPT** (`_two_opt`, 227): first-improvement; every segment reversal tried, an improving one applied at once; a move must beat `IMPROVEMENT_EPS` = 1e-9 m (line 65); at most `DEFAULT_MAX_PASSES` = 1,000 passes (line 70). Not optimal; the tests check it against brute force for n ≤ 7.
- **Restarts** (`best_order`, `restarts`, `seed`): extra 2-OPT runs from seeded random orders, as FedEx does. The plan search uses **none** (`order_contacts(stops, pose, end=dock)`, `plan_search.py:650`); D4 uses 8 (§7).
- **Determinism:** `order_contacts` sorts contacts by (position, devices) first, so the route is a function of the contact *set*, not the order listed.

The router minimises **length**. It does not see dwell, deadlines or the band; those enter through the predicate and V when the search scores what the tour gives it. That is why the local search's 2-OPT start is only a seed.

---

## 7. The plan commit

`PlanCommit` (`hermes/types/scheduler.py:451`, constructed at `fl_scheduler.py:1665`) is the build plan's "(b̄, ordered queue, budget) for this mule's slice". It holds: `band`, `band_index`, `band_class_policy`; `queue` (S3a stops, some reduced, plus one-device hover stops); `budget_end` (absolute, mission clock; None without a budget); `demand` and `weights`; `score` (the nine `PLAN_SCORE_KEYS`: v, delta_s, time, coverage, link, energy_j, energy, served_weight, demand_weight, plus `mission_s`, the predicted whole mission, recorded for every arm); `constants` (c₁..c₄ resolved) and `t_ref_s`; `search_mode`, `n_candidates`, one JSON `per_class` summary per class searched (mode, stops offered, candidates, V, served, cap key, served share, applied rank, and for `local` evaluations, passes, `bounded`); and the cap: `cap_s`, `cap_lookahead`, `ages`, `capped`, `violations`.

**Frozen and checked.** The constructor refuses a device in two stops, serving outside the demand, weights that do not match the demand, a `fixed:<c>` policy that disagrees with the flown band (FB+c flies only class c), wall-time keys, and a capped set that is not exactly the devices aged ≥ S − L. `close` returns a checked copy at mission end (visited set; `dropped_in_flight` and `not_merged` violations).

**The commit step** (`fl_scheduler.py:1687-1705`): the class's model becomes the scheduler's model, so the departure check, the re-plan and Pass 2 all price b̄; `last_feasibility` holds the route and every demanded device the plan leaves out, labelled with the first predicate clause that refuses it alone at its offered stop (`overdue`, `budget`, `energy`) or `plan` (a choice; kept out of S3c's planned count). Wall time never enters a decision: it goes to `last_plan_wall_s`, outside the commit, so `describe()` is identical on a repeated trial (critic B12).

**The guard** (1641-1650): the chosen route is folded again under S3b's predicate as flown, exempt stops protected, the set computed on the route itself. A failure raises `FLSchedulerError` and commits nothing; it would be a bug in the search.

---

## 8. The arms that vary this decision

| Arm | What changes from F | What it isolates | Study |
|---|---|---|---|
| **F** | Nothing: search every class; subset admission; committed flight slot; age cap at S\*; weights age × (1 + miss streak) | The system under test, without the learned slot | all |
| **FB+wide / medium / narrow** | `band_class_policy = fixed:<c>`: only class c is searched and flown, committed slot only | **Cuts the band→range edge**: same clustering, search, subsets and cap, one class | 5.4, 5.8 |
| **F-cov** | κ = 0, c₃ = 0 | Coverage term out of V; the weighted rank applies; serves capped devices only ("cap-only service") | 5.8; 5.7 as FX-cov |
| **F-cap** | `age_cap_missions = None` | No hard cap | 5.8 |
| **F-prio** | `miss_priority` off | Weights by age alone (≈ a_j, not ≈ a_j²) | 5.8 |
| **FX** | Flight slot `cross_heuristic` | Same plan; in-flight reorder and band switch by a fixed rule (document 05) | 5.3, 5.5 |

Arm definitions: `experiments/exp4/driver.py:153` and `226-237`, Configuration Reference §18.8 (rows for F, FB+, F-cov, F-cap, F-prio). All plan arms run the `trim` re-plan fallback. A pinned band refuses any flight slot but `committed` (`PlanOptions`, `plan/types.py:1286`).

### 8.1 D4 — FedEx/CARP, the closest route prior

`policies/fedex_carp.py` ports Bian et al. (IEEE TMC 24(6), 2025). It differs from F on every axis this document describes:

| | F | D4 (FedEx tour) |
|---|---|---|
| Stops | Chosen: S3a per class + hover, with member subsets | Every S3a contact, whole |
| Band | Searched per mission | One fixed band (the cell's `contact_band`) |
| Skipping | Yes (budget, deadline, cap) | Never; no gates; `in_flight_check = IN_FLIGHT_NONE` (`budget_walk.py:67`), so the tour is flown even past the budget |
| Route | Search over orders with a score | 2-OPT closed tour (fixed end at the dock on the mission clock), **8 restarts**, seeded (`DEFAULT_TOUR_RESTARTS`, line 241; `_admit_on_the_mission_clock`, 919) |
| Objective | V (hand-set, FedEx's form) | Shortest tour; for K > 1, Σ R_k·Δ_k² via Gibbs assignment |
| Assignment | The harness's split (§9) | **Gibbs sampling** (`carp_search`, 594; `carp_assign`, 727): geometric schedule q_s = q0·0.975^s, 200 sweeps, initial acceptance 0.8, then a greedy polish; tours priced with 5 restarts |

Honest limits of the CARP port, from its own docstring: tuned on 200 seeded instances; on a fresh family it missed the async optimum 6 of 400 times (1.5 %) and the sync optimum 11 of 400 (2.8 %), 0.3 to 9.0 % above optimum, the loss being the assignment, not the tours; a 40-client, 4-transporter instance takes about 5 s. D4 declares 13 deviations (one transporter per mission; no energy gate; contact-level stops priced with S3b's model; seconds not slots; return leg but no depot time; and so on). It runs twice, with FeRRy's merge (`agg:cutoff`, arm D4) and with FedEx's (`agg:fedex`, D4fedex), so route and merge are judged separately.

### 8.2 E3 — the trajectory baseline contrast

E3 (`policies/chen_dqn.py`, `policies/next_stop.py`) is the learned **trajectory-only** competitor, after Chen et al. (GLOBECOM Workshops 2023). Against J1: it has **no plan, no band choice, no cap, no deadline, no coverage term**; it admits every S3a contact and names the **next stop** at takeoff and at every Pass-1 departure among the stops S3b's single-contact budget predicate admits (Chen's "safety controller", no deadline clause, `next_stop.py` docstring), on the cell's one band, rewarded in bytes. So where F commits a whole order and a class at the dock, E3 decides one stop at a time in flight and never decides reach. It belongs to the flight-clock story (overview §2.7) and is the learned counterpart to J1's non-learned plan, not a variant of it. Study 5.6 flies it against FX.

### 8.3 O1 — the offline oracle, "after Zhai"

`experiments/analysis/o1_oracle.py` (built; **not yet run on the study's cells**). It answers: how far is F's plan from the best plan over **band class × clustering × route × per-stop band**, for N ≤ 6 (`MAX_DEVICES = 6`, line 78), a planning-level tool never flown.

- **Inputs:** exactly the inputs of F's own plan, captured by an `on_mule` hook that wraps `build_ferry_plan` in FerrySim episodes (`capture`, `oracle_hook`, 433), so the gap is read on F's own mission states, ages and caps.
- **Family:** an ordered sequence of disjoint groups of demanded devices; a group is served at one position on one class whose planar reach covers every member from it. Positions offered to a group: its centroid, each member's position, and the position of every stop F was offered on any class that holds all its members (`_options`, 199). So F's family lies inside the oracle's, and the oracle adds any grouping, any order and **per-stop bands** (the committed class prices Pass 2 and nothing else).
- **Scoring:** F's `score` and `cap_key`; two optima per mission, the best under **F's own plan key** (the gap the arm could close) and the best V alone. Reported: `gap_v` (≥ 0), `gap_v_at_key`, `gap_share_at_key`, whether the key-best flies a stop off its committed class, and search size.
- **Pruning:** a branch is dropped when an earlier one served the same devices at the same position no later, on no more energy, with no more link loss; or when an upper bound on every extension's V (home, energy and link held, nothing left uncovered, cheapest Pass 2) cannot beat either best (`_dominates`, `_v_upper`, `_hopeless`, 284-320); a test checks the bound changes nothing. Capped at 20,000,000 expansions.
- **Checks on every mission:** the oracle must price F's committed plan at F's own V (else it raises), and its best must not be worse than F's under F's key.
- **Cost:** about 1 minute per N = 6 mission on one core (Configuration Reference §20.11); the smoke run's oracle took 50 min at the full 30 episodes (Readiness, step 3).
- **"After Zhai" is in spirit only.** Zhai et al. (IEEE TWC 24(3), 2025) design trajectory, selection and transmit amplitudes jointly and *offline* by alternating convex approximation to a local optimum, with no band classes, no per-stop band and no clustering (Related Work §3.3). O1 is exhaustive and searches those; it is "an offline full-information joint design, in the spirit of Zhai", never "Zhai's method". The build plan's file table still lists it as `oracle/zhai_oracle.py` (deferred); it was built as `experiments/analysis/o1_oracle.py`.

---

## 9. Multi-mule slicing

Multi-mule here means **disjoint `MissionSlice`s, one plan per mule**; there is no joint multi-mule planner. A `MissionSlice` (`mule_id`, `device_ids`, `issued_round`, `issued_at`) is the planner's whole world: S1 admits only slice members (`s1_eligibility.py:31`), the age cap counts the device's own mule's missions since its last merge, and each mule commits its own `PlanCommit`.

- **The registry's rule** (`DeviceRegistry.rebalance`, `device_registry.py:123`): clear assignments, sort devices (new ones first, then by id), assign **round-robin**. This is disjoint and deterministic but **not spatial**.
- **What the experiments fly.** The cluster seeds the registry round-robin and then overrides each device with the topology's own assignment (`processes/cluster.py:747-760`). The harness default is **angular sectors** around the dock (`angular_slices`, `topology_builder.py:171`: K contiguous runs by angle, cut at the widest empty arc, sizes differing by at most one), so each mule tours its own part of the field. Arm D4 overrides this with **CARP's Gibbs assignment**, computed once per trial over the seeded positions (`d4_slice_assignment`, `driver.py:516`, used at 2217), since devices are wired to one mule's RF link at launch and the split cannot change mid-trial.
- **Consequences for the plan.** Each mule's search is over its own slice, so N in the search is the slice size, not the deployment. Study 5.11 (b) varies K (weak: 6 devices per mule, K = 1..3; strong: N = 12, K = 1..3); one batch 1 point: N = 12, 3 mules, FX 102 s to τ against F 151 s, a claim for FX (batch 1 commit message). Scaling with mules is measured on the stack only (FerrySim is one mule).

---

## 10. Decision cost and scaling (Study 5.11 (a))

**What is measured.** `plan_wall_s` per plan-mode mission, recorded outside the commit. 5.11 (a) forces the search mode through `--plan-search-params` (`subsets`: `exact_max_devices = 0`; `local`: both 0) at N ∈ {6, 12, 24, 48, 96} in FerrySim, 20 episodes, 4 missions, FX, one mule (`results/exp5/s511a_quiet/b1/s511a/{auto,subsets,local}.json`).

**Reporting rule (Readiness, `batch1` row).** Report 5.11 (a) from the 7 Oct **rerun alone and unthrottled**, not batch 1's run (started beside batch 1's other jobs, possibly throttled; every episode made the same plans; planning ran 1.5 to 2.5 times faster in the rerun).

| N (cell, budget) | `auto` mean (p95), s | forced `subsets` mean | forced `local` mean | decisions per sortie (auto) |
|---|---|---|---|---|
| 6 (`jit-n6-150`, 150 s) | 0.037 (0.064) | 0.0083 | 0.0061 | 1.0 |
| 12 (`jit-n12-90`, 90 s) | 0.022 (0.043) | 0.021 | 0.011 | 1.78 |
| 24 (`scl-n24-350`, 350 s) | 0.142 (0.265) | 0.138 | 0.053 | 10.4 |
| 48 (`scl-n48-680`, 680 s) | 0.409 (0.826) | 0.393 | 0.311 | 20.5 |
| 96 (`scl-n96-1330`, 1330 s) | **1.531 (1.996)** | 1.508 | 1.500 | 42.8 |

*Values read from the three JSON reports' `table`; the 1.53 s mean and 2.0 s p95 at N = 96 and 0.037 s at N = 6 match the Readiness sheet.*

**Reading it.**

- **Not monotone at the small end.** N = 6 under `auto` is the *exact* search, which is slower than N = 12's heuristic search (0.037 against 0.022 s). Forcing the heuristic at N = 6 makes it 4 to 6 times faster (0.037 s to 0.0083 s forced `subsets`, 0.0061 s forced `local`) with identical plans on these cells (the same mean return and served share in all three modes at N = 6). Which mode `auto` used at each larger N is **unverified** here: the per-class `mode` is in each trace's `per_class` summary but was not tallied.
- **The heuristic costs quality at N = 12.** Forced `local` serves a mean share of 0.424 against 0.457 under `auto` (return mean −0.152 against 0.052); at N = 24 it is 0.540 against 0.5375, at N = 48 and 96 the same as `auto`. N = 12 is the only cell where forcing the heuristic visibly lowers served share (a 20-episode, one-family result; no test of significance).
- **Cost against a mission.** At N = 96 the mean plan is 1.53 s against a 1,330 s budget (about 0.1 %). This compares wall seconds with simulated mission seconds; the build plan states the criterion as "decision cost stays a small share of a mission at the largest N" (5.11 "Reads as"). Part (a) of the study as designed also reports share of missions per search mode, V of local against exact where exact runs, and per-decision slot time; the slot's mask and score time is empty for FX here (`decide_s` null).
- **The earlier worst case is a different condition.** The Phase 4 probe, N = 96 on a **500 m** field, 1 MB, **no budget**, all three classes at the 2,000-walk bound, took **3.1 to 3.4 s** per plan (`plan_search.py` docstring, `PlanSearchParams`); and at N = 6 a whole plan took at most about 0.25 s where nothing prunes (three classes of six one-device stops, 5,871 candidates; 0.21 s on the mule's own classes) and about 20 ms under 30–120 s budgets. Do not set these against the FerrySim table: field size, budget and payload differ. The N = 96 cell's first run in batch 1 took 3.70 s on average (p95 5.16 s) against the rerun's 1.53 s; Readiness reads the first as probably power-throttled ("may also have been throttled"), which is why the rerun is the one to report.

### 10.1 Complexity bounds (derived from the code; no recorded proof)

| Piece | Bound | Basis |
|---|---|---|
| S3a clustering, one class | O(N² log N) worst case (N singletons: sort per anchor) | `s3a_cluster.py:99` loop with a re-sort per cluster |
| Hover point, per capped device | ≤ 64 + 64·8 halved cells priced; 65–118 pricings observed | `hover.py:108-112`; docstring |
| `exact`, per class | ≤ Σ over subsets S of stops of \|S\|! · Π_{i∈S}(2^{m_i} − 1) predicate calls, with prefix and down-closure pruning; six one-device stops: 1,957 candidates per class, 5,871 over three | `_exact`; 1+6+30+120+360+720+720 = 1,957 |
| `stop_subsets`, per class | ≤ 1,957 ordered stop sequences (≤ 6 stops), each stop admitted by a member walk | `_stop_subsets` |
| `local`, per class | ≤ 2,000 walks (all scans), ≤ 50 passes; a walk folds one route with admissions memoised by (state, stop, mask); neighbours per pass ≈ k + (s−k)(k+1) + k(k−1)/2 (+ member drops) for a route of k of s stops | `_local`, `moves`; `PlanSearchParams` |
| 2-OPT start | cheapest insertion O(n³); 2-OPT pass O(n²), ≤ 1,000 passes | `two_opt.py` |
| Pass-2 pricing | One fold per class | `price_pass_2` |
| Whole plan | 3 classes × the above + one guard fold + the drop labelling | `build_ferry_plan` |
| O1 | exhaustive over ordered disjoint groups: (2^N − 1) groups × (≤ 1 + \|g\| + offered) positions × classes; dominance and V-bound pruning; hard cap 20,000,000 expansions | `o1_oracle.py:78, 178-260` |
| CARP Gibbs | 200 sweeps × N clients × ≤ K candidate moves, each pricing two tours with 5 restarts | `fedex_carp.py:241-267` |

The 2,000-walk bound makes the *count* of walks constant, not the time: a walk costs more on a longer route.

---

## 11. Layer interfaces (L1 ↔ L2 ↔ L3)

**L1 → L2 (what the radio layer hands the planner).** Per class: a radius R_planar(c) (S3a's radius and the contact gate's range); a `FeasibilityModel` with ferry physics bound to that class and to the scheduler's device states (an unbound model would price every member at its stop, critic B3), whose `leg`, `dwell_s`, `member_distances_m`, `home_at`, `fold` and `admit` supply transit, per-member dwell at the class's *mean* predicted rate, the Pass-1 upload and the energy; and `outage(d)`, the probability a member at planar distance d is below the SNR floor at the class's mean SNR with the channel's spread. The planner never sees the seeded phase. L1 learns nothing here.

**L2 internal (stages around the search).** S1 → S3 → age cap/weights → S3a (+ hover) per class → search → guard fold → commit. Gates always run before anything ranks anything: the S3b predicate is the one admission rule used by the search, the guard, the departure check and the re-plan.

**L2 → L1 (what the commit sets).** The commit makes b̄ the mule's band: the scheduler's model becomes b̄'s, and the mule sets its runtime's band (`FerryRuntime.set_band`, Configuration Reference §18 step 7). Pass 2 flies b̄ too. The arrival times the route implies set the phase each in-flight observation lands in (J2).

**L3 → L2.** (i) The **age cap** sits first in the plan key, makes an all-capped stop exempt from its own deadline clause, and drives the hover rule. (ii) **Coverage weights** from ages and miss streaks define U, L and the served share. (iii) **Deadlines** from S3 bound each stop through the predicate. (iv) S\* is a pilot output (smallest S covering 90 % of layouts at both pilot budgets, never below 2; S\* = 2 at every N).

**L2 → L3.** Devices a plan leaves out are widened as misses, so ages and streaks feed the next mission's demand and weights; `PlanCommit.visited` and close-time violations feed the cap's accounting; the plan's `served` set is what the merge later sees. The plan is the *same* quantity the merge weight prices only by construction (overview J4), not by a derivation.

**Slice interface.** The only cross-mule coupling in planning is the slice boundary (§9); ages are per mule.

---

## 12. Failure modes

| Failure | Mechanism | Where documented | Mitigation or status |
|---|---|---|---|
| **Empty missions at a tight budget** | (a) `whole` admission and the narrow 0-or-N cliff; (b) under `weighted` rank V can prefer flying empty (every serving plan pays a whole Pass 2); (c) lexicographic rank: empty only if nothing that serves anyone is admitted; (d) before the cap binds a mission can still fly empty, most at 30 s; (e) a capped device no class serves alone is `unplannable` | `member_subset.py` docstring; `plan_score.py` "The rank"; build plan 5.8 caveats | Subsets; lexicographic default. κ sweep: at κ = 0.15, 2 of 30 plans already fly empty at 30 s and 1 MB; at κ = 0.1, 1 to 9 of 30 (the critic's 0.1–0.25 range was rejected for this) |
| **Starvation by the widen loop** | A far device's deadline recedes, S3a anchors it last at its own position, no plan can serve it, the cap labels it `unplannable` | `hover.py` docstring (PLAN-1, E2E2-01) | Hover rule (§5) |
| **T_nom priced on wide** | The deadline unit is T_nom, priced on the cell's reference class (wide) for every arm, so an arm with shorter missions fits more of them into a deadline window: F's narrow missions last about 55 s against a T_nom of 172–250 s, so about 4× as many fit (critic C4's estimate) | Configuration Reference §18.8; build plan 5.4, 5.8 caveats | **Documented, not corrected.** Bears on every F-vs-wide-arm comparison |
| **Crowding at S\* + 1** | S3a re-clusters every mission, so the S\* tool's S + 1 is no guarantee for a *pinned* class at the stress budget: FB+medium crowded 7 times in 291 missions at 45 s, FB+wide twice in 360. A capped far device is often served from its hover point, where in-flight misses are likelier. F itself, on a deterministic loopback at S = S\* + 1, left no capped device out on any of the critic's 30 layouts at 45 and 60 s | Build plan 5.4, 5.8 caveats; Phase 4 tests | Reported as cap violations by cause (`crowded` vs `unplannable`); no pass mark. About 15 % of device-missions miss at S = 3 from availability alone |
| **Longer missions to serve one more device** | Lexicographic rank: time only breaks ties | Build plan 5.4 caveats; Freeze §5k gives the measured cost | Reported; the `weighted` rank is the other end of the κ sweep |
| **Mean-SNR pricing** | Dwell is convex in SNR, so realised missions run longer than planned: +0.8 % wide, +7.1 % wide at 10 MB, +19.2 % narrow at 1 MB (per member up to 1.39×). Budgets bind in flight more on narrow | Build plan 5.4 caveats | The re-plan `trim` repairs it; not an error of the search |
| **Local-search optimality gap** | No swap move; the plan-key scan alone misses `weighted`'s plan on 3 of 600 random local problems; worse cap key than exact in 6.1 % of small capped problems | `plan_search.py`; Configuration Reference §18.4 | Up to three scans; O1's gap is the measure, N ≤ 6 only, so **above 6 devices the gap is unmeasured** |
| **Hover stops** | Time-minimal, not energy-minimal; often at the dock or reach edge; untraced | `hover.py` docstring | `hover_stops` switch for 5.14 |
| **Pass-2 pricing is by one queue** | Pass 2 is priced and flown on S3a stops, nearest-first, not searched | `price_pass_2`; `s3a_cluster.py:187` | By design |
| **Guard fold fails** | A bug in the search | `fl_scheduler.py:1645` | Raises, commits nothing |
| **Oracle disagrees with F's pricing** | O1 prices F's committed plan and must match to 1e-6 | `o1_oracle.py` `mission_gap` | Raises |
| **D4 overruns** | Never truncated; `in_flight_check = none` | `fedex_carp.py` deviation 11 | Reported (`last_tour_overrun_s`) |
| **Empty plan's band** | With several classes the empty plan's band is the first searched class (R8) | Configuration Reference §18.4 | By design |

---

## 13. Evidence

### 13.1 The decision-memo argument (why this layer is built the way it is)

The memo (`architecture review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md`, built 2026-08-26 to 09-16) starts from the team's diagnosis that the solution method was "too simple to justify RL" and concludes: **adding a trajectory layer will not justify reinforcement learning; adding a coupling will, and the difference is testable.** The mechanism (§1): modules that each optimise their own objective and hand a result downstream compose into a *chain*; a chain decomposes; a cross-heuristic solves a decomposed problem. A *coupling* is a loop in which a downstream quantity changes an upstream one.

- **Q1, "include a trajectory navigation layer?"**: *Yes — half of it.* Distance and order: add, because travel cost is the binding constraint and the machinery already existed (Exp 3 `sim_env`, S3b), disconnected rather than missing. Environment and obstacles: don't; it is solved motion planning, adds path length you could sample, and buys complexity without coupling. **This document is the "half" that was added:** the stop set and order, with no motion planner.
- **Q3, "jointly optimise band and navigation?"**: yes, and this is the contribution, as *two* chains hinged: `demand → contact band → range → clustering → route → commit` at plan time, and `arrive → observed phase → band → rate → dwell → clock` per stop. Test: cut the band→range edge (FB+).
- **The RL test (a) "decomposition"** lives here: "offline oracle on N ≤ 6, exhaustive over band class × clustering × route × per-stop band, against each decomposition; report the optimality gap of each. Falsified by: gap ≈ 0. A large gap on the cross-clock split but a small one *within* plan time is the likely and useful outcome: keep a heuristic planner, learn only the flight-time policy." That is O1. **(b) at plan time** (γ swept across missions) is untested; flat on the flight side means no hierarchical policy is entered, so the plan-time column stays untested only because nothing justified building a learned planner (overview §3.1).

**Where the build stands against that argument.** The plan layer was built as the *heuristic/exact* filling of the PLAN slot (the memo's learned filling, V(class, route | demand) trained on round closure, was not built). The only learned filling tested, in the FLY slot, came out flat. So the record supports "the architecture is right and a fixed rule suffices to exploit it" for flight time, and is silent for plan time until O1 and 5.4 run.

### 13.2 What batch 1 scored (time to τ = 0.71, simulated s, 20 paired trials per cell; `results/exp5/scores/b1/s53.md`, `s59.md`)

| Cell | F | H1 | D3 | D4 (FedEx tour, FeRRy merge) | FX | Reading |
|---|---|---|---|---|---|---|
| N = 6, 1 mule, knee | **194** | 282 | 280 | 264 | 178 | F beats H1, D1 (282), D2 (266), D3, D4, each a claim (Holm p ≤ 0.008); FX 16 s ahead of F, not a claim (Holm 0.146) |
| N = 6, 1 mule, stress | **178** | 220 | 209 | 264 | 175 | **Only D4 is a claim** (diff −86.4, Holm 2.67e-4); H1 (reach 0.9), D1, D2, D3 are not (CIs include 0) |
| N = 6, 3 mules | **52** | 65 | 65 | 55 | 56 | Every arm ties with F except D4fedex (101, a claim): the gap is FedEx's *merge*, not its route |
| N = 12, 1 mule | 486 | 589 | 507 | 459 | 424 | **No claim at all** against F (D4: diff +27.9, CI [−50.1, 127]) |
| N = 24, 1 mule | **749** | 951 | 911 | **752** | 789 | F beats H1 and D3 (claims); **ties D4** (diff −3.1, Holm 1) |

Three things the table says about the plan layer. (1) F's lead over the schedulers is real at N = 6 and N = 24 against H1 and D3, but **the lead over FedEx's visit-all tour exists only at N = 6** (F's route beats FedEx's by 26 % at the knee and 33 % under stress, Related Work §3.1) and is gone by N = 12. The overview states N = 24 as "F 749 against H1 951 and D3 911"; D4's 752 is the omitted comparator that ties. (2) The F-versus-baseline gap at N = 6 bundles **reach, subset admission, the cap, deadlines and the route**. Every rival flies the one wide band. The bundle cannot be split without Study 5.4. (3) At N = 6 and 3 mules everything ties; the cells give the planner little to decide (Readiness: "one stop serves all six devices in most missions").

### 13.3 What F actually commits (tallied here from `results/exp5/b1/s514/*_scored*.csv`, variant `capS` = F at S\*; 40 trials, 4 missions each; mean of per-trial `band_shares`)

| Cell | narrow | medium | wide | mean served share | search mode |
|---|---|---|---|---|---|
| N = 6, knee | 0.826 | 0.174 | 0 | 1.0 | `exact` in every mission |
| N = 6, stress | 0.709 | 0.281 | 0.010 | 0.918 | `exact` in every mission |
| N = 6, stress, `whole` admission | 0.350 | 0.443 | 0.207 | 0.878 | `exact` |

So at N = 6 the planner almost never chooses the class every baseline flies (wide), and at the knee it serves every demanded device. Whether that choice *pays* is Study 5.4's question.

### 13.4 Batch 1's plan-component ablations (Study 5.14, network AoU, lower is better, each variant against F in its cell, 40 paired trials; `results/exp5/scores/b1/s514.md`)

| Variant | N = 6 knee | N = 6 stress |
|---|---|---|
| F (cap at S\*) | 0.715 | 0.810 |
| whole stops | 0.715 (diff 0) | 0.897 (diff −0.0865, Holm 0.0775, **no**) |
| local search forced | 0.715 (diff 0) | 0.810 (diff ≈ 0) |
| weighted coverage rank | 0.717 (diff −0.0021) | 0.805 (diff +0.0052) |
| cap at S\* + 1 / cap off | 0.715 / 0.715 | 0.810 / 0.810 |
| hover stops off | 0.715 | 0.810 |

**No plan mechanism is a claim at N = 6.** Identical means mean those switches did not change the plans the cells make, which fits one stop serving everyone at the knee and 4 missions at S\* = 2. Only F+L1 (adaptive backhaul, L1) against F on the seconds backhaul is a claim (batch 1 commit message; not a plan decision). The ablations at larger N, where the plan has real choices, are the runs that would separate them; 5.14 as scored is N = 6 only.

### 13.5 What is not yet evidence

- **Study 5.4** (F against FB+wide, FB+medium, FB+narrow; sweeps `narrow_range_ratios` [2.0, 3.0] beside the derivation's 3.87, `far_shares` [0.25, 0.5, 0.75], payloads measured and 1 MB; 20 trials per cell, knee budget; ~960 trials, ~3.6 h; `scripts/exp5/params.toml` `[s54]`): **not run** (batch 2).
- **O1's gap** (30 episodes on the N = 6 FerrySim cells at the base and at each setting): built, **not run**.
- **Study 5.8** (F, F-cov, F-cap, F-prio, FB+wide, D3, D4, D1; knee and stress; 4 and 8 missions): **not run** (batch 2). It is where the cap and priority attribution is tested; at N = 6 the cap never binds in 5.14.
- **Study 5.11 (b)** beyond the one N = 12, K = 3 point, and **(c)** FerrySim at N ≥ 24: batch 2 and batch 3.
- A plan-time learned value: no. Nothing was built because Study 5.5 gave no slope on the flight side.

---

## 14. Notes against other records (not contradictions of the overview unless stated)

1. **Related Work §3.1** says FedEx's Gibbs assignment "is never exercised" and "D4 tests FedEx's tour and FedEx's merge, not its assignment". True of the *per-mission policy* (`FedExCarpPolicy` is CARP's inner level, K = 1; its docstring says the outer level "is not called by the policy"), but the Exp 4/5 driver **does** run `carp_assign` for D4 whenever `n_mules > 1` (`driver.py:516`, call at 2217). For the 3-mule batch 1 cells, D4's device split is CARP's; the sentence should say "not its assignment at K = 1".
2. **Overview §1.4** gives N = 24 as F 749 against H1 951 and D3 911; the same table has D4 at 752, a tie (§13.2). Not wrong, but incomplete for a claim about the route.
3. **Build plan**: lists O1 as `oracle/zhai_oracle.py`, *deferred*, "under visit-within-S". The oracle is built as `experiments/analysis/o1_oracle.py` and ranks by F's plan key (the cap key first), not a literal visit-within-S constraint.
4. **Cluster registry**: its docstring and `rebalance` are round-robin by id; the experiments' spatially coherent slices come from the harness (angular sectors or CARP), not the registry (§9). Anyone reading `device_registry.py` alone would infer a non-spatial split.
5. **`member_admission` default** is `whole` on `MuleConfig` and `subset` on `PlanOptions`; both are right (the driver passes `subset` for plan arms). The Configuration Reference counts the search settings as "four" (`hover_stops` is omitted from `as_dict` at its default of True; `plan/types.py:487-491` has five fields).
6. **Planner time**: the Phase 4 worst case (3.1–3.4 s at N = 96) and 5.11 (a)'s 1.53 s are different conditions (§10) and are not a regression or an improvement of each other.

---

## 15. Open items

- [ ] **Run Study 5.4 and O1** (batch 2, ~50 h with the rest): the only direct test of C1 and the only measure of how far the plan search is from optimal. Until then "reach pays" is untested and the N = 6 lead is a bundle (§13.2).
- [ ] **A gap measure above N = 6.** O1 is bounded to N ≤ 6, exactly where the search is already exact. Whether the heuristic above 6 devices loses anything *on the study's cells* is measured only indirectly (5.11 (a)'s forced-local vs `auto`, N = 12: share 0.424 against 0.457) and its mode per N is unverified.
- [ ] **A swap move** for the local search (two capped stops competing for one slot), or a stated reason to leave the gap.
- [ ] **A larger-N ablation set for 5.14**; at N = 6 every plan switch but `whole` is identical to F.
- [ ] **Fix the T_nom pricing or keep it documented** (critic C4: about 4× as many F missions as wide-arm missions fit a deadline window).
- [ ] **Hover on energy** if an energy capacity is ever set (the rule minimises time).
- [ ] **Per-decision timing** for the slot and E3 (5.11 (a) reports `decide_s` null for FX) and the share of missions per search mode.
- [ ] **A spatial registry rule** if the cluster is ever to slice without the harness; and an honest K > 1 statement on D4's Gibbs (§14.1).
- [ ] **Bound-derived V constants** (overview J4): hand-set and swept today; a decision pending before batch 2's 5.1 and 5.7.

---

## 16. Sources

Code: `hermes/scheduler/plan/{plan_search,plan_score,member_subset,hover,types}.py`; `hermes/scheduler/routing/two_opt.py`; `hermes/scheduler/fl_scheduler.py` (`build_ferry_plan`, 1454-1705); `hermes/types/scheduler.py` (`PlanCommit`); `hermes/scheduler/stages/{s3a_cluster,s3d_age_cap}.py`; `hermes/scheduler/policies/{fedex_carp,next_stop,chen_dqn,cross_heuristic}.py`; `hermes/cluster/device_registry.py`; `hermes/processes/cluster.py`; `experiments/exp4/{driver,topology_builder}.py`; `experiments/analysis/o1_oracle.py`; `scripts/exp5/params.toml`.

Records: [FeRRy build plan](../FeRRy_Build_Plan.html) (Phase 4; Studies 5.4, 5.8, 5.11, 5.14; arms table) · [Configuration Reference](../HERMES_Configuration_Reference.md) §18, §20.11 · [Related Work notes](../HERMES_Related_Work_Notes.md) §3.1, §3.3 · [RL decision memo](../architecture%20review/rl-decision-memo/HERMES_Layer_Redefinition_and_RL_Decision.md) §1, §3, §5 · [Experiment 5 readiness](../Experiment_5_Readiness.md) · [Scheduler Freeze](../HERMES_Scheduler_Freeze.md) §5k · batch 1 scores `results/exp5/scores/b1/{s53,s59,s514}.md` and `*_arms.csv`, per-trial CSVs `results/exp5/b1/s514/`, 5.11 (a) rerun `results/exp5/s511a_quiet/b1/s511a/`, commit message of `c784699`.

*Numbers are copied from those records as of 7 Oct 2026; none was re-run. The band shares and served shares in §4 and §13.3 are means tallied here from the per-trial scored CSVs and do not appear in a score report.*
